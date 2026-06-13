# Transformer NPU Architecture v1

## 1. Target / 目标

Transformer NPU v1 turns the project from a CNN/MNIST regression SoC into a
Transformer-oriented tensor NPU baseline. The goal is a verified and PPA-visible
foundation for edge LLM inference, not a full LLaMA runtime.

V1 scope:

- single-batch tiny decoder-only Transformer envelope;
- `seq_len = 32 / 128`, `hidden = 64 / 128`, `heads = 4`,
  `head_dim = 16 / 32`;
- int8 activation/weight, int32 accumulator;
- fixed-point softmax internals;
- int8 KV cache v1 as spec/counters first;
- primitive uops and micro-kernels before macro-op hardware;
- MNIST/CNN remains a regression workload.

Out of v1 scope: complete LLaMA, fused attention pipeline, hardware macro-op
expansion, reorder scheduler, real LPDDR controller, INT4/FP8, multi-core NPU.

## 2. Overall Design / 整体设计思路

The architecture stays unified:

```text
workload graph
  -> static Compiler: legalization / fusion / tiling / schedule
  -> compiled execution package
  -> Runtime: bind / stage / submit / synchronize
  -> CPU / firmware
  -> wrapper / CSR
  -> core Command Processor / descriptor engine
  -> uop scheduler
  -> memory + data mover
  -> matrix / accumulator / vector / reduction / SFU engines
  -> perf + PPA report
```

CNN and Transformer jobs share wrapper, descriptor identity, workload manifest,
perf report, and PPA Level 0 report. Transformer support is added as new
primitive capabilities and workload metadata rather than as a second core.

The first Transformer path uses current K-stream matmul where possible, adds
golden/model-only coverage for vector/reduction/SFU micro-kernels, and exposes
utilization fields so decode GEMV/skinny-GEMM waste is visible before building
new datapaths.

The system-level ownership contract is defined in
`software_runtime_compiler_attention.md`. Static Compiler owns graph lowering,
fusion, M/N/K tiling, and executable schedule generation. Runtime binds and
submits that schedule. The Host Wrapper is intentionally thin and must not
parse graphs, split large operators, or perform fusion.

系统级职责由`software_runtime_compiler_attention.md`定义：静态Compiler负责
计算图lowering、融合、M/N/K分块和执行计划生成；Runtime负责绑定并提交；
Host Wrapper保持轻量，不解析计算图、不拆分大算子，也不负责融合。

## 3. Key Details / 重点细节

Canonical v1 references:

| File | Role |
| --- | --- |
| `arch/configs/npu_transformer_v1.jsonc` | v1 architecture config |
| `arch/specs/transformer/v1/transformer_npu_v1.md` | module, counter, and PPA contract |
| `arch/specs/transformer/v1/transformer_numerical_v1.md` | fixed-point softmax/RMSNorm numerical contract |
| `arch/specs/transformer/v1/csr_map_v1.md` | wrapper CSR map |
| `arch/specs/transformer/v1/descriptor_v1.md` | job descriptor and job types |
| `arch/specs/transformer/v1/uop_isa_v1.md` | primitive uop ISA |

Primitive uop, micro-kernel, macro-op expansion, and fused hardware pipeline
are distinct:

- primitive operation: a lowest-level scheduler-visible building block
  executed by one engine, such as vector subtract, reduce max, SFU EXP, or one
  matrix tile multiply;
- primitive uop: an encoded command that directly issues one primitive
  operation;
- micro-kernel: compiler/software sequence of primitive uops;
- macro-op expansion: future scheduler/compiler expansion of compact row ops;
- fused pipeline: dedicated multi-stage hardware datapath, out of v1 scope.

For example, attention softmax is a micro-kernel assembled from reduction,
vector, and SFU primitive operations. The primitive valid/ready contract
defines how those operations are accepted and how their results are returned
without relying on fixed engine latency.

The accumulator file becomes an explicit architectural module:

| Field | v1 value |
| --- | --- |
| dtype | int32 |
| tile | 8 x 8 |
| banks | 2 |
| operations | clear, accumulate/write, read, store path |
| counters | read, write, clear, residency cycles, spill count |

Current RTL keeps the verified `hw/` layout. The staged target layout remains
`rtl/npu/matrix/accumulator_file.sv`; the current implementation equivalent is
`hw/npu_core/rtl/matrix/accumulator_file.sv`.

### 3.1 Large Matrix Execution On The 8x8 Matrix Engine / 基于8x8 Matrix Engine的大矩阵执行

#### Problem / 要解决的问题

The Matrix Engine physically computes one `8x8x8` multiplication, while real
Transformer matrices commonly have `M`, `N`, or `K` larger than eight. Making
the physical array as large as every logical matrix would sharply increase
area, routing, SRAM bandwidth, and idle capacity for small/decode shapes.

物理Matrix Engine一次只能计算`8x8x8`，但真实Transformer矩阵的M、N、K
通常都大于8。直接把硬件阵列扩大到逻辑矩阵大小会显著增加面积、布线和
带宽，而且在小矩阵或decode场景下会产生大量闲置。因此大矩阵应通过分块
调度复用现有阵列。

#### Proposed baseline for review / 待评审基础方案

For `C[M,N] = A[M,K] * B[K,N]`, Compiler tiles all three dimensions:

```text
TM = ceil(M / 8)
TN = ceil(N / 8)
TK = ceil(K / 8)

for tm in range(TM):
  for tn in range(TN):
    acc[8,8] = 0
    for tk in range(TK):
      acc += A_tile[tm,tk] * B_tile[tk,tn]
    C_tile[tm,tn] = acc
```

One output tile `(tm,tn)` is one `MATMUL_K_STREAM` descriptor. Compiler and
Runtime own the outer M/N loops by emitting and launching multiple descriptors.
The Command Processor owns the descriptor-local K loop through `k_chunks`.
Matrix Engine only executes physical tile primitives; Accumulator File keeps
the partial output tile resident across K chunks.

每个输出tile对应一个`MATMUL_K_STREAM` descriptor。Compiler和Runtime生成
并启动M/N方向的多个descriptor；Command Processor根据`k_chunks`完成当前
输出tile内部的K循环；Matrix Engine只执行物理tile乘法；Accumulator File
在K循环期间保存int32部分和。该方案复用现有K-stream，不新增矩阵ISA。
当前只实现了单个输出tile的K-stream；M/N多tile和边界tile仍需评审后编码。

#### Example: 16x16x16 / 示例

```text
A[16,16] * B[16,16] -> C[16,16]
TM=2, TN=2, TK=2
```

Runtime launches four output-tile descriptors:

| Descriptor | Accumulated physical operations | Output |
| --- | --- | --- |
| `C[0:8,0:8]` | `A[0:8,0:8]*B[0:8,0:8]` + `A[0:8,8:16]*B[8:16,0:8]` | C tile 0,0 |
| `C[0:8,8:16]` | two K chunks | C tile 0,1 |
| `C[8:16,0:8]` | two K chunks | C tile 1,0 |
| `C[8:16,8:16]` | two K chunks | C tile 1,1 |

This produces eight physical Matrix Engine invocations. With the current
one-K-slice-per-cycle design, theoretical Matrix Engine active time is
`8 invocations * 8 cycles = 64 cycles`. This does not include movement,
descriptor, accumulator-control, store, or stall cycles.

#### Dataflow and overlap / 数据流与并行

For one output tile:

```text
clear accumulator
load A0/B0 into bank0
compute chunk0 while Data Mover prefetches A1/B1 into bank1
compute chunk1 while optional next work is prepared
store completed accumulator tile
```

Double buffering allows Data Mover and Matrix Engine to overlap only when they
access different banks and bandwidth is sufficient. If an A/B chunk takes
longer to load than eight Matrix cycles, Matrix Engine must show an explicit
input-wait stall between chunks. PPA timelines must not imply uninterrupted
compute when the next bank is not ready.

双缓冲允许Data Mover搬运下一K chunk时Matrix Engine计算当前chunk，但前提
是使用不同bank且带宽足够。如果搬运一组A/B tile需要的时间超过8个Matrix
cycle，Matrix Engine必须等待，PPA时间轴应明确显示input-wait stall，而
不能把理论上的重叠画成已经充分流水。

#### Boundary tiles / 边界tile

The first correct baseline zero-fills invalid A/B lanes for shapes not divisible
by eight and stores only logical C rows/columns. Compiler records valid M/N/K
extents per tile. This is simple and deterministic but wastes Matrix capacity;
later hardware lane gating or compiler tile skipping requires measured PPA
benefit before adoption.

首个正确版本对不足8的A/B边界lane补零，并只写回有效C区域。这样会浪费
部分MAC能力，但实现简单且容易验证。后续是否增加lane gating或跳过tile，
必须通过PPA证明收益大于控制、面积和功耗代价。

#### Descriptor and ownership / Descriptor与职责

No new matrix instruction is required for the baseline:

- Compiler chooses tile order, source/output offsets, valid extents, and emits
  one descriptor per output tile;
- Runtime allocates/stages buffers and launches descriptors;
- Command Processor validates the descriptor and sequences `k_chunks`;
- Data Mover loads A/B chunks and stores C;
- Matrix Engine performs exactly one physical tile operation per chunk;
- Accumulator File clears once, accumulates all chunks, and stores once.

The current descriptor can express one full `8x8` output tile and its K chunks,
but does not yet carry general M/N tile offsets or boundary valid extents.
The first implementation may materialize per-tile base addresses in each
descriptor. Before executable boundary tiles are added, the canonical
descriptor spec must define valid extents rather than relying on fixture
padding assumptions.

#### Performance and PPA acceptance / 性能与PPA验收

For each logical GEMM, the report must show:

```text
physical_tile_invocations = ceil(M/8) * ceil(N/8) * ceil(K/8)
theoretical_matrix_cycles = physical_tile_invocations * 8
useful_mac_ops = M * N * K
issued_mac_capacity = physical_tile_invocations * 512
```

It must separately show descriptor cycles, A/B load cycles, Matrix active and
input-wait cycles, accumulator clear/commit cycles, C store cycles, overlap,
and tail waste. Initial acceptance workloads are `8x8x16`, `16x8x8`,
`8x16x8`, `16x16x16`, and one non-multiple-of-eight boundary case.

#### Review decisions before coding / 编码前待评审项

1. Accept one descriptor per output tile for the first correct baseline.
2. Decide the canonical descriptor representation for boundary valid extents.
3. Confirm whether M/N descriptor sequencing remains in Runtime until command
   lists are introduced.
4. Define the required SRAM layout and stride fields before non-contiguous
   model tensors are supported.
5. Use measured stalls to decide whether larger buffers, wider movement, or
   command-list scheduling is the next optimization.

## 4. Verification / 验证测试

V1 acceptance keeps existing gates:

```text
make test
make firmware-smoke
make perf-report
make ppa-proxy-report
```

New Transformer coverage must add:

- Python golden for micro workloads;
- at least one executable Transformer micro workload in `perf-report`;
- shape metadata in workload manifest;
- matrix/GEMV/skinny-GEMM utilization fields in perf and PPA reports;
- null/unavailable fields when a metric is not implemented rather than guessed;
- explicit Level 0 proxy wording for modeled area/energy.

## 5. Implementation Priority / 实现优先级

1. Land v1 architecture/spec/numerical/CSR/descriptor/uop documents and config.
2. Add standalone accumulator file RTL and keep current K-stream regression
   behavior stable.
3. Extend Transformer micro workload metadata and Python golden coverage.
4. Add report-derived utilization metrics from manifest shape metadata.
5. Feed utilization and KV traffic into Level 0 PPA output.
6. Add primitive vector/reduction/SFU RTL blocks and tests after the numerical
   golden contract is stable.
