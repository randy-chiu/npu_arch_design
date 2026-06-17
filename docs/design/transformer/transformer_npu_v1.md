# Transformer NPU Architecture v1

## 1. Target / 目标

Transformer NPU v1 turns the project from a CNN/MNIST regression SoC into a
Transformer-oriented tensor NPU baseline. The goal is a verified and PPA-visible
foundation for edge LLM inference, not a full LLaMA runtime.

The longer-term target is not merely functional Transformer support. This
architecture must support a repeatable exploration loop where representative
LLM Prefill/Decode workloads identify a measured bottleneck, a spec-first
candidate changes the architecture, and candidate-versus-baseline PPA evidence
determines whether the change is retained. A feature that passes RTL but lacks
a workload-level PPA comparison is functional progress, not an accepted
architecture optimization.

长期目标不仅是支持Transformer功能，而是建立可持续迭代的LLM NPU架构探索
平台：由代表性Prefill/Decode workload暴露量化瓶颈，先修改spec和设计，再
实现Compiler/Runtime/RTL，最后通过候选方案与保留基线的PPA对比决定是否
接受。仅RTL功能正确但没有workload级PPA收益证据，只能算功能进展，不能算
已经完成的架构优化。

V1 scope:

- single-batch tiny decoder-only Transformer envelope;
- `seq_len = 32 / 128`, `hidden = 64 / 128`, `heads = 4`,
  `head_dim = 16 / 32`;
- int8 activation/weight, int32 accumulator;
- fixed-point softmax internals;
- int8 KV cache v1 as spec/counters first;
- primitive uops and micro-kernels before macro-op hardware;
- MNIST/CNN remains a regression workload.

The next functional acceptance target is one complete tiny LLaMA-like Prefill
Decoder Block followed by two chained blocks. Local hardware optimization is
deferred until block-level PPA identifies the dominant bottleneck. Micro
workloads remain diagnostic tools, not the sole optimization decision surface.

下一阶段首先让NPU完整执行一个tiny LLaMA-like Prefill Decoder Block，再串联
两个Block。局部硬件优化暂缓，直到Block级PPA识别出主要瓶颈；微算子只用于
定位问题，不能单独决定优化优先级。

The frozen B0/B1 functional workload is TinyLlama-derived:

```text
S=8, H=16, Q_heads=2, KV_heads=1, head_dim=8, FFN=32
B0 = one block
B1 = two chained blocks with distinct weights
```

This deliberately preserves GQA, RoPE, RMSNorm, SwiGLU, causal Attention, and
both residual connections. It excludes embeddings, LM head, tokenization, and
sampling because the current decision surface is the Decoder Block datapath.

Out of v1 scope: complete LLaMA, fused attention pipeline, hardware macro-op
expansion, reorder scheduler, real LPDDR controller, INT4/FP8, multi-core NPU.

### Architecture decision evidence / 架构决策证据

The fast iteration loop uses the RTL workload view: RTL-measured performance
plus normalized structural area/energy estimates. Substantial resource changes
such as wider datapaths, additional storage ports, larger arrays, and fused
pipelines also require the mapped area/timing view before acceptance.
Power/energy conclusions require later activity-power evidence. Every
comparison must declare workload suite, baseline variant, process/library and
clock assumptions, SRAM/external-memory accounting, and activity scope.

快速迭代阶段继续使用RTL workload view；但扩大数据通路、增加存储端口、
扩大阵列或增加融合流水线等重大资源修改，在正式接受前必须经过mapped
area/timing view验证。功耗与能耗结论还需要后续基于activity的证据。

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

### 3.2 Complete Decoder Block Plan / 完整Decoder Block执行计划

#### Problem / 要解决的问题

The current executable path proves individual Attention stages, but it cannot
identify the dominant bottleneck of a real Decoder Block. Projection/FFN
matrices, normalization, residual movement, activation, repeated heads, and
block boundaries may dominate the measured result. A Python-computed
intermediate would hide exactly those costs.

当前可执行路径只证明了Attention局部阶段，无法判断完整Decoder Block真正的
瓶颈。若由Python或fixture生成中间结果，会隐藏Norm、Projection、FFN、
Residual和Block边界的真实硬件开销，因此不能算完整Block执行。

#### B0 stage order and buffer contract / B0阶段与Buffer约束

```text
x
 -> rmsnorm_attn
 -> q_proj, k_proj, v_proj
 -> rope_q, rope_k
 -> head0_attention, head1_attention using shared K/V head
 -> concat_heads
 -> o_proj
 -> residual_attn
 -> rmsnorm_ffn
 -> gate_proj, up_proj
 -> silu_gate
 -> gate_mul_up
 -> down_proj
 -> residual_ffn
 -> block_output
```

The first baseline materializes every named intermediate in workspace SRAM.
This is intentionally expensive but observable. Residency, fusion, concurrent
heads, fused QKV, and command lists remain later candidates measured against
this baseline.

Each BlockPlan stage records:

```text
stage_id, logical_op, inputs, outputs
logical shape and dtype
lowered tile jobs or primitive program
execution_state and provenance
```

Allowed aggregate states are:

- `planned_not_executable`: plan/golden exists but no complete RTL path;
- `partially_executable`: some measured RTL stages and explicit gaps;
- `executable`: every stage has measured RTL provenance and complete output
  matches golden.

The Compiler may calculate expected results for verification, but Runtime and
firmware must never substitute those expected intermediates in an accepted
`executable` run.

#### Numerical bring-up policy / 数值Bring-up策略

- activation and weights are deterministic signed INT8;
- matrix accumulation is INT32, followed by explicit saturating INT8
  requantization at BlockPlan stage boundaries;
- RMSNorm uses the existing `SUMSQ -> RSQRT -> Vector scale` bring-up contract;
- RoPE uses a deterministic fixed-point table owned by the Compiler fixture,
  then must execute through reviewed Vector primitives before B0 acceptance;
- causal Attention reuses the existing `S=8,D=8` numerical contract per head;
- SwiGLU uses `SiLU(gate) * up`; its fixed-point approximation must be reviewed
  before RTL execution is accepted.

These rules favor deterministic functional closure. Their accuracy is not a
claim that the final quantization policy is selected.

#### B1 chaining contract / B1串联约束

B1 contains two B0-shaped plans. Block 0 `block_output` is the sole input of
Block 1; no CPU recomputation, fixture replacement, or reset to the original
input is legal. Block-level reporting must preserve stage identity as
`block0/...` and `block1/...` and also publish the combined run.

#### Current executable subset / 当前可执行子集

The first executable subsets are the B0 matrix subgraph, RMSNorm vector
subgraph, Attention-head subgraph, gate-multiply vector subgraph, and residual
vector subgraph. The matrix subgraph intentionally covers only the seven
matrix stages:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Compiler lowering emits 16 output-tile `MATMUL_K_STREAM` descriptor jobs, which
contain 36 physical `8x8x8` Matrix Engine invocations. The measured RTL result
matches the Compiler golden for every tile.

This subset is useful because it makes projection and FFN movement/control cost
visible before complete B0 is ready. It is not accepted as complete B0 because
RoPE and SiLU still lack measured RTL provenance, and the executable subgraphs
are still submitted as separate groups rather than one block execution package.

Measured matrix-subgraph evidence:

| Metric | Value |
| --- | ---: |
| descriptor jobs | 16 |
| physical Matrix invocations | 36 |
| effective MAC ops | 18432 |
| theoretical Matrix cycles | 288 |
| measured Matrix cycles | 288 |
| total cycles | 2008 |
| Data Mover active cycles | 1472 |
| Matrix utilization | 1.0 |
| end-to-end efficiency | 0.143426 |

The conclusion is functional, not yet an optimization decision: the Matrix
Engine is fully utilized while active, but the current per-tile descriptor and
external SRAM movement boundaries dominate elapsed cycles.

The RMSNorm vector subgraph covers `rmsnorm_attn` and `rmsnorm_ffn` through
32 `DESC_VECTOR_TILE_V1` descriptor jobs. Each output segment job loads both
`H=16` row segments, runs `SUMSQ_SRC0 + SUMSQ_SRC1 + RSQRT + SCALE_SRCx`, and
stores one eight-lane output segment. Measured RTL evidence:

| Metric | Value |
| --- | ---: |
| descriptor jobs | 32 |
| effective vector lane ops | 256 |
| theoretical Reduction cycles | 64 |
| theoretical SFU cycles | 32 |
| theoretical Vector cycles | 32 |
| measured compute cycles | 128 |
| total cycles | 1344 |
| Data Mover active cycles | 320 |
| primitive compute efficiency | 1.0 |
| end-to-end efficiency | 0.095238 |

中文说明：RMSNorm现在已经通过Reduction、SFU和Vector三个执行单元完成，
不是CPU预先计算`inv_rms`后塞给硬件。当前baseline每个输出segment都重复
计算整行`sumsq/rsqrt`，性能不优，但这正好给后续row-state cache或
command-list fanout提供了可量化基线。

The Attention-head subgraph covers the two B0 query heads with shared K/V
head. For each head, firmware submits the measured stage sequence:

```text
QK matmul -> Scale/Mask -> Softmax -> PV matmul
```

The Q and K inputs are currently `rope_q` and `rope_k` from the compiler
golden, so this subgraph does not prove RoPE execution. Softmax expected output
uses the current RTL bring-up LUT contract, not the final BlockPlan fixed-spec
golden. It does prove that the B0 two-head causal Attention dataflow can run
through the existing Matrix, Vector, Reduction, SFU, Data Mover, Scheduler, and
Command Processor paths and can be attributed as one B0 Attention workload in
PPA.

中文说明：B0 Attention子图把两个query head都跑过QK、Scale/Mask、Softmax、
PV，并在PPA中聚合显示该子图的模块开销。但它的输入仍是compiler golden生成的
`rope_q/rope_k`，Softmax数值也仍是当前RTL bring-up LUT合同，因此RoPE和最终
Softmax数值仍是完整B0的缺口，不能把这个子图说成完整Attention Block已经闭合。

Measured Attention-subgraph evidence:

| Metric | Value |
| --- | ---: |
| descriptor jobs | 8 |
| heads | 2 |
| effective MAC ops | 2048 |
| theoretical Matrix cycles | 32 |
| measured Matrix cycles | 32 |
| total cycles | 1550 |
| Data Mover active cycles | 406 |
| SFU active cycles | 144 |
| Matrix utilization | 1.0 |
| end-to-end efficiency | 0.020645 |

Per head, the measured stage costs are `QK=83 cycles`,
`Scale/Mask=85 cycles`, `Softmax=526 cycles`, and `PV=81 cycles`. This makes
the current bottleneck explicit: Matrix work is not the limiting factor for
the fixed `S=8,D=8` B0 Attention subgraph; separate descriptor boundaries and
serialized row Softmax dominate elapsed cycles.

中文说明：B0 Attention子图的Matrix部分活跃时利用率为1.0，问题不在QK/PV矩阵
乘本身，而在每个stage单独提交带来的搬运/控制开销，以及当前Softmax逐行串行
执行。后续优化应先完成完整B0功能闭环，再用这个PPA基线判断是否需要Softmax
row pipeline、command-list或stage fusion。

The gate-multiply vector subgraph covers `gate_mul_up` through 32
`DESC_VECTOR_TILE_V1` descriptor jobs. Each FFN row has width 32 and is split
into four eight-lane jobs. The primitive program is:

```text
VMUL silu_gate_segment, up_segment
VREQUANT arg1=INT8_SHIFT4_CLAMP
HALT
```

This does not make the preceding `silu_gate` stage executable. SiLU still needs
a reviewed SFU/vector approximation before complete B0 acceptance. The gate
multiply slice is still valuable because it exposes the FFN elementwise
multiply/requant movement and control cost instead of hiding it in the Python
golden.

Measured gate-multiply evidence:

| Metric | Value |
| --- | ---: |
| descriptor jobs | 32 |
| effective vector lane ops | 256 |
| theoretical Vector cycles | 64 |
| measured Vector cycles | 64 |
| total cycles | 1088 |
| Data Mover active cycles | 320 |
| Vector compute efficiency | 1.0 |
| end-to-end efficiency | 0.058824 |

中文说明：`gate_mul_up`已经由NPU执行乘法和`>>4`后int8 clamp，不再由Python
直接生成该stage结果。但`SiLU(gate)`本身仍是前置golden输入；后续必须评审
SiLU近似的SFU/Vector primitive序列，才能把完整SwiGLU标为NPU执行。

The residual vector subgraph covers `residual_attn` and `residual_ffn` through
32 `DESC_VECTOR_TILE_V1` descriptor jobs. Each `H=16` row is split into two
eight-lane `VADD + HALT` primitive programs. Measured RTL evidence:

| Metric | Value |
| --- | ---: |
| descriptor jobs | 32 |
| effective vector lane ops | 256 |
| theoretical Vector cycles | 32 |
| measured Vector cycles | 32 |
| total cycles | 960 |
| Data Mover active cycles | 320 |
| Vector compute efficiency | 1.0 |
| end-to-end efficiency | 0.033333 |

中文说明：Residual Add已经通过通用`DESC_VECTOR_TILE_V1`路径执行和验证，
不是完整B0的剩余缺口。它暴露出的主要问题是每个8-lane segment都独立提交
descriptor，导致Vector Engine实际只工作32 cycle，但端到端耗时960 cycle。
这类边界开销应在完整B0 PPA之后作为command-list或融合候选来评估。

### 3.3 Vector Tile And Segmented-Row Path / Vector Tile与分段行路径

#### Problem / 要解决的问题

B0 now has measured matrix, RMSNorm, gate-multiply, and residual-vector
subgraphs, but the remaining
stages still cannot be accepted as executable:

```text
RoPE, Attention head composition, SiLU
```

These stages are not Matrix Engine work. They are row/vector operations over
`S=8, H=16` or `S=8, FFN=32` tensors. The current measured Vector/Reduction/SFU
path is tied to Attention row storage and fixed eight-lane Attention programs.
It can execute `S=8,D=8` Softmax rows, but it is not a general way to run a
16-wide hidden row or a 32-wide FFN row.

中文说明：B0剩余算子不是矩阵乘，而是对`H=16`或`FFN=32`的行向量做归一化、
旋转、激活和逐元素乘法，并把两个Attention head组合进Block执行流。
当前Attention专用row path只能自然表达
8-lane score row，不能把这些更宽的行向量作为通用硬件工作负载执行。如果继续
在fixture或CPU里生成这些中间结果，就会隐藏B0真正的数据搬运和Vector/Reduction
开销，因此不能算完整B0。

#### Why not add one descriptor per operator? / 为什么不为每个算子新增专用descriptor

Adding `RMSNORM_OP`, `ROPE_OP`, `RESIDUAL_OP`, and `SWIGLU_OP` as private
Compute-cluster state machines would repeat the old Softmax problem: operator
sequencing would live in RTL control instead of the common Scheduler, PPA would
show opaque control buckets, and later changes to numerical policy would
require RTL rewrites.

The accepted direction is one common descriptor family:

```text
DESC_VECTOR_TILE_V1 descriptor
  -> Data Mover loads one or two tensor segments
  -> Uop Scheduler executes a compiler-expanded primitive program
  -> Vector / Reduction / SFU engines execute primitive commands
  -> Data Mover stores the produced segment or row result
```

This keeps complex operators as compiler micro-kernels, not hardware macros.
It also gives PPA one comparable surface for all non-matrix B0 stages.

#### Logical row segmentation / 逻辑行分段

The physical Vector Engine has eight lanes. B0 rows may be wider:

| Tensor row | Width | Physical segments |
| --- | ---: | ---: |
| hidden row | 16 | 2 x 8-lane segments |
| FFN intermediate row | 32 | 4 x 8-lane segments |
| attention head row | 8 | 1 x 8-lane segment |

Compiler lowers a logical row into segment descriptors:

```text
segment_width = 8
segment_count = ceil(row_width / 8)
for row in rows:
  for segment in segments:
    emit DESC_VECTOR_TILE_V1 descriptor or command-list entry
```

For operations requiring whole-row state, Compiler emits a two-pass segmented
micro-kernel. RMSNorm is the first example:

```text
Pass 1:
  for each segment:
    REDUCE_SUMSQ(segment)
  accumulate row_sumsq across segments
  SFU_RSQRT(row_sumsq)

Pass 2:
  for each segment:
    VEC_SCALE(segment, inv_rms)
```

The first implementation may materialize `row_sumsq` and `inv_rms` in SRAM or
a small local scalar file. It must report that movement/storage explicitly.
Keeping these scalars resident is a later optimization candidate.

#### Workload/planner contract check / Workload与Planner一致性检查

B0/B1 deliberately do not use a general graph parser yet. The checked-in
workload JSONC declares the reviewed shape, topology summary, and planner
entry point; `npu_compiler.block` expands that fixed workload into the
BlockPlan, golden tensors, tile jobs, and primitive segment jobs. To prevent
two independent definitions from drifting apart, the compiler regression
validates:

- B0/B1 JSONC shape equals the generated planner shape;
- B0 topology summary equals the expanded stage order;
- planner names and execution states match the invoked planner;
- B1 boundary metadata matches the generated two-block plan.

中文说明：这里不做通用graph parser，但必须防止“workload里一套图、compiler里
另一套图”。后续修改B0/B1网络结构时，JSONC声明和planner展开逻辑必须一起改；
否则contract test会失败。

#### RMSNorm primitive lowering boundary / RMSNorm primitive降低边界

RMSNorm is the next B0 blocker after residual add. The Reduction Engine already
supports `REDUCE_SUMSQ`, and the SFU already supports `SFU_RSQRT`, but the
current `npu_v0` 4-bit UOP encoding does not expose these two operations
through `DESC_VECTOR_TILE_V1`. Therefore the compiler may emit a planned
RMSNorm segmented-row plan, but it must remain non-executable until the
primitive ISA/descriptor extension is reviewed and implemented.

The first executable baseline for one `H=16` row uses two descriptor jobs, one
per output segment. Each job loads both row segments, recomputes the full
row sum of squares, computes `rsqrt`, and scales only the selected output
segment:

```text
job for output segment0:
  input0 = row[0:8], input1 = row[8:16]
  VREDSUM arg1=SUMSQ_SRC0
  VREDSUM arg1=SUMSQ_SRC1
  VDIV    arg1=RSQRT_ROW_ACCUM
  VNORM   arg1=SCALE_SRC0_BY_SFU -> output segment0

job for output segment1:
  input0 = row[0:8], input1 = row[8:16]
  VREDSUM arg1=SUMSQ_SRC0
  VREDSUM arg1=SUMSQ_SRC1
  VDIV    arg1=RSQRT_ROW_ACCUM
  VNORM   arg1=SCALE_SRC1_BY_SFU -> output segment1
```

The first accepted RTL path must expose these as measured Reduction, SFU, and
Vector cycles. A dedicated `RMSNORM_OP` FSM is rejected for the same reason
dedicated Softmax sequencing was rejected: it would hide primitive scheduling
and make PPA less useful.

This baseline intentionally repeats `SUMSQ/RSQRT` for each output segment. It
does not hide row-state in CPU preprocessing, and it creates a clear PPA
baseline for later row-state caching or command-list fanout.

中文说明：RMSNorm不能通过新增一个黑盒`RMSNORM`硬件状态机来绕过。正确方向是
让compiler生成`REDUCE_SUMSQ -> SFU_RSQRT -> VEC_SCALE`这类primitive序列，
PPA中能看到Reduction、SFU、Vector分别工作了多少cycle。当前先生成
每个输出segment一个descriptor job的baseline：每个job都加载同一行的两个
segment，在NPU里重复计算完整row的`sumsq/rsqrt`，然后只输出当前segment。
这样性能不最优，但不会把`inv_rms`藏在CPU里，PPA也能真实显示后续优化空间。

#### B0 stage mapping / B0阶段映射

| Stage | Width | First `DESC_VECTOR_TILE_V1` mapping | Whole-row state |
| --- | ---: | --- | --- |
| `rmsnorm_attn` | 16 | segmented `REDUCE_SUMSQ`, `SFU_RSQRT`, `VEC_SCALE` | `sumsq`, `inv_rms` |
| `rope_q` | 16 | two segments per row, pairwise rotate using compiler-provided sin/cos constants | none across segments |
| `rope_k` | 8 | one segment per row | none |
| `residual_attn` | 16 | `VEC_ADD` per segment | none |
| `rmsnorm_ffn` | 16 | same as `rmsnorm_attn` | `sumsq`, `inv_rms` |
| `silu_gate` | 32 | bring-up LUT/approximation through SFU or reviewed vector approximation | none across segments |
| `gate_mul_up` | 32 | `VEC_MUL` + `VEC_REQUANT arg1=INT8_SHIFT4_CLAMP` per segment | none |
| `residual_ffn` | 16 | `VEC_ADD` per segment | none |

RoPE and SiLU require reviewed numerical approximations before they are marked
`executable`. Until then, their BlockPlan stages remain
`planned_not_executable` even if the descriptor transport exists.

#### Descriptor and storage model / Descriptor与存储模型

The first `DESC_VECTOR_TILE_V1` descriptor is intentionally segment-oriented:

```text
input0_addr/input0_words = first segment tensor
input1_addr/input1_words = optional second segment tensor or constants
input2_addr             = optional scalar/state table in descriptor v1
output_addr/output_words = output segment tensor
program_addr/program_words = compiler-expanded primitive program
m = rows in this job
n = logical row width for metadata/PPA
k = segment offset or segment count, depending on descriptor flags
flags = op class, input count, segment count, row-state policy
```

The current v0 ABI lacks `input2`, `m/n/k`, and flags, so the first executable
implementation may use fixed firmware-generated descriptors for bring-up. The
canonical v1 spec still records the full contract now so implementation does
not depend on hidden fixture assumptions.

#### v0 generic primitive carrying boundary / v0通用primitive承载边界

The current RTL cannot honestly be called a generic primitive-program carrier
until two Attention-specific constraints are removed:

| Constraint | Why it blocks B0 |
| --- | --- |
| Compute-cluster primitive routing reads and writes the Attention score row in `C` | RMSNorm, residual, RoPE, and SwiGLU need arbitrary tensor segments, not only `score[row][lane]` |
| Matrix operand windows are `A16` and `B8` while vector outputs are `C32` | a general vector program needs two 32-bit source segments and one 32-bit destination segment; using `A16/B8` would silently truncate non-Attention data |

中文说明：这里的“通用”不是指硬件能理解`RMSNorm`、`RoPE`、`SwiGLU`这些
模型算子名字，而是指硬件能接收Compiler生成的primitive program，并把descriptor
中声明的两个输入segment搬到通用32-bit向量源窗口，再由Scheduler发给
Vector/Reduction/SFU执行，最后把结果写回32-bit输出窗口。

The accepted v0 bring-up mechanism is:

```text
DESC_VECTOR_TILE_V1 op_type
  input0 -> vector_src0[8] 32-bit
  input1 -> vector_src1[8] 32-bit when binary/constants are needed
  output <- vector_dst[8] / C window
  program -> existing 16-word primitive instruction memory
```

This boundary now supports simple 8-lane `VADD`, segmented RMSNorm
Reduction/SFU/Vector programs, and `VMUL + VREQUANT` gate-multiply programs.
It does not yet claim RoPE swizzle or SiLU approximation support. Those remain
explicit lowering steps after the vector-tile carrier path is proven.

#### PPA acceptance / PPA验收

For every `DESC_VECTOR_TILE_V1` workload, PPA must show:

```text
logical_elements
physical_segment_count
useful_vector_lane_ops
vector_active_cycles
reduction_active_cycles
sfu_active_cycles
data_mover_active_cycles
descriptor/control cycles
```

For segmented whole-row operations such as RMSNorm, PPA must additionally show:

```text
segment_reduce_cycles
row_state_accumulate_cycles
rsqrt_cycles
segment_scale_cycles
scalar/state movement cycles
```

B0 can be marked `executable` only when these non-matrix stages appear with
measured RTL provenance and the final `block_output` matches the documented
golden. Model-only or CPU-filled intermediates must remain visible as gaps.

#### Deliberate non-goals / 非目标

- no dedicated RMSNorm, RoPE, or SwiGLU hardware macro in this step;
- no multi-row Vector parallelism yet;
- no command-list fusion yet;
- no claim that the bring-up RoPE/SiLU numerical approximation is final;
- no optimization decision from the matrix-subgraph PPA alone.

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
