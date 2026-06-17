# Software Compiler, Runtime, And Hardware Execution Architecture

## Scope / 范围

This is the system-level ownership document for converting a logical workload
graph into NPU execution. Attention is the first complete example, but the
Compiler/Runtime/hardware boundary also applies to large GEMM, CNN, and future
operators.

本文档说明从workload计算图到NPU执行的完整系统分层。Attention是第一个完整
示例，但Compiler、Runtime和硬件之间的职责边界同样适用于大矩阵乘、CNN和
未来算子。

| Document | Owns |
| --- | --- |
| `attention_operators_v1.md` | stable attention operator names, dtypes, layouts, primitive-engine mapping, numerical contracts |
| `attention_compiler_v1.md` | lowering `scaled_dot_product_attention_v1` into QK, scale/mask, softmax, and PV stages |
| `attention_runtime_v1.md` | descriptor launch, SRAM buffers, firmware dispatch, perf/PPA capture |
| `transformer_npu_v1.md` | hardware architecture, physical tile execution, and PPA contracts |

The key architecture rule remains unchanged: attention v1 is not a dedicated
attention RTL macro. It is a compiler/runtime sequence over shared matrix,
vector, reduction, and SFU primitives.

## End-To-End Layering / 端到端系统分层

```text
workload graph / imported model
  -> graph importer and operator legalization
  -> static Compiler lowering, fusion selection, tiling, and scheduling
  -> compiled execution package
       operators + buffers + primitive programs + descriptor/command templates
  -> Runtime binds addresses and actual legal shape values
  -> Runtime submits descriptors or a command list
  -> Host Wrapper accepts CPU-visible launch and exposes status/perf
  -> NPU Core Command Processor validates and executes descriptor-local control
  -> Uop Scheduler fetches/decodes/issues primitive programs
  -> Data Mover and compute engines execute
```

首先由图导入层解析workload计算图并识别大shape算子。静态Compiler根据硬件
spec完成算子合法化、融合选择、M/N/K分块、buffer规划、原语程序生成和任务
依赖排序，输出可执行计划。Runtime不重新发明这些算法，只在执行时绑定实际
地址和合法动态shape、选择Compiler已生成的变体并提交任务。Wrapper只负责
CPU可见控制、status和perf，不负责解析计算图或拆分大算子。

## Ownership Contract / 职责契约

| Layer | Owns | Must not own |
| --- | --- | --- |
| Workload/model importer | parse graph, recover logical operators/shapes/constants | hardware tile scheduling |
| Static Compiler | operator legalization, fusion decisions, M/N/K tiling, boundary masks, buffer lifetime/allocation plan, primitive programs, descriptor/command templates, dependency DAG, theoretical work/PPA metadata | runtime MMIO polling, RTL handshake |
| Runtime | bind SRAM addresses, bind legal dynamic dimensions, select a precompiled variant, stage data/programs, submit work, wait/synchronize, handle errors, collect perf | change numerical semantics, invent a new tiling/fusion schedule |
| Host Wrapper | CPU-visible launch/status/interrupt/perf snapshot and descriptor-pointer forwarding | graph parsing, operator splitting, fusion, K-loop scheduling, primitive issue |
| Command Processor | descriptor fetch/validation, descriptor-local loops such as `k_chunks`, Data Mover orchestration, launch/completion control | high-level graph optimization or arbitrary cross-descriptor fusion |
| Uop Scheduler | primitive program fetch/decode, dependency handling, engine issue and completion wait | causal/padding policy, logical matrix tiling |
| Engines/Data Mover | execute accepted movement/arithmetic operations | decide operator graph or schedule |

Static does not mean every address and dimension is fixed. For bounded dynamic
shapes, Compiler may emit parameterized templates or several legal variants.
Runtime binds or selects them. If a shape falls outside all compiled contracts,
Runtime rejects it or invokes an explicit JIT/compiler service; it must not
silently perform ad hoc tiling itself.

静态编译不代表所有地址和shape都固定。Compiler可以针对受限动态shape生成
参数化模板或多个合法变体，Runtime执行时只负责绑定或选择。如果实际shape
超出已编译契约，应明确拒绝或调用JIT/Compiler，而不是让Runtime临时发明一
套未经验证的tiling。

For architecture exploration, a convenient test API may accept a logical
shape such as `matmul(M,N,K)` and appear to "run it automatically". The
implementation still invokes the Compiler/planner first:

```text
test requests matmul(M,N,K)
  -> Compiler/planner emits M/N/K tile descriptors and buffers
  -> thin Runtime/submitter binds addresses and launches descriptors
```

The project does not need a production-grade Runtime to test this behavior.
It needs a small, deterministic submit path that does not hard-code one shape
and therefore does not distort hardware PPA experiments.

在架构探索测试中，接口可以表现为输入一个`matmul(M,N,K)`就自动执行，但
真正负责拆分tile的仍是Compiler/planner。Runtime只绑定地址并提交descriptor。
当前不需要建设生产级Runtime，只需要一个不会硬编码单一shape、不会干扰
硬件PPA分析的轻量提交路径。

## Large GEMM Example / 大矩阵示例

For logical `C[16,16] = A[16,16] * B[16,16]` on an `8x8x8` Matrix Engine:

- Compiler sees one logical GEMM and emits four output-tile jobs;
- each output-tile job carries two K chunks and correct buffer offsets;
- Runtime binds A/B/C addresses and submits four jobs in Compiler-defined order;
- Command Processor iterates two K chunks inside each job;
- Matrix Engine sees eight physical `8x8x8` operations.

因此“把大shape算子拆成硬件可执行流”的责任属于Compiler。Runtime负责把
执行流送到硬件；Command Processor只执行descriptor规定的局部循环；
Wrapper不参与拆分。

## Fusion And Submission Models / 融合与提交方式

Three decisions must not be confused:

1. **Compiler fusion decision**: combine Scale and Mask in one pass, or keep an
   intermediate on chip.
2. **Submission granularity**: submit many descriptors or one command list.
3. **Hardware fusion**: add a dedicated pipeline or macro-op connecting stages.

The current multi-descriptor baseline is simple and measurable, but is not
expected to be the highest-performance implementation. It pays repeated
CPU/runtime launch, descriptor fetch, synchronization, and intermediate SRAM
round-trip costs. It remains the correctness and PPA reference.

The next performance step keeps fusion policy in Compiler while reducing
submission and movement overhead:

```text
Compiler emits grouped command list and buffer-residency plan
  -> Runtime submits once
  -> thin Host Wrapper forwards launch
  -> Core Command Processor/Uop Scheduler execute commands
  -> intermediates remain in local storage where legal
```

Mask illustrates this distinction. Compiler fuses regular mask application
into the existing Score Scale pass and emits compact row-mask metadata.
Descriptor transport makes the metadata available to hardware. Wrapper does
not split or fuse Mask, and no standalone `MASK` ISA instruction is required.

当前多descriptor方式便于验证和统计PPA，但会产生重复launch、descriptor
读取、同步以及中间结果写回SRAM的开销，因此不是最终性能最优方案。性能
优化应由Compiler决定融合和数据驻留，再生成command list；Runtime一次
提交；NPU Core内部执行。Wrapper仍保持轻量。

A dedicated fused Attention pipeline is considered only after measured PPA
shows command-list execution and on-chip residency are insufficient. It may
improve performance and energy, but adds area, control complexity, and reduced
reuse. Fusion is evidence-driven, not automatically assigned to the Wrapper.

## Performance Strategy / 性能策略

Preferred evolution:

1. multi-descriptor baseline for correctness and transparent PPA;
2. Compiler-generated command list to remove repeated Runtime/CPU launches;
3. on-chip intermediate residency and cross-command pipelining;
4. scheduler macro-op expansion when instruction traffic dominates;
5. dedicated fused hardware only when measured benefit justifies area/power.

PPA must compare every optimized path with the baseline and separately expose
launch, descriptor, movement, compute, stall, and writeback cycles.

## Current Implementation Versus Target / 当前实现与目标差距

Current implementation:

- workload manifests are hand-authored; there is no general model/graph
  importer yet;
- `sw/tools/npu_compiler/attention.py` statically lowers the current fixed
  Attention shape and emits ordered runtime jobs;
- `sw/tools/npu_compiler/k_stream.py` plans K chunks for one output tile;
- generated firmware data and CPU Runtime submit descriptor jobs;
- Core Command Processor performs descriptor-local K-stream control;
- Host Wrapper remains a CPU-visible launch/status/perf boundary;
- there is no Compiler-generated M/N multi-tile command list or cross-stage
  on-chip residency yet.

当前已有静态Compiler雏形，但还不是完整模型编译器。当前workload manifest是
手写输入，Attention planner可以生成固定shape的任务序列，K-stream planner
可以拆分单个输出tile的K维；尚未实现通用计算图导入、M/N多tile lowering、
整组command list和跨算子片上驻留。

Therefore the architectural next step is to extend static Compiler output, not
to move graph lowering into Runtime or Wrapper.

### B0 non-matrix lowering / B0非矩阵lowering

The same ownership rule applies to `DESC_VECTOR_TILE_V1`:

```text
logical RMSNorm/RoPE/Residual/SwiGLU stage
  -> Compiler emits row/segment plan and primitive program
  -> Runtime binds input/output/state addresses
  -> Command Processor moves segment data
  -> Uop Scheduler issues Vector/Reduction/SFU primitives
```

Runtime must not decide how many segments a row has, when to combine RMSNorm
partial sums, or which SiLU approximation to use. Those are Compiler and spec
decisions. Hardware must not infer operator identity from buffer names; it
executes the descriptor and primitive program.

During bring-up, a partial executable state is allowed only when clearly
reported. For example, if NPU computes RMSNorm segment partial sums but CPU
combines them, the workload is not complete B0 and PPA must show the CPU gap.

## B0 Operator Lowering To Primitive Programs / B0算子到Primitive程序的编译

### Problem / 问题

B0 high-level operators are model semantics:

```text
RMSNorm, RoPE, Residual Add, SiLU, Gate Multiply
```

The NPU should not grow one RTL macro for each of these names. The correct
software/hardware split is:

```text
Compiler understands operator math and emits primitive programs
Runtime binds addresses and submits descriptors
RTL executes generic primitive uops
```

If operator lowering is left implicit in firmware or test fixtures, B0 cannot
be reused for shape changes and PPA cannot attribute work to actual engines.

### Compiler artifact

For every B0 non-matrix stage, the Compiler emits:

| Field | Meaning |
| --- | --- |
| `stage_id` | stable B0 stage name |
| `logical_op` | RMSNorm/RoPE/etc. |
| `input_buffers` | logical buffers consumed |
| `output_buffers` | logical buffers produced |
| `segment_plan` | rows, row width, segment width, segment count, valid masks |
| `primitive_program` | ordered uops consumed by the common Scheduler |
| `constants` | RoPE tables, shifts, clamp bounds, SiLU coefficients |
| `row_state` | scalar temporaries such as `sumsq` and `inv_rms` |
| `golden_contract` | fixed-point model name and tolerance |
| `ppa_theory` | useful lane ops and theoretical cycles |

The generated firmware/header is only a serialization of this compiler
artifact. It must not invent the primitive program.

### RMSNorm lowering

For one row of width `H=16`:

```text
for segment in 0..1:
  REDUCE_SUMSQ row, segment -> partial_sumsq[segment]
REDUCE_SUM partial_sumsq[0:2] -> row_sumsq
SFU_RSQRT row_sumsq -> inv_rms
for segment in 0..1:
  VEC_SCALE row_segment by inv_rms -> output_segment
```

Bring-up may implement `partial0 + partial1` using a small scalar vector row
and `REDUCE_SUM`, so it still uses existing Reduction/SFU primitives. Complete
B0 requires all four steps to be in the measured NPU path.

### Residual Add lowering

For `H=16`:

```text
for segment in 0..1:
  VEC_ADD input_segment, residual_segment -> output_segment
```

This is the first simple `DESC_VECTOR_TILE_V1` acceptance test because it needs no
whole-row state.

Compiler emission for the first RTL carrier test is:

```text
program = [VADD row=0 segment=0, HALT]
descriptor.op_type = DESC_VECTOR_TILE_V1
input0 = first 8-lane 32-bit segment
input1 = second 8-lane 32-bit segment
output = 8-lane 32-bit segment
```

This is intentionally not a `RESIDUAL_ADD` ISA instruction. The stage name
remains a compiler/PPA label; the hardware sees only `DESC_VECTOR_TILE_V1` plus
primitive uops.

Layer naming rule:

```text
logical operator:     Residual Add
descriptor/job type:  desc_vector_tile_v1 / DESC_VECTOR_TILE_V1
primitive program:    VADD, HALT
engine-local op:      OP_VEC_ADD
```

Software artifacts should keep these names separate so a PPA page or test
failure immediately shows whether the issue is operator lowering, descriptor
transport, Scheduler ISA decode, or Vector Engine execution.

### RoPE lowering

RoPE is pairwise rotation:

```text
y0 = round(x0 * cos - x1 * sin)
y1 = round(x0 * sin + x1 * cos)
```

The compiler owns the sin/cos table and fixed-point scale. The first hardware
implementation may use a reviewed primitive expansion such as:

```text
VEC_MUL x_even, cos
VEC_MUL x_odd, sin
VEC_SUB products -> y_even
VEC_MUL x_even, sin
VEC_MUL x_odd, cos
VEC_ADD products -> y_odd
VEC_REQUANT rotated values
```

If the current primitive set cannot express even/odd lane packing without
extra swizzle support, the Compiler must mark RoPE `planned_not_executable`
until a reviewed `VEC_PERMUTE` or pairwise-rotate primitive is added. It must
not silently compute RoPE in CPU for an accepted B0 run.

### SwiGLU lowering

SwiGLU in B0 is:

```text
gated = SiLU(gate) * up
```

`gate` and `up` are `S=8, FFN=32`, so each row has four vector segments.

First accepted partial lowering:

```text
for segment in 0..3:
  input0 = compiler-golden silu_segment
  input1 = up_segment
  VEC_MUL silu_segment, up_segment -> product
  VEC_REQUANT arg1=INT8_SHIFT4_CLAMP -> gated_segment
```

This makes `gate_mul_up` executable and measurable, but it does not make the
preceding `silu_gate` stage executable. Complete B0 still requires:

```text
SFU/Vector approximation for SiLU(gate_segment) -> silu_segment
```

SiLU approximation must be named in the numerical contract. A coarse LUT or
piecewise-linear approximation is acceptable for bring-up only after its error
and PPA cost are documented.

### Compiler legality

The Compiler must reject or mark `planned_not_executable` when:

- row width cannot be represented by 8-lane segments;
- required primitive opcodes are not available in the target RTL variant;
- RoPE needs an unsupported lane swizzle;
- SiLU approximation is not reviewed;
- RMSNorm row-state accumulation would be CPU-materialized in a supposedly
  executable B0 run.

### First coding order

1. complete: residual add, because it proves generic binary vector segment routing;
2. complete: RMSNorm segmented row-state through repeated per-segment row reduction;
3. complete: gate multiply without SiLU, as a second binary vector/requant check;
4. RoPE lane-pair support;
5. SiLU approximation;
6. full B0 connection and PPA.

## Current Executable State

The SoC can already measure three attention-related stage jobs:

| Stage | Executable op | Primitive RTL used | Status |
| --- | --- | --- | --- |
| QK | `matmul_k_stream` | matrix array | measured as int8 x int8 QK |
| Softmax | `attention_softmax_v1` | vector + reduction + SFU | measured as current simplified Q0.15 softmax |
| PV | `matmul_u16s8_q15` | shared matrix mixed mode | measured as Q0.15 probability x int8 value |

The full attention parent workload is still not a single measured runtime
descriptor execution. The current CPU firmware does consume a compiler-produced
runtime-job table for QK, softmax, and PV, so the group can be treated as
software-sequenced measured stages. Scale/mask remains materialized by fixture
data.

## Target Software Flow

```text
operator metadata
  -> compiler lowers scaled_dot_product_attention_v1
  -> compiler emits AttentionPlan
  -> firmware data generator emits tensors, buffers, programs, runtime jobs
  -> CPU runtime launches generated jobs in order
  -> wrapper/core execute shared primitive RTL
  -> perf/PPA reports stage and group results
```

The fixture generator should not be the owner of attention execution semantics.
It should become a consumer of compiler output and a producer of deterministic
test tensors/golden data.

## Attention Formula Mapping

Mathematical attention:

```text
O = softmax((Q * K^T) / sqrt(D_k) + mask) * V
```

Software/primitive mapping:

| Formula part | Operator | Current execution |
| --- | --- | --- |
| `Q * K^T` | `matmul_s8s8_i32_tile` | measured QK descriptor |
| `/ sqrt(D_k)` | `attention_score_scale_mask_v1` scale policy | currently metadata/pre-materialized bridge |
| `+ mask` | `attention_score_scale_mask_v1` mask policy | currently none/pre-materialized bridge |
| `softmax(...)` | `attention_softmax_q15_v1` | measured simplified row/tile descriptor |
| `P * V` | `matmul_u16s8_q15_i32_tile` | measured mixed matrix descriptor |

The missing software work is not another RTL attention block. The missing work
is making these boundaries explicit in operators, compiler output, runtime
buffers, generated firmware launch data, and PPA provenance.

## Near-Term Coding Order After Review

1. Add `sw/npu_core/operators/transformer_attention_v1.json`.
2. Add an attention compiler planner under `sw/tools/npu_compiler`.
3. Change Transformer fixture generation to consume the compiler plan.
4. Add generated runtime-job metadata for QK, softmax, and PV.
5. Move firmware smoke toward table-driven runtime launch.
6. Promote the parent attention PPA row from model-only to software-grouped only
   after generated runtime launches the full sequence.

## Non-Goals For The Next Patch

- no new monolithic attention RTL module;
- no command-list scheduler RTL yet;
- no claim that scale/mask is measured NPU compute until it has an executable
  stage;
- no silent numerical-contract change for softmax or PV rounding;
- no removal of existing CNN/MNIST regression paths.
