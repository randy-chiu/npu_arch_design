# Transformer NPU v1 Next Steps

## Goal

Build an iteratively measurable NPU architecture that can eventually satisfy
the performance, power, and area requirements of representative LLM
inference. Transformer Attention is the current architecture driver, but it is
one part of the target workload rather than the final product or a dedicated
RTL macro. The implementation target remains a compiler/runtime scheduled
sequence over matrix, vector, reduction, SFU, data mover, scheduler, and
memory-system primitives.

最终目标是构建一个面向典型LLM推理、能够持续进行架构探索并以PPA数据驱动
取舍的NPU，而不是仅仅让固定`8x8` Attention用例跑通。Attention是当前阶段
用于建立编译、调度、数据通路和PPA闭环的主要载体；后续架构选择还必须同时
面对Prefill、Decode、FFN、Norm、KV-cache和系统存储流量。

The immediate architectural question is:

```text
Which bottleneck prevents the current architecture from meeting a
representative LLM inference target, what candidate mechanism addresses it,
and does measured performance plus credible area/power evidence prove that the
candidate is better than the retained baseline?
```

## North-Star Architecture Exploration Contract / 总体架构探索约束

### Current distance from the final goal / 当前差距

The project already has a useful exploration foundation:

- executable RTL, Compiler/Runtime/Firmware integration, and golden checking;
- CNN regression plus end-to-end causal `S=8,D=8` Attention;
- measured cycle/event timelines exposing module order, overlap, and waits;
- Level-0 structural area and event-energy models with baseline comparison.

This can detect functional regressions and several performance bottlenecks. It
cannot yet prove an LLM inference NPU or support final PPA choices:

| Dimension | Current evidence | Gap to the final goal |
| --- | --- | --- |
| Workloads | tiny Attention stages, matrix micro-workloads, CNN regression, model-only decode/KV views | executable decoder block, realistic Prefill/Decode shapes, FFN/RMSNorm/residual, multi-head/multi-layer traces |
| Performance | RTL-measured cycles and module timelines | model/block latency, tokens/s, sustained utilization, bandwidth limits, runtime and external-memory inclusion |
| Area | Level-0 structural normalized model | synthesized/mapped module area, timing closure, SRAM/macro policy, later physical trend |
| Power/energy | Level-0 event-coefficient model | activity-driven on-chip power, credible external-memory energy, energy/token |
| Comparison | one frozen serial K-stream baseline and local comparisons | repeatable candidate matrix across a representative workload suite with regression limits |
| Scalability | correct fixed `8x8` causal Attention | large/tail shapes, multi-tile reduction, buffer allocation, command lists, executable decode/KV |

当前工程已经能发现功能错误、模块工作顺序错误和部分cycle瓶颈，但仍不能证明
某个架构能够满足典型LLM推理PPA。尤其面积和功耗仍属于L0模型，workload也
以微型用例为主。后续不能把“某个小算子cycle下降”直接等同于“LLM架构更优”。

### Mandatory iteration loop / 每轮迭代必须遵守的闭环

Every architecture-changing package must use this sequence:

```text
representative workload or trace
  -> measured baseline and identified bottleneck
  -> architecture hypothesis and spec-first design
  -> Compiler/Runtime/RTL implementation
  -> functional and compatibility regression
  -> candidate-versus-baseline PPA report
  -> accept, revise, or reject the candidate
```

Each candidate design must state:

1. the measured problem and affected LLM scenario;
2. why software scheduling or current hardware is inadequate;
3. the proposed mechanism and required resources;
4. expected performance benefit and expected area/power cost;
5. metrics, workloads, baseline identity, and pass/fail threshold;
6. a retained reference path or other valid before/after comparison.

No architecture feature is complete merely because its RTL works. It is
complete only when the report explains whether measured behavior matches
theory, where remaining cycles/traffic are spent, and whether the PPA tradeoff
is accepted.

每次架构修改必须从可量化的问题开始，并保留修改前基线。功能验证通过只是
必要条件；还必须回答实际性能是否符合理论、瓶颈是否真的消除、增加的面积和
功耗是否值得。没有对比数据的硬件功能不能作为架构优化结论。

### Required LLM decision workloads / 架构决策工作负载

Architecture decisions must eventually pass a common suite rather than one
favorable operator:

| Suite level | Required scenarios | Primary decision metrics |
| --- | --- | --- |
| Primitive | GEMM, GEMV/skinny GEMM, vector, reduction, SFU, movement | active/stall cycles, utilization, bandwidth, energy/event |
| Operator/group | QKV projection, causal Attention, FFN, RMSNorm/residual, KV traffic | latency, movement, intermediate residency, useful work/waste |
| Decoder block | one executable tiny decoder block in Prefill and Decode | block latency, cycles/token, bytes/token, engine balance |
| Trace/model view | representative sequence/context/hidden-size traces | estimated/measured tokens/s, energy/token, capacity and bandwidth pressure |

MNIST/CNN remains a compatibility regression, not an optimization target.
Every candidate may improve its target LLM scenario, but must report
regressions across the common suite and explain any accepted loss.

### Hardware-first scope control / 硬件优先的范围控制

The project is an NPU architecture exploration project, not a general-purpose
compiler/runtime product. Software is implemented only when it is needed to:

- generate representative and repeatable hardware workloads;
- express tiling, scheduling, fusion, and data placement candidates;
- submit those candidates without shape-specific firmware distortion;
- produce golden results and trustworthy PPA attribution.

Near-term work explicitly excludes a general graph importer, dynamic memory
allocator, broad operator framework, and production runtime APIs unless a
measured hardware experiment requires them. The main output of each iteration
is a hardware architecture conclusion, not software feature count.

本项目以NPU硬件架构探索为核心。Compiler/planner和轻量submitter只用于生成
代表性硬件执行流、避免固件硬编码干扰测试、并获得可信PPA；不以建设大而全
的软件栈为近期目标。每轮工作的最终产出必须是硬件瓶颈和架构取舍结论。

### PPA evidence policy / PPA证据策略

- `rtl_workload_view` remains the fast inner loop: RTL-measured cycles/traffic plus
  structural area and event-energy models.
- `mapped_area_timing_view` is an early feasibility gate, not final polish. It
  must precede acceptance of major datapath widening, new storage ports,
  larger arrays, or fusion structures.
- `activity_power_view` evidence is required before making power/energy
  conclusions or selecting among candidates with similar performance.
- External-memory assumptions, process/library, clock target, SRAM accounting,
  and activity scope must be explicit; incomparable reports must not publish a
  numerical winner.

The exact final PPA target requires a declared product envelope: model
class/size, context length, batch, latency or tokens/s target, power budget,
area/process target, and memory system. Until that envelope is fixed, the
project optimizes transparent trend metrics and does not claim product-level
sufficiency.

最终PPA目标需要明确模型规模、上下文长度、batch、吞吐/时延目标、功耗预算、
芯片面积/工艺和外部存储条件。在这些条件确定前，报告用于比较架构趋势，不
声称已经满足产品级指标。

## Current Attention Validation Summary

### Legacy Phase-0 Softmax retirement

The old `op=0` Softmax path duplicated Attention Softmax with immediate
whole-vector RTL tasks and separate X/Y windows. It did not represent the
vector/reduction/SFU pipeline and no model depended on it. That path has been
removed while retaining `op=0` MatMul for MNIST/CNN and the shared Softmax
primitive opcodes used by the common Scheduler.

Acceptance:

- no old whole-vector Softmax tasks, X/Y fixture, firmware job, or descriptor
  op type remains;
- Attention Softmax numerical and PPA regressions pass;
- digits and real-MNIST CNN CPU-to-NPU RTL regressions pass.

The current SoC path has passed stage-level end-to-end validation for the basic
attention building blocks:

```text
QK score:     Q_s8 * K_t_s8          -> int32 score tile
Softmax row:  scaled/masked score    -> Q0.15 probability row/tile
PV:           P_q15 * V_s8           -> int32 output tile
```

This is enough evidence that the basic attention primitives can execute through
the CPU-to-NPU descriptor path. It is also now enough evidence that the current
compiler/runtime path can generate a software-sequenced attention stage group
for the fixed `S=8,D=8` bring-up case.

Current boundary:

- QK, causal scale/mask, masked softmax, and PV are launched and measured as
  separate descriptor jobs,
  but the firmware launch order now comes from a compiler-generated runtime-job
  table rather than direct hand-written stage calls.
- QK output SRAM plus a packed row-mask table feed an executable causal
  scale/mask descriptor, the
  produced scaled-score tile feeds softmax, and the produced probability tile
  feeds PV for the fixed `S=8,D=8` case.
- The parent `attention_prefill_s8_d8` workload is now
  `software_group_measured_stages` for QK/scale-mask/softmax/PV stage execution, not a
  command-list or single-descriptor measured full-attention operation.

Therefore the next software task is to remove stage-specific firmware
descriptor filling and fixed `S=8,D=8` assumptions, then reduce the launch and
external-memory boundaries between the already connected stages.

## Command-Processor Trace Contract

### Implemented semantic-event design

P0 replaced the report dependency on private `cmd_state` values with stable
architectural event IDs generated from `arch/configs/npu_v0.jsonc`:

```text
event_id: DESC_DECODE | PROGRAM_MOVE | INPUT_MOVE | ACC_CLEAR |
          CHUNK_LAUNCH | PREFETCH_BANK_SELECT | OUTPUT_MOVE | JOB_RETIRE
arguments: chunk_id, source_bank, destination_bank, operator_id
cycle, active, wait_reason
```

The RTL command processor, Scheduler, and Compute cluster emit the events they
are actually performing. The testbench records them and PPA renders them
without reading private FSM encodings. Strict report validation rejects
Scheduler active/wait contradictions, typed-wait mismatches, out-of-range
spans, and Attention Compute-cluster parent/child cycle mismatches.

Acceptance gates:

- renumbering an internal FSM state does not change PPA labels;
- a new job using existing events needs no workload-specific timeline code;
- a new architectural action requires an explicit event definition and test;
- every displayed command action can be traced to a measured RTL event.

This contract is now the baseline prerequisite for later command-list, fusion,
and generalized-shape timelines.

## Industry Comparison And Next-Phase Direction

The useful lessons from TPU and NVDLA are architectural patterns, not a request
to copy either design:

| Reference pattern | Relevant lesson | Current project implication |
| --- | --- | --- |
| TPU matrix unit plus software-managed on-chip buffer and deterministic instruction schedule | regular compute is effective only when data placement and scheduling keep it fed | do not enlarge the Matrix engine before measuring scratchpad bandwidth, bank conflicts, and wait reasons |
| NVDLA independent engines, per-engine configuration, and fused mode through small FIFOs | adjacent engines can avoid external-memory round trips while preserving modular ownership | connect matrix/vector/reduction/SFU through explicit dependency and valid/ready contracts; keep a non-fused fallback |
| NVDLA ping-pong configuration and data buffering | configuration and next-tile movement can overlap current execution | add command/config double buffering and measured overlap only after the event contract is stable |
| NVDLA per-engine stall and latency counters | optimization decisions require bottleneck evidence at engine boundaries | report data wait, engine wait, bank conflict, and output backpressure separately |

The current fixed attention path proves functional composition, but it still
resembles independent-mode execution: firmware launches multiple descriptors
and intermediate tiles cross externally visible SRAM boundaries. The next
phase should improve scheduling and data residency before adding a dedicated
attention macro or a larger Matrix engine.

## Recommended Next-Phase Plan

Each package below first states the problem being solved, then the mechanism
and its acceptance gate.

### Functional-baseline-first decision / 功能基线优先决策

Architecture optimization candidates are deferred until the NPU executes a
recognizable complete tiny Decoder Block, then two chained blocks, with all
major operator work attributed honestly to NPU RTL. Optimizing the current
isolated Attention path first could select the wrong bottleneck because a real
block also contains projection GEMMs, RMSNorm, residual operations, FFN
activation/gating, output movement, and repeated block boundaries.

架构优化候选暂缓。当前首要目标是让NPU真实执行一个完整tiny Decoder Block，
随后串联执行两个Block。只有完整Block的PPA才能判断主要瓶颈究竟来自矩阵计算、
Softmax、Norm、FFN、数据搬运还是任务边界；仅根据当前Attention微用例优化，
可能会优化错误的对象。

For this milestone, "complete block" means a LLaMA-like prefill block:

```text
input
  -> RMSNorm
  -> Q/K/V projections
  -> position transform (RoPE or an explicitly measured NPU fallback)
  -> causal Attention: QK -> Scale/Mask -> Softmax -> PV
  -> Attention output projection
  -> residual add
  -> RMSNorm
  -> FFN gate/up projections
  -> activation and gate multiply
  -> FFN down projection
  -> residual add
  -> block output
```

No CPU-computed operator may be hidden inside an executable block result.
Model-only stages may remain visible as gaps during development, but the block
is not accepted as executable until every listed stage has measured NPU RTL
provenance. Numerical behavior may use documented bring-up fixed-point
contracts; it must still match the corresponding end-to-end golden.

The first two functional milestones use a scaled TinyLlama-derived structure.
The dimensions are deliberately small, but the operator topology retains the
important LLaMA-family behavior: RMSNorm, RoPE, causal Attention, GQA, SwiGLU,
and residual connections.

首个完整Block工作负载参考TinyLlama/LLaMA结构。尺寸缩小是为了尽快闭合RTL
功能路径，而不是删除关键模型语义；B0仍保留RMSNorm、RoPE、因果Attention、
GQA、SwiGLU和两次Residual Add。

| Milestone | Shape/scope | Purpose | Acceptance |
| --- | --- | --- | --- |
| B0 one-block bring-up | `S=8, H=16, Q_heads=2, KV_heads=1, head_dim=8, FFN=32`, one Prefill block | close a complete TinyLlama-derived block while forcing useful projection/FFN tiling | one NPU-launched block plan matches golden; every stage has measured RTL cycles and visible buffers/movement |
| B1 two-block representative baseline | two chained B0-shape blocks with distinct weights | expose repeated-block scheduling, movement, storage lifetime, and resource balance | block 0 output feeds block 1 input without CPU recomputation or fixture replacement; grouped PPA reports per-stage/per-block/full-run totals |

The accepted B0/B1 workload contract is:

- `Q_heads=2, KV_heads=1` preserves GQA: both query heads share one K/V head;
- Q/K/V are separate projection jobs first; fused QKV is a later measured
  optimization candidate;
- heads execute sequentially first; concurrent-head execution is a later
  candidate selected from block PPA;
- deterministic synthetic INT8 inputs and weights are used first so fixed-point
  failures are reproducible; importing real model weights is a later accuracy
  milestone;
- embeddings, tokenizer, sampling, and LM head are outside the block workload;
- Compiler/planner owns all `8x8x8` M/N/K tiling; Runtime only binds buffers and
  submits the generated jobs.

B0 may use small dimensions to close functionality, but it is not architecture
optimization evidence. B1 and later representative shapes become the first
valid surface for selecting hardware optimization candidates.

| Priority | Problem | Mechanism | Acceptance gate |
| --- | --- | --- | --- |
| P0 | PPA timelines can contain raw-FSM labels, residual control buckets, out-of-range terminal events, irrelevant empty lanes, and missing conservation checks | stable semantic command/compute-control/engine/wait events, one job interval contract, timeline conservation validator, and report contract tests | every displayed span is measured and in range; compute-cluster child spans reconcile with parent activity; FSM renumbering cannot alter report meaning |
| P1 | Expanded Softmax and Scale/Mask now use the common Scheduler, but the in-order start/done integration and large expanded program expose control and movement overhead | add typed wait reasons and valid-ready commands; measure expanded program against a generic loop candidate; reduce routing handoff cycles | both jobs show real Scheduler/engine spans, no operator-specific Compute-cluster sequence FSM, functional outputs unchanged, and before/after control/movement costs measured |
| P2 | Some current one-cycle storage operations were visible in RTL/PPA but their required parallel hardware was not an explicit architecture contract | performance-first local-storage contract declaring accumulator, Attention-row, and Matrix-feed lanes/buses/latencies; RTL elaboration checks and PPA transaction conservation | every one-cycle operation names the required parallel resource; RTL dimensions match the contract; PPA rejects duration and overlap violations |
| P3 | The NPU executes Attention stages but not a complete Decoder Block, so current bottleneck conclusions are incomplete | complete one tiny LLaMA-like Prefill block, chain two blocks, and expose honest block-level PPA | every major block operator executes in NPU RTL, matches golden, and appears in per-stage/per-block/full-run PPA |
| P4 | Separate descriptor launches and SRAM-visible intermediate boundaries add control and movement overhead | grouped command list, dependency tokens, on-chip score/probability tile residency, and optional matrix-to-vector/reduction/SFU streaming | full-attention group cycles include a stated runtime policy and demonstrate reduced launch/movement cycles against the unfused path |
| P5 | Future storage sharing and widening can introduce unreported port conflicts | explicit bank ownership, allocator rules, double buffering, bank-conflict and wait-reason counters | timeline explains every compute idle interval using measured data, dependency, conflict, or backpressure reasons |
| P6 | Softmax remains a bring-up numerical path and limits correctness claims | target EXP/RECIP implementation, reviewed scale/requant widths, saturation and error bounds | masked attention output meets documented tolerance over directed and randomized cases |
| P7 | Decode behavior and KV traffic are not executable architecture evidence | decode workload suite first, then KV-cache streamer and GEMV/skinny-GEMM changes only where measured evidence justifies them | tokens/s, bytes/token, utilization, and energy/token compare baseline and proposed decode paths |
| P8 | RTL workload performance and normalized coefficients cannot validate physical feasibility or final PPA tradeoffs | introduce mapped area/timing as an early gate for substantial resource changes, then add per-engine activity-driven power | architecture decisions correlate clearly identified workload, mapped area/timing, and activity-power views of the same variant |

P0 through P2 are complete and P3 remains the active
functional/scalability package. P8 is a parallel evidence track and must begin
before P4 fusion or any larger-array/storage-port decision is accepted. P4
fusion must retain an unfused reference path so its benefit and additional
control/storage cost can be measured rather than assumed.

### Revised Near-Term Execution Order / 修订后的近期执行顺序

The immediate work proceeds on two coordinated tracks:

| Order | Track | Deliverable | Why it is required now |
| --- | --- | --- | --- |
| 1 | Decision framework | freeze an initial LLM decision suite and report scorecard covering causal Attention, projection/FFN-shaped GEMM, decode skinny GEMM/KV traffic, and CNN regression | prevents optimization against only the current favorable `8x8` case |
| 2 | P3b complete one-block baseline | implement the missing projection, RMSNorm, residual, FFN gate multiply, RoPE, SiLU, Attention composition, and block-plan execution path using existing shared engines where possible | creates the first honest full-block bottleneck profile |
| 3 | P8a physical feasibility | executable mapped area/timing view for the current retained baseline, with module hierarchy and SRAM-accounting policy | establishes a physical cost baseline before fusion, wider ports, or larger arrays |
| 4 | P3c two-block and shape expansion | chain two complete blocks, then add multi-tile dimensions and a first Decode path | supplies representative measured evidence for architecture choices |
| 5 | Candidate optimization | Softmax parallelism, residency, command-list, buffering, fusion, or datapath candidates selected from measured block bottlenecks | ensures mechanisms are evidence-driven rather than roadmap-driven |
| 6 | P8b power evidence | activity-power view for shortlisted variants | supports energy/token and final candidate selection |

The next coding package is redefined as P3b complete-block functionality.
Software changes remain limited to generating and submitting the block's
hardware-relevant schedule. P8a remains a parallel evidence task, but no
hardware optimization candidate is selected until B0/B1 PPA exists.

下一步P3b改为先完成完整Block功能基线。软件只生成并提交必要的硬件执行流，
不扩展为通用软件平台。P8a可并行建立mapped area/timing基线，但在B0/B1完整
Block PPA出现前，不选择Softmax并行、融合、扩大阵列或宽存储端口等优化方案。

P3b and P8a do not modify the same architectural responsibility:

- P3b changes the test Compiler/planner, minimal submit path, and required RTL
  behavior for correct partial-tile execution, then measures workload behavior
  in the RTL workload view;
- P8a builds a repeatable synthesis/mapping flow and first records the retained
  pre-P3b/current baseline; it can then map the post-P3b candidate;
- RTL workload and mapped area/timing results are associated only when they
  name the same architecture variant and RTL/config revision.

They may progress in parallel at the tooling level. P8a must not block P3b
correctness work, while P3b must not silently overwrite the L1 baseline
identity.

### Active Package: P3 Mask And Shape Generalization

P0 semantic-event instrumentation, P1 common-Scheduler integration, and P2
performance-first resource contracts are complete. The active package is P3.

Problem being solved:

- the executable AttentionPlan silently assumes every lane in one `8x8` tile
  is valid;
- causal prefill, padding, and tile tails all require invalid lanes to be
  excluded from scale/mask, row max, row sum, probability output, and PV;
- adding larger shapes before defining this contract would make compiler,
  runtime, RTL, golden, and PPA disagree about which work is useful.

Accepted first-step design:

1. Compiler lowering converts logical mask policy into one
   `valid_lane_mask` bit vector per physical query-tile row. Bit `j=1` means
   key lane `j` is visible to that row; rows beyond logical `seq_q` have a zero
   mask and are excluded by `valid_query_mask`.
2. `causal`, `padding`, and `tile_tail` are compiler-side rules that compose
   into the same row-mask representation; they are not separate hardware
   engines.
3. Current hardware tile width remains eight lanes. Logical `seq_q` and
   `seq_k` up to eight may be represented in a plan; larger shapes are
   explicitly rejected until multi-tile lowering exists.
4. Full physical eight-row plans with at least one valid lane per row are
   executable, including causal `S=8,D=8`. Plans containing all-invalid
   physical rows remain `planned_not_executable` until tail-row scheduling is
   implemented.
5. Golden behavior must prove invalid lanes cannot affect row max, row sum,
   probability, or PV output before masked RTL is enabled.

P3 implementation order:

1. add the canonical mask/shape fields and legality rules to compiler design
   and AttentionPlan validation;
2. emit row `valid_lane_mask` values for none, causal, padding, and tile-tail
   cases and add directed golden tests;
3. specify the descriptor-referenced row-mask table and implicit row-indexed
   uop selection, then implement Scale/Mask, Reduction, and Softmax RTL
   consumption;
4. make descriptor filling buffer-driven and add executable masked/tail
   fixtures;
5. generalize to multi-tile shapes only after the single-tile mask contract
   passes end-to-end.

First-step acceptance:

- mask policy and every row mask are explicit in AttentionPlan;
- causal, padding, and tile-tail composition is covered by directed tests;
- unsupported executable shapes and masked plans fail or remain explicitly
  non-executable instead of silently using the unmasked fixed case;
- existing unmasked Transformer and CNN execution remains unchanged.

Current P3 progress:

- Compiler emits `valid_query_mask` and one `valid_lane_mask` per physical
  query-tile row for `none`, `causal`, `padding`, and `causal_padding`.
- AttentionPlan schema recomputes the expected masks and rejects inconsistent
  metadata, shapes larger than one tile, and masked/tail plans incorrectly
  marked executable.
- Firmware-data emission rejects `planned_not_executable` plans before
  producing a runtime-job table.
- Golden coverage composes causal, padding, and tail visibility rules.
- Causal `S=8,D=8` Transformer execution, Transformer PPA, full CNN execution,
  and the complete unit/RTL regression pass.

Next P3 action:

- finish hardware-visible malformed-descriptor/all-invalid-row reporting;
- make tail physical rows executable through valid-row-aware scheduling;
- begin P3b buffer-driven runtime and edge-tile work before P3c multi-tile
  `S=16,D=16`.

P3 mask implementation decision:

Status: P3a causal `S=8,D=8` is implemented end to end. Descriptor/uop specs
select a packed descriptor-referenced row-mask table and no new `MASK` uop.
Tail/all-invalid physical rows and hardware error status remain pending.

- Masking remains Compiler-planned but is consumed in the NPU execution path.
  This does not require a separate Mask engine.
- Regular causal/tail/padding cases use compact row metadata or a generated
  lane-valid predicate. Scale/Mask writes `SCORE_NEG_INF` for invalid lanes;
  Reduction excludes invalid lanes from max/sum; Softmax emits zero
  probability for them.
- The expected hardware cost is lane-valid compare/gating and mask metadata
  transport. The benefit is avoiding a CPU pass over the score tile, avoiding
  an additional score-matrix SRAM read/write round trip, and allowing fused or
  resident-score execution later.
- A software-only fallback may materialize `SCORE_NEG_INF` into the score
  matrix before Softmax, but it must be reported as CPU/materialization work
  and is not the performance target.
- Arbitrary dense masks are deferred. They require mask-tensor movement and
  may not be worthwhile for this small baseline until a workload requires
  them.

### P3 Complete-Block Functional Roadmap / 完整Block功能路线

The earlier P3b Softmax row-throughput candidates remain documented below as
future optimization candidates, but they are no longer the active coding
package. First complete the block and use its PPA to decide whether Softmax
parallelism is actually the highest-value change.

P3b complete-block implementation order:

1. **Freeze B0 block operator and numerical contract**
   - use `S=8, H=16, Q_heads=2, KV_heads=1, head_dim=8, FFN=32`;
   - define every stage, dtype, scale, buffer, and golden boundary;
   - use deterministic synthetic INT8 weights and a reviewed fixed-point RoPE
     table for the bring-up workload;
   - reject any CPU-hidden stage in the executable block.
2. **Reuse existing Matrix path for all projections**
   - Q/K/V projection;
   - Attention output projection;
   - FFN gate/up/down projections;
   - first use tile-compatible shapes, then reuse Compiler tiling for B1.
3. **Make shared non-matrix primitives executable**
   - RMSNorm through Reduction `SUMSQ`, SFU `RSQRT`, Vector scale/multiply;
   - Residual Add through Vector Add;
   - FFN activation and gate multiply through reviewed Vector/SFU primitives;
   - RoPE through reviewed Vector primitive sequence or a clearly named
     temporary NPU implementation.
4. **Generate and submit one complete block plan**
   - Compiler/planner emits ordered descriptors/programs/buffers;
   - thin submitter launches them without operator-specific computation;
   - produced intermediate/output buffers feed their real consumers.
5. **Chain two blocks**
   - block 1 output becomes block 2 input;
   - no CPU recomputation or fixture replacement at the boundary.
6. **Publish block-level PPA**
   - graph view for every block stage;
   - theoretical versus measured operator cycles;
   - per-engine timeline and movement;
   - per-stage, per-block, and two-block totals;
   - measured bottleneck ranking that selects the first hardware optimization.

P3b B0 acceptance:

- all listed block stages have measured NPU RTL provenance;
- complete block output matches the documented fixed-point golden;
- PPA contains no unexplained or model-only stage inside the accepted block;
- existing Attention and CNN regressions remain unchanged;
- missing performance is reported but not optimized in the same package.

Current P3b implementation status:

- B0/B1 are frozen as the TinyLlama-derived shape
  `S=8,H=16,Q_heads=2,KV_heads=1,head_dim=8,FFN=32`;
- B0/B1 workload JSONC declarations are now contract-checked against the
  compiler planner. Shape, topology summary, planner name, execution state, and
  B1 block-boundary metadata must match generated BlockPlan data; this prevents
  workload declarations and compiler lowering from drifting apart without
  introducing a general graph parser.
- the Compiler emits an 18-stage BlockPlan with explicit inputs, outputs,
  provenance, and execution state;
- generic M/N/K tiling now emits one K-stream job per output tile, including
  zero-filled boundary-tile metadata;
- the seven matrix stages in one B0 Block lower to 16 output-tile descriptor
  jobs, 36 physical tile invocations, and 288 theoretical Matrix-active
  cycles;
- deterministic fixed-point golden carries the real Block 0 output into
  Block 1 without CPU recomputation or fixture replacement;
- B0/B1 remain `planned_not_executable`. RoPE, SiLU, and complete block-level
  submission still need
  measured RTL paths. This state is intentional and must not be presented as
  completed block execution.
- B0 matrix subgraph is now executable through the CPU-to-NPU descriptor path:
  16 `MATMUL_K_STREAM` descriptor jobs cover Q/K/V, output projection, and
  FFN gate/up/down matrices. Firmware submits every tile job and checks RTL
  output against the Compiler golden.
- Measured B0 matrix-subgraph PPA:
  `effective_mac_ops=18432`, `physical_tile_invocations=36`,
  `theoretical_matrix_cycles=288`, `measured_matrix_cycles=288`,
  `total_cycles=2008`, `data_mover_active_cycles=1472`,
  `matrix_utilization=1.0`, `end_to_end_efficiency=0.143426`.
  This proves the Matrix Engine math is fully utilized for these tile shapes,
  but descriptor/data-movement boundaries dominate elapsed cycles.
- Complete B0 remains incomplete. The remaining non-matrix stages still
  require hardware-visible primitive sequences or composition paths; Python
  golden intermediates are not substituted into an accepted complete-block
  workload.
- The first `DESC_VECTOR_TILE_V1 + VADD` carrier slice is now implemented and
  measured as `operator_smoke_vector_tile_vadd`. It proves descriptor-driven
  program movement, two 32-bit vector input segments, Scheduler primitive
  issue, Vector Engine execution, and C-window output writeback. Measured
  transformer-profile smoke result: `total_cycles=30`, `core.total=4`,
  `data_mover.active_cycles=10`, `uop_scheduler.active_cycles=4`, and the
  PPA timeline labels the Vector Engine span as `Vector add segment: src0 + src1`.
- The Compiler BlockPlan now lowers `residual_attn` and `residual_ffn` into
  `desc_vector_tile_v1` segment jobs: each `H=16` residual row becomes two
  eight-lane `VADD + HALT` primitive programs.
- B0 residual vector subgraph is now executable through the CPU-to-NPU
  descriptor path: 32 `DESC_VECTOR_TILE_V1` jobs cover the two residual adds
  across `S=8,H=16`. Firmware submits every segment job and checks RTL output
  against the Compiler golden.
- Measured B0 residual-vector subgraph PPA:
  `effective_vector_lane_ops=256`, `theoretical_vector_cycles=32`,
  `measured_vector_cycles=32`, `total_cycles=960`,
  `data_mover_active_cycles=320`, `compute_efficiency=1.0`,
  `end_to_end_efficiency=0.033333`.
  This proves the Vector Engine primitive math is correct and fully utilized
  for each segment, but the current per-segment descriptor boundary dominates
  elapsed cycles.
- RMSNorm stages now execute through the CPU-to-NPU descriptor path using the
  `DESC_VECTOR_TILE_V1` arg1-mode primitive extension. `rmsnorm_attn` and
  `rmsnorm_ffn` lower to 32 descriptor jobs: each output segment job loads both
  `H=16` row segments, executes `SUMSQ_SRC0 + SUMSQ_SRC1 + RSQRT + SCALE_SRCx`,
  and checks RTL output against the Compiler golden.
- Measured B0 RMSNorm-vector subgraph PPA:
  `effective_vector_lane_ops=256`, `theoretical_reduction_cycles=64`,
  `theoretical_sfu_cycles=32`, `theoretical_vector_cycles=32`,
  `measured_compute_cycles=128`, `total_cycles=1344`,
  `data_mover_active_cycles=320`, `compute_efficiency=1.0`,
  `end_to_end_efficiency=0.095238`.
  This baseline intentionally repeats row reduction for each output segment so
  later row-state caching or command-list fanout has a measured reference.
- B0 gate-multiply vector subgraph is now executable through the CPU-to-NPU
  descriptor path: 32 `DESC_VECTOR_TILE_V1` jobs cover `gate_mul_up` across
  `S=8,FFN=32` using `VMUL + VREQUANT(INT8_SHIFT4_CLAMP) + HALT`. The input
  `silu_gate` is still a compiler golden input for this subgraph, so this does
  not complete the SiLU stage or full SwiGLU.
- Measured B0 gate-multiply subgraph PPA:
  `effective_vector_lane_ops=256`, `theoretical_vector_cycles=64`,
  `measured_vector_cycles=64`, `total_cycles=1088`,
  `data_mover_active_cycles=320`, `compute_efficiency=1.0`,
  `end_to_end_efficiency=0.058824`.
  This proves the multiply/requant portion of FFN gating is executable and
  exposes the current per-segment descriptor overhead.
- B0 two-head Attention subgraph is now executable through the CPU-to-NPU
  descriptor path as eight stage jobs: `QK -> Scale/Mask -> Softmax -> PV` for
  each query head with shared K/V. This uses B0 `rope_q/rope_k/v` tensors from
  the compiler golden and the current RTL Softmax bring-up LUT numerical
  contract, so it proves multi-head Attention stage composition and PPA
  attribution, not RoPE execution or final Softmax numerical closure.
- Measured B0 Attention subgraph PPA:
  `stage_jobs=8`, `effective_mac_ops=2048`,
  `theoretical_matrix_cycles=32`, `measured_matrix_cycles=32`,
  `total_cycles=1550`, `data_mover_active_cycles=406`,
  `sfu_active_cycles=144`, `compute_efficiency=1.0`,
  `end_to_end_efficiency=0.020645`. Per head, the measured stages are
  `QK=83 cycles`, `Scale/Mask=85 cycles`, `Softmax=526 cycles`, and
  `PV=81 cycles`. The immediate bottleneck remains the current Softmax path:
  Matrix work is fully utilized while active, but elapsed cycles are dominated
  by separate descriptor boundaries and serialized row Softmax work.

Immediate next implementation order:

1. complete: review and accept the detailed `DESC_VECTOR_TILE_V1`/segmented-row design in
   `transformer_npu_v1.md`, `descriptor_v1.md`, `uop_isa_v1.md`,
   `vector_engine_v1.md`, and `reduction_engine_v1.md`;
2. complete: implement the first common `DESC_VECTOR_TILE_V1 + VADD` carrier path so a
   primitive program consumes descriptor-selected 32-bit vector source
   segments instead of implicit Attention score rows;
3. complete: submit the compiler-lowered multi-segment residual `DESC_VECTOR_TILE_V1`
   jobs so rows wider than
   eight lanes can execute through Vector/Reduction/SFU without CPU
   materialization;
4. complete: execute RMSNorm primitive sequence over segmented `H=16` rows and
   publish Reduction/SFU/Vector PPA;
5. complete: execute `gate_mul_up` through `VMUL + VREQUANT` and publish Vector PPA;
6. complete: compose the two measured Attention heads as a B0 Attention subgraph;
7. add RoPE and SiLU primitive sequences;
8. submit one complete B0 execution package without CPU-filled intermediate stages;
9. submit Block 0 output directly as Block 1 input and publish B1 PPA.

Coding gate:

- do not add RMSNorm/RoPE/SwiGLU operator-specific Compute-cluster FSMs;
- do not mark B0 executable while any non-matrix stage is CPU/materialized;
- do not hide row-state accumulation or constants in testbench code;
- review descriptor flags, row-state storage, and PPA fields before RTL.

P3c B1 acceptance:

- two blocks execute and chain correctly;
- at least one important dimension exceeds the physical tile and is
  Compiler-tiled;
- full-run PPA identifies the top bottlenecks by elapsed cycles, movement,
  utilization, and available area/energy evidence;
- only then is the first hardware optimization candidate selected.

### Deferred Attention-Only Optimization Roadmap

Mask completion is not the end of P3. It removes the correctness blocker for
generalized shapes; the next work is to stop treating `8x8` as the logical
Attention size while retaining it as the physical compute tile.

Execution order:

1. **P3a: executable single-tile masks**
   - add packed row-mask fixture/runtime data;
   - load two mask words through descriptor `input1`;
   - integrate lane gating in Scale/Mask, Reduction, and normalization;
   - execute causal full-physical-row cases with `seq_q/seq_k <= 8`;
   - leave all-invalid physical query rows to P3b valid-row scheduling.
2. **P3b: Softmax row-throughput architecture**
   - treat `seq_q<8` valid-row omission as a small Compiler correctness change,
     not the main optimization;
   - quantify the current one-row-at-a-time Scheduler/Vector/Reduction/SFU
     bottleneck;
   - design and compare `row_parallelism = 1, 2, 4` candidates;
   - keep the current serial path as the functional/PPA baseline;
   - use mapped area/timing before accepting replicated or widened resources.
3. **P3c: multi-tile Attention baseline**
   - review and accept the M/N/K tiling, descriptor ownership, boundary-tile,
     and PPA contract in `transformer_npu_v1.md` before coding;
   - first target `S=16,D=16`, then `S=32,D=32`;
   - make the test Compiler/planner accept logical operator shapes and
     automatically emit M/N/K tile descriptors; Runtime/submitter does not
     perform tiling;
   - tile QK over query, key, and head-dimension axes;
   - use K-stream accumulation for `D > 8`;
   - materialize multi-tile score rows in SRAM for the first correct baseline;
   - implement segmented row max/sum across key tiles;
   - stream/accumulate PV across probability/value K chunks;
   - skip fully invisible causal future-key tiles and report saved work.
4. **P4: reduce the baseline overhead**
   - compare the correct multi-descriptor/multi-tile baseline against grouped
     command lists, on-chip score/probability residency, and streaming/online
     Softmax;
   - keep graph lowering, fusion choice, and command-list generation in the
     Compiler; Runtime binds/submits, Host Wrapper remains thin, and Core
     Command Processor/Uop Scheduler executes the list;
   - retain the unfused baseline so PPA benefit is measured rather than
     assumed.

Required multi-tile workloads:

| Workload | Purpose |
| --- | --- |
| causal `S=8,D=8` | first executable masked RTL and regression |
| tail `S=5,D=8` | valid query/key lane behavior |
| K-stream `S=8,D=16` | larger head dimension without larger Matrix array |
| multi-key-tile `S=16,D=8` | segmented Softmax correctness |
| multi-axis `S=16,D=16` | complete Compiler/runtime/buffer tiling baseline |

The physical Matrix/Vector tile remains `8x8`/eight lanes. Larger logical
Attention is primarily a Compiler, Runtime, buffer-allocation, segmented
reduction, and scheduling problem before it is a reason to enlarge the RTL
array.

#### P3b corrected priority / P3b修订后的优先级

Matrix edge-tile waste is an accounting result, not a new execution mechanism.
The Compiler/planner already knows logical `M/N/K`; RTL performance counters
already expose Matrix active cycles and physical issue capacity. Therefore:

```text
useful_mac_ops = logical M * N * K
issued_mac_capacity = measured_matrix_active_cycles * peak_macs_per_cycle
tail_waste = issued_mac_capacity - useful_mac_ops
```

Adding these fields requires workload metadata/report validation, but no
Matrix RTL control and no Runtime decision logic. The first baseline continues
to execute a full physical `8x8` tile when it contains padding.

Matrix边界tile中的无效MAC属于PPA统计结果，不需要新增软件执行逻辑或Matrix
控制逻辑。Compiler提供逻辑shape，RTL提供实测Matrix active cycle，报告据此
计算有效MAC、物理发射容量和tail waste。只有数据证明tail waste成为主要瓶颈
后，才评审Matrix lane gating。

The more important current hardware limitation is serialized Softmax:

```text
one in-order Uop Scheduler
  -> at most one primitive command in flight
  -> one 8-lane Vector Engine processes one score row
  -> one Reduction Engine processes one row
  -> one scalar SFU processes EXP lanes serially
  -> next row begins only after the prior primitive responses complete
```

The current eight-lane Vector Engine already computes all eight elements of
one row in parallel. It does **not** compute multiple rows concurrently.
Commercial accelerators commonly expose more parallel execution capacity
through multiple SIMD/vector/SFU lanes or clusters and multiple outstanding
rows/heads/tiles. The useful architecture question is not simply "how many
Vector Engines", but how many rows can be in flight and which resources must
be replicated, widened, banked, or pipelined to sustain them.

当前Vector Engine的8个lane并行处理一行中的8个score，但8行Softmax仍完全
串行。真实高性能加速器通常允许多个row、head或tile同时在途，并配置多个
SIMD/SFU执行资源或执行簇。我们下一步应探索“同时在途多少行、复制哪些资源、
共享哪些资源”，而不是只处理无效行。

P3b architecture candidates:

| Candidate | Mechanism | Expected benefit | Main cost/risk |
| --- | --- | --- | --- |
| S0 serial baseline | current one-command-in-flight Scheduler and one row datapath | retained reference | measured Softmax latency is dominated by serialized issue/response and scalar EXP |
| S1 multi-row scheduler contexts | allow multiple independent row states in flight while retaining shared engines | overlaps Scheduler/control and prepares engine parallelism with modest state cost | little benefit if engines remain single-issue; requires dependency/scoreboard contract |
| S2 duplicated row pipelines | instantiate 2 or 4 Vector/Reduction/SFU row pipelines with banked row storage | near-proportional row throughput when bandwidth and issue keep up | area/power growth; replicated SFU/Reduction may hurt timing |
| S3 asymmetric shared pipeline | multiple Vector/Reduction row contexts plus a pipelined or multi-lane SFU shared across rows | targets the likely scalar-SFU bottleneck with less duplication | arbitration, buffering, and backpressure complexity |
| S4 fused/online Attention pipeline | overlap tiled QK, online Softmax, and PV | highest long-context movement/latency opportunity | much larger architecture change; defer until S1-S3 evidence and multi-tile baseline |

P3b first design package must measure S0 and specify S1-S3 before coding a
candidate. Required report fields:

- row throughput and rows simultaneously in flight;
- per-engine issue rate, active, stall, and backpressure cycles;
- Scheduler issue/response overhead per primitive and per row;
- Vector, Reduction, and SFU utilization;
- theoretical and measured Softmax cycles for 1, 2, 4, and 8 rows;
- normalized resource delta followed by mapped area/timing delta;
- full Attention group cycle improvement, not only isolated Softmax speedup.

Initial acceptance decision:

1. add only the small valid-row Compiler omission needed for correctness;
2. establish the S0 serial Softmax bottleneck report;
3. review S1-S3 resource/handshake/storage design and theoretical benefit;
4. implement the selected candidate only after review;
5. accept it only if full-Attention benefit justifies mapped area/timing and
   later power cost.

#### Edge-tile terminology and correctness work / 边界tile术语与正确性工作

`tail` describes the logical remainder when a dimension does not fill the
physical hardware tile. For example, `seq_q=5` on an eight-row physical tile
has five valid query rows and three padding rows that must not execute as real
Softmax rows.

`edge tile` is the physical tile containing such a tail. The first correct
baseline still stores and moves the current `8x8` container, but it carries
explicit valid extents:

```text
logical shape: seq_q=5, seq_k=5, head_dim=8
physical QK tile: 8x8
valid query rows: 0..4
valid key columns: 0..4
physical rows/columns outside those extents: padding, never logical output
```

`minimal plan-driven submission` means the test Compiler/planner emits the
descriptor sequence, buffer references, and tile offsets needed by a hardware
test. The thin submitter sends those descriptors without knowing Attention
stage names. Current firmware still contains branches equivalent to:

```text
if stage == QK:       use fixed Q/K/score addresses
if stage == Softmax:  use fixed score/probability addresses
if stage == PV:       use fixed probability/V/output addresses
```

That cannot launch another shape or tile without new C code. P3b changes only
the submission contract needed for architecture experiments:

```text
runtime_job.input0_buffer_id -> allocated address/word count
runtime_job.input1_buffer_id -> allocated address/word count
runtime_job.output_buffer_id -> allocated address/word count
runtime_job.program_buffer_id -> generated program address/word count
runtime_job.op/shape metadata -> generic descriptor fields
```

The test Compiler/planner owns shape-to-tile conversion. The submitter binds
addresses and submits jobs; it does not recalculate masks, tile the graph, or
choose a different execution order. A user-facing test API may accept
`matmul(M,N,K)`, but that API invokes the Compiler/planner to create tiles
before submission. Tiling is not a Runtime algorithm.

P3b contains three separately reviewable changes:

| P3b item | Problem | Design and ownership | Initial acceptance |
| --- | --- | --- | --- |
| P3b.1 valid-row scheduling | fixed Scale/Mask and Softmax programs issue all eight physical rows; for `seq_q<8`, padding rows have no valid lane and would execute invalid Softmax work | Compiler emits row-indexed uops only for valid query rows; Scheduler executes only emitted rows; row masks still gate invalid key columns | causal `seq_q=5,seq_k=5,D=8` executes rows 0..4 only; no Vector/Reduction/SFU activity appears for rows 5..7 |
| P3b.2 single edge-tile execution | Matrix and storage paths use physical `8x8` containers, but logical rows/columns may be smaller and useful work/tail waste are not executable/reportable end to end | Compiler records valid M/N/K extents; first baseline zero-fills invalid input lanes, masks invalid Attention columns, and checks/stores only valid logical output; hardware Matrix lane-skipping is not assumed | `S=5,D=8`, `S=8,D=5`, and one combined edge case match golden; PPA reports useful MACs, issued capacity, Matrix tail waste, transferred padding, and invalid Softmax-row work as zero |
| P3b.3 minimal plan-driven submitter | generated job order exists, but firmware still maps each Attention stage to fixed C arrays and addresses | test Compiler/planner emits concrete descriptor records and buffer references; a thin submitter launches them without stage-specific behavior | a second shape runs without adding a new stage-specific firmware branch; no general allocator, graph importer, or dynamic Runtime tiler is required |

P3b does **not** execute a logical dimension larger than eight. It makes one
partially filled physical tile correct and removes fixed-shape Runtime
assumptions. P3c then composes multiple full or edge tiles for `S=16,D=16` and
beyond.

P3b acceptance reports must show:

- logical shape and physical tile shape;
- valid rows/columns and emitted row-uop count;
- theoretical Matrix cycles, measured Matrix cycles, and tail-waste capacity;
- Data Mover words, including explicit padding movement;
- absence of engine activity for invalid query rows;
- RTL workload-view candidate-versus-baseline results;
- CNN and existing causal `S=8,D=8` regression results.

### P2 Performance-First Physical Resource Contract

Problem being solved:

- functional RTL can create wide same-cycle array operations without making
  their required hardware resources explicit;
- PPA correctly reports those RTL cycles, but an architecture reviewer cannot
  tell whether the cycle is a deliberate performance choice or an accidental
  unlimited-bandwidth assumption;
- later storage refactors could silently serialize or overlap transactions and
  invalidate the established performance baseline.

Accepted policy:

- prioritize performance for the current baseline;
- explicitly declare 64-lane accumulator clear/commit/readout, eight-lane
  Attention row access, and eight-A/eight-B Matrix slice feed;
- accept the expected wide-routing, storage-port, area, power, and timing cost;
- keep one-cycle latency unless mapped timing, area, or power evidence
  justifies changing the spec;
- any width reduction requires spec-first change and a new measured baseline.

Implementation and verification:

1. generate RTL constants from the canonical performance contract;
2. fail RTL elaboration when tile/row dimensions do not match the contract;
3. record the active contract and unverified mapped-timing status in perf/PPA;
4. reject accumulator clear/commit/readout events that overlap or last longer
   than their declared transaction latency;
5. retain Transformer and CNN full regressions as compatibility gates.

### P1 Common Primitive Scheduler Integration

Problem being solved:

- Softmax now uses the common Uop Scheduler and a Compiler-expanded program.
- Scale/Mask now uses Scheduler-issued row-level primitives, proving the
  two-level scheduling boundary, but its per-primitive routing/start/response
  handoff still consumes three control cycles around each one-cycle Vector op.
- Compute cluster no longer contains the private Softmax operator sequence FSM.
- The expanded baseline exposes `113` program words and serialized
  Scheduler/route/engine completion overhead that now needs optimization.
- Adding new micro-kernels would duplicate more control logic and PPA
  instrumentation.

Target design:

```text
compiler-generated primitive uop program
  -> common Uop scheduler fetch/decode/dependency/issue/wait
  -> Compute cluster routing/arbitration
  -> Vector / Reduction / SFU engines
  -> response back to Uop scheduler
```

Implementation order:

1. Make the Compiler emit the fully expanded Softmax primitive program and
   report required words versus current instruction-memory capacity.
2. Compare expanded-program instruction-memory/movement cost against a generic
   counted-loop ISA candidate before accepting any hardware loop.
3. Finalize primitive command/response valid-ready payloads and wait reasons.
4. Extend Uop Scheduler decode/issue/response ports for Vector, Reduction, and
   SFU while retaining the existing Matrix path.
5. Add the minimum local-buffer and scalar operand references required by
   Scale/Mask and Softmax primitive uops.
6. Select the accepted program representation and provide sufficient
   instruction-memory capacity.
7. Switch Softmax to the Scheduler path, keeping the current
   private FSM as a temporary regression reference.
8. Remove private operator sequence states after functional and PPA
   equivalence/benefit gates pass.

PPA acceptance:

- Uop Scheduler lane shows each primitive fetch/decode/issue and dependency
  wait using measured semantic events.
- Compute-cluster control no longer decides primitive order; remaining cycles
  are only routing/arbitration/response handoff and are named accordingly.
- Scale/Mask and Softmax before/after reports include total, scheduler,
  integration-control, Vector/Reduction/SFU active, and wait-reason cycles.
- The first in-order integration need not overlap engines, but it must reduce
  duplicated control and establish a path that can later support overlap.

Current P1 progress:

- Score Scale primitive is named `VSCALE_FIXED`, because it keeps the signed
  `int32` score domain while approximating `1/sqrt(D_k)` with
  multiply-round-shift. `VEC_REQUANT` remains reserved for conversion into a
  target quantized representation.
- The earlier `LOOP_ROWS` implementation was withdrawn because the hardware
  loop had not been compared against a Compiler-expanded program.
- Scale/Mask uses the Compiler/generated-program baseline of eight row-indexed
  `VSCALE_FIXED` uops plus `HALT`.
- Softmax executes the Compiler-expanded 113-word program from a selected
  128-word instruction memory. Row/lane sequencing is no longer hardcoded in
  Compute cluster.
- Clamp bounds, normalization shift, and Score Scale multiplier/shift are
  generated from canonical architecture config rather than retyped RTL
  literals.
- Scheduler-to-Compute-cluster primitive issue now uses a held one-in-flight
  valid-ready command and response contract with typed Matrix-response,
  primitive-accept, and primitive-response wait reasons.
- The extra route-start state is removed. Current engines still use an internal
  registered start/done adapter; its real one-cycle latency is reported as
  `ENGINE_START_ADAPTER`, not hidden in an inferred control bucket.
- Scale/Mask improved from `92` to `84` total cycles. Scheduler changed from
  `10 active + 32 wait` to `18 active + 16 wait`; Compute-cluster child
  activity reconciles as `24 control + 8 Vector = 32` cycles.
- Expanded Softmax improved from `637` to `525` total cycles. Scheduler changed
  from `114 active + 448 wait` to `226 active + 224 wait`; Compute-cluster
  child activity reconciles as `336 control + 24 Vector + 16 Reduction + 72
  SFU = 448` cycles.
- The generic counted-loop candidate remains a measured-design decision, not
  an implemented hardware loop.

Compiler-expanded Softmax capacity result:

| Item | Expanded baseline | Generic counted-loop candidate |
| --- | ---: | ---: |
| Primitive/program words for fixed `S=8` | `112 + HALT = 113` | approximately `15 primitives/control + HALT = 16` |
| Program bytes | `452` | `64` |
| Fits selected `128`-word instruction memory | yes | yes |
| Practical power-of-two instruction-memory capacity | `128` words | candidate could use `16` words |
| Theoretical program transfer lower bound at current `4 words/cycle` mover width | `ceil(113/4) = 29` cycles | `ceil(16/4) = 4` cycles |
| Additional per-row loop-control cycles | none | modeled `8` cycles for fixed `S=8` |
| Additional hardware | larger instruction memory/addressing | generic loop counter/branch control |
| Fixed-case total-cycle comparison | measured `525` cycles | modeled `525 - (29 - 4) + 8 = 508` cycles |

The candidate saves a modeled `17` cycles, or about `3.2%`, for fixed `S=8`,
before accounting for loop-control area/timing cost. P1 therefore retains the
Compiler-expanded program as the accepted implementation and does not add a
hardware loop. The loop decision can be reopened for larger generalized shapes
with measured area and timing evidence.

Completed P0 instrumentation correction:

- Attention Softmax PPA shows the measured 113-word program movement,
  Scheduler primitive issue/wait spans, and engine row/lane work.
- Scale/Mask PPA shows its measured primitive-uop program movement, eight
  Scheduler issue/wait pairs, eight Vector operations, and remaining
  Compute-cluster routing/handoff cycles.
- Their Vector, Reduction, and SFU spans come from measured primitive
  active/op/row/lane events. Ratio-derived occupancy blocks are not acceptable.
- The current eight-row softmax serializes eight scalar EXP operations per row;
  this is expected to dominate the measured compute latency and is a concrete
  optimization target, not an unexplained report artifact.
- Residual control attribution is named `Compute cluster control`, with
  semantic accept/adapter/response events.
- The timeline validator rejects out-of-range spans, unexplained
  compute-active cycles, active/wait contradictions, and lane/counter
  mismatches before a PPA report is accepted.
- Add a PPA review checklist to every architecture work package: expected
  module order, theoretical cycle basis, measured timeline, conservation
  result, bottleneck explanation, and before/after comparison.

## Current PPA Status And Gaps

Current PPA evidence is Level 0 only:

- performance cycles and movement counters come from RTL/SoC simulation;
- normalized area is a structural L0 model, not synthesized area;
- normalized energy is an event/coefficient L0 modeled estimate, not measured power;
- external-memory energy is modeled from workload metadata.

Measured stage PPA exists for:

| Stage | Current PPA evidence | Main gap |
| --- | --- | --- |
| QK | measured cycles, data mover words, useful MACs, matrix utilization, L0 modeled energy; output feeds scale input | larger `D_k`/tile support remains |
| Scale/mask | measured primitive-uop program movement, row-mask movement, typed Scheduler issue/wait pairs, eight fixed-point Vector scale/mask operations, and semantic accept/adapter/response cycles for causal `8x8 int32`; output feeds tile softmax | tail/all-invalid-row scheduling and internal start/done adapter remain |
| Softmax | measured 113-word program movement, typed Scheduler issue/wait spans, semantic control events, and per-engine row/lane timeline events, L0 modeled energy | expanded program and serialized one-in-flight execution dominate; per-engine energy counters remain incomplete |
| PV | measured cycles through shared mixed matrix mode, useful MACs, matrix utilization, L0 modeled energy | mixed `u16 x s8` area/energy still uses generic MAC model coefficients |
| Full attention parent | buffer-chained measured QK/scale-mask/eight-row-softmax/PV stages through generated runtime table | no command-list/full-descriptor snapshot, runtime overhead not measured |

Missing before claiming complete attention PPA:

- group total cycles and energy provenance for `attention_prefill_s8_d8` that
  includes or explicitly excludes runtime overhead;
- submodule counters or derived events for matrix, vector, reduction, SFU, data
  mover, and scheduler/control;
- event-energy coefficients split by `int8xint8` MAC, `u16xint8` MAC,
  vector lane op, reduction element op, SFU EXP, and SFU RECIP;
- structural area model split by matrix, mixed-precision multiplier delta,
  vector lanes, reduction tree, SFU LUT/table, accumulator/local buffers, and
  wrapper/data mover;
- later mapped area/timing and activity-driven power if architecture
  choices need ASIC trend validation.

## V1 Module Status

| V1 module | Design doc | RTL / tooling status | Verification status | Next action |
| --- | --- | --- | --- | --- |
| wrapper / CSR | `arch/specs/transformer/v1/csr_map_v1.md` | v0 wrapper exists; v1 CSR fields are spec-only | v0 CSR perf path passes | add v1 counters only after event sources are reviewed |
| descriptor engine | `arch/specs/transformer/v1/descriptor_v1.md` | v0 descriptors execute current jobs; v1 descriptor is spec-only | v0 firmware profiles pass | add v1 fields when a new executable job needs them |
| uop scheduler | `arch/specs/transformer/v1/uop_isa_v1.md` | v0 in-order sequencer exists; v1 primitive scheduler is not integrated | v0 core sim passes | connect standalone primitive engines after module tests stabilize |
| matrix engine | `docs/design/npu_core.md` | `matrix/matmul_array.sv` implemented | `make npu-core-sim` passes | add GEMV/valid-lane support after primitive vector path |
| accumulator file | `docs/design/transformer/transformer_npu_v1.md` | `matrix/accumulator_file.sv` integrated into `npu_v0_compute_cluster` | core, quick SoC, full CNN pass | expose counters through perf only after CSR plan update |
| vector engine | `docs/design/transformer/vector_engine_v1.md` | standalone `vector/vector_engine.sv` implemented | primitive op and softmax/RMSNorm sequence tests pass | connect to scheduler/uop path |
| reduction engine | `docs/design/transformer/reduction_engine_v1.md` | standalone `reduction/reduction_engine.sv` implemented | primitive op and softmax/RMSNorm sequence tests pass | broaden row-length coverage before scheduler integration |
| SFU | `docs/design/transformer/sfu_v1.md` | standalone `sfu/sfu_lut.sv` implemented | EXP/RECIP/RSQRT and sequence tests pass | refine LUT/tolerance before model accuracy claims |
| memory / scratchpad / data mover | common docs in `docs/design/` | v0 core data mover exists | perf/PPA pass | add v1 internal scratchpad contract before widening |
| KV cache subsystem | `arch/specs/transformer/v1/transformer_npu_v1.md` | spec/model-only counters only | perf/PPA model-only traffic visible | no RTL until decode traffic evidence justifies it |
| Transformer attention workloads | `docs/design/transformer/attention_workload_ppa.md` | QK, causal scale/mask, masked attention softmax, and mixed PV stage jobs execute from a generated runtime-job table with fixed-case buffer chaining | `make cpu-soc-transformer` and `make ppa-l0-report WORKLOAD_PROFILE=transformer` pass | remove fixed-shape/stage-specific runtime assumptions, then reduce launch and external-memory boundaries |

## Completed Foundations

The following foundations are complete and are not competing next-step plans:

- current NPU-core module ownership and status are documented;
- primitive valid/ready shims and local event counters have directed tests;
- the fixed `S=8,D=8` attention planner, generated runtime table, and
  QK -> scale/mask -> softmax -> PV buffer chain execute;
- causal Scale/Mask, masked tile Softmax, mixed-precision PV, and grouped
  Transformer PPA are measurable;
- PPA distinguishes measured events from modeled area/energy evidence.

All new implementation priority and acceptance gates are defined only by
`Recommended Next-Phase Plan`. Scale/Mask and Softmax migration to the common
Uop Scheduler is complete. P0 semantic-event instrumentation, P1
valid-ready/typed-wait integration, and P2 performance-first resource
contracts are complete. The active package is P3 mask and shape
generalization.

### Complete Attention Subnetwork Gap

For this project, a "complete attention subnetwork" means the executable
subgraph:

```text
Q, K, V inputs
  -> QK score matmul
  -> score scale and mask
  -> row softmax
  -> PV weighted sum
  -> O output
```

The current implementation exercises the hardware for QK, one-row softmax, and
PV, and software-sequences those measured stages through a generated runtime
table. It does not yet execute the full formula end-to-end with produced
intermediate buffers.

Current state versus complete subnetwork:

| Requirement | Current state | Remaining work |
| --- | --- | --- |
| Input graph/operator source | Transformer workload manifest drives the fixed `attention_prefill_s8_d8` plan | add operator metadata and eventually model/graph import beyond manifest-only tests |
| Compiler lowering | manifest-driven plan emits QK, scale/mask, softmax, PV stages for `S=8,D=8` | generalize shape/tile lowering and reject/handle larger rows explicitly |
| Runtime launch | CPU firmware iterates generated runtime jobs for QK, scale/mask, softmax, PV | make descriptor filling buffer-driven instead of stage-specific C switch |
| QK stage | measured through `matmul_k_stream` | support larger `D_k`/tiles and expose matrix mode counters |
| Scale/mask stage | executable measured causal `8x8 int32` descriptor; QK output and packed row mask feed scaled/masked output | add tail/all-invalid-row scheduling and generalize tiling |
| Softmax stage | measured masked eight-row tile through current vector/reduction/SFU sequence | upgrade SFU numerical contract and generalize segmented/tiled rows |
| PV stage | measured through mixed `u16 x s8` matrix mode using produced probability tile | generalize tiling and review mixed-path PPA coefficients |
| Intermediate buffers | fixed `S=8,D=8` SRAM buffers are producer-to-consumer chained | make descriptor filling generic and allocate buffers for generalized shapes |
| Parent group PPA | `software_group_measured_stages` for QK/scale-mask/softmax/PV | add runtime-overhead policy; eventually one command-list snapshot |
| Numerical contract | QK exact, softmax bring-up, PV Q0.15 mixed path | unify scale/mask/SFU/requant/PV under one target attention contract |

These fixed-case milestones are complete. Remaining work follows the unique
P0-P7 order above; it must not reuse the earlier bring-up priority numbers.

### 0. Attention Sequence Contract

Status:

- Implemented in documentation: `attention_sequence_v1.md` defines attention
  as QK matrix, score scale/mask/clamp, row softmax, and PV sequence over
  existing primitives.
- Implemented in documentation: attention workload/PPA and compiler/runtime
  documents define parent/stage grouping, model-only versus measured evidence,
  and software-owned lowering.
- Implemented in current SoC path: QK, causal scale/mask, masked softmax, and PV
  stages are separately executable and visible in PPA.
- Still deferred: generalized intermediate-buffer allocation, tail/multi-tile semantics,
  command-list ABI, and measured full-attention parent row.

Acceptance criteria:

- Attention formula and primitive mapping are documented.
- Matrix/vector/reduction/SFU docs state current support and attention-driven
  gaps.
- No v1 plan introduces a dedicated attention RTL macro.

### 0.1 Attention Golden And Workload Identity

Detailed design:

- `docs/design/transformer/attention_sequence_v1.md`
- `docs/design/transformer/attention_workload_ppa.md`

Acceptance criteria:

- Add Python golden functions for:
  - QK score matmul;
  - score scale/mask/clamp;
  - row softmax Q0.15;
  - PV policy;
  - full `attention_prefill_s8_d8`.
- Add manifest entries with parent/stage identity:
  - `transformer_attention_qk_s8_d8`;
  - `transformer_attention_softmax_s8`;
  - `transformer_attention_pv_s8_d8`;
  - `transformer_attention_prefill_s8_d8`;
  - decode KV traffic attention view.
- Any field that is not launched through firmware/runtime remains explicitly
  labeled with model-only provenance.
- Numerical contract IDs identify whether a workload uses simplified bring-up
  policies or target attention policies.

### 0.2 Measured QK Stage On Current Matrix Path

Acceptance criteria:

- Execute `attention_qk_s8_d8` using current `8x8x8` matrix path.
- Compiler/generator emits K-transposed tile layout and exact int32 expected
  score tile.
- Perf/PPA reports show QK stage cycles, useful MACs, data movement, and
  matrix utilization.
- Existing CNN, operator smoke, and current Transformer quick workloads keep
  passing.

### 0.3 SFU EXP/RECIP Contract For Attention Softmax

Detailed design:

- `docs/design/transformer/sfu_v1.md`

Acceptance criteria:

- Generate the 257-entry Q0.15 EXP table from a deterministic numerical source
  and config scale/Q fields.
- Replace current 9-segment `bringup_exp_q15_segments` RTL path only after
  golden and RTL agree.
- Define RECIP input range, Q0.24 output, normalization shift, zero behavior,
  and test vectors.
- Keep target fixed-spec and current RTL approximation functions separately
  named.
- Attention softmax remains bring-up-labeled, even when measured, until this
  contract is implemented.

Upgrade trigger:

- implement this before measured attention softmax is used as target PPA or
  accuracy evidence; simplified SFU is acceptable only for plumbing bring-up.

### 0.4 Valid/Ready and Counter Expansion

Detailed design: `docs/design/transformer/primitive_valid_ready_v1.md`.

Acceptance criteria:

- Add an engine-level issue/accept/done contract before scheduler integration:
  `valid`, `ready`, `done`, and stable input hold rules.
- Define active/stall/idle counter increments in the spec before adding CSRs.
- Preserve the existing single-start bring-up tests as compatibility tests
  until the scheduler path consumes the valid/ready interface.

### 0.5 Requant v2 And Attention Mask Semantics

Detailed design:

- `docs/design/transformer/vector_engine_v1.md`

Acceptance criteria:

- Add a mode field or distinct op contract for `mul_round_shift_clamp`.
- Extend config/spec/golden with multiplier width, rounding mode, shift, clamp,
  and optional zero-point behavior.
- Keep current `shift_clamp` as a named v1 mode with regression coverage.
- Only switch Transformer fixed-spec softmax/RMSNorm paths to v2 requant after
  RTL-like golden and primitive RTL tests agree.
- Define attention mask semantics: unmasked first, then causal/padding mask
  through mask-select or compiler-materialized negative sentinel.

Upgrade trigger:

- implement fixed-multiplier requant before claiming `1/sqrt(D)` target
  scaling;
- implement mask semantics before claiming decoder prefill or tiled-tail
  attention correctness.

### 0.6 Row Reduction Contract For Attention Softmax

Detailed design: `docs/design/transformer/reduction_engine_v1.md`.

Acceptance criteria:

- Define row length, valid-lane, masked-lane, and empty-row behavior.
- Define segmented row max/sum behavior for `S > lanes`.
- Add measured or modeled `reduction_element_ops` semantics before PPA uses
  reduction energy as evidence.

### 1. Accumulator File Integration

Acceptance criteria:

- `npu_v0_compute_cluster` uses `hw/npu_core/rtl/matrix/accumulator_file.sv` for matmul and
  K-stream partial-sum residency instead of the legacy internal `acc_buf`.
- Existing `MATMUL` and `MATMUL_K_STREAM` firmware paths keep passing.
- Accumulator counters are wired into a local test or documented as internal
  module counters pending CSR exposure.

Status:

- Implemented in the current work item. `acc_buf` is removed from
  `npu_v0_compute_cluster`; accumulator storage is now provided by
  `matrix/accumulator_file.sv`.

Verification:

```text
make npu-core-sim
make cpu-soc-cnn-full
make ppa-l0-report WORKLOAD_PROFILE=transformer
```

### 2. Primitive Vector / Reduction / SFU Blocks

Acceptance criteria:

- Add standalone RTL modules under:
  - `hw/npu_core/rtl/vector/`
  - `hw/npu_core/rtl/reduction/`
  - `hw/npu_core/rtl/sfu/`
- Cover `VEC_CLAMP`, `REDUCE_MAX`, `REDUCE_SUM`, `REDUCE_SUMSQ`,
  `SFU_EXP`, `SFU_RECIP`, and `SFU_RSQRT` with deterministic RTL unit tests.
- Match Python golden in `sw/tools/transformer/micro_golden.py` within stated
  fixed-point tolerances.

Status:

- Standalone RTL and directed testbench are implemented:
  - `hw/npu_core/rtl/vector/vector_engine.sv`
  - `hw/npu_core/rtl/reduction/reduction_engine.sv`
  - `hw/npu_core/rtl/sfu/sfu_lut.sv`
  - `hw/npu_core/tb/primitive_engines_tb.sv`
- Scheduler integration is still pending.

Design contracts:

- `docs/design/transformer/vector_engine_v1.md`
- `docs/design/transformer/reduction_engine_v1.md`
- `docs/design/transformer/sfu_v1.md`

Verification:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_transformer_micro_fixtures -v
make primitive-engines-sim
make npu-core-sim
```

### 3. Attention Softmax Primitive Micro-Kernel Fixtures

Acceptance criteria:

- `attention_softmax_s8` has generated fixture data and expected Q0.15 outputs.
- It executes through the CPU-to-NPU `attention_softmax_v1` descriptor path.
- Perf/PPA reports measured stage cycles for the current bring-up numerical
  contract.
- Vector/reduction/SFU submodule event split remains a PPA gap until counters
  or derived event accounting are added.
- RMSNorm remains covered but is not the next PPA driver.

Status:

- Implemented for standalone primitive sequence validation and SoC stage-level
  attention softmax bring-up.
- Python golden functions:
  - `softmax_row_primitive_lut_q15()`;
  - `rmsnorm_primitive_sequence()`.
- RTL sequence coverage lives in `hw/npu_core/tb/primitive_engines_tb.sv`.
- The attention softmax smoke uses the Compiler-expanded primitive program;
  `npu_v0_compute_cluster.sv` only routes issued primitives. It remains labeled
  as the bring-up numerical contract until SFU/RECIP are upgraded to the target
  fixed spec.

### 4. PV Policy And GEMV / Skinny GEMM Execution Path

Acceptance criteria:

- Decide PV policy:
  - selected current direction: mixed Q0.15 x int8 weighted-sum/matrix support;
  - deferred approximation: probability requantized to int8 and reused by the
    old matrix mode.
- Add a true `GEMV_TILE` or valid-row/valid-column skinny-GEMM path.
- Report distinguishes:
  - full tile GEMM;
  - skinny GEMM;
  - GEMV.
- `effective_mac_ops`, `peak_mac_capacity`, `matrix_utilization`,
  `gemv_utilization`, and `tail_waste_mac_capacity` are still visible in
  perf/PPA output.

Upgrade trigger:

- int8 probability PV is acceptable as an explicitly labeled approximation;
- mixed Q0.15 x int8 PV is now the attention PV direction and should keep using
  the shared matrix path, with upgraded area/energy coefficients before making
  mixed-precision PPA claims.

### 5. Attention Runtime / Command-List Path

Detailed design:

- `docs/design/transformer/software_runtime_compiler_attention.md`
- `docs/design/transformer/attention_operators_v1.md`
- `docs/design/transformer/attention_compiler_v1.md`
- `docs/design/transformer/attention_runtime_v1.md`

Acceptance criteria:

- Add operator metadata for QK, score scale/mask, softmax, PV, and composite
  SDPA.
- Add compiler lowering from logical attention to primitive stage sequence.
- Add runtime launch model:
  - multi-descriptor firmware loop for generated grouped stage execution first;
  - command-list executor before production-like full measured attention.
- Define intermediate score/probability/output buffer allocation.
- Preserve generated `job_id` to manifest correlation.
- Promote `attention_prefill_s8_d8` from model-only only after generated runtime
  launches QK -> scale/mask -> softmax -> PV and PPA reports the group
  provenance.

PPA acceptance additions:

- report measured stage cycles and group total separately;
- expose or derive matrix/vector/reduction/SFU/data-mover events;
- split modeled energy coefficients for `int8xint8` MAC and `u16xint8` mixed MAC;
- split structural area model by matrix, vector, reduction, SFU, buffers, and
  wrapper/data mover before using submodule area conclusions.

### 6. KV Cache Counter Path

Acceptance criteria:

- Keep v1 KV cache as counters/spec first.
- Add CSR/report plumbing for `kv_read_bytes` and `kv_write_bytes` only after
  the event source is reviewed.
- Do not implement a complex KV streamer until decode bytes/token dominates a
  measured or modeled PPA comparison.

## Non-Goals For The Next Round

- No full LLaMA.
- No full Transformer block.
- No dedicated fused-attention RTL macro; standard engine-to-engine streaming
  and command-list fusion remain valid measured optimization work.
- No macro-op hardware expansion.
- No dedicated attention RTL macro.
- No INT4/FP8.
- No real LPDDR controller.
- No CNN regression deletion.
