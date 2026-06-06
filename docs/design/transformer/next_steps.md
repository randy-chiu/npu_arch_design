# Transformer NPU v1 Next Steps

## Goal

Prepare the next implementation rounds around Transformer attention. Attention
is the primary workload and PPA driver for the next phase, but it is not a
dedicated RTL macro. The implementation target is a compiler/runtime scheduled
sequence over matrix, vector, reduction, SFU, data mover, and scheduler
primitives.

The immediate architectural question is:

```text
Which existing primitive blocks are sufficient for attention, which contracts
are underspecified, and which missing features must be added before attention
measurements are valid PPA evidence?
```

## Current Attention Validation Summary

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

- QK, unmasked scale/mask, softmax, and PV are launched and measured as
  separate descriptor jobs,
  but the firmware launch order now comes from a compiler-generated runtime-job
  table rather than direct hand-written stage calls.
- QK output SRAM feeds an executable unmasked scale/mask descriptor, the
  produced scaled-score tile feeds softmax, and the produced probability tile
  feeds PV for the fixed `S=8,D=8` case.
- The parent `attention_prefill_s8_d8` workload is now
  `software_group_measured_stages` for QK/scale-mask/softmax/PV stage execution, not a
  command-list or single-descriptor measured full-attention operation.

Therefore the next software task is to remove stage-specific firmware
descriptor filling and fixed `S=8,D=8` assumptions, then reduce the launch and
external-memory boundaries between the already connected stages.

## Command-Processor Trace Contract

### Current behavior

The current PPA command-processor timeline is measured, but its text labels are
not yet a generic architectural trace contract.

The SoC testbench records the real command-processor state and arguments on
every active cycle:

```text
cmd_state, stream_chunk, target_bank, movement phase, uop event,
matrix event, accumulator event
```

Consequently, the displayed ordering and duration of actions such as
accumulator clear and chunk launch come from the actual RTL execution. They are
not a synthetic sequence inserted only for the Transformer report.

However, `sw/tools/perf/report.py` currently converts raw numeric
`cmd_state` values into labels such as:

```text
Enable and clear accumulator
Launch chunk <stream_chunk>; latch selected compute bank
Select alternate prefetch/next-compute bank
```

That numeric-state-to-label table is hardcoded in the report. The descriptor
provides addresses, lengths, mode, and chunk count; the fixed RTL
command-processor state machine uses those fields to decide the sequence. The
command processor does not currently parse the descriptor into a generic,
recorded list of semantic operations.

This means:

- a new job that reuses the same command-processor state machine is displayed
  correctly without a workload-specific PPA sequence;
- changing state encoding or adding a different command sequence can silently
  produce missing or incorrect labels until the report mapping is updated;
- the raw internal FSM encoding has incorrectly become part of the PPA
  interface.

### Required semantic-event design

Before adding more command sequences, replace the raw-state dependency with a
stable semantic command-event contract:

```text
event_id: DESC_DECODE | PROGRAM_MOVE | INPUT_MOVE | ACC_CLEAR |
          CHUNK_LAUNCH | PREFETCH_BANK_SELECT | OUTPUT_MOVE | JOB_RETIRE
arguments: chunk_id, source_bank, destination_bank, operator_id
cycle, active, wait_reason
```

The RTL command processor emits the event that it is actually performing. The
testbench records the event ID and arguments. PPA renders those semantic events
and must not infer them from an internal state number. The event definition
must have one generated source shared by SystemVerilog and Python, or an
equivalent contract test that prevents the two definitions from drifting.

Acceptance gates:

- renumbering an internal FSM state does not change PPA labels;
- a new job using existing events needs no workload-specific timeline code;
- a new architectural action requires an explicit event definition and test;
- every displayed command action can be traced to a measured RTL event.

This trace-contract work is a prerequisite for claiming that command-list,
fusion, or generalized-shape timelines are reliable.

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

| Priority | Problem | Mechanism | Acceptance gate |
| --- | --- | --- | --- |
| P0 | PPA timelines can contain raw-FSM labels, residual control buckets, out-of-range terminal events, irrelevant empty lanes, and missing conservation checks | stable semantic command/compute-control/engine/wait events, one job interval contract, timeline conservation validator, and report contract tests | every displayed span is measured and in range; compute-cluster child spans reconcile with parent activity; FSM renumbering cannot alter report meaning |
| P1 | Expanded Softmax and Scale/Mask now use the common Scheduler, but the in-order start/done integration and large expanded program expose control and movement overhead | add typed wait reasons and valid-ready commands; measure expanded program against a generic loop candidate; reduce routing handoff cycles | both jobs show real Scheduler/engine spans, no operator-specific Compute-cluster sequence FSM, functional outputs unchanged, and before/after control/movement costs measured |
| P2 | Current attention is only functionally complete for unmasked fixed `S=8,D=8`; it cannot represent real causal prefill or tails | causal/padding/tail mask contract, generalized shape/tile lowering, buffer-driven descriptor construction | multiple `S`, `D`, head, and tail cases match the golden model without stage-specific firmware switches |
| P3 | Separate descriptor launches and SRAM-visible intermediate boundaries add control and movement overhead | grouped command list, dependency tokens, on-chip score/probability tile residency, and optional matrix-to-vector/reduction/SFU streaming | full-attention group cycles include a stated runtime policy and demonstrate reduced launch/movement cycles against the unfused path |
| P4 | Current ping-pong behavior lacks a physical scratchpad bandwidth/conflict contract | banked scratchpad model, explicit ports, allocator rules, double buffering, bank-conflict and wait-reason counters | timeline explains every compute idle interval using measured data, dependency, conflict, or backpressure reasons |
| P5 | Softmax remains a bring-up numerical path and limits correctness claims | target EXP/RECIP implementation, reviewed scale/requant widths, saturation and error bounds | masked attention output meets documented tolerance over directed and randomized cases |
| P6 | Decode behavior and KV traffic are not executable architecture evidence | decode workload suite first, then KV-cache streamer and GEMV/skinny-GEMM changes only where measured evidence justifies them | tokens/s, bytes/token, utilization, and energy/token compare baseline and proposed decode paths |
| P7 | L0 performance and modeled coefficients cannot validate final PPA tradeoffs | per-engine event-based power model followed by mapped area/timing and activity-driven power | architecture decisions cite measured performance plus clearly identified L1/L2 area and power provenance |

Recommended execution order is P0 through P7. In particular, P3 fusion must
retain an unfused reference path so its benefit and additional control/storage
cost can be measured rather than assumed.

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
- Measured expanded Softmax is `637` total cycles: program movement is `29`
  cycles, Scheduler is `114` active plus `448` engine-wait cycles, and measured
  engine activity is Vector `24`, Reduction `16`, and SFU `72` cycles.
- Next optimization work is typed wait reasons, lower routing/handoff latency,
  and measured expanded-versus-loop PPA comparison.

Compiler-expanded Softmax capacity result:

| Item | Expanded baseline | Generic counted-loop candidate |
| --- | ---: | ---: |
| Primitive/program words for fixed `S=8` | `112 + HALT = 113` | approximately `15 primitives/control + HALT = 16` |
| Program bytes | `452` | `64` |
| Fits selected `128`-word instruction memory | yes | yes |
| Practical power-of-two instruction-memory capacity | `128` words | candidate could use `16` words |
| Theoretical program transfer lower bound at current `4 words/cycle` mover width | `ceil(113/4) = 29` cycles | `ceil(16/4) = 4` cycles |
| Additional per-row loop-control cycles | none | approximately `8` cycles for fixed `S=8` |
| Additional hardware | larger instruction memory/addressing | generic loop counter/branch control |

This table is an architecture estimate, not an accepted loop decision. The
expanded program is now generated by the Compiler and is the functional source
for future execution. Before accepting a loop ISA, PPA must compare measured
expanded execution with sufficient instruction memory against a generic loop
candidate, including instruction-memory area and loop-control area.

Immediate P0 instrumentation correction:

- Attention Softmax PPA must show the measured 113-word program movement,
  Scheduler primitive issue/wait spans, and engine row/lane work.
- Scale/Mask PPA must show its measured primitive-uop program movement, eight
  Scheduler issue/wait pairs, eight Vector operations, and remaining
  Compute-cluster routing/handoff cycles.
- Their Vector, Reduction, and SFU spans must come from measured primitive
  active/op/row/lane events. Ratio-derived occupancy blocks are not acceptable.
- The current eight-row softmax serializes eight scalar EXP operations per row;
  this is expected to dominate the measured compute latency and is a concrete
  optimization target, not an unexplained report artifact.
- Rename residual `Primitive sequencer` attribution to `Compute cluster
  control`: it is internal FSM overhead, not a standalone hardware module.
- Add a timeline validator that rejects out-of-range spans, unexplained
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
| Scale/mask | measured primitive-uop program movement, eight Scheduler issue/wait pairs, eight fixed-point Vector scale operations, and routing/handoff cycles for unmasked `8x8 int32`; output feeds tile softmax | no causal/padding/tail mask; per-row routing/start/response handoff remains three cycles |
| Softmax | measured 113-word program movement, Scheduler issue/wait spans, and per-engine row/lane timeline events, L0 modeled energy | expanded program and serialized routing/wait overhead dominate; per-engine energy counters remain incomplete |
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
- later L1 mapped area/timing and L2 activity-driven power if architecture
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
| Transformer attention workloads | `docs/design/transformer/attention_workload_ppa.md` | QK, unmasked scale/mask, attention softmax, and mixed PV stage jobs execute from a generated runtime-job table with fixed-case buffer chaining | `make cpu-soc-transformer` and `make ppa-l0-report WORKLOAD_PROFILE=transformer` pass | remove fixed-shape/stage-specific runtime assumptions, then reduce launch and external-memory boundaries |

## Completed Foundations

The following foundations are complete and are not competing next-step plans:

- current NPU-core module ownership and status are documented;
- primitive valid/ready shims and local event counters have directed tests;
- the fixed `S=8,D=8` attention planner, generated runtime table, and
  QK -> scale/mask -> softmax -> PV buffer chain execute;
- unmasked Scale/Mask, tile Softmax, mixed-precision PV, and grouped
  Transformer PPA are measurable;
- PPA distinguishes measured events from modeled area/energy evidence.

All new implementation priority and acceptance gates are defined only by
`Recommended Next-Phase Plan`. The active package is P1, beginning with
Scale/Mask migration to the common Uop Scheduler and then Softmax migration.

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
| Scale/mask stage | executable measured unmasked `8x8 int32` descriptor; QK output feeds scale and scale output feeds softmax | add causal/padding/tail mask semantics |
| Softmax stage | measured eight-row tile through current vector/reduction/SFU sequence | upgrade SFU/requant/mask numerical contract and generalize tiling |
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
- Implemented in current SoC path: QK, unmasked scale/mask, softmax, and PV
  stages are separately executable and visible in PPA.
- Still deferred: generalized intermediate-buffer allocation, mask semantics,
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
