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
the CPU-to-NPU descriptor path. It is not yet evidence that a single firmware
or runtime-generated attention operation executes the full formula as one
grouped workload.

Current boundary:

- QK, softmax, and PV are launched and measured as separate descriptor jobs.
- The scale/mask boundary is explicit in the design, but the current executable
  smoke path may still materialize or pre-stage the softmax input.
- The parent `attention_prefill_s8_d8` workload remains model-only until
  compiler output drives runtime launch of the full sequence and owns the
  intermediate buffers.

Therefore the next software task is to make operators, compiler, and runtime
generate the stage sequence and runtime jobs, rather than continuing to encode
attention execution behavior inside fixture-specific code.

## Current PPA Status And Gaps

Current PPA evidence is Level 0 only:

- performance cycles and movement counters come from RTL/SoC simulation;
- normalized area is a structural proxy, not synthesized area;
- normalized energy is event/coefficient proxy, not measured power;
- external-memory energy is modeled from workload metadata.

Measured stage PPA exists for:

| Stage | Current PPA evidence | Main gap |
| --- | --- | --- |
| QK | measured cycles, data mover words, useful MACs, matrix utilization, L0 energy proxy | no grouped attention parent execution yet |
| Softmax | measured cycles through vector/reduction/SFU sequence, L0 energy proxy | vector/reduction/SFU events are not yet split into reliable per-submodule active-cycle/energy counters |
| PV | measured cycles through shared mixed matrix mode, useful MACs, matrix utilization, L0 energy proxy | mixed `u16 x s8` area/energy still uses generic MAC proxy coefficients |
| Full attention parent | model-only/group metadata | no generated runtime plan, no measured group total, no measured scale/mask cost |

Missing before claiming complete attention PPA:

- generated runtime group execution for QK -> scale/mask -> softmax -> PV;
- explicit PPA row for scale/mask bridge, measured or clearly model-only;
- group total cycles and energy provenance for `attention_prefill_s8_d8`;
- submodule counters or derived events for matrix, vector, reduction, SFU, data
  mover, and scheduler/control;
- event-energy coefficients split by `int8xint8` MAC, `u16xint8` MAC,
  vector lane op, reduction element op, SFU EXP, and SFU RECIP;
- structural area proxy split by matrix, mixed-precision multiplier delta,
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
| accumulator file | `docs/design/transformer/transformer_npu_v1.md` | `matrix/accumulator_file.sv` integrated into `npu_v0_top` | core, quick SoC, full CNN pass | expose counters through perf only after CSR plan update |
| vector engine | `docs/design/transformer/vector_engine_v1.md` | standalone `vector/vector_engine.sv` implemented | primitive op and softmax/RMSNorm sequence tests pass | connect to scheduler/uop path |
| reduction engine | `docs/design/transformer/reduction_engine_v1.md` | standalone `reduction/reduction_engine.sv` implemented | primitive op and softmax/RMSNorm sequence tests pass | broaden row-length coverage before scheduler integration |
| SFU | `docs/design/transformer/sfu_v1.md` | standalone `sfu/sfu_lut.sv` implemented | EXP/RECIP/RSQRT and sequence tests pass | refine LUT/tolerance before model accuracy claims |
| memory / scratchpad / data mover | common docs in `docs/design/` | v0 wrapper data mover exists | perf/PPA pass | add v1 internal scratchpad contract before widening |
| KV cache subsystem | `arch/specs/transformer/v1/transformer_npu_v1.md` | spec/model-only counters only | perf/PPA model-only traffic visible | no RTL until decode traffic evidence justifies it |
| Transformer attention workloads | `docs/design/transformer/attention_workload_ppa.md` | QK, attention softmax, and mixed PV stage jobs execute; full attention parent is still model-only | stage-level SoC/PPA path passes; grouped attention runtime is pending | make compiler/runtime generated attention group the main PPA surface |

## Ordered Work

### 0. Attention Sequence Contract

Status:

- Implemented in documentation: `attention_sequence_v1.md` defines attention
  as QK matrix, score scale/mask/clamp, row softmax, and PV sequence over
  existing primitives.
- Implemented in documentation: attention workload/PPA and compiler/runtime
  documents define parent/stage grouping, model-only versus measured evidence,
  and software-owned lowering.
- Implemented in current SoC path: QK, softmax, and PV stages are separately
  executable and visible in PPA.
- Still deferred: compiler-generated grouped attention runtime, executable
  scale/mask bridge, command-list ABI, and measured full-attention parent row.

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
- Model-only fields remain model-only until launched through firmware/runtime.
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
- `docs/design/transformer/sfu_exp_lut257_design.md`

Acceptance criteria:

- Generate the 257-entry Q0.15 EXP table from a deterministic numerical source
  and config scale/Q fields.
- Replace current 9-segment `bringup_exp_q15_segments` RTL path only after
  golden and RTL agree.
- Define RECIP input range, Q0.24 output, normalization shift, zero behavior,
  and test vectors.
- Keep target fixed-spec and current RTL approximation functions separately
  named.
- Attention softmax remains model-only or bring-up-labeled until this contract
  is implemented.

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
- `docs/design/transformer/requant_v2_design.md`

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

- `npu_v0_top` uses `hw/npu_core/rtl/matrix/accumulator_file.sv` for matmul and
  K-stream partial-sum residency instead of the legacy internal `acc_buf`.
- Existing `MATMUL` and `MATMUL_K_STREAM` firmware paths keep passing.
- Accumulator counters are wired into a local test or documented as internal
  module counters pending CSR exposure.

Status:

- Implemented in the current work item. `acc_buf` is removed from
  `npu_v0_top`; accumulator storage is now provided by
  `matrix/accumulator_file.sv`.

Verification:

```text
make npu-core-sim
make cpu-soc-cnn-full
make ppa-proxy-report
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
- The attention softmax smoke uses the hardwired current V1 sequence in
  `npu_v0_top.sv`; it is measured but still labeled as the bring-up numerical
  contract until SFU/RECIP are upgraded to the target fixed spec.

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
- split energy proxy coefficients for `int8xint8` MAC and `u16xint8` mixed MAC;
- split structural area proxy by matrix, vector, reduction, SFU, buffers, and
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
- No fused attention pipeline.
- No macro-op hardware expansion.
- No dedicated attention RTL macro.
- No INT4/FP8.
- No real LPDDR controller.
- No CNN regression deletion.
