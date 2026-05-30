# Transformer NPU v1 Next Steps

## Goal

Prepare the next implementation rounds after the directory cleanup. The
priority is still a verifiable Transformer-oriented NPU baseline, not a full
LLM or fused attention pipeline.

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
| Transformer micro workloads | `docs/design/transformer/workloads.md` | executable tiny matmul plus model-only/golden entries and primitive sequence golden | quick profile enters perf/PPA; primitive sequences pass standalone RTL | connect primitive sequences to scheduler later |

## Ordered Work

### 0. Primitive Contract Cleanup

Status:

- Implemented: Transformer primitive dimensions, local op encodings, current
  SFU bring-up LUT constants, and RECIP/RSQRT Q formats are sourced from
  `arch/configs/npu_transformer_v1.jsonc`.
- Implemented: `make transformer-config` emits SV/C/Python constants under
  `build/generated/`.
- Implemented: `hw/npu_core/rtl/transformer_primitive_engines.sv` is the
  design-side primitive integration point and explicitly passes generated
  parameters into vector/reduction/SFU RTL.
- Still deferred: production SFU LUT/Newton, valid/ready pipeline, full
  requant.

### 0.1 SFU EXP 257-entry LUT Expansion

Detailed design: `docs/design/transformer/sfu_exp_lut257_design.md`.

Acceptance criteria:

- Generate the 257-entry Q0.15 EXP table from the numerical spec source and
  config scale/Q fields.
- Replace current `bringup_exp_q15_segments` usage in RTL with the generated
  257-entry table.
- Keep `softmax_reference_*`, `softmax_fixed_spec_*`, and
  `softmax_rtl_model_*` separate while the old and new SFU models coexist.
- Update README status from 9-segment bring-up to 257-entry implemented only
  after RTL/golden/tests match.

### 0.2 Valid/Ready and Counter Expansion

Detailed design: `docs/design/transformer/primitive_valid_ready_v1.md`.

Acceptance criteria:

- Add an engine-level issue/accept/done contract before scheduler integration:
  `valid`, `ready`, `done`, and stable input hold rules.
- Define active/stall/idle counter increments in the spec before adding CSRs.
- Preserve the existing single-start bring-up tests as compatibility tests
  until the scheduler path consumes the valid/ready interface.

### 0.3 Requant v2 Expansion

Detailed design: `docs/design/transformer/requant_v2_design.md`.

Acceptance criteria:

- Add a mode field or distinct op contract for `mul_round_shift_clamp`.
- Extend config/spec/golden with multiplier width, rounding mode, shift, clamp,
  and optional zero-point behavior.
- Keep current `shift_clamp` as a named v1 mode with regression coverage.
- Only switch Transformer fixed-spec softmax/RMSNorm paths to v2 requant after
  RTL-like golden and primitive RTL tests agree.

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

### 3. Softmax/RMSNorm Primitive Micro-Kernel Fixtures

Acceptance criteria:

- `softmax_row` and `rmsnorm_row` have generated fixture data and expected
  outputs.
- The workload manifest keeps these as `model_only` or `primitive_rtl_unit`
  until firmware execution exists.
- Perf/PPA fields stay `null` for unavailable runtime cycles rather than
  pretending they are measured.

Status:

- Implemented for standalone primitive sequence validation.
- Python golden functions:
  - `softmax_row_primitive_lut_q15()`;
  - `rmsnorm_primitive_sequence()`.
- RTL sequence coverage lives in `hw/npu_core/tb/primitive_engines_tb.sv`.
- These are not yet firmware-executable workloads and do not report measured
  runtime cycles in perf/PPA.

### 4. GEMV / Skinny GEMM Execution Path

Acceptance criteria:

- Add a true `GEMV_TILE` or valid-row/valid-column skinny-GEMM path.
- Report distinguishes:
  - full tile GEMM;
  - skinny GEMM;
  - GEMV.
- `effective_mac_ops`, `peak_mac_capacity`, `matrix_utilization`,
  `gemv_utilization`, and `tail_waste_mac_capacity` are still visible in
  perf/PPA output.

### 5. KV Cache Counter Path

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
- No INT4/FP8.
- No real LPDDR controller.
- No CNN regression deletion.
