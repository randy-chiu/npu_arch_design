# Transformer V1 Test Plan

## Purpose

Transformer V1 verification proves that attention-oriented workloads can be
generated, executed, checked, and reported without adding a dedicated attention
RTL macro. The test plan keeps three views separate:

- mathematical golden behavior;
- current simplified RTL bring-up behavior;
- final target numerical behavior.

The current acceptance target is not full Transformer model accuracy. It is a
measurable, repeatable CPU-to-NPU RTL path for the attention stages that current
hardware can execute, with explicit labels for every approximation.

## Verification Targets

| Target | Current status | Main evidence |
| --- | --- | --- |
| QK score tile | CPU-to-NPU RTL measured | `transformer_attention_qk_s8_d8` |
| attention row softmax | CPU-to-NPU RTL measured V1 primitive path | `transformer_attention_softmax_s8` |
| PV weighted sum | CPU-to-NPU RTL measured shared mixed matrix path | `transformer_attention_pv_s8_d8` |
| executable causal score scale/mask | CPU-to-NPU RTL measured fixed-point vector scale plus lane mask-select | `transformer_attention_scale_mask_s8_d8` |
| full attention group | software-sequenced measured stage grouping | `transformer_attention_prefill_s8_d8` |
| target Q0.15 softmax | Python golden and standalone primitive sequence | `attention_numerical_v1.md`, primitive TB |
| mixed Q0.15-by-int8 PV | CPU-to-NPU RTL measured for `S=8,D=8` | `matmul_u16s8_q15` |

## Workload Constraints

Current CPU-to-NPU RTL tests must obey these constraints:

| Constraint | Reason |
| --- | --- |
| matrix executable stages use `8x8x8` or K-stream multiples of 8 | current matrix array and fixture generator only support full 8-wide physical tiles |
| attention QK uses `Q int8 * K^T int8 -> int32 scores` | maps exactly to the current matrix datapath |
| attention softmax measured path uses an `8x8` int32 score tile and Q0.15 outputs | current descriptor loops over eight V1 primitive softmax rows |
| softmax measured row count is eight rows | current fixed-shape descriptor consumes and produces one complete tile |
| PV measured path uses Q0.15 probabilities and int8 V | current matrix path has a mixed `u16s8_q15` mode |
| causal `S=8,D=8` is measured; tail/all-invalid physical rows are not claimed | current Scheduler still issues all eight physical rows |
| model-only rows have zero measured cycles | zero cycles must not be interpreted as measured hardware performance |

## Numerical Contracts

Every test vector must declare one contract:

| Contract | Meaning | Allowed use |
| --- | --- | --- |
| `attention_bringup_v0_qk_exact` | exact int8 dot product into int32 | measured QK |
| `attention_bringup_v0_shift_scale_sfu9seg` | simplified softmax/SFU bring-up behavior | measured or standalone softmax bring-up |
| `attention_numerical_v1_q15_prob_q24_recip_lut257` | Q0.15 softmax and mixed Q0.15-by-int8 PV contract | golden and measured PV RTL |

The golden, fixture generator, firmware expected data, and PPA metadata must
all use the same contract string. A test must fail if a measured workload loses
its `attention_stage`, `stage_provenance`, or `numerical_contract` metadata.

## Test Levels

### Python Golden And Fixture Tests

Goals:

- check QK score generation and deterministic scaling examples;
- check target Q0.15 softmax intermediate values;
- check target Q0.15-by-int8 PV golden;
- check executable/model-only workload classification;
- check generated firmware manifest job ordering and metadata.

Command:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_transformer_micro_fixtures test.rtl.test_perf_report -v
```

### Standalone Primitive RTL Tests

Goals:

- verify vector/reduction/SFU primitive modules still pass their own contract;
- verify the attention softmax primitive sequence:
  `reduce max -> subtract -> clamp -> EXP -> reduce sum -> RECIP -> scale`;
- keep this separate from CPU-to-NPU claims until runtime can launch these
  primitive engines directly.

Command:

```text
make primitive-engines-sim
```

### CPU-To-NPU RTL Tests

Goals:

- boot firmware on the PicoRV32 SoC simulation;
- copy workload tensors/programs into SoC SRAM;
- launch NPU jobs through the opsched descriptor path;
- check output buffers against generated expected data;
- emit architectural `PERF_JOB` records.

Transformer V1 executable jobs in the transformer profile:

| Job | Op type | Checks |
| --- | --- | --- |
| `transformer_prefill_gemm_tiny` | `matmul_k_stream` | int32 tile |
| `transformer_attention_qk_s8_d8` | `matmul_k_stream` | int32 score tile |
| `transformer_attention_scale_mask_s8_d8` | `attention_scale_mask_v1` | exact scaled int32 score tile |
| `transformer_attention_softmax_s8` | `attention_softmax_v1` | Q0.15 row output within tolerance |
| `transformer_attention_pv_s8_d8` | `matmul_u16s8_q15` | int32 mixed PV tile |
| `transformer_decode_skinny_gemm_m8_compat` | `matmul_k_stream` | int32 tile |

Command:

```text
make perf-l0-transformer
```

### PPA Contract Tests

Goals:

- expose attention stage, provenance, and numerical contract fields in perf and
  PPA reports;
- derive QK/PV useful MACs from logical shape;
- report softmax as measured cycles but zero matrix MACs;
- report the full attention group as `software_group_measured_stages` when the
  generated runtime table launches measured QK, scale/mask, softmax, and PV
  stages; keep incomplete buffer chaining explicit.

Command:

```text
make ppa-l0-report WORKLOAD_PROFILE=transformer
```

## Current Acceptance Criteria

The current implementation is accepted only if:

- `make test` passes;
- `make ppa-l0-report WORKLOAD_PROFILE=transformer` passes schema validation;
- QK, scale/mask, softmax, and PV appear as `transformer_micro` workloads in
  `build/ppa/ppa.json`;
- measured attention stage fields are non-null where appropriate:
  `qk_cycles` for QK, `attention_softmax_cycles` for softmax, and `pv_cycles`
  for PV;
- the full attention parent is clearly labeled as a software-sequenced measured
  stage group, with incomplete buffer chaining and runtime-overhead policy
  visible in metadata.

## Deferred Verification

The following are deliberately not accepted in the current simplified version:

- target higher-accuracy Q0.15 softmax measured through CPU-to-NPU runtime;
- reciprocal/EXP target LUT accuracy in SoC RTL;
- causal and padding mask behavior;
- rounded/saturating mixed Q0.15 probability by int8 value PV output;
- multi-row softmax launched as one descriptor;
- a single grouped attention descriptor that sequences QK, softmax, and PV with
  explicit intermediate buffers.

These become the next acceptance targets after the vector/reduction/SFU runtime
issue path and reviewed PV probability policy are implemented.

### Mask And Larger-Attention Acceptance Progression

The current Transformer profile has no executable mask behavior:

- `transformer_attention_scale_mask_s8_d8` declares `mask_policy=none`;
- generated Scale/Mask expected data applies scaling only;
- no packed row-mask table is emitted to firmware;
- no descriptor points `input1` at mask data;
- current RTL drives all vector lanes valid.

Mask implementation and later shape generalization add these tests in order:

| Workload | Required proof |
| --- | --- |
| causal `S=8,D=8` | future key lanes do not affect max/sum/probability/PV |
| padding `S=8,D=8,valid_k=5` | padded lanes produce exact zero probability |
| tail `S=5,D=8` | invalid query rows and key lanes are not treated as valid |
| K-stream `S=8,D=16` | larger head dimension accumulates QK correctly |
| multi-key-tile `S=16,D=8` | segmented Softmax max/sum matches golden |
| multi-axis `S=16,D=16` | Compiler, Runtime, buffers, QK, Softmax, and PV compose end to end |

For each masked workload, PPA must expose mask-table movement, lane-mask
control cost, valid versus masked Reduction element operations, skipped future
tiles, and comparison against a CPU-materialized fallback.
