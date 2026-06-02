# Attention Runtime v1

## Scope

This document defines how CPU firmware and host-side tooling launch a
compiler-produced Transformer attention plan on the current SoC/NPU wrapper. It
focuses on runtime sequencing, descriptor generation, intermediate buffers, and
PPA capture.

Runtime includes:

- host-side firmware data emission;
- CPU firmware staging and NPU descriptor launch;
- NPU wrapper descriptor ABI usage;
- perf/PPA collection for attention stages and groups.

## Current Runtime State

Implemented today:

- CPU firmware launches descriptor jobs through `soc_npu_job_desc_t`;
- QK launches through `SOC_NPU_JOB_OP_MATMUL_K_STREAM`;
- attention softmax launches through `SOC_NPU_JOB_OP_ATTENTION_SOFTMAX_V1`;
- PV launches through `SOC_NPU_JOB_OP_MATMUL_U16S8_Q15`;
- firmware data generation emits a compiler-produced runtime-job table for the
  current attention group;
- CPU firmware iterates that runtime-job table to launch QK, softmax, and PV in
  compiler order;
- firmware checks each stage output;
- perf report captures per-job counters and attention stage metadata.

Current limitation:

```text
Scale/mask is still materialized by fixture data, not an executable NPU stage.
```

The parent full-attention workload can now be reported as
`software_group_measured_stages` for measured QK/softmax/PV stage execution,
but it is not target full-attention numerical/PPA evidence until scale/mask is
executable or explicitly accounted with measured runtime overhead.

## Runtime Responsibilities

The runtime owns:

- mapping compiler `runtime_jobs[]` to descriptor structs;
- copying or referencing tensors/programs in SRAM;
- launching jobs in compiler order;
- enforcing stage barriers;
- checking status and timeouts;
- collecting perf snapshots per stage;
- reporting group-level PPA metadata;
- preserving descriptor ABI compatibility.

The runtime does not own attention mathematical lowering, Q/K/V layout
decisions, scale policy, mask semantics, or golden values. Those belong to the
operator, compiler, and numerical specs.

## Launch Models

### Model A: Multi-Descriptor Attention Sequence

This is the next implementation target.

```text
CPU runtime receives AttentionPlan
  -> launch QK descriptor and wait
  -> run or materialize scale/mask bridge
  -> launch softmax descriptor and wait
  -> launch PV descriptor and wait
  -> collect stage perf snapshots
  -> report grouped attention result
```

Benefits:

- reuses current wrapper descriptor ABI;
- no command-list RTL is required;
- makes software stack real without destabilizing current RTL;
- keeps stage PPA comparable to existing reports.

Limits:

- CPU launch overhead is outside the NPU completed-job snapshot;
- intermediate tensors round-trip through SRAM/staging buffers;
- group execution is software-sequenced, not a hardware command list;
- full attention parent PPA must state whether runtime overhead is included.

Model A acceptance criteria:

- the compiler emits one `AttentionPlan` with ordered `runtime_jobs[]`;
- firmware/data generation consumes that plan instead of hard-coding stage
  execution behavior in fixture-specific smoke code;
- QK, scale/mask, softmax, and PV all appear as stage entries even when
  scale/mask is materialized or model-only;
- parent group metadata records whether group cycles are the sum of measured
  stage snapshots, measured stage snapshots plus CPU runtime overhead, or still
  model-only;
- intermediate buffers have typed producer/consumer metadata.

The parent `attention_prefill_s8_d8` may move from `model_only_full_attention`
to `software_group_measured_stages` only after these criteria are met.

### Model B: Grouped Command-List Descriptor

This is a later target.

```text
CPU runtime stages one attention command list
  -> launches one descriptor
  -> wrapper/scheduler walks primitive commands
  -> perf scopes attribute QK/softmax/PV internally
```

Required before Model B:

- command-list ABI;
- wrapper primitive scheduler;
- intermediate scratchpad or explicit SRAM buffer references;
- perf scope counters inside one descriptor execution.

## Model A Runtime ABI

Minimum runtime job fields:

| Field | Meaning |
| --- | --- |
| `job_id` | numeric job ID emitted from manifest/config |
| `job_name` | stable report name |
| `attention_group` | parent group key |
| `attention_stage` | `qk`, `scale_mask`, `softmax`, `pv`, or `full_attention` |
| `descriptor_op` | SoC descriptor op enum name |
| `program` | optional program words for uop-based ops |
| `input0` | SRAM buffer symbol/address |
| `input1` | SRAM buffer symbol/address or zero |
| `output` | SRAM buffer symbol/address |
| `input0_words` | descriptor word count |
| `input1_words` | descriptor word count |
| `output_words` | descriptor word count |
| `k_chunks` | K-stream chunk count when used |
| `check_policy` | exact, absolute tolerance, model-only, or none |
| `perf_scope` | report grouping key |

Descriptor op mapping:

| Compiler `descriptor_op` | Firmware enum | Core behavior |
| --- | --- | --- |
| `matmul_k_stream` | `SOC_NPU_JOB_OP_MATMUL_K_STREAM` | int8 matrix K-stream |
| `attention_softmax_v1` | `SOC_NPU_JOB_OP_ATTENTION_SOFTMAX_V1` | vector/reduction/SFU softmax sequence |
| `matmul_u16s8_q15` | `SOC_NPU_JOB_OP_MATMUL_U16S8_Q15` | mixed probability-value matrix |

## Runtime Attention Sequence

### QK Stage

Input staging:

```text
input0 = Q tile, int8 packed in SRAM words
input1 = K^T tile, int8 packed in SRAM words
program = matmul program when required
```

Descriptor:

```text
op_type      = MATMUL_K_STREAM
input0_addr  = q_tile
input1_addr  = k_t_tile
output_addr  = score_raw
k_chunks     = compiler planned chunk count
```

Output and check:

```text
score_raw_i32[8,8], exact int32 check
```

### Scale/Mask Stage

Current first runtime step:

- may be materialized by fixture data for the smoke workload;
- must still exist in compiler plan and generated metadata.

Target Model A step:

```text
input0 = score_raw_i32
input1 = mask or scale/mask metadata
output = score_softmax_in_i32
op_type = future vector/requant/mask descriptor or command-list primitive
```

This boundary is required because:

- QK produces unscaled dot products;
- attention requires division by `sqrt(D_k)`;
- decode and padding correctness require mask handling;
- softmax should not silently own score scaling and mask semantics.

Until executable, runtime marks it:

```text
execution = materialized_or_model_only_bridge
```

and PPA must not count it as measured NPU compute.

Executable bridge acceptance criteria:

- unmasked `S=8,D=8` scale path is generated from compiler metadata;
- if the bridge is CPU/materialized, report provenance is `model_only` or
  `materialized_by_fixture`, not measured NPU;
- if launched on NPU, descriptor/runtime job metadata names the vector/requant
  op sequence and exposes a measured stage snapshot;
- causal, padding, and tail masks are not claimed until invalid-lane behavior
  is tested against the numerical contract.

### Softmax Stage

Input staging:

```text
input0 = one scaled/masked score row or tile
```

Descriptor:

```text
op_type      = ATTENTION_SOFTMAX_V1
input0_addr  = score_softmax_in
output_addr  = prob_q15
```

Output and check:

```text
prob_q15_u16[8], absolute tolerance from numerical contract
```

If the full `8x8` probability matrix is tested row-by-row, runtime must emit
eight softmax jobs or use a future row-loop descriptor/command list.

### PV Stage

Input staging:

```text
input0 = P probability tile, uint16 Q0.15
input1 = V tile, int8
```

Descriptor:

```text
op_type      = MATMUL_U16S8_Q15
input0_addr  = prob_q15_tile
input1_addr  = v_tile
output_addr  = o_i32
k_chunks     = compiler planned sequence chunks
```

Output and check:

```text
o_i32[8,8], exact against current truncating shift15 fixed spec
```

When RTL changes to rounded/saturating PV output, runtime checks and golden must
move together under a new numerical contract.

## Firmware Data Generation

Current file:

```text
sw/tools/firmware/emit_soc_cpu_smoke_data.py
```

Target behavior:

1. Read fixture tensors and compiler `AttentionPlan`.
2. Emit SRAM arrays for each logical buffer.
3. Emit generated runtime job descriptors or descriptor setup data.
4. Emit expected outputs and check policies from the plan.
5. Keep tensor generation separate from execution planning.

Generated artifacts should make the firmware smoke app less attention-specific.
The C firmware should iterate generated jobs where possible instead of adding a
new hand-written `run_transformer_attention_*` function for every stage.

## CPU Firmware Runtime

Current files:

```text
sw/soc_cpu/runtime/npu_driver.c
sw/soc_cpu/runtime/npu_driver.h
sw/soc_cpu/apps/soc_cpu_smoke/main.c
```

Near-term additions:

- generated `npu_runtime_job_t` representation;
- helper to fill `soc_npu_job_desc_t` from a runtime job;
- helper to launch one job and read perf snapshot;
- helper to launch all jobs in an attention group in order;
- timeout support around `npu_wait_done`;
- per-stage check dispatch based on `check_policy`.

Pseudo-code:

```c
for each runtime_job in attention_group.jobs:
    stage_inputs(runtime_job)
    fill_soc_npu_job_desc(&desc, runtime_job)
    npu_set_desc_addr(ptr32(&desc))
    npu_start()
    if (!npu_wait_done_timeout(timeout_cycles)):
        fail(runtime_job.job_id, TIMEOUT)
    npu_read_perf_snapshot(&snapshot)
    record_perf(runtime_job.perf_scope, snapshot)
    if (!check_output(runtime_job)):
        fail(runtime_job.job_id, MISMATCH)
```

The current implementation uses a generated runtime-job table for the attention
group. More generic descriptor filling is still pending for additional shapes
and operators.

## Intermediate Buffer Contract

Model A uses SRAM-resident buffers between descriptor jobs.

Required metadata per buffer:

| Field | Meaning |
| --- | --- |
| `name` | logical buffer name |
| `dtype` | int8, int32, uint16_q0.15 |
| `shape` | logical dimensions |
| `layout` | row-major, transposed, packed words |
| `word_count` | number of 32-bit words staged |
| `producer_stage` | producing stage or input |
| `consumer_stages` | consuming stages |
| `lifetime` | first/last stage index |

Initial packing rules:

- int8 tensors use existing 32-bit word packing path;
- int32 tensors are one element per word;
- uint16 Q0.15 tensors must not be truncated to 8 bits;
- mixed PV input0 must preserve 16-bit probability values through SRAM and
  wrapper/core staging.

## Perf And PPA

Runtime must preserve per-stage measured rows:

```text
attention_qk_s8_d8
attention_softmax_s8
attention_pv_s8_d8
```

The parent group row:

```text
attention_prefill_s8_d8
```

should transition through these states:

| State | Meaning |
| --- | --- |
| `model_only_full_attention` | parent is metadata/golden only; stages may be measured separately |
| `software_group_measured_stages` | compiler/runtime launches grouped sequence; group total is sum of measured stage jobs, with CPU overhead reported separately or explicitly excluded |
| `command_list_measured_full_attention` | one descriptor/command list executes the full group with internal perf scopes |

PPA requirements:

- stage provenance says whether each stage is measured or model-only;
- scale/mask bridge is visible even if not measured;
- group total does not silently mix model-only and measured stages;
- runtime overhead is explicitly included or excluded.
- full attention parent state transitions are monotonic:
  `model_only_full_attention` -> `software_group_measured_stages` ->
  `command_list_measured_full_attention`;
- if any stage in a group is model-only, the parent group cannot be labeled as
  fully measured.

## Verification

Runtime tests:

- generated runtime job table maps descriptor op names to firmware enums;
- QK, softmax, and PV launch in compiler order;
- buffers consumed by a stage were produced by an earlier stage or input;
- uint16 Q0.15 probability buffer is preserved for PV;
- per-stage perf snapshots retain job ID and op type;
- failed checks report failing job/stage identity.

SoC tests:

- existing CPU-controlled SoC smoke continues to pass;
- transformer attention QK/softmax/PV pass when launched from generated runtime
  metadata;
- PPA report contains the same stage rows as before;
- parent group uses `software_group_measured_stages` only when the runtime-job
  table launches the measured stage jobs in compiler order.

Regression tests:

- MNIST/CNN descriptor jobs are unchanged;
- legacy matmul/softmax operator smoke remains valid;
- descriptor ABI changes fail schema tests unless docs/config/firmware/RTL are
  updated together.

## Iteration Plan

### Runtime v1.0: Generated Multi-Descriptor Stage Jobs

Deliverables:

- compiler emits runtime jobs for current attention smoke;
- firmware data emitter consumes runtime jobs;
- existing hand-written stage functions can still launch them;
- PPA stage rows remain measured.

Trigger: after operator and compiler docs are reviewed.

### Runtime v1.1: Table-Driven Firmware Dispatch

Deliverables:

- firmware iterates generated job table;
- descriptor filling is generic for matmul, attention softmax, and mixed PV;
- output check policy is table-driven.

Trigger: before adding more Transformer operators or more attention shapes.

### Runtime v1.2: Executable Scale/Mask Bridge

Deliverables:

- score scale and mask stage is launched or generated as an executable primitive
  sequence;
- causal/padding/tail masks are tested;
- PPA reports scale/mask cost instead of hiding it in fixture generation.

Trigger: before claiming full attention numerical correctness.

### Runtime v2: Command-List Group Execution

Deliverables:

- one attention group descriptor points to a command list;
- wrapper/core scheduler executes QK -> scale/mask -> softmax -> PV;
- internal perf scopes report stage counters;
- parent `attention_prefill_s8_d8` becomes measured full attention.

Trigger: when CPU launch overhead or SRAM round-trip traffic dominates
attention PPA, or before claiming production-like full attention execution.
