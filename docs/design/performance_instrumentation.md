# Performance Instrumentation Design

[TOC]

This document describes how cycle-level performance data is collected and
rendered today.

## 1. Goal

The performance system should show where time goes in a CPU-launched NPU job:

```text
CPU firmware start/poll
NPU wrapper descriptor and control phases
Data mover load/store phases
NPU core execution phases
```

The main UI requirement is a timeline where the x-axis is cycle time and the
y-axis is module/resource. It should make overlap, waiting, and wasted time
visible.

## 2. Current Collection Method

The report source is the CPU-visible completed-job snapshot implemented in
`hw/npu_wrapper/rtl/npu_v0_opsched.sv`. Firmware reads it through MMIO after
every descriptor job. `hw/soc/tb/soc_cpu_tb.sv` serializes those actual bus
read responses into `PERF_JOB` JSON records labeled
`architectural_perf_csr_snapshot`.

The TB continues to accumulate a minimal hierarchical event reference solely
to assert that the CSR implementation matches the event sources. It is no
longer a production report data source.

## 3. Job Boundary

Start condition:

```text
CPU writes NPU_OPSCHED_CTRL with bit 0 set
```

End condition:

```text
wrapper reaches completion, publishes a snapshot, and firmware reads it
```

Current measured interval starts at NPU launch, not at CPU/DMA input staging.
Copying tensors/programs into SRAM and checking outputs are outside the current
cycle report.

当前测量区间从 NPU launch 开始，不包含 CPU/DMA input staging。tensor/program
搬到 SRAM 以及 firmware 检查输出都不在当前 `PERF_JOB` cycle 统计内。

## 4. Sampled Signals

| Counter group | Architectural source |
| --- | --- |
| job identity | `PERF_JOB_ID`, `PERF_OP_TYPE` |
| total/core cycles | `PERF_TOTAL_CYCLES`, `PERF_CORE_*_CYCLES` |
| data mover cycles/words | `PERF_DATA_MOVER_*` |
| data mover direction words | `PERF_DATA_MOVER_READ_WORDS`, `PERF_DATA_MOVER_WRITE_WORDS` |
| SRAM boundary traffic | `PERF_SRAM_READ_WORDS`, `PERF_SRAM_WRITE_WORDS` |

The testbench prints one JSON line:

```text
PERF_JOB {"source":"architectural_perf_csr_snapshot", ...}
```

`sw/tools/perf/report.py` parses these lines and generates:

```text
build/perf/perf.json
build/perf/perf_report.html
```

For Transformer-oriented v1 reporting, `report.py` also joins each workload
with manifest shape metadata and derives analysis fields that are not yet all
hardware CSRs:

| Field | Source |
| --- | --- |
| `effective_mac_ops` | manifest logical shape, usually `M*N*K` for measured executable jobs |
| `peak_mac_capacity` | `matrix_active_cycles * peak_macs_per_cycle` |
| `matrix_utilization` | derived from useful MACs over peak capacity |
| `gemv_utilization` | same utilization only for `M=1` or `N=1` shapes |
| `skinny_gemm_utilization` | same utilization for skinny current-array-compatible shapes |
| `kv_read_bytes` / `kv_write_bytes` | manifest external-memory fields |
| `bytes_per_token` | manifest/model-only KV traffic normalization when available |

When a workload is model-only or a hardware engine is not implemented yet, the
corresponding utilization/cycle field is `null` or zero rather than reported as
measured hardware behavior.

### Current Code Walkthrough

`make perf-report` does not run a separate profiler. It rebuilds the
CPU-controlled SoC simulation, redirects simulator stdout to
`build/perf/cpu_soc_perf.log`, then runs `sw/tools/perf/report.py`.

#### Data Mover Event Source

`hw/npu_wrapper/rtl/npu_v0_data_mover.sv` exports the event signals consumed by
both collection paths:

| Signal | Current meaning | Consumer |
| --- | --- | --- |
| `perf_active` | mover is busy or accepting `start` | CSR and TB |
| `perf_setup` | setup cycle is active | CSR and TB |
| `perf_transfer` | one transfer beat is active | CSR and TB |
| `perf_words` | words transferred by the active beat | CSR and TB |
| `perf_stall` | reserved stall event; currently tied to zero | CSR and TB schema |

These signals are not obsolete because CSRs exist. They are the RTL event
source from which the CSR data-mover snapshot fields are accumulated.
`perf_stall` is intentionally retained as a stable zero-valued field until
backpressure/stall behavior is implemented or the public counter schema is
revised.

#### Wrapper CSR Aggregation

`hw/npu_wrapper/rtl/npu_v0_opsched.sv` implements the software-readable
completed-job snapshot:

1. A write to `NPU_OPSCHED_CTRL.start` asserts `perf_start_event`, sets
   `perf_running`, and clears the private `perf_work_*` accumulator bank. The
   previous `perf_snap_*` values remain readable until completion or explicit
   idle clear.
2. While `perf_running` is set, the wrapper saturating-adds total cycles,
   selected core cycles, data-mover events, and SRAM boundary words into
   `perf_work_*`; any saturated increment latches working overflow.
3. On `DESC_DONE`, or on a legacy idle-path `npu_done`, `perf_complete_event`
   atomically copies working counters to `perf_snap_*`, asserts valid, and
   exposes overflow.
4. The MMIO read mux returns `PERF_STATUS` and `PERF_*` snapshot registers from
   the generated register offsets.

The core now exports explicit `perf_active`, `perf_fetch_active`,
`perf_matmul_active`, and `perf_done_active` events. The CSR bank consumes
these events rather than comparing a copied numeric internal state encoding.
This makes legacy direct-window and descriptor launches use the same core
counter semantics; the legacy MMIO smoke checks nonzero matmul cycles.

#### Firmware Readout, Testbench Serialization And Correlation

Inside the current flow:

1. Firmware launches a descriptor and waits for completion.
2. Firmware calls `npu_read_perf_snapshot()` and reads summary, identity and
   directional mover CSRs through CPU-visible MMIO.
3. `soc_cpu_tb` observes those bus read responses and emits one `PERF_JOB`
   record only when the snapshot read sequence is complete.
4. Independently, a small TB reference accumulator checks the internal
   snapshot implementation for total/core/mover/SRAM summary equivalence.

Detailed wrapper FSM and inferred host-window phases are no longer emitted as
formal production data. A future fine-grain timeline must come from a reviewed
architectural event/trace contract rather than restored ad hoc observation.

#### Report Post-Processing

`report.py` parses `PERF_JOB ` records, adds analytical estimates, and writes
JSON and HTML. It can reconstruct timeline spans for legacy records containing
old TB phase counters; the production
build supplies `build/perf/workload_manifest.json`, so workload grouping uses
explicit `job_id` declarations rather than job order. `infer_workloads()` is
still retained as a warned legacy-log fallback and its tests still encode the
historical fixed job ordering.

The generated manifest describes 53 jobs / 7 workloads when only the earlier
single full-`fc1`-tile external fixture is present:

```text
operator_smoke_matmul: 1 job
operator_smoke_softmax: 1 job
digits_linear_classifier: 16 jobs
real_mnist_cnn_fc1_tile0: 1 job
real_mnist_cnn_fc1_k_stream_smoke: 1 job
real_mnist_cnn_fc1_full_k_stream_tile0: 1 job
real_mnist_cnn_fc2: 32 jobs
```

After full `fc1` was extended from one output N tile to all 16 output N tiles,
the active generated manifest describes 68 jobs / 7 workloads:

```text
operator_smoke_matmul: 1 job
operator_smoke_softmax: 1 job
digits_linear_classifier: 16 jobs
real_mnist_cnn_fc1_tile0: 1 job
real_mnist_cnn_fc1_k_stream_smoke: 1 job
real_mnist_cnn_fc1_full_k_stream_layer: 16 jobs
real_mnist_cnn_fc2: 32 jobs
```

The full `fc1` K-stream tile is a single descriptor with `k_chunks=1152`.
Current measured wrapper counters for that job are still dominated by movement,
but A/B ping-pong buffering now overlaps the next K-chunk prefetch with current
chunk execution. The NPU-side SRAM/data-mover/core-host path moves four words
per cycle:

```text
total_cycles: 39218
input0_words: 73728
input1_words: 73728
fetch_input0 cycles: 16
fetch_input1 cycles: 16
core matmul cycles: 11520
data_mover transfer cycles: 36884
data_mover read_words: 147472
data_mover write_words: 64
```

完整 `fc1` K-stream tile 是一个 `k_chunks=1152` 的 descriptor。当前测得的 wrapper
计数仍主要由搬运主导，但 A/B ping-pong buffer 已经把下一 K chunk 的预取与当前
chunk 执行重叠起来。NPU 侧 SRAM/data-mover/core-host 路径当前每拍搬运 4 word。

Important consequence: the stable summary values in the current report and
PPA proxy are values consumed through wrapper snapshot CSRs. The TB equality
check is now an implementation regression, not the measurement provenance.

The SoC now has a ROM-to-SRAM DMA for firmware staging. This reduces
CPU-controlled simulation finish time, but it does not change the NPU job
cycle counts above because those jobs are timed only after firmware starts the
NPU wrapper.

SoC 当前已有 ROM-to-SRAM DMA 用于 firmware staging。它会降低 CPU-controlled
simulation 的整体 finish time，但不会改变上面的 NPU job cycle，因为这些 job 只从
firmware 启动 NPU wrapper 后开始计时。

## 5. Current JSON Shape

Each job has:

| Field | Meaning |
| --- | --- |
| `job_id` | Stable manifest join key; current SoC run retains one-based numbering. |
| `id` | Legacy alias retained for old report consumers. |
| `name` | `matmul`, `softmax`, or `unknown` |
| `total_cycles` | measured launch-to-wrapper-done cycles |
| `wrapper` | legacy-only wrapper phase cycles, when present |
| `core` | core phase cycles |
| `movement` | legacy-only detailed SRAM/core-host phase counters, when present |
| `data_mover` | explicit counters sampled from `npu_v0_data_mover` |
| `estimates` | matmul compute model, if applicable |
| `movement_estimates` | burst-style movement model |
| `timeline` | legacy-only reconstructed phase lanes, when inputs exist |

The schema is intentionally small and additive. New counters should be added as
nested fields rather than breaking existing fields.

The manifest contract and mismatch behavior are specified in
`docs/design/workload_manifest.md`. The CPU-visible counter contract and
validation reference boundary are specified in `docs/design/perf_counter_csr_plan.md`.

## 6. Timeline Reconstruction

The report builds lanes:

| Lane | Current source |
| --- | --- |
| `CPU firmware` | synthetic start + poll/wait span |
| `NPU wrapper` | wrapper phase counters |
| `Data mover` | currently reconstructed from wrapper movement phases |
| `NPU core` | core phase counters offset to wrapper core-wait position |

For legacy phase-rich records, the data mover lane is placed on wrapper
fetch/write phases and `matmul_k_stream` records can render a `K prefetch
overlap` span. Production CSR snapshots establish aggregate cycle/word
reduction only. A future report step requires architectural trace events before
asserting fine-grain overlap spans.

旧 phase-rich 记录仍可按 wrapper fetch/write phase 回放 Data mover timeline
与 `K prefetch overlap` span。正式 CSR snapshot 目前只声明 aggregate
cycle/word 结果；若后续需要正式展示重叠子阶段，应先定义架构化 trace event。

## 7. Report Panels

Legacy phase-rich HTML panels can include:

- summary metrics;
- cycle timeline;
- matmul compute model;
- movement model;
- wrapper phase timeline;
- data mover phase timeline;
- core phase timeline;
- raw JSON.

Production CSR-sourced reports show summary/highlight information and keep
per-job details compact; complete raw JSON remains available for audit. The
FC1 overlap
highlight obtains its named serial comparison baseline from workload-manifest
metadata instead of embedding that baseline in rendering logic.

## 8. Movement Model

The current movement model estimates a burst data mover:

```text
ideal_burst_cycles = ceil(total_words / 4)
conservative_burst_cycles =
  sum(ceil(segment_words / 4)) + active_segments * 1 setup cycle
```

For the current `WORDS_PER_CYCLE=4`, `SETUP_CYCLES=0` RTL path, the measured
matmul movement phases are:

```text
fetch_program: 4 cycles
fetch_input0: 16 cycles
fetch_input1: 16 cycles
write_output: 16 cycles
```

This means the 4-word grouping is now real RTL behavior. Nonzero setup cycles,
stall cycles, and overlap are still future work.

当前 `WORDS_PER_CYCLE=4`、`SETUP_CYCLES=0` 的 RTL 路径中，普通 matmul 的搬运
phase 已经是 4 word/cycle。也就是说，4-word grouping 已经是真实 RTL 行为；非零
setup cycle、stall cycle 和搬运/计算 overlap 仍是后续工作。

The explicit data mover counters for a normal matmul job are:

```text
data_mover.active_cycles: 52
data_mover.transfer_cycles: 52
data_mover.words: 208
data_mover.read_cycles: 36
data_mover.write_cycles: 16
data_mover.read_words: 144
data_mover.write_words: 64
```

These 208 words are program + input0 + input1 + output. Descriptor reads are
not counted as data mover work.

普通 matmul job 的显式 data mover 计数如上。208 个 word 包含 program、input0、
input1 和 output；descriptor read 不计入 data mover work。

## 9. Known Limitations

- CPU staging/check work is not measured.
- DMA staging work is not measured yet.
- CPU polling is modeled as one synthetic wait span.
- Fine-grain data mover/wrapper/core timeline placement is not an
  architectural measurement and is omitted from CSR-sourced production jobs.
- SRAM `ready`/stall cycles are not separated.
- The TB-versus-CSR equality checker reads internal snapshot storage only as a
  verification reference; firmware and report consume MMIO-visible values.
- No global multi-job timeline yet.

## 10. Counter Placement Policy

Short term:

- add only stable, reviewable completed-job metrics to the CSR snapshot;
- keep `PERF_JOB` JSON additive and label the architectural CSR provenance.

Current transition:

- expose the stable job summary and identity through wrapper perf CSRs;
- cross-check snapshot implementation against minimal TB event references;
- consume firmware MMIO-read CSR values as report/PPA performance provenance.

Signal retention review:

| Signal group | Remove now? | Reason |
| --- | --- | --- |
| `npu_v0_data_mover.perf_*` | No | It is the RTL event source feeding the CSR bank and validation reference. |
| `mover_perf_stall` | No, but document as zero | It preserves the first-batch counter schema while no stall behavior exists. |
| TB detailed wrapper/core phase sampling | Removed from production path | It has no architectural timeline contract. |
| TB direct hierarchical access to `perf_snap_*` | Retain for validation | It independently checks the CSR implementation and is not report input. |

## 11. Next Work

Next performance work follows the stabilized Level 0/PPA execution order:

1. Extend workload identity and external-memory accounting for Transformer
   comparisons.
2. Retire order-based inference once legacy-log
   replay is no longer required.
3. Define committed `mac_ops`/`instr_count` or error/timeout events only when
   Transformer execution requires those architectural counters.
