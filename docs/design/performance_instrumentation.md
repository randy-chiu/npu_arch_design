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

The performance page must also show the tested computation structure before
the timelines:

```text
workload/profile -> graph/model -> layer or attention group -> operator -> shape
```

Each executable operator card places its theoretical-cycle formula beside
measured compute and total cycles. This lets a user distinguish datapath
inefficiency from descriptor/data movement overhead without reading JSON.

### Required Timeline Contract

Every measured descriptor job must expose one aligned timeline with:

| Lane | Purpose |
| --- | --- |
| command processor | descriptor read, fetch issue, compute launch/wait, writeback issue, done |
| data mover | program/input/output transfer intervals and stalls |
| matrix | active matrix interval |
| vector | active vector intervals |
| reduction | active reduction intervals |
| SFU | active EXP/RECIP/RSQRT intervals |

The timeline must distinguish measured spans from analytically reconstructed
spans. Reconstructed spans are allowed only when their ordering and duration
come from the reviewed descriptor/core state machine; arbitrary placement from
summary totals is not allowed.

K-stream reports require cycle-event placement because summary totals cannot
tell where scheduler-only gaps occur inside a Data-mover prefetch interval.
In particular, this subtraction is invalid for absolute placement:

```text
prefetch - compute overlap - wait for data
```

The remainder means that Data mover was active while Compute cluster was not;
it does not prove that those cycles all occurred before compute started.
Simulation reports therefore use `PERF_TRACE` cycle events for K-stream lane
placement and retain CSR snapshots as the authoritative aggregate totals.

Timeline lanes must also expose the hardware hierarchy instead of presenting
every lane as an unrelated peer:

```text
CPU firmware
NPU wrapper                    CPU-visible host transactions only
NPU core                       architecture group, not an additive active lane
  Command processor            descriptor decode and load/compute/store issue
  Data mover                   external-memory/local-storage movement
  Compute cluster              compute execution boundary
    Matrix/Vector/Reduction/SFU compute-cluster child execution units
```

The wrapper lane must not reuse command-processor state residency. Accepting a
CPU launch is wrapper work; descriptor fetch, movement issue, compute wait, and
job retirement are NPU-core command-processor work. During data movement or
compute, the wrapper is normally idle even though the NPU core remains busy.
The `NPU core` row is a visual group boundary and has no additive active-cycle
total. Matrix/vector/reduction/SFU lanes refine compute-cluster activity.

Command-processor state residency is also not automatically active work. A
fetch or writeback state that has issued a data-mover command and is waiting
for completion must be shown as wait. Until individual command-issue events
are traced, the report counts only confirmed descriptor handling, compute
launch, and completion control as command-processor work.

The production snapshot must expose these measured counters:

| Counter | Event definition |
| --- | --- |
| command-processor active | descriptor word accepted, explicit control/config/launch/done state |
| command-processor wait | waiting for data mover or compute cluster completion |
| data-mover/compute overlap | `data_mover_active && compute_cluster_active` in the same cycle |
| uop-scheduler active | common uop fetch/decode/issue/completion-control work |
| uop-scheduler wait | an issued execution-engine command has not completed |
| matrix datapath active | Matrix engine's internal MAC iteration state is active |
| wait for prefetched data | K-stream command remains active after a chunk completes while the next chunk is still moving |
| compute-cluster local active | accepted local LOAD/STORE or fixed primitive/local-storage operation; command launch is excluded |
| data-mover program load | external SRAM to core-local `instr_mem` while the command processor is in program-fetch state |
| data-mover initial input load | external SRAM to the first A/B preload bank before its first uop-program launch |
| data-mover next-chunk prefetch | external SRAM to the alternate A/B preload bank while the current chunk is executing or waiting |
| data-mover output store | mover transfer while writing the completed output |

Reports must use movement labels that preserve the two scheduling levels:

```text
Data mover:         external SRAM -> preload bank
Uop scheduler:      bind selected MatMul operand bank
Local storage path: only explicitly modeled physical local movement
```

The generic word `LOAD` must not be used for both without identifying the
source and destination.

K-stream overlap must use the measured overlap counter. It must not be inferred
from aggregate read cycles or drawn as an exact span without this event.

Movement spans must expose overlap rather than stretching an initial-load
label across compute. For the current K-stream Prefill Projection GEMM,
`data_mover_active && compute_cluster_active` measures 9 overlap cycles. The
report shows those cycles as `Measured K prefetch overlap`; it must not derive
an overlap amount by subtracting aggregate read and wait counters.

For K-stream timelines, aggregate compute-active cycles must not be drawn as
one continuous span. The report uses the measured wait-for-prefetched-data
counter to render:

```text
chunk 0 compute -> wait for prefetched A/B -> chunk 1 compute
```

Matrix-engine spans use the measured datapath-active event, not time spent by
the scheduler waiting for the Matrix engine transaction.

Compute-cluster active excludes command-processor launch and Uop-scheduler
decode/issue-only cycles. For the common matmul path it is the union of:

```text
accumulator commit/add || local STORE execution || Matrix datapath active
```

The MatMul `LOAD A/B` uops are scheduler work that bind the selected operand
bank; they are not Compute-cluster local-movement active cycles. For one chunk
the current active-cycle breakdown is:

```text
non-final chunk: 8-cycle Matrix -> 1-cycle accumulator commit/add
final chunk:     8-cycle Matrix -> 1-cycle accumulator commit/add
                                -> 1-cycle accumulator-to-C-window copy
```

The accumulator commit and accumulator-to-C-window copy are full-tile-wide RTL
events backed by the performance-first contract in `arch/configs/npu_v0.jsonc`.
The baseline deliberately declares 64 int32 commit/add lanes and 2048-bit
read/write paths, so both operations are architectural one-cycle transactions.
Mapped timing remains unverified. Reports expose both in the `Accumulator
file` lane and reject transaction durations or overlaps that violate the
declared contract.

The strict cycle-trace validator also checks:

- every Matrix-active transaction lasts exactly one declared operand-feed
  cycle per K slice;
- primitive accept and response transactions match the declared Attention-row
  read and write latency;
- accumulator clear, commit, and readout transactions are mutually exclusive
  and match their declared latency.

Data-mover phase spans must come from state-qualified measured counters. The
report must not reconstruct initial-load length by subtracting aggregate
overlap and wait counters from total reads.

Timeline lane totals must distinguish:

```text
elapsed span = wall-clock interval covered by the lane
active cycles = cycles in work spans only
wait cycles = cycles in wait spans only
```

A polling CPU/firmware lane may cover the full descriptor interval, but its
poll/wait span must not be counted or colored as continuous firmware work.
Until individual MMIO poll reads are traced, the report shows the interval as
wait time and reports only the known launch write as firmware active time.

The page must summarize bottlenecks, including the longest active lane,
non-compute overhead, and whether movement overlaps compute.

## 2. Current Collection Method

The report source is the CPU-visible completed-job snapshot implemented in
`hw/npu_core/rtl/npu_v0_core_system.sv`. Firmware reads it through MMIO after
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
build/ppa/data/perf.json
```

`perf.json` is the single measured-performance data source. The selected PPA
test-case page, for example `build/ppa/cases/transformer.html`, is the single
HTML location for computation graphs, operator details, and pipeline timelines.
The perf tool retains an optional standalone HTML mode for isolated tool tests,
but the normal PPA build does not generate a duplicate pipeline report.

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

### Theoretical Versus Measured Cycle Analysis

Raw measured cycles are insufficient for architecture analysis because they do
not explain whether an observed cost is expected. Every executable Transformer
stage should therefore report a theoretical compute lower bound beside the
measured values.

Required fields:

| Field | Meaning |
| --- | --- |
| `theoretical_compute_cycles` | mathematical lower bound from logical work and declared peak engine parallelism |
| `measured_compute_cycles` | measured active cycles of the relevant engine, or core-active model when no per-engine CSR exists |
| `compute_overhead_cycles` | `measured_compute_cycles - theoretical_compute_cycles` |
| `compute_efficiency` | `theoretical_compute_cycles / measured_compute_cycles` |
| `measured_total_cycles` | descriptor launch-to-completion cycles |
| `non_compute_overhead_cycles` | `measured_total_cycles - measured_compute_cycles` |
| `end_to_end_efficiency` | `theoretical_compute_cycles / measured_total_cycles` |
| `theoretical_cycle_basis` | human-readable formula and assumptions |

Current formulas:

```text
matrix QK/PV:
  theoretical = ceil(M * N * K * jobs / peak_macs_per_cycle)

scale/mask:
  theoretical = ceil(score_elements * jobs / vector_lanes)

measured softmax rows:
  theoretical primitive issues =
      rows * (REDUCE_MAX + VEC_SUB + VEC_CLAMP + EXP_per_lane
            + REDUCE_SUM + RECIP + VEC_SCALE)
```

The softmax formula is a primitive-issue lower bound, not a final latency
promise. It assumes one cycle per primitive issue and no stalls. Until
vector/reduction/SFU active-cycle CSRs are connected, scale/mask and softmax
use measured core-active cycles as an explicitly labeled model.

Example for current `8x8x8` QK:

```text
useful MACs                  = 8 * 8 * 8 = 512
peak MACs/cycle              = 64
theoretical compute cycles   = ceil(512 / 64) = 8
measured matrix cycles       = 10
compute overhead             = 2 cycles
compute efficiency           = 8 / 10 = 80%
measured descriptor cycles   = 84
end-to-end efficiency        = 8 / 84 = 9.52%
```

This separates datapath efficiency from wrapper/data-movement overhead.

When a workload is model-only or a hardware engine is not implemented yet, the
corresponding utilization/cycle field is `null` or zero rather than reported as
measured hardware behavior.

### Current Code Walkthrough

`make perf-report` does not run a separate profiler. It rebuilds the
CPU-controlled SoC simulation, redirects simulator stdout to
`build/ppa/data/cpu_soc_perf.log`, then runs `sw/tools/perf/report.py`.

#### Data Mover Event Source

`hw/npu_core/rtl/memory/npu_v0_data_mover.sv` exports the event signals consumed by
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

`hw/npu_core/rtl/npu_v0_core_system.sv` implements the software-readable
completed-job snapshot:

1. A write to `NPU_OPSCHED_CTRL.start` asserts `perf_start_event`, sets
   `perf_running`, and clears the private `perf_work_*` accumulator bank. The
   previous `perf_snap_*` values remain readable until completion or explicit
   idle clear.
2. While `perf_running` is set, the NPU core counter block saturating-adds total cycles,
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

The earlier MNIST report displayed detailed timelines because testbench-only
phase counters were present. After production reporting moved to CPU-readable
CSR snapshots, only summary counters remained, and `report.py` intentionally
fell back to a CPU-only lane rather than inventing phase placement. Therefore
the missing timeline is a collection-contract regression, not an HTML
rendering limitation.

The correction is to add reviewed phase/module trace fields to the
architectural report contract and render them for every workload. Until all
new CSRs exist, fixed descriptor paths may emit state-machine-derived spans
whose provenance is explicitly `derived_from_reviewed_state_machine`.

#### Report Post-Processing

`report.py` parses `PERF_JOB ` records, adds analytical estimates, and writes
JSON and HTML. It can reconstruct timeline spans for legacy records containing
old TB phase counters; the production
build supplies `build/ppa/data/workload_manifest.json`, so workload grouping uses
explicit `job_id` declarations rather than job order. `infer_workloads()` is
still retained as a warned legacy-log fallback and its tests still encode the
historical fixed job ordering.

The generated manifest describes 53 jobs / 7 workloads when only the earlier
single full-`fc1`-tile external fixture is present:

```text
operator_smoke_matmul: 1 job
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
PPA are values consumed through wrapper snapshot CSRs. The TB equality
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

### 6.1 Command semantic-event boundary

The per-cycle SoC trace records stable semantic command, Scheduler-wait, and
Compute-cluster-control event IDs defined by `arch/configs/npu_v0.jsonc`.
RTL emits those IDs with row/lane/chunk/bank arguments. The PPA report consumes
the generated architecture event map and must not translate private FSM state
numbers. Workload-specific synthetic command sequences are not allowed.

Primitive execution timelines follow the same rule. Scale/mask and Attention
Softmax run through the common Uop Scheduler. Their PPA timelines must:

- show typed primitive-accept/primitive-response waits;
- use measured vector/reduction/SFU active events and operation arguments;
- show compute-cluster-control prepare/start/wait/result-handoff
  cycles that are not execution-engine active cycles;
- identify softmax row and SFU lane where applicable;
- never split total compute time into estimated engine occupancy ratios.

The current attention softmax is intentionally serial per row:

```text
reduce-max
-> vector subtract
-> vector clamp
-> eight scalar SFU EXP operations
-> reduce-sum
-> scalar SFU reciprocal
-> vector normalize
```

For eight rows this makes scalar SFU EXP issue the dominant latency. The
timeline must expose that serialization so a future vectorized/pipelined SFU
proposal can be compared against measured evidence.

`Compute cluster control` is not a separate execution engine. It is the
control FSM physically implemented inside `npu_v0_compute_cluster`. It is shown
as a child lane only to account for compute-cluster-active cycles during which
Matrix/Vector/Reduction/SFU are not active.

Compute-cluster control cycles are semantic `PRIMITIVE_ACCEPT` and
`PRIMITIVE_RESPONSE` events, plus the measured `ENGINE_START_ADAPTER` cycle
required by current internal start/done engines. Engine-active cycles are not
also labeled as control work.

The Vector engine itself is active for only one cycle per row, or eight cycles
total. This `25 control / 8 execution` ratio is measured behavior but poor
microarchitectural efficiency. A future scheduler/handshake path should issue
consecutive rows without three control cycles around every one-cycle Vector
operation.

The report builds lanes:

| Lane | Parent | Current source |
| --- | --- | --- |
| `CPU firmware` | none | synthetic start + poll/wait span |
| `NPU wrapper` | none | confirmed CPU-visible launch transaction |
| `NPU core` | none | visual architecture group; no additive cycle total |
| `Command processor` | `NPU core` | descriptor/scheduler state-machine placement |
| `Data mover` | `NPU core` | measured movement totals plus reviewed placement |
| `Compute cluster` | `NPU core` | compute active counters offset to command-processor wait position |
| `Compute cluster control` | `Compute cluster` | measured internal control-FSM cycles when no execution engine is active |
| matrix/vector/reduction/SFU engines and local-storage path | `Compute cluster` | engine active counters or reviewed state-machine placement |

For legacy phase-rich records, the data mover lane is placed on wrapper
fetch/write phases and `matmul_k_stream` records can render a `K prefetch
overlap` span. Production CSR snapshots use state-qualified phase counters and
the measured overlap counter; exact absolute span placement remains derived
from the reviewed command/core state-machine sequence.

旧 phase-rich 记录仍可按 command-processor fetch/write phase 回放 Data mover timeline
与 `K prefetch overlap` span。正式 CSR snapshot 的阶段时长与 overlap
均来自实测计数器，绝对起止位置仍按已评审的 command/core 状态机顺序放置。

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
- The strict per-job validator rejects out-of-range spans, same-lane overlap,
  cycle-length mismatch, Scheduler active/wait contradictions, typed-wait
  mismatches, and uncovered Attention Compute-cluster cycles. It does not yet
  validate every cross-module dependency or overlap rule.
- Primitive engine adapters still expose internal start/done latency. This is
  reported as `ENGINE_START_ADAPTER`; replacing it requires a separate measured
  native-engine valid-ready change.
- Legacy phase-reconstructed logs can contain internally inconsistent summary
  values that produce out-of-range spans. They are now marked
  `legacy_not_accepted_as_architectural_evidence`; only architectural
  cycle-event timelines can pass the strict truthfulness validator.

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
