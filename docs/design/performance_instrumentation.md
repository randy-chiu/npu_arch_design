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

Counters are currently collected in `hw/soc/tb/soc_cpu_tb.sv`.

This is testbench-side profiling:

- no CPU-visible perf registers yet;
- no synthesizable counter block yet;
- no counter logic inserted into every RTL module;
- the testbench observes existing hierarchical RTL signals once per cycle.

This choice keeps RTL simple while phase definitions are still changing.

## 3. Job Boundary

Start condition:

```text
CPU writes NPU_OPSCHED_CTRL with bit 0 set
```

End condition:

```text
wrapper reaches DESC_DONE and PERF_JOB is printed
```

Current measured interval starts at NPU launch, not at CPU/DMA input staging.
Copying tensors/programs into SRAM and checking outputs are outside the current
cycle report.

当前测量区间从 NPU launch 开始，不包含 CPU/DMA input staging。tensor/program
搬到 SRAM 以及 firmware 检查输出都不在当前 `PERF_JOB` cycle 统计内。

## 4. Sampled Signals

| Counter group | Source signal |
| --- | --- |
| wrapper phases | `dut.u_npu_wrapper.desc_state` |
| core phases | `dut.u_npu_wrapper.u_npu.state` |
| SRAM read/write cycles | `dut.u_npu_wrapper.sram_req/sram_we` |
| core host writes | `dut.u_npu_wrapper.desc_host_we` |
| core host reads | inferred during `DESC_WRITE_OUTPUT` |
| moved words | SRAM request counts by wrapper state |
| explicit data mover | `dut.u_npu_wrapper.u_data_mover.perf_*` signals |

The testbench prints one JSON line:

```text
PERF_JOB {...}
```

`sw/tools/perf/report.py` parses these lines and generates:

```text
build/perf/perf.json
build/perf/perf_report.html
```

### Current Code Walkthrough

`make perf-report` does not run a separate profiler. It rebuilds the
CPU-controlled SoC simulation, redirects the simulator stdout to
`build/perf/cpu_soc_perf.log`, then runs `sw/tools/perf/report.py`.

Inside `hw/soc/tb/soc_cpu_tb.sv`, the profiling block is an `always` block on
`posedge clk`. It behaves like this:

1. When the CPU bus writes `NPU_OPSCHED_CTRL` with `wdata[0] == 1`, the
   testbench starts a new perf job, increments `perf_job_id`, and clears all
   counters.
2. While `perf_active` is true, `perf_total_cycles` increments once per clock.
3. The testbench samples `dut.u_npu_wrapper.desc_state` and increments exactly
   one wrapper phase counter for the current state.
4. While the wrapper is launching or waiting for the core, the testbench samples
   `dut.u_npu_wrapper.u_npu.state` and increments the core phase counters.
5. Wrapper SRAM requests are counted from `sram_req/sram_we`; the same request
   is classified into descriptor/program/input/output word counters by the
   current wrapper state.
6. Core host-window writes are counted from `desc_host_we`; host-window reads
   are inferred during `DESC_WRITE_OUTPUT`.
7. The testbench also samples explicit data mover perf signals:
   `perf_active`, `perf_setup`, `perf_transfer`, `perf_stall`, and
   `perf_words`.
8. When `desc_state == DESC_DONE`, the testbench prints one `PERF_JOB` JSON
   record and closes the active job.

`report.py` is post-processing only. It parses lines beginning with
`PERF_JOB `, adds analytical estimates, reconstructs timeline spans from the
phase counters, and writes JSON and HTML reports. The current build supplies
`build/perf/workload_manifest.json`, so workload grouping uses explicit
`job_id` declarations rather than job order. Order-based inference remains
only as a warned fallback for legacy logs.

Current workload inference recognizes 53 jobs / 7 workloads when the real MNIST
external fixtures are present:

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
current workload inference recognizes 68 jobs / 7 workloads:

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
total_cycles: 39217
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

Important consequence: the current numbers are cycle counts observed by the
simulation testbench, not timestamp events emitted by synthesizable RTL logic.
They are valid for bring-up and bottleneck classification, but they are not yet
CPU-readable hardware counters.

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
| `wrapper` | wrapper phase cycles |
| `core` | core phase cycles |
| `movement` | SRAM/core-host movement counters |
| `data_mover` | explicit counters sampled from `npu_v0_data_mover` |
| `estimates` | matmul compute model, if applicable |
| `movement_estimates` | burst-style movement model |
| `timeline` | reconstructed lanes and spans |

The schema is intentionally small and additive. New counters should be added as
nested fields rather than breaking existing fields.

The manifest contract and mismatch behavior are specified in
`docs/design/workload_manifest.md`. The planned transition from testbench
sampling to CPU-visible counter registers is specified in
`docs/design/perf_counter_csr_plan.md`.

## 6. Timeline Reconstruction

The report builds lanes:

| Lane | Current source |
| --- | --- |
| `CPU firmware` | synthetic start + poll/wait span |
| `NPU wrapper` | wrapper phase counters |
| `Data mover` | currently reconstructed from wrapper movement phases |
| `NPU core` | core phase counters offset to wrapper core-wait position |

Current data mover lane is still placed on the wrapper fetch/write phases, but
the numeric counters now come from explicit `npu_v0_data_mover` perf signals.
For `matmul_k_stream`, the report also renders a `K prefetch overlap` span
inside the wrapper `wait_core` interval. This makes the ping-pong behavior
visible as timeline overlap instead of only as aggregate cycle reduction. A
future report step can use explicit setup/stall counters to render finer
subspans directly.

当前 Data mover lane 的时间位置仍由 wrapper fetch/write phase 放置，但数值计数
已经来自 `npu_v0_data_mover` 的显式 perf signal。对于 `matmul_k_stream`，report
还会在 wrapper `wait_core` 区间内渲染 `K prefetch overlap` span，让 ping-pong
行为在 timeline 上可见，而不只是通过总 cycle 下降间接体现。后续可以继续用显式
setup/stall counter 渲染更细的子阶段。

## 7. Report Panels

Current HTML panels:

- summary metrics;
- cycle timeline;
- matmul compute model;
- movement model;
- wrapper phase timeline;
- data mover phase timeline;
- core phase timeline;
- raw JSON.

The UI should prefer timeline views over plain tables because the main question
is whether phases overlap or block each other.

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
- Data mover lane placement is still reconstructed from wrapper phases, though
  numeric counters now come from explicit data mover signals.
- SRAM `ready`/stall cycles are not separated.
- Core host-window reads are inferred by wrapper state.
- Counters are not software-readable.
- No global multi-job timeline yet.

## 10. Counter Placement Policy

Short term:

- keep rapidly changing counters in `soc_cpu_tb`;
- expose enough hierarchy for testbench sampling;
- keep `PERF_JOB` JSON additive.

Medium term:

- move stable counters into an optional RTL perf-counter/debug block;
- expose CPU-readable counters through wrapper debug CSRs;
- keep simulation report able to consume either testbench or RTL counter source.

## 11. Next Work

Next performance work should match A2:

1. Drive the `Data mover` lane from explicit setup/transfer/stall counters.
2. Add DMA staging counters if staging should appear in the global timeline.
3. Add burst-mode comparison against the movement model.
4. Later add scratchpad bank conflict and core input-stall counters.
