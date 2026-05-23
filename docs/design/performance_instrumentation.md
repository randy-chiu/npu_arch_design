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

Current measured interval starts at NPU launch, not at CPU input staging. CPU
copying tensors/programs into SRAM and checking outputs are outside the current
cycle report.

## 4. Sampled Signals

| Counter group | Source signal |
| --- | --- |
| wrapper phases | `dut.u_npu_wrapper.desc_state` |
| core phases | `dut.u_npu_wrapper.u_npu.state` |
| SRAM read/write cycles | `dut.u_npu_wrapper.sram_req/sram_we` |
| core host writes | `dut.u_npu_wrapper.desc_host_we` |
| core host reads | inferred during `DESC_WRITE_OUTPUT` |
| moved words | SRAM request counts by wrapper state |

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
7. When `desc_state == DESC_DONE`, the testbench prints one `PERF_JOB` JSON
   record and closes the active job.

`report.py` is post-processing only. It parses lines beginning with
`PERF_JOB `, adds analytical estimates, reconstructs timeline spans from the
phase counters, infers known multi-job workloads by job order, and writes JSON
and HTML reports.

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

The full `fc1` K-stream tile is a single descriptor with `k_chunks=1152`.
Current measured wrapper counters for that job are still dominated by movement,
but the NPU-side SRAM/data-mover/core-host path now moves four words per cycle:

```text
total_cycles: 58784
input0_words: 73728
input1_words: 73728
fetch_input0 cycles: 18432
fetch_input1 cycles: 18432
core matmul cycles: 11520
```

完整 `fc1` K-stream tile 是一个 `k_chunks=1152` 的 descriptor。当前测得的 wrapper
计数仍主要由搬运主导，但 NPU 侧 SRAM/data-mover/core-host 路径已经从每拍 1 word
变为每拍 4 word。

Important consequence: the current numbers are cycle counts observed by the
simulation testbench, not timestamp events emitted by synthesizable RTL logic.
They are valid for bring-up and bottleneck classification, but they are not yet
CPU-readable hardware counters.

## 5. Current JSON Shape

Each job has:

| Field | Meaning |
| --- | --- |
| `id` | job sequence number |
| `name` | `matmul`, `softmax`, or `unknown` |
| `total_cycles` | measured launch-to-wrapper-done cycles |
| `wrapper` | wrapper phase cycles |
| `core` | core phase cycles |
| `movement` | SRAM/core-host movement counters |
| `estimates` | matmul compute model, if applicable |
| `movement_estimates` | burst-style movement model |
| `timeline` | reconstructed lanes and spans |

The schema is intentionally small and additive. New counters should be added as
nested fields rather than breaking existing fields.

## 6. Timeline Reconstruction

The report builds lanes:

| Lane | Current source |
| --- | --- |
| `CPU firmware` | synthetic start + poll/wait span |
| `NPU wrapper` | wrapper phase counters |
| `Data mover` | currently reconstructed from wrapper movement phases |
| `NPU core` | core phase counters offset to wrapper core-wait position |

Current data mover lane is a bridge step. After A2 adds explicit data mover
counters/state, the lane should come from real data mover measurements.

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

## 9. Known Limitations

- CPU staging/check work is not measured.
- CPU polling is modeled as one synthetic wait span.
- Data mover lane is reconstructed, not from independent data mover counters.
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

1. Add explicit data mover counters:
   `dm_active_cycles`, `dm_setup_cycles`, `dm_transfer_cycles`, `dm_words`,
   `dm_stall_cycles`.
2. Drive the `Data mover` lane from those counters.
3. Add burst-mode comparison against the movement model.
4. Later add scratchpad bank conflict and core input-stall counters.
