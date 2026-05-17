# Performance Report Tools

This directory contains host-side tools for turning RTL simulation performance
records into machine-readable data and a browser UI.

## Current Flow

```text
make perf-report
```

Outputs:

```text
build/perf/cpu_soc_perf.log
build/perf/perf.json
build/perf/perf_report.html
```

The current report consumes `PERF_JOB` JSON lines printed by `soc_cpu_tb`.
The schema is intentionally small: each job has total cycles plus nested module
phase counters. Later RTL modules can add more nested counters without changing
the basic report flow.

## Current Status

The first report is tied to the CPU-controlled SoC smoke simulation. It measures
the interval from CPU firmware writing `CTRL.start` to the NPU wrapper finishing
the job. The HTML report currently shows:

- a cycle timeline with `CPU firmware`, `NPU wrapper`, and `NPU core` lanes;
- active work spans as solid blocks;
- wait/blocked spans as patterned blocks;
- per-job total cycles;
- wrapper phase counters;
- core phase counters;
- raw JSON for debugging and future tooling.

Current wrapper phases:

| Phase | Meaning |
| --- | --- |
| `desc_read` | wrapper reads `soc_npu_job_desc_t` from SRAM |
| `fetch_program` | wrapper fetches NPU program words from SRAM into the core host window |
| `fetch_input0` | wrapper fetches A/X input data |
| `fetch_input1` | wrapper fetches B input data for matmul |
| `start_core` | wrapper pulses core start |
| `wait_core` | wrapper waits while the core is executing |
| `write_output` | wrapper writes C/Y output data back to SRAM |
| `done` | wrapper done-latch phase |

Current core phases:

| Phase | Meaning |
| --- | --- |
| `fetch` | core fetch/decode cycles, including current single-cycle vector tasks |
| `matmul` | iterative matmul engine cycles |
| `done` | core done assertion phase |

The initial measured baseline is:

| Job | Total cycles | Core cycles | Dominant cost |
| --- | ---: | ---: | --- |
| `matmul` | 738 | 520 | 512-cycle iterative matmul |
| `softmax` | 53 | 11 | wrapper movement and single-cycle vector tasks |

The 512-cycle matmul compute cost is expected for the current RTL. It is one
MAC update per clock over an 8x8x8 tile, not a tensor/cube matrix engine. The
target architecture direction is described in `docs/target_architecture.md`.

## Timeline Semantics

The timeline is a reconstruction from the counters emitted by `soc_cpu_tb`.
For the current sequential wrapper FSM this is exact enough to show ordering and
overlap:

```text
CPU firmware:  MMIO start -> poll/wait for done
NPU wrapper:   desc/program/input -> start core -> wait core -> output
NPU core:                              fetch/execute/done
```

The key use is spotting pipeline structure and wasted time. For example,
`wait_core` overlapping with core `matmul` is expected. A long CPU wait with no
matching wrapper/core work would indicate missing accounting or a real stall.

## Current Limitations

- CPU time before `CTRL.start` is not measured yet. Descriptor construction,
  SRAM buffer writes, and result checking are outside the current timeline.
- The CPU lane currently models firmware as `MMIO start` followed by polling
  wait; it does not yet break down individual firmware instructions or bus
  transactions.
- The wrapper is still mostly sequential, so there is little true overlap except
  wrapper wait overlapping core execution.
- Core vector operations are still single-cycle RTL tasks, so softmax timing is
  not a realistic vector/SFU pipeline model yet.
- Counters are collected in the testbench through hierarchy. This is useful for
  bring-up, but synthesizable performance counters should later move into RTL
  registers.

## Extension Points

Near-term extensions:

- Add CPU-side spans before and after NPU launch: input staging, descriptor
  write, program copy, result check, and test-status write.
- Split CPU polling into MMIO read transactions and idle cycles.
- Add SRAM NPU-port read/write spans and bandwidth counters.
- Add bus-level spans for CPU accesses to SRAM, wrapper registers, and test
  status.
- Emit per-job workload metadata such as tensor shape, program words, input
  words, and output words.

NPU wrapper extensions:

- Separate SRAM request, SRAM ready, host-window write, and output readback.
- Track descriptor validation, error handling, timeout, IRQ, and command queue
  phases once those features exist.
- Show overlap when future DMA/data movement can run concurrently with core
  compute.

NPU core extensions:

- Split fetch, decode, issue, execute, writeback when the core grows a real
  pipeline.
- Track MAC active cycles, accumulator write cycles, vector/SFU active cycles,
  and stalls.
- Add utilization counters: MAC lane utilization, vector lane utilization,
  scratchpad bank conflicts, and instruction bubbles.

Report/UI extensions:

- Add filtering by job/operator/module.
- Add stacked global timeline across multiple jobs instead of one independent
  timeline per job.
- Add critical-path highlighting and idle-gap warnings.
- Add compare mode between two `perf.json` files for architecture experiments.
