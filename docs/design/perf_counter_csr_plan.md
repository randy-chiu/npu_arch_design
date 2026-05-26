# Performance Counter CSR Plan

## Current Boundary

The current bring-up path is allowed to collect per-job counters in
`hw/soc/tb/soc_cpu_tb.sv` by observing wrapper, data mover, and core signals.
`PERF_JOB` and reports must describe those values as
`measured_rtl_perf_job_counters` or otherwise mark their testbench provenance.
They are not yet CPU-visible architectural counters.

## Planned Counter Set

| Counter / register | Meaning |
| --- | --- |
| `total_cycles` | Launch-to-completion elapsed cycles for the selected job window. |
| `core_active_cycles` | Cycles in which the core is executing or completing work. |
| `core_matmul_cycles` | Cycles attributed to the matrix engine. |
| `data_mover_active_cycles` | Cycles in any data mover phase. |
| `data_mover_setup_cycles` | Descriptor/setup overhead cycles in the mover. |
| `data_mover_transfer_cycles` | Cycles transferring valid words. |
| `data_mover_stall_cycles` | Active cycles blocked by memory/backpressure. |
| `data_mover_words` | Valid on-chip words moved. |
| `sram_read_words` | Words read through the NPU SRAM/memory boundary. |
| `sram_write_words` | Words written through that boundary. |
| `mac_ops` | Count of architecturally committed MAC operations. |
| `instr_count` | Completed NPU instructions/uops. |
| `error/status` | Counter validity, overflow, illegal command and completion state. |

Counter widths, snapshot semantics, clear/start control and any overflow
policy must be specified in the wrapper source-of-truth register configuration
before RTL-visible registers are introduced.

## Staged Integration

1. Keep current testbench counters as the regression reference and preserve
   provenance in perf/PPA output.
2. Define wrapper-visible counter register offsets and snapshot/clear
   semantics in `arch/configs/npu_wrapper_v0.jsonc` or its successor.
3. Aggregate stable wrapper/data-mover/core event signals in a synthesizable
   counter block and cross-check CSR reads against existing testbench samples.
4. Switch firmware or testbench report input to CSR/perf-register reads once
   matching is covered by regression.

The long-term interface is a CPU/firmware or testbench read of defined
CSR/perf registers, not hierarchical access to internal RTL state.
