# Design Docs

This directory contains active module-level design documents. These are
implementation contracts and should be updated before and after non-trivial
development work.

| Document | Scope |
| --- | --- |
| `soc_architecture.md` | SoC top, memory map, bus, ROM/SRAM, NPU attachment |
| `npu_wrapper.md` | NPU wrapper, descriptor FSM, host windows, A2 data mover |
| `npu_core.md` | NPU core memories, uop execution, matmul array, softmax path |
| `software_hardware_flow.md` | compiler, assembler, firmware, descriptor ABI, CPU/NPU flow |
| `performance_instrumentation.md` | cycle counters, PERF_JOB schema, report lanes, perf-counter strategy |
| `verification_strategy.md` | verification layers, current baselines, gaps, and test update rules |

If a change touches one of these areas, update the matching design document
before implementing the change. After implementation, record the actual result,
test command, and any remaining gap.
