# Design Docs

This directory contains active module-level design documents. These are
implementation contracts and should be updated before and after non-trivial
development work.

Each RTL module should have one module-level design source. Subfeatures such as
an SFU EXP table, vector requant mode, or matrix mixed-precision mode live
inside their owning module document unless they become a separate RTL module.
Overview, workload, compiler, runtime, and PPA documents may reference module
contracts, but must not redefine them.

| Document | Scope |
| --- | --- |
| `soc_architecture.md` | SoC top, memory map, bus, ROM/SRAM, NPU attachment |
| `soc_dma.md` | SoC ROM-to-SRAM DMA/preload engine |
| `npu_wrapper.md` | CPU-visible NPU Host wrapper and forwarded register/descriptor ABI |
| `npu_subsystem.md` | primary PPA top around Host wrapper and complete NPU core with external memory boundary |
| `npu_core.md` | NPU core memories, uop execution, matmul array, softmax path |
| `npu_core_module_status.md` | NPU core module status matrix: design source, RTL status, counters, tests, and blockers |
| `software_hardware_flow.md` | compiler, assembler, firmware, descriptor ABI, CPU/NPU flow |
| `performance_instrumentation.md` | cycle counters, PERF_JOB schema, report lanes, perf-counter strategy |
| `verification_strategy.md` | verification layers, current baselines, gaps, and test update rules |
| `ppa_methodology.md` | ASIC-oriented PPA boundaries, metrics, targets, activity and result contracts |
| `workload_manifest.md` | Explicit `job_id` to workload contract for repeatable perf/PPA grouping |
| `directory_structure.md` | Current implementation paths and staged unified-NPU target layout |
| `perf_counter_csr_plan.md` | Migration from testbench sampled counters to wrapper-visible counters |

## Domain-Specific Subdirectories

| Directory | Scope |
| --- | --- |
| `transformer/` | Transformer-oriented NPU v1 architecture, workloads, integration notes, and next-step plan |
| `v0_cnn/` | V0/CNN-specific MNIST `fc1` K-streaming, ping-pong, and selected-layer quantization notes |

If a change touches one of these areas, update the matching design document
before implementing the change. After implementation, record the actual result,
test command, and any remaining gap.

Current Transformer planning is attention-centered. Attention is treated as a
compiler/runtime sequence over shared matrix, vector, reduction, SFU, memory,
and scheduler primitives, not as a separate RTL macro.
