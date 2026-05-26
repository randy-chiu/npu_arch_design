# Design Docs

This directory contains active module-level design documents. These are
implementation contracts and should be updated before and after non-trivial
development work.

| Document | Scope |
| --- | --- |
| `soc_architecture.md` | SoC top, memory map, bus, ROM/SRAM, NPU attachment |
| `soc_dma.md` | SoC ROM-to-SRAM DMA/preload engine |
| `npu_wrapper.md` | NPU wrapper, descriptor FSM, host windows, A2 data mover |
| `npu_subsystem.md` | primary PPA top around wrapper/data mover/core with external memory boundary |
| `npu_core.md` | NPU core memories, uop execution, matmul array, softmax path |
| `software_hardware_flow.md` | compiler, assembler, firmware, descriptor ABI, CPU/NPU flow |
| `quantization_strategy.md` | selected-layer quantization boundary, scale policy, bias handling |
| `fc1_k_streaming_matmul.md` | K-axis streaming matmul contract for real MNIST CNN `fc1` |
| `k_stream_ping_pong_buffer.md` | K-streaming ping-pong buffer overlap design |
| `performance_instrumentation.md` | cycle counters, PERF_JOB schema, report lanes, perf-counter strategy |
| `verification_strategy.md` | verification layers, current baselines, gaps, and test update rules |
| `ppa_methodology.md` | ASIC-oriented PPA boundaries, metrics, targets, activity and result contracts |
| `transformer_workloads.md` | LLM prefill/decode workload levels, initial kernels, precision and traffic requirements |
| `workload_manifest.md` | Explicit `job_id` to workload contract for repeatable perf/PPA grouping |
| `directory_structure.md` | Current implementation paths and staged unified-NPU target layout |
| `perf_counter_csr_plan.md` | Migration from testbench sampled counters to wrapper-visible counters |

If a change touches one of these areas, update the matching design document
before implementing the change. After implementation, record the actual result,
test command, and any remaining gap.
