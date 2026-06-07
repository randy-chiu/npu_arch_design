# NPU Core Module Status

This document tracks the current NPU core module review state. It is the
short-form checklist that ties design docs, RTL, integration status, counters,
tests, and next blockers together.

## Review Boundary

The current core is still `hw/npu_core/rtl/npu_v0_compute_cluster.sv`. Transformer v1 does
not introduce a separate attention core. It adds shared primitive capabilities
and runtime/compiler sequencing over the same wrapper/core/perf/PPA framework.

Status labels:

| Label | Meaning |
| --- | --- |
| Implemented | RTL/tooling exists in the current tree |
| Integrated | reachable through `npu_v0_compute_cluster`, wrapper descriptors, or SoC firmware |
| Standalone | RTL exists but is not a production scheduler path |
| Spec-only | documented/configured, no RTL path yet |
| Model-only | report/golden metadata exists, no measured hardware execution |

## Module Matrix

| Module | Design source | RTL/tooling source | Integration status | Counter status | Verification entry | Blocking gaps |
| --- | --- | --- | --- | --- | --- | --- |
| Core top / launch modes | `docs/design/npu_core.md` | `hw/npu_core/rtl/npu_v0_compute_cluster.sv` | Integrated through wrapper `start/op` and host windows | core activity and wait-for-prefetched-data events | `make npu-core-sim`, `make perf-l0-transformer` | primitive routing still uses serialized start/done handoff |
| Uop scheduler / dispatcher | `docs/design/npu_core.md` | `hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv` | Integrated for Matrix, Scale/Mask, and Compiler-expanded Softmax with one-in-flight primitive valid-ready command/response | scheduler active plus typed Matrix/primitive accept/response waits published in PPA | core, Transformer SoC, and Transformer PPA pass | add multi-entry issue/response capacity only when measured overlap justifies it |
| Matrix array | `docs/design/npu_core.md` | `hw/npu_core/rtl/matrix/matmul_array.sv` | Integrated for int8 matmul, K-stream, and mixed PV mode; `docs/matmul_array_a1.md` is supporting A1 timing background | real matrix datapath active cycles are separate from scheduler transaction wait | `make npu-core-sim`, `make perf-l0-transformer` | no GEMV/valid-row/valid-column support, no per-mode event counters |
| Accumulator file | `docs/design/transformer/transformer_npu_v1.md` | `hw/npu_core/rtl/matrix/accumulator_file.sv` | Integrated into `npu_v0_compute_cluster` for matmul/K-stream residency | local read/write/clear/residency/spill counters exist but are not wrapper CSR surfaced | `make npu-core-sim`, SoC perf profiles | CSR/report exposure pending reviewed event-source plan |
| Vector engine | `docs/design/transformer/vector_engine_v1.md` | `hw/npu_core/rtl/vector/vector_engine.sv` | Used by Scheduler-issued Softmax and Scale/Mask primitives; valid/ready shim is standalone only | measured active/op/row timeline events plus shim-local counters; no per-op CSR exposure | `make primitive-engines-sim`, `make perf-l0-transformer` | native valid/ready Scheduler command/response and mask-select semantics remain |
| Reduction engine | `docs/design/transformer/reduction_engine_v1.md` | `hw/npu_core/rtl/reduction/reduction_engine.sv` | Used by Scheduler-issued Softmax primitives; valid/ready shim is standalone only | measured active/op/row timeline events plus shim-local counters; no per-op CSR exposure | `make primitive-engines-sim`, `make perf-l0-transformer` | native valid/ready command/response, segmented-row contract, and explicit mask input remain |
| SFU | `docs/design/transformer/sfu_v1.md` | `hw/npu_core/rtl/sfu/sfu_lut.sv` | Used by Scheduler-issued Softmax primitives; valid/ready shim is standalone only | measured active/op/row/lane timeline events plus shim-local counters; no per-op CSR exposure | `make primitive-engines-sim`, `make perf-l0-transformer` | current EXP is bring-up approximation, 257-entry target LUT not implemented, and native valid/ready remains |
| Primitive valid/ready | `docs/design/transformer/primitive_valid_ready_v1.md` | `hw/npu_core/rtl/primitive_handshake_shims.sv`, `hw/npu_core/rtl/npu_v0_compute_cluster.sv` | Production Scheduler boundary uses valid-ready; current engines retain internal start/done adapters | semantic accept/adapter/response events and standalone shim counters implemented | primitive, core, Transformer SoC, and Transformer PPA pass | replace internal engine adapters only when latency/overlap benefit is measured |
| Uop/primitive scheduler boundary | `arch/specs/transformer/v1/uop_isa_v1.md` | `hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv` | Common Matrix, Scale/Mask, and Compiler-expanded Softmax paths integrated; Compute cluster routes one accepted primitive | active plus typed wait reasons and row/lane issue semantics integrated | core, Transformer SoC, and Transformer PPA pass | generalized response payloads and loop-vs-expanded PPA decision remain |
| Memory / scratchpad boundary | `docs/design/npu_core.md` | internal arrays in `npu_v0_compute_cluster`; `hw/npu_core/rtl/memory/npu_v0_data_mover.sv` | NPU core command processor and data mover own SoC-side movement | data mover counters are published through Host-visible perf CSRs; no core bank conflict counters | `make perf-report`, `make perf-l0-transformer` | no production memory protocol, no bank conflict model, no double buffering contract |
| Attention runtime group flow | `docs/design/transformer/attention_runtime_v1.md` | host/compiler/runtime tooling | Generated runtime table launches QK, executable unmasked scale/mask, eight-row softmax, and PV with fixed-case SRAM buffer chaining | measured stage perf includes all four stages; parent group remains software-sequenced and runtime overhead is not measured | `make cpu-soc-transformer`, `make ppa-l0-report WORKLOAD_PROFILE=transformer`, PPA schema tests | intermediate tiles still cross SRAM boundaries, descriptor filling remains stage-specific, and shape is fixed |
| KV cache boundary | `docs/design/transformer/workloads.md` | `hw/npu_core/rtl/kv_cache/README.md` only | Model-only traffic/counters; implementation order tracked in `next_steps.md` | external-memory model fields only | PPA model/report checks | no RTL until decode traffic evidence justifies hardware |

## Immediate Review Actions

1. Keep this matrix updated whenever a module changes from spec-only to
   standalone, standalone to integrated, or integrated to CSR/PPA visible.
2. Do not add a new PPA field unless its provenance is one of:
   measured CSR, measured RTL event source validated against CSR, or explicitly
   labeled model-only/proxy.
3. Do not connect primitive engines to a scheduler until the valid/ready
   contract and directed handshake tests are reviewed.
4. Treat `op=1` attention softmax, `op=2` mixed PV, and `op=3` unmasked score
   scale as stage-level bring-up evidence. Full attention PPA still requires
   complete intermediate-buffer chaining and runtime-overhead provenance.
