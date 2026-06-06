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
| Core top / launch modes | `docs/design/npu_core.md` | `hw/npu_core/rtl/npu_v0_compute_cluster.sv` | Integrated through wrapper `start/op` and host windows | core activity and wait-for-prefetched-data events | `make npu-core-sim`, `make perf-l0-transformer` | fixed attention sequencing still colocated with cluster integration |
| Uop scheduler / dispatcher | `docs/design/npu_core.md` | `hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv` | Integrated for common `op=0` LOAD/MATMUL/STORE/HALT fetch, decode, issue, and completion wait | scheduler active/wait counters published in PPA snapshot | `make npu-core-sim`, `make perf-l0-transformer` | vector/reduction/SFU issue migration remains |
| Matrix array | `docs/design/npu_core.md` | `hw/npu_core/rtl/matrix/matmul_array.sv` | Integrated for int8 matmul, K-stream, and mixed PV mode; `docs/matmul_array_a1.md` is supporting A1 timing background | real matrix datapath active cycles are separate from scheduler transaction wait | `make npu-core-sim`, `make perf-l0-transformer` | no GEMV/valid-row/valid-column support, no per-mode event counters |
| Accumulator file | `docs/design/transformer/transformer_npu_v1.md` | `hw/npu_core/rtl/matrix/accumulator_file.sv` | Integrated into `npu_v0_compute_cluster` for matmul/K-stream residency | local read/write/clear/residency/spill counters exist but are not wrapper CSR surfaced | `make npu-core-sim`, SoC perf profiles | CSR/report exposure pending reviewed event-source plan |
| Vector engine | `docs/design/transformer/vector_engine_v1.md` | `hw/npu_core/rtl/vector/vector_engine.sv` | Standalone and used by `op=1`; valid/ready shim is standalone only | shim-local cycle, accepted-op, and lane-op counters; no CSR exposure | `make primitive-engines-sim`, `make perf-l0-transformer` | no scheduler issue path, no requant v2, no mask-select semantics |
| Reduction engine | `docs/design/transformer/reduction_engine_v1.md` | `hw/npu_core/rtl/reduction/reduction_engine.sv` | Standalone and used by `op=1`; valid/ready shim is standalone only | shim-local cycle, accepted-op, and element-op counters; no CSR exposure | `make primitive-engines-sim`, `make perf-l0-transformer` | no scheduler issue path, no segmented-row contract, no explicit mask input |
| SFU | `docs/design/transformer/sfu_v1.md` | `hw/npu_core/rtl/sfu/sfu_lut.sv` | Standalone and used by `op=1`; valid/ready shim is standalone only | shim-local cycle and EXP/RECIP/RSQRT counters; no CSR exposure | `make primitive-engines-sim`, `make perf-l0-transformer` | current EXP is bring-up approximation, 257-entry target LUT not implemented, no production RECIP/RSQRT latency model |
| Primitive valid/ready | `docs/design/transformer/primitive_valid_ready_v1.md` | `hw/npu_core/rtl/primitive_handshake_shims.sv` | Standalone compatibility shims; current SoC path still uses start/done | local handshake/event counters implemented and tested; no CSR exposure | `make primitive-engines-sim` directed handshake, counter, and reset tests | connect reviewed scheduler/native issue path before CSR integration |
| Uop/primitive scheduler boundary | `arch/specs/transformer/v1/uop_isa_v1.md` | `hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv` | Common v0 matmul uop path integrated; primitive issue remains incomplete | active/wait counters integrated | v0 `make npu-core-sim` only | no issue queue, response queue, or vector/reduction/SFU command stream |
| Memory / scratchpad boundary | `docs/design/npu_core.md` | internal arrays in `npu_v0_compute_cluster`; `hw/npu_core/rtl/memory/npu_v0_data_mover.sv` | NPU core command processor and data mover own SoC-side movement | data mover counters are published through Host-visible perf CSRs; no core bank conflict counters | `make perf-report`, `make perf-l0-transformer` | no production memory protocol, no bank conflict model, no double buffering contract |
| Attention runtime group flow | `docs/design/transformer/attention_runtime_v1.md` | host/compiler/runtime tooling | Generated runtime table launches QK, executable unmasked scale/mask, softmax, and PV; QK output SRAM feeds scale input | measured stage perf includes scale/mask; parent group remains software-sequenced and runtime overhead is not measured | `make cpu-soc-transformer`, `make ppa-l0-report WORKLOAD_PROFILE=transformer`, PPA schema tests | scale output is not yet connected to all softmax rows, probabilities are independently staged for PV, descriptor filling remains stage-specific |
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
