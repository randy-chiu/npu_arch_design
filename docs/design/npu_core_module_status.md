# NPU Core Module Status

This document tracks the current NPU core module review state. It is the
short-form checklist that ties design docs, RTL, integration status, counters,
tests, and next blockers together.

## Review Boundary

The current core is still `hw/npu_core/rtl/npu_v0_top.sv`. Transformer v1 does
not introduce a separate attention core. It adds shared primitive capabilities
and runtime/compiler sequencing over the same wrapper/core/perf/PPA framework.

Status labels:

| Label | Meaning |
| --- | --- |
| Implemented | RTL/tooling exists in the current tree |
| Integrated | reachable through `npu_v0_top`, wrapper descriptors, or SoC firmware |
| Standalone | RTL exists but is not a production scheduler path |
| Spec-only | documented/configured, no RTL path yet |
| Model-only | report/golden metadata exists, no measured hardware execution |

## Module Matrix

| Module | Design source | RTL/tooling source | Integration status | Counter status | Verification entry | Blocking gaps |
| --- | --- | --- | --- | --- | --- | --- |
| Core top / launch modes | `docs/design/npu_core.md` | `hw/npu_core/rtl/npu_v0_top.sv` | Integrated through wrapper `start/op` and host windows | core phase perf signals only | `make npu-core-sim`, `make perf-l0-transformer` | no primitive scheduler, no grouped attention launch, no input/output backpressure |
| Matrix array | `docs/design/npu_core.md` | `hw/npu_core/rtl/matrix/matmul_array.sv` | Integrated for int8 matmul, K-stream, and mixed PV mode; `docs/matmul_array_a1.md` is supporting A1 timing background | matrix active phase is visible through core phase counters; mixed-mode MAC cost is proxy only | `make npu-core-sim`, `make perf-l0-transformer` | no GEMV/valid-row/valid-column support, no per-mode event counters |
| Accumulator file | `docs/design/transformer/transformer_npu_v1.md` | `hw/npu_core/rtl/matrix/accumulator_file.sv` | Integrated into `npu_v0_top` for matmul/K-stream residency | local read/write/clear/residency/spill counters exist but are not wrapper CSR surfaced | `make npu-core-sim`, SoC perf profiles | CSR/report exposure pending reviewed event-source plan |
| Vector engine | `docs/design/transformer/vector_engine_v1.md` | `hw/npu_core/rtl/vector/vector_engine.sv` | Standalone and used by the `op=1` attention softmax bring-up sequence | `active` only; no real stall/op counters | `make primitive-engines-sim`, `make perf-l0-transformer` | no valid/ready, no scheduler issue path, no requant v2, no mask-select semantics |
| Reduction engine | `docs/design/transformer/reduction_engine_v1.md` | `hw/npu_core/rtl/reduction/reduction_engine.sv` | Standalone and used by the `op=1` attention softmax bring-up sequence | `active` only; no element/op counters | `make primitive-engines-sim`, `make perf-l0-transformer` | no valid/ready, no segmented-row contract, no explicit mask input, no measured `reduction_element_ops` |
| SFU | `docs/design/transformer/sfu_v1.md` | `hw/npu_core/rtl/sfu/sfu_lut.sv` | Standalone and used by the `op=1` attention softmax bring-up sequence | `active` only; no EXP/RECIP/RSQRT op counters | `make primitive-engines-sim`, `make perf-l0-transformer` | current EXP is bring-up approximation, 257-entry target LUT not implemented, no production RECIP/RSQRT latency model |
| Primitive valid/ready | `docs/design/transformer/primitive_valid_ready_v1.md` | none yet | Spec-only | counter semantics proposed, not implemented | future directed RTL handshake tests | review must complete before scheduler/CSR integration |
| Uop/primitive scheduler boundary | `arch/specs/transformer/v1/uop_isa_v1.md` | `hw/npu_core/rtl/scheduler/README.md` only | Spec-only; v0 uop fetch remains in-order and limited; `next_steps.md` tracks work order | no scheduler/control counters | v0 `make npu-core-sim` only | no issue queue, no response handling, no primitive command stream |
| Memory / scratchpad boundary | `docs/design/npu_core.md` | internal arrays in `npu_v0_top`; `hw/npu_core/rtl/memory/README.md` planned | wrapper owns SoC memory movement; wrapper/data-mover docs own SoC-side movement | data mover counters exist at wrapper level; no core bank conflict counters | `make perf-report`, `make perf-l0-transformer` | no real core SRAM master, no bank conflict model, no double buffering contract |
| Attention runtime group flow | `docs/design/transformer/attention_runtime_v1.md` | host/compiler/runtime tooling | Stage jobs measured; compiler lowers manifest attention into a generated runtime-job table for QK, softmax, and PV; scale/mask is still materialized | stage perf exists; parent group is `software_group_measured_stages`; runtime overhead and scale/mask cost are not measured | `make cpu-soc-transformer`, `make ppa-l0-report WORKLOAD_PROFILE=transformer`, PPA schema tests | scale/mask bridge not executable, descriptor filling remains stage-specific, intermediate buffers are not yet producer-to-consumer chained |
| KV cache boundary | `docs/design/transformer/workloads.md` | `hw/npu_core/rtl/kv_cache/README.md` only | Model-only traffic/counters; implementation order tracked in `next_steps.md` | external-memory model fields only | PPA model/report checks | no RTL until decode traffic evidence justifies hardware |

## Immediate Review Actions

1. Keep this matrix updated whenever a module changes from spec-only to
   standalone, standalone to integrated, or integrated to CSR/PPA visible.
2. Do not add a new PPA field unless its provenance is one of:
   measured CSR, measured RTL event source validated against CSR, or explicitly
   labeled model-only/proxy.
3. Do not connect primitive engines to a scheduler until the valid/ready
   contract and directed handshake tests are reviewed.
4. Treat `op=1` attention softmax and `op=2` mixed PV as stage-level bring-up
   evidence. Full attention PPA still requires executable scale/mask,
   intermediate-buffer chaining, and runtime-overhead provenance.
