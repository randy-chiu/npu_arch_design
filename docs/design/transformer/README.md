# Transformer Design Docs

This directory owns Transformer-oriented NPU v1 design work. It keeps v1
architecture, workload, integration, and execution-plan material separate from
the public SoC/wrapper/perf/PPA contracts in `docs/design/`.

Each RTL module should have one module-level design source. Subfeatures such as
SFU EXP LUT generation or vector requantization live inside their owning module
document unless they become a separate RTL module.

| Document | Scope |
| --- | --- |
| `transformer_npu_v1.md` | Unified tensor NPU v1 architecture baseline, primitive uops, counters, and staged implementation |
| `workloads.md` | LLM prefill/decode workload progression, tiny micro workloads, and metrics |
| `workload_integration.md` | Fixture/golden/manifest/firmware/perf/PPA integration path |
| `attention_sequence_v1.md` | Attention as a compiler/runtime sequence over matrix/vector/reduction/SFU primitives |
| `attention_numerical_v1.md` | Fixed-point attention math, examples, Q formats, and golden/RTL consistency rules |
| `attention_workload_ppa.md` | Attention workload grouping, measured/model-only boundaries, and PPA fields |
| `transformer_v1_test_plan.md` | Transformer V1 verification targets, constraints, acceptance criteria, and deferred tests |
| `software_runtime_compiler_attention.md` | Overview for attention software execution ownership and next coding order |
| `attention_operators_v1.md` | Operator contracts for attention QK, score scale/mask, softmax, PV, and composite SDPA |
| `attention_compiler_v1.md` | Compiler plan for lowering SDPA to primitive stages, buffers, and runtime jobs |
| `attention_runtime_v1.md` | Runtime design for descriptor launch, SRAM buffers, firmware dispatch, and grouped PPA |
| `vector_engine_v1.md` | Vector primitive engine contract for ADD/SUB/MUL/SCALE/REQUANT/CLAMP |
| `reduction_engine_v1.md` | Reduction primitive engine contract for MAX/SUM/SUMSQ |
| `sfu_v1.md` | Fixed-point EXP/RECIP/RSQRT SFU contract |
| `next_steps.md` | Ordered implementation plan for the next development rounds |

V1 keeps MNIST/CNN as regression and does not fork a separate Transformer core.
Attention v1 is also not a separate RTL macro; it is a software/compiler
scheduled sequence over the shared primitive engines.

Current standalone primitive RTL modules:

```text
hw/npu_core/rtl/vector/vector_engine.sv
hw/npu_core/rtl/reduction/reduction_engine.sv
hw/npu_core/rtl/sfu/sfu_lut.sv
```
