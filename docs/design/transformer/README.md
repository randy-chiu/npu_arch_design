# Transformer Design Docs

This directory owns Transformer-oriented NPU v1 design work. It keeps v1
architecture, workload, integration, and execution-plan material separate from
the public SoC/wrapper/perf/PPA contracts in `docs/design/`.

Each RTL module should have one module-level design source. Subfeatures such as
SFU EXP LUT generation or vector requantization live inside their owning module
document unless they become a separate RTL module.

Core architecture documents are maintained bilingually. This means their
major sections must contain both an English contract and a Chinese explanation
of the same design intent, ownership, tradeoffs, and status. A detached Chinese
summary at the beginning is not sufficient because it becomes stale and makes
the later English-only decisions difficult to review.

核心架构文档采用中英文双语维护。每个主要章节都需要同时说明设计目标、模块
职责、方案取舍和当前状态；不能只在文档开头增加一段中文摘要，而后续正式
设计仍全部使用英文。当前核心文档包括：

- `transformer_npu_v1.md`
- `attention_sequence_v1.md`
- `attention_compiler_v1.md`
- `attention_runtime_v1.md`
- 对应的 `arch/specs/transformer/v1/` 芯片架构契约

| Document | Scope |
| --- | --- |
| `transformer_npu_v1.md` | Unified tensor NPU v1 architecture baseline, large-matrix tiling, primitive uops, counters, and staged implementation |
| `workloads.md` | LLM prefill/decode workload progression, tiny micro workloads, and metrics |
| `workload_integration.md` | Fixture/golden/manifest/firmware/perf/PPA integration path |
| `attention_sequence_v1.md` | Attention sequence and formal end-to-end mask architecture decision/tradeoff source |
| `attention_numerical_v1.md` | Fixed-point attention math, examples, Q formats, and golden/RTL consistency rules |
| `attention_workload_ppa.md` | Attention workload grouping, measured/model-only boundaries, and PPA fields |
| `transformer_v1_test_plan.md` | Transformer V1 verification targets, constraints, acceptance criteria, and deferred tests |
| `software_runtime_compiler_attention.md` | System-level Compiler/Runtime/Wrapper/Core ownership, fusion/submission models, and attention execution overview |
| `attention_operators_v1.md` | Operator contracts for attention QK, score scale/mask, softmax, PV, and composite SDPA |
| `attention_compiler_v1.md` | Compiler plan for lowering SDPA to primitive stages, buffers, and runtime jobs |
| `attention_runtime_v1.md` | Runtime design for descriptor launch, SRAM buffers, firmware dispatch, and grouped PPA |
| `vector_engine_v1.md` | Vector primitive engine contract for ADD/SUB/MUL/SCALE/REQUANT/CLAMP |
| `reduction_engine_v1.md` | Reduction primitive engine contract for MAX/SUM/SUMSQ |
| `sfu_v1.md` | Fixed-point EXP/RECIP/RSQRT SFU contract |
| `primitive_valid_ready_v1.md` | Meaning of primitive operations; why start/done is insufficient; scheduler command/response handshake |
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
