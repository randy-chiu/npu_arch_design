# Transformer Design Docs

This directory owns Transformer-oriented NPU v1 design work. It keeps v1
architecture, workload, integration, and execution-plan material separate from
the public SoC/wrapper/perf/PPA contracts in `docs/design/`.

| Document | Scope |
| --- | --- |
| `transformer_npu_v1.md` | Unified tensor NPU v1 architecture baseline, primitive uops, counters, and staged implementation |
| `workloads.md` | LLM prefill/decode workload progression, tiny micro workloads, and metrics |
| `workload_integration.md` | Fixture/golden/manifest/firmware/perf/PPA integration path |
| `vector_engine_v1.md` | Vector primitive engine contract for ADD/SUB/MUL/SCALE/REQUANT/CLAMP |
| `reduction_engine_v1.md` | Reduction primitive engine contract for MAX/SUM/SUMSQ |
| `sfu_v1.md` | Fixed-point EXP/RECIP/RSQRT SFU contract |
| `next_steps.md` | Ordered implementation plan for the next development rounds |

V1 keeps MNIST/CNN as regression and does not fork a separate Transformer core.

Current standalone primitive RTL modules:

```text
hw/npu_core/rtl/vector/vector_engine.sv
hw/npu_core/rtl/reduction/reduction_engine.sv
hw/npu_core/rtl/sfu/sfu_lut.sv
```
