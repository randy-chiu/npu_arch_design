# Transformer Workload Plan / Transformer 工作负载计划

[TOC]

## 1. Role / 定位

Transformer/LLM inference is the long-term architecture driver for this
project. The existing real MNIST CNN path remains a useful CPU/NPU integration
regression, but future matrix, vector, precision, and memory-system decisions
must be evaluated against Transformer-shaped workloads.

Integration details for adding Transformer fixtures, generated firmware data,
workload manifest entries, perf report metadata, and PPA proxy fields are in
`docs/design/transformer_workload_integration.md`. This document defines the
workload progression and metrics; the integration document defines how those
workloads enter the existing verified NPU path.

Transformer/LLM 推理分为两类显著不同的压力：

```text
prefill: 处理输入序列，GEMM 较大，计算利用率通常更高
decode:  每次生成 token，GEMV/skinny GEMM 与 KV-cache 流量占主导
```

评估中必须区分这两类场景，不能仅使用一个 matmul shape 代表 LLM 推理。

Terminology note: earlier drafts used `proxy` for decode micro workloads. In
this document, `proxy` means an approximation that uses a current RTL-compatible
shape to represent pressure from an operation that the hardware does not yet
implement directly. For clarity, executable decode workloads should prefer
names such as `m8_compat` instead of bare `proxy`.

术语说明：早期草案里 `proxy` 表示“用当前 RTL 能跑的形状近似一个尚未直接支持的
真实场景”。为了避免被理解成软件代理或硬件代理，后续可执行 decode workload 命名
优先使用 `m8_compat` 这类名字，而不是单独使用 `proxy`。

## 2. Workload Levels / 工作负载层级

| Level | Purpose | Initial content |
| --- | --- | --- |
| Micro kernel | isolate architectural resources | GEMM, GEMV, softmax, RMSNorm, KV-cache traffic |
| Attention/FFN group | measure coupled operations and movement | QKV, `Q*K^T`, softmax, `P*V`, FFN up/down |
| Tiny decoder block | verify a recognizable inference block | norm + attention + residual + FFN |
| Trace-driven model view | study realistic shape/traffic trends | scaled prefill/decode manifests |

Directory contract:

```text
workloads/transformer/micro/
workloads/transformer/block/
workloads/transformer/traces/prefill/
workloads/transformer/traces/decode/
```

## 3. First Micro Workloads / 首批微工作负载

| Workload | Hardware pressure | Required measurements |
| --- | --- | --- |
| GEMM | matrix throughput and tiling | cycles, utilization, MACs, movement, energy/MAC |
| GEMV / skinny GEMM | decode utilization | cycles/token estimate, lane utilization, bytes/MAC |
| QKV projection | repeated linear layers | reuse, data layout, weight traffic |
| `Q*K^T` | attention score compute | sequence-shape scaling, accumulator behavior |
| attention softmax | reduction/SFU path | vector latency, accuracy, energy |
| attention value matmul | score/value movement | bandwidth and buffer need |
| RMSNorm | vector + reduction | missing-op requirements and cost |
| FFN up/down | dominant projection compute | GEMM scaling and activation path |
| KV-cache traffic | decode memory pressure | bytes/token and external-energy estimate |

The current RTL supports only part of this list. A manifest may be introduced
before RTL implementation to define required shapes and metrics. Adding an RTL
op still requires an architecture/spec decision backed by a measured need.

KV-cache traffic is included because autoregressive decode repeatedly reads
previous keys/values and writes the new token's key/value state. This often
turns decode into a memory-traffic and external-energy problem rather than a
pure matrix-throughput problem. The first KV-cache workload is model-only: it
does not require an RTL cache or memory-interface implementation, but it makes
bytes/token and estimated external-memory energy visible next to measured
on-chip NPU activity.

加入 KV-cache traffic 的原因是：自回归 decode 每生成一个 token 都要读取历史
key/value，并写入当前 token 的 key/value。这个场景经常受 memory traffic 和
外部存储能耗主导，而不是单纯受矩阵吞吐限制。第一版 KV-cache workload 只是
model-only，不要求已经实现 RTL cache 或外部 memory interface；它的作用是让
bytes/token 和估算外部 memory energy 出现在报告里，和实测 on-chip NPU activity
分开比较。

## 4. Precision Progression / 精度路线

The current datapath baseline remains signed INT8 matmul with INT32
accumulation. Transformer evaluation should introduce precision decisions in
stages:

1. INT8 baseline for infrastructure and comparable PPA results.
2. Define accuracy/golden checks for Transformer kernels and tiny block.
3. Evaluate whether weight-only lower precision such as W4 is required by
   decode traffic/energy evidence.
4. Do not add precision formats without including conversion, scale metadata,
   accumulator, memory-traffic, and PPA impact.

## 5. Memory And Energy Requirements / 存储与能耗要求

Transformer manifests must eventually identify:

```text
activation bytes
weight bytes
KV-cache read bytes
KV-cache write bytes
on-chip resident bytes
external-memory bytes/token
```

NPU RTL power estimation covers on-chip implementation activity. Weight and
KV-cache external-memory energy may initially be modeled rather than
synthesized, but it must be reported as a distinct contribution.

## 6. Near-Term Deliverables / 近期交付

1. Establish workload manifest format and directory entry point.
2. Add initial prefill/decode micro-kernel manifests for GEMM/GEMV, RMSNorm,
   and KV-cache traffic.
3. Reuse current matmul/softmax simulation and perf mechanisms wherever the
   operator contract already exists.
4. Use resulting workload gaps and PPA baselines to select the first
   Transformer-driven RTL extension.

## 7. Post-CSR Baseline Decision / CSR 基线后的执行顺序

The production performance path now reads architectural completed-job CSR
snapshots. Transformer support should start without adding speculative RTL
operators:

1. Extend the workload manifest identity with `scenario` (`prefill`/`decode`),
   logical shape, precision, activity scope, and external/KV-cache traffic
   fields.
2. Add executable INT8 projection proxies using the existing matmul and
   K-stream execution path:
   - prefill projection GEMM, tiled into current `8x8x8` jobs;
   - decode skinny-GEMM `m8_compat` workload with `M=8`, explicitly labeled as
     current-array-compatible rather than true single-token GEMV.
3. Add model-only KV-cache read/write traffic accounting alongside decode
   results before treating latency/energy as architecture evidence.
4. Use measured projection utilization and modeled KV traffic to choose the
   first RTL extension.

Candidate RTL extensions, in evidence order:

| Candidate | Triggering evidence | Do not implement before |
| --- | --- | --- |
| descriptor/command-list support for repeated tiles and tensor layout | projection jobs are dominated by CPU/control/staging or descriptor traffic | executable projection manifests and traffic identity |
| skinny-GEMM/GEMV utilization support such as valid-row/valid-column handling | decode `m8_compat` shows poor useful-MAC ratio on the `8x8` array | compare prefill versus decode `m8_compat` results |
| KV-cache/external movement accounting or interface support | modeled decode bytes/token dominates event energy | external traffic fields are reportable |
| RMSNorm/reduction/SFU extension | a tiny block cannot be represented using current operators | projection and traffic baselines are stable |

`mac_ops`, `instr_count`, and execution error/timeout CSRs remain planned
contracts. `mac_ops` becomes high priority before comparing shapes whose useful
work differs from fully occupied `8x8x8` tiles; it should count committed useful
MAC work rather than infer it from observed matmul phase cycles.

## 8. Current Executable Baseline / 当前可执行基线

The first executable Transformer baseline is intentionally tiny and uses only
current RTL-supported K-stream matmul:

| Workload | Scenario | Shape | Meaning |
| --- | --- | ---: | --- |
| `transformer_prefill_gemm_tiny` | `transformer_prefill` | `8x8x16` | two-K-chunk projection GEMM smoke for the Transformer generation/report path |
| `transformer_decode_skinny_gemm_m8_compat` | `transformer_decode` | `8x8x8` | current-array-compatible decode skinny GEMM approximation |
| `transformer_kv_cache_traffic_tiny` | `transformer_decode` | model-only | KV-cache external-memory bytes/token pressure |

These shapes validate the workload/perf/PPA evidence path. They are not yet a
claim about full model-scale Transformer performance.

当前第一版可执行 Transformer baseline 故意很小，只使用当前 RTL 已支持的
K-stream matmul。它的作用是验证 workload 生成、firmware 执行、perf report 和 PPA
proxy 证据链，不代表完整模型规模性能结论。
