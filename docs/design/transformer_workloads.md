# Transformer Workload Plan / Transformer 工作负载计划

[TOC]

## 1. Role / 定位

Transformer/LLM inference is the long-term architecture driver for this
project. The existing real MNIST CNN path remains a useful CPU/NPU integration
regression, but future matrix, vector, precision, and memory-system decisions
must be evaluated against Transformer-shaped workloads.

Transformer/LLM 推理分为两类显著不同的压力：

```text
prefill: 处理输入序列，GEMM 较大，计算利用率通常更高
decode:  每次生成 token，GEMV/skinny GEMM 与 KV-cache 流量占主导
```

评估中必须区分这两类场景，不能仅使用一个 matmul shape 代表 LLM 推理。

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
| GEMV / skinny GEMM | decode utilization | cycles/token proxy, lane utilization, bytes/MAC |
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

