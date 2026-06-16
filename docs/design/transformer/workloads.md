# Transformer Workload Plan / Transformer 工作负载计划

[TOC]

## 1. Role / 定位

Transformer/LLM inference is the long-term architecture driver for this
project. The existing real MNIST CNN path remains a useful CPU/NPU integration
regression, but future matrix, vector, precision, and memory-system decisions
must be evaluated against Transformer-shaped workloads.

Integration details for adding Transformer fixtures, generated firmware data,
workload manifest entries, perf report metadata, and PPA proxy fields are in
`docs/design/transformer/workload_integration.md`. This document defines the
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
| Attention group | measure coupled operations and movement | `Q*K^T`, row softmax, `P*V`, KV traffic |
| FFN group | measure dense projection pressure | FFN up/down |
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

The current RTL supports only part of this list. Attention is now the primary
Transformer workload family for upcoming PPA decisions. It is represented as a
sequence over matrix/vector/reduction/SFU primitives rather than as a dedicated
RTL attention macro.

A manifest may be introduced before RTL implementation to define required
shapes and metrics. Adding an RTL op still requires an architecture/spec
decision backed by a measured need.

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

1. Define and execute one complete tiny LLaMA-like Prefill Decoder Block.
2. Chain two complete blocks without CPU recomputation between them.
3. Reuse current matrix/vector/reduction/SFU mechanisms and add only missing
   primitives required by the complete block.
4. Keep micro workloads as diagnosis tests, but select hardware optimizations
   only from the complete-block bottleneck report.
5. Add Decode/KV execution after the Prefill block baseline is complete.

近期首先让NPU真实执行一个完整tiny Decoder Block，再串联两个Block。微算子
测试继续用于定位问题，但不能单独决定硬件优化优先级；首次架构优化必须由
完整Block PPA暴露的主要瓶颈驱动。

### Complete tiny block contract / 完整tiny block约束

The accepted executable block must include:

```text
RMSNorm
Q/K/V projection
position transform
causal Attention
Attention output projection
residual add
RMSNorm
FFN gate/up projection
activation and gate multiply
FFN down projection
residual add
```

Every stage must execute through NPU RTL and appear in PPA. Fixture-generated
inputs and expected outputs are allowed; fixture- or CPU-computed intermediate
operator results are not allowed in an accepted complete-block run.

### B0/B1 TinyLlama-derived workload / B0/B1 TinyLlama派生工作负载

The first block workload scales down TinyLlama/LLaMA structure without
removing the architectural behaviors that matter to the NPU:

| Field | B0 | B1 |
| --- | ---: | ---: |
| Prefill sequence `S` | 8 | 8 |
| Hidden size `H` | 16 | 16 |
| Query heads | 2 | 2 |
| KV heads | 1 | 1 |
| Head dimension | 8 | 8 |
| FFN intermediate | 32 | 32 |
| Decoder blocks | 1 | 2 |
| Input/weight source | deterministic synthetic INT8 | deterministic synthetic INT8, distinct per block |

`Q_heads=2, KV_heads=1` intentionally preserves grouped-query Attention. The
two query heads initially execute sequentially and share one K/V head. This is
a functional baseline, not the final scheduling policy.

B0/B1 use separate Q, K, and V projections; separate gate and up projections;
and materialized intermediate buffers. Fused projections, concurrent heads,
resident intermediates, and command lists remain later candidates whose value
must be measured against this retained baseline.

The Compiler/planner must lower every logical matrix operation into physical
`8x8x8` M/N/K tiles. The Runtime/submitter must not infer shapes or perform
tiling. B0 already exercises N-axis tiling (`H=16`, `FFN=32`) and K-axis
streaming (`H=16` or `FFN=32`); B1 additionally proves that the first block's
actual output buffer is the second block's input.

Acceptance state is explicit:

| State | Meaning |
| --- | --- |
| `planned_not_executable` | BlockPlan, buffers, numerical golden, and tile jobs exist, but at least one stage lacks RTL execution provenance |
| `partially_executable` | some stages execute in RTL; gaps remain visible and the block is not accepted |
| `executable` | every stage executes through measured NPU RTL and the complete output matches golden |

Neither a Python golden nor fixture-produced intermediates may promote a block
to `executable`.

Current executable subset:

- `transformer_tinyllama_b0_matrix_subgraph` executes the seven B0 matrix
  stages through NPU RTL as 16 `MATMUL_K_STREAM` descriptor jobs.
- `transformer_tinyllama_b0_residual_vector_subgraph` executes the two B0
  residual-add stages through NPU RTL as 32 `DESC_VECTOR_TILE_V1` descriptor
  jobs, each running an eight-lane `VADD + HALT` primitive program.
- Their status is `partially_executable` for the executable subgraphs and
  `planned_not_executable` for the complete block.
- PPA must display them as Block subgraph workloads, not as complete B0.

## 7. Post-CSR Baseline Decision / CSR 基线后的执行顺序

The production performance path now reads architectural completed-job CSR
snapshots. Transformer support should now pivot to attention without adding a
dedicated attention RTL macro:

1. Extend the workload manifest identity with attention parent/stage fields,
   `scenario` (`prefill`/`decode`), logical shape, precision, activity scope,
   and external/KV-cache traffic fields.
2. Add attention-centered workloads:
   - measured/current-RTL-compatible QK stage first;
   - model-only row softmax until vector/reduction/SFU scheduler path exists;
   - model-only PV until probability format policy is reviewed;
   - grouped `attention_prefill_s8_d8` report view.
3. Keep current projection and decode skinny-GEMM workloads as supporting
   matrix-utilization evidence, not the main Transformer decision surface.
4. Add model-only KV-cache read/write traffic accounting alongside decode
   attention workloads.
5. Use measured QK utilization, softmax/PV gaps, and modeled KV traffic to
   choose the first datapath/runtime extension.

Candidate RTL extensions, in evidence order:

| Candidate | Triggering evidence | Do not implement before |
| --- | --- | --- |
| descriptor/command-list support for repeated tiles and tensor layout | attention stage jobs are dominated by CPU/control/staging or descriptor traffic | executable attention QK manifests and traffic identity |
| vector/reduction/SFU scheduler path | attention softmax is measured only through the current bring-up path and lacks reviewed scheduler/counter semantics for target PPA | row softmax golden and stage metadata are stable |
| PV probability format support | `P*V` cannot be measured without excessive precision loss or unsupported mixed precision | attention softmax policy is reviewed |
| skinny-GEMM/GEMV utilization support such as valid-row/valid-column handling | decode attention shows poor useful-MAC ratio on the `8x8` array | compare prefill QK versus decode `m8_compat` results |
| KV-cache/external movement accounting or interface support | modeled decode bytes/token dominates event energy | external traffic fields are reportable |

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

The v1 workload surface also defines golden/model-only entries under
`workloads/transformer/micro/` for `qkv_projection`, `qk_matmul`,
`softmax_row`, `attn_pv`, `rmsnorm_row`, `ffn_up_down`, and
`kv_cache_read_write`. These entries prepare manifest, golden, and PPA fields
before the corresponding vector/reduction/SFU/KV streamer RTL is accepted.

Upcoming attention-specific entries should use the parent/stage naming and
provenance rules in `attention_workload_ppa.md`.
