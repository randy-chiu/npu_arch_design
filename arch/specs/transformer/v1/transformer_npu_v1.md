# Transformer NPU v1 Architecture Spec

## Target

Transformer NPU v1 upgrades the current CNN/MNIST regression platform into a
unified tensor NPU baseline for edge LLM inference experiments. It is not a
complete LLaMA implementation and does not introduce a fused attention
pipeline.

Initial model envelope:

| Field | v1 target |
| --- | --- |
| Batch | single batch |
| Model | tiny decoder-only Transformer first |
| Sequence lengths | 32 and 128 |
| Hidden sizes | 64 and 128 |
| Heads | 4 |
| Head dimensions | 16 and 32 |
| Activation / weight | int8 |
| Accumulator | int32 |
| Softmax internal | fixed-point |
| KV cache | int8 v1, specified and counted before complex RTL |

The v1 rule is primitive uops first. Software/compiler micro-kernels may expand
`SOFTMAX_ROW` and `RMSNORM_ROW` into primitive uops, but hardware v1 does not
need to recognize those rows as fused macro-ops.

## Module Split

The architecture remains one tensor NPU, not separate CNN and Transformer
cores.

| Module | Responsibility |
| --- | --- |
| wrapper / CSR | CPU-visible control, status, descriptor pointers, perf CSR snapshots |
| descriptor engine | reads job descriptors and validates shape/workspace fields |
| uop scheduler | issues primitive uops in program order; v1 is in-order with barriers |
| matrix engine | int8 x int8 tile GEMM, K-stream accumulation, GEMV/skinny-GEMM accounting |
| accumulator file | explicit int32 tile storage for partial sums and store path |
| vector engine | lane-wise add/sub/mul/scale/requant/clamp primitives |
| reduction engine | max/sum/sumsq reductions up to 128 elements |
| SFU | fixed-point EXP/RECIP/RSQRT LUT primitives |
| memory / scratchpad / data mover | system SRAM workspace movement and internal scratchpad staging |
| KV cache streamer | v1 specifies layout, bytes, and counters; complex cache RTL is later |

## Activity Accounting

Every engine counter uses the same state definitions:

| Counter class | Definition |
| --- | --- |
| `active_cycles` | engine has assigned work and makes forward progress |
| `stall_cycles` | engine has assigned work but cannot make progress |
| `idle_cycles` | engine has no assigned work |

Counters are per engine. A cycle may be active for one engine and idle or stall
for another. `PERF_CORE_ACTIVE` is the union-like core busy interval reported by
the wrapper/core contract; detailed utilization uses per-engine counters.

## PPA Metrics

Required v1 perf/PPA fields:

| Metric | Meaning |
| --- | --- |
| `matrix_active_cycles` | matrix engine progress cycles |
| `vector_active_cycles` | vector engine progress cycles |
| `reduction_active_cycles` | reduction engine progress cycles |
| `sfu_active_cycles` | SFU progress cycles |
| `data_mover_active_cycles` | data mover transfer/progress cycles |
| `stall_cycles_by_engine` | stall cycles for matrix/vector/reduction/SFU/data mover |
| `effective_mac_ops` | useful MAC work from workload shape, normally `M*N*K` |
| `peak_mac_capacity` | `matrix_active_cycles * PEAK_MACS_PER_CYCLE` |
| `matrix_utilization` | `effective_mac_ops / peak_mac_capacity` |
| `gemv_utilization` | utilization for workloads with `M=1` or `N=1`, otherwise null |
| `skinny_gemm_utilization` | utilization for skinny decode shapes, otherwise null |
| `tail_waste_mac_capacity` | `peak_mac_capacity - effective_mac_ops` |
| `kv_read_bytes` | KV cache bytes read per workload or token model |
| `kv_write_bytes` | KV cache bytes written per workload or token model |
| `bytes_per_token` | decode traffic normalized per token when available |
| `energy_proxy_per_token` | Level 0 normalized energy proxy per token, not joules |

The report must distinguish measured RTL cycle/movement evidence from modeled
external-memory traffic and normalized proxy energy.

## Transformer Terminology

Primitive uop:
: A directly scheduled hardware primitive such as `MATMUL`, `GEMV`,
  `REDUCE_MAX`, or `SFU_EXP`.

Micro-kernel:
: A compiler/software sequence of primitive uops implementing an operator row,
  for example stable softmax or RMSNorm.

Macro-op expansion:
: A future scheduler/compiler feature where `SOFTMAX_ROW` or `RMSNORM_ROW` is
  accepted as a compact program operation and expanded into primitive uops.

Fused hardware pipeline:
: A dedicated datapath that fuses multiple logical stages such as
  `QK -> softmax -> PV`. This is explicitly out of v1 scope.
