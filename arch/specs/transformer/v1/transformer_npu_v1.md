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
| wrapper / CSR | thin CPU-visible control, status, descriptor-pointer forwarding, perf CSR snapshots; no graph lowering, tiling, or fusion |
| descriptor engine / Command Processor | reads and validates job descriptors; executes descriptor-local control such as K chunks and movement launch |
| uop scheduler | issues primitive uops in program order; v1 is in-order with barriers |
| matrix engine | int8 x int8 tile GEMM, K-stream accumulation, GEMV/skinny-GEMM accounting |
| accumulator file | explicit int32 tile storage for partial sums and store path |
| vector engine | lane-wise add/sub/mul/scale/requant/clamp primitives |
| reduction engine | max/sum/sumsq reductions up to 128 elements |
| SFU | fixed-point EXP/RECIP/RSQRT LUT primitives |
| memory / scratchpad / data mover | system SRAM workspace movement and internal scratchpad staging |
| KV cache streamer | v1 specifies layout, bytes, and counters; complex cache RTL is later |

Static Compiler owns graph/operator lowering, fusion choice, logical tiling,
and descriptor/command generation. Runtime owns address binding, legal dynamic
shape binding or variant selection, submission, synchronization, and
error/perf collection. Hardware executes the supplied contract; it does not
reconstruct the logical graph.

## Large Matrix Tiling Contract

Status: proposed architecture contract for review; only the single-output-tile
K-stream subset is currently implemented.

The physical Matrix Engine contract remains one `8x8x8` tile operation:

```text
A_tile[8,8] * B_tile[8,8] -> C_partial[8,8]
```

A logical `C[M,N] = A[M,K] * B[K,N]` is represented by:

```text
TM = ceil(M / 8)
TN = ceil(N / 8)
TK = ceil(K / 8)

for m_tile in 0 .. TM-1:
  for n_tile in 0 .. TN-1:
    clear accumulator tile
    for k_tile in 0 .. TK-1:
      load/prefetch A[m_tile,k_tile] and B[k_tile,n_tile]
      execute one physical matrix tile
      accumulate into the same int32 accumulator tile
    store C[m_tile,n_tile]
```

The first executable baseline uses one descriptor per logical output tile
`C[m_tile,n_tile]`. The descriptor's existing `k_chunks` field controls the
inner K loop. Compiler/runtime generate the outer M/N descriptor sequence.
This extends the existing K-stream mechanism without adding a matrix ISA
instruction.

中文说明：物理Matrix Engine仍然只计算一次`8x8x8`。大矩阵由Compiler沿
M、N、K三个维度切块。每个输出`C` tile对应一个descriptor；硬件现有
K-stream在同一个Accumulator File tile中累加多个K chunk；Runtime依次启动
不同M/N位置的descriptor并把结果写回对应地址。

Boundary tiles use zero-filled invalid A/B lanes for the first baseline.
Compiler records logical valid rows/columns so useful work and tail waste are
distinguishable. Hardware must never read outside the logical source tensor.
Skipping zero-only/totally masked tiles is a later compiler optimization and
must not change numerical results.

For the current one-K-slice-per-cycle Matrix Engine:

```text
physical_tile_invocations = TM * TN * TK
ideal_matrix_active_cycles = physical_tile_invocations * 8
issued_mac_capacity = physical_tile_invocations * 8 * 8 * 8
useful_mac_ops = M * N * K
tail_waste_mac_capacity = issued_mac_capacity - useful_mac_ops
```

These formulas cover Matrix Engine active time only. Descriptor processing,
Data Mover load/store, accumulator clear/commit, bank conflicts, and stalls
must be reported separately and must not be hidden inside theoretical Matrix
cycles.

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
