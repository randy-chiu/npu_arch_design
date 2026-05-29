# Primitive Uop ISA v1

## Target

Uop ISA v1 defines primitive operations for a unified tensor NPU. Complex
operators are expressed as micro-kernels unless a later architecture decision
adds macro-op expansion or fused hardware.

## Primitive Uops

| Uop | Semantics |
| --- | --- |
| `LOAD_TILE` | load tensor tile from scratch/system workspace into internal buffer |
| `STORE_TILE` | store internal tile or accumulator tile to scratch/system workspace |
| `MATMUL` | int8 x int8 matrix tile accumulate to int32 accumulator |
| `GEMV` | decode-oriented matrix/vector primitive, utilization-accounted separately |
| `VEC_ADD` | lane-wise vector add |
| `VEC_SUB` | lane-wise vector subtract |
| `VEC_MUL` | lane-wise vector multiply |
| `VEC_SCALE` | lane-wise scale by fixed-point scalar |
| `VEC_REQUANT` | requantize wider intermediate to target integer format |
| `VEC_CLAMP` | `y[i] = min(max(x[i], low), high)` |
| `REDUCE_MAX` | maximum over a row/vector |
| `REDUCE_SUM` | sum over a row/vector |
| `REDUCE_SUMSQ` | sum of squares over a row/vector |
| `SFU_EXP` | fixed-point exponential LUT |
| `SFU_RECIP` | fixed-point reciprocal LUT/approximation |
| `SFU_RSQRT` | fixed-point reciprocal square-root LUT/approximation |
| `CLEAR_ACC` | clear accumulator tile/bank |
| `BARRIER` | in-order dependency and visibility boundary |
| `HALT` | end program |

## Non-Primitive Operators

`SOFTMAX_ROW` and `RMSNORM_ROW` are v1 compiler/software micro-kernels. They
may appear in descriptor `job_type` for scheduling and report identity, but the
v1 primitive program expands them into reduction/vector/SFU uops.

Macro-op expansion and fused hardware pipelines are later architecture
decisions and must be justified by perf/PPA evidence.
