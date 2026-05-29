# Reduction Engine v1 Design

## 1. Target / 目标

Reduction Engine v1 provides row/vector reductions for Transformer
micro-kernels. It is a primitive engine, not a softmax or RMSNorm macro-op.

Required v1 reductions:

- `REDUCE_MAX` for stable softmax;
- `REDUCE_SUM` for softmax denominator;
- `REDUCE_SUMSQ` for RMSNorm.

## 2. Overall Design / 整体设计思路

The engine consumes vector rows in chunks of `REDUCE_LANES=8` and accumulates a
scalar result:

```text
vector/source buffer
  -> lane reduction stage
  -> scalar accumulator
  -> scalar result register
```

The first module should live under:

```text
hw/npu_core/rtl/reduction/
```

It should support a row length up to 128 through repeated chunks.

## 3. Key Details / 重点细节

Parameters:

| Field | v1 value |
| --- | --- |
| `REDUCE_LANES` | 8 |
| `MAX_LEN` | 128 |
| input type | int16 or int32 |
| sum output | int32/uint32 |
| max output | signed int32 |
| issue model | start + length + op, then done |

Ops:

| Op | Semantics |
| --- | --- |
| `REDUCE_MAX` | signed maximum over valid elements |
| `REDUCE_SUM` | arithmetic sum over valid elements |
| `REDUCE_SUMSQ` | sum of `x[i] * x[i]` over valid elements |

Counter semantics:

| Counter | Meaning |
| --- | --- |
| `reduction_active_cycles` | at least one chunk consumed or scalar state updated |
| `reduction_stall_cycles` | reduction assigned but input chunk unavailable |
| `reduction_idle_cycles` | no reduction op assigned |

For v1, overflow behavior must be deterministic. RMSNorm `SUMSQ` should use a
wide enough intermediate for hidden sizes 64/128 with int8/int16 inputs before
downstream approximation.

## 4. Verification / 验证测试

Initial tests:

- `REDUCE_MAX` on mixed signed rows;
- `REDUCE_SUM` with valid lengths 1, 8, 16, 64, 128;
- `REDUCE_SUMSQ` for RMSNorm-sized vectors;
- valid-lane tail behavior.

Golden source:

```text
sw/tools/transformer/micro_golden.py
```

## 5. Implementation Priority / 实现优先级

1. Build standalone reduction RTL and testbench. Status: implemented as
   `hw/npu_core/rtl/reduction/reduction_engine.sv`.
2. Cover row lengths up to 128 before connecting to scheduler. Status:
   module supports `MAX_LEN=128`; directed tests currently cover short rows.
3. Add softmax/RMSNorm primitive fixture coverage. Status: implemented in
   `hw/npu_core/tb/primitive_engines_tb.sv`.
4. Expose measured reduction cycles only after integrated execution exists.
   Status: pending.
