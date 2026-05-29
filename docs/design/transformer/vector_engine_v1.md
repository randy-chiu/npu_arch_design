# Vector Engine v1 Design

## 1. Target / 目标

Vector Engine v1 provides primitive lane-wise integer operations used by
Transformer micro-kernels. It is not a fused softmax or RMSNorm unit.

Initial use cases:

- stable softmax row: `VEC_SUB`, `VEC_CLAMP`, `VEC_MUL`;
- RMSNorm row: lane-wise multiply/scale after `REDUCE_SUMSQ` and `SFU_RSQRT`;
- future requantization after matrix or vector intermediate results.

## 2. Overall Design / 整体设计思路

The v1 vector engine is an in-order primitive execution block behind the uop
scheduler:

```text
uop scheduler
  -> vector operand buffer
  -> VECTOR_LANES=8 lane datapath
  -> vector result buffer
```

The first RTL module should be standalone under:

```text
hw/npu_core/rtl/vector/
```

It should be tested independently before connecting to `npu_v0_top`.

## 3. Key Details / 重点细节

Parameters:

| Field | v1 value |
| --- | --- |
| `VECTOR_LANES` | 8 |
| input types | int16 and int32 paths |
| output types | int16/int32 internal, later int8/uint16 through requant |
| issue model | one primitive op per issued uop |
| edge handling | valid-lane mask, default all lanes valid |

Primitive ops:

| Op | Semantics |
| --- | --- |
| `VEC_ADD` | `y[i] = a[i] + b[i]` |
| `VEC_SUB` | `y[i] = a[i] - b[i]` |
| `VEC_MUL` | `y[i] = a[i] * b[i]` |
| `VEC_SCALE` | `y[i] = (a[i] * scale) >> shift` |
| `VEC_REQUANT` | clamp/round/shift wider input to target integer format |
| `VEC_CLAMP` | `y[i] = min(max(x[i], low), high)` |

Counter semantics:

| Counter | Meaning |
| --- | --- |
| `vector_active_cycles` | vector op accepted and produces progress |
| `vector_stall_cycles` | vector op assigned but input/output path unavailable |
| `vector_idle_cycles` | no vector op assigned |

V1 may implement one-cycle combinational lane ops for ADD/SUB/CLAMP and a
registered one-cycle MUL/SCALE path. The report must use measured active cycles
once the module is connected; standalone tests can check only functionality.

## 4. Verification / 验证测试

Initial tests:

- directed ADD/SUB/MUL with signed values;
- SCALE with positive and negative inputs;
- CLAMP for softmax range `[-256, 0]`;
- REQUANT saturation behavior.

Golden source:

```text
sw/tools/transformer/micro_golden.py
```

Required gates after integration:

```text
make npu-core-sim
make test
```

## 5. Implementation Priority / 实现优先级

1. Add standalone vector RTL and testbench. Status: implemented as
   `hw/npu_core/rtl/vector/vector_engine.sv`.
2. Match Python golden for softmax clamp and RMSNorm scale cases. Status:
   directed RTL cases and standalone softmax/RMSNorm primitive sequences pass
   through `make primitive-engines-sim`.
3. Add uop decode constants only after standalone module passes. Status:
   pending.
4. Connect perf counters after the scheduler path is reviewed. Status:
   pending.
