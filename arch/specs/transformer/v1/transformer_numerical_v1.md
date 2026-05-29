# Transformer Numerical v1 Spec

## Target

Numerical v1 defines deterministic fixed-point behavior for Transformer
micro-kernels. The goal is reproducible verification and PPA comparison, not
full model accuracy.

Baseline data types:

| Path | Type |
| --- | --- |
| activation | int8 |
| weight | int8 |
| matrix accumulator | int32 |
| KV cache | int8 |
| softmax output v1 | uint16 |
| RMSNorm intermediate | int32 / fixed-point reciprocal square root |

## Softmax Row v1

Stable softmax is implemented as a primitive-uop micro-kernel:

```text
max_x = REDUCE_MAX(x)
shifted = VEC_SUB(x, max_x)
clamped = VEC_CLAMP(shifted, -8 * input_scale, 0)
exp_q = SFU_EXP(clamped)
sum_q = REDUCE_SUM(exp_q)
recip_q = SFU_RECIP(sum_q)
y = VEC_MUL(exp_q, recip_q)
```

Parameters:

| Field | Value |
| --- | --- |
| input scale | 32 |
| clamp range | `[-8, 0]` in real units |
| clamp integer range | `[-256, 0]` |
| EXP LUT entries | 257 |
| EXP output | uint16 Q0.15 |
| sum accumulator | uint32 |
| reciprocal output | uint16 or uint24 |
| output | uint16 first |

The first RTL path may use LUT approximation. Reports and tests must identify
this as fixed-point approximate math, not floating-point softmax.

## RMSNorm Row v1

RMSNorm is implemented as a primitive-uop micro-kernel:

```text
sumsq = REDUCE_SUMSQ(x)
mean = sumsq / hidden_size
r = SFU_RSQRT(mean + eps)
y_i = x_i * r * weight_i
```

Initial constraints:

| Field | Value |
| --- | --- |
| hidden sizes | 64 and 128 |
| reduction max length | 128 |
| epsilon | fixed compile-time constant in golden/test metadata |
| output | fixed-point/int16 intermediate, later requantized |

The v1 acceptance path is Python golden plus primitive RTL/unit coverage. Full
Transformer block execution is a later milestone.
