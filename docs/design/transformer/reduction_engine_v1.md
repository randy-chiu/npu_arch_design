# Reduction Engine v1

## Scope

Primitive standalone row/vector reductions for Transformer bring-up. Covered ops
are `REDUCE_MAX`, `REDUCE_SUM`, and `REDUCE_SUMSQ`.

For attention v1, reduction is primarily required by row softmax:

```text
P[i,j] = exp(x[i,j] - max_j x[i,j]) / sum_j exp(x[i,j] - max_j x[i,j])
```

This formula requires a row maximum over masked scores and a row sum over EXP
outputs. RMSNorm continues to use `REDUCE_SUMSQ`, but RMSNorm is separate from
the attention softmax acceptance path.

## Attention-derived requirements

### Row max

For each attention score row:

```text
row_max = max_j masked_score[j]
```

The reduction engine must define:

- row length;
- valid lane behavior;
- masked element behavior;
- result dtype;
- behavior when no lane is valid.

Current RTL can compute `REDUCE_MAX` over `[0, length)` but has no explicit mask
input. For causal attention, invalid positions must not become zero-valued
participants in max. They must be excluded or replaced with a reviewed negative
sentinel before reduction.

### Row sum

After SFU EXP:

```text
row_sum = sum_j exp_q15[j]
```

For initial attention:

```text
0 <= exp_q15[j] <= 32767
row_sum width >= ceil(log2(row_len * 32767))
```

Current `RESULT_WIDTH=64` is sufficient for the v1 model envelope. The spec must
still state the expected input Q format because row sum feeds SFU RECIP.

### Segmented rows

For `S > reduction lanes`, softmax rows may be processed in segments:

```text
row_max = max(segment_max_0, segment_max_1, ...)
row_sum = sum(segment_sum_0, segment_sum_1, ...)
```

The current standalone RTL accepts a packed vector up to `MAX_LEN`, but the
future scheduler/runtime must define whether rows are issued whole or segmented.
PPA counters must count useful reduced elements, not only command count.

## Parameters and source-of-truth config fields

Source of truth: `arch/configs/npu_transformer_v1.jsonc`.

| Parameter | Config field |
| --- | --- |
| `MAX_LEN` | `modules.reduction_engine.max_len` |
| `DATA_WIDTH` | `modules.reduction_engine.data_width` |
| `RESULT_WIDTH` | `modules.reduction_engine.result_width` |
| v1 logical lanes | `modules.reduction_engine.lanes` |
| `OP_REDUCE_*` | `primitive_op_encodings.reduction.*` |

## Input/output dtype and Q format

Inputs are signed integer elements of `DATA_WIDTH` bits packed into `x_flat`.
`result` is signed `RESULT_WIDTH`. Current v1 bring-up uses int32 inputs and
int64 result accumulation.

## Operation semantics

The engine consumes elements `[0, length)`, capped structurally by `MAX_LEN`.
`REDUCE_MAX` returns the signed maximum, `REDUCE_SUM` returns signed sum, and
`REDUCE_SUMSQ` returns the sum of signed element squares.

## Latency model

Current RTL computes combinationally inside one clocked start transaction and
pulses `done` in the cycle after `start` is sampled. This is not the final
multi-cycle reduction tree or streaming latency model.

## active/stall/done semantics

`active` mirrors `start`. `done` pulses for one cycle when `start` is sampled.
There is no valid/ready pipeline and no real stall counter.

## Rounding/saturation behavior

`REDUCE_MAX` sign-extends the selected input. `REDUCE_SUM` accumulates in
`RESULT_WIDTH`. `REDUCE_SUMSQ` accumulates integer products in `RESULT_WIDTH`.
Current RTL does not saturate on accumulator overflow.

## PPA counters

Required v1 reporting includes `reduction_active_cycles` and
`stall_cycles_by_engine`. Current standalone RTL exposes only `active`; real
counter integration is deferred.

## Current RTL status

Implemented as `hw/npu_core/rtl/reduction/reduction_engine.sv`. The design-side
integration point is `hw/npu_core/rtl/transformer_primitive_engines.sv`, which
imports generated config and passes reduction parameters/op encodings
explicitly.

## Known gaps

- Primitive standalone bring-up RTL only.
- No valid/ready pipeline.
- No real stall counters.
- No balanced tree or streaming implementation.
- No production overflow policy.
- No explicit attention mask semantics.
- No segmented-row scheduler contract.
- No measured `reduction_element_ops` counter.
