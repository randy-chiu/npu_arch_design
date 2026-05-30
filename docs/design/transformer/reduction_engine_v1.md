# Reduction Engine v1

## Scope

Primitive standalone row/vector reductions for Transformer bring-up. Covered ops
are `REDUCE_MAX`, `REDUCE_SUM`, and `REDUCE_SUMSQ`.

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
