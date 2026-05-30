# Requant v1

## Scope

Requantization contract for Transformer primitive vector bring-up. This document
defines the current v1 mode and the planned v2 target mode.

## Parameters and source-of-truth config fields

Source of truth for vector width and lanes is `arch/configs/npu_transformer_v1.jsonc`:

| Parameter | Config field |
| --- | --- |
| `LANES` | `modules.vector_engine.lanes` |
| `DATA_WIDTH` | `modules.vector_engine.data_width` |
| `OP_VEC_REQUANT` | `primitive_op_encodings.vector.VEC_REQUANT` |

Per-operation shift and clamp bounds are uop/runtime fields, not fixed global
config constants.

## Input/output dtype and Q format

Current RTL consumes signed `DATA_WIDTH` inputs and emits signed `DATA_WIDTH`
outputs. The current implementation does not encode a full scale Q format; it
performs arithmetic shift and clamp only.

## Operation semantics

Current v1 mode is `shift_clamp`:

```text
y = clamp(a >>> shift, clamp_low, clamp_high)
```

This is bring-up mode and is not final full requant semantics.

The v2 target mode is `mul_round_shift_clamp`:

```text
wide = a * multiplier
rounded = round(wide, shift)
shifted = rounded >>> shift
biased = shifted + optional_zero_point
y = clamp(biased, clamp_low, clamp_high)
```

## Latency model

Current v1 `shift_clamp` is single-cycle in `vector_engine.sv`. v2 multiply and
rounding may require a different latency model.

## active/stall/done semantics

Current semantics follow `vector_engine_v1.md`: `active` mirrors `start`, `done`
pulses for one cycle, and no valid/ready or stall counter exists yet.

## Rounding/saturation behavior

Current RTL uses arithmetic right shift with truncation toward negative
infinity for signed values, then clamps to explicit bounds. It does not perform
round-to-nearest, zero-point addition, or full output dtype saturation unless
those limits are passed as clamp bounds.

## PPA counters

Current `shift_clamp` is counted only through future `vector_active_cycles`.
There is no dedicated requant counter in standalone RTL.

## Current RTL status

`VEC_REQUANT` in `hw/npu_core/rtl/vector/vector_engine.sv` implements only
`shift_clamp`. Tests cover this behavior directly.

## Known gaps

- No full requant multiply.
- No rounding mode selection.
- No zero-point path.
- No final output dtype policy.
- No valid/ready pipeline or real stall counters.
