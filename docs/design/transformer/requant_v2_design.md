# Requant v2 Design

## Scope

This document defines the proposed full requantization target for
`VEC_REQUANT`. It extends the current v1 `shift_clamp` bring-up behavior with a
multiply, rounding, shift, optional zero-point, and clamp path.

No RTL implementation should start from this document until review is complete.

## Current status

Current `VEC_REQUANT` in `vector_engine.sv` implements only:

```text
y = clamp(a >>> shift, clamp_low, clamp_high)
```

This mode is named `shift_clamp`. It remains valid as a bring-up/compatibility
mode and must keep regression coverage.

## Target mode

The v2 target mode is named `mul_round_shift_clamp`:

```text
wide = a * multiplier
rounded = round(wide, rounding_mode, shift)
shifted = rounded >>> shift
biased = shifted + zero_point
y = clamp(biased, clamp_low, clamp_high)
```

`zero_point` may be disabled by mode/config/uop field. If disabled, it is
treated as zero.

## Parameters and source of truth

Static parameters belong in `arch/configs/npu_transformer_v1.jsonc` before RTL
implementation. Runtime fields belong in the uop/descriptor encoding before
scheduler integration.

Proposed static config fields:

| Parameter | Proposed config field |
| --- | --- |
| Input data width | `modules.vector_engine.data_width` |
| Multiplier width | `modules.vector_engine.requant.multiplier_width` |
| Wide product width | `modules.vector_engine.requant.product_width` |
| Supported rounding modes | `modules.vector_engine.requant.rounding_modes` |
| Zero-point support | `modules.vector_engine.requant.zero_point_supported` |
| Requant modes | `modules.vector_engine.requant.modes` |

Proposed runtime/uop fields:

| Field | Meaning |
| --- | --- |
| `requant_mode` | `shift_clamp` or `mul_round_shift_clamp` |
| `multiplier` | signed or unsigned fixed-point multiplier |
| `shift` | right shift amount |
| `rounding_mode` | selected rounding policy |
| `zero_point` | optional output offset |
| `clamp_low` | minimum output value |
| `clamp_high` | maximum output value |

The signedness of `multiplier` must be fixed in the reviewed config/spec before
RTL implementation. Initial recommendation: signed multiplier, signed input,
signed wide product.

## Input/output dtype and Q format

Input:

- Signed integer `a` with `DATA_WIDTH` bits.

Multiplier:

- Signed fixed-point multiplier.
- Exact Q format is owned by the producing compiler/kernel metadata. RTL treats
  it as an integer multiplier plus shift.

Output:

- Signed integer with `DATA_WIDTH` bits for the current vector engine port.
- Final dtype saturation is represented by `clamp_low` and `clamp_high`.

## Rounding modes

Initial implementation should support one mode only unless there is a clear
workload need for more. Recommended first mode:

```text
round_nearest_away_from_zero
```

For signed `wide` and positive `shift`:

```text
offset = 1 << (shift - 1)
if wide >= 0:
    rounded = wide + offset
else:
    rounded = wide - offset
```

Then:

```text
shifted = rounded >>> shift
```

For `shift = 0`, no rounding offset is applied.

Alternative rounding modes, if later needed, must be added to this document and
golden tests before RTL.

## Saturation and clamp behavior

Clamp is explicit:

```text
if biased < clamp_low: y = clamp_low
else if biased > clamp_high: y = clamp_high
else y = biased
```

Clamp occurs after zero-point addition. Overflow behavior before clamp must be
defined by product/intermediate width. Initial recommendation: product and
biased intermediates are wide enough for all supported runtime fields in the
target config; no silent wrap should be used as numerical policy.

## Latency model

`shift_clamp` remains single-cycle in the current bring-up RTL.

`mul_round_shift_clamp` target latency is one accepted operation to one response
if timing allows a single-cycle multiply path. If the multiplier must be
registered, the vector engine latency must be updated before implementation.

The valid/ready interface in `primitive_valid_ready_v1.md` should be used for
any multi-cycle implementation.

## active/stall/done semantics

Until valid/ready is implemented, `VEC_REQUANT` follows `vector_engine_v1.md`:

- `start` accepts one operation.
- `done` pulses when output is valid.
- `active` mirrors accepted work.
- no real stall counter exists.

After valid/ready, it follows `primitive_valid_ready_v1.md`.

## Golden model alignment

`sw/tools/transformer/micro_golden.py` should expose explicit names:

- `requant_shift_clamp_rtl_model_*` for current behavior.
- `requant_mul_round_shift_clamp_fixed_spec_*` for target behavior.

Softmax/RMSNorm fixed-spec models should not silently switch to v2 requant until
RTL-like golden and RTL tests agree.

## Verification plan

Required tests:

- Existing `shift_clamp` regressions remain unchanged.
- Positive value with multiplier and shift rounds correctly.
- Negative value rounds according to the reviewed signed rounding rule.
- `shift = 0` path applies no rounding offset.
- Zero-point enabled path adds offset before clamp.
- Zero-point disabled path matches zero offset.
- Clamp low and clamp high both saturate correctly.
- Extreme multiplier/input cases do not silently wrap outside the reviewed
  intermediate width policy.

## Known risks

- Rounding for negative signed values is easy to mismatch between Python and
  RTL.
- Product/intermediate width must be reviewed before coding to avoid accidental
  wrap semantics.
- The uop encoding may need more fields than the current primitive testbench
  exposes.
