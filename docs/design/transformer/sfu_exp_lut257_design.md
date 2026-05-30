# SFU EXP 257-entry LUT Design

## Scope

This document defines the target v1 replacement for the current SFU EXP
bring-up model. It only covers `SFU_EXP`. `SFU_RECIP` and `SFU_RSQRT` remain
bring-up division/isqrt models until their own LUT/Newton design is reviewed.

No RTL implementation should start from this document until review is complete.

## Current status

Current RTL in `hw/npu_core/rtl/sfu/sfu_lut.sv` implements a 9-segment coarse
Q0.15 EXP approximation. Its segment constants are sourced from
`arch/configs/npu_transformer_v1.jsonc` as `bringup_exp_q15_segments` and
passed through `transformer_primitive_engines.sv`.

That implementation is a bring-up model. It is not the final numerical policy.

## Target parameters and source of truth

Source of truth: `arch/configs/npu_transformer_v1.jsonc`.

| Parameter | Config field | Target value |
| --- | --- | --- |
| Input data width | `modules.sfu.data_width` | 32 |
| EXP input scale | `modules.sfu.exp_input_scale` | 32 |
| EXP LUT entries | `modules.sfu.exp_lut_entries` | 257 |
| EXP output Q | `modules.sfu.exp_output_q` | 15 |
| `SFU_EXP` encoding | `primitive_op_encodings.sfu.SFU_EXP` | 0 |

The 257-entry table should be generated, not handwritten in RTL. The generator
may emit either an SV include/table package or constants inside the generated
Transformer config package. The table generator must record the source config
path and generation formula in the emitted file header.

## Input/output dtype and Q format

Input:

- Signed integer `x`.
- Real interpretation: `x_real = x / exp_input_scale`.
- Legal target table domain: integer `x` clamped to `[-256, 0]`, representing
  real values `[-8.0, 0.0]` when `exp_input_scale = 32`.

Output:

- Unsigned Q0.15 in the low 16 bits of the `DATA_WIDTH` output port.
- `0.0` maps to `0`.
- Values near `1.0` saturate to `(1 << exp_output_q) - 1 = 32767`.

## Operation semantics

Target EXP computes:

```text
x_clamped = clamp(x, -8 * exp_input_scale, 0)
index = x_clamped + 8 * exp_input_scale
y_real = exp(x_clamped / exp_input_scale)
y_q15 = saturate_u16(round(y_real * ((1 << exp_output_q) - 1)))
```

With `exp_input_scale = 32`, `index` spans `0..256`.

Table ordering:

- `table[0]` corresponds to `x = -256`, real `-8.0`.
- `table[256]` corresponds to `x = 0`, real `0.0`.

Lookup:

```text
y_q15 = table[index]
```

No interpolation is part of the v1 target. Interpolation may be considered in a
later revision only after PPA/accuracy evidence.

## Rounding and saturation

Table generation uses round-to-nearest ties-to-away-from-zero unless a later
numerical spec changes it before RTL implementation.

Generation pseudocode:

```text
scale = (1 << exp_output_q) - 1
raw = exp(x / exp_input_scale) * scale
rounded = floor(raw + 0.5)
y = min(max(rounded, 0), scale)
```

The runtime lookup itself has no rounding. Runtime clamp occurs before index
calculation.

## Latency model

Target v1 latency is one accepted operation to one `done` pulse for ROM-style
lookup. If synthesis or timing later requires a registered ROM read, the
latency must be updated in this document and in the valid/ready contract before
RTL changes are accepted.

## active/stall/done semantics

For the current start/done bring-up interface:

- `start` sampled high accepts one operation.
- `active` is high for the accepted operation cycle.
- `done` pulses when `y` is valid.
- No stall is represented in the current interface.

For the future valid/ready interface, semantics are deferred to
`primitive_valid_ready_v1.md`.

## PPA counters

The EXP LUT contributes to future `sfu_active_cycles`. There is no dedicated
EXP counter in v1. A future stall counter should count cycles where SFU input is
valid but the SFU cannot accept the operation.

## Golden model alignment

`sw/tools/transformer/micro_golden.py` should keep three visible paths:

- `softmax_reference_*`: floating-point algorithm reference.
- `softmax_fixed_spec_*`: 257-entry fixed-point EXP target.
- `softmax_rtl_model_*`: current RTL model while RTL still uses the 9-segment
  bring-up LUT.

When RTL moves to the 257-entry LUT, `softmax_rtl_model_*` should be updated to
match the implemented lookup, while old 9-segment behavior remains covered only
if a legacy compatibility test is still useful.

## Verification plan

Required tests before implementation is considered complete:

- Generator test: table has exactly 257 entries.
- Endpoint test: `x = 0` returns `32767`.
- Clamp test: values greater than `0` use index `256`; values less than `-256`
  use index `0`.
- Monotonicity test: table values are nondecreasing with index.
- RTL directed test: selected indices `0`, `1`, `32`, `64`, `128`, `192`,
  `255`, `256` match the fixed spec model.
- Softmax primitive sequence test updated to the 257-entry RTL model.

## Migration plan

1. Add table generation and fixed-spec model tests.
2. Keep current RTL unchanged until table generation is reviewed.
3. Replace 9-segment lookup in `sfu_lut.sv` with 257-entry lookup.
4. Update SFU README from "9-segment bring-up" to "257-entry implemented".
5. Keep RECIP/RSQRT documented as bring-up until their own design is reviewed.

## Known risks

- A 257-entry inline case statement is simple but may be noisy; an initialized
  ROM-style array may be cleaner but needs simulator/synthesis compatibility
  checks.
- Exact rounding mode must stay identical across generator, Python golden, and
  RTL test expectations.
- The valid/ready latency decision can affect how the LUT is registered.
