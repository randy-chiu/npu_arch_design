# Vector Engine v1

## Scope

Primitive standalone vector RTL for Transformer bring-up. Covered ops are
`VEC_ADD`, `VEC_SUB`, `VEC_MUL`, `VEC_SCALE`, `VEC_REQUANT`, and `VEC_CLAMP`.
This is not a full vector pipeline or scheduler integration contract.

For attention v1, the vector engine is the glue between matrix scores,
row-softmax reductions/SFU, and the probability-value path. Attention does not
need a dedicated RTL macro, but it does require the vector primitive contract to
be precise enough for compiler-scheduled attention sequences.

## Attention-derived requirements

Attention uses:

```text
scores = Q * K^T
scaled_scores = scores / sqrt(D)
P = softmax(masked_scores)
O = P * V
```

The vector engine is needed in the following steps.

### Score scale

Initial v1 represents `1 / sqrt(D)` as a power-of-two shift:

```text
scaled_score = score >>> score_shift
```

This can use current `VEC_REQUANT` when no multiplier is required. If accuracy
requires a non-power-of-two scale, attention needs `requant_v2`:

```text
scaled_score = round(score * multiplier / 2^shift)
```

### Mask and clamp

After row max subtraction:

```text
delta = masked_score - row_max
exp_input = clamp(delta, -256, 0)
```

Current `VEC_SUB` and `VEC_CLAMP` cover this for unmasked rows. Causal or
padding masks are not yet covered because inactive lanes currently produce
zero, while attention mask semantics require invalid positions to behave like
large negative scores before softmax.

Therefore attention needs one of:

- explicit mask-select op;
- compiler materializes masked scores before vector issue;
- command-list field that maps invalid lanes to `SOFTMAX_NEG_INF` before max.

### Probability normalization

Softmax normalization computes:

```text
P_q15 = requant(exp_q15 * recip_q24)
```

Current `VEC_MUL` keeps only low `DATA_WIDTH` bits and is not enough as a
reviewed Q0.15 normalization path. Attention therefore needs `VEC_REQUANT`
v2 or a named vector op that defines multiply width, rounding, shift, and
clamp.

### PV input policy

For:

```text
O = P * V
```

current matrix RTL can only consume int8 x int8. Attention must choose whether
vector logic requantizes `P_q15` to int8 before PV, or whether matrix/PV adds a
mixed Q0.15 x int8 path. This decision affects vector clamp ranges, golden
accuracy, and PPA area/energy.

## Parameters and source-of-truth config fields

Source of truth: `arch/configs/npu_transformer_v1.jsonc`.

| Parameter | Config field |
| --- | --- |
| `LANES` | `modules.vector_engine.lanes` |
| `DATA_WIDTH` | `modules.vector_engine.data_width` |
| `OP_VEC_*` | `primitive_op_encodings.vector.*` |

Generated integration constants are emitted in
`build/generated/npu_transformer_v1_config_pkg.sv`. RTL integration must pass
both shape parameters and op encodings explicitly; module defaults are only a
standalone fallback.

## Input/output dtype and Q format

Inputs and outputs are signed integer lanes of `DATA_WIDTH` bits. Current
bring-up tests use signed 32-bit lanes. `VEC_REQUANT` current mode is
`shift_clamp`; the target full mode is `mul_round_shift_clamp`.

## Operation semantics

For active lanes in `valid_mask`, operations are lane-wise:

| Op | Semantics |
| --- | --- |
| `VEC_ADD` | `a + b` |
| `VEC_SUB` | `a - b` |
| `VEC_MUL` | low `DATA_WIDTH` bits of `a * b` |
| `VEC_SCALE` | `(a * scalar) >>> shift` |
| `VEC_REQUANT` | `(a >>> shift)` clamped to `[clamp_low, clamp_high]` |
| `VEC_CLAMP` | `a` clamped to `[clamp_low, clamp_high]` |

Inactive lanes produce zero in current RTL.

### Requantization Modes

`VEC_REQUANT` is owned by this vector engine document. It does not have a
separate module-level design source.

Current mode:

```text
requant_mode = shift_clamp
y = clamp(a >>> shift, clamp_low, clamp_high)
```

This is the current RTL behavior. It uses arithmetic right shift with
truncation, then clamps to explicit bounds. It does not provide a multiplier,
round-to-nearest, zero-point, or final dtype saturation beyond the explicit
clamp limits.

Target mode:

```text
requant_mode = mul_round_shift_clamp
wide = a * multiplier
rounded = round(wide, rounding_mode, shift)
shifted = rounded >>> shift
biased = shifted + zero_point
y = clamp(biased, clamp_low, clamp_high)
```

`zero_point` may be disabled by mode/config/uop field. If disabled, it is
treated as zero.

Static config fields must be added before RTL implementation:

| Parameter | Proposed config field |
| --- | --- |
| Multiplier width | `modules.vector_engine.requant.multiplier_width` |
| Wide product width | `modules.vector_engine.requant.product_width` |
| Supported rounding modes | `modules.vector_engine.requant.rounding_modes` |
| Zero-point support | `modules.vector_engine.requant.zero_point_supported` |
| Requant modes | `modules.vector_engine.requant.modes` |

Runtime/uop fields:

| Field | Meaning |
| --- | --- |
| `requant_mode` | `shift_clamp` or `mul_round_shift_clamp` |
| `multiplier` | signed fixed-point multiplier |
| `shift` | right shift amount |
| `rounding_mode` | selected rounding policy |
| `zero_point` | optional output offset |
| `clamp_low` | minimum output value |
| `clamp_high` | maximum output value |

Initial v2 recommendation: signed multiplier, signed input, signed wide
product, and one rounding mode:

```text
round_nearest_away_from_zero
```

For signed `wide` and positive `shift`:

```text
offset = 1 << (shift - 1)
if wide >= 0:
    shifted = (wide + offset) >>> shift
else:
    shifted = -(((-wide) + offset) >>> shift)
```

For `shift = 0`, no rounding offset is applied.

The first executable consumer is unmasked `S=8,D_k=8` attention score scaling.
It uses a distinct `VEC_REQUANT_V2` operation so the existing shift-only
`VEC_REQUANT` behavior and regressions remain unchanged.

### Executable Attention Score Scale

#### Problem being solved

The compiler plan declares:

```text
QK -> scale_mask -> softmax -> PV
```

Previously, scale/mask was fixture-materialized. That meant the measured QK
output was not consumed by another measured NPU stage, scale cost was absent
from PPA, and the executable graph did not implement `score / sqrt(D_k)`.

The existing shift-only `VEC_REQUANT` also cannot accurately implement
`1/sqrt(8) = 0.353553...`, because this value is not a power-of-two shift.

#### Vector-engine design

For the first unmasked `D_k=8` path:

```text
scale_multiplier = round((1 / sqrt(8)) * 2^15) = 11585
scale_shift      = 15
rounding         = round_nearest_away_from_zero
mask_policy      = none
```

Each score uses:

```text
wide = score_raw * 11585
if wide >= 0:
    score_scaled = (wide + 2^14) >>> 15
else:
    score_scaled = -(((-wide) + 2^14) >>> 15)
```

The core issues one eight-lane `VEC_REQUANT_V2` operation per score row. One
`8x8` tile therefore requires eight vector operations. Causal, padding, and
tail mask-select behavior remains deferred.

#### Why this mechanism is effective

- it implements the reviewed non-power-of-two scale;
- it reuses the shared vector engine instead of adding attention-specific
  compute RTL;
- positive and negative values use a defined symmetric rounding rule;
- the original shift-only requant behavior remains unchanged;
- measured vector-stage cost can be compared against the theoretical minimum
  of eight row operations.

Descriptor routing, SRAM ownership, and QK-to-scale buffer chaining belong to
`attention_runtime_v1.md`, not to the vector-engine module contract.

Review is complete only when config fields, runtime/uop fields, golden model
names, rounding mode, intermediate widths, and valid/ready latency are accepted
together. Requant v2 is a numerical contract change, not just a datapath
change.

## Latency model

Current RTL is single-cycle start-to-done in the cycle after the sampled
`start` edge. This is a bring-up model, not a timing target for production.

The standalone `vector_engine_handshake` compatibility shim adds the accepted
valid/ready boundary with one command in flight and one held response slot.
Its `cmd_fire` to `rsp_valid` latency is two cycles and it does not overlap
commands. The underlying engine and current SoC path remain start/done.

## active/stall/done semantics

`active` mirrors `start`. `done` pulses for one cycle when `start` is sampled.
There is no valid/ready interface, no back-pressure, and no real stall counter.

Before scheduler integration, replace this bring-up contract with
`primitive_valid_ready_v1.md` semantics:

- `cmd_valid/cmd_ready` accepts one vector command and stable payload;
- `rsp_valid/rsp_ready` transfers the packed lane result;
- `vector_active_cycles`, `vector_input_stall_cycles`, and
  `vector_output_stall_cycles` are counted from handshake events;
- response latency and initiation interval are declared in this document before
  RTL changes.

The old start/done tests should remain as compatibility tests through a shim,
but production scheduler integration must not depend on start/done pulses.

## Rounding/saturation behavior

`VEC_SCALE` and current `VEC_REQUANT` use arithmetic right shift with truncation.
`VEC_CLAMP` and `VEC_REQUANT` saturate only to explicit clamp bounds. Add/sub/mul
do not provide full overflow saturation.

Target `mul_round_shift_clamp` clamps after zero-point addition:

```text
if biased < clamp_low: y = clamp_low
else if biased > clamp_high: y = clamp_high
else y = biased
```

Product and biased intermediates must be wide enough for all supported runtime
fields in the target config. Silent wrap must not become the numerical policy.

## PPA counters

Required v1 reporting includes `vector_active_cycles` and
`stall_cycles_by_engine`. The compatibility shim exposes local active, input
stall, output stall, idle, accepted-op, and accepted-lane-op counters. CSR
integration is deferred to the wrapper/scheduler path.

Counter exposure order:

1. implement handshake-visible local event sources;
2. verify event counts in primitive directed tests;
3. aggregate through scheduler/wrapper;
4. expose CSR/report fields only after the event source is stable.

`VEC_REQUANT` does not get a dedicated standalone counter in the first plan.
Its operations are counted through vector accepted-op/lane events once the
valid/ready scheduler path exists.

## Current RTL status

Implemented as `hw/npu_core/rtl/vector/vector_engine.sv`. The design-side
integration point is `hw/npu_core/rtl/transformer_primitive_engines.sv`, which
imports generated config and passes vector parameters/op encodings explicitly.
Integrated tests instantiate that wrapper and include a smaller direct override
instance to prove parameters are not relying on defaults.

Golden model names should remain explicit:

- `requant_shift_clamp_rtl_model_*` for current behavior.
- `requant_mul_round_shift_clamp_fixed_spec_*` for target behavior.

Softmax/RMSNorm fixed-spec models should not silently switch to target requant
until RTL-like golden and RTL tests agree. Attention scale/mask and softmax
normalization must record the selected requant mode in workload metadata. A
measured workload using `shift_clamp` cannot be labeled with the target
`mul_round_shift_clamp` numerical contract.

Target requant verification requires:

- existing `shift_clamp` regressions remain unchanged;
- positive and negative rounded values match the reviewed signed rounding rule;
- `shift = 0` applies no rounding offset;
- zero-point enabled and disabled paths are both tested;
- clamp low and clamp high both saturate correctly;
- extreme multiplier/input cases do not silently wrap outside the reviewed
  intermediate width policy;
- workload metadata test confirms numerical contract and requant mode agree
  across golden, firmware expected data, and PPA report fields.

## Known gaps

- Primitive standalone bring-up RTL only.
- No valid/ready pipeline.
- No real stall counters.
- Requant v2 is implemented for signed multiply, round-nearest-away-from-zero,
  shift, and explicit clamp; zero-point and runtime-selectable fields remain.
- No scheduler-owned issue/retire protocol.
- No attention mask-select semantics.
- Current `VEC_MUL` is not a reviewed softmax normalization multiply.
- Probability-to-int8 policy for PV is superseded for the first measured path
  by mixed `Q0.15 x int8` matrix mode, but final output requant policy remains
  undefined.
