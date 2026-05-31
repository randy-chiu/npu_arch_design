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
`shift_clamp`; see `requant_v1.md`.

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

## Latency model

Current RTL is single-cycle start-to-done in the cycle after the sampled
`start` edge. This is a bring-up model, not a timing target for production.

## active/stall/done semantics

`active` mirrors `start`. `done` pulses for one cycle when `start` is sampled.
There is no valid/ready interface, no back-pressure, and no real stall counter.

## Rounding/saturation behavior

`VEC_SCALE` and current `VEC_REQUANT` use arithmetic right shift with truncation.
`VEC_CLAMP` and `VEC_REQUANT` saturate only to explicit clamp bounds. Add/sub/mul
do not provide full overflow saturation.

## PPA counters

Required v1 reporting includes `vector_active_cycles` and
`stall_cycles_by_engine`. Current standalone RTL exposes only `active`; counter
integration is deferred to the wrapper/scheduler path.

## Current RTL status

Implemented as `hw/npu_core/rtl/vector/vector_engine.sv`. The design-side
integration point is `hw/npu_core/rtl/transformer_primitive_engines.sv`, which
imports generated config and passes vector parameters/op encodings explicitly.
Integrated tests instantiate that wrapper and include a smaller direct override
instance to prove parameters are not relying on defaults.

## Known gaps

- Primitive standalone bring-up RTL only.
- No valid/ready pipeline.
- No real stall counters.
- No full requant policy.
- No scheduler-owned issue/retire protocol.
- No attention mask-select semantics.
- Current `VEC_MUL` is not a reviewed softmax normalization multiply.
- Probability-to-int8 policy for PV is not defined.
