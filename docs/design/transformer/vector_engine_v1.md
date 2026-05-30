# Vector Engine v1

## Scope

Primitive standalone vector RTL for Transformer bring-up. Covered ops are
`VEC_ADD`, `VEC_SUB`, `VEC_MUL`, `VEC_SCALE`, `VEC_REQUANT`, and `VEC_CLAMP`.
This is not a full vector pipeline or scheduler integration contract.

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
