# Vector Engine RTL

Design: `docs/design/transformer/vector_engine_v1.md`.

## Current status

`vector_engine.sv` is primitive standalone bring-up RTL for `VEC_ADD`,
`VEC_SUB`, `VEC_MUL`, `VEC_SCALE`, `VEC_REQUANT`, and `VEC_CLAMP`.
`VEC_REQUANT` currently implements `shift_clamp` only.

## Config hookup status

Integrated paths must pass generated config parameters and op encodings
explicitly. The design-side integration point is
`hw/npu_core/rtl/transformer_primitive_engines.sv`; it imports
`build/generated/npu_transformer_v1_config_pkg.sv` and instantiates:

```systemverilog
vector_engine #(
    .LANES(CFG_VECTOR_LANES),
    .DATA_WIDTH(CFG_VECTOR_DATA_WIDTH),
    .OP_VEC_REQUANT(CFG_VEC_REQUANT)
)
```

## Known gaps

- No valid/ready pipeline yet.
- No real stall counters yet.
- No full requant yet; final mode is planned as multiply + rounding + shift +
  optional zero-point + clamp.
- No scheduler integration.

## Tests

`make primitive-engines-sim` covers configured instantiation, parameter
override instantiation, arithmetic ops, clamp, and current `VEC_REQUANT`
`shift_clamp` behavior.
