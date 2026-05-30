# Reduction Engine RTL

Design: `docs/design/transformer/reduction_engine_v1.md`.

## Current status

`reduction_engine.sv` is primitive standalone bring-up RTL for `REDUCE_MAX`,
`REDUCE_SUM`, and `REDUCE_SUMSQ`.

## Config hookup status

Integrated paths must pass generated config parameters and op encodings
explicitly. The design-side integration point is
`hw/npu_core/rtl/transformer_primitive_engines.sv`; it imports
`build/generated/npu_transformer_v1_config_pkg.sv` and instantiates:

```systemverilog
reduction_engine #(
    .MAX_LEN(CFG_REDUCTION_MAX_LEN),
    .DATA_WIDTH(CFG_REDUCTION_DATA_WIDTH),
    .RESULT_WIDTH(CFG_REDUCTION_RESULT_WIDTH),
    .OP_REDUCE_SUM(CFG_REDUCE_SUM)
)
```

## Known gaps

- No valid/ready pipeline yet.
- No real stall counters yet.
- Single-cycle bring-up model, not a production reduction tree.
- No production overflow/saturation policy.

## Tests

`make primitive-engines-sim` covers `REDUCE_MAX`, `REDUCE_SUM`, and
`REDUCE_SUMSQ` on directed signed inputs.
