# SFU RTL

Design: `docs/design/transformer/sfu_v1.md`.

## Current status

`sfu_lut.sv` is primitive standalone bring-up RTL for `SFU_EXP`, `SFU_RECIP`,
and `SFU_RSQRT`.

## Config hookup status

Integrated paths must pass generated config parameters and op encodings
explicitly. The design-side integration point is
`hw/npu_core/rtl/transformer_primitive_engines.sv`; it imports
`build/generated/npu_transformer_v1_config_pkg.sv` and instantiates:

```systemverilog
sfu_lut #(
    .DATA_WIDTH(CFG_SFU_DATA_WIDTH),
    .EXP_LUT_ENTRIES(CFG_SFU_EXP_LUT_ENTRIES),
    .BRINGUP_EXP_SEG_0(CFG_SFU_BRINGUP_EXP_SEG_0),
    .RECIP_OUTPUT_Q(CFG_SFU_RECIP_OUTPUT_Q),
    .OP_SFU_EXP(CFG_SFU_EXP)
)
```

## Known gaps

- Current EXP is a 9-segment coarse LUT.
- Spec target is a 257-entry Q0.15 LUT.
- RECIP/RSQRT currently use division/isqrt for bring-up.
- Final LUT/Newton reciprocal and rsqrt implementation is deferred.
- No valid/ready pipeline yet.
- No real stall counters yet.
- No full softmax pipeline yet.

## Tests

`make primitive-engines-sim` covers current EXP segment values, RECIP, RSQRT,
and directed softmax/RMSNorm primitive sequences matched to
`sw/tools/transformer/micro_golden.py` RTL-model functions.
