# SFU RTL

Planned Transformer NPU v1 location for fixed-point approximate special
functions:

- `SFU_EXP`
- `SFU_RECIP`
- `SFU_RSQRT`

The first implementation should use deterministic LUT-style approximations
matched against `sw/tools/transformer/micro_golden.py`.

Current module:

| File | Scope |
| --- | --- |
| `sfu_lut.sv` | Standalone v1 SFU for EXP/RECIP/RSQRT approximations |
