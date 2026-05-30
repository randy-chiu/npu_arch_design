# SFU v1

## Scope

Primitive standalone special-function unit for Transformer bring-up. Covered ops
are `SFU_EXP`, `SFU_RECIP`, and `SFU_RSQRT`.

## Parameters and source-of-truth config fields

Source of truth: `arch/configs/npu_transformer_v1.jsonc`.

| Parameter | Config field |
| --- | --- |
| `DATA_WIDTH` | `modules.sfu.data_width` |
| EXP input scale | `modules.sfu.exp_input_scale` |
| EXP LUT entries | `modules.sfu.exp_lut_entries` |
| EXP output Q | `modules.sfu.exp_output_q` |
| Bring-up EXP segments | `modules.sfu.bringup_exp_q15_segments` |
| RECIP output Q | `modules.sfu.recip_output_q` |
| RSQRT output Q | `modules.sfu.rsqrt_output_q` |
| `OP_SFU_*` | `primitive_op_encodings.sfu.*` |

## Input/output dtype and Q format

Current RTL input and output ports are `DATA_WIDTH` bits. EXP takes signed
integer input scaled by `exp_input_scale = 32` and returns Q0.15 in the low
16 bits. RECIP and RSQRT return unsigned Q24-style approximations.

## Operation semantics

Target v1 EXP semantics are a 257-entry Q0.15 LUT over clamped integer input
`[-256, 0]`. Current RTL does not implement that target table. Current EXP is a
9-segment coarse LUT matched by `softmax_rtl_model_row_q15` in
`sw/tools/transformer/micro_golden.py`.

Current RECIP computes `(1 << 24) / x` with integer division and returns zero
for zero input. Current RSQRT computes `isqrt(x)` and then `(1 << 24) / root`.
These are bring-up models, not production SFU implementations.

## Latency model

Current RTL is single-cycle start-to-done. Final LUT/Newton paths may have
different latency and must update this spec before integration.

## active/stall/done semantics

`active` mirrors `start`. `done` pulses for one cycle when `start` is sampled.
There is no valid/ready pipeline and no real stall counter.

## Rounding/saturation behavior

EXP clamps input to `[-256, 0]` and selects a coarse segment. RECIP/RSQRT use
integer truncating division. Zero input returns zero for reciprocal-like ops.

## PPA counters

Required v1 reporting includes `sfu_active_cycles` and
`stall_cycles_by_engine`. Current standalone RTL exposes only `active`; real
counter integration is deferred.

## Current RTL status

Implemented as `hw/npu_core/rtl/sfu/sfu_lut.sv`. The design-side integration
point is `hw/npu_core/rtl/transformer_primitive_engines.sv`, which imports
generated config and passes SFU parameters/op encodings explicitly. The file
name is provisional: current EXP is a 9-segment coarse LUT, not the 257-entry
target LUT.

## Known gaps

- Primitive standalone bring-up RTL only.
- No valid/ready pipeline.
- No real stall counters.
- No full softmax pipeline.
- EXP target 257-entry Q0.15 LUT is not implemented.
- Production reciprocal/rsqrt LUT/Newton implementation is deferred.
