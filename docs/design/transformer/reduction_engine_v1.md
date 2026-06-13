# Reduction Engine v1

## Scope

Primitive standalone row/vector reductions for Transformer bring-up. Covered ops
are `REDUCE_MAX`, `REDUCE_SUM`, and `REDUCE_SUMSQ`.

For attention v1, reduction is primarily required by row softmax:

```text
P[i,j] = exp(x[i,j] - max_j x[i,j]) / sum_j exp(x[i,j] - max_j x[i,j])
```

This formula requires a row maximum over masked scores and a row sum over EXP
outputs. RMSNorm continues to use `REDUCE_SUMSQ`, but RMSNorm is separate from
the attention softmax acceptance path.

## Attention-derived requirements

### Row max

For each attention score row:

```text
row_max = max_j masked_score[j]
```

The reduction engine must define:

- row length;
- valid lane behavior;
- masked element behavior;
- result dtype;
- behavior when no lane is valid.

Current RTL computes reductions over lanes satisfying both `[0, length)` and
the explicit valid-lane mask. Causal invalid positions are therefore excluded
instead of becoming zero-valued participants.

### Valid-lane reduction contract

Implementation status: valid-lane gating is implemented for the single-tile
Attention path. Hardware error/status reporting for all-invalid rows remains
pending. The end-to-end decision and alternatives are owned by
`attention_sequence_v1.md`.

Each row reduction command receives:

```text
length
valid_lane_mask
```

An element participates only when:

```text
lane < length and valid_lane_mask[lane] == 1
```

| Operation | Invalid-lane behavior |
| --- | --- |
| `REDUCE_MAX` | exclude lane; do not substitute zero |
| `REDUCE_SUM` | exclude lane; equivalent arithmetic contribution is zero |
| `REDUCE_SUMSQ` | exclude lane when mask is enabled |

If no lane participates, the engine returns an architectural error rather than
a numerical result. Compiler/runtime must reject such rows before issue, but
hardware retains the check so malformed commands cannot silently produce a
probability distribution.

The mask is an input qualifier, not a new reduction operation. PPA counts only
participating elements as `reduction_element_ops` and separately reports
mask-control cost.

Expected cost is lane-valid gating, all-invalid detection, and mask transport.
The benefit is correct max/sum behavior and reuse of the same Reduction
primitive for causal, padding, and tile-tail rows.

### Row sum

After SFU EXP:

```text
row_sum = sum_j exp_q15[j]
```

For initial attention:

```text
0 <= exp_q15[j] <= 32767
row_sum width >= ceil(log2(row_len * 32767))
```

Current `RESULT_WIDTH=64` is sufficient for the v1 model envelope. The spec must
still state the expected input Q format because row sum feeds SFU RECIP.

### Segmented rows

For `S > reduction lanes`, softmax rows may be processed in segments:

```text
row_max = max(segment_max_0, segment_max_1, ...)
row_sum = sum(segment_sum_0, segment_sum_1, ...)
```

The current standalone RTL accepts a packed vector up to `MAX_LEN`, but the
future scheduler/runtime must define whether rows are issued whole or segmented.
PPA counters must count useful reduced elements, not only command count.

## Parameters and source-of-truth config fields

Source of truth: `arch/configs/npu_transformer_v1.jsonc`.

| Parameter | Config field |
| --- | --- |
| `MAX_LEN` | `modules.reduction_engine.max_len` |
| `DATA_WIDTH` | `modules.reduction_engine.data_width` |
| `RESULT_WIDTH` | `modules.reduction_engine.result_width` |
| v1 logical lanes | `modules.reduction_engine.lanes` |
| `OP_REDUCE_*` | `primitive_op_encodings.reduction.*` |

## Input/output dtype and Q format

Inputs are signed integer elements of `DATA_WIDTH` bits packed into `x_flat`.
`result` is signed `RESULT_WIDTH`. Current v1 bring-up uses int32 inputs and
int64 result accumulation.

## Operation semantics

The engine consumes elements `[0, length)`, capped structurally by `MAX_LEN`.
`REDUCE_MAX` returns the signed maximum, `REDUCE_SUM` returns signed sum, and
`REDUCE_SUMSQ` returns the sum of signed element squares.

## Latency model

Current RTL computes combinationally inside one clocked start transaction and
pulses `done` in the cycle after `start` is sampled. This is not the final
multi-cycle reduction tree or streaming latency model.

## active/stall/done semantics

`active` mirrors `start`. `done` pulses for one cycle when `start` is sampled.
There is no valid/ready pipeline and no real stall counter.

Before scheduler integration, replace this bring-up contract with
`primitive_valid_ready_v1.md` semantics:

- `cmd_valid/cmd_ready` accepts one reduction command and stable payload;
- `rsp_valid/rsp_ready` transfers the reduction result;
- `reduction_active_cycles`, `reduction_input_stall_cycles`, and
  `reduction_output_stall_cycles` are counted from handshake events;
- `reduction_element_ops` counts valid reduced elements, not just command
  count, and must define segmented-row accounting before PPA use.

The start/done path can remain as a compatibility shim for primitive bring-up
tests, but it is not the production scheduler contract.

## Rounding/saturation behavior

`REDUCE_MAX` sign-extends the selected input. `REDUCE_SUM` accumulates in
`RESULT_WIDTH`. `REDUCE_SUMSQ` accumulates integer products in `RESULT_WIDTH`.
Current RTL does not saturate on accumulator overflow.

## PPA counters

Required v1 reporting includes `reduction_active_cycles` and
`stall_cycles_by_engine`. The compatibility shim exposes local active, input
stall, output stall, idle, accepted-op, and accepted-element-op counters. CSR
integration is deferred.

Counter exposure order:

1. review and accept row-length, mask, empty-row, and segmented-row behavior;
2. implement handshake-visible local event sources;
3. verify event and element counts in primitive directed tests;
4. aggregate through scheduler/wrapper and expose CSR/report fields.

## Current RTL status

Implemented as `hw/npu_core/rtl/reduction/reduction_engine.sv`. The design-side
integration point is `hw/npu_core/rtl/transformer_primitive_engines.sv`, which
imports generated config and passes reduction parameters/op encodings
explicitly.

`hw/npu_core/rtl/primitive_handshake_shims.sv` also provides the standalone
`reduction_engine_handshake` compatibility boundary. It holds commands and
responses according to `primitive_valid_ready_v1.md`; the current SoC path
still uses start/done.

## Known gaps

- Primitive standalone bring-up RTL only.
- No valid/ready pipeline.
- No real stall counters.
- No balanced tree or streaming implementation.
- No production overflow policy.
- No explicit attention mask semantics.
- Valid-lane semantics are documented and accepted as the architecture
  direction but not implemented.
- No segmented-row scheduler contract.
- No measured `reduction_element_ops` counter.
