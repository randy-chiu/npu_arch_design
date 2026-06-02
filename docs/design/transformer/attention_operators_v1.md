# Attention Operators v1

## Scope

This document defines the software-visible operators needed to execute
Transformer attention v1 on the existing NPU primitive RTL. It does not define a
new monolithic attention RTL block. The operator layer names stable operations,
their tensor contracts, numerical contracts, and the primitive engines they use.

Operator definitions live under:

```text
sw/npu_core/operators/
```

Attention v1 should add a Transformer-oriented operator metadata file instead
of hiding attention-specific behavior inside fixture generation.

## Layer Responsibility

The operator layer owns:

- operator names and versions;
- input/output tensor ranks, layouts, dtypes, and quantization formats;
- required primitive engines;
- legal shape/tile constraints;
- numerical contract IDs;
- metadata needed by compiler, runtime, golden, and PPA.

The operator layer does not own concrete tensor values, SRAM addresses, job IDs,
runtime buffer allocation, or RTL state-machine implementation details.

## Current Implemented Surface

| Operation | Current path | Status | Notes |
| --- | --- | --- | --- |
| `matmul_k_stream` | wrapper descriptor + matrix array | implemented | int8 x int8 -> int32, K chunks accumulate into one 8x8 tile |
| `attention_softmax_v1` | wrapper descriptor + vector/reduction/SFU sequence | implemented | one 8-lane row/tile, approximate exp/recip path, Q0.15 output |
| `matmul_u16s8_q15` | wrapper descriptor + shared matrix mixed mode | implemented | Q0.15 probability x int8 value -> int32, current RTL returns `acc >> 15` |
| `softmax` | Phase 0 operator template | legacy | generic smoke path, not the attention numerical contract |
| `matmul` | Phase 0 operator template | legacy | single-tile template used by older core tests |

Attention v1 must reuse these paths and make their contracts explicit in
operator metadata.

## Primitive Operators

### `matmul_s8s8_i32_tile`

Formula:

```text
C_i32[M,N] = A_s8[M,K] * B_s8[K,N]
```

Attention QK use:

```text
Score_raw_i32[S_q,S_k] = Q_s8[S_q,D_k] * K_t_s8[D_k,S_k]
```

Contract:

| Item | Value |
| --- | --- |
| Inputs | `a:int8 row-major MxK`, `b:int8 row-major KxN` |
| Output | `out:int32 row-major MxN` |
| Current tile | `8 x 8 x 8` |
| Larger K | descriptor K-stream chunks |
| Primitive RTL | matrix array |
| Attention contract | `attention_bringup_v0_qk_exact` |

No scale, mask, or requant is applied in this operator.

### `attention_score_scale_mask_v1`

Formula:

```text
Score_scaled = Score_raw / sqrt(D_k)
Score_masked = apply_mask(Score_scaled, mask)
```

This boundary is required even if the first implementation materializes the
scaled/masked softmax input in the fixture. Without it, the attention formula is
not traceable from QK to softmax.

Target fixed-point scale policy:

```text
scale_multiplier = round((1 / sqrt(D_k)) * 2^scale_shift)
Score_scaled = round_shift(Score_raw * scale_multiplier, scale_shift)
```

The compiler records:

| Field | Meaning |
| --- | --- |
| `scale_policy` | `power_of_two`, `multiplier_shift`, or `pre_scaled_fixture` |
| `scale_multiplier` | integer multiplier when used |
| `scale_shift` | right shift amount |
| `rounding` | `truncate`, `round_to_nearest`, or future mode |
| `score_q_format` | fixed-point interpretation consumed by softmax |

Example scale table:

| `D_k` | `1/sqrt(D_k)` | Example fixed policy |
| ---: | ---: | --- |
| 8 | 0.353553 | multiplier/shift table entry |
| 16 | 0.25 | exact `>> 2` |
| 64 | 0.125 | exact `>> 3` |

Mask kinds:

| Mask kind | Meaning |
| --- | --- |
| `none` | all lanes valid |
| `causal` | key positions greater than current query position are invalid |
| `padding` | padded sequence lanes are invalid |
| `tile_tail` | lanes outside logical sequence length are invalid |

Mask application:

```text
if lane_invalid:
    Score_masked = SCORE_NEG_INF
else:
    Score_masked = Score_scaled
```

`SCORE_NEG_INF` must be chosen so subtract-max plus clamp makes masked lanes
produce zero probability. It belongs in the numerical contract and must match
golden, fixtures, and RTL.

### `attention_softmax_q15_v1`

Formula:

```text
P_q15[row,j] = softmax(Score_masked[row,:])[j]
```

`P_q15` is an unsigned Q0.15 probability:

```text
0      -> 0.0
32768  -> 1.0 in arithmetic, with stored values saturating when required
```

Current executable simplified algorithm:

```text
row_max      = max(score)
delta        = score - row_max
delta_clamp  = clamp(delta, clamp_min, 0)
exp_q15      = sfu_exp(delta_clamp)
sum_exp      = sum(exp_q15)
recip_q24    = sfu_recip(sum_exp)
prob_q15     = (exp_q15 * recip_q24) >> 9
```

Notes:

- clamp is part of softmax implementation, not a separate attention formula
  stage;
- current SFU is approximate and LUT-based;
- current smoke accepts bounded tolerance against the fixed spec;
- future LUT/shift changes require a new numerical contract ID.

Contract:

| Item | Value |
| --- | --- |
| Input | `score_softmax_in:int32 row tile` |
| Output | `prob:uint16 Q0.15 row tile` |
| Primitive RTL | reduction max/sum, vector sub/clamp/mul-shift, SFU exp/recip |
| Current contract | `attention_bringup_v0_shift_scale_sfu9seg` |
| Target contract | `attention_numerical_v1_q15_prob_q24_recip_lut257` |

### `matmul_u16s8_q15_i32_tile`

Formula:

```text
O_i32[M,N] = sum_k P_q15[M,K] * V_s8[K,N] / 2^15
```

Attention PV use:

```text
O_i32[S_q,D_v] = P_q15[S_q,S_k] * V_s8[S_k,D_v]
```

Contract:

| Item | Value |
| --- | --- |
| Inputs | `a:uint16 Q0.15 row-major MxK`, `b:int8 row-major KxN` |
| Output | `out:int32 row-major MxN` |
| Current tile | `8 x 8 x 8` |
| Larger K | K-stream chunks along sequence dimension `S_k` |
| Primitive RTL | shared matrix array mixed mode |
| Current output policy | accumulated int32 then truncating `>> 15` |

This is not a separate PV RTL macro. It is a mixed-precision mode of the shared
matrix mechanism. Rounding/saturation is deferred and must be introduced with a
numerical-contract update.

### `scaled_dot_product_attention_v1`

Formula:

```text
O = softmax((Q * K^T) / sqrt(D_k) + mask) * V
```

This is a logical composite operator, not an RTL macro. It lowers to:

```text
score_raw     = matmul_s8s8_i32_tile(q, k_t)
softmax_in    = attention_score_scale_mask_v1(score_raw, mask)
prob_q15      = attention_softmax_q15_v1(softmax_in)
o_i32         = matmul_u16s8_q15_i32_tile(prob_q15, v)
```

Current status:

- QK, softmax, and PV are measured as separate SoC jobs;
- the parent logical attention workload is a compiler-produced
  `software_group_measured_stages` group for measured QK, softmax, and PV;
- scale/mask is still materialized, and the group does not yet prove a fully
  chained attention subnetwork.

## Proposed Metadata File

Add:

```text
sw/npu_core/operators/transformer_attention_v1.json
```

Required fields per operator:

- `kind`: `primitive` or `composite`;
- `descriptor_op`: current SoC descriptor op name when directly executable;
- `inputs` and `outputs`;
- `dtypes`;
- `layouts`;
- `tile_constraints`;
- `numerical_contract`;
- `primitive_engines`;
- `ppa_stage`;
- `current_status`.

The composite operator should reference primitive stages by name instead of
duplicating all stage details.

## Acceptance Criteria

- Every attention stage maps to an existing or planned shared primitive module.
- No stage requires a new `attention` RTL block.
- QK, scale/mask, softmax, and PV all have explicit dtype/layout contracts.
- The composite operator records that it lowers to primitive operators.
- Fixture generation can read operator metadata instead of hardcoding all
  attention semantics locally.
- PPA can identify `qk`, `scale_mask`, `softmax`, `pv`, and `full_attention`
  from operator metadata.
