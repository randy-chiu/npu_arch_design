# Attention Sequence v1

## Scope

This document defines how Transformer attention is represented on the current
unified NPU path. Attention v1 is a software/compiler scheduled primitive
sequence. It is not a dedicated `attention_engine` RTL macro.

The goal is to make attention the main Transformer workload and PPA driver while
reusing and extending the existing primitive engines:

- matrix engine;
- accumulator file;
- vector engine;
- reduction engine;
- SFU;
- data mover and local tensor memory;
- wrapper descriptor/runtime/perf path.

## Mathematical Contract

For one attention head:

```text
Q: S x D
K: S x D
V: S x D

scores = Q * K^T
scaled_scores = scores / sqrt(D)
masked_scores = apply_mask(scaled_scores)
P = softmax(masked_scores)
O = P * V
```

Expanded per element:

```text
scores[i,j] =
  sum(k=0..D-1) Q[i,k] * K[j,k]

scaled_scores[i,j] =
  scores[i,j] / sqrt(D)

masked_scores[i,j] =
  scaled_scores[i,j]                 if position j is visible to query i
  SOFTMAX_NEG_INF                    otherwise

P[i,j] =
  exp(masked_scores[i,j]) /
  sum(t=0..S-1) exp(masked_scores[i,t])

O[i,d] =
  sum(j=0..S-1) P[i,j] * V[j,d]
```

The scale by `1 / sqrt(D)` is part of scaled dot-product attention. It prevents
the dot-product magnitude from growing with `D`, which would otherwise push
softmax toward saturated one-hot outputs and make quantized attention unstable.
For v1 hardware-facing fixed point, the compiler must lower this scale into an
explicit vector/requant operation before softmax.

Masking is part of the attention semantics, not an optimization:

- prefill self-attention with a causal decoder model must prevent query position
  `i` from attending to future keys `j > i`;
- decode attention for one newly generated token usually has no future tokens in
  the current context, but it may still need a padding/valid-length mask when a
  row contains unused cache slots or tile tail lanes;
- encoder-style bidirectional attention may use no causal mask, but can still
  use padding masks.

The first measured attention target may be unmasked `S=8,D=8` to validate QK
and softmax mechanics. Causal/padding mask support must be added before claiming
decoder-style attention semantics.

## Fixed-Point Policy

Detailed fixed-point derivations, examples, and golden/RTL consistency rules are
owned by `docs/design/transformer/attention_numerical_v1.md`. This section keeps
the execution-level summary.

The mathematical attention formula uses real numbers:

```text
scores / sqrt(D)
exp(x)
1 / sum(exp(x))
P * V
```

The RTL path cannot leave these as abstract floating-point operations. Every
real-valued step must be lowered to an explicit fixed-point representation,
rounding rule, clamp rule, and testable golden model.

### Probability Q0.15 convention

`Q0.15` means a fixed-point value with zero integer magnitude bits and fifteen
fractional bits. In many DSP/NPU designs, softmax probabilities are stored as
unsigned fixed-point integers because probabilities are in `[0, 1]`.

This project uses an unsigned Q0.15-like softmax probability convention:

```text
PROB_ONE = 32767
real_probability ~= p_q15 / PROB_ONE
p_q15 range = 0..32767
```

So:

```text
0.0   -> 0
0.5   -> about 16384
1.0   -> 32767
```

This is close to common Q0.15 usage. Some industry implementations instead use
Q1.15 signed values, int8 probabilities with a tensor scale, bfloat/FP16, or
block-floating formats depending on model accuracy and hardware cost. For this
project, Q0.15 is the first reviewed softmax probability format because it is
simple, deterministic, and keeps probability precision higher than int8 before
the `P*V` stage.

### Score scale policy

The real scale is:

```text
scale_real = 1 / sqrt(D)
scaled_score_real = score_int32 * scale_real
```

This floating-point value is not stored or computed as floating point in v1.
The compiler must select one of these explicit policies and record it in
workload metadata:

| Policy | Formula | Status |
| --- | --- | --- |
| `power_of_two_shift` | `scaled = score >>> score_shift` | current bring-up compatible |
| `fixed_multiplier_shift` | `scaled = round(score * multiplier / 2^shift)` | target for better `1/sqrt(D)` approximation |
| `model_only_float_reference` | Python float reference only | allowed for accuracy studies, not RTL evidence |

For `D=8`, `1/sqrt(8) ~= 0.353553`. A plain right shift cannot represent this
exactly. A fixed multiplier can approximate it, for example:

```text
multiplier = round((1 / sqrt(8)) * 2^15) = 11585
scaled = round(score * 11585 / 2^15)
```

The exact multiplier width, rounding mode, and shift belong to the vector
requant v2 contract before RTL implementation. Until that is implemented,
measured attention QK may use a documented shift approximation, while full
attention accuracy claims must use the reviewed fixed multiplier policy.

### EXP and reciprocal policy

Softmax is implemented as:

```text
delta = masked_scaled_score - row_max
exp_input = clamp(delta, -256, 0)
e_q15 = EXP_Q0_15(exp_input / 32)
sum_q15 = sum(e_q15)
recip_q24 = floor((1 << 24) / sum_q15)
p_q15 = normalize(e_q15, recip_q24)
```

The initial target is:

- EXP: generated 257-entry LUT for integer inputs `[-256, 0]`, scale `32`;
- RECIP: integer reciprocal returning unsigned Q0.24;
- normalization: vector multiply/requant from `e_q15 * recip_q24` back to
  Q0.15.

Current RTL has only a coarse 9-segment EXP bring-up model and a simple integer
division reciprocal. That path is useful for primitive wiring tests but is not
the final attention softmax numerical contract.

### Golden/RTL consistency rule

Every attention workload must identify the numerical contract used by both
Python golden and RTL. The target path must compare stage-by-stage
intermediates, not only final output:

```text
scores -> scaled_scores -> masked_scores -> row_max -> exp_input
-> e_q15 -> sum_q15 -> recip_q24 -> P_q15 -> O
```

The same generated EXP table and constants must feed both golden and RTL. If RTL
uses a coarser approximation, the workload must be labeled as bring-up or
model-only for attention PPA.

The first executable target is fixed-point, single-head attention:

| Symbol | v1 initial value | Type |
| --- | --- | --- |
| `S` | 8 first, then 16 | sequence length |
| `D` | 8 first, then 16 | head dimension |
| `Q`, `K`, `V` | int8 | activation/KV tensor |
| `scores` | int32 | exact dot-product accumulator |
| `scaled_scores` | int32 | shifted/fixed-point score |
| `P` | unsigned Q0.15 | softmax probability matrix |
| `O` | int32 first, optional int8 after requant | attention output |

The `S=8,D=8` target maps directly onto the current `8x8x8` matrix tile. Larger
`S` or `D` values are represented as tiled primitive sequences.

## Primitive Decomposition

Attention decomposes into five primitive groups.

### 1. QK Score Matmul

Formula:

```text
scores[i,j] = sum(k=0..D-1) Q[i,k] * K[j,k]
```

Hardware implication:

- matrix engine executes `Q * K^T`;
- compiler/runtime must supply `K^T` layout or a tile loader that presents K as
  `D x S`;
- accumulator stores int32 score tiles.

Current status:

- `matmul_array.sv` supports one `8x8x8` int8/int8/int32 tile;
- K-streaming accumulation supports larger K in the wrapper/core path;
- current workload generator can emit deterministic `matmul_k_stream`
  fixtures.

Needed for attention:

- document and test `K^T` tile layout for QK;
- preserve logical shape metadata in workload manifests;
- expose QK cycles/MACs separately from PV cycles/MACs in perf/PPA;
- add true skinny/GEMV handling later for decode attention.

### 2. Score Scale

Formula:

```text
scaled_scores[i,j] = scores[i,j] / sqrt(D)
```

For initial fixed-point v1 this is represented as:

```text
scaled_scores[i,j] = requant(scores[i,j], scale = 1 / sqrt(D))
```

The simplest current-compatible approximation is a power-of-two shift:

```text
scaled_scores[i,j] = scores[i,j] >>> score_shift
```

For `D=8`, `1 / sqrt(D)` is not exactly a power-of-two. Therefore this shift is
a bring-up approximation unless the workload explicitly chooses `score_shift`
as its fixed-point numerical policy. A more faithful implementation needs
`mul_round_shift_clamp`:

```text
wide = scores[i,j] * score_scale_multiplier
scaled_scores[i,j] = round(wide) >>> score_scale_shift
```

Hardware implication:

- vector engine needs a reviewed scale/requant operation for int32 score lanes;
- current `VEC_REQUANT` can only do arithmetic shift plus clamp;
- `requant_v2` is needed when attention accuracy requires non-power-of-two
  `1/sqrt(D)` scaling.

Current status:

- current vector standalone RTL has `VEC_SCALE` and `VEC_REQUANT`;
- current `VEC_REQUANT` is `a >>> shift` with clamp;
- no scheduler-integrated vector path exists yet.

Needed for attention:

- workload metadata must state `head_dim`, `score_scale_policy`,
  `score_shift` or multiplier/shift fields;
- Python golden must use the same score scale policy as the intended RTL path;
- PPA reports must identify whether scale is exact fixed-point or a bring-up
  shift approximation.
- full attention accuracy claims require the fixed multiplier policy or another
  reviewed approximation for `1/sqrt(D)`.

### 3. Attention Mask

Formula:

```text
masked_scores[i,j] = mask[i,j] ? scaled_scores[i,j] : SOFTMAX_NEG_INF
```

Mask meaning:

```text
mask[i,j] = 1  means key/value position j is visible to query position i
mask[i,j] = 0  means it must not affect softmax or output O
```

Common masks:

| Mask | Condition | Why it exists |
| --- | --- | --- |
| none | all `mask[i,j]=1` | useful for first prefill bring-up and non-causal attention |
| causal | `mask[i,j]=1` only when `j <= i` | decoder prefill must not see future tokens |
| valid length / padding | mask out unused sequence or cache slots | decode/cache tiles may include invalid tail lanes |
| tile tail | mask out lanes beyond logical `S` | tiled rows may be wider than logical row length |

Masking must happen before row max. If an invalid score is simply zeroed after
softmax, it can still affect row max, row sum, and output probability. The
correct behavior is to exclude invalid positions from softmax. Hardware can
implement this in one of three ways:

1. Compiler materializes `SOFTMAX_NEG_INF` into invalid score lanes before the
   reduction.
2. Vector engine adds a mask-select op:

   ```text
   masked_score = valid ? scaled_score : SOFTMAX_NEG_INF
   ```

3. Reduction/softmax primitives accept a valid mask and ignore invalid lanes.

The first v1 implementation should use unmasked rows for measured QK/softmax
bring-up, then add causal/tail mask support as an explicit feature. The chosen
mask policy must be recorded in workload metadata.

Current status:

- vector engine has `valid_mask`, but inactive lanes currently produce zero;
- reduction engine has `length` but no general per-lane mask;
- no reviewed `SOFTMAX_NEG_INF` sentinel is defined.

Needed for attention:

- define `SOFTMAX_NEG_INF` in the numerical contract;
- decide whether mask is compiler-materialized, vector mask-select, or
  reduction-valid-mask;
- add mask/tail golden vectors before claiming decoder attention behavior.

### 4. Row Softmax

Formula per row:

```text
row_max[i] = max_j masked_scores[i,j]
delta[i,j] = masked_scores[i,j] - row_max[i]
e[i,j] = exp(delta[i,j])
row_sum[i] = sum_j e[i,j]
P[i,j] = e[i,j] / row_sum[i]
```

Clamp is part of the fixed-point softmax implementation, not a separate
attention semantic. It exists because the SFU EXP target only supports a bounded
input interval:

```text
exp_input[i,j] = clamp(delta[i,j], -256, 0)
```

This clamp happens after row max subtraction. Since `row_max` makes
`delta <= 0`, the high clamp to zero is mainly a safety/saturation rule. The
low clamp to `-256` maps all very small probabilities to the minimum EXP table
entry instead of requiring an unbounded LUT.

Fixed-point v1 target:

```text
exp_input = clamp(masked_scores - row_max, -256, 0)
e_q15 = EXP_Q0_15(exp_input / 32)
sum_q15 = sum(e_q15)
recip_q24 = RECIP_Q24(sum_q15)
P_q15 = requant(e_q15 * recip_q24)
```

Hardware implication:

- reduction engine provides row max and row sum;
- SFU provides EXP and reciprocal;
- vector engine performs subtract, clamp, multiply/requant;
- row length and lane tail behavior must be explicit.

Current status:

- reduction standalone RTL supports max/sum over packed int32 inputs;
- SFU standalone RTL supports coarse EXP and integer reciprocal;
- vector standalone RTL supports subtract, clamp, and simple requant;
- old `npu_v0_compute_cluster` softmax is Phase 0-specific and should not be treated as
  the Transformer attention softmax implementation.

Needed for attention:

- replace coarse SFU EXP with reviewed 257-entry Q0.15 LUT or keep reports
  clearly labeled as bring-up accuracy;
- define reciprocal Q format and normalization shift;
- add row-softmax golden vectors and tolerance;
- add valid/ready/counter semantics before scheduler integration;
- report reduction/SFU/vector cycles separately.

### 5. Probability-Value Matmul

Formula:

```text
O[i,d] = sum(j=0..S-1) P[i,j] * V[j,d]
```

`Probability-Value` means the final attention weighted sum: the softmax
probability matrix `P` weights the value matrix `V`. This is often abbreviated
as `P*V` or attention-value matmul. `P` is not a learned weight matrix; it is
the runtime probability distribution produced from QK scores.

The mixed-precision issue comes directly from the datatypes:

```text
P[i,j]  = softmax probability, naturally fractional in [0, 1]
V[j,d]  = int8 value activation/cache element
O[i,d]  = sum of fractional probability times int8 value
```

The current matrix engine accepts int8 x int8. But the numerically natural
fixed-point probability is Q0.15, not int8. Therefore `P*V` has two possible
hardware-facing representations.

Initial implementation options:

1. Convert or requant `P_q15` to int8 and reuse the int8 matrix engine:

   ```text
   P_i8 = requant(P_q15)
   O_i32 = P_i8 * V_i8
   ```

   This is current-matrix compatible but loses probability precision and needs
   a reviewed scale/zero-point policy.

2. Add mixed precision support for Q0.15 probability times int8 value:

   ```text
   O_acc[i,d] = sum_j P_q15[i,j] * V_i8[j,d]
   O[i,d] = O_acc[i,d] >>> 15
   ```

   This better matches attention math but requires a new weighted-sum or
   mixed-precision matrix path.

3. Model PV until a reviewed mixed-precision or requant path exists.

The first option is simplest for current RTL compatibility but loses
probability precision. The second option better represents attention but adds
hardware cost and PPA implications.

Current status:

- matrix engine only supports int8 x int8 -> int32;
- vector engine can shift/clamp values but does not define probability-to-int8
  attention policy;
- no mixed Q0.15 x int8 matrix datapath exists.

Needed for attention:

- choose the first executable PV policy before RTL work:
  `P_q15_to_i8_then_matmul` or `mixed_q15_i8_weighted_sum`;
- if using int8 probability, define rounding/clamp and expected error;
- if using mixed precision, update matrix/vector docs, area proxy, and tests.

## Current Implementation Inventory

| Capability | Current implementation | Attention readiness |
| --- | --- | --- |
| int8 tile matmul | `hw/npu_core/rtl/matrix/matmul_array.sv` | ready for QK `S=8,D=8`; PV only if probability is int8 |
| K-stream accumulation | wrapper/core K-stream descriptor path | useful for larger `D`; not attention-specific |
| accumulator file | `matrix/accumulator_file.sv` | stores matrix tile accumulations; score/probability lifetime needs memory contract |
| vector primitive ops | `vector/vector_engine.sv` | standalone only; needs mask/requant details and scheduler path |
| reduction primitive ops | `reduction/reduction_engine.sv` | standalone only; row softmax length/mask/latency needs contract |
| SFU primitive ops | `sfu/sfu_lut.sv` | coarse bring-up only; EXP/RECIP detail must be upgraded |
| primitive integration wrapper | `transformer_primitive_engines.sv` | standalone test integration only |
| core command processor/runtime descriptors | `npu_v0_core_system.sv` | matmul/k-stream focused; no command-list attention sequence |
| perf/PPA | perf CSR snapshots and L0 report | matmul/data mover visible; attention group counters missing |

## Planned Execution Stages

The numerical iteration plan is defined in
`docs/design/transformer/attention_numerical_v1.md`. The execution stages below
are intentionally compatible with starting from simplified primitives and later
upgrading score scale, EXP, RECIP, normalization, mask, and PV policy. The key
rule is that each simplification must be explicit in workload metadata and
fixture contract IDs.

### Stage A: Model And Documentation

- Add attention golden functions.
- Add attention workload manifest entries.
- Keep softmax/PV model-only where RTL cannot yet execute them.
- Define PPA grouping and expected fields.
- Add numerical policy fields for any simplified behavior, such as
  `score_scale=shift_approx` or `sfu_exp=bringup_9_segment`.

Exit criteria:

- Python tests validate QK, row softmax, PV policy, and full attention output.
- PPA report can include attention model-only sections without pretending they
  are measured.

### Stage B: Measured QK Matmul

- Execute `attention_qk_s8_d8` using current K-stream or single-tile matmul.
- Track Q/K bytes, output score bytes, effective MACs, and matrix utilization.

Exit criteria:

- `perf-report` and `ppa-l0-report` show measured QK cycles.
- Existing CNN and operator smoke regressions still pass.

### Stage C: Scheduler-Visible Softmax Primitives

- Connect vector/reduction/SFU primitive issue path.
- Add row-softmax micro-kernel tests.
- Report vector/reduction/SFU active cycles.
- Start with simplified primitives only if reports label them as bring-up.
- Upgrade to target attention softmax before using output accuracy or SFU PPA as
  architecture evidence.

Exit criteria:

- row softmax has RTL-measured primitive cycles and matches the active
  numerical contract stage-by-stage.

### Stage D: PV Path

- Choose and implement the reviewed PV policy.
- Report PV MACs/cycles separately.
- Integrate full `attention_prefill_s8_d8` as measured sequence.

Exit criteria:

- QK, softmax, and PV are all represented in one attention workload group.

### Stage E: Decode And KV Traffic

- Add decode shape such as `S=1, context=32, D=16`.
- Keep KV cache as external-memory accounting until RTL streamer is justified.
- Use PPA evidence to decide whether true GEMV/skinny-GEMM or KV streaming is
  the next hardware priority.

## Non-Goals

- No dedicated attention RTL macro in v1.
- No fused attention pipeline.
- No full decoder block until attention sequence evidence is stable.
- No multi-head hardware fusion.
- No real LPDDR/KV cache controller.
