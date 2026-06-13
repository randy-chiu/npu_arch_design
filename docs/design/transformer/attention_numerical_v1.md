# Attention Numerical v1

## Scope

This document explains the fixed-point math behind attention v1. It is the
numerical source that attention golden models, RTL, fixtures, and PPA
provenance must agree on.

The goal is not to prove model accuracy for a large LLM. The goal is a small,
deterministic, reviewable numerical contract for:

```text
QK score scale
masking
softmax EXP
softmax reciprocal
softmax normalization
P*V
```

## Fixed-Point Basics

Hardware stores integers. A fixed-point value is an integer plus an agreed
scale.

Example:

```text
real_value ~= integer_value / 2^F
```

If `F = 15`, then the format has fifteen fractional bits. That is the common
idea behind Q formats.

In this project:

| Name | Stored as | Real interpretation |
| --- | --- | --- |
| Q0.15 probability | unsigned integer `0..32767` | `p_real ~= p_q15 / 32767` |
| Q0.24 reciprocal | unsigned integer | `r_real ~= r_q24 / 2^24` |
| EXP input scale 32 | signed integer | `x_real = x_int / 32` |

The project uses `32767` as probability one because the output is a
uint16-like Q0.15 probability with maximum positive value `2^15 - 1`.

## QK Score Scale

Mathematical attention uses:

```text
scaled_score = score / sqrt(D)
```

where:

```text
score = sum(k=0..D-1) Q[k] * K[k]
```

`1 / sqrt(D)` is generally not exactly representable as a shift. For example:

```text
D = 8
1 / sqrt(8) = 0.353553390593...
```

### Bring-up shift policy

The simplest current-compatible approximation is:

```text
scaled_score = score >>> score_shift
```

This is cheap, but approximate. For `score_shift = 1`, scale is `0.5`. For
`score_shift = 2`, scale is `0.25`. Neither equals `0.353553` for `D=8`.

This policy is allowed only when metadata labels it as a bring-up approximation.

### Fixed multiplier policy

The target v1 policy represents the real scale as:

```text
scale_real ~= multiplier / 2^shift
scaled_score = round(score * multiplier / 2^shift)
```

Example for `D=8`, choosing `shift = 15`:

```text
scale_real = 1 / sqrt(8) = 0.353553390593...
multiplier = round(scale_real * 2^15)
           = round(0.353553390593 * 32768)
           = 11585
```

For `score = 1000`:

```text
floating reference:
  1000 / sqrt(8) = 353.553...

fixed-point:
  product = 1000 * 11585 = 11585000
  rounded = product + 2^14 = 11601384
  scaled = rounded >>> 15 = 354
```

So the fixed result is `354`, close to the floating result `353.553`.

This is why attention needs `VSCALE_FIXED`: current `VEC_REQUANT` can shift
and clamp, but it cannot express a reviewed fixed-point scale multiplier with
reviewed rounding. Because the score remains in the signed `int32` score
domain, this operation is scaling rather than a conversion to a new quantized
representation.

## Softmax Derivation

For one row:

```text
p[j] = exp(x[j]) / sum_t exp(x[t])
```

Stable softmax subtracts row max:

```text
m = max_j x[j]
d[j] = x[j] - m
p[j] = exp(d[j]) / sum_t exp(d[t])
```

Since `m` is the maximum, every `d[j] <= 0`. This is the reason the EXP input
range can be bounded to negative values.

## EXP Input Scale 32

The SFU EXP LUT does not take a floating-point input. It takes an integer
`x_int` interpreted as:

```text
x_real = x_int / 32
```

The softmax row supplies:

```text
x_int = clamp(d_int, -256, 0)
```

Therefore:

```text
x_int = 0      -> x_real = 0
x_int = -32    -> x_real = -1
x_int = -64    -> x_real = -2
x_int = -256   -> x_real = -8
```

The scale value `32` means there are 32 integer steps per real unit. It is a
precision/cost choice for the table index. With clamp range `[-256, 0]`, this
creates `257` entries:

```text
-256, -255, ..., -1, 0
```

### 中文说明：EXP 输入定点编码

这里的 `32` 是 SFU EXP 输入的定点编码 scale，不是 attention 公式中的
`sqrt(D)`。attention 先需要完成：

```text
scaled_score_real = score / sqrt(D)
```

然后再把这个 real 值编码成 EXP LUT 可寻址的整数：

```text
scaled_score_int = round(scaled_score_real * EXP_INPUT_SCALE)
```

实际硬件或编译器实现通常不会先生成浮点 `scaled_score_real`，而是把两个
scale 合并成一个整数 requant：

```text
scaled_score_int ~= round(score * EXP_INPUT_SCALE / sqrt(D))
```

也就是说，`EXP_INPUT_SCALE=32` 定义的是“一个整数 code 代表 `1/32` 个 real
EXP 输入单位”。它不会保证所有 real 值都能精确表示。比如：

```text
scaled_score_real = -0.701
scaled_score_real * 32 = -22.432
scaled_score_int = round(-22.432) = -22
scaled_score_real_approx = -22 / 32 = -0.6875
```

所以这是一个量化近似。更大的 `EXP_INPUT_SCALE` 会让相邻 real 输入格点更密，
但同样 `[-8.0, 0.0]` 范围下 LUT 表项也会更多。

LUT 不能直接用 real score 查询，因为 real 值不是有限硬件地址。表项在生成
阶段按照整数 grid 预先计算：

```text
table[index] = round(exp((index - 256) / 32) * 32767)
```

运行时 RTL 只使用整数 delta 做 clamp 和查表：

```text
delta_int = scaled_score_int - row_max_int
x_int = clamp(delta_int, -256, 0)
index = x_int + 256
```

忽略量化误差时：

```text
delta_int / 32 = scaled_score_real - row_max_real
```

因此 SFU 近似计算的是 `exp(scaled_score_real - row_max_real)`。这和直接用
`exp(scaled_score_real)` 做 softmax 得到的最终概率等价，因为 row max 引入的
`exp(-row_max_real)` 是整行共享因子，会在 softmax 分母中抵消。差异来自
round 量化误差和低于 `-8.0` 的 delta 被 clamp 到 `-8.0`。

## EXP Output Q0.15

EXP output is represented as:

```text
e_q15 = round(exp(x_real) * 32767)
```

Examples:

| `x_int` | `x_real` | `exp(x_real)` | `e_q15` |
| ---: | ---: | ---: | ---: |
| `0` | `0.0` | `1.000000` | `32767` |
| `-32` | `-1.0` | `0.367879` | `12054` |
| `-64` | `-2.0` | `0.135335` | `4435` |
| `-128` | `-4.0` | `0.018316` | `600` |
| `-256` | `-8.0` | `0.000335` | `11` |

These are target fixed-spec values for the `round(exp(x) * 32767)` convention.
Some existing bring-up segment constants differ by one count because they were
coarse hand-selected approximations. Attention v1 requires a generated
257-entry table so neighboring inputs do not collapse into the same coarse
segment and golden/RTL do not carry two hand-maintained tables.

## Reciprocal Q0.24

After EXP:

```text
sum_q15 = sum_j e_q15[j]
```

The reciprocal is:

```text
r_q24 = floor(2^24 / sum_q15)
r_real ~= r_q24 / 2^24
```

The left shift by 24 creates fractional precision for the reciprocal. Without
that scale, integer `1 / sum_q15` would be zero for almost every softmax row.

Example with two entries:

```text
e_q15 = [32767, 12055]
sum_q15 = 44822

real reciprocal:
  1 / 44822 = 0.000022310...

Q0.24 reciprocal:
  r_q24 = floor(16777216 / 44822) = 374
  r_q24 / 2^24 = 374 / 16777216 = 0.000022292...
```

The reciprocal integer looks small, but it represents a small fractional value
because it is Q0.24.

## Softmax Normalization Back To Q0.15

The target probability is:

```text
p_q15[j] ~= (e_q15[j] / sum_q15) * 32767
```

Using `r_q24 ~= 2^24 / sum_q15`:

```text
p_q15[j] = round(e_q15[j] * r_q24 * 32767 / 2^24)
```

Then clamp:

```text
p_q15[j] = clamp(p_q15[j], 0, 32767)
```

Continuing the example:

```text
e_q15 = [32767, 12055]
sum_q15 = 44822
r_q24 = 374
PROB_ONE = 32767
```

First probability:

```text
numerator = 32767 * 374 * 32767 = 401,554,932,086  (wide integer)
p0 = round(numerator / 2^24)
   ~= 23935
```

Second probability:

```text
p1 = round(12055 * 374 * 32767 / 2^24)
   ~= 8806
```

The sum is near `32767`; small error comes from reciprocal truncation and
rounding:

```text
p0 + p1 ~= 32741
```

This example shows why the vector normalization path needs a wide intermediate.
`e_q15 * r_q24 * 32767` can exceed 32 bits. Current `VEC_MUL`, which keeps only
low `DATA_WIDTH` bits, is not a valid attention-softmax normalization
implementation.

## P Times V

The final attention output is:

```text
O[i,d] = sum_j P[i,j] * V[j,d]
```

With Q0.15 probability:

```text
O_acc[i,d] = sum_j p_q15[i,j] * V_i8[j,d]
O[i,d] = round(O_acc[i,d] / 32767)
```

Example:

```text
p_q15 = [24575, 8192]        # roughly [0.75, 0.25]
V      = [20, -12]

O_acc = 24575 * 20 + 8192 * (-12)
      = 491500 - 98304
      = 393196

O = round(393196 / 32767)
  ~= 12

floating reference:
  0.75 * 20 + 0.25 * (-12) = 12
```

If the project instead requantizes `P` to int8 and reuses the current int8
matrix engine, this formula changes and accuracy must be reviewed separately.

## Golden And RTL Consistency

Golden and RTL must not implement these formulas independently from memory.
The following rules are required before attention numerical work is accepted.

### One named numerical contract

Each attention fixture must carry a numerical contract ID, for example:

```text
attention_numerical_v1_q15_prob_q24_recip_lut257
```

The contract ID identifies:

- score scale policy;
- EXP input scale and table version;
- reciprocal Q format;
- normalization formula;
- rounding mode;
- clamp bounds;
- mask policy;
- PV policy.

### Generated constants and tables

EXP LUT contents must be generated from the numerical spec/config. RTL must
include the generated table or package. Python golden must use the same
generator or the same checked output table. Hand-written duplicate EXP tables
are not allowed for the target path.

### Stage-by-stage fixtures

Fixtures must include intermediate expected values:

```text
QK scores
scaled scores
masked scores
row max
delta / EXP input
EXP Q0.15
row sum
RECIP Q0.24
P Q0.15
P*V output
```

Tests should compare stages, not only final `O`. This catches mismatches such
as correct final output hiding a wrong EXP table or reciprocal shift.

### Named golden functions

Python helpers must use names that identify their role:

```text
attention_softmax_fixed_spec_v1(...)
attention_softmax_rtl_model_v1(...)
attention_softmax_float_reference(...)
```

The float reference is for understanding and tolerance studies. It is not a
substitute for the fixed-spec golden used by RTL tests.

### RTL comparison rule

An RTL attention primitive or sequence passes only if it matches the fixed-spec
golden for the reviewed contract. If RTL intentionally uses a coarser
approximation, the workload and PPA report must label the stage as bring-up or
approximate, not as the target attention numerical implementation.

## Iteration And Upgrade Plan

Attention v1 may start with simplified primitive behavior so the project can
measure the sequence early. That is acceptable only if each simplification is
named in the numerical contract and has a planned replacement path.

The goal is to keep interfaces and fixture structure stable while improving
individual numerical policies.

### Iteration 0: current bring-up primitives

Purpose:

- prove workload generation, stage metadata, scheduler/runtime plumbing, and
  PPA grouping;
- avoid blocking QK/PV/control work on perfect softmax math.

Allowed simplifications:

| Area | Simplified behavior | Report label |
| --- | --- | --- |
| score scale | power-of-two shift | `score_scale=shift_approx` |
| EXP | 9-segment bring-up LUT | `sfu_exp=bringup_9_segment` |
| RECIP | direct integer expression | `recip=direct_div_bringup` |
| normalization | current RTL-like approximation | `softmax_norm=bringup` |
| mask | unmasked rows only | `mask=none` |
| PV | model-only or probability requant to int8 | `pv=model_only` or `pv=p_i8_approx` |

Exit condition:

- stage-by-stage golden exists for the simplified contract;
- reports clearly distinguish simplified numerical evidence from target
  attention evidence.

### Iteration 1: target softmax numerical path

Purpose:

- make row softmax accurate and deterministic enough for attention PPA.

Required upgrades:

- score scale uses fixed multiplier/shift for `1/sqrt(D)`;
- EXP uses generated 257-entry LUT;
- RECIP output is Q0.24 with reviewed zero behavior;
- normalization uses the reviewed Q0.15 formula from this document;
- row-softmax fixtures include intermediate values.

Trigger:

- before reporting measured `attention_softmax_s8` as target numerical
  evidence;
- before using full attention output accuracy to choose architecture variants.

Exit condition:

- fixed-spec golden and RTL match stage-by-stage;
- old bring-up functions remain only as compatibility tests or are removed from
  active attention workloads.

### Iteration 2: mask and tile-tail correctness

Purpose:

- support decoder-style attention semantics and tiled rows.

Required upgrades:

- define `SOFTMAX_NEG_INF`;
- retain the implemented compact row-mask contract through Scale/Mask,
  Reduction, and Softmax normalization;
- add causal, padding, and tile-tail fixtures.

#### Numerical mask contract

Design status: implemented for executable causal `S=8,D=8`.

Masking is a validity operation, not ordinary addition of a finite bias. The
architectural result must satisfy:

```text
invalid lane:
  does not participate in row_max
  does not participate in row_sum
  probability_q15 = 0
```

`SOFTMAX_NEG_INF` is the stored representation used by the selected
Scale/Mask path for invalid score lanes. Its canonical value is signed int32
minimum, `-2147483648`, emitted from architecture configuration. It is not
subtracted as ordinary valid data: the same row mask excludes it from
Reduction, gates later vector operations, and forces invalid probabilities to
zero. This avoids depending on sentinel arithmetic or risking signed overflow.

Selection criteria:

1. subtraction from every legal valid-row maximum cannot overflow int32;
2. the reviewed EXP contract maps the resulting clamped delta to zero, or
   Softmax explicitly gates the invalid probability to zero;
3. RTL, golden, fixtures, and compiler-generated constants use one canonical
   value;
4. the value is not used as a substitute for Reduction validity gating.

The selected implementation therefore uses both:

- a sentinel in the stored masked-score tile, for traceability and safe
  downstream vector behavior;
- `valid_lane_mask` in Reduction/normalization, so invalid lanes are excluded
  exactly rather than relying only on a finite sentinel approximation.

A row with no valid lane is invalid input in v1. Compiler/runtime reject it and
hardware reports an error if it reaches execution. Defining an arbitrary
uniform or zero distribution for such a row is rejected because it can hide a
shape/mask bug.

Trigger:

- before claiming decoder prefill semantics;
- before using decode attention rows with unused cache/tile lanes.

Exit condition:

- invalid positions do not affect row max, row sum, probability, or output;
- PPA metadata records mask policy.

### Iteration 3: PV accuracy path

Purpose:

- replace model-only or int8-probability approximation with a reviewed
  probability-value implementation.

Options:

| Option | Meaning | Trigger |
| --- | --- | --- |
| `P_q15_to_i8_then_matmul` | reuse current matrix path after probability requant | early measured PV with known accuracy loss |
| `mixed_q15_i8_weighted_sum` | compute Q0.15 probability times int8 value | when PV error or PPA evidence justifies new datapath |

Exit condition:

- chosen PV policy has fixed-spec golden, RTL/model agreement, and PPA area/ops
  accounting.

### Iteration 4: implementation-quality SFU

Purpose:

- improve area/timing/power after numerical behavior is stable.

Allowed implementation replacements:

- direct reciprocal divider to LUT or LUT+Newton;
- combinational EXP LUT to registered ROM;
- multi-cycle SFU with valid/ready.

Rule:

The external numerical contract should remain unchanged unless there is a
reviewed spec change. If the contract changes, all golden functions, fixtures,
RTL tests, workload metadata, and PPA provenance must change in the same patch.

## Upgrade Complexity Assessment

Starting with simplified vector/SFU behavior is manageable if the project keeps
these boundaries stable:

- tensor shapes and stage names;
- fixture intermediate fields;
- workload numerical contract ID;
- command-list fields for scale, clamp, mask, and stage identity;
- PPA provenance fields.

It becomes expensive if early RTL hard-codes hidden constants without exposing
them in metadata. Therefore every simplification must be represented as an
explicit policy field rather than an implicit implementation detail.

Examples:

```text
Good:
  score_scale_policy = "power_of_two_shift"
  score_shift = 1

Bad:
  RTL shifts scores by 1 internally and golden happens to match.
```

```text
Good:
  exp_table_id = "bringup_9_segment"

Bad:
  Python and RTL each contain hand-written EXP constants with no table ID.
```
