# SFU v1

## Scope

SFU v1 provides primitive special functions required by Transformer attention
softmax and RMSNorm. The immediate attention driver is row softmax:

```text
P[i,j] = exp(x[i,j] - max_j x[i,j]) / sum_j exp(x[i,j] - max_j x[i,j])
```

From this formula, attention needs:

- `EXP` for each shifted/clamped score element;
- `RECIP` for the reciprocal of the row EXP sum;
- vector multiply/requant after reciprocal;
- `RSQRT` remains for RMSNorm, not for attention softmax.

This document defines why the current SFU is insufficient for attention
evidence and what must be implemented before measured attention softmax can be
used as PPA evidence.

## Source Of Truth

Static configuration lives in `arch/configs/npu_transformer_v1.jsonc`.

| Parameter | Config field | Target/current value |
| --- | --- | --- |
| `DATA_WIDTH` | `modules.sfu.data_width` | 32 |
| EXP input scale | `modules.sfu.exp_input_scale` | 32 |
| EXP LUT entries | `modules.sfu.exp_lut_entries` | target 257 |
| EXP output Q | `modules.sfu.exp_output_q` | 15 |
| Bring-up EXP segments | `modules.sfu.bringup_exp_q15_segments` | current 9-segment path |
| RECIP output Q | `modules.sfu.recip_output_q` | 24 |
| RSQRT output Q | `modules.sfu.rsqrt_output_q` | 24 |
| `OP_SFU_*` | `primitive_op_encodings.sfu.*` | generated op encodings |

Generated integration constants are emitted by `make transformer-config` into
`build/generated/npu_transformer_v1_config_pkg.sv`.

## Attention Softmax Derivation

The fixed-point background and worked examples for EXP scale, Q0.15 output, and
Q0.24 reciprocal are in these sections of
`docs/design/transformer/attention_numerical_v1.md`:

- `Softmax Derivation`
- `EXP Input Scale 32`
- `EXP Output Q0.15`
- `Reciprocal Q0.24`
- `Softmax Normalization Back To Q0.15`

This section defines the SFU-specific contract.

For one score row:

```text
x[j] = masked_scaled_score[j]
m = max_j x[j]
d[j] = x[j] - m
e[j] = exp(d[j])
s = sum_j e[j]
p[j] = e[j] / s
```

The max subtraction keeps `d[j] <= 0`, which bounds EXP input and avoids large
positive exponential values. For v1 fixed-point:

```text
d_int[j] = clamp(x_int[j] - m_int, -256, 0)
e_q15[j] = EXP_LUT_Q0_15(d_int[j] / 32)
s_q15 = sum_j e_q15[j]
r_q24 = RECIP_Q24(s_q15)
p_q15[j] = requant(e_q15[j] * r_q24)
```

This derivation implies concrete SFU requirements:

1. EXP only needs the negative range `[-256, 0]` for attention softmax.
2. EXP output must be accurate enough that row probability ordering and sum are
   stable across tested rows.
3. RECIP input is the positive row sum of Q0.15 EXP values.
4. RECIP output Q format must match the vector normalization shift.
5. EXP/RECIP latency and issue rate must be measurable for PPA.

## Current RTL Status

Current implementation:

```text
hw/npu_core/rtl/sfu/sfu_lut.sv
```

Implemented today:

- `SFU_EXP`: coarse 9-segment Q0.15 LUT over clamped negative input;
- `SFU_RECIP`: integer truncating `(1 << 24) / x`;
- `SFU_RSQRT`: integer `isqrt(x)` followed by `(1 << 24) / root`;
- start/done single-cycle bring-up interface;
- standalone primitive testbench coverage.

Current gaps for attention:

- EXP config says `257` entries but RTL still uses 9 segments.
- EXP accuracy is a bring-up approximation, not a reviewed attention softmax
  numerical target.
- RECIP has no stated input range, rounding policy, or overflow behavior beyond
  zero returning zero.
- There is no valid/ready contract, so scheduler integration cannot model
  stalls or multi-cycle SFU latency.
- There are no measured `sfu_active_cycles`, `sfu_exp_ops`, or
  `sfu_recip_ops` counters.

Before scheduler integration, the SFU must adopt
`primitive_valid_ready_v1.md` semantics:

- `cmd_valid/cmd_ready` accepts one SFU command and stable operand/op payload;
- `rsp_valid/rsp_ready` transfers the result;
- EXP, RECIP, and RSQRT may have different declared response latencies and
  initiation intervals;
- `sfu_active_cycles`, `sfu_input_stall_cycles`, `sfu_output_stall_cycles`,
  `sfu_exp_ops`, `sfu_recip_ops`, and `sfu_rsqrt_ops` are counted from
  handshake events and accepted op type.

CSR/report exposure must wait until these local event sources are tested.

## EXP Target

The SFU `EXP` target is not a general-purpose `exp()` implementation. It is the
bounded exponential primitive needed by stable attention softmax after row-max
subtraction. The `DATA_WIDTH=32` operand width is the transport/interface width;
the EXP numerical domain is the post-clamp domain.

For attention, software or the row-softmax primitive sequence supplies a
score-delta input:

```text
delta = masked_scaled_score - row_max
```

This delta is always non-positive before clamping because `row_max` is the
largest valid element in the row. It is not guaranteed to be greater than or
equal to `-256`; values below `-256` are intentionally saturated to `-256`.
With `EXP_INPUT_SCALE=32`, that clamp means all lower real inputs are treated as
`-8.0`.

The division by `EXP_INPUT_SCALE` is the fixed-point interpretation of the
integer EXP input, not a runtime divider requirement:

```text
x_real = x_int / EXP_INPUT_SCALE
```

This scale does not preserve softmax equivalence by itself. Softmax is
invariant to subtracting the row maximum, but it is sensitive to multiplying or
dividing all logits by a constant. Therefore `EXP_INPUT_SCALE=32` is part of
the reviewed attention numerical contract: the upstream score scale/requant
path must produce `x_int` values whose intended real value is `x_int / 32`.
If the upstream path instead produced integer logits in a different unit, using
this EXP table would change the softmax temperature and the probabilities.

For attention, the upstream producer is the score path:

```text
score = Q * K^T
scaled_score ~= score / sqrt(D)
delta = masked_scaled_score - row_max
```

The scale/requant stage must encode `masked_scaled_score` and `delta` in the
same `1/EXP_INPUT_SCALE` units expected by the SFU. This makes the v1 SFU EXP
an attention-softmax primitive with a fixed input unit, not a universal softmax
block for arbitrary logit scales. A different producer may reuse it only if it
emits the same fixed-point unit or explicitly converts into that unit.

`EXP_INPUT_SCALE` is separate from the attention `sqrt(D)` score scale. For a
head dimension `D`, the compiler/numerical contract must first apply the
attention scale:

```text
scaled_score_real = score / sqrt(D)
```

Then it must encode that real scaled score into the EXP input unit:

```text
scaled_score_int = round(scaled_score_real * EXP_INPUT_SCALE)
```

This multiplication does not make every real value exactly representable. It is
a quantization rule: `scaled_score_int` is an integer code whose real
interpretation is `scaled_score_int / EXP_INPUT_SCALE`. With
`EXP_INPUT_SCALE=32`, the representable values are spaced by `1/32`, so the
encoding error is at most about half a step before any later clamp.

`EXP_INPUT_SCALE=32` is not a proof that every attention score delta keeps
enough precision. A value such as `scaled_score_real=-0.701` still produces a
non-integer product:

```text
scaled_score_real * 32 = -22.432
scaled_score_int = round(-22.432) = -22
scaled_score_real_approx = -22 / 32 = -0.6875
```

Very small real-score differences can therefore collapse to the same integer
code. For example, values whose encoded products round to `-22` all use the
same EXP table entry. This is the normal fixed-point quantization tradeoff:
larger `EXP_INPUT_SCALE` gives finer real-input spacing but also increases the
number of LUT entries for the same real range, or reduces the covered real
range for a fixed entry count.

The selected scale is acceptable only if golden tests and workload error
analysis show that probability ordering, row sums, and downstream `P*V` error
remain within the reviewed tolerance. If `1/32` spacing is too coarse, the
architecture must change `EXP_INPUT_SCALE`, the clamp range, or use
interpolation/piecewise approximation, then regenerate RTL tables, golden data,
fixtures, and metadata together.

The hardware/compiler path does not need to materialize a floating-point
`scaled_score_real`. It can combine the attention scale and EXP input scale into
one fixed-point requantization:

```text
scaled_score_int ~= round(score * EXP_INPUT_SCALE / sqrt(D))
```

An implementation should realize that expression with a reviewed integer
multiplier, rounding offset, and shift, for example:

```text
scaled_score_int = round(score * SCALE_MUL / 2^SCALE_SHIFT)
SCALE_MUL ~= round((EXP_INPUT_SCALE / sqrt(D)) * 2^SCALE_SHIFT)
```

The chosen `SCALE_MUL`, `SCALE_SHIFT`, rounding mode, and clamp behavior are
part of the attention numerical contract and must be generated or recorded with
the workload metadata.

The table is not indexed by the unencoded real value `scaled_score_real` because
that value is not a finite hardware address. The multiply-by-scale step chooses
a discrete grid for the real EXP input:

```text
scaled_score_int = round(scaled_score_real * EXP_INPUT_SCALE)
scaled_score_real_approx = scaled_score_int / EXP_INPUT_SCALE
```

The EXP table then stores the exponential for each grid point. For example, if
`scaled_score_real=-0.7`, then with `EXP_INPUT_SCALE=32`:

```text
scaled_score_int = round(-0.7 * 32) = -22
table index = -22 + 256
table entry = round(exp(-22 / 32) * 32767)
```

This approximates `exp(-0.7)` with `exp(-0.6875)`. A table that was indexed
directly by raw matrix scores or unscaled score deltas would be much larger and
would depend on head dimension, quantization scales, and the chosen
`1/sqrt(D)` requantization. Keeping the EXP LUT indexed by a fixed-point EXP
input code separates the upstream score conversion from the reusable bounded
EXP primitive.

The row max and delta must be computed in this same encoded integer domain:

```text
row_max_int = max_j scaled_score_int[j]
delta_int[j] = scaled_score_int[j] - row_max_int
```

Ignoring rounding for clarity:

```text
delta_int[j] / EXP_INPUT_SCALE
  = (scaled_score_real[j] * EXP_INPUT_SCALE
     - row_max_real * EXP_INPUT_SCALE) / EXP_INPUT_SCALE
  = scaled_score_real[j] - row_max_real
```

Therefore the SFU computes an approximation of:

```text
exp(scaled_score_real[j] - row_max_real)
```

not `exp(scaled_score_real[j] / EXP_INPUT_SCALE)`. This is equivalent to
`exp(scaled_score_real[j])` for softmax normalization because the shared
`exp(-row_max_real)` factor cancels in the row sum:

```text
exp(score[j] - row_max) / sum_t exp(score[t] - row_max)
  = exp(score[j]) / sum_t exp(score[t])
```

The remaining differences from real-valued softmax are intentional fixed-point
effects: rounding when `scaled_score_real` is encoded to integer units and
clamping deltas below `-8.0` to the `-256` table endpoint.

For example, if `D=1024`, then `sqrt(D)=32`, but this numerical coincidence
does not mean `EXP_INPUT_SCALE` came from `sqrt(D)`. If a later workload uses
`D=2048`, the upstream `1/sqrt(D)` multiplier changes, while the SFU EXP LUT
can remain at `EXP_INPUT_SCALE=32` as long as the requant stage still emits
values in `1/32` EXP-input units. Changing the EXP input unit itself is a
separate spec/config change and requires regenerating the LUT plus golden data.

`EXP_INPUT_SCALE=32` is also a sampling choice for the LUT. One integer step
corresponds to `1/32` in real EXP input space, so adjacent table entries differ
by `exp(1/32)` rather than by `exp(1)`. This avoids a table where neighboring
integer deltas are too far apart, while keeping the bounded `[-8.0, 0.0]`
domain small enough for 257 entries.

The RTL lookup does not compute `x_real` or divide by `32` at runtime. The table
is generated ahead of time from the mathematical definition:

```text
table[index] = round(exp((index - 256) / 32) * 32767)
```

Equivalently, generation follows this pseudo-code:

```text
for x_int in [-256, -255, ..., 0]:
    index = x_int + 256
    table[index] = round(exp(x_int / EXP_INPUT_SCALE) * 32767)
```

At runtime the SFU uses the clamped integer input as the table address:

```text
x_clamped = clamp(x_int, -256, 0)
index = x_clamped + 256
y_q15 = table[index]
```

For example, runtime input `x_int=-32` addresses `table[224]`. That table entry
was generated from `exp(-32 / 32) = exp(-1.0)`, so no divider is needed in the
EXP datapath.

Changing `EXP_INPUT_SCALE` changes the softmax temperature/numerical contract
unless score scaling, fixtures, golden data, and metadata are updated together.

### 中文说明：EXP 输入缩放与查表

这里的 SFU `EXP` 不是通用 `exp()`，而是为 attention softmax 服务的有界
指数单元。`DATA_WIDTH=32` 表示接口上传输的是 32-bit 整数；真正用于 EXP
查表的数学输入域，是 clamp 之后的 `[-256, 0]`。

attention softmax 前面的 score 路径是：

```text
score = Q * K^T
scaled_score ~= score / sqrt(D)
delta = masked_scaled_score - row_max
```

`EXP_INPUT_SCALE=32` 不是 `sqrt(D)`，也不是说系统默认 `D=1024`。`sqrt(D)`
属于 attention score scaling；`EXP_INPUT_SCALE` 属于 SFU EXP 输入的定点
编码单位。对于不同的 `D`，上游 `1/sqrt(D)` 的 requant 参数会变；只要上游
仍然把结果编码成 `1/32` 为单位的整数，SFU EXP LUT 可以不变。

如果把除以 `sqrt(D)` 之后的实数 score 记为 `A`，那么上游需要把它量化成
SFU 能接收的整数编码：

```text
B = round(A * EXP_INPUT_SCALE)
```

这不是无损转换。`A * 32` 仍然可能是小数，所以必须 round 到最近的整数格点。
例如：

```text
A = -0.701
B = round(-0.701 * 32) = -22
B / 32 = -0.6875
```

因此 `EXP_INPUT_SCALE=32` 只是选择了 `1/32` 的 real 输入间隔。间隔越小，
LUT 越精细，但同样 `[-8.0, 0.0]` 范围下表项越多。`1/32` 是否够用必须由
golden test、概率排序、row sum 误差和下游 `P*V` 误差来验证。

运行时 RTL 不会真的计算 `B / 32`，也不会用浮点数查表。LUT 在生成阶段已经
把 `/32` 折进表项：

```text
table[index] = round(exp((index - 256) / 32) * 32767)
```

运行时只做整数 clamp 和地址计算：

```text
x_clamped = clamp(B1, -256, 0)
index = x_clamped + 256
y_q15 = table[index]
```

其中 `B1 = B - row_max_int`。忽略 round 误差时：

```text
B1 / 32 = A - max(A)
```

所以 SFU 实际近似计算的是：

```text
exp(A - max(A))
```

这和直接用 `exp(A)` 做 softmax 的最终概率等价，因为 `exp(-max(A))` 是整行
共享因子，会在归一化分母中抵消。剩下的误差来自 fixed-point round 和
`[-8.0, 0.0]` clamp。

### Input

EXP input is signed integer `x` with `DATA_WIDTH` bits.

For attention softmax:

```text
x = clamp(masked_scaled_score - row_max, -256, 0)
real_input = x / EXP_INPUT_SCALE
```

Initial config:

```text
EXP_INPUT_SCALE = 32
EXP clamp range = [-256, 0]
real input range = [-8.0, 0.0]
```

Inputs above zero saturate to zero. Inputs below `-256` saturate to `-256`.

### Output

EXP output is unsigned Q0.15 in the low 16 bits of the `DATA_WIDTH` output:

```text
e_q15 = round(exp(real_input) * (2^15 - 1))
```

The output is zero-extended to `DATA_WIDTH`.

### 257-entry LUT

Target LUT has one entry per integer input value from `-256` through `0`
inclusive:

```text
index = x_clamped + 256
lut[0]   = exp(-256 / 32) in Q0.15
lut[256] = exp(0) in Q0.15
```

The table must be generated from a deterministic numerical source, not hand
typed. The generator should be owned by the Transformer config/tool path and
should produce both:

- an RTL include/package table;
- Python golden data or a golden helper using the same formula.

Target EXP operation:

```text
x_clamped = clamp(x, -8 * EXP_INPUT_SCALE, 0)
index = x_clamped + 8 * EXP_INPUT_SCALE
y_real = exp(x_clamped / EXP_INPUT_SCALE)
y_q15 = saturate_u16(round(y_real * ((1 << EXP_OUTPUT_Q) - 1)))
```

With `EXP_INPUT_SCALE=32`, `index` spans `0..256`.

Table ordering:

- `table[0]` corresponds to `x=-256`, real `-8.0`.
- `table[256]` corresponds to `x=0`, real `0.0`.

Lookup:

```text
y_q15 = table[index]
```

No interpolation is part of the v1 target. Interpolation may be considered in a
later revision only after PPA/accuracy evidence.

Table generation uses round-to-nearest via:

```text
scale = (1 << EXP_OUTPUT_Q) - 1
raw = exp(x / EXP_INPUT_SCALE) * scale
rounded = floor(raw + 0.5)
y = min(max(rounded, 0), scale)
```

The runtime lookup itself has no rounding. Runtime clamp occurs before index
calculation.

Implementation plan:

1. Add a generator that reads `EXP_INPUT_SCALE`, clamp range, and output Q from
   `arch/configs/npu_transformer_v1.jsonc`.
2. Emit a 257-entry table for inputs `[-256, 0]`.
3. Replace the current segment `case` in `sfu_lut.sv` with indexed table lookup:

   ```text
   x_clamped = clamp(x, -256, 0)
   index = x_clamped + 256
   y = exp_lut_q15[index]
   ```

4. Keep the old 9-segment RTL model function in Python only as a named
   compatibility model until all tests migrate.
5. Update perf/PPA provenance from bring-up SFU to target SFU only after RTL
   and golden agree.

Migration plan:

1. Add table generation and fixed-spec model tests.
2. Keep current RTL unchanged until table generation is reviewed.
3. Replace the 9-segment lookup in `sfu_lut.sv` with 257-entry lookup.
4. Update `hw/npu_core/rtl/sfu/README.md` from "9-segment bring-up" to
   "257-entry implemented".
5. Keep RECIP/RSQRT documented as bring-up until their own implementation
   strategy is reviewed.

### Why The 9-Segment LUT Is Not Enough

The current 9-segment LUT maps broad score deltas to only nine EXP values. For
operator smoke tests this is acceptable because it proves SFU wiring, but for
attention it hides architectural tradeoffs:

- softmax probability error can dominate PV output error;
- many score rows with different distributions collapse to the same EXP
  pattern;
- PPA comparisons involving SFU accuracy versus cost cannot be defended;
- workload reports would treat an approximation artifact as an architecture
  result.

Therefore attention PPA may use the current SFU only if reports label softmax
as bring-up/model-only. Measured attention softmax requires either the 257-entry
LUT or another reviewed SFU numerical target.

## RECIP Target

### Input

For attention softmax:

```text
s_q15 = sum_j e_q15[j]
```

For row length `L`, range is:

```text
1 <= s_q15 <= L * 32767
```

The lower bound assumes at least one valid softmax element. If a row has zero
valid elements, the softmax kernel should mark the row invalid before RECIP; SFU
zero-input behavior remains defined for robustness.

### Output

RECIP output is unsigned Q0.24:

```text
r_q24 = floor((1 << 24) / s_q15)
```

Current truncation is acceptable for bring-up. If attention accuracy requires
rounding, the rounding policy must be added to this document and golden tests
before RTL changes.

Implementation plan:

1. Keep the first reviewed RTL behavior simple and deterministic:

   ```text
   if s_q15 == 0:
       r_q24 = 0
   else:
       r_q24 = floor((1 << 24) / s_q15)
   ```

2. Use this as the fixed-spec reciprocal for `attention_softmax_s8` until PPA
   or accuracy evidence requires a different implementation.
3. If a single-cycle divider is not acceptable for synthesis/PPA, replace it
   with a reviewed LUT/Newton or piecewise approximation. That replacement must
   preserve the same external Q0.24 contract or update compiler/golden/tests
   together.

Future implementation options:

| Option | Role | Tradeoff |
| --- | --- | --- |
| direct integer divider | simplest golden/RTL match | area/timing may be poor |
| reciprocal LUT | predictable latency | table size and interpolation error |
| LUT seed + Newton step | better accuracy/area tradeoff | multi-cycle control and more tests |
| model-only reciprocal | useful before RTL | cannot be reported as measured SFU |

### Normalization Coupling

Vector normalization consumes:

```text
p_q15 = requant(e_q15 * r_q24)
```

With this document's convention:

```text
e_q15 ~= exp(delta) * 32767
r_q24 ~= 2^24 / sum(e_q15)
raw = e_q15 * r_q24
```

The v1 recommended normalization target is:

```text
p_q15 = clamp(round(e_q15 * r_q24 * PROB_ONE / 2^24), 0, PROB_ONE)
PROB_ONE = 32767
```

This formula exists because `r_q24` approximates `1 / sum(e_q15)` while the
output probability is also stored on a `0..32767` scale:

```text
p_q15 ~= (e_q15 / sum(e_q15)) * 32767
```

The vector/requant path therefore needs enough intermediate width for:

```text
e_q15 * r_q24 * PROB_ONE
```

or an algebraically equivalent staged implementation with reviewed rounding.
This is one reason current `VEC_MUL` is not enough for attention softmax
normalization.

Alternative conventions are possible:

1. Treat Q0.15 values as scaled by `32767`, then normalize explicitly to
   `PROB_ONE = 32767`. This is the v1 recommendation above.
2. Treat Q0.15 values as scaled by `2^15`, then use a pure binary shift path.

Changing from the v1 recommendation to the binary `2^15` convention is a spec
change because it affects golden results, vector normalization, PV input scale,
and PPA accuracy interpretation.

Current `micro_golden.py` uses a bring-up approximation. Attention v1 must
replace it with named functions:

- `softmax_attention_fixed_spec_*`;
- `softmax_attention_rtl_model_*`.

The names must make it clear whether a test is using target fixed-spec math or
the current RTL approximation.

## RSQRT Target

`RSQRT` is not required for attention softmax. It is retained for RMSNorm:

```text
y = x * rsqrt(sum(x^2) / N + eps)
```

Current `RSQRT` is a bring-up integer approximation. It should not block
attention work, but its documentation and tests must remain separated from
attention softmax evidence.

## Interface Contract

Current RTL:

```text
start
op
x
done
active
y
```

Production scheduler target:

```text
cmd_valid
cmd_ready
cmd_op
cmd_x
rsp_valid
rsp_ready
rsp_y
```

Handshake semantics are defined in `primitive_valid_ready_v1.md`.

Attention requires the valid/ready form before full measured softmax because
the row kernel issues multiple EXP operations and one RECIP operation per row.
Without back-pressure and response stability rules, scheduler integration cannot
correctly count stalls or handle a multi-cycle SFU.

## Latency And Throughput

Current bring-up latency:

| Op | Current latency | Initiation interval |
| --- | --- | --- |
| EXP 9-segment | 1 cycle | 1 cycle in standalone start/done test |
| RECIP integer divide expression | modeled as 1 cycle in RTL | 1 cycle in standalone start/done test |
| RSQRT integer loop expression | modeled as 1 cycle in RTL | 1 cycle in standalone start/done test |

Target attention latency must be reviewed before implementation:

| Op | Target expectation |
| --- | --- |
| EXP 257 LUT | 1 cycle if combinational table is acceptable, 2 cycles if table output is registered |
| RECIP | implementation-dependent; may need LUT/Newton or multi-cycle divider |
| RSQRT | deferred unless RMSNorm becomes measured workload gate |

Any latency change must update:

- this document;
- primitive valid/ready tests;
- row-softmax golden/RTL sequence tests;
- perf counter expectations.

## Counters

Required attention SFU counters:

| Counter | Increment condition |
| --- | --- |
| `sfu_active_cycles` | SFU has accepted work not yet retired, or same-cycle accept/retire |
| `sfu_input_stall_cycles` | `cmd_valid && !cmd_ready` |
| `sfu_output_stall_cycles` | `rsp_valid && !rsp_ready` |
| `sfu_exp_ops` | accepted EXP command |
| `sfu_recip_ops` | accepted RECIP command |
| `sfu_rsqrt_ops` | accepted RSQRT command |

The first PPA report may expose only `sfu_active_cycles` and op counts. It must
label unavailable stall counters as unavailable rather than zero.

## Verification Plan

### Unit vectors

EXP target vectors must include:

```text
x = 0      -> Q0.15 exp(0)
x = -1     -> near exp(-1/32)
x = -32    -> near exp(-1)
x = -64    -> near exp(-2)
x = -128   -> near exp(-4)
x = -256   -> near exp(-8)
x < -256   -> same as -256
x > 0      -> same as 0
```

RECIP vectors must include:

```text
x = 0
x = 1
x = 32767
x = 8 * 32767
```

### Row softmax vectors

At least these rows are required:

```text
uniform:      [0, 0, 0, 0, 0, 0, 0, 0]
one-hot-ish:  [64, 0, -64, -128, -256, -512, -1, -32]
monotonic:    [0, -32, -64, -96, -128, -160, -192, -224]
masked tail:  valid first N lanes, invalid remaining lanes
```

For each row, tests must check:

- probability ordering;
- sum close to Q0.15 one, within reviewed tolerance;
- invalid lanes produce zero probability;
- Python target fixed-spec and RTL model are named separately.

### Integration tests

- Existing standalone primitive tests keep passing.
- New SFU LUT table generation is deterministic.
- EXP table tests cover entry count, endpoints, clamp behavior, monotonicity,
  and selected indices `0`, `1`, `32`, `64`, `128`, `192`, `255`, `256`.
- Row-softmax primitive sequence matches fixed-point golden.
- Perf/PPA reports include SFU provenance and counters only after scheduler
  integration exists.

## Acceptance Criteria

Before SFU can support measured attention softmax:

- 257-entry EXP LUT or reviewed alternative is implemented.
- Python golden and RTL use the same EXP target.
- RECIP input/output Q formats and normalization shift are documented.
- valid/ready or an equivalent reviewed scheduler contract exists.
- SFU active/op counters are visible to perf/PPA.
- `attention_softmax_s8` passes row-level golden tests.

## Known Gaps

- Current RTL is standalone bring-up only.
- EXP target table is not implemented.
- RECIP/RSQRT production implementation is not reviewed.
- No scheduler issue path exists.
- No SFU counters are exposed through wrapper perf CSR.
