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

Generated integration constants are emitted by `make transformer-config` into
`build/generated/npu_transformer_v1_config_pkg.sv`.

## Attention Softmax Derivation

The fixed-point background and worked examples for EXP scale, Q0.15 output, and
Q0.24 reciprocal are in
`docs/design/transformer/attention_numerical_v1.md`. This section defines the
SFU-specific contract.

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

## EXP Target

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
