# Matrix Mixed Precision V1

## Purpose

Attention PV should reuse the shared matrix/MAC mechanism instead of adding a
temporary PV-only matrix. The requirement is:

```text
P = softmax(scores)      unsigned Q0.15 probability
V = value matrix         signed int8
O = P * V               signed int32 output for the current bring-up
```

For one output element:

```text
O_real[i,j] = sum_k P_real[i,k] * V_int8[k,j]
P_real     = P_q15 / 32768

acc_q15[i,j] = sum_k P_q15[i,k] * V_int8[k,j]
O_int[i,j]   = acc_q15[i,j] >>> 15
```

The first implementation truncates with arithmetic shift. A later target
requant revision may add rounding and int8 activation output.

## Existing Module Reuse Decision

Do not build a standalone attention PV RTL macro. Extend the current matrix
array so the same module supports two operand modes:

| Mode | A operand | B operand | Accumulate | Output |
| --- | --- | --- | --- | --- |
| `MATMUL_S8S8` | signed int8 | signed int8 | int32 | int32 |
| `MATMUL_U16S8_Q15` | unsigned Q0.15 | signed int8 | int32 Q15-scaled | int32 shifted by 15 |

This keeps PV as a matrix capability and makes mixed precision reusable by
future attention workloads.

## Interface Changes

`matmul_array.sv` gains:

```text
input mixed_u16s8_q15
input [(M*K*16)-1:0] a_flat
input [(K*N*8)-1:0]  b_flat
```

In normal `s8s8` mode, `a_flat` uses only each lane's low 8 bits and sign
extends them. In mixed mode, `a_flat` uses all 16 bits as unsigned Q0.15.

`npu_v0_top.sv` stores A as 16-bit lanes so the same A host window can carry
either int8 activations or Q0.15 probabilities. Existing int8 workloads keep
writing low 8 bits; mixed PV writes 16-bit values.

`soc_v0.jsonc` adds a descriptor op type:

```text
matmul_u16s8_q15
```

The wrapper routes this op through the same matrix fetch/execute/writeback path
as current matmul, but drives the core matrix mode to mixed precision.

## Fixed-Point Example

For:

```text
P_q15 = [24576, 8192]    // [0.75, 0.25]
V     = [20, -12]
```

Then:

```text
acc_q15 = 24576 * 20 + 8192 * (-12)
        = 393216
O_int   = 393216 >>> 15
        = 12
```

The output is an integer value in the V domain. It is not an int8 activation
until a separate output requant rule is applied.

## Verification

Required tests:

- Python golden for `attention_pv_q15_i8_i32`.
- Matrix RTL test for direct mixed `u16s8_q15` behavior.
- CPU-to-NPU transformer workload `transformer_attention_pv_s8_d8` using
  `SOC_NPU_JOB_OP_MATMUL_U16S8_Q15`.
- PPA report labels PV provenance as measured mixed matrix path and no longer
  as int8 proxy.

## PPA Impact

Mixed mode changes the matrix datapath cost model because A-side multiplier
width grows from 8 to 16 bits. L0 proxy reporting must keep the measured cycles
separate from true synthesized area/power. Until the area model is upgraded,
reports should state that mixed-mode area/energy uses the existing generic MAC
proxy and underestimates the real `16x8` multiplier cost.

## Limitations

- Only `K=8` single-tile PV is accepted in the initial SoC workload.
- Output is int32 after `>>> 15`; no rounded requant or saturation is applied.
- The softmax output is still staged through generated fixtures, not yet wired
  as an in-RTL buffer from the preceding softmax stage.
- Full attention parent remains model-only until QK, softmax, and mixed PV are
  launched as one grouped command sequence.
