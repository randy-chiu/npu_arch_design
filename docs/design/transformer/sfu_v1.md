# SFU v1 Design

## 1. Target / 目标

SFU v1 provides deterministic fixed-point approximations needed by Transformer
micro-kernels. It does not implement floating-point exp/div/sqrt.

Required v1 functions:

- `SFU_EXP` for softmax;
- `SFU_RECIP` for softmax normalization;
- `SFU_RSQRT` for RMSNorm.

## 2. Overall Design / 整体设计思路

The first SFU should be LUT-oriented and deterministic:

```text
uop scheduler
  -> scalar/vector SFU input
  -> LUT or small approximation stage
  -> fixed-point result
```

The first module should live under:

```text
hw/npu_core/rtl/sfu/
```

V1 prefers simple LUTs over iterative math unless PPA evidence later justifies
more complex approximations.

## 3. Key Details / 重点细节

Softmax EXP:

| Field | v1 value |
| --- | --- |
| input | signed fixed-point delta |
| input scale | 32 |
| clamp range | `[-256, 0]`, representing `[-8, 0]` |
| LUT entries | 257 |
| output | uint16 Q0.15 |

RECIP:

| Field | v1 value |
| --- | --- |
| input | uint32 sum |
| output | uint16 or uint24 fixed-point reciprocal |
| zero behavior | return zero and flag invalid input if exposed later |

RSQRT:

| Field | v1 value |
| --- | --- |
| input | fixed-point mean square plus epsilon |
| output | fixed-point reciprocal square root |
| use case | RMSNorm row |

Counter semantics:

| Counter | Meaning |
| --- | --- |
| `sfu_active_cycles` | SFU accepts input or produces approximation progress |
| `sfu_stall_cycles` | SFU op assigned but input/result path unavailable |
| `sfu_idle_cycles` | no SFU op assigned |

## 4. Verification / 验证测试

Initial tests:

- EXP LUT endpoints: `0`, `-1`, `-8`, below clamp range;
- monotonic EXP output over clamped range;
- RECIP simple denominators and zero behavior;
- RSQRT simple positive inputs;
- softmax row golden comparison through Python Q15 reference.

Golden source:

```text
sw/tools/transformer/micro_golden.py
arch/specs/transformer/v1/transformer_numerical_v1.md
```

## 5. Implementation Priority / 实现优先级

1. Add standalone EXP LUT module or combined SFU module. Status: implemented
   as `hw/npu_core/rtl/sfu/sfu_lut.sv`.
2. Add RECIP and RSQRT approximations with documented fixed-point formats.
   Status: implemented as simple deterministic approximations for standalone
   validation.
3. Connect to reduction/vector micro-kernel fixtures. Status: implemented in
   the standalone primitive testbench; scheduler integration is still pending.
4. Add measured SFU cycles only when scheduler integration exists. Status:
   pending.
