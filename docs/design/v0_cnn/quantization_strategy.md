# Quantization Strategy

[TOC]

This document defines the current quantization policy for mapping selected
float32 model layers onto the Phase 0 signed-int8 NPU datapath.

## 1. Scope

The current source models remain float32 models. Quantization is a
hardware-facing view used only at the selected layer boundary being mapped to
the NPU.

For the real MNIST CNN, the active boundaries are:

```text
fc1:
  float pool/flat activation + float fc1.weight
  -> int8 matmul view
  -> int32 accumulator
  -> dequantized float bias/ReLU

fc2:
  float or approximated fc1_relu activation + float fc2.weight
  -> int8 matmul view
  -> int32 accumulator
  -> dequantized float bias/logits
```

The model graph and tensor/parameter shapes are recorded in:

```text
test/graphs/real_mnist_cnn.json
```

The code must derive feature counts, class counts, and padded NPU matrix shapes
from this graph and the checked safetensors weight metadata, not from unrelated
hard-coded constants.

## 2. Phase 0 Policy

Use symmetric signed-int8 quantization for both activations and weights:

```text
scale = 127 / max(abs(x))
q     = clamp(round(x * scale), -128, 127)
```

Current granularity:

| Tensor | Granularity | Reason |
| --- | --- | --- |
| activation | per-tensor, per sample | simple boundary validation before calibration data exists |
| weight | per-tensor, per layer | matches the current single-scale test and firmware path |
| accumulator | int32 | matches current RTL matmul accumulation |
| bias | dequantized float for tool tests; scaled int32 for existing firmware smoke where needed | avoids changing RTL before numerical behavior is understood |

The dequantized matmul value is:

```text
float_acc = int32_acc / (activation_scale * weight_scale)
```

For `fc1`, ReLU is applied after dequantization and bias addition. The resulting
`fc1_relu` vector is then requantized as the activation input to the existing
`fc2` mapping.

## 3. Why Not Asymmetric Yet

Asymmetric quantization is not just a different scale choice. It introduces
zero-points:

```text
sum((qa - za) * (qw - zw))
```

which expands into correction terms:

```text
sum(qa * qw)
- zw * sum(qa)
- za * sum(qw)
+ K * za * zw
```

The Phase 0 RTL, assembler, descriptor ABI, and firmware path currently expose
only signed-int8 multiply-accumulate. They do not carry activation zero-points,
weight zero-points, row sums, column sums, or correction instructions. Adding
asymmetric quantization is therefore a hardware/software contract change, not a
local test change.

## 4. Upgrade Path

The next quantization improvements should be evaluated in this order:

1. Keep symmetric activation quantization but change weights to per-output-channel
   symmetric scales.
2. Replace per-sample activation scales with calibration-derived fixed scales so
   firmware does not need dynamic max-abs scanning.
3. Move bias into the int32 accumulator domain where the scale is fixed enough
   for firmware/RTL contracts.
4. Add asymmetric quantization only after the NPU contract supports zero-point
   correction metadata and arithmetic.

Each change must state the affected workload, expected accuracy benefit,
firmware/RTL contract impact, and verification commands.

## 5. Verification Plan

Quantization changes must be verified at three levels before they are treated
as part of the active hardware/software contract.

### Tool-Level Numerical Checks

The first gate validates the numerical boundary against the original float32
model:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_real_mnist_cnn -v
```

This covers:

- source graph fixture matches `real_mnist_cnn_graph()`;
- graph op attributes, tensor shapes, and safetensors parameter shapes are
  internally consistent;
- original float CNN predicts the expected labels for the checked MNIST smoke
  samples;
- `fc2` symmetric-int8 view preserves the original model class prediction;
- `fc1 -> fc2` tool flow preserves the class prediction after:
  - quantizing `flat` and `fc1.weight`;
  - running logical `8x9216 * 9216x128` micro-op matmul;
  - adding `fc1.bias`;
  - applying ReLU;
  - requantizing into the existing `fc2` mapping.

### SoC RTL Checks

The SoC RTL gate verifies that selected quantized layer tiles use the same data
layout and arithmetic once staged by firmware:

```text
make cpu-soc-sim
make perf-report
```

Current SoC RTL coverage:

- `fc2` full current-RTL-compatible tiled path for MNIST sample 0.
- `fc1` first hardware-facing tile for MNIST sample 0: this validates real
  `fc1` quantized activation/weight tile staging, descriptor launch, RTL
  matmul, output writeback, and firmware-side comparison against the tool
  expected tile.
- `fc1` K-streaming smoke for MNIST sample 0: this validates that multiple real
  `fc1` K chunks can be streamed within one descriptor and accumulated in the
  NPU core before one output writeback.

The current `fc1` SoC tile is intentionally not a full-layer claim. Full `fc1`
needs NPU-side K streaming and accumulator residency before it should replace
the tool-level layer check.

### Regression Gates

For any quantization policy change, run:

```text
make test
make perf-report
```

If timing, job counts, or workload grouping change, update
`docs/design/performance_instrumentation.md`,
`docs/design/verification_strategy.md`, and
`docs/collaboration_journal.md`.
