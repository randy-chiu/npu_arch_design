# Real MNIST CNN Workload

[TOC]

This workload replaces the temporary MNIST prototype with an open-source
pretrained CNN and real MNIST test images.

## Source

Model source:

```text
https://huggingface.co/cmaeti/mnist-cnn
```

The model card states:

- license: Apache-2.0;
- framework: PyTorch;
- architecture: Conv1 32x3x3, Conv2 64x3x3, maxpool, FC1 128, FC2 10;
- dataset: MNIST;
- training accuracy: 99%.

The local external fixtures are:

```text
test/external/mnist_cnn/mnist-cnn.safetensors
test/external/mnist_cnn/README.md
test/external/mnist/t10k-images-idx3-ubyte.gz
test/external/mnist/t10k-labels-idx1-ubyte.gz
```

`test/external/` is ignored by git. The tests skip this workload if the
external files are absent.

## Graph

Checked-in graph fixture:

```text
test/graphs/real_mnist_cnn.json
```

Inference graph:

```text
28x28 grayscale image
-> scale to float32 [0, 1]
-> conv1: 1 -> 32, 3x3 valid
-> ReLU
-> conv2: 32 -> 64, 3x3 valid
-> ReLU
-> maxpool 2x2
-> flatten 64x12x12 = 9216
-> fc1: 9216 -> 128
-> ReLU
-> fc2: 128 -> 10
-> argmax
```

The current implementation is a non-RTL golden/reference flow in:

```text
sw/tools/npu_phase0/real_mnist_cnn.py
```

It includes a minimal safetensors F32 reader and a numpy forward path.

The graph and downloaded float weights are treated as the source of truth. RTL
mapping work must not rewrite the model topology or replace trained weights.
When a layer is mapped to the current int8 NPU matmul path, the test creates a
hardware-facing quantized view from the original float activation/weight
tensors and compares that view back to the original model's predicted class.

## Graph And Lowering Path

The downloaded `mnist-cnn.safetensors` file contains tensor weights only. It is
not parsed as a full neural-network program. The model topology is represented
explicitly in `real_mnist_cnn_graph()` and checked against
`test/graphs/real_mnist_cnn.json`.

Current flow:

```text
safetensors weights + MNIST IDX image
-> explicit Python CNN forward path
-> original float intermediates and logits
-> selected layer hardware-facing quantized view
-> current RTL-compatible 8x8x8 matmul tile jobs
-> compiler/assembler emits micro-ops for the fixed tile graph
```

For `fc2`, the code runs the original float model up to `fc1_relu`, quantizes
that activation and `fc2.weight`, then lowers the resulting `8x128 * 128x16`
hardware-facing view into 32 independent `8x8x8` jobs. The fixed tile graph is
compiled to micro-ops; the whole CNN graph is not yet lowered automatically end
to end.

## Quantization Boundary

The original downloaded model remains a float32 model. Quantization is only a
hardware-facing view used when a selected layer is mapped onto the current NPU
RTL.

This is necessary today because the Phase 0 matmul datapath accepts int8
activation/weight inputs and produces int32 accumulators. It does not implement
float32 multiply-accumulate. Therefore, mapping `fc1` or `fc2` to the current
NPU requires:

- deriving int8 activations from the original float intermediate tensor;
- deriving int8 weights from the original float layer weights;
- accumulating in int32 on the NPU;
- comparing the dequantized or scaled result against the original model's class
  decision.

This is not intended to replace the source model or retrain weights. The source
of truth remains the float graph and float safetensors weights. The quantized
view is a verification bridge for the current int8 RTL. If the NPU later grows
a float or mixed-precision datapath, this boundary can change.

## Current Validation

Current test:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_real_mnist_cnn -v
```

It verifies:

- graph fixture matches the tool graph;
- downloaded safetensors tensor shapes match the expected CNN;
- first 10 MNIST test images predict the expected labels;
- first 100 MNIST test images meet the smoke accuracy threshold.
- tool-level `fc2` mapping uses the original model's `fc1_relu` activation and
  `fc2.weight/bias`, builds a current-NPU-compatible int8 matmul view, lowers it
  into 32 `8x8x8` tile jobs, and checks that the tiled result preserves the
  original float model prediction for the first 10 MNIST test images;
- CPU-controlled SoC RTL smoke runs the same `fc2` hardware-facing view for
  MNIST test sample 0:
  - firmware stages the precomputed quantized `fc1_relu` activation and
    quantized `fc2.weight` tiles;
  - PicoRV32 launches 32 descriptor jobs through the NPU wrapper;
  - NPU RTL executes the 32 `8x8x8` matmul tiles;
  - firmware accumulates partial sums, adds scaled original `fc2.bias`, checks
    scaled logits, and validates expected label 7.

`make perf-report` now reports:

```text
jobs: 50
workloads: 4
real_mnist_cnn_fc2: 32 jobs, 7552 cycles
```

Next steps:

1. map `fc1: 9216 -> 128`, likely requiring broader tiling and storage support
   than current Phase 0:
   - define job count and accumulation policy before changing firmware;
   - avoid blindly emitting huge static tile arrays if a compact feature/weight
     staging scheme is needed;
   - update perf grouping so `fc1` and `fc2` appear as separate real CNN model
     layers.
2. only after `fc1/fc2` are stable, decide whether convolution should be a
   direct NPU op or lowered through `im2col -> matmul` tiles.
3. move more of the original model's runtime preprocessing into firmware when
   the data movement and SRAM footprint are understood.

## FC1 Mapping Plan

### Problem

The original CNN `fc1` layer is:

```text
flat[9216] * fc1.weight[128, 9216]^T + fc1.bias -> fc1[128]
```

The current RTL-compatible matmul tile is only `8x8x8`. If `fc1` is exposed as
one descriptor per current tile, the job count is:

```text
K tiles = 9216 / 8 = 1152
N tiles = 128 / 8 = 16
jobs    = 1152 * 16 = 18432
```

That path is not a useful hardware direction. It mostly measures descriptor
launch, program/input reload, output writeback, and CPU-side partial-sum
accumulation overhead.

### Step 1: Logical Micro-Op Simulation

First implement a CPU-side micro-op simulation for `fc1` without changing RTL:

```text
original float CNN forward to pool/flat
-> derive int8 hardware-facing flat activation
-> derive int8 hardware-facing fc1 weight matrix
-> compile logical MATMUL shape 8x128x9216
-> run MicroOpFunctionalSimulator
-> add scaled fc1.bias and ReLU
-> feed resulting fc1_relu view into the existing fc2 mapping
-> compare final predicted label with the original float model
```

This validates the numerical boundary, scale policy, accumulator range, and
compiler/simulator micro-op behavior. It does not claim current RTL can store
or execute the full `8x128x9216` matmul in one job.

### Step 2: Hardware Direction

Do not map `fc1` by forcing CPU firmware to accumulate 18432 independent tile
outputs. Instead, evolve the NPU core/wrapper contract so K-axis accumulation
stays inside the NPU side.

Preferred direction:

```text
one FC1 N-tile job
  -> load/stream A[8 x K_total] and B[K_total x 8]
  -> NPU core keeps acc[8 x 8]
  -> matmul engine consumes K in internal chunks
  -> final output tile is written once
```

With `N=128`, this makes the SoC-visible layer work closer to 16 output-tile
jobs instead of 18432 current micro-tile jobs.

This does require a larger internal K path or a streaming K loop inside the
NPU. The important boundary is that partial sums should remain in the NPU
accumulator, not in CPU firmware.

Two hardware options are under consideration:

1. Increase the core logical matmul K capacity for a resident tile, for example
   `8x8x64` or `8x8x128`, and let wrapper/data mover feed larger K chunks.
2. Keep the physical array at `8x8` output lanes, but add an internal K-stream
   loop where the core repeatedly consumes K chunks and accumulates into the
   same `acc_buf` before one final store.

Option 2 is more scalable for `K=9216`, because a fully resident
`8x8x9216` tile would require much larger A/B storage and long preload time.
Option 1 may still be useful as an intermediate step if paired with data-mover
changes and measured through `make perf-report`.

### Design Rule

Expanding MAC lanes alone is not enough. After A1, current `8x8x8` matmul jobs
are already dominated by wrapper/data movement rather than core compute.
Therefore `fc1` hardware work should combine:

- larger or streamed K handling inside the NPU core;
- accumulator residency across K chunks;
- wrapper/data-mover changes that avoid per-8-K descriptor relaunch;
- perf grouping for `real_mnist_cnn_fc1` separate from `fc2`.
