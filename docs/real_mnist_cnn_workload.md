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
