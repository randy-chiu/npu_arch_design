# Digits Classifier Workload

[TOC]

This document captures the next workload step before further NPU core
optimization. The goal is to move from operator smoke tests to a small but
complete inference task.

## Goal

Run a complete handwritten-digit classification workload:

```text
8x8 grayscale image -> int8 quantization -> linear classifier on NPU
-> logits -> CPU/tool argmax -> digit
```

The first version is intentionally small. It should prove that graph fixtures,
compiler lowering, simulator execution, firmware data generation, and later SoC
execution can carry a real model-style workload from image input to predicted
class.

## Why This Comes Before More Core Optimization

The current closed loop verifies:

- `matmul`
- `softmax`
- `matmul -> softmax`

That is enough for operator bring-up, but it does not exercise realistic model
concerns:

- image-like input layout;
- checked-in model weights;
- class logits and label validation;
- multi-tile or padded tensor shapes;
- runtime ownership of final post-processing such as `argmax`;
- performance reporting over a model-level workload.

A small classifier should expose these system contracts before we add more core
complexity.

## First Implementation Shape

Keep the first workload compatible with the current Phase 0 ISA:

```text
A: 8 x 64   int8 activations
B: 64 x 16  int8 classifier weights
C: 8 x 16   int32 logits
```

Only `A[0]` contains the flattened 8x8 image. Rows `A[1:7]` are zero padding.
Only logits `C[0][0:10]` are real digit classes. Columns `10:15` are padding.

This shape is legal for the current no-edge-tile rule because it is a multiple
of the Phase 0 `8x8x8` matmul tile.

The first NPU-visible graph can remain:

```text
logits = matmul(image_batch, classifier_weights)
```

Graph JSON:

```json
{
  "tensors": {
    "A": {"shape": [8, 64], "dtype": "int8"},
    "W": {"shape": [64, 16], "dtype": "int8"}
  },
  "ops": [
    {"type": "matmul", "a": "A", "b": "W", "out": "Logits"}
  ]
}
```

`argmax(logits[0][0:10])` is CPU/tool-side post-processing in the first
version. A later architecture change can add NPU `argmax` or `topk` if the
workload justifies it.

Softmax is not required for classification correctness here. `argmax(logits)`
and `argmax(softmax(logits))` pick the same class, so this workload uses only
`matmul` on the NPU-visible path and leaves final `argmax` to CPU/tool code.

## Graph Lowering

There are two lowering views:

1. Logical classifier graph:

   ```text
   A[8x64] x W[64x16] -> Logits[8x16]
   ```

   Compiler output:

   ```text
   LOAD A -> spad_a
   LOAD W -> spad_b
   MATMUL shape 8x16x64 -> acc_Logits
   STORE acc_Logits -> Logits
   HALT
   ```

   This is valid in the Python micro-op simulator, but it is larger than the
   current RTL core windows.

2. RTL-compatible tiled lowering:

   ```text
   for n in {0, 8}:
     for k in {0, 8, 16, 24, 32, 40, 48, 56}:
       C_partial[8x8] = matmul(A[:, k:k+8], W[k:k+8, n:n+8])
       Logits[:, n:n+8] += C_partial
   ```

   This creates 16 independent `8x8x8` matmul tile jobs. Each tile job uses the
   same graph shape the current RTL matmul path already supports:

   ```json
   {
     "tensors": {
       "A": {"shape": [8, 8], "dtype": "int8"},
       "B": {"shape": [8, 8], "dtype": "int8"}
     },
     "ops": [
       {"type": "matmul", "a": "A", "b": "B", "out": "C"}
     ]
   }
   ```

The tiled path is the intended bridge to firmware/SoC execution.

## Tiny MLP Step

The second workload step introduces a more realistic multi-layer graph while
still using only matmul on the NPU-visible path:

```text
image[8x8]
-> quantize/flatten
-> FC1: matmul A[8x64] * W1[64x16] -> Hidden[8x16]
-> CPU/tool relu_requantize Hidden -> HiddenInt8[8x16]
-> FC2: matmul HiddenInt8[8x16] * W2[16x16] -> Logits[8x16]
-> CPU/tool argmax Logits[0][0:10]
```

Graph JSON lives in:

```text
test/graphs/digits_tiny_mlp.json
```

This graph deliberately marks placement:

- FC1 and FC2 are `placement: npu`;
- `relu_requantize` and `argmax` are `placement: cpu`.

Both FC matmuls are lowered into current-RTL-compatible `8x8x8` tile jobs in
the tool tests. This is the preferred bridge before adding activation or top-k
instructions to the NPU ISA.

## Scope For V1

In scope:

- deterministic checked-in PGM digit images and classifier weights;
- golden classifier reference;
- graph/input fixture generation;
- compiler and micro-op simulator tests that predict expected labels;
- RTL-compatible tile-job lowering in tools;
- CPU-controlled SoC execution of the linear classifier tiled workload;
- Tiny MLP graph and tiled tool execution;
- documentation of the current RTL/firmware gap.

Out of scope for V1:

- CNN layers;
- online training;
- downloading datasets during tests;
- new NPU ISA operations;
- RTL support for `8x64 * 64x16` storage and tiled execution;
- firmware execution of the Tiny MLP.

## Implemented V1 State

Implemented files:

- `sw/tools/npu_phase0/digits_classifier.py`: deterministic weights, PGM image
  loading, graph construction, reference logits, tiled lowering, and `argmax`
  prediction.
- `test/graphs/digits_classifier.json`: checked-in graph shape for the
  classifier workload.
- `test/graphs/digits_classifier_rtl_tile.json`: one RTL-compatible tile job.
- `test/graphs/digits_tiny_mlp.json`: FC1/ReLU/FC2/argmax graph with CPU/NPU
  placement.
- `test/assets/digits/digit_*.pgm`: checked-in 8x8 grayscale digit images.
- `test/inputs/digits_classifier_samples.json`: image paths, visible 8x8
  glyphs, expected 10-class logits, and expected predictions.
- `test/rtl/test_digits_classifier.py`: golden and compiler/simulator tests for
  all labels `0` through `9`.
- `npu_phase0.cli digits-demo`: command-line smoke demo for one selected label.
- `make digits-demo`: Makefile entry for the CLI smoke demo.

Current validation:

```text
make digits-demo: PASS, label 2 predicted 2
make cpu-soc-sim: PASS, includes 16 tiled classifier matmul jobs
make test: PASS
```

The emitted program for the first workload is:

```text
LOAD A
LOAD W
MATMUL shape 8x16x64
STORE Logits
HALT
```

The linear classifier now runs through CPU-controlled SoC simulation: firmware
launches 16 current-RTL-shaped `8x8x8` matmul jobs and accumulates partial sums
in SRAM before checking logits and predicted label.

The Tiny MLP currently runs through the tool/compiler/simulator tile path. Its
firmware path is intentionally deferred until the linear classifier path is
reviewed.

## Follow-Up Steps

Tomorrow's likely third step:

1. Review the linear classifier firmware path and decide whether to keep it in
   the smoke app or split it into a dedicated model app.
2. Move Tiny MLP orchestration into firmware if the linear path looks clean.
3. Start the Tiny CNN design note only after the MLP firmware path is stable:
   choose input size, decide whether convolution lowers through `im2col` into
   matmul tiles, and define which CPU-side ops are allowed before adding NPU
   ISA support.
4. Extend performance reporting from operator jobs to model-level workloads so
   the 16-job classifier and later MLP/CNN appear as grouped model runs.
