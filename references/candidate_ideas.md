# Candidate Architecture Ideas

These ideas are extracted from public architecture references. They are not
accepted requirements until they are converted into hardware spec changes.

## Near-Term Candidates

### Explicit DMA And Scratchpad

Use software-managed data movement for Phase 0/1. This matches the current
minimal compiler and keeps verification deterministic.

Potential spec fields:

- DMA alignment.
- Max burst bytes.
- Scratchpad bank count.
- Supported transfer ranks.

### Output-Stationary Matmul

Keep partial sums in the accumulator buffer while streaming `A` and `B` tiles.
This is easier to verify than a full systolic timing model and still exposes
the core data-reuse problem.

Potential spec fields:

- `compute.dataflow = "output_stationary"`
- accumulator capacity.
- tile dimensions.

### Deterministic Vector SFU

Use a simple vector unit for softmax micro-ops. `VEXP` and `VDIV` can start as
functional approximations in simulator and later become LUT or piecewise-linear
RTL.

Potential spec fields:

- vector lanes.
- exp approximation mode.
- allowed softmax tolerance.

### Utilization Counters From Day One

Even the first simulator should report approximate:

- DMA bytes.
- MAC operations.
- vector operations.
- estimated cycles.

These counters let architecture changes be judged by measurement.

## Later Candidates

### Double Buffering

Add after single-buffer execution passes. Requires compiler scheduling,
scoreboarding, simulator overlap, and RTL handshake support.

### Mixed Precision And Scaling Metadata

Add INT8/FP16/BF16/FP8/FP4 only after the baseline numerical path is stable.
Scaling metadata should be part of compiler and runtime artifacts.

### Sparse Acceleration

Add only with explicit sparsity format and compiler legality checks.

### Banked Scratchpad

Add once cycle counters show memory conflicts or insufficient read bandwidth.

### Multi-Core Or NoC

Out of scope until one-core shared-bus designs show a measured bottleneck.

