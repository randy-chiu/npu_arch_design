# Target NPU Architecture

[TOC]

Status: active planning note.

This document records the target direction for the project after the Phase 0
CPU/NPU functional loop and the first cycle report. It is not a commitment to
implement every feature immediately. Architecture changes still follow the
measurement rule in `docs/project_plan.md`: do not add complexity without a
measured bottleneck and a verification plan.

## 1. Research Snapshot

Checked on 2026-05-17. Primary public references:

| Source | Relevant architecture signals |
| --- | --- |
| NVIDIA Blackwell architecture | Tensor Core / Transformer Engine, low precision including FP4, fine-grained scaling, LLM/MoE focus, very high chip-to-chip bandwidth |
| Google Cloud TPU v6e / Trillium | TensorCore contains matrix multiply units, vector unit, scalar unit; optimized for transformers, text-to-image, CNN training/fine-tuning/serving; 2D torus scale-out |
| Google TPU v4 paper | DSA for production ML, optical reconfigurable interconnect, SparseCore for embeddings, large scale-out system view |
| AMD MI300X public specs | Many matrix cores, high HBM capacity/bandwidth, FP8/BF16/INT8 support, structured sparsity peak modes |
| Intel Gaudi 3 white paper page | Accelerator system view includes host interface, compute, software, networking, and product specs |
| Gemmini paper | Open-source RISC-V integrated systolic-array generator for matrix multiplication and inference design-space evaluation |
| Eyeriss v2 paper | Flexible on-chip network and compressed-domain sparse processing for varied layer shapes |
| SCALE-Sim v3 paper | Modern cycle modeling needs multi-core, sparse matmul, hierarchical memory, DRAM stalls, data layout, and energy/power analysis |

Public trend summary:

- The core compute primitive is no longer a single scalar multiplier. Modern
  accelerators use matrix/tensor units, usually systolic or tensor-core-like
  arrays, so many MACs happen every cycle after pipeline fill.
- Memory movement is often the real limiter. HBM bandwidth, SRAM hierarchy,
  data layout, DMA overlap, and interconnect topology are first-class
  architecture features.
- Transformer workloads dominate current design pressure: GEMM, attention,
  softmax/norm, KV cache movement, and MoE routing all matter.
- Low precision is a central axis: INT8/BF16/FP8 are mainstream, and FP4-style
  formats with scaling are now an explicit frontier direction.
- Sparsity and embeddings are not optional long-term topics, but they should
  enter only after dense matmul/dataflow and counters are trustworthy.
- The compiler/runtime stack is part of the architecture. Dataflow, tiling,
  buffering, layout, and scheduling determine whether the hardware is utilized.

## 2. Current Matmul Reality

The current `npu_v0_top` matmul path is not a tensor/cube engine yet.

`arch/configs/npu_v0.jsonc` describes a logical Phase 0 tile:

```text
array_m = 8
array_n = 8
k_step  = 8
mac_lanes = 64
```

But the current hand-written RTL executes the tile with one multiply-accumulate
update per clock inside `ST_MATMUL`:

```text
for i in 0..7:
  for j in 0..7:
    for k in 0..7:
      acc += A[i,k] * B[k,j]
```

That is:

```text
8 * 8 * 8 = 512 MAC cycles
```

So the current 512-cycle measurement is expected. It is the baseline behavior of
a single-lane iterative engine, not a bug in the performance report.

What people often call a "cube", "tensor core", or "MXU" is closer to a
matrix-multiply array. In an ideal 8x8 output-stationary array, the same logical
8x8x8 tile would be closer to `K + pipeline_fill/drain` cycles for the inner
compute after data is resident, because 64 output elements are updated in
parallel each k step. Real cycle count still includes fill/drain, load/store,
bank conflicts, and control overhead.

## 3. Long-Term Target

The project target should be a small, explainable NPU SoC that can grow toward
modern accelerator principles without losing verification discipline.

Target block diagram:

```text
CPU firmware/runtime
  -> MMIO command queue / descriptors
  -> NPU wrapper scheduler
      -> DMA / data mover
      -> SRAM / scratchpad banks
      -> NPU core cluster
          -> instruction front-end
          -> matrix/tensor array
          -> accumulator / reduction path
          -> vector/SFU pipeline
          -> local counter block
      -> SRAM/HBM-like memory model
  -> result / IRQ / performance counters
```

Target properties:

| Area | Target |
| --- | --- |
| Control | descriptor queue, status/error/timeout, optional IRQ |
| Memory | explicit DMA/data mover, banked scratchpad, accumulator SRAM, future HBM-like model |
| Matrix compute | parameterized systolic/tensor array, output-stationary first |
| Vector/SFU | multi-cycle vector pipeline for softmax/norm/activation/reduction |
| ISA | stable uop stream with load/compute/store/barrier and shape metadata |
| Compiler | tile selection, data layout, double buffering, operator fusion |
| Runtime | job descriptor allocation, program artifact loading, perf counter collection |
| Performance model | RTL counters, cycle simulator, timeline UI, bottleneck classification |
| Scalability | single-core first, then multi-core / NoC / partitioning only after counters justify it |

## 4. Proposed Architecture Milestones

### A0: Counter-Driven Baseline

Status: in progress.

Goal:

- Keep current functional behavior stable.
- Use `make perf-report` as the baseline for every architecture change.
- Make the report show pipeline overlap, waits, and module-level timeline.

Exit criteria:

- matmul and softmax show per-job CPU/wrapper/core timelines.
- wrapper and core phase timelines share a common cycle axis.
- current 512-cycle matmul baseline is documented.

### A1: Parallel Matmul Engine

Goal:

- Replace the single-lane `ST_MATMUL` loop with a small parameterized matrix
  engine.
- First target: 8x8 output-stationary compute lanes, matching the current
  logical tile.

Expected behavior:

- Current compute component moves from about 512 cycles toward about 8 plus
  fill/drain/control cycles for resident data.
- Total job time will not improve by 64x because wrapper input/output movement
  and launch overhead remain visible.

Required updates:

- RTL: matrix engine module, accumulator writeback, core FSM integration.
- Tooling: cycle model estimates for scalar vs array matmul.
- Tests: exact matmul output, perf regression comparing old/new cycle trend.
- Docs: update architecture and perf baseline.

### A2: Real Data Movement And Scratchpad Banking

Goal:

- Stop treating wrapper host-window writes as the long-term memory path.
- Introduce explicit data mover semantics and banked scratchpad access.

Key work:

- define DMA uops or descriptor fields;
- model SRAM read/write bandwidth and bank conflicts;
- add timeline lanes for SRAM port, DMA, scratchpad, and core input stalls;
- measure whether compute becomes memory-bound after A1.

### A3: Multi-Cycle Vector/SFU Pipeline

Goal:

- Replace softmax's current single-cycle task behavior with visible vector and
  reduction pipeline stages.

Key work:

- vector lanes, reduction tree/loop, exp approximation latency, reciprocal/div;
- pipeline counters for active, stall, and dependency wait;
- softmax/norm workloads in compiler and tests.

### A4: Compiler Scheduling And Overlap

Goal:

- Move from "run one op serially" to scheduled load/compute/store overlap.

Key work:

- tiled graph lowering;
- double buffering;
- barriers and dependencies;
- operator fusion where useful;
- report critical path and idle gaps.

### A5: Advanced Features

Only after A1-A4 are measured:

- lower precision modes: INT4/FP8/FP4-style scaled formats;
- structured sparsity and compressed-domain execution;
- multi-core partitioning and NoC;
- command queues and IRQ-driven runtime;
- energy/area estimation.

## 5. Immediate Next Plan

Recommended next implementation sequence:

1. Freeze the current perf baseline in docs and generated report examples.
2. Add a cycle-model estimate for current scalar matmul vs ideal 8x8 array
   matmul, without changing RTL yet.
3. Create a separate `matmul_array` RTL module behind the same core interface.
4. Add a config switch or new architecture version for scalar vs array matmul.
5. Update `perf-report` to show expected-vs-measured cycles for matmul.
6. Only then replace the default core matmul path.

This keeps the next step narrow: prove that the matrix engine changes the
measured bottleneck before adding DMA, banking, or vector pipeline complexity.

## 6. Source Links

- NVIDIA Blackwell architecture: https://www.nvidia.com/en-gb/data-center/technologies/blackwell-architecture/
- Google Cloud TPU v6e documentation: https://docs.cloud.google.com/tpu/docs/v6e
- Google TPU v4 paper: https://arxiv.org/abs/2304.01433
- AMD MI300X public specs: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- Intel Gaudi 3 white paper page: https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html
- Gemmini paper page: https://alonamid.github.io/publication/gemmini/
- Eyeriss v2 paper: https://arxiv.org/abs/1807.07928
- SCALE-Sim v3 paper: https://arxiv.org/abs/2504.15377
