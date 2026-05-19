# Project Plan

[TOC]

Status note:

This document is for milestones and ownership. For the current NPU SoC
architecture, use `docs/architecture.md`.

This document is the current planning entry point. It combines milestone
planning, ownership boundaries, and change rules that were previously split
between the roadmap and agent plan.

## Operating Model

The project uses a chief-architect model:

- the human owner sets goals and reviews architecture direction;
- the AI keeps contracts, docs, tests, and implementation aligned;
- module work is accepted only when the relevant closed-loop verification
  passes.

Every architecture fact should have one canonical source. Changes to ISA,
memory map, launch protocol, or program format are system changes and must
update compiler/tooling, simulator, RTL, tests, and docs together.

## Current Module Ownership

| Area | Ownership |
| --- | --- |
| `arch` | Architecture config and source-of-truth contracts |
| `hw/soc` | SoC top, bus, memories, debug peripherals, later CPU integration |
| `hw/npu_wrapper` | CPU-visible NPU wrapper, `opsched`, register map, launch protocol |
| `hw/npu_core` | NPU core RTL and core-level tests |
| `sw/soc_cpu` | Firmware, NPU-wrapper driver, runtime, CPU-side apps |
| `sw/npu_core` | Programs or operator code consumed by the NPU core |
| `sw/tools` | Host-side compilers, assemblers, simulators, fixture generation |
| `test` | Top-level graph, RTL, SoC, and fixture verification |
| `docs` | Current design notes, work rules, journal, and bring-up plans |

## Near-Term Milestones

Architecture evolution after the current Phase 0 loop is tracked in
`docs/target_architecture.md`. The short version is: keep the performance report
as the baseline, replace the scalar iterative matmul with a measured
matrix-array engine first, then add real data movement, scratchpad banking,
vector/SFU pipeline timing, and compiler overlap only when counters justify the
complexity.

### W1: Digits Classifier Workload

Status: linear classifier SoC path implemented; Tiny MLP tool path implemented;
real MNIST CNN `fc2` SoC path implemented. Tiny MLP SoC path is no longer the
active priority.

Purpose:

- pause further core optimization long enough to add a realistic model-level
  workload;
- run an image-like input through NPU-visible classifier math;
- validate predicted digit labels rather than only operator tensors.

Exit condition:

- checked-in 8x8 digit inputs and int8 classifier weights exist;
- host-side golden predicts expected labels;
- compiler and micro-op simulator run the classifier graph end to end;
- CPU firmware launches 16 RTL-compatible `8x8x8` matmul tile jobs and checks
  classifier logits/predicted label;
- Tiny MLP graph exists with CPU/NPU placement and tiled matmul tests;
- real open-source MNIST CNN is documented in `docs/real_mnist_cnn_workload.md`;
- CPU firmware launches 32 RTL-compatible `8x8x8` matmul tile jobs for the
  original MNIST CNN `fc2` quantized hardware-facing view and checks expected
  label;
- next step is mapping the original MNIST CNN `fc1: 9216 -> 128` while keeping
  conv/maxpool on the CPU/tool side.

### W2: Real MNIST CNN Layer Mapping

Status: `fc2` SoC RTL path implemented; `fc1` planned.

Purpose:

- use a real open-source pretrained CNN and real MNIST images as the model-level
  workload;
- preserve the original graph and trained float weights as source of truth;
- map only layers that current or next-step NPU RTL can verify.

Exit condition:

- `fc2` quantized view passes tool-level tiled simulator and CPU-controlled SoC
  RTL;
- `fc1` tiling is specified with job count, SRAM footprint, accumulation policy,
  and perf grouping;
- `fc1` passes tool-level tiled simulator against original float-model class
  prediction;
- `fc1` then passes CPU-controlled SoC RTL, with conv/maxpool still CPU/tool
  side until a conv lowering plan is chosen.

### M0: Phase 0 Core Loop

Status: implemented.

Exit condition:

- graph compiles to JSON micro-ops;
- Python micro-op simulator matches CPU golden;
- standalone NPU RTL simulation passes generated fixtures.

### M1: Minimal SoC Wrapper Loop

Status: first smoke path implemented.

Exit condition:

- `opsched` exposes NPU control/status and tensor/program windows;
- SoC testbench launches NPU only through MMIO;
- matmul and softmax can be launched through the SoC path;
- generated fixture metadata remains deterministic.

### M2: Real SoC CPU Firmware

Status: first PicoRV32 simulation path implemented.

Exit condition:

- an open-source RV32 CPU is integrated under `hw/soc/cpu`;
- SoC simulation loads firmware artifacts instead of only testbench-driven bus
  tasks;
- firmware uses an NPU-wrapper driver/runtime shape to launch NPU work;
- a later C firmware flow with a linker script replaces the temporary Python
  RV32I firmware emitter.

### M3: Tool Artifact Split

Status: planned.

Exit condition:

- `sw/tools/npu_phase0` compatibility package is split into compiler,
  assembler, simulator, golden, and fixture modules;
- NPU assembler emits binary program artifacts for both RTL tests and CPU
  firmware;
- graph-to-operator-stream and operator-stream-to-uop lowering boundaries are
  explicit.

### M4: FPGA Candidate Cleanup

Status: planned.

Exit condition:

- board wrapper, clocks/resets, memory init, constraints, and vendor build flow
  are defined;
- host or firmware path can load tensors/programs and read results;
- FPGA bring-up uses the same register map and program artifacts as SoC
  simulation.

## PPA Decision Rules

Do not accept architecture complexity without measurement. Each proposed change
should state:

- which workload improves;
- which bottleneck it addresses;
- expected performance gain;
- area/power/cost increase;
- compiler/runtime impact;
- verification impact.

Examples:

| Change | Accept when |
| --- | --- |
| Wider bus | DMA bandwidth is measured as a bottleneck and memory can consume it |
| Larger MAC array | Utilization remains high and memory can feed it |
| More scratchpad | Tile reuse reduces external traffic enough to justify area |
| More DMA channels | Overlap improves wall-clock cycles despite arbitration cost |
| Better SFU | Softmax/GELU dominates measured workload time or accuracy |

## Agent Roles

| Agent | Primary responsibility |
| --- | --- |
| Chief Architect | System contracts, milestones, PPA tradeoffs, review decisions |
| ISA Agent | Instruction semantics, encoding, assembler/disassembler tests |
| Hardware Agent | RTL modules, integration, testbenches |
| Compiler Agent | Graph lowering, scheduling, NPU program emission |
| Runtime Agent | CPU-side driver/runtime and launch flow |
| Simulator Agent | Functional and cycle models, traces, counters |
| Verification Agent | Golden models, tests, simulator-vs-RTL comparison |
| DevTools Agent | CLI, build system, packaging, docs |

## Change Process

1. State the measured problem or missing capability.
2. Update the architecture contract if shared behavior changes.
3. Update affected hardware, software, tools, tests, and docs together.
4. Run the relevant verification loop.
5. Record the decision and validation result in
   `docs/collaboration_journal.md`.
