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

## Current Program Goal

The project goal is now an ASIC-oriented NPU architecture exploration platform
for inference workloads, with Transformer/LLM inference as the primary
long-term application direction.

The platform must connect:

```text
architecture variant
  -> functional verification
  -> workload execution and cycle counters
  -> structural area / event-energy proxy
  -> lightweight ASIC mapping and timing when justified
  -> physical/activity-driven estimates for selected variants
  -> comparable PPA report
  -> next architecture decision
```

No FPGA board or private ASIC technology is assumed. Initial PPA results use
measured RTL counters with normalized structural-area and event-energy proxy
models. Public/open ASIC flows enter after that comparison loop is useful;
they are estimates, not signoff claims. The first intended physical ASIC
reference target remains `sky130hd`.

MNIST CNN remains an important compatibility and end-to-end regression
workload. It is no longer the primary driver for future NPU capabilities.
Transformer kernels and eventually a tiny decoder block will drive new
matrix/vector/reduction/memory architecture decisions.

## Current Module Ownership

| Area | Ownership |
| --- | --- |
| `arch` | Architecture config and source-of-truth contracts |
| `hw/soc` | SoC top, bus, memories, debug peripherals, later CPU integration |
| `hw/npu_wrapper` | CPU-visible NPU wrapper, `opsched`, register map, launch protocol |
| `hw/npu_core` | NPU core RTL and core-level tests |
| `hw/npu_subsystem` | Primary PPA top: wrapper/data mover/local-memory boundary/core without simulation SoC baggage |
| `sw/soc_cpu` | Firmware, NPU-wrapper driver, runtime, CPU-side apps |
| `sw/npu_core` | Programs or operator code consumed by the NPU core |
| `sw/tools` | Host-side compilers, assemblers, simulators, fixture generation |
| `sw/tools/ppa` | PPA result normalization, comparison, and reporting |
| `workloads` | Versioned benchmark manifests, Transformer traces, and compatibility workload definitions |
| `flows/asic` | ASIC synthesis/timing/power flow configuration and technology targets |
| `ppa` | Measurement schema, constraints, and checked-in baseline summaries |
| `test` | Top-level graph, RTL, SoC, and fixture verification |
| `docs` | Current design notes, work rules, journal, and bring-up plans |

## Near-Term Milestones

Architecture evolution after the current Phase 0 loop is tracked in
`docs/target_architecture.md`. A1/A2 work established cycle-level performance
visibility and a real data-movement optimization case. Before adding further
datapath features, the next program phase establishes lightweight PPA proxy
comparison and Transformer workload measurement, then upgrades selected
comparisons to mapped and physical ASIC estimates.

### PPA0: Methodology And Stable Evaluation Boundary

Status: implemented first cut and being extended with Level 0 proxy reporting.
The methodology, primary subsystem boundary, result schema, Transformer
manifest, and structural elaboration gate are present.

Purpose:

- define repeatable ASIC-oriented PPA measurement while no private process or
  board target is available;
- establish `npu_subsystem` as the primary architecture-comparison boundary;
- prevent simulation-only ROM/SRAM size from being misreported as NPU area;
- establish a lightweight comparison path before ASIC toolchain integration;
- make Transformer inference the long-term workload direction.

Exit condition:

- `docs/design/ppa_methodology.md` defines metrics, tops, memory accounting,
  activity windows, technology targets, and result interpretation;
- `docs/design/transformer_workloads.md` defines prefill/decode kernel and
  block-level workload progression;
- PPA result schema and target configuration have stable locations;
- a synthesizable `npu_subsystem_top` boundary exists independently of the
  CPU/boot-ROM simulation system.

### PPA1: Structural And Event Proxy Baseline

Status: implemented first cut. `make ppa-proxy-report` consumes current
`PERF_JOB`-derived workload results plus explicit structural/event coefficient
configs and generates labeled Level 0 JSON/HTML output.

Purpose:

- combine real RTL cycle/movement metrics with structural area and
  event-energy proxy models;
- quantify compute/data-movement/buffering tradeoffs before choosing heavier
  implementation analysis.

Exit condition:

- `make ppa-proxy-report` generates machine-readable and readable output;
- output labels performance as RTL-measured and area/energy as normalized
  proxy results;
- existing full `fc1` K-stream counters produce a reportable proxy baseline;
- candidate reports expose improvement and cost deltas against a named
  baseline, beginning with serial-to-ping-pong K-streaming;
- Transformer manifests can supply future traffic events.

### PPA2: Lightweight ASIC Mapping And Timing Baseline

Status: planned after PPA1.

Purpose:

- introduce Yosys/ABC and a public Liberty library for pre-layout mapped-area
  comparison;
- add OpenSTA timing where practical without requiring physical implementation.

Exit condition:

- `npu_core` and `npu_subsystem` produce mapped-area summaries;
- local-memory implementation/accounting is explicit;
- timing results clearly state their pre-layout interpretation.

### PPA3: Activity Power And Selected Physical Validation

Status: planned after PPA2.

Purpose:

- add activity-driven on-chip power estimates;
- use `sky130hd` OpenROAD/OpenLane physical flow only for selected variants;
- turn PPA into a required architecture decision gate.

Exit condition:

- one report compares latency, throughput, area, timing, power, energy, and
  memory traffic against a named baseline;
- area/power results state whether they are mapped or physical estimates;
- external-memory energy remains separately modeled for Transformer analysis.

### TR0: Transformer Workload Baseline

Status: planned in parallel with PPA1 once the proxy contract is stable.

Purpose:

- add workload definitions that represent LLM prefill and decode pressure
  before expanding the RTL ISA without evidence.

Initial coverage:

- GEMM and skinny GEMM/GEMV;
- QKV projection, `Q*K^T`, attention softmax, and attention-value matmul;
- RMSNorm and FFN projections;
- KV-cache traffic microbenchmarks;
- a small decoder-only block after the kernel metrics are stable.

### W1: Digits Classifier Workload

Status: linear classifier SoC path implemented; real MNIST CNN `fc2` SoC path
implemented. The earlier Tiny MLP branch has been removed because the active
model-level path is the real MNIST CNN.

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
- real open-source MNIST CNN is documented in `docs/real_mnist_cnn_workload.md`;
- CPU firmware launches 32 RTL-compatible `8x8x8` matmul tile jobs for the
  original MNIST CNN `fc2` quantized hardware-facing view and checks expected
  label;
- next step is mapping the original MNIST CNN `fc1: 9216 -> 128` while keeping
  conv/maxpool on the CPU/tool side.

### W2: Real MNIST CNN Layer Mapping

Status: `fc2` SoC RTL path implemented; `fc1` tool-level layer mapping, first
SoC RTL tile checkpoint, multi-chunk K-streaming SoC smoke, full single-N-tile
`fc1` K-streaming SoC smoke, DMA staging, explicit data mover counters,
K-streaming A/B ping-pong overlap, and full 16-output-N-tile `fc1` K-streaming
SoC coverage are implemented. Next work is applying `fc1` bias/ReLU and feeding
the NPU-produced `fc1_relu` into the existing `fc2` path instead of using the
precomputed tool-side `fc1_relu`.

Role after PPA0: compatibility and end-to-end functional regression workload.
New NPU features should be justified primarily by PPA evidence and Transformer
workload requirements rather than only by this CNN shape.

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
- first real `fc1` tile passes CPU-controlled SoC RTL as a data-layout and
  arithmetic checkpoint;
- `fc1` K-streaming smoke passes CPU-controlled SoC RTL with multiple K chunks
  accumulated inside one descriptor;
- full single-N-tile `fc1` passes CPU-controlled SoC RTL through an NPU-side
  K-streaming contract, with conv/maxpool still CPU/tool side until a conv
  lowering plan is chosen;
- K-streaming ping-pong overlap reduces full single-N-tile `fc1` total cycles
  while preserving `data_mover.words` and `core.matmul` work.
- full 16-output-N-tile `fc1` K-streaming layer passes CPU-controlled SoC RTL,
  still with conv/maxpool on the CPU/tool side and `fc1` bias/ReLU as the next
  integration step.

### W3: Retire Linear Digits Classifier

Status: planned after real MNIST CNN `fc1/fc2` SoC coverage is stable.

Purpose:

- remove the temporary 8x8 hand-constructed digit workload once the real 28x28
  MNIST CNN path provides enough always-on model-level coverage;
- avoid maintaining two classifier workload stories with different image sizes
  and unrelated model semantics.

Do not delete only the image files. Retirement must remove or replace the whole
workload surface:

- `test/assets/digits/`;
- `test/assets/digits_realistic/`;
- `test/inputs/digits_classifier_samples.json`;
- `test/rtl/test_digits_classifier.py`;
- firmware smoke `DIGITS_*` data generation and C runtime path;
- perf report grouping for `digits_linear_classifier`;
- `docs/digits_classifier_workload.md` active references.

Precondition:

- real MNIST CNN external fixture setup is reliable enough for the normal test
  environment, or an equivalent checked-in minimal real-MNIST fixture exists;
- `fc1/fc2` have stable SoC RTL coverage and perf grouping.

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

Status: deferred; the primary implementation-estimation direction is now a
public ASIC flow rather than board bring-up.

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
