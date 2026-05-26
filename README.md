# NPU Architecture Design System

An open, from-scratch NPU/AI accelerator architecture lab: RTL, compiler,
firmware, SoC simulation, and ASIC-oriented PPA analysis, evolving toward
Transformer inference accelerator designs.

This repository is not just a toy RTL block. The goal is to build a complete,
inspectable, and extensible hardware/software loop for NPU architecture
exploration:

- graph/operator lowering to NPU micro-ops;
- assembler-generated program artifacts;
- RTL NPU core and CPU-visible wrapper;
- CPU firmware-controlled SoC simulation;
- descriptor/SRAM-based CPU-NPU launch protocol;
- golden-model and RTL verification;
- cycle-level performance timeline reports;
- layered ASIC-oriented PPA analysis, starting with structural and event-energy
  proxies before mapped/physical estimates;
- Transformer/LLM prefill and decode workload evolution.

Current smoke coverage focuses on `matmul`, `softmax`, and a real MNIST CNN
compatibility path under CPU-controlled SoC simulation. Matrix-array and
data-movement improvements are already measured in the cycle report. The
current program phase establishes a lightweight PPA comparison framework and
Transformer-shaped workloads before escalating selected variants to public
ASIC synthesis or physical implementation.

## GitHub About

Suggested short description:

```text
Open NPU architecture lab: RTL + compiler + firmware + SoC simulation + cycle timeline UI, evolving toward tensor-array accelerator designs.
```

Suggested topics:

```text
npu
ai-accelerator
rtl
systemverilog
riscv
soc
compiler
systolic-array
tensor-core
hardware-architecture
performance-modeling
cycle-simulator
fpga
machine-learning-systems
```

## Why This Project Exists

Modern AI accelerators are full systems. ISA, compiler, runtime, memory
hierarchy, data movement, RTL, firmware, verification, and performance analysis
all have to evolve together.

This project is built around that idea. Every architecture change should be
measurable, documented, and verified end to end.

If you are interested in NPU architecture, systolic arrays, AI accelerator RTL,
compiler lowering, cycle simulators, performance counters, or SoC bring-up,
this project is intended to be a practical playground and collaboration base.

## Contributions Welcome

Contributions are welcome across multiple areas:

- NPU RTL and microarchitecture;
- systolic/tensor-array matmul engines;
- vector/SFU pipelines for softmax and normalization;
- compiler lowering and scheduling;
- cycle models and performance visualization;
- firmware/runtime and descriptor protocols;
- verification, golden models, and randomized tests;
- documentation and architecture research notes.

The project values measured progress: propose the bottleneck, update the
relevant contracts, implement the change, run the verification loop, and record
the result.

## Start Here

Read these first when entering the project:

| Entry | Purpose |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Current SoC/NPU architecture, NPU compute model, CPU/NPU interaction protocol |
| [docs/design/](docs/design/) | Detailed module design docs for SoC, wrapper, NPU core, perf, verification, and software/hardware flow |
| [docs/target_architecture.md](docs/target_architecture.md) | Research-backed target NPU architecture and staged evolution plan |
| [docs/design/ppa_methodology.md](docs/design/ppa_methodology.md) | ASIC-oriented PPA boundaries, metrics, public target assumptions, and result contract |
| [docs/design/transformer_workloads.md](docs/design/transformer_workloads.md) | LLM prefill/decode workload progression and architecture requirements |
| [docs/README.md](docs/README.md) | Full docs map and current status of planning/history documents |
| [docs/work_rules.md](docs/work_rules.md) | Collaboration and source-of-truth rules |
| [docs/collaboration_journal.md](docs/collaboration_journal.md) | Current project snapshot, next plan, important decisions, and implementation history |
| [docs/bugfix_list.md](docs/bugfix_list.md) | Bring-up bugs, root causes, fixes, and rules learned |

## Source-Of-Truth Specs

Generated RTL/software metadata should come from these specs instead of being
hand-coded in multiple places:

| Spec | Owns |
| --- | --- |
| [arch/configs/npu_v0.jsonc](arch/configs/npu_v0.jsonc) | NPU core architecture shape, ISA/uop constants, tensor dimensions |
| [arch/configs/npu_wrapper_v0.jsonc](arch/configs/npu_wrapper_v0.jsonc) | CPU-visible NPU wrapper register map and legacy debug windows |
| [arch/configs/soc_v0.jsonc](arch/configs/soc_v0.jsonc) | CPU reset vector, SoC memory map, boot image path, CPU/NPU descriptor ABI |
| [arch/configs/ppa/area_proxy_v0.jsonc](arch/configs/ppa/area_proxy_v0.jsonc) | Level 0 structural area-proxy resources and normalized coefficients |
| [arch/configs/ppa/energy_proxy_v0.jsonc](arch/configs/ppa/energy_proxy_v0.jsonc) | Level 0 event-energy proxy coefficients and matmul event derivation |

Useful generated outputs:

```text
build/soc/soc_v0_addr.h
build/soc/soc_v0_addr.svh
build/soc/soc_v0.ld
build/npu_wrapper/npu_v0_regs.h
build/npu_wrapper/npu_v0_regs.svh
build/rtl_fixture/*
```

## Main Implementation Paths

```text
hw/soc/                  SoC top, bus, boot ROM, SRAM, debug status
hw/soc/cpu/              PicoRV32 integration
hw/npu_wrapper/          CPU-visible NPU wrapper and descriptor/SRAM launch FSM
hw/npu_core/             Phase 0 NPU core RTL
hw/npu_subsystem/        Primary PPA boundary around wrapper/data mover/core
sw/soc_cpu/              CPU firmware, NPU driver, runtime smoke app
sw/npu_core/             NPU-consumed operator/program design source area
sw/tools/                Host-side compiler, assembler, fixture, firmware tools
workloads/               Benchmark and Transformer workload manifests
flows/asic/              ASIC synthesis/timing/power flow integration
ppa/                     PPA schema, constraints, and checked-in baselines
test/                    Python and RTL/SoC verification entry points
docs/                    Architecture, review, process, and bring-up documents
```

## Verification Quick Start

The Phase 0 skeleton is intentionally dependency-free.

```text
cd npu_arch_design
make validate-arch
make demo
make test
```

Key simulation targets:

| Command | What it checks |
| --- | --- |
| `make validate-arch` | NPU architecture spec validity |
| `make demo` | Graph compile + Python simulator + golden reference |
| `make npu-core-sim` | Standalone NPU core RTL fixture |
| `make npu-subsystem-elab` | Elaborate the wrapper/data-mover/core PPA boundary without simulation SoC memories |
| `make soc-sim` | Legacy direct-wrapper-window SoC smoke |
| `make cpu-soc-sim` | PicoRV32 firmware-controlled descriptor/SRAM SoC smoke |
| `make perf-report` | CPU-controlled SoC smoke plus cycle JSON/HTML report |
| `make ppa-proxy-report` | RTL-measured performance plus normalized structural-area/event-energy proxy report |
| `make test` | Python unit tests plus available RTL/SoC smoke tests |

If a RISC-V bare-metal GCC is available in `PATH`, `make cpu-soc-sim` builds
the C/ASM firmware under `sw/soc_cpu`. Otherwise the Makefile can fall back to
the generated smoke image path used during early bring-up.

`make perf-report` writes:

```text
build/perf/workload_manifest.json
build/perf/perf.json
build/perf/perf_report.html
```

`make ppa-proxy-report` is the current PPA entry point. It intentionally
reports:

```text
performance/traffic: measured from RTL PERF_JOB counters
area:                structural normalized proxy, not synthesized area
energy:              event-based normalized proxy, not measured power
```

The current SoC run is grouped using the explicit workload manifest generated at
`build/perf/workload_manifest.json`; PPA report compatibility includes its
manifest identity. The Level 0 JSON contract and validation rules live in
`ppa/schema/ppa_proxy_schema_v0.md` and are enforced during
`make ppa-proxy-report`.

For report/schema-only iteration after `build/perf/perf.json` already exists,
`make ppa-proxy-from-perf` regenerates and validates the Level 0 proxy output
without rerunning the SoC simulation.

After a baseline exists, every NPU architecture iteration must report deltas
against that baseline. Improvements and costs are both part of the decision:
latency reductions must remain visible alongside area/buffer growth, energy
changes, and metrics not yet covered by the current evidence level.

Public-library mapping and OpenROAD/OpenLane physical estimates are later
evidence levels, used after representative variants and workload boundaries
are worth the heavier analysis.

## Current End-To-End Flow

```text
graph/input fixture
  -> host tools generate NPU program words and firmware fixture data
  -> RISC-V firmware copies input/program buffers into SRAM
  -> firmware fills soc_npu_job_desc_t in SRAM
  -> firmware writes NPU wrapper DESC_ADDR and CTRL.start
  -> NPU wrapper fetches descriptor/program/input data from SRAM
  -> wrapper loads NPU core and pulses start
  -> NPU core executes matmul or softmax
  -> wrapper writes output data back to SRAM
  -> firmware checks output and writes test_status
```

## Planning And History

The current architecture entry is [docs/architecture.md](docs/architecture.md).
Detailed active module docs live under [docs/design/](docs/design/). Planning
and historical notes are kept only when they add context:

- [docs/project_plan.md](docs/project_plan.md): milestones and ownership notes.
- [docs/fpga_bringup.md](docs/fpga_bringup.md): FPGA direction and gaps.
- [docs/archive/](docs/archive/): archived design drafts.

Refresh research references when planning a new architecture iteration:

```text
make refresh-references
```
