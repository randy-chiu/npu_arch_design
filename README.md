# NPU Architecture Design System

This repository builds a small NPU-centered SoC for architecture validation.
The current goal is not a full production NPU. It is a hardware/software
closed loop where a CPU firmware program stages NPU inputs and program streams
in SRAM, launches the NPU wrapper, lets the NPU core run, and checks results.

Current smoke coverage focuses on `matmul`, `softmax`, and the CPU-controlled
SoC path around them.

## Start Here

Read these first when entering the project:

| Entry | Purpose |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Current SoC/NPU architecture, NPU compute model, CPU/NPU interaction protocol |
| [docs/target_architecture.md](docs/target_architecture.md) | Research-backed target NPU architecture and staged evolution plan |
| [docs/code_structure_review.md](docs/code_structure_review.md) | File-level code map, RTL walkthrough, software flow, and verification details |
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
sw/soc_cpu/              CPU firmware, NPU driver, runtime smoke app
sw/npu_core/             NPU-consumed operator/program design source area
sw/tools/                Host-side compiler, assembler, fixture, firmware tools
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
| `make soc-sim` | Legacy direct-wrapper-window SoC smoke |
| `make cpu-soc-sim` | PicoRV32 firmware-controlled descriptor/SRAM SoC smoke |
| `make perf-report` | CPU-controlled SoC smoke plus cycle JSON/HTML report |
| `make test` | Python unit tests plus available RTL/SoC smoke tests |

If a RISC-V bare-metal GCC is available in `PATH`, `make cpu-soc-sim` builds
the C/ASM firmware under `sw/soc_cpu`. Otherwise the Makefile can fall back to
the generated smoke image path used during early bring-up.

`make perf-report` writes:

```text
build/perf/perf.json
build/perf/perf_report.html
```

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
Planning and historical notes are still kept for context:

- [docs/project_plan.md](docs/project_plan.md): milestones and ownership notes.
- [docs/soc_bringup.md](docs/soc_bringup.md): earlier SoC bring-up plan.
- [docs/fpga_bringup.md](docs/fpga_bringup.md): FPGA direction and gaps.
- [docs/archive/](docs/archive/): archived design drafts.

Refresh research references when planning a new architecture iteration:

```text
make refresh-references
```
