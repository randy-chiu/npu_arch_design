# NPU Architecture Design System

This directory contains the top-level architecture plan for a from-scratch NPU
architecture design and verification system.

The goal is to build a closed-loop hardware/software co-design platform:

1. Describe a configurable NPU architecture, including ISA, compute array,
   scratchpad, DMA, internal bus, control path, and memory map.
2. Generate RTL and software-facing metadata from the same architecture spec.
3. Compile neural network operators into NPU programs.
4. Run the same program on simulator, RTL testbench, and later FPGA/ASIC targets.
5. Iterate the architecture while tracking PPA: performance, power, and area/cost.

Start with a minimal self-checking system that supports only `matmul` and
`softmax`, then expand in controlled increments.

For developers reviewing the codebase, start with the
[code structure review](docs/code_structure_review.md). It maps the current
repository layout, graph-to-micro-op compiler path, Python micro-op functional
model, RTL simulation fixture path, and the main testing entry points.

## Project Process

For demonstrations, start here:

- [Collaboration journal](docs/collaboration_journal.md): records the human/AI
  teamwork process, major decisions, reasoning, and validation milestones.
- [Mandatory agent rules](AGENT_RULES.md): rules every future module agent must
  follow.
- [Current architecture spec](arch/configs/npu_v0.jsonc): human-readable and
  tool-readable hardware/ISA contract.
- [Code structure review](docs/code_structure_review.md): practical map of the
  current code, data flow, tests, and verification entry points.
- [FPGA bring-up notes](docs/fpga_bringup.md): explains what exists today and
  what is still required before running on a real FPGA board.

## Documents

- [Mandatory agent rules](AGENT_RULES.md)
- [Collaboration journal](docs/collaboration_journal.md)
- [Overall architecture](docs/overall_architecture.md)
- [Code structure review](docs/code_structure_review.md)
- [Minimal closed-loop system](docs/minimal_closed_loop.md)
- [Iteration roadmap](docs/roadmap.md)
- [Agent ownership plan](docs/agent_plan.md)
- [Work rules for module agents](docs/work_rules.md)
- [FPGA bring-up notes](docs/fpga_bringup.md)

## Phase 0 Quick Start

The Phase 0 skeleton is intentionally dependency-free.

```text
cd npu_arch_design
make validate-arch
make demo
make test
```

The current hardware description file is:

```text
arch/configs/npu_v0.jsonc
```

Refresh research references when planning a new architecture iteration:

```text
make refresh-references
```

Run the Phase 0 RTL test on a machine with `iverilog` installed. The make
target first emits deterministic program/input/expected-output fixtures from
the Python tooling, then the SystemVerilog testbench loads those files:

```text
make rtl-sim
```
