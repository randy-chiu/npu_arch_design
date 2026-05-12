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

## Project Process

For demonstrations, start here:

- [Collaboration journal](docs/collaboration_journal.md): records the human/AI
  teamwork process, major decisions, reasoning, and validation milestones.
- [Mandatory agent rules](AGENT_RULES.md): rules every future module agent must
  follow.
- [Current architecture spec](arch/configs/npu_v0.jsonc): human-readable and
  tool-readable hardware/ISA contract.

## Documents

- [Mandatory agent rules](AGENT_RULES.md)
- [Collaboration journal](docs/collaboration_journal.md)
- [Overall architecture](docs/overall_architecture.md)
- [Minimal closed-loop system](docs/minimal_closed_loop.md)
- [Iteration roadmap](docs/roadmap.md)
- [Agent ownership plan](docs/agent_plan.md)
- [Work rules for module agents](docs/work_rules.md)

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

Run the Phase 0 RTL smoke test on a machine with `iverilog` installed:

```text
make rtl-sim
```
