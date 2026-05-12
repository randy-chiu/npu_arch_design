# Agent Ownership Plan

## Operating Model

Use one architecture owner and multiple module agents. The architecture owner
keeps interfaces stable, accepts or rejects cross-cutting changes, and prevents
compiler, runtime, simulator, and RTL from drifting apart.

Every agent should own concrete files or modules, publish interface changes
through architecture specs, and provide tests before changing behavior.

## Core Agents

| Agent | Ownership | Primary Outputs |
| --- | --- | --- |
| Chief Architect | System architecture, milestones, interface contracts, PPA tradeoffs | architecture spec, roadmap, review decisions |
| ISA Agent | instruction semantics, encoding, assembler/disassembler | ISA schema, encoder/decoder, ISA tests |
| Hardware Generator Agent | RTL generation, module templates, integration | generated RTL, synthesis-ready modules |
| Compute Agent | MAC array, accumulator, vector/SFU units | compute RTL, latency model, unit tests |
| Memory/Bus Agent | scratchpad, DMA, interconnect, banking | memory RTL/model, bandwidth counters |
| Compiler Agent | graph IR, tiling, scheduling, lowering | compiler passes, emitted programs |
| Runtime Agent | host API, memory allocation, command submission | runtime library, backend abstraction |
| Simulator Agent | functional and cycle simulators | simulator engines, traces, counters |
| Verification Agent | golden models, test generation, RTL comparison | test suites, diff tools, CI checks |
| PPA Agent | performance/power/area models and reports | PPA dashboard, sweep tooling |
| DevTools Agent | CLI, build system, developer workflow | commands, packaging, docs |

## Interface Contracts

All agents must consume the same architecture config. Any cross-module change
must update the relevant contract:

- `arch schema`: legal hardware parameters.
- `ISA spec`: instruction fields and semantics.
- `memory map`: host-visible addresses and buffers.
- `program format`: compiler-to-runtime artifact.
- `trace format`: simulator/RTL comparison artifact.
- `counter schema`: performance and PPA metrics.

## Recommended Agent Prompts

### Chief Architect

```text
You are the chief architect for the NPU architecture design system. Own the
architecture spec, system interfaces, roadmap, and PPA decision process. Review
module proposals for correctness, integration risk, and measurable value.
```

### ISA Agent

```text
You own the NPU ISA. Define instruction semantics, fields, assembler format,
binary encoding, decoder tables, and ISA-level tests. Coordinate with compiler,
simulator, and RTL agents before changing instruction behavior.
```

### Hardware Generator Agent

```text
You own RTL generation and top-level hardware integration. Generate parameterized
RTL from the architecture config. Keep generated module interfaces stable and
provide RTL testbenches for each generated block.
```

### Compiler Agent

```text
You own graph lowering to NPU programs. Implement shape validation, tiling,
scratchpad planning, scheduling, and ISA emission for supported operators.
Consume the architecture config and emit runtime-compatible program artifacts.
```

### Simulator Agent

```text
You own functional and cycle simulation. Execute compiler-emitted programs,
produce deterministic outputs, expose traces and counters, and maintain
agreement with ISA semantics and RTL behavior.
```

### Verification Agent

```text
You own correctness closure. Build CPU golden models, randomized tests,
simulator-vs-RTL comparison, tolerances, failure minimization, and CI commands.
```

## Change Process

1. Propose change against one measured problem.
2. Update architecture config or schema.
3. Update ISA/memory/program contract if needed.
4. Implement module changes.
5. Update compiler/simulator/runtime compatibility.
6. Add or update tests.
7. Produce correctness and PPA report.
8. Chief Architect accepts or requests revision.

## First Agent Assignments

For `npu_v0`, assign work in this order:

1. Chief Architect: freeze `npu_v0` spec and minimal ISA.
2. ISA Agent: implement assembly/JSON instruction schema.
3. Simulator Agent: implement functional simulator for handwritten programs.
4. Compiler Agent: emit programs for `matmul` and `softmax`.
5. Verification Agent: build golden tests and comparison harness.
6. Hardware Generator Agent: generate first RTL and testbench.
7. Compute Agent: implement matmul and vector/SFU modules.
8. Memory/Bus Agent: implement scratchpad and DMA model/RTL.
9. Runtime Agent: unify simulator and RTL launch flow.
10. PPA Agent: add counters and first reports.

