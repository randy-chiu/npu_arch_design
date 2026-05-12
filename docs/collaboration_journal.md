# Collaboration Journal

This journal records the project process and AI collaboration flow. It avoids
low-level patch history and focuses on goals, decisions, reasoning, and team
workflow.

## Session 1: Project Framing

The project started with a broad goal: build an NPU architecture design and
verification system from scratch. The desired system should eventually support
custom NPU architecture definition, ISA design, compute units, internal bus,
RTL generation, compiler, runtime, operators, simulators, verification, and PPA
iteration.

The initial engineering judgment was to avoid starting with a large design.
Instead, the project should first build a minimal self-verifying loop that can
compile and run only `matmul` and `softmax`. This gives the team a working
hardware/software contract before adding complexity.

Key decision:

- The useful unit of progress is an end-to-end verified loop, not an isolated
  hardware block or compiler pass.

Initial architecture split:

- Architecture spec as the source of truth.
- Hardware modules: sequencer, ISA decoder, DMA, scratchpad, accumulator,
  matmul engine, vector/SFU, internal bus.
- Software modules: compiler, runtime, functional simulator, cycle simulator,
  golden tests, PPA tooling.
- Verification loop: compile graph, run simulator/RTL, compare with CPU golden,
  report counters.

Artifacts created:

- Overall architecture design.
- Minimal closed-loop design.
- Roadmap.
- Agent ownership plan.

## Session 2: Team Rules And Phase 0

The user clarified that this will become a long-running multi-agent project, so
shared rules must be defined before scaling implementation.

Rules established:

- Hardware spec changes are system changes.
- ISA changes require compiler, simulator, RTL, runtime, tests, and
  documentation to be updated together.
- The project should keep the first NPU simple: only micro-ops needed by
  `matmul` and `softmax`.
- Architecture research should be collected separately and should not directly
  become implementation work until reviewed through a spec change.

The AI then built a small Phase 0 software loop:

- Architecture spec validation.
- Minimal compiler from graph to JSON micro-ops.
- Functional simulator.
- CPU golden `matmul` and `softmax`.
- Unit tests and demo command.

The Phase 0 loop proved that the project can already perform:

```text
spec -> validate -> compile -> simulate -> golden compare -> tests
```

Validation result:

- `make validate-arch`: PASS.
- `make demo`: PASS.
- `make test`: PASS.

## Session 3: Human-Readable Spec, Research Refresh, And RTL

The user pointed out important gaps:

- The team rules needed to be more visible.
- The architecture spec must be readable by both humans and AI.
- Research refresh should be executable by humans.
- The project needs a real RTL hardware target, not only software simulation.

Responses and decisions:

- Added root-level mandatory agent rules.
- Changed the canonical architecture spec to JSONC so humans can review
  comments while tools still parse it deterministically.
- Added a reference refresh script that can query public research sources and
  write discovered references into the project.
- Added a minimal Phase 0 RTL implementation.

Current canonical architecture spec:

```text
arch/configs/npu_v0.jsonc
```

Current mandatory rules entry:

```text
AGENT_RULES.md
```

Current RTL target:

```text
hw/rtl/npu_v0_top.sv
```

The RTL is intentionally small:

- 8x8 INT8 matmul with INT32 output.
- 8-element softmax approximation with Q0.8 output.
- Simple host write/read interface.
- Start/done control.

The local environment did not have a SystemVerilog simulator installed, so RTL
simulation was prepared but not executed successfully in that environment.

Validation result after the spec and RTL updates:

- `make validate-arch`: PASS.
- `make demo`: PASS.
- `make test`: PASS.
- `make rtl-sim`: blocked by missing `iverilog`.

## Current Collaboration Workflow

The project uses a chief-architect style AI collaboration model:

1. The human owner sets goals and reviews architecture direction.
2. The AI turns goals into concrete project rules, specs, module boundaries,
   tests, and implementation steps.
3. Shared contracts are kept visible and versioned.
4. Implementation changes are accepted only when the relevant verification loop
   passes.
5. Research ideas remain candidates until promoted through a reviewed spec
   change.

## Working Principles

- Prefer a small verified system over a large speculative design.
- Make architecture decisions explicit in the spec.
- Keep human review and AI automation pointed at the same source of truth.
- Treat compiler, simulator, RTL, runtime, and tests as one system.
- Use PPA counters to justify complexity.

