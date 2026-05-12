# Work Rules For Module Agents

## 1. Architecture Spec Is The Contract

The hardware description file is the source of truth for the whole project.
For Phase 0, the canonical file is:

```text
arch/configs/npu_v0.jsonc
```

The file uses JSONC instead of plain JSON so humans can review detailed
architecture comments while AI tools can still parse it deterministically after
comment stripping.

It must describe at least:

- ISA opcodes and operand constraints.
- Supported data types and accumulator types.
- MAC array shape and compute limits.
- Scratchpad and accumulator memory sizes.
- DMA channels, burst size, alignment, and transfer limits.
- Internal bus width and bandwidth assumptions.
- Vector/SFU lanes and supported micro-ops.
- Runtime memory map and launch registers.
- Verification tolerances.

No module agent may silently hard-code architecture facts that belong in this
file. If a value affects compiler legality, simulator behavior, RTL generation,
runtime layout, or verification, it must be represented in the spec.

## 2. Spec Change Protocol

Every architecture spec change must be treated as a system change.

Required steps:

1. Update the hardware spec.
2. Update spec validation if a new field or constraint is introduced.
3. If ISA changes, update compiler emission, simulator execution, RTL
   control/decode, tests, and documentation in the same change.
4. Update compiler target logic if scheduling, tiling, dtype, memory, or ISA
   behavior changes.
5. Update simulator semantics or latency model if hardware behavior changes.
6. Update RTL and RTL testbench if the change affects hardware-visible
   behavior.
7. Update runtime metadata if memory map or launch protocol changes.
8. Update golden tests or tolerances if numerical behavior changes.
9. Run the full test suite.
10. Record the reason for the change and the test result.

The minimum acceptance rule is simple: a spec change is not accepted unless the
closed-loop tests pass.

## 3. Minimal Current Scope

Keep the first NPU intentionally small. Phase 0 and Phase 1 only need the
micro-ops required for:

- `matmul`
- `softmax`
- `matmul -> softmax`

Initial micro-op set:

- `LOAD`
- `STORE`
- `MATMUL`
- `VREDMAX`
- `VSUB`
- `VEXP`
- `VREDSUM`
- `VDIV`
- `HALT`

Do not add complex operators, cache coherence, advanced NoC behavior, dynamic
shapes, sparsity, or quantization refinements until the minimal loop is stable.

## 4. Verification Rule

Every module must provide tests at the level it changes:

- ISA change: instruction validation tests.
- Compiler change: emitted program tests.
- Simulator change: instruction and end-to-end tests.
- Runtime change: launch and memory-layout tests.
- Hardware generator change: generated RTL interface and testbench checks.
- PPA change: counter and report consistency tests.

The first verification target is functional correctness. Cycle and PPA accuracy
come after the functional loop is stable.

## 5. Reference Research Rule

Architecture research references live under:

```text
references/
```

Each reference entry should contain:

- Source title.
- Link.
- Architecture topic.
- Key ideas.
- Potential use in this project.
- Current status: `candidate`, `adopted`, `rejected`, or `watch`.

Research ideas do not become implementation requirements until the chief
architect accepts them through a spec change.

## 6. Agent Coordination

Each module agent owns its implementation area, but cross-module contracts are
owned by the chief architect. Agents must not change shared contracts without
updating the affected downstream modules and tests.

Shared contracts:

- Hardware spec.
- ISA schema.
- Program format.
- Memory map.
- Runtime metadata.
- Trace format.
- Counter schema.
