# Work Rules For Module Agents

[TOC]

## 1. Architecture Spec Is The Contract

The hardware description file is the source of truth for the whole project.
For Phase 0, the canonical file is:

```text
arch/configs/npu_v0.jsonc
```

For SoC-level reset and address-map facts, the canonical file is:

```text
arch/configs/soc_v0.jsonc
```

For CPU-visible NPU-wrapper register and data-window offsets, the canonical
file is:

```text
arch/configs/npu_wrapper_v0.jsonc
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

No module agent may silently hard-code SoC base addresses, memory-region sizes,
or CPU reset-vector facts that belong in `arch/configs/soc_v0.jsonc`.

No module agent may silently hard-code NPU-wrapper register or window offsets
that belong in `arch/configs/npu_wrapper_v0.jsonc`.

Represent each fact once. Opcode tables, instruction bit fields, tensor IDs,
buffer IDs, memory-map constants, fixture paths, expected-output lengths, and
tolerances must be consumed from their canonical source or generated metadata,
not retyped independently in compiler, RTL, testbench, and tests.

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

For SoC-visible launch, register, bus, or NPU-wrapper behavior, `make soc-sim`
is part of the relevant verification loop.

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

## 7. Documentation Entry Points

The root `README.md` is the project entry point for new developers. Any change
that adds a key subsystem, command, verification flow, document, architecture
contract, or review-critical behavior must update the README with a short link
and description.

The active design entry points are:

```text
docs/architecture.md
docs/design/*.md
docs/target_architecture.md
docs/data_mover_a2.md
docs/matmul_array_a1.md
```

Historical notes belong under `docs/archive/` only when they still have useful
context. Do not keep duplicate active design documents that describe an older
version of the system.

## 8. Design-Before-Implementation Rule

Before implementing a non-trivial change, write the design intent into the
appropriate document under `docs/`.

The design note should be detailed enough that another developer can understand
the intended behavior before reading the patch. Include, as applicable:

- problem statement and scope;
- affected modules and ownership boundaries;
- interface or ABI changes;
- state-machine or timing changes;
- expected performance/cycle impact;
- verification plan and expected test commands;
- known limitations and follow-up work.

Use the most specific active document:

| Change area | Preferred document |
| --- | --- |
| SoC top, bus, memory map | `docs/design/soc_architecture.md` |
| Wrapper, descriptor path, data mover | `docs/design/npu_wrapper.md` |
| NPU core, uops, matmul/vector datapath | `docs/design/npu_core.md` |
| Firmware, compiler artifacts, CPU/NPU ABI | `docs/design/software_hardware_flow.md` |
| Perf counters, report schema, UI lanes | `docs/design/performance_instrumentation.md` |
| Tests and coverage | `docs/design/verification_strategy.md` |
| Long-term direction | `docs/target_architecture.md` |

After implementation, update the same document with what actually changed,
the observed test/perf result, and any gap between the plan and the result.
If the work changes a public entry point, update `README.md` and
`docs/README.md` as well.
