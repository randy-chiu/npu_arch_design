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

## Session 4: Correcting RTL Abstraction Level

The user reviewed the RTL and found an important architecture violation: the
hardware had implemented a high-level softmax operation, while the spec defines
micro-ops such as `LOAD`, `STORE`, `VREDMAX`, `VSUB`, `VEXP`, `VREDSUM`, and
`VDIV`.

The team decision was that the user was correct. RTL must follow the spec's
micro-op contract, not bypass it with high-level behavior.

The RTL was changed to include a minimal instruction memory and micro-op
sequencer. The testbench now writes explicit programs:

```text
LOAD A -> spad_a
LOAD B -> spad_b
MATMUL
STORE acc -> C
HALT
```

and:

```text
LOAD X -> vec
VREDMAX
VSUB
VEXP
VREDSUM
VDIV
STORE vec -> Y
HALT
```

Validation result:

- `make rtl-sim`: PASS with `PASS npu_v0 RTL micro-op smoke tests`.

This reinforced a core project rule: implementation convenience must not
override the architecture spec.

## Session 5: Persistent Collaboration Records And FPGA Reality Check

The user asked that every future collaboration step be recorded so the project
can demonstrate not only the technical result but also the human/AI teamwork
process.

The rule was promoted into `AGENT_RULES.md`: meaningful collaboration turns
must update `docs/collaboration_journal.md` with goals, decisions, reasoning,
validation outcomes, and open risks. Secrets and patch-level noise must not be
recorded.

The user also asked whether the current RTL can already be compiled and burned
onto an FPGA board. The clarification was:

- The current RTL is a real micro-op-level hardware model and passes
  `make rtl-sim`.
- `iverilog` is a simulation compiler, not an FPGA synthesis/bitstream tool.
- Running on a physical FPGA still requires a board wrapper, constraints,
  vendor synthesis/place/route/bitstream generation, and a host/runtime path to
  load tensors and instruction memory.

The intended future board flow was captured:

```text
compiler emits instruction stream
host/runtime writes tensors and instructions into FPGA-visible memory
host/runtime asserts start
NPU executes micro-ops
host/runtime polls done and reads outputs
```

Open next step:

- Define a `phase0_fpga_min` milestone with an RTL assembler, program memory
  files, board wrapper, and vendor-specific project target.

## Session 6: Code Review Map, Test Entry, And Verification Naming

The user reported that the repository had grown enough to make code review
difficult. The requested response was not a patch to one module, but a review
map: a document that explains the current code structure, how the architecture
spec drives the graph-to-micro-op path, how the Python execution path compares
against CPU golden results, and how RTL simulation fits into later cycle-level
verification.

The AI added:

- `docs/code_structure_review.md`: a practical repository map with Mermaid
  flows for compiler/micro-op verification and RTL simulation.
- README links to the code structure review so developers can find it before
  reading implementation files.
- Documentation rules in `docs/work_rules.md` requiring future key subsystems,
  commands, verification paths, and review-critical behavior to update the
  README and code structure document.

The user then identified several review issues and corrected the AI's framing:

- `tests/inputs_matmul_softmax.json` existed but was not actually used by
  `tests/test_phase0.py`; the test had duplicated input data in code.
- The Python "functional simulator" name was too broad. It was really a
  compiler-emitted micro-op functional model, not a real RTL simulator.
- The Python micro-op model compared to CPU golden mainly verifies compiler
  lowering and abstract instruction semantics. It does not prove real RTL
  logic correctness.
- The user asked whether RTL functional verification existed and whether a
  simple version could be added immediately.

The AI changed the implementation accordingly:

- `tests/test_phase0.py` now reads `tests/graphs/matmul_softmax.json` and
  `tests/inputs_matmul_softmax.json` instead of duplicating the same graph and
  matrices in test code.
- `FunctionalSimulator` was renamed to `MicroOpFunctionalSimulator`, with a
  compatibility alias left behind for older callers.
- `rtl_fixture.py` was added to generate deterministic RTL fixtures from the
  shared graph/input files.
- `make rtl-fixtures` was added, and `make rtl-sim` now regenerates fixtures
  before running the SystemVerilog testbench.
- `tests/test_phase0.py` gained an optional RTL test that runs `make rtl-sim`
  when `iverilog` and `vvp` are installed.

The user then found another important testing flaw: although the test now read
the graph and inputs from JSON files, the expected golden result was still
hard-coded as:

```text
softmax(matmul(inputs["A"], inputs["B"]))
```

The user also asked whether `tests/test_phase0.py` is intended to be the whole
test entry point, because they prefer to start code review from a top-level
test file and follow the high-level API behavior from there.

The resolution was:

- `tests/test_phase0.py` is the Phase 0 unified top-level test entry.
- `make test` runs it through unittest discovery.
- The file is organized by verification level:
  - `ArchitectureAndGoldenTests`
  - `CompilerMicroOpFunctionalTests`
  - `RTLFunctionalTests`
- Future cycle-level validation should be added as another top-level test class
  in this same entry, for example `CycleModelTests`.
- Lower-level module tests may be added later, but this file should remain the
  readable top-level path for developers who want to understand the project by
  following tests.

The hard-coded golden formula was removed. The compiler/micro-op test, renamed
to `test_compiler_micro_ops_match_graph_golden`, now uses a graph-driven golden
helper that walks `graph["ops"]` and computes expected tensors according to
each op's own input/output fields. This means changes to tensor names or graph
order are reflected in the expected output instead of being hidden behind a
stale test formula.

Validation result:

- `make test`: PASS, including the optional RTL simulation test in the local
  environment where `iverilog` and `vvp` were available.

Remaining limitation:

- RTL functional verification currently covers standalone 8x8 matmul and
  standalone 8-element softmax. It does not yet execute the full
  `matmul -> softmax` graph inside RTL as a single chained program.

## Canonical Encoding And Fixture Metadata Tightening

The user identified another source-of-truth violation:

- `rtl_fixture.py` carried a separate hard-coded opcode table instead of reading
  opcode and operand encodings from the architecture spec.
- `hw/tb/npu_v0_tb.sv` hard-coded generated fixture paths and expected-output
  lengths, including a fixed 64-element matmul comparison.
- The same ISA facts appeared independently in the spec, Python encoder, and
  RTL decode constants.

The fix was to make `arch/configs/npu_v0.jsonc` carry the Phase 0 binary
micro-op encoding, including opcode field bits, arg field bits, opcode values,
tensor IDs, buffer IDs, and buffer aliases. `rtl_fixture.py` now consumes this
encoding directly when producing binary uops.

The RTL fixture generator now also emits generated SystemVerilog include files:

- `npu_v0_spec.svh`: opcode, tensor, buffer, instruction field, and RTL tile
  constants derived from the spec.
- `npu_v0_tb_params.svh`: generated fixture file paths and output comparison
  lengths derived from the produced fixture artifacts.

`npu_v0_top.sv` and `npu_v0_tb.sv` include those generated files instead of
duplicating opcode constants, tensor IDs, buffer IDs, fixture paths, and output
lengths. `Makefile` passes the fixture directory as an include path during RTL
simulation.

Project rules were also tightened in `AGENT_RULES.md` and `docs/work_rules.md`:
each architecture fact must have one canonical representation, and downstream
compiler/RTL/testbench constants must be generated or consumed from that source
instead of being retyped independently.

Validation result:

- `make validate-arch`: PASS.
- `make demo`: PASS.
- `make test`: PASS.
- `make rtl-sim`: PASS.

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

## Session 8: Minimal SoC Bring-Up Planning

The user raised an important system-level gap in the current RTL validation
path: `npu_v0_top` is launched by the testbench directly toggling `start`, not
by software writing a control register through a CPU-visible bus. That is enough
for early RTL smoke tests, but it does not prove the hardware/software control
loop needed for FPGA bring-up or later real workloads.

The agreed direction is to add a minimal SoC around the NPU:

```text
RISC-V CPU softcore
  -> simple memory-mapped bus
  -> ROM/SRAM/debug status
  -> opsched
  -> npu_v0_top
```

The CPU should be reused from an existing open-source core instead of designed
locally. PicoRV32 is the first candidate because it is small, open-source, and
offers simple native memory, AXI4-Lite, and Wishbone-style integration options.
The CPU compiler should also be reused, using a bare-metal RISC-V GNU toolchain
such as `riscv32-unknown-elf-gcc` or an RV32-capable
`riscv64-unknown-elf-gcc`.

The CPU/NPU boundary module was renamed from a generic "NPU MMIO wrapper" to
`opsched`, meaning operator scheduler. For the first implementation, `opsched`
is intentionally thin:

- expose CPU-visible control/status registers;
- translate CPU MMIO tensor/program accesses into the current NPU host
  interface;
- generate a one-cycle NPU `start` pulse from a `CTRL.start` register write;
- expose `done`, `busy`, and `idle` status to CPU firmware.

The naming leaves room for later growth into a real operator/job scheduler with
command queues, descriptors, interrupts, DMA launch, and performance counters.

The project directory plan was updated to converge toward a clearer
`sw/hw/docs/test/build` top-level structure:

- `hw/soc`: CPU subsystem, bus, memories, debug peripherals, SoC top;
- `hw/npu/rtl`: NPU core RTL;
- `hw/npu/opsched`: CPU-visible NPU operator scheduler and register interface;
- `sw/cpu`: boot code, linker scripts, bare-metal NPU driver, CPU firmware
  tests;
- `sw/npu`: graph compiler, operator lowering, uop assembler, NPU program
  metadata;
- `sw/tools`: toolchain scripts and third-party tool notes;
- `test`: graph tests, golden models, RTL tests, SoC tests;
- `build`: generated artifacts only.

The SoC bring-up plan was documented in `docs/soc_bringup.md`, including:

- a hardware architecture diagram;
- a separate program/data-flow diagram;
- hardware component specifications for CPU, bus, ROM, SRAM, `opsched`, NPU
  core, optional UART, and test status register;
- software components including boot stub, linker script, CPU-side NPU driver,
  RISC-V toolchain, NPU graph compiler, uop assembler, and golden model;
- open-source reuse plan for PicoRV32 and the RISC-V GNU toolchain;
- a staged verification plan: `opsched` unit test, assembler artifact
  unification, CPU-controlled matmul, CPU-controlled softmax, compiler-fed
  firmware, then FPGA candidate cleanup.

The immediate next implementation milestone is:

1. create `opsched` RTL around the existing `npu_v0_top`;
2. verify it can launch the NPU only through register writes;
3. refactor the uop encoder into a reusable assembler artifact;
4. add the minimal CPU SoC simulation path after the `opsched` boundary is
   stable.
