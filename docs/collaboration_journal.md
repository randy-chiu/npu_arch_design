# Collaboration Journal

[TOC]

This journal records the project process and AI collaboration flow. It avoids
low-level patch history and focuses on goals, decisions, reasoning, and team
workflow.

## Current Project Snapshot

本章节是重启 Codex 或重新打开项目时的恢复入口。开始继续工作前，优先阅读：

1. 本章节；
2. `README.md`；
3. `docs/architecture.md`；
4. `docs/code_structure_review.md`；
5. `docs/bugfix_list.md`。

### 当前目标

当前项目目标是构建一个小型 NPU-centered SoC，用来做 NPU 架构验证闭环，而不是
只验证单独 NPU core。最小闭环应该覆盖：

- CPU 从 reset vector 取指启动；
- CPU firmware 通过 NPU wrapper MMIO 控制 NPU；
- tensor input/output、NPU program stream、job descriptor 放在 SRAM；
- NPU wrapper 根据 descriptor 从 SRAM fetch program/input，驱动 NPU core；
- NPU core 执行 `matmul`、`softmax`；
- wrapper 写 output 回 SRAM；
- CPU firmware 校验结果并写 `test_status`；
- Python/unit test、NPU core RTL sim、SoC sim、CPU-controlled SoC sim 都能跑通。

### 当前已完成

- 顶层目录已按 SoC/NPU wrapper/NPU core/CPU/software/tools/test/docs 方向整理。
- `docs/architecture.md` 是当前架构入口。
- `docs/code_structure_review.md` 记录 RTL 和软件路径走读。
- `arch/configs/soc_v0.jsonc` 是 SoC memory map、CPU reset/stack、boot image、
  CPU/NPU descriptor ABI 的 source of truth。
- `arch/configs/npu_wrapper_v0.jsonc` 是 NPU wrapper register map 的 source of
  truth。
- `arch/configs/npu_v0.jsonc` 是当前 NPU core ISA/uop/tensor/tile 配置的 source
  of truth。
- `make soc-spec` 生成 `soc_v0_addr.h`、`soc_v0_addr.svh`、`soc_v0.ld`。
- `make npu-wrapper-spec` 生成 wrapper C/SV register headers。
- PicoRV32 已接入 `soc_cpu_top`。
- `cpu-soc-sim` 已经使用真实 C/ASM firmware 路径，firmware 会把 matmul/softmax
  input、program、descriptor 放入 SRAM，然后通过 `DESC_ADDR` + `CTRL.start`
  启动 wrapper。
- NPU wrapper 已有 descriptor/SRAM fetch/writeback 状态机，同时保留 legacy
  wrapper-window path 供 `soc-sim` 覆盖。
- descriptor ABI 已经从 `arch/configs/soc_v0.jsonc` 单源生成，C 和 RTL 不再各自
  手写字段顺序/op id。
- operator/compiler/assembler 初步分层已落地：
  `sw/npu_core/operators/phase0_intrinsics.json` 描述 matmul/softmax 的 Phase 0
  ISA/uop intent，`sw/tools/npu_compiler/phase0.py` 负责 graph/operator lowering，
  `sw/tools/npu_assembler/phase0.py` 负责 uop encoding，`sw/tools/npu_phase0`
  保留为兼容入口。

### 当前验证状态

最近一次完整验证结果：

```text
make cpu-soc-sim: PASS
make soc-sim: PASS
make test: PASS, 8 tests
```

新增 cycle 级性能报告入口：

```text
make perf-report
```

它运行 CPU-controlled SoC simulation，采集每个 NPU job 的 wrapper/core cycle
分解和 CPU/wrapper/core timeline，并生成：

```text
build/perf/perf.json
build/perf/perf_report.html
```

当前 perf 现状、timeline 语义、限制和后续扩展点记录在：

```text
sw/tools/perf/README.md
```

长期 NPU 目标架构、业界资料提炼、matmul array 化路线记录在：

```text
docs/target_architecture.md
```

本地 RISC-V GCC 曾使用：

```text
thirdparty/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-gcc
```

如果 shell 没有继承 `.bashrc`，运行 CPU firmware 相关目标前需要确保该工具链在
`PATH` 中。

### 当前设计约束

- 当前 boot ROM 是仿真简化模型，里面放完整 firmware image，不是最终真实 SoC
  的小 bootloader 模型。
- SRAM 当前承担 stack、locals、descriptor、runtime tensor/program buffers。
- NPU wrapper 的 SRAM 访问是简单第二端口模型，不是完整 DMA/crossbar。
- NPU core 内部仍使用 host-loadable memories，不是真正 streaming datapath。
- 当前 matmul/softmax 固定在 Phase 0 小规模 ISA/uop 模型。
- `soc-sim` 是 legacy wrapper-window path；`cpu-soc-sim` 是当前更重要的
  firmware descriptor/SRAM path。

### 下一步计划

短期下一步：

1. 继续完善 operator template schema，增加字段校验，避免 template 写错后到
   RTL/firmware 阶段才暴露。
2. 把 `npu_phase0` compatibility package 中剩余的 Phase 0 专用职责逐步迁到
   `npu_compiler`、`npu_assembler`、simulator/golden 等更清晰目录。
3. 让固定 operator program artifact 在合适时进入 `sw/npu_core/programs`，区分
   “operator intent/source”和“已编码 program artifact”。
4. descriptor 增加错误码、状态码、更清晰的 job validation。

中期计划：

- wrapper SRAM fetch 从固定 op 特判逐步演进为更通用的 program/data movement。
- 引入更真实的 NPU core memory/streaming 接口，减少 wrapper 对 core 内部 host
  window 的依赖。
- 加 command queue、IRQ、performance counter、timeout。

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

## Session 9: Minimal SoC Skeleton And Directory Semantics Correction

The user asked to move from standalone NPU-core validation toward a small SoC
validation loop. The first implementation step built a minimal hardware
skeleton around the existing NPU core:

- `opsched`, a thin CPU-visible NPU wrapper that exposes control/status
  registers and tensor/program windows;
- a simple 32-bit local bus decoder;
- ROM, SRAM, and simulation test-status peripheral wrappers;
- a `soc_top` shell with a CPU-side local-bus master port;
- a SoC smoke testbench that acts like firmware by writing tensor inputs and
  program words through the `opsched` MMIO window, launching the NPU, polling
  done, and checking output data.

Validation result:

- `make test`: PASS, including the existing NPU RTL simulation and the new
  SoC `opsched` smoke test.

The user then corrected an important repository semantics issue. The previous
directory plan used `sw/npu` for compiler, assembler, and other host/tooling
code. That was ambiguous and wrong for this project. The corrected split is:

- `sw/soc_cpu`: software that executes on the SoC CPU, including NPU-wrapper
  drivers, runtime, boot code, and firmware apps;
- `sw/npu_core`: code or programs consumed by the NPU core itself, such as
  operator programs and later NPU-side operator implementations;
- `sw/tools`: software tools that run on the development host, including CPU
  toolchain integration, NPU graph compiler, NPU assembler, fixture generation,
  and simulators.

Hardware should also be separated by module role under `hw`:

- `hw/soc`: SoC top, bus, memory, debug peripherals, and later CPU integration;
- `hw/npu_wrapper`: CPU-visible wrapper/scheduler around the NPU;
- `hw/npu_core`: the NPU core RTL and core-level testbench.

The user also clarified that CPU linker scripts are only needed once real
bare-metal firmware is introduced. They should not be created prematurely.
Temporary fixture generation remains useful for RTL/SoC simulation, but fixture
artifacts are test infrastructure, not product software. Generated fixture hex
files should stay under `build`, while `test/fixtures` can document this
temporary verification bridge.

Documentation cleanup decision:

- keep `docs/work_rules.md` as the core co-work rules;
- keep and continuously update `docs/collaboration_journal.md`;
- keep `docs/soc_bringup.md` and `docs/fpga_bringup.md`;
- merge `docs/roadmap.md` and `docs/agent_plan.md` into a single project plan;
- archive early or superseded design notes instead of deleting them;
- update the code structure review so it matches the real tree.

## Session 10: Detailed SoC Test Flow Documentation

The user asked for a clearer description of the complete current test flow:
compile a graph into NPU micro-ops, encode the micro-op stream, launch work
through the NPU wrapper, let the NPU core run, read outputs, and check results
in the testbench.

The clarification was added to `docs/code_structure_review.md` under
"Complete SoC Smoke Test Flow". The document now explains:

- `make soc-sim` first runs `rtl-fixtures`;
- fixture generation reads the architecture config, graph JSON, and input JSON;
- the matmul subgraph is compiled to JSON micro-ops and encoded into 32-bit
  RTL uops;
- generated hex and SystemVerilog include files are written under
  `build/rtl_fixture`;
- `hw/soc/tb/soc_tb.sv` acts as the temporary CPU model by driving the
  `soc_top` CPU-side bus port;
- the testbench writes Matrix A, Matrix B, and program words through the
  `opsched` MMIO window;
- `opsched` translates CPU-visible byte offsets into the NPU core's existing
  word-addressed host interface;
- writing `CTRL.start` creates a one-cycle `start_pulse` into `npu_v0_top`;
- the testbench polls `STATUS.done`, reads Matrix C, and compares each value
  against `matmul_expected_c.hex`.

The document also calls out the current limitation: SoC smoke verification
currently covers wrapper-controlled matmul only. The fixture generator and
standalone NPU core testbench already cover softmax, but the SoC testbench has
not yet launched softmax through `opsched`. The next useful SoC verification
step is to add the softmax path to `hw/soc/tb/soc_tb.sv`.

## Session 11: Add SoC Softmax Wrapper Verification

The user asked to fill the missing SoC-level softmax verification and asked
whether the older verification targets are still needed now that `soc-sim`
exists.

Implementation change:

- `hw/soc/tb/soc_tb.sv` now runs two wrapper-controlled launches:
  1. matmul through the `opsched` A/B/program/C windows;
  2. softmax through the `opsched` X/program/Y windows.
- The softmax path loads `softmax_x.hex`, `softmax_program.hex`, and
  `softmax_expected_y.hex`, writes X and program words through the SoC bus,
  writes `CTRL.start`, polls `STATUS.done`, reads Y, and compares each low byte
  against the expected output.

Validation result:

- `make soc-sim`: PASS.

Verification target decision:

- Keep `make validate-arch`, `make demo`, `make rtl-sim`, `make soc-sim`, and
  `make test` for now because they cover different failure scopes.
- `validate-arch` catches contract/schema issues quickly.
- `demo` gives a readable compiler/simulator smoke path.
- `rtl-sim` isolates NPU core behavior without SoC bus or wrapper variables.
- `soc-sim` verifies the bus, NPU wrapper launch protocol, and NPU core
  together.
- `make test` remains the aggregate developer check.

The code structure review was updated with this rationale and with the new
SoC softmax sequence. The remaining SoC gap is no longer "softmax through
wrapper"; it is a firmware-like chained graph flow where matmul output becomes
softmax input through runtime-managed tensor metadata.

## Session 12: Rename Standalone Core Simulation Target

The user pointed out that `rtl-sim` was too generic now that the project has
both standalone NPU-core simulation and SoC-level simulation. The agreed naming
is:

- `make npu-core-sim`: standalone NPU core RTL simulation;
- `make soc-sim`: SoC bus + NPU wrapper + NPU core simulation.

`Makefile` now exposes `npu-core-sim` as the primary target. `rtl-sim` remains
as a compatibility alias for existing habits or older notes. Active docs and
tests were updated to use `npu-core-sim`.

## Session 13: Share Wrapper Register Map With SoC Testbench

The user reviewed `hw/soc/tb/soc_tb.sv` and pointed out that the testbench was
hard-coding offsets for the NPU wrapper address space, even though those
offsets are owned by the wrapper register map.

The fix was to include `hw/npu_wrapper/rtl/npu_v0_regs.svh` from `soc_tb`.
The testbench now defines only the SoC-level `OPSCHED_BASE` and derives full
addresses from wrapper-owned constants such as `NPU_OPSCHED_CTRL`,
`NPU_OPSCHED_A_BASE`, `NPU_OPSCHED_X_BASE`, and
`NPU_OPSCHED_PROGRAM_BASE`.

The code structure review was also extended with Mermaid diagrams showing:

- the SoC connection graph from the temporary CPU bus master through
  `soc_top`, `simple_bus`, `npu_v0_opsched`, and `npu_v0_top`;
- how a CPU-visible program-window address is decoded by the SoC bus and then
  translated by `opsched` into the NPU core's word-addressed host interface.

## Session 14: Add PicoRV32 CPU-Controlled SoC Simulation

The user agreed that it was time to add a CPU to the minimal SoC. The project
kept the earlier architectural decision not to design a local CPU from scratch.
Instead, PicoRV32 was vendored under:

```text
hw/soc/cpu/third_party/picorv32/picorv32.v
```

New hardware integration:

- `hw/soc/cpu/rtl/picorv32_native_cpu.sv` adapts PicoRV32's native memory
  interface to the project local bus.
- `hw/soc/rtl/soc_cpu_top.sv` integrates PicoRV32, boot ROM, SRAM, test status,
  `opsched`, and `npu_v0_top`.
- `hw/soc/tb/soc_cpu_tb.sv` runs the CPU-controlled SoC simulation and watches
  the simulation test-status register.

The local environment did not have `riscv32-unknown-elf-gcc` or an RV32-capable
`riscv64-unknown-elf-gcc`, so the first firmware path uses a temporary host
tool instead of C firmware:

```text
sw/tools/firmware/emit_soc_cpu_smoke.py
```

This tool emits `build/firmware/soc_cpu_smoke.hex`, a small RV32I boot ROM
program. The firmware performs the same MMIO flow as the previous direct-bus
SoC test:

1. write matmul A/B and matmul program through the `opsched` windows;
2. write `CTRL.start`;
3. poll `STATUS.done`;
4. read and check Matrix C;
5. write softmax X and softmax program;
6. launch and poll again;
7. read and check softmax Y;
8. write `0x0000_0001` to the simulation test-status register on success, or
   `0xffff_ffff` on failure.

New make target:

```text
make cpu-soc-sim
```

Validation result:

- `make cpu-soc-sim`: PASS with
  `PASS PicoRV32 firmware-controlled SoC smoke test`.

Remaining work:

- Replace the temporary Python RV32I firmware emitter with normal C firmware,
  startup code, a linker script, and a RISC-V GCC build once the toolchain is
  available.

## Session 15: Clarify CPU Testbench Startup And Timeout

The user asked how `soc_cpu_tb` starts the CPU, whether the `repeat (20000)`
loop is the normal way to run CPU tests, and where the CPU fetches instructions
before writing the NPU wrapper `CTRL` register.

Clarification and updates:

- `soc_cpu_tb` now uses a named `CPU_SOC_TIMEOUT_CYCLES` constant instead of a
  raw `repeat (20000)`.
- The repeat loop is only a simulation watchdog; it does not start the CPU.
- The CPU starts when `soc_cpu_tb` releases reset by driving `rst_n` high.
- PicoRV32 starts at `PROGADDR_RESET = 0x0000_0000`.
- `soc_cpu_top` maps `0x0000_0000` to `boot_rom`.
- `boot_rom` is initialized from `build/firmware/soc_cpu_smoke.hex`.
- The firmware instructions then perform ordinary RV32I stores and loads to
  the memory-mapped `opsched` address range, including the write to
  `CTRL.start`.

`docs/code_structure_review.md` was updated with a CPU-controlled test sequence
section explaining reset, boot ROM instruction fetch, MMIO routing, NPU launch,
status polling, output checking, and the role of the timeout watchdog.

## Session 16: Align NPU Wrapper Naming And Add Boot Flow Diagram

The user noticed that active RTL still used many `opsched` names even though
the higher-level module boundary is now called `npu_wrapper`. The naming
decision is:

- keep `npu_v0_opsched` as the wrapper-internal scheduler module name for now,
  because it describes the block's function;
- use `npu_wrapper_*` for SoC bus signals and top-level instance names, because
  those describe the SoC boundary.

RTL updates:

- `simple_bus` now exposes `npu_wrapper_req/we/addr/wdata/rdata/ready`.
- `soc_top` and `soc_cpu_top` use `npu_wrapper_*` interconnect signals.
- the wrapper instance name changed from `u_opsched` to `u_npu_wrapper`.

Documentation updates:

- active docs now include `[TOC]` markers for easier navigation;
- `docs/code_structure_review.md` now includes a Mermaid flow from
  `soc_cpu_tb` clock/reset through PicoRV32 reset release, boot ROM instruction
  fetch at `0x0000_0000`, firmware execution, the first store to
  `0x1000_0000`, and the resulting `start_pulse` into `npu_v0_top`.

## Session 17: Generate Readable RV32I Firmware Assembly

The user asked for a readable RISC-V assembly view of the generated CPU boot
firmware, because the existing `emit_soc_cpu_smoke.py` path only emitted raw
machine-code hex.

The firmware emitter now generates both:

```text
build/firmware/soc_cpu_smoke.hex
build/firmware/soc_cpu_smoke.S
```

Both files come from the same `Program` builder in
`sw/tools/firmware/emit_soc_cpu_smoke.py`, so the readable `.S` listing mirrors
the boot ROM image used by `cpu-soc-sim`.

The generated assembly makes the important CPU/MMIO behavior visible:

- stores tensor/program words into NPU wrapper windows;
- stores `1` to `0x1000_0000` to write `CTRL.start`;
- loops on `0x1000_0004` until `STATUS.done` is set;
- reads output windows and compares expected values;
- writes `0x0000_0001` or `0xffff_ffff` to `0x3000_0000` for pass/fail.

`docs/code_structure_review.md` now points to the generated `.S` file and
includes key assembly snippets for matmul launch, softmax launch, and final
pass/fail reporting. `sw/soc_cpu/firmware_smoke/README.md` documents the
generated firmware artifacts for source-tree readers.

Validation result:

- `make cpu-soc-sim`: PASS.
- `make test`: PASS.

## Session 18: Add SoC Memory Map Spec

The user pointed out that SoC-level addresses were still hard-coded in several
places, especially `OPSCHED_BASE = 0x1000_0000` in the temporary firmware
emitter. The project now has a dedicated SoC spec:

```text
arch/configs/soc_v0.jsonc
```

It defines:

- CPU reset vector;
- CPU stack pointer;
- boot ROM base and size;
- SRAM base and size;
- NPU wrapper base and size;
- reserved UART window;
- simulation test-status base and size.

`make soc-spec` generates:

```text
build/soc/soc_v0_addr.svh
```

`simple_bus.sv` and `soc_tb.sv` consume the generated include instead of
hard-coding SoC base addresses. `emit_soc_cpu_smoke.py` now reads
`arch/configs/soc_v0.jsonc` for SoC base addresses before generating boot ROM
hex and assembly.

The NPU wrapper internal offsets are still owned by:

```text
hw/npu_wrapper/rtl/npu_v0_regs.svh
```

The review document was updated to describe the full SoC memory map and the
pass/fail path. CPU firmware writes `CTRL.start` and polls `STATUS.done` in the
NPU wrapper, but final smoke-test pass/fail is reported by writing the separate
`test_status` peripheral at `0x3000_0000`, which `soc_cpu_tb` observes through
the top-level `sim_status` signal.

## Session 19: Start Real CPU Firmware Flow

The user started downloading a RISC-V GCC toolchain and asked to begin the
formal firmware-code path in parallel.

The project now has a real bare-metal PicoRV32 firmware layout:

```text
sw/soc_cpu/boot/start.S
sw/soc_cpu/runtime/npu_driver.c
sw/soc_cpu/runtime/npu_driver.h
sw/soc_cpu/apps/soc_cpu_smoke/main.c
sw/soc_cpu/linker/soc_v0.ld
```

The firmware startup sets the stack pointer from generated SoC metadata and
calls `main()`. The C smoke app uses an NPU-wrapper driver to:

- write matmul inputs and micro-op program words;
- write `CTRL.start`;
- poll `STATUS.done`;
- compare Matrix C;
- write softmax input and program words;
- launch and poll the NPU again;
- compare Softmax Y low bytes;
- report pass/fail through the separate `test_status` peripheral.

The NPU wrapper register map is no longer duplicated between RTL, firmware,
and host tooling. A new canonical spec was added:

```text
arch/configs/npu_wrapper_v0.jsonc
```

`make npu-wrapper-spec` generates:

```text
build/npu_wrapper/npu_v0_regs.svh
build/npu_wrapper/npu_v0_regs.h
```

`make soc-spec` now also generates:

```text
build/soc/soc_v0_addr.h
```

The Makefile now prefers real C/ASM firmware when one of these toolchains is
available:

```text
riscv-none-elf-gcc
riscv32-unknown-elf-gcc
riscv64-unknown-elf-gcc
```

The firmware is compiled as RV32I/ILP32:

```text
-march=rv32i -mabi=ilp32
```

If no toolchain is present, `make firmware-smoke` keeps using the temporary
Python RV32I emitter so `make cpu-soc-sim` remains runnable while the local
toolchain is being installed.

Validation result in the no-GCC environment:

- `make soc-sim`: PASS.
- `make cpu-soc-sim`: PASS through the fallback emitter.

## Session 20: Verify Real GCC-Built Firmware

The user installed the xPack RISC-V GCC toolchain under:

```text
thirdparty/xpack-riscv-none-elf-gcc-15.2.0-1/bin
```

The first `make firmware-smoke-c` build succeeded and produced:

```text
build/firmware/soc_cpu_smoke.elf
build/firmware/soc_cpu_smoke.bin
build/firmware/soc_cpu_smoke.hex
build/firmware/soc_cpu_smoke.dump
build/firmware/soc_cpu_smoke.map
```

The first `make cpu-soc-sim` run with real C/ASM firmware failed with a firmware
mismatch. Debug showed:

- C firmware successfully wrote NPU input/program windows;
- NPU core computed the correct first matmul result internally;
- CPU readback through the wrapper returned the correct value;
- the value was lost when C code stored the readback into a stack local.

The root cause was the SoC SRAM address map. `simple_bus` currently decodes
regions with a power-of-two mask, so a 128 KiB SRAM region must be aligned to
128 KiB. The old map used:

```text
base = 0x0001_0000
size = 0x0002_0000
stack = 0x0002_fff0
```

That is not aligned for the mask decoder. Stack accesses near `0x0002_fff0`
therefore missed SRAM and returned default zero.

The SoC spec was corrected to:

```text
SRAM base = 0x0002_0000
SRAM size = 0x0002_0000
stack     = 0x0003_fff0
```

`sw/soc_cpu/linker/soc_v0.ld` was updated to match. `simple_sram.sv` was also
changed to provide same-cycle combinational reads with synchronous writes,
matching the current simple local bus ready protocol.

The firmware smoke app now reports high-bit failure codes through
`test_status_fail_code()`, and `soc_cpu_tb` treats any status with bit 31 set
as a firmware failure while printing the code.

Validation result with real GCC-built firmware:

- `make firmware-smoke-c`: PASS.
- `make cpu-soc-sim`: PASS.
- `make soc-sim`: PASS.
- `make test`: PASS, 7 tests.

## Session 21: Add Bugfix List

The user asked for a dedicated docs file to record representative bugs and
their solutions.

Added:

```text
docs/bugfix_list.md
```

The first entry documents the GCC-built firmware mismatch caused by the
misaligned SRAM base address. It records the symptom, debug path, root cause,
fix, verification result, and follow-up rule that SoC mask-decoded regions must
be aligned to their size.

The user then asked for the bugfix list to use Chinese descriptions. The file
was rewritten in Chinese while preserving the same technical structure and
details.

## Session 22: Generate Linker Script From SoC Spec And Add RTL Walkthrough

The user noticed that `sw/soc_cpu/linker/soc_v0.ld` still hard-coded ROM/SRAM
addresses, which violated the SoC source-of-truth rule.

The linker script is now generated by `make soc-spec` from:

```text
arch/configs/soc_v0.jsonc
```

Generated outputs are now:

```text
build/soc/soc_v0_addr.svh
build/soc/soc_v0_addr.h
build/soc/soc_v0.ld
```

`make firmware-smoke-c` links with:

```text
-T build/soc/soc_v0.ld
```

The old handwritten `sw/soc_cpu/linker/soc_v0.ld` was removed. SoC spec
generation now validates memory-map regions before emitting generated files:

- `size_bytes` must be a positive power of two;
- `base` must be aligned to `size_bytes`.

`docs/code_structure_review.md` was also expanded with a Chinese RTL walkthrough
for readers who are not familiar with RTL. The walkthrough follows the actual
path:

```text
soc_cpu_tb
  -> soc_cpu_top
  -> picorv32_native_cpu
  -> simple_bus
  -> boot_rom / simple_sram / npu_v0_opsched / test_status
  -> npu_v0_top
```

It explains reset, CPU instruction fetch from address `0x0000_0000`, firmware
MMIO stores to `CTRL.start`, wrapper-to-core `start_pulse`, NPU program
execution, `STATUS.done` polling, result readback, and final pass/fail reporting
through `test_status`.

Validation result:

- `make firmware-smoke-c`: PASS.
- `make cpu-soc-sim`: PASS.
- `make soc-sim`: PASS.

## Session 23: Clarify CPU SoC RTL Mental Model

The user restated their understanding of `soc_cpu_tb` and the CPU-controlled
SoC flow and asked for corrections plus documentation updates.

Clarifications added to `docs/code_structure_review.md`:

- `soc_cpu_tb` instantiates `soc_cpu_top`, creates `clk`, drives reset, and
  observes `sim_status`/`cpu_trap`; it does not directly operate the NPU
  wrapper.
- CPU execution starts when clock exists and reset is released. Clock alone is
  not enough.
- `soc_cpu_top` instantiates CPU, bus, boot ROM, SRAM, NPU wrapper, and
  `test_status`. These RTL modules already exist after elaboration; they
  respond when CPU bus signals select them.
- CPU reset vector is `0x0000_0000`, which the bus decodes to boot ROM.
- Current `soc_cpu_smoke.hex` is a full simulation firmware image, including
  startup code, NPU driver, `main()`, and generated test data. It is not only a
  tiny boot ROM stub.
- This is a bring-up simplification. A more production-like SoC may have a
  small boot ROM that loads user firmware from flash or another non-volatile
  image into SRAM/DRAM before jumping to it. That flow is not modeled yet.
- `soc_cpu_top.sv` does not directly instantiate `npu_v0_top` because the NPU
  core is instantiated inside `npu_v0_opsched` as `u_npu`. The SoC top connects
  only to the CPU-visible NPU wrapper.

`sw/soc_cpu/firmware_smoke/README.md` was also updated to clarify the current
boot ROM image scope.

## Session 24: Clarify ROM/SRAM Roles And Future NPU Fetch Model

The user asked to add comments around boot ROM and SRAM usage and pointed out
that the current smoke firmware writes matmul input tensors and NPU program
words directly into NPU wrapper windows. The user expects the future model to
place operator inputs/outputs and NPU program streams in SRAM, then let the NPU
wrapper/core fetch them by address.

Clarifications added:

- `arch/configs/soc_v0.jsonc` now describes current boot ROM as a full
  simulation firmware image location, not only a tiny boot stub.
- `hw/soc/rtl/mem/boot_rom.sv` comments explain that current `INIT_HEX`
  includes startup code, NPU driver, `main()`, and generated test data. ROM
  size growth must be watched.
- `hw/soc/rtl/mem/simple_sram.sv` comments explain that current SRAM is writable
  CPU data memory for stack and locals, and future tensor/program
  buffers/descriptors should live there.
- `sw/soc_cpu/apps/soc_cpu_smoke/main.c` comments mark the direct wrapper-window
  preload as a Phase 0 bring-up shortcut.
- `docs/code_structure_review.md`, `docs/soc_bringup.md`, and
  `sw/soc_cpu/firmware_smoke/README.md` now document the target address-based
  flow:

```text
CPU firmware writes tensor buffers/program descriptors to SRAM
  -> CPU writes descriptor/program address to NPU wrapper
  -> CPU writes CTRL.start
  -> NPU wrapper/core fetches data and program words from SRAM
  -> NPU writes outputs back to SRAM
  -> CPU checks SRAM outputs
```

Ownership split recorded:

- CPU firmware/tests own runtime input generation and SRAM buffer allocation.
- NPU compiler/assembler owns operator instruction stream generation.
- `sw/npu_core/programs` and `sw/npu_core/operators` hold NPU-consumed program
  descriptions or operator code when checked in as design source.
- `sw/tools` holds host-side compiler, assembler, simulator, and fixture
  generators.

## Session 25: Implement Descriptor/SRAM CPU-NPU Launch Path

The user approved starting the descriptor/SRAM interaction cleanup before
expanding the NPU hardware architecture.

Implemented a minimal descriptor-driven launch path while retaining the legacy
wrapper-window path:

- Added `DESC_ADDR` at NPU wrapper offset `0x020` in
  `arch/configs/npu_wrapper_v0.jsonc`.
- Updated generated/fallback wrapper register headers to expose
  `NPU_OPSCHED_DESC_ADDR`.
- Extended `simple_sram.sv` with a second NPU port. The CPU port remains used
  through `simple_bus`; the wrapper uses the second port to fetch descriptors,
  programs, and input data and to write output data.
- Wired the NPU SRAM port through `soc_cpu_top.sv`.
- Updated `npu_v0_opsched.sv` with a descriptor FSM:

```text
DESC_READ
  -> DESC_FETCH_PROGRAM
  -> DESC_FETCH_INPUT0
  -> DESC_FETCH_INPUT1
  -> DESC_START_CORE
  -> DESC_WAIT_CORE
  -> DESC_WRITE_OUTPUT
  -> DESC_DONE
```

The descriptor layout used by firmware is:

```c
struct npu_job_desc {
    uint32_t op_type;        // 1 = matmul, 2 = softmax
    uint32_t program_addr;
    uint32_t program_words;
    uint32_t input0_addr;
    uint32_t input0_words;
    uint32_t input1_addr;
    uint32_t input1_words;
    uint32_t output_addr;
    uint32_t output_words;
};
```

Updated `sw/soc_cpu/apps/soc_cpu_smoke/main.c` so the GCC-built firmware now:

1. copies matmul inputs and program stream to SRAM buffers;
2. fills a descriptor in SRAM;
3. writes `DESC_ADDR`;
4. writes `CTRL.start`;
5. waits for `STATUS.done`;
6. checks output buffers in SRAM;
7. repeats the flow for softmax.

The direct-bus `soc-sim` still uses the legacy wrapper-window preload path, so
that path remains covered.

Validation result:

- `make cpu-soc-sim`: PASS through the new descriptor/SRAM path.
- `make soc-sim`: PASS through the legacy wrapper-window path.

## Session 26: Add Current Architecture Document And Docs Map

The user asked for a clearer architecture document that describes the NPU basic
framework, compute logic, and CPU software interaction logic. The user also
noted that some docs are not updated often and asked whether they are still
useful.

Added:

```text
docs/architecture.md
docs/README.md
```

`docs/architecture.md` is now the current architecture entry point. It covers:

- current SoC framework;
- memory map and generated metadata;
- NPU wrapper register model;
- descriptor/SRAM CPU-NPU launch protocol;
- NPU core internal compute/storage model;
- matmul and softmax execution logic;
- compiler/program ownership boundaries;
- verification loops;
- current limitations.

`docs/README.md` classifies docs into:

- current entry points;
- planning and bring-up notes;
- process notes;
- archived notes.

README and existing docs were updated so the main reading order is:

```text
docs/architecture.md
docs/code_structure_review.md
docs/work_rules.md
docs/collaboration_journal.md
docs/bugfix_list.md
```

`docs/project_plan.md`, `docs/soc_bringup.md`, and `docs/fpga_bringup.md` are
still useful, but they are now clearly marked as planning/bring-up/future notes
rather than the current architecture source.

## Session 27: Move NPU Job Descriptor ABI To SoC Spec

The user pointed out that `npu_job_desc` is produced by CPU firmware and
consumed by the NPU wrapper, so defining the format separately in C and RTL is
unsafe. The user also clarified the desired operator/compiler layering:

- `sw/npu_core/operators` should describe matmul, softmax, and later operators
  against the NPU core ISA/intrinsics.
- `sw/tools/npu_compiler` should lower graph/operators to NPU ISA/uop streams.
- `sw/tools/npu_assembler` should encode those streams into program words.
- CPU firmware should only stage data/program/descriptor into SRAM and launch
  the wrapper.

Implemented the descriptor ABI source-of-truth cleanup:

- Added `abi.npu_job_desc` to `arch/configs/soc_v0.jsonc`.
- Extended `sw/tools/soc/emit_soc_spec.py` to generate descriptor field word
  offsets, op ids, and C typedef `soc_npu_job_desc_t`.
- Generated C header content is guarded with `__ASSEMBLER__` so `start.S` can
  still include address constants without seeing C typedefs.
- Updated `sw/soc_cpu/apps/soc_cpu_smoke/main.c` to use
  `soc_npu_job_desc_t` and `SOC_NPU_JOB_OP_*` from the generated SoC header.
- Updated `hw/npu_wrapper/rtl/npu_v0_opsched.sv` to use descriptor field
  offsets and op ids from the generated SoC SVH.
- Updated architecture/review docs to describe descriptor ABI ownership and the
  target operator/compiler layering.

Validation:

- `make cpu-soc-sim`: PASS after fixing the C header assembly guard.

## Session 28: Rework Top-Level README As Project Entry

The user asked for the repository root README to expose the important entry
points immediately, so new readers can find architecture, specs, implementation
review, and verification flow from the top level.

Updated `README.md` to make it a project entry page:

- Added a "Start Here" table linking to current architecture, code structure
  review, docs map, work rules, collaboration journal, and bugfix list.
- Added a "Source-Of-Truth Specs" section for:
  - `arch/configs/npu_v0.jsonc`
  - `arch/configs/npu_wrapper_v0.jsonc`
  - `arch/configs/soc_v0.jsonc`
- Added the generated metadata outputs users should expect under `build/`.
- Added the main implementation path map for `hw/`, `sw/`, `test/`, and `docs/`.
- Reworked verification quick start around `make validate-arch`, `make demo`,
  `make npu-core-sim`, `make soc-sim`, `make cpu-soc-sim`, and `make test`.
- Added a short current end-to-end flow showing graph/input fixture through
  firmware descriptor launch, wrapper SRAM fetch, NPU core execution, SRAM
  output writeback, and firmware `test_status`.

## Session 29: Add Restart Snapshot And Begin Compiler/Assembler Split

The user asked for project goal, current state, and next plan to be kept in one
place so future Codex sessions can recover context after closing and reopening
the project.

Added `Current Project Snapshot` near the top of this journal. It now records:

- current project goal;
- current implemented state;
- latest validation status;
- active design constraints;
- short-term and mid-term next plans.

Also started the second software layering cleanup:

- Added `sw/npu_core/operators/phase0_intrinsics.json` as the Phase 0 operator
  intent/template source for matmul and softmax.
- Added `sw/tools/npu_compiler/phase0.py` for graph/operator-to-uop lowering.
- Added `sw/tools/npu_assembler/phase0.py` for uop-to-32-bit-word encoding.
- Kept `sw/tools/npu_phase0/compiler.py` as a compatibility wrapper.
- Updated `sw/tools/npu_phase0/rtl_fixture.py` to use `npu_assembler.phase0`
  for program encoding.
- Added a unit test that checks the new compiler/assembler path matches the old
  compatibility path.
- Updated architecture/review/operator/compiler/assembler docs to reflect the
  new ownership split.

Validation:

- `make demo`: PASS through the compatibility CLI with the new compiler path.
- `make test`: PASS, 8 tests.

## Session 30: Add Cycle Performance Report UI

The user decided that before adding NPU core architectural complexity, the
project should first expose cycle-level runtime cost. The report must have a UI,
not only terminal text, so it can grow with later fine-grained module counters.

Implemented a first CPU-controlled SoC performance path:

- `hw/soc/tb/soc_cpu_tb.sv` now observes the NPU wrapper/core through hierarchy
  and emits one `PERF_JOB` JSON line per firmware-launched job.
- Current counters cover wrapper phases:
  descriptor read, program fetch, input fetch, core launch/wait, output
  writeback.
- Current core counters cover fetch/vector-style cycles, matmul cycles, and
  done cycles.
- Added `sw/tools/perf/report.py` to parse the simulation log and generate both
  `build/perf/perf.json` and `build/perf/perf_report.html`.
- Added `make perf-report`.

Initial measured baseline:

```text
matmul:  738 total cycles, 520 core cycles, 512 matmul cycles
softmax:  53 total cycles,  11 core cycles
```

Validation:

- `make perf-report`: PASS, generated JSON and HTML report.

## Session 31: Define Research-Backed Target Architecture

The user asked why current matmul takes 512 cycles and whether cube-style
accelerators should do a matrix multiply in one cycle. The answer is that the
current Phase 0 RTL is still a single-lane iterative MAC baseline: an 8x8x8 tile
performs 512 multiply-accumulate updates. It does not yet implement a
systolic/tensor/cube matrix engine.

Researched public architecture directions from NVIDIA Blackwell, Google TPU
v6e/v4, AMD MI300X, Intel Gaudi 3, Gemmini, Eyeriss v2, and SCALE-Sim v3. The
shared direction is:

- matrix/tensor arrays for many MACs per cycle;
- explicit SRAM/HBM-style memory hierarchy and data movement;
- compiler-controlled tiling/dataflow/overlap;
- vector/SFU pipelines for transformer support;
- low precision and sparsity as later measured extensions.

Added `docs/target_architecture.md` with:

- current 512-cycle matmul explanation;
- long-term target block diagram;
- staged architecture milestones A0-A5;
- immediate next implementation plan focused on a measured parallel matmul
  engine before DMA/banking/vector-pipeline complexity.
