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

Design documents must be written from architecture intent to verification
evidence. Do not start with a file list or patch plan. Use this structure unless
there is a clear reason to deviate:

1. **Target / 目标**: state the problem, goal, non-goals, and what decision the
   design should enable.
2. **Overall Design / 整体设计思路**: explain the architecture flow, ownership
   boundaries, source-of-truth inputs, generated artifacts, and how data/control
   moves through the system.
3. **Key Details / 重点细节**: describe the important interfaces, formats,
   state transitions, memory layout, timing model, naming choices, and
   compatibility constraints.
4. **Verification / 验证测试**: state what must be proven, which tests or
   reports prove it, what metrics are measured versus modeled, and what
   existing regressions must remain unchanged.
5. **Implementation Priority / 实现优先级**: break the work into ordered steps
   with acceptance criteria.

设计文档必须从架构意图写到验证证据，不要一开始就列文件和 patch。除非有明确
理由，新增或大幅更新的设计文档应按下面结构组织：

1. **目标**：说明问题、目标、非目标，以及这个设计要支撑什么决策。
2. **整体设计思路**：说明架构流程、ownership 边界、source of truth、生成产物，
   以及数据/控制如何在系统中流动。
3. **重点细节**：说明关键接口、格式、状态转换、memory layout、时序模型、命名
   选择和兼容性约束。
4. **验证测试**：说明要证明什么、用哪些测试或报告证明、哪些指标是实测、哪些
   是模型估计，以及哪些既有 regression 必须保持不变。
5. **实现优先级**：拆成有顺序的步骤，并给出 acceptance criteria。

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
| ASIC PPA boundary, metric or result schema | `docs/design/ppa_methodology.md` |
| Transformer workload, trace or metric choice | `docs/design/transformer_workloads.md` |
| Tests and coverage | `docs/design/verification_strategy.md` |
| Long-term direction | `docs/target_architecture.md` |

After implementation, update the same document with what actually changed,
the observed test/perf result, and any gap between the plan and the result.
If the work changes a public entry point, update `README.md` and
`docs/README.md` as well.

## 9. Bilingual Design Documentation Rule

New or substantially updated design documents should be understandable in both
English and Chinese. Prefer one bilingual document over two separate files so
the contract cannot diverge.

新增或大幅更新的设计文档应同时提供英文和中文说明。优先使用同一份双语文档，
不要拆成两份文件，避免合同内容漂移。

Minimum expectation:

- section titles should include both English and Chinese when practical;
- key problem statements, interface contracts, timing behavior, memory/buffer
  maps, verification plans, limitations, and follow-up work should be bilingual;
- code symbols, filenames, commands, and register names stay in their original
  spelling.

最低要求：

- 章节标题尽量同时包含英文和中文；
- 问题定义、接口合同、时序行为、memory/buffer 划分、验证计划、限制和后续工作
  需要双语说明；
- 代码符号、文件名、命令、寄存器名保持原始拼写。

## 10. Performance Iteration Record Rule

Every NPU performance or PPA optimization iteration must leave a reviewable
record in the relevant design document. The goal is to make later architecture
review possible without reconstructing the reasoning from patches and logs.

Required content:

- measured bottleneck or missing capability before the change;
- design idea and why it should improve performance;
- affected modules and interface/control changes;
- expected perf impact and any expected tradeoff in area, power, verification,
  or software complexity;
- actual measured perf result after implementation;
- commands used for verification;
- whether the result came from real RTL behavior, testbench-side profiling, or
  report/model accounting;
- remaining gap and recommended next performance step.

For changes that intentionally improve cycle count, also add or update a perf
regression check where practical. The check should guard the intended
invariant, for example:

```text
total cycles drops below the previous baseline
data_mover.words remains stable
core.matmul cycles remains stable
```

Once a Level 0 baseline exists, a PPA-affecting RTL change is not complete
until its result summary names the evidence level, candidate, baseline,
workload, metric provenance, improvement deltas, and tradeoff/regression
deltas. The comparison must not suppress unfavorable metrics. At Level 0,
area/energy fields are normalized proxies; until higher-level extraction is
available, timing or real power fields may be marked unavailable but must not
be silently inferred from cycle results.

一旦建立 Level 0 baseline，影响 PPA 的 RTL 变更必须在结果摘要中明确证据层级、
candidate、baseline、workload、指标来源、收益差异和代价/退化差异，才能视为
完成；报告不得隐藏不利指标。在 Level 0 阶段，area/energy 字段是 normalized
proxy；在更高级提取流程可用之前，可以把真实时序或功耗字段标为不可用，但不能
把 cycle 改善直接当成功耗或面积结论。

每次 NPU 性能或 PPA 优化迭代，都必须在对应设计文档中留下可复盘记录。目标是后续
架构 review 时可以直接看到当时的问题、思路、收益和代价，而不是从 patch 和 log
里反推。

必须记录：

- 优化前测到的瓶颈或缺失能力；
- 设计改进思路，以及为什么它应该提升性能；
- 影响的模块、接口和控制逻辑；
- 预期性能收益，以及面积、功耗、验证、软件复杂度等代价；
- 实现后的实测 perf 结果；
- 使用的验证命令；
- 该结果来自真实 RTL 行为、testbench-side profiling，还是 report/model
  accounting；
- 剩余 gap 和推荐的下一步性能优化。

如果改动有意改善 cycle count，应尽量增加或更新 perf regression check。测试应守住
真正的设计不变量，例如：

```text
total cycles 低于旧 baseline
data_mover.words 保持稳定
core.matmul cycles 保持稳定
```
