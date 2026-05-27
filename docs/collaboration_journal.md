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
4. `docs/design/README.md`；
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
make test: PASS, 27 tests
make perf-report: PASS
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

当前真实 workload 记录在：

```text
docs/digits_classifier_workload.md
docs/real_mnist_cnn_workload.md
```

它已经跑通三个 active workload 阶段：

1. Linear classifier 已进入 CPU-controlled SoC：firmware 对 digit_2 PGM 输入
   发起 16 个 `8x8x8` matmul tile job，在 SRAM 中累加 logits，并校验 expected
   logits / predicted label。
2. 真实 MNIST CNN 的 `fc2` 已进入 CPU-controlled SoC：firmware 对真实
   MNIST sample 0 的 `fc1_relu` activation 发起 32 个 `8x8x8` matmul tile
   job，并校验 expected logits / predicted label。
3. 真实 MNIST CNN 的 `fc1` 已进入第一个 CPU-controlled SoC tile checkpoint：
   firmware 对真实 MNIST sample 0 的 quantized `flat`/`fc1.weight` 在
   `K=56, N=0` 发起 1 个 `8x8x8` matmul tile job，并校验 RTL output tile。
   这不是 full `fc1` layer。
4. 真实 MNIST CNN 的 `fc1` 已进入 K-streaming SoC smoke：新增
   `SOC_NPU_JOB_OP_MATMUL_K_STREAM`，firmware 对真实 MNIST sample 0 的 4 个
   selected nonzero K chunks 发起 1 个 descriptor，wrapper 在 descriptor 内
   循环搬运 A/B tile，core 在同一个 `acc_buf` 中累加并最终写回一次。

最新补充：

- 图片输入路径已从阈值二值化升级为灰度 PGM 到 int8 的多级量化；原二值 PGM
  仍保持兼容。
- 新增 `test/assets/digits_realistic/digit_*_gray.pgm`，用于覆盖带抗锯齿和灰度
  变化的离线确定性测试图片。
- CPU firmware smoke 的 linear classifier 数据生成改为使用灰度 `digit_2_gray.pgm`。
- 临时 MNIST tiny CNN prototype 已删除，改为接入真实开源预训练 CNN：
  `docs/real_mnist_cnn_workload.md`。模型来自
  `https://huggingface.co/cmaeti/mnist-cnn`，使用 Apache-2.0 safetensors
  权重。当前已跑通非 RTL golden 流程：读取真实 MNIST IDX gzip 测试图片，
  执行 `conv1 -> relu -> conv2 -> relu -> maxpool -> fc1 -> relu -> fc2 ->
  argmax`，并校验权重 shape、前 10 张预测和前 100 张 accuracy smoke。
  新增 `fc2` 的当前 NPU tile 映射测试：CPU/tool 按原始 float 模型跑到
  `fc1_relu`，用原始 `fc2.weight/bias` 生成 int8 hardware-facing view，
  lower 成 32 个 `8x8x8` tile jobs，通过 micro-op simulator 与原始模型
  argmax 对齐。该真实 CNN 的 `fc2` hardware-facing view 已接入完整
  CPU-controlled SoC RTL：firmware 对 MNIST test sample 0 发起 32 个 NPU
  descriptor jobs，NPU RTL 执行 tile matmul，firmware 累加 partial sums、
  加 scaled 原始 `fc2.bias` 并校验 expected label 7。当前还新增了
  `fc1` 的 first nonzero tile SoC checkpoint 和 K-streaming smoke。
  `make perf-report` 当前识别 52 个 job / 6 个 workload，其中
  `real_mnist_cnn_fc1_tile0` 为 1 job，
  `real_mnist_cnn_fc1_k_stream_smoke` 为 1 job，
  `real_mnist_cnn_fc2` 为 32 jobs。

### Resume Snapshot: 2026-05-19

今天完成的关键工作：

1. 删除临时 MNIST tiny CNN prototype，避免把本地派生权重误认为真实模型。
2. 接入真实开源预训练 MNIST CNN：
   - 来源：`https://huggingface.co/cmaeti/mnist-cnn`
   - license：Apache-2.0
   - 权重：`test/external/mnist_cnn/mnist-cnn.safetensors`
   - 数据：MNIST IDX gzip 测试集，位于 `test/external/mnist/`
   - `test/external/` 已在 `.gitignore` 中忽略，缺外部文件时相关测试 skip。
3. 新增 `sw/tools/npu_phase0/real_mnist_cnn.py`：
   - 最小 safetensors F32 reader；
   - 原始 float CNN forward；
   - `fc2` 的 hardware-facing int8 quantized view；
   - `fc2` lower 到 32 个 current-RTL-compatible `8x8x8` tile jobs。
4. 保持原始 CNN graph 和 float 权重为 source of truth：
   - 不改模型拓扑；
   - 不替换训练权重；
   - 当前只为 NPU RTL 验证派生量化视图。
5. `fc2` 已接入 CPU-controlled SoC RTL：
   - data generator 按原始 float 模型跑 MNIST test sample 0 到 `fc1_relu`；
   - firmware stage quantized `fc1_relu` 和 quantized 原始 `fc2.weight`；
   - PicoRV32 发 32 个 descriptor；
   - wrapper/NPU RTL 执行 32 个 `8x8x8` matmul tile；
   - firmware 累加 partial sums，加 scaled 原始 `fc2.bias`，校验 scaled logits
     和 expected label 7。
6. `perf-report` 识别新增 workload：
   - total jobs: 50
   - workloads: 4
   - `real_mnist_cnn_fc2`: 32 jobs, 7552 cycles

今天验证命令和结果：

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_real_mnist_cnn -v: PASS
make cpu-soc-sim: PASS
make test: PASS, 24 tests
make perf-report: PASS
```

明天最直接的下一步：

1. 旧 Tiny MLP 分支已从 active code/docs 中移除；当前主线切到真实 MNIST CNN。
2. 把真实 CNN 的 `fc1: 9216 -> 128` 作为下一层 NPU 映射目标：
   - 先在工具层定义 tiling/accumulation 方案；
   - 评估 9216 K 维、128 N 维对当前 8x8x8 tile path 的 job 数、SRAM
     footprint、firmware runtime 和 perf report 的压力；
   - 可能需要 grouped workload metadata 或更紧凑的 data staging，避免直接
     在 firmware 中堆大量静态 tile arrays。
3. `fc1` 之前的 conv/maxpool 暂时保持 CPU/tool 侧原始 float 逻辑；等
   `fc1/fc2` 都稳定后，再决定 conv 是 direct op 还是 `im2col -> matmul`。

本地 RISC-V GCC 曾使用：

```text
thirdparty/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-gcc
```

如果 shell 没有继承 `.bashrc`，运行 CPU firmware 相关目标前需要确保该工具链在
`PATH` 中。

### Resume Snapshot: 2026-05-21

今天完成的关键工作：

1. 清理旧 Tiny MLP 分支：
   - 删除 `test/graphs/digits_tiny_mlp.json`；
   - 删除 `sw/tools/npu_phase0/digits_classifier.py` 中 Tiny MLP graph、input、
     FC2 权重、reference logits 和不再使用的 `relu/relu_requantize`；
   - 删除 `test/rtl/test_digits_classifier.py` 中 3 个 Tiny MLP 测试；
   - 更新 `docs/digits_classifier_workload.md`、`docs/project_plan.md`、
     `docs/design/verification_strategy.md`，把 active workload 收敛为
     linear digits classifier + real MNIST CNN。
2. 修正恢复入口：
   - `docs/collaboration_journal.md` 不再要求读取已删除的
     `docs/code_structure_review.md`；
   - 恢复入口改为 `docs/design/README.md`。
3. 补充 perf 采集机制说明：
   - `docs/design/performance_instrumentation.md` 新增 current code walkthrough；
   - `sw/tools/perf/README.md` 新增 “How One Job Is Timed”；
   - 明确当前 perf 是 testbench-side profiling，不是 CPU-readable hardware
     perf counter，也不是 RTL 内部 timestamp packet。
4. 补充真实 MNIST CNN 文档：
   - `docs/real_mnist_cnn_workload.md` 明确 safetensors 只包含权重，不包含可自动
     解析的网络程序；
   - topology 由 `real_mnist_cnn_graph()` 和 `test/graphs/real_mnist_cnn.json`
     显式表达；
   - 当前不是 whole-CNN graph lowering，而是 selected layer hardware-facing
     view lowering；
   - 量化只是当前 int8 RTL 的 hardware-facing view，source of truth 仍是原始
     float graph + safetensors 权重。
5. 讨论并记录 `fc1` 方案：
   - Step 1 先做 CPU-side micro-op simulation：
     float CNN forward 到 `pool/flat`，构建 `fc1` int8 view，编译 logical
     `MATMUL 8x128x9216`，用 `MicroOpFunctionalSimulator` 验证 `fc1 -> relu ->
     fc2` 预测与原始 float 模型一致；
   - Step 2 不采用 CPU firmware 对 18432 个 `8x8x8` tile job 做 partial-sum
     累加；
   - 硬件方向应让 K-axis partial sum 留在 NPU 内部 accumulator；
   - 对 `fc1`，M/N 可以切，但 K=9216 仍是主要问题。推荐语义是按 N tile 发
     job，NPU 内部按 K chunk stream/accumulate，最后只写回一次 output tile。

今天验证命令和结果：

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_digits_classifier -v: PASS, 8 tests
PYTHONPATH=sw/tools python -m unittest test.rtl.test_real_mnist_cnn -v: PASS, 5 tests
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 3 tests
make test: PASS, 24 tests
```

明天最直接的下一步：

1. 实现 `fc1` Step 1 工具层闭环：
   - 在 `sw/tools/npu_phase0/real_mnist_cnn.py` 增加 `fc1` hardware-facing
     quantized view；
   - 构造 logical matmul graph/input：`A[8x9216] * W[9216x128] -> C[8x128]`；
   - 用 `compile_graph()` + `MicroOpFunctionalSimulator` 跑 logical micro-op；
   - 加 scaled `fc1.bias` 和 ReLU；
   - 接现有 `fc2` mapping，校验 sample 0 和前若干 sample 的 predicted label。
2. 扩展 `test/rtl/test_real_mnist_cnn.py`，加入 `fc1 -> fc2` tool-level test。
3. 暂时不要改 RTL/firmware，等 Step 1 确认数值闭环后，再设计
   `fc1` NPU-side K streaming / accumulator residency / data mover contract。

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

1. 在继续优化 NPU core 之前，先加入真实 workload：
   `docs/digits_classifier_workload.md` 记录 8x8 手写数字分类闭环。第一版使用
   `8x64 * 64x16` 的 Phase 0 合法 matmul 形状，NPU-visible 部分输出 logits，
   CPU/tool 侧做 `argmax` 校验 expected digit。
2. 继续完善 operator template schema，增加字段校验，避免 template 写错后到
   RTL/firmware 阶段才暴露。
3. 把 `npu_phase0` compatibility package 中剩余的 Phase 0 专用职责逐步迁到
   `npu_compiler`、`npu_assembler`、simulator/golden 等更清晰目录。
4. 让固定 operator program artifact 在合适时进入 `sw/npu_core/programs`，区分
   “operator intent/source”和“已编码 program artifact”。
5. descriptor 增加错误码、状态码、更清晰的 job validation。

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

## Session 32: Implement A1 Matmul Array

Implemented A1.0 through A1.3:

- Added matmul model estimates to `perf-report`:
  measured compute, scalar baseline, ideal 8x8 array, conservative array, and
  projected total.
- Added `docs/matmul_array_a1.md` describing the module interface, timing, and
  verification plan.
- Added `hw/npu_core/rtl/matmul_array.sv`, an 8x8 output-parallel matmul engine
  that consumes one K slice per active cycle.
- Updated `hw/npu_core/rtl/npu_v0_top.sv` so `UOP_MATMUL` launches the array
  engine and commits `result_flat` into `acc_buf`.
- Updated Makefile RTL/SoC/perf targets to compile the new module.

Measured result:

```text
matmul total cycles: 738 -> 236
core matmul cycles:  512 -> 10
softmax total cycles: unchanged at 53
```

Interpretation:

- A1 achieved the intended compute reduction.
- The current matmul job is now dominated by wrapper program/input/output
  movement rather than the core compute phase.
- The next architecture milestone should be A2: real data movement,
  scratchpad/banking, and overlap.

## Session 33: Add A2.0 Data-Movement Profiling

Started A2 with profiling rather than a DMA rewrite.

Added movement counters to `soc_cpu_tb` PERF_JOB records:

- SRAM NPU-port read/write cycles;
- core host-window write/read cycles;
- descriptor/program/input/output word counts.

Updated `perf-report` to render:

- movement annotations inside `Wrapper phases`, such as
  `SRAM read program -> core host write instr_mem`;
- raw movement counters in `perf.json` for later tooling.

Measured post-A1 matmul movement profile:

```text
total job cycles:              236
core matmul cycles:             10
SRAM read cycles:              153
SRAM write cycles:              64
core host write cycles:        144
core host read cycles:          64
```

Interpretation:

- Compute is no longer the bottleneck for the Phase 0 matmul.
- The wrapper still performs one-word-per-cycle movement through the SRAM port
  and core host window.
- `Core host window` means wrapper-to-core host-interface accesses, not
  independent core-to-SRAM writes.
- The next A2 design step should define a real data mover/burst path and
  scratchpad banking model.

## Session 34: Define A2.1 Movement Direction

Clarified the temporary nature of the current wrapper/core host-window path:

- wrapper preload/readback through core host windows is an A0/A1 bring-up
  mechanism;
- core A/B/X/program/output windows are small NPU-core internal memories;
- wrapper currently reads SRAM and writes those internal memories one word at a
  time;
- core output is read back by wrapper and then written to SRAM;
- fixed-size `instr_mem` preload is not suitable for future variable-length
  programs.

Added `docs/data_mover_a2.md` as the A2 working plan. It defines the intended
direction: data mover commands, burst SRAM movement, banked scratchpad,
instruction buffer/prefetch, movement/stall timeline lanes, and double-buffer
overlap later.

Updated `perf-report` with a `Movement model` panel. It uses current movement
word counts and a simple 4-word-per-cycle burst estimate to compare today's
one-word wrapper movement against the first A2 data-mover target. This is a
projection, not RTL behavior yet.

## Session 35: Start A2 RTL Data Mover

Documented how perf counters are collected:

- current counters live in `soc_cpu_tb`;
- the testbench samples wrapper/core state and SRAM/host-window signals once
  per clock;
- `PERF_JOB` lines are simulation records, not CPU-readable hardware counters;
- once the taxonomy stabilizes, these counters should move into optional RTL
  perf registers or debug CSRs.

Started A2 RTL with a structural data mover:

- added `hw/npu_wrapper/rtl/npu_v0_data_mover.sv`;
- routed wrapper program/input/output linear transfers through the module;
- preserved the current one-word-per-cycle timing;
- added `Data mover` as a report timeline lane, currently reconstructed from
  wrapper movement phases.

Verification:

```text
make test        PASS
make perf-report PASS
matmul total cycles: 236
softmax total cycles: 53
```

## Session 36: Freeze Next-Step Context

Before pausing, recorded the next-session plan in `docs/data_mover_a2.md`.

Resume point:

- A2 RTL has started.
- The current data mover is structural and still behaves as one word per cycle.
- The report has a `Data mover` lane, but that lane is currently reconstructed
  from wrapper movement phases.
- Perf counters are still collected in `soc_cpu_tb`, not CPU-readable RTL CSRs.

Next work:

1. Add data mover timing parameters: `WORDS_PER_CYCLE` and `SETUP_CYCLES`.
2. Preserve current `1 word/cycle` default behavior and pass tests.
3. Add a profiled burst mode matching the current report model:
   `4 words/cycle + 1 setup cycle per transfer segment`.
4. Emit explicit data mover counters in `PERF_JOB`.
5. Drive the report `Data mover` lane from those counters rather than wrapper
   phase reconstruction.
6. Compare measured movement cycles against the existing matmul movement model:
   current measured SRAM movement is 217 cycles; conservative burst estimate is
   about 60 cycles.
7. Start scratchpad banking only after data mover counters and burst timing are
   stable.

Do not jump directly to double buffering, bank conflicts, or variable-length
program streaming in the next session.

## Session 37: Add Module Design Documentation

Paused feature work to improve design documentation before the system grows
further.

Added detailed design docs:

- `docs/design/soc_architecture.md`: SoC top, memory map, bus semantics,
  ROM/SRAM, NPU attachment, simulation top, source-of-truth rules.
- `docs/design/npu_wrapper.md`: wrapper register interface, descriptor ABI,
  FSM, core host-window map, A2.1 data mover, timing semantics, status/error
  limitations.
- `docs/design/npu_core.md`: core interface, internal memories, host-window map,
  uop execution, A1 matmul array, softmax path, timing baseline and limitations.
- `docs/design/software_hardware_flow.md`: compiler/assembler/firmware/runtime
  flow, descriptor setup, program format, pass/fail contract.
- `docs/design/performance_instrumentation.md`: testbench-side perf collection,
  sampled signals, `PERF_JOB` schema, report lanes, movement model, counter
  placement policy.
- `docs/design/verification_strategy.md`: verification layers, current
  baselines, gaps, and test update rules.

Updated `docs/README.md` and `docs/architecture.md` so these files are formal
entry points rather than scattered notes.

## Session 38: Prune Redundant Docs And Strengthen Doc Rules

Cleaned up confusing historical/redundant documentation:

- deleted `docs/code_structure_review.md`; its active content is now covered by
  `docs/architecture.md` and the focused `docs/design/*.md` documents;
- deleted `docs/soc_bringup.md`; the current SoC design now lives in
  `docs/design/soc_architecture.md` and `docs/design/npu_wrapper.md`;
- updated `README.md`, `docs/README.md`, `docs/archive/README.md`,
  `docs/fpga_bringup.md`, and `docs/bugfix_list.md` to stop pointing developers
  at deleted active-entry files;
- added `docs/design/README.md` as the index for active module design docs.

Updated `docs/work_rules.md` with a design-before-implementation rule:

- before non-trivial implementation, write the design intent into the matching
  active document under `docs/`;
- include problem, scope, affected modules, interfaces, timing/perf impact,
  verification plan, limitations, and follow-up work;
- after implementation, update the same document with actual behavior, test
  results, and gaps.

This keeps future contributors from relying on stale chat context or scattered
historical notes.

## Session 39: Complete Real CNN Graph Metadata And FC1 Quantized Tool Loop

User review found two issues in the real MNIST CNN workload:

- `test/graphs/real_mnist_cnn.json` did not fully describe operator attributes,
  especially convolution kernel shapes, stride/padding, parameter shapes, and
  linear layer feature counts.
- The `fc1` quantization boundary and policy needed an explicit design
  document before implementation.

Implemented changes:

- Expanded `test/graphs/real_mnist_cnn.json` so it now records:
  - all parameter shapes for `conv1`, `conv2`, `fc1`, and `fc2`;
  - all active tensor shapes, including ReLU outputs, `Flat`, and `Predicted`;
  - conv2d `input_shape`, `weight_shape`, `bias_shape`, `output_shape`,
    `kernel_shape`, `strides`, and `pads`;
  - maxpool/flatten shape attributes;
  - linear `in_features`, `out_features`, `weight_shape`, `bias_shape`, and
    `output_shape`;
  - the Phase 0 default quantization metadata.
- Added `docs/design/quantization_strategy.md`:
  - current policy is symmetric signed-int8 activation and weight quantization;
  - activation/weight are per-tensor for the first tool loop;
  - accumulators are int32;
  - bias/ReLU stay in dequantized float for the tool-level `fc1` test;
  - asymmetric quantization is deferred because zero-point correction requires
    new hardware/software contract fields and arithmetic.
- Updated `docs/real_mnist_cnn_workload.md`, `docs/README.md`, and
  `docs/design/README.md` to point at the complete graph and quantization
  strategy.
- Added graph helper/validation code in `sw/tools/npu_phase0/real_mnist_cnn.py`:
  - `real_mnist_cnn_op()`;
  - `validate_real_mnist_cnn_graph()`;
  - shape consistency checks against graph metadata and safetensors weights.
- Updated `fc2` mapping to derive its feature count and real class count from
  the graph, padding only to the Phase 0 tile width.
- Added `fc1` tool-level mapping helpers:
  - `fc1_npu_inputs_from_flat()`;
  - `fc1_logical_matmul_graph()`;
  - `fc1_relu_from_int32()`.
- Extended `test/rtl/test_real_mnist_cnn.py` with a `fc1 -> fc2` logical
  micro-op test. It runs `8x9216 * 9216x128` through
  `MicroOpFunctionalSimulator`, applies bias/ReLU, feeds that result into the
  existing tiled `fc2` path, and verifies the class prediction for the first
  three MNIST test samples.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_real_mnist_cnn -v: PASS, 6 tests
make test: PASS, 25 tests
make perf-report: PASS
```

Remaining boundary:

- This is still a tool-level `fc1` numerical closure. It does not claim the
  current RTL can execute a resident `8x9216 * 9216x128` matmul.
- The next hardware design step remains NPU-side K streaming / accumulator
  residency for `fc1`; do not implement `fc1` as 18432 CPU-launched
  `8x8x8` descriptor jobs.

## Session 40: Add FC1 SoC RTL Tile Checkpoint

User follow-up:

1. Record linear digits classifier retirement as a separate future cleanup task.
2. Make quantization verification explicit in the quantization design doc.
3. Start moving real MNIST CNN `fc1` into SoC RTL verification.

Decisions:

- The 8x8 linear digits classifier should not be deleted piecemeal. It remains
  the checked-in no-external-fixture smoke workload for now. Retirement is now
  tracked in `docs/project_plan.md` as its own task after real MNIST CNN
  `fc1/fc2` SoC coverage is stable.
- Quantization validation is now documented in
  `docs/design/quantization_strategy.md`, including tool-level numerical tests,
  SoC RTL checks, and regression gates.
- Full `fc1` should still not be implemented as 18432 CPU-launched
  `8x8x8` descriptor jobs. The first SoC RTL step is a real `fc1` tile
  checkpoint using current hardware, while full-layer execution waits for
  NPU-side K streaming and accumulator residency.

Implemented:

- `sw/tools/firmware/emit_soc_cpu_smoke_data.py` now emits
  `REAL_MNIST_CNN_FC1_TILE_*` data when real MNIST external fixtures are
  available:
  - sample: MNIST test sample 0;
  - K offset: 56;
  - N offset: 0;
  - shape: one current RTL-compatible `8x8x8` tile;
  - expected output: generated by the existing Python golden matmul over the
    quantized `fc1` activation/weight tile.
- `sw/soc_cpu/apps/soc_cpu_smoke/main.c` stages that tile in SRAM, launches one
  normal matmul descriptor through the NPU wrapper, and checks the RTL output
  against the generated expected tile.
- `sw/tools/perf/report.py` now groups the extra job as
  `real_mnist_cnn_fc1_tile0` before the existing 32-job
  `real_mnist_cnn_fc2` workload.
- `test/rtl/test_perf_report.py` covers both the previous 50-job report shape
  and the new 51-job shape with the FC1 tile checkpoint.
- Docs updated:
  - `docs/project_plan.md`;
  - `docs/real_mnist_cnn_workload.md`;
  - `docs/design/quantization_strategy.md`;
  - `docs/design/verification_strategy.md`;
  - `docs/design/performance_instrumentation.md`.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 4 tests
make cpu-soc-sim: PASS, 51 PERF_JOB records
make test: PASS, 26 tests
make perf-report: PASS
build/perf/perf.json summary: jobs=51, workloads=5, total_cycles=11853
```

Remaining boundary:

- `real_mnist_cnn_fc1_tile0` verifies real `fc1` tile staging, current RTL
  arithmetic, wrapper writeback, and firmware comparison.
- It is not full `fc1` layer execution. The next architecture step is a new
  wrapper/core contract that streams K chunks and preserves accumulators inside
  the NPU side so full `fc1` can be represented as approximately 16 output-tile
  jobs instead of 18432 micro-tile jobs.

## Session 41: Implement MATMUL_K_STREAM Smoke

User asked to proceed with the K-axis streaming direction: keep the physical
`8x8x8` MAC tile, but expose a larger K matmul job where the wrapper streams K
chunks and the core accumulates partial sums internally.

Design document added:

```text
docs/design/fc1_k_streaming_matmul.md
```

Key design decisions:

- Do not enlarge the core buffer to `8x9216`.
- Keep the physical tile at `M=8, N=8, K_STEP=8`.
- Add `SOC_NPU_JOB_OP_MATMUL_K_STREAM`.
- Add descriptor field `k_chunks`.
- First version uses packed A/B streams rather than natural-stride tensor
  layout to avoid adding a full address generator before the data mover grows.
- Add a core host control register at `0x500`:
  - bit 0: matmul accumulate enable;
  - bit 1: clear accumulator pulse.
- Wrapper behavior for K-stream jobs:
  - configure accumulator clear/accumulate;
  - loop over `k_chunks`;
  - fetch one A tile and one B tile per chunk;
  - start the core once per chunk;
  - write output once at the end;
  - disable accumulate mode.

Implemented:

- `arch/configs/soc_v0.jsonc` descriptor ABI now has `k_chunks` and
  `matmul_k_stream`.
- `hw/npu_core/rtl/npu_v0_top.sv` supports accumulate mode:
  `acc_buf += matmul_result` when enabled, otherwise normal overwrite behavior.
- `hw/npu_wrapper/rtl/npu_v0_opsched.sv` supports descriptor-internal K-loop
  execution for `MATMUL_K_STREAM`.
- `sw/tools/firmware/emit_soc_cpu_smoke_data.py` emits a real MNIST CNN
  `fc1` K-stream smoke:
  - sample 0;
  - 4 selected nonzero K chunks;
  - packed A/B streams;
  - expected C tile accumulated by the Python golden matmul.
- `sw/soc_cpu/apps/soc_cpu_smoke/main.c` stages the packed streams, launches
  one `MATMUL_K_STREAM` descriptor, and checks the single output tile.
- `hw/soc/tb/soc_cpu_tb.sv` emits `PERF_JOB` name `matmul_k_stream`.
- `sw/tools/perf/report.py` groups the new job as
  `real_mnist_cnn_fc1_k_stream_smoke`.
- `test/rtl/test_perf_report.py` covers the 52-job report shape.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 5 tests
make cpu-soc-sim: PASS, 52 PERF_JOB records
make perf-report: PASS
make test: PASS, 27 tests
```

Current perf summary with real MNIST external fixtures:

```text
jobs: 52
workloads: 6
total_cycles: 12584
operator_smoke_matmul: 237 cycles
operator_smoke_softmax: 54 cycles
digits_linear_classifier: 16 jobs, 3792 cycles
real_mnist_cnn_fc1_tile0: 1 job, 237 cycles
real_mnist_cnn_fc1_k_stream_smoke: 1 job, 680 cycles
real_mnist_cnn_fc2: 32 jobs, 7584 cycles
```

Important limitation:

- The new K-streaming hardware contract is real, but the smoke uses 4 selected
  real FC1 chunks, not all 1152 chunks.
- Full `fc1` N-tile execution needs compact staging or an external load path.
  A full packed stream for one N tile would exceed the current small boot
  ROM/SRAM budget.

## Session 42: Clarify Core Parallelism And Bilingual Design Docs

User asked to further clarify:

1. how many MACs the current NPU core performs per cycle;
2. how internal SRAM/buffers are partitioned for A, B, accumulator, and output;
3. that future design documents should be bilingual.

Documentation updates:

- Rewrote `docs/design/fc1_k_streaming_matmul.md` as a bilingual English/Chinese
  design document.
- Added the current compute parallelism:
  - physical tile remains `M=8, N=8, K=8`;
  - each active matmul cycle updates all `M*N=64` output elements in parallel;
  - each active cycle performs 64 signed int8-by-int8 MACs into int32;
  - `k_idx` advances across cycles, so one `8x8x8` tile needs 8 active MAC
    cycles, observed as about 10 core matmul cycles with start/done/commit
    overhead.
- Added the current core storage map:
  - `dram_a`: 64 int8 host preload entries for A tile;
  - `dram_b`: 64 int8 host preload entries for B tile;
  - `spad_a`: 64 int8 scratchpad entries loaded by `LOAD A`;
  - `spad_b`: 64 int8 scratchpad entries loaded by `LOAD B`;
  - `acc_buf`: 64 int32 resident accumulator/output staging entries;
  - `dram_c`: 64 int32 host-readable output entries;
  - plus `instr_mem`, `dram_x`, `vec_buf`, and `dram_y`.
- Clarified that K-streaming does not add `8x9216` or `9216x8` buffers inside
  the core. A/B tile buffers are overwritten for each K chunk while `acc_buf`
  stays resident.
- Updated `docs/design/npu_core.md` with the same core parallelism and buffer
  partition details.
- Added a bilingual design documentation rule to `docs/work_rules.md`:
  new or substantially updated design docs should include both English and
  Chinese explanations in the same file.

No RTL or software behavior changed in this session; this was documentation
clarification only.

Follow-up clarification:

- Added a cycle-by-cycle `8x8 * 8x8` example to
  `docs/design/fc1_k_streaming_matmul.md`.
- The example shows that each cycle fixes one `k_idx` and performs an outer
  product:
  `C[i,j] += A[i,k_idx] * B[k_idx,j]` for all 64 output coordinates in
  parallel.
- It explicitly lists cycle 0 through cycle 7 and the partial terms accumulated
  in every `C[i,j]`.

## Session 43: Extract K-Stream Planner

User asked to continue coding from the documented K-streaming direction.

Design update:

- Extended `docs/design/fc1_k_streaming_matmul.md` with a bilingual planner
  section.
- Documented that compiler-side planning now produces `k_chunks`, `k_offsets`,
  packed `a_stream`/`b_stream`, and one accumulated `expected_c`.
- Documented the full real MNIST CNN `fc1` single-N-tile artifact:
  `A[8,9216] * B[9216,8] -> C[8,8]`, implemented as 1152 physical
  `8x8x8` chunks.
- Kept the SoC boundary explicit: the current boot ROM/SRAM path still runs the
  4-chunk smoke; the full 1152-chunk artifact is generated and checked in tool
  tests until a host-preload or compact tensor-stride staging path exists.

Implemented:

- Added `sw/tools/npu_compiler/k_stream.py` with `plan_matmul_k_stream()`.
- Exported the planner from `sw/tools/npu_compiler/__init__.py`.
- Refactored `sw/tools/firmware/emit_soc_cpu_smoke_data.py` so both the
  single-tile FC1 checkpoint and the 4-chunk K-stream smoke consume the shared
  planner instead of local ad hoc chunk-selection helpers.
- Added `test/rtl/test_k_stream_planner.py` for small matrix planner behavior.
- Extended `test/rtl/test_real_mnist_cnn.py` to build the full 1152-chunk FC1
  single-N-tile plan and compare it against direct logical matmul.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_k_stream_planner -v: PASS, 2 tests
PYTHONPATH=sw/tools python -m unittest test.rtl.test_real_mnist_cnn -v: PASS, 7 tests
make test: PASS, 30 tests
make cpu-soc-sim: PASS, 52 PERF_JOB records
make perf-report: PASS
build/perf/perf.json summary: jobs=52, workloads=6, total_cycles=12584
```

## Session 44: Full FC1 Single N-Tile SoC K-Stream

User asked to implement the first host/SRAM preload-style checkpoint by keeping
the data expansion simple in `main.c` and enlarging SRAM as needed, with the
focus on NPU architecture iteration rather than CPU-side elegance.

Implemented:

- Enlarged the simulation memory map in `arch/configs/soc_v0.jsonc`:
  - boot ROM: 2 MiB;
  - SRAM: 2 MiB at `0x0020_0000`;
  - stack pointer moved to the top of the enlarged SRAM.
- Parameterized `hw/soc/rtl/soc_cpu_top.sv` so `boot_rom` and `simple_sram`
  word counts come from generated SoC size constants.
- Increased `hw/soc/tb/soc_cpu_tb.sv` timeout to cover CPU staging plus the
  full K-stream job.
- Increased `Makefile` firmware ROM padding to `524288` words.
- Extended `sw/tools/firmware/emit_soc_cpu_smoke_data.py` to emit a full
  `REAL_MNIST_CNN_FC1_FULL_K_STREAM_*` data set:
  - sample 0;
  - `k_chunks=1152`;
  - packed A stream: `1152 * 64` words;
  - packed B stream: `1152 * 64` words;
  - expected accumulated `C[8,8]`.
- Extended `sw/soc_cpu/apps/soc_cpu_smoke/main.c` with enlarged SRAM arrays,
  CPU-side copy loops, one `SOC_NPU_JOB_OP_MATMUL_K_STREAM` descriptor, and
  output comparison for the full single-N-tile result.
- Extended `sw/tools/perf/report.py` and `test/rtl/test_perf_report.py` to
  recognize `real_mnist_cnn_fc1_full_k_stream_tile0`.
- Updated `docs/design/fc1_k_streaming_matmul.md`,
  `docs/design/performance_instrumentation.md`, and
  `docs/design/verification_strategy.md`.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 6 tests
make firmware-smoke-c: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make test: PASS, 31 tests
make perf-report: PASS
build/perf/perf.json summary: jobs=53, workloads=7, total_cycles=182020
```

Key measured result:

```text
real_mnist_cnn_fc1_full_k_stream_tile0:
  jobs: 1
  total_cycles: 169436
  k_chunks: 1152
  input0_words: 73728
  input1_words: 73728
  core matmul cycles: 11520
```

Boundary:

- This verifies one full `fc1` output N tile, not all 128 `fc1` output
  channels.
- The data path is intentionally simple: large C firmware data is copied into
  enlarged SRAM by CPU code. This is acceptable for the current architecture
  checkpoint, but should later be replaced by host preload, loader support, or
  stride-based compact staging.

## Session 45: Start Data-Movement Improvement Plan

User asked to write the four-step data-movement improvement plan into docs and
then execute each step in order with design, coding, and verification.

Design update:

- Added the full FC1 data-movement improvement roadmap to
  `docs/design/npu_wrapper.md`:
  1. keep the physical `8x8x8` MAC tile unchanged;
  2. replace the debug-style wrapper-to-core host-window preload path with a
     real movement path;
  3. parameterize and widen movement bandwidth with `WORDS_PER_CYCLE` and
     `SETUP_CYCLES`;
  4. add double buffering so fetch of chunk `i+1` can overlap compute of
     chunk `i`.
- Updated `docs/design/fc1_k_streaming_matmul.md` follow-up work with the same
  ordered plan.

Step 1 implementation:

- Added `WORDS_PER_CYCLE` and `SETUP_CYCLES` parameters to
  `hw/npu_wrapper/rtl/npu_v0_data_mover.sv`.
- Preserved the current verified default behavior:
  `WORDS_PER_CYCLE=1`, `SETUP_CYCLES=0`.
- Added a guard so `WORDS_PER_CYCLE > 1` is not accidentally enabled before the
  core preload/readback interface is widened; the current core host-window
  interface can only accept one word per cycle.
- Explicitly bound those defaults in `hw/npu_wrapper/rtl/npu_v0_opsched.sv`.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 6 tests
make cpu-soc-sim: PASS, 53 PERF_JOB records
make test: PASS, 31 tests
make perf-report: PASS
build/perf/perf.json summary: jobs=53, workloads=7, total_cycles=182020
```

Measured full FC1 single-N-tile job remains unchanged, as intended for Step 1:

```text
real_mnist_cnn_fc1_full_k_stream_tile0:
  total_cycles: 169436
  k_chunks: 1152
  input0_words: 73728
  input1_words: 73728
  core matmul cycles: 11520
```

Next step:

- Design and implement a widened core preload/readback interface so
  `WORDS_PER_CYCLE > 1` can become functionally correct instead of only a
  future parameter.

## Session 46: Widen Core Host Preload Interface Shape

User confirmed the plan to keep CPU staging unchanged for now and continue with
the NPU-side data path. This session implemented Step 2 of the documented
movement roadmap.

Design update:

- Updated `docs/design/npu_wrapper.md` with the Step 2 contract:
  - `CORE_HOST_LANES=4`;
  - `host_we[3:0]`;
  - one base `host_addr`;
  - packed `host_wdata[127:0]`;
  - packed `host_rdata[127:0]`;
  - lane `i` maps to host word address `host_addr + i`.
- Updated `docs/design/npu_core.md` with the same widened preload/readback
  interface.
- Kept the coding boundary explicit: wrapper still drives lane 0 only in this
  checkpoint, so behavior and timing remain equivalent.

Implemented:

- `hw/npu_core/rtl/npu_v0_top.sv` now has parameterized
  `CORE_HOST_LANES=4` host write/read lanes.
- Core preload writes now iterate over active lanes and write consecutive A/B/X
  or program host-window addresses.
- Core readback now returns consecutive C/Y words across the packed lanes.
- `hw/npu_wrapper/rtl/npu_v0_opsched.sv` instantiates the core with four lanes
  but maps the existing scalar wrapper/data-mover path to lane 0 only.
- `hw/npu_core/tb/npu_v0_tb.sv` was updated for the packed interface and now
  includes a direct 4-lane host-window smoke check.

Validation:

```text
make npu-core-sim: PASS
make soc-sim: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make test: PASS, 31 tests
make perf-report: PASS
build/perf/perf.json summary: jobs=53, workloads=7, total_cycles=182020
```

The full FC1 single-N-tile job remains unchanged, as intended:

```text
real_mnist_cnn_fc1_full_k_stream_tile0:
  total_cycles: 169436
  k_chunks: 1152
  input0_words: 73728
  input1_words: 73728
  core matmul cycles: 11520
```

Next step:

- Connect the 4-lane core preload/readback interface to a multi-lane data mover
  and SRAM model so `WORDS_PER_CYCLE=4` can become real instead of lane-0
  compatibility mode.

## Session 47: Enable WORDS_PER_CYCLE=4 Movement

User confirmed the movement bandwidth target: set `WORDS_PER_CYCLE=4`.

Design update:

- Updated `docs/design/npu_wrapper.md` to mark Step 3 as real RTL behavior:
  - `DATA_MOVER_WORDS_PER_CYCLE=4`;
  - `CORE_HOST_LANES=4`;
  - SRAM NPU port is 4 lanes / 128-bit packed data;
  - CPU SRAM port remains scalar 32-bit;
  - lane `i` transfers word `base + i`;
  - partial tails use per-lane masks.
- Updated performance and verification docs with the new measured baselines.

Implemented:

- `simple_sram` now exposes a 4-lane NPU port while preserving the scalar CPU
  port.
- `npu_v0_data_mover` now supports `WORDS_PER_CYCLE` up to the configured lane
  count and drives packed SRAM/core-host data.
- `npu_v0_opsched` instantiates the data mover with
  `DATA_MOVER_WORDS_PER_CYCLE=4` and connects the packed movement path to the
  4-lane core preload/readback interface.
- SoC integration and perf counters were updated for vector SRAM write masks
  and multi-word movement counts.

Validation:

```text
make npu-core-sim: PASS
make soc-sim: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make test: PASS, 31 tests
make perf-report: PASS
build/perf/perf.json summary: jobs=53, workloads=7, total_cycles=63100
```

Measured workload cycles after Step 3:

```text
operator_smoke_matmul: 81 cycles
operator_smoke_softmax: 30 cycles
digits_linear_classifier: 16 jobs, 1296 cycles
real_mnist_cnn_fc1_tile0: 1 job, 81 cycles
real_mnist_cnn_fc1_k_stream_smoke: 1 job, 236 cycles
real_mnist_cnn_fc1_full_k_stream_tile0: 1 job, 58784 cycles
real_mnist_cnn_fc2: 32 jobs, 2592 cycles
```

Full FC1 single-N-tile detail:

```text
total_cycles: 58784
k_chunks: 1152
input0_words: 73728
input1_words: 73728
fetch_input0 cycles: 18432
fetch_input1 cycles: 18432
core matmul cycles: 11520
```

Next step:

- Add explicit data mover counters to `PERF_JOB`, then drive the report
  `Data mover` lane from real data mover state/counters instead of only wrapper
  phase reconstruction.

## Session 48: Move Bandwidth Knobs Into SoC Spec And Widen CPU SRAM Port Shape

User asked to put the `WORDS_PER_CYCLE`/lane parameters into spec and then start
modifying the CPU bus access width.

Design boundary:

- PicoRV32 remains an RV32 core and still issues one 32-bit load/store per CPU
  request.
- The SoC SRAM CPU port is now structurally widened to 4 lanes / 128-bit packed
  data.
- `simple_bus` maps each scalar PicoRV32 SRAM access onto one lane of that wide
  SRAM CPU port.
- This does not accelerate CPU staging by itself; it prepares the SRAM-side
  interface for a later preload/copy engine that can drive multiple lanes per
  cycle.

Spec update:

- Added to `arch/configs/soc_v0.jsonc`:
  - `bus.sram_cpu_lanes = 4`;
  - `bus.sram_cpu_data_width_bits = 128`;
  - `npu_data_mover.core_host_lanes = 4`;
  - `npu_data_mover.sram_npu_lanes = 4`;
  - `npu_data_mover.words_per_cycle = 4`;
  - `npu_data_mover.setup_cycles = 0`.
- `sw/tools/soc/emit_soc_spec.py` now emits generated constants for these
  fields and validates:
  - PicoRV32 bus width is still 32-bit;
  - SRAM CPU data width equals `sram_cpu_lanes * 32`;
  - current RTL requires `core_host_lanes == sram_npu_lanes`;
  - `words_per_cycle` is in range.

Implemented:

- `simple_sram` now has parameterized CPU and NPU lane counts.
- `simple_bus` now exposes a packed multi-lane SRAM CPU port and performs
  scalar lane select/readback for PicoRV32 requests.
- `soc_cpu_top` and `soc_top` instantiate bus/SRAM/NPU wrapper with generated
  SoC lane constants.
- `npu_v0_opsched` now takes `DATA_MOVER_WORDS_PER_CYCLE` and
  `DATA_MOVER_SETUP_CYCLES` from generated SoC constants rather than a local
  hard-coded value.
- `soc_cpu_tb` lane counting now uses `SOC_NPU_SRAM_LANES`.

Validation so far:

```text
make soc-spec: PASS
make npu-core-sim: PASS
make soc-sim: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make test: PASS, 31 tests
make perf-report: PASS
build/perf/perf.json summary: jobs=53, workloads=7, total_cycles=63100
```

Observed timing remains the same as expected because PicoRV32 still drives one
32-bit lane per request:

```text
operator_smoke_matmul: 81 cycles
real_mnist_cnn_fc1_full_k_stream_tile0: 58784 cycles
```

Next step:

- Add a preload/copy engine or loader path that can drive all SRAM CPU lanes per
  cycle for ROM/flash-to-SRAM staging. That is the step expected to reduce CPU
  staging simulation time.

## Session 49: Add ROM-To-SRAM DMA For Firmware Staging

User clarified the CPU-bus-width question and approved adding a DMA. The goal
was to reduce firmware staging time without changing PicoRV32's 32-bit ISA or
the NPU job contract.

Design:

- Added `docs/design/soc_dma.md`.
- DMA is a CPU-configured ROM-to-SRAM copy engine:
  - CPU writes source ROM address, destination SRAM address, word count, and
    start;
  - DMA reads from a second packed boot-ROM read port;
  - DMA writes up to `SOC_SRAM_CPU_LANES` words per cycle into the SRAM
    CPU/preload port;
  - CPU polls done.
- This optimizes staging before NPU jobs. It does not change NPU wrapper/core
  cycle counts.

Implemented:

- Added `SOC_DMA_BASE/SIZE/MASK` to `arch/configs/soc_v0.jsonc` and generated
  headers.
- Added `hw/soc/rtl/dma/soc_dma.sv`.
- Extended `simple_bus` with a DMA MMIO target.
- Extended `boot_rom` with a second packed DMA read port.
- Reused the widened SRAM CPU/preload port through a mux in `soc_cpu_top`.
- Added `dma_copy_words()` to firmware runtime and routed the existing
  `copy_words()` helper through DMA.
- Added `soc_dma.sv` to RTL simulation builds.
- Updated SoC/perf docs to state that DMA staging is outside current
  `PERF_JOB` timing.

Validation:

```text
make soc-spec: PASS
make soc-sim: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make test: PASS, 31 tests, 45.793s
make perf-report: PASS
build/perf/perf.json summary: jobs=53, workloads=7, total_cycles=63100
```

Observed CPU-controlled simulation finish time:

```text
before DMA: 39712785000 ps
after DMA:   3765375000 ps
```

NPU job timing remains unchanged, as expected:

```text
real_mnist_cnn_fc1_full_k_stream_tile0: 58784 cycles
```

Next step:

- Add DMA counters/error bits if we want the performance report to show staging
  time explicitly.
- Then return to NPU-side work: explicit data mover counters, overlap, and
  scratchpad banking.

## Session 50: Add Explicit Data Mover Counters And Ping-Pong Design

User asked to first implement explicit data mover counters, then write the
K-streaming ping-pong buffer design document with current-problem explanation,
overlap advantages, and diagrams.

Implemented explicit data mover visibility:

- `npu_v0_data_mover` now exposes per-cycle perf signals:
  - `perf_active`;
  - `perf_setup`;
  - `perf_transfer`;
  - `perf_stall`;
  - `perf_words`.
- `soc_cpu_tb` samples those signals and emits a new `data_mover` object in
  every `PERF_JOB`.
- The old `movement` object remains for compatibility and SRAM/core-host
  counters.
- `sw/tools/perf/report.py` now carries `data_mover` into job and workload
  summaries and includes data mover measured values in movement estimates.
- `test/rtl/test_perf_report.py` now covers the new JSON field.

Measured full FC1 single-N-tile explicit data mover counters:

```text
real_mnist_cnn_fc1_full_k_stream_tile0:
  total_cycles: 58784
  data_mover.active_cycles: 36884
  data_mover.transfer_cycles: 36884
  data_mover.stall_cycles: 0
  data_mover.words: 147536
  data_mover.read_cycles: 36868
  data_mover.write_cycles: 16
  data_mover.read_words: 147472
  data_mover.write_words: 64
  core.matmul cycles: 11520
```

Design documentation:

- Added `docs/design/k_stream_ping_pong_buffer.md`.
- The document explains the current serial K-streaming problem:
  `load_A + load_B + compute`.
- It includes ASCII timelines for:
  - current serial execution;
  - proposed ping-pong overlap;
  - A/B bank ownership;
  - wrapper barrier conditions.
- It defines the first RTL scope:
  - keep physical `8x8x8` MAC tile;
  - keep one resident `acc_buf`;
  - add A/B ping-pong banks;
  - overlap prefetch of chunk `i+1` with compute of chunk `i`;
  - keep normal matmul/softmax behavior unchanged.

Docs updated:

- `docs/design/performance_instrumentation.md`;
- `docs/design/npu_wrapper.md`;
- `docs/design/README.md`;
- `docs/README.md`.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 6 tests
make npu-core-sim: PASS
make soc-sim: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make perf-report: PASS
build/perf/perf.json summary: jobs=53, workloads=7, total_cycles=63100
make test: PASS, 31 tests, 44.071s
```

Next step:

- Start coding K-streaming ping-pong overlap according to
  `docs/design/k_stream_ping_pong_buffer.md`.

### Resume Snapshot: 2026-05-23

Stop point:

- DMA staging is implemented and verified.
- Explicit data mover counters are implemented and verified.
- K-streaming ping-pong buffer overlap design is written, but RTL implementation
  has not started.

Current verified commands:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 6 tests
make npu-core-sim: PASS
make soc-sim: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make perf-report: PASS
build/perf/perf.json summary: jobs=53, workloads=7, total_cycles=63100
make test: PASS, 31 tests, 44.071s
```

Current key performance facts:

```text
CPU SoC sim finish time after DMA: 3765375000 ps
real_mnist_cnn_fc1_full_k_stream_tile0 total_cycles: 58784
data_mover.transfer_cycles: 36884
data_mover.words: 147536
data_mover.read_words: 147472
data_mover.write_words: 64
core.matmul cycles: 11520
```

Important interpretation:

- DMA reduced firmware staging wall-clock simulation time.
- NPU job cycles are unchanged by DMA because DMA staging happens before
  `PERF_JOB` starts.
- Explicit data mover counters now prove that full FC1 K-streaming still spends
  a large serial block in movement before/around compute.
- Next optimization should attack overlap, not bus width again.

Primary design doc for tomorrow:

```text
docs/design/k_stream_ping_pong_buffer.md
```

Tomorrow's recommended task order:

1. Re-read `docs/design/k_stream_ping_pong_buffer.md`.
2. Inspect current core host window and buffer layout in
   `hw/npu_core/rtl/npu_v0_top.sv`.
3. Decide the bank-select ABI:
   recommended first version is a core control register, not host address bits.
4. Add A/B bank storage in the core while keeping the normal matmul/softmax path
   unchanged.
5. Update wrapper K-stream FSM so prefetch of chunk `i+1` can run while core
   computes chunk `i`.
6. Run `make npu-core-sim`, `make soc-sim`, `make cpu-soc-sim`, `make test`, and
   `make perf-report`.

Expected success signal after overlap:

```text
data_mover.words stays roughly unchanged
core.matmul cycles stays roughly unchanged
real_mnist_cnn_fc1_full_k_stream_tile0 total_cycles drops
```

Do not start with:

- full 16-N-tile `fc1` layer expansion;
- conv lowering;
- changing `WORDS_PER_CYCLE` beyond 4;
- replacing PicoRV32;
- DMA counters, unless staging visibility becomes the immediate question.

## Session 51: Implement K-Streaming Ping-Pong Overlap

User asked whether the current NPU internal buffers already have banks and how
bank ownership should be understood. Decision:

- current `dram_a`, `dram_b`, `spad_a`, `spad_b`, and `acc_buf` were simple
  register arrays with no bank concept;
- a bank is an access-partitioning mechanism, not a new data type;
- bank ownership is temporal:
  - core owns the compute bank for the current chunk;
  - data mover owns the load bank for the next chunk;
  - ownership swaps at the chunk boundary after compute and prefetch are both
    complete;
- first implementation should bank only A/B staging and keep `acc_buf` single,
  because K-streaming requires one resident accumulator.

Implemented:

- Added bank principle and ownership explanation to
  `docs/design/k_stream_ping_pong_buffer.md`.
- Added A/B bank 1 in `hw/npu_core/rtl/npu_v0_top.sv`:
  - bank 0 keeps historical names `dram_a` / `dram_b`;
  - bank 1 uses `dram_a_bank1` / `dram_b_bank1`;
  - `spad_a`, `spad_b`, and `acc_buf` remain single-copy.
- Extended core control register `0x500`:
  - bit 0: `matmul_accumulate_enable`;
  - bit 1: `clear_accumulator` pulse;
  - bit 2: `host_write_bank`;
  - bit 3: `compute_bank_select`.
- The core latches `compute_bank_select` at launch so wrapper can switch
  `host_write_bank` for prefetch while the current program is executing.
- Updated `hw/npu_wrapper/rtl/npu_v0_opsched.sv` K-streaming flow:
  - chunk 0 is prefetched normally;
  - after launching chunk `i`, wrapper configures the next bank;
  - wrapper prefetches chunk `i+1` during `DESC_WAIT_CORE`;
  - wrapper advances only after `core_done_seen && next_prefetch_done`.

Validation:

```text
make npu-core-sim: PASS
make soc-sim: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make perf-report: PASS
make test: PASS, 31 tests, 48.635s
```

Measured result:

```text
real_mnist_cnn_fc1_full_k_stream_tile0 total_cycles:
  before: 58784
  after:  39217

data_mover.transfer_cycles: 36884 unchanged
data_mover.words:           147536 unchanged
data_mover.read_words:      147472 unchanged
data_mover.write_words:     64 unchanged
core.matmul cycles:         11520 unchanged
perf summary total_cycles:  43482
```

Interpretation:

- Functional output remains correct.
- The optimization did not reduce transferred data or MAC work.
- The cycle reduction comes from overlapping A/B chunk movement with core
  execution.

Next recommended work:

1. Add a perf regression assertion for this overlap result.
2. Improve perf timeline rendering so prefetch and compute overlap is visible
   as overlapping spans, not only aggregate counters.
3. Then extend real MNIST CNN `fc1` from one output N tile to all 16 N tiles and
   add bias/ReLU handling.

## Session 52: Add Performance Iteration Rule And Ping-Pong Regression

User requested a standing rule: every NPU performance optimization iteration
must document the design idea and measured perf benefit for later review.

Updated:

- Added `Performance Iteration Record Rule` to `docs/work_rules.md`.
- The rule requires each NPU perf/PPA iteration to document:
  - measured bottleneck before the change;
  - design idea and expected improvement;
  - affected modules and interface/control changes;
  - expected tradeoffs;
  - actual measured perf result;
  - verification commands;
  - whether the result is real RTL behavior, testbench-side profiling, or
    report/model accounting;
  - remaining gap and next performance step.

Also added the first ping-pong perf regression assertion:

- `test/rtl/test_perf_report.py` now models the full `fc1` single-N-tile
  K-stream PERF_JOB with the ping-pong result:
  - total cycles: 39217, below old serial baseline 58784;
  - `core_matmul_cycles`: 11520 stable;
  - `data_mover.transfer_cycles`: 36884 stable;
  - `data_mover.words`: 147536 stable;
  - `data_mover.read_words`: 147472 stable;
  - `data_mover.write_words`: 64 stable.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 6 tests
```

Next recommended work:

- Improve perf timeline rendering so overlapped prefetch and compute are visible
  as overlapping spans, not only aggregate counters.

Follow-up completed in the same session:

- `sw/tools/perf/report.py` now adds a `K prefetch overlap` span to the Data
  mover timeline for `matmul_k_stream` jobs.
- The span is placed inside the wrapper `wait_core` interval, starting at the
  same point as the NPU core lane. This makes ping-pong overlap visible in the
  HTML/JSON timeline instead of only in aggregate counters.
- `test/rtl/test_perf_report.py` asserts that the full `fc1` K-stream synthetic
  PERF_JOB includes this overlap span.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 6 tests
make test: PASS, 31 tests, 45.544s
```

## Session 53: Extend Real MNIST CNN FC1 To 16 Output N Tiles

User approved extending `fc1` from one output N tile to all 16 output N tiles
and asked where the split should live.

Decision:

- The split belongs in the tool-side firmware data emitter, not primarily in
  `main.c`.
- `sw/tools/firmware/emit_soc_cpu_smoke_data.py` plans
  `n_offset = 0, 8, ..., 120` with `plan_matmul_k_stream()`.
- Firmware should only loop over generated tile plans, stage the streams, launch
  descriptors, and check expected output tiles.

Implemented:

- `emit_soc_cpu_smoke_data.py` now generates 16 full `fc1` K-stream N-tile
  plans.
- `main.c` now runs 16 `SOC_NPU_JOB_OP_MATMUL_K_STREAM` descriptors and checks
  every `8x8` output tile.
- The A stream is staged once because it is shared across N tiles; each B stream
  is staged per N tile.
- The first attempted compact `uint8_t` ROM representation was functionally
  reasonable but made PicoRV32 unpacking too slow in simulation. It was replaced
  with `uint32_t` generated arrays so existing DMA staging can be used.
- `arch/configs/soc_v0.jsonc` now temporarily uses an 8 MiB simulation boot ROM
  and 4 MiB SRAM at `0x0080_0000` to hold the full-layer generated smoke data
  and staging buffers.
- `Makefile` now emits `soc_cpu_smoke.hex` with 2097152 words for the 8 MiB
  boot ROM.
- `sw/tools/perf/report.py` groups the 16 K-stream jobs as
  `real_mnist_cnn_fc1_full_k_stream_layer`.

Validation:

```text
make cpu-soc-sim: PASS, 68 PERF_JOB records
make perf-report: PASS
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report -v: PASS, 6 tests
make test: PASS, 31 tests, 213.655s
```

Perf result:

```text
jobs: 68
workloads: 7
total_cycles: 631737
real_mnist_cnn_fc1_full_k_stream_layer:
  jobs: 16
  total_cycles: 627472
  core_matmul_cycles: 184320
  data_mover.words: 2360576
  ping-pong baseline saved cycles vs old serial estimate: 313072
```

Important limitation:

- Full `fc1` matmul layer is now covered by CPU-controlled SoC RTL.
- `fc1` bias/ReLU is not yet applied to these NPU-produced tiles.
- Existing `fc2` smoke still uses tool-side precomputed `fc1_relu`.
- Current full-layer data staging embeds packed streams in the simulation boot
  ROM. This is a checkpoint, not the final loader/layout model.

Next recommended work:

1. Apply `fc1` bias/ReLU in firmware after the 16 output tiles.
2. Feed the resulting `fc1_relu` into the existing `fc2` tile path.
3. Replace packed-stream firmware bloat with stride/layout fields or a loader
   path.

## Session 54: Shift Mainline To ASIC-Oriented PPA And Transformer Workloads

User clarified the long-term project objective:

- there is no current FPGA board or private ASIC technology target;
- the desired end state is ASIC-oriented PPA analysis, accepting public-flow
  estimates before signoff-grade accuracy is possible;
- `npu_core` breakdown remains important, but the main NPU evaluation boundary
  must include wrapper/data mover and core together;
- Transformer/LLM inference, rather than MNIST CNN, is the long-term target
  application.

Decision:

- establish a public-ASIC PPA framework before further major datapath
  expansion;
- use `npu_subsystem` as the primary PPA boundary, with `npu_core` reported for
  attribution and `soc_reference` kept only as a system reference;
- retain real MNIST CNN as a functional/compatibility regression workload;
- make Transformer prefill/decode workloads the driver for future matrix,
  vector/reduction, precision, and memory-system decisions;
- retain the existing `hw/` layout and add stable future-facing directories
  incrementally rather than moving working RTL before a baseline exists.

Documentation added or updated:

- `docs/design/ppa_methodology.md`: public ASIC targets, measurement tops,
  area/timing/power/energy metrics, activity windows, memory accounting, and
  result contract;
- `docs/design/transformer_workloads.md`: prefill/decode distinction,
  micro-kernel progression, precision and external-memory reporting
  requirements;
- `docs/design/npu_subsystem.md`: primary PPA RTL top and external-memory
  boundary contract;
- `docs/project_plan.md`, `docs/target_architecture.md`, `docs/architecture.md`,
  `docs/work_rules.md`, and entry-point READMEs updated to reflect the new
  mainline.

Engineering scaffold implemented:

- added `hw/npu_subsystem/rtl/npu_subsystem_top.sv`, a structural top around
  the already verified `npu_v0_opsched`/data-mover/core hierarchy; it exposes
  memory externally and does not include CPU or oversized simulation memories;
- added `make npu-subsystem-elab` as an immediate structural verification gate;
- added initial PPA target assumptions in
  `arch/configs/ppa/sky130hd_v0.jsonc`;
- added result contract in `ppa/schema/ppa_result.schema.json`;
- added Transformer initial workload manifest in
  `workloads/manifests/transformer/transformer_micro_v0.jsonc`;
- added initial OpenROAD Flow Scripts design inputs for `npu_core` and
  `npu_subsystem`, each with a 100 MHz starting constraint;
- added PPA contract tests, now located at
  `test/ppa_contract/test_ppa_contract.py`, to verify schema, target, Transformer
  prefill/decode split, and subsystem elaboration.

Limitations:

- no OpenROAD, OpenLane, Yosys, or OpenSTA executable is currently available
  in the local environment, so no area/timing/power number is claimed in this
  session;
- the OpenROAD files are flow inputs only until the toolchain is installed and
  an actual report is produced;
- local-memory macro accounting and activity-driven power remain PPA1/PPA2
  work.

Validation performed while introducing the scaffold:

```text
make npu-subsystem-elab: PASS
make npu-core-sim: PASS
PYTHONPATH=sw/tools python -m unittest test.ppa_contract.test_ppa_contract -v: PASS, 4 tests
make test: PASS, 35 tests, 206.773s (includes the 4 new PPA tests)
```

Next work:

1. Make the first executable `sky130hd` area/timing flow available and extract
   `npu_core` and `npu_subsystem` machine-readable summaries.
2. Define job-scoped switching-activity capture for future power/energy
   extraction.
3. Add initial Transformer workload/golden handling once the PPA baseline
   execution path is usable.

## Session 55: Make Lightweight PPA Proxy The Immediate Execution Path

User questioned whether starting with full SKY130HD/OpenROAD/OpenLane
implementation was too heavy before the PPA framework and Transformer
workloads were mature.

Decision:

- keep ASIC PPA as the end goal, but use layered evidence;
- make `L0_proxy` the immediate executable level:
  - performance and traffic from real RTL `PERF_JOB` counters;
  - structural-area proxy from explicit hardware resource counts;
  - event-energy proxy from replaceable normalized coefficients;
- defer Yosys/ABC/OpenSTA mapped-area/timing to `L1_mapped`;
- defer activity-driven power to `L2_power`;
- retain OpenROAD/OpenLane and `sky130hd` as `L3_physical` selected-variant
  validation, not an immediate prerequisite.

Updated planning/docs:

- `docs/design/ppa_methodology.md` now defines `L0_proxy`, `L1_mapped`,
  `L2_power`, and `L3_physical`, including the claim boundary of each level;
- `docs/project_plan.md`, `docs/target_architecture.md`, `README.md`, and
  `docs/design/verification_strategy.md` now identify Level 0 proxy reporting
  as the current task.

Implemented Level 0 flow:

- `arch/configs/ppa/area_proxy_v0.jsonc`: current `npu_subsystem` structural
  resources and normalized coefficients;
- `arch/configs/ppa/energy_proxy_v0.jsonc`: event-energy coefficients and
  verified `8x8x8` matmul event derivation;
- `ppa/schema/ppa_proxy_report.schema.json`: Level 0 report contract;
- `sw/tools/ppa/proxy_report.py`: consumes `build/perf/perf.json`, produces
  JSON and HTML proxy reports;
- `make ppa-proxy-report`: full report entry point after RTL performance
  collection;
- `test/ppa_contract/test_ppa_contract.py`: tests proxy labeling, structural
  calculation, MAC derivation, and ping-pong summary.

First generated Level 0 results from the current RTL perf baseline:

```text
npu_subsystem structural area proxy: 6998.4 normalized_area_units
npu_subsystem local-state storage:    7968 bits
operator_smoke_matmul:
  measured cycles:                    81
  measured data_mover.words:          208
  event-energy proxy:                 1428.25 normalized_energy_units
real_mnist_cnn_fc1_full_k_stream_layer:
  measured cycles:                    627472
  measured data_mover.words:          2360576
  derived int8 MAC operations:        9437184
  event-energy proxy:                 19037380.0 normalized_energy_units
ping-pong versus recorded serial baseline:
  measured cycles saved:              313072
  modeled active-duration energy saved only: 78268.0 normalized_energy_units
```

Interpretation:

- cycles and movement are real RTL measurements from the current SoC
  simulation;
- area is not `um^2` and is not synthesized area;
- energy is not joules or watts and currently excludes external-memory energy;
- the ping-pong result deliberately does not claim MAC or moved-data energy
  savings, because those work counters remain unchanged.

Validation:

```text
make ppa-proxy-report: PASS
PYTHONPATH=sw/tools python -m unittest test.ppa_contract.test_ppa_contract -v: PASS, 5 tests
make test: PASS, 36 tests, 208.466s
```

Implementation note:

- the PPA contract test package is named `test/ppa_contract/` rather than
  `test/ppa/`, because a top-level test package named `ppa` shadows the
  implementation package `sw/tools/ppa` during `unittest discover`.

## Session 56: Require Candidate-Versus-Baseline PPA Deltas

User specified that each future NPU iteration should show its PPA difference
relative to the older architecture, especially where the new version improves.

Decision:

- every PPA-affecting NPU iteration after a named baseline exists must emit a
  candidate-versus-baseline delta report;
- the report must show advantages and costs together rather than filtering out
  unfavorable metrics;
- Level 0 delta reports distinguish real RTL performance counters from
  normalized area/energy proxies.

Implemented:

- added the baseline delta rule to `docs/design/ppa_methodology.md` and
  `docs/work_rules.md`;
- added a checked-in first baseline:
  `ppa/baselines/l0/npu_v0_a2_serial_k_stream.json`;
- added serial K-stream structural configuration:
  `arch/configs/ppa/area_proxy_v0_serial_k_stream.jsonc`;
- extended `sw/tools/ppa/proxy_report.py` and `make ppa-proxy-report` with
  baseline comparison support;
- extended the HTML/JSON output with `comparison`, `improvements`, and `costs`
  sections;
- extended `test/ppa_contract/test_ppa_contract.py` to protect delta behavior.

First named comparison:

```text
baseline:  npu_v0_a2_serial_k_stream
candidate: npu_v0_a2_ping_pong
workload:  real_mnist_cnn_fc1_full_k_stream_layer

measured cycles:       940544 -> 627472, -313072 (-33.286%), improvement
measured mover words:  2360576 -> 2360576, invariant
derived MAC work:      9437184 -> 9437184, invariant
energy proxy:          19115648.0 -> 19037380.0, -78268.0 (-0.409%), improvement
area proxy:            6947.2 -> 6998.4, +51.2 (+0.737%), cost
```

Interpretation:

- ping-pong is favorable for this workload because it removes a large serial
  latency component for a small normalized storage-area increase;
- mover words and MAC work do not fall, so this is an overlap improvement, not
  a reduction in arithmetic or transfer work;
- external-memory energy remains unknown at Level 0 and is explicitly listed
  as an unavailable decision metric.

Validation:

```text
make ppa-proxy-report: PASS, generated candidate-versus-baseline JSON/HTML
PYTHONPATH=sw/tools python -m unittest test.ppa_contract.test_ppa_contract -v: PASS, 6 tests
make test: PASS, 37 tests, 204.151s
```

## Session 57: Stable Manifest And Frozen Level 0 Baseline

Goal:

- remove report-side ownership of current firmware workload ordering;
- make the active Level 0 baseline a validated frozen report;
- reduce report-only iteration time without weakening the full SoC/PPA gate;
- document remaining debt before introducing further datapath or Transformer
  implementation.

Implemented:

- `sw/tools/firmware/emit_soc_cpu_smoke_data.py` now emits
  `build/perf/workload_manifest.json` from the same enabled fixture counts
  that build the C firmware data; the full current run emits 68 jobs and
  carries derived `fc1` `k_chunks=1152`;
- `sw/tools/firmware/emit_soc_cpu_smoke.py` emits a matching two-job manifest
  for the minimal generated-firmware fallback;
- `make perf-report` consumes the generated manifest directly rather than
  copying a separately maintained fixed job list;
- added active frozen baseline
  `ppa/baselines/l0/npu_v0_a2_serial_k_stream_proxy.json`; the older
  `npu_v0_a2_serial_k_stream.json` remains only as recorded source evidence;
- `sw/tools/ppa/schema_check.py` now checks non-negative metrics, area/energy
  contribution sums, duplicate workload names, and comparable delta
  consistency in addition to required fields;
- added `make ppa-proxy-from-perf` for derived-report work from an existing
  valid perf artifact; `make ppa-proxy-report` still reruns the complete
  RTL-performance path and invokes the fast target only after it succeeds;
- recorded the baseline stabilization debt/order in `docs/project_plan.md`.

Deliberately deferred:

- wrapper-visible perf CSR RTL was not introduced in this batch. The next
  implementation must first define snapshot/clear/overflow semantics and
  compare CSR values against the current testbench-sampled reference counters
  before changing report provenance.

Validation:

```text
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report test.ppa_contract.test_ppa_contract -v: PASS, 18 tests
make firmware-smoke: PASS
make perf-report: PASS, generated manifest drives 68 jobs / 7 workloads
make ppa-proxy-from-perf: PASS, frozen baseline and candidate validation pass
make ppa-proxy-report: PASS, full perf-to-proxy gate and validation pass
make test: PASS, 43 tests, 204.889s
```

## Session 58: First Wrapper-Visible Performance Snapshot CSRs

Goal:

- reduce documentation drift around the immediate execution path;
- introduce the first CPU-visible architectural perf counter subset without
  silently changing existing `PERF_JOB`/PPA provenance;
- correlate new RTL counters against the established testbench-sampled
  reference.

Documentation alignment:

- updated `docs/target_architecture.md`,
  `docs/design/performance_instrumentation.md`,
  `docs/design/perf_counter_csr_plan.md`, `docs/project_plan.md`, and related
  active entry points so the immediate sequence is now consistently:
  perf CSR correlation, then Transformer workload/external-memory identity,
  then later mapped/physical evidence;
- marked the old A2 next-step checklist as historical completed work rather
  than an active implementation queue.

Implemented:

- extended `arch/configs/npu_wrapper_v0.jsonc` with first-batch perf CSRs:
  `PERF_CTRL`, `PERF_STATUS`, cycle counters for total/core/data mover, data
  mover words, and NPU SRAM read/write words;
- defined completed-job snapshot semantics: a new launch clears private
  running accumulators while preserving the previous visible snapshot;
  completion atomically replaces the snapshot; idle `PERF_CTRL.clear` removes
  the retained snapshot; counters are 32-bit saturating with an overflow bit;
- implemented the synthesizable accumulator/snapshot bank and MMIO read path
  in `hw/npu_wrapper/rtl/npu_v0_opsched.sv`;
- deliberately deferred architectural `mac_ops` and `instr_count` until
  committed-event signals have stable contracts;
- kept `PERF_JOB` and Level 0 report data TB-sampled, preserving the existing
  measurement provenance during this correlation step.

Verification changes:

- `soc_tb` now reads the new CSR window through MMIO and verifies snapshot
  valid/clear behavior on the legacy direct-window smoke path;
- `soc_cpu_tb` now checks every descriptor job snapshot against its existing
  TB reference for total, core, data-mover and SRAM-boundary summary counters;
- the full active firmware workload still emits 68 jobs / 7 workloads and its
  existing cycle results are unchanged.

Validation:

```text
make npu-subsystem-elab: PASS
make soc-sim: PASS, CSR MMIO read/clear smoke covered
make cpu-soc-sim: PASS, 68 descriptor job CSR correlations covered
make validate-arch: PASS
make demo: PASS
make npu-core-sim: PASS
make ppa-proxy-report: PASS, 68 jobs / 7 workloads, Level 0 baseline comparison validated
make test: PASS, 43 tests, 224.150s
```

Remaining next work:

1. Extend workload identity with shape, precision, activity scope, and
   explicit external-memory/KV-cache accounting before using Transformer
   results as decision evidence.
2. Keep report provenance TB-sampled until a separate reviewed migration to
   CSR consumption is made.
3. Introduce `L1_mapped` only after those workload identities are sufficiently
   comparable.

## Session 59: Pre-Transformer Contract And Perf Review

Goal:

- audit handwritten implementation/test/report assumptions before adding
  Transformer workloads or extending the core surface;
- document the current perf signal-to-CSR-to-report path for code review;
- decide whether existing internal performance signals are removable now that
  snapshot CSRs exist.

Findings:

- first-batch CSR core-cycle qualification is complete for descriptor jobs but
  undercounts legacy direct-window execution because it only samples the
  sustained core interval while `desc_state == DESC_WAIT_CORE`;
- `npu_v0_opsched.sv` and `npu_v0_top.sv` independently hardcode the internal
  core host window/control ABI and use a repeated numeric core state for perf
  observation; these facts are not generated from `arch/configs`;
- firmware job launch order and generated workload-manifest order remain
  parallel conventions, while the warned report fallback and its tests retain
  old fixed order/count inference;
- descriptor-job CSR correlation currently inspects internal snapshot storage,
  not CPU-visible MMIO reads; the legacy MMIO smoke covers status/total/clear
  only;
- DMA register offsets and test-status values are still mirrored manually
  across RTL and firmware/test surfaces.

Performance signal decision:

- keep `npu_v0_data_mover.perf_*`: it feeds both the architectural CSR
  accumulator and the existing TB report reference;
- retain zero-valued `perf_stall` as a declared schema field until a reviewed
  schema change or real stall modeling;
- do not delete TB phase probes until report provenance moves to CSR summaries
  or an explicit phase/event contract.

Documentation changes:

- expanded `docs/design/performance_instrumentation.md` with the data-mover
  event, wrapper CSR aggregation, TB correlation and report-processing code
  walk-through, along with signal-retention and report-compaction decisions;
- updated `docs/project_plan.md` to place CSR semantics/MMIO coverage,
  generated internal ABI, and generated job identity ahead of Transformer
  expansion.

Scope:

- review/documentation only in this session; no RTL, software or test behavior
  was changed.

## Session 60: Generated Contracts And Perf CSR Repair

Goal:

- repair the consistency issues identified before Transformer work;
- keep production report inputs compact and reviewable;
- rerun the audit after implementation to identify intentional residual debt.

Implemented:

- moved the NPU core host windows, host control bits, DMA registers/bit fields,
  test-status values, wrapper register fields, and report timing model inputs
  under generated `arch/configs` ownership;
- deleted the stale checked-in `hw/npu_wrapper/rtl/npu_v0_regs.svh` duplicate;
  builds now consume the generated `build/npu_wrapper/npu_v0_regs.svh` only;
- added descriptor-carried generated `job_id`, so firmware, `PERF_JOB`, and
  the generated workload manifest share one workload identity contract;
- exposed explicit core perf events and made wrapper CSR accumulation consume
  those events for both descriptor and legacy launch paths, fixing legacy
  matmul undercount and removing observation of core FSM encodings;
- made firmware read and validate the full completed-job perf CSR snapshot over
  MMIO after every descriptor; TB correlation remains as an independent
  reference until report provenance is deliberately migrated;
- changed the production report model to load architecture/SoC config values,
  moved the named FC1 comparison baseline into generated manifest metadata,
  and made per-job HTML content lazy/collapsed by default.

Current measured result:

```text
make ppa-proxy-report: PASS
jobs/workloads/total_cycles: 68 / 7 / 631805
real_mnist_cnn_fc1_full_k_stream_layer: 627488 cycles
FC1 candidate energy proxy: 19037384.0 normalized_energy_units
serial-to-ping-pong delta: -313056 cycles, -78264.0 energy proxy units
```

The descriptor now reads one additional `job_id` word, adding one cycle per
descriptor job relative to the previous current-result documentation.

Validation:

```text
make validate-arch demo: PASS
make npu-core-sim soc-sim npu-subsystem-elab: PASS
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report test.ppa_contract.test_ppa_contract -v: PASS, 19 tests
make ppa-proxy-report: PASS
make test: PASS, 44 tests, 225.539s
git diff --check: PASS
```

Residual debt:

1. `sw/tools/perf/report.py::infer_workloads()` still preserves fixed legacy
   log ordering/counts and the recorded serial baseline when invoked without a
   manifest. Production reporting supplies the generated manifest and does not
   use that path.
2. `PERF_JOB`/PPA report provenance remains TB-produced JSON; firmware now
   proves CSR accessibility and basic validity, but switching report ingestion
   to CSR values remains a separate reviewable change.
3. `mac_ops`, `instr_count`, timeout, and execution-error CSRs remain deferred
   until committed-event/error contracts exist.

## Session 61: CSR-Sourced Performance Provenance And Transformer Entry

Goal:

- retire TB-aggregated production performance records in favor of CPU-visible
  architectural CSR snapshot reads;
- classify the two remaining review items before beginning Transformer work.

Implemented:

- extended the wrapper snapshot contract with `PERF_JOB_ID`, `PERF_OP_TYPE`,
  `PERF_DATA_MOVER_READ_WORDS`, and `PERF_DATA_MOVER_WRITE_WORDS`;
- made firmware read those fields through MMIO and validate identity plus
  directional-word conservation for every descriptor job;
- replaced the `soc_cpu_tb` production profiler output: it now observes the
  firmware's actual CSR read responses and emits `PERF_JOB` values labeled
  `architectural_perf_csr_snapshot`;
- retained a reduced hierarchical event accumulator only as a CSR
  implementation equality assertion, not as report/PPA input;
- changed `perf.json` and PPA proxy provenance to
  `measured_architectural_perf_csr_snapshot`;
- stopped CSR-sourced reports from drawing unmeasured fine-grain phase or
  overlap spans; legacy phase-rich log replay remains supported.

Remaining scoped work:

1. `infer_workloads()` remains only for warned legacy-log replay without a
   manifest. Production report targets already use generated manifest identity;
   removing fallback now would primarily be compatibility/test cleanup.
2. `mac_ops`, `instr_count`, timeout and execution-error CSRs need stable
   committed-event semantics. `mac_ops` becomes important before comparing
   Transformer shapes with partially utilized matrix tiles; it is planned
   rather than inferred from existing phase cycles.

Transformer entry decision:

- begin with executable INT8 prefill projection GEMM and `M=8` decode
  skinny-GEMM proxy workloads on the existing matmul/K-stream path;
- add manifest/report identity for scenario, shape, precision, activity scope,
  and external/KV-cache traffic before using results for architecture choices;
- model KV-cache bytes/token alongside decode results, then choose whether the
  first RTL extension is command/layout support, skinny utilization, memory
  movement, or reduction/SFU support.

Validation:

```text
make npu-wrapper-spec soc-spec rtl-fixtures npu-core-sim soc-sim npu-subsystem-elab firmware-smoke-c: PASS
make ppa-proxy-report: PASS
perf/PPA performance provenance: measured_architectural_perf_csr_snapshot
jobs/workloads/total_cycles: 68 / 7 / 631805
PYTHONPATH=sw/tools python -m unittest test.rtl.test_perf_report test.ppa_contract.test_ppa_contract -v: PASS, 21 tests
make test: PASS, 46 tests, 226.158s
git diff --check: PASS
```
