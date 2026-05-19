# Docs Map

[TOC]

本文档说明 `docs/` 目录下各文件的当前定位。

## Current Entry Points

| Document | Purpose |
| --- | --- |
| `architecture.md` | 当前 NPU SoC 架构、计算逻辑、CPU/NPU 软件硬件交互协议 |
| `design/README.md` | 当前活跃模块级设计文档索引 |
| `design/soc_architecture.md` | SoC 顶层、memory map、bus、ROM/SRAM、NPU 接入方式的详细设计 |
| `design/npu_wrapper.md` | NPU wrapper、descriptor FSM、core host window、A2 data mover 的详细设计 |
| `design/npu_core.md` | NPU core、内部 memory、uop 执行、matmul array、softmax 路径的详细设计 |
| `design/software_hardware_flow.md` | compiler/assembler/firmware/descriptor/wrapper/core 的软硬件交互流程 |
| `design/performance_instrumentation.md` | cycle 级 perf 计数、PERF_JOB、HTML timeline、counter 下沉策略 |
| `design/verification_strategy.md` | 当前测试层级、验证入口、coverage 边界和下一步测试计划 |
| `digits_classifier_workload.md` | 下一阶段真实 workload：8x8 手写数字分类闭环 |
| `real_mnist_cnn_workload.md` | 真实开源 MNIST CNN：外部 safetensors 权重、MNIST 图片、`fc2` SoC RTL 验证 |
| `target_architecture.md` | 业界架构资料提炼、长期目标 NPU 架构、分阶段扩展计划 |
| `matmul_array_a1.md` | A1 matmul array 的接口、时序、性能目标和验证标准 |
| `data_mover_a2.md` | A2 data mover、scratchpad banking、program streaming 的临时约束和下一步计划 |
| `work_rules.md` | 协作规则、source-of-truth 规则、spec change protocol |
| `collaboration_journal.md` | 每次重要讨论和实现决策的纪要 |
| `bugfix_list.md` | 代表性 bug、定位过程、根因和修复经验 |

## Planning And Bring-Up Notes

| Document | Status |
| --- | --- |
| `project_plan.md` | 里程碑和模块 ownership，低频更新 |
| `fpga_bringup.md` | FPGA 方向说明，当前不是日常入口 |

## Process Notes

| Document | Purpose |
| --- | --- |
| `process/github_publish.md` | 发布到 GitHub 的流程说明 |

## Archived Notes

`docs/archive/` 下是历史设计草案，不是当前入口。需要了解演进背景时可以看，
但实现和 review 应优先参考 Current Entry Points。
