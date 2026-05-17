# Docs Map

[TOC]

本文档说明 `docs/` 目录下各文件的当前定位。

## Current Entry Points

| Document | Purpose |
| --- | --- |
| `architecture.md` | 当前 NPU SoC 架构、计算逻辑、CPU/NPU 软件硬件交互协议 |
| `target_architecture.md` | 业界架构资料提炼、长期目标 NPU 架构、分阶段扩展计划 |
| `matmul_array_a1.md` | A1 matmul array 的接口、时序、性能目标和验证标准 |
| `data_mover_a2.md` | A2 data mover、scratchpad banking、program streaming 的临时约束和下一步计划 |
| `code_structure_review.md` | 当前代码结构、逐文件走读、验证流程和 review 辅助说明 |
| `work_rules.md` | 协作规则、source-of-truth 规则、spec change protocol |
| `collaboration_journal.md` | 每次重要讨论和实现决策的纪要 |
| `bugfix_list.md` | 代表性 bug、定位过程、根因和修复经验 |

## Planning And Bring-Up Notes

| Document | Status |
| --- | --- |
| `project_plan.md` | 里程碑和模块 ownership，低频更新 |
| `soc_bringup.md` | 最小 SoC bring-up 计划和历史设计说明，部分内容已被 `architecture.md` 固化为当前态 |
| `fpga_bringup.md` | FPGA 方向说明，当前不是日常入口 |

## Process Notes

| Document | Purpose |
| --- | --- |
| `process/github_publish.md` | 发布到 GitHub 的流程说明 |

## Archived Notes

`docs/archive/` 下是历史设计草案，不是当前入口。需要了解演进背景时可以看，
但实现和 review 应优先参考 Current Entry Points。
