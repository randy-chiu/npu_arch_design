# NPU Subsystem PPA Boundary / NPU 子系统 PPA 边界

[TOC]

## 1. Purpose / 目的

`npu_subsystem_top` is the primary synthesis and PPA comparison top. It places
the current CPU-visible wrapper, descriptor/data mover control, and NPU core
behind stable external control and memory interfaces, without pulling in the
simulation CPU, boot ROM, oversized staging SRAM, or test-status peripheral.

`npu_subsystem_top` 是主要的综合与 PPA 对比 top。它保留 NPU 执行一次 job 所需的
wrapper、descriptor/data mover 与 core 代价，但不把当前功能仿真用的 CPU、
超大 boot ROM/SRAM 与 test status 外设误计入 NPU 主面积和功耗。

## 2. First Implementation Boundary / 第一版实现边界

The first implementation is deliberately a structural boundary only:

```text
npu_subsystem_top
  -> npu_v0_opsched
      -> npu_v0_data_mover
      -> npu_v0_top
          -> matmul_array
```

It does not change the descriptor ABI, movement behavior, ping-pong behavior,
core uops, or SoC simulation path. This keeps the first PPA baseline tied to
already verified RTL behavior.

第一版只新增结构边界，不修改 descriptor ABI、搬运行为、ping-pong 行为、core
uop 或现有 SoC 仿真路径。这样第一份 PPA baseline 可对应当前已验证功能。

## 3. External Interface / 外部接口

| Interface | Direction | Meaning |
| --- | --- | --- |
| `ctrl_*` | host to/from subsystem | current 32-bit MMIO control/register request path |
| `mem_*` | subsystem to/from external memory boundary | descriptor/program/tensor SRAM-side movement interface |
| `clk`, `rst_n` | input | synthesis clock and active-low reset |

The memory interface is intentionally exposed outside the PPA top. A later
flow may bind it to an SRAM macro or traffic/activity harness, but it must not
silently instantiate the current multi-megabyte simulation SRAM as part of the
primary NPU result.

memory 接口刻意暴露在 PPA top 外部。后续可以绑定 SRAM macro 或 activity
harness，但不得默认把当前多 MiB 的仿真 SRAM 综合进主要 NPU 结果。

## 4. Measurement Use / 测量用途

| Result | Interpretation |
| --- | --- |
| `npu_core` PPA | compute/core-state attribution |
| `npu_subsystem` PPA | primary decision result including wrapper/data mover/core |
| SRAM/external model | separately named memory cost or energy estimate |
| `soc_reference` PPA | optional integration reference only |

## 5. Follow-Up / 后续工作

1. Add ASIC flow inputs for `npu_subsystem_top`.
2. Decide local buffer macro-accounting policy before publishing area claims.
3. Add activity harness/capture window for kernel-scoped power analysis.
4. Introduce stable debug/perf counter visibility only when the PPA flow needs
   it; do not modify the existing functional launch contract unnecessarily.

