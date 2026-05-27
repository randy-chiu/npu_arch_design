# SoC DMA / Preload Engine Design

[TOC]

This document describes the first SoC DMA block used to reduce firmware staging
time for large NPU test tensors.

本文档描述第一版 SoC DMA，用于降低大规模 NPU 测试 tensor 从 ROM 搬到 SRAM 的
firmware staging 时间。

## 1. Goal / 目标

Before this block, firmware copied generated constant arrays from boot ROM to
SRAM with scalar PicoRV32 stores:

```text
for each word:
  load 32-bit word from ROM
  store 32-bit word to SRAM
```

即使 SRAM CPU 端口已经是 4-lane / 128-bit 形态，PicoRV32 本身仍然只能每条
load/store 访问 32-bit，所以大数组 staging 仍然很慢。

The DMA changes the staging path to:

```text
CPU writes DMA registers
DMA reads boot ROM words
DMA writes up to SOC_SRAM_CPU_LANES words per cycle into SRAM
CPU polls DMA done
```

DMA 的目标不是替代 NPU wrapper/data mover，也不改变 NPU job 内部 cycle。它只
减少 NPU job 启动之前 CPU 准备 SRAM 输入数据的时间。

## 2. Register Contract / 寄存器合同

The DMA base address, register offsets, and control/status bit fields are
owned by `arch/configs/soc_v0.jsonc`; `make soc-spec` emits the constants
consumed by RTL and firmware:

```text
SOC_DMA_BASE = 0x4000_0000
```

Register map:

| Offset | Name | Direction | Meaning |
| ---: | --- | --- | --- |
| `0x000` | `CTRL` | CPU write | bit 0 starts a copy when idle |
| `0x004` | `STATUS` | CPU read | bit 0 done, bit 1 busy, bit 2 idle |
| `0x008` | `SRC_ADDR` | CPU RW | absolute boot ROM source byte address |
| `0x00c` | `DST_ADDR` | CPU RW | absolute SRAM destination byte address |
| `0x010` | `WORDS` | CPU RW | number of 32-bit words to copy |

第一版只支持 boot ROM 到 SRAM 的 copy。`SRC_ADDR` 和 `DST_ADDR` 都使用 CPU
看到的绝对 SoC 地址。

## 3. Datapath / 数据通路

Current RTL modules:

| Module | Role |
| --- | --- |
| `soc_dma` | MMIO control, copy state, ROM read address, SRAM lane mask/data |
| `boot_rom` | adds a second packed DMA read port |
| `simple_sram` | reuses the widened CPU/preload write port |
| `soc_cpu_top` | muxes CPU scalar SRAM requests and DMA wide SRAM writes |

当前数据路径：

```text
PicoRV32
  -> simple_bus
  -> DMA MMIO registers

soc_dma
  -> boot_rom DMA read port
  -> simple_sram CPU/preload wide write port
```

The CPU can continue executing from the scalar ROM port while DMA reads through
the second ROM port. This is a simulation simplification; a later flash/ROM bus
model should add arbitration or explicit bandwidth limits.

CPU 可以继续通过标量 ROM port 取指，同时 DMA 通过第二个 ROM port 读数据。这是当前
仿真模型的简化；后续真实 flash/ROM bus 模型需要加入仲裁或带宽限制。

## 4. Lane Behavior / Lane 行为

DMA writes up to `SOC_SRAM_CPU_LANES` words per cycle. For the current spec:

```text
SOC_SRAM_CPU_LANES = 4
```

Unaligned destination addresses are supported at word granularity. If the
destination starts at lane 1, the first DMA cycle writes lanes 1..3, then later
cycles write lanes 0..3.

DMA 每拍最多写 `SOC_SRAM_CPU_LANES` 个 word。目标地址支持 word 粒度非 4-lane
对齐：如果目标从 lane 1 开始，第一拍写 lane 1..3，后续拍再写 lane 0..3。

## 5. Firmware API / Firmware API

Firmware uses:

```c
void dma_copy_words(uint32_t *dst, const uint32_t *src, uint32_t len);
```

`soc_cpu_smoke` routes its existing `copy_words()` helper through this API, so
the high-level workload code remains unchanged.

`soc_cpu_smoke` 当前把已有的 `copy_words()` helper 接到 `dma_copy_words()`，因此
上层 workload 代码不需要知道每个数组是 CPU 标量 copy 还是 DMA copy。

## 6. Verification / 验证

Acceptance criteria for this step:

```text
make soc-spec
make soc-sim
make cpu-soc-sim
make test
make perf-report
```

Expected NPU job cycle counts should remain unchanged because DMA staging is
outside the current `PERF_JOB` interval. CPU-controlled simulation finish time
should drop because large ROM-to-SRAM copies no longer execute as PicoRV32
scalar store loops.

预期 NPU job cycle 不变，因为 DMA staging 发生在当前 `PERF_JOB` 区间之外。但
CPU-controlled simulation 的 finish time 应明显下降，因为大数组 ROM-to-SRAM copy
不再由 PicoRV32 逐 word store 完成。

## 7. Limitations / 限制

- only ROM-to-SRAM copy is supported;
- no source/destination range error reporting;
- no interrupt;
- no contention model between DMA and CPU SRAM access beyond a simple mux;
- no performance counters for DMA cycles/words yet;
- boot ROM has a second ideal DMA read port.

- 当前只支持 ROM-to-SRAM copy；
- 还没有源/目标地址范围错误上报；
- 还没有 interrupt；
- DMA 和 CPU SRAM 访问之间只有简单 mux，没有完整仲裁模型；
- 还没有 DMA cycle/word 计数器；
- boot ROM 目前有一个理想化的第二 DMA read port。

## 8. Next Work / 后续工作

Near-term follow-up:

1. Add DMA counters to the testbench and performance report.
2. Add error bits for unsupported source/destination ranges.
3. Replace the ideal second ROM port with a bus/arbitration model when external
   flash/DDR modeling starts.
4. Return to NPU-side optimization: explicit data mover counters, overlap, and
   scratchpad banking.

近期后续工作：

1. 在 testbench 和 performance report 中加入 DMA 计数。
2. 为不支持的源/目标地址范围增加 error bit。
3. 后续引入外部 flash/DDR 模型时，用 bus/arbitration 替代理想第二 ROM port。
4. 回到 NPU 侧优化：显式 data mover counters、搬运/计算 overlap、scratchpad
   banking。
