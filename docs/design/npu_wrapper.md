# NPU Host Wrapper Design

[TOC]

This document describes the thin CPU-visible `npu_v0_wrapper`.

## 1. Role

The wrapper is the host boundary between the CPU-visible SoC and the internal
NPU core system. Its responsibilities are:

- expose memory-mapped control/status registers;
- forward CPU commands and descriptor addresses into the NPU core system;
- publish `done/busy/idle`, IRQ, error, and performance information produced
  by the NPU core system;
- isolate the CPU-facing bus contract from future internal NPU changes.

The wrapper does not move tensor data, execute descriptor state machines, or
schedule compute engines. Those responsibilities belong to the NPU core
system. The wrapper owns the CPU-visible register file and translates an
accepted `CTRL.start` write into an explicit core command.

```text
CPU / SoC bus
  -> npu_v0_wrapper
      -> cmd_valid + cmd_desc_addr
      <- cmd_ready + busy + done + perf_snapshot
      -> npu_v0_core_system
```

Formal wrapper/core command interface:

| Signal | Direction | Meaning |
| --- | --- | --- |
| `cmd_valid` | wrapper to core | one-cycle descriptor command submission |
| `cmd_desc_addr` | wrapper to core | descriptor SRAM address captured by wrapper |
| `cmd_ready` | core to wrapper | core can accept a new descriptor command |
| `core_busy` | core to wrapper | submitted command remains in progress |
| `core_done` | core to wrapper | one-cycle completed-job event |
| `core_irq` | core to wrapper | completed/error event eligible for interrupt |
| `perf_snapshot_*` | core to wrapper | completed-job measurement snapshot |

`STATUS.done`, `STATUS.busy`, IRQ state, and visible perf CSRs are wrapper
state derived from this interface. They are not implemented by forwarding the
CPU bus into the core.

## 2. Register Interface

Source of truth:

```text
arch/configs/npu_wrapper_v0.jsonc
```

Generated files:

```text
build/npu_wrapper/npu_v0_regs.svh
build/npu_wrapper/npu_v0_regs.h
```

Key registers:

| Register | Offset | Direction | Meaning |
| --- | ---: | --- | --- |
| `CTRL` | `0x000` | CPU write | bit 0 starts a job |
| `STATUS` | `0x004` | CPU read | bit 0 done, bit 1 busy, bit 2 idle |
| `VERSION` | `0x008` | CPU read | wrapper version |
| `IRQ_ENABLE` | `0x00c` | CPU RW | reserved for interrupt flow |
| `IRQ_STATUS` | `0x010` | CPU RW | reserved for interrupt flow |
| `DESC_ADDR` | `0x020` | CPU write | absolute SRAM address of job descriptor |
| `PERF_CTRL` | `0x040` | CPU write | bit 0 clears retained perf snapshot while idle |
| `PERF_STATUS` | `0x044` | CPU read | bit 0 valid, bit 1 running, bit 2 overflow |
| `PERF_TOTAL_CYCLES` | `0x048` | CPU read | completed job elapsed cycles |
| `PERF_CORE_ACTIVE_CYCLES` | `0x04c` | CPU read | completed job core active cycles |
| `PERF_CORE_MATMUL_CYCLES` | `0x050` | CPU read | completed job matmul cycles |
| `PERF_DATA_MOVER_*` | `0x054` - `0x064` | CPU read | completed job mover cycle/word counters |
| `PERF_SRAM_*_WORDS` | `0x068` - `0x06c` | CPU read | completed job NPU SRAM-boundary traffic |
| `PERF_JOB_ID` / `PERF_OP_TYPE` | `0x070` - `0x074` | CPU read | completed descriptor identity |
| `PERF_DATA_MOVER_READ_WORDS` / `PERF_DATA_MOVER_WRITE_WORDS` | `0x078` - `0x07c` | CPU read | completed directional mover traffic |

Legacy A/B/C/X/Y/program windows are retained for older direct-window smoke
tests. New firmware should use the descriptor path.

The first perf CSR bank uses internal running accumulators and a completed-job
snapshot. A new launch does not destroy the previous visible snapshot; normal
descriptor completion atomically publishes the new one. Counters saturate at
32 bits and report overflow through `PERF_STATUS`. `mac_ops`, uop counts, and
phase-detail counters remain outside this first stable CSR contract.

第一批 perf CSR 使用“内部运行计数 + 已完成 job 快照”的结构。新 job 启动时不会破坏
上一个可见快照，descriptor 完成后才原子发布新值。计数器为 32 位饱和计数，并通过
`PERF_STATUS` 报告 overflow；`mac_ops`、uop 数量和更细的 phase counter 暂不进入
本批合同。

## 3. Forwarded Descriptor Contract

The descriptor ABI is owned by `arch/configs/soc_v0.jsonc` because it is shared
between CPU firmware and RTL.

Current layout:

| Word | Field | Meaning |
| ---: | --- | --- |
| 0 | `op_type` | `SOC_NPU_JOB_OP_MATMUL` or `SOC_NPU_JOB_OP_SOFTMAX` |
| 1 | `program_addr` | SRAM base address of encoded uops |
| 2 | `program_words` | uop word count |
| 3 | `input0_addr` | SRAM base of A or X |
| 4 | `input0_words` | input0 word count |
| 5 | `input1_addr` | SRAM base of B for matmul |
| 6 | `input1_words` | input1 word count |
| 7 | `output_addr` | SRAM output buffer |
| 8 | `output_words` | output word count |
| 9 | `k_chunks` | K-stream chunk count; zero for non-stream operators |
| 10 | `job_id` | generated workload identity emitted in `PERF_JOB` |

The NPU core command processor assumes word-aligned 32-bit addresses and currently truncates
transfer lengths through 8-bit counters in the movement path. This is acceptable
for Phase 0/A2 bring-up and must be widened before larger tiles.

## 4. Wrapper Activity Semantics

The wrapper is active only while accepting or returning a CPU-visible bus
transaction. The NPU may remain busy after launch while the wrapper is idle.

```text
wrapper active != NPU busy
wrapper active != command processor active
wrapper active != data mover active
```

Legacy A/B/C/X/Y/program windows remain a compatibility path. They use a
separate internal debug-window request interface and are not part of the
descriptor command protocol or production scheduler timing.

## 5. Internal Core Host Window Mapping

The NPU core command processor converts descriptor movement into compute-cluster host addresses:

| Core host window | Address range | Meaning |
| --- | ---: | --- |
| A | `0x000` - `0x03f` | matmul input A |
| B | `0x100` - `0x13f` | matmul input B |
| C | `0x200` - `0x23f` | matmul output C |
| X | `0x300` - `0x307` | softmax input X |
| Y | `0x380` - `0x387` | softmax output Y |
| program | `0x400` - `0x40f` | encoded uop `instr_mem` |

The internal map and accumulator/bank control bits are owned by
`arch/configs/npu_v0.jsonc` under `rtl.host_map` and `rtl.control_bits`. They
are consumed by the core implementation; the wrapper does not decode or
schedule these internal addresses.

This host window is an internal preload/readback path. It is not the long-term
NPU memory architecture.

## 6. Timing Semantics

Current transfer timing is:

```text
cycles ~= setup_cycles + ceil(words / words_per_cycle)
```

With `WORDS_PER_CYCLE=4` and `SETUP_CYCLES=0`, and with the descriptor ABI
including the generated `job_id` word, the verified launch-to-done baseline is:

```text
matmul total cycles: 81
softmax total cycles: 31
```

当前传输时序为：

```text
cycles ~= setup_cycles + ceil(words / words_per_cycle)
```

对于 `WORDS_PER_CYCLE=4`、`SETUP_CYCLES=0`，当前验证过的 NPU job 基线为：

```text
matmul total cycles: 81
softmax total cycles: 31
```

## 7. Full FC1 Data-Movement Improvement Plan / 完整 FC1 数据搬运改进计划

The full FC1 single-N-tile SoC checkpoint shows that the current bottleneck is
not the `8x8x8` MAC array. The bottleneck is movement from SRAM through the
core data mover into the compute-cluster preload windows.

完整 FC1 single-N-tile 的 SoC checkpoint 表明，当前瓶颈不是 `8x8x8` MAC array，
而是从 SRAM 经 wrapper 到 core preload window 的数据搬运。

The implementation roadmap is:

实施路线如下：

1. Keep the physical MAC tile unchanged.
   The core remains `M=8, N=8, K=8`, and K-streaming continues to accumulate
   many physical chunks in the core accumulator file.
2. Replace the debug-style wrapper-to-core preload path with a real movement
   path.
   The wrapper should configure movement; the movement path should fill core
   local storage without acting like a CPU writing one host-window word at a
   time.
3. Parameterize and then widen movement bandwidth.
   Add explicit `WORDS_PER_CYCLE` and `SETUP_CYCLES` knobs first, then back
   those knobs with a wider core preload interface and a wider SRAM/DMEM read
   model.
4. Add double buffering and overlap.
   While the core computes K chunk `i`, the movement path should prefetch K
   chunk `i+1` into the inactive buffer.

1. 保持物理 MAC tile 不变。
   core 仍然是 `M=8, N=8, K=8`，K-streaming 继续在 core accumulator file 中
   累加多个物理 chunks。
2. 把 debug 风格的 wrapper-to-core preload 路径替换成真正的数据搬运路径。
   wrapper 应该只配置搬运；搬运路径应直接填充 core local storage，而不是像 CPU
   一样逐 word 写 host window。
3. 先参数化再加宽搬运带宽。
   先加入显式 `WORDS_PER_CYCLE` 和 `SETUP_CYCLES` 参数，再用更宽的 core preload
   interface 和更宽的 SRAM/DMEM read model 支撑这些参数。
4. 增加双缓冲和重叠。
   core 计算 K chunk `i` 时，搬运路径预取 K chunk `i+1` 到另一个 buffer。

Step 1 status:

第 1 步状态：

- `npu_v0_data_mover` exposes `WORDS_PER_CYCLE` and `SETUP_CYCLES` parameters;
- default values preserve the current verified behavior:
  `WORDS_PER_CYCLE=1`, `SETUP_CYCLES=0`;
- Step 1 originally kept `WORDS_PER_CYCLE=1` until the host interface was
  widened; this is now superseded by Step 3.

- `npu_v0_data_mover` 暴露 `WORDS_PER_CYCLE` 和 `SETUP_CYCLES` 参数；
- 默认值保持当前已验证行为：`WORDS_PER_CYCLE=1`、`SETUP_CYCLES=0`；
- 第 1 步最初保持 `WORDS_PER_CYCLE=1`，直到 host interface 加宽；该限制已被第
  3 步取代。

Step 2 widens the wrapper-to-core preload/readback interface shape without yet
changing SRAM layout or CPU staging.

第 2 步会先加宽 wrapper-to-core preload/readback 接口形态，但暂时不改变 SRAM 布局
或 CPU staging。

Contract:

合同：

```text
CORE_HOST_LANES = 4
host_we[3:0]
host_addr       // base word address
host_wdata[4*32-1:0]
host_rdata[4*32-1:0]
```

Lane `i` maps to host window word address `host_addr + i`.

lane `i` 映射到 host window word 地址 `host_addr + i`。

Step 2 widened the interface shape first. In that checkpoint, the wrapper still
drove only lane 0:

第 2 步先加宽接口形态。在该 checkpoint 中，wrapper 仍然只驱动 lane 0：

```text
host_we = 4'b0001 when a scalar preload/write is active
host_wdata[31:0] carries the existing word
host_rdata[31:0] is consumed by the existing scalar readback path
```

This kept functionality and timing unchanged while moving the core boundary
away from a scalar-only debug host port.

这保持了功能和时序不变，同时先把 core 边界从只能标量访问的 debug host port 迁出。

Step 3 connects that 4-lane shape to actual movement:

第 3 步已经把 4-lane 接口接到真实搬运路径：

```text
DATA_MOVER_WORDS_PER_CYCLE = 4
CORE_HOST_LANES            = 4
SRAM NPU port              = 4 lanes, 128-bit packed data
CPU SRAM port              = 4 lanes, PicoRV32 drives one 32-bit lane/request
```

For SRAM-to-core transfers, the mover reads up to four consecutive SRAM words
and writes them to four consecutive core host-window words in the same cycle.
For core-to-SRAM transfers, it reads up to four consecutive core output words
and writes them back to consecutive SRAM words. Partial tails are handled with
per-lane write masks.

SRAM 到 core 的搬运中，data mover 每拍最多读取 4 个连续 SRAM word，并写入 4 个连续
core host-window word。core 到 SRAM 的搬运中，data mover 每拍最多读取 4 个连续
core output word，并写回连续 SRAM word。尾部不足 4 word 的 segment 通过 per-lane
write mask 处理。

These values now come from `arch/configs/soc_v0.jsonc` through generated SoC
constants:

```text
SOC_NPU_CORE_HOST_LANES
SOC_NPU_SRAM_LANES
SOC_NPU_DATA_MOVER_WORDS_PER_CYCLE
SOC_NPU_DATA_MOVER_SETUP_CYCLES
```

Current RTL requires `SOC_NPU_CORE_HOST_LANES == SOC_NPU_SRAM_LANES`, and
`WORDS_PER_CYCLE` must be in `1..lanes`.

These are actual RTL parameters, not only report-model knobs. CPU firmware
staging remains scalar at the PicoRV32 transaction level. The SRAM CPU port has
been widened structurally, but the current CPU still drives one lane per
request, so CPU copy time before each NPU job is not accelerated by this step.

这些值现在来自 `arch/configs/soc_v0.jsonc`，并通过生成的 SoC 常量进入 RTL。当前
RTL 要求 `SOC_NPU_CORE_HOST_LANES == SOC_NPU_SRAM_LANES`，且 `WORDS_PER_CYCLE`
必须落在 `1..lanes`。这些是实际 RTL 参数，不只是 report-model knob。CPU firmware
在 PicoRV32 事务层面仍然是标量访问；SRAM CPU port 已经做成宽口形态，但当前 CPU
每次仍只驱动一个 lane，所以这一步不会加速每个 NPU job 之前的 CPU copy 时间。

## 9. Status Bits

`STATUS` currently returns:

```text
bit 0: done_latched
bit 1: busy
bit 2: !busy
```

Firmware currently polls `done_latched`. IRQ registers exist but are not wired
into a CPU interrupt flow yet.

## 10. K-Streaming Ping-Pong Control

For `SOC_NPU_JOB_OP_MATMUL_K_STREAM`, the NPU core command processor overlaps prefetch of the
next K chunk with core execution of the current K chunk.

The NPU core control host register at `0x500` is used as:

| Bit | Meaning |
| ---: | --- |
| 0 | `matmul_accumulate_enable` |
| 1 | `clear_accumulator` pulse |
| 2 | `host_write_bank` for A/B host-window writes |
| 3 | `compute_bank_select`, latched by the core at launch |

The first chunk is loaded into bank 0. After launching chunk `i`, the command processor
configures both `host_write_bank` and the next `compute_bank_select` to the
opposite bank, then uses the data mover to fetch chunk `i+1` while the core is
active. The command processor advances to the next chunk only after both conditions hold:

```text
core_done_seen && next_prefetch_done
```

The current K-streaming contract uses one resident accumulator bank for all K
chunks in the descriptor. The v1 `accumulator_file` module has two physical
banks, but the v0 path currently selects bank 0 only.

Current measured result for the full FC1 single-N-tile smoke:

```text
before ping-pong: 58784 total cycles
after ping-pong:  39218 total cycles
data_mover.words: 147536 unchanged
core.matmul:      11520 unchanged
```

## 11. Error Handling

Current wrapper error handling is minimal:

- unknown op types are not rejected early;
- invalid descriptor addresses are not trapped;
- transfer length overflow is not reported;
- no timeout exists for a stuck core;
- no status error code is exposed to firmware.

These should be added before larger programs or untrusted descriptors are used.

## 12. Next Work

Immediate next work after the perf CSR snapshot implementation:

1. Extend workload identity and external-memory accounting for Transformer
   comparison evidence.
2. Render overlapped prefetch and compute spans more explicitly in the perf
   report timeline.
3. Consider nonzero setup/stall modeling after the current `4 words/cycle`
   path is stable.
4. Add richer bank/stall counters if they become necessary for the next
   scheduling decision.
