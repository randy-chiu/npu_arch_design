# SoC Architecture Design

[TOC]

This document is the detailed design note for the current CPU-controlled SoC.
For the shorter architecture overview, see `docs/architecture.md`.

## 1. Scope

The current SoC exists to make the NPU path executable end to end:

```text
PicoRV32 firmware
  -> memory-mapped NPU wrapper registers
  -> descriptor and tensors in SRAM
  -> NPU wrapper/data mover
  -> NPU core
  -> SRAM output
  -> firmware result check
```

The SoC is intentionally small. It is a verification platform and architecture
bring-up vehicle, not yet a production interconnect, cache, or DMA subsystem.

## 2. Top-Level Modules

Current CPU-controlled top:

| Module | File | Responsibility |
| --- | --- | --- |
| `soc_cpu_top` | `hw/soc/rtl/soc_cpu_top.sv` | CPU SoC top, instantiates CPU, bus, memories, NPU wrapper, test status |
| `picorv32_native_cpu` | `hw/soc/cpu/rtl/picorv32_native_cpu.sv` | Adapts PicoRV32 native memory interface to local bus signals |
| `simple_bus` | `hw/soc/rtl/bus/simple_bus.sv` | Address decode and single-master routing |
| `boot_rom` | `hw/soc/rtl/mem/boot_rom.sv` | Read-only firmware image for simulation |
| `simple_sram` | `hw/soc/rtl/mem/simple_sram.sv` | CPU data memory plus independent NPU port |
| `soc_dma` | `hw/soc/rtl/dma/soc_dma.sv` | CPU-configured ROM-to-SRAM staging DMA |
| `npu_v0_opsched` | `hw/npu_wrapper/rtl/npu_v0_opsched.sv` | CPU-visible NPU wrapper |
| `test_status` | `hw/soc/rtl/debug/test_status.sv` | Simulation pass/fail register |

## 3. Memory Map

The source of truth is `arch/configs/soc_v0.jsonc`.

Generated outputs:

```text
build/soc/soc_v0_addr.svh
build/soc/soc_v0_addr.h
build/soc/soc_v0.ld
```

Current regions:

| Region | Address range | Accessor | Use |
| --- | --- | --- | --- |
| Boot ROM | `0x0000_0000` - `0x0000_7fff` | CPU | Firmware image |
| SRAM | `0x0002_0000` - `0x0003_ffff` | CPU + NPU port | stack, locals, descriptor, tensor buffers, program buffers |
| NPU wrapper | `0x1000_0000` - `0x1000_0fff` | CPU | NPU control/status/register windows |
| UART | `0x2000_0000` - `0x2000_0fff` | reserved | Not implemented |
| Test status | `0x3000_0000` - `0x3000_000f` | CPU | Simulation pass/fail |
| DMA | `0x4000_0000` - `0x4000_0fff` | CPU | ROM-to-SRAM staging DMA |

Important detail: `simple_bus` subtracts region bases for ROM/SRAM local
addresses. The NPU wrapper receives only `m_addr[11:0]` as its local register
offset. The NPU SRAM port in `soc_cpu_top` converts the wrapper's absolute SRAM
address back to SRAM-local addressing:

```text
npu_addr = npu_sram_addr - SOC_SRAM_BASE
```

## 4. Bus Semantics

`simple_bus` is a single-master, single-cycle-style local bus.

Signals:

| Signal | Meaning |
| --- | --- |
| `m_req` | CPU request valid |
| `m_we` | write enable |
| `m_addr` | byte address |
| `m_wdata` | write data |
| `m_rdata` | read data |
| `m_ready` | target accepted/completed request |

Current targets return `ready` combinationally from `req` or through simple
target logic. There is no bus arbitration because PicoRV32 is the only bus
master. The NPU wrapper uses a separate SRAM port, so wrapper movement does not
arbitrate with CPU on `simple_bus` in this model.

The SoC spec now separates the RV32 CPU transaction width from the physical
SRAM CPU port shape:

```text
SOC_BUS_DATA_WIDTH_BITS      = 32
SOC_SRAM_CPU_LANES           = 4
SOC_SRAM_CPU_DATA_WIDTH_BITS = 128
```

PicoRV32 still issues one 32-bit load/store per request. `simple_bus` maps that
request onto one lane of the wider SRAM CPU port:

```text
sram_lane = (sram_local_addr >> 2) % SOC_SRAM_CPU_LANES
sram_addr = aligned group base address
sram_we[sram_lane] = CPU write enable
sram_wdata[32*sram_lane +: 32] = CPU write data
CPU read data = sram_rdata[32*sram_lane +: 32]
```

This is a structural staging step. It makes the SRAM CPU side compatible with a
future preload/copy engine that can drive multiple lanes per cycle, but it does
not make scalar PicoRV32 stores faster by itself.

SoC spec 现在把 RV32 CPU 事务宽度和 SRAM CPU 物理端口形态分开：

```text
SOC_BUS_DATA_WIDTH_BITS      = 32
SOC_SRAM_CPU_LANES           = 4
SOC_SRAM_CPU_DATA_WIDTH_BITS = 128
```

PicoRV32 仍然每次只发起一个 32-bit load/store。`simple_bus` 会根据地址把这个
请求映射到 SRAM CPU 宽口的某一个 lane。这个改动是结构准备：它让 SRAM CPU 侧
可以被未来的 preload/copy engine 多 lane 驱动，但不会单独加速 PicoRV32 标量
store。

Current limitations:

- no burst transactions;
- no wait-state memory model beyond target `ready`;
- no bus errors;
- no CPU/NPU arbitration for the same SRAM port;
- no cache or instruction/data bus split.

## 5. Memory Modules

### Boot ROM

`boot_rom` is initialized from:

```text
build/firmware/soc_cpu_smoke.hex
```

This image currently contains the whole smoke firmware. In a more realistic
system, boot ROM would contain a smaller loader that fetches code/data from
external storage into SRAM/DRAM.

The current boot ROM also exposes a second packed DMA read port. This is an
idealized simulation shortcut so the CPU can continue fetching instructions
while the DMA stages generated arrays into SRAM.

当前 boot ROM 还暴露第二个 packed DMA read port。这是仿真简化，用来让 CPU 继续
取指，同时 DMA 将生成数组搬到 SRAM。

### SRAM

`simple_sram` has two ports:

| Port | User | Current purpose |
| --- | --- | --- |
| CPU port | `simple_bus` | stack, globals, descriptor construction, tensor/program staging; physically `SOC_SRAM_CPU_LANES` lanes but PicoRV32 currently uses one lane per request |
| NPU port | `npu_v0_opsched` | descriptor/program/input reads and output writes; `SOC_NPU_SRAM_LANES` lanes |

Both ports are still single-cycle ready in the current model. There is no bank
conflict model yet. A2 work should gradually replace this with a bandwidth and
bank-aware memory model.

## 6. NPU Placement In SoC

The NPU is attached as a CPU-visible peripheral:

```text
CPU
  -> simple_bus
  -> NPU wrapper registers
  -> wrapper reads/writes SRAM through NPU SRAM port
  -> wrapper drives NPU core host interface
```

The CPU does not write tensors directly into the NPU core in the main firmware
path. It stages data in SRAM, builds a descriptor, and starts the wrapper.

Large generated arrays are now staged with `soc_dma` instead of scalar
PicoRV32 copy loops. Firmware configures source ROM address, destination SRAM
address, word count, and start; the DMA writes up to `SOC_SRAM_CPU_LANES` words
per cycle into the SRAM CPU/preload port.

大规模生成数组现在通过 `soc_dma` staging，而不是 PicoRV32 逐 word copy loop。
Firmware 配置 ROM 源地址、SRAM 目标地址、word 数和 start；DMA 每拍最多向 SRAM
CPU/preload port 写入 `SOC_SRAM_CPU_LANES` 个 word。

Legacy debug windows still exist under the NPU wrapper register space for older
`soc-sim` tests. The firmware-controlled path should use descriptor/SRAM launch.

## 7. Simulation Top

`soc_cpu_tb` is the CPU-controlled SoC testbench. It:

- provides clock/reset;
- waits for `test_status`;
- fails on CPU trap or timeout;
- collects performance counters by observing existing RTL hierarchy;
- prints `PERF_JOB` JSON lines for `make perf-report`.

The testbench should remain observation-oriented. Functional behavior should be
driven by firmware, not by direct testbench pokes into the NPU.

## 8. Source-Of-Truth Rules

- SoC memory map and descriptor ABI: `arch/configs/soc_v0.jsonc`.
- NPU wrapper register map: `arch/configs/npu_wrapper_v0.jsonc`.
- Generated headers are build artifacts and should not be manually edited.
- Firmware should include generated headers rather than duplicate constants.

## 9. Known Limitations

- ROM is simulation firmware storage, not a real boot flow.
- SRAM has no realistic latency, arbitration, or banking.
- NPU SRAM port bypasses the CPU bus and does not model contention.
- DMA has an ideal second boot-ROM read port and only a simple SRAM-port mux.
- PicoRV32 itself remains a 32-bit scalar bus master; the wider SRAM CPU port is
  only useful for scalar lane mapping today and for a future preload/copy
  engine.
- UART is reserved but not implemented.
- Interrupts are provisioned in wrapper registers but firmware still polls.
- Performance counters are testbench-side, not CPU-visible CSRs.

## 10. Next Design Work

Near-term SoC work should support A2 without overbuilding:

- add explicit data mover counters and optional debug registers;
- add DMA counters and error bits;
- model SRAM bandwidth and later bank conflicts;
- keep descriptor ABI stable while the internal data mover changes;
- decide when IRQ replaces polling in the firmware loop.
