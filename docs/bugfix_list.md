# Bugfix List

[TOC]

本文档记录 bring-up 过程中遇到的代表性 bug。每条记录都应该包含现象、定位
过程、根因、修复方式、验证结果和后续规则，避免后续重启 Codex 或其他 agent
时重复踩同一个问题。

## Bug 1: SRAM Base 未对齐导致 GCC 编译的 Firmware 读回失败

### 背景

这个问题出现在项目第一次从临时 Python RV32I firmware emitter 切换到
xPack `riscv-none-elf-gcc` 编译的真实 C/ASM firmware 时。

当时 CPU 控制的 SoC 测试流程是：

```text
make firmware-smoke-c
make cpu-soc-sim
```

firmware 执行路径是：

```text
PicoRV32
  -> boot_rom at 0x0000_0000
  -> C firmware in sw/soc_cpu
  -> NPU wrapper MMIO at 0x1000_0000
  -> NPU core
  -> read output window
  -> compare expected data
  -> write test_status
```

### 现象

`make firmware-smoke-c` 可以成功构建，但 `make cpu-soc-sim` 失败：

```text
FAIL firmware reported mismatch
```

临时给 firmware 增加失败码后，第一次失败状态是：

```text
0x80000100
```

这表示 matmul 第一个输出元素比较失败。

### 定位过程

当时按下面几个阶段逐步缩小范围：

1. 确认 GCC 输出使用的是当前 CPU 支持的指令：
   `-march=rv32i -mabi=ilp32`.
2. 检查 `soc_cpu_smoke.dump`，确认没有生成当前 PicoRV32 配置不支持的 `M`
   扩展、`C` 扩展或压缩指令。
3. 在 `soc_cpu_tb` 中临时探测 NPU wrapper 写操作：
   - Matrix A 写入已经到达 wrapper。
   - program 写入已经到达 wrapper。
   - `CTRL.start` 写入已经到达 wrapper。
4. 临时探测 NPU core 内部状态：
   - `dram_a[0] = 1`
   - `dram_b[0] = 1`
   - `instr_mem[0] = 0x10000000`
   - `acc_buf[0] = -4`
   - `dram_c[0] = -4`
5. 临时探测 wrapper 读回路径：
   - 从 wrapper C window 读回 `0xfffffffc`，这是正确结果。
6. 但 C firmware 里参与比较的实际值仍然是 `0`，说明数据从 wrapper 读回后，
   在写入/读出栈上的局部变量时丢失了。

### 根因

旧的 SoC SRAM map 是：

```text
SRAM base = 0x0001_0000
SRAM size = 0x0002_0000
stack     = 0x0002_fff0
```

但当前 `simple_bus` 的地址译码方式是 power-of-two mask：

```systemverilog
assign sel_sram = ((m_addr & SOC_SRAM_MASK) == SOC_SRAM_BASE);
```

对于 128 KiB 的 region，base 必须按照 128 KiB 对齐。`0x0001_0000` 只按
64 KiB 对齐，不满足当前 mask decoder 的要求。

所以栈地址附近的访问，例如 `0x0002_fff0` 一带，没有命中 SRAM，而是落到了
bus default 路径，读回默认值 `0`。

临时 Python RV32I emitter 之前没有暴露这个问题，因为它基本没有使用正常 C
栈。真实 C firmware 一运行，就会使用局部变量、保存寄存器和函数调用栈，因此
立刻触发了这个问题。

### 修复

把 SRAM region 移到 128 KiB 对齐的地址范围：

```text
SRAM base = 0x0002_0000
SRAM size = 0x0002_0000
stack     = 0x0003_fff0
```

同步更新文件：

```text
arch/configs/soc_v0.jsonc
docs/architecture.md
docs/design/soc_architecture.md
```

现在 linker script 已经改为由 `make soc-spec` 从 `arch/configs/soc_v0.jsonc`
生成：

```text
build/soc/soc_v0.ld
```

同时把 `simple_sram.sv` 改成与当前 simple local bus 协议匹配的行为：

```text
read:  同周期组合读
write: 同步写
ready: 同周期 req
```

这样 SRAM 的行为与当前仿真里 PicoRV32 使用的简单 bus ready 协议保持一致。

### 验证

修复后验证结果：

```text
make firmware-smoke-c: PASS
make cpu-soc-sim: PASS
make soc-sim: PASS
make test: PASS, 7 tests
```

### 后续规则

当 bus region 使用 `(addr & mask) == base` 这种方式译码时，region base
必须按 region size 对齐。

SoC spec validation 现在已经在 `sw/tools/soc/emit_soc_spec.py` 中检查：

```text
base % size_bytes == 0
size_bytes is a power of two
```

这样非法 memory map 会在 `make soc-spec` 生成阶段失败，而不是等到 firmware
simulation 时才暴露。

## Bug 2: 生成的 SoC C Header 被汇编文件 Include 后编译失败

### 背景

在把 `npu_job_desc` descriptor ABI 收敛到 `arch/configs/soc_v0.jsonc` 后，
`make soc-spec` 会生成：

```text
build/soc/soc_v0_addr.h
build/soc/soc_v0_addr.svh
```

C firmware 需要从 `soc_v0_addr.h` 获取 `soc_npu_job_desc_t`、descriptor 字段
offset 和 `SOC_NPU_JOB_OP_*`。同时 `sw/soc_cpu/boot/start.S` 也 include 这个
header，用里面的 reset/stack 地址常量。

### 现象

第一次运行：

```text
make cpu-soc-sim
```

GCC 在编译 `start.S` 时失败，报错类似：

```text
Error: unrecognized opcode `typedef signed char int8_t'
Error: unrecognized opcode `typedef struct{'
```

### 根因

`.S` 文件会经过 C preprocessor，所以它可以 include `#define` 常量。但
preprocess 后的结果仍然要交给 assembler。生成的 `soc_v0_addr.h` 新增了：

```c
#include <stdint.h>
typedef struct {
    uint32_t op_type;
    ...
} soc_npu_job_desc_t;
```

这些 C typedef 对 assembler 来说不是合法汇编语法，因此被当成非法 opcode。

### 修复

生成 header 时把 C-only 内容包在 `__ASSEMBLER__` guard 里：

```c
#ifndef __ASSEMBLER__
#include <stdint.h>
#endif

...

#ifndef __ASSEMBLER__
typedef struct {
    ...
} soc_npu_job_desc_t;
#endif
```

这样：

- `start.S` 仍然能看到 SoC 地址和 stack/reset 常量；
- C firmware 能看到 `soc_npu_job_desc_t`；
- descriptor ABI 仍然只从 SoC spec 生成一份。

### 验证

修复后验证结果：

```text
make cpu-soc-sim: PASS
make soc-sim: PASS
make test: PASS, 7 tests
```

### 后续规则

任何可能被 `.S` 文件 include 的 generated header，只能无条件暴露 assembler
可接受的 `#define`。C typedef、inline function、`stdint.h` include 等内容都
必须放在 `#ifndef __ASSEMBLER__` 保护下。
