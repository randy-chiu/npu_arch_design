# NPU Core Design

[TOC]

This document describes the current `npu_v0_top` compute core and the A1 matmul
array. It focuses on what the RTL implements today, not the final target NPU.

## 1. Role

The NPU core consumes a preloaded micro-op program and preloaded tensor data,
executes the supported operators, and exposes results through host-readable
output windows.

Current core properties:

- no direct SRAM/DRAM master interface;
- no autonomous program fetch from SoC memory;
- no queue or multi-job context;
- no real vector/SFU pipeline yet;
- matmul compute path has A1 output-parallel array behavior.

The wrapper owns all SoC memory movement today.

## 2. External Interface

`npu_v0_top` ports:

| Signal | Direction | Meaning |
| --- | --- | --- |
| `clk/rst_n` | input | clock/reset |
| `start` | input | one-cycle start pulse from wrapper |
| `op` | input | reserved; execution is currently driven by uops |
| `done` | output | asserted when program reaches done state |
| `host_we[CORE_HOST_LANES-1:0]` | input | lane write enables for host preload windows |
| `host_addr` | input | 12-bit base host window word address |
| `host_wdata[CORE_HOST_LANES*32-1:0]` | input | lane-packed host write data |
| `host_rdata[CORE_HOST_LANES*32-1:0]` | output | lane-packed host read data from output windows |

For the current checkpoint, `CORE_HOST_LANES=4`, but the wrapper still drives
only lane 0. Lane `i` maps to `host_addr + i`. This prepares the core boundary
for a wider data mover while preserving the current scalar preload behavior.

当前 checkpoint 中，`CORE_HOST_LANES=4`，但 wrapper 仍然只驱动 lane 0。lane `i`
映射到 `host_addr + i`。这为更宽的数据搬运边界做准备，同时保持当前标量 preload
行为。

Host writes are accepted only while the core is idle:

```text
host_we[lane] && state == ST_IDLE
```

This prevents the wrapper from changing inputs/program while the core is
executing.

## 3. Internal Memories

| Storage | Width | Entries | Purpose |
| --- | ---: | ---: | --- |
| `dram_a` | signed 8 | 64 | matmul A preload window |
| `dram_b` | signed 8 | 64 | matmul B preload window |
| `dram_c` | signed 32 | 64 | matmul C output window |
| `dram_x` | signed 8 | 8 | softmax X preload window |
| `dram_y` | unsigned 8 | 8 | softmax Y output window |
| `instr_mem` | 32 | 16 | encoded uop program |
| `spad_a` | signed 8 | 64 | matmul A scratchpad |
| `spad_b` | signed 8 | 64 | matmul B scratchpad |
| `acc_buf` | signed 32 | 64 | matmul accumulator/output staging |
| `vec_buf` | signed 16 | 8 | softmax vector staging |

Names like `dram_a` are historical. These are internal core arrays in the
current RTL, not external DRAM.

For K-streaming matmul, `acc_buf` is the resident partial-sum buffer. The A/B
preload and scratchpad arrays remain one `8x8` tile each; they are overwritten
for every K chunk while `acc_buf` persists until the final store.

对于 K-streaming matmul，`acc_buf` 是常驻 partial-sum buffer。A/B preload 和
scratchpad 数组仍然各自只保存一个 `8x8` tile；每个 K chunk 都会覆盖它们，而
`acc_buf` 会一直保持到最终 store。

## 4. Host Window Map

| Address range | Access | Storage |
| ---: | --- | --- |
| `0x000` - `0x03f` | write | `dram_a` |
| `0x100` - `0x13f` | write | `dram_b` |
| `0x200` - `0x23f` | read | `dram_c` |
| `0x300` - `0x307` | write | `dram_x` |
| `0x380` - `0x387` | read | `dram_y` |
| `0x400` - `0x40f` | write | `instr_mem` |
| `0x500` | write | matmul accumulate control: bit 0 enable, bit 1 clear pulse |

The core does not validate window overflows beyond these simple address ranges.

## 5. Program Execution

State machine:

```text
ST_IDLE
  -> ST_FETCH on start
  -> ST_MATMUL when MATMUL uop starts
  -> ST_FETCH when matmul array completes
  -> ST_DONE on HALT or unknown opcode
  -> ST_IDLE after wrapper drops start
```

In `ST_FETCH`, the core reads `instr_mem[pc]`, increments `pc`, and executes or
dispatches the uop.

Supported uops:

| Uop | Current behavior |
| --- | --- |
| `LOAD` | copy preloaded tensor window into scratch/vector buffer |
| `MATMUL` | start A1 matmul array |
| `STORE` | copy accumulator/vector buffer into output window |
| `VREDMAX` | reduce max over `vec_buf` |
| `VSUB` | subtract scalar max |
| `VEXP` | approximate exp through small LUT |
| `VREDSUM` | reduce sum over low 8-bit vector values |
| `VDIV` | normalize to Q0.8-like output |
| `HALT` | finish program |

Most non-matmul uops are implemented as single-cycle RTL tasks. This is useful
for functional bring-up but is not a realistic vector pipeline timing model.

## 6. Matmul A1 Array

`matmul_array.sv` is parameterized by `M`, `N`, and `K`. Current config is
8x8x8.

Data shape:

```text
A: M x K, signed int8
B: K x N, signed int8
C: M x N, signed int32
```

Behavior:

- on `start`, clear 64 result accumulators;
- for each active `k_idx`, update all `M*N = 64` output accumulators in
  parallel;
- each active cycle performs 64 signed int8-by-int8 MACs into int32 results;
- after `K` slices, assert `done`;
- `npu_v0_top` commits `result_flat` into `acc_buf`.

The nested `for i/j` loops inside the clocked block describe many same-cycle
register updates, not software-style serial loop execution. Only `k_idx`
advances across cycles. This is why the measured matmul compute phase moved
from the old 512-cycle scalar baseline to about 10 cycles.

For a cycle-by-cycle diagram of the current 64-MAC/cycle behavior, see
`docs/design/fc1_k_streaming_matmul.md`, section
`2.1 Cycle-By-Cycle Example / 逐拍计算例子`.

K-streaming matmul does not change this physical parallelism. It repeats the
same `8x8x8` array operation for multiple K chunks and changes the commit
semantics from:

```text
acc_buf = tile_result
```

to:

```text
acc_buf += tile_result
```

when `matmul_accumulate_enable` is set through host address `0x500`.

K-streaming matmul 不改变物理并行度。它只是对多个 K chunk 重复执行同一个
`8x8x8` array operation，并在 `matmul_accumulate_enable` 置位时把提交语义从
覆盖改为累加。

Detailed A1 explanation is in `docs/matmul_array_a1.md`.

## 7. Softmax Path

Current softmax program:

```text
LOAD X -> VREDMAX -> VSUB -> VEXP -> VREDSUM -> VDIV -> STORE Y -> HALT
```

The RTL implements each vector operation as an immediate task over the whole
8-element vector. This means softmax timing is not yet representative of a real
multi-cycle vector/SFU pipeline.

A3 should replace this with:

- vector lane active cycles;
- reduction latency;
- exp approximation latency;
- reciprocal/div latency;
- dependency stalls between vector stages.

## 8. Current Timing Baseline

From `make perf-report` after enabling the 4-lane core host interface and
`WORDS_PER_CYCLE=4` NPU-side movement:

```text
matmul total cycles:       81
core total cycles:         18
core matmul cycles:        10
softmax total cycles:      30
softmax core cycles:       11
```

Matmul job time is now dominated by wrapper/data movement rather than core
compute.

## 9. Limitations

- `instr_mem` has only 16 entries.
- Program is preloaded before launch; no instruction streaming/prefetch.
- Core cannot directly access SoC SRAM.
- Vector/SFU operations are single-cycle tasks.
- No issue queue, hazard tracking, or pipeline backpressure.
- No scratchpad bank conflict model.
- Host writes are blocked during execution instead of using double buffering.

## 10. Next Work

Core changes should follow the movement work:

1. Keep A1 matmul array stable while A2 data mover counters are added.
2. Add scratchpad/bank visibility once data mover timing is real.
3. Add input-ready/core-stall counters before changing compute scheduling.
4. Move softmax to A3 multi-cycle vector/SFU pipeline after movement bottlenecks
   are visible.
