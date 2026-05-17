# NPU Wrapper And Data Mover Design

[TOC]

This document describes `npu_v0_opsched` and the first A2 data mover structure.

## 1. Role

The wrapper is the boundary between the CPU-visible SoC and the internal NPU
core. Its current responsibilities are:

- expose memory-mapped control/status registers;
- receive a descriptor address from CPU firmware;
- read descriptor/program/tensors from SRAM;
- preload NPU core internal memories through the core host interface;
- launch the core;
- wait for core completion;
- read core output windows and write output back to SRAM;
- publish `done/busy/idle` status.

The wrapper is not yet a full scheduler, command queue, DMA, or interrupt
controller. It is the place where those features should be introduced
incrementally.

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

Legacy A/B/C/X/Y/program windows are retained for older direct-window smoke
tests. New firmware should use the descriptor path.

## 3. Descriptor Contract

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

The wrapper assumes word-aligned 32-bit addresses and currently truncates
transfer lengths through 8-bit counters in the movement path. This is acceptable
for Phase 0/A2 bring-up and must be widened before larger tiles.

## 4. Wrapper State Machine

Main state machine in `npu_v0_opsched.sv`:

```text
DESC_IDLE
  -> DESC_READ
  -> DESC_FETCH_PROGRAM
  -> DESC_FETCH_INPUT0
  -> DESC_FETCH_INPUT1      // matmul only
  -> DESC_START_CORE
  -> DESC_WAIT_CORE
  -> DESC_WRITE_OUTPUT
  -> DESC_DONE
  -> DESC_IDLE
```

State responsibilities:

| State | Responsibility |
| --- | --- |
| `DESC_IDLE` | wait for CPU start or service legacy direct-window access |
| `DESC_READ` | read descriptor words from SRAM |
| `DESC_FETCH_PROGRAM` | load encoded uops into core `instr_mem` window |
| `DESC_FETCH_INPUT0` | load A or X into core input window |
| `DESC_FETCH_INPUT1` | load B for matmul |
| `DESC_START_CORE` | issue one-cycle `start_pulse` |
| `DESC_WAIT_CORE` | wait for `npu_done` |
| `DESC_WRITE_OUTPUT` | read C/Y from core output window and store to SRAM |
| `DESC_DONE` | clear busy and latch done |

## 5. Core Host Window Mapping

The wrapper converts descriptor movement into NPU core host addresses:

| Core host window | Address range | Meaning |
| --- | ---: | --- |
| A | `0x000` - `0x03f` | matmul input A |
| B | `0x100` - `0x13f` | matmul input B |
| C | `0x200` - `0x23f` | matmul output C |
| X | `0x300` - `0x307` | softmax input X |
| Y | `0x380` - `0x387` | softmax output Y |
| program | `0x400` - `0x40f` | encoded uop `instr_mem` |

This host window is an internal preload/readback path. It is not the long-term
NPU memory architecture.

## 6. Data Mover A2.1

`npu_v0_data_mover.sv` is the first A2 structural split. It owns linear
transfers between SRAM and a core host window.

Interface summary:

| Signal | Meaning |
| --- | --- |
| `start` | begin a transfer segment |
| `direction_store` | `0`: SRAM -> core host, `1`: core host -> SRAM |
| `sram_base_addr` | absolute SRAM source/destination |
| `host_base_addr` | core host window base |
| `words` | number of 32-bit words |
| `busy` | transfer is in progress |
| `complete` | current segment is complete |
| `index` | current word index |

Current behavior:

- one word per cycle;
- no setup latency;
- no burst grouping;
- no stalls from `sram_ready`;
- no independent counters inside the module yet.

The wrapper drives the data mover during:

- `DESC_FETCH_PROGRAM`;
- `DESC_FETCH_INPUT0`;
- `DESC_FETCH_INPUT1`;
- `DESC_WRITE_OUTPUT`.

Descriptor read still lives directly in the wrapper because it also populates
wrapper job registers.

## 7. Timing Semantics

Current transfer timing is intentionally equivalent to the old wrapper loops:

```text
cycles ~= words
```

Therefore the A2.1 structural patch should not change functional output or
cycle baselines. The verified baseline is:

```text
matmul total cycles: 236
softmax total cycles: 53
```

Next A2 step will add `WORDS_PER_CYCLE` and `SETUP_CYCLES`. At that point,
transfer timing becomes:

```text
cycles ~= setup_cycles + ceil(words / words_per_cycle)
```

## 8. Status Bits

`STATUS` currently returns:

```text
bit 0: done_latched
bit 1: busy
bit 2: !busy
```

Firmware currently polls `done_latched`. IRQ registers exist but are not wired
into a CPU interrupt flow yet.

## 9. Error Handling

Current wrapper error handling is minimal:

- unknown op types are not rejected early;
- invalid descriptor addresses are not trapped;
- transfer length overflow is not reported;
- no timeout exists for a stuck core;
- no status error code is exposed to firmware.

These should be added before larger programs or untrusted descriptors are used.

## 10. Next Work

Immediate next work:

1. Add data mover parameters `WORDS_PER_CYCLE` and `SETUP_CYCLES`.
2. Preserve `1 word/cycle` default behavior.
3. Add explicit data mover counters to `PERF_JOB`.
4. Drive the report `Data mover` lane from real data mover state/counters.
5. Add burst-profile mode: `4 words/cycle + 1 setup cycle`.
6. Only then start scratchpad banking and overlap work.
