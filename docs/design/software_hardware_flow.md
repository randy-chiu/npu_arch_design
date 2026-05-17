# Software And Hardware Interaction Design

[TOC]

This document describes how generated software artifacts, CPU firmware, wrapper
RTL, and NPU core RTL cooperate in the current system.

## 1. End-To-End Flow

Current smoke flow:

```text
graph/operator intent
  -> compiler emits NPU micro-ops
  -> assembler encodes 32-bit program words
  -> fixture/firmware data generator emits C header and RTL hex files
  -> firmware copies tensors/programs into SRAM buffers
  -> firmware fills soc_npu_job_desc_t
  -> firmware writes DESC_ADDR and CTRL.start
  -> wrapper reads descriptor/data/program from SRAM
  -> wrapper preloads core windows and starts core
  -> core executes preloaded uops
  -> wrapper writes output back to SRAM
  -> firmware checks output and writes test_status
```

## 2. Source Ownership

| Area | Directory | Owns |
| --- | --- | --- |
| NPU operator intent | `sw/npu_core/operators` | operator-to-uop templates/intents |
| Compiler | `sw/tools/npu_compiler` | graph/operator lowering to uops |
| Assembler | `sw/tools/npu_assembler` | uop encoding to 32-bit words |
| Compatibility tools | `sw/tools/npu_phase0` | historical CLI, simulator, fixture glue |
| CPU firmware | `sw/soc_cpu` | boot, runtime, driver, smoke app |
| Generated firmware data | `build/firmware` | C arrays for tensors/programs/expected outputs |
| RTL fixture data | `build/rtl_fixture` | hex files and generated SV include params |

## 3. Build Artifacts

Important generated artifacts:

| Command | Output |
| --- | --- |
| `make soc-spec` | SoC headers/linker script from `arch/configs/soc_v0.jsonc` |
| `make npu-wrapper-spec` | NPU wrapper headers from `arch/configs/npu_wrapper_v0.jsonc` |
| `make rtl-fixtures` | NPU core test hex files and SV spec include |
| `make firmware-data` | `soc_cpu_smoke_data.h` |
| `make firmware-smoke` | boot ROM hex for CPU simulation |

Generated files should not be manually edited.

## 4. Firmware Runtime

Key files:

| File | Purpose |
| --- | --- |
| `sw/soc_cpu/boot/start.S` | reset entry, stack setup, call `main` |
| `sw/soc_cpu/runtime/npu_driver.c` | MMIO helpers, start/status polling, test status writes |
| `sw/soc_cpu/apps/soc_cpu_smoke/main.c` | current matmul then softmax smoke app |

Driver functions:

| Function | Effect |
| --- | --- |
| `npu_set_desc_addr(addr)` | write descriptor SRAM address to wrapper |
| `npu_start()` | write `CTRL.start` |
| `npu_status()` | read wrapper status |
| `npu_wait_done()` | poll `STATUS.done` |
| `test_status_pass/fail` | write simulation status register |

The driver still includes legacy `npu_write_words` and `npu_read_words` helpers
for wrapper windows. New descriptor-based firmware should not use them for main
data movement.

## 5. Firmware Job Sequence

For each job, firmware performs:

1. Copy input tensors into SRAM-resident arrays.
2. Copy encoded NPU program into an SRAM-resident array.
3. Fill `soc_npu_job_desc_t`.
4. Write `DESC_ADDR`.
5. Write `CTRL.start`.
6. Poll `STATUS.done`.
7. Check output buffer in SRAM.

Matmul descriptor fields:

```text
op_type      = SOC_NPU_JOB_OP_MATMUL
program_addr = matmul_program_sram
input0_addr  = matmul_a_sram
input1_addr  = matmul_b_sram
output_addr  = matmul_c_sram
```

Softmax descriptor fields:

```text
op_type      = SOC_NPU_JOB_OP_SOFTMAX
program_addr = softmax_program_sram
input0_addr  = softmax_x_sram
input1_addr  = 0
output_addr  = softmax_y_sram
```

## 6. ABI Details

The descriptor struct is generated from `arch/configs/soc_v0.jsonc`.

Current ABI rules:

- all fields are 32-bit words;
- addresses are absolute SoC addresses as seen by CPU firmware;
- wrapper subtracts/uses SRAM base through SoC wiring;
- program/input/output lengths are word counts;
- current wrapper movement path effectively supports small 8-bit transfer
  counters and should be widened before larger workloads.

Any descriptor field change is a hardware/software contract change. It must
update:

- `arch/configs/soc_v0.jsonc`;
- generated C/SV headers;
- firmware setup code;
- wrapper descriptor read logic;
- docs and tests.

## 7. Program Format

The current NPU core consumes 32-bit encoded uops. The source-level lowering is
owned by compiler/assembler tools, not by CPU firmware.

Firmware treats program words as opaque data:

```text
firmware copies program words into SRAM
descriptor points wrapper to those words
wrapper preloads core instr_mem
core fetches from instr_mem after start
```

Current limitation: the whole program must fit into 16-word `instr_mem`. Future
variable-length programs need instruction streaming or an instruction buffer.

## 8. Pass/Fail Contract

The firmware writes `test_status`:

| Value | Meaning |
| ---: | --- |
| `0x0000_0001` | pass |
| `0xffff_ffff` | generic fail |
| `0x8000_0000 | code` | fail with encoded mismatch/status code |

`soc_cpu_tb` watches this register and ends simulation.

## 9. Current Limitations

- firmware uses polling, not interrupts;
- no timeout in firmware `npu_wait_done`;
- no cache coherency concerns are modeled;
- no dynamic allocator or runtime job queue;
- program/tensor staging is hardcoded for smoke tests;
- CPU-side staging/check cycles are not yet included in the perf timeline.

## 10. Next Work

Near-term software/hardware interaction work:

- add driver timeout around `npu_wait_done`;
- expose wrapper error/status code once RTL supports it;
- decide whether perf counters become CPU-readable debug CSRs;
- keep descriptor ABI stable while A2 data mover internals change;
- later add IRQ-driven completion path.
