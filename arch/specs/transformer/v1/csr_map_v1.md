# NPU CSR Map v1

## Target

CSR v1 extends the current wrapper-visible control/perf contract for
Transformer-oriented profiling. Addresses are byte offsets from the NPU wrapper
MMIO base.

`SCRATCH_BASE` is the base address of the NPU workspace in SoC SRAM or system
memory. It is not the internal scratchpad storage itself. `DM` means data
mover.

## Registers

| Offset | Name | Purpose |
| ---: | --- | --- |
| `0x000` | `CTRL` | start/reset/interrupt control bits |
| `0x004` | `STATUS` | busy/done/error state |
| `0x008` | `DESC_BASE_LO` | descriptor base address bits 31:0 |
| `0x00C` | `DESC_BASE_HI` | descriptor base address bits 63:32 |
| `0x010` | `PROGRAM_BASE_LO` | program base address bits 31:0 |
| `0x014` | `PROGRAM_BASE_HI` | program base address bits 63:32 |
| `0x018` | `SCRATCH_BASE_LO` | system workspace base bits 31:0 |
| `0x01C` | `SCRATCH_BASE_HI` | system workspace base bits 63:32 |
| `0x020` | `KV_BASE_LO` | KV cache base bits 31:0 |
| `0x024` | `KV_BASE_HI` | KV cache base bits 63:32 |
| `0x028` | `ERROR_CODE` | descriptor/uop/runtime error code |

## Performance Registers

| Offset | Name | Purpose |
| ---: | --- | --- |
| `0x100` | `PERF_TOTAL_CYCLES` | job total cycles |
| `0x104` | `PERF_CORE_ACTIVE` | core busy/progress interval |
| `0x108` | `PERF_MATRIX_ACTIVE` | matrix active cycles |
| `0x10C` | `PERF_VECTOR_ACTIVE` | vector active cycles |
| `0x110` | `PERF_REDUCTION_ACTIVE` | reduction active cycles |
| `0x114` | `PERF_SFU_ACTIVE` | SFU active cycles |
| `0x118` | `PERF_DM_ACTIVE` | data mover active cycles |
| `0x11C` | `PERF_DM_STALL` | data mover stall cycles |
| `0x120` | `PERF_SRAM_READ_WORDS` | SRAM/system read words |
| `0x124` | `PERF_SRAM_WRITE_WORDS` | SRAM/system write words |
| `0x128` | `PERF_MAC_OPS_LO` | effective MAC ops bits 31:0 |
| `0x12C` | `PERF_MAC_OPS_HI` | effective MAC ops bits 63:32 |
| `0x130` | `PERF_KV_READ_BYTES` | KV/cache read bytes |
| `0x134` | `PERF_KV_WRITE_BYTES` | KV/cache write bytes |

`PERF_MAC_OPS` is effective useful work. `PERF_MATRIX_ACTIVE` is time. The two
together define utilization and must not be substituted for each other.
