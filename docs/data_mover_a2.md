# A2 Data Mover And Scratchpad Plan

This document captures the next architecture step after A1 matmul-array bring-up.

## 1. Why A2 Exists

A1 reduced the core matmul compute phase from 512 cycles to about 10 cycles for
the current 8x8x8 tile. The measured matmul job is now dominated by movement:

```text
total job cycles:              236
core matmul cycles:             10
SRAM read cycles:              153
SRAM write cycles:              64
core host write cycles:        144
core host read cycles:          64
```

The current wrapper path is intentionally simple: it reads one word per cycle
from SoC SRAM and writes one word per cycle into NPU core host windows. This was
good for functional bring-up and cycle visibility, but it is not the final NPU
memory path.

## 2. Current Temporary Mechanism

Current data/program flow:

```text
CPU stages descriptor/program/tensors in SRAM
  -> wrapper reads SRAM descriptor
  -> wrapper reads program words from SRAM
  -> wrapper writes core instr_mem through host window
  -> wrapper reads tensor words from SRAM
  -> wrapper writes core A/B/X windows
  -> wrapper starts core
  -> core executes from preloaded instr_mem and internal buffers
  -> wrapper reads core C/Y output window
  -> wrapper writes SRAM output buffer
```

Important constraints:

- `instr_mem`, A/B/X windows, C/Y windows, and accumulator storage are inside
  the current NPU core.
- wrapper host-window access is a preload/readback interface, not a scalable DMA
  or scratchpad interface.
- program words currently must fit in fixed-size `instr_mem` before launch.
- tensor and output movement is serialized around compute, so overlap is almost
  absent.

## 3. A2 Target Shape

A2 should replace wrapper word-copy behavior with explicit movement resources:

```text
descriptor
  -> data mover command queue
  -> burst SRAM reads/writes
  -> banked scratchpad / instruction buffer
  -> compute engine consumes tiles
```

The first implementation does not need a full production DMA. It should model
the key architectural effects:

- multi-word burst movement;
- setup latency per transfer;
- separate input, output, and instruction streams;
- scratchpad bank conflicts;
- compute stalls when required data or uops are unavailable;
- timeline lanes for SRAM, data mover, scratchpad, instruction fetch, compute,
  and stalls.

## 4. Proposed A2.1 Interface

Descriptor fields to add or model:

| Field | Meaning |
| --- | --- |
| `program_addr` | SRAM base of encoded uop stream |
| `program_words` | number of uop words |
| `input0_addr` | SRAM base of first input tensor/tile |
| `input1_addr` | SRAM base of second input tensor/tile |
| `output_addr` | SRAM base of output tensor/tile |
| `shape_m/n/k` | tile shape for matmul-like ops |
| `stride_*` | later tensor layout support |
| `flags` | streaming, double-buffer, or profiling controls |

Data mover command model:

| Command | Direction |
| --- | --- |
| `LOAD_UOP` | SRAM -> instruction buffer |
| `LOAD_TILE_A` | SRAM -> scratchpad A banks |
| `LOAD_TILE_B` | SRAM -> scratchpad B banks |
| `STORE_TILE_C` | accumulator/scratchpad C -> SRAM |

## 5. Performance Model

The current perf report now includes a `Movement model` panel. It uses the
measured word counts and a simple 4-word-per-cycle burst assumption:

```text
conservative_burst_cycles =
  sum(ceil(segment_words / 4)) + active_segments * setup_cycles
```

This is not claiming the RTL already has a DMA. It is a target estimate that
lets us compare:

- current one-word wrapper movement;
- ideal burst movement;
- conservative burst movement with setup overhead.

If the estimate shows a large win, the next RTL change is justified. If it does
not, we should focus on banking, overlap, or operator scheduling instead.

## 6. A2.1 Exit Criteria

- perf report shows current movement and burst-model movement side by side;
- docs define which current paths are temporary;
- RTL has an explicit data mover module for linear SRAM/core-window transfers,
  initially preserving the current one-word-per-cycle behavior;
- next RTL patch can add burst timing or banking without changing the
  CPU-visible launch protocol unnecessarily;
- tests still pass for matmul and softmax functional behavior.

## 7. Later A2.x Work

- Add a cycle-visible data mover FSM.
- Add scratchpad bank selection and conflict counters.
- Add instruction buffer or prefetch path for variable-length programs.
- Add double buffering so input fetch for tile N+1 can overlap compute for tile
  N.
- Add report lanes for DMA active, DMA wait, scratchpad conflict, uop fetch
  stall, and compute active.

## 8. A2.1 RTL Status

First RTL step:

- added `hw/npu_wrapper/rtl/npu_v0_data_mover.sv`;
- connected wrapper program load, input load, and output store phases through
  the data mover;
- preserved current one-word-per-cycle behavior and CPU-visible descriptor
  protocol;
- kept matmul and softmax cycle baselines unchanged;
- added a report `Data mover` lane reconstructed from current movement phases.

This is a structural change, not the burst implementation yet. The next RTL
step can add configurable transfer bandwidth or setup latency inside the data
mover while keeping the wrapper-level state names stable.

## 9. Next Session Checklist

Continue from A2.1, not from A1. The current verified baseline is:

```text
make test        PASS
make perf-report PASS
matmul total cycles: 236
softmax total cycles: 53
```

Current important files:

- `hw/npu_wrapper/rtl/npu_v0_data_mover.sv`: new structural data mover;
- `hw/npu_wrapper/rtl/npu_v0_opsched.sv`: wrapper now routes program/input/output
  linear transfers through the data mover;
- `sw/tools/perf/report.py`: report has `Data mover` timeline lane and
  `Movement model` panel;
- `hw/soc/tb/soc_cpu_tb.sv`: perf counters are still testbench-side sampling.

Next implementation order:

1. Add configurable data mover timing parameters:
   `WORDS_PER_CYCLE` and `SETUP_CYCLES`.
2. Keep default behavior equivalent to today's `1 word/cycle` path until tests
   are stable.
3. Add a simulation/profile mode using the documented target model:
   `4 words/cycle + 1 setup cycle per segment`.
4. Extend `PERF_JOB` with explicit data mover counters:
   `dm_active_cycles`, `dm_setup_cycles`, `dm_transfer_cycles`,
   `dm_words`, and later `dm_stall_cycles`.
5. Change the report `Data mover` lane to use real data mover counters instead
   of reconstructing spans from wrapper phases.
6. Compare measured RTL cycles with the existing `Movement model` estimate:
   matmul conservative burst target is currently about 60 movement cycles
   versus 217 measured SRAM movement cycles.
7. Only after burst timing is visible, start scratchpad banking work:
   bank mapping, bank conflict counters, and compute input-stall counters.

Acceptance criteria for the next patch:

- functional matmul and softmax still pass;
- `make test` and `make perf-report` pass;
- report shows whether A2 burst-mode reduces movement cycles or only changes
  accounting;
- docs state clearly whether the result is still a model or actual RTL timing.

Do not start with double buffering or variable-length program streaming yet.
Those should wait until the single data mover path has real counters and stable
reporting.
