# NPU Core Design

[TOC]

This document describes the complete `npu_v0_core_system`, including its
command processor, data mover, local storage boundary, and compute cluster.

## 1. Role

The NPU core is the autonomous execution side of the NPU IP after a CPU command
crosses the host wrapper. It owns:

- descriptor fetch/decode and load/compute/store scheduling;
- data movement between external SRAM and core-local windows/storage;
- compute-cluster launch and completion handling;
- job-scoped performance counter collection;
- matrix, vector, reduction, and SFU execution.

```text
npu_v0_core_system
  -> command processor
  -> uop scheduler / dispatcher
  -> npu_v0_data_mover
  -> npu_v0_compute_cluster
      -> matrix / vector / reduction / SFU
```

Current core properties:

- the core system has an SRAM-side movement interface;
- no autonomous program fetch from SoC memory;
- no queue or multi-job context;
- no production valid/ready vector/SFU pipeline yet;
- matmul compute path has A1 output-parallel array behavior.
- Transformer attention stage bring-up uses explicit `op` modes for row
  softmax, mixed probability-value matmul, and unmasked score scale.

The data mover and command processor own all SoC memory movement today.

The command processor and uop scheduler are distinct control levels:

| Module | Responsibility |
| --- | --- |
| Command processor | descriptor fetch/decode, operator-level movement sequencing, K-chunk sequencing, compute-cluster launch, and job completion |
| Uop scheduler / dispatcher | common uop fetch/decode, local LOAD/STORE dispatch, execution-engine issue, and engine-completion wait |
| Compute cluster | execution engines and core-local operand/result storage |

This is a deliberate two-level scheduling contract for the current
architecture:

```text
descriptor + Command processor = external-memory movement and chunk scheduling
uop program + Uop scheduler     = core-local movement and execution scheduling
```

The current encoded `UOP_LOAD` contains only a tensor-window identifier and a
local-buffer identifier. It does not contain an external SRAM address, transfer
length, destination preload bank, or asynchronous completion token. Therefore
it cannot issue the current Data mover transaction. Those fields come from the
descriptor, so the Command processor can preload chunk 0 before launching the
uop scheduler and can prefetch chunk 1 while chunk 0 executes.

For MatMul, the current `UOP_LOAD A/B` should be read as an operand-bank bind
and dependency check. The opcode name is retained for Phase 0 binary
compatibility, but it does not copy a second full tile:

```text
selected preload/operand bank -> Matrix input binding
```

The former RTL copied each full A/B tile from a preload bank into duplicate
`spad_a/spad_b` arrays in one cycle. That copy had no modeled port width,
banking, area, or multi-cycle cost, while the selected operand bank was already
stable for the whole Matrix operation. It was therefore removed rather than
reported as a credible one-cycle local-memory transfer.

Softmax `UOP_LOAD X` remains a real local movement into `vec_buf`. A future
Matrix operand feed/register module is valid only when its storage, bandwidth,
latency, and active events are explicitly modeled.

A future ISA where a decoded load instruction directly issues the Data mover
would be a different contract. It would require address/length operands or a
descriptor-reference operand, an asynchronous load token, dependency tracking,
and explicit bank selection. Until that contract exists, reports must not
describe descriptor-directed external movement as execution of `UOP_LOAD`.

Execution engines do not fetch or decode the common uop stream. Matrix,
vector, reduction, and SFU engines retain only engine-local control such as
iteration counters, pipeline progress, accepted-operation state, and done
generation.

The word `program` in the descriptor means the encoded uop stream, not tensor
data and not CPU firmware. For the current matmul path the meaningful uops are:

```text
LOAD A bind selected operand bank A
LOAD B bind selected operand bank B
MATMUL selected operand bank A/B -> accumulator
STORE accumulator -> output window
HALT
```

The current descriptor points to a fixed 16-word program image containing
these uops plus HALT padding. With the four-word-per-cycle Data mover it costs
four cycles to load. The v0 command processor reloads this image for every
descriptor job; retaining or caching an unchanged program is a future
optimization, not current behavior.

Before the uop scheduler starts, the command processor must use the Data mover
to copy both the uop program and the first A/B chunk from external SRAM into
core-local storage. The initial A/B movement is descriptor-directed external
movement; it is not caused by decoding the uop `LOAD A/B` instructions.

For K-stream matmul, external movement and local uop execution are ordered as:

```text
descriptor read
-> Data mover: uop program SRAM -> instr_mem
-> Data mover: chunk 0 A/B SRAM -> preload bank 0
-> launch chunk 0 uop program
   -> uop LOAD A/B: preload bank 0 -> scratchpads
   -> Matrix: compute chunk 0
   -> uop STORE: accumulator remains resident
|| Data mover: chunk 1 A/B SRAM -> preload bank 1
-> wait if chunk 1 prefetch is incomplete
-> launch chunk 1 uop program using preload bank 1
```

After chunk 0 A/B movement, the Command processor performs two visible control
cycles before local execution begins:

```text
DESC_START_CORE       -> pulse compute-cluster start; chunk 0 latches bank 0
DESC_CONFIG_NEXT_BANK -> select bank 1 as the Data-mover prefetch target and
                         as the compute bank for the next chunk launch
```

`DESC_CONFIG_NEXT_BANK` does not change the bank already latched by the active
chunk. This ordering prevents the chunk 1 prefetch from overwriting chunk 0.

The alternate preload bank is what makes chunk `N+1` external movement legal
while chunk `N` uses its selected bank and scratchpads. The uop scheduler does
not fetch or decode a second program for every chunk; it reruns the same
preloaded uop program against the selected A/B bank.

## 2. Core-System Command Interface

The core system does not decode CPU MMIO registers. It accepts descriptor
commands from the Host wrapper:

```text
cmd_valid && cmd_ready
  -> capture cmd_desc_addr
  -> descriptor fetch/decode
  -> movement / compute / writeback
  -> core_done pulse + completed perf snapshot
```

The command processor owns descriptor memory reads and descriptor-field
decode. The Host wrapper owns only the descriptor-address register and command
submission. This split keeps the CPU ABI outside the autonomous execution
engine while allowing the scheduler to understand operator dependencies and
memory addresses.

## 3. Scheduler And Compute-Cluster Boundary

The v0 uop scheduler is an explicit RTL module between the command processor
and compute cluster. For the legacy `op=0` path it owns:

```text
start
  -> fetch instr_mem[pc]
  -> decode LOAD / MATMUL / STORE / HALT
  -> issue one local-storage or engine command
  -> wait for engine completion when required
  -> done
```

The scheduler reports separate active and wait events. Fetch/decode/issue and
completion handling are scheduler active work. Waiting for an issued engine is
scheduler wait time. These cycles must not be attributed to Matrix engine
datapath activity.

The current fixed `op=1` attention-softmax and `op=3` scale paths still use a
bring-up sequencer colocated with the compute-cluster integration module. They
must migrate to the common scheduler before being treated as the final
primitive scheduling architecture.

`npu_v0_compute_cluster` ports:

`npu_v0_compute_cluster` ports:

| Signal | Direction | Meaning |
| --- | --- | --- |
| `clk/rst_n` | input | clock/reset |
| `start` | input | one-cycle launch pulse from the command processor |
| `op[1:0]` | input | launch mode: uop program, attention softmax v1, or mixed PV matmul |
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

`op` mode behavior:

| `op` | Mode | Current behavior |
| ---: | --- | --- |
| `0` | uop program | fetch and execute the preloaded Phase 0 uop stream |
| `1` | attention softmax v1 | run the integrated row-softmax primitive sequence over all eight rows of `dram_c` and write Q0.15-style words back to `dram_c` |
| `2` | mixed PV matmul | run the shared matmul path in `u16 x s8 -> int32` Q15-shifted mode |
| `3` | attention score scale/mask v1 | scale one unmasked `8x8 int32` score tile through vector requant v2 |

The `op=1`, `op=2`, and `op=3` paths are Transformer bring-up entry points. They reuse
the current host preload/output windows and wrapper descriptor launch path; they
are not yet a general primitive scheduler or grouped attention command list.

## 4. Internal Memories

| Storage | Width | Entries | Purpose |
| --- | ---: | ---: | --- |
| `dram_a` | unsigned/signed 16 | 64 | matmul A/probability preload window; low 8 bits are used for normal int8 matmul |
| `dram_a_bank1` | unsigned/signed 16 | 64 | alternate host preload bank for A/probability staging |
| `dram_b` | signed 8 | 64 | matmul B preload window |
| `dram_b_bank1` | signed 8 | 64 | alternate host preload bank for B staging |
| `dram_c` | signed 32 | 64 | matmul C output window |
| `dram_x` | signed 8 | 8 | softmax X preload window |
| `dram_y` | unsigned 32 | 8 | legacy softmax Y output window |

Attention softmax v1 reuses the 64-word C window because scaled scores are
int32 and contain eight rows, while the legacy X window is only eight signed
bytes. It processes each C row through the shared vector/reduction/SFU sequence
and overwrites that dead score row with Q0.15 probabilities. The wrapper then
stores C to the runtime probability buffer. This preserves the legacy softmax
X/Y behavior and avoids adding another tile storage module.
| `instr_mem` | 32 | 16 | encoded uop program |
| `accumulator_file` | signed 32 | 2 banks x 64 | matmul accumulator/output staging; v0 currently uses bank 0 |
| `vec_buf` | signed 16 | 8 | softmax vector staging |

Names like `dram_a` are historical. These are internal core arrays in the
current RTL, not external DRAM.

For K-streaming matmul, `matrix/accumulator_file.sv` is the resident
partial-sum storage. A/B operand banks alternate across K chunks while
accumulator bank 0 persists for the complete descriptor.

对于 K-streaming matmul，`matrix/accumulator_file.sv` 是常驻 partial-sum
storage。A/B operand bank 在 K chunk 之间交替使用，而 accumulator bank 0
在整个 descriptor 期间保持。

Each chunk commits independently; there is no final operation that combines
two operand-bank results:

```text
descriptor start -> clear accumulator bank 0
chunk 0 Matrix done -> accumulator bank 0 += chunk 0 result
chunk 1 Matrix done -> accumulator bank 0 += chunk 1 result
...
final output -> accumulator bank 0 -> C output window -> external SRAM
```

The current RTL implements each accumulator commit/add as one full-tile-wide
clocked operation over 64 int32 elements. It reruns the same uop program for
every K chunk, but the Command processor suppresses `STORE C, ACC` execution
for non-final chunks; only the final resident sum is copied into the C output
window before external writeback. The commit/add and final copy are valid
current RTL cycle events but do not yet model realistic
accumulator/output-window port width or banking cost.

## 5. Host Window Map

| Address range | Access | Storage |
| ---: | --- | --- |
| `0x000` - `0x03f` | write | `dram_a` |
| `0x100` - `0x13f` | write | `dram_b` |
| `0x200` - `0x23f` | read/write while idle | `dram_c`; write is used by int32 scale/mask descriptor input |
| `0x300` - `0x307` | write | `dram_x` |
| `0x380` - `0x387` | read | `dram_y` |
| `0x400` - `0x40f` | write | `instr_mem` |
| `0x500` | write | matmul accumulate control: bit 0 enable, bit 1 clear pulse |

The core does not validate window overflows beyond these simple address ranges.

## 6. Program Execution

For `op=0`, the uop scheduler runs:

```text
ST_IDLE
  -> ST_FETCH on start
  -> ST_MATMUL when MATMUL uop starts
  -> ST_FETCH when matmul array completes
  -> ST_DONE on HALT or unknown opcode
  -> ST_IDLE after wrapper drops start
```

In `ST_FETCH`, the scheduler reads `instr_mem[pc]`, increments `pc`, and
dispatches the uop. LOAD/STORE commands operate on compute-cluster local
storage through explicit scheduler commands. MATMUL is issued to Matrix engine
and the scheduler waits for its completion.

Supported uops:

| Uop | Current behavior |
| --- | --- |
| `LOAD A/B` | bind the selected MatMul operand bank; no duplicate full-tile copy |
| `LOAD X` | copy the preloaded X window into the softmax vector buffer |
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

For `op=1`, the core bypasses the uop stream and runs a fixed attention
row-softmax sequence:

```text
prepare input row
  -> REDUCE_MAX
  -> vector subtract row max
  -> vector clamp to EXP range
  -> per-lane SFU EXP
  -> REDUCE_SUM
  -> SFU RECIP
  -> vector normalization
  -> ST_DONE
```

This path uses the standalone `vector_engine`, `reduction_engine`, and
`sfu_lut` modules through start/done pulses. It is descriptor-visible and useful
for stage-level attention bring-up, but it is still not the final scheduler
contract because there is no valid/ready backpressure, response queue, or
per-engine stall counter.

For `op=2`, the core uses the same matmul FSM but drives
`matmul_array.mixed_u16s8_q15=1`. This lets the A-side operand carry unsigned
Q0.15 probabilities while B remains signed int8. The array accumulates Q15
products internally and exposes `result >>> 15` as int32 output.

For `op=3`, the wrapper loads an int32 score tile into `dram_c`. The core
processes eight rows through `VEC_REQUANT_V2` using multiplier `11585`, shift
`15`, and round-nearest-away-from-zero, then writes scaled int32 values back to
`dram_c`. Mask policy is `none`; causal/padding/tail masks are not claimed.

## 6. Matmul A1 Array

`matmul_array.sv` is parameterized by `M`, `N`, and `K`. Current config is
8x8x8.

Normal `op=0` data shape:

```text
A: M x K, signed int8
B: K x N, signed int8
C: M x N, signed int32
```

Mixed `op=2` data shape:

```text
A/P: M x K, unsigned Q0.15 in 16-bit lanes
B:   K x N, signed int8
C:   M x N, signed int32 after arithmetic shift by 15
```

For attention PV:

```text
P = softmax(scores)      unsigned Q0.15 probability
V = value matrix         signed int8
O = P * V               signed int32 output for current bring-up
```

For one output element:

```text
P_real          = P_q15 / 32768
acc_q15[i,j]    = sum_k P_q15[i,k] * V_int8[k,j]
O_int[i,j]      = acc_q15[i,j] >>> 15
```

Example:

```text
P_q15 = [24576, 8192]    // [0.75, 0.25]
V     = [20, -12]
acc_q15 = 24576 * 20 + 8192 * (-12) = 393216
O_int   = 393216 >>> 15 = 12
```

The output is an integer value in the V domain. It is not an int8 activation
until a separate output requant rule is applied.

Behavior:

- on `start`, clear 64 result accumulators;
- for each active `k_idx`, update all `M*N = 64` output accumulators in
  parallel;
- each active cycle performs 64 signed int8-by-int8 MACs into int32 results;
- after `K` slices, assert `done`;
- `npu_v0_compute_cluster` commits `result_flat` into `accumulator_file`.

The nested `for i/j` loops inside the clocked block describe many same-cycle
register updates, not software-style serial loop execution. Only `k_idx`
advances across cycles. This is why the measured matmul compute phase moved
from the old 512-cycle scalar baseline to about 10 cycles.

For a cycle-by-cycle diagram of the current 64-MAC/cycle behavior, see
`docs/design/v0_cnn/fc1_k_streaming_matmul.md`, section
`2.1 Cycle-By-Cycle Example / 逐拍计算例子`.

K-streaming matmul does not change this physical parallelism. It repeats the
same `8x8x8` array operation for multiple K chunks and changes the commit
semantics from:

```text
accumulator_file = tile_result
```

to:

```text
accumulator_file += tile_result
```

when `matmul_accumulate_enable` is set through host address `0x500`.

K-streaming matmul 不改变物理并行度。它只是对多个 K chunk 重复执行同一个
`8x8x8` array operation，并在 `matmul_accumulate_enable` 置位时把提交语义从
覆盖改为累加。

Mixed mode changes the operand width and output scaling, not the physical
parallelism: the array still updates 64 output elements per active K slice.
Until the PPA model is upgraded, mixed `16x8` multiplier area/energy is reported
as an L0 model limitation, not a real ASIC cost.

Detailed A1 explanation is in `docs/matmul_array_a1.md`.

Mixed PV verification requirements:

- Python golden for `attention_pv_q15_i8_i32`;
- matrix RTL test for direct mixed `u16s8_q15` behavior;
- CPU-to-NPU transformer workload `transformer_attention_pv_s8_d8` using
  `SOC_NPU_JOB_OP_MATMUL_U16S8_Q15`;
- PPA report labels PV provenance as measured mixed matrix path and states that
  current L0 area/energy uses generic MAC model coefficients.

## 7. Softmax Path

Legacy `op=0` softmax uop program:

```text
LOAD X -> VREDMAX -> VSUB -> VEXP -> VREDSUM -> VDIV -> STORE Y -> HALT
```

The RTL implements each vector operation as an immediate task over the whole
8-element vector. This means softmax timing is not yet representative of a real
multi-cycle vector/SFU pipeline.

Attention `op=1` softmax is a newer bring-up path. It sequences the standalone
primitive modules from the core FSM and writes Q0.15-style outputs to `dram_y`.
It is more representative of the intended decomposition than the legacy uop
tasks, but its primitive engines still use start/done pulses and simplified SFU
EXP/RECIP behavior.

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
Matrix datapath cycles:     8
softmax total cycles:      30
softmax core cycles:       11
```

Matmul job time is now dominated by wrapper/data movement rather than core
compute.

## 9. Limitations

- `instr_mem` has only 16 entries.
- Program is preloaded before launch; no instruction streaming/prefetch.
- Core cannot directly access SoC SRAM.
- Legacy uop vector/SFU operations are single-cycle tasks.
- Attention primitive engines still use start/done, not valid/ready.
- `op=1` attention softmax is a fixed `8x8` tile loop, not a general tiled or
  grouped full-attention command.
- `op=2` mixed PV has measured cycles but uses a generic modeled `16x8`
  area/energy estimate.
- SFU EXP still uses the current bring-up approximation unless the target
  257-entry LUT path is explicitly implemented and selected.
- No issue queue, hazard tracking, or pipeline backpressure.
- No scratchpad bank conflict model.
- Host writes are blocked during execution instead of using double buffering.

## 10. Next Work

Core changes should now follow the Transformer v1 evidence path:

1. Keep the A1 matrix array stable while mixed PV and attention stage reports
   remain regression-covered.
2. Review and implement the primitive valid/ready contract before adding a
   scheduler-visible vector/reduction/SFU issue path.
3. Add per-engine active/stall/op counters from reviewed event sources before
   exposing new Transformer CSRs or PPA fields.
4. Replace fixture-specific attention smoke sequencing with compiler/runtime
   generated QK -> scale/mask -> softmax -> PV stage groups.
5. Upgrade requant, mask semantics, and SFU EXP/RECIP numerical contracts
   before using measured softmax as target accuracy/PPA evidence.
6. Add scratchpad/bank visibility and conflict modeling before widening the
   core memory path or claiming overlap improvements.
