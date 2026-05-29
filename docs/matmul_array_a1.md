# A1 Matmul Array Design

[TOC]

Status: implementation note for A1.

## Goal

Replace the Phase 0 scalar iterative matmul execution path with a small,
measurable matrix-array engine while preserving the current ISA, fixtures,
firmware launch path, and golden output checks.

The current matmul compute cost is:

```text
8 * 8 * 8 = 512 cycles
```

because `npu_v0_top` performs one MAC update per cycle. A1 changes only the
internal execution unit for `UOP_MATMUL`.

## First Target

Shape:

```text
M = 8
N = 8
K = 8
```

Data:

```text
A/B: int8
accumulator/result: int32
```

Dataflow:

```text
output-stationary
```

Compute behavior:

- keep 64 accumulators for the 8x8 output tile;
- each active cycle consumes one `k` slice;
- all 64 output accumulators update in parallel for that `k`;
- after 8 `k` cycles, result is available for store/writeback.

This is not yet a full systolic array with explicit wavefront fill/drain. It is
the smallest verifiable array-style matmul engine that exposes the architectural
shift from one MAC per cycle to many MACs per cycle.

## RTL Module

New module:

```text
hw/npu_core/rtl/matrix/matmul_array.sv
```

Interface:

```systemverilog
module matmul_array #(
    parameter int M = 8,
    parameter int N = 8,
    parameter int K = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done,
    input  logic [(M*K*8)-1:0]   a_flat,
    input  logic [(K*N*8)-1:0]   b_flat,
    output logic [(M*N*32)-1:0]  result_flat
);
```

The module uses packed flat ports. `npu_v0_top` flattens `spad_a/spad_b` and
unpacks `result_flat` into the core accumulator storage. This keeps the module internal interface
simulator-friendly and closer to a future synthesizable boundary.

## Core Integration

`npu_v0_top` keeps the existing states:

```text
ST_IDLE
ST_FETCH
ST_MATMUL
ST_DONE
```

On `UOP_MATMUL`:

1. pulse `matmul_start`;
2. enter `ST_MATMUL`;
3. wait for `matmul_done`;
4. copy/commit array results into accumulator storage;
5. return to `ST_FETCH`.

The program format does not change. Existing matmul fixtures, SoC firmware, and
wrapper descriptor flow should keep working.

## Expected Cycles

For the current 8x8x8 tile:

| Engine | Compute cycles |
| --- | ---: |
| scalar iterative baseline | 512 |
| ideal 8x8 array | 8 |
| conservative report estimate | 12 |

## Old vs New Execution

### How To Read RTL `for` Loops

For software readers, the most important rule is:

```text
An RTL `for` loop is not automatically a runtime loop.
```

In synthesizable/SystemVerilog-style hardware descriptions, a `for` loop inside
an `always_ff @(posedge clk)` block often means "generate this repeated hardware
assignment at this clock edge". Whether work is serial or parallel depends on
where the loop variable lives:

| Pattern | Meaning |
| --- | --- |
| loop variable is an RTL register updated each clock, such as `i_idx <= i_idx + 1` | serial work across cycles |
| loop variable is an `integer` used only inside the clocked block's `for` statement | repeated hardware assignments in the same cycle |

So software big-O intuition is not enough. The key question is:

```text
Does i/j/k advance as state across clocks, or does the RTL describe many
assignments that happen on the same clock edge?
```

### Old Scalar Baseline

The old implementation lived directly inside `npu_v0_top.sv` under
`ST_MATMUL`. It used loop indices equivalent to:

```text
i_idx
j_idx
k_idx
acc
```

Those were hardware state registers. On every clock, the FSM used the current
`i_idx/j_idx/k_idx` values to update exactly one multiply-accumulate:

```text
acc += spad_a[(i * K) + k] * spad_b[(k * N) + j]
```

Then the RTL advanced `k_idx`, and when K was done it advanced `j_idx`, and when
N was done it advanced `i_idx`.

Conceptually:

```text
cycle 0:  i=0, j=0, k=0  -> one MAC
cycle 1:  i=0, j=0, k=1  -> one MAC
...
cycle 7:  i=0, j=0, k=7  -> finish C[0,0]
cycle 8:  i=0, j=1, k=0  -> one MAC
...
cycle 511: i=7, j=7, k=7 -> finish C[7,7]
```

Hardware shape:

```text
one multiplier
one adder
one accumulator register
i/j/k state registers
```

Only one output element was being accumulated at a time. The hardware was small,
but time was proportional to all three loop dimensions:

```text
outputs = 8 * 8 = 64
MACs per output = 8
total cycles ~= 64 * 8 = 512
```

This was useful as a functional baseline because it was easy to inspect and
debug, but it did not model a modern tensor/cube-style matrix engine.

### New Array-Style Engine

The new implementation moves matmul into `hw/npu_core/rtl/matrix/matmul_array.sv`.
`npu_v0_top.sv` now only:

1. flattens `spad_a` and `spad_b`;
2. pulses `matmul_start`;
3. waits for `matmul_done`;
4. commits `result_flat` back into accumulator storage.

Inside `matmul_array.sv`, one active cycle processes one K slice:

```systemverilog
for i in 0..7:
  for j in 0..7:
    result[i,j] += A[i,k] * B[k,j]
```

Here `i` and `j` are not state registers that advance over many clocks. They are
`integer` loop variables inside the clocked block. The synthesis/simulation
meaning is closer to writing 64 assignments by hand:

```systemverilog
result[0,0] <= result[0,0] + A[0,k_idx] * B[k_idx,0];
result[0,1] <= result[0,1] + A[0,k_idx] * B[k_idx,1];
...
result[7,7] <= result[7,7] + A[7,k_idx] * B[k_idx,7];
```

Those assignments all happen on the same clock edge. The remaining serial state
is only `k_idx`, which advances once per cycle.

Hardware shape:

```text
64 multipliers
64 adders
64 accumulator registers
one k_idx state register
```

Cycle view:

```text
cycle 0: k=0 -> update all 64 C[i,j] accumulators
cycle 1: k=1 -> update all 64 C[i,j] accumulators
...
cycle 7: k=7 -> update all 64 C[i,j] accumulators
```

Therefore the core compute work changes from:

```text
64 outputs * 8 K steps = 512 scalar MAC cycles
```

to:

```text
8 K steps + start/done/FSM observation overhead ~= 10 measured cycles
```

This is why `make perf-report` now shows:

```text
core.matmul: 512 -> 10 cycles
matmul job:  738 -> 236 cycles
```

The job did not drop to roughly 8 cycles because the wrapper still spends time
fetching descriptor/program/input data and writing output back to SRAM:

```text
desc_read + fetch_program + fetch_input0 + fetch_input1
+ start_core + wait_core + write_output
```

After A1, those movement phases dominate the matmul timeline.

The full `matmul` job will not drop by 64x because wrapper descriptor fetch,
program fetch, input fetch, output writeback, and CPU polling are still visible.
That is the point of keeping `make perf-report` in the loop.

## Verification

Required checks:

- `make npu-core-sim`: exact matmul output still passes.
- `make cpu-soc-sim`: firmware-controlled matmul/softmax still passes.
- `make perf-report`: report shows measured matmul compute cycles moving from
  the scalar baseline toward the array estimate.
- `make test`: all unit and RTL/SoC smoke tests pass.

## Follow-Up

A1 intentionally does not solve memory movement. Once compute cycles shrink, the
timeline should make wrapper input/output movement dominate more clearly. That
will justify A2: real data movement, scratchpad banking, and overlap.
