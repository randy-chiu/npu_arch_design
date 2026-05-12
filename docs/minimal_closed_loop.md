# Minimal Closed-Loop System

## Goal

Build `npu_v0`: a small NPU system that can compile and run:

1. `C = matmul(A, B)`
2. `Y = softmax(X)`
3. `Y = softmax(matmul(A, B))`

The system is successful when the same generated program runs on the functional
simulator and generated RTL testbench, and both match CPU golden output.

## Scope

Supported data and shapes:

- `matmul`: INT8 x INT8 -> INT32 accumulator -> INT8 or FP16 output later.
- `softmax`: FP16 or fixed-point approximation. For the first version, FP32 in
  simulator is acceptable if RTL uses a deterministic approximate format.
- Static 2D tensors only.
- Batch size 1.
- Shapes must be multiples of tile size in the first milestone. Edge tiles come
  later.

Recommended first hardware configuration:

```yaml
name: npu_v0
compute:
  array_m: 8
  array_n: 8
  k_step: 8
  input_dtype: int8
  accum_dtype: int32
memory:
  scratchpad_bytes: 65536
  accumulator_bytes: 16384
bus:
  data_width_bits: 128
dma:
  channels: 1
isa:
  encoding: json_assembly_first
```

## Minimal Hardware Modules

```text
npu_top
  cmd_regs
  sequencer
  instruction_memory
  dma_engine
  scratchpad
  accumulator_buffer
  matmul_engine
  vector_sfu
  result_checker_hooks
```

Required behavior:

- Host writes program and tensor data into simulated DRAM.
- Host writes start register.
- Sequencer executes instructions until `HALT`.
- DMA moves data between DRAM and scratchpad.
- Matmul engine reads tiles and writes accumulator buffer.
- Vector SFU performs softmax sequence on scratchpad rows.
- DMA stores result to DRAM.
- Done bit is set.

## Minimal ISA Program Example

Readable assembly form:

```text
LOAD    dst=spad0, src=dram_A, bytes=tile_A
LOAD    dst=spad1, src=dram_B, bytes=tile_B
MATMUL  a=spad0, b=spad1, acc=acc0, m=8, n=8, k=8
STORE   dst=dram_C, src=acc0, bytes=tile_C
HALT
```

Softmax program:

```text
LOAD     dst=spad0, src=dram_X, bytes=row_bytes
VREDMAX  src=spad0, dst=scalar0, len=N
VSUB     src=spad0, scalar=scalar0, dst=spad0, len=N
VEXP     src=spad0, dst=spad0, len=N
VREDSUM  src=spad0, dst=scalar1, len=N
VDIV     src=spad0, scalar=scalar1, dst=spad0, len=N
STORE    dst=dram_Y, src=spad0, bytes=row_bytes
HALT
```

## Minimal Compiler

Compiler input:

```json
{
  "ops": [
    {
      "type": "matmul",
      "a": "A",
      "b": "B",
      "out": "C",
      "shape": {"m": 64, "n": 64, "k": 64},
      "dtype": "int8"
    }
  ]
}
```

Compiler output:

- NPU assembly or JSON instruction list.
- Tensor memory layout.
- Runtime launch metadata.
- Expected output dtype and tolerance.

First compiler passes:

1. Shape validation.
2. Tile selection from architecture config.
3. Scratchpad allocation.
4. Load/compute/store schedule.
5. ISA emission.

## Minimal Simulator

Functional simulator:

- Executes instruction list.
- Models DRAM, scratchpad, accumulator buffer.
- Implements exact or reference-like matmul.
- Implements softmax with deterministic approximation mode.

Cycle simulator:

- Can initially be a latency table:
  - DMA cycles = bytes / bus_bytes_per_cycle + setup.
  - MATMUL cycles = ceil(m*n*k / mac_lanes) + pipeline.
  - Vector cycles = ceil(elements / lanes) * op_latency.
- Later add bank conflicts and overlapping DMA/compute.

## Minimal RTL

RTL should be generated from the architecture config, but the first generator
can be simple templates with parameters.

Minimum RTL verification:

- Reset/start/done test.
- Single `LOAD`/`STORE`.
- Single `MATMUL` tile.
- Single `SOFTMAX` row.
- End-to-end `softmax(matmul(A, B))`.

## Golden Tests

Initial tests:

| Test | Purpose |
| --- | --- |
| `test_matmul_8x8x8` | Single tile correctness |
| `test_matmul_64x64x64` | Multi-tile schedule |
| `test_softmax_1x16` | Single row SFU correctness |
| `test_softmax_8x64` | Multiple rows |
| `test_matmul_softmax` | End-to-end graph |

Tolerance:

- INT8/INT32 matmul: exact match before quantized output.
- Softmax: start with absolute tolerance `1e-2`, then tighten as SFU improves.

## First Milestone Definition Of Done

`npu_v0` is done when one command can run:

```text
npu run --arch configs/npu_v0.yaml --graph tests/graphs/matmul_softmax.json --backend rtl
```

and produce:

- Compiler program dump.
- Simulator output.
- RTL output.
- Golden comparison pass/fail.
- Cycle count.
- Basic utilization report.

