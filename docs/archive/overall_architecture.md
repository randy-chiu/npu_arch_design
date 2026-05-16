# Overall Architecture

## 1. Design Principles

The system should not be built as separate hardware and software projects. It
should be built around one source of truth: an architecture description that
drives RTL generation, compiler target description, runtime memory layout,
simulator behavior, tests, and PPA analysis.

The first milestone must be small enough to verify end to end. The useful unit
of progress is not "a better MAC array" or "a compiler pass" in isolation. The
useful unit is a program that compiles, runs on the simulator, runs on generated
RTL, and matches a CPU golden result.

## 2. Top-Level System

```text
                 model / op graph
                       |
                       v
              +------------------+
              | frontend / importer
              +------------------+
                       |
                       v
              +------------------+
              | compiler / mapper|
              +------------------+
                       |
        program binary | metadata
                       v
+--------------------------------------------------+
| runtime                                          |
| - memory allocation                              |
| - command submission                             |
| - host/NPU synchronization                       |
+--------------------------------------------------+
                       |
             command queue / MMIO
                       v
+--------------------------------------------------+
| generated NPU hardware                           |
| - control processor / sequencer                  |
| - instruction decoder                            |
| - DMA / load-store                               |
| - scratchpad / buffers                           |
| - compute array                                  |
| - special function units                         |
| - internal interconnect                          |
+--------------------------------------------------+
                       |
        trace / counters / waveform / results
                       v
+--------------------------------------------------+
| verification and PPA loop                        |
| - ISA simulator                                  |
| - cycle simulator                                |
| - RTL simulation                                 |
| - golden CPU reference                           |
| - performance counters                           |
| - area/power/timing estimates                    |
+--------------------------------------------------+
```

## 3. Repository Shape

Recommended future implementation layout:

```text
npu/
  arch/
    schema/                 # architecture schema and validation
    configs/                # concrete NPU configurations
    isa/                    # instruction definitions and encoders
  hw/
    generators/             # RTL generators
    rtl/                    # generated or checked-in RTL
    tb/                     # RTL testbench
  compiler/
    ir/                     # internal IR
    passes/                 # tiling, scheduling, allocation, lowering
    targets/                # target-specific lowering rules
  runtime/
    host/                   # host API, command submission, memory management
    driver_model/           # simulator or MMIO driver abstraction
  kernels/
    matmul/
    softmax/
  simulator/
    isa/                    # functional simulator
    cycle/                  # timing/cycle model
  verification/
    golden/                 # CPU golden models
    tests/                  # end-to-end tests
    traces/                 # trace comparison tools
  ppa/
    perf/                   # performance model and reports
    power/                  # activity and power estimation
    area/                   # area/cost estimation
  tools/
    cli/                    # build, compile, simulate, verify commands
```

## 4. Architecture Source Of Truth

The architecture spec should be a structured file, for example YAML or JSON:

```yaml
name: npu_v0
data_types: [int8, int32, fp16]
memory:
  host_dram_bytes: 1073741824
  scratchpad_bytes: 262144
  accumulator_bytes: 65536
compute:
  array_m: 16
  array_n: 16
  mac_lanes: 256
  accumulator_width: 32
isa:
  encoding_bits: 64
  instructions:
    - load
    - store
    - matmul
    - reduce_max
    - exp_approx
    - reduce_sum
    - div_approx
    - sync
bus:
  data_width_bits: 256
  topology: shared_crossbar
dma:
  channels: 2
  max_burst_bytes: 256
```

This spec should generate or configure:

- ISA encoder/decoder tables.
- Compiler target constraints.
- Runtime memory map and command format.
- Simulator instruction semantics.
- RTL module parameters.
- Test generators and PPA model inputs.

## 5. Hardware Architecture

### 5.1 Control Plane

The minimal control plane should be a deterministic micro-sequencer, not a full
CPU. It fetches instructions from program memory, decodes fixed-width
instructions, issues commands to DMA, compute, and SFU blocks, and updates
status registers.

Key modules:

- `npu_top`: top-level integration, host interface, reset, clocks.
- `cmd_regs`: MMIO registers for start, done, error, program base, tensor bases.
- `sequencer`: program counter, fetch, decode, dependency tracking.
- `scoreboard`: tracks outstanding DMA and compute operations.
- `irq/status`: optional completion interrupt.

### 5.2 Memory System

Start with explicit software-managed memory. Avoid hardware caches in the first
version.

Memory hierarchy:

- Host DRAM: tensor storage and program binary.
- Program memory: read-only instruction stream visible to sequencer.
- Scratchpad: input/output tiles.
- Accumulator buffer: widened partial sums.

DMA should support:

- 1D contiguous copy for the first version.
- 2D strided copy in the second version.
- Optional transpose/layout conversion later.

### 5.3 Compute Path

Initial compute block:

- 2D systolic or output-stationary MAC array.
- INT8 input operands.
- INT32 accumulation.
- Configurable tile dimensions.
- Deterministic latency model.

Softmax path:

- For the first version, implement softmax as microcoded vector passes:
  `reduce_max`, `sub`, `exp_approx`, `reduce_sum`, `div_approx`.
- `exp_approx` and `div_approx` can initially be coarse LUT or piecewise linear.
- Correctness threshold should be relaxed for approximate math, then tightened
  as the SFU improves.

### 5.4 Internal Bus

First version:

- Shared valid/ready crossbar or simple arbiter.
- One DMA master and one compute master are enough.
- Fixed data width, for example 256 bits.

Later versions:

- Separate read/write networks.
- Banked scratchpad arbitration.
- QoS and conflict counters.
- NoC-style topology only after shared interconnect becomes a measured bottleneck.

## 6. ISA

Use a small fixed-width instruction encoding first. Keep instruction semantics
coarse enough for compiler productivity, but not so high-level that hardware
cannot be realistically scheduled.

Minimal ISA:

| Instruction | Purpose |
| --- | --- |
| `LOAD` | Copy tensor tile from DRAM to scratchpad |
| `STORE` | Copy tensor tile from scratchpad to DRAM |
| `MATMUL` | Matrix tile multiply-accumulate |
| `VREDMAX` | Vector reduce max |
| `VSUB` | Vector subtract scalar |
| `VEXP` | Approximate vector exponential |
| `VREDSUM` | Vector reduce sum |
| `VDIV` | Vector divide by scalar |
| `SYNC` | Wait for prior async work |
| `HALT` | Finish program |

Instruction fields should include opcode, buffer IDs, offsets, dimensions,
strides, datatype, and flags. The first implementation can use a readable
assembly or JSON program representation and add binary packing later.

## 7. Compiler Architecture

Compiler pipeline:

```text
frontend graph
  -> operator IR
  -> shape/type inference
  -> tiling
  -> memory planning
  -> schedule generation
  -> ISA lowering
  -> assembly/binary emission
```

Initial frontend:

- Direct Python API or JSON graph.
- Only `matmul` and `softmax`.
- Static shapes only.

Compiler responsibilities:

- Validate supported shapes and dtypes.
- Choose tile sizes based on architecture spec.
- Allocate scratchpad regions.
- Emit explicit load/compute/store/sync instructions.
- Emit metadata for runtime input/output buffers.

## 8. Runtime Architecture

Runtime should hide target differences while keeping the command path explicit.

Host API:

```text
compile(graph, arch_config) -> program
load_program(program)
allocate_tensor(shape, dtype)
copy_to_device(tensor)
run(program, inputs, outputs)
copy_from_device(tensor)
```

Driver backends:

- Functional simulator backend.
- Cycle simulator backend.
- RTL simulator backend.
- Future FPGA/ASIC MMIO backend.

## 9. Simulator Architecture

Use two simulators:

- Functional ISA simulator: fast correctness check, instruction-level behavior.
- Cycle simulator: timing model, stalls, bus conflicts, utilization counters.

Both must consume the same program emitted by the compiler and the same
architecture config used by RTL generation.

## 10. Verification Strategy

Every operator should have:

- CPU golden implementation.
- Compiler-generated NPU program.
- Functional simulator result.
- RTL simulation result.
- Trace or counter comparison when failures occur.

End-to-end pass condition:

```text
input tensors
  -> compiler
  -> NPU program
  -> simulator / RTL
  -> output tensors
  -> compare with CPU golden
```

## 11. PPA Loop

Track PPA from the first milestone, even if estimates are rough.

Performance metrics:

- Cycles per operator.
- MAC utilization.
- DMA bandwidth utilization.
- Scratchpad bank conflicts.
- Softmax SFU utilization.

Area/cost metrics:

- MAC count.
- SRAM size.
- bus width and crossbar size.
- SFU LUT/table size.
- control logic estimate.

Power metrics:

- Toggle/activity estimate per module.
- SRAM access counts.
- MAC operation counts.
- DMA bytes moved.

Architecture changes should be accepted only when they improve a measured
target or unblock required functionality.

