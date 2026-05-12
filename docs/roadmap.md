# Iteration Roadmap

## Phase 0: Foundations

Deliverables:

- Architecture schema.
- `npu_v0.yaml` config.
- ISA assembly/JSON format.
- CPU golden operators.
- Functional simulator memory model.
- CLI skeleton.

Exit condition:

- A handwritten `LOAD/MATMUL/STORE/HALT` program runs in the simulator.

## Phase 1: Minimal Closed Loop

Deliverables:

- Matmul compiler path.
- Softmax compiler path.
- Runtime simulator backend.
- RTL template generator for v0 modules.
- RTL testbench for generated programs.
- End-to-end tests for `matmul`, `softmax`, and `matmul -> softmax`.

Exit condition:

- Compiler-generated program passes functional simulator and RTL simulation.

## Phase 2: Realistic Scheduling

Deliverables:

- Multi-tile matmul.
- Edge tile handling.
- Double buffering.
- DMA/compute overlap.
- Cycle simulator with bus and SRAM contention.
- Performance counters.

Exit condition:

- Cycle model identifies utilization bottlenecks and matches RTL trends.

## Phase 3: Datatype And Quantization

Deliverables:

- INT8 quantized matmul.
- FP16 or BF16 path.
- Scale/zero-point metadata.
- Requantization instruction or unit.
- Accuracy reports versus CPU reference.

Exit condition:

- Quantized `matmul -> softmax` runs with defined accuracy tolerance.

## Phase 4: Architecture Exploration

Deliverables:

- Parameter sweep tool.
- MAC array size sweep.
- Scratchpad size/banking sweep.
- Bus width/topology sweep.
- DMA channel sweep.
- PPA dashboard.

Exit condition:

- Architecture changes can be compared by throughput, SRAM cost, MAC
  utilization, bandwidth pressure, and estimated power.

## Phase 5: Operator Expansion

Candidate operators:

- Elementwise add/mul.
- ReLU/GELU.
- LayerNorm.
- Convolution lowered to matmul or direct convolution.
- Attention block primitives.

Exit condition:

- A small transformer or CNN block compiles and verifies end to end.

## Phase 6: Production-Grade Tooling

Deliverables:

- Stable IR.
- Better diagnostics.
- Binary instruction encoding.
- Trace viewer.
- Differential debugging between simulator and RTL.
- FPGA bring-up path.
- ASIC-oriented synthesis reports.

Exit condition:

- The system can support repeated architecture experiments without manual
  rewiring of compiler, runtime, simulator, and RTL.

## PPA Decision Rules

Do not accept architecture complexity without measurement. Each proposed change
should state:

- Which workload improves.
- Which bottleneck it addresses.
- Expected performance gain.
- Area/power/cost increase.
- Compiler/runtime impact.
- Verification impact.

Examples:

| Change | Accept when |
| --- | --- |
| Wider bus | DMA bandwidth is measured bottleneck and SRAM ports can consume it |
| Larger MAC array | Utilization remains high and memory system can feed it |
| More scratchpad | Tile reuse improves enough to reduce DRAM traffic |
| More DMA channels | Overlap improves wall-clock cycles despite arbitration cost |
| Better SFU | Softmax/GELU dominates measured workload time or accuracy |

