# Transformer NPU Architecture v1

## 1. Target / 目标

Transformer NPU v1 turns the project from a CNN/MNIST regression SoC into a
Transformer-oriented tensor NPU baseline. The goal is a verified and PPA-visible
foundation for edge LLM inference, not a full LLaMA runtime.

V1 scope:

- single-batch tiny decoder-only Transformer envelope;
- `seq_len = 32 / 128`, `hidden = 64 / 128`, `heads = 4`,
  `head_dim = 16 / 32`;
- int8 activation/weight, int32 accumulator;
- fixed-point softmax internals;
- int8 KV cache v1 as spec/counters first;
- primitive uops and micro-kernels before macro-op hardware;
- MNIST/CNN remains a regression workload.

Out of v1 scope: complete LLaMA, fused attention pipeline, hardware macro-op
expansion, reorder scheduler, real LPDDR controller, INT4/FP8, multi-core NPU.

## 2. Overall Design / 整体设计思路

The architecture stays unified:

```text
CPU / firmware
  -> wrapper / CSR
  -> descriptor engine
  -> uop scheduler
  -> memory + data mover
  -> matrix / accumulator / vector / reduction / SFU engines
  -> perf + PPA report
```

CNN and Transformer jobs share wrapper, descriptor identity, workload manifest,
perf report, and PPA Level 0 report. Transformer support is added as new
primitive capabilities and workload metadata rather than as a second core.

The first Transformer path uses current K-stream matmul where possible, adds
golden/model-only coverage for vector/reduction/SFU micro-kernels, and exposes
utilization fields so decode GEMV/skinny-GEMM waste is visible before building
new datapaths.

## 3. Key Details / 重点细节

Canonical v1 references:

| File | Role |
| --- | --- |
| `arch/configs/npu_transformer_v1.jsonc` | v1 architecture config |
| `arch/specs/transformer/v1/transformer_npu_v1.md` | module, counter, and PPA contract |
| `arch/specs/transformer/v1/transformer_numerical_v1.md` | fixed-point softmax/RMSNorm numerical contract |
| `arch/specs/transformer/v1/csr_map_v1.md` | wrapper CSR map |
| `arch/specs/transformer/v1/descriptor_v1.md` | job descriptor and job types |
| `arch/specs/transformer/v1/uop_isa_v1.md` | primitive uop ISA |

Primitive uop, micro-kernel, macro-op expansion, and fused hardware pipeline
are distinct:

- primitive uop: directly issued hardware primitive;
- micro-kernel: compiler/software sequence of primitive uops;
- macro-op expansion: future scheduler/compiler expansion of compact row ops;
- fused pipeline: dedicated multi-stage hardware datapath, out of v1 scope.

The accumulator file becomes an explicit architectural module:

| Field | v1 value |
| --- | --- |
| dtype | int32 |
| tile | 8 x 8 |
| banks | 2 |
| operations | clear, accumulate/write, read, store path |
| counters | read, write, clear, residency cycles, spill count |

Current RTL keeps the verified `hw/` layout. The staged target layout remains
`rtl/npu/matrix/accumulator_file.sv`; the current implementation equivalent is
`hw/npu_core/rtl/matrix/accumulator_file.sv`.

## 4. Verification / 验证测试

V1 acceptance keeps existing gates:

```text
make test
make firmware-smoke
make perf-report
make ppa-proxy-report
```

New Transformer coverage must add:

- Python golden for micro workloads;
- at least one executable Transformer micro workload in `perf-report`;
- shape metadata in workload manifest;
- matrix/GEMV/skinny-GEMM utilization fields in perf and PPA reports;
- null/unavailable fields when a metric is not implemented rather than guessed;
- explicit Level 0 proxy wording for modeled area/energy.

## 5. Implementation Priority / 实现优先级

1. Land v1 architecture/spec/numerical/CSR/descriptor/uop documents and config.
2. Add standalone accumulator file RTL and keep current K-stream regression
   behavior stable.
3. Extend Transformer micro workload metadata and Python golden coverage.
4. Add report-derived utilization metrics from manifest shape metadata.
5. Feed utilization and KV traffic into Level 0 PPA output.
6. Add primitive vector/reduction/SFU RTL blocks and tests after the numerical
   golden contract is stable.
