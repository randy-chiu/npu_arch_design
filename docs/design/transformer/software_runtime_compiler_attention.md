# Software Runtime And Compiler Attention v1

## Scope

This document defines how attention is lowered and launched without adding a
dedicated attention RTL macro. Compiler and runtime own the attention sequence;
hardware exposes primitive engines and movement/counter contracts.

## Compiler Responsibility

The compiler lowers a logical attention op:

```text
attention(Q, K, V, mask) -> O
```

into a primitive command sequence:

```text
QK matmul tiles
score scale/mask/clamp vector ops
row softmax reduction/SFU/vector ops
PV matmul or weighted-sum tiles
output requant/store
```

The compiler must decide:

- tensor tile layout;
- whether K is pre-transposed or transposed during staging;
- score and probability intermediate allocation;
- score scale and Q format;
- mask policy;
- PV policy;
- command ordering and synchronization points;
- workload metadata for perf/PPA.

## Runtime Responsibility

The runtime launches the compiler-produced primitive sequence. In v1 there are
two possible launch models.

### Model A: Multi-Descriptor Firmware Loop

Firmware emits one descriptor per measurable stage or tile. This reuses the
current descriptor machinery and is suitable for measured QK first.

Benefits:

- minimal RTL change;
- uses existing job identity and perf snapshot flow;
- easy to compare stage timings.

Limits:

- high CPU/control overhead;
- intermediate tensors must be written to and read from SRAM between stages;
- not a final attention execution model.

### Model B: Command List Executor

Firmware launches one descriptor pointing to a command list. The wrapper/core
scheduler walks primitive commands.

Benefits:

- attention sequence is still software-defined, not a hardware macro;
- lower launch overhead;
- cleaner place to issue vector/reduction/SFU primitives;
- better PPA visibility for stage overlap and stalls.

Limits:

- requires new command-list ABI;
- requires primitive scheduler integration;
- requires counters beyond current matmul/data mover fields.

Recommendation:

1. Use Model A for measured QK and model-only softmax/PV.
2. Move to Model B before claiming full measured attention.

## Command List Requirements

A command list entry should be able to express:

| Field | Purpose |
| --- | --- |
| `opcode` | matrix, vector, reduction, SFU, load, store, barrier |
| `src0`, `src1`, `dst` | tensor/buffer identifiers |
| `shape` | tile shape, row length, valid lanes |
| `dtype` | int8, int32, Q0.15, Q24 |
| `scale`/`shift` | fixed-point scale policy |
| `clamp_low`/`clamp_high` | vector/requant clamp |
| `mask_id` | optional attention mask source |
| `stage_id` | QK, softmax, PV, output |
| `perf_scope` | counter grouping key |

The command list is a software-visible ABI. Any field addition must update
compiler, runtime, RTL scheduler, tests, and docs together.

## Tensor Layout

Initial prefill layout:

```text
Q tile: row-major S x D
K tile for QK: pre-transposed D x S
V tile: row-major S x D
score tile: row-major S x S int32
probability tile: row-major S x S Q0.15 or int8 after requant
O tile: row-major S x D
```

For `S=8,D=8`, each tensor fits one tile. For larger shapes:

```text
QK:
  M tile = query rows
  N tile = key rows
  K chunks = head_dim chunks

PV:
  M tile = query rows
  N tile = value dimensions
  K chunks = key/value sequence positions
```

The compiler must record logical shape and tile shape separately. PPA derives
useful MACs from logical shape, not just physical tile count.

## Intermediate Buffers

Attention introduces intermediate tensors that the current CNN/matmul path does
not model explicitly.

| Buffer | Dtype | Producer | Consumer |
| --- | --- | --- | --- |
| `score` | int32 | QK matrix | scale/mask/softmax |
| `exp` | Q0.15 | SFU EXP | row sum/normalize |
| `probability` | Q0.15 or int8 | softmax normalize | PV |
| `output_acc` | int32 | PV | output requant/store |

Current RTL host windows are sized around one tile and are not a reviewed
intermediate-memory contract. Before full measured attention, the project needs
a local tensor memory or scratchpad contract that describes:

- capacity;
- banks/ports;
- element width;
- row/tile addressing;
- read/write ownership between primitive engines;
- spill behavior to SRAM;
- perf counters for intermediate traffic.

## Current Software Status

Implemented today:

- Transformer workload manifest with logical shapes and external-memory fields.
- Deterministic matmul fixture generation for current `8x8x8` compatible
  workloads.
- Model-only metadata for softmax, RMSNorm, KV traffic, QK, and PV-like
  workloads.
- Perf/PPA can group workloads by generated `job_id` and manifest metadata.

Missing for attention:

- full attention golden function;
- attention parent/stage workload identity;
- command-list format;
- compiler lowering from logical attention to primitive commands;
- runtime support for vector/reduction/SFU commands;
- intermediate buffer allocator;
- stage-level perf scopes.

## Verification Plan

### Compiler tests

- Logical attention lowers to expected primitive stage list.
- K transpose/layout is deterministic and matches QK golden.
- Score/probability/output buffers do not overlap incorrectly.
- Shape metadata is emitted once at workload level.

### Runtime tests

- Generated descriptors and manifest share the same job/stage identity.
- Measured QK job produces exact int32 score tile.
- Model-only softmax/PV fields remain model-only until runtime can launch them.

### End-to-end tests

- `attention_prefill_s8_d8` Python golden passes.
- `perf-report` includes attention group metadata.
- `ppa-l0-report` includes attention bytes and stage provenance.

## Open Decisions

| Decision | Options | Required before |
| --- | --- | --- |
| PV numerical policy | int8 probability requant or mixed Q0.15 x int8 | measured PV RTL |
| mask support | unmasked first, causal mask, padding mask | softmax attention claims |
| launch model | multi-descriptor loop or command list | full measured attention |
| intermediate memory | host window, SRAM spill, local scratchpad | scheduler integration |
| score scale | power-of-two shift or multiplier requant | softmax/PV accuracy claims |
