# NPU Descriptor v1

## Target

Descriptor v1 carries enough shape and memory identity for CNN regression,
Transformer micro workloads, and Level 0 PPA grouping. The first RTL
implementation may execute only a subset of job types, but encodings and
semantics are reserved here.

## Layout

```c
struct npu_job_desc_v1 {
    uint32_t job_id;
    uint32_t job_type;
    uint64_t program_addr;
    uint64_t input0_addr;
    uint64_t input1_addr;
    uint64_t input2_addr;
    uint64_t output_addr;
    uint64_t kv_base_addr;

    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t head_dim;
    uint32_t num_heads;

    uint32_t input0_words;
    uint32_t input1_words;
    uint32_t output_words;
    uint32_t flags;
};
```

## Job Types

| Name | v1 purpose |
| --- | --- |
| `MATMUL_TILE` | current 8x8x8 tile GEMM |
| `MATMUL_K_STREAM` | K-streaming tile with accumulator residency |
| `GEMV_TILE` | decode-oriented vector/matrix tile; may initially map to skinny GEMM |
| `SOFTMAX_ROW` | compiler micro-kernel target, not required as hardware macro-op in v1 |
| `RMSNORM_ROW` | compiler micro-kernel target, not required as hardware macro-op in v1 |
| `QK_ATTENTION_SCORE` | Q x K^T attention score tile |
| `ATTN_PV` | attention probability x V tile |
| `FFN_TILE` | feed-forward projection tile |
| `KV_CACHE_READ` | KV read traffic or streamer job |
| `KV_CACHE_WRITE` | KV write traffic or streamer job |

The first implementation can execute only `MATMUL_TILE`, `MATMUL_K_STREAM`,
and current-compatible Transformer matmul micro workloads. Unsupported job
types must be reported as planned or unavailable, not silently treated as
measured hardware support.

## Attention Row-Mask Descriptor Contract

Architecture decision: regular causal, padding, and tile-tail masks use a
compact descriptor-referenced row-mask table. They do not add a standalone
`MASK` uop.

Implementation status: the `input1_words=0/2` transport path, local row-mask
registers, Scheduler launch dependency, and Vector/Reduction lane gating are
implemented for single-tile Scale/Mask and Softmax. Compiler/runtime reject
unsupported plans. Hardware-visible descriptor rejection/status for malformed
values other than `0/2`, bad addresses, and all-invalid rows remains pending.

中文说明：Mask不通过新增ISA指令表达，而是作为当前Attention tile共享的
descriptor数据输入。`op_type`说明当前任务如何解释`input1`；
`input1_words`说明是否存在Mask以及长度。Command Processor解析descriptor，
Data Mover搬运Mask，Uop Scheduler仍执行原有row-indexed指令。

For Attention Scale/Mask and Softmax jobs:

```text
input0_addr/input0_words = score or scaled-score tile
input1_addr/input1_words = packed row-mask table
```

The current physical tile is `8x8`. Each physical query row owns one eight-bit
`valid_lane_mask`; the eight row masks pack into one 64-bit table, stored as
two 32-bit words:

```text
word 0 = row_mask[0] | row_mask[1] << 8 | row_mask[2] << 16 | row_mask[3] << 24
word 1 = row_mask[4] | row_mask[5] << 8 | row_mask[6] << 16 | row_mask[7] << 24
```

Descriptor interpretation is selected by `op_type`; no ABI field is added:

| `op_type` | `input1_words=0` | `input1_words=2` | Other value |
| --- | --- | --- | --- |
| `ATTENTION_SCALE_MASK_V1` | implicit all-valid mask | packed row-mask table | descriptor error |
| `ATTENTION_SOFTMAX_V1` | implicit all-valid mask | packed row-mask table | descriptor error |
| other existing operations | preserve the operation's existing `input1` meaning | preserve existing meaning | preserve existing validation |

For a masked job, `input1_addr` must be 32-bit aligned and reference two
readable words. Scale/Mask and Softmax descriptors reference the same table.
For an unmasked job, `input1_words=0`; `input1_addr` is ignored and local mask
registers are initialized to all ones. A physically stored all-ones table is
legal but unnecessary.

### Required hardware sequence

```text
Command Processor finishes descriptor read
  -> validate op_type/input1_words/address combination
  -> Data Mover reads mask word 0 and word 1 when input1_words=2
  -> unpack into local_row_mask[0..7]
  -> assert row_mask_ready
  -> launch Uop Scheduler
  -> each accepted row-indexed primitive selects local_row_mask[row_index]
```

`row_mask_ready` is a launch dependency, not a new ISA-visible instruction.
The Command Processor must not launch Scale/Mask or Softmax before the table is
ready. The local table remains stable until that descriptor completes. A new
Attention descriptor overwrites or reinitializes it; other descriptor types
must not consume it.

The initial local implementation is eight 8-bit registers with one
combinational row-indexed read port. The selected mask accompanies the
Scheduler-issued primitive into Vector/Reduction routing. Mask-table movement
cycles belong to Data Mover activity; descriptor validation/unpack/launch
cycles belong to Command Processor activity. Neither is reported as Vector,
Reduction, or Uop execution.

### Error behavior and PPA

The descriptor fails before Scheduler launch when:

- an Attention mask job uses an unsupported `input1_words` value;
- a required mask address is unaligned or cannot be read;
- a row selected for execution has no valid lane, until an all-invalid-row
  numerical contract is approved.

Perf/PPA must report `mask_table_words`, mask-load interval, mask policy, valid
lane count, and any descriptor rejection. The unmasked path must not invent a
two-word movement interval.

Why descriptor metadata instead of a new `MASK` uop:

- mask policy is job/tile metadata shared by many row primitives;
- existing Softmax and Scale/Mask uops already carry row index;
- adding a mask uop per row would increase instruction words and serialized
  Scheduler/handshake cycles without performing useful arithmetic;
- two packed words add a small, explicitly measured movement cost.

Arbitrary dense masks are not covered by this v1 contract. They require a
separate reviewed tensor-storage and movement contract.
