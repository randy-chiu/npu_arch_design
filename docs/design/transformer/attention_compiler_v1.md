# Attention Compiler v1

## Scope / 范围

This document defines the compiler design for lowering a logical Transformer
attention operator to existing NPU primitive jobs. It is the implementation plan
for `sw/tools/npu_compiler` and its interaction with `sw/npu_core/operators`.

The compiler must replace the current attention-specific fixture logic as the
owner of execution planning. The fixture generator should only generate
deterministic tensor contents, expected outputs, and C/hex artifacts from
compiler output.

中文说明：

Compiler负责把逻辑Attention和大矩阵运算转换成硬件能够执行的tile、
primitive program、buffer和descriptor序列。Compiler决定Mask语义、M/N/K
分块、tile顺序和边界有效范围；它不负责RTL内部握手，也不能依赖fixture中
硬编码的数据布局或任务顺序。

## Current State

Implemented today:

- `sw/tools/npu_compiler/phase0.py` lowers simple graph `matmul` and `softmax`
  ops using `sw/npu_core/operators/phase0_intrinsics.json`;
- `sw/tools/npu_compiler/k_stream.py` plans K-stream chunks for one matmul
  output tile;
- `sw/tools/transformer/generate_transformer_micro_fixtures.py` directly
  selects attention QK, softmax, and PV executable jobs from the workload
  manifest;
- QK, softmax, and PV are measured as separate SoC jobs;
- full attention parent metadata exists, but no compiler-produced plan drives
  runtime execution of the full group.

Missing:

- typed Transformer attention operator IR;
- full lowering from `scaled_dot_product_attention_v1` to primitive stages;
- generated runtime command descriptors;
- compiler-owned grouped PPA metadata.

Implemented starter pieces:

- `sw/tools/npu_compiler/attention.py` emits the current `S=8,D=8`
  QK -> scale/mask -> softmax -> PV plan from the Transformer workload
  manifest. Unsupported shapes fail validation instead of silently using the
  bring-up plan.
- `sw/tools/npu_compiler/attention_plan_schema.py` validates stage order,
  typed intermediate buffers, and current runtime descriptor ops.
- `sw/tools/transformer/generate_transformer_micro_fixtures.py` now attaches
  the plan, stage metadata, and runtime-job metadata to generated Transformer
  fixture records.
- The plan marks the parent group as `software_group_measured_stages` once
  firmware runtime consumes the generated runtime-job table.
- The runtime-job table includes executable unmasked `scale_mask` using fixed
  multiplier `11585`, shift `15`, and round-nearest-away-from-zero.

## Compiler Inputs

| Input | Source | Used for |
| --- | --- | --- |
| Operator metadata | `sw/npu_core/operators/transformer_attention_v1.json` | operator names, dtype/layout checks, primitive sequence, numerical contracts |
| Workload manifest | `workloads/manifests/transformer/transformer_micro_v0.jsonc` | workload name, logical shape, attention group, PPA role |
| Tensor metadata | fixtures now, future model import | tensor shape, dtype, layout, quantization |
| Architecture config | `arch/configs/soc_v0.jsonc`, wrapper/core configs | descriptor op values, tile sizes, SRAM/program limits |

## Compiler Output

The compiler emits an `AttentionPlan`. The first coding version can be a Python
dictionary validated by tests; a stable JSON schema can follow after the shape
settles.

```text
AttentionPlan
  workload_name
  attention_group
  logical_op = scaled_dot_product_attention_v1
  shape
  numerical_contract
  tensors
  buffers
  stages[]
  runtime_jobs[]
  ppa
```

Stage entry example:

```json
{
  "stage_id": "qk",
  "operator": "matmul_s8s8_i32_tile",
  "inputs": ["q", "k_t"],
  "outputs": ["score_raw"],
  "shape": {"m": 8, "n": 8, "k": 8},
  "dtype": {"inputs": ["int8", "int8"], "output": "int32"},
  "numerical_contract": "attention_bringup_v0_qk_exact"
}
```

Runtime job entry example:

```json
{
  "job_id_symbol": "JOB_ID_TRANSFORMER_ATTENTION_QK_S8_D8",
  "stage_id": "qk",
  "descriptor_op": "matmul_k_stream",
  "input0": "q_tile_sram",
  "input1": "k_t_tile_sram",
  "output": "score_raw_sram",
  "program": "matmul_program",
  "k_chunks": 1,
  "perf_scope": "attention_prefill_s8_d8/qk"
}
```

Required stage order:

| Stage | Operator | Current executable? | Output |
| --- | --- | --- | --- |
| `qk` | `matmul_s8s8_i32_tile` | yes | `score_raw_i32` |
| `scale_mask` | `attention_score_scale_mask_v1` | yes for unmasked `S=8,D_k=8` | `score_softmax_in_i32` |
| `softmax` | `attention_softmax_q15_v1` | yes | `prob_q15_u16` |
| `pv` | `matmul_u16s8_q15_i32_tile` | yes | `o_i32` |

## Lowering Algorithm

### 1. Validate Logical Shape

The first executable attention target is:

```text
S_q = 8
S_k = 8
D_k = 8
D_v = 8
```

Validation rules:

- Q shape is `S_q x D_k`;
- K shape is `S_k x D_k` or already transposed `D_k x S_k`;
- V shape is `S_k x D_v`;
- physical tile sizes match current matrix and softmax contracts;
- PV `K` dimension equals `S_k`.

Unsupported shapes are rejected unless the workload is explicitly marked
`model_only`.

### 2. Choose K Layout

QK requires:

```text
Q[S_q,D_k] * K^T[D_k,S_k]
```

Compiler policy:

- if K is tagged `layout = transposed_d_by_s`, use it directly;
- otherwise materialize a staged tensor `k_t`;
- record the transform in the plan so fixture, golden, and runtime agree.

Firmware should not own the transpose decision.

### 3. Plan QK

For each query/key tile:

```text
score_raw_tile = matmul_s8s8_i32_tile(q_tile, k_t_tile)
```

Current shape:

```text
M = 8, N = 8, K = 8, k_chunks = 1
```

For future `D_k > 8`, the compiler calls the existing K-stream planner and
emits one runtime job with multiple chunks when the descriptor supports it.

For future `S_q > 8` or `S_k > 8`, Compiler emits one QK output-tile
descriptor per `(query_tile, key_tile)` pair. Each output-tile descriptor uses
K-stream for `D_k > 8`. This is the M/N/K tiling contract defined in
`transformer_npu_v1.md`; it must be reviewed before multi-tile Attention
lowering is coded.

中文说明：

`D_k > 8`时，一个QK输出tile内部通过K-stream累加；`S_q > 8`或`S_k > 8`
时，Compiler为每个Query tile和Key tile组合生成独立输出tile descriptor。
因此大Attention不是扩大Matrix Engine，而是由Compiler生成多个8x8输出
tile，并由Runtime按计划执行。

### 4. Plan Score Scale And Mask

Mathematical attention requires:

```text
score_scaled = score_raw / sqrt(D_k)
```

Compiler fixed-point policy:

```text
scale_multiplier = round((1 / sqrt(D_k)) * 2^scale_shift)
score_scaled = round_shift(score_raw * scale_multiplier, scale_shift)
```

The plan must record:

| Field | Meaning |
| --- | --- |
| `scale_policy` | `power_of_two`, `multiplier_shift`, or `pre_scaled_fixture` |
| `scale_multiplier` | integer multiplier when used |
| `scale_shift` | right shift amount |
| `rounding` | truncate or round-to-nearest |
| `score_q_format` | fixed-point interpretation for softmax input |

Current executable limitation:

- the softmax descriptor consumes a pre-staged row;
- first compiler integration may use `scale_policy = pre_scaled_fixture` for
  the smoke workload while still emitting target multiplier/shift metadata.

Mask planning:

| Mask kind | Compiler action |
| --- | --- |
| `none` | all lanes valid |
| `causal` | invalid when `key_position > query_position` |
| `padding` | invalid from sequence valid length |
| `tile_tail` | invalid outside logical sequence length |

Invalid lanes become `SCORE_NEG_INF`. The compiler must record mask policy even
when the first smoke workload uses `none`.

#### Canonical single-tile mask representation

Design status: Compiler planning and causal single-tile RTL execution are
implemented. The hardware uses a descriptor-referenced packed row-mask table
without a new `MASK` uop.

The first generalized lowering uses one `valid_lane_mask` integer per physical
query-tile row plus one `valid_query_mask`. For the current eight-lane tile,
bit `j` corresponds to key lane `j`:

```text
bit j = 1: lane participates in scale/mask, reduction, softmax, and PV
bit j = 0: lane is invalid and must contribute zero probability
```

Logical policies compose before hardware issue:

```text
valid(query, key) =
    key < seq_k
    and key < valid_k
    and (not causal or key <= query)
```

Here `key < seq_k` also materializes the tile-tail rule. The plan records:

| Field | Meaning |
| --- | --- |
| `mask_policy` | `none`, `causal`, `padding`, or `causal_padding` |
| `valid_k` | valid key count used by padding lowering |
| `tile_rows` / `tile_cols` | current physical tile dimensions |
| `valid_query_mask` | physical rows corresponding to logical query rows |
| `valid_lane_masks` | one integer bit mask per physical query-tile row; tail rows are zero |
| `execution_state` | `executable` only when current RTL supports the emitted mask |

For the first P3 implementation, `seq_q` and `seq_k` may be represented only
when both are at most eight. Shapes larger than one tile require Compiler v2
multi-tile lowering and are rejected. The current RTL executes full physical
eight-row plans whose rows are non-empty, including causal `S=8,D=8`.
Plans containing zero-mask physical rows remain `planned_not_executable` until
tail-row scheduling is implemented.

The Compiler owns policy composition, not the hardware mechanism:

- it may emit explicit row masks for the current single tile;
- a future multi-tile planner may omit fully invisible causal tiles;
- it must not assume a hardware mask transport that the selected target
  contract does not declare;
- arbitrary dense masks remain unsupported until their storage and movement
  contract is reviewed.

For the current `8x8` tile, Compiler output adds:

```text
row_mask_words[2]
```

It does not add primitive instructions. Existing Scale/Mask and Softmax
programs remain row-indexed and unchanged in length. Runtime jobs for both
stages reference the same row-mask buffer through descriptor `input1`.

Current executable workload status:

- `transformer_attention_prefill_s8_d8` declares `mask_policy=causal`;
- fixture generation emits the packed row-mask table and masked score golden;
- Scale/Mask and Softmax runtime jobs share the generated `row_mask` buffer;
- CPU-to-NPU RTL execution verifies causal masked scores, probabilities, and
  produced PV output.

The remaining single-tile gap is tail-query execution where physical rows have
no valid lane.

### 5. Plan Softmax

The compiler emits softmax over scaled/masked score rows:

```text
prob_q15 = attention_softmax_q15_v1(score_masked)
```

The stage records:

- clamp range;
- exp SFU contract;
- reciprocal SFU contract;
- probability output dtype `uint16_q0.15`;
- test tolerance policy.

For a full `8x8` probability matrix, runtime either launches one row/tile
descriptor per row or later uses a row-loop descriptor/command list.

### 6. Plan PV

PV computes:

```text
O_i32[S_q,D_v] = P_q15[S_q,S_k] * V_s8[S_k,D_v]
```

Compiler action:

- use `matmul_u16s8_q15_i32_tile`;
- set descriptor op to `matmul_u16s8_q15`;
- tag input0 as `uint16_q0.15`;
- tag input1 as `int8`;
- tag current output policy as `acc_shift15_truncate`.

Future larger sequence length uses K-stream chunks along `S_k`.

### 7. Emit Group Metadata

The plan includes:

```text
attention_group = attention_prefill_s8_d8
logical_op      = scaled_dot_product_attention_v1
stages          = qk, scale_mask, softmax, pv
```

PPA rules:

- stage rows report measured jobs when executable;
- scale/mask is visible even if materialized/model-only;
- parent remains `model_only` until runtime launches the compiler plan as a
  grouped attention workload.

## Buffer Planning

| Buffer | Dtype | Producer | Consumer | Current location |
| --- | --- | --- | --- | --- |
| `q_tile` | int8 | fixture/model input | QK | SRAM input0 |
| `k_t_tile` | int8 | compiler transpose/fixture | QK | SRAM input1 |
| `score_raw` | int32 | QK | scale/mask | SRAM output |
| `score_softmax_in` | int32 | scale/mask | softmax | SRAM/input window |
| `prob_q15` | uint16 Q0.15 | softmax | PV | SRAM input0 |
| `v_tile` | int8 | fixture/model input | PV | SRAM input1 |
| `o_i32` | int32 | PV | output check/store | SRAM output |

Initial allocation rules:

- buffers from different attention groups must not alias;
- `score_raw` and `prob_q15` must be separate because they have different
  dtypes and consumers;
- output buffers must not alias live input buffers;
- every buffer records element width and word packing.

## Modules To Add

Proposed files:

```text
sw/tools/npu_compiler/attention.py
sw/tools/npu_compiler/attention_plan_schema.py
```

`attention.py` owns metadata loading, attention lowering, K-stream planning for
QK/PV, runtime job plan emission, and PPA metadata. The current implementation
is a manifest-driven `S=8,D=8` plan that emits a software-sequenced runtime job
table for QK, unmasked scale/mask, softmax, and PV.

`attention_plan_schema.py` owns required fields and validation for shape, dtype,
layout, stage dependencies, and buffer lifetimes. The current schema validates
the initial plan shape and must grow before supporting larger/tiled attention.

## Verification

Compiler unit tests:

- lowering `scaled_dot_product_attention_v1` emits `qk`, `scale_mask`,
  `softmax`, `pv` in order;
- QK uses transposed K layout and records whether transpose was materialized;
- score scale metadata for `D_k=8` matches the executable fixed-scale path;
- causal mask plan marks future key lanes invalid for decode rows;
- PV uses `matmul_u16s8_q15` and `uint16_q0.15` input0;
- generated runtime job stage names match manifest attention stages;
- buffer validator rejects accidental alias between `score_raw` and `prob_q15`.

Integration tests:

- transformer fixture generation consumes `AttentionPlan`;
- existing QK/softmax/PV smoke data remains bit-compatible unless a reviewed
  numerical contract changes;
- PPA report still emits per-stage measured rows;
- parent full attention is reported as `software_group_measured_stages` once
  generated runtime jobs launch measured QK, scale/mask, softmax, and PV
  stages; full producer-to-consumer chaining remains incomplete.

## Iteration Plan

### Compiler v1.0: Plan-Only Integration

Deliverables:

- operator metadata loaded;
- `AttentionPlan` generated for current `S=8,D=8` prefill smoke;
- fixture generator consumes plan stage metadata;
- no RTL changes required.

Current status:

- manifest-driven plan/schema implemented for current `S=8,D=8` workload;
- fixture generator consumes plan metadata without changing tensor/golden
  values;
- firmware data emitter emits the generated attention runtime-job table;
- operator metadata loading remains pending.

Trigger: before adding more attention shapes or decode masks.

### Compiler v1.1: Scale/Mask Materialization

Deliverables:

- fixed-point scale multiplier/shift emitted from `D_k`;
- mask policy emitted and tested;
- golden/fixture include `score_raw`, `score_scaled`, and `score_masked`.

Trigger: before claiming full attention numerical correctness.

### Compiler v1.2: Grouped Runtime Command Planning

Deliverables:

- one grouped runtime plan for QK -> scale/mask -> softmax -> PV;
- intermediate buffer allocation in the plan;
- group-level perf scope and parent PPA row.

Trigger: before promoting `attention_prefill_s8_d8` from model-only to measured
software-grouped attention.

### Compiler v2: Larger/Tiled Attention

Deliverables:

- multi-tile QK;
- row-wise softmax across multiple key tiles;
- PV across multiple sequence chunks;
- spill/reload policy for score/probability buffers.

Trigger: when sequence length or head dimension exceeds one 8x8 tile.
