# Attention Workload And PPA v1

## Scope

Attention becomes the primary Transformer workload family for the Level 0 PPA
loop. This document defines workload structure, measured/model-only boundaries,
and PPA fields for attention sequences composed from primitive engines.

## Workload Levels

| Level | Purpose | Example |
| --- | --- | --- |
| primitive | validate one engine | `softmax_exp_row`, `reduce_max_row` |
| attention stage | measure a stage | `attention_qk_s8_d8`, `attention_softmax_s8`, `attention_pv_s8_d8` |
| attention group | compare architecture choices | `attention_prefill_s8_d8` |
| decode memory view | expose KV pressure | `attention_decode_s1_ctx32_d16_kv` |

The PPA report should group attention stages under a parent attention workload,
while still preserving per-stage counters. This prevents a single total cycle
number from hiding whether bottlenecks are matrix, vector/reduction/SFU, or
memory traffic.

## Initial Workloads

### `transformer_attention_qk_s8_d8`

```text
Q: 8 x 8 int8
K: 8 x 8 int8
scores: 8 x 8 int32
```

Status target:

- executable with current matrix engine;
- measured cycles and data movement;
- exact int32 golden.

Required metadata:

```json
{
  "attention_stage": "qk",
  "logical_shape": {"seq_len": 8, "head_dim": 8},
  "matrix_shape": {"m": 8, "n": 8, "k": 8},
  "precision": {"q": "int8", "k": "int8", "scores": "int32"},
  "layout": {"k_runtime_layout": "transposed_tile"}
}
```

### `transformer_attention_softmax_s8`

```text
scores tile: 8 x 8 x int32
probability tile: 8 x 8 x Q0.15
```

Status target:

- measured through CPU-to-NPU RTL using the V1 vector/reduction/SFU primitive
  engines;
- all eight 8-element rows per descriptor in the current fixed-shape path;
- Q0.15 output checked with fixed tolerance against the bring-up golden.

Required metadata:

```json
{
  "attention_stage": "softmax",
  "rows": 8,
  "row_len": 8,
  "score_shift": 0,
  "exp_input_scale": 32,
  "probability_q": 15
}
```

### `transformer_attention_pv_s8_d8`

```text
P: 8 x 8 Q0.15 or requantized int8
V: 8 x 8 int8
O: 8 x 8 int32 or int8
```

Status target:

- measured through the shared matrix path in `matmul_u16s8_q15` mixed mode;
- `P` remains Q0.15 and is not requantized to int8;
- output is int32 after the current `>>> 15` bring-up normalization.

Required metadata:

```json
{
  "attention_stage": "pv",
  "probability_policy": "q0_15_u16_measured_mixed_matrix",
  "logical_shape": {"seq_len": 8, "head_dim": 8}
}
```

### `transformer_attention_prefill_s8_d8`

Parent group:

```text
QK stage + softmax stage + PV stage
```

The group may contain measured and model-only stage evidence. The PPA report
must label provenance per stage.

Group-state policy:

| State | Allowed evidence | Parent cycle policy |
| --- | --- | --- |
| `model_only_full_attention` | stage rows may be measured separately, but no compiler/runtime group launched them as one plan | parent cycles are model-only or null |
| `software_group_measured_stages` | compiler/runtime launched ordered stage jobs from one plan | parent cycles are sum of measured stage snapshots, with CPU overhead included or excluded by an explicit field |
| `command_list_measured_full_attention` | one descriptor/command list launched the group and internal scopes measured stages | parent cycles come from the command-list job snapshot |

If scale/mask remains materialized by fixtures or CPU-only preprocessing, the
parent can be `software_group_measured_stages` but not target full-attention
accuracy/PPA evidence.

### `transformer_attention_decode_s1_ctx32_d16`

Decode target:

```text
Q: 1 x 16
K cache: 32 x 16
V cache: 32 x 16
```

Initial status:

- model-only or current-array-compatible skinny approximation;
- KV bytes are explicit external-memory events.

## PPA Fields

Every attention workload should expose:

| Field | Meaning |
| --- | --- |
| `attention_group` | parent attention workload name |
| `attention_stage` | `qk`, `scale_mask`, `softmax`, `pv`, `kv_cache`, or `full_attention` |
| `stage_provenance` | measured CSR, standalone RTL, model-only |
| `qk_cycles` | cycles attributed to QK matrix stage |
| `scale_mask_cycles` | cycles attributed to executable score scale/mask bridge |
| `softmax_cycles` | cycles attributed to row softmax stage |
| `pv_cycles` | cycles attributed to PV stage |
| `matrix_active_cycles` | measured matrix active cycles |
| `vector_active_cycles` | measured vector active cycles |
| `reduction_active_cycles` | measured reduction active cycles |
| `sfu_active_cycles` | measured SFU active cycles |
| `effective_mac_ops` | useful MAC operations for QK/PV |
| `peak_mac_capacity` | active matrix cycles times peak MAC lanes |
| `matrix_utilization` | useful MACs divided by capacity |
| `q_bytes`, `k_bytes`, `v_bytes`, `o_bytes` | logical tensor traffic |
| `score_intermediate_bytes` | score buffer footprint/traffic |
| `probability_intermediate_bytes` | probability buffer footprint/traffic |
| `kv_read_bytes`, `kv_write_bytes` | decode KV traffic |
| `external_memory_bytes` | modeled external traffic sum |
| `runtime_overhead_cycles` | CPU/runtime launch overhead when measured; otherwise `null` |
| `group_cycle_policy` | `sum_measured_stages`, `sum_plus_runtime_overhead`, `command_list_snapshot`, or `model_only` |

Unavailable measured fields must be `null`, not zero, when a stage is
model-only. Zero is reserved for a measured event count that is actually zero.

Scale/mask bridge fields:

| Field | Meaning |
| --- | --- |
| `scale_mask_provenance` | measured NPU, CPU/materialized, fixture/materialized, or model-only |
| `scale_mask_cycles` | measured cycles only when a runtime job or command-list scope exists |
| `scale_policy` | power-of-two, multiplier-shift, or pre-scaled fixture |
| `mask_policy` | none, causal, padding, tile-tail, or model-only |

## Energy Events

Level 0 event-energy should break attention into groups:

| Event | Source |
| --- | --- |
| `int8_mac_accumulate_qk` | measured or derived QK matrix MACs |
| `int8_mac_accumulate_pv` | measured or derived PV MACs when int8 PV is used |
| `vector_lane_op` | vector active lanes, once measured |
| `reduction_element_op` | reduction row elements, once measured |
| `sfu_exp_op` | EXP invocations |
| `sfu_recip_op` | RECIP invocations |
| `data_mover_read_word` | existing measured data mover counter |
| `data_mover_write_word` | existing measured data mover counter |
| `external_memory_byte` | manifest-modeled Q/K/V/O/KV bytes |

Current `energy_model_v0.jsonc` does not yet split matrix/vector/reduction/SFU
events this way. Until it does, reports must state which attention events are
folded into existing generic coefficients.

## Current Level 0 Result

Generated with:

```text
make ppa-l0-report WORKLOAD_PROFILE=transformer
```

Report artifacts:

- `build/ppa/data/perf.json`
- `build/ppa/ppa.json`
- `build/ppa/ppa_overview.html`
- `build/ppa/cases/transformer.html`
- `build/ppa/perf.html`
- `build/ppa/power.html`
- `build/ppa/area.html`

Measured attention stages:

| Workload | Stage | Provenance | Cycles | Matrix cycles | Data mover words | Useful MACs | Matrix utilization |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `transformer_attention_qk_s8_d8` | `qk` | `measured_current_matmul_path` | 84 | 10 | 208 | 512 | 0.8 |
| `transformer_attention_softmax_s8` | `softmax` | `measured_current_softmax_path` | 67 | 0 | 32 | 0 | n/a |
| `transformer_attention_pv_s8_d8` | `pv` | `measured_mixed_matrix_path` | 82 | 10 | 208 | 512 | 0.8 |

Level 0 energy model for measured attention stages:

| Workload | Normalized energy | On-chip event contribution | Modeled external-memory contribution | External bytes | Energy per useful MAC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `transformer_attention_qk_s8_d8` | 9109.0 | 1429.0 | 7680.0 | 384 | 17.790039 |
| `transformer_attention_softmax_s8` | 7832.75 | 152.75 | 7680.0 | 384 | n/a |
| `transformer_attention_pv_s8_d8` | 6548.5 | 1428.5 | 5120.0 | 256 | 12.790039 |

`transformer_attention_softmax_s8` is measured through a hardwired
CPU-to-NPU `attention_softmax_v1` descriptor path that uses
`reduction_engine.sv`, `vector_engine.sv`, and `sfu_lut.sv` inside
`npu_v0_compute_cluster.sv`. It is still the bring-up numerical contract, because EXP and
normalization use the current coarse SFU/scale behavior, but it is no longer
the old Phase 0 Q0.8 softmax path.

`transformer_attention_pv_s8_d8` is measured through the shared matrix path in
mixed `u16(Q0.15) x s8` mode. This removes the previous int8-probability model.
The Level 0 area/energy model still uses generic MAC coefficients, so it does not
yet model the larger `16x8` multiplier cost.

The parent `transformer_attention_prefill_s8_d8` row is now
`software_group_measured_stages`: a generated runtime table launches and
buffer-chains measured QK, unmasked scale/mask, eight-row softmax, and PV
stages. It is not yet target attention evidence because masks, general tiling,
target SFU accuracy, and runtime-overhead measurement remain incomplete.

## Acceptance Gates

### Documentation gate

- Attention sequence doc explains how formulas map to primitives.
- SFU/vector/reduction docs state attention-specific requirements.
- Workload manifest has parent/stage identity.

### Model-only gate

- Python golden covers full `S=8,D=8` attention.
- PPA report can include attention model-only sections with external bytes.
- No model-only field is mislabeled as measured performance.

### Measured QK gate

- QK stage executes through current firmware/descriptor path.
- `perf-report` carries attention stage metadata.
- L0 PPA derives QK MACs from logical shape and measured cycles.

### Measured softmax gate

- vector/reduction/SFU primitive sequence executes through reviewed runtime
  path.
- row-softmax outputs match fixed-point golden tolerance.
- PPA exposes vector/reduction/SFU active cycles.

### Full attention gate

- QK, softmax, and PV are represented in one attention group.
- The report shows total group cost and stage-level cost.
- Decode KV traffic remains explicit even if no RTL KV streamer exists.
