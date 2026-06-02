# Software Runtime And Compiler Attention v1

## Scope

This is the overview document for executing Transformer attention through
software-defined primitive sequencing. The detailed module designs are split by
ownership:

| Document | Owns |
| --- | --- |
| `attention_operators_v1.md` | stable attention operator names, dtypes, layouts, primitive-engine mapping, numerical contracts |
| `attention_compiler_v1.md` | lowering `scaled_dot_product_attention_v1` into QK, scale/mask, softmax, and PV stages |
| `attention_runtime_v1.md` | descriptor launch, SRAM buffers, firmware dispatch, perf/PPA capture |

The key architecture rule remains unchanged: attention v1 is not a dedicated
attention RTL macro. It is a compiler/runtime sequence over shared matrix,
vector, reduction, and SFU primitives.

## Current Executable State

The SoC can already measure three attention-related stage jobs:

| Stage | Executable op | Primitive RTL used | Status |
| --- | --- | --- | --- |
| QK | `matmul_k_stream` | matrix array | measured as int8 x int8 QK |
| Softmax | `attention_softmax_v1` | vector + reduction + SFU | measured as current simplified Q0.15 softmax |
| PV | `matmul_u16s8_q15` | shared matrix mixed mode | measured as Q0.15 probability x int8 value |

The full attention parent workload is still not a single measured runtime
descriptor execution. The current CPU firmware does consume a compiler-produced
runtime-job table for QK, softmax, and PV, so the group can be treated as
software-sequenced measured stages. Scale/mask remains materialized by fixture
data.

## Target Software Flow

```text
operator metadata
  -> compiler lowers scaled_dot_product_attention_v1
  -> compiler emits AttentionPlan
  -> firmware data generator emits tensors, buffers, programs, runtime jobs
  -> CPU runtime launches generated jobs in order
  -> wrapper/core execute shared primitive RTL
  -> perf/PPA reports stage and group results
```

The fixture generator should not be the owner of attention execution semantics.
It should become a consumer of compiler output and a producer of deterministic
test tensors/golden data.

## Attention Formula Mapping

Mathematical attention:

```text
O = softmax((Q * K^T) / sqrt(D_k) + mask) * V
```

Software/primitive mapping:

| Formula part | Operator | Current execution |
| --- | --- | --- |
| `Q * K^T` | `matmul_s8s8_i32_tile` | measured QK descriptor |
| `/ sqrt(D_k)` | `attention_score_scale_mask_v1` scale policy | currently metadata/pre-materialized bridge |
| `+ mask` | `attention_score_scale_mask_v1` mask policy | currently none/pre-materialized bridge |
| `softmax(...)` | `attention_softmax_q15_v1` | measured simplified row/tile descriptor |
| `P * V` | `matmul_u16s8_q15_i32_tile` | measured mixed matrix descriptor |

The missing software work is not another RTL attention block. The missing work
is making these boundaries explicit in operators, compiler output, runtime
buffers, generated firmware launch data, and PPA provenance.

## Near-Term Coding Order After Review

1. Add `sw/npu_core/operators/transformer_attention_v1.json`.
2. Add an attention compiler planner under `sw/tools/npu_compiler`.
3. Change Transformer fixture generation to consume the compiler plan.
4. Add generated runtime-job metadata for QK, softmax, and PV.
5. Move firmware smoke toward table-driven runtime launch.
6. Promote the parent attention PPA row from model-only to software-grouped only
   after generated runtime launches the full sequence.

## Non-Goals For The Next Patch

- no new monolithic attention RTL module;
- no command-list scheduler RTL yet;
- no claim that scale/mask is measured NPU compute until it has an executable
  stage;
- no silent numerical-contract change for softmax or PV rounding;
- no removal of existing CNN/MNIST regression paths.
