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
