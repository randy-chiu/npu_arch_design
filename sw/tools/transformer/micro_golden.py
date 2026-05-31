from __future__ import annotations

import math

SOFTMAX_INPUT_SCALE = 32
SOFTMAX_CLAMP_MIN = -8 * SOFTMAX_INPUT_SCALE
SOFTMAX_CLAMP_MAX = 0
SOFTMAX_EXP_Q = 15
SFU_EXP_Q15_SEGMENTS = [32767, 12055, 4435, 1632, 600, 221, 81, 30, 11]
PROB_ONE_Q15 = (1 << SOFTMAX_EXP_Q) - 1
RECIP_Q = 24
ATTENTION_NUMERICAL_CONTRACT_V1 = "attention_numerical_v1_q15_prob_q24_recip_lut257"
ATTENTION_BRINGUP_CONTRACT_V0 = "attention_bringup_v0_shift_scale_sfu9seg"


def classify_matrix_shape(m: int, n: int, k: int) -> str:
    if m == 1 or n == 1:
        return "gemv"
    if m <= 8 or n <= 8:
        return "skinny_gemm"
    return "full_tile_gemm"


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    if not matrix:
        raise ValueError("matrix must be non-empty")
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("matrix rows must have a consistent length")
    return [[int(matrix[row][col]) for row in range(len(matrix))] for col in range(cols)]


def attention_qk_scores_i8_i32(q: list[list[int]], k: list[list[int]]) -> list[list[int]]:
    if not q or not k:
        raise ValueError("Q and K must be non-empty")
    head_dim = len(q[0])
    if any(len(row) != head_dim for row in q):
        raise ValueError("Q rows must have a consistent head dimension")
    if any(len(row) != head_dim for row in k):
        raise ValueError("K rows must match Q head dimension")
    return [
        [
            sum(int(q_row[d]) * int(k_row[d]) for d in range(head_dim))
            for k_row in k
        ]
        for q_row in q
    ]


def scale_scores_fixed_multiplier(
    scores: list[list[int]],
    head_dim: int,
    shift: int = 15,
) -> dict[str, int | list[list[int]]]:
    if head_dim <= 0 or shift < 0:
        raise ValueError("invalid score scale parameters")
    multiplier = int(round((1.0 / math.sqrt(float(head_dim))) * (1 << shift)))
    rounded_offset = 0 if shift == 0 else 1 << (shift - 1)
    scaled = []
    for row in scores:
        scaled_row = []
        for value in row:
            product = int(value) * multiplier
            if shift == 0:
                scaled_value = product
            elif product >= 0:
                scaled_value = (product + rounded_offset) >> shift
            else:
                scaled_value = -(((-product) + rounded_offset) >> shift)
            scaled_row.append(int(scaled_value))
        scaled.append(scaled_row)
    return {"policy": "fixed_multiplier_shift", "multiplier": multiplier, "shift": shift, "scaled": scaled}


def scale_scores_power_of_two(scores: list[list[int]], shift: int) -> dict[str, int | list[list[int]]]:
    if shift < 0:
        raise ValueError("shift must be non-negative")
    return {
        "policy": "power_of_two_shift",
        "shift": shift,
        "scaled": [[int(value) >> shift for value in row] for row in scores],
    }


def apply_attention_mask(
    scores: list[list[int]],
    mask: list[list[bool]] | None = None,
    neg_inf: int = -256,
) -> list[list[int]]:
    if mask is None:
        return [[int(value) for value in row] for row in scores]
    if len(mask) != len(scores):
        raise ValueError("mask row count must match scores")
    masked = []
    for score_row, mask_row in zip(scores, mask):
        if len(mask_row) != len(score_row):
            raise ValueError("mask row width must match scores")
        masked.append([int(value) if visible else int(neg_inf) for value, visible in zip(score_row, mask_row)])
    return masked


def causal_mask(seq_len: int) -> list[list[bool]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    return [[key <= query for key in range(seq_len)] for query in range(seq_len)]


def softmax_reference_row_q15(row: list[int], input_scale: int = SOFTMAX_INPUT_SCALE) -> list[int]:
    """Algorithm reference model using Python floating-point exp."""
    if not row:
        raise ValueError("softmax row must be non-empty")
    max_value = max(int(value) for value in row)
    exp_values = []
    for value in row:
        shifted = int(value) - max_value
        clamped = min(max(shifted, SOFTMAX_CLAMP_MIN), SOFTMAX_CLAMP_MAX)
        exp_real = math.exp(clamped / float(input_scale))
        exp_values.append(int(round(exp_real * (1 << SOFTMAX_EXP_Q))))
    total = sum(exp_values)
    if total == 0:
        return [0 for _ in exp_values]
    return [int(round(value * ((1 << SOFTMAX_EXP_Q) - 1) / total)) for value in exp_values]


def softmax_fixed_spec_row_q15(row: list[int], input_scale: int = SOFTMAX_INPUT_SCALE) -> list[int]:
    """Fixed spec model for transformer_numerical_v1.md Q0.15 softmax."""
    return softmax_reference_row_q15(row, input_scale)


def sfu_exp_rtl_model_q15(value: int) -> int:
    """Current RTL bring-up EXP model: 9-segment coarse Q0.15 LUT."""
    clamped = min(max(int(value), SOFTMAX_CLAMP_MIN), SOFTMAX_CLAMP_MAX)
    segment = (-clamped + 16) // 32
    if segment < 0:
        segment = 0
    if segment >= len(SFU_EXP_Q15_SEGMENTS):
        segment = len(SFU_EXP_Q15_SEGMENTS) - 1
    return SFU_EXP_Q15_SEGMENTS[segment]


def sfu_recip_rtl_model_q24(value: int) -> int:
    """Current RTL bring-up reciprocal model: integer division to Q24."""
    return 0 if int(value) == 0 else (1 << 24) // int(value)


def sfu_rsqrt_rtl_model_q24(value: int) -> int:
    """Current RTL bring-up rsqrt model: isqrt followed by integer division."""
    root = math.isqrt(int(value))
    return 0 if root == 0 else (1 << 24) // root


def softmax_rtl_model_row_q15(row: list[int]) -> dict[str, list[int] | int]:
    """RTL-like primitive sequence matched to current bring-up RTL."""
    if not row:
        raise ValueError("softmax row must be non-empty")
    max_value = max(int(value) for value in row)
    shifted = [int(value) - max_value for value in row]
    clamped = [min(max(value, SOFTMAX_CLAMP_MIN), SOFTMAX_CLAMP_MAX) for value in shifted]
    exp_q15 = [sfu_exp_rtl_model_q15(value) for value in clamped]
    exp_sum = sum(exp_q15)
    reciprocal = sfu_recip_rtl_model_q24(exp_sum)
    # Current standalone RTL uses a compact reciprocal value from SFU_RECIP.
    # Shifting by 9 maps exp_q15 * recip_q24 back near Q0.15 for the directed
    # primitive sequence; this is a v1 bring-up approximation, not final
    # softmax numerical policy.
    output_q15 = [(value * reciprocal) >> 9 for value in exp_q15]
    return {
        "max": max_value,
        "shifted": shifted,
        "clamped": clamped,
        "exp_q15": exp_q15,
        "sum": exp_sum,
        "recip_q24": reciprocal,
        "output_q15": output_q15,
    }


def softmax_attention_fixed_spec_row_q15(row: list[int]) -> dict[str, list[int] | int | str]:
    """Target fixed-spec row softmax using Q0.15 EXP and Q0.24 reciprocal."""
    if not row:
        raise ValueError("softmax row must be non-empty")
    max_value = max(int(value) for value in row)
    shifted = [int(value) - max_value for value in row]
    clamped = [min(max(value, SOFTMAX_CLAMP_MIN), SOFTMAX_CLAMP_MAX) for value in shifted]
    exp_q15 = [
        int(round(math.exp(value / float(SOFTMAX_INPUT_SCALE)) * PROB_ONE_Q15))
        for value in clamped
    ]
    exp_sum = sum(exp_q15)
    recip_q24 = 0 if exp_sum == 0 else (1 << RECIP_Q) // exp_sum
    output_q15 = [
        min(
            max(
                int(round((value * recip_q24 * PROB_ONE_Q15) / float(1 << RECIP_Q))),
                0,
            ),
            PROB_ONE_Q15,
        )
        for value in exp_q15
    ]
    return {
        "contract": ATTENTION_NUMERICAL_CONTRACT_V1,
        "max": max_value,
        "shifted": shifted,
        "clamped": clamped,
        "exp_q15": exp_q15,
        "sum": exp_sum,
        "recip_q24": recip_q24,
        "output_q15": output_q15,
    }


def attention_softmax_fixed_spec_q15(scores: list[list[int]]) -> dict[str, list]:
    rows = [softmax_attention_fixed_spec_row_q15(row) for row in scores]
    return {
        "contract": ATTENTION_NUMERICAL_CONTRACT_V1,
        "rows": rows,
        "prob_q15": [row["output_q15"] for row in rows],
    }


def attention_pv_q15_i8_i32(prob_q15: list[list[int]], v: list[list[int]]) -> list[list[int]]:
    if not prob_q15 or not v:
        raise ValueError("probability and V matrices must be non-empty")
    seq_len = len(prob_q15[0])
    if any(len(row) != seq_len for row in prob_q15):
        raise ValueError("probability rows must have a consistent sequence length")
    if len(v) != seq_len:
        raise ValueError("V row count must match probability column count")
    head_dim = len(v[0])
    if any(len(row) != head_dim for row in v):
        raise ValueError("V rows must have a consistent head dimension")
    output = []
    for prob_row in prob_q15:
        out_row = []
        for dim in range(head_dim):
            acc = sum(int(prob_row[j]) * int(v[j][dim]) for j in range(seq_len))
            if acc >= 0:
                out = (acc + (PROB_ONE_Q15 // 2)) // PROB_ONE_Q15
            else:
                out = -(((-acc) + (PROB_ONE_Q15 // 2)) // PROB_ONE_Q15)
            out_row.append(int(out))
        output.append(out_row)
    return output


def attention_head_fixed_spec(
    q: list[list[int]],
    k: list[list[int]],
    v: list[list[int]],
    *,
    score_scale_shift: int = 15,
    mask: list[list[bool]] | None = None,
) -> dict[str, object]:
    scores = attention_qk_scores_i8_i32(q, k)
    scaled_info = scale_scores_fixed_multiplier(scores, len(q[0]), shift=score_scale_shift)
    scaled = scaled_info["scaled"]
    assert isinstance(scaled, list)
    masked = apply_attention_mask(scaled, mask)
    softmax = attention_softmax_fixed_spec_q15(masked)
    output = attention_pv_q15_i8_i32(softmax["prob_q15"], v)
    return {
        "contract": ATTENTION_NUMERICAL_CONTRACT_V1,
        "scores": scores,
        "score_scale": {
            "policy": scaled_info["policy"],
            "multiplier": scaled_info["multiplier"],
            "shift": scaled_info["shift"],
        },
        "scaled_scores": scaled,
        "masked_scores": masked,
        "softmax": softmax,
        "output": output,
    }


def softmax_row_fixed_q15(row: list[int], input_scale: int = SOFTMAX_INPUT_SCALE) -> list[int]:
    return softmax_fixed_spec_row_q15(row, input_scale)


def sfu_exp_lut_q15(value: int) -> int:
    return sfu_exp_rtl_model_q15(value)


def recip_q24(value: int) -> int:
    return sfu_recip_rtl_model_q24(value)


def rsqrt_q24(value: int) -> int:
    return sfu_rsqrt_rtl_model_q24(value)


def softmax_row_primitive_lut_q15(row: list[int]) -> dict[str, list[int] | int]:
    return softmax_rtl_model_row_q15(row)


def rmsnorm_primitive_sequence(row: list[int], shift: int = 20) -> dict[str, list[int] | int]:
    if not row:
        raise ValueError("rmsnorm row must be non-empty")
    sumsq = sum(int(value) * int(value) for value in row)
    inv_rms_q24 = sfu_rsqrt_rtl_model_q24(sumsq)
    scaled = [(int(value) * inv_rms_q24) >> shift for value in row]
    return {
        "sumsq": sumsq,
        "rsqrt_q24": inv_rms_q24,
        "shift": shift,
        "scaled": scaled,
    }


def rmsnorm_row_reference(
    row: list[int],
    weight: list[int],
    eps: float = 1.0e-5,
    output_scale: int = 128,
) -> list[int]:
    if not row or len(row) != len(weight):
        raise ValueError("rmsnorm row and weight must be non-empty and equal length")
    mean_square = sum(float(value) * float(value) for value in row) / len(row)
    inv_rms = 1.0 / math.sqrt(mean_square + eps)
    result = []
    for value, scale in zip(row, weight):
        normalized = float(value) * inv_rms * (float(scale) / output_scale)
        result.append(int(round(normalized * output_scale)))
    return result


def kv_cache_bytes(seq_len: int, heads: int, head_dim: int, bytes_per_elem: int = 1) -> dict[str, int]:
    if seq_len < 0 or heads <= 0 or head_dim <= 0 or bytes_per_elem <= 0:
        raise ValueError("invalid KV cache shape")
    per_token_kv = 2 * heads * head_dim * bytes_per_elem
    return {
        "kv_cache_read_bytes": seq_len * per_token_kv,
        "kv_cache_write_bytes": per_token_kv,
        "bytes_per_token": (seq_len + 1) * per_token_kv,
    }
