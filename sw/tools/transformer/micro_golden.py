from __future__ import annotations

import math

SOFTMAX_INPUT_SCALE = 32
SOFTMAX_CLAMP_MIN = -8 * SOFTMAX_INPUT_SCALE
SOFTMAX_CLAMP_MAX = 0
SOFTMAX_EXP_Q = 15
SFU_EXP_Q15_SEGMENTS = [32767, 12055, 4435, 1632, 600, 221, 81, 30, 11]


def classify_matrix_shape(m: int, n: int, k: int) -> str:
    if m == 1 or n == 1:
        return "gemv"
    if m <= 8 or n <= 8:
        return "skinny_gemm"
    return "full_tile_gemm"


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
