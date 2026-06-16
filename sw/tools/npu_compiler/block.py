"""TinyLlama-derived Decoder Block planning and deterministic golden model."""

from __future__ import annotations

import math
from typing import Any

from npu_compiler.k_stream import plan_tiled_matmul
from transformer.golden import deterministic_i8_matrix, matmul_i8_i32
from transformer.micro_golden import attention_head_fixed_spec, rmsnorm_primitive_sequence


B0_SHAPE = {
    "seq_len": 8,
    "hidden": 16,
    "query_heads": 2,
    "kv_heads": 1,
    "head_dim": 8,
    "ffn_intermediate": 32,
}
MATRIX_STAGE_IDS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
VECTOR_SEGMENT_WIDTH = 8


def build_b0_block_plan(seed: int = 1, input_x: list[list[int]] | None = None, block_index: int = 0) -> dict[str, Any]:
    return _build_block_plan(seed=seed, input_x=input_x, block_index=block_index)


def build_b1_two_block_plan(seed: int = 1) -> dict[str, Any]:
    block0 = _build_block_plan(seed=seed, input_x=None, block_index=0)
    block1 = _build_block_plan(seed=seed + 101, input_x=block0["golden"]["block_output"], block_index=1)
    return {
        "name": "tinyllama_derived_b1_two_block_prefill",
        "shape": dict(B0_SHAPE),
        "blocks": [block0, block1],
        "input_binding": "block0/input_x",
        "block_boundary": {
            "producer": "block0/block_output",
            "consumer": "block1/input_x",
            "cpu_recomputation": False,
            "fixture_replacement": False,
        },
        "execution_state": "planned_not_executable",
        "block_output": block1["golden"]["block_output"],
    }


def build_b0_matrix_subgraph_workload(seed: int = 1) -> dict[str, Any]:
    """Return executable K-stream tile jobs for B0 matrix stages only."""

    plan = build_b0_block_plan(seed=seed)
    tile_jobs = []
    for stage in plan["stages"]:
        matrix_plan = stage.get("matrix_plan")
        if matrix_plan is None:
            continue
        for job in matrix_plan["output_tile_jobs"]:
            tile_jobs.append(
                {
                    "stage_id": stage["stage_id"],
                    "logical_op": stage["logical_op"],
                    "tile_job_index": len(tile_jobs),
                    "stage_tile_index": job["job_index"],
                    "m_offset": job["m_offset"],
                    "n_offset": job["n_offset"],
                    "valid_m": job["valid_m"],
                    "valid_n": job["valid_n"],
                    "k_chunks": job["k_chunks"],
                    "a_stream": [_flatten(chunk) for chunk in job["a_stream"]],
                    "b_stream": [_flatten(chunk) for chunk in job["b_stream"]],
                    "expected_c": _flatten(job["expected_c"]),
                }
            )
    return {
        "name": "transformer_tinyllama_b0_matrix_subgraph",
        "kind": "transformer_block",
        "op": "block_matmul_k_stream_group",
        "role": "transformer_block_matrix_subgraph",
        "metadata": {
            "scenario": "transformer_prefill",
            "logical_op": "tinyllama_decoder_block_matrix_subgraph",
            "block": "B0",
            "block_execution_state": "partially_executable",
            "full_block_execution_state": "planned_not_executable",
            "shape": dict(B0_SHAPE),
            "stage_count": len(plan["stages"]),
            "matrix_stage_count": len([stage for stage in plan["stages"] if "matrix_plan" in stage]),
            "tile_jobs": len(tile_jobs),
            "physical_tile_invocations": sum(
                stage["matrix_plan"]["physical_tile_invocations"]
                for stage in plan["stages"]
                if "matrix_plan" in stage
            ),
            "effective_mac_ops": sum(
                stage["matrix_plan"]["useful_mac_ops"]
                for stage in plan["stages"]
                if "matrix_plan" in stage
            ),
            "shape_class": "mixed_block_matrix_subgraph",
            "provenance": "measured_current_matmul_k_stream_path",
            "non_matrix_gap": [
                "rmsnorm_attn",
                "rope_q",
                "rope_k",
                "attention_head0",
                "attention_head1",
                "concat_heads",
                "residual_attn",
                "rmsnorm_ffn",
                "silu_gate",
                "gate_mul_up",
                "residual_ffn",
            ],
        },
        "tile_jobs": tile_jobs,
        "max_k_chunks": max(job["k_chunks"] for job in tile_jobs),
        "tile_words": 64,
    }


def build_b0_residual_vector_subgraph_workload(seed: int = 1) -> dict[str, Any]:
    """Return executable DESC_VECTOR_TILE_V1 jobs for B0 residual add stages."""

    plan = build_b0_block_plan(seed=seed)
    tile_jobs = []
    for stage in plan["stages"]:
        vector_plan = stage.get("vector_plan")
        if vector_plan is None:
            continue
        for job in vector_plan["jobs"]:
            tile_jobs.append(
                {
                    "stage_id": stage["stage_id"],
                    "logical_op": stage["logical_op"],
                    "tile_job_index": len(tile_jobs),
                    "row": job["row"],
                    "segment": job["segment"],
                    "segment_offset": job["segment_offset"],
                    "valid_lanes": job["valid_lanes"],
                    "input0": job["input0"],
                    "input1": job["input1"],
                    "expected_output": job["expected_output"],
                    "primitive_program": job["primitive_program"],
                    "ppa_theory": job["ppa_theory"],
                }
            )
    return {
        "name": "transformer_tinyllama_b0_residual_vector_subgraph",
        "kind": "transformer_block",
        "op": "block_desc_vector_tile_group",
        "role": "transformer_block_residual_vector_subgraph",
        "metadata": {
            "scenario": "transformer_prefill",
            "logical_op": "tinyllama_decoder_block_residual_vector_subgraph",
            "block": "B0",
            "block_execution_state": "partially_executable",
            "full_block_execution_state": "planned_not_executable",
            "shape": dict(B0_SHAPE),
            "stage_count": 2,
            "tile_jobs": len(tile_jobs),
            "descriptor_op": "desc_vector_tile_v1",
            "primitive_program": "VADD + HALT",
            "effective_vector_lane_ops": sum(job["valid_lanes"] for job in tile_jobs),
            "theoretical_vector_cycles": sum(job["ppa_theory"]["theoretical_vector_cycles"] for job in tile_jobs),
            "shape_class": "mixed_block_residual_vector_subgraph",
            "provenance": "measured_current_desc_vector_tile_path",
            "non_matrix_gap": [
                "rmsnorm_attn",
                "rope_q",
                "rope_k",
                "attention_head0",
                "attention_head1",
                "concat_heads",
                "rmsnorm_ffn",
                "silu_gate",
                "gate_mul_up",
            ],
        },
        "tile_jobs": tile_jobs,
        "tile_words": VECTOR_SEGMENT_WIDTH,
    }


def lower_residual_add_vector_tiles(
    lhs: list[list[int]],
    rhs: list[list[int]],
    *,
    stage_id: str = "residual_add",
    segment_width: int = VECTOR_SEGMENT_WIDTH,
) -> dict[str, Any]:
    """Lower row-wise residual add into DESC_VECTOR_TILE_V1 primitive jobs."""

    _validate_same_shape(lhs, rhs, "lhs", "rhs")
    if segment_width != VECTOR_SEGMENT_WIDTH:
        raise ValueError(f"v0 DESC_VECTOR_TILE_V1 segment width must be {VECTOR_SEGMENT_WIDTH}")
    rows = len(lhs)
    width = len(lhs[0]) if rows else 0
    if width == 0 or width % segment_width:
        raise ValueError("v0 residual add lowering requires a non-empty width that is a multiple of 8")

    jobs = []
    for row_idx in range(rows):
        for segment_idx, offset in enumerate(range(0, width, segment_width)):
            lhs_segment = lhs[row_idx][offset : offset + segment_width]
            rhs_segment = rhs[row_idx][offset : offset + segment_width]
            jobs.append(
                {
                    "stage_id": stage_id,
                    "logical_op": "residual_add",
                    "descriptor_op": "desc_vector_tile_v1",
                    "row": row_idx,
                    "segment": segment_idx,
                    "segment_offset": offset,
                    "valid_lanes": segment_width,
                    "input0": lhs_segment,
                    "input1": rhs_segment,
                    "expected_output": [_sat_i8(a + b) for a, b in zip(lhs_segment, rhs_segment)],
                    "primitive_program": [
                        {"op": "VADD", "row": 0, "segment": 0},
                        {"op": "HALT"},
                    ],
                    "ppa_theory": {
                        "useful_vector_lane_ops": segment_width,
                        "theoretical_vector_cycles": 1,
                        "primitive_passes": 1,
                    },
                }
            )
    return {
        "stage_id": stage_id,
        "logical_op": "residual_add",
        "descriptor_op": "desc_vector_tile_v1",
        "segment_width": segment_width,
        "rows": rows,
        "row_width": width,
        "segment_count": len(jobs),
        "execution_state": "compiler_lowered_not_full_block_submitted",
        "jobs": jobs,
    }


def _build_block_plan(*, seed: int, input_x: list[list[int]] | None, block_index: int) -> dict[str, Any]:
    shape = dict(B0_SHAPE)
    if input_x is None:
        input_x = deterministic_i8_matrix(shape["seq_len"], shape["hidden"], seed)
    _validate_matrix(input_x, shape["seq_len"], shape["hidden"], "input_x")
    weights = _weights(shape, seed)
    golden = _block_golden(input_x, weights, shape)
    stages = []
    for stage_id, logical_op, inputs, outputs in _stage_defs():
        stage: dict[str, Any] = {
            "stage_id": stage_id,
            "logical_op": logical_op,
            "inputs": inputs,
            "outputs": outputs,
            "execution_state": "planned_not_executable",
            "provenance": "compiler_golden_only",
        }
        if stage_id in MATRIX_STAGE_IDS:
            matrix_input, weight_name = _matrix_stage_operands(stage_id, golden, weights)
            stage["matrix_plan"] = plan_tiled_matmul(matrix_input, weights[weight_name])
            stage["provenance"] = "compiler_tiled_current_matrix_contract_not_submitted"
        elif logical_op == "vector_add":
            lhs, rhs = _vector_add_stage_operands(stage_id, golden)
            stage["vector_plan"] = lower_residual_add_vector_tiles(lhs, rhs, stage_id=stage_id)
            stage["provenance"] = "compiler_desc_vector_tile_lowered_not_submitted"
        stages.append(stage)
    return {
        "name": f"tinyllama_derived_b0_block{block_index}_prefill",
        "block_index": block_index,
        "shape": shape,
        "topology": "rmsnorm_rope_causal_gqa_swiglu_residual",
        "input_source": "previous_block_output" if input_x is not None and block_index else "deterministic_fixture",
        "execution_state": "planned_not_executable",
        "stages": stages,
        "buffers": _buffers(shape),
        "weights": weights,
        "golden": golden,
    }


def _block_golden(x: list[list[int]], weights: dict[str, list[list[int]]], shape: dict[str, int]) -> dict[str, Any]:
    norm_attn = _rmsnorm_rows(x)
    q = _requant_matmul(norm_attn, weights["wq"])
    k = _requant_matmul(norm_attn, weights["wk"])
    v = _requant_matmul(norm_attn, weights["wv"])
    q_rope = _rope_q14(q, shape["head_dim"])
    k_rope = _rope_q14(k, shape["head_dim"])
    head_outputs = []
    for head in range(shape["query_heads"]):
        q_head = _slice_cols(q_rope, head * shape["head_dim"], shape["head_dim"])
        attention = attention_head_fixed_spec(q_head, k_rope, v, mask=_causal_mask(shape["seq_len"]))
        head_outputs.append(attention["output"])
    attention_concat = _concat_cols(head_outputs)
    attention_projected = _requant_matmul(attention_concat, weights["wo"])
    residual_attn = _sat_add(x, attention_projected)
    norm_ffn = _rmsnorm_rows(residual_attn)
    gate = _requant_matmul(norm_ffn, weights["w_gate"])
    up = _requant_matmul(norm_ffn, weights["w_up"])
    silu_gate = [[_silu_i8(value) for value in row] for row in gate]
    gated = [[_sat_i8((g * u) >> 4) for g, u in zip(g_row, u_row)] for g_row, u_row in zip(silu_gate, up)]
    down = _requant_matmul(gated, weights["w_down"])
    output = _sat_add(residual_attn, down)
    return {
        "input_x": x,
        "rmsnorm_attn": norm_attn,
        "q": q,
        "k": k,
        "v": v,
        "rope_q": q_rope,
        "rope_k": k_rope,
        "attention_heads": head_outputs,
        "attention_concat": attention_concat,
        "attention_projected": attention_projected,
        "residual_attn": residual_attn,
        "rmsnorm_ffn": norm_ffn,
        "gate": gate,
        "up": up,
        "silu_gate": silu_gate,
        "gate_mul_up": gated,
        "down": down,
        "block_output": output,
    }


def _weights(shape: dict[str, int], seed: int) -> dict[str, list[list[int]]]:
    h = shape["hidden"]
    d = shape["head_dim"]
    f = shape["ffn_intermediate"]
    return {
        "wq": deterministic_i8_matrix(h, h, seed + 11),
        "wk": deterministic_i8_matrix(h, d, seed + 12),
        "wv": deterministic_i8_matrix(h, d, seed + 13),
        "wo": deterministic_i8_matrix(h, h, seed + 14),
        "w_gate": deterministic_i8_matrix(h, f, seed + 15),
        "w_up": deterministic_i8_matrix(h, f, seed + 16),
        "w_down": deterministic_i8_matrix(f, h, seed + 17),
    }


def _requant_matmul(a: list[list[int]], b: list[list[int]], shift: int = 6) -> list[list[int]]:
    return [[_sat_i8(value >> shift) for value in row] for row in matmul_i8_i32(a, b)]


def _rmsnorm_rows(rows: list[list[int]]) -> list[list[int]]:
    return [[_sat_i8(value) for value in rmsnorm_primitive_sequence(row)["scaled"]] for row in rows]


def _rope_q14(rows: list[list[int]], head_dim: int) -> list[list[int]]:
    output = []
    for position, row in enumerate(rows):
        rotated = list(row)
        for head_base in range(0, len(row), head_dim):
            for pair in range(0, head_dim, 2):
                angle = position / float(1 << pair)
                cos_q14 = int(round(math.cos(angle) * (1 << 14)))
                sin_q14 = int(round(math.sin(angle) * (1 << 14)))
                x0 = row[head_base + pair]
                x1 = row[head_base + pair + 1]
                rotated[head_base + pair] = _sat_i8((x0 * cos_q14 - x1 * sin_q14) >> 14)
                rotated[head_base + pair + 1] = _sat_i8((x0 * sin_q14 + x1 * cos_q14) >> 14)
        output.append(rotated)
    return output


def _silu_i8(value: int) -> int:
    return _sat_i8(int(round(value / (1.0 + math.exp(-value / 16.0)))))


def _sat_add(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[_sat_i8(x + y) for x, y in zip(a_row, b_row)] for a_row, b_row in zip(a, b)]


def _sat_i8(value: int) -> int:
    return min(max(int(value), -128), 127)


def _slice_cols(matrix: list[list[int]], offset: int, width: int) -> list[list[int]]:
    return [row[offset : offset + width] for row in matrix]


def _concat_cols(matrices: list[list[list[int]]]) -> list[list[int]]:
    return [sum((matrix[row] for matrix in matrices), []) for row in range(len(matrices[0]))]


def _causal_mask(seq_len: int) -> list[list[bool]]:
    return [[key <= query for key in range(seq_len)] for query in range(seq_len)]


def _matrix_stage_operands(
    stage_id: str,
    golden: dict[str, Any],
    weights: dict[str, list[list[int]]],
) -> tuple[list[list[int]], str]:
    mapping = {
        "q_proj": ("rmsnorm_attn", "wq"),
        "k_proj": ("rmsnorm_attn", "wk"),
        "v_proj": ("rmsnorm_attn", "wv"),
        "o_proj": ("attention_concat", "wo"),
        "gate_proj": ("rmsnorm_ffn", "w_gate"),
        "up_proj": ("rmsnorm_ffn", "w_up"),
        "down_proj": ("gate_mul_up", "w_down"),
    }
    input_name, weight_name = mapping[stage_id]
    return golden[input_name], weight_name


def _vector_add_stage_operands(stage_id: str, golden: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    mapping = {
        "residual_attn": ("input_x", "attention_projected"),
        "residual_ffn": ("residual_attn", "down"),
    }
    if stage_id not in mapping:
        raise ValueError(f"unsupported vector add stage {stage_id!r}")
    lhs_name, rhs_name = mapping[stage_id]
    return golden[lhs_name], golden[rhs_name]


def _stage_defs() -> list[tuple[str, str, list[str], list[str]]]:
    return [
        ("rmsnorm_attn", "rmsnorm", ["input_x"], ["norm_attn"]),
        ("q_proj", "matmul", ["norm_attn", "wq"], ["q"]),
        ("k_proj", "matmul", ["norm_attn", "wk"], ["k"]),
        ("v_proj", "matmul", ["norm_attn", "wv"], ["v"]),
        ("rope_q", "rope", ["q"], ["rope_q"]),
        ("rope_k", "rope", ["k"], ["rope_k"]),
        ("attention_head0", "causal_attention_gqa", ["rope_q", "rope_k", "v"], ["head0"]),
        ("attention_head1", "causal_attention_gqa", ["rope_q", "rope_k", "v"], ["head1"]),
        ("concat_heads", "concat", ["head0", "head1"], ["attention_concat"]),
        ("o_proj", "matmul", ["attention_concat", "wo"], ["attention_projected"]),
        ("residual_attn", "vector_add", ["input_x", "attention_projected"], ["residual_attn"]),
        ("rmsnorm_ffn", "rmsnorm", ["residual_attn"], ["norm_ffn"]),
        ("gate_proj", "matmul", ["norm_ffn", "w_gate"], ["gate"]),
        ("up_proj", "matmul", ["norm_ffn", "w_up"], ["up"]),
        ("silu_gate", "silu", ["gate"], ["silu_gate"]),
        ("gate_mul_up", "vector_mul", ["silu_gate", "up"], ["gated"]),
        ("down_proj", "matmul", ["gated", "w_down"], ["down"]),
        ("residual_ffn", "vector_add", ["residual_attn", "down"], ["block_output"]),
    ]


def _buffers(shape: dict[str, int]) -> list[dict[str, Any]]:
    s, h, d, f = shape["seq_len"], shape["hidden"], shape["head_dim"], shape["ffn_intermediate"]
    return [
        {"name": "input_x", "shape": [s, h], "dtype": "int8"},
        {"name": "q", "shape": [s, h], "dtype": "int8"},
        {"name": "k", "shape": [s, d], "dtype": "int8"},
        {"name": "v", "shape": [s, d], "dtype": "int8"},
        {"name": "attention_concat", "shape": [s, h], "dtype": "int8"},
        {"name": "ffn_intermediate", "shape": [s, f], "dtype": "int8"},
        {"name": "block_output", "shape": [s, h], "dtype": "int8"},
    ]


def _validate_matrix(matrix: list[list[int]], rows: int, cols: int, name: str) -> None:
    if len(matrix) != rows or any(len(row) != cols for row in matrix):
        raise ValueError(f"{name} must have shape {rows}x{cols}")


def _validate_same_shape(lhs: list[list[int]], rhs: list[list[int]], lhs_name: str, rhs_name: str) -> None:
    if len(lhs) != len(rhs):
        raise ValueError(f"{lhs_name}/{rhs_name} row count mismatch: {len(lhs)} != {len(rhs)}")
    for row_idx, (lhs_row, rhs_row) in enumerate(zip(lhs, rhs)):
        if len(lhs_row) != len(rhs_row):
            raise ValueError(
                f"{lhs_name}/{rhs_name} row {row_idx} width mismatch: "
                f"{len(lhs_row)} != {len(rhs_row)}"
            )


def _flatten(value: list[Any]) -> list[int]:
    out = []
    for item in value:
        if isinstance(item, list):
            out.extend(_flatten(item))
        else:
            out.append(int(item))
    return out
