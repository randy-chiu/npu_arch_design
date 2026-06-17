"""TinyLlama-derived Decoder Block planning and deterministic golden model."""

from __future__ import annotations

import math
import json
import re
from pathlib import Path
from typing import Any

from npu_compiler.k_stream import plan_tiled_matmul
from transformer.golden import deterministic_i8_matrix, matmul_i8_i32
from transformer.micro_golden import (
    SOFTMAX_NEG_INF,
    attention_head_fixed_spec,
    attention_qk_scores_i8_i32,
    rmsnorm_primitive_sequence,
    scale_scores_fixed_multiplier,
    softmax_row_primitive_lut_q15,
    transpose,
)


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
B0_WORKLOAD_SPEC_PATH = Path("workloads/transformer/block/tinyllama_derived_b0.jsonc")
B1_WORKLOAD_SPEC_PATH = Path("workloads/transformer/block/tinyllama_derived_b1.jsonc")


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


def load_block_workload_spec(path: Path | str) -> dict[str, Any]:
    spec_path = Path(path)
    text = spec_path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    return json.loads(text)


def validate_b0_workload_contract(
    spec: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> None:
    """Validate that the B0 workload declaration matches the compiler planner.

    This is intentionally a contract check, not a general graph importer.
    B0 remains a fixed architecture-driver workload, but the checked-in
    workload declaration must not drift away from the BlockPlan generator.
    """

    if spec is None:
        spec = load_block_workload_spec(B0_WORKLOAD_SPEC_PATH)
    if plan is None:
        plan = build_b0_block_plan()
    _validate_workload_shape(spec, plan["shape"], expected_blocks=1)
    _require_equal(spec.get("planner"), "npu_compiler.block.build_b0_block_plan", "B0 planner")
    _require_equal(spec.get("execution_state"), plan["execution_state"], "B0 execution_state")
    _require_equal(spec.get("topology"), _aggregate_stage_topology(plan["stages"]), "B0 topology")


def validate_b1_workload_contract(
    spec: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> None:
    if spec is None:
        spec = load_block_workload_spec(B1_WORKLOAD_SPEC_PATH)
    if plan is None:
        plan = build_b1_two_block_plan()
    _validate_workload_shape(spec, plan["shape"], expected_blocks=2)
    _require_equal(spec.get("planner"), "npu_compiler.block.build_b1_two_block_plan", "B1 planner")
    _require_equal(spec.get("execution_state"), plan["execution_state"], "B1 execution_state")
    _require_equal(spec.get("block_boundary"), "block0/block_output -> block1/input_x", "B1 block_boundary")
    _require_equal(bool(spec.get("cpu_recomputation")), False, "B1 cpu_recomputation")
    _require_equal(bool(spec.get("fixture_replacement")), False, "B1 fixture_replacement")


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


def build_b0_rmsnorm_vector_subgraph_workload(seed: int = 1) -> dict[str, Any]:
    """Return executable DESC_VECTOR_TILE_V1 jobs for B0 RMSNorm stages."""

    plan = build_b0_block_plan(seed=seed)
    tile_jobs = []
    for stage in plan["stages"]:
        rmsnorm_plan = stage.get("rmsnorm_plan")
        if rmsnorm_plan is None:
            continue
        for job in rmsnorm_plan["jobs"]:
            tile_jobs.append(
                {
                    "stage_id": stage["stage_id"],
                    "logical_op": stage["logical_op"],
                    "tile_job_index": len(tile_jobs),
                    "row": job["row"],
                    "segment": job["segment"],
                    "segment_offset": job["segment_offset"],
                    "program_select": job["program_select"],
                    "input0": job["input0"],
                    "input1": job["input1"],
                    "expected_output": job["expected_output"],
                    "ppa_theory": job["ppa_theory"],
                }
            )
    return {
        "name": "transformer_tinyllama_b0_rmsnorm_vector_subgraph",
        "kind": "transformer_block",
        "op": "block_desc_vector_tile_group",
        "role": "transformer_block_rmsnorm_vector_subgraph",
        "metadata": {
            "scenario": "transformer_prefill",
            "logical_op": "tinyllama_decoder_block_rmsnorm_vector_subgraph",
            "block": "B0",
            "block_execution_state": "partially_executable",
            "full_block_execution_state": "planned_not_executable",
            "shape": dict(B0_SHAPE),
            "stage_count": 2,
            "tile_jobs": len(tile_jobs),
            "descriptor_op": "desc_vector_tile_v1",
            "primitive_program": "SUMSQ_SRC0 + SUMSQ_SRC1 + RSQRT + SCALE_SRCx + HALT",
            "effective_vector_lane_ops": sum(job["ppa_theory"]["useful_vector_lane_ops"] for job in tile_jobs),
            "theoretical_reduction_cycles": sum(job["ppa_theory"]["theoretical_reduction_cycles"] for job in tile_jobs),
            "theoretical_sfu_cycles": sum(job["ppa_theory"]["theoretical_sfu_cycles"] for job in tile_jobs),
            "theoretical_vector_cycles": sum(job["ppa_theory"]["theoretical_vector_cycles"] for job in tile_jobs),
            "shape_class": "mixed_block_rmsnorm_vector_subgraph",
            "provenance": "measured_current_desc_vector_tile_path",
            "non_matrix_gap": [
                "rope_q",
                "rope_k",
                "attention_head0",
                "attention_head1",
                "concat_heads",
                "silu_gate",
                "gate_mul_up",
            ],
        },
        "tile_jobs": tile_jobs,
        "tile_words": VECTOR_SEGMENT_WIDTH,
    }


def build_b0_gate_mul_vector_subgraph_workload(seed: int = 1) -> dict[str, Any]:
    """Return executable DESC_VECTOR_TILE_V1 jobs for B0 gate multiply."""

    plan = build_b0_block_plan(seed=seed)
    tile_jobs = []
    for stage in plan["stages"]:
        gate_mul_plan = stage.get("gate_mul_plan")
        if gate_mul_plan is None:
            continue
        for job in gate_mul_plan["jobs"]:
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
        "name": "transformer_tinyllama_b0_gate_mul_vector_subgraph",
        "kind": "transformer_block",
        "op": "block_desc_vector_tile_group",
        "role": "transformer_block_gate_mul_vector_subgraph",
        "metadata": {
            "scenario": "transformer_prefill",
            "logical_op": "tinyllama_decoder_block_gate_mul_vector_subgraph",
            "block": "B0",
            "block_execution_state": "partially_executable",
            "full_block_execution_state": "planned_not_executable",
            "shape": dict(B0_SHAPE),
            "stage_count": 1,
            "tile_jobs": len(tile_jobs),
            "descriptor_op": "desc_vector_tile_v1",
            "primitive_program": "VMUL + VREQUANT(INT8_SHIFT4_CLAMP) + HALT",
            "effective_vector_lane_ops": sum(job["valid_lanes"] for job in tile_jobs),
            "theoretical_vector_cycles": sum(job["ppa_theory"]["theoretical_vector_cycles"] for job in tile_jobs),
            "shape_class": "mixed_block_gate_mul_vector_subgraph",
            "provenance": "measured_current_desc_vector_tile_path",
            "explicit_gap": [
                "silu_gate is still compiler golden input for this subgraph",
            ],
            "non_matrix_gap": [
                "rope_q",
                "rope_k",
                "attention_head0",
                "attention_head1",
                "concat_heads",
                "silu_gate",
            ],
        },
        "tile_jobs": tile_jobs,
        "tile_words": VECTOR_SEGMENT_WIDTH,
    }


def build_b0_attention_subgraph_workload(seed: int = 1) -> dict[str, Any]:
    """Return executable stage jobs for the two B0 causal Attention heads."""

    plan = build_b0_block_plan(seed=seed)
    shape = plan["shape"]
    golden = plan["golden"]
    valid_lane_masks = [sum(1 << key for key in range(query + 1)) for query in range(shape["seq_len"])]
    row_mask_words = [
        sum(valid_lane_masks[row + offset] << (offset * 8) for offset in range(4))
        for row in range(0, shape["seq_len"], 4)
    ]
    heads = []
    stage_jobs = []
    for head in range(shape["query_heads"]):
        q_head = _slice_cols(golden["rope_q"], head * shape["head_dim"], shape["head_dim"])
        k_head = golden["rope_k"]
        v_head = golden["v"]
        scores = attention_qk_scores_i8_i32(q_head, k_head)
        scaled_info = scale_scores_fixed_multiplier(scores, shape["head_dim"], shift=15)
        scaled = scaled_info["scaled"]
        assert isinstance(scaled, list)
        masked_scores = [
            [
                value if valid_lane_masks[row_idx] & (1 << lane) else SOFTMAX_NEG_INF
                for lane, value in enumerate(scaled_row)
            ]
            for row_idx, scaled_row in enumerate(scaled)
        ]
        probabilities = [
            softmax_row_primitive_lut_q15(
                row,
                [bool(valid_lane_masks[row_idx] & (1 << lane)) for lane in range(shape["seq_len"])],
            )["output_q15"]
            for row_idx, row in enumerate(masked_scores)
        ]
        pv_output = _attention_pv_q15_i8_shift15(probabilities, v_head)
        head_jobs = [
            {
                "stage": "qk",
                "op": "matmul_k_stream",
                "head": head,
                "a_stream": [_flatten(q_head)],
                "b_stream": [_flatten(transpose(k_head))],
                "expected_output": _flatten(scores),
                "k_chunks": 1,
            },
            {
                "stage": "scale_mask",
                "op": "attention_scale_mask_v1",
                "head": head,
                "expected_output": _flatten(masked_scores),
            },
            {
                "stage": "softmax",
                "op": "attention_softmax_v1",
                "head": head,
                "expected_output": _flatten(probabilities),
            },
            {
                "stage": "pv",
                "op": "matmul_u16s8_q15",
                "head": head,
                "a_stream": [_flatten(probabilities)],
                "b_stream": [_flatten(v_head)],
                "expected_output": _flatten(pv_output),
                "k_chunks": 1,
            },
        ]
        for job in head_jobs:
            job["stage_job_index"] = len(stage_jobs)
            stage_jobs.append(job)
        heads.append(
            {
                "head": head,
                "q_input_source": "compiler_golden_rope_q",
                "k_input_source": "compiler_golden_rope_k",
                "v_input_source": "compiler_golden_v_projection",
                "stages": [job["stage"] for job in head_jobs],
            }
        )
    return {
        "name": "transformer_tinyllama_b0_attention_subgraph",
        "kind": "transformer_block",
        "op": "block_attention_head_group",
        "role": "transformer_block_attention_subgraph",
        "metadata": {
            "scenario": "transformer_prefill",
            "logical_op": "tinyllama_decoder_block_attention_subgraph",
            "block": "B0",
            "block_execution_state": "partially_executable",
            "full_block_execution_state": "planned_not_executable",
            "shape": dict(B0_SHAPE),
            "stage_count": 4,
            "heads": shape["query_heads"],
            "stage_jobs": len(stage_jobs),
            "attention_group": "tinyllama_b0_attention",
            "attention_stage": "block_attention_subgraph",
            "shape_class": "mixed_block_attention_subgraph",
            "provenance": "measured_current_attention_stage_paths",
            "effective_mac_ops": shape["query_heads"] * 2 * shape["seq_len"] * shape["seq_len"] * shape["head_dim"],
            "theoretical_matrix_cycles": shape["query_heads"] * 2 * 8,
            "theoretical_scale_mask_vector_cycles": shape["query_heads"] * 8,
            "theoretical_softmax_cycles": shape["query_heads"] * 112,
            "valid_lane_masks": valid_lane_masks,
            "row_mask_words": row_mask_words,
            "explicit_gap": [
                "rope_q and rope_k are compiler golden inputs for this subgraph",
                "softmax uses current RTL bring-up LUT contract, not final BlockPlan fixed-spec golden",
            ],
            "non_matrix_gap": [
                "rope_q",
                "rope_k",
                "concat_heads",
                "silu_gate",
            ],
        },
        "heads": heads,
        "stage_jobs": stage_jobs,
        "row_mask_words": row_mask_words,
        "tile_words": 64,
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


def lower_gate_mul_vector_tiles(
    silu_gate: list[list[int]],
    up: list[list[int]],
    *,
    stage_id: str = "gate_mul_up",
    segment_width: int = VECTOR_SEGMENT_WIDTH,
) -> dict[str, Any]:
    """Lower B0 SwiGLU gate multiply into VMUL/VREQUANT primitive jobs."""

    _validate_same_shape(silu_gate, up, "silu_gate", "up")
    if segment_width != VECTOR_SEGMENT_WIDTH:
        raise ValueError(f"v0 gate multiply segment width must be {VECTOR_SEGMENT_WIDTH}")
    rows = len(silu_gate)
    width = len(silu_gate[0]) if rows else 0
    if width == 0 or width % segment_width:
        raise ValueError("gate multiply lowering requires a non-empty width that is a multiple of 8")

    jobs = []
    for row_idx in range(rows):
        for segment_idx, offset in enumerate(range(0, width, segment_width)):
            lhs_segment = silu_gate[row_idx][offset : offset + segment_width]
            rhs_segment = up[row_idx][offset : offset + segment_width]
            jobs.append(
                {
                    "stage_id": stage_id,
                    "logical_op": "vector_mul_requant",
                    "descriptor_op": "desc_vector_tile_v1",
                    "row": row_idx,
                    "segment": segment_idx,
                    "segment_offset": offset,
                    "valid_lanes": segment_width,
                    "input0": lhs_segment,
                    "input1": rhs_segment,
                    "expected_output": [_sat_i8((a * b) >> 4) for a, b in zip(lhs_segment, rhs_segment)],
                    "primitive_program": [
                        {"op": "VMUL", "row": 0, "segment": 0},
                        {"op": "VREQUANT", "mode": "INT8_SHIFT4_CLAMP"},
                        {"op": "HALT"},
                    ],
                    "ppa_theory": {
                        "useful_vector_lane_ops": segment_width,
                        "theoretical_vector_cycles": 2,
                        "primitive_passes": 2,
                    },
                }
            )
    return {
        "stage_id": stage_id,
        "logical_op": "vector_mul_requant",
        "descriptor_op": "desc_vector_tile_v1",
        "segment_width": segment_width,
        "rows": rows,
        "row_width": width,
        "segment_count": len(jobs),
        "execution_state": "compiler_lowered_executable_subgraph",
        "required_primitives": ["VEC_MUL", "VEC_REQUANT"],
        "jobs": jobs,
        "ppa_theory": {
            "theoretical_vector_cycles": len(jobs) * 2,
            "effective_vector_lane_ops": rows * width,
        },
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
        elif logical_op == "rmsnorm":
            source, expected = _rmsnorm_stage_operands(stage_id, golden)
            stage["rmsnorm_plan"] = lower_rmsnorm_segmented_rows(
                source,
                expected,
                stage_id=stage_id,
            )
            stage["provenance"] = "compiler_desc_vector_tile_rmsnorm_lowered_not_full_block_submitted"
        elif logical_op == "vector_add":
            lhs, rhs = _vector_add_stage_operands(stage_id, golden)
            stage["vector_plan"] = lower_residual_add_vector_tiles(lhs, rhs, stage_id=stage_id)
            stage["provenance"] = "compiler_desc_vector_tile_lowered_not_submitted"
        elif logical_op == "vector_mul":
            lhs, rhs = _gate_mul_stage_operands(stage_id, golden)
            stage["gate_mul_plan"] = lower_gate_mul_vector_tiles(lhs, rhs, stage_id=stage_id)
            stage["provenance"] = "compiler_desc_vector_tile_gate_mul_lowered_not_full_block_submitted"
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


def lower_rmsnorm_segmented_rows(
    source: list[list[int]],
    expected: list[list[int]],
    *,
    stage_id: str = "rmsnorm",
    segment_width: int = VECTOR_SEGMENT_WIDTH,
) -> dict[str, Any]:
    _validate_same_shape(source, expected, "source", "expected")
    if segment_width != VECTOR_SEGMENT_WIDTH:
        raise ValueError(f"v0 RMSNorm segment width must be {VECTOR_SEGMENT_WIDTH}")
    rows = len(source)
    width = len(source[0]) if rows else 0
    if width == 0 or width % segment_width:
        raise ValueError("RMSNorm lowering requires a non-empty width that is a multiple of 8")

    row_plans = []
    jobs = []
    total_segments = 0
    for row_idx, row in enumerate(source):
        sequence = rmsnorm_primitive_sequence(row)
        reduce_segments = []
        scale_segments = []
        for segment_idx, offset in enumerate(range(0, width, segment_width)):
            segment = row[offset : offset + segment_width]
            expected_segment = expected[row_idx][offset : offset + segment_width]
            reduce_segments.append(
                {
                    "segment": segment_idx,
                    "segment_offset": offset,
                    "input": segment,
                    "primitive": "REDUCE_SUMSQ",
                    "partial_sumsq": sum(int(value) * int(value) for value in segment),
                }
            )
            scale_segments.append(
                {
                    "segment": segment_idx,
                    "segment_offset": offset,
                    "input": segment,
                    "scalar": sequence["rsqrt_q24"],
                    "primitive_program": [
                        {"op": "VEC_SCALE", "scalar": "row_rsqrt_q24", "shift": sequence["shift"]},
                        {"op": "HALT"},
                    ],
                    "expected_output": expected_segment,
                }
            )
            jobs.append(
                {
                    "stage_id": stage_id,
                    "logical_op": "rmsnorm",
                    "descriptor_op": "desc_vector_tile_v1",
                    "row": row_idx,
                    "segment": segment_idx,
                    "segment_offset": offset,
                    "program_select": "src0" if segment_idx == 0 else "src1",
                    "input0": row[0:segment_width],
                    "input1": row[segment_width : segment_width * 2],
                    "expected_output": expected_segment,
                    "primitive_program": [
                        {"op": "VREDSUM", "mode": "SUMSQ_SRC0"},
                        {"op": "VREDSUM", "mode": "SUMSQ_SRC1"},
                        {"op": "VDIV", "mode": "RSQRT_ROW_ACCUM"},
                        {"op": "VNORM", "mode": "SCALE_SRC0_BY_SFU" if segment_idx == 0 else "SCALE_SRC1_BY_SFU"},
                        {"op": "HALT"},
                    ],
                    "ppa_theory": {
                        "useful_vector_lane_ops": segment_width,
                        "theoretical_reduction_cycles": 2,
                        "theoretical_sfu_cycles": 1,
                        "theoretical_vector_cycles": 1,
                    },
                }
            )
            total_segments += 1
        row_plans.append(
            {
                "row": row_idx,
                "sumsq": sequence["sumsq"],
                "rsqrt_q24": sequence["rsqrt_q24"],
                "shift": sequence["shift"],
                "reduce_segments": reduce_segments,
                "sfu": {"primitive": "SFU_RSQRT", "input": sequence["sumsq"], "output": sequence["rsqrt_q24"]},
                "scale_segments": scale_segments,
            }
        )

    return {
        "stage_id": stage_id,
        "logical_op": "rmsnorm",
        "descriptor_op": "desc_vector_tile_v1",
        "execution_state": "compiler_lowered_executable_subgraph",
        "required_primitives": ["REDUCE_SUMSQ", "SFU_RSQRT", "VEC_SCALE"],
        "arg1_mode_extension": {
            "VREDSUM": ["SUMSQ_SRC0", "SUMSQ_SRC1"],
            "VDIV": ["RSQRT_ROW_ACCUM"],
            "VNORM": ["SCALE_SRC0_BY_SFU", "SCALE_SRC1_BY_SFU"],
        },
        "segment_width": segment_width,
        "rows": rows,
        "row_width": width,
        "segment_count": total_segments,
        "jobs": jobs,
        "row_plans": row_plans,
        "ppa_theory": {
            "theoretical_reduction_cycles": total_segments * 2,
            "theoretical_sfu_cycles": total_segments,
            "theoretical_vector_cycles": total_segments,
            "effective_vector_lane_ops": rows * width,
        },
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


def _attention_pv_q15_i8_shift15(prob_q15: list[list[int]], v: list[list[int]]) -> list[list[int]]:
    output = []
    for prob_row in prob_q15:
        out_row = []
        for dim in range(len(v[0])):
            acc = sum(int(prob_row[j]) * int(v[j][dim]) for j in range(len(prob_row)))
            out_row.append(acc >> 15)
        output.append(out_row)
    return output


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


def _rmsnorm_stage_operands(stage_id: str, golden: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    mapping = {
        "rmsnorm_attn": ("input_x", "rmsnorm_attn"),
        "rmsnorm_ffn": ("residual_attn", "rmsnorm_ffn"),
    }
    if stage_id not in mapping:
        raise ValueError(f"unsupported RMSNorm stage {stage_id!r}")
    input_name, output_name = mapping[stage_id]
    return golden[input_name], golden[output_name]


def _vector_add_stage_operands(stage_id: str, golden: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    mapping = {
        "residual_attn": ("input_x", "attention_projected"),
        "residual_ffn": ("residual_attn", "down"),
    }
    if stage_id not in mapping:
        raise ValueError(f"unsupported vector add stage {stage_id!r}")
    lhs_name, rhs_name = mapping[stage_id]
    return golden[lhs_name], golden[rhs_name]


def _gate_mul_stage_operands(stage_id: str, golden: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    mapping = {
        "gate_mul_up": ("silu_gate", "up"),
    }
    if stage_id not in mapping:
        raise ValueError(f"unsupported gate multiply stage {stage_id!r}")
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


def _aggregate_stage_topology(stages: list[dict[str, Any]]) -> list[str]:
    stage_ids = [stage["stage_id"] for stage in stages]
    expected_stage_ids = [stage_id for stage_id, _, _, _ in _stage_defs()]
    _require_equal(stage_ids, expected_stage_ids, "BlockPlan stage order")
    return [
        "rmsnorm",
        "qkv_linear_projection",
        "rope",
        "causal_gqa_attention",
        "output_projection",
        "residual_add",
        "rmsnorm",
        "swiglu",
        "down_projection",
        "residual_add",
    ]


def _validate_workload_shape(spec: dict[str, Any], shape: dict[str, int], *, expected_blocks: int) -> None:
    spec_shape = dict(spec.get("shape", {}))
    blocks = spec_shape.pop("blocks", None)
    _require_equal(blocks, expected_blocks, f"{spec.get('name', '<unnamed>')} blocks")
    _require_equal(spec_shape, shape, f"{spec.get('name', '<unnamed>')} shape")


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


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
