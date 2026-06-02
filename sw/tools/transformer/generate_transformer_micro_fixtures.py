from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from npu_compiler.attention import build_attention_plan_from_manifest
from npu_phase0.rtl_fixture import softmax_q0_8
from transformer.golden import (
    TILE_K,
    TILE_M,
    TILE_N,
    TILE_WORDS,
    deterministic_i8_matrix,
    tile_k_stream,
)
from transformer.micro_golden import (
    ATTENTION_BRINGUP_CONTRACT_V0,
    ATTENTION_NUMERICAL_CONTRACT_V1,
    PROB_ONE_Q15,
    attention_head_fixed_spec,
    attention_pv_q15_i8_i32,
    attention_qk_scores_i8_i32,
    attention_softmax_fixed_spec_q15,
    classify_matrix_shape,
    softmax_row_primitive_lut_q15,
    transpose,
)


TRANSFORMER_SPEC_PATH = Path("workloads/manifests/transformer/transformer_micro_v0.jsonc")


def read_jsonc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(re.sub(r"//.*$", "", text, flags=re.MULTILINE))


def validate_transformer_micro_spec(spec: dict[str, Any]) -> None:
    if spec.get("name") != "transformer_micro_v0":
        raise ValueError("unsupported transformer micro spec name")
    if "precision_baseline" not in spec:
        raise ValueError("transformer spec is missing precision_baseline")
    workloads = spec.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("transformer spec workloads must be a non-empty array")
    scenarios = set()
    for workload in workloads:
        for field in ("name", "scenario", "op", "shape", "status", "metrics"):
            if field not in workload:
                raise ValueError(f"transformer workload {workload.get('name', '<unnamed>')} missing {field}")
        if "activity_scope" not in workload:
            raise ValueError(f"transformer workload {workload['name']} missing activity_scope")
        if "external_memory" not in workload:
            raise ValueError(f"transformer workload {workload['name']} missing external_memory")
        scenarios.add(workload["scenario"])
    if "transformer_prefill" not in scenarios or "transformer_decode" not in scenarios:
        raise ValueError("transformer spec must include both prefill and decode scenarios")


def generate_transformer_micro_fixtures(spec_path: Path = TRANSFORMER_SPEC_PATH) -> dict[str, Any]:
    spec = read_jsonc(spec_path)
    validate_transformer_micro_spec(spec)
    precision = spec["precision_baseline"]
    attention_plans = _build_attention_plans(spec)
    executable = []
    model_only = []
    for index, workload in enumerate(spec["workloads"]):
        if workload["op"] in ("matmul", "matmul_u16s8_q15") and workload["status"] == "planned_current_matmul_extension":
            executable.append(_generate_matmul_workload(workload, precision, seed=index + 1))
        elif workload["op"] in ("softmax", "attention_softmax_v1") and workload["status"] == "planned_current_softmax_extension":
            executable.append(_generate_softmax_workload(workload, precision, seed=index + 1))
        else:
            model_only.append(_model_only_metadata(workload, precision))
    _attach_attention_plan_metadata(executable, attention_plans)
    _attach_attention_plan_metadata(model_only, attention_plans)
    return {
        "spec_name": spec["name"],
        "version": int(spec["version"]),
        "attention_plans": list(attention_plans.values()),
        "executable_workloads": executable,
        "model_only_workloads": model_only,
    }


def _build_attention_plans(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = {
        workload.get("attention_group")
        for workload in spec.get("workloads", [])
        if workload.get("logical_op") == "scaled_dot_product_attention"
    }
    plans: dict[str, dict[str, Any]] = {}
    for group in sorted(group for group in groups if group):
        if group == "attention_prefill_s8_d8":
            plans[group] = build_attention_plan_from_manifest(spec, group)
    return plans


def _attach_attention_plan_metadata(items: list[dict[str, Any]], plans: dict[str, dict[str, Any]]) -> None:
    for item in items:
        metadata = item.get("metadata", {})
        group = metadata.get("attention_group")
        if not group or group not in plans:
            continue
        plan = plans[group]
        stage_id = metadata.get("attention_stage")
        metadata["attention_plan"] = {
            "workload_name": plan["workload_name"],
            "attention_group": plan["attention_group"],
            "group_state": plan["group_state"],
            "group_cycle_policy": plan["group_cycle_policy"],
        }
        if stage_id == "full_attention":
            metadata["attention_plan"]["stages"] = [stage["stage_id"] for stage in plan["stages"]]
            metadata["attention_plan"]["runtime_jobs"] = [job["stage_id"] for job in plan["runtime_jobs"]]
            continue
        for index, stage in enumerate(plan["stages"]):
            if stage["stage_id"] == stage_id:
                metadata["attention_plan_stage"] = {
                    "stage_index": index,
                    "operator": stage["operator"],
                    "inputs": stage["inputs"],
                    "outputs": stage["outputs"],
                    "execution": stage.get("execution", "descriptor_job"),
                }
                break
        for job in plan["runtime_jobs"]:
            if job["stage_id"] == stage_id:
                metadata["attention_plan_runtime_job"] = {
                    "job_id_symbol": job["job_id_symbol"],
                    "descriptor_op": job["descriptor_op"],
                    "input0": job["input0"],
                    "input1": job["input1"],
                    "output": job["output"],
                    "perf_scope": job["perf_scope"],
                }
                break


def _generate_matmul_workload(workload: dict[str, Any], precision: dict[str, str], seed: int) -> dict[str, Any]:
    shape = workload["shape"]
    m_dim = int(shape["m"])
    n_dim = int(shape["n"])
    k_dim = int(shape["k"])
    if m_dim != TILE_M or n_dim != TILE_N:
        raise ValueError(
            f"{workload['name']}: first executable Transformer fixtures require one 8x8 output tile"
        )
    if k_dim % TILE_K != 0:
        raise ValueError(f"{workload['name']}: K must be a multiple of {TILE_K}")

    if workload.get("logical_op") == "attention_qk_score":
        q = deterministic_i8_matrix(m_dim, k_dim, seed)
        k = deterministic_i8_matrix(n_dim, k_dim, seed + 17)
        a = q
        b = transpose(k)
        attention_scores = attention_qk_scores_i8_i32(q, k)
    elif workload.get("logical_op") == "attention_probability_value":
        probabilities = _deterministic_probability_q15(m_dim, k_dim)
        v = deterministic_i8_matrix(k_dim, n_dim, seed + 17)
        expected_c = [value for row in _attention_pv_q15_i8_shift15(probabilities, v) for value in row]
        metadata = {
            "scenario": workload["scenario"],
            "logical_op": workload.get("logical_op", workload["op"]),
            "logical_shape": {"m": m_dim, "n": n_dim, "k": k_dim},
            "workload_family": _workload_family(workload),
            "shape_class": classify_matrix_shape(m_dim, n_dim, k_dim),
            "rtl_tile_shape": {"m": TILE_M, "n": TILE_N, "k": TILE_K},
            "precision": precision,
            "activity_scope": workload["activity_scope"],
            "external_memory": workload["external_memory"],
            "attention_group": workload.get("attention_group"),
            "attention_stage": workload.get("attention_stage"),
            "numerical_contract": workload.get("numerical_contract"),
            "stage_provenance": "measured_mixed_matrix_path",
            "kv_read_bytes": int(workload["external_memory"].get("kv_cache_read_bytes", 0)),
            "kv_write_bytes": int(workload["external_memory"].get("kv_cache_write_bytes", 0)),
            "k_chunks": 1,
            "tile_jobs": 1,
            "attention": {
                "probability_policy": "q0.15_u16",
                "p_layout": "row_major_s_by_s",
                "v_runtime_layout": "row_major_s_by_d",
                "output_dtype": "int32_after_q15_shift",
            },
        }
        return {
            "name": f"transformer_{workload['name']}",
            "kind": "transformer_micro",
            "op": "matmul_u16s8_q15",
            "role": "transformer_micro",
            "metadata": metadata,
            "k_chunks": 1,
            "tile_words": TILE_WORDS,
            "a_bits": 16,
            "a_stream": [[value for row in probabilities for value in row]],
            "b_stream": [[value for row in v for value in row]],
            "expected_c": expected_c,
            "attention_scores": None,
        }
    else:
        a = deterministic_i8_matrix(m_dim, k_dim, seed)
        b = deterministic_i8_matrix(k_dim, n_dim, seed + 17)
        attention_scores = None
    tiled = tile_k_stream(a, b)
    metadata = {
        "scenario": workload["scenario"],
        "logical_op": workload.get("logical_op", workload["op"]),
        "logical_shape": {"m": m_dim, "n": n_dim, "k": k_dim},
        "workload_family": _workload_family(workload),
        "shape_class": classify_matrix_shape(m_dim, n_dim, k_dim),
        "rtl_tile_shape": {"m": TILE_M, "n": TILE_N, "k": TILE_K},
        "precision": precision,
        "activity_scope": workload["activity_scope"],
        "external_memory": workload["external_memory"],
        "attention_group": workload.get("attention_group"),
        "attention_stage": workload.get("attention_stage"),
        "numerical_contract": workload.get("numerical_contract"),
        "stage_provenance": "measured_current_matmul_path",
        "kv_read_bytes": int(workload["external_memory"].get("kv_cache_read_bytes", 0)),
        "kv_write_bytes": int(workload["external_memory"].get("kv_cache_write_bytes", 0)),
        "k_chunks": tiled["k_chunks"],
        "tile_jobs": 1,
    }
    if attention_scores is not None:
        metadata["attention"] = {
            "q_layout": "row_major_s_by_d",
            "k_runtime_layout": "transposed_d_by_s",
            "score_dtype": "int32",
        }
    if workload.get("logical_op") == "attention_probability_value":
        metadata["attention"] = {
            "probability_policy": "bringup_int8_proxy",
            "p_layout": "row_major_s_by_s",
            "v_runtime_layout": "row_major_s_by_d",
            "output_dtype": "int32",
        }
    return {
        "name": f"transformer_{workload['name']}",
        "kind": "transformer_micro",
        "op": "matmul_k_stream",
        "role": "transformer_micro",
        "metadata": metadata,
        "k_chunks": tiled["k_chunks"],
        "tile_words": TILE_WORDS,
        "a_stream": tiled["a_stream"],
        "b_stream": tiled["b_stream"],
        "expected_c": tiled["expected_c"],
        "attention_scores": attention_scores,
    }


def _generate_softmax_workload(workload: dict[str, Any], precision: dict[str, str], seed: int) -> dict[str, Any]:
    shape = workload["shape"]
    elements = int(shape["elements"])
    if elements != 8:
        raise ValueError(f"{workload['name']}: current softmax RTL path requires exactly 8 elements")

    if workload.get("logical_op") == "attention_row_softmax":
        row = [32, 0, -32, -64, -96, -128, -128, -128]
        expected_y = softmax_row_primitive_lut_q15(row)["output_q15"]
        output_dtype = "q0.15_uint16"
        implementation = "npu_v1_vector_reduction_sfu_sequence"
        op = "attention_softmax_v1"
    else:
        row = _deterministic_softmax_row_i8(elements, seed)
        expected_y = softmax_q0_8(row)
        output_dtype = "q0.8_uint8"
        implementation = "npu_v0_micro_op_softmax_lut"
        op = "softmax"
    metadata = {
        "scenario": workload["scenario"],
        "logical_op": workload.get("logical_op", workload["op"]),
        "logical_shape": shape,
        "workload_family": _workload_family(workload),
        "precision": precision,
        "activity_scope": workload["activity_scope"],
        "external_memory": workload["external_memory"],
        "attention_group": workload.get("attention_group"),
        "attention_stage": workload.get("attention_stage"),
        "numerical_contract": workload.get("numerical_contract"),
        "stage_provenance": "measured_current_softmax_path",
        "kv_read_bytes": int(workload["external_memory"].get("kv_cache_read_bytes", 0)),
        "kv_write_bytes": int(workload["external_memory"].get("kv_cache_write_bytes", 0)),
        "softmax": {
            "input_dtype": "int8",
            "output_dtype": output_dtype,
            "row_count_measured": 1,
            "row_count_logical": int(shape.get("rows", 1)),
            "implementation": implementation,
        },
    }
    return {
        "name": f"transformer_{workload['name']}",
        "kind": "transformer_micro",
        "op": op,
        "role": "transformer_micro",
        "metadata": metadata,
        "x_words": elements,
        "y_words": elements,
        "x": row,
        "expected_y": expected_y,
    }


def _model_only_metadata(workload: dict[str, Any], precision: dict[str, str]) -> dict[str, Any]:
    shape = workload["shape"]
    metadata = {
        "scenario": workload["scenario"],
        "logical_op": workload.get("logical_op", workload["op"]),
        "logical_shape": shape,
        "workload_family": _workload_family(workload),
        "precision": precision,
        "activity_scope": workload["activity_scope"],
        "external_memory": workload["external_memory"],
        "attention_group": workload.get("attention_group"),
        "attention_stage": workload.get("attention_stage"),
        "numerical_contract": workload.get("numerical_contract"),
        "kv_read_bytes": int(workload["external_memory"].get("kv_cache_read_bytes", 0)),
        "kv_write_bytes": int(workload["external_memory"].get("kv_cache_write_bytes", 0)),
        "model_only": True,
        "status": workload["status"],
    }
    if workload["op"] == "matmul":
        metadata["shape_class"] = classify_matrix_shape(
            int(shape["m"]),
            int(shape["n"]),
            int(shape["k"]),
        )
    if workload["op"] == "memory_traffic":
        external = workload["external_memory"]
        metadata["bytes_per_token"] = int(external.get("kv_cache_read_bytes", 0)) + int(
            external.get("kv_cache_write_bytes", 0)
        )
    if workload.get("logical_op") == "attention_row_softmax":
        row = [32, 0, -32, -64, -96, -128, -192, -256]
        metadata["golden"] = attention_softmax_fixed_spec_q15([row])
        metadata["stage_provenance"] = "model_only_fixed_spec"
    if workload.get("logical_op") == "scaled_dot_product_attention":
        seq_len = int(shape["seq_len"])
        head_dim = int(shape["head_dim"])
        q = deterministic_i8_matrix(seq_len, head_dim, 31)
        k = deterministic_i8_matrix(seq_len, head_dim, 37)
        v = deterministic_i8_matrix(seq_len, head_dim, 41)
        metadata["golden"] = attention_head_fixed_spec(q, k, v)
        metadata["stage_provenance"] = "model_only_full_attention"
        metadata["numerical_contract"] = metadata["numerical_contract"] or ATTENTION_NUMERICAL_CONTRACT_V1
    if workload.get("logical_op") == "attention_probability_value":
        metadata["stage_provenance"] = "model_only_pv_policy_pending"
        metadata["numerical_contract"] = metadata["numerical_contract"] or ATTENTION_NUMERICAL_CONTRACT_V1
    if workload.get("attention_stage") == "softmax" and metadata.get("numerical_contract") is None:
        metadata["numerical_contract"] = ATTENTION_BRINGUP_CONTRACT_V0
    return {
        "name": f"transformer_{workload['name']}",
        "kind": "transformer_model_only",
        "metadata": metadata,
    }


def _workload_family(workload: dict[str, Any]) -> str:
    if workload["scenario"] == "transformer_prefill":
        return "transformer_prefill"
    if workload["scenario"] == "transformer_decode":
        return "transformer_decode"
    return "transformer_micro"


def _deterministic_softmax_row_i8(elements: int, seed: int) -> list[int]:
    base = [4, 3, 2, 1, 0, -1, -2, -3]
    if elements != len(base):
        raise ValueError("current deterministic softmax row is defined for 8 elements")
    rotate = seed % elements
    return base[rotate:] + base[:rotate]


def _deterministic_probability_proxy_i8(rows: int, cols: int) -> list[list[int]]:
    if cols != 8:
        raise ValueError("current probability proxy is defined for 8 columns")
    patterns = [
        [64, 32, 16, 8, 4, 2, 1, 0],
        [0, 64, 32, 16, 8, 4, 2, 1],
        [1, 0, 64, 32, 16, 8, 4, 2],
        [2, 1, 0, 64, 32, 16, 8, 4],
        [4, 2, 1, 0, 64, 32, 16, 8],
        [8, 4, 2, 1, 0, 64, 32, 16],
        [16, 8, 4, 2, 1, 0, 64, 32],
        [32, 16, 8, 4, 2, 1, 0, 64],
    ]
    return [patterns[row % len(patterns)][:cols] for row in range(rows)]


def _deterministic_probability_q15(rows: int, cols: int) -> list[list[int]]:
    if cols != 8:
        raise ValueError("current probability Q15 fixture is defined for 8 columns")
    base = [16384, 8192, 4096, 2048, 1024, 512, 256, 255]
    out = []
    for row in range(rows):
        rotated = base[row % cols :] + base[: row % cols]
        total = sum(rotated)
        normalized = [(value * PROB_ONE_Q15) // total for value in rotated]
        normalized[-1] += PROB_ONE_Q15 - sum(normalized)
        out.append(normalized)
    return out


def _attention_pv_q15_i8_shift15(prob_q15: list[list[int]], v: list[list[int]]) -> list[list[int]]:
    output = []
    for prob_row in prob_q15:
        out_row = []
        for dim in range(len(v[0])):
            acc = sum(int(prob_row[j]) * int(v[j][dim]) for j in range(len(prob_row)))
            out_row.append(acc >> 15)
        output.append(out_row)
    return output
