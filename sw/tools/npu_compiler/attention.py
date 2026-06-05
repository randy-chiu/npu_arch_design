"""Attention planning helpers for Transformer v1 bring-up workloads."""

from __future__ import annotations

from typing import Any

from .attention_plan_schema import validate_attention_plan


STAGE_BY_MANIFEST_LOGICAL_OP = {
    "attention_qk_score": "qk",
    "attention_score_scale_mask": "scale_mask",
    "attention_row_softmax": "softmax",
    "attention_probability_value": "pv",
}

STAGE_OPERATORS = {
    "qk": "matmul_s8s8_i32_tile",
    "scale_mask": "attention_score_scale_mask_v1",
    "softmax": "attention_softmax_q15_v1",
    "pv": "matmul_u16s8_q15_i32_tile",
}

DESCRIPTOR_OP_BY_STAGE = {
    "qk": "matmul_k_stream",
    "scale_mask": "attention_scale_mask_v1",
    "softmax": "attention_softmax_v1",
    "pv": "matmul_u16s8_q15",
}


def build_attention_plan_from_manifest(spec: dict[str, Any], attention_group: str) -> dict[str, Any]:
    """Lower one manifest attention group into a compiler AttentionPlan.

    Current hardware support is intentionally narrow: one 8x8 QK tile, one
    8-element softmax row descriptor, and one 8x8 mixed-PV tile. The shape and
    contracts come from the manifest; unsupported shapes fail validation instead
    of being silently treated as the fixed bring-up case.
    """

    workloads = spec.get("workloads", [])
    parent = _find_parent_workload(workloads, attention_group)
    stage_workloads = _find_stage_workloads(workloads, attention_group)
    _validate_current_attention_group(parent, stage_workloads)

    qk_shape = stage_workloads["qk"]["shape"]
    pv_shape = stage_workloads["pv"]["shape"]
    softmax_shape = stage_workloads["softmax"]["shape"]
    shape = {
        "seq_q": int(parent["shape"]["seq_len"]),
        "seq_k": int(parent["shape"]["seq_len"]),
        "head_dim": int(parent["shape"]["head_dim"]),
        "value_dim": int(pv_shape["n"]),
        "softmax_rows": int(softmax_shape.get("rows", 1)),
        "softmax_elements": int(softmax_shape["elements"]),
    }
    numerical_contract = parent.get("numerical_contract") or stage_workloads["pv"].get("numerical_contract")
    stages = _build_stages(stage_workloads, numerical_contract)
    buffers = [
        _buffer("q_tile", "int8", [int(qk_shape["m"]), int(qk_shape["k"])], "input", [0]),
        _buffer("k_t_tile", "int8", [int(qk_shape["k"]), int(qk_shape["n"])], "input", [0], layout="transposed_d_by_s"),
        _buffer("score_raw", "int32", [int(qk_shape["m"]), int(qk_shape["n"])], "qk", [1], producer_stage_index=0),
        _buffer("score_softmax_in", "int32", [int(qk_shape["m"]), int(qk_shape["n"])], "scale_mask", [2], producer_stage_index=1),
        _buffer("prob_q15", "uint16_q0.15", [int(pv_shape["m"]), int(pv_shape["k"])], "softmax", [3], producer_stage_index=2),
        _buffer("v_tile", "int8", [int(pv_shape["k"]), int(pv_shape["n"])], "input", [3]),
        _buffer("o_i32", "int32", [int(pv_shape["m"]), int(pv_shape["n"])], "pv", [], producer_stage_index=3),
    ]
    runtime_jobs = _build_runtime_jobs(stage_workloads, attention_group)
    plan = {
        "workload_name": f"transformer_{parent['name']}",
        "attention_group": attention_group,
        "logical_op": "scaled_dot_product_attention_v1",
        "shape": shape,
        "numerical_contract": numerical_contract,
        "group_state": "software_group_measured_stages",
        "group_cycle_policy": "sum_measured_stages",
        "scale_mask_provenance": "measured_npu_vector_bridge",
        "stages": stages,
        "buffers": buffers,
        "runtime_jobs": runtime_jobs,
    }
    validate_attention_plan(plan)
    return plan


def build_attention_prefill_plan_s8_d8(
    *,
    spec: dict[str, Any] | None = None,
    attention_group: str = "attention_prefill_s8_d8",
) -> dict[str, Any]:
    """Compatibility wrapper for the current manifest-driven bring-up plan."""

    if spec is None:
        raise ValueError("build_attention_prefill_plan_s8_d8 now requires a manifest spec")
    return build_attention_plan_from_manifest(spec, attention_group)


def _find_parent_workload(workloads: list[dict[str, Any]], attention_group: str) -> dict[str, Any]:
    matches = [
        workload
        for workload in workloads
        if workload.get("attention_group") == attention_group
        and workload.get("logical_op") == "scaled_dot_product_attention"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one parent attention workload for {attention_group}, got {len(matches)}")
    return matches[0]


def _find_stage_workloads(workloads: list[dict[str, Any]], attention_group: str) -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = {}
    for workload in workloads:
        if workload.get("attention_group") != attention_group:
            continue
        stage = STAGE_BY_MANIFEST_LOGICAL_OP.get(workload.get("logical_op"))
        if stage is None:
            continue
        if stage in stages:
            raise ValueError(f"duplicate attention stage {stage} for {attention_group}")
        stages[stage] = workload
    missing = [stage for stage in ("qk", "scale_mask", "softmax", "pv") if stage not in stages]
    if missing:
        raise ValueError(f"attention group {attention_group} missing stages {missing}")
    return stages


def _validate_current_attention_group(parent: dict[str, Any], stage_workloads: dict[str, dict[str, Any]]) -> None:
    parent_shape = parent["shape"]
    if int(parent_shape["seq_len"]) != 8 or int(parent_shape["head_dim"]) != 8:
        raise ValueError("current executable attention lowering supports only seq_len=8, head_dim=8")
    qk = stage_workloads["qk"]["shape"]
    softmax = stage_workloads["softmax"]["shape"]
    pv = stage_workloads["pv"]["shape"]
    if (int(qk["m"]), int(qk["n"]), int(qk["k"])) != (8, 8, 8):
        raise ValueError("current QK lowering requires m=n=k=8")
    if int(softmax["elements"]) != 8:
        raise ValueError("current softmax lowering requires 8 elements")
    if (int(pv["m"]), int(pv["n"]), int(pv["k"])) != (8, 8, 8):
        raise ValueError("current PV lowering requires m=n=k=8")


def _build_stages(stage_workloads: dict[str, dict[str, Any]], parent_contract: str) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": "qk",
            "operator": STAGE_OPERATORS["qk"],
            "inputs": ["q_tile", "k_t_tile"],
            "outputs": ["score_raw"],
            "descriptor_op": DESCRIPTOR_OP_BY_STAGE["qk"],
            "workload_name": f"transformer_{stage_workloads['qk']['name']}",
            "numerical_contract": stage_workloads["qk"].get("numerical_contract"),
        },
        {
            "stage_id": "scale_mask",
            "operator": STAGE_OPERATORS["scale_mask"],
            "inputs": ["score_raw"],
            "outputs": ["score_softmax_in"],
            "descriptor_op": DESCRIPTOR_OP_BY_STAGE["scale_mask"],
            "workload_name": f"transformer_{stage_workloads['scale_mask']['name']}",
            "execution": "descriptor_job",
            "scale_policy": "fixed_multiplier_shift",
            "scale_multiplier": 11585,
            "scale_shift": 15,
            "rounding": "round_nearest_away_from_zero",
            "mask_policy": "none",
            "numerical_contract": stage_workloads["scale_mask"].get("numerical_contract"),
        },
        {
            "stage_id": "softmax",
            "operator": STAGE_OPERATORS["softmax"],
            "inputs": ["score_softmax_in"],
            "outputs": ["prob_q15"],
            "descriptor_op": DESCRIPTOR_OP_BY_STAGE["softmax"],
            "workload_name": f"transformer_{stage_workloads['softmax']['name']}",
            "numerical_contract": stage_workloads["softmax"].get("numerical_contract"),
        },
        {
            "stage_id": "pv",
            "operator": STAGE_OPERATORS["pv"],
            "inputs": ["prob_q15", "v_tile"],
            "outputs": ["o_i32"],
            "descriptor_op": DESCRIPTOR_OP_BY_STAGE["pv"],
            "workload_name": f"transformer_{stage_workloads['pv']['name']}",
            "numerical_contract": stage_workloads["pv"].get("numerical_contract"),
        },
    ]


def _build_runtime_jobs(stage_workloads: dict[str, dict[str, Any]], attention_group: str) -> list[dict[str, Any]]:
    return [
        _runtime_job(
            "qk",
            _job_id_symbol(stage_workloads["qk"]["name"]),
            DESCRIPTOR_OP_BY_STAGE["qk"],
            "q_tile",
            "k_t_tile",
            "score_raw",
            attention_group=attention_group,
        ),
        _runtime_job(
            "scale_mask",
            _job_id_symbol(stage_workloads["scale_mask"]["name"]),
            DESCRIPTOR_OP_BY_STAGE["scale_mask"],
            "score_raw",
            None,
            "score_softmax_in",
            attention_group=attention_group,
        ),
        _runtime_job(
            "softmax",
            _job_id_symbol(stage_workloads["softmax"]["name"]),
            DESCRIPTOR_OP_BY_STAGE["softmax"],
            "score_softmax_in",
            None,
            "prob_q15",
            check_policy="absolute_tolerance",
            attention_group=attention_group,
        ),
        _runtime_job(
            "pv",
            _job_id_symbol(stage_workloads["pv"]["name"]),
            DESCRIPTOR_OP_BY_STAGE["pv"],
            "prob_q15",
            "v_tile",
            "o_i32",
            attention_group=attention_group,
        ),
    ]


def _job_id_symbol(workload_name: str) -> str:
    return f"JOB_ID_TRANSFORMER_{workload_name.upper()}"


def _buffer(
    name: str,
    dtype: str,
    shape: list[int],
    producer: str,
    consumers: list[int],
    *,
    producer_stage_index: int = -1,
    layout: str = "row_major",
) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "layout": layout,
        "producer": producer,
        "producer_stage_index": producer_stage_index,
        "consumer_stage_indices": consumers,
    }


def _runtime_job(
    stage_id: str,
    job_id_symbol: str,
    descriptor_op: str,
    input0: str,
    input1: str | None,
    output: str,
    *,
    check_policy: str = "exact",
    attention_group: str,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "job_id_symbol": job_id_symbol,
        "descriptor_op": descriptor_op,
        "input0": input0,
        "input1": input1,
        "output": output,
        "k_chunks": 1 if descriptor_op.startswith("matmul") else 0,
        "check_policy": check_policy,
        "perf_scope": f"{attention_group}/{stage_id}",
    }
