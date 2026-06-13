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

CURRENT_RTL_PROGRAM_CAPACITY_WORDS = 128
CURRENT_ATTENTION_TILE_ROWS = 8
CURRENT_ATTENTION_TILE_COLS = 8
SUPPORTED_MASK_POLICIES = {"none", "causal", "padding", "causal_padding"}


def build_attention_mask_plan(
    *,
    seq_q: int,
    seq_k: int,
    mask_policy: str = "none",
    valid_k: int | None = None,
    tile_rows: int = CURRENT_ATTENTION_TILE_ROWS,
    tile_cols: int = CURRENT_ATTENTION_TILE_COLS,
) -> dict[str, Any]:
    """Lower logical Attention mask rules into per-row physical lane masks."""

    if mask_policy not in SUPPORTED_MASK_POLICIES:
        raise ValueError(f"unsupported attention mask_policy {mask_policy}")
    if seq_q <= 0 or seq_k <= 0:
        raise ValueError("attention seq_q and seq_k must be positive")
    if seq_q > tile_rows or seq_k > tile_cols:
        raise ValueError("current attention mask lowering supports only one physical tile")
    if valid_k is None:
        valid_k = seq_k
    if valid_k < 0 or valid_k > seq_k:
        raise ValueError("attention valid_k must be in 0..seq_k")
    if mask_policy in ("none", "causal") and valid_k != seq_k:
        raise ValueError(f"mask_policy {mask_policy} cannot shorten valid_k")

    causal = mask_policy in ("causal", "causal_padding")
    padding = mask_policy in ("padding", "causal_padding")
    key_limit = valid_k if padding else seq_k
    valid_lane_masks = []
    for query in range(tile_rows):
        row_mask = 0
        if query < seq_q:
            for key in range(tile_cols):
                visible = key < seq_k and key < key_limit and (not causal or key <= query)
                if visible:
                    row_mask |= 1 << key
        valid_lane_masks.append(row_mask)

    valid_query_mask = (1 << seq_q) - 1
    full_lane_mask = (1 << tile_cols) - 1
    row_mask_words = [
        sum(valid_lane_masks[row + offset] << (offset * 8) for offset in range(4))
        for row in (0, 4)
    ]
    current_rtl_executable = (
        seq_q == tile_rows
        and seq_k == tile_cols
        and all(row_mask != 0 for row_mask in valid_lane_masks)
    )
    return {
        "mask_policy": mask_policy,
        "seq_q": seq_q,
        "seq_k": seq_k,
        "valid_k": valid_k,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "valid_query_mask": valid_query_mask,
        "valid_lane_masks": valid_lane_masks,
        "row_mask_words": row_mask_words,
        "execution_state": "executable" if current_rtl_executable else "planned_not_executable",
    }


def build_softmax_expanded_primitive_program(
    rows: int,
    elements: int,
    *,
    capacity_words: int = CURRENT_RTL_PROGRAM_CAPACITY_WORDS,
) -> dict[str, Any]:
    """Expand row Softmax into primitive uops and report program capacity."""

    if rows <= 0 or elements <= 0:
        raise ValueError("softmax rows and elements must be positive")
    program: list[dict[str, int | str]] = []
    for row in range(rows):
        program.extend(
            [
                {"op": "REDUCE_MAX", "row": row},
                {"op": "VEC_SUB", "row": row},
                {"op": "VEC_CLAMP", "row": row},
            ]
        )
        program.extend({"op": "SFU_EXP", "row": row, "lane": lane} for lane in range(elements))
        program.extend(
            [
                {"op": "REDUCE_SUM", "row": row},
                {"op": "SFU_RECIP", "row": row},
                {"op": "VEC_SCALE", "row": row},
            ]
        )
    program.append({"op": "HALT"})
    required_words = len(program)
    return {
        "representation": "compiler_expanded_primitives",
        "program": program,
        "required_words": required_words,
        "required_bytes": required_words * 4,
        "capacity_words": capacity_words,
        "fits_current_capacity": required_words <= capacity_words,
        "shortfall_words": max(0, required_words - capacity_words),
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
    parent_shape = parent["shape"]
    shape = {
        "seq_q": int(parent_shape.get("seq_q", parent_shape["seq_len"])),
        "seq_k": int(parent_shape.get("seq_k", parent_shape["seq_len"])),
        "head_dim": int(parent["shape"]["head_dim"]),
        "value_dim": int(pv_shape["n"]),
        "softmax_rows": int(softmax_shape.get("rows", 1)),
        "softmax_elements": int(softmax_shape["elements"]),
    }
    mask_plan = build_attention_mask_plan(
        seq_q=shape["seq_q"],
        seq_k=shape["seq_k"],
        mask_policy=str(parent.get("mask_policy", "none")),
        valid_k=int(parent.get("valid_k", shape["seq_k"])),
    )
    numerical_contract = parent.get("numerical_contract") or stage_workloads["pv"].get("numerical_contract")
    stages = _build_stages(stage_workloads, numerical_contract, mask_plan)
    softmax_program = build_softmax_expanded_primitive_program(
        shape["softmax_rows"], shape["softmax_elements"]
    )
    next(stage for stage in stages if stage["stage_id"] == "softmax")["primitive_program"] = softmax_program
    buffers = [
        _buffer("q_tile", "int8", [int(qk_shape["m"]), int(qk_shape["k"])], "input", [0]),
        _buffer("k_t_tile", "int8", [int(qk_shape["k"]), int(qk_shape["n"])], "input", [0], layout="transposed_d_by_s"),
        _buffer("score_raw", "int32", [int(qk_shape["m"]), int(qk_shape["n"])], "qk", [1], producer_stage_index=0),
        _buffer("score_softmax_in", "int32", [int(qk_shape["m"]), int(qk_shape["n"])], "scale_mask", [2], producer_stage_index=1),
        _buffer("row_mask", "uint32", [2], "compiler", [1, 2]),
        _buffer("prob_q15", "uint16_q0.15", [int(pv_shape["m"]), int(pv_shape["k"])], "softmax", [3], producer_stage_index=2),
        _buffer("v_tile", "int8", [int(pv_shape["k"]), int(pv_shape["n"])], "input", [3]),
        _buffer("o_i32", "int32", [int(pv_shape["m"]), int(pv_shape["n"])], "pv", [], producer_stage_index=3),
    ]
    runtime_jobs = _build_runtime_jobs(stage_workloads, attention_group, mask_plan["execution_state"])
    plan = {
        "workload_name": f"transformer_{parent['name']}",
        "attention_group": attention_group,
        "logical_op": "scaled_dot_product_attention_v1",
        "shape": shape,
        "numerical_contract": numerical_contract,
        "group_state": "software_group_measured_stages",
        "group_cycle_policy": "sum_measured_stages",
        "scale_mask_provenance": "measured_npu_vector_bridge",
        "execution_state": mask_plan["execution_state"],
        "mask": mask_plan,
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
    seq_q = int(parent_shape.get("seq_q", parent_shape["seq_len"]))
    seq_k = int(parent_shape.get("seq_k", parent_shape["seq_len"]))
    if not 0 < seq_q <= 8 or not 0 < seq_k <= 8 or int(parent_shape["head_dim"]) != 8:
        raise ValueError("current single-tile attention planning requires seq_q/seq_k in 1..8 and head_dim=8")
    qk = stage_workloads["qk"]["shape"]
    softmax = stage_workloads["softmax"]["shape"]
    pv = stage_workloads["pv"]["shape"]
    if (int(qk["m"]), int(qk["n"]), int(qk["k"])) != (8, 8, 8):
        raise ValueError("current QK lowering requires m=n=k=8")
    if int(softmax["elements"]) != 8:
        raise ValueError("current softmax lowering requires 8 elements")
    if (int(pv["m"]), int(pv["n"]), int(pv["k"])) != (8, 8, 8):
        raise ValueError("current PV lowering requires m=n=k=8")


def _build_stages(
    stage_workloads: dict[str, dict[str, Any]],
    parent_contract: str,
    mask_plan: dict[str, Any],
) -> list[dict[str, Any]]:
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
            "inputs": ["score_raw", "row_mask"],
            "outputs": ["score_softmax_in"],
            "descriptor_op": DESCRIPTOR_OP_BY_STAGE["scale_mask"],
            "workload_name": f"transformer_{stage_workloads['scale_mask']['name']}",
            "execution": "descriptor_job",
            "scale_policy": "fixed_multiplier_shift",
            "scale_multiplier": 11585,
            "scale_shift": 15,
            "rounding": "round_nearest_away_from_zero",
            "mask_policy": mask_plan["mask_policy"],
            "valid_k": mask_plan["valid_k"],
            "valid_query_mask": mask_plan["valid_query_mask"],
            "valid_lane_masks": mask_plan["valid_lane_masks"],
            "execution_state": mask_plan["execution_state"],
            "numerical_contract": stage_workloads["scale_mask"].get("numerical_contract"),
        },
        {
            "stage_id": "softmax",
            "operator": STAGE_OPERATORS["softmax"],
            "inputs": ["score_softmax_in", "row_mask"],
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


def _build_runtime_jobs(
    stage_workloads: dict[str, dict[str, Any]],
    attention_group: str,
    execution_state: str,
) -> list[dict[str, Any]]:
    return [
        _runtime_job(
            "qk",
            _job_id_symbol(stage_workloads["qk"]["name"]),
            DESCRIPTOR_OP_BY_STAGE["qk"],
            "q_tile",
            "k_t_tile",
            "score_raw",
            attention_group=attention_group,
            execution_state=execution_state,
        ),
        _runtime_job(
            "scale_mask",
            _job_id_symbol(stage_workloads["scale_mask"]["name"]),
            DESCRIPTOR_OP_BY_STAGE["scale_mask"],
            "score_raw",
            "row_mask",
            "score_softmax_in",
            attention_group=attention_group,
            execution_state=execution_state,
        ),
        _runtime_job(
            "softmax",
            _job_id_symbol(stage_workloads["softmax"]["name"]),
            DESCRIPTOR_OP_BY_STAGE["softmax"],
            "score_softmax_in",
            "row_mask",
            "prob_q15",
            check_policy="absolute_tolerance",
            attention_group=attention_group,
            execution_state=execution_state,
        ),
        _runtime_job(
            "pv",
            _job_id_symbol(stage_workloads["pv"]["name"]),
            DESCRIPTOR_OP_BY_STAGE["pv"],
            "prob_q15",
            "v_tile",
            "o_i32",
            attention_group=attention_group,
            execution_state=execution_state,
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
    execution_state: str,
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
        "execution_state": execution_state,
    }
