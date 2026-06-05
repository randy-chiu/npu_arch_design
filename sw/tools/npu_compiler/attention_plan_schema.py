"""Validation helpers for compiler-generated attention plans."""

from __future__ import annotations

from typing import Any


REQUIRED_STAGE_ORDER = ["qk", "scale_mask", "softmax", "pv"]


def validate_attention_plan(plan: dict[str, Any]) -> None:
    """Validate the minimal AttentionPlan shape used by current S=8,D=8 bring-up."""

    for field in ("workload_name", "attention_group", "logical_op", "shape", "stages", "buffers", "runtime_jobs"):
        if field not in plan:
            raise ValueError(f"attention plan missing {field}")

    stages = plan["stages"]
    if [stage.get("stage_id") for stage in stages] != REQUIRED_STAGE_ORDER:
        raise ValueError("attention stages must be qk -> scale_mask -> softmax -> pv")

    buffers = {buffer["name"]: buffer for buffer in plan["buffers"]}
    for name in ("q_tile", "k_t_tile", "score_raw", "score_softmax_in", "prob_q15", "v_tile", "o_i32"):
        if name not in buffers:
            raise ValueError(f"attention plan missing buffer {name}")

    produced = {"q_tile", "k_t_tile", "v_tile"}
    for stage_index, stage in enumerate(stages):
        for input_name in stage.get("inputs", []):
            if input_name not in produced:
                raise ValueError(f"stage {stage['stage_id']} consumes {input_name} before it is produced")
        for output_name in stage.get("outputs", []):
            produced.add(output_name)
            buffer = buffers.get(output_name)
            if buffer is None:
                raise ValueError(f"stage {stage['stage_id']} produces undeclared buffer {output_name}")
            if int(buffer["producer_stage_index"]) != stage_index:
                raise ValueError(f"buffer {output_name} has incorrect producer_stage_index")

    runtime_jobs = plan["runtime_jobs"]
    runtime_stage_ids = [job["stage_id"] for job in runtime_jobs]
    if runtime_stage_ids != ["qk", "scale_mask", "softmax", "pv"]:
        raise ValueError("current Model A runtime jobs must be qk, scale_mask, softmax, pv")

    descriptor_ops = {job["stage_id"]: job["descriptor_op"] for job in runtime_jobs}
    expected_ops = {
        "qk": "matmul_k_stream",
        "scale_mask": "attention_scale_mask_v1",
        "softmax": "attention_softmax_v1",
        "pv": "matmul_u16s8_q15",
    }
    if descriptor_ops != expected_ops:
        raise ValueError(f"unexpected runtime descriptor ops: {descriptor_ops}")

    if plan.get("group_state") not in ("model_only_full_attention", "software_group_measured_stages"):
        raise ValueError("unsupported attention group_state")
    if plan.get("group_state") == "software_group_measured_stages":
        if plan.get("group_cycle_policy") != "sum_measured_stages":
            raise ValueError("software grouped attention must use sum_measured_stages cycle policy")
        if plan.get("scale_mask_provenance") not in ("measured_npu_vector_bridge",):
            raise ValueError("software grouped attention must state scale/mask provenance")
