"""Validation helpers for compiler-generated attention plans."""

from __future__ import annotations

from typing import Any


REQUIRED_STAGE_ORDER = ["qk", "scale_mask", "softmax", "pv"]


def validate_attention_plan(plan: dict[str, Any]) -> None:
    """Validate the minimal AttentionPlan shape used by current S=8,D=8 bring-up."""

    for field in (
        "workload_name",
        "attention_group",
        "logical_op",
        "shape",
        "mask",
        "execution_state",
        "stages",
        "buffers",
        "runtime_jobs",
    ):
        if field not in plan:
            raise ValueError(f"attention plan missing {field}")

    _validate_mask_plan(plan)

    stages = plan["stages"]
    if [stage.get("stage_id") for stage in stages] != REQUIRED_STAGE_ORDER:
        raise ValueError("attention stages must be qk -> scale_mask -> softmax -> pv")

    buffers = {buffer["name"]: buffer for buffer in plan["buffers"]}
    for name in ("q_tile", "k_t_tile", "score_raw", "score_softmax_in", "row_mask", "prob_q15", "v_tile", "o_i32"):
        if name not in buffers:
            raise ValueError(f"attention plan missing buffer {name}")

    produced = {"q_tile", "k_t_tile", "row_mask", "v_tile"}
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
    if any(job.get("execution_state") != plan["execution_state"] for job in runtime_jobs):
        raise ValueError("runtime job execution_state must match attention plan")

    softmax_stage = next(stage for stage in stages if stage["stage_id"] == "softmax")
    softmax_program = softmax_stage.get("primitive_program")
    if not isinstance(softmax_program, dict):
        raise ValueError("softmax stage missing expanded primitive_program")
    program = softmax_program.get("program")
    if not isinstance(program, list) or not program or program[-1].get("op") != "HALT":
        raise ValueError("softmax expanded primitive_program must end with HALT")
    if int(softmax_program.get("required_words", -1)) != len(program):
        raise ValueError("softmax primitive_program required_words mismatch")
    capacity_words = int(softmax_program.get("capacity_words", 0))
    expected_fit = len(program) <= capacity_words
    if bool(softmax_program.get("fits_current_capacity")) != expected_fit:
        raise ValueError("softmax primitive_program capacity result mismatch")

    if plan.get("group_state") not in ("model_only_full_attention", "software_group_measured_stages"):
        raise ValueError("unsupported attention group_state")
    if plan.get("group_state") == "software_group_measured_stages":
        if plan.get("group_cycle_policy") != "sum_measured_stages":
            raise ValueError("software grouped attention must use sum_measured_stages cycle policy")
        if plan.get("scale_mask_provenance") not in ("measured_npu_vector_bridge",):
            raise ValueError("software grouped attention must state scale/mask provenance")


def _validate_mask_plan(plan: dict[str, Any]) -> None:
    mask = plan["mask"]
    required = {
        "mask_policy",
        "seq_q",
        "seq_k",
        "valid_k",
        "tile_rows",
        "tile_cols",
        "valid_query_mask",
        "valid_lane_masks",
        "row_mask_words",
        "execution_state",
    }
    missing = sorted(required - set(mask))
    if missing:
        raise ValueError(f"attention mask plan missing fields {missing}")
    tile_rows = int(mask["tile_rows"])
    tile_cols = int(mask["tile_cols"])
    row_masks = mask["valid_lane_masks"]
    if tile_rows <= 0 or tile_cols <= 0 or len(row_masks) != tile_rows:
        raise ValueError("attention mask plan must provide one mask per physical query row")
    if any(int(row_mask) < 0 or int(row_mask) >= (1 << tile_cols) for row_mask in row_masks):
        raise ValueError("attention valid_lane_mask exceeds physical tile width")
    if mask["execution_state"] != plan["execution_state"]:
        raise ValueError("attention mask execution_state must match plan")
    seq_q = int(mask["seq_q"])
    seq_k = int(mask["seq_k"])
    valid_k = int(mask["valid_k"])
    if not 0 < seq_q <= tile_rows or not 0 < seq_k <= tile_cols:
        raise ValueError("attention logical shape exceeds one physical tile")
    if not 0 <= valid_k <= seq_k:
        raise ValueError("attention valid_k must be in 0..seq_k")
    if int(mask["valid_query_mask"]) != (1 << seq_q) - 1:
        raise ValueError("attention valid_query_mask does not match seq_q")
    causal = mask["mask_policy"] in ("causal", "causal_padding")
    padding = mask["mask_policy"] in ("padding", "causal_padding")
    if mask["mask_policy"] not in ("none", "causal", "padding", "causal_padding"):
        raise ValueError("unsupported attention mask_policy")
    key_limit = valid_k if padding else seq_k
    expected_row_masks = []
    for query in range(tile_rows):
        expected = 0
        if query < seq_q:
            for key in range(tile_cols):
                if key < seq_k and key < key_limit and (not causal or key <= query):
                    expected |= 1 << key
        expected_row_masks.append(expected)
    if [int(row_mask) for row_mask in row_masks] != expected_row_masks:
        raise ValueError("attention valid_lane_masks do not match logical mask policy")
    expected_words = [
        sum(expected_row_masks[row + offset] << (offset * 8) for offset in range(4))
        for row in (0, 4)
    ]
    if [int(word) for word in mask["row_mask_words"]] != expected_words:
        raise ValueError("attention row_mask_words do not match valid_lane_masks")

    full_mask = (1 << tile_cols) - 1
    current_executable = (
        seq_q == tile_rows
        and seq_k == tile_cols
        and all(int(row_mask) != 0 for row_mask in row_masks)
    )
    expected_state = "executable" if current_executable else "planned_not_executable"
    if plan["execution_state"] != expected_state:
        raise ValueError("attention execution_state does not match current RTL mask capability")
