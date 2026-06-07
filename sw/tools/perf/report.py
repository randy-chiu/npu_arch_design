from __future__ import annotations

import argparse
import html
import json
import re
import warnings
from pathlib import Path


PERF_PREFIX = "PERF_JOB "
PERF_TRACE_PREFIX = "PERF_TRACE "
DEFAULT_MODEL = {
    "matmul_tile": [8, 8, 8],
    "peak_macs_per_cycle": 64,
    "vector_lanes": 8,
    "matmul_array_control_cycles": 4,
    "data_mover_words_per_cycle": 4,
    "data_mover_setup_cycles_per_segment": 1,
    "trace_contract": {
        "command_events": {
            "NONE": 0, "DESC_DECODE": 1, "PROGRAM_MOVE": 2, "INPUT0_MOVE": 3,
            "INPUT1_MOVE": 4, "CHUNK_LAUNCH": 5, "COMPUTE_WAIT": 6,
            "OUTPUT_MOVE": 7, "JOB_RETIRE": 8, "ACC_CLEAR": 9,
            "ACC_DISABLE": 10, "PREFETCH_BANK_SELECT": 11,
        },
        "scheduler_wait_reasons": {
            "NONE": 0, "MATRIX_RESPONSE": 1, "PRIMITIVE_ACCEPT": 2,
            "PRIMITIVE_RESPONSE": 3,
        },
        "compute_control_events": {
            "NONE": 0, "PRIMITIVE_ACCEPT": 1, "PRIMITIVE_RESPONSE": 2,
            "ENGINE_START_ADAPTER": 3,
        },
    },
    "performance_contract": {},
}

TIMELINE_HIERARCHY = {
    "CPU firmware": {"parent": None, "depth": 0, "role": "software"},
    "NPU wrapper": {"parent": None, "depth": 0, "role": "host interface"},
    "NPU core": {"parent": None, "depth": 0, "role": "architecture group"},
    "Command processor": {"parent": "NPU core", "depth": 1, "role": "schedule/control"},
    "Uop scheduler": {"parent": "NPU core", "depth": 1, "role": "uop fetch/decode/dispatch"},
    "Data mover": {"parent": "NPU core", "depth": 1, "role": "data transfer"},
    "Compute cluster": {"parent": "NPU core", "depth": 1, "role": "compute"},
    "Compute cluster control": {"parent": "Compute cluster", "depth": 2, "role": "internal control FSM"},
    "Accumulator file": {"parent": "Compute cluster", "depth": 2, "role": "partial-sum storage"},
    "Local storage path": {"parent": "Compute cluster", "depth": 2, "role": "operand/result movement"},
    "Matrix engine": {"parent": "Compute cluster", "depth": 2, "role": "execution unit"},
    "Vector engine": {"parent": "Compute cluster", "depth": 2, "role": "execution unit"},
    "Reduction engine": {"parent": "Compute cluster", "depth": 2, "role": "execution unit"},
    "SFU": {"parent": "Compute cluster", "depth": 2, "role": "execution unit"},
}


def load_measurement_model(arch_path: Path, soc_path: Path) -> dict:
    arch = _read_jsonc(arch_path)
    soc = _read_jsonc(soc_path)
    return {
        "matmul_tile": [int(value) for value in arch["rtl"]["matmul_tile"]],
        "peak_macs_per_cycle": int(arch["compute"].get("mac_lanes", 64)),
        "vector_lanes": int(arch["rtl"]["softmax_vector_len"]),
        "matmul_array_control_cycles": int(arch["rtl"]["matmul_array_control_cycles"]),
        "data_mover_words_per_cycle": int(soc["npu_data_mover"]["words_per_cycle"]),
        "data_mover_setup_cycles_per_segment": int(
            soc["npu_data_mover"]["model_setup_cycles_per_segment"]
        ),
        "trace_contract": arch.get("trace_contract", {}),
        "performance_contract": arch.get("performance_contract", {}),
    }


def add_estimates(job: dict, model: dict = DEFAULT_MODEL) -> dict:
    if job.get("name") != "matmul":
        return job

    matmul_m, matmul_n, matmul_k = model["matmul_tile"]
    scalar_compute = matmul_m * matmul_n * matmul_k
    ideal_array_compute = matmul_k
    conservative_array_compute = ideal_array_compute + model["matmul_array_control_cycles"]
    measured_compute = int(job.get("core", {}).get("matmul", 0))
    non_matmul_cycles = int(job["total_cycles"]) - measured_compute

    job["estimates"] = {
        "matmul_shape": [matmul_m, matmul_n, matmul_k],
        "scalar_compute_cycles": scalar_compute,
        "ideal_array_compute_cycles": ideal_array_compute,
        "conservative_array_compute_cycles": conservative_array_compute,
        "measured_compute_cycles": measured_compute,
        "projected_total_with_conservative_array": non_matmul_cycles + conservative_array_compute,
    }
    return job


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def add_movement_estimates(job: dict, model: dict = DEFAULT_MODEL) -> dict:
    movement = job.get("movement")
    if not movement:
        return job

    transfer_fields = [
        "desc_words",
        "program_words",
        "input0_words",
        "input1_words",
        "output_words",
    ]
    words = {field: int(movement.get(field, 0)) for field in transfer_fields}
    active_segments = sum(1 for value in words.values() if value > 0)
    total_words = sum(words.values())
    measured_sram_cycles = int(movement.get("sram_read_cycles", 0)) + int(
        movement.get("sram_write_cycles", 0)
    )
    measured_host_window_cycles = int(movement.get("core_host_write_cycles", 0)) + int(
        movement.get("core_host_read_cycles", 0)
    )
    data_mover = job.get("data_mover", {})
    words_per_cycle = model["data_mover_words_per_cycle"]
    setup_cycles = model["data_mover_setup_cycles_per_segment"]
    ideal_burst_cycles = ceil_div(total_words, words_per_cycle)
    conservative_burst_cycles = sum(
        ceil_div(value, words_per_cycle)
        for value in words.values()
        if value > 0
    ) + active_segments * setup_cycles

    job["movement_estimates"] = {
        "total_words": total_words,
        "measured_sram_cycles": measured_sram_cycles,
        "measured_host_window_cycles": measured_host_window_cycles,
        "measured_data_mover_transfer_cycles": int(data_mover.get("transfer_cycles", 0)),
        "measured_data_mover_words": int(data_mover.get("words", 0)),
        "model_words_per_cycle": words_per_cycle,
        "model_setup_cycles_per_segment": setup_cycles,
        "ideal_burst_cycles": ideal_burst_cycles,
        "conservative_burst_cycles": conservative_burst_cycles,
    }
    return job


def add_timeline(job: dict) -> dict:
    cpu_lane = {
        "module": "CPU firmware",
        "spans": [
            {"label": "MMIO start", "start": 0, "end": 1, "cycles": 1, "kind": "work"},
            {
                "label": "Poll/wait for done",
                "start": 1,
                "end": int(job["total_cycles"]),
                "cycles": max(0, int(job["total_cycles"]) - 1),
                "kind": "wait",
            },
        ],
    }
    if job.get("source") == "architectural_perf_csr_snapshot":
        job["timeline"] = _with_timeline_hierarchy(_architectural_timeline(job, cpu_lane))
        job["timeline_provenance"] = {
            "summary_counters": "measured_architectural_perf_csr_snapshot",
            "span_placement": (
                "measured_cycle_event_trace"
                if job.get("cycle_trace")
                else "derived_from_reviewed_state_machine"
            ),
        }
        _validate_timeline(job, strict=True)
        job["timeline_validation"] = {"status": "passed"}
        return job

    wrapper_order = [
        ("desc_read", "Descriptor read", "work"),
        ("fetch_program", "Program movement wait", "wait"),
        ("fetch_input0", "Input0 movement wait", "wait"),
        ("fetch_input1", "Input1 movement wait", "wait"),
        ("start_core", "Core launch", "work"),
        ("wait_core", "Wait for core", "wait"),
        ("write_output", "Output movement wait", "wait"),
        ("done", "Done latch", "work"),
    ]
    core_order = [
        ("fetch", "Uop fetch/execute", "work"),
        ("matmul", "Matmul", "work"),
        ("done", "Done", "work"),
    ]

    wrapper_spans = []
    cursor = 0
    core_start = 0
    for key, label, kind in wrapper_order:
        value = int(job.get("wrapper", {}).get(key, 0))
        if key == "wait_core":
            core_start = cursor
        if value > 0:
            wrapper_spans.append(
                {
                    "label": label,
                    "start": cursor,
                    "end": cursor + value,
                    "cycles": value,
                    "kind": kind,
                }
            )
        cursor += value

    core_spans = []
    cursor = core_start
    for key, label, kind in core_order:
        value = int(job.get("core", {}).get(key, 0))
        if value > 0:
            core_spans.append(
                {
                    "label": label,
                    "start": cursor,
                    "end": cursor + value,
                    "cycles": value,
                    "kind": kind,
                }
            )
        cursor += value

    data_mover_spans = []
    wrapper_span_by_label = {}
    movement_labels = {
        "Program movement wait": "Program load",
        "Input0 movement wait": "Input0 load",
        "Input1 movement wait": "Input1 load",
        "Output movement wait": "Output store",
    }
    for span in wrapper_spans:
        wrapper_span_by_label[span["label"]] = span
        label = movement_labels.get(span["label"])
        if label is not None:
            data_mover_spans.append(
                {
                    "label": label,
                    "start": span["start"],
                    "end": span["end"],
                    "cycles": span["cycles"],
                    "kind": "work",
                }
            )
    job["timeline"] = _with_timeline_hierarchy([
        cpu_lane,
        {"module": "NPU wrapper", "spans": [_span("Forward CPU launch", 0, 1, "work")]},
        {"module": "NPU core", "spans": []},
        {"module": "Command processor", "spans": wrapper_spans},
        {"module": "Data mover", "spans": data_mover_spans},
        {"module": "Compute cluster", "spans": core_spans},
    ])
    issues = _validate_timeline(job, strict=False)
    job["timeline_validation"] = {
        "status": "legacy_not_accepted_as_architectural_evidence",
        "issues": issues,
    }
    return job


def _validate_timeline(job: dict, strict: bool) -> list[str]:
    total = int(job["total_cycles"])
    lanes = {lane["module"]: lane for lane in job.get("timeline", [])}
    issues = []

    def reject(message: str) -> None:
        if strict:
            raise ValueError(message)
        issues.append(message)

    for lane in lanes.values():
        previous_end = 0
        for span in lane.get("spans", []):
            start = int(span["start"])
            end = int(span["end"])
            cycles = int(span["cycles"])
            if start < 0 or end > total or end < start:
                reject(
                    f"timeline span outside job interval job={_job_id(job)} "
                    f"module={lane['module']} span={start}-{end} total={total}"
                )
            if cycles != end - start:
                reject(
                    f"timeline cycle mismatch job={_job_id(job)} module={lane['module']}"
                )
            if start < previous_end:
                reject(
                    f"overlapping spans in one lane job={_job_id(job)} module={lane['module']}"
                )
            previous_end = end

    trace = job.get("cycle_trace", [])
    performance_contract = job.get("_performance_contract", {})
    trace_contract = job.get("_trace_contract", {})
    for event in trace:
        cycle = int(event["cycle"])
        wait_reason = int(event.get("sched_wait_reason", 0))
        if bool(event.get("uop_wait")) != (wait_reason != 0):
            reject(
                f"scheduler wait reason mismatch job={_job_id(job)} cycle={cycle}"
            )
        if event.get("uop_active") and event.get("uop_wait"):
            reject(
                f"scheduler active/wait overlap job={_job_id(job)} cycle={cycle}"
            )
        if event.get("cmd_active") and event.get("cmd_wait"):
            reject(
                f"command active/wait overlap job={_job_id(job)} cycle={cycle}"
            )
        accumulator_events = [
            name for name in ("acc_clear", "acc_commit", "acc_readout") if event.get(name)
        ]
        if len(accumulator_events) > 1:
            reject(
                f"accumulator transaction overlap job={_job_id(job)} cycle={cycle} "
                f"events={accumulator_events}"
            )

    accumulator_contract = performance_contract.get("accumulator", {})
    for field, cycles_key in (
        ("acc_clear", "clear_cycles"),
        ("acc_commit", "commit_cycles"),
        ("acc_readout", "readout_cycles"),
    ):
        expected_cycles = accumulator_contract.get(cycles_key)
        if expected_cycles is None:
            continue
        for span in _event_spans(
            trace, lambda event, field=field: field if event.get(field) else None
        ):
            if int(span["cycles"]) != int(expected_cycles):
                reject(
                    f"{field} violates performance contract job={_job_id(job)} "
                    f"measured={span['cycles']} expected={expected_cycles}"
                )

    matrix_contract = performance_contract.get("matrix_operand_feed", {})
    expected_matrix_cycles = (
        int(matrix_contract.get("feed_cycles_per_k_slice", 0))
        * int(job.get("_matmul_k", 0))
    )
    if expected_matrix_cycles:
        for span in _event_spans(
            trace, lambda event: "matrix_active" if event.get("matrix_active") else None
        ):
            if int(span["cycles"]) != expected_matrix_cycles:
                reject(
                    f"matrix operand-feed transaction violates performance contract "
                    f"job={_job_id(job)} measured={span['cycles']} "
                    f"expected={expected_matrix_cycles}"
                )

    attention_contract = performance_contract.get("attention_row_storage", {})
    compute_events = trace_contract.get("compute_control_events", {})
    for event_name, cycles_key in (
        ("PRIMITIVE_ACCEPT", "row_read_cycles"),
        ("PRIMITIVE_RESPONSE", "row_write_cycles"),
    ):
        event_id = compute_events.get(event_name)
        expected_cycles = attention_contract.get(cycles_key)
        if event_id is None or expected_cycles is None:
            continue
        for span in _event_spans(
            trace,
            lambda event, event_id=event_id, event_name=event_name: (
                event_name if int(event.get("compute_ctrl_event", 0)) == int(event_id) else None
            ),
        ):
            if int(span["cycles"]) != int(expected_cycles):
                reject(
                    f"{event_name} violates Attention row performance contract "
                    f"job={_job_id(job)} measured={span['cycles']} expected={expected_cycles}"
                )

    if job.get("name") in ("attention_scale_mask_v1", "attention_softmax_v1"):
        parent_cycles = {
            cycle
            for span in lanes["Compute cluster"]["spans"]
            if span.get("kind", "work") == "work"
            for cycle in range(int(span["start"]), int(span["end"]))
        }
        child_cycles = {
            cycle
            for module in (
                "Compute cluster control",
                "Vector engine",
                "Reduction engine",
                "SFU",
            )
            for span in lanes.get(module, {}).get("spans", [])
            if span.get("kind", "work") == "work"
            for cycle in range(int(span["start"]), int(span["end"]))
        }
        if parent_cycles != child_cycles:
            reject(
                f"compute-cluster child timeline does not conserve active cycles "
                f"job={_job_id(job)} parent={len(parent_cycles)} children={len(child_cycles)}"
            )
    return issues


def _with_timeline_hierarchy(lanes: list[dict]) -> list[dict]:
    for lane in lanes:
        lane.update(TIMELINE_HIERARCHY.get(lane["module"], {"parent": None, "depth": 0, "role": "module"}))
    return lanes


def _event_spans(trace: list[dict], label_for_event, kind: str = "work") -> list[dict]:
    spans = []
    active_label = None
    active_start = 0
    previous_cycle = -2
    for event in trace:
        cycle = int(event["cycle"])
        label = label_for_event(event)
        if label != active_label or cycle != previous_cycle + 1:
            if active_label is not None:
                spans.append(_span(active_label, active_start, previous_cycle + 1, kind))
            active_label = label
            active_start = cycle
        previous_cycle = cycle
    if active_label is not None:
        spans.append(_span(active_label, active_start, previous_cycle + 1, kind))
    return spans


def _cycle_trace_timeline(job: dict, cpu_lane: dict) -> list[dict]:
    trace = job["cycle_trace"]
    job_name = job.get("name")
    trace_contract = job.get("_trace_contract", {})

    def semantic_name(group, value):
        return next(
            (
                name
                for name, event_id in trace_contract.get(group, {}).items()
                if int(event_id) == int(value)
            ),
            None,
        )

    def command_label(event):
        chunk = int(event["stream_chunk"])
        labels = {
            "DESC_DECODE": "Read/decode job descriptor",
            "PROGRAM_MOVE": "Wait for uop-program movement",
            "INPUT0_MOVE": "Wait for external A movement",
            "INPUT1_MOVE": "Wait for external B movement",
            "CHUNK_LAUNCH": f"Launch chunk {chunk}; latch selected compute bank",
            "COMPUTE_WAIT": "Wait for compute/prefetch completion",
            "OUTPUT_MOVE": "Wait for output movement",
            "JOB_RETIRE": "Retire job / publish done",
            "ACC_CLEAR": "Enable and clear accumulator",
            "ACC_DISABLE": "Disable accumulator",
            "PREFETCH_BANK_SELECT": "Select alternate prefetch/next-compute bank",
        }
        return labels.get(semantic_name("command_events", event.get("cmd_event", 0)))

    def mover_label(event):
        bank = int(event["dm_target_bank"])
        if event["dm_program"]:
            return (
                "Primitive-uop program: external SRAM -> instruction memory"
                if job_name in ("attention_scale_mask_v1", "attention_softmax_v1")
                else "Uop program: external SRAM -> instr_mem"
            )
        if event["dm_input_a"]:
            return (
                "Score tile: external SRAM -> compute-cluster local storage"
                if job_name in ("attention_scale_mask_v1", "attention_softmax_v1")
                else f"Chunk 0 A: external SRAM -> preload bank {bank}"
            )
        if event["dm_input_b"]:
            return f"Chunk 0 B: external SRAM -> preload bank {bank}"
        if event["dm_prefetch_a"]:
            return f"Chunk 1 A prefetch: external SRAM -> preload bank {bank}"
        if event["dm_prefetch_b"]:
            return f"Chunk 1 B prefetch: external SRAM -> preload bank {bank}"
        if event["dm_output"]:
            return (
                "Produced score/probability tile -> external SRAM"
                if job_name in ("attention_scale_mask_v1", "attention_softmax_v1")
                else "Output window -> external SRAM"
            )
        return None

    def scheduler_label(event):
        if event["uop_load"]:
            tensor = "A" if int(event["uop_tensor"]) == 0 else "B"
            return f"Fetch/decode/issue LOAD {tensor}: bind selected operand bank"
        if event["matrix_issue"]:
            return "Fetch/decode/issue MATMUL"
        if event.get("uop_exec"):
            opcode = int(event.get("uop_opcode", 0))
            row = int(event.get("uop_tensor", 0))
            lane = int(event.get("uop_buffer", 0))
            labels = {
                4: f"Fetch/decode/issue reduction max row {row}",
                5: f"Fetch/decode/issue vector subtract row {row}",
                6: f"Fetch/decode/issue SFU EXP row {row}, lane {lane}",
                7: f"Fetch/decode/issue reduction sum row {row}",
                8: f"Fetch/decode/issue SFU reciprocal row {row}",
                9: f"Fetch/decode/issue fixed score scale row {row}",
                10: f"Fetch/decode/issue vector clamp row {row}",
                11: f"Fetch/decode/issue vector normalize row {row}",
            }
            return labels.get(opcode, f"Fetch/decode/issue primitive opcode {opcode}")
        if event["uop_store"]:
            return (
                "Fetch/decode/issue final accumulator -> C-window STORE"
                if event["output_store_enable"]
                else "Decode/skip non-final accumulator STORE"
            )
        if event["uop_wait"]:
            wait_labels = {
                "MATRIX_RESPONSE": "Wait for Matrix response",
                "PRIMITIVE_ACCEPT": "Wait for Compute cluster command acceptance",
                "PRIMITIVE_RESPONSE": "Wait for issued primitive response",
            }
            return wait_labels.get(
                semantic_name("scheduler_wait_reasons", event.get("sched_wait_reason", 0)),
                "Wait for typed Scheduler dependency",
            )
        if event["uop_active"]:
            return "HALT/completion control"
        return None

    command_spans = []
    for span in _event_spans(trace, command_label):
        span["kind"] = "wait" if span["label"].startswith("Wait") else "work"
        command_spans.append(span)
    scheduler_spans = []
    for span in _event_spans(trace, scheduler_label):
        span["kind"] = "wait" if span["label"].startswith("Wait") else "work"
        scheduler_spans.append(span)
    compute_spans = _event_spans(
        trace,
        lambda event: (
            "Compute-cluster active"
            if event["core_active"]
            else ("Wait for next-chunk A/B prefetch" if event["core_wait_data"] else None)
        ),
    )
    for span in compute_spans:
        span["kind"] = "wait" if span["label"].startswith("Wait") else "work"

    module_lanes = []
    if job_name in ("attention_scale_mask_v1", "attention_softmax_v1"):
        active_core_cycles = [int(event["cycle"]) for event in trace if event["core_active"]]

        vector_names = {
            1: "Vector subtract row {row}: score - row maximum",
            3: "Vector normalize row {row}: exp * reciprocal(sum)",
            5: "Vector clamp row {row}: limit shifted scores to EXP input range",
            6: "Vector fixed scale row {row}: apply 1/sqrt(head_dim)",
        }
        reduction_names = {
            0: "Reduction max row {row}: find stable-softmax row maximum",
            1: "Reduction sum row {row}: sum exponentials",
            2: "Reduction sum-of-squares row {row}",
        }
        sfu_names = {
            0: "SFU EXP row {row}, lane {lane}: exponentiate one shifted score",
            1: "SFU reciprocal row {row}: compute 1/sum(exp)",
            2: "SFU reciprocal-sqrt row {row}",
        }

        def primitive_label(event, active_field, op_field, names):
            if not event.get(active_field):
                return None
            template = names.get(int(event[op_field]), f"Unknown primitive op {event[op_field]}")
            return template.format(
                row=int(event.get("primitive_row", 0)),
                lane=int(event.get("primitive_lane", 0)),
            )

        def control_label(event):
            event_name = semantic_name(
                "compute_control_events", event.get("compute_ctrl_event", 0)
            )
            row = int(event.get("primitive_row", 0))
            lane = int(event.get("primitive_lane", 0))
            if event_name == "PRIMITIVE_ACCEPT":
                return f"Accept/route/start primitive row {row}, lane {lane}"
            if event_name == "PRIMITIVE_RESPONSE":
                return f"Capture/retire primitive response row {row}, lane {lane}"
            if event_name == "ENGINE_START_ADAPTER":
                return f"Internal start/done adapter latency row {row}, lane {lane}"
            return None

        module_lanes.extend(
            [
                {
                    "module": "Compute cluster control",
                    "spans": _event_spans(trace, control_label),
                },
                {
                    "module": "Vector engine",
                    "spans": _event_spans(
                        trace,
                        lambda event: primitive_label(
                            event, "vector_active", "vector_op", vector_names
                        ),
                    ),
                },
                {
                    "module": "Reduction engine",
                    "spans": _event_spans(
                        trace,
                        lambda event: primitive_label(
                            event, "reduction_active", "reduction_op", reduction_names
                        ),
                    ),
                },
                {
                    "module": "SFU",
                    "spans": _event_spans(
                        trace,
                        lambda event: primitive_label(event, "sfu_active", "sfu_op", sfu_names),
                    ),
                },
            ]
        )

    base_lanes = [
        cpu_lane,
        {"module": "NPU wrapper", "spans": [_span("Forward CPU launch", 0, 1, "work")]},
        {"module": "NPU core", "spans": []},
        {"module": "Command processor", "spans": command_spans},
        {
            "module": "Uop scheduler",
            "spans": scheduler_spans,
            "measured_active_cycles": int(job.get("uop_scheduler", {}).get("active_cycles", 0)),
            "measured_wait_cycles": int(job.get("uop_scheduler", {}).get("wait_cycles", 0)),
        },
        {"module": "Data mover", "spans": _event_spans(trace, mover_label)},
        {"module": "Compute cluster", "spans": compute_spans},
    ]
    matrix_spans = _event_spans(
        trace, lambda event: "Matrix datapath active" if event["matrix_active"] else None
    )
    accumulator_spans = _event_spans(
        trace,
        lambda event: (
            "Clear resident partial sum"
            if event["acc_clear"]
            else (
                "Commit/add Matrix result into resident partial sum"
                if event["acc_commit"]
                else (
                    "Read/copy resident sum into C output window"
                    if event["acc_readout"]
                    else None
                )
            )
        ),
    )
    if matrix_spans:
        base_lanes.append({"module": "Matrix engine", "spans": matrix_spans})
    if accumulator_spans:
        base_lanes.append({"module": "Accumulator file", "spans": accumulator_spans})
    return base_lanes + module_lanes


def _architectural_timeline(job: dict, cpu_lane: dict) -> list[dict]:
    if job.get("cycle_trace"):
        return _cycle_trace_timeline(job, cpu_lane)

    total = int(job["total_cycles"])
    core_cycles = int(job.get("core", {}).get("total", 0))
    matmul_cycles = int(job.get("core", {}).get("matmul", 0))
    mover = job.get("data_mover", {})
    command = job.get("command_processor", {})
    uop_scheduler = job.get("uop_scheduler", {})
    measured_overlap_cycles = int(mover.get("compute_overlap_cycles", 0))
    wait_data_cycles = int(job.get("core", {}).get("wait_data_cycles", 0))
    local_active_cycles = int(job.get("core", {}).get("local_active_cycles", 0))
    program_cycles = int(mover.get("program_cycles", 0))
    initial_input_cycles = int(mover.get("initial_input_cycles", 0))
    prefetch_cycles = int(mover.get("prefetch_cycles", 0))
    read_cycles = ceil_div(int(mover.get("read_words", 0)), 4)
    write_cycles = ceil_div(int(mover.get("write_words", 0)), 4)
    output_cycles = int(mover.get("output_cycles", write_cycles))
    initial_read_cycles = program_cycles + initial_input_cycles
    descriptor_cycles = min(11, max(1, total - core_cycles - read_cycles - write_cycles - 2))
    prefetch_transition_cycles = max(
        0, prefetch_cycles - measured_overlap_cycles - wait_data_cycles
    )
    core_start = min(
        max(descriptor_cycles + initial_read_cycles + prefetch_transition_cycles, 1),
        max(1, total - core_cycles - write_cycles - 1),
    )
    core_end = min(total, core_start + core_cycles + wait_data_cycles)
    write_start = max(core_end, total - write_cycles - 1)

    wrapper_spans = [
        _span("Descriptor read", 0, descriptor_cycles, "work"),
        _span("Wait for input/program movement", descriptor_cycles, core_start, "wait"),
        _span("Wait for execution", core_start, core_end, "wait"),
        _span("Wait for output movement", write_start, min(total - 1, write_start + write_cycles), "wait"),
        _span("Done latch", max(0, total - 1), total, "work"),
    ]
    program_start = descriptor_cycles
    initial_input_start = program_start + program_cycles
    prefetch_transition_start = initial_input_start + initial_input_cycles
    data_mover_spans = []
    if program_cycles:
        data_mover_spans.append(
            _span("Program load", program_start, program_start + program_cycles, "work")
        )
    if initial_input_cycles:
        data_mover_spans.append(
            _span("Initial chunk A/B load", initial_input_start, initial_input_start + initial_input_cycles, "work")
        )
    if prefetch_transition_cycles:
        data_mover_spans.append(
            _span("Next-chunk prefetch during control transition", prefetch_transition_start, prefetch_transition_start + prefetch_transition_cycles, "work")
        )
    if measured_overlap_cycles > 0:
        overlap_label = "Measured K prefetch overlap" if job.get("name") == "matmul_k_stream" else "Measured movement/compute overlap"
        data_mover_spans.append(
            _span(overlap_label, core_start, min(core_end, core_start + measured_overlap_cycles), "work")
        )
    if wait_data_cycles > 0:
        data_mover_spans.append(
            _span(
                "Prefetch blocking next chunk",
                core_start + measured_overlap_cycles,
                core_start + measured_overlap_cycles + wait_data_cycles,
                "work",
            )
        )
    data_mover_spans.append(
        _span("Write/store", write_start, min(total, write_start + output_cycles), "work")
    )

    module_lanes = []
    compute_spans = [_span("Measured compute active", core_start, core_start + core_cycles, "work")]
    matrix_spans = []
    accumulator_spans = []
    local_spans = []
    scheduler_spans = []
    scheduler_active_cycles = int(uop_scheduler.get("active_cycles", 0))
    scheduler_wait_cycles = int(uop_scheduler.get("wait_cycles", 0))
    if job.get("name") == "matmul_k_stream" and wait_data_cycles > 0:
        chunk_matrix_cycles = matmul_cycles // 2
        first_chunk_compute_cycles = chunk_matrix_cycles + 1
        second_chunk_compute_cycles = core_cycles - first_chunk_compute_cycles
        first_end = core_start + first_chunk_compute_cycles
        second_start = first_end + wait_data_cycles
        compute_spans = [
            _span("Chunk 0 compute active", core_start, first_end, "work"),
            _span("Wait for prefetched A/B", first_end, second_start, "wait"),
            _span("Chunk 1 compute active", second_start, second_start + second_chunk_compute_cycles, "work"),
        ]
        matrix_spans = [
            _span("Chunk 0 matrix datapath active", core_start, core_start + chunk_matrix_cycles, "work"),
            _span("Chunk 1 matrix datapath active", second_start, second_start + chunk_matrix_cycles, "work"),
        ]
        accumulator_spans = [
            _span("Chunk 0 commit/add into resident partial sum", first_end - 1, first_end, "work"),
            _span("Chunk 1 commit/add into resident partial sum", second_start + chunk_matrix_cycles, second_start + chunk_matrix_cycles + 1, "work"),
            _span("Chunk 1 read/copy resident sum into C output window", second_start + second_chunk_compute_cycles - 1, second_start + second_chunk_compute_cycles, "work"),
        ]
        active_per_chunk = scheduler_active_cycles // 2
        wait_per_chunk = scheduler_wait_cycles // 2
        issue_cycles = active_per_chunk // 2
        complete_cycles = active_per_chunk - issue_cycles
        scheduler_spans = [
            _span("Chunk 0 fetch/decode/issue", core_start, core_start + issue_cycles, "work"),
            _span("Chunk 0 wait for Matrix", core_start + issue_cycles, core_start + issue_cycles + wait_per_chunk, "wait"),
            _span("Chunk 0 completion control", core_start + issue_cycles + wait_per_chunk, core_start + issue_cycles + wait_per_chunk + complete_cycles, "work"),
            _span("Chunk 1 fetch/decode/issue", second_start, second_start + issue_cycles, "work"),
            _span("Chunk 1 wait for Matrix", second_start + issue_cycles, second_start + issue_cycles + wait_per_chunk, "wait"),
            _span("Chunk 1 completion control", second_start + issue_cycles + wait_per_chunk, second_start + issue_cycles + wait_per_chunk + complete_cycles, "work"),
        ]
    if (scheduler_active_cycles or scheduler_wait_cycles) and not scheduler_spans:
        issue_cycles = scheduler_active_cycles // 2
        complete_cycles = scheduler_active_cycles - issue_cycles
        scheduler_spans = [
            _span("Fetch/decode/issue", core_start, core_start + issue_cycles, "work"),
            _span("Wait for execution engine", core_start + issue_cycles, core_start + issue_cycles + scheduler_wait_cycles, "wait"),
            _span("Completion control", core_start + issue_cycles + scheduler_wait_cycles, core_start + issue_cycles + scheduler_wait_cycles + complete_cycles, "work"),
        ]
    if matmul_cycles:
        if not matrix_spans:
            matrix_spans = [_span("Measured matrix datapath active", core_start, core_start + matmul_cycles, "work")]
            accumulator_spans = [
                _span("Measured commit/add into resident partial sum", core_start + matmul_cycles, core_start + matmul_cycles + 1, "work"),
                _span("Measured read/copy resident sum into C output window", core_start + core_cycles - 1, core_start + core_cycles, "work"),
            ]
        module_lanes.append(
            {"module": "Matrix engine", "spans": matrix_spans}
        )
        module_lanes.append(
            {"module": "Accumulator file", "spans": accumulator_spans}
        )
    elif job.get("name") == "attention_scale_mask_v1":
        module_lanes.append(
            {"module": "Vector engine", "spans": [_span("8 row requant operations", core_start, core_end, "work")]}
        )
    elif job.get("name") == "attention_softmax_v1":
        # The current state machine is serial: vector -> reduction -> SFU per row.
        # Split measured core duration by reviewed primitive issue counts.
        issue_counts = [("Vector engine", 32), ("Reduction engine", 16), ("SFU", 72)]
        issue_total = sum(count for _, count in issue_counts)
        cursor = core_start
        for index, (module, count) in enumerate(issue_counts):
            end = core_end if index == len(issue_counts) - 1 else cursor + round(core_cycles * count / issue_total)
            module_lanes.append(
                {"module": module, "spans": [_span(f"Derived {module.lower()} occupancy", cursor, end, "work")]}
            )
            cursor = end

    return [
        cpu_lane,
        {"module": "NPU wrapper", "spans": [_span("Forward CPU launch", 0, 1, "work")]},
        {"module": "NPU core", "spans": []},
        {
            "module": "Command processor",
            "spans": [span for span in wrapper_spans if span["cycles"] > 0],
            "measured_active_cycles": int(command.get("active_cycles", 0)),
            "measured_wait_cycles": int(command.get("wait_cycles", 0)),
        },
        {
            "module": "Uop scheduler",
            "spans": scheduler_spans,
            "measured_active_cycles": scheduler_active_cycles,
            "measured_wait_cycles": scheduler_wait_cycles,
        },
        {"module": "Data mover", "spans": [span for span in data_mover_spans if span["cycles"] > 0]},
        {"module": "Compute cluster", "spans": compute_spans},
        *module_lanes,
    ]


def _span(label: str, start: int, end: int, kind: str) -> dict:
    end = max(start, end)
    return {"label": label, "start": start, "end": end, "cycles": end - start, "kind": kind}


def parse_perf_log(path: Path, manifest_path: Path | None = None, model: dict = DEFAULT_MODEL) -> dict:
    raw_jobs = []
    traces: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(PERF_PREFIX):
                raw_jobs.append(json.loads(line[len(PERF_PREFIX) :]))
            elif line.startswith(PERF_TRACE_PREFIX):
                event = json.loads(line[len(PERF_TRACE_PREFIX) :])
                traces.setdefault(int(event["job_id"]), []).append(event)
    jobs = []
    for job in raw_jobs:
        job_trace = [
            event
            for event in traces.get(_job_id(job), [])
            if int(event["cycle"]) < int(job.get("total_cycles", 0))
        ]
        if job_trace and len(job_trace) >= max(1, int(job.get("total_cycles", 0)) - 1):
            job["cycle_trace"] = job_trace
            job["_trace_contract"] = model.get("trace_contract", {})
            job["_performance_contract"] = model.get("performance_contract", {})
            job["_matmul_k"] = int(model.get("matmul_tile", [0, 0, 0])[2])
        job = add_timeline(add_movement_estimates(add_estimates(job, model), model))
        job.pop("cycle_trace", None)
        job.pop("_trace_contract", None)
        job.pop("_performance_contract", None)
        job.pop("_matmul_k", None)
        jobs.append(job)
    if not jobs:
        raise ValueError(f"no {PERF_PREFIX.strip()} records found in {path}")
    workload_manifest = None
    if manifest_path is not None:
        workload_manifest = load_workload_manifest(manifest_path)
        workloads = workloads_from_manifest(jobs, workload_manifest)
        model_only_workloads = model_only_workloads_from_manifest(workload_manifest)
    else:
        warnings.warn(
            "no workload manifest provided; falling back to order-based workload inference",
            UserWarning,
        )
        workloads = infer_workloads(jobs)
        model_only_workloads = []
    highlights = build_highlights(workloads, jobs)
    performance_source = (
        "measured_architectural_perf_csr_snapshot"
        if all(job.get("source") == "architectural_perf_csr_snapshot" for job in jobs)
        else "measured_rtl_perf_job_counters"
    )
    return {
        "schema": "npu_perf_report_v0",
        "source_log": str(path),
        "source": {"performance": performance_source},
        "performance_contract": model.get("performance_contract", {}),
        "workload_manifest": (
            {
                "schema": workload_manifest["schema"],
                "id": workload_manifest["manifest_id"],
                "run_name": workload_manifest["run_name"],
                "workload_profile": workload_manifest.get("workload_profile", "unspecified"),
                "source": str(manifest_path),
            }
            if workload_manifest is not None
            else None
        ),
        "summary": {
            "jobs": len(jobs),
            "workloads": len(workloads),
            "total_cycles": sum(job["total_cycles"] for job in jobs),
            "max_job_cycles": max(job["total_cycles"] for job in jobs),
            "transformer": transformer_summary(workloads, model_only_workloads),
        },
        "highlights": highlights,
        "workloads": workloads,
        "model_only_workloads": model_only_workloads,
        "jobs": jobs,
    }


def load_workload_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "npu_workload_manifest_v0":
        raise ValueError(f"{path}: unsupported workload manifest schema {manifest.get('schema')!r}")
    for required in ("manifest_id", "run_name", "jobs"):
        if required not in manifest:
            raise ValueError(f"{path}: workload manifest is missing required field {required!r}")
    if not isinstance(manifest["jobs"], list) or not manifest["jobs"]:
        raise ValueError(f"{path}: workload manifest jobs must be a non-empty array")
    seen: set[int] = set()
    for entry in manifest["jobs"]:
        for required in ("job_id", "workload", "op", "role"):
            if required not in entry:
                raise ValueError(f"{path}: workload manifest job is missing required field {required!r}")
        job_id = entry["job_id"]
        if not isinstance(job_id, int) or job_id in seen:
            raise ValueError(f"{path}: workload manifest has invalid or duplicate job_id {job_id!r}")
        seen.add(job_id)
    return manifest


def _read_jsonc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return json.loads("\n".join(_strip_line_comment(line) for line in text.splitlines()))


def _strip_line_comment(line: str) -> str:
    in_string = False
    escaped = False
    for i in range(len(line) - 1):
        ch = line[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string and line[i : i + 2] == "//":
            return line[:i]
    return line


def _job_id(job: dict, require_explicit: bool = False) -> int:
    if "job_id" in job:
        return int(job["job_id"])
    if not require_explicit and "id" in job:
        return int(job["id"])
    raise ValueError("PERF_JOB record is missing required job_id for manifest correlation")


def workloads_from_manifest(jobs: list[dict], manifest: dict) -> list[dict]:
    job_by_id: dict[int, dict] = {}
    for job in jobs:
        job_id = _job_id(job, require_explicit=True)
        if job_id in job_by_id:
            raise ValueError(f"duplicate PERF_JOB job_id {job_id}")
        job_by_id[job_id] = job

    entry_by_id = {entry["job_id"]: entry for entry in manifest["jobs"]}
    missing = sorted(set(entry_by_id) - set(job_by_id))
    unexpected = sorted(set(job_by_id) - set(entry_by_id))
    if missing or unexpected:
        raise ValueError(
            "workload manifest/PERF_JOB mismatch: "
            f"missing job_id(s) {missing}; unexpected job_id(s) {unexpected}"
        )

    ordered_workloads: list[str] = []
    grouped_jobs: dict[str, list[dict]] = {}
    first_entry: dict[str, dict] = {}
    for entry in manifest["jobs"]:
        job_id = entry["job_id"]
        job = job_by_id[job_id]
        if job.get("name") != entry["op"]:
            raise ValueError(
                f"workload manifest/PERF_JOB mismatch for job_id {job_id}: "
                f"manifest op {entry['op']!r}, PERF_JOB name {job.get('name')!r}"
            )
        workload = entry["workload"]
        if workload not in grouped_jobs:
            ordered_workloads.append(workload)
            grouped_jobs[workload] = []
            first_entry[workload] = entry
        grouped_jobs[workload].append(job)

    definitions = manifest.get("workload_metadata", {})
    summaries = []
    for workload in ordered_workloads:
        definition = definitions.get(workload, {})
        entry = first_entry[workload]
        summaries.append(
            _workload_summary(
                workload,
                grouped_jobs[workload],
                definition.get("kind", entry["role"]),
                metadata=definition.get("metadata", {}),
            )
        )
    return summaries


def model_only_workloads_from_manifest(manifest: dict) -> list[dict]:
    summaries = []
    for name, definition in manifest.get("workload_metadata", {}).items():
        metadata = definition.get("metadata", {})
        if not metadata.get("model_only"):
            continue
        summary = {
            "name": name,
            "kind": definition.get("kind", "model_only"),
            "job_ids": [],
            "jobs": 0,
            "total_cycles": 0,
            "max_job_cycles": 0,
            "core_matmul_cycles": 0,
            "movement_sram_cycles": 0,
            "wrapper": {},
            "core": {},
            "movement": {},
            "data_mover": {},
            "metadata": metadata,
        }
        summary["transformer_metrics"] = transformer_metrics(summary, DEFAULT_MODEL)
        summaries.append(summary)
    return summaries


def transformer_summary(workloads: list[dict], model_only_workloads: list[dict]) -> dict:
    all_workloads = workloads + model_only_workloads
    prefill_cycles = sum(
        int(workload.get("total_cycles", 0))
        for workload in all_workloads
        if workload.get("metadata", {}).get("workload_family") == "transformer_prefill"
    )
    decode_cycles = sum(
        int(workload.get("total_cycles", 0))
        for workload in all_workloads
        if workload.get("metadata", {}).get("workload_family") == "transformer_decode"
    )
    kv_read = sum(
        int(workload.get("transformer_metrics", {}).get("kv_read_bytes") or 0)
        for workload in all_workloads
    )
    kv_write = sum(
        int(workload.get("transformer_metrics", {}).get("kv_write_bytes") or 0)
        for workload in all_workloads
    )
    return {
        "prefill_cycles": prefill_cycles if prefill_cycles else None,
        "decode_cycles_per_token": decode_cycles if decode_cycles else None,
        "kv_read_bytes": kv_read,
        "kv_write_bytes": kv_write,
        "bytes_per_token": kv_read + kv_write if (kv_read or kv_write) else None,
    }


def build_highlights(workloads: list[dict], jobs: list[dict]) -> list[dict]:
    highlights = []
    workload_by_name = {workload["name"]: workload for workload in workloads}
    fc1_full = workload_by_name.get("real_mnist_cnn_fc1_full_k_stream_layer")
    if not fc1_full:
        fc1_full = workload_by_name.get("real_mnist_cnn_fc1_full_k_stream_tile0")
    baseline = fc1_full.get("metadata", {}).get("comparison_baseline") if fc1_full else None
    if fc1_full and baseline:
        old_serial_baseline = int(baseline["cycles_per_job"]) * int(fc1_full.get("jobs", 1))
        total_cycles = int(fc1_full["total_cycles"])
        cycles_saved = old_serial_baseline - total_cycles
        improvement_pct = (cycles_saved * 100.0 / old_serial_baseline) if old_serial_baseline else 0.0
        overlap_cycles = None
        for job in jobs:
            if _job_id(job) in fc1_full.get("job_ids", []):
                for lane in job.get("timeline", []):
                    if lane.get("module") == "Data mover":
                        for span in lane.get("spans", []):
                            if span.get("label") == "Measured K prefetch overlap":
                                if overlap_cycles is None:
                                    overlap_cycles = 0
                                overlap_cycles += int(span.get("cycles", 0))
        highlights.append(
            {
                "title": "FC1 K-stream ping-pong overlap",
                "workload": fc1_full["name"],
                "baseline_id": baseline["id"],
                "before_cycles": old_serial_baseline,
                "after_cycles": total_cycles,
                "cycles_saved": cycles_saved,
                "improvement_pct": round(improvement_pct, 1),
                "overlap_cycles": overlap_cycles,
                "core_matmul_cycles": int(fc1_full.get("core_matmul_cycles", 0)),
                "data_mover_words": int(fc1_full.get("data_mover", {}).get("words", 0)),
                "data_mover_transfer_cycles": int(
                    fc1_full.get("data_mover", {}).get("transfer_cycles", 0)
                ),
                "summary": (
                    "A/B ping-pong overlaps K-chunk prefetch with core execution; "
                    "moved words and core matmul cycles stay stable."
                ),
            }
        )
    return highlights


def infer_workloads(jobs: list[dict]) -> list[dict]:
    workloads = []
    if not jobs:
        return workloads

    if len(jobs) >= 1:
        workloads.append(_workload_summary("operator_smoke_matmul", jobs[0:1], "operator_smoke"))
    cursor = 1
    if cursor < len(jobs) and jobs[cursor].get("name") == "softmax":
        # Historical log replay only: the obsolete Phase-0 Softmax job is not
        # promoted into a current workload.
        cursor += 1
    classifier_tile_count = 16
    if cursor + classifier_tile_count <= len(jobs):
        candidate = jobs[cursor : cursor + classifier_tile_count]
        if all(job.get("name") == "matmul" for job in candidate):
            workloads.append(
                _workload_summary(
                    "digits_linear_classifier",
                    candidate,
                    "model",
                    metadata={
                        "input": "test/assets/digits_realistic/digit_2_gray.pgm",
                        "graph": "test/graphs/digits_classifier.json",
                        "tile_graph": "test/graphs/digits_classifier_rtl_tile.json",
                        "tile_jobs": classifier_tile_count,
                        "description": "8x8 grayscale digit image, 16 matmul tiles, CPU-side partial-sum accumulation and argmax",
                    },
                )
            )
            cursor += classifier_tile_count

    real_mnist_fc1_tile_count = 1
    real_mnist_fc1_k_stream_count = 1
    real_mnist_fc2_tile_count = 32
    if cursor + real_mnist_fc1_tile_count + real_mnist_fc2_tile_count <= len(jobs):
        candidate = jobs[cursor : cursor + real_mnist_fc1_tile_count]
        if all(job.get("name") == "matmul" for job in candidate):
            workloads.append(
                _workload_summary(
                    "real_mnist_cnn_fc1_tile0",
                    candidate,
                    "model_layer_tile",
                    metadata={
                        "input": "test/external/mnist/t10k-images-idx3-ubyte.gz sample 0",
                        "graph": "test/graphs/real_mnist_cnn.json",
                        "weights": "test/external/mnist_cnn/mnist-cnn.safetensors",
                        "tile_jobs": real_mnist_fc1_tile_count,
                        "description": "First quantized fc1 8x8x8 tile from the original CNN sample 0 path; validates SoC RTL staging/arithmetic, not full fc1 layer execution",
                    },
                )
            )
            cursor += real_mnist_fc1_tile_count

    if cursor + real_mnist_fc1_k_stream_count + real_mnist_fc2_tile_count <= len(jobs):
        candidate = jobs[cursor : cursor + real_mnist_fc1_k_stream_count]
        if all(job.get("name") == "matmul_k_stream" for job in candidate):
            workloads.append(
                _workload_summary(
                    "real_mnist_cnn_fc1_k_stream_smoke",
                    candidate,
                    "model_layer_tile",
                    metadata={
                        "input": "test/external/mnist/t10k-images-idx3-ubyte.gz sample 0",
                        "graph": "test/graphs/real_mnist_cnn.json",
                        "weights": "test/external/mnist_cnn/mnist-cnn.safetensors",
                        "tile_jobs": real_mnist_fc1_k_stream_count,
                        "description": "Multi-chunk quantized fc1 K-streaming descriptor smoke; validates accumulator residency within one NPU job, not full fc1 layer execution",
                    },
                )
            )
            cursor += real_mnist_fc1_k_stream_count

    if cursor + 1 + real_mnist_fc2_tile_count <= len(jobs):
        remaining_before_fc2 = len(jobs) - cursor - real_mnist_fc2_tile_count
        candidate = jobs[cursor : cursor + remaining_before_fc2]
        if remaining_before_fc2 > 0 and all(job.get("name") == "matmul_k_stream" for job in candidate):
            workload_name = (
                "real_mnist_cnn_fc1_full_k_stream_layer"
                if remaining_before_fc2 == 16
                else "real_mnist_cnn_fc1_full_k_stream_tile0"
            )
            workloads.append(
                _workload_summary(
                    workload_name,
                    candidate,
                    "model_layer",
                    metadata={
                        "input": "test/external/mnist/t10k-images-idx3-ubyte.gz sample 0",
                        "graph": "test/graphs/real_mnist_cnn.json",
                        "weights": "test/external/mnist_cnn/mnist-cnn.safetensors",
                        "tile_jobs": remaining_before_fc2,
                        "k_chunks": 1152,
                        "comparison_baseline": {
                            "id": "npu_v0_a2_serial_k_stream_proxy",
                            "cycles_per_job": 58784,
                        },
                        "description": (
                            "Full quantized fc1 layer across all 16 output N tiles"
                            if remaining_before_fc2 == 16
                            else "Full quantized fc1 single N-tile K-streaming descriptor; validates 1152 K chunks accumulated inside one NPU job"
                        ),
                    },
                )
            )
            cursor += remaining_before_fc2

    if cursor + real_mnist_fc2_tile_count <= len(jobs):
        candidate = jobs[cursor : cursor + real_mnist_fc2_tile_count]
        if all(job.get("name") == "matmul" for job in candidate):
            workloads.append(
                _workload_summary(
                    "real_mnist_cnn_fc2",
                    candidate,
                    "model_layer",
                    metadata={
                        "input": "test/external/mnist/t10k-images-idx3-ubyte.gz sample 0",
                        "graph": "test/graphs/real_mnist_cnn.json",
                        "weights": "test/external/mnist_cnn/mnist-cnn.safetensors",
                        "tile_jobs": real_mnist_fc2_tile_count,
                        "description": "Original CNN conv/fc1 path precomputed on CPU/tool side; fc2 quantized into 32 current-RTL-compatible matmul tiles",
                    },
                )
            )
            cursor += real_mnist_fc2_tile_count

    if cursor < len(jobs):
        workloads.append(_workload_summary(f"unclassified_jobs_{cursor + 1}", jobs[cursor:], "unknown"))
    return workloads


def _workload_summary(
    name: str,
    jobs: list[dict],
    kind: str,
    metadata: dict | None = None,
) -> dict:
    movement_totals: dict[str, int] = {}
    data_mover_totals: dict[str, int] = {}
    wrapper_totals: dict[str, int] = {}
    core_totals: dict[str, int] = {}
    for job in jobs:
        _accumulate_counter_dict(wrapper_totals, job.get("wrapper", {}))
        _accumulate_counter_dict(core_totals, job.get("core", {}))
        _accumulate_counter_dict(movement_totals, job.get("movement", {}))
        _accumulate_counter_dict(data_mover_totals, job.get("data_mover", {}))

    total_cycles = sum(int(job["total_cycles"]) for job in jobs)
    core_matmul_cycles = int(core_totals.get("matmul", 0))
    movement_cycles = int(movement_totals.get("sram_read_cycles", 0)) + int(
        movement_totals.get("sram_write_cycles", 0)
    )
    summary = {
        "name": name,
        "kind": kind,
        "job_ids": [_job_id(job) for job in jobs],
        "jobs": len(jobs),
        "total_cycles": total_cycles,
        "max_job_cycles": max(int(job["total_cycles"]) for job in jobs),
        "core_matmul_cycles": core_matmul_cycles,
        "movement_sram_cycles": movement_cycles,
        "wrapper": wrapper_totals,
        "core": core_totals,
        "movement": movement_totals,
        "data_mover": data_mover_totals,
        "metadata": metadata or {},
    }
    summary["transformer_metrics"] = transformer_metrics(summary, DEFAULT_MODEL)
    return summary


def transformer_metrics(workload: dict, model: dict = DEFAULT_MODEL) -> dict:
    metadata = workload.get("metadata", {})
    shape = metadata.get("logical_shape", {})
    matrix_active = int(workload.get("core_matmul_cycles", 0))
    peak_macs_per_cycle = int(model.get("peak_macs_per_cycle", 64))
    effective_mac_ops = None
    shape_class = metadata.get("shape_class")
    if all(key in shape for key in ("m", "n", "k")):
        m_dim = int(shape["m"])
        n_dim = int(shape["n"])
        k_dim = int(shape["k"])
        effective_mac_ops = m_dim * n_dim * k_dim * max(1, int(workload.get("jobs", 0)))
        if shape_class is None:
            shape_class = classify_matrix_shape(m_dim, n_dim, k_dim)

    peak_mac_capacity = matrix_active * peak_macs_per_cycle if matrix_active else None
    matrix_utilization = (
        round(float(effective_mac_ops) / float(peak_mac_capacity), 6)
        if effective_mac_ops is not None and peak_mac_capacity
        else None
    )
    is_model_only = bool(metadata.get("model_only")) or int(workload.get("jobs", 0)) == 0
    effective_mac_ops_for_report = None if is_model_only else effective_mac_ops
    if is_model_only:
        peak_mac_capacity = None
        matrix_utilization = None

    cycle_analysis = _transformer_cycle_analysis(
        workload,
        metadata,
        effective_mac_ops_for_report,
        model,
        is_model_only,
    )

    external_memory = metadata.get("external_memory", {})
    kv_read = int(metadata.get("kv_read_bytes", external_memory.get("kv_cache_read_bytes", 0)))
    kv_write = int(metadata.get("kv_write_bytes", external_memory.get("kv_cache_write_bytes", 0)))
    bytes_per_token = metadata.get("bytes_per_token")
    if bytes_per_token is None and (kv_read or kv_write):
        bytes_per_token = kv_read + kv_write
    attention_stage = metadata.get("attention_stage")
    qk_cycles = None
    scale_mask_cycles = None
    softmax_cycles = None
    pv_cycles = None
    if not is_model_only:
        if attention_stage == "qk":
            qk_cycles = int(workload.get("total_cycles", 0))
        elif attention_stage == "scale_mask":
            scale_mask_cycles = int(workload.get("total_cycles", 0))
        elif attention_stage == "softmax":
            softmax_cycles = int(workload.get("total_cycles", 0))
        elif attention_stage == "pv":
            pv_cycles = int(workload.get("total_cycles", 0))

    return {
        "workload_family": metadata.get("workload_family"),
        "attention_group": metadata.get("attention_group"),
        "attention_stage": attention_stage,
        "numerical_contract": metadata.get("numerical_contract"),
        "stage_provenance": metadata.get("stage_provenance"),
        "shape_class": shape_class,
        "matrix_active_cycles": matrix_active,
        "vector_active_cycles": int(workload.get("core", {}).get("vector", 0)),
        "reduction_active_cycles": int(workload.get("core", {}).get("reduction", 0)),
        "sfu_active_cycles": int(workload.get("core", {}).get("sfu", 0)),
        "data_mover_active_cycles": int(workload.get("data_mover", {}).get("transfer_cycles", 0)),
        "stall_cycles_by_engine": {
            "matrix": int(workload.get("core", {}).get("matrix_stall", 0)),
            "vector": int(workload.get("core", {}).get("vector_stall", 0)),
            "reduction": int(workload.get("core", {}).get("reduction_stall", 0)),
            "sfu": int(workload.get("core", {}).get("sfu_stall", 0)),
            "data_mover": int(workload.get("data_mover", {}).get("stall_cycles", 0)),
        },
        "effective_mac_ops": effective_mac_ops_for_report,
        "peak_mac_capacity": peak_mac_capacity,
        "matrix_utilization": matrix_utilization,
        "gemv_utilization": matrix_utilization if shape_class == "gemv" else None,
        "skinny_gemm_utilization": matrix_utilization if shape_class == "skinny_gemm" else None,
        "tail_waste_mac_capacity": (
            peak_mac_capacity - effective_mac_ops_for_report
            if peak_mac_capacity is not None and effective_mac_ops_for_report is not None
            else None
        ),
        "kv_read_bytes": kv_read,
        "kv_write_bytes": kv_write,
        "bytes_per_token": bytes_per_token,
        "qk_cycles": qk_cycles,
        "scale_mask_cycles": scale_mask_cycles,
        "attention_softmax_cycles": softmax_cycles,
        "pv_cycles": pv_cycles,
        "softmax_cycles": int(workload.get("core", {}).get("softmax", 0)) if not is_model_only else None,
        "rmsnorm_cycles": int(workload.get("core", {}).get("rmsnorm", 0)) if not is_model_only else None,
        "sfu_cycles": int(workload.get("core", {}).get("sfu", 0)) if not is_model_only else None,
        **cycle_analysis,
    }


def _transformer_cycle_analysis(
    workload: dict,
    metadata: dict,
    effective_mac_ops: int | None,
    model: dict,
    is_model_only: bool,
) -> dict:
    empty = {
        "theoretical_compute_cycles": None,
        "measured_compute_cycles": None,
        "compute_overhead_cycles": None,
        "compute_efficiency": None,
        "measured_total_cycles": None,
        "non_compute_overhead_cycles": None,
        "end_to_end_efficiency": None,
        "theoretical_cycle_basis": None,
        "measured_compute_provenance": None,
    }
    if is_model_only:
        return empty

    stage = metadata.get("attention_stage")
    jobs = max(1, int(workload.get("jobs", 0)))
    core = workload.get("core", {})
    theoretical = None
    measured_compute = None
    basis = None
    measured_provenance = None

    if effective_mac_ops is not None:
        peak = int(model.get("peak_macs_per_cycle", 64))
        theoretical = ceil_div(effective_mac_ops, peak)
        measured_compute = int(workload.get("core_matmul_cycles", 0))
        basis = f"ceil(effective_mac_ops={effective_mac_ops}/peak_macs_per_cycle={peak})"
        measured_provenance = "measured_matrix_active_cycles"
    elif stage == "scale_mask":
        shape = metadata.get("logical_shape", {})
        elements = int(shape.get("m", 0)) * int(shape.get("n", 0))
        lanes = int(model.get("vector_lanes", 8))
        theoretical = ceil_div(elements * jobs, lanes) if elements else None
        measured_compute = int(core.get("total", 0))
        basis = f"ceil(score_elements={elements * jobs}/vector_lanes={lanes})"
        measured_provenance = "measured_core_active_cycles_aggregate"
    elif stage == "softmax":
        softmax = metadata.get("softmax", {})
        elements = int(metadata.get("logical_shape", {}).get("elements", 0))
        rows = int(softmax.get("row_count_measured", jobs))
        fixed_ops_per_row = 6
        theoretical = rows * (elements + fixed_ops_per_row) if elements else None
        measured_compute = int(core.get("total", 0))
        basis = (
            f"rows={rows}*(EXP_per_lane={elements}+"
            "REDUCE_MAX+VEC_SUB+VEC_CLAMP+REDUCE_SUM+RECIP+VEC_SCALE=6)"
        )
        measured_provenance = "measured_core_active_cycles_aggregate"

    if theoretical is None or measured_compute is None or measured_compute <= 0:
        return empty
    measured_total = int(workload.get("total_cycles", 0))
    return {
        "theoretical_compute_cycles": theoretical,
        "measured_compute_cycles": measured_compute,
        "compute_overhead_cycles": measured_compute - theoretical,
        "compute_efficiency": round(theoretical / measured_compute, 6),
        "measured_total_cycles": measured_total,
        "non_compute_overhead_cycles": measured_total - measured_compute,
        "end_to_end_efficiency": (
            round(theoretical / measured_total, 6) if measured_total > 0 else None
        ),
        "theoretical_cycle_basis": basis,
        "measured_compute_provenance": measured_provenance,
    }


def classify_matrix_shape(m_dim: int, n_dim: int, k_dim: int) -> str:
    if m_dim == 1 or n_dim == 1:
        return "gemv"
    if m_dim <= 8 or n_dim <= 8:
        return "skinny_gemm"
    return "full_tile_gemm"


def _accumulate_counter_dict(dst: dict[str, int], src: dict) -> None:
    for key, value in src.items():
        dst[key] = int(dst.get(key, 0)) + int(value)


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def write_html(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs_json = json.dumps(report["jobs"])
    workloads_json = json.dumps(report.get("workloads", []))
    report_json = json.dumps(report, indent=2)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NPU Cycle Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #647085;
      --line: #d9dee8;
      --accent: #2068d8;
      --accent-2: #1a9a7a;
      --accent-3: #c46b1f;
      --accent-4: #7b61d1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    header {{
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 650; letter-spacing: 0; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 16px; }}
    h3 {{ font-size: 14px; }}
    main {{ padding: 24px 28px 40px; max-width: 1200px; margin: 0 auto; }}
    .subtle {{ color: var(--muted); margin-top: 6px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric, .job, .raw {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    details.job summary {{
      cursor: pointer;
      list-style: none;
    }}
    details.job summary::-webkit-details-marker {{ display: none; }}
    .highlight {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent-2);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .highlight-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .highlight-item {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .highlight-item .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .highlight-item .value {{
      font-size: 18px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .metric .value {{ font-size: 26px; font-weight: 700; margin-top: 8px; }}
    .grid {{ display: grid; gap: 16px; }}
    .job-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(110px, 160px) minmax(180px, 1fr) max-content;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
    }}
    .bar-value {{
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .bar-track {{
      height: 16px;
      background: #eef1f6;
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      min-width: 2px;
      border-radius: 4px;
    }}
    .module-title {{ margin-top: 12px; margin-bottom: 6px; color: var(--muted); }}
    .timeline-scroll {{
      width: 100%;
      overflow-x: auto;
      overflow-y: visible;
      padding-bottom: 4px;
    }}
    .timeline {{
      display: grid;
      grid-template-columns: max-content minmax(360px, 1fr) max-content;
      gap: 12px 12px;
      align-items: center;
      min-width: 680px;
      margin-top: 12px;
    }}
    .phase-timeline {{
      grid-template-columns: max-content minmax(360px, 1fr) max-content;
    }}
    .axis {{
      grid-column: 2;
      position: relative;
      height: 34px;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .axis-value-spacer {{ grid-column: 3; }}
    .axis::after {{
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: 4px;
      border-bottom: 1px solid var(--line);
    }}
    .tick {{
      position: absolute;
      bottom: 10px;
      transform: translateX(-50%);
      white-space: nowrap;
      background: var(--panel);
      padding: 0 4px;
    }}
    .lane-label {{ color: var(--ink); font-weight: 600; }}
    .timeline .lane-label {{
      white-space: nowrap;
    }}
    .timeline .lane-label.child {{
      border-left: 2px solid #c7d2e2;
      color: #3c4b61;
    }}
    .lane-role {{
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 500;
      margin-top: 1px;
    }}
    .phase-timeline .lane-label {{ font-weight: 500; color: #324056; }}
    .lane-value {{
      text-align: right;
      color: var(--ink);
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .lane {{
      position: relative;
      height: 34px;
      background: #eef1f6;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }}
    .span {{
      position: absolute;
      top: 4px;
      height: 24px;
      border-radius: 5px;
      color: #fff;
      display: flex;
      align-items: center;
      padding: 0 8px;
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 3px;
    }}
    .span.wait {{
      color: #253044;
      background: repeating-linear-gradient(
        45deg,
        #dce2ec,
        #dce2ec 6px,
        #cbd4e2 6px,
        #cbd4e2 12px
      );
      border: 1px solid #bcc7d8;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend i {{
      display: inline-block;
      width: 18px;
      height: 10px;
      border-radius: 3px;
      margin-right: 5px;
      vertical-align: -1px;
    }}
    .estimate-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin: 10px 0 6px;
    }}
    .estimate {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .workload-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-variant-numeric: tabular-nums;
    }}
    .workload-table th, .workload-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }}
    .workload-table th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .estimate .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .estimate .value {{
      font-size: 18px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.45;
    }}
    @media (max-width: 680px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .bar-row {{ grid-template-columns: minmax(140px, 1fr) max-content; gap: 5px; }}
      .bar-row > :first-child {{ grid-column: 1 / -1; }}
      .job-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>NPU Cycle Report</h1>
    <div class="subtle">Source: {html.escape(report["source_log"])}</div>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><h2>Jobs</h2><div class="value">{report["summary"]["jobs"]}</div></div>
      <div class="metric"><h2>Workloads</h2><div class="value">{report["summary"].get("workloads", 0)}</div></div>
      <div class="metric"><h2>Total Cycles</h2><div class="value">{report["summary"]["total_cycles"]}</div></div>
      <div class="metric"><h2>Max Job Cycles</h2><div class="value">{report["summary"]["max_job_cycles"]}</div></div>
    </section>
    <section id="highlights"></section>
    <section class="job" id="workloads">
      <div class="job-head">
        <h2>Workload Summary</h2>
        <div class="subtle">Grouped model/operator runs from workload manifest when supplied; legacy logs fall back to inference</div>
      </div>
    </section>
    <section class="grid" id="jobs"></section>
    <section class="raw" style="margin-top:16px">
      <h2 style="margin-bottom:10px">Raw JSON</h2>
      <pre>{html.escape(report_json)}</pre>
    </section>
  </main>
  <script>
    const jobs = {jobs_json};
    const workloads = {workloads_json};
    const highlights = {json.dumps(report.get("highlights", []))};
    const colors = ["var(--accent)", "var(--accent-2)", "var(--accent-3)", "var(--accent-4)"];
    const timelineColors = {{
      "CPU firmware": "#7b61d1",
      "NPU wrapper": "#2068d8",
      "NPU core": "#526174",
      "Command processor": "#4380b8",
      "Uop scheduler": "#6f5aa8",
      "Data mover": "#c46b1f",
      "Compute cluster": "#1a9a7a",
      "Compute cluster control": "#68758a",
      "Accumulator file": "#8a6f3d",
      "Local storage path": "#5b8f78",
      "Matrix engine": "#1666b1",
      "Vector engine": "#8b5a2b",
      "Reduction engine": "#b04759",
      "SFU": "#7b61d1"
    }};
    const wrappers = [
      ["desc_read", "Descriptor read"],
      ["fetch_program", "Program fetch"],
      ["fetch_input0", "Input0 fetch"],
      ["fetch_input1", "Input1 fetch"],
      ["start_core", "Core launch"],
      ["wait_core", "Core wait"],
      ["write_output", "Output writeback"],
      ["done", "Done latch"]
    ];
    const cores = [
      ["fetch", "Uop fetch/execute"],
      ["matmul", "Matmul"],
      ["done", "Done"]
    ];

    function row(key, label, value, total, colorIndex) {{
      const pct = total > 0 ? Math.max(0, Math.min(100, (value / total) * 100)) : 0;
      const div = document.createElement("div");
      div.className = "bar-row";
      div.innerHTML = `
        <div>${{label}}</div>
        <div class="bar-track"><div class="bar" style="width:${{pct}}%; background:${{colors[colorIndex % colors.length]}}"></div></div>
        <div class="bar-value">${{value}} cycles</div>
      `;
      return div;
    }}

    function renderModule(parent, title, entries, data, total, colorOffset) {{
      const h = document.createElement("h3");
      h.className = "module-title";
      h.textContent = title;
      parent.appendChild(h);
      entries.forEach(([key, label], idx) => parent.appendChild(row(key, label, data[key] || 0, total, idx + colorOffset)));
    }}

    function tickValues(total) {{
      if (total <= 10) return [0, total];
      const step = Math.max(1, Math.ceil(total / 4 / 10) * 10);
      const ticks = [0];
      for (let t = step; t < total; t += step) ticks.push(t);
      ticks.push(total);
      return ticks;
    }}

    function renderTimeline(parent, job) {{
      const title = document.createElement("h3");
      title.className = "module-title";
      title.textContent = "Cycle timeline";
      parent.appendChild(title);

      const scroll = document.createElement("div");
      scroll.className = "timeline-scroll";
      const timeline = document.createElement("div");
      timeline.className = "timeline";
      const spacer = document.createElement("div");
      const axis = document.createElement("div");
      axis.className = "axis";
      tickValues(job.total_cycles).forEach((tick) => {{
        const t = document.createElement("span");
        t.className = "tick";
        t.style.left = `${{(tick / job.total_cycles) * 100}}%`;
        t.textContent = tick;
        axis.appendChild(t);
      }});
      timeline.appendChild(spacer);
      timeline.appendChild(axis);
      const axisValueSpacer = document.createElement("div");
      axisValueSpacer.className = "axis-value-spacer";
      timeline.appendChild(axisValueSpacer);

      job.timeline.forEach((laneData) => {{
        const label = document.createElement("div");
        const depth = laneData.depth || 0;
        label.className = `lane-label ${{depth ? "child" : "root"}}`;
        label.style.paddingLeft = `${{depth * 16 + 4}}px`;
        label.innerHTML = `<span>${{depth ? "↳ " : ""}}${{laneData.module}}</span><small class="lane-role">${{laneData.role || "module"}}</small>`;
        const lane = document.createElement("div");
        lane.className = "lane";
        laneData.spans.forEach((span) => {{
          const el = document.createElement("div");
          const width = ((span.end - span.start) / job.total_cycles) * 100;
          el.className = `span ${{span.kind}}`;
          el.style.left = `${{(span.start / job.total_cycles) * 100}}%`;
          el.style.width = `${{Math.max(width, 0.6)}}%`;
          if (span.kind !== "wait") el.style.background = timelineColors[laneData.module] || "var(--accent)";
          el.title = `${{laneData.module}}: ${{span.label}}\\n${{span.start}}-${{span.end}} cycles (${{span.cycles}})`;
          el.textContent = width >= 7 ? `${{span.label}} (${{span.cycles}})` : "";
          lane.appendChild(el);
        }});
        const laneValue = document.createElement("div");
        laneValue.className = "lane-value";
        const activeCycles = laneData.measured_active_cycles ?? laneData.spans.filter((span) => span.kind === "work").reduce((sum, span) => sum + span.cycles, 0);
        const waitCycles = laneData.measured_wait_cycles ?? laneData.spans.filter((span) => span.kind === "wait").reduce((sum, span) => sum + span.cycles, 0);
        laneValue.textContent = laneData.role === "architecture group"
          ? "group"
          : (waitCycles ? `${{activeCycles}} active / ${{waitCycles}} wait` : `${{activeCycles}} active`);
        timeline.appendChild(label);
        timeline.appendChild(lane);
        timeline.appendChild(laneValue);
      }});
      scroll.appendChild(timeline);
      parent.appendChild(scroll);

      const legend = document.createElement("div");
      legend.className = "legend";
      legend.innerHTML = `
        <span><i style="background:#2068d8"></i>active work</span>
        <span><i style="background:repeating-linear-gradient(45deg,#dce2ec,#dce2ec 6px,#cbd4e2 6px,#cbd4e2 12px); border:1px solid #bcc7d8"></i>wait/blocked</span>
        <span>NPU core is a group; command processor, data mover, and compute cluster are its children. Group and child totals are not additive.</span>
      `;
      parent.appendChild(legend);
    }}

    function renderPhaseTimeline(parent, titleText, laneData, total, moduleColor) {{
      const title = document.createElement("h3");
      title.className = "module-title";
      title.textContent = titleText;
      parent.appendChild(title);

      const scroll = document.createElement("div");
      scroll.className = "timeline-scroll";
      const timeline = document.createElement("div");
      timeline.className = "timeline phase-timeline";
      const spacer = document.createElement("div");
      const axis = document.createElement("div");
      axis.className = "axis";
      tickValues(total).forEach((tick) => {{
        const t = document.createElement("span");
        t.className = "tick";
        t.style.left = `${{(tick / total) * 100}}%`;
        t.textContent = tick;
        axis.appendChild(t);
      }});
      timeline.appendChild(spacer);
      timeline.appendChild(axis);
      const axisValueSpacer = document.createElement("div");
      axisValueSpacer.className = "axis-value-spacer";
      timeline.appendChild(axisValueSpacer);

      laneData.spans.forEach((span) => {{
        const label = document.createElement("div");
        label.className = "lane-label";
        label.textContent = span.label;
        const lane = document.createElement("div");
        lane.className = "lane";
        const el = document.createElement("div");
        const width = ((span.end - span.start) / total) * 100;
        const detail = phaseDetail(laneData.module, span.label);
        el.className = `span ${{span.kind}}`;
        el.style.left = `${{(span.start / total) * 100}}%`;
        el.style.width = `${{Math.max(width, 0.6)}}%`;
        if (span.kind !== "wait") el.style.background = moduleColor;
        el.title = `${{laneData.module}}: ${{span.label}}\\n${{detail}}\\n${{span.start}}-${{span.end}} cycles (${{span.cycles}})`;
        el.textContent = width >= 7 ? `${{span.start}}-${{span.end}}` : "";
        lane.appendChild(el);
        const laneValue = document.createElement("div");
        laneValue.className = "lane-value";
        laneValue.textContent = detail ? `${{span.cycles}} cycles - ${{detail}}` : `${{span.cycles}} cycles`;
        timeline.appendChild(label);
        timeline.appendChild(lane);
        timeline.appendChild(laneValue);
      }});
      scroll.appendChild(timeline);
      parent.appendChild(scroll);
    }}

    function phaseDetail(moduleName, label) {{
      if (moduleName !== "Command processor") return "";
      const details = {{
        "Descriptor read": "wrapper reads job descriptor words from SRAM",
        "Program fetch": "wrapper reads program words from SRAM and writes core instr_mem through host window",
        "Input0 fetch": "wrapper reads input0 tensor from SRAM and writes core A/X window",
        "Input1 fetch": "wrapper reads input1 tensor from SRAM and writes core B window",
        "Core launch": "command processor starts the compute cluster",
        "Core wait": "command processor waits while the compute cluster executes",
        "Output writeback": "wrapper reads core C/Y output window and writes result words to SRAM",
        "Done latch": "wrapper updates done/status"
      }};
      return details[label] || "";
    }}

    function renderEstimates(parent, job) {{
      if (!job.estimates) return;
      const title = document.createElement("h3");
      title.className = "module-title";
      title.textContent = "Matmul model";
      parent.appendChild(title);

      const grid = document.createElement("div");
      grid.className = "estimate-grid";
      const items = [
        ["Measured compute", `${{job.estimates.measured_compute_cycles}} cycles`],
        ["Scalar baseline", `${{job.estimates.scalar_compute_cycles}} cycles`],
        ["Ideal 8x8 array", `${{job.estimates.ideal_array_compute_cycles}} cycles`],
        ["Conservative array", `${{job.estimates.conservative_array_compute_cycles}} cycles`],
        ["Projected total", `${{job.estimates.projected_total_with_conservative_array}} cycles`]
      ];
      items.forEach(([label, value]) => {{
        const div = document.createElement("div");
        div.className = "estimate";
        div.innerHTML = `<div class="label">${{label}}</div><div class="value">${{value}}</div>`;
        grid.appendChild(div);
      }});
      parent.appendChild(grid);
    }}

    function renderMovementEstimates(parent, job) {{
      if (!job.movement_estimates) return;
      const title = document.createElement("h3");
      title.className = "module-title";
      title.textContent = "Movement model";
      parent.appendChild(title);

      const grid = document.createElement("div");
      grid.className = "estimate-grid";
      const m = job.movement_estimates;
      const items = [
        ["Moved words", `${{m.total_words}} words`],
        ["Measured SRAM", `${{m.measured_sram_cycles}} cycles`],
        ["Measured host window", `${{m.measured_host_window_cycles}} cycles`],
        ["Model bandwidth", `${{m.model_words_per_cycle}} words/cycle`],
        ["Ideal burst", `${{m.ideal_burst_cycles}} cycles`],
        ["Conservative burst", `${{m.conservative_burst_cycles}} cycles`]
      ];
      items.forEach(([label, value]) => {{
        const div = document.createElement("div");
        div.className = "estimate";
        div.innerHTML = `<div class="label">${{label}}</div><div class="value">${{value}}</div>`;
        grid.appendChild(div);
      }});
      parent.appendChild(grid);
    }}

    const root = document.getElementById("jobs");
    const highlightRoot = document.getElementById("highlights");
    const workloadRoot = document.getElementById("workloads");
    if (highlights.length) {{
      highlights.forEach((h) => {{
        const section = document.createElement("article");
        section.className = "highlight";
        section.innerHTML = `
          <div class="job-head">
            <h2>${{h.title}}</h2>
            <div class="subtle">${{h.workload}}</div>
          </div>
          <div class="subtle">${{h.summary}}</div>
          <div class="highlight-grid">
            <div class="highlight-item"><div class="label">Before</div><div class="value">${{h.before_cycles}} cycles</div></div>
            <div class="highlight-item"><div class="label">After</div><div class="value">${{h.after_cycles}} cycles</div></div>
            <div class="highlight-item"><div class="label">Saved</div><div class="value">${{h.cycles_saved}} cycles (${{h.improvement_pct}}%)</div></div>
            ${{h.overlap_cycles === null ? "" : `<div class="highlight-item"><div class="label">Overlap</div><div class="value">${{h.overlap_cycles}} cycles</div></div>`}}
            <div class="highlight-item"><div class="label">Core matmul</div><div class="value">${{h.core_matmul_cycles}} cycles</div></div>
            <div class="highlight-item"><div class="label">Moved words</div><div class="value">${{h.data_mover_words}}</div></div>
          </div>
        `;
        highlightRoot.appendChild(section);
      }});
    }}
    if (workloads.length) {{
      const graphGroups = new Map();
      workloads.forEach((w) => {{
        const m = w.metadata || {{}};
        const group = m.attention_group || m.graph || m.workload_family || w.kind;
        if (!graphGroups.has(group)) graphGroups.set(group, []);
        graphGroups.get(group).push(w);
      }});
      graphGroups.forEach((items, group) => {{
        const graph = document.createElement("div");
        graph.className = "highlight";
        graph.innerHTML = `<h3>${{group}}</h3><div class="subtle">Computation graph / tested operator sequence</div>`;
        const flow = document.createElement("div");
        flow.className = "legend";
        items.forEach((w, index) => {{
          const m = w.metadata || {{}};
          const shape = m.logical_shape ? JSON.stringify(m.logical_shape) : "shape not declared";
          const op = m.logical_op || w.name;
          const box = document.createElement("span");
          box.innerHTML = `<i style="background:${{colors[index % colors.length]}}"></i><strong>${{op}}</strong> ${{shape}}`;
          flow.appendChild(box);
          if (index + 1 < items.length) flow.append(" -> ");
        }});
        graph.appendChild(flow);
        workloadRoot.appendChild(graph);
      }});
      const table = document.createElement("table");
      table.className = "workload-table";
      table.innerHTML = `
        <thead>
          <tr>
            <th>Name</th>
            <th>Kind</th>
            <th>Jobs</th>
            <th>Total cycles</th>
            <th>Core matmul</th>
            <th>SRAM movement</th>
            <th>Theoretical compute</th>
            <th>Measured compute</th>
            <th>Efficiency</th>
            <th>Operator / shape / formula</th>
          </tr>
        </thead>
        <tbody>
          ${{workloads.map((w) => `
            <tr>
              <td>${{w.name}}</td>
              <td>${{w.kind}}</td>
              <td>#${{w.job_ids.join(", #")}}</td>
              <td>${{w.total_cycles}}</td>
              <td>${{w.core_matmul_cycles}}</td>
              <td>${{w.movement_sram_cycles}}</td>
              <td>${{w.transformer_metrics && w.transformer_metrics.theoretical_compute_cycles !== null ? w.transformer_metrics.theoretical_compute_cycles : "-"}}</td>
              <td>${{w.transformer_metrics && w.transformer_metrics.measured_compute_cycles !== null ? w.transformer_metrics.measured_compute_cycles : "-"}}</td>
              <td>${{w.transformer_metrics && w.transformer_metrics.compute_efficiency !== null ? `${{(w.transformer_metrics.compute_efficiency * 100).toFixed(2)}}%` : "-"}}</td>
              <td>${{w.metadata && w.metadata.logical_op ? w.metadata.logical_op : ""}} ${{w.metadata && w.metadata.logical_shape ? JSON.stringify(w.metadata.logical_shape) : ""}}<br><span class="subtle">${{w.transformer_metrics && w.transformer_metrics.theoretical_cycle_basis ? w.transformer_metrics.theoretical_cycle_basis : ""}}</span></td>
            </tr>
          `).join("")}}
        </tbody>
      `;
      workloadRoot.appendChild(table);
    }}

    jobs.forEach((job) => {{
      const section = document.createElement("details");
      section.className = "job";
      section.innerHTML = `
        <summary>
        <div class="job-head">
          <h2>#${{job.id}} ${{job.name}}</h2>
          <div class="subtle">${{job.total_cycles}} cycles - open detail</div>
        </div>
        </summary>
      `;
      section.addEventListener("toggle", () => {{
        if (!section.open || section.dataset.rendered) return;
        renderTimeline(section, job);
        if (job.timeline_provenance) {{
          const provenance = document.createElement("div");
          provenance.className = "subtle";
          provenance.textContent = `Timeline counters: ${{job.timeline_provenance.summary_counters}}; placement: ${{job.timeline_provenance.span_placement}}`;
          section.appendChild(provenance);
        }}
        renderEstimates(section, job);
        renderMovementEstimates(section, job);
        job.timeline.slice(1).forEach((laneData) => {{
          renderPhaseTimeline(section, `${{laneData.module}} phases`, laneData, job.total_cycles, timelineColors[laneData.module]);
        }});
        section.dataset.rendered = "true";
      }});
      root.appendChild(section);
    }});
  </script>
</body>
</html>
"""
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NPU cycle report UI from simulation log.")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--workload-manifest", type=Path)
    parser.add_argument("--arch-config", type=Path)
    parser.add_argument("--soc-config", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--html-out", type=Path)
    args = parser.parse_args()

    if args.arch_config is None or args.soc_config is None:
        model = DEFAULT_MODEL
    else:
        model = load_measurement_model(args.arch_config, args.soc_config)
    report = parse_perf_log(args.log, args.workload_manifest, model)
    write_json(report, args.json_out)
    print(f"Wrote {args.json_out}")
    if args.html_out is not None:
        write_html(report, args.html_out)
        print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
