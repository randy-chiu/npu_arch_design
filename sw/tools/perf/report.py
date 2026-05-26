from __future__ import annotations

import argparse
import html
import json
import warnings
from pathlib import Path


PERF_PREFIX = "PERF_JOB "
PHASE0_MATMUL_M = 8
PHASE0_MATMUL_N = 8
PHASE0_MATMUL_K = 8
PHASE0_MATMUL_ARRAY_CONTROL_CYCLES = 4
PHASE0_DATA_MOVER_WORDS_PER_CYCLE = 4
PHASE0_DATA_MOVER_SETUP_CYCLES = 1


def add_estimates(job: dict) -> dict:
    if job.get("name") != "matmul":
        return job

    scalar_compute = PHASE0_MATMUL_M * PHASE0_MATMUL_N * PHASE0_MATMUL_K
    ideal_array_compute = PHASE0_MATMUL_K
    conservative_array_compute = ideal_array_compute + PHASE0_MATMUL_ARRAY_CONTROL_CYCLES
    measured_compute = int(job.get("core", {}).get("matmul", 0))
    non_matmul_cycles = int(job["total_cycles"]) - measured_compute

    job["estimates"] = {
        "matmul_shape": [PHASE0_MATMUL_M, PHASE0_MATMUL_N, PHASE0_MATMUL_K],
        "scalar_compute_cycles": scalar_compute,
        "ideal_array_compute_cycles": ideal_array_compute,
        "conservative_array_compute_cycles": conservative_array_compute,
        "measured_compute_cycles": measured_compute,
        "projected_total_with_conservative_array": non_matmul_cycles + conservative_array_compute,
    }
    return job


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def add_movement_estimates(job: dict) -> dict:
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
    ideal_burst_cycles = ceil_div(total_words, PHASE0_DATA_MOVER_WORDS_PER_CYCLE)
    conservative_burst_cycles = sum(
        ceil_div(value, PHASE0_DATA_MOVER_WORDS_PER_CYCLE)
        for value in words.values()
        if value > 0
    ) + active_segments * PHASE0_DATA_MOVER_SETUP_CYCLES

    job["movement_estimates"] = {
        "total_words": total_words,
        "measured_sram_cycles": measured_sram_cycles,
        "measured_host_window_cycles": measured_host_window_cycles,
        "measured_data_mover_transfer_cycles": int(data_mover.get("transfer_cycles", 0)),
        "measured_data_mover_words": int(data_mover.get("words", 0)),
        "model_words_per_cycle": PHASE0_DATA_MOVER_WORDS_PER_CYCLE,
        "model_setup_cycles_per_segment": PHASE0_DATA_MOVER_SETUP_CYCLES,
        "ideal_burst_cycles": ideal_burst_cycles,
        "conservative_burst_cycles": conservative_burst_cycles,
    }
    return job


def add_timeline(job: dict) -> dict:
    wrapper_order = [
        ("desc_read", "Descriptor read", "work"),
        ("fetch_program", "Program fetch", "work"),
        ("fetch_input0", "Input0 fetch", "work"),
        ("fetch_input1", "Input1 fetch", "work"),
        ("start_core", "Core launch", "work"),
        ("wait_core", "Wait for core", "wait"),
        ("write_output", "Output writeback", "work"),
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
        "Program fetch": "Program load",
        "Input0 fetch": "Input0 load",
        "Input1 fetch": "Input1 load",
        "Output writeback": "Output store",
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
    if job.get("name") == "matmul_k_stream":
        wait_span = wrapper_span_by_label.get("Wait for core")
        if wait_span is not None:
            initial_read_cycles = sum(
                int(job.get("wrapper", {}).get(key, 0))
                for key in ("fetch_program", "fetch_input0", "fetch_input1")
            )
            prefetch_cycles = max(
                0,
                int(job.get("data_mover", {}).get("read_cycles", 0)) - initial_read_cycles,
            )
            overlap_cycles = min(prefetch_cycles, int(wait_span["cycles"]))
            if overlap_cycles > 0:
                data_mover_spans.append(
                    {
                        "label": "K prefetch overlap",
                        "start": wait_span["start"],
                        "end": wait_span["start"] + overlap_cycles,
                        "cycles": overlap_cycles,
                        "kind": "work",
                    }
                )

    job["timeline"] = [
        {
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
        },
        {"module": "NPU wrapper", "spans": wrapper_spans},
        {"module": "Data mover", "spans": data_mover_spans},
        {"module": "NPU core", "spans": core_spans},
    ]
    return job


def parse_perf_log(path: Path, manifest_path: Path | None = None) -> dict:
    jobs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(PERF_PREFIX):
                jobs.append(
                    add_timeline(
                        add_movement_estimates(add_estimates(json.loads(line[len(PERF_PREFIX) :])))
                    )
                )
    if not jobs:
        raise ValueError(f"no {PERF_PREFIX.strip()} records found in {path}")
    workload_manifest = None
    if manifest_path is not None:
        workload_manifest = load_workload_manifest(manifest_path)
        workloads = workloads_from_manifest(jobs, workload_manifest)
    else:
        warnings.warn(
            "no workload manifest provided; falling back to order-based workload inference",
            UserWarning,
        )
        workloads = infer_workloads(jobs)
    highlights = build_highlights(workloads, jobs)
    return {
        "schema": "npu_perf_report_v0",
        "source_log": str(path),
        "workload_manifest": (
            {
                "schema": workload_manifest["schema"],
                "id": workload_manifest["manifest_id"],
                "run_name": workload_manifest["run_name"],
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
        },
        "highlights": highlights,
        "workloads": workloads,
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


def build_highlights(workloads: list[dict], jobs: list[dict]) -> list[dict]:
    highlights = []
    workload_by_name = {workload["name"]: workload for workload in workloads}
    fc1_full = workload_by_name.get("real_mnist_cnn_fc1_full_k_stream_layer")
    if not fc1_full:
        fc1_full = workload_by_name.get("real_mnist_cnn_fc1_full_k_stream_tile0")
    if fc1_full:
        old_serial_baseline = 58784 * int(fc1_full.get("jobs", 1))
        total_cycles = int(fc1_full["total_cycles"])
        cycles_saved = old_serial_baseline - total_cycles
        improvement_pct = (cycles_saved * 100.0 / old_serial_baseline) if old_serial_baseline else 0.0
        overlap_cycles = 0
        for job in jobs:
            if _job_id(job) in fc1_full.get("job_ids", []):
                for lane in job.get("timeline", []):
                    if lane.get("module") == "Data mover":
                        for span in lane.get("spans", []):
                            if span.get("label") == "K prefetch overlap":
                                overlap_cycles += int(span.get("cycles", 0))
        highlights.append(
            {
                "title": "FC1 K-stream ping-pong overlap",
                "workload": fc1_full["name"],
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
    if len(jobs) >= 2:
        workloads.append(_workload_summary("operator_smoke_softmax", jobs[1:2], "operator_smoke"))

    cursor = 2
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
    return {
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
      "Data mover": "#c46b1f",
      "NPU core": "#1a9a7a"
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
        label.className = "lane-label";
        label.textContent = laneData.module;
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
        const activeCycles = laneData.spans.reduce((sum, span) => sum + span.cycles, 0);
        laneValue.textContent = `${{activeCycles}} cycles`;
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
      if (moduleName !== "NPU wrapper") return "";
      const details = {{
        "Descriptor read": "wrapper reads job descriptor words from SRAM",
        "Program fetch": "wrapper reads program words from SRAM and writes core instr_mem through host window",
        "Input0 fetch": "wrapper reads input0 tensor from SRAM and writes core A/X window",
        "Input1 fetch": "wrapper reads input1 tensor from SRAM and writes core B window",
        "Core launch": "wrapper pulses NPU core start",
        "Core wait": "wrapper waits while NPU core executes its already-loaded uops/data",
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
            <div class="highlight-item"><div class="label">Overlap</div><div class="value">${{h.overlap_cycles}} cycles</div></div>
            <div class="highlight-item"><div class="label">Core matmul</div><div class="value">${{h.core_matmul_cycles}} cycles</div></div>
            <div class="highlight-item"><div class="label">Moved words</div><div class="value">${{h.data_mover_words}}</div></div>
          </div>
        `;
        highlightRoot.appendChild(section);
      }});
    }}
    if (workloads.length) {{
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
            <th>Metadata</th>
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
              <td>${{w.metadata && w.metadata.description ? w.metadata.description : ""}}</td>
            </tr>
          `).join("")}}
        </tbody>
      `;
      workloadRoot.appendChild(table);
    }}

    jobs.forEach((job) => {{
      const section = document.createElement("article");
      section.className = "job";
      section.innerHTML = `
        <div class="job-head">
          <h2>#${{job.id}} ${{job.name}}</h2>
          <div class="subtle">${{job.total_cycles}} cycles</div>
        </div>
      `;
      renderTimeline(section, job);
      renderEstimates(section, job);
      renderMovementEstimates(section, job);
      renderPhaseTimeline(section, "Wrapper phases", job.timeline[1], job.total_cycles, timelineColors["NPU wrapper"]);
      renderPhaseTimeline(section, "Data mover phases", job.timeline[2], job.total_cycles, timelineColors["Data mover"]);
      renderPhaseTimeline(section, "Core phases", job.timeline[3], job.total_cycles, timelineColors["NPU core"]);
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
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--html-out", required=True, type=Path)
    args = parser.parse_args()

    report = parse_perf_log(args.log, args.workload_manifest)
    write_json(report, args.json_out)
    write_html(report, args.html_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
