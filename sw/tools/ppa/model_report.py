from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_NAME = "npu_ppa_report_v0"
EVIDENCE_LEVEL = "L0_model"


def read_jsonc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(re.sub(r"//.*$", "", text, flags=re.MULTILINE))


def build_ppa_report(
    perf: dict[str, Any],
    area_cfg: dict[str, Any],
    energy_cfg: dict[str, Any],
    baseline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    area_model = build_area_model(area_cfg)
    performance_provenance = perf.get("source", {}).get(
        "performance", "measured_rtl_perf_job_counters"
    )
    workload_results = [
        build_workload_model(workload, energy_cfg, performance_provenance)
        for workload in perf.get("workloads", [])
    ]
    workload_results.extend(
        build_workload_model(workload, energy_cfg, "modeled_manifest_only")
        for workload in perf.get("model_only_workloads", [])
    )
    manifest = perf.get("workload_manifest") or {}
    report = {
        "schema": SCHEMA_NAME,
        "evidence_level": EVIDENCE_LEVEL,
        "source_perf_report": perf.get("source_log", ""),
        "workload_manifest_id": manifest.get("id") or perf.get("workload_manifest_id"),
        "test_case": {
            "name": manifest.get("workload_profile", "unspecified"),
            "run_name": manifest.get("run_name", "unspecified"),
        },
        "design": area_cfg["design"],
        "model_config": {
            "area_coefficient_version": int(area_cfg["version"]),
            "area_units": area_cfg["units"],
            "area_coefficients": area_cfg["coefficients"],
            "energy_coefficient_version": int(energy_cfg["version"]),
            "energy_units": energy_cfg["units"],
            "energy_coefficients": energy_cfg["event_coefficients"],
        },
        "metric_provenance": {
            "performance": performance_provenance,
            "area": area_cfg["interpretation"],
            "energy": energy_cfg["interpretation"],
            "timing": "unavailable_until_mapped_or_physical_analysis",
            "power": "unavailable_until_activity_based_tool_flow",
        },
        "area_model": area_model,
        "pipeline_jobs": perf.get("jobs", []),
        "workloads": workload_results,
        "highlights": build_highlights(perf, workload_results, energy_cfg),
        "limitations": [
            "Normalized area units are not synthesized cell area or physical area.",
            "Normalized energy units are not joules or measured power.",
            "External-memory byte events are modeled from workload manifests, not measured RTL power.",
        ],
    }
    report["comparison"] = build_comparison(baseline_report, report) if baseline_report else None
    return report


def build_area_model(config: dict[str, Any]) -> dict[str, Any]:
    coeff = config["coefficients"]
    resources = config["resources"]
    storage_bits = resources["storage_bits"]
    storage_total = sum(int(value) for value in storage_bits.values())
    contributions = {
        "int8_mac_lanes": resources["int8_mac_lanes"] * coeff["int8_mac_lane"],
        "stored_bits": storage_total * coeff["stored_bit"],
        "data_mover_lanes": resources["data_mover_lanes"] * coeff["data_mover_lane"],
        "wrapper_control": resources["wrapper_control_units"] * coeff["wrapper_control_unit"],
    }
    return {
        "units": config["units"],
        "interpretation": config["interpretation"],
        "config": config["name"],
        "resources": resources,
        "storage_bits_total": storage_total,
        "coefficients": coeff,
        "contributions": contributions,
        "normalized_area_units": round(sum(contributions.values()), 3),
        "excludes": config.get("excludes", []),
    }


def build_workload_model(
    workload: dict[str, Any],
    energy_cfg: dict[str, Any],
    performance_provenance: str = "measured_rtl_perf_job_counters",
) -> dict[str, Any]:
    coefficients = energy_cfg["event_coefficients"]
    events = derive_workload_events(workload, energy_cfg)
    contributions = {
        name: events[name] * coefficients[name]
        for name in (
            "int8_mac_accumulate",
            "data_mover_read_word",
            "data_mover_write_word",
            "active_subsystem_cycle",
            "external_memory_byte",
        )
    }
    total_energy = sum(contributions.values())
    mac_ops = events["int8_mac_accumulate"]
    transformer_metrics = workload.get("transformer_metrics", {})
    energy_per_token = None
    bytes_per_token = transformer_metrics.get("bytes_per_token")
    if bytes_per_token is not None and bytes_per_token != 0:
        energy_per_token = round(total_energy, 3)
    return {
        "name": workload["name"],
        "kind": workload.get("kind", "unknown"),
        "job_ids": workload.get("job_ids", []),
        "jobs": int(workload.get("jobs", 0)),
        "performance": {
            "cycles": int(workload["total_cycles"]),
            "core_matmul_cycles": int(workload.get("core_matmul_cycles", 0)),
            "data_mover_words": int(workload.get("data_mover", {}).get("words", 0)),
            "attention_group": transformer_metrics.get("attention_group"),
            "attention_stage": transformer_metrics.get("attention_stage"),
            "numerical_contract": transformer_metrics.get("numerical_contract"),
            "stage_provenance": transformer_metrics.get("stage_provenance"),
            "qk_cycles": transformer_metrics.get("qk_cycles"),
            "scale_mask_cycles": transformer_metrics.get("scale_mask_cycles"),
            "attention_softmax_cycles": transformer_metrics.get("attention_softmax_cycles"),
            "pv_cycles": transformer_metrics.get("pv_cycles"),
            "effective_mac_per_cycle": (
                round(float(transformer_metrics["effective_mac_ops"]) / float(workload["total_cycles"]), 6)
                if transformer_metrics.get("effective_mac_ops") is not None and int(workload["total_cycles"]) > 0
                else None
            ),
            "matrix_utilization": transformer_metrics.get("matrix_utilization"),
            "gemv_utilization": transformer_metrics.get("gemv_utilization"),
            "skinny_gemm_utilization": transformer_metrics.get("skinny_gemm_utilization"),
            "kv_read_bytes": transformer_metrics.get("kv_read_bytes", 0),
            "kv_write_bytes": transformer_metrics.get("kv_write_bytes", 0),
            "bytes_per_token": bytes_per_token,
            "softmax_cycles": transformer_metrics.get("softmax_cycles"),
            "rmsnorm_cycles": transformer_metrics.get("rmsnorm_cycles"),
            "sfu_cycles": transformer_metrics.get("sfu_cycles"),
            "theoretical_compute_cycles": transformer_metrics.get("theoretical_compute_cycles"),
            "measured_compute_cycles": transformer_metrics.get("measured_compute_cycles"),
            "compute_overhead_cycles": transformer_metrics.get("compute_overhead_cycles"),
            "compute_efficiency": transformer_metrics.get("compute_efficiency"),
            "measured_total_cycles": transformer_metrics.get("measured_total_cycles"),
            "non_compute_overhead_cycles": transformer_metrics.get("non_compute_overhead_cycles"),
            "end_to_end_efficiency": transformer_metrics.get("end_to_end_efficiency"),
            "theoretical_cycle_basis": transformer_metrics.get("theoretical_cycle_basis"),
            "measured_compute_provenance": transformer_metrics.get("measured_compute_provenance"),
            "provenance": performance_provenance,
        },
        "energy_model": {
            "units": energy_cfg["units"],
            "interpretation": energy_cfg["interpretation"],
            "events": events,
            "coefficients": coefficients,
            "contributions": contributions,
            "contribution_groups": {
                "measured_onchip_events": round(
                    contributions["int8_mac_accumulate"]
                    + contributions["data_mover_read_word"]
                    + contributions["data_mover_write_word"]
                    + contributions["active_subsystem_cycle"],
                    3,
                ),
                "modeled_external_memory": round(contributions["external_memory_byte"], 3),
            },
            "normalized_energy_units": round(total_energy, 3),
            "normalized_energy_per_mac": (
                round(total_energy / mac_ops, 6) if mac_ops else None
            ),
            "normalized_energy_per_token": energy_per_token,
        },
        "metadata": workload.get("metadata", {}),
        "transformer_metrics": transformer_metrics,
    }


def build_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    compatibility = comparison_compatibility(baseline, candidate)
    if not compatibility["comparable"]:
        return {
            "comparable": False,
            "compatibility": compatibility,
            "evidence_level": candidate["evidence_level"],
            "baseline": baseline["design"],
            "candidate": candidate["design"],
            "area_delta": None,
            "workload_deltas": [],
            "improvements": [],
            "costs": [
                "Direct comparison is unavailable because baseline and candidate are incompatible."
            ],
        }
    baseline_area = float(baseline["area_model"]["normalized_area_units"])
    candidate_area = float(candidate["area_model"]["normalized_area_units"])
    area_delta = delta_metric(baseline_area, candidate_area, lower_is_better=True)
    area_delta["metric"] = "normalized_area_units"
    area_delta["interpretation"] = "structural_model_not_synthesized_area"

    baseline_by_name = {item["name"]: item for item in baseline["workloads"]}
    candidate_by_name = {item["name"]: item for item in candidate["workloads"]}
    common_names = sorted(set(baseline_by_name) & set(candidate_by_name))
    workload_deltas = []
    improvements: list[str] = []
    costs: list[str] = []
    for name in common_names:
        old = baseline_by_name[name]
        new = candidate_by_name[name]
        cycles = delta_metric(
            old["performance"]["cycles"], new["performance"]["cycles"], lower_is_better=True
        )
        energy = delta_metric(
            old["energy_model"]["normalized_energy_units"],
            new["energy_model"]["normalized_energy_units"],
            lower_is_better=True,
        )
        moved_words = delta_metric(
            old["performance"]["data_mover_words"],
            new["performance"]["data_mover_words"],
            lower_is_better=True,
        )
        mac_ops = delta_metric(
            old["energy_model"]["events"]["int8_mac_accumulate"],
            new["energy_model"]["events"]["int8_mac_accumulate"],
            lower_is_better=False,
        )
        workload_deltas.append(
            {
                "name": name,
                "cycles": cycles,
                "energy_model": energy,
                "data_mover_words": moved_words,
                "int8_mac_accumulate": mac_ops,
            }
        )
        if cycles["classification"] == "improvement":
            improvements.append(
                f"{name}: latency decreases by {abs(cycles['delta'])} cycles "
                f"({abs(cycles['delta_pct']):.1f}%)."
            )
        if energy["classification"] == "improvement":
            improvements.append(
                f"{name}: event-energy model decreases by {abs(energy['delta'])} normalized units "
                f"({abs(energy['delta_pct']):.3f}%)."
            )
    if area_delta["classification"] == "regression":
        costs.append(
            f"Structural area model increases by {area_delta['delta']} normalized units "
            f"({area_delta['delta_pct']:.3f}%) due to candidate resources."
        )
    costs.append(
        "External-memory energy is unavailable at Level 0 and is not part of this preference decision."
    )
    return {
        "comparable": True,
        "compatibility": compatibility,
        "evidence_level": EVIDENCE_LEVEL,
        "baseline": baseline["design"],
        "candidate": candidate["design"],
        "metric_provenance": {
            "baseline_performance": baseline["metric_provenance"]["performance"],
            "candidate_performance": candidate["metric_provenance"]["performance"],
            "area": "structural_model_not_synthesized_area",
            "energy": "event_model_not_measured_power",
        },
        "area_delta": area_delta,
        "workload_deltas": workload_deltas,
        "improvements": improvements,
        "costs": costs,
    }


def comparison_compatibility(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if baseline.get("schema") != candidate.get("schema"):
        issues.append("schema differs")
    if baseline.get("evidence_level") != candidate.get("evidence_level"):
        issues.append("evidence_level differs")
    baseline_cfg = baseline.get("model_config", {})
    candidate_cfg = candidate.get("model_config", {})
    for key in (
        "area_coefficient_version",
        "area_units",
        "area_coefficients",
        "energy_coefficient_version",
        "energy_units",
        "energy_coefficients",
    ):
        if baseline_cfg.get(key) != candidate_cfg.get(key):
            issues.append(f"model_config.{key} differs")
    baseline_manifest = baseline.get("workload_manifest_id")
    candidate_manifest = candidate.get("workload_manifest_id")
    if baseline_manifest != candidate_manifest and (baseline_manifest or candidate_manifest):
        issues.append("workload_manifest_id differs or is missing on one report")
    baseline_names = {item["name"] for item in baseline.get("workloads", [])}
    candidate_names = {item["name"] for item in candidate.get("workloads", [])}
    common_names = sorted(baseline_names & candidate_names)
    if not common_names:
        issues.append("no common workload names")
    return {
        "comparable": not issues,
        "issues": issues,
        "common_workloads": common_names,
    }


def delta_metric(baseline: float, candidate: float, lower_is_better: bool) -> dict[str, Any]:
    delta = candidate - baseline
    delta_pct = (delta * 100.0 / baseline) if baseline else 0.0
    if delta == 0:
        classification = "invariant"
    elif (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better):
        classification = "improvement"
    else:
        classification = "regression"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": round(delta, 3),
        "delta_pct": round(delta_pct, 3),
        "classification": classification,
    }


def derive_workload_events(workload: dict[str, Any], energy_cfg: dict[str, Any]) -> dict[str, int]:
    derivation = energy_cfg["matmul_event_derivation"]
    matmul_cycles_per_tile = int(derivation["measured_core_matmul_cycles_per_tile"])
    mac_ops_per_tile = int(derivation["mac_ops_per_tile"])
    core_matmul_cycles = int(workload.get("core_matmul_cycles", 0))
    if core_matmul_cycles % matmul_cycles_per_tile != 0:
        raise ValueError(
            f"{workload['name']}: core_matmul_cycles={core_matmul_cycles} "
            f"is not divisible by verified tile cycles={matmul_cycles_per_tile}"
        )
    matmul_tiles = core_matmul_cycles // matmul_cycles_per_tile
    data_mover = workload.get("data_mover", {})
    external_bytes = external_memory_bytes(workload.get("metadata", {}))
    transformer_mac_ops = workload.get("transformer_metrics", {}).get("effective_mac_ops")
    return {
        "int8_mac_accumulate": int(transformer_mac_ops)
        if transformer_mac_ops is not None
        else matmul_tiles * mac_ops_per_tile,
        "data_mover_read_word": int(data_mover.get("read_words", 0)),
        "data_mover_write_word": int(data_mover.get("write_words", 0)),
        "active_subsystem_cycle": int(workload["total_cycles"]),
        "external_memory_byte": external_bytes,
    }


def external_memory_bytes(metadata: dict[str, Any]) -> int:
    external = metadata.get("external_memory", {})
    if not isinstance(external, dict):
        return int(metadata.get("external_memory_bytes", 0))
    return sum(int(value) for value in external.values())


def build_highlights(
    perf: dict[str, Any],
    workload_results: list[dict[str, Any]],
    energy_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    result_by_name = {item["name"]: item for item in workload_results}
    highlights = []
    for perf_highlight in perf.get("highlights", []):
        if perf_highlight.get("title") != "FC1 K-stream ping-pong overlap":
            continue
        current = result_by_name.get(perf_highlight["workload"])
        if current is None:
            continue
        coeff = energy_cfg["event_coefficients"]
        before_cycles = int(perf_highlight["before_cycles"])
        after_energy = current["energy_model"]["normalized_energy_units"]
        cycles_saved = int(perf_highlight["cycles_saved"])
        modeled_active_cycle_energy_saved = cycles_saved * coeff["active_subsystem_cycle"]
        highlights.append(
            {
                "title": perf_highlight["title"],
                "workload": current["name"],
                "performance_provenance": current["performance"]["provenance"],
                "before_cycles": before_cycles,
                "after_cycles": current["performance"]["cycles"],
                "cycles_saved": cycles_saved,
                "core_matmul_cycles": int(perf_highlight["core_matmul_cycles"]),
                "data_mover_words": int(perf_highlight["data_mover_words"]),
                "energy_model_interpretation": energy_cfg["interpretation"],
                "after_normalized_energy_units": after_energy,
                "modeled_energy_saved_from_shorter_active_duration_only": round(
                    modeled_active_cycle_energy_saved, 3
                ),
                "summary": (
                    "RTL counters prove latency reduction with stable MAC and moved-word work; "
                    "the only modeled energy reduction here is shorter active duration."
                ),
            }
        )
    return highlights


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_html(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_dimension_pages(report, path.parent)
    area = report["area_model"]
    case_name = report.get("test_case", {}).get("name", "unspecified")
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td>{item['performance']['cycles']}</td>"
        f"<td>{html.escape(str(item['performance'].get('theoretical_compute_cycles')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('measured_compute_cycles')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('compute_overhead_cycles')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('compute_efficiency')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('end_to_end_efficiency')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('attention_stage')))}</td>"
        f"<td>{item['energy_model']['events']['int8_mac_accumulate']}</td>"
        f"<td>{html.escape(str(item['performance'].get('matrix_utilization')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('gemv_utilization')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('kv_read_bytes')))}</td>"
        f"<td>{item['performance']['data_mover_words']}</td>"
        f"<td>{item['energy_model']['normalized_energy_units']}</td>"
        "</tr>"
        for item in report["workloads"]
    )
    limitations = "".join(f"<li>{html.escape(text)}</li>" for text in report["limitations"])
    comparison_html = ""
    comparison = report.get("comparison")
    if comparison and comparison.get("comparable"):
        delta_rows = "".join(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td>{item['cycles']['baseline']} -> {item['cycles']['candidate']} "
            f"({item['cycles']['delta_pct']}%)</td>"
            f"<td>{item['energy_model']['baseline']} -> {item['energy_model']['candidate']} "
            f"({item['energy_model']['delta_pct']}%)</td>"
            f"<td>{item['data_mover_words']['classification']}</td>"
            f"<td>{item['int8_mac_accumulate']['classification']}</td>"
            "</tr>"
            for item in comparison["workload_deltas"]
        )
        benefits = "".join(f"<li>{html.escape(text)}</li>" for text in comparison["improvements"])
        costs = "".join(f"<li>{html.escape(text)}</li>" for text in comparison["costs"])
        area_delta = comparison["area_delta"]
        comparison_html = f"""
  <section>
    <h2>Candidate Versus Baseline</h2>
    <p><code>{html.escape(comparison['candidate']['variant'])}</code> compared with
       <code>{html.escape(comparison['baseline']['variant'])}</code>.</p>
    <p>Structural area model: {area_delta['baseline']} -> {area_delta['candidate']}
       ({area_delta['delta_pct']}%, {area_delta['classification']}).</p>
    <table>
      <thead><tr><th>Common workload</th><th>Measured cycles</th><th>Energy model</th><th>Moved words</th><th>MAC work</th></tr></thead>
      <tbody>{delta_rows}</tbody>
    </table>
    <h3>Improvements</h3><ul>{benefits}</ul>
    <h3>Costs And Unknowns</h3><ul>{costs}</ul>
  </section>"""
    elif comparison:
        issues = "".join(
            f"<li>{html.escape(text)}</li>" for text in comparison["compatibility"]["issues"]
        )
        comparison_html = f"""
  <section>
    <h2>Candidate Versus Baseline</h2>
    <p>Direct comparison is not valid for this report pair.</p>
    <ul>{issues}</ul>
  </section>"""
    else:
        comparison_html = """
  <section>
    <h2>Candidate Versus Baseline</h2>
    <p>No baseline provided; <code>comparison</code> is null.</p>
  </section>"""
    highlights = "".join(
        f"<section><h2>{html.escape(item['title'])}</h2>"
        f"<p>{html.escape(item['summary'])}</p>"
        f"<p>Cycles: {item['before_cycles']} -> {item['after_cycles']} "
        f"(saved {item['cycles_saved']}); modeled active-duration energy saved: "
        f"{item['modeled_energy_saved_from_shorter_active_duration_only']} normalized units.</p></section>"
        for item in report["highlights"]
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NPU PPA Overview</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #17202a; }}
    .warning {{ background: #fff4df; border: 1px solid #dfb052; padding: 12px; }}
    .metrics {{ display: flex; gap: 16px; margin: 18px 0; }}
    .metric, section {{ border: 1px solid #d9dee8; padding: 14px; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #d9dee8; }}
    code {{ background: #f2f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>NPU PPA Overview</h1>
  <p class="warning"><strong>Interpretation:</strong> Performance and movement are measured from architectural
  perf CSR snapshot values carried in <code>PERF_JOB</code>. Area and energy are normalized models, not synthesized area,
  watts, or joules.</p>
  <section>
    <h2>Detailed Reports</h2>
    <p><strong>Test case:</strong> <a href="cases/{html.escape(case_name)}.html">{html.escape(case_name)}</a></p>
    <p><a href="perf.html">Performance</a> - computation graphs, shapes, theoretical versus measured cycles, and module timelines.</p>
    <p><a href="power.html">Power</a> - workload event and energy breakdown with current evidence level.</p>
    <p><a href="area.html">Area</a> - resource and storage contribution breakdown with current evidence level.</p>
  </section>
  <div class="metrics">
    <div class="metric">Area model<strong>{area['normalized_area_units']}</strong>normalized units</div>
    <div class="metric">Stored bits<strong>{area['storage_bits_total']}</strong>current local state</div>
    <div class="metric">MAC lanes<strong>{area['resources']['int8_mac_lanes']}</strong>INT8 lanes</div>
  </div>
  {comparison_html}
  {highlights}
  <section>
    <h2>Workloads</h2>
    <table>
      <thead><tr><th>Name</th><th>Measured total</th><th>Theoretical compute</th><th>Measured compute</th><th>Compute overhead</th><th>Compute efficiency</th><th>End-to-end efficiency</th><th>Attention stage</th><th>Derived MAC ops</th><th>Matrix util</th><th>GEMV util</th><th>KV read bytes</th><th>Moved words</th><th>Energy model</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Limitations</h2>
    <ul>{limitations}</ul>
  </section>
</body>
</html>
""",
        encoding="utf-8",
    )


def _write_dimension_pages(report: dict[str, Any], out_dir: Path) -> None:
    nav = '<nav><a href="ppa_overview.html">Overview</a> | <a href="perf.html">Performance</a> | <a href="power.html">Power</a> | <a href="area.html">Area</a></nav>'
    style = """
    body{font-family:system-ui,sans-serif;margin:32px;color:#17202a}
    nav{margin-bottom:24px} section{border:1px solid #d9dee8;padding:14px;border-radius:7px;margin:16px 0}
    table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #d9dee8}
    .flow{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.node{border:1px solid #7ea4d8;border-radius:6px;padding:10px;background:#f3f7fc}
    .note{background:#fff4df;border:1px solid #dfb052;padding:12px}code{background:#f2f4f8;padding:2px 4px}
    """
    perf_rows = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in report["workloads"]:
        metadata = item.get("metadata", {})
        group = metadata.get("attention_group") or metadata.get("graph") or metadata.get("workload_family") or item["kind"]
        groups.setdefault(str(group), []).append(item)
        p = item["performance"]
        if p.get("attention_stage") == "softmax":
            bottleneck = "Core serial vector/reduction/SFU sequence; SFU issue count dominates"
        elif p.get("measured_compute_cycles") and p.get("cycles", 0) > p["measured_compute_cycles"] * 2:
            bottleneck = "Descriptor and data movement dominate end-to-end latency"
        elif p.get("measured_compute_cycles"):
            bottleneck = "Compute dominates"
        else:
            bottleneck = "Model-only or insufficient measured module counters"
        perf_rows.append(
            "<tr>"
            f"<td>{html.escape(item['name'])}</td>"
            f"<td>{html.escape(str(metadata.get('logical_op', item['kind'])))}</td>"
            f"<td><code>{html.escape(json.dumps(metadata.get('logical_shape', {})))}</code></td>"
            f"<td>{p.get('theoretical_compute_cycles')}</td><td>{p.get('measured_compute_cycles')}</td>"
            f"<td>{p.get('cycles')}</td><td>{p.get('compute_efficiency')}</td>"
            f"<td>{html.escape(bottleneck)}</td>"
            f"<td><code>{html.escape(str(p.get('theoretical_cycle_basis')))}</code></td></tr>"
        )
    flows = []
    for group, items in groups.items():
        nodes = " <strong>→</strong> ".join(
            f'<span class="node"><strong>{html.escape(str(item.get("metadata", {}).get("logical_op", item["name"])))}</strong><br>'
            f'{html.escape(json.dumps(item.get("metadata", {}).get("logical_shape", {})))}</span>'
            for item in items
        )
        flows.append(f"<section><h2>{html.escape(group)}</h2><div class=\"flow\">{nodes}</div></section>")
    case_name = str(report.get("test_case", {}).get("name", "unspecified"))
    perf_page = f"""<!doctype html><html><head><meta charset="utf-8"><title>PPA Performance</title><style>{style}</style></head><body>
    {nav}<h1>Performance</h1><p class="note">Cycle counters are measured from architectural CSR snapshots. Detailed job timelines distinguish measured counters from state-machine-derived span placement.</p>
    {''.join(flows)}
    <section><h2>Theoretical Versus Measured</h2><table><thead><tr><th>Workload</th><th>Operator/layer</th><th>Shape</th><th>Theoretical compute</th><th>Measured compute</th><th>Measured total</th><th>Efficiency</th><th>Primary bottleneck</th><th>Formula</th></tr></thead><tbody>{''.join(perf_rows)}</tbody></table></section>
    <section><h2>Per-job Module Timelines</h2><p><a href="cases/{html.escape(case_name)}.html">Open the {html.escape(case_name)} test-case report</a>. It includes wrapper, data mover, core, matrix, vector, reduction, and SFU lanes where applicable.</p></section>
    </body></html>"""
    (out_dir / "perf.html").write_text(perf_page, encoding="utf-8")

    power_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td><td>{item['energy_model']['normalized_energy_units']}</td>"
        f"<td>{item['energy_model']['contribution_groups']['measured_onchip_events']}</td>"
        f"<td>{item['energy_model']['contribution_groups']['modeled_external_memory']}</td>"
        f"<td><code>{html.escape(json.dumps(item['energy_model']['events']))}</code></td></tr>"
        for item in report["workloads"]
    )
    (out_dir / "power.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>PPA Power</title><style>{style}</style></head><body>{nav}
        <h1>Power And Energy</h1><p class="note">Current Level 0 evidence is an event-energy model in normalized units. Watts and joules require an activity-based implementation flow.</p>
        <section><table><thead><tr><th>Workload</th><th>Total energy model</th><th>On-chip events</th><th>External memory</th><th>Event counts</th></tr></thead><tbody>{power_rows}</tbody></table></section>
        </body></html>""",
        encoding="utf-8",
    )

    area = report["area_model"]
    area_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value}</td></tr>"
        for name, value in area["contributions"].items()
    )
    storage_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value}</td></tr>"
        for name, value in area["resources"].get("storage_bits", {}).items()
    )
    (out_dir / "area.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>PPA Area</title><style>{style}</style></head><body>{nav}
        <h1>Area</h1><p class="note">Current Level 0 evidence is a structural area model in normalized units. Physical area requires synthesis or physical implementation.</p>
        <section><h2>Contribution Breakdown</h2><table><tbody>{area_rows}</tbody></table></section>
        <section><h2>Storage Bits</h2><table><tbody>{storage_rows}</tbody></table></section>
        </body></html>""",
        encoding="utf-8",
    )
    _write_case_page(report, out_dir, style)


def _write_case_page(report: dict[str, Any], out_dir: Path, style: str) -> None:
    case_name = str(report.get("test_case", {}).get("name", "unspecified"))
    case_dir = out_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    items = [
        item
        for item in report["workloads"]
        if item.get("metadata", {}).get("model_only") is not True
    ]
    if case_name == "transformer":
        graph_items = [
            item for item in items
            if str(item.get("metadata", {}).get("workload_family", "")).startswith("transformer_")
        ]
    elif case_name == "cnn-full":
        graph_items = [
            item for item in items
            if item.get("metadata", {}).get("graph") or item.get("kind") in ("model", "model_layer", "model_layer_tile")
        ]
    else:
        graph_items = items
    graph_groups: dict[str, list[dict[str, Any]]] = {}
    for item in graph_items:
        metadata = item.get("metadata", {})
        group = metadata.get("attention_group") or metadata.get("workload_family") or item["kind"]
        graph_groups.setdefault(str(group), []).append(item)
    graph = "".join(
        f'<h3>{html.escape(group)}</h3><div class="flow">'
        + " <strong>→</strong> ".join(
            f'<span class="node"><strong>{html.escape(str(item.get("metadata", {}).get("logical_op", item["name"])))}</strong><br>'
            f'{html.escape(json.dumps(item.get("metadata", {}).get("logical_shape", {})))}</span>'
            for item in group_items
        )
        + "</div>"
        for group, group_items in graph_groups.items()
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td>{html.escape(str(item.get('metadata', {}).get('logical_op', item['kind'])))}</td>"
        f"<td><code>{html.escape(json.dumps(item.get('metadata', {}).get('logical_shape', {})))}</code></td>"
        f"<td>{item['performance'].get('theoretical_compute_cycles')}</td>"
        f"<td>{item['performance'].get('measured_compute_cycles')}</td>"
        f"<td>{item['performance'].get('cycles')}</td>"
        f"<td>{item['energy_model'].get('normalized_energy_units')}</td>"
        f"<td>{report['area_model'].get('normalized_area_units')}</td>"
        "</tr>"
        for item in graph_items
    )
    case_job_ids = {
        int(job_id)
        for item in graph_items
        for job_id in item.get("job_ids", [])
    }
    timeline_jobs = [
        job for job in report.get("pipeline_jobs", [])
        if int(job.get("job_id", job.get("id", -1))) in case_job_ids
    ]
    timelines = "".join(_render_static_timeline(job) for job in timeline_jobs)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>PPA Case {html.escape(case_name)}</title>
    <style>{style}.timeline{{display:grid;grid-template-columns:150px 1fr 100px;gap:7px;align-items:center}}.track{{position:relative;height:28px;background:#eef1f6;border-radius:5px}}.bar{{position:absolute;top:4px;height:20px;background:#2068d8;border-radius:4px;min-width:2px}}.lane-value{{text-align:right}}</style></head><body>
    <nav><a href="../ppa_overview.html">Overview</a> | <a href="../perf.html">Performance</a> | <a href="../power.html">Power</a> | <a href="../area.html">Area</a></nav>
    <h1>Test Case: {html.escape(case_name)}</h1>
    <section><h2>Computation Graph</h2><div class="flow">{graph}</div></section>
    <section><h2>Per-operator PPA</h2><table><thead><tr><th>Workload</th><th>Operator/layer</th><th>Shape</th><th>Theoretical cycles</th><th>Measured compute</th><th>Measured total</th><th>Energy model</th><th>Area model</th></tr></thead><tbody>{rows}</tbody></table></section>
    <section><h2>Pipeline Timeline</h2><p>Only jobs belonging to this test-case graph are shown. Span placement provenance is displayed per job.</p>{timelines}</section>
    </body></html>"""
    (case_dir / f"{case_name}.html").write_text(page, encoding="utf-8")


def _render_static_timeline(job: dict[str, Any]) -> str:
    total = max(1, int(job.get("total_cycles", 0)))
    lanes = []
    for lane in job.get("timeline", []):
        bars = "".join(
            f'<span class="bar" title="{html.escape(span["label"])}: {span["start"]}-{span["end"]}" '
            f'style="left:{span["start"] * 100 / total:.3f}%;width:{max(0.5, span["cycles"] * 100 / total):.3f}%"></span>'
            for span in lane.get("spans", [])
        )
        active = sum(int(span.get("cycles", 0)) for span in lane.get("spans", []))
        lanes.append(
            f'<div>{html.escape(lane["module"])}</div><div class="track">{bars}</div><div class="lane-value">{active} cycles</div>'
        )
    provenance = job.get("timeline_provenance", {})
    return (
        f'<h3>#{job.get("job_id", job.get("id"))} {html.escape(job.get("name", "unknown"))} - {total} cycles</h3>'
        f'<p><small>Counters: {html.escape(str(provenance.get("summary_counters", "legacy")))}; '
        f'placement: {html.escape(str(provenance.get("span_placement", "measured_or_legacy")))}</small></p>'
        f'<div class="timeline">{"".join(lanes)}</div>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an NPU Level 0 PPA report.")
    parser.add_argument("--perf-json", required=True, type=Path)
    parser.add_argument("--area-config", required=True, type=Path)
    parser.add_argument("--energy-config", required=True, type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--html-out", required=True, type=Path)
    args = parser.parse_args()

    perf = json.loads(args.perf_json.read_text(encoding="utf-8"))
    baseline_report = None
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        if baseline.get("schema") == SCHEMA_NAME:
            baseline_report = baseline
        else:
            baseline_report = build_ppa_report(
                baseline,
                read_jsonc(Path(baseline["area_config"])),
                read_jsonc(Path(baseline["energy_config"])),
            )
    report = build_ppa_report(
        perf,
        read_jsonc(args.area_config),
        read_jsonc(args.energy_config),
        baseline_report=baseline_report,
    )
    write_json(report, args.json_out)
    write_html(report, args.html_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
