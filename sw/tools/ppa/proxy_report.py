from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_NAME = "npu_ppa_proxy_report_v0"
EVIDENCE_LEVEL = "L0_proxy"


def read_jsonc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(re.sub(r"//.*$", "", text, flags=re.MULTILINE))


def build_proxy_report(
    perf: dict[str, Any],
    area_cfg: dict[str, Any],
    energy_cfg: dict[str, Any],
    baseline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    area_proxy = build_area_proxy(area_cfg)
    performance_provenance = perf.get("source", {}).get(
        "performance", "measured_rtl_perf_job_counters"
    )
    workload_results = [
        build_workload_proxy(workload, energy_cfg, performance_provenance)
        for workload in perf.get("workloads", [])
    ]
    workload_results.extend(
        build_workload_proxy(workload, energy_cfg, "modeled_manifest_only")
        for workload in perf.get("model_only_workloads", [])
    )
    manifest = perf.get("workload_manifest") or {}
    report = {
        "schema": SCHEMA_NAME,
        "evidence_level": EVIDENCE_LEVEL,
        "source_perf_report": perf.get("source_log", ""),
        "workload_manifest_id": manifest.get("id") or perf.get("workload_manifest_id"),
        "design": area_cfg["design"],
        "proxy_config": {
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
        "area_proxy": area_proxy,
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


def build_area_proxy(config: dict[str, Any]) -> dict[str, Any]:
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


def build_workload_proxy(
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
        "jobs": int(workload.get("jobs", 0)),
        "performance": {
            "cycles": int(workload["total_cycles"]),
            "core_matmul_cycles": int(workload.get("core_matmul_cycles", 0)),
            "data_mover_words": int(workload.get("data_mover", {}).get("words", 0)),
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
            "provenance": performance_provenance,
        },
        "energy_proxy": {
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
    baseline_area = float(baseline["area_proxy"]["normalized_area_units"])
    candidate_area = float(candidate["area_proxy"]["normalized_area_units"])
    area_delta = delta_metric(baseline_area, candidate_area, lower_is_better=True)
    area_delta["metric"] = "normalized_area_units"
    area_delta["interpretation"] = "structural_proxy_not_synthesized_area"

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
            old["energy_proxy"]["normalized_energy_units"],
            new["energy_proxy"]["normalized_energy_units"],
            lower_is_better=True,
        )
        moved_words = delta_metric(
            old["performance"]["data_mover_words"],
            new["performance"]["data_mover_words"],
            lower_is_better=True,
        )
        mac_ops = delta_metric(
            old["energy_proxy"]["events"]["int8_mac_accumulate"],
            new["energy_proxy"]["events"]["int8_mac_accumulate"],
            lower_is_better=False,
        )
        workload_deltas.append(
            {
                "name": name,
                "cycles": cycles,
                "energy_proxy": energy,
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
                f"{name}: event-energy proxy decreases by {abs(energy['delta'])} normalized units "
                f"({abs(energy['delta_pct']):.3f}%)."
            )
    if area_delta["classification"] == "regression":
        costs.append(
            f"Structural area proxy increases by {area_delta['delta']} normalized units "
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
            "area": "structural_proxy_not_synthesized_area",
            "energy": "event_proxy_not_measured_power",
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
    baseline_cfg = baseline.get("proxy_config", {})
    candidate_cfg = candidate.get("proxy_config", {})
    for key in (
        "area_coefficient_version",
        "area_units",
        "area_coefficients",
        "energy_coefficient_version",
        "energy_units",
        "energy_coefficients",
    ):
        if baseline_cfg.get(key) != candidate_cfg.get(key):
            issues.append(f"proxy_config.{key} differs")
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
        after_energy = current["energy_proxy"]["normalized_energy_units"]
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
                "energy_proxy_interpretation": energy_cfg["interpretation"],
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
    area = report["area_proxy"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['name'])}</td>"
        f"<td>{item['performance']['cycles']}</td>"
        f"<td>{item['energy_proxy']['events']['int8_mac_accumulate']}</td>"
        f"<td>{html.escape(str(item['performance'].get('matrix_utilization')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('gemv_utilization')))}</td>"
        f"<td>{html.escape(str(item['performance'].get('kv_read_bytes')))}</td>"
        f"<td>{item['performance']['data_mover_words']}</td>"
        f"<td>{item['energy_proxy']['normalized_energy_units']}</td>"
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
            f"<td>{item['energy_proxy']['baseline']} -> {item['energy_proxy']['candidate']} "
            f"({item['energy_proxy']['delta_pct']}%)</td>"
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
    <p>Structural area proxy: {area_delta['baseline']} -> {area_delta['candidate']}
       ({area_delta['delta_pct']}%, {area_delta['classification']}).</p>
    <table>
      <thead><tr><th>Common workload</th><th>Measured cycles</th><th>Energy proxy</th><th>Moved words</th><th>MAC work</th></tr></thead>
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
  <title>NPU Level 0 PPA Proxy Report</title>
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
  <h1>NPU Level 0 PPA Proxy Report</h1>
  <p class="warning"><strong>Interpretation:</strong> Performance and movement are measured from architectural
  perf CSR snapshot values carried in <code>PERF_JOB</code>. Area and energy are normalized proxies, not synthesized area,
  watts, or joules.</p>
  <div class="metrics">
    <div class="metric">Area proxy<strong>{area['normalized_area_units']}</strong>normalized units</div>
    <div class="metric">Stored bits<strong>{area['storage_bits_total']}</strong>current local state</div>
    <div class="metric">MAC lanes<strong>{area['resources']['int8_mac_lanes']}</strong>INT8 lanes</div>
  </div>
  {comparison_html}
  {highlights}
  <section>
    <h2>Workloads</h2>
    <table>
      <thead><tr><th>Name</th><th>Measured cycles</th><th>Derived MAC ops</th><th>Matrix util</th><th>GEMV util</th><th>KV read bytes</th><th>Moved words</th><th>Energy proxy</th></tr></thead>
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an NPU Level 0 PPA proxy report.")
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
            baseline_report = build_proxy_report(
                baseline,
                read_jsonc(Path(baseline["area_config"])),
                read_jsonc(Path(baseline["energy_config"])),
            )
    report = build_proxy_report(
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
