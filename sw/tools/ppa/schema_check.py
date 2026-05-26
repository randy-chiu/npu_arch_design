from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_NAME = "npu_ppa_proxy_report_v0"
EVIDENCE_LEVEL = "L0_proxy"


def validate_proxy_report(report: dict[str, Any]) -> None:
    errors: list[str] = []

    def require_object(parent: dict[str, Any], name: str, prefix: str = "") -> dict[str, Any]:
        value = parent.get(name)
        if not isinstance(value, dict):
            errors.append(f"{prefix}{name} must be an object")
            return {}
        return value

    if report.get("schema") != SCHEMA_NAME:
        errors.append(f"schema must be {SCHEMA_NAME!r}")
    if report.get("evidence_level") != EVIDENCE_LEVEL:
        errors.append(f"evidence_level must be {EVIDENCE_LEVEL!r}")
    require_object(report, "design")
    require_object(report, "metric_provenance")
    config = require_object(report, "proxy_config")
    for key in (
        "area_coefficient_version",
        "area_units",
        "area_coefficients",
        "energy_coefficient_version",
        "energy_units",
        "energy_coefficients",
    ):
        if key not in config:
            errors.append(f"proxy_config.{key} is required")
    area = require_object(report, "area_proxy")
    if area.get("units") != "normalized_area_units":
        errors.append("area_proxy.units must be 'normalized_area_units'")
    if "normalized_area_units" not in area:
        errors.append("area_proxy.normalized_area_units is required")
    else:
        _require_nonnegative(errors, "area_proxy.normalized_area_units", area["normalized_area_units"])
    area_contributions = area.get("contributions")
    if isinstance(area_contributions, dict) and "normalized_area_units" in area:
        _check_sum(
            errors,
            "area_proxy.normalized_area_units",
            area["normalized_area_units"],
            area_contributions,
        )
    resources = area.get("resources", {})
    if isinstance(resources, dict):
        for key in ("int8_mac_lanes", "data_mover_lanes", "wrapper_control_units"):
            if key in resources:
                _require_nonnegative(errors, f"area_proxy.resources.{key}", resources[key])
        for key, value in resources.get("storage_bits", {}).items():
            _require_nonnegative(errors, f"area_proxy.resources.storage_bits.{key}", value)
    workloads = report.get("workloads")
    if not isinstance(workloads, list):
        errors.append("workloads must be an array")
        workloads = []
    workload_names: set[str] = set()
    for index, workload in enumerate(workloads):
        if not isinstance(workload, dict) or "name" not in workload:
            errors.append(f"workloads[{index}].name is required")
            continue
        if workload["name"] in workload_names:
            errors.append(f"workloads[{index}].name duplicates {workload['name']!r}")
        workload_names.add(workload["name"])
        performance = require_object(workload, "performance", f"workloads[{index}].")
        for key in ("cycles", "core_matmul_cycles", "data_mover_words", "provenance"):
            if key not in performance:
                errors.append(f"workloads[{index}].performance.{key} is required")
            elif key != "provenance":
                _require_nonnegative(errors, f"workloads[{index}].performance.{key}", performance[key])
        energy = require_object(workload, "energy_proxy", f"workloads[{index}].")
        if energy.get("units") != "normalized_energy_units":
            errors.append(
                f"workloads[{index}].energy_proxy.units must be 'normalized_energy_units'"
            )
        for key in ("events", "coefficients", "normalized_energy_units"):
            if key not in energy:
                errors.append(f"workloads[{index}].energy_proxy.{key} is required")
        events = energy.get("events", {})
        if isinstance(events, dict):
            for key, value in events.items():
                _require_nonnegative(errors, f"workloads[{index}].energy_proxy.events.{key}", value)
        if "normalized_energy_units" in energy:
            _require_nonnegative(
                errors,
                f"workloads[{index}].energy_proxy.normalized_energy_units",
                energy["normalized_energy_units"],
            )
            if isinstance(energy.get("contributions"), dict):
                _check_sum(
                    errors,
                    f"workloads[{index}].energy_proxy.normalized_energy_units",
                    energy["normalized_energy_units"],
                    energy["contributions"],
                )
    if not isinstance(report.get("limitations"), list) or not report["limitations"]:
        errors.append("limitations must be a non-empty array")
    if "comparison" not in report:
        errors.append("comparison is required and must be null or an object")
    elif isinstance(report["comparison"], dict) and report["comparison"].get("comparable"):
        comparison = report["comparison"]
        if comparison.get("compatibility", {}).get("issues"):
            errors.append("comparable comparison must not declare compatibility issues")
        area_delta = comparison.get("area_delta")
        if isinstance(area_delta, dict):
            _check_delta(errors, "comparison.area_delta", area_delta)
        for index, workload_delta in enumerate(comparison.get("workload_deltas", [])):
            for metric in ("cycles", "energy_proxy", "data_mover_words", "int8_mac_accumulate"):
                if metric in workload_delta:
                    _check_delta(
                        errors,
                        f"comparison.workload_deltas[{index}].{metric}",
                        workload_delta[metric],
                    )

    if errors:
        raise ValueError("invalid PPA proxy report:\n- " + "\n- ".join(errors))


def _require_nonnegative(errors: list[str], path: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or value < 0:
        errors.append(f"{path} must be a non-negative number")


def _check_sum(errors: list[str], path: str, total: Any, contributions: dict[str, Any]) -> None:
    if not isinstance(total, (int, float)):
        return
    if not all(isinstance(value, (int, float)) for value in contributions.values()):
        errors.append(f"{path} contributions must be numeric")
        return
    expected = round(sum(contributions.values()), 3)
    if abs(float(total) - expected) > 1e-6:
        errors.append(f"{path} must equal contribution sum {expected}")


def _check_delta(errors: list[str], path: str, delta: dict[str, Any]) -> None:
    for key in ("baseline", "candidate", "delta"):
        if not isinstance(delta.get(key), (int, float)):
            errors.append(f"{path}.{key} must be numeric")
            return
    expected = round(delta["candidate"] - delta["baseline"], 3)
    if abs(float(delta["delta"]) - expected) > 1e-6:
        errors.append(f"{path}.delta must equal candidate - baseline ({expected})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an NPU Level 0 PPA proxy report.")
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.json.read_text(encoding="utf-8"))
    validate_proxy_report(report)
    print(f"Validated {args.json} ({SCHEMA_NAME}, {EVIDENCE_LEVEL})")


if __name__ == "__main__":
    main()
