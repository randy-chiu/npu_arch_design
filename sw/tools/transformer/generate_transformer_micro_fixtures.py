from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from transformer.golden import (
    TILE_K,
    TILE_M,
    TILE_N,
    TILE_WORDS,
    deterministic_i8_matrix,
    tile_k_stream,
)
from transformer.micro_golden import classify_matrix_shape


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
    executable = []
    model_only = []
    for index, workload in enumerate(spec["workloads"]):
        if workload["op"] == "matmul" and workload["status"] == "planned_current_matmul_extension":
            executable.append(_generate_matmul_workload(workload, precision, seed=index + 1))
        else:
            model_only.append(_model_only_metadata(workload, precision))
    return {
        "spec_name": spec["name"],
        "version": int(spec["version"]),
        "executable_workloads": executable,
        "model_only_workloads": model_only,
    }


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

    a = deterministic_i8_matrix(m_dim, k_dim, seed)
    b = deterministic_i8_matrix(k_dim, n_dim, seed + 17)
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
        "kv_read_bytes": int(workload["external_memory"].get("kv_cache_read_bytes", 0)),
        "kv_write_bytes": int(workload["external_memory"].get("kv_cache_write_bytes", 0)),
        "k_chunks": tiled["k_chunks"],
        "tile_jobs": 1,
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
