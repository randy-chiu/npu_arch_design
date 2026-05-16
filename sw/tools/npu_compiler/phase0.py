"""Phase 0 graph-to-uop compiler."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from npu_phase0.isa import validate_program


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPERATOR_TEMPLATE_PATH = _REPO_ROOT / "sw/npu_core/operators/phase0_intrinsics.json"


def compile_graph(
    graph: dict[str, Any],
    arch: dict[str, Any],
    operator_template_path: Path = DEFAULT_OPERATOR_TEMPLATE_PATH,
) -> dict[str, Any]:
    tensors = deepcopy(graph.get("tensors", {}))
    operators = _read_operator_templates(operator_template_path)
    program: list[dict[str, Any]] = []
    live_buffer = 0

    for op in graph.get("ops", []):
        op_type = op.get("type")
        if op_type == "matmul":
            a = op["a"]
            b = op["b"]
            out = op["out"]
            shape = _matmul_shape(tensors, a, b, op.get("shape"))
            program.extend(
                _instantiate_operator(
                    operators,
                    "matmul",
                    {
                        "a": a,
                        "b": b,
                        "out": out,
                        "shape": shape,
                    },
                )
            )
            tensors[out] = {"shape": [shape["m"], shape["n"]], "dtype": "int32"}
            live_buffer += 1
        elif op_type == "softmax":
            src = op["x"]
            out = op["out"]
            program.extend(
                _instantiate_operator(
                    operators,
                    "softmax",
                    {
                        "x": src,
                        "out": out,
                    },
                )
            )
            tensors[out] = {"shape": tensors[src]["shape"], "dtype": "fp32"}
            live_buffer += 1
        else:
            raise ValueError(f"unsupported op type: {op_type}")

    if live_buffer == 0:
        raise ValueError("graph must contain at least one op")

    program.append({"op": "HALT"})
    validate_program(program, arch)
    return {
        "arch": arch["name"],
        "format": arch["isa"]["program_format"],
        "program": program,
        "tensors": tensors,
    }


def _read_operator_templates(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["operators"]


def _instantiate_operator(
    operators: dict[str, Any],
    name: str,
    bindings: dict[str, Any],
) -> list[dict[str, Any]]:
    if name not in operators:
        raise ValueError(f"missing operator template: {name}")
    return [_substitute_value(inst, bindings) for inst in operators[name]["instructions"]]


def _substitute_value(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _substitute_value(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_value(item, bindings) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith("$") and value[1:] in bindings:
        return bindings[value[1:]]
    out = value
    for key, bound in bindings.items():
        if isinstance(bound, str):
            out = out.replace(f"${key}", bound)
    return out


def _matmul_shape(
    tensors: dict[str, Any],
    a: str,
    b: str,
    explicit: Optional[dict[str, int]],
) -> dict[str, int]:
    if explicit:
        return {"m": explicit["m"], "n": explicit["n"], "k": explicit["k"]}
    a_shape = tensors[a]["shape"]
    b_shape = tensors[b]["shape"]
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("matmul tensors must be 2D")
    if a_shape[1] != b_shape[0]:
        raise ValueError("matmul shape mismatch")
    return {"m": a_shape[0], "n": b_shape[1], "k": a_shape[1]}
