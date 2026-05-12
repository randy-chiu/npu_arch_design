"""Minimal graph-to-micro-op compiler."""

from __future__ import annotations

from typing import Any, Optional

from .isa import validate_program


def compile_graph(graph: dict[str, Any], arch: dict[str, Any]) -> dict[str, Any]:
    tensors = graph.get("tensors", {})
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
                [
                    {"op": "LOAD", "tensor": a, "buffer": "spad_a"},
                    {"op": "LOAD", "tensor": b, "buffer": "spad_b"},
                    {
                        "op": "MATMUL",
                        "a": "spad_a",
                        "b": "spad_b",
                        "out": f"acc_{out}",
                        "shape": shape,
                    },
                    {"op": "STORE", "tensor": out, "buffer": f"acc_{out}"},
                ]
            )
            tensors[out] = {"shape": [shape["m"], shape["n"]], "dtype": "int32"}
            live_buffer += 1
        elif op_type == "softmax":
            src = op["x"]
            out = op["out"]
            in_buffer = "spad_softmax"
            program.extend(
                [
                    {"op": "LOAD", "tensor": src, "buffer": in_buffer},
                    {"op": "VREDMAX", "src": in_buffer, "dst": "scalar_max"},
                    {"op": "VSUB", "src": in_buffer, "dst": in_buffer, "scalar": "scalar_max"},
                    {"op": "VEXP", "src": in_buffer, "dst": in_buffer},
                    {"op": "VREDSUM", "src": in_buffer, "dst": "scalar_sum"},
                    {"op": "VDIV", "src": in_buffer, "dst": in_buffer, "scalar": "scalar_sum"},
                    {"op": "STORE", "tensor": out, "buffer": in_buffer},
                ]
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
