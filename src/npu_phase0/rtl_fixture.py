"""Generate deterministic RTL simulation fixtures for Phase 0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .compiler import compile_graph
from .golden import matmul


OPCODES = {
    "LOAD": 0x1,
    "STORE": 0x2,
    "MATMUL": 0x3,
    "VREDMAX": 0x4,
    "VSUB": 0x5,
    "VEXP": 0x6,
    "VREDSUM": 0x7,
    "VDIV": 0x8,
    "HALT": 0xF,
}

TENSOR_IDS = {
    "A": 0x0,
    "B": 0x1,
    "C": 0x2,
    "X": 0x3,
    "Y": 0x4,
}

BUFFER_IDS = {
    "spad_a": 0x0,
    "spad_b": 0x1,
    "acc": 0x2,
    "vec": 0x3,
}

BUFFER_ALIASES = {
    "spad_softmax": "vec",
    "scalar_max": "vec",
    "scalar_sum": "vec",
}

DEFAULT_GRAPH_PATH = Path("tests/graphs/matmul_softmax.json")
DEFAULT_INPUTS_PATH = Path("tests/inputs_matmul_softmax.json")


def encode_uop(opcode: str, arg0: int = 0, arg1: int = 0) -> int:
    """Encode the temporary Phase 0 RTL micro-op format."""

    return (OPCODES[opcode] << 28) | ((arg0 & 0xF) << 24) | ((arg1 & 0xF) << 20)


def encode_program(program: list[dict[str, Any]]) -> list[int]:
    """Encode compiler-emitted Phase 0 JSON micro-ops for the RTL testbench."""

    encoded: list[int] = []
    for inst in program:
        op = inst["op"]
        if op in {"LOAD", "STORE"}:
            encoded.append(
                encode_uop(op, TENSOR_IDS[inst["tensor"]], _buffer_id(inst["buffer"]))
            )
        elif op == "MATMUL":
            encoded.append(encode_uop(op))
        elif op in {"VREDMAX", "VEXP", "VREDSUM"}:
            encoded.append(encode_uop(op, _buffer_id(inst["src"]), _buffer_id(inst["dst"])))
        elif op in {"VSUB", "VDIV"}:
            encoded.append(encode_uop(op, _buffer_id(inst["src"]), _buffer_id(inst["dst"])))
        elif op == "HALT":
            encoded.append(encode_uop(op))
        else:
            raise ValueError(f"unsupported RTL op: {op}")
    return encoded


def generate_default_fixtures(
    out_dir: Path,
    arch: dict[str, Any],
    graph_path: Path = DEFAULT_GRAPH_PATH,
    inputs_path: Path = DEFAULT_INPUTS_PATH,
) -> None:
    """Emit compiled RTL fixtures under ``out_dir``."""

    out_dir.mkdir(parents=True, exist_ok=True)

    graph = _read_json(graph_path)
    inputs = _read_json(inputs_path)
    matmul_op = _first_op(graph, "matmul")
    a = inputs[matmul_op["a"]]
    b = inputs[matmul_op["b"]]
    c = matmul(a, b)
    x = c[0]
    y = softmax_q0_8(x)

    _write_hex(out_dir / "matmul_a.hex", _flatten(a), 2)
    _write_hex(out_dir / "matmul_b.hex", _flatten(b), 2)
    _write_hex(out_dir / "matmul_expected_c.hex", _flatten(c), 8)
    matmul_artifact = compile_graph(_single_op_graph(graph, matmul_op), arch)
    _write_hex(
        out_dir / "matmul_program.hex",
        _pad_program(encode_program(matmul_artifact["program"])),
        8,
    )

    _write_hex(out_dir / "softmax_x.hex", x, 2)
    _write_hex(out_dir / "softmax_expected_y.hex", y, 2)
    softmax_artifact = compile_graph(
        {
            "tensors": {
                "X": {"shape": [1, 8], "dtype": "int8"},
            },
            "ops": [{"type": "softmax", "x": "X", "out": "Y"}],
        },
        arch,
    )
    _write_hex(
        out_dir / "softmax_program.hex",
        _pad_program(encode_program(softmax_artifact["program"])),
        8,
    )


def softmax_q0_8(values: list[int]) -> list[int]:
    """Match the tiny integer softmax approximation in ``npu_v0_top.sv``."""

    max_v = max(values)
    exp_values = [_exp_lut_q8(v - max_v) for v in values]
    denom = sum(exp_values)
    if denom == 0:
        return [0 for _ in values]
    return [((v * 255) // denom) & 0xFF for v in exp_values]


def _exp_lut_q8(delta: int) -> int:
    if delta >= 0:
        return 255
    return {
        -1: 94,
        -2: 35,
        -3: 13,
        -4: 5,
        -5: 2,
        -6: 1,
    }.get(delta, 0)


def _flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def _pad_program(program: list[int], depth: int = 16) -> list[int]:
    if len(program) > depth:
        raise ValueError(f"program length {len(program)} exceeds RTL instruction depth {depth}")
    return program + [encode_uop("HALT")] * (depth - len(program))


def _buffer_id(name: str) -> int:
    canonical = BUFFER_ALIASES.get(name, name)
    if canonical.startswith("acc_"):
        canonical = "acc"
    return BUFFER_IDS[canonical]


def _write_hex(path: Path, values: Iterable[int], width: int) -> None:
    mask = (1 << (width * 4)) - 1
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{int(value) & mask:0{width}x}\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_op(graph: dict[str, Any], op_type: str) -> dict[str, Any]:
    for op in graph.get("ops", []):
        if op.get("type") == op_type:
            return op
    raise ValueError(f"graph does not contain op type {op_type!r}")


def _single_op_graph(graph: dict[str, Any], op: dict[str, Any]) -> dict[str, Any]:
    if op["type"] != "matmul":
        raise ValueError("only matmul single-op graph extraction is supported")
    return {
        "tensors": {
            op["a"]: graph["tensors"][op["a"]],
            op["b"]: graph["tensors"][op["b"]],
        },
        "ops": [op],
    }
