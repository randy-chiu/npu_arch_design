"""Phase 0 JSON micro-op validation."""

from __future__ import annotations

from typing import Any


class ISAError(ValueError):
    """Raised when a program violates the Phase 0 ISA contract."""


def validate_program(program: list[dict[str, Any]], arch: dict[str, Any]) -> None:
    legal = set(arch["isa"]["instructions"])
    if not program:
        raise ISAError("program must not be empty")
    if program[-1].get("op") != "HALT":
        raise ISAError("program must end with HALT")

    for pc, inst in enumerate(program):
        op = inst.get("op")
        if op not in legal:
            raise ISAError(f"pc {pc}: illegal op {op!r}")
        _validate_instruction(pc, inst, arch)


def _validate_instruction(pc: int, inst: dict[str, Any], arch: dict[str, Any]) -> None:
    op = inst["op"]
    if op in {"LOAD", "STORE"}:
        _require_fields(pc, inst, ["tensor", "buffer"])
    elif op == "MATMUL":
        _require_fields(pc, inst, ["a", "b", "out"])
        shape = inst.get("shape", {})
        _require_shape(pc, shape, ["m", "n", "k"])
        if not arch["scope"].get("edge_tiles", False):
            tile_m = arch["compute"]["array_m"]
            tile_n = arch["compute"]["array_n"]
            tile_k = arch["compute"]["k_step"]
            if shape["m"] % tile_m or shape["n"] % tile_n or shape["k"] % tile_k:
                raise ISAError(f"pc {pc}: MATMUL shape must be a multiple of tile size")
    elif op == "HALT":
        return


def _require_fields(pc: int, inst: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if field not in inst]
    if missing:
        raise ISAError(f"pc {pc}: missing fields {missing}")


def _require_shape(pc: int, shape: dict[str, Any], fields: list[str]) -> None:
    for field in fields:
        value = shape.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ISAError(f"pc {pc}: shape.{field} must be a positive integer")
