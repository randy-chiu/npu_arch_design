"""Phase 0 uop assembler."""

from __future__ import annotations

from typing import Any


def encode_uop(arch: dict[str, Any], opcode: str, arg0: int = 0, arg1: int = 0) -> int:
    """Encode the temporary Phase 0 RTL micro-op format."""

    encoding = arch["isa"]["encoding"]
    opcode_field = encoding["opcode"]
    arg0_field = encoding["arg0"]
    arg1_field = encoding["arg1"]
    return (
        (_lookup_encoding(encoding["opcodes"], opcode) << opcode_field["lsb"])
        | ((arg0 & _field_mask(arg0_field)) << arg0_field["lsb"])
        | ((arg1 & _field_mask(arg1_field)) << arg1_field["lsb"])
    )


def encode_program(program: list[dict[str, Any]], arch: dict[str, Any]) -> list[int]:
    """Encode compiler-emitted Phase 0 JSON micro-ops for RTL/firmware use."""

    encoded: list[int] = []
    for inst in program:
        op = inst["op"]
        if op in {"LOAD", "STORE"}:
            encoded.append(
                encode_uop(
                    arch,
                    op,
                    _tensor_id(arch, inst["tensor"]),
                    _buffer_id(arch, inst["buffer"]),
                )
            )
        elif op == "MATMUL":
            encoded.append(encode_uop(arch, op))
        elif op in {"VREDMAX", "VEXP", "VREDSUM"}:
            encoded.append(
                encode_uop(arch, op, _buffer_id(arch, inst["src"]), _buffer_id(arch, inst["dst"]))
            )
        elif op in {"VSUB", "VDIV"}:
            encoded.append(
                encode_uop(arch, op, _buffer_id(arch, inst["src"]), _buffer_id(arch, inst["dst"]))
            )
        elif op == "HALT":
            encoded.append(encode_uop(arch, op))
        else:
            raise ValueError(f"unsupported RTL op: {op}")
    return encoded


def _tensor_id(arch: dict[str, Any], name: str) -> int:
    return _lookup_encoding(arch["isa"]["encoding"]["tensors"], name)


def _buffer_id(arch: dict[str, Any], name: str) -> int:
    encoding = arch["isa"]["encoding"]
    canonical = encoding.get("buffer_aliases", {}).get(name, name)
    if canonical.startswith("acc_"):
        canonical = "acc"
    return _lookup_encoding(encoding["buffers"], canonical)


def _lookup_encoding(values: dict[str, int], name: str) -> int:
    if name not in values:
        raise ValueError(f"missing ISA encoding for {name!r}")
    return values[name]


def _field_mask(field: dict[str, int]) -> int:
    return (1 << (field["msb"] - field["lsb"] + 1)) - 1
