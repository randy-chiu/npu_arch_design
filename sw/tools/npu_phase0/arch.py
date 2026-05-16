"""Architecture spec loading and validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union


REQUIRED_TOP_LEVEL = {
    "name",
    "version",
    "scope",
    "data_types",
    "isa",
    "compute",
    "vector_sfu",
    "memory",
    "dma",
    "bus",
    "runtime",
    "verification",
}

REQUIRED_INSTRUCTIONS = {
    "LOAD",
    "STORE",
    "MATMUL",
    "VREDMAX",
    "VSUB",
    "VEXP",
    "VREDSUM",
    "VDIV",
    "HALT",
}


class ArchSpecError(ValueError):
    """Raised when an architecture spec is invalid."""


def load_arch(path: Union[str, Path]) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        text = f.read()
    if spec_path.suffix == ".jsonc":
        text = _strip_jsonc_comments(text)
    spec = json.loads(text)
    validate_arch(spec)
    return spec


def validate_arch(spec: dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL - set(spec)
    if missing:
        raise ArchSpecError(f"missing top-level fields: {sorted(missing)}")

    instructions = set(spec["isa"].get("instructions", []))
    missing_instr = REQUIRED_INSTRUCTIONS - instructions
    if missing_instr:
        raise ArchSpecError(f"missing ISA instructions: {sorted(missing_instr)}")
    _validate_isa_encoding(spec["isa"], instructions)

    compute = spec["compute"]
    for key in ("array_m", "array_n", "k_step", "mac_lanes"):
        _require_positive_int(compute, key)

    if compute["mac_lanes"] != compute["array_m"] * compute["array_n"]:
        raise ArchSpecError("compute.mac_lanes must equal array_m * array_n in Phase 0")

    memory = spec["memory"]
    for key in ("dram_bytes", "scratchpad_bytes", "accumulator_bytes", "alignment_bytes"):
        _require_positive_int(memory, key)

    bus = spec["bus"]
    _require_positive_int(bus, "data_width_bits")
    if bus["data_width_bits"] % 8 != 0:
        raise ArchSpecError("bus.data_width_bits must be byte aligned")

    dma = spec["dma"]
    _require_positive_int(dma, "channels")
    _require_positive_int(dma, "max_burst_bytes")
    if 1 not in dma.get("ranks", []):
        raise ArchSpecError("Phase 0 DMA must support rank-1 transfers")

    vector = spec["vector_sfu"]
    _require_positive_int(vector, "lanes")

    rtl = spec.get("rtl", {})
    _require_positive_int(rtl, "host_data_width_bits")
    _require_positive_int(rtl, "host_addr_width_bits")
    if rtl.get("matmul_tile") != [
        compute["array_m"],
        compute["array_n"],
        compute["k_step"],
    ]:
        raise ArchSpecError("rtl.matmul_tile must match compute tile shape in Phase 0")

    supported_ops = set(spec["scope"].get("operators", []))
    if not {"matmul", "softmax"}.issubset(supported_ops):
        raise ArchSpecError("Phase 0 scope must support matmul and softmax")


def _require_positive_int(parent: dict[str, Any], key: str) -> None:
    value = parent.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ArchSpecError(f"{key} must be a positive integer")


def _validate_isa_encoding(isa: dict[str, Any], instructions: set[str]) -> None:
    encoding = isa.get("encoding")
    if not isinstance(encoding, dict):
        raise ArchSpecError("isa.encoding must be defined")

    _require_positive_int(encoding, "word_bits")
    for field in ("opcode", "arg0", "arg1"):
        _validate_bit_field(encoding, field)

    opcodes = encoding.get("opcodes")
    if not isinstance(opcodes, dict):
        raise ArchSpecError("isa.encoding.opcodes must be defined")
    missing_opcode = instructions - set(opcodes)
    if missing_opcode:
        raise ArchSpecError(f"missing opcode encodings: {sorted(missing_opcode)}")
    _validate_unique_nibble_map("isa.encoding.opcodes", opcodes)

    for map_name in ("tensors", "buffers"):
        values = encoding.get(map_name)
        if not isinstance(values, dict) or not values:
            raise ArchSpecError(f"isa.encoding.{map_name} must be a non-empty map")
        _validate_unique_nibble_map(f"isa.encoding.{map_name}", values)

    aliases = encoding.get("buffer_aliases", {})
    if not isinstance(aliases, dict):
        raise ArchSpecError("isa.encoding.buffer_aliases must be a map")
    buffers = set(encoding["buffers"])
    bad_aliases = {alias: target for alias, target in aliases.items() if target not in buffers}
    if bad_aliases:
        raise ArchSpecError(f"buffer aliases target unknown buffers: {bad_aliases}")


def _validate_bit_field(parent: dict[str, Any], key: str) -> None:
    field = parent.get(key)
    if not isinstance(field, dict):
        raise ArchSpecError(f"{key} bit field must be a map")
    for bound in ("msb", "lsb"):
        value = field.get(bound)
        if not isinstance(value, int) or value < 0:
            raise ArchSpecError(f"{key}.{bound} must be a non-negative integer")
    if field["msb"] < field["lsb"]:
        raise ArchSpecError(f"{key}.msb must be >= {key}.lsb")
    if field["msb"] >= parent["word_bits"]:
        raise ArchSpecError(f"{key}.msb must fit in isa.encoding.word_bits")


def _validate_unique_nibble_map(name: str, values: dict[str, Any]) -> None:
    seen: dict[int, str] = {}
    for key, value in values.items():
        if not isinstance(value, int) or value < 0 or value > 0xF:
            raise ArchSpecError(f"{name}.{key} must be a 4-bit unsigned integer")
        if value in seen:
            raise ArchSpecError(f"{name} duplicates value {value} for {seen[value]} and {key}")
        seen[value] = key


def _strip_jsonc_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        lines.append(_strip_line_comment(line))
    return "\n".join(lines)


def _strip_line_comment(line: str) -> str:
    in_string = False
    escaped = False
    for i in range(len(line) - 1):
        ch = line[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string and line[i : i + 2] == "//":
            return line[:i]
    return line
