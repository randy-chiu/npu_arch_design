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
