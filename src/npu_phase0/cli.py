"""Command-line entry points for Phase 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

from .arch import load_arch
from .compiler import compile_graph
from .golden import assert_close, matmul, softmax
from .simulator import FunctionalSimulator


def main() -> int:
    parser = argparse.ArgumentParser(prog="npu-phase0")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate-arch")
    validate.add_argument("--arch", required=True)

    run = sub.add_parser("run")
    run.add_argument("--arch", required=True)
    run.add_argument("--graph", required=True)
    run.add_argument("--inputs", required=True)
    run.add_argument("--output", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--arch", required=True)

    args = parser.parse_args()

    if args.cmd == "validate-arch":
        arch = load_arch(args.arch)
        print(f"PASS arch={arch['name']} version={arch['version']}")
        return 0

    if args.cmd == "run":
        arch = load_arch(args.arch)
        graph = _read_json(args.graph)
        inputs = _read_json(args.inputs)
        artifact = compile_graph(graph, arch)
        result = FunctionalSimulator(arch).run(artifact, inputs)
        print(json.dumps(result["dram"][args.output], indent=2))
        print(json.dumps({"counters": result["counters"]}, indent=2))
        return 0

    if args.cmd == "demo":
        _run_demo(Path(args.arch))
        return 0

    raise AssertionError("unreachable")


def _run_demo(arch_path: Path) -> None:
    arch = load_arch(arch_path)
    graph = {
        "tensors": {
            "A": {"shape": [8, 8], "dtype": "int8"},
            "B": {"shape": [8, 8], "dtype": "int8"},
        },
        "ops": [
            {"type": "matmul", "a": "A", "b": "B", "out": "C"},
            {"type": "softmax", "x": "C", "out": "Y"},
        ],
    }
    inputs = {
        "A": [[(i + j) % 5 - 2 for j in range(8)] for i in range(8)],
        "B": [[(i * 2 + j) % 7 - 3 for j in range(8)] for i in range(8)],
    }
    artifact = compile_graph(graph, arch)
    result = FunctionalSimulator(arch).run(artifact, inputs)
    expected = softmax(matmul(inputs["A"], inputs["B"]))
    assert_close(result["dram"]["Y"], expected, arch["verification"]["softmax_abs_tolerance"])
    print("PASS demo matmul -> softmax")
    print(json.dumps({"program": artifact["program"], "counters": result["counters"]}, indent=2))


def _read_json(path: Union[str, Path]):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    raise SystemExit(main())
