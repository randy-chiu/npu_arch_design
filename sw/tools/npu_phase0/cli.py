"""Command-line entry points for Phase 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

from .arch import load_arch
from .compiler import compile_graph
from .digits_classifier import (
    classifier_graph,
    classifier_inputs,
    classifier_inputs_from_image,
    glyph_rows,
    image_to_glyph_rows,
    predict_label,
)
from .golden import assert_close, matmul, softmax
from .rtl_fixture import generate_default_fixtures
from .simulator import MicroOpFunctionalSimulator


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

    digits_demo = sub.add_parser("digits-demo")
    digits_demo.add_argument("--arch", required=True)
    digits_demo.add_argument("--label", type=int, default=2)
    digits_demo.add_argument("--image")

    fixtures = sub.add_parser("emit-rtl-fixtures")
    fixtures.add_argument("--arch", required=True)
    fixtures.add_argument("--out-dir", required=True)

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
        result = MicroOpFunctionalSimulator(arch).run(artifact, inputs)
        print(json.dumps(result["dram"][args.output], indent=2))
        print(json.dumps({"counters": result["counters"]}, indent=2))
        return 0

    if args.cmd == "demo":
        _run_demo(Path(args.arch))
        return 0

    if args.cmd == "digits-demo":
        _run_digits_demo(Path(args.arch), args.label, Path(args.image) if args.image else None)
        return 0

    if args.cmd == "emit-rtl-fixtures":
        arch = load_arch(args.arch)
        generate_default_fixtures(Path(args.out_dir), arch)
        print(f"PASS rtl fixtures written to {args.out_dir}")
        return 0

    raise AssertionError("unreachable")


def _run_demo(arch_path: Path) -> None:
    arch = load_arch(arch_path)
    tile_m, tile_n, tile_k = arch["rtl"]["matmul_tile"]
    graph = {
        "tensors": {
            "A": {"shape": [tile_m, tile_k], "dtype": "int8"},
            "B": {"shape": [tile_k, tile_n], "dtype": "int8"},
        },
        "ops": [
            {"type": "matmul", "a": "A", "b": "B", "out": "C"},
            {"type": "softmax", "x": "C", "out": "Y"},
        ],
    }
    inputs = {
        "A": [[(i + j) % 5 - 2 for j in range(tile_k)] for i in range(tile_m)],
        "B": [[(i * 2 + j) % 7 - 3 for j in range(tile_n)] for i in range(tile_k)],
    }
    artifact = compile_graph(graph, arch)
    result = MicroOpFunctionalSimulator(arch).run(artifact, inputs)
    expected = softmax(matmul(inputs["A"], inputs["B"]))
    assert_close(result["dram"]["Y"], expected, arch["verification"]["softmax_abs_tolerance"])
    print("PASS demo matmul -> softmax")
    print(json.dumps({"program": artifact["program"], "counters": result["counters"]}, indent=2))


def _run_digits_demo(arch_path: Path, label: int, image_path: Path | None) -> None:
    arch = load_arch(arch_path)
    graph = classifier_graph()
    artifact = compile_graph(graph, arch)
    if image_path is None:
        inputs = classifier_inputs(label)
        input_glyph = glyph_rows(label)
    else:
        inputs = classifier_inputs_from_image(image_path)
        input_glyph = image_to_glyph_rows(image_path)
    result = MicroOpFunctionalSimulator(arch).run(artifact, inputs)
    logits_10 = result["dram"]["Logits"][0][:10]
    predicted = predict_label(result["dram"]["Logits"])
    if predicted != label:
        raise AssertionError(f"digits classifier predicted {predicted}, expected {label}")
    print(f"PASS digits classifier label={label} predicted={predicted}")
    print(
        json.dumps(
            {
                "input_image": str(image_path) if image_path else None,
                "input_glyph": input_glyph,
                "class_logits_0_to_9": logits_10,
                "predicted_label": predicted,
                "program": artifact["program"],
                "counters": result["counters"],
            },
            indent=2,
        )
    )


def _read_json(path: Union[str, Path]):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    raise SystemExit(main())
