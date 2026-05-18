import unittest
import json
from pathlib import Path

from npu_phase0.arch import load_arch
from npu_phase0.compiler import compile_graph
from npu_phase0.digits_classifier import (
    CLASS_COLUMNS,
    PIXELS,
    REAL_CLASSES,
    accumulate_tile_output,
    classifier_graph,
    classifier_inputs,
    classifier_inputs_from_image,
    glyph_rows,
    image_to_glyph_rows,
    lower_classifier_to_rtl_tiles,
    lower_matmul_to_rtl_tiles,
    predict_label,
    reference_logits,
    reference_logits_from_image,
    relu_requantize,
    rtl_tile_graph,
    tiny_mlp_graph,
    tiny_mlp_inputs_from_image,
    tiny_mlp_reference_logits_from_image,
)
from npu_phase0.simulator import MicroOpFunctionalSimulator


ARCH_PATH = "arch/configs/npu_v0.jsonc"
SAMPLES_PATH = Path("test/inputs/digits_classifier_samples.json")


class DigitsClassifierWorkloadTests(unittest.TestCase):
    def test_classifier_shape_matches_phase0_tile_rules(self):
        graph = classifier_graph()
        self.assertEqual(graph["tensors"]["A"]["shape"], [8, PIXELS])
        self.assertEqual(graph["tensors"]["W"]["shape"], [PIXELS, CLASS_COLUMNS])
        self.assertEqual(REAL_CLASSES, 10)

    def test_golden_classifier_predicts_each_digit(self):
        for label in range(REAL_CLASSES):
            with self.subTest(label=label):
                self.assertEqual(predict_label(reference_logits(label)), label)

    def test_checked_in_samples_match_golden_outputs(self):
        samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]
        self.assertEqual(len(samples), REAL_CLASSES)
        for sample in samples:
            label = sample["label"]
            with self.subTest(label=label):
                logits = reference_logits(label)
                self.assertEqual(sample["image"], glyph_rows(label))
                self.assertEqual(sample["expected_logits_0_to_9"], logits[0][:REAL_CLASSES])
                self.assertEqual(sample["expected_prediction"], predict_label(logits))

    def test_real_pgm_images_match_checked_in_samples(self):
        samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]
        for sample in samples:
            label = sample["label"]
            image_path = Path(sample["image_path"])
            with self.subTest(label=label):
                self.assertEqual(image_to_glyph_rows(image_path), sample["image"])
                logits = reference_logits_from_image(image_path)
                self.assertEqual(logits[0][:REAL_CLASSES], sample["expected_logits_0_to_9"])
                self.assertEqual(predict_label(logits), sample["expected_prediction"])

    def test_compiler_simulator_classifier_predicts_each_digit(self):
        arch = load_arch(ARCH_PATH)
        graph = classifier_graph()
        artifact = compile_graph(graph, arch)
        simulator = MicroOpFunctionalSimulator(arch)

        for label in range(REAL_CLASSES):
            with self.subTest(label=label):
                result = simulator.run(artifact, classifier_inputs(label))
                logits = result["dram"]["Logits"]
                self.assertEqual(logits, reference_logits(label))
                self.assertEqual(predict_label(logits), label)

    def test_tiled_lowering_uses_only_rtl_compatible_matmul_jobs(self):
        graph = rtl_tile_graph()
        self.assertEqual(graph["tensors"]["A"]["shape"], [8, 8])
        self.assertEqual(graph["tensors"]["B"]["shape"], [8, 8])
        self.assertEqual(graph["ops"], [{"type": "matmul", "a": "A", "b": "B", "out": "C"}])

        jobs = lower_classifier_to_rtl_tiles(classifier_inputs(2))
        self.assertEqual(len(jobs), 16)
        self.assertEqual({job["n_offset"] for job in jobs}, {0, 8})
        self.assertEqual({job["k_offset"] for job in jobs}, set(range(0, 64, 8)))
        for job in jobs:
            self.assertEqual(job["graph"], graph)
            self.assertEqual(len(job["inputs"]["A"]), 8)
            self.assertEqual(len(job["inputs"]["A"][0]), 8)
            self.assertEqual(len(job["inputs"]["B"]), 8)
            self.assertEqual(len(job["inputs"]["B"][0]), 8)

    def test_tiled_compiler_simulator_path_matches_full_classifier(self):
        arch = load_arch(ARCH_PATH)
        tile_artifact = compile_graph(rtl_tile_graph(), arch)
        simulator = MicroOpFunctionalSimulator(arch)
        samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]

        for sample in samples:
            label = sample["label"]
            image_path = Path(sample["image_path"])
            with self.subTest(label=label):
                logits = [[0 for _ in range(CLASS_COLUMNS)] for _ in range(8)]
                for job in lower_classifier_to_rtl_tiles(classifier_inputs_from_image(image_path)):
                    result = simulator.run(tile_artifact, job["inputs"])
                    accumulate_tile_output(logits, result["dram"]["C"], job["n_offset"])
                self.assertEqual(logits, reference_logits_from_image(image_path))
                self.assertEqual(logits[0][:REAL_CLASSES], sample["expected_logits_0_to_9"])
                self.assertEqual(predict_label(logits), label)

    def test_tiny_mlp_graph_exposes_cpu_and_npu_placement(self):
        graph = tiny_mlp_graph()
        self.assertEqual([op["type"] for op in graph["ops"]], ["matmul", "relu_requantize", "matmul", "argmax"])
        self.assertEqual([op["placement"] for op in graph["ops"]], ["npu", "cpu", "npu", "cpu"])
        self.assertEqual(graph["tensors"]["W1"]["shape"], [64, 16])
        self.assertEqual(graph["tensors"]["W2"]["shape"], [16, 16])

    def test_tiny_mlp_tiled_path_predicts_each_digit(self):
        arch = load_arch(ARCH_PATH)
        tile_artifact = compile_graph(rtl_tile_graph(), arch)
        simulator = MicroOpFunctionalSimulator(arch)
        samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]

        for sample in samples:
            label = sample["label"]
            image_path = Path(sample["image_path"])
            with self.subTest(label=label):
                inputs = tiny_mlp_inputs_from_image(image_path)
                hidden = _run_tiled_matmul(simulator, tile_artifact, inputs["A"], inputs["W1"])
                hidden_int8 = relu_requantize(hidden)
                self.assertTrue(all(0 <= value <= 127 for row in hidden_int8 for value in row))
                logits = _run_tiled_matmul(simulator, tile_artifact, hidden_int8, inputs["W2"])
                self.assertEqual(logits, tiny_mlp_reference_logits_from_image(image_path))
                self.assertEqual(predict_label(logits), label)


def _run_tiled_matmul(simulator, tile_artifact, a, b):
    out = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for job in lower_matmul_to_rtl_tiles(a, b):
        result = simulator.run(tile_artifact, job["inputs"])
        accumulate_tile_output(out, result["dram"]["C"], job["n_offset"])
    return out


if __name__ == "__main__":
    unittest.main()
