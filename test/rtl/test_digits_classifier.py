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
    flatten_quantized_image,
    glyph_rows,
    image_to_glyph_rows,
    lower_classifier_to_rtl_tiles,
    predict_label,
    reference_logits,
    reference_logits_from_image,
    rtl_tile_graph,
)
from npu_phase0.simulator import MicroOpFunctionalSimulator


ARCH_PATH = "arch/configs/npu_v0.jsonc"
SAMPLES_PATH = Path("test/inputs/digits_classifier_samples.json")
REALISTIC_DIGITS_DIR = Path("test/assets/digits_realistic")


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

    def test_realistic_grayscale_pgm_images_exercise_quantized_input(self):
        for label in range(REAL_CLASSES):
            image_path = REALISTIC_DIGITS_DIR / f"digit_{label}_gray.pgm"
            with self.subTest(label=label):
                quantized = flatten_quantized_image(image_path)
                self.assertGreater(len(set(quantized)), 2)
                self.assertTrue(all(-1 <= value <= 3 for value in quantized))
                self.assertEqual(predict_label(reference_logits_from_image(image_path)), label)

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


if __name__ == "__main__":
    unittest.main()
