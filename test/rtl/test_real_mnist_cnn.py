import json
import unittest
from pathlib import Path

from npu_phase0.real_mnist_cnn import (
    MODEL_README_PATH,
    MODEL_WEIGHTS_PATH,
    TEST_IMAGES_PATH,
    TEST_LABELS_PATH,
    fc2_npu_inputs_from_activation,
    fc2_quantized_logits_from_int32,
    forward_intermediates,
    load_mnist_images,
    load_mnist_labels,
    load_safetensors_f32,
    lower_fc2_to_rtl_tiles,
    numpy_available,
    predict,
    real_mnist_cnn_graph,
)
from npu_phase0.arch import load_arch
from npu_phase0.compiler import compile_graph
from npu_phase0.digits_classifier import CLASS_COLUMNS, accumulate_tile_output, predict_label, rtl_tile_graph
from npu_phase0.simulator import MicroOpFunctionalSimulator


GRAPH_PATH = Path("test/graphs/real_mnist_cnn.json")
EXTERNAL_AVAILABLE = (
    numpy_available()
    and MODEL_WEIGHTS_PATH.exists()
    and MODEL_README_PATH.exists()
    and TEST_IMAGES_PATH.exists()
    and TEST_LABELS_PATH.exists()
)


@unittest.skipUnless(EXTERNAL_AVAILABLE, "real MNIST CNN external fixtures are not available")
class RealMnistCnnGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = load_safetensors_f32()
        cls.images = load_mnist_images()
        cls.labels = load_mnist_labels()

    def test_graph_fixture_matches_tool_graph(self):
        self.assertEqual(json.loads(GRAPH_PATH.read_text(encoding="utf-8")), real_mnist_cnn_graph())

    def test_open_source_weight_shapes_match_expected_cnn(self):
        self.assertEqual(self.weights["conv1.weight"].shape, (32, 1, 3, 3))
        self.assertEqual(self.weights["conv1.bias"].shape, (32,))
        self.assertEqual(self.weights["conv2.weight"].shape, (64, 32, 3, 3))
        self.assertEqual(self.weights["conv2.bias"].shape, (64,))
        self.assertEqual(self.weights["fc1.weight"].shape, (128, 9216))
        self.assertEqual(self.weights["fc1.bias"].shape, (128,))
        self.assertEqual(self.weights["fc2.weight"].shape, (10, 128))
        self.assertEqual(self.weights["fc2.bias"].shape, (10,))

    def test_first_ten_mnist_test_images_predict_expected_labels(self):
        expected = [7, 2, 1, 0, 4, 1, 4, 9, 5, 9]
        actual = [predict(self.images[idx], self.weights) for idx in range(10)]
        self.assertEqual(actual, expected)
        self.assertEqual(actual, [int(label) for label in self.labels[:10]])

    def test_first_hundred_mnist_test_images_meet_accuracy_smoke_threshold(self):
        correct = 0
        for idx in range(100):
            correct += predict(self.images[idx], self.weights) == int(self.labels[idx])
        self.assertGreaterEqual(correct, 98)

    def test_fc2_mapping_keeps_original_model_logic_and_uses_rtl_tiles(self):
        arch = load_arch("arch/configs/npu_v0.jsonc")
        tile_artifact = compile_graph(rtl_tile_graph(), arch)
        simulator = MicroOpFunctionalSimulator(arch)

        for idx in range(10):
            with self.subTest(idx=idx, label=int(self.labels[idx])):
                intermediates = forward_intermediates(self.images[idx], self.weights)
                npu_inputs = fc2_npu_inputs_from_activation(intermediates["fc1_relu"], self.weights)
                jobs = lower_fc2_to_rtl_tiles(npu_inputs)
                self.assertEqual(len(jobs), 32)
                self.assertEqual({job["n_offset"] for job in jobs}, {0, 8})
                self.assertEqual({job["k_offset"] for job in jobs}, set(range(0, 128, 8)))

                acc = [[0 for _ in range(CLASS_COLUMNS)] for _ in range(8)]
                for job in jobs:
                    result = simulator.run(tile_artifact, job["inputs"])
                    accumulate_tile_output(acc, result["dram"]["C"], job["n_offset"])

                quantized_logits = fc2_quantized_logits_from_int32(acc, npu_inputs)
                self.assertEqual(predict_label([quantized_logits]), int(self.labels[idx]))
                self.assertEqual(predict_label([quantized_logits]), predict_label([intermediates["logits"].tolist()]))


if __name__ == "__main__":
    unittest.main()
