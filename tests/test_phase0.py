import unittest

from npu_phase0.arch import load_arch
from npu_phase0.compiler import compile_graph
from npu_phase0.golden import assert_close, matmul, softmax
from npu_phase0.simulator import FunctionalSimulator


ARCH_PATH = "arch/configs/npu_v0.jsonc"


class Phase0Tests(unittest.TestCase):
    def test_arch_validates(self):
        arch = load_arch(ARCH_PATH)
        self.assertEqual(arch["name"], "npu_v0")

    def test_matmul_golden(self):
        self.assertEqual(
            matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
            [[19, 22], [43, 50]],
        )

    def test_compile_and_simulate_matmul_softmax(self):
        arch = load_arch(ARCH_PATH)
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
        self.assertGreater(result["counters"]["mac_ops"], 0)
        self.assertGreater(result["counters"]["dma_transfers"], 0)


if __name__ == "__main__":
    unittest.main()
