import unittest
import json
import shutil
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path

from npu_phase0.arch import load_arch
from npu_phase0.compiler import compile_graph
from npu_phase0.golden import assert_close, matmul, softmax
from npu_phase0.rtl_fixture import encode_program, encode_uop, generate_default_fixtures, softmax_q0_8
from npu_phase0.simulator import MicroOpFunctionalSimulator


ARCH_PATH = "arch/configs/npu_v0.jsonc"
GRAPH_PATH = Path("tests/graphs/matmul_softmax.json")
INPUTS_PATH = Path("tests/inputs_matmul_softmax.json")


class ArchitectureAndGoldenTests(unittest.TestCase):
    def test_arch_validates(self):
        arch = load_arch(ARCH_PATH)
        self.assertEqual(arch["name"], "npu_v0")

    def test_matmul_golden(self):
        self.assertEqual(
            matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
            [[19, 22], [43, 50]],
        )


class CompilerMicroOpFunctionalTests(unittest.TestCase):
    def test_compiler_micro_ops_match_graph_golden(self):
        arch = load_arch(ARCH_PATH)
        graph = _read_json(GRAPH_PATH)
        inputs = _read_json(INPUTS_PATH)
        artifact = compile_graph(graph, arch)
        result = MicroOpFunctionalSimulator(arch).run(artifact, inputs)
        expected_tensors = _run_golden_graph(graph, inputs)
        output_tensor = graph["ops"][-1]["out"]

        assert_close(
            result["dram"][output_tensor],
            expected_tensors[output_tensor],
            arch["verification"]["softmax_abs_tolerance"],
        )
        self.assertGreater(result["counters"]["mac_ops"], 0)
        self.assertGreater(result["counters"]["dma_transfers"], 0)


class RTLFunctionalTests(unittest.TestCase):
    def test_rtl_fixture_generator_emits_expected_files(self):
        arch = load_arch(ARCH_PATH)
        self.assertEqual(encode_uop(arch, "LOAD", 0, 1), 0x10100000)
        self.assertEqual(
            softmax_q0_8([0] * arch["rtl"]["softmax_vector_len"]),
            [31] * arch["rtl"]["softmax_vector_len"],
        )
        graph = _read_json(GRAPH_PATH)
        artifact = compile_graph(_single_matmul_graph(graph), arch)
        self.assertEqual(
            encode_program(artifact["program"], arch)[:5],
            [0x10000000, 0x11100000, 0x30000000, 0x22200000, 0xF0000000],
        )
        with TemporaryDirectory() as tmp:
            generate_default_fixtures(Path(tmp), arch)
            expected = {
                "matmul_a.hex",
                "matmul_b.hex",
                "matmul_expected_c.hex",
                "matmul_program.hex",
                "softmax_x.hex",
                "softmax_expected_y.hex",
                "softmax_program.hex",
                "npu_v0_spec.svh",
                "npu_v0_tb_params.svh",
            }
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, expected)

    @unittest.skipUnless(
        shutil.which("iverilog") and shutil.which("vvp"),
        "iverilog/vvp not installed",
    )
    def test_rtl_simulation_matches_generated_fixtures(self):
        result = subprocess.run(
            ["make", "rtl-sim"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("PASS npu_v0 RTL generated-fixture tests", result.stdout)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _single_matmul_graph(graph):
    op = next(op for op in graph["ops"] if op["type"] == "matmul")
    return {
        "tensors": {
            op["a"]: graph["tensors"][op["a"]],
            op["b"]: graph["tensors"][op["b"]],
        },
        "ops": [op],
    }


def _run_golden_graph(graph, inputs):
    tensors = dict(inputs)
    for op in graph["ops"]:
        op_type = op["type"]
        if op_type == "matmul":
            tensors[op["out"]] = matmul(tensors[op["a"]], tensors[op["b"]])
        elif op_type == "softmax":
            tensors[op["out"]] = softmax(tensors[op["x"]])
        else:
            raise ValueError(f"unsupported golden op type: {op_type}")
    return tensors


if __name__ == "__main__":
    unittest.main()
