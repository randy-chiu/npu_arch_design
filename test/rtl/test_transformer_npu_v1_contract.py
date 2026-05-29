import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path


def read_jsonc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    return json.loads(text)


class TransformerNpuV1ContractTests(unittest.TestCase):
    def test_transformer_v1_config_names_required_primitives_and_metrics(self):
        config = read_jsonc(Path("arch/configs/npu_transformer_v1.jsonc"))

        self.assertEqual(config["name"], "npu_transformer_v1")
        self.assertIn("GEMV", config["primitive_uops"])
        self.assertIn("REDUCE_SUMSQ", config["primitive_uops"])
        self.assertIn("SFU_RSQRT", config["primitive_uops"])
        self.assertIn("matrix_utilization", config["perf_metrics"]["required"])
        self.assertEqual(config["modules"]["accumulator_file"]["banks"], 2)

    def test_required_transformer_v1_specs_exist(self):
        for path in (
            "docs/design/transformer/transformer_npu_v1.md",
            "docs/design/transformer/next_steps.md",
            "docs/design/transformer/vector_engine_v1.md",
            "docs/design/transformer/reduction_engine_v1.md",
            "docs/design/transformer/sfu_v1.md",
            "arch/specs/transformer/v1/transformer_npu_v1.md",
            "arch/specs/transformer/v1/transformer_numerical_v1.md",
            "arch/specs/transformer/v1/csr_map_v1.md",
            "arch/specs/transformer/v1/descriptor_v1.md",
            "arch/specs/transformer/v1/uop_isa_v1.md",
        ):
            self.assertTrue(Path(path).is_file(), path)

    @unittest.skipUnless(shutil.which("iverilog"), "iverilog not installed")
    def test_accumulator_file_elaborates(self):
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-t",
                "null",
                "-s",
                "accumulator_file",
                "hw/npu_core/rtl/matrix/accumulator_file.sv",
            ],
            check=True,
        )

    @unittest.skipUnless(shutil.which("iverilog"), "iverilog not installed")
    def test_primitive_engines_simulation_passes(self):
        subprocess.run(["make", "primitive-engines-sim"], check=True)
