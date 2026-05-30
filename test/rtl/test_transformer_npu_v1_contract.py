import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from transformer.emit_transformer_config import emit
from transformer.micro_golden import softmax_rtl_model_row_q15, sfu_exp_rtl_model_q15


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
        self.assertEqual(config["modules"]["vector_engine"]["data_width"], 32)
        self.assertEqual(config["modules"]["reduction_engine"]["result_width"], 64)
        self.assertEqual(config["modules"]["sfu"]["exp_lut_entries"], 257)
        self.assertEqual(config["primitive_op_encodings"]["vector"]["VEC_REQUANT"], 4)
        self.assertEqual(config["primitive_op_encodings"]["reduction"]["REDUCE_SUMSQ"], 2)
        self.assertEqual(config["primitive_op_encodings"]["sfu"]["SFU_RSQRT"], 2)

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
            "docs/design/transformer/vector_engine_v1.md",
            "docs/design/transformer/reduction_engine_v1.md",
            "docs/design/transformer/sfu_v1.md",
            "docs/design/transformer/requant_v1.md",
            "docs/design/transformer/sfu_exp_lut257_design.md",
            "docs/design/transformer/primitive_valid_ready_v1.md",
            "docs/design/transformer/requant_v2_design.md",
        ):
            self.assertTrue(Path(path).is_file(), path)

    def test_transformer_config_generator_emits_required_constants(self):
        out_dir = Path("build/test_generated")
        outputs = emit(Path("arch/configs/npu_transformer_v1.jsonc"), out_dir)
        sv_text = outputs["sv"].read_text(encoding="utf-8")
        h_text = outputs["h"].read_text(encoding="utf-8")
        py_text = outputs["py"].read_text(encoding="utf-8")

        for name in (
            "CFG_VECTOR_LANES",
            "CFG_VECTOR_DATA_WIDTH",
            "CFG_REDUCTION_LANES",
            "CFG_REDUCTION_MAX_LEN",
            "CFG_REDUCTION_DATA_WIDTH",
            "CFG_REDUCTION_RESULT_WIDTH",
            "CFG_SFU_DATA_WIDTH",
            "CFG_SFU_EXP_INPUT_SCALE",
            "CFG_SFU_EXP_LUT_ENTRIES",
            "CFG_SFU_EXP_OUTPUT_Q",
            "CFG_SFU_BRINGUP_EXP_SEG_0",
            "CFG_SFU_RECIP_OUTPUT_Q",
            "CFG_SFU_RSQRT_OUTPUT_Q",
            "CFG_VEC_REQUANT",
            "CFG_REDUCE_SUMSQ",
            "CFG_SFU_RSQRT",
        ):
            self.assertIn(name, sv_text)
            self.assertIn(name, h_text)
            self.assertIn(name, py_text)

        self.assertIn("localparam int CFG_SFU_EXP_LUT_ENTRIES = 257;", sv_text)

    def test_micro_golden_rtl_model_matches_current_sfu_lut_points(self):
        self.assertEqual(sfu_exp_rtl_model_q15(0), 32767)
        self.assertEqual(sfu_exp_rtl_model_q15(-32), 12055)
        self.assertEqual(sfu_exp_rtl_model_q15(-64), 4435)
        self.assertEqual(sfu_exp_rtl_model_q15(-256), 11)

        modeled = softmax_rtl_model_row_q15([32, 0, -32, -512])
        self.assertEqual(modeled["exp_q15"], [32767, 12055, 4435, 11])
        self.assertEqual(modeled["sum"], 49268)
        self.assertEqual(modeled["recip_q24"], 340)
        self.assertEqual(modeled["output_q15"], [21759, 8005, 2945, 7])

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
