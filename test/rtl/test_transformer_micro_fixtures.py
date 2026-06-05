import unittest
from pathlib import Path

from transformer.generate_transformer_micro_fixtures import generate_transformer_micro_fixtures
from transformer.golden import matmul_i8_i32
from transformer.micro_golden import (
    ATTENTION_NUMERICAL_CONTRACT_V1,
    attention_head_fixed_spec,
    attention_pv_q15_i8_i32,
    attention_qk_scores_i8_i32,
    attention_softmax_fixed_spec_q15,
    kv_cache_bytes,
    rmsnorm_primitive_sequence,
    rmsnorm_row_reference,
    scale_scores_fixed_multiplier,
    softmax_row_fixed_q15,
    softmax_row_primitive_lut_q15,
)


class TransformerMicroFixtureTests(unittest.TestCase):
    def test_transformer_micro_spec_generates_executable_and_model_only_workloads(self):
        generated = generate_transformer_micro_fixtures(
            Path("workloads/manifests/transformer/transformer_micro_v0.jsonc")
        )

        executable = generated["executable_workloads"]
        model_only = generated["model_only_workloads"]
        self.assertEqual(
            [plan["attention_group"] for plan in generated["attention_plans"]],
            ["attention_prefill_s8_d8"],
        )
        self.assertEqual(
            [stage["stage_id"] for stage in generated["attention_plans"][0]["stages"]],
            ["qk", "scale_mask", "softmax", "pv"],
        )
        self.assertEqual(
            [item["name"] for item in executable],
            [
                "transformer_prefill_gemm_tiny",
                "transformer_attention_qk_s8_d8",
                "transformer_attention_scale_mask_s8_d8",
                "transformer_attention_softmax_s8",
                "transformer_attention_pv_s8_d8",
                "transformer_decode_skinny_gemm_m8_compat",
            ],
        )
        self.assertEqual(executable[0]["k_chunks"], 2)
        self.assertEqual(executable[1]["k_chunks"], 1)
        self.assertEqual(executable[1]["metadata"]["attention_stage"], "qk")
        self.assertEqual(executable[1]["metadata"]["attention_plan_runtime_job"]["descriptor_op"], "matmul_k_stream")
        self.assertEqual(executable[2]["op"], "attention_scale_mask_v1")
        self.assertEqual(executable[2]["metadata"]["attention_stage"], "scale_mask")
        self.assertEqual(executable[2]["metadata"]["attention"]["scale_multiplier"], 11585)
        self.assertEqual(executable[2]["input_scores"], executable[1]["expected_c"])
        self.assertEqual(executable[3]["op"], "attention_softmax_v1")
        self.assertEqual(executable[3]["x"], executable[2]["expected_scores"])
        self.assertEqual(len(executable[3]["expected_y"]), 64)
        self.assertEqual(executable[4]["a_stream"][0], executable[3]["expected_y"])
        self.assertEqual(executable[3]["metadata"]["attention_stage"], "softmax")
        self.assertEqual(executable[3]["metadata"]["stage_provenance"], "measured_current_softmax_path")
        self.assertEqual(
            executable[3]["metadata"]["attention_plan_stage"]["inputs"],
            ["score_softmax_in"],
        )
        self.assertEqual(
            executable[3]["metadata"]["softmax"]["implementation"],
            "npu_v1_vector_reduction_sfu_sequence",
        )
        self.assertEqual(executable[4]["k_chunks"], 1)
        self.assertEqual(executable[4]["op"], "matmul_u16s8_q15")
        self.assertEqual(executable[4]["metadata"]["attention_stage"], "pv")
        self.assertEqual(
            executable[4]["metadata"]["attention_plan_runtime_job"]["descriptor_op"],
            "matmul_u16s8_q15",
        )
        self.assertEqual(
            executable[4]["metadata"]["attention"]["probability_policy"],
            "q0.15_u16",
        )
        self.assertEqual(executable[4]["a_bits"], 16)
        self.assertEqual(executable[5]["k_chunks"], 1)
        self.assertTrue(any(item["name"] == "transformer_kv_cache_traffic_tiny" for item in model_only))
        self.assertTrue(any(item["name"] == "transformer_softmax_row" for item in model_only))
        self.assertTrue(any(item["name"] == "transformer_attention_prefill_s8_d8" for item in model_only))
        parent = next(item for item in model_only if item["name"] == "transformer_attention_prefill_s8_d8")
        self.assertEqual(parent["metadata"]["attention_plan"]["group_state"], "software_group_measured_stages")
        self.assertEqual(parent["metadata"]["attention_plan"]["runtime_jobs"], ["qk", "scale_mask", "softmax", "pv"])
        self.assertEqual(executable[0]["metadata"]["shape_class"], "skinny_gemm")
        self.assertEqual(executable[5]["metadata"]["workload_family"], "transformer_decode")

    def test_integer_golden_matmul_uses_full_k_dimension(self):
        a = [[1, -2, 3], [4, 5, -6]]
        b = [[7, 8], [-9, 10], [11, -12]]

        self.assertEqual(matmul_i8_i32(a, b), [[58, -48], [-83, 154]])

    def test_softmax_fixed_q15_is_stable_and_normalized(self):
        result = softmax_row_fixed_q15([32, 0, -32, -512])

        self.assertEqual(len(result), 4)
        self.assertGreater(result[0], result[1])
        self.assertLess(result[-1], 10)
        self.assertLessEqual(abs(sum(result) - 32767), 1)

    def test_softmax_primitive_lut_sequence_matches_current_sfu_contract(self):
        result = softmax_row_primitive_lut_q15([32, 0, -32, -512])

        self.assertEqual(result["max"], 32)
        self.assertEqual(result["clamped"], [0, -32, -64, -256])
        self.assertEqual(result["exp_q15"], [32767, 12055, 4435, 11])
        self.assertEqual(result["output_q15"], [21759, 8005, 2945, 7])

    def test_rmsnorm_reference_preserves_shape(self):
        result = rmsnorm_row_reference([1, -2, 3, -4], [128, 128, 128, 128])

        self.assertEqual(len(result), 4)
        self.assertGreater(result[2], 0)
        self.assertLess(result[3], 0)

    def test_rmsnorm_primitive_sequence_matches_current_sfu_contract(self):
        result = rmsnorm_primitive_sequence([8, 8, 8, 8])

        self.assertEqual(result["sumsq"], 256)
        self.assertEqual(result["rsqrt_q24"], 1048576)
        self.assertEqual(result["scaled"], [8, 8, 8, 8])

    def test_kv_cache_byte_accounting(self):
        self.assertEqual(
            kv_cache_bytes(seq_len=32, heads=4, head_dim=16),
            {
                "kv_cache_read_bytes": 4096,
                "kv_cache_write_bytes": 128,
                "bytes_per_token": 4224,
            },
        )

    def test_attention_qk_scores_and_scale_are_deterministic(self):
        q = [[1, -2, 3], [4, 0, -1]]
        k = [[2, 1, -3], [-1, 5, 2]]

        scores = attention_qk_scores_i8_i32(q, k)
        scaled = scale_scores_fixed_multiplier(scores, head_dim=3, shift=15)

        self.assertEqual(scores, [[-9, -5], [11, -6]])
        self.assertEqual(scaled["policy"], "fixed_multiplier_shift")
        self.assertEqual(scaled["multiplier"], 18919)
        self.assertEqual(scaled["scaled"], [[-5, -3], [6, -3]])

    def test_attention_softmax_fixed_spec_exposes_intermediates(self):
        result = attention_softmax_fixed_spec_q15([[32, 0, -32, -64]])
        row = result["rows"][0]

        self.assertEqual(result["contract"], ATTENTION_NUMERICAL_CONTRACT_V1)
        self.assertEqual(row["max"], 32)
        self.assertEqual(row["clamped"], [0, -32, -64, -96])
        self.assertEqual(row["exp_q15"], [32767, 12054, 4435, 1631])
        self.assertLessEqual(abs(sum(row["output_q15"]) - 32767), 128)

    def test_attention_pv_q15_i8_matches_weighted_sum(self):
        prob = [[24575, 8192]]
        value = [[20, 4], [-12, 8]]

        self.assertEqual(attention_pv_q15_i8_i32(prob, value), [[12, 5]])

    def test_attention_head_fixed_spec_carries_stage_outputs(self):
        q = [[1, 0], [0, 1]]
        k = [[1, 0], [0, 1]]
        v = [[10, -2], [-4, 8]]

        result = attention_head_fixed_spec(q, k, v)

        self.assertEqual(result["contract"], ATTENTION_NUMERICAL_CONTRACT_V1)
        self.assertEqual(result["scores"], [[1, 0], [0, 1]])
        self.assertIn("softmax", result)
        self.assertEqual(len(result["output"]), 2)
