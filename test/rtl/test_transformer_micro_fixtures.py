import unittest
from pathlib import Path

from transformer.generate_transformer_micro_fixtures import generate_transformer_micro_fixtures
from transformer.golden import matmul_i8_i32
from transformer.micro_golden import (
    kv_cache_bytes,
    rmsnorm_primitive_sequence,
    rmsnorm_row_reference,
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
            [item["name"] for item in executable],
            [
                "transformer_prefill_gemm_tiny",
                "transformer_decode_skinny_gemm_m8_compat",
            ],
        )
        self.assertEqual(executable[0]["k_chunks"], 2)
        self.assertEqual(executable[1]["k_chunks"], 1)
        self.assertTrue(any(item["name"] == "transformer_kv_cache_traffic_tiny" for item in model_only))
        self.assertTrue(any(item["name"] == "transformer_softmax_row" for item in model_only))
        self.assertEqual(executable[0]["metadata"]["shape_class"], "skinny_gemm")
        self.assertEqual(executable[1]["metadata"]["workload_family"], "transformer_decode")

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
