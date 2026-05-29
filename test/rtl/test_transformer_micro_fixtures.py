import unittest
from pathlib import Path

from transformer.generate_transformer_micro_fixtures import generate_transformer_micro_fixtures
from transformer.golden import matmul_i8_i32


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

    def test_integer_golden_matmul_uses_full_k_dimension(self):
        a = [[1, -2, 3], [4, 5, -6]]
        b = [[7, 8], [-9, 10], [11, -12]]

        self.assertEqual(matmul_i8_i32(a, b), [[58, -48], [-83, 154]])
