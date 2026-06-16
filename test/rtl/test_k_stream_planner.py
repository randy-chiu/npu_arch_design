import unittest

from npu_compiler.k_stream import plan_matmul_k_stream, plan_tiled_matmul
from npu_phase0.golden import matmul


class KStreamPlannerTests(unittest.TestCase):
    def test_plan_accumulates_all_k_chunks_for_one_n_tile(self):
        a = [[row + col for col in range(16)] for row in range(8)]
        b = [[row * 3 + col for col in range(8)] for row in range(16)]

        plan = plan_matmul_k_stream(a, b)

        self.assertEqual(plan["m"], 8)
        self.assertEqual(plan["n"], 8)
        self.assertEqual(plan["k_step"], 8)
        self.assertEqual(plan["k_chunks"], 2)
        self.assertEqual(plan["k_offsets"], [0, 8])
        self.assertEqual(plan["input0_words"], 64)
        self.assertEqual(plan["input1_words"], 64)
        self.assertEqual(plan["output_words"], 64)
        self.assertEqual(plan["expected_c"], matmul(a, b))

    def test_plan_can_select_first_nonzero_chunks_for_smoke(self):
        a = [[0 for _ in range(24)] for _ in range(8)]
        b = [[1 for _ in range(8)] for _ in range(24)]
        for row in range(8):
            a[row][8 + row] = row + 1
            a[row][16 + row] = row + 2

        plan = plan_matmul_k_stream(a, b, max_chunks=2, require_nonzero=True)

        self.assertEqual(plan["k_chunks"], 2)
        self.assertEqual(plan["k_offsets"], [8, 16])
        self.assertEqual(plan["expected_c"], matmul([row[8:24] for row in a], b[8:24]))

    def test_plan_tiled_matmul_lowers_m_n_k_and_boundary_tiles(self):
        a = [[row + col for col in range(16)] for row in range(9)]
        b = [[row * 3 + col for col in range(10)] for row in range(16)]

        plan = plan_tiled_matmul(a, b)

        self.assertEqual(plan["logical_shape"], {"m": 9, "n": 10, "k": 16})
        self.assertEqual(plan["output_tile_count"], 4)
        self.assertEqual(plan["physical_tile_invocations"], 8)
        self.assertEqual(plan["theoretical_matrix_cycles"], 64)
        self.assertEqual(plan["output_tile_jobs"][-1]["valid_m"], 1)
        self.assertEqual(plan["output_tile_jobs"][-1]["valid_n"], 2)
        self.assertEqual(plan["expected_c"], matmul(a, b))


if __name__ == "__main__":
    unittest.main()
