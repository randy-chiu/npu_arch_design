import unittest

from npu_compiler.block import (
    B0_SHAPE,
    build_b0_block_plan,
    build_b0_matrix_subgraph_workload,
    build_b1_two_block_plan,
    lower_residual_add_vector_tiles,
)


class TransformerBlockPlanTests(unittest.TestCase):
    def test_b0_preserves_tinyllama_derived_topology_and_marks_rtl_gaps(self):
        plan = build_b0_block_plan()

        self.assertEqual(plan["shape"], B0_SHAPE)
        self.assertEqual(plan["execution_state"], "planned_not_executable")
        self.assertEqual(len(plan["stages"]), 18)
        self.assertEqual(plan["stages"][0]["stage_id"], "rmsnorm_attn")
        self.assertEqual(plan["stages"][-1]["stage_id"], "residual_ffn")
        self.assertEqual(len(plan["golden"]["attention_heads"]), 2)
        self.assertEqual(len(plan["golden"]["block_output"]), 8)
        self.assertEqual(len(plan["golden"]["block_output"][0]), 16)
        self.assertTrue(all(stage["execution_state"] == "planned_not_executable" for stage in plan["stages"]))

    def test_b0_matrix_stages_are_compiler_tiled(self):
        plan = build_b0_block_plan()
        stages = {stage["stage_id"]: stage for stage in plan["stages"]}

        self.assertEqual(stages["q_proj"]["matrix_plan"]["output_tile_count"], 2)
        self.assertEqual(stages["q_proj"]["matrix_plan"]["physical_tile_invocations"], 4)
        self.assertEqual(stages["gate_proj"]["matrix_plan"]["output_tile_count"], 4)
        self.assertEqual(stages["gate_proj"]["matrix_plan"]["physical_tile_invocations"], 8)
        self.assertEqual(stages["down_proj"]["matrix_plan"]["output_tile_count"], 2)
        self.assertEqual(stages["down_proj"]["matrix_plan"]["physical_tile_invocations"], 8)

    def test_b0_residual_stages_lower_to_desc_vector_tile_segments(self):
        plan = build_b0_block_plan()
        stages = {stage["stage_id"]: stage for stage in plan["stages"]}

        for stage_id in ("residual_attn", "residual_ffn"):
            vector_plan = stages[stage_id]["vector_plan"]
            self.assertEqual(vector_plan["descriptor_op"], "desc_vector_tile_v1")
            self.assertEqual(vector_plan["rows"], 8)
            self.assertEqual(vector_plan["row_width"], 16)
            self.assertEqual(vector_plan["segment_count"], 16)
            self.assertEqual(vector_plan["jobs"][0]["primitive_program"][0]["op"], "VADD")

    def test_b1_chains_real_block0_output_into_block1(self):
        plan = build_b1_two_block_plan()

        self.assertEqual(len(plan["blocks"]), 2)
        self.assertFalse(plan["block_boundary"]["cpu_recomputation"])
        self.assertFalse(plan["block_boundary"]["fixture_replacement"])
        self.assertEqual(
            plan["blocks"][1]["golden"]["input_x"],
            plan["blocks"][0]["golden"]["block_output"],
        )
        self.assertEqual(plan["block_output"], plan["blocks"][1]["golden"]["block_output"])

    def test_b0_matrix_subgraph_exports_executable_tile_jobs(self):
        workload = build_b0_matrix_subgraph_workload()

        self.assertEqual(workload["op"], "block_matmul_k_stream_group")
        self.assertEqual(workload["metadata"]["full_block_execution_state"], "planned_not_executable")
        self.assertEqual(len(workload["tile_jobs"]), 16)
        self.assertEqual(workload["max_k_chunks"], 4)
        self.assertEqual(workload["metadata"]["physical_tile_invocations"], 36)
        self.assertEqual(workload["metadata"]["effective_mac_ops"], 18432)
        self.assertEqual(workload["tile_jobs"][0]["stage_id"], "q_proj")
        self.assertEqual(workload["tile_jobs"][-1]["stage_id"], "down_proj")

    def test_residual_add_lowers_to_vector_tile_vadd_segments(self):
        lhs = [[i for i in range(16)]]
        rhs = [[100 - i for i in range(16)]]

        lowered = lower_residual_add_vector_tiles(lhs, rhs, stage_id="residual_attn")

        self.assertEqual(lowered["descriptor_op"], "desc_vector_tile_v1")
        self.assertEqual(lowered["segment_count"], 2)
        self.assertEqual(lowered["jobs"][0]["primitive_program"], [{"op": "VADD", "row": 0, "segment": 0}, {"op": "HALT"}])
        self.assertEqual(lowered["jobs"][0]["input0"], list(range(8)))
        self.assertEqual(lowered["jobs"][1]["input0"], list(range(8, 16)))
        self.assertEqual(lowered["jobs"][0]["expected_output"], [100 for _ in range(8)])


if __name__ == "__main__":
    unittest.main()
