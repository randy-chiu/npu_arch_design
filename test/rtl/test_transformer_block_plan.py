import unittest

from npu_compiler.block import (
    B0_SHAPE,
    build_b0_attention_subgraph_workload,
    build_b0_gate_mul_vector_subgraph_workload,
    load_block_workload_spec,
    build_b0_block_plan,
    build_b0_matrix_subgraph_workload,
    build_b1_two_block_plan,
    lower_residual_add_vector_tiles,
    lower_gate_mul_vector_tiles,
    lower_rmsnorm_segmented_rows,
    validate_b0_workload_contract,
    validate_b1_workload_contract,
)


class TransformerBlockPlanTests(unittest.TestCase):
    def test_checked_in_block_workloads_match_compiler_planner_contract(self):
        validate_b0_workload_contract()
        validate_b1_workload_contract()

    def test_b0_workload_topology_drift_is_rejected(self):
        spec = load_block_workload_spec("workloads/transformer/block/tinyllama_derived_b0.jsonc")
        spec["topology"] = list(spec["topology"])
        spec["topology"][1] = "qk_score_matmul"

        with self.assertRaisesRegex(ValueError, "B0 topology mismatch"):
            validate_b0_workload_contract(spec=spec)

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

    def test_b0_rmsnorm_stages_have_segmented_plan_but_are_not_executable(self):
        plan = build_b0_block_plan()
        stages = {stage["stage_id"]: stage for stage in plan["stages"]}

        for stage_id in ("rmsnorm_attn", "rmsnorm_ffn"):
            rmsnorm_plan = stages[stage_id]["rmsnorm_plan"]
            self.assertEqual(
                rmsnorm_plan["execution_state"],
                "compiler_lowered_executable_subgraph",
            )
            self.assertEqual(rmsnorm_plan["required_primitives"], ["REDUCE_SUMSQ", "SFU_RSQRT", "VEC_SCALE"])
            self.assertEqual(rmsnorm_plan["rows"], 8)
            self.assertEqual(rmsnorm_plan["row_width"], 16)
            self.assertEqual(rmsnorm_plan["segment_count"], 16)
            self.assertEqual(rmsnorm_plan["ppa_theory"]["theoretical_reduction_cycles"], 32)
            self.assertEqual(rmsnorm_plan["ppa_theory"]["theoretical_sfu_cycles"], 16)
            self.assertEqual(rmsnorm_plan["ppa_theory"]["theoretical_vector_cycles"], 16)
            self.assertEqual(
                stages[stage_id]["provenance"],
                "compiler_desc_vector_tile_rmsnorm_lowered_not_full_block_submitted",
            )

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

    def test_b0_gate_mul_stage_lowers_to_desc_vector_tile_segments(self):
        plan = build_b0_block_plan()
        stages = {stage["stage_id"]: stage for stage in plan["stages"]}

        gate_mul_plan = stages["gate_mul_up"]["gate_mul_plan"]
        self.assertEqual(gate_mul_plan["descriptor_op"], "desc_vector_tile_v1")
        self.assertEqual(gate_mul_plan["rows"], 8)
        self.assertEqual(gate_mul_plan["row_width"], 32)
        self.assertEqual(gate_mul_plan["segment_count"], 32)
        self.assertEqual(gate_mul_plan["required_primitives"], ["VEC_MUL", "VEC_REQUANT"])
        self.assertEqual(gate_mul_plan["jobs"][0]["primitive_program"][0]["op"], "VMUL")
        self.assertEqual(gate_mul_plan["jobs"][0]["primitive_program"][1]["op"], "VREQUANT")

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

    def test_gate_mul_lowers_to_vector_tile_vmul_requant_segments(self):
        silu_gate = [[i - 4 for i in range(32)]]
        up = [[2 for _ in range(32)]]

        lowered = lower_gate_mul_vector_tiles(silu_gate, up, stage_id="gate_mul_up")

        self.assertEqual(lowered["descriptor_op"], "desc_vector_tile_v1")
        self.assertEqual(lowered["execution_state"], "compiler_lowered_executable_subgraph")
        self.assertEqual(lowered["segment_count"], 4)
        self.assertEqual(lowered["jobs"][0]["primitive_program"], [
            {"op": "VMUL", "row": 0, "segment": 0},
            {"op": "VREQUANT", "mode": "INT8_SHIFT4_CLAMP"},
            {"op": "HALT"},
        ])
        self.assertEqual(lowered["jobs"][0]["expected_output"], [-1, -1, -1, -1, 0, 0, 0, 0])

    def test_b0_gate_mul_subgraph_exports_executable_tile_jobs(self):
        workload = build_b0_gate_mul_vector_subgraph_workload()

        self.assertEqual(workload["op"], "block_desc_vector_tile_group")
        self.assertEqual(workload["metadata"]["full_block_execution_state"], "planned_not_executable")
        self.assertEqual(len(workload["tile_jobs"]), 32)
        self.assertEqual(workload["metadata"]["effective_vector_lane_ops"], 256)
        self.assertEqual(workload["metadata"]["theoretical_vector_cycles"], 64)
        self.assertEqual(workload["tile_jobs"][0]["stage_id"], "gate_mul_up")

    def test_b0_attention_subgraph_exports_two_measured_heads(self):
        workload = build_b0_attention_subgraph_workload()

        self.assertEqual(workload["op"], "block_attention_head_group")
        self.assertEqual(workload["metadata"]["full_block_execution_state"], "planned_not_executable")
        self.assertEqual(workload["metadata"]["heads"], 2)
        self.assertEqual(len(workload["stage_jobs"]), 8)
        self.assertEqual([job["stage"] for job in workload["stage_jobs"][:4]], ["qk", "scale_mask", "softmax", "pv"])
        self.assertEqual(workload["stage_jobs"][0]["op"], "matmul_k_stream")
        self.assertEqual(workload["stage_jobs"][3]["op"], "matmul_u16s8_q15")
        self.assertEqual(workload["metadata"]["valid_lane_masks"], [1, 3, 7, 15, 31, 63, 127, 255])
        self.assertEqual(workload["row_mask_words"], [0x0f070301, 0xff7f3f1f])
        self.assertEqual(
            workload["metadata"]["explicit_gap"],
            [
                "rope_q and rope_k are compiler golden inputs for this subgraph",
                "softmax uses current RTL bring-up LUT contract, not final BlockPlan fixed-spec golden",
            ],
        )

    def test_rmsnorm_lowers_to_reduce_rsqrt_scale_segments(self):
        source = [[8 for _ in range(16)]]
        expected = [[8 for _ in range(16)]]

        lowered = lower_rmsnorm_segmented_rows(source, expected, stage_id="rmsnorm_attn")

        self.assertEqual(lowered["descriptor_op"], "desc_vector_tile_v1")
        self.assertEqual(lowered["execution_state"], "compiler_lowered_executable_subgraph")
        self.assertEqual(lowered["segment_count"], 2)
        self.assertEqual(lowered["row_plans"][0]["sumsq"], 1024)
        self.assertEqual(lowered["row_plans"][0]["rsqrt_q24"], 524288)
        self.assertEqual(lowered["row_plans"][0]["reduce_segments"][0]["partial_sumsq"], 512)
        self.assertEqual(lowered["row_plans"][0]["sfu"]["primitive"], "SFU_RSQRT")
        self.assertEqual(lowered["row_plans"][0]["scale_segments"][0]["primitive_program"][0]["op"], "VEC_SCALE")
        self.assertEqual(lowered["row_plans"][0]["scale_segments"][1]["expected_output"], [8 for _ in range(8)])


if __name__ == "__main__":
    unittest.main()
