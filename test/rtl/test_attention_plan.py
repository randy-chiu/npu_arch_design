import unittest
from copy import deepcopy
from pathlib import Path

from npu_compiler.attention import (
    build_attention_mask_plan,
    build_attention_plan_from_manifest,
    build_softmax_expanded_primitive_program,
)
from npu_compiler.attention_plan_schema import validate_attention_plan
from firmware.emit_soc_cpu_smoke_data import _append_transformer_runtime_plan_data
from transformer.generate_transformer_micro_fixtures import read_jsonc


MANIFEST = read_jsonc(Path("workloads/manifests/transformer/transformer_micro_v0.jsonc"))


class AttentionPlanTests(unittest.TestCase):
    def test_attention_plan_has_expected_stage_order_and_runtime_jobs(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")

        self.assertEqual([stage["stage_id"] for stage in plan["stages"]], ["qk", "scale_mask", "softmax", "pv"])
        self.assertEqual([job["stage_id"] for job in plan["runtime_jobs"]], ["qk", "scale_mask", "softmax", "pv"])
        self.assertEqual(plan["group_state"], "software_group_measured_stages")
        self.assertEqual(plan["group_cycle_policy"], "sum_measured_stages")
        self.assertEqual(plan["scale_mask_provenance"], "measured_npu_vector_bridge")
        self.assertEqual(plan["execution_state"], "executable")
        self.assertEqual(plan["mask"]["valid_lane_masks"], [1, 3, 7, 15, 31, 63, 127, 255])
        self.assertEqual(plan["mask"]["row_mask_words"], [0x0F070301, 0xFF7F3F1F])
        self.assertTrue(all(job["execution_state"] == "executable" for job in plan["runtime_jobs"]))

    def test_attention_plan_preserves_typed_intermediate_buffers(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")
        buffers = {buffer["name"]: buffer for buffer in plan["buffers"]}

        self.assertEqual(buffers["score_raw"]["dtype"], "int32")
        self.assertEqual(buffers["score_softmax_in"]["producer"], "scale_mask")
        self.assertEqual(buffers["prob_q15"]["dtype"], "uint16_q0.15")
        self.assertEqual(buffers["prob_q15"]["consumer_stage_indices"], [3])

    def test_attention_plan_uses_current_descriptor_ops(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")
        jobs = {job["stage_id"]: job for job in plan["runtime_jobs"]}

        self.assertEqual(jobs["qk"]["descriptor_op"], "matmul_k_stream")
        self.assertEqual(jobs["scale_mask"]["descriptor_op"], "attention_scale_mask_v1")
        self.assertEqual(jobs["scale_mask"]["input0"], "score_raw")
        self.assertEqual(jobs["softmax"]["descriptor_op"], "attention_softmax_v1")
        self.assertEqual(jobs["pv"]["descriptor_op"], "matmul_u16s8_q15")
        self.assertEqual(jobs["pv"]["input0"], "prob_q15")

    def test_compiler_expands_softmax_and_fits_selected_capacity(self):
        expanded = build_softmax_expanded_primitive_program(rows=8, elements=8)

        self.assertEqual(expanded["representation"], "compiler_expanded_primitives")
        self.assertEqual(expanded["required_words"], 113)
        self.assertEqual(expanded["required_bytes"], 452)
        self.assertEqual(expanded["capacity_words"], 128)
        self.assertTrue(expanded["fits_current_capacity"])
        self.assertEqual(expanded["shortfall_words"], 0)
        self.assertEqual(expanded["program"][0], {"op": "REDUCE_MAX", "row": 0})
        self.assertEqual(expanded["program"][3], {"op": "SFU_EXP", "row": 0, "lane": 0})
        self.assertEqual(expanded["program"][-1], {"op": "HALT"})

    def test_attention_plan_carries_compiler_expanded_softmax_program(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")
        softmax = next(stage for stage in plan["stages"] if stage["stage_id"] == "softmax")

        self.assertEqual(softmax["primitive_program"]["required_words"], 113)
        self.assertTrue(softmax["primitive_program"]["fits_current_capacity"])

    def test_validator_rejects_missing_scale_mask_stage(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")
        plan["stages"] = [stage for stage in plan["stages"] if stage["stage_id"] != "scale_mask"]

        with self.assertRaisesRegex(ValueError, "qk -> scale_mask -> softmax -> pv"):
            validate_attention_plan(plan)

    def test_manifest_lowering_rejects_unsupported_parent_shape(self):
        spec = deepcopy(MANIFEST)
        for workload in spec["workloads"]:
            if workload.get("logical_op") == "scaled_dot_product_attention":
                workload["shape"]["seq_len"] = 16

        with self.assertRaisesRegex(ValueError, "seq_q/seq_k in 1..8"):
            build_attention_plan_from_manifest(spec, "attention_prefill_s8_d8")

    def test_compiler_composes_causal_and_padding_masks(self):
        mask = build_attention_mask_plan(seq_q=4, seq_k=6, mask_policy="causal_padding", valid_k=3)

        self.assertEqual(mask["valid_query_mask"], 0b1111)
        self.assertEqual(mask["valid_lane_masks"], [0b000001, 0b000011, 0b000111, 0b000111, 0, 0, 0, 0])
        self.assertEqual(mask["execution_state"], "planned_not_executable")

    def test_tail_plan_is_not_silently_executable(self):
        spec = deepcopy(MANIFEST)
        for workload in spec["workloads"]:
            if workload.get("logical_op") == "scaled_dot_product_attention":
                workload["shape"]["seq_q"] = 5
                workload["shape"]["seq_k"] = 5
                workload["mask_policy"] = "padding"
                workload["valid_k"] = 5

        plan = build_attention_plan_from_manifest(spec, "attention_prefill_s8_d8")

        self.assertEqual(plan["execution_state"], "planned_not_executable")
        self.assertEqual(plan["mask"]["valid_query_mask"], 0b11111)
        self.assertEqual(plan["mask"]["valid_lane_masks"][:5], [0b11111] * 5)
        self.assertEqual(plan["mask"]["valid_lane_masks"][5:], [0, 0, 0])
        self.assertTrue(all(job["execution_state"] == "planned_not_executable" for job in plan["runtime_jobs"]))

    def test_validator_rejects_all_invalid_row_marked_executable(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")
        plan["mask"] = build_attention_mask_plan(seq_q=8, seq_k=8, mask_policy="causal")
        plan["mask"]["valid_lane_masks"][0] = 0
        plan["mask"]["row_mask_words"][0] &= 0xFFFFFF00
        plan["mask"]["execution_state"] = "executable"

        with self.assertRaisesRegex(ValueError, "valid_lane_masks do not match"):
            validate_attention_plan(plan)

    def test_firmware_rejects_non_executable_mask_plan(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")
        plan["execution_state"] = "planned_not_executable"

        with self.assertRaisesRegex(ValueError, "cannot emit non-executable attention plan"):
            _append_transformer_runtime_plan_data([], {"attention_plans": [plan]})


if __name__ == "__main__":
    unittest.main()
