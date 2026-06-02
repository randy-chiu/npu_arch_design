import unittest
from pathlib import Path

from npu_compiler.attention import build_attention_plan_from_manifest
from npu_compiler.attention_plan_schema import validate_attention_plan
from transformer.generate_transformer_micro_fixtures import read_jsonc


MANIFEST = read_jsonc(Path("workloads/manifests/transformer/transformer_micro_v0.jsonc"))


class AttentionPlanTests(unittest.TestCase):
    def test_attention_plan_has_expected_stage_order_and_runtime_jobs(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")

        self.assertEqual([stage["stage_id"] for stage in plan["stages"]], ["qk", "scale_mask", "softmax", "pv"])
        self.assertEqual([job["stage_id"] for job in plan["runtime_jobs"]], ["qk", "softmax", "pv"])
        self.assertEqual(plan["group_state"], "software_group_measured_stages")
        self.assertEqual(plan["group_cycle_policy"], "sum_measured_stages")
        self.assertEqual(plan["scale_mask_provenance"], "materialized_by_fixture")

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
        self.assertEqual(jobs["softmax"]["descriptor_op"], "attention_softmax_v1")
        self.assertEqual(jobs["pv"]["descriptor_op"], "matmul_u16s8_q15")
        self.assertEqual(jobs["pv"]["input0"], "prob_q15")

    def test_validator_rejects_missing_scale_mask_stage(self):
        plan = build_attention_plan_from_manifest(MANIFEST, "attention_prefill_s8_d8")
        plan["stages"] = [stage for stage in plan["stages"] if stage["stage_id"] != "scale_mask"]

        with self.assertRaisesRegex(ValueError, "qk -> scale_mask -> softmax -> pv"):
            validate_attention_plan(plan)

    def test_manifest_lowering_rejects_unsupported_parent_shape(self):
        spec = dict(MANIFEST)
        spec["workloads"] = [dict(workload) for workload in MANIFEST["workloads"]]
        for workload in spec["workloads"]:
            if workload.get("logical_op") == "scaled_dot_product_attention":
                workload["shape"] = dict(workload["shape"])
                workload["shape"]["seq_len"] = 16

        with self.assertRaisesRegex(ValueError, "seq_len=8"):
            build_attention_plan_from_manifest(spec, "attention_prefill_s8_d8")


if __name__ == "__main__":
    unittest.main()
