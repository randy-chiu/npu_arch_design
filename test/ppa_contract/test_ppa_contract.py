import json
import re
import shutil
import subprocess
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from ppa.model_report import build_ppa_report, read_jsonc
from ppa.schema_check import validate_ppa_report


PPA_TARGET_PATH = Path("arch/configs/ppa/sky130hd_v0.jsonc")
PPA_SCHEMA_PATH = Path("ppa/schema/ppa_result.schema.json")
PPA_PROXY_SCHEMA_PATH = Path("ppa/schema/ppa_report.schema.json")
AREA_PROXY_PATH = Path("arch/configs/ppa/area_model_v0.jsonc")
ENERGY_PROXY_PATH = Path("arch/configs/ppa/energy_model_v0.jsonc")
SERIAL_BASELINE_PATH = Path("ppa/baselines/l0/npu_v0_a2_serial_k_stream_l0.json")
TRANSFORMER_MANIFEST_PATH = Path("workloads/manifests/transformer/transformer_micro_v0.jsonc")
SUBSYSTEM_RTL_PATH = Path("hw/npu_subsystem/rtl/npu_subsystem_top.sv")


def _read_jsonc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(re.sub(r"//.*$", "", text, flags=re.MULTILINE))


class PPAContractTests(unittest.TestCase):
    def test_initial_target_names_primary_subsystem_boundary(self):
        target = _read_jsonc(PPA_TARGET_PATH)

        self.assertEqual(target["flow"]["platform"], "sky130hd")
        self.assertEqual(target["tops"]["primary"], "npu_subsystem_top")
        self.assertEqual(target["interpretation"], "estimate_not_signoff")
        self.assertTrue(
            target["memory_accounting"]["simulation_soc_sram_is_excluded_from_primary_npu_ppa"]
        )

    def test_result_schema_requires_core_measurement_identity(self):
        schema = json.loads(PPA_SCHEMA_PATH.read_text(encoding="utf-8"))
        model_schema = json.loads(PPA_PROXY_SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(schema["title"], "NPU PPA Result")
        self.assertIn("design", schema["required"])
        self.assertIn("performance", schema["required"])
        self.assertIn("npu_subsystem", schema["properties"]["design"]["properties"]["top"]["enum"])
        self.assertEqual(model_schema["properties"]["evidence_level"]["const"], "L0_model")

    def test_model_report_labels_modeled_metrics_and_uses_measured_events(self):
        perf = {
            "source_log": "synthetic.log",
            "source": {"performance": "measured_architectural_perf_csr_snapshot"},
            "workloads": [
                {
                    "name": "real_mnist_cnn_fc1_full_k_stream_tile0",
                    "kind": "model_layer",
                    "jobs": 1,
                    "total_cycles": 39217,
                    "core_matmul_cycles": 11520,
                    "data_mover": {
                        "words": 147536,
                        "read_words": 147472,
                        "write_words": 64,
                    },
                    "metadata": {},
                }
            ],
            "highlights": [
                {
                    "title": "FC1 K-stream ping-pong overlap",
                    "workload": "real_mnist_cnn_fc1_full_k_stream_tile0",
                    "before_cycles": 58784,
                    "after_cycles": 39217,
                    "cycles_saved": 19567,
                    "core_matmul_cycles": 11520,
                    "data_mover_words": 147536,
                }
            ],
        }

        report = build_ppa_report(perf, read_jsonc(AREA_PROXY_PATH), read_jsonc(ENERGY_PROXY_PATH))
        workload = report["workloads"][0]

        self.assertEqual(report["evidence_level"], "L0_model")
        self.assertEqual(report["area_model"]["storage_bits_total"], 7968)
        self.assertEqual(report["area_model"]["normalized_area_units"], 6998.4)
        self.assertEqual(workload["performance"]["provenance"], "measured_architectural_perf_csr_snapshot")
        self.assertEqual(workload["energy_model"]["events"]["int8_mac_accumulate"], 1152 * 512)
        self.assertEqual(workload["energy_model"]["events"]["data_mover_read_word"], 147472)
        self.assertEqual(
            report["highlights"][0]["modeled_energy_saved_from_shorter_active_duration_only"],
            19567 * 0.25,
        )

    def test_model_report_compares_ping_pong_candidate_against_serial_baseline(self):
        baseline_report = json.loads(SERIAL_BASELINE_PATH.read_text(encoding="utf-8"))
        validate_ppa_report(baseline_report)
        candidate_perf = {
            "workload_manifest_id": "soc_cpu_smoke_v0",
            "workloads": [
                {
                    "name": "real_mnist_cnn_fc1_full_k_stream_layer",
                    "kind": "model_layer",
                    "jobs": 16,
                    "total_cycles": 627472,
                    "core_matmul_cycles": 184320,
                    "data_mover": {
                        "words": 2360576,
                        "read_words": 2359552,
                        "write_words": 1024,
                    },
                    "metadata": {},
                }
            ],
            "highlights": [],
        }

        report = build_ppa_report(
            candidate_perf,
            read_jsonc(AREA_PROXY_PATH),
            read_jsonc(ENERGY_PROXY_PATH),
            baseline_report=baseline_report,
        )
        comparison = report["comparison"]
        delta = comparison["workload_deltas"][0]

        self.assertEqual(comparison["baseline"]["variant"], "npu_v0_a2_serial_k_stream")
        self.assertEqual(comparison["candidate"]["variant"], "npu_v0_a2_ping_pong")
        self.assertEqual(comparison["area_delta"]["classification"], "regression")
        self.assertEqual(comparison["area_delta"]["delta"], 51.2)
        self.assertEqual(delta["cycles"]["classification"], "improvement")
        self.assertEqual(delta["cycles"]["delta"], -313072)
        self.assertEqual(delta["data_mover_words"]["classification"], "invariant")
        self.assertEqual(delta["int8_mac_accumulate"]["classification"], "invariant")
        self.assertEqual(delta["energy_model"]["classification"], "improvement")
        self.assertTrue(comparison["improvements"])
        self.assertTrue(any("area model increases" in item for item in comparison["costs"]))

    def test_model_schema_validator_requires_critical_fields(self):
        report = build_ppa_report(
            {"workloads": [], "highlights": []},
            read_jsonc(AREA_PROXY_PATH),
            read_jsonc(ENERGY_PROXY_PATH),
        )
        validate_ppa_report(report)
        del report["area_model"]["normalized_area_units"]
        with self.assertRaisesRegex(ValueError, "normalized_area_units is required"):
            validate_ppa_report(report)

    def test_model_schema_validator_rejects_inconsistent_or_negative_metrics(self):
        report = build_ppa_report(
            {
                "workloads": [
                    {
                        "name": "bad_workload",
                        "total_cycles": 1,
                        "core_matmul_cycles": 0,
                        "data_mover": {"words": 0},
                    }
                ],
                "highlights": [],
            },
            read_jsonc(AREA_PROXY_PATH),
            read_jsonc(ENERGY_PROXY_PATH),
        )
        report["area_model"]["normalized_area_units"] += 1
        report["workloads"][0]["performance"]["cycles"] = -1
        with self.assertRaisesRegex(ValueError, "contribution sum"):
            validate_ppa_report(report)
        with self.assertRaisesRegex(ValueError, "non-negative number"):
            validate_ppa_report(report)

    def test_model_comparison_marks_mismatched_manifest_incomparable(self):
        baseline = build_ppa_report(
            {"workload_manifest_id": "baseline_manifest", "workloads": [], "highlights": []},
            read_jsonc(AREA_PROXY_PATH),
            read_jsonc(ENERGY_PROXY_PATH),
        )
        candidate = build_ppa_report(
            {"workload_manifest_id": "candidate_manifest", "workloads": [], "highlights": []},
            read_jsonc(AREA_PROXY_PATH),
            read_jsonc(ENERGY_PROXY_PATH),
            baseline_report=baseline,
        )
        self.assertFalse(candidate["comparison"]["comparable"])
        self.assertIn(
            "workload_manifest_id differs or is missing on one report",
            candidate["comparison"]["compatibility"]["issues"],
        )

    def test_transformer_manifest_separates_prefill_and_decode(self):
        manifest = _read_jsonc(TRANSFORMER_MANIFEST_PATH)
        scenarios = {workload["scenario"] for workload in manifest["workloads"]}

        self.assertIn("transformer_prefill", scenarios)
        self.assertIn("transformer_decode", scenarios)
        self.assertTrue(any(workload["op"] == "memory_traffic" for workload in manifest["workloads"]))
        self.assertTrue(
            any(
                workload.get("attention_stage") == "qk"
                and workload["status"] == "planned_current_matmul_extension"
                for workload in manifest["workloads"]
            )
        )
        self.assertTrue(any(workload.get("attention_group") == "attention_prefill_s8_d8" for workload in manifest["workloads"]))
        for workload in manifest["workloads"]:
            self.assertIn("activity_scope", workload)
            self.assertIn("external_memory", workload)

    def test_model_report_keeps_transformer_external_memory_modeled_separately(self):
        perf = {
            "source": {"performance": "measured_architectural_perf_csr_snapshot"},
            "workloads": [
                {
                    "name": "transformer_prefill_gemm_tiny",
                    "kind": "transformer_micro",
                    "jobs": 1,
                    "total_cycles": 128,
                    "core_matmul_cycles": 20,
                    "data_mover": {"words": 512, "read_words": 448, "write_words": 64},
                    "transformer_metrics": {
                        "effective_mac_ops": 1024,
                        "matrix_utilization": 0.8,
                        "gemv_utilization": None,
                        "skinny_gemm_utilization": 0.8,
                        "kv_read_bytes": 0,
                        "kv_write_bytes": 0,
                        "bytes_per_token": None,
                        "softmax_cycles": 0,
                        "rmsnorm_cycles": 0,
                        "sfu_cycles": 0,
                    },
                    "metadata": {
                        "external_memory": {
                            "activation_read_bytes": 128,
                            "activation_write_bytes": 256,
                            "weight_read_bytes": 128,
                            "kv_cache_read_bytes": 0,
                            "kv_cache_write_bytes": 0,
                        }
                    },
                }
            ],
            "model_only_workloads": [
                {
                    "name": "transformer_kv_cache_traffic_tiny",
                    "kind": "transformer_model_only",
                    "jobs": 0,
                    "total_cycles": 0,
                    "core_matmul_cycles": 0,
                    "data_mover": {},
                    "metadata": {
                        "model_only": True,
                        "external_memory": {
                            "kv_cache_read_bytes": 1024,
                            "kv_cache_write_bytes": 512,
                        },
                    },
                }
            ],
            "highlights": [],
        }

        report = build_ppa_report(perf, read_jsonc(AREA_PROXY_PATH), read_jsonc(ENERGY_PROXY_PATH))
        validate_ppa_report(report)
        prefill = report["workloads"][0]
        kv = report["workloads"][1]

        self.assertEqual(prefill["energy_model"]["events"]["external_memory_byte"], 512)
        self.assertEqual(prefill["energy_model"]["events"]["int8_mac_accumulate"], 1024)
        self.assertEqual(prefill["performance"]["matrix_utilization"], 0.8)
        self.assertEqual(prefill["performance"]["skinny_gemm_utilization"], 0.8)
        self.assertEqual(
            prefill["energy_model"]["contribution_groups"]["modeled_external_memory"],
            512 * 20.0,
        )
        self.assertEqual(kv["performance"]["provenance"], "modeled_manifest_only")
        self.assertEqual(kv["energy_model"]["events"]["external_memory_byte"], 1536)

    def test_model_report_exposes_attention_stage_metadata(self):
        perf = {
            "source": {"performance": "measured_architectural_perf_csr_snapshot"},
            "workloads": [
                {
                    "name": "transformer_attention_qk_s8_d8",
                    "kind": "transformer_micro",
                    "jobs": 1,
                    "total_cycles": 96,
                    "core_matmul_cycles": 10,
                    "data_mover": {"words": 256, "read_words": 192, "write_words": 64},
                    "metadata": {
                        "workload_family": "transformer_prefill",
                        "attention_group": "attention_prefill_s8_d8",
                        "attention_stage": "qk",
                        "numerical_contract": "attention_bringup_v0_qk_exact",
                        "stage_provenance": "measured_current_matmul_path",
                        "logical_shape": {"m": 8, "n": 8, "k": 8},
                        "shape_class": "skinny_gemm",
                        "external_memory": {
                            "activation_read_bytes": 64,
                            "activation_write_bytes": 256,
                            "weight_read_bytes": 64,
                            "kv_cache_read_bytes": 0,
                            "kv_cache_write_bytes": 0,
                        },
                    },
                }
            ],
            "model_only_workloads": [
                {
                    "name": "transformer_attention_softmax_s8",
                    "kind": "transformer_model_only",
                    "jobs": 0,
                    "total_cycles": 0,
                    "core_matmul_cycles": 0,
                    "data_mover": {},
                    "metadata": {
                        "model_only": True,
                        "attention_group": "attention_prefill_s8_d8",
                        "attention_stage": "softmax",
                        "numerical_contract": "attention_bringup_v0_shift_scale_sfu9seg",
                        "external_memory": {
                            "activation_read_bytes": 256,
                            "activation_write_bytes": 128,
                        },
                    },
                }
            ],
            "highlights": [],
        }

        perf["workloads"][0]["transformer_metrics"] = {
            "attention_group": "attention_prefill_s8_d8",
            "attention_stage": "qk",
            "numerical_contract": "attention_bringup_v0_qk_exact",
            "stage_provenance": "measured_current_matmul_path",
            "effective_mac_ops": 512,
            "matrix_utilization": 0.8,
            "gemv_utilization": None,
            "skinny_gemm_utilization": 0.8,
            "kv_read_bytes": 0,
            "kv_write_bytes": 0,
            "bytes_per_token": None,
            "qk_cycles": 96,
            "theoretical_compute_cycles": 8,
            "measured_compute_cycles": 10,
            "compute_overhead_cycles": 2,
            "compute_efficiency": 0.8,
            "non_compute_overhead_cycles": 86,
            "end_to_end_efficiency": 0.083333,
            "theoretical_cycle_basis": "ceil(effective_mac_ops=512/peak_macs_per_cycle=64)",
            "measured_compute_provenance": "measured_matrix_active_cycles",
            "attention_softmax_cycles": None,
            "pv_cycles": None,
        }
        perf["model_only_workloads"][0]["transformer_metrics"] = {
            "attention_group": "attention_prefill_s8_d8",
            "attention_stage": "softmax",
            "numerical_contract": "attention_bringup_v0_shift_scale_sfu9seg",
            "stage_provenance": "model_only_fixed_spec",
            "effective_mac_ops": None,
            "kv_read_bytes": 0,
            "kv_write_bytes": 0,
            "bytes_per_token": None,
            "qk_cycles": None,
            "attention_softmax_cycles": None,
            "pv_cycles": None,
        }

        report = build_ppa_report(perf, read_jsonc(AREA_PROXY_PATH), read_jsonc(ENERGY_PROXY_PATH))
        validate_ppa_report(report)
        qk = report["workloads"][0]
        softmax = report["workloads"][1]

        self.assertEqual(qk["performance"]["attention_stage"], "qk")
        self.assertEqual(qk["performance"]["qk_cycles"], 96)
        self.assertEqual(qk["performance"]["theoretical_compute_cycles"], 8)
        self.assertEqual(qk["performance"]["compute_overhead_cycles"], 2)
        self.assertEqual(qk["performance"]["compute_efficiency"], 0.8)
        self.assertEqual(qk["energy_model"]["events"]["int8_mac_accumulate"], 512)
        self.assertEqual(softmax["performance"]["attention_stage"], "softmax")
        self.assertEqual(softmax["performance"]["provenance"], "modeled_manifest_only")

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            from ppa.model_report import write_html

            write_html(report, out / "ppa_overview.html")
            self.assertTrue((out / "perf.html").exists())
            self.assertTrue((out / "power.html").exists())
            self.assertTrue((out / "area.html").exists())
            self.assertTrue((out / "cases" / "unspecified.html").exists())
            self.assertIn("Theoretical Versus Measured", (out / "perf.html").read_text())

    @unittest.skipUnless(shutil.which("iverilog"), "iverilog not installed")
    def test_npu_subsystem_boundary_elaborates(self):
        self.assertIn("module npu_subsystem_top", SUBSYSTEM_RTL_PATH.read_text(encoding="utf-8"))
        subprocess.run(
            ["make", "npu-subsystem-elab"],
            check=True,
            capture_output=True,
            text=True,
        )
