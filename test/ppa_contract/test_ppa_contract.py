import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from ppa.proxy_report import build_proxy_report, read_jsonc
from ppa.schema_check import validate_proxy_report


PPA_TARGET_PATH = Path("arch/configs/ppa/sky130hd_v0.jsonc")
PPA_SCHEMA_PATH = Path("ppa/schema/ppa_result.schema.json")
PPA_PROXY_SCHEMA_PATH = Path("ppa/schema/ppa_proxy_report.schema.json")
AREA_PROXY_PATH = Path("arch/configs/ppa/area_proxy_v0.jsonc")
ENERGY_PROXY_PATH = Path("arch/configs/ppa/energy_proxy_v0.jsonc")
SERIAL_BASELINE_PATH = Path("ppa/baselines/l0/npu_v0_a2_serial_k_stream_proxy.json")
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
        proxy_schema = json.loads(PPA_PROXY_SCHEMA_PATH.read_text(encoding="utf-8"))

        self.assertEqual(schema["title"], "NPU PPA Result")
        self.assertIn("design", schema["required"])
        self.assertIn("performance", schema["required"])
        self.assertIn("npu_subsystem", schema["properties"]["design"]["properties"]["top"]["enum"])
        self.assertEqual(proxy_schema["properties"]["evidence_level"]["const"], "L0_proxy")

    def test_proxy_report_labels_modeled_metrics_and_uses_measured_events(self):
        perf = {
            "source_log": "synthetic.log",
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

        report = build_proxy_report(perf, read_jsonc(AREA_PROXY_PATH), read_jsonc(ENERGY_PROXY_PATH))
        workload = report["workloads"][0]

        self.assertEqual(report["evidence_level"], "L0_proxy")
        self.assertEqual(report["area_proxy"]["storage_bits_total"], 7968)
        self.assertEqual(report["area_proxy"]["normalized_area_units"], 6998.4)
        self.assertEqual(workload["performance"]["provenance"], "measured_rtl_perf_job_counters")
        self.assertEqual(workload["energy_proxy"]["events"]["int8_mac_accumulate"], 1152 * 512)
        self.assertEqual(workload["energy_proxy"]["events"]["data_mover_read_word"], 147472)
        self.assertEqual(
            report["highlights"][0]["modeled_energy_saved_from_shorter_active_duration_only"],
            19567 * 0.25,
        )

    def test_proxy_report_compares_ping_pong_candidate_against_serial_baseline(self):
        baseline_report = json.loads(SERIAL_BASELINE_PATH.read_text(encoding="utf-8"))
        validate_proxy_report(baseline_report)
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

        report = build_proxy_report(
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
        self.assertEqual(delta["energy_proxy"]["classification"], "improvement")
        self.assertTrue(comparison["improvements"])
        self.assertTrue(any("area proxy increases" in item for item in comparison["costs"]))

    def test_proxy_schema_validator_requires_critical_fields(self):
        report = build_proxy_report(
            {"workloads": [], "highlights": []},
            read_jsonc(AREA_PROXY_PATH),
            read_jsonc(ENERGY_PROXY_PATH),
        )
        validate_proxy_report(report)
        del report["area_proxy"]["normalized_area_units"]
        with self.assertRaisesRegex(ValueError, "normalized_area_units is required"):
            validate_proxy_report(report)

    def test_proxy_schema_validator_rejects_inconsistent_or_negative_metrics(self):
        report = build_proxy_report(
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
        report["area_proxy"]["normalized_area_units"] += 1
        report["workloads"][0]["performance"]["cycles"] = -1
        with self.assertRaisesRegex(ValueError, "contribution sum"):
            validate_proxy_report(report)
        with self.assertRaisesRegex(ValueError, "non-negative number"):
            validate_proxy_report(report)

    def test_proxy_comparison_marks_mismatched_manifest_incomparable(self):
        baseline = build_proxy_report(
            {"workload_manifest_id": "baseline_manifest", "workloads": [], "highlights": []},
            read_jsonc(AREA_PROXY_PATH),
            read_jsonc(ENERGY_PROXY_PATH),
        )
        candidate = build_proxy_report(
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

    @unittest.skipUnless(shutil.which("iverilog"), "iverilog not installed")
    def test_npu_subsystem_boundary_elaborates(self):
        self.assertIn("module npu_subsystem_top", SUBSYSTEM_RTL_PATH.read_text(encoding="utf-8"))
        subprocess.run(
            ["make", "npu-subsystem-elab"],
            check=True,
            capture_output=True,
            text=True,
        )
