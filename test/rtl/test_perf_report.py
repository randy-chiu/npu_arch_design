import json
import importlib.util
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional

from perf.report import build_highlights, load_measurement_model, parse_perf_log, write_html, write_json


class PerfReportTests(unittest.TestCase):
    def test_production_measurement_model_comes_from_arch_configs(self):
        model = load_measurement_model(
            Path("arch/configs/npu_v0.jsonc"),
            Path("arch/configs/soc_v0.jsonc"),
        )
        self.assertEqual(model["matmul_tile"], [8, 8, 8])
        self.assertEqual(model["data_mover_words_per_cycle"], 4)

    def test_csr_records_mark_architectural_report_provenance(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "csr.log"
            log_path.write_text(
                'PERF_JOB {"source":"architectural_perf_csr_snapshot","job_id":1,"id":1,'
                '"name":"matmul","total_cycles":82,"core":{"total":17,"matmul":10},'
                '"data_mover":{"active_cycles":52,"setup_cycles":0,"transfer_cycles":52,'
                '"stall_cycles":0,"words":208,"read_words":144,"write_words":64},'
                '"sram":{"read_words":155,"write_words":64}}\n',
                encoding="utf-8",
            )

            report = parse_perf_log(log_path)

        self.assertEqual(
            report["source"]["performance"],
            "measured_architectural_perf_csr_snapshot",
        )
        self.assertEqual(report["workloads"][0]["data_mover"]["read_words"], 144)
        self.assertEqual([lane["module"] for lane in report["jobs"][0]["timeline"]], ["CPU firmware"])

    def test_csr_highlight_does_not_claim_unmeasured_overlap_span(self):
        highlights = build_highlights(
            [
                {
                    "name": "real_mnist_cnn_fc1_full_k_stream_layer",
                    "jobs": 1,
                    "job_ids": [1],
                    "total_cycles": 39218,
                    "core_matmul_cycles": 11520,
                    "data_mover": {"words": 147536, "transfer_cycles": 36884},
                    "metadata": {
                        "comparison_baseline": {
                            "id": "npu_v0_a2_serial_k_stream_proxy",
                            "cycles_per_job": 58784,
                        }
                    },
                }
            ],
            [{"source": "architectural_perf_csr_snapshot", "job_id": 1, "timeline": []}],
        )

        self.assertIsNone(highlights[0]["overlap_cycles"])

    def test_firmware_fixture_tool_emits_manifest_from_job_counts(self):
        tool_path = Path("sw/tools/firmware/emit_soc_cpu_smoke_data.py")
        spec = importlib.util.spec_from_file_location("emit_soc_cpu_smoke_data", tool_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            module._write_workload_manifest(manifest_path, 16, 1, 1, 16, 1152, 32)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["generated_by"], "sw/tools/firmware/emit_soc_cpu_smoke_data.py")
        self.assertEqual(len(manifest["jobs"]), 68)
        self.assertEqual(manifest["jobs"][20]["workload"], "real_mnist_cnn_fc1_full_k_stream_layer")
        self.assertEqual(manifest["jobs"][-1]["job_id"], 68)
        self.assertEqual(
            manifest["workload_metadata"]["real_mnist_cnn_fc1_full_k_stream_layer"]["metadata"]["k_chunks"],
            1152,
        )

    def test_manifest_groups_jobs_by_job_id_not_log_order(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            manifest_path = tmp_path / "manifest.json"
            first = _perf_job_line(7, "softmax", 53, matmul_cycles=0).replace(
                '{"id":7', '{"job_id":7,"id":7'
            )
            second = _perf_job_line(4, "matmul", 236).replace(
                '{"id":4', '{"job_id":4,"id":4'
            )
            log_path.write_text(first + second, encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "npu_workload_manifest_v0",
                        "manifest_id": "reordered_v0",
                        "run_name": "unit",
                        "jobs": [
                            {
                                "job_id": 4,
                                "workload": "matrix_regression",
                                "op": "matmul",
                                "role": "regression",
                            },
                            {
                                "job_id": 7,
                                "workload": "vector_regression",
                                "op": "softmax",
                                "role": "regression",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = parse_perf_log(log_path, manifest_path)

            self.assertEqual(report["workload_manifest"]["id"], "reordered_v0")
            self.assertEqual(report["workloads"][0]["name"], "matrix_regression")
            self.assertEqual(report["workloads"][0]["job_ids"], [4])
            self.assertEqual(report["workloads"][1]["name"], "vector_regression")
            self.assertEqual(report["workloads"][1]["job_ids"], [7])

    def test_manifest_requires_explicit_perf_job_id(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            manifest_path = tmp_path / "manifest.json"
            log_path.write_text(_perf_job_line(1, "matmul", 236), encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "npu_workload_manifest_v0",
                        "manifest_id": "unit_v0",
                        "run_name": "unit",
                        "jobs": [
                            {
                                "job_id": 1,
                                "workload": "operator_smoke_matmul",
                                "op": "matmul",
                                "role": "smoke",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required job_id"):
                parse_perf_log(log_path, manifest_path)

    def test_perf_log_generates_json_and_html_report(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            json_path = tmp_path / "perf.json"
            html_path = tmp_path / "perf.html"
            log_path.write_text(
                'PERF_JOB {"id":1,"name":"matmul","total_cycles":10,'
                '"wrapper":{"desc_read":1},"core":{"total":5,"matmul":4},'
                '"movement":{"sram_read_cycles":1,"sram_write_cycles":0,'
                '"core_host_write_cycles":0,"core_host_read_cycles":0,'
                '"desc_words":1,"program_words":0,"input0_words":0,'
                '"input1_words":0,"output_words":0},'
                '"data_mover":{"active_cycles":0,"setup_cycles":0,'
                '"transfer_cycles":0,"stall_cycles":0,"words":0,'
                '"read_cycles":0,"write_cycles":0,"read_words":0,'
                '"write_words":0}}\n',
                encoding="utf-8",
            )

            report = parse_perf_log(log_path)
            write_json(report, json_path)
            write_html(report, html_path)

            self.assertEqual(report["summary"]["jobs"], 1)
            self.assertEqual(report["summary"]["workloads"], 1)
            self.assertEqual(report["summary"]["total_cycles"], 10)
            self.assertEqual(report["workloads"][0]["name"], "operator_smoke_matmul")
            self.assertEqual(report["workloads"][0]["job_ids"], [1])
            self.assertEqual(report["jobs"][0]["estimates"]["scalar_compute_cycles"], 512)
            self.assertEqual(report["jobs"][0]["estimates"]["ideal_array_compute_cycles"], 8)
            self.assertEqual(report["jobs"][0]["movement_estimates"]["total_words"], 1)
            self.assertEqual(report["jobs"][0]["movement_estimates"]["conservative_burst_cycles"], 2)
            self.assertEqual(report["jobs"][0]["movement_estimates"]["measured_data_mover_words"], 0)
            self.assertEqual(report["workloads"][0]["data_mover"]["active_cycles"], 0)
            self.assertEqual(report["jobs"][0]["timeline"][0]["module"], "CPU firmware")
            self.assertEqual(report["jobs"][0]["timeline"][1]["module"], "NPU wrapper")
            self.assertEqual(report["jobs"][0]["timeline"][2]["module"], "Data mover")
            self.assertEqual(report["jobs"][0]["timeline"][3]["module"], "NPU core")
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["schema"], "npu_perf_report_v0")
            self.assertIn("timeline", json_path.read_text(encoding="utf-8"))
            self.assertIn("movement", json_path.read_text(encoding="utf-8"))
            self.assertIn("NPU Cycle Report", html_path.read_text(encoding="utf-8"))
            self.assertIn("Workload Summary", html_path.read_text(encoding="utf-8"))
            self.assertIn("Cycle timeline", html_path.read_text(encoding="utf-8"))
            self.assertIn("Movement model", html_path.read_text(encoding="utf-8"))
            self.assertIn("renderPhaseTimeline(section", html_path.read_text(encoding="utf-8"))
            self.assertIn('document.createElement("details")', html_path.read_text(encoding="utf-8"))
            self.assertIn("highlights", json_path.read_text(encoding="utf-8"))
            self.assertIn("wrapper reads job descriptor words from SRAM", html_path.read_text(encoding="utf-8"))

    def test_perf_log_groups_digits_classifier_tile_jobs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            lines = [
                _perf_job_line(1, "matmul", 236),
                _perf_job_line(2, "softmax", 53, matmul_cycles=0, input1_words=0, output_words=8),
            ]
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(3, 19))
            log_path.write_text("".join(lines), encoding="utf-8")

            report = parse_perf_log(log_path)

            self.assertEqual(report["summary"]["jobs"], 18)
            self.assertEqual(report["summary"]["workloads"], 3)
            self.assertEqual(report["summary"]["total_cycles"], 4065)
            classifier = report["workloads"][2]
            self.assertEqual(classifier["name"], "digits_linear_classifier")
            self.assertEqual(classifier["kind"], "model")
            self.assertEqual(classifier["jobs"], 16)
            self.assertEqual(classifier["job_ids"], list(range(3, 19)))
            self.assertEqual(classifier["total_cycles"], 3776)
            self.assertEqual(classifier["core_matmul_cycles"], 160)
            self.assertEqual(classifier["movement"]["input0_words"], 16 * 64)
            self.assertEqual(classifier["metadata"]["tile_jobs"], 16)

    def test_perf_log_groups_real_mnist_cnn_fc2_tile_jobs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            lines = [
                _perf_job_line(1, "matmul", 236),
                _perf_job_line(2, "softmax", 53, matmul_cycles=0, input1_words=0, output_words=8),
            ]
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(3, 19))
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(19, 51))
            log_path.write_text("".join(lines), encoding="utf-8")

            report = parse_perf_log(log_path)

            self.assertEqual(report["summary"]["jobs"], 50)
            self.assertEqual(report["summary"]["workloads"], 4)
            fc2 = report["workloads"][3]
            self.assertEqual(fc2["name"], "real_mnist_cnn_fc2")
            self.assertEqual(fc2["kind"], "model_layer")
            self.assertEqual(fc2["jobs"], 32)
            self.assertEqual(fc2["job_ids"], list(range(19, 51)))
            self.assertEqual(fc2["total_cycles"], 32 * 236)
            self.assertEqual(fc2["core_matmul_cycles"], 32 * 10)
            self.assertEqual(fc2["movement"]["input0_words"], 32 * 64)
            self.assertEqual(fc2["metadata"]["tile_jobs"], 32)

    def test_perf_log_groups_real_mnist_cnn_fc1_tile_before_fc2(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            lines = [
                _perf_job_line(1, "matmul", 236),
                _perf_job_line(2, "softmax", 53, matmul_cycles=0, input1_words=0, output_words=8),
            ]
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(3, 19))
            lines.append(_perf_job_line(19, "matmul", 236))
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(20, 52))
            log_path.write_text("".join(lines), encoding="utf-8")

            report = parse_perf_log(log_path)

            self.assertEqual(report["summary"]["jobs"], 51)
            self.assertEqual(report["summary"]["workloads"], 5)
            fc1 = report["workloads"][3]
            fc2 = report["workloads"][4]
            self.assertEqual(fc1["name"], "real_mnist_cnn_fc1_tile0")
            self.assertEqual(fc1["kind"], "model_layer_tile")
            self.assertEqual(fc1["jobs"], 1)
            self.assertEqual(fc1["job_ids"], [19])
            self.assertEqual(fc1["metadata"]["tile_jobs"], 1)
            self.assertEqual(fc2["name"], "real_mnist_cnn_fc2")
            self.assertEqual(fc2["job_ids"], list(range(20, 52)))

    def test_perf_log_groups_real_mnist_cnn_fc1_k_stream_before_fc2(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            lines = [
                _perf_job_line(1, "matmul", 236),
                _perf_job_line(2, "softmax", 53, matmul_cycles=0, input1_words=0, output_words=8),
            ]
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(3, 19))
            lines.append(_perf_job_line(19, "matmul", 236))
            lines.append(_perf_job_line(20, "matmul_k_stream", 727, matmul_cycles=40, input0_words=256, input1_words=256))
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(21, 53))
            log_path.write_text("".join(lines), encoding="utf-8")

            report = parse_perf_log(log_path)

            self.assertEqual(report["summary"]["jobs"], 52)
            self.assertEqual(report["summary"]["workloads"], 6)
            fc1_tile = report["workloads"][3]
            fc1_stream = report["workloads"][4]
            fc2 = report["workloads"][5]
            self.assertEqual(fc1_tile["name"], "real_mnist_cnn_fc1_tile0")
            self.assertEqual(fc1_stream["name"], "real_mnist_cnn_fc1_k_stream_smoke")
            self.assertEqual(fc1_stream["kind"], "model_layer_tile")
            self.assertEqual(fc1_stream["jobs"], 1)
            self.assertEqual(fc1_stream["job_ids"], [20])
            self.assertEqual(fc1_stream["movement"]["input0_words"], 256)
            self.assertEqual(fc1_stream["movement"]["input1_words"], 256)
            self.assertEqual(fc2["name"], "real_mnist_cnn_fc2")
            self.assertEqual(fc2["job_ids"], list(range(21, 53)))

    def test_perf_log_groups_real_mnist_cnn_fc1_full_k_stream_before_fc2(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            lines = [
                _perf_job_line(1, "matmul", 236),
                _perf_job_line(2, "softmax", 53, matmul_cycles=0, input1_words=0, output_words=8),
            ]
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(3, 19))
            lines.append(_perf_job_line(19, "matmul", 236))
            lines.append(_perf_job_line(20, "matmul_k_stream", 727, matmul_cycles=40, input0_words=256, input1_words=256))
            for job_id in range(21, 37):
                lines.append(
                    _perf_job_line(
                        job_id,
                        "matmul_k_stream",
                        39217,
                        matmul_cycles=11520,
                        input0_words=73728,
                        input1_words=73728,
                        data_mover_read_words=147472,
                        data_mover_write_words=64,
                        data_mover_transfer_cycles=36884,
                        wrapper_wait_core=36850,
                    )
                )
            lines.extend(_perf_job_line(job_id, "matmul", 236) for job_id in range(37, 69))
            log_path.write_text("".join(lines), encoding="utf-8")

            report = parse_perf_log(log_path)

            self.assertEqual(report["summary"]["jobs"], 68)
            self.assertEqual(report["summary"]["workloads"], 7)
            fc1_tile = report["workloads"][3]
            fc1_smoke = report["workloads"][4]
            fc1_full = report["workloads"][5]
            fc2 = report["workloads"][6]
            self.assertEqual(fc1_tile["name"], "real_mnist_cnn_fc1_tile0")
            self.assertEqual(fc1_smoke["name"], "real_mnist_cnn_fc1_k_stream_smoke")
            self.assertEqual(fc1_full["name"], "real_mnist_cnn_fc1_full_k_stream_layer")
            self.assertEqual(fc1_full["kind"], "model_layer")
            self.assertEqual(fc1_full["jobs"], 16)
            self.assertEqual(fc1_full["job_ids"], list(range(21, 37)))
            self.assertEqual(fc1_full["metadata"]["k_chunks"], 1152)
            self.assertEqual(fc1_full["movement"]["input0_words"], 16 * 73728)
            self.assertEqual(fc1_full["movement"]["input1_words"], 16 * 73728)
            self.assertLess(fc1_full["total_cycles"], 16 * 58784)
            self.assertEqual(fc1_full["core_matmul_cycles"], 16 * 11520)
            self.assertEqual(fc1_full["data_mover"]["transfer_cycles"], 16 * 36884)
            self.assertEqual(fc1_full["data_mover"]["words"], 16 * 147536)
            self.assertEqual(fc1_full["data_mover"]["read_words"], 16 * 147472)
            self.assertEqual(fc1_full["data_mover"]["write_words"], 16 * 64)
            highlight = report["highlights"][0]
            self.assertEqual(highlight["title"], "FC1 K-stream ping-pong overlap")
            self.assertEqual(highlight["before_cycles"], 16 * 58784)
            self.assertEqual(highlight["after_cycles"], 16 * 39217)
            self.assertEqual(highlight["cycles_saved"], 16 * 19567)
            self.assertEqual(highlight["data_mover_words"], 16 * 147536)
            self.assertEqual(highlight["core_matmul_cycles"], 16 * 11520)
            full_job = report["jobs"][20]
            data_mover_lane = next(lane for lane in full_job["timeline"] if lane["module"] == "Data mover")
            core_lane = next(lane for lane in full_job["timeline"] if lane["module"] == "NPU core")
            prefetch_span = next(
                span for span in data_mover_lane["spans"] if span["label"] == "K prefetch overlap"
            )
            core_fetch_span = core_lane["spans"][0]
            self.assertEqual(prefetch_span["start"], core_fetch_span["start"])
            self.assertGreater(prefetch_span["cycles"], full_job["core"]["matmul"])
            self.assertEqual(fc2["name"], "real_mnist_cnn_fc2")
            self.assertEqual(fc2["job_ids"], list(range(37, 69)))

def _perf_job_line(
    job_id: int,
    name: str,
    total_cycles: int,
    matmul_cycles: int = 10,
    input0_words: int = 64,
    input1_words: int = 64,
    output_words: int = 64,
    data_mover_read_words: Optional[int] = None,
    data_mover_write_words: Optional[int] = None,
    data_mover_transfer_cycles: Optional[int] = None,
    wrapper_wait_core: int = 18,
) -> str:
    read_words = input0_words + input1_words if data_mover_read_words is None else data_mover_read_words
    write_words = output_words if data_mover_write_words is None else data_mover_write_words
    transfer_cycles = (
        input0_words + input1_words + output_words
        if data_mover_transfer_cycles is None
        else data_mover_transfer_cycles
    )
    data_mover_words = read_words + write_words
    return (
        f'PERF_JOB {{"id":{job_id},"name":"{name}","total_cycles":{total_cycles},'
        '"wrapper":{"desc_read":9,"fetch_program":16,"fetch_input0":64,'
        f'"fetch_input1":64,"start_core":1,"wait_core":{wrapper_wait_core},'
        '"write_output":64,"done":0},'
        f'"core":{{"total":18,"fetch":5,"matmul":{matmul_cycles},"done":1}},'
        '"movement":{"sram_read_cycles":153,"sram_write_cycles":64,'
        '"core_host_write_cycles":144,"core_host_read_cycles":64,'
        f'"desc_words":9,"program_words":16,"input0_words":{input0_words},'
        f'"input1_words":{input1_words},"output_words":{output_words}}},'
        f'"data_mover":{{"active_cycles":{transfer_cycles},'
        '"setup_cycles":0,'
        f'"transfer_cycles":{transfer_cycles},'
        '"stall_cycles":0,'
        f'"words":{data_mover_words},'
        f'"read_cycles":{read_words},'
        f'"write_cycles":{write_words},'
        f'"read_words":{read_words},'
        f'"write_words":{write_words}}}}}\n'
    )


if __name__ == "__main__":
    unittest.main()
