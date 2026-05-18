import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from perf.report import parse_perf_log, write_html, write_json


class PerfReportTests(unittest.TestCase):
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
                '"input1_words":0,"output_words":0}}\n',
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
            self.assertIn("Data mover phases", html_path.read_text(encoding="utf-8"))
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

def _perf_job_line(
    job_id: int,
    name: str,
    total_cycles: int,
    matmul_cycles: int = 10,
    input1_words: int = 64,
    output_words: int = 64,
) -> str:
    return (
        f'PERF_JOB {{"id":{job_id},"name":"{name}","total_cycles":{total_cycles},'
        '"wrapper":{"desc_read":9,"fetch_program":16,"fetch_input0":64,'
        '"fetch_input1":64,"start_core":1,"wait_core":18,"write_output":64,"done":0},'
        f'"core":{{"total":18,"fetch":5,"matmul":{matmul_cycles},"done":1}},'
        '"movement":{"sram_read_cycles":153,"sram_write_cycles":64,'
        '"core_host_write_cycles":144,"core_host_read_cycles":64,'
        f'"desc_words":9,"program_words":16,"input0_words":64,'
        f'"input1_words":{input1_words},"output_words":{output_words}}}}}\n'
    )


if __name__ == "__main__":
    unittest.main()
