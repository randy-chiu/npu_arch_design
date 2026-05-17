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
            self.assertEqual(report["summary"]["total_cycles"], 10)
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
            self.assertIn("Cycle timeline", html_path.read_text(encoding="utf-8"))
            self.assertIn("Movement model", html_path.read_text(encoding="utf-8"))
            self.assertIn("Data mover phases", html_path.read_text(encoding="utf-8"))
            self.assertIn("wrapper reads job descriptor words from SRAM", html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
