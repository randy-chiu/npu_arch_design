import json
import importlib.util
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Optional

from perf.report import build_highlights, load_measurement_model, parse_perf_log, write_html, write_json
from transformer.generate_transformer_micro_fixtures import generate_transformer_micro_fixtures


class PerfReportTests(unittest.TestCase):
    def test_production_measurement_model_comes_from_arch_configs(self):
        model = load_measurement_model(
            Path("arch/configs/npu_v0.jsonc"),
            Path("arch/configs/soc_v0.jsonc"),
        )
        self.assertEqual(model["matmul_tile"], [8, 8, 8])
        self.assertEqual(model["data_mover_words_per_cycle"], 4)
        self.assertEqual(
            model["performance_contract"]["accumulator"]["commit_add_lanes"],
            64,
        )
        self.assertEqual(
            model["performance_contract"]["attention_row_storage"]["read_bus_bits"],
            256,
        )

    def test_accumulator_transaction_must_match_performance_contract(self):
        defaults = {
            "cmd_event": 6, "cmd_active": 0, "cmd_wait": 1, "stream_chunk": 0,
            "dm_program": 0, "dm_input_a": 0, "dm_input_b": 0, "dm_prefetch_a": 0,
            "dm_prefetch_b": 0, "dm_output": 0, "dm_target_bank": 0,
            "core_active": 1, "core_wait_data": 0, "uop_active": 0, "uop_wait": 0,
            "sched_wait_reason": 0, "uop_load": 0, "uop_tensor": 0, "uop_buffer": 0,
            "uop_exec": 0, "uop_opcode": 0, "uop_store": 0, "output_store_enable": 1,
            "matrix_issue": 0, "matrix_active": 0, "compute_ctrl_event": 0,
            "acc_clear": 0, "acc_commit": 0, "acc_readout": 0,
        }
        events = [
            {"job_id": 1, "cycle": 0, **defaults, "acc_commit": 1},
            {"job_id": 1, "cycle": 1, **defaults, "acc_commit": 1},
            {"job_id": 1, "cycle": 2, **defaults, "core_active": 0},
        ]
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trace.log"
            lines = ["PERF_TRACE " + json.dumps(event) for event in events]
            lines.append(
                'PERF_JOB {"source":"architectural_perf_csr_snapshot","job_id":1,"id":1,'
                '"name":"matmul","total_cycles":3,"core":{"total":2,"matmul":0},'
                '"data_mover":{"active_cycles":0,"read_words":0,"write_words":0},"sram":{}}'
            )
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "acc_commit violates performance contract"):
                parse_perf_log(
                    log_path,
                    model=load_measurement_model(
                        Path("arch/configs/npu_v0.jsonc"),
                        Path("arch/configs/soc_v0.jsonc"),
                    ),
                )

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
        self.assertEqual(
            [lane["module"] for lane in report["jobs"][0]["timeline"]],
            [
                "CPU firmware",
                "NPU wrapper",
                "NPU core",
                "Command processor",
                "Uop scheduler",
                "Data mover",
                "Compute cluster",
                "Matrix engine",
                "Accumulator file",
            ],
        )
        lanes = {lane["module"]: lane for lane in report["jobs"][0]["timeline"]}
        self.assertIsNone(lanes["NPU core"]["parent"])
        self.assertEqual(lanes["NPU core"]["depth"], 0)
        self.assertEqual(lanes["Command processor"]["parent"], "NPU core")
        self.assertEqual(lanes["Uop scheduler"]["parent"], "NPU core")
        self.assertEqual(lanes["Data mover"]["parent"], "NPU core")
        self.assertEqual(lanes["Compute cluster"]["parent"], "NPU core")
        self.assertEqual(lanes["Matrix engine"]["parent"], "Compute cluster")
        self.assertEqual(lanes["Matrix engine"]["depth"], 2)
        self.assertEqual(lanes["Accumulator file"]["parent"], "Compute cluster")
        self.assertEqual(
            report["jobs"][0]["timeline_provenance"]["span_placement"],
            "derived_from_reviewed_state_machine",
        )

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

    def test_k_stream_timeline_splits_initial_load_from_prefetch_overlap(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "csr.log"
            log_path.write_text(
                'PERF_JOB {"source":"architectural_perf_csr_snapshot","job_id":19,"id":19,'
                '"name":"matmul_k_stream","total_cycles":117,'
                '"core":{"total":19,"matmul":16,"wait_data_cycles":17,"local_active_cycles":3},'
                '"command_processor":{"active_cycles":16,"wait_cycles":102},'
                '"uop_scheduler":{"active_cycles":12,"wait_cycles":18},'
                '"data_mover":{"active_cycles":84,"setup_cycles":0,"transfer_cycles":84,'
                '"stall_cycles":0,"words":336,"read_words":272,"write_words":64,'
                '"compute_overlap_cycles":9,"program_cycles":4,'
                '"initial_input_cycles":32,"prefetch_cycles":32,"output_cycles":16},'
                '"sram":{"read_words":283,"write_words":64}}\n',
                encoding="utf-8",
            )

            report = parse_perf_log(log_path)

        lanes = {lane["module"]: lane for lane in report["jobs"][0]["timeline"]}
        wrapper_fetch = next(
            span
            for span in lanes["Command processor"]["spans"]
            if span["label"] == "Wait for input/program movement"
        )
        initial_load = next(
            span for span in lanes["Data mover"]["spans"] if span["label"] == "Initial chunk A/B load"
        )
        overlap = next(
            span
            for span in lanes["Data mover"]["spans"]
            if span["label"] == "Measured K prefetch overlap"
        )

        self.assertEqual(initial_load["cycles"], 32)
        self.assertGreater(wrapper_fetch["cycles"], initial_load["cycles"])
        self.assertEqual(overlap["cycles"], 9)
        self.assertEqual(overlap["start"], lanes["Compute cluster"]["spans"][0]["start"])
        self.assertEqual(
            next(span for span in lanes["Compute cluster"]["spans"] if span["kind"] == "wait")["cycles"],
            17,
        )
        self.assertEqual(
            [span["cycles"] for span in lanes["Matrix engine"]["spans"]],
            [8, 8],
        )
        matrix_spans = lanes["Matrix engine"]["spans"]
        accumulator_spans = lanes["Accumulator file"]["spans"]
        self.assertEqual([span["cycles"] for span in accumulator_spans], [1, 1, 1])
        for matrix_span in matrix_spans:
            self.assertTrue(
                all(
                    accumulator_span["end"] <= matrix_span["start"]
                    or accumulator_span["start"] >= matrix_span["end"]
                    for accumulator_span in accumulator_spans
                )
            )

    def test_k_stream_cycle_trace_places_external_and_local_loads_in_order(self):
        defaults = {
            "cmd_event": 6, "cmd_active": 0, "cmd_wait": 1, "stream_chunk": 0,
            "dm_program": 0, "dm_input_a": 0,
            "dm_input_b": 0, "dm_prefetch_a": 0, "dm_prefetch_b": 0, "dm_output": 0,
            "dm_target_bank": 0, "core_active": 0, "core_wait_data": 0, "uop_active": 0,
            "uop_wait": 0, "sched_wait_reason": 0, "uop_load": 0, "uop_tensor": 0, "uop_store": 0,
            "output_store_enable": 1, "matrix_issue": 0, "matrix_active": 0,
            "acc_clear": 0, "acc_commit": 0, "acc_readout": 0,
        }
        events = []
        for cycle, changes in [
            (1, {"dm_program": 1}),
            (2, {"dm_input_a": 1}),
            (3, {"dm_input_b": 1}),
            (4, {"dm_prefetch_a": 1, "dm_target_bank": 1, "core_active": 1,
                 "uop_active": 1, "uop_load": 1}),
            (5, {"dm_prefetch_a": 1, "dm_target_bank": 1, "uop_active": 1,
                 "matrix_issue": 1}),
            (6, {"dm_prefetch_b": 1, "dm_target_bank": 1, "core_active": 1,
                 "uop_wait": 1, "sched_wait_reason": 1, "matrix_active": 1}),
        ]:
            events.append({"job_id": 1, "cycle": cycle, **defaults, **changes})

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trace.log"
            lines = ["PERF_TRACE " + json.dumps(event) for event in events]
            lines.append(
                'PERF_JOB {"source":"architectural_perf_csr_snapshot","job_id":1,"id":1,'
                '"name":"matmul_k_stream","total_cycles":7,"core":{"total":2,"matmul":1},'
                '"data_mover":{"active_cycles":6,"read_words":24,"write_words":0},"sram":{}}'
            )
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = parse_perf_log(log_path)

        lanes = {lane["module"]: lane for lane in report["jobs"][0]["timeline"]}
        self.assertEqual(
            [span["label"] for span in lanes["Data mover"]["spans"]],
            [
                "Uop program: external SRAM -> instr_mem",
                "Chunk 0 A: external SRAM -> preload bank 0",
                "Chunk 0 B: external SRAM -> preload bank 0",
                "Chunk 1 A prefetch: external SRAM -> preload bank 1",
                "Chunk 1 B prefetch: external SRAM -> preload bank 1",
            ],
        )
        self.assertEqual(
            lanes["Uop scheduler"]["spans"][0]["label"],
            "Fetch/decode/issue BIND A operand bank (encoded UOP_LOAD)",
        )
        self.assertEqual(lanes["Matrix engine"]["spans"][0]["start"], 6)
        self.assertEqual(report["jobs"][0]["timeline_provenance"]["span_placement"], "measured_cycle_event_trace")

    def test_attention_softmax_trace_describes_scheduler_and_primitive_work(self):
        defaults = {
            "cmd_event": 6, "cmd_active": 0, "cmd_wait": 1, "stream_chunk": 0,
            "dm_program": 0, "dm_input_a": 0, "dm_input_b": 0, "dm_prefetch_a": 0,
            "dm_prefetch_b": 0, "dm_output": 0, "dm_target_bank": 0,
            "core_active": 1, "core_wait_data": 0, "uop_active": 0, "uop_wait": 0,
            "sched_wait_reason": 0, "compute_ctrl_event": 0,
            "uop_load": 0, "uop_tensor": 0, "uop_store": 0, "output_store_enable": 1,
            "matrix_issue": 0, "matrix_active": 0, "acc_clear": 0, "acc_commit": 0,
            "acc_readout": 0, "vector_active": 0, "vector_op": 0,
            "reduction_active": 0, "reduction_op": 0, "sfu_active": 0, "sfu_op": 0,
            "primitive_row": 2, "primitive_lane": 0,
        }
        events = [
            {"job_id": 1, "cycle": 0, **defaults, "uop_active": 1, "uop_exec": 1, "uop_opcode": 4,
             "reduction_active": 1, "reduction_op": 0},
            {"job_id": 1, "cycle": 1, **defaults, "vector_active": 1, "vector_op": 1},
            {"job_id": 1, "cycle": 2, **defaults, "sfu_active": 1, "sfu_op": 0, "primitive_lane": 3},
            {"job_id": 1, "cycle": 3, **defaults, "reduction_active": 1, "reduction_op": 1},
            {"job_id": 1, "cycle": 4, **defaults, "sfu_active": 1, "sfu_op": 1},
            {"job_id": 1, "cycle": 5, **defaults, "compute_ctrl_event": 3},
        ]
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trace.log"
            lines = ["PERF_TRACE " + json.dumps(event) for event in events]
            lines.append(
                'PERF_JOB {"source":"architectural_perf_csr_snapshot","job_id":1,"id":1,'
                '"name":"attention_softmax_v1","total_cycles":7,"core":{"total":6,"matmul":0},'
                '"data_mover":{"active_cycles":0,"read_words":0,"write_words":0},"sram":{}}'
            )
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = parse_perf_log(log_path)

        lanes = {lane["module"]: lane for lane in report["jobs"][0]["timeline"]}
        self.assertEqual(
            lanes["Uop scheduler"]["spans"][0]["label"],
            "Fetch/decode/issue reduction max row 0",
        )
        self.assertIn("start/done adapter", lanes["Compute cluster control"]["spans"][0]["label"])
        self.assertEqual(
            lanes["Reduction engine"]["spans"][0]["label"],
            "Reduction max row 2: find stable-softmax row maximum",
        )
        self.assertEqual(
            lanes["SFU"]["spans"][0]["label"],
            "SFU EXP row 2, lane 3: exponentiate one shifted score",
        )

    def test_attention_mask_input_is_not_reported_as_matrix_b(self):
        defaults = {
            "cmd_event": 4, "cmd_active": 1, "cmd_wait": 0, "stream_chunk": 0,
            "dm_program": 0, "dm_input_a": 0, "dm_input_b": 1, "dm_prefetch_a": 0,
            "dm_prefetch_b": 0, "dm_output": 0, "dm_target_bank": 0,
            "core_active": 0, "core_wait_data": 0, "uop_active": 0, "uop_wait": 0,
            "sched_wait_reason": 0, "uop_load": 0, "uop_tensor": 0, "uop_store": 0,
            "output_store_enable": 1, "matrix_issue": 0, "matrix_active": 0,
            "acc_clear": 0, "acc_commit": 0, "acc_readout": 0,
        }
        event = {"job_id": 1, "cycle": 0, **defaults}
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trace.log"
            log_path.write_text(
                "PERF_TRACE " + json.dumps(event) + "\n"
                'PERF_JOB {"source":"architectural_perf_csr_snapshot","job_id":1,"id":1,'
                '"name":"attention_softmax_v1","total_cycles":1,"core":{"total":0,"matmul":0},'
                '"data_mover":{"active_cycles":1,"read_words":2,"write_words":0},"sram":{}}\n',
                encoding="utf-8",
            )
            report = parse_perf_log(log_path)

        lanes = {lane["module"]: lane for lane in report["jobs"][0]["timeline"]}
        self.assertEqual(
            lanes["Command processor"]["spans"][0]["label"],
            "Wait for row-mask table movement",
        )
        self.assertEqual(
            lanes["Data mover"]["spans"][0]["label"],
            "Row-mask table: external SRAM -> core-local mask registers",
        )

    def test_attention_scale_trace_shows_scheduler_issue_wait_and_vector_work(self):
        defaults = {
            "cmd_event": 6, "cmd_active": 0, "cmd_wait": 1, "stream_chunk": 0,
            "dm_program": 0, "dm_input_a": 0, "dm_input_b": 0, "dm_prefetch_a": 0,
            "dm_prefetch_b": 0, "dm_output": 0, "dm_target_bank": 0,
            "core_active": 1, "core_wait_data": 0, "uop_active": 0, "uop_wait": 0,
            "sched_wait_reason": 0, "compute_ctrl_event": 0,
            "uop_load": 0, "uop_tensor": 3, "uop_exec": 0, "uop_opcode": 9,
            "uop_store": 0, "output_store_enable": 1, "matrix_issue": 0,
            "matrix_active": 0, "acc_clear": 0, "acc_commit": 0, "acc_readout": 0,
            "vector_active": 0, "vector_op": 6, "reduction_active": 0,
            "reduction_op": 0, "sfu_active": 0, "sfu_op": 0,
            "primitive_row": 3, "primitive_lane": 0,
        }
        events = [
            {"job_id": 1, "cycle": 0, **defaults, "uop_active": 1, "uop_exec": 1,
             "compute_ctrl_event": 1},
            {"job_id": 1, "cycle": 1, **defaults, "uop_wait": 1,
             "sched_wait_reason": 3, "compute_ctrl_event": 3},
            {"job_id": 1, "cycle": 2, **defaults, "uop_wait": 1,
             "sched_wait_reason": 3, "vector_active": 1},
            {"job_id": 1, "cycle": 3, **defaults, "uop_active": 1,
             "compute_ctrl_event": 2},
            {"job_id": 1, "cycle": 4, **defaults, "core_active": 0, "uop_active": 1, "uop_opcode": 15},
            {"job_id": 1, "cycle": 5, **defaults, "core_active": 0},
        ]
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trace.log"
            lines = ["PERF_TRACE " + json.dumps(event) for event in events]
            lines.append(
                'PERF_JOB {"source":"architectural_perf_csr_snapshot","job_id":1,"id":1,'
                '"name":"attention_scale_mask_v1","total_cycles":7,"core":{"total":4,"matmul":0},'
                '"uop_scheduler":{"active_cycles":2,"wait_cycles":3},'
                '"data_mover":{"active_cycles":0,"read_words":0,"write_words":0},"sram":{}}'
            )
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = parse_perf_log(log_path)

        lanes = {lane["module"]: lane for lane in report["jobs"][0]["timeline"]}
        self.assertEqual(
            lanes["Uop scheduler"]["spans"][0]["label"],
            "Fetch/decode/issue fixed score scale row 3",
        )
        self.assertEqual(
            lanes["Uop scheduler"]["spans"][1]["label"],
            "Wait for issued primitive response",
        )
        self.assertEqual(
            lanes["Vector engine"]["spans"][0]["label"],
            "Vector fixed scale row 3: apply 1/sqrt(head_dim)",
        )

    def test_incomplete_cycle_trace_is_not_used_for_timeline_placement(self):
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trace.log"
            log_path.write_text(
                'PERF_TRACE {"job_id":1,"cycle":0}\n'
                'PERF_JOB {"source":"architectural_perf_csr_snapshot","job_id":1,"id":1,'
                '"name":"attention_scale_mask_v1","total_cycles":80,'
                '"core":{"total":33,"matmul":0},"command_processor":{"active_cycles":12,"wait_cycles":68},'
                '"data_mover":{"active_cycles":33,"read_words":64,"write_words":64,'
                '"program_cycles":1,"initial_input_cycles":16,"output_cycles":16},"sram":{}}\n',
                encoding="utf-8",
            )
            report = parse_perf_log(log_path)

        lanes = {lane["module"]: lane for lane in report["jobs"][0]["timeline"]}
        self.assertTrue(lanes["Data mover"]["spans"])
        self.assertTrue(lanes["Compute cluster"]["spans"])
        self.assertEqual(
            report["jobs"][0]["timeline_provenance"]["span_placement"],
            "derived_from_reviewed_state_machine",
        )

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
        self.assertEqual(len(manifest["jobs"]), 67)
        self.assertEqual(manifest["jobs"][19]["workload"], "real_mnist_cnn_fc1_full_k_stream_layer")
        self.assertEqual(manifest["jobs"][-1]["job_id"], 67)
        self.assertEqual(
            manifest["workload_metadata"]["real_mnist_cnn_fc1_full_k_stream_layer"]["metadata"]["k_chunks"],
            1152,
        )

    def test_firmware_fixture_tool_can_append_transformer_manifest_entries(self):
        tool_path = Path("sw/tools/firmware/emit_soc_cpu_smoke_data.py")
        spec = importlib.util.spec_from_file_location("emit_soc_cpu_smoke_data", tool_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        transformer_micro = generate_transformer_micro_fixtures()
        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            module._write_workload_manifest(
                manifest_path,
                16,
                1,
                1,
                16,
                1152,
                32,
                transformer_micro=transformer_micro,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(manifest["jobs"]), 73)
        self.assertEqual(manifest["manifest_id"], "soc_cpu_smoke_quick_v0")
        self.assertEqual(manifest["jobs"][-6]["workload"], "transformer_prefill_gemm_tiny")
        self.assertEqual(manifest["jobs"][-5]["workload"], "transformer_attention_qk_s8_d8")
        self.assertEqual(manifest["jobs"][-4]["workload"], "transformer_attention_scale_mask_s8_d8")
        self.assertEqual(manifest["jobs"][-3]["workload"], "transformer_attention_softmax_s8")
        self.assertEqual(manifest["jobs"][-2]["workload"], "transformer_attention_pv_s8_d8")
        self.assertEqual(manifest["jobs"][-1]["workload"], "transformer_decode_skinny_gemm_m8_compat")
        self.assertEqual(
            manifest["workload_metadata"]["transformer_prefill_gemm_tiny"]["metadata"]["scenario"],
            "transformer_prefill",
        )
        self.assertEqual(
            manifest["workload_metadata"]["transformer_attention_qk_s8_d8"]["metadata"]["attention_stage"],
            "qk",
        )
        self.assertEqual(
            manifest["workload_metadata"]["transformer_attention_scale_mask_s8_d8"]["metadata"]["attention_stage"],
            "scale_mask",
        )
        self.assertEqual(
            manifest["workload_metadata"]["transformer_attention_softmax_s8"]["metadata"]["attention_stage"],
            "softmax",
        )
        self.assertEqual(
            manifest["workload_metadata"]["transformer_attention_pv_s8_d8"]["metadata"]["attention_stage"],
            "pv",
        )
        self.assertTrue(
            manifest["workload_metadata"]["transformer_kv_cache_traffic_tiny"]["metadata"]["model_only"]
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

    def test_manifest_preserves_transformer_metadata_and_model_only_workloads(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "perf.log"
            manifest_path = tmp_path / "manifest.json"
            log_path.write_text(
                _perf_job_line(1, "matmul_k_stream", 128, matmul_cycles=20).replace(
                    '{"id":1', '{"job_id":1,"id":1'
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "npu_workload_manifest_v0",
                        "manifest_id": "transformer_unit_v0",
                        "run_name": "unit",
                        "workload_metadata": {
                            "transformer_prefill_gemm_tiny": {
                                "kind": "transformer_micro",
                                "metadata": {
                                    "scenario": "transformer_prefill",
                                    "logical_shape": {"m": 8, "n": 8, "k": 16},
                                    "external_memory": {"weight_read_bytes": 128},
                                },
                            },
                            "transformer_kv_cache_traffic_tiny": {
                                "kind": "transformer_model_only",
                                "metadata": {
                                    "scenario": "transformer_decode",
                                    "model_only": True,
                                    "external_memory": {"kv_cache_read_bytes": 1024},
                                },
                            },
                        },
                        "jobs": [
                            {
                                "job_id": 1,
                                "workload": "transformer_prefill_gemm_tiny",
                                "op": "matmul_k_stream",
                                "role": "transformer_micro",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = parse_perf_log(log_path, manifest_path)

        self.assertEqual(report["workloads"][0]["metadata"]["scenario"], "transformer_prefill")
        self.assertEqual(report["workloads"][0]["transformer_metrics"]["effective_mac_ops"], 1024)
        self.assertEqual(report["workloads"][0]["transformer_metrics"]["matrix_utilization"], 0.8)
        self.assertEqual(report["workloads"][0]["transformer_metrics"]["skinny_gemm_utilization"], 0.8)
        self.assertEqual(report["workloads"][0]["transformer_metrics"]["theoretical_compute_cycles"], 16)
        self.assertEqual(report["workloads"][0]["transformer_metrics"]["measured_compute_cycles"], 20)
        self.assertEqual(report["workloads"][0]["transformer_metrics"]["compute_overhead_cycles"], 4)
        self.assertEqual(report["workloads"][0]["transformer_metrics"]["compute_efficiency"], 0.8)
        self.assertEqual(
            report["workloads"][0]["transformer_metrics"]["measured_compute_provenance"],
            "measured_matrix_active_cycles",
        )
        self.assertEqual(report["model_only_workloads"][0]["name"], "transformer_kv_cache_traffic_tiny")
        self.assertEqual(report["model_only_workloads"][0]["metadata"]["external_memory"]["kv_cache_read_bytes"], 1024)
        self.assertIsNone(report["model_only_workloads"][0]["transformer_metrics"]["matrix_utilization"])
        self.assertEqual(report["summary"]["transformer"]["kv_read_bytes"], 1024)

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
            self.assertEqual(report["jobs"][0]["timeline"][2]["module"], "NPU core")
            self.assertEqual(report["jobs"][0]["timeline"][3]["module"], "Command processor")
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["schema"], "npu_perf_report_v0")
            self.assertIn("timeline", json_path.read_text(encoding="utf-8"))
            self.assertIn("movement", json_path.read_text(encoding="utf-8"))
            self.assertIn("NPU Cycle Report", html_path.read_text(encoding="utf-8"))
            self.assertIn("Workload Summary", html_path.read_text(encoding="utf-8"))
            self.assertIn("Cycle timeline", html_path.read_text(encoding="utf-8"))
            self.assertIn("Movement model", html_path.read_text(encoding="utf-8"))
            self.assertIn("renderPhaseTimeline(section", html_path.read_text(encoding="utf-8"))
            self.assertIn('document.createElement("details")', html_path.read_text(encoding="utf-8"))
            self.assertIn('span.kind === "work"', html_path.read_text(encoding="utf-8"))
            self.assertIn('span.kind === "wait"', html_path.read_text(encoding="utf-8"))
            self.assertIn("laneData.depth", html_path.read_text(encoding="utf-8"))
            self.assertIn("laneData.role", html_path.read_text(encoding="utf-8"))
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
            self.assertEqual(report["summary"]["workloads"], 2)
            self.assertEqual(report["summary"]["total_cycles"], 4065)
            classifier = report["workloads"][1]
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
            self.assertEqual(report["summary"]["workloads"], 3)
            fc2 = report["workloads"][2]
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
            self.assertEqual(report["summary"]["workloads"], 4)
            fc1 = report["workloads"][2]
            fc2 = report["workloads"][3]
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
            self.assertEqual(report["summary"]["workloads"], 5)
            fc1_tile = report["workloads"][2]
            fc1_stream = report["workloads"][3]
            fc2 = report["workloads"][4]
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
            self.assertEqual(report["summary"]["workloads"], 6)
            fc1_tile = report["workloads"][2]
            fc1_smoke = report["workloads"][3]
            fc1_full = report["workloads"][4]
            fc2 = report["workloads"][5]
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
            self.assertFalse(
                any("overlap" in span["label"].lower() for span in data_mover_lane["spans"])
            )
            self.assertIsNone(highlight["overlap_cycles"])
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
