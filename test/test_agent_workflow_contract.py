from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AgentWorkflowContractTests(unittest.TestCase):
    def test_root_agent_contract_preserves_hardware_first_rules(self):
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        for required in (
            "North Star",
            "Design before code",
            "Spec first",
            "Hardware first",
            "Representative functionality before local optimization",
            "Do not hide executable-workload stages in CPU or fixture preprocessing",
            "Tiling belongs to the Compiler/planner",
            "make check-workflow",
            "candidate-versus-retained-baseline PPA evidence",
        ):
            self.assertIn(required, text)

    def test_make_test_runs_workflow_gate_first(self):
        text = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("check-workflow:", text)
        self.assertIn("test: check-workflow", text)

    def test_workflow_checker_exists(self):
        self.assertTrue((ROOT / "scripts/check_workflow.py").is_file())


if __name__ == "__main__":
    unittest.main()
