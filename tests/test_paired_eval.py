from dataclasses import dataclass
import unittest

from email_agent.paired_eval import (
    paired_scenario_rows,
    paired_summary,
    parse_model_ref,
    select_scenarios,
)


@dataclass
class Scenario:
    id: int


class ParseModelRefTest(unittest.TestCase):
    def test_parses_serverless_artifact_ref(self):
        ref = parse_model_ref(
            "wandb-artifact:///wandb/email-search-agent/email-agent-smoke-r1:step2"
        )
        self.assertEqual(ref.entity, "wandb")
        self.assertEqual(ref.project, "email-search-agent")
        self.assertEqual(ref.collection, "email-agent-smoke-r1")
        self.assertEqual(ref.alias, "step2")
        self.assertEqual(
            ref.artifact_path,
            "wandb/email-search-agent/email-agent-smoke-r1:step2",
        )

    def test_rejects_unpinned_ref(self):
        with self.assertRaises(ValueError):
            parse_model_ref("email-agent-smoke-r1")


class SelectScenariosTest(unittest.TestCase):
    def test_selection_is_deterministic_without_mutating_input(self):
        scenarios = [Scenario(i) for i in range(10)]
        original = list(scenarios)
        first = select_scenarios(scenarios, limit=4, seed=42)
        second = select_scenarios(scenarios, limit=4, seed=42)
        self.assertEqual([s.id for s in first], [s.id for s in second])
        self.assertEqual(scenarios, original)

    def test_explicit_manifest_preserves_order(self):
        scenarios = [Scenario(i) for i in range(5)]
        selected = select_scenarios(
            scenarios, limit=3, seed=999, explicit_ids=[4, 1, 3]
        )
        self.assertEqual([s.id for s in selected], [4, 1, 3])

    def test_rejects_missing_manifest_id(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            select_scenarios(
                [Scenario(1)], limit=1, seed=42, explicit_ids=[2]
            )


class PairedSummaryTest(unittest.TestCase):
    def test_computes_paired_delta_and_failure_rates(self):
        rows = [
            # Scenario 1: treatment improves from 0/2 to 2/2.
            {"scenario_id": 1, "model_label": "step0", "correct": 0, "empty_answer": True},
            {"scenario_id": 1, "model_label": "step0", "correct": 0, "empty_answer": True},
            {"scenario_id": 1, "model_label": "step2", "correct": 1, "empty_answer": False},
            {"scenario_id": 1, "model_label": "step2", "correct": 1, "empty_answer": False},
            # Scenario 2: both checkpoints score 1/2.
            {"scenario_id": 2, "model_label": "step0", "correct": 1},
            {"scenario_id": 2, "model_label": "step0", "correct": 0},
            {"scenario_id": 2, "model_label": "step2", "correct": 1},
            {"scenario_id": 2, "model_label": "step2", "correct": 0},
        ]
        summary = paired_summary(rows, bootstrap_samples=200, bootstrap_seed=7)
        self.assertEqual(summary["scenario_count"], 2)
        self.assertEqual(summary["control_trials"], 4)
        self.assertEqual(summary["treatment_trials"], 4)
        self.assertAlmostEqual(summary["control_accuracy"], 0.25)
        self.assertAlmostEqual(summary["treatment_accuracy"], 0.75)
        self.assertAlmostEqual(summary["accuracy_delta"], 0.5)
        self.assertAlmostEqual(summary["control_empty_answer_rate"], 0.5)
        self.assertAlmostEqual(summary["treatment_empty_answer_rate"], 0.0)

    def test_rejects_unpaired_scenario(self):
        with self.assertRaisesRegex(ValueError, "missing a paired"):
            paired_summary([
                {"scenario_id": 1, "model_label": "step0", "correct": 1},
                {"scenario_id": 2, "model_label": "step2", "correct": 1},
            ])

    def test_reports_per_scenario_delta(self):
        rows = [
            {"scenario_id": 9, "question": "Q", "model_label": "step0", "correct": 0, "empty_answer": 1, "exception": 0, "tool_calls": 2},
            {"scenario_id": 9, "question": "Q", "model_label": "step2", "correct": 1, "empty_answer": 0, "exception": 0, "tool_calls": 3},
        ]
        result = paired_scenario_rows(rows)
        self.assertEqual(result[0]["scenario_id"], 9)
        self.assertEqual(result[0]["accuracy_delta"], 1.0)
        self.assertEqual(result[0]["treatment_mean_tool_calls"], 3.0)


if __name__ == "__main__":
    unittest.main()