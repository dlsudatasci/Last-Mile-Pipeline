"""Tests that only reviewed examples can enter real training."""

import json
import tempfile
import unittest
from pathlib import Path

from graph_test_data import build_test_graph_data
from stgat_lstm.graph_data import (load_approved_decisions, load_approved_preferences,
                                   path_choice_edge_ids)


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = build_test_graph_data()

    def test_review_required_preferences_cannot_enter_training(self) -> None:
        document = {"status": "review_required_not_training_ready", "candidates": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not been approved"):
                load_approved_preferences(path, self.data)

    def test_unapproved_decisions_cannot_enter_training(self) -> None:
        document = {"status": "review_required_not_training_ready", "examples": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not been approved"):
                load_approved_decisions(path, self.data)

    def test_only_roads_with_an_accessible_branch_become_choice_targets(self) -> None:
        choices = path_choice_edge_ids(self.data, ("1|2|0", "2|4|0"))
        self.assertEqual(choices, ("2|4|0",))


if __name__ == "__main__":
    unittest.main()
