"""Tests that only reviewed examples can enter real training."""

import json
import tempfile
import unittest
from pathlib import Path

from graph_test_data import build_test_graph_data
from stgat_lstm.preprocessing.graph_data import (load_approved_decisions, load_approved_preferences,
                                   path_choice_edge_ids, RealDecisionExample,
                                   training_graph_buffer_m, build_real_graph_data_from_graph,
                                   NODE_CONTEXT_COLUMNS)
from stgat_lstm.modeling.train_model import decision_choice_targets


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

    def test_deviation_target_names_observed_outgoing_road(self) -> None:
        decision = RealDecisionExample(
            "example", "rider", "ride", 1, "1", "4", 1,
            ("1|2|0", "2|4|0"),
            ("1|2|0", "2|3|0", "3|4|0"),
        )
        target = decision_choice_targets(decision, self.data)[0]
        self.assertEqual(target.source_node_id, "2")
        self.assertEqual(set(target.candidate_edge_ids), {"2|3|0", "2|4|0"})
        self.assertEqual(target.suggested_edge_id, "2|4|0")
        self.assertEqual(target.chosen_edge_id, "2|3|0")
        self.assertEqual(target.deviated, 1)

    def test_explicit_outgoing_choice_does_not_require_common_path_endpoints(self) -> None:
        decision = RealDecisionExample(
            "local", "rider", "ride", 1, "2", "6", 1,
            ("2|4|0",), ("2|3|0",),
            suggested_choice_edge_id="2|4|0", chosen_choice_edge_id="2|3|0",
            has_route_pair=False,
        )
        targets = decision_choice_targets(decision, self.data)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].chosen_edge_id, "2|3|0")
        self.assertEqual(targets[0].suggested_edge_id, "2|4|0")

    def test_model_crop_expands_to_cover_retained_roads_and_destination(self) -> None:
        graph = self.data.graph.copy()
        graph.add_node("far", x=3000.0, y=0.0)
        graph.add_edge("3", "far", key="0", length=2900.0,
                       geometry="LINESTRING (100 100, 3000 0)")
        document = {"status": "approved_for_training", "examples": [{
            "status": "approved",
            "routing_context": {"origin_node_id": "2", "destination_node_id": "far"},
            "suggested_path": {"edges": [{"u": "2", "v": "4", "key": "0"}]},
            "observed_path_label_evidence": {"edges": [
                {"u": "2", "v": "3", "key": "0"}, {"u": "3", "v": "far", "key": "0"}]},
        }]}
        buffer = training_graph_buffer_m(graph, document)
        rows = {node: {column: 1.0 for column in NODE_CONTEXT_COLUMNS} for node in graph}
        expanded = build_real_graph_data_from_graph(graph, rows, corridor_buffer_m=buffer)
        self.assertGreater(buffer, 1000)
        self.assertIn("3|far|0", expanded.edge_ids)
        self.assertIn("far", expanded.node_ids)
        self.assertEqual(expanded.graph.graph["model_corridor_buffer_m"], buffer)


if __name__ == "__main__":
    unittest.main()
