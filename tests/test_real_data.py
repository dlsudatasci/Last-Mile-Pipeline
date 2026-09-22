"""Tests for the OSM-to-model feature contract and training approval guard."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import networkx as nx
import torch

from stgat_lstm.model import DecisionPreferenceModel, PreferenceModel
from stgat_lstm.mapbox_traffic import build_edge_snapshot
from stgat_lstm.osm_audit import EdgeMatch
from stgat_lstm.real_data import (
    NODE_CONTEXT_COLUMNS,
    build_real_graph_data_from_graph,
    load_approved_decisions,
    load_approved_preferences,
)
from stgat_lstm.real_routing import shortest_real_preference_path


class RealDataTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = nx.MultiDiGraph(crs="EPSG:32651")
        graph.add_node("1", x=0.0, y=0.0, highway="traffic_signals")
        graph.add_node("2", x=100.0, y=0.0)
        graph.add_node("3", x=100.0, y=100.0)
        graph.add_node("4", x=200.0, y=0.0)
        graph.add_node("far", x=5_000.0, y=5_000.0)
        graph.add_edge("1", "2", key="0", name="Taft Avenue", highway="primary", length=100.0, oneway="True", geometry="LINESTRING (0 0, 100 0)")
        graph.add_edge("2", "4", key="0", name="Main Road", highway="secondary", length=100.0, geometry="LINESTRING (100 0, 200 0)")
        graph.add_edge("2", "3", key="0", name="Side Road", highway="residential", length=100.0, geometry="LINESTRING (100 0, 100 100)")
        graph.add_edge("3", "4", key="0", name="Return Road", highway="residential", length=141.4, geometry="LINESTRING (100 100, 200 0)")
        graph.add_edge("far", "far", key="0", name="Far Road", highway="service", length=1.0, geometry="LINESTRING (5000 5000, 5001 5000)")
        rows = {
            node_id: {column: float(index + 1) for index, column in enumerate(NODE_CONTEXT_COLUMNS)}
            for node_id in ("1", "2", "3", "4", "far")
        }
        self.data = build_real_graph_data_from_graph(graph, rows, corridor_buffer_m=500.0)

    def test_builds_variable_dimension_model_input(self) -> None:
        self.assertNotIn("far", self.data.node_ids)
        inputs = self.data.build_model_input("4")
        self.assertEqual(inputs.node_features.shape[1], len(self.data.schema.node))
        self.assertEqual(inputs.edge_static.shape[1], len(self.data.schema.edge_static))
        age_index = self.data.schema.edge_dynamic.index("traffic_age_scaled")
        self.assertTrue(torch.all(inputs.temporal_edge_features[0][:, age_index] == 1))
        model = PreferenceModel(**self.data.model_configuration())
        scores = model(inputs)
        self.assertEqual(scores.shape, (len(self.data.edge_ids),))
        self.assertTrue(torch.all(scores > 0))

    def test_review_required_artifact_cannot_enter_training(self) -> None:
        document = {"status": "review_required_not_training_ready", "candidates": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not been approved"):
                load_approved_preferences(path, self.data)

    def test_real_router_uses_canonical_directed_edges(self) -> None:
        costs = {edge_id: 1.0 for edge_id in self.data.edge_ids}
        route = shortest_real_preference_path(self.data, costs, "1", "4")
        self.assertEqual(route.edge_ids, ("1|2|0", "2|4|0"))
        self.assertEqual(route.total_preference_cost, 2.0)

    def test_multitask_model_scores_edges_and_deviation_probability(self) -> None:
        inputs = self.data.build_model_input("4")
        model = DecisionPreferenceModel(**self.data.model_configuration())
        costs, logit = model(inputs, ("1|2|0", "2|4|0"))
        self.assertEqual(costs.shape, (len(self.data.edge_ids),))
        self.assertEqual(logit.shape, ())
        (costs.mean() + logit).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_unapproved_decisions_cannot_enter_training(self) -> None:
        document = {"status": "review_required_not_training_ready", "examples": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not been approved"):
                load_approved_decisions(path, self.data)

    def test_mapbox_speed_and_sparse_congestion_are_separate_features(self) -> None:
        observation = {
            "requested_at": "1970-01-01T00:00:00Z",
            "route": {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[120.99, 14.55], [120.991, 14.551], [120.992, 14.552]],
                },
                "legs": [
                    {
                        "annotation": {
                            "speed": [5.0, 15.0],
                            "congestion_numeric": [None, 40],
                            "congestion": ["unknown", "low"],
                            "duration": [10.0, 10.0],
                        }
                    }
                ],
            },
        }
        fake_index = Mock()
        fake_index.nearest.return_value = EdgeMatch("1", "2", "0", 2.0, "Taft Avenue")
        with patch("stgat_lstm.mapbox_traffic.RoadEdgeIndex", return_value=fake_index):
            snapshot = build_edge_snapshot(self.data, observation, now_epoch_ms=1_800_000)

        row = self.data.edge_ids.index("1|2|0")
        columns = {name: index for index, name in enumerate(self.data.schema.edge_dynamic)}
        self.assertAlmostEqual(float(snapshot[row, columns["congestion_normalized"]]), 0.4)
        self.assertAlmostEqual(float(snapshot[row, columns["speed_ratio_to_reference"]]), 0.9, places=4)
        self.assertEqual(float(snapshot[row, columns["congestion_observed"]]), 1.0)
        self.assertEqual(float(snapshot[row, columns["speed_observed"]]), 1.0)
        self.assertAlmostEqual(float(snapshot[row, columns["traffic_age_scaled"]]), 0.5)
        unobserved = self.data.edge_ids.index("2|4|0")
        self.assertEqual(float(snapshot[unobserved, columns["speed_observed"]]), 0.0)
        self.assertEqual(float(snapshot[unobserved, columns["traffic_age_scaled"]]), 1.0)


if __name__ == "__main__":
    unittest.main()
