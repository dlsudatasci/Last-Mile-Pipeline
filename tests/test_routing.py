"""Tests for directed routing with learned edge costs."""

import unittest
from unittest.mock import Mock

import torch
from pyproj import Transformer

from graph_test_data import build_test_graph_data
from stgat_lstm.audit_road_network import RoadEdgeIndex
from stgat_lstm.mapbox_traffic import match_traffic_segment
from stgat_lstm.predict_route import route_request, shortest_real_preference_path, snap_coordinate


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = build_test_graph_data()

    def test_real_router_uses_canonical_directed_edges(self) -> None:
        costs = {edge_id: 1.0 for edge_id in self.data.edge_ids}
        route = shortest_real_preference_path(self.data, costs, "1", "4")
        self.assertEqual(route.edge_ids, ("1|2|0", "2|4|0"))
        self.assertEqual(route.total_preference_cost, 2.0)

    def test_new_route_uses_learned_costs_without_approved_examples(self) -> None:
        scores = torch.tensor([
            1.0 if edge_id in ("1|2|0", "2|3|0", "3|4|0") else 100.0
            for edge_id in self.data.edge_ids
        ])
        model = Mock(return_value=(scores, torch.zeros(len(self.data.edge_ids))))
        result = route_request(model, self.data, "1", "4")
        self.assertEqual(result["distance_baseline"]["edge_ids"], ["1|2|0", "2|4|0"])
        self.assertEqual(result["recommended_route"]["edge_ids"], ["1|2|0", "2|3|0", "3|4|0"])
        self.assertGreater(result["recommended_route"]["distance_m"], result["distance_baseline"]["distance_m"])
        self.assertEqual(result["recommended_route"]["geometry"]["type"], "MultiLineString")
        road_predictions = result["recommended_route"]["road_deviation_predictions"]
        self.assertEqual(len(road_predictions), 3)
        choices = [item for item in road_predictions if item["is_decision_point"]]
        self.assertTrue(choices)
        self.assertAlmostEqual(choices[0]["deviation_probability"], 0.5)
        with self.assertRaisesRegex(ValueError, "same node"):
            route_request(model, self.data, "1", "1")
        with self.assertRaisesRegex(ValueError, "No routable path"):
            route_request(model, self.data, "4", "1")

    def test_coordinate_snapping_rejects_far_points(self) -> None:
        transform = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        longitude, latitude = transform.transform(0, 0)
        node, distance = snap_coordinate(self.data, longitude, latitude)
        self.assertEqual(node, "1")
        self.assertLess(distance, 0.01)
        longitude, latitude = transform.transform(10_000, 10_000)
        with self.assertRaisesRegex(ValueError, "closer"):
            snap_coordinate(self.data, longitude, latitude)

    def test_traffic_matching_respects_direction_on_coincident_edges(self) -> None:
        graph = self.data.graph.copy()
        graph.add_edge("2", "1", key="0", name="Taft Avenue", highway="primary", length=100.0,
                       geometry="LINESTRING (0 0, 100 0)")
        index = RoadEdgeIndex(graph)
        transform = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        left, right = transform.transform(10, 0), transform.transform(90, 0)
        forward = match_traffic_segment(index, self.data, left, right, 30)
        reverse = match_traffic_segment(index, self.data, right, left, 30)
        self.assertEqual((forward.u, forward.v), ("1", "2"))
        self.assertEqual((reverse.u, reverse.v), ("2", "1"))


if __name__ == "__main__":
    unittest.main()
