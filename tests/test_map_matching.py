"""Behavior tests for directed HMM map matching."""

from __future__ import annotations

import unittest

import networkx as nx

from stgat_lstm.geometry import GeoPoint
from stgat_lstm.map_matching import HiddenMarkovMatcher, TimedObservation
from stgat_lstm.osm_audit import RoadEdgeIndex


class HiddenMarkovMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = nx.MultiDiGraph(crs="EPSG:32651")
        self.graph.add_node("a", x=283_900.0, y=1_610_000.0)
        self.graph.add_node("b", x=284_000.0, y=1_610_000.0)
        self.graph.add_node("c", x=284_100.0, y=1_610_000.0)
        self.graph.add_edge("a", "b", key="ab", length=100.0, geometry="LINESTRING (283900 1610000, 284000 1610000)")
        self.graph.add_edge("b", "c", key="bc", length=100.0, geometry="LINESTRING (284000 1610000, 284100 1610000)")
        self.index = RoadEdgeIndex(self.graph)

    def observation(self, x: float, y: float, timestamp_ms: int) -> TimedObservation:
        longitude, latitude = self.index.transformer.transform(x, y, direction="INVERSE")
        return TimedObservation(GeoPoint(latitude, longitude), timestamp_ms)

    def test_connected_directed_edges_are_decoded(self) -> None:
        matcher = HiddenMarkovMatcher(self.graph, self.index, search_radius_m=30.0)
        result = matcher.match(
            [self.observation(283_920.0, 1_610_005.0, 0), self.observation(284_080.0, 1_610_004.0, 10_000)]
        )
        self.assertEqual(result.traversed_edge_ids, (("a", "b", "ab"), ("b", "c", "bc")))
        self.assertLess(result.mean_observation_distance_m, 6.0)

    def test_observation_outside_search_radius_is_rejected(self) -> None:
        matcher = HiddenMarkovMatcher(self.graph, self.index, search_radius_m=20.0)
        with self.assertRaisesRegex(ValueError, "No OSM edge candidate"):
            matcher.match(
                [self.observation(283_920.0, 1_610_100.0, 0), self.observation(284_080.0, 1_610_100.0, 10_000)]
            )

    def test_timestamps_must_increase(self) -> None:
        matcher = HiddenMarkovMatcher(self.graph, self.index)
        with self.assertRaisesRegex(ValueError, "timestamps must increase"):
            matcher.match(
                [self.observation(283_920.0, 1_610_000.0, 10_000), self.observation(283_950.0, 1_610_000.0, 10_000)]
            )


if __name__ == "__main__":
    unittest.main()
