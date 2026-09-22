"""Tests for projected OSM edge indexing."""

from __future__ import annotations

import unittest

import networkx as nx

from stgat_lstm.geometry import GeoPoint
from stgat_lstm.osm_audit import RoadEdgeIndex


class OsmEdgeIndexTests(unittest.TestCase):
    def test_nearest_edge_uses_graph_crs_and_preserves_direction(self) -> None:
        graph = nx.MultiDiGraph(crs="EPSG:32651")
        graph.add_node("west", x=283_900.0, y=1_610_000.0)
        graph.add_node("east", x=284_100.0, y=1_610_000.0)
        graph.add_edge(
            "west",
            "east",
            key="forward",
            name="Test Road",
            geometry="LINESTRING (283900 1610000, 284100 1610000)",
        )
        index = RoadEdgeIndex(graph)
        longitude, latitude = index.transformer.transform(284_000.0, 1_610_010.0, direction="INVERSE")
        match = index.nearest(GeoPoint(latitude, longitude))
        self.assertEqual((match.u, match.v, match.key, match.name), ("west", "east", "forward", "Test Road"))
        self.assertAlmostEqual(match.distance_m, 10.0, delta=0.1)

    def test_missing_crs_is_rejected(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge("a", "b")
        with self.assertRaisesRegex(ValueError, "CRS"):
            RoadEdgeIndex(graph)


if __name__ == "__main__":
    unittest.main()
