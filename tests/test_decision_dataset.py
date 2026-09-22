"""Tests for local follow-versus-deviate decision construction."""

from __future__ import annotations

import unittest

import networkx as nx

from stgat_lstm.build_decision_dataset import _agreement, _follow_choice_window
from stgat_lstm.geometry import GeoPoint
from stgat_lstm.map_matching import MapMatchResult, MatchState, TimedObservation


def match(edges) -> MapMatchResult:
    states = tuple(MatchState(edge, 1.0, 1.0, 10.0) for edge in edges)
    return MapMatchResult(states, tuple(edges), tuple(range(len(edges))), -1.0, 1.0, 2.0)


class DecisionDatasetTests(unittest.TestCase):
    def test_follow_window_begins_at_a_real_choice_and_uses_pre_choice_time(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge("a", "b", key="ab", length=10)
        graph.add_edge("a", "x", key="ax", length=10)
        graph.add_edge("b", "c", key="bc", length=10)
        graph.add_edge("c", "d", key="cd", length=10)
        edges = (("a", "b", "ab"), ("b", "c", "bc"), ("c", "d", "cd"))
        result = match(edges)
        gps = [
            TimedObservation(GeoPoint(14.55, 120.99), 1000),
            TimedObservation(GeoPoint(14.55, 120.99), 2000),
            TimedObservation(GeoPoint(14.55, 120.99), 3000),
        ]

        choice = _follow_choice_window(graph, result, result, gps, window_edges=3)

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice[0], edges)
        self.assertEqual(choice[1], 1000)
        self.assertEqual(choice[2:], ("a", "d"))

    def test_agreement_reports_both_path_directions(self) -> None:
        observed = match((("a", "b", "ab"), ("b", "c", "bc")))
        suggested = match((("a", "b", "ab"), ("b", "c", "bc"), ("c", "d", "cd")))

        quality = _agreement(observed, suggested)

        self.assertEqual(quality["observed_edge_agreement"], 1.0)
        self.assertAlmostEqual(quality["suggested_edge_coverage"], 2 / 3)
        self.assertAlmostEqual(quality["edge_jaccard"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
