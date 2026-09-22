"""Tests for common-origin/common-destination pair extraction."""

from __future__ import annotations

import unittest

from stgat_lstm.map_matching import MapMatchResult, MatchState
from stgat_lstm.preference_pairs import extract_divergence_pair


def result(edges, offsets) -> MapMatchResult:
    states = tuple(MatchState(edges[offset], 1.0, 1.0, 10.0) for offset in offsets)
    return MapMatchResult(states, tuple(edges), tuple(offsets), -1.0, 1.0, 1.0)


class PreferencePairTests(unittest.TestCase):
    def test_extracts_distinct_paths_with_common_od(self) -> None:
        before = ("a", "b", "ab")
        after = ("e", "f", "ef")
        observed = result(
            [before, ("b", "c", "bc"), ("c", "e", "ce"), after],
            [0, 1, 2, 3],
        )
        rejected = result(
            [before, ("b", "d", "bd"), ("d", "e", "de"), after],
            [0, 1, 2, 3],
        )
        pair = extract_divergence_pair(observed, rejected, event_state_index=1)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual((pair.origin_node_id, pair.destination_node_id), ("b", "e"))
        self.assertEqual(pair.preferred_edge_ids, (("b", "c", "bc"), ("c", "e", "ce")))
        self.assertEqual(pair.rejected_edge_ids, (("b", "d", "bd"), ("d", "e", "de")))

    def test_requires_a_rejoin_after_the_event(self) -> None:
        observed = result([("a", "b", "ab"), ("b", "c", "bc")], [0, 1])
        rejected = result([("a", "b", "ab"), ("b", "d", "bd")], [0, 1])
        self.assertIsNone(extract_divergence_pair(observed, rejected, event_state_index=1))

    def test_rejects_invalid_event_index(self) -> None:
        matched = result([("a", "b", "ab"), ("b", "c", "bc")], [0, 1])
        with self.assertRaises(IndexError):
            extract_divergence_pair(matched, matched, event_state_index=2)


if __name__ == "__main__":
    unittest.main()
