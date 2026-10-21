"""Tests for local follow-versus-deviate decision construction."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import networkx as nx
from shapely.geometry import LineString

from stgat_lstm.preprocessing.build_decision_candidates import (_agreement, _follow_choice_window, build_decision_candidates,
                                                refine_real_decisions)
from stgat_lstm.preprocessing.build_deviation_candidates import build_candidates
from stgat_lstm.preprocessing.coordinates import Transformer
from stgat_lstm.preprocessing.geometry import GeoPoint
from stgat_lstm.preprocessing.map_matching import MapMatchResult, MatchState, TimedObservation
from stgat_lstm.preprocessing.preference_pairs import (extract_outgoing_choice, extract_divergence_pair,
                                       infer_recorded_road_choice, parse_navigation_instruction)
from stgat_lstm.preprocessing.audit_road_network import RoadEdgeIndex
from stgat_lstm.preprocessing.review_decisions import apply_automatic_rules


def match(edges) -> MapMatchResult:
    states = tuple(MatchState(edge, 1.0, 1.0, 10.0) for edge in edges)
    return MapMatchResult(states, tuple(edges), tuple(range(len(edges))), -1.0, 1.0, 2.0)


class DecisionDatasetTests(unittest.TestCase):
    def _instruction_choice(self, *, instruction="Drive east.", ambiguous=False,
                            restricted=False, reverse=False, report_ms=6000, integer_keys=False):
        graph = nx.MultiDiGraph(crs="EPSG:32651")
        positions = {"a": (283900, 1610000), "b": (284000, 1610000),
                     "c": (284000, 1610100), "d": (284100, 1610000),
                     "e": (284100, 1610000)}
        for node, (x, y) in positions.items():
            graph.add_node(node, x=x, y=y)
        for u, v, name in (("a", "b", "Approach Street"), ("b", "c", "North Avenue"),
                           ("b", "d", "East Street")):
            graph.add_edge(u, v, key=0 if integer_keys else "0", name=name, length=100,
                           access="private" if restricted and v == "d" else "yes",
                           geometry=LineString([positions[u], positions[v]]).wkt)
        if ambiguous:
            graph.add_edge("b", "e", key="0", name="Another East Road", length=100,
                           geometry=LineString([positions["b"], positions["e"]]).wkt)
        inverse = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        locations = [(283990, 1610000), (284010, 1610000), (284025, 1610000)]
        if reverse:
            locations = [(283990, 1610000), (284025, 1610000), (284010, 1610000)]
        gps = []
        for (x, y), stamp in zip(locations, (1000, 3000, 5000)):
            lon, lat = inverse.transform(x, y)
            gps.append(TimedObservation(GeoPoint(lat, lon), stamp))
        diagnostics = {}
        choice = infer_recorded_road_choice(
            graph, RoadEdgeIndex(graph), match((("a", "b", "0"), ("b", "c", "0"))),
            gps, report_ms, gps[-1].point, original_instruction="Turn left onto North Avenue. in 100 m",
            deviated_instruction=instruction, earliest_choice_timestamp_ms=0, diagnostics=diagnostics)
        return choice, diagnostics

    def test_navigation_parser_preserves_future_maneuver_distance(self) -> None:
        self.assertEqual(parse_navigation_instruction("Turn right onto Quirino Avenue/140. in 330 m"),
                         {"road_name": "Quirino Avenue/140", "bearing_degrees": None,
                          "maneuver_distance_m": 330})
        self.assertEqual(parse_navigation_instruction("Drive north-east.")["bearing_degrees"], 45)
        self.assertIsNone(parse_navigation_instruction("Your destination is on the right. in 60 m")["road_name"])

    def test_recorded_heading_resolves_a_unique_geometry_supported_outgoing_road(self) -> None:
        choice, diagnostics = self._instruction_choice()
        self.assertEqual(choice.chosen_edge_id, ("b", "d", "0"))
        self.assertEqual(choice.suggested_edge_id, ("b", "c", "0"))
        self.assertEqual(choice.decision_timestamp_ms, 1000)
        self.assertEqual(diagnostics["status"], "resolved_unique_choice")
        self.assertTrue(diagnostics["original_named_maneuver_on_active_route"])

    def test_ambiguous_heading_is_not_broken_by_graph_iteration_order(self) -> None:
        choice, diagnostics = self._instruction_choice(ambiguous=True)
        self.assertIsNone(choice)
        self.assertEqual(diagnostics["status"], "ambiguous_choices")
        choice, diagnostics = self._instruction_choice(ambiguous=True, instruction="Drive east on East St.")
        self.assertEqual(choice.chosen_edge_id, ("b", "d", "0"))

    def test_navigation_inference_rejects_conflicts_and_noncausal_or_reverse_movement(self) -> None:
        for kwargs in ({"instruction": "Drive east on Missing Street."}, {"instruction": "Drive north."},
                       {"restricted": True}, {"reverse": True}, {"report_ms": 500}):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(self._instruction_choice(**kwargs)[0])

    def test_navigation_inference_handles_integer_graphml_edge_keys(self) -> None:
        self.assertEqual(self._instruction_choice(integer_keys=True)[0].chosen_edge_id, ("b", "d", "0"))

    def test_late_report_can_use_a_turn_from_the_same_active_route_period(self) -> None:
        approach, expected, chosen = ("a", "b", "0"), ("b", "c", "0"), ("b", "d", "0")
        observed = match((approach, chosen, chosen))
        gps = [TimedObservation(GeoPoint(14.55, 120.99), stamp) for stamp in (1000, 2000, 4000)]
        diagnostics = {}
        self.assertIsNone(extract_outgoing_choice(observed, match((approach, expected)), gps, 120000,
                                                 diagnostics=diagnostics))
        self.assertEqual(diagnostics["transition_rejections"], {"choice_outside_report_time_window": 1})
        choice = extract_outgoing_choice(observed, match((approach, expected)), gps, 120000,
                                         earliest_choice_timestamp_ms=0)
        self.assertEqual(choice.chosen_edge_id, chosen)
        self.assertEqual(choice.decision_timestamp_ms, 1000)
        self.assertIsNone(extract_outgoing_choice(observed, match((approach, expected)), gps, 120000,
                                                 earliest_choice_timestamp_ms=3000))

    def test_local_choice_is_supported_without_rejoining_the_suggestion(self) -> None:
        approach, expected, chosen = ("a", "b", "0"), ("b", "c", "0"), ("b", "d", "0")
        states = tuple(MatchState(edge, 2.0, 10.0, 30.0) for edge in (approach, chosen, chosen))
        observed = MapMatchResult(states, (approach, chosen), (0, 1, 1), -1, 2, 2)
        suggested = match((approach, expected))
        gps = [TimedObservation(GeoPoint(14.55, 120.99), stamp) for stamp in (1000, 2000, 4000)]
        choice = extract_outgoing_choice(observed, suggested, gps, 3000)
        self.assertIsNone(extract_divergence_pair(observed, suggested, 1))
        self.assertIsNotNone(choice)
        self.assertEqual(choice.chosen_edge_id, chosen)
        self.assertEqual(choice.suggested_edge_id, expected)
        self.assertEqual(choice.decision_timestamp_ms, 1000)

    def test_short_outgoing_edge_can_be_supported_by_gps_on_its_continuation(self) -> None:
        approach, chosen, continuation = ("a", "b", "0"), ("b", "d", "0"), ("d", "e", "0")
        states = tuple(MatchState(edge, 2.0, 10.0, 30.0) for edge in (approach, continuation, continuation))
        observed = MapMatchResult(states, (approach, chosen, continuation), (0, 2, 2), -1, 2, 2)
        suggested = match((approach, ("b", "c", "0")))
        gps = [TimedObservation(GeoPoint(14.55, 120.99), stamp) for stamp in (1000, 2000, 4000)]
        choice = extract_outgoing_choice(observed, suggested, gps, 3000)
        self.assertIsNotNone(choice)
        self.assertEqual(choice.chosen_edge_id, chosen)
        self.assertEqual(choice.observed_continuation, (chosen, continuation))

    def test_connector_only_detour_without_off_suggestion_gps_is_not_a_label(self) -> None:
        approach, expected = ("a", "b", "0"), ("b", "c", "0")
        edges = (approach, ("b", "d", "0"), ("d", "b", "0"), expected)
        states = tuple(MatchState(edge, 2.0, 10.0, 30.0) for edge in (approach, expected, expected))
        observed = MapMatchResult(states, edges, (0, 3, 3), -1, 2, 2)
        gps = [TimedObservation(GeoPoint(14.55, 120.99), stamp) for stamp in (1000, 2000, 4000)]
        self.assertIsNone(extract_outgoing_choice(observed, match((approach, expected)), gps, 3000))
    def _write_distant_ride(self, directory: Path, *, deviated: bool) -> Path:
        graph = nx.MultiDiGraph(crs="EPSG:32651")
        positions = {
            "a": (283900, 1610000), "b": (284000, 1610000),
            "c": (284100, 1610000), "d": (284200, 1610000),
            "x": (283900, 1610100),
            "taft_start": (280000, 1610000), "taft_end": (280100, 1610000),
        }
        for node, (x, y) in positions.items():
            graph.add_node(node, x=float(x), y=float(y))
        for u, v in (("a", "b"), ("b", "c"), ("c", "d"), ("a", "x"),
                     ("taft_start", "taft_end")):
            left, right = positions[u], positions[v]
            graph.add_edge(u, v, key="0", length=100.0,
                           name="Taft Avenue" if u == "taft_start" else "Side Road",
                           geometry=f"LINESTRING ({left[0]} {left[1]}, {right[0]} {right[1]})")
        graph_path = directory / "graph.graphml"
        nx.write_graphml(graph, graph_path)
        inverse = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        route = []
        points = []
        for index, x in enumerate((283920, 284020, 284120, 284180)):
            lon, lat = inverse.transform(x, 1610000)
            route.append({"latitude": lat, "longitude": lon})
            points.append({"rideId": "ride", "latitude": lat, "longitude": lon,
                           "timestamp": 1000 + index * 4000})
        tables = {
            "rides_clean.csv": (["id", "userId"], [{"id": "ride", "userId": "rider"}]),
            "generated_routes_clean.csv": (
                ["routeId", "rideId", "type", "generatedAt", "routePoints"],
                [{"routeId": "route", "rideId": "ride", "type": "Initial Route",
                  "generatedAt": 0, "routePoints": json.dumps(route)}]),
            "map_points_clean.csv": (["rideId", "latitude", "longitude", "timestamp"], points),
            "deviations_clean.csv": (["deviationId", "rideId", "routeId", "timestamp"],
                                      [{"deviationId": "event", "rideId": "ride", "routeId": "route",
                                        "timestamp": 20000}] if deviated else []),
            "deviationResponses_clean.csv": (["deviationId", "primaryReason"],
                [{"deviationId": "event",
                  "primaryReason": "Shortcut/Faster Route/Personal Preference/Familiar Road"}]
                if deviated else []),
        }
        for name, (fields, rows) in tables.items():
            with (directory / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        return graph_path

    def test_verified_followed_choice_outside_taft_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            graph = self._write_distant_ride(directory, deviated=False)
            document = build_decision_candidates(directory, graph)
        self.assertEqual(document["summary"]["candidate_examples"], 1)
        self.assertEqual(document["examples"][0]["common_od"]["origin_node_id"], "a")
        self.assertIsNone(document["rules"]["study_corridor_limit_m"])

    def test_refinement_uses_real_destination_and_keeps_validation_labels_fixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            graph_path = self._write_distant_ride(folder, deviated=False)
            original = apply_automatic_rules(json.dumps(build_decision_candidates(folder, graph_path)).encode())
            example = original["examples"][0]
            # Recreate the old local-window destination while keeping its valid road evidence.
            example.pop("routing_context")
            example["common_od"]["destination_node_id"] = "c"
            for field in ("suggested_path", "observed_path_label_evidence"):
                example[field]["edges"] = example[field]["edges"][:2]
            refined = refine_real_decisions(folder, graph_path, original,
                                           {example["rider_group"]: "validation"})
            actual = refined["examples"][0]
            self.assertEqual(actual["routing_context"]["destination_node_id"], "d")
            self.assertEqual(actual["common_od"]["destination_node_id"], "c")
            self.assertEqual(actual["example_key"], example["example_key"])
            self.assertEqual(actual["target"], example["target"])
            self.assertEqual(refined["refinement"]["additional_real_choices"], 0)
            self.assertEqual(refined["refinement"]["changed_followed_destinations"], 1)

    def test_additional_real_choices_are_training_only_and_not_duplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            graph_path = self._write_distant_ride(folder, deviated=False)
            graph = nx.read_graphml(graph_path, force_multigraph=True)
            graph.add_node("z", x=283800., y=1610000.)
            graph.add_edge("z", "a", key="0", length=100., geometry="LINESTRING (283800 1610000, 283900 1610000)")
            nx.write_graphml(graph, graph_path)
            original = apply_automatic_rules(json.dumps(build_decision_candidates(folder, graph_path)).encode())
            example = original["examples"][0]
            for field in ("suggested_path", "observed_path_label_evidence"):
                example[field]["edges"] = example[field]["edges"][1:]
            example["common_od"]["origin_node_id"] = "b"
            example["routing_context"]["origin_node_id"] = "b"
            edges = (("z", "a", "0"), ("a", "b", "0"), ("b", "c", "0"), ("c", "d", "0"))
            states = tuple(MatchState(edge, 1., along, 100.) for edge, along in
                           ((edges[0], 90.), (edges[1], 10.), (edges[1], 60.), (edges[3], 20.)))
            matched = MapMatchResult(states, edges, (0, 1, 1, 3), -1., 1., 1.)
            with patch("stgat_lstm.preprocessing.build_decision_candidates.HiddenMarkovMatcher") as matcher:
                matcher.return_value.match.return_value = matched
                expanded = refine_real_decisions(folder, graph_path, original, {example["rider_group"]: "training"})
                repeated = refine_real_decisions(folder, graph_path, expanded, {example["rider_group"]: "training"})
                held_out = refine_real_decisions(folder, graph_path, original, {example["rider_group"]: "test"})
            self.assertEqual(expanded["refinement"]["additional_real_choices"], 1)
            self.assertEqual(repeated["refinement"]["additional_real_choices"], 0)
            self.assertEqual(held_out["refinement"]["additional_real_choices"], 0)
            additional = expanded["examples"][-1]
            self.assertEqual(additional["status"], "approved")
            self.assertEqual(additional["decision_timestamp_ms"], 1000)
            self.assertEqual(additional["suggested_path"]["edges"][0]["u"], "a")

    def test_deviation_geography_does_not_replace_gps_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            graph = self._write_distant_ride(directory, deviated=True)
            document = build_candidates(directory, graph)
        self.assertEqual(document["summary"]["eligible_intentional_events"], 1)
        self.assertEqual(document["summary"]["failure_counts"], {"no_directly_supported_outgoing_choice": 1})
        self.assertEqual(document["summary"]["review_candidates"], 0)

    def test_report_after_last_gps_can_retain_a_supported_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            graph_path = self._write_distant_ride(directory, deviated=True)
            graph = nx.read_graphml(graph_path, force_multigraph=True)
            graph.add_edge("b", "x", key="0", length=141.42, name="Alternative Road",
                           geometry="LINESTRING (284000 1610000, 283900 1610100)")
            nx.write_graphml(graph, graph_path)
            inverse = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
            with (directory / "map_points_clean.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["rideId", "latitude", "longitude", "timestamp"])
                writer.writeheader()
                for x, y, stamp in ((283950, 1610000, 1000), (283970, 1610030, 5000),
                                    (283930, 1610070, 9000)):
                    lon, lat = inverse.transform(x, y)
                    writer.writerow({"rideId": "ride", "latitude": lat, "longitude": lon,
                                     "timestamp": stamp})
            document = build_candidates(directory, graph_path)
            deviation_path = directory / "deviations_clean.csv"
            with deviation_path.open(encoding="utf-8", newline="") as stream:
                source_rows = list(csv.DictReader(stream))
            source_rows[0].update({"originalRouteEdge": "Drive east on Side Road.",
                                   "deviatedEdge": "Drive northwest on Alternative Road.",
                                   "gpsLocation": f"{lat},{lon}"})
            with deviation_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
                writer.writeheader()
                writer.writerows(source_rows)
            # Simulate the old extractor finding no exact shared incoming edge.
            with patch("stgat_lstm.preprocessing.build_deviation_candidates.extract_outgoing_choice", return_value=None):
                inferred = build_candidates(directory, graph_path)
            artifact = directory / "deviation_candidates.json"
            artifact.write_text(json.dumps(inferred), encoding="utf-8")
            decisions = build_decision_candidates(directory, graph_path, artifact)
            approved = apply_automatic_rules(json.dumps(decisions).encode())
            resolved = approved["examples"][0]
            self.assertEqual(approved["review"]["approved"], 1)
            self.assertEqual(resolved["road_choice"]["label_method"], "recorded_navigation_and_local_geometry")
            self.assertIn("recorded_navigation", resolved["feature_boundary"]["label_only_never_input"])
            self.assertLess(resolved["decision_timestamp_ms"], resolved["reported_event_timestamp_ms"])
        self.assertEqual(document["summary"]["review_candidates"], 1)
        self.assertEqual(document["candidates"][0]["road_choice"]["chosen_edge_id"], "b|x|0")
        self.assertLess(document["candidates"][0]["decision_timestamp_ms"], 20000)
        self.assertFalse(document["rules"]["require_gps_at_or_after_report"])

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
