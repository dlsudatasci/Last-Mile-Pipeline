"""Contract and smoke tests for the first synthetic training slice."""

from __future__ import annotations

import copy
import hashlib
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from stgat_lstm.features import build_model_input
from stgat_lstm.fixture import TrafficSnapshot, prepare_choices, prepare_routing_request, read_fixture
from stgat_lstm.model import PreferenceModel, preference_loss
from stgat_lstm.predict import route_from_checkpoint
from stgat_lstm.routing import shortest_preference_path
from stgat_lstm.training import train_preference_model


FIXTURE = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "rider_preference_fixture.json"


class PreferencePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.graph, cls.scenarios = read_fixture(FIXTURE)
        cls.choices = prepare_choices(cls.graph, cls.scenarios)
        cls.by_id = {choice.case_id: choice for choice in cls.choices}

    def test_expected_outcomes_are_reconstructed_without_reading_test_oracle(self) -> None:
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario["case_id"]):
                choice = self.by_id[scenario["case_id"]]
                expected = scenario["expected"]
                self.assertEqual(choice.deviation_label, expected["deviation_label"])
                self.assertEqual(len(choice.traffic_history), expected["usable_traffic_count"])
                if expected["preference_pair"] is None:
                    self.assertIsNone(choice.preferred_edge_ids)
                else:
                    wanted = self.graph.candidate_paths[expected["preference_pair"]["preferred_path_id"]]
                    rejected = self.graph.candidate_paths[expected["preference_pair"]["rejected_path_id"]]
                    self.assertEqual((choice.preferred_edge_ids, choice.rejected_edge_ids), (wanted, rejected))

        changed = copy.deepcopy(self.scenarios)
        for scenario in changed:
            scenario["expected"] = {"deliberately_wrong": "never used as training data"}
        reconstructed = prepare_choices(self.graph, changed)
        self.assertEqual(reconstructed, self.choices)

    def test_late_traffic_and_missing_traffic_have_missingness_mask(self) -> None:
        for case_id in ("syn_case_06", "syn_case_09"):
            choice = self.by_id[case_id]
            self.assertEqual(choice.traffic_history, ())
            inputs = build_model_input(self.graph, choice)
            self.assertTrue(torch.all(inputs.temporal_edge_features[0][:, 1] == 0))
        self.assertIn("exclude_late_arriving_traffic", self.by_id["syn_case_09"].quality_flags)

    def test_routing_request_does_not_consume_post_choice_information(self) -> None:
        original = self.scenarios[0]
        changed = copy.deepcopy(original)
        changed["input"].pop("observed_edge_ids")
        changed["input"].pop("trajectory_complete")
        changed["input"].pop("event")
        changed.pop("expected")
        request = prepare_routing_request(original)
        self.assertEqual(request, prepare_routing_request(changed))
        self.assertTrue(torch.equal(
            build_model_input(self.graph, request).edge_static,
            build_model_input(self.graph, prepare_routing_request(changed)).edge_static,
        ))

    def test_wrong_turn_and_stop_are_not_routing_preference_pairs(self) -> None:
        for case_id in ("syn_case_04", "syn_case_05"):
            choice = self.by_id[case_id]
            self.assertIs(choice.deviation_label, True)
            self.assertIsNone(choice.preferred_edge_ids)

    def test_gps_and_prior_route_gaps_do_not_generate_labels(self) -> None:
        for case_id in ("syn_case_07", "syn_case_08"):
            choice = self.by_id[case_id]
            self.assertIsNone(choice.deviation_label)
            self.assertIsNone(choice.preferred_edge_ids)

    def test_router_excludes_forbidden_shortcut_and_rejects_invalid_costs(self) -> None:
        costs = {edge.edge_id: 1.0 for edge in self.graph.edges if edge.motorcycle_allowed}
        costs["syn_AD_restricted"] = 0.001
        route = shortest_preference_path(self.graph, costs, "syn_A", "syn_D")
        self.assertNotIn("syn_AD_restricted", route.edge_ids)
        self.assertEqual(self.graph.check_path(route.edge_ids), ("syn_A", "syn_D"))
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            shortest_preference_path(self.graph, {**costs, "syn_AB": 0.0}, "syn_A", "syn_D")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            shortest_preference_path(self.graph, {**costs, "syn_AB": math.nan}, "syn_A", "syn_D")

    def test_spatiotemporal_model_and_baselines_produce_positive_gradients(self) -> None:
        choice = self.by_id["syn_case_01"]
        older = TrafficSnapshot(10, 20, "low", "moderate")
        two_frame_choice = replace(choice, traffic_history=(older, *choice.traffic_history))
        inputs = build_model_input(self.graph, two_frame_choice)
        for architecture in ("gcn", "gat", "stgat_lstm"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(7)
                model = PreferenceModel(architecture=architecture)
                scores = model(inputs)
                self.assertEqual(scores.shape, (6,))
                self.assertTrue(torch.all(scores > 0))
                loss = preference_loss(scores, choice.preferred_edge_ids, choice.rejected_edge_ids, inputs.edge_ids)
                loss.backward()
                self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
                if architecture == "stgat_lstm":
                    self.assertTrue(all(parameter.grad is not None for parameter in model.temporal.parameters()))

    def test_trained_scores_route_to_both_observed_toy_choices(self) -> None:
        model, summary = train_preference_model(self.graph, self.choices, epochs=60)
        self.assertEqual(summary.examples, 4)
        self.assertLess(summary.final_loss, summary.initial_loss)
        self.assertEqual(summary.in_sample_correct, 4)
        for case_id in ("syn_case_01", "syn_case_02"):
            choice = self.by_id[case_id]
            inputs = build_model_input(self.graph, choice)
            with torch.no_grad():
                scores = model(inputs)
            costs = {edge_id: float(scores[index]) for index, edge_id in enumerate(inputs.edge_ids)}
            route = shortest_preference_path(self.graph, costs, choice.origin_node_id, choice.destination_node_id)
            self.assertEqual(route.edge_ids, choice.observed_edge_ids)
        self.assertEqual(
            self.by_id["syn_case_01"].preferred_edge_ids,
            self.graph.candidate_paths["side"],
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "synthetic.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": "stgat_lstm",
                    "hidden_channels": model.hidden_channels,
                    "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                    "provenance": "synthetic-only",
                },
                checkpoint_path,
            )
            for case_id in ("syn_case_01", "syn_case_02"):
                restored = route_from_checkpoint(checkpoint_path, FIXTURE, case_id)
                self.assertEqual(tuple(restored["route_edge_ids"]), self.by_id[case_id].observed_edge_ids)


if __name__ == "__main__":
    unittest.main()
