"""Tests for the human-review approval boundary."""

from __future__ import annotations

import json
import unittest

from stgat_lstm.preprocessing.review_decisions import apply_automatic_rules, apply_decisions, create_decision_template


class ReviewDecisionTests(unittest.TestCase):
    def test_local_choice_can_be_approved_without_rejoin_or_new_route_overlap(self) -> None:
        artifact = {
            "status": "review_required_not_training_ready",
            "examples": [{
                "example_key": "local",
                "target": {"label": "deviated"},
                "has_route_pair": False,
                "routing_context": {"origin_node_id": "1", "destination_node_id": "9"},
                "suggested_path": {"edges": [{"u": "1", "v": "2", "key": "0"}]},
                "observed_path_label_evidence": {"edges": [{"u": "1", "v": "3", "key": "0"}]},
                "road_choice": {"source_node_id": "1", "suggested_edge_id": "1|2|0",
                                "chosen_edge_id": "1|3|0", "supporting_gps_points": 2},
                "survey_evidence": {"primary_reason": "Traffic Congestion", "traffic_severity": "4"},
                "quality": {"mean_gps_to_matched_edge_m": 5,
                            "gps_edge_overlap_with_prior": 0.8,
                            "gps_edge_overlap_with_regenerated": None},
            }],
        }
        result = apply_automatic_rules(json.dumps(artifact).encode())
        self.assertEqual(result["review"]["approved"], 1)
        record = artifact["examples"][0]
        record["decision_timestamp_ms"] = 1000
        record["recorded_navigation"] = {"deviatedEdge": "Drive east."}
        record["road_choice"]["label_method"] = "recorded_navigation_and_local_geometry"
        evidence = {"method": "recorded_navigation_and_local_geometry", "status": "resolved_unique_choice",
                    "candidate_choices": [{"suggested_edge_id": "1|2|0", "chosen_edge_id": "1|3|0",
                                           "decision_timestamp_ms": 1000}]}
        record["road_choice"]["inference_evidence"] = evidence
        self.assertEqual(apply_automatic_rules(json.dumps(artifact).encode())["review"]["approved"], 1)
        evidence["candidate_choices"][0]["decision_timestamp_ms"] = 2000
        with self.assertRaisesRegex(ValueError, "rejected every"):
            apply_automatic_rules(json.dumps(artifact).encode())
        evidence["candidate_choices"][0]["decision_timestamp_ms"] = 1000
        evidence["status"] = "ambiguous_choices"
        with self.assertRaisesRegex(ValueError, "rejected every"):
            apply_automatic_rules(json.dumps(artifact).encode())
        evidence["status"] = "resolved_unique_choice"
        artifact["examples"][0]["road_choice"]["supporting_gps_points"] = 1
        with self.assertRaisesRegex(ValueError, "rejected every"):
            apply_automatic_rules(json.dumps(artifact).encode())

    def setUp(self) -> None:
        self.candidates = json.dumps(
            {
                "status": "review_required_not_training_ready",
                "candidates": [{"event_key": "one", "status": "review_required"}],
            },
            separators=(",", ":"),
        ).encode()

    def test_pending_decision_cannot_create_training_artifact(self) -> None:
        decisions = create_decision_template(self.candidates)
        with self.assertRaisesRegex(ValueError, "approve or reject"):
            apply_decisions(self.candidates, decisions)

    def test_approved_decision_is_auditable(self) -> None:
        decisions = create_decision_template(self.candidates)
        decisions["decisions"][0].update(decision="approve", review_note="Paths match the event")
        approved = apply_decisions(self.candidates, decisions)
        self.assertEqual(approved["status"], "approved_for_training")
        self.assertEqual(approved["candidates"][0]["status"], "approved")
        self.assertEqual(approved["review"]["approved"], 1)

    def test_decisions_are_bound_to_exact_candidate_bytes(self) -> None:
        decisions = create_decision_template(self.candidates)
        decisions["decisions"][0]["decision"] = "approve"
        with self.assertRaisesRegex(ValueError, "different candidate artifact"):
            apply_decisions(self.candidates + b"\n", decisions)

    def test_decision_examples_use_example_keys(self) -> None:
        artifact = json.dumps(
            {
                "status": "review_required_not_training_ready",
                "examples": [{"example_key": "choice-one", "status": "review_required"}],
            },
            separators=(",", ":"),
        ).encode()
        decisions = create_decision_template(artifact)
        self.assertEqual(decisions["decisions"][0]["example_key"], "choice-one")
        decisions["decisions"][0]["decision"] = "approve"
        approved = apply_decisions(artifact, decisions)
        self.assertEqual(approved["examples"][0]["status"], "approved")

    def test_unchanged_decisions_can_be_carried_forward(self) -> None:
        decisions = create_decision_template(self.candidates)
        decisions["decisions"][0].update(decision="approve", review_note="GPS supports this pair")
        approved = apply_decisions(self.candidates, decisions)
        updated_candidates = json.dumps({
            "status": "review_required_not_training_ready",
            "candidates": [
                {"event_key": "one", "status": "review_required"},
                {"event_key": "two", "status": "review_required"},
            ],
        }, separators=(",", ":")).encode()
        template = create_decision_template(updated_candidates, json.dumps(approved).encode())
        self.assertEqual(template["summary"], {"candidates": 2, "carried_forward": 1, "pending": 1})
        self.assertEqual(template["decisions"][0]["decision"], "approve")
        self.assertEqual(template["decisions"][0]["review_note"], "GPS supports this pair")
        self.assertEqual(template["decisions"][1]["decision"], "pending")

    def test_changed_candidate_is_not_carried_forward(self) -> None:
        decisions = create_decision_template(self.candidates)
        decisions["decisions"][0]["decision"] = "approve"
        approved = apply_decisions(self.candidates, decisions)
        changed = json.dumps({
            "status": "review_required_not_training_ready",
            "candidates": [{"event_key": "one", "status": "review_required", "quality": {"score": 0.8}}],
        }, separators=(",", ":")).encode()
        template = create_decision_template(changed, json.dumps(approved).encode())
        self.assertEqual(template["summary"]["carried_forward"], 0)
        self.assertEqual(template["decisions"][0]["decision"], "pending")

    def test_carry_forward_requires_an_approved_artifact(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an approved"):
            create_decision_template(self.candidates, self.candidates)

    def test_automatic_rules_use_survey_and_geometry_evidence(self) -> None:
        def path():
            return {"edges": [{"u": "1", "v": "2", "key": "0"}]}

        artifact = json.dumps({
            "status": "review_required_not_training_ready",
            "examples": [
                {
                    "example_key": "traffic",
                    "target": {"label": "deviated"},
                    "common_od": {"origin_node_id": "1", "destination_node_id": "2"},
                    "suggested_path": path(),
                    "observed_path_label_evidence": path(),
                    "survey_evidence": {"primary_reason": "Traffic Congestion", "traffic_severity": "4 = Heavy"},
                    "quality": {"gps_edge_overlap_with_prior": 0.2,
                                "gps_edge_overlap_with_regenerated": 0.8,
                                "mean_gps_to_matched_edge_m": 4.0},
                },
                {
                    "example_key": "personal-stop",
                    "target": {"label": "deviated"},
                    "common_od": {"origin_node_id": "1", "destination_node_id": "2"},
                    "suggested_path": path(),
                    "observed_path_label_evidence": path(),
                    "survey_evidence": {"primary_reason": "Personal Stop (Meal, Restroom, Break, Refueling, etc.)"},
                    "quality": {"gps_edge_overlap_with_prior": 0.2,
                                "gps_edge_overlap_with_regenerated": 0.8,
                                "mean_gps_to_matched_edge_m": 4.0},
                },
            ],
        }, separators=(",", ":")).encode()
        approved = apply_automatic_rules(artifact)
        self.assertEqual(approved["status"], "approved_for_training")
        self.assertEqual(approved["review"]["approved"], 1)
        self.assertEqual(approved["review"]["rejected"], 1)
        self.assertEqual(approved["examples"][0]["status"], "approved")
        self.assertEqual(approved["examples"][1]["status"], "rejected")

    def test_automatic_rules_reject_missing_traffic_severity(self) -> None:
        path = {"edges": [{"u": "1", "v": "2", "key": "0"}]}
        artifact = json.dumps({
            "status": "review_required_not_training_ready",
            "examples": [{
                "example_key": "traffic",
                "target": {"label": "deviated"},
                "common_od": {"origin_node_id": "1", "destination_node_id": "2"},
                "suggested_path": path,
                "observed_path_label_evidence": path,
                "survey_evidence": {"primary_reason": "Traffic Congestion", "traffic_severity": None},
                "quality": {"gps_edge_overlap_with_prior": 0.2,
                            "gps_edge_overlap_with_regenerated": 0.8,
                            "mean_gps_to_matched_edge_m": 4.0},
            }],
        }, separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError, "rejected every"):
            apply_automatic_rules(artifact)


if __name__ == "__main__":
    unittest.main()
