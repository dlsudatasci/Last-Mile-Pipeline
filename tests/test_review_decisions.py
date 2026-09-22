"""Tests for the human-review approval boundary."""

from __future__ import annotations

import json
import unittest

from stgat_lstm.review_decisions import apply_decisions, create_decision_template


class ReviewDecisionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
