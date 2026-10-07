"""Tests for rider-disjoint evaluation helpers."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import torch

from stgat_lstm.evaluate_model import (classification_metrics, rider_disjoint_folds,
    temporal_evidence_report, latest_only_history, evaluate_rider_disjoint,
    validation_threshold)
from stgat_lstm.train_model import PreparedDecision
from stgat_lstm.model import ModelInput


class DecisionEvaluationTests(unittest.TestCase):
    TRAFFIC_SCHEMA = ("congestion_normalized", "speed_ratio_to_reference", "congestion_observed",
                      "speed_observed", "traffic_age_scaled")

    def history_examples(self):
        inputs = ModelInput(torch.zeros((1, 1)), torch.zeros((2, 1), dtype=torch.long),
            torch.ones((1, 1)), (torch.tensor([[0., .2, 0., 1., .1]]),
                               torch.tensor([[0., .8, 0., 1., 0.]])), torch.zeros((1, 1)), ("edge",))
        return [PreparedDecision(str(i), f"rider-{i}", i % 2, ("edge",),
                                 (("edge", i % 2),), None, None, inputs)
                for i in range(4)]

    def test_metrics_are_computed_from_held_out_probabilities(self) -> None:
        predictions = [
            {"label": 1, "probability_deviated": 0.9},
            {"label": 0, "probability_deviated": 0.2},
            {"label": 1, "probability_deviated": 0.4},
            {"label": 0, "probability_deviated": 0.7},
        ]
        metrics = classification_metrics(predictions)
        self.assertEqual(metrics["confusion"], {
            "true_positive": 1,
            "true_negative": 1,
            "false_positive": 1,
            "false_negative": 1,
        })
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertGreater(metrics["log_loss"], 0)
        self.assertEqual(metrics["roc_auc"], 0.75)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["class_support"], {"followed": 2, "deviated": 2})

    def test_auc_ties_single_class_and_invalid_predictions(self):
        tied = [{"label": label, "probability_deviated": .4} for label in (0, 1, 0, 1)]
        self.assertEqual(classification_metrics(tied)["roc_auc"], 0.5)
        self.assertEqual(classification_metrics(tied)["average_precision"], 0.5)
        absent = classification_metrics([{"label": 0, "probability_deviated": .1}])
        self.assertIsNone(absent["roc_auc"])
        self.assertIsNone(absent["balanced_accuracy"])
        self.assertIsNone(absent["average_precision"])
        for value in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "probabilities"):
                classification_metrics([{"label": 0, "probability_deviated": value}])
        with self.assertRaisesRegex(ValueError, "labels"):
            classification_metrics([{"label": 2, "probability_deviated": .5}])

    def test_validation_threshold_separates_ordered_probabilities(self):
        predictions = [
            {"label": 0, "probability_deviated": 0.55},
            {"label": 0, "probability_deviated": 0.60},
            {"label": 1, "probability_deviated": 0.75},
            {"label": 1, "probability_deviated": 0.80},
        ]
        threshold = validation_threshold(predictions)
        self.assertGreater(threshold, 0.60)
        self.assertLess(threshold, 0.75)
        self.assertEqual(classification_metrics(predictions, threshold)["balanced_accuracy"], 1.0)

    def test_history_diagnostics_ignore_age_and_missingness_only_changes(self):
        examples = self.history_examples()
        report = temporal_evidence_report(examples, self.TRAFFIC_SCHEMA)
        self.assertEqual(report["examples_with_observed_value_changes"], 4)
        self.assertEqual(report["suggested_paths_with_observed_value_changes"], 4)
        unobserved = torch.tensor([[0., 0., 0., 0., 1.]])
        latest = examples[0].inputs.temporal_edge_features[-1]
        example = replace(examples[0], inputs=replace(examples[0].inputs,
            temporal_edge_features=(unobserved, latest)))
        report = temporal_evidence_report([example], self.TRAFFIC_SCHEMA)
        self.assertEqual(report["examples_with_observed_value_changes"], 0)
        self.assertEqual(report["status"], "insufficient_temporal_evidence")
        same_value = latest.clone()
        same_value[0, -1] = .2
        example = replace(example, inputs=replace(example.inputs, temporal_edge_features=(same_value, latest)))
        self.assertEqual(temporal_evidence_report([example], self.TRAFFIC_SCHEMA)
                         ["examples_with_observed_value_changes"], 0)

    def test_latest_only_control_preserves_length_and_does_not_modify_full_history(self):
        examples = self.history_examples()
        controlled = latest_only_history(examples)
        self.assertEqual(len(controlled[0].inputs.temporal_edge_features), 2)
        self.assertTrue(torch.equal(controlled[0].inputs.temporal_edge_features[0],
                                    examples[0].inputs.temporal_edge_features[-1]))
        self.assertFalse(torch.equal(examples[0].inputs.temporal_edge_features[0],
                                     examples[0].inputs.temporal_edge_features[-1]))

    def test_temporal_control_trains_on_same_rider_folds_and_seed(self):
        examples = self.history_examples()
        fake_model = Mock(return_value=(torch.ones(1), torch.zeros(1)))
        with patch("stgat_lstm.evaluate_model.train_decision_model",
                   return_value=(fake_model, None)) as train:
            report = evaluate_rider_disjoint(examples, {}, epochs=1,
                temporal_ablation=True, traffic_schema=self.TRAFFIC_SCHEMA)
        self.assertEqual(train.call_count, 20)
        self.assertEqual(report["lstm_comparison"]["latest_only_control"], "run")
        self.assertIsNone(report["lstm_comparison"]["full_minus_control"]
                          ["stgat_lstm_latest_only"]["preference_ranking_accuracy"])
        control = report["models"]["stgat_lstm_latest_only"]
        self.assertEqual(len(control["predictions"]), 4)
        for fold in range(4):
            full, latest = train.call_args_list[fold * 5 + 3:fold * 5 + 5]
            self.assertEqual(full.kwargs["seed"], latest.kwargs["seed"])
            self.assertEqual(full.args[1], latest.args[1])
            self.assertEqual([e.example_key for e in full.args[0]],
                             [e.example_key for e in latest.args[0]])
        with patch("stgat_lstm.evaluate_model.train_decision_model", return_value=(fake_model, None)) as train:
            report = evaluate_rider_disjoint(latest_only_history(examples), {}, epochs=1,
                temporal_ablation=True, traffic_schema=self.TRAFFIC_SCHEMA)
        self.assertEqual(train.call_count, 16)
        self.assertEqual(report["lstm_comparison"]["latest_only_control"],
                         "skipped_no_observed_history_variation")

    def test_folds_hold_out_complete_riders_and_require_both_training_classes(self) -> None:
        inputs = ModelInput(
            node_features=torch.zeros((1, 1)),
            edge_index=torch.zeros((2, 1), dtype=torch.long),
            edge_static=torch.zeros((1, 1)),
            temporal_edge_features=(torch.zeros((1, 1)),),
            destination_features=torch.zeros((1, 1)),
            edge_ids=("edge",),
        )

        def example(key: str, rider: str, label: int) -> PreparedDecision:
            return PreparedDecision(key, rider, label, ("edge",), (("edge", label),),
                                    None, None, inputs)

        prepared = [
            example("a", "rider-a", 0),
            example("b", "rider-a", 0),
            example("c", "rider-b", 1),
            example("d", "rider-c", 0),
            example("e", "rider-d", 1),
        ]
        folds = rider_disjoint_folds(prepared)
        self.assertEqual(len(folds), 4)
        for rider, train, test in folds:
            self.assertTrue(all(item.rider_group == rider for item in test))
            self.assertTrue(all(item.rider_group != rider for item in train))
            self.assertEqual({item.label for item in train}, {0, 1})


if __name__ == "__main__":
    unittest.main()
