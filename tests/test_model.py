"""Tests for model inputs, outputs, and gradient flow."""

import unittest

import torch

from graph_test_data import build_test_graph_data
from stgat_lstm.model import DecisionPreferenceModel, PreferenceModel


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = build_test_graph_data()

    def test_builds_variable_dimension_model_input(self) -> None:
        self.assertNotIn("far", self.data.node_ids)
        inputs = self.data.build_model_input("4")
        self.assertEqual(inputs.node_features.shape[1], len(self.data.schema.node))
        self.assertEqual(inputs.edge_static.shape[1], len(self.data.schema.edge_static))
        age_index = self.data.schema.edge_dynamic.index("traffic_age_scaled")
        self.assertTrue(torch.all(inputs.temporal_edge_features[0][:, age_index] == 1))
        model = PreferenceModel(**self.data.model_configuration())
        scores = model(inputs)
        self.assertEqual(scores.shape, (len(self.data.edge_ids),))
        self.assertTrue(torch.all(scores > 0))

    def test_multitask_model_scores_edges_and_road_deviation_probabilities(self) -> None:
        inputs = self.data.build_model_input("4")
        model = DecisionPreferenceModel(**self.data.model_configuration())
        costs, logits = model(inputs)
        self.assertEqual(costs.shape, (len(self.data.edge_ids),))
        self.assertEqual(logits.shape, (len(self.data.edge_ids),))
        (costs.mean() + logits.mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
