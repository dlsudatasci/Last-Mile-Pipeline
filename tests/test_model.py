"""Tests for model inputs, outputs, and gradient flow."""

import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import torch

from graph_test_data import build_test_graph_data
from stgat_lstm.modeling.model import (DecisionPreferenceModel, PreferenceModel,
                              outgoing_choice_log_probabilities)
from stgat_lstm.modeling.train_model import (PreparedDecision, PreparedRoadChoice, _forward_batch,
                                    train_decision_model, mask_traffic_history, first_junction_visits)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = build_test_graph_data()

    def test_first_junction_visit_preserves_examples_and_other_rider_preferences(self):
        inputs = self.data.build_model_input("4")
        first = PreparedRoadChoice("2", ("2|3|0", "2|4|0"), "2|3|0", "2|3|0", 0)
        later = replace(first, chosen_edge_id="2|4|0", suggested_edge_id="2|4|0")
        original = PreparedDecision("repeat", "rider-a", 0, ("2|3|0", "2|4|0"),
                                    (first, later), None, None, inputs)
        other = replace(original, example_key="other", rider_group="rider-b", choice_targets=(later,))
        filtered, audit = first_junction_visits([original, other])
        self.assertEqual(filtered[0].choice_targets, (first,))
        self.assertEqual(filtered[1].choice_targets, (later,))
        self.assertEqual(original.choice_targets, (first, later))
        self.assertIs(filtered[0].inputs, inputs)
        self.assertEqual(audit["choices_before"], 3)
        self.assertEqual(audit["choices_after"], 2)
        self.assertEqual(audit["examples_before"], audit["examples_after"])

    def test_first_junction_visit_refuses_to_discard_later_alternative(self):
        inputs = self.data.build_model_input("4")
        first = PreparedRoadChoice("2", ("2|3|0", "2|4|0"), "2|4|0", "2|4|0", 0)
        alternative = replace(first, chosen_edge_id="2|3|0", deviated=1)
        example = PreparedDecision("later-alternative", "rider-a", 1, ("2|4|0",),
                                   (first, alternative), None, None, inputs)
        with self.assertRaisesRegex(ValueError, "do not discard"):
            first_junction_visits([example])

    def test_fixed_epoch_training_monitors_validation_without_selecting_on_it(self):
        torch.set_num_threads(1)
        inputs = self.data.build_model_input("4")
        examples = [PreparedDecision(str(label), f"rider-{label}", label, ("2|4|0",),
            (PreparedRoadChoice("2", ("2|3|0", "2|4|0"), "2|3|0" if label else "2|4|0",
                                "2|4|0", label),), None, None, inputs) for label in (0, 1)]
        validation = [replace(examples[0], rider_group="validation-rider")]
        states = []
        def score(model, *args):
            states.append(deepcopy(model.state_dict()))
            return {"accuracy": 1. if len(states) == 1 else .5, "negative_log_likelihood": .1 * len(states)}
        with patch("stgat_lstm.modeling.train_model._validation_choice_score", side_effect=score):
            model, summary = train_decision_model(examples, self.data.model_configuration(), epochs=3,
                validation=validation, patience=1, validation_interval=1, select_validation_checkpoint=False)
        self.assertEqual(summary.epochs, 3)
        self.assertEqual(summary.selected_epoch, 3)
        self.assertEqual(len(summary.validation_history), 3)
        self.assertTrue(all(torch.equal(value, states[-1][name]) for name, value in model.state_dict().items()))

    def test_repeated_frames_preserve_outputs_and_parameter_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        inputs = self.data.build_model_input("4")
        first = inputs.temporal_edge_features[0].clone()
        second = first.clone()
        second[0, 0] += .25
        inputs = replace(inputs, temporal_edge_features=(first, second, first.clone(),
                                                        second.clone(), first.clone(), first.clone()))
        cached = DecisionPreferenceModel(**self.data.model_configuration())
        reference = deepcopy(cached)

        def uncached(backbone_inputs):
            sequence = torch.stack([reference.backbone._spatial_snapshot(backbone_inputs, frame)
                for frame in backbone_inputs.temporal_edge_features], dim=1)
            return reference.backbone.temporal(sequence)[0][:, -1, :]

        for training in (True, False):
            cached.train(training)
            reference.train(training)
            cached.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            with patch.object(cached.backbone, "_spatial_snapshot",
                              wraps=cached.backbone._spatial_snapshot) as spatial:
                actual = cached(inputs)
            self.assertEqual(spatial.call_count, 2)
            with patch.object(reference.backbone, "encode_edges", side_effect=uncached):
                expected = reference(inputs)
            for value, target in zip(actual, expected):
                torch.testing.assert_close(value, target, rtol=1e-6, atol=1e-7)
            sum(value.square().mean() for value in actual).backward()
            sum(value.square().mean() for value in expected).backward()
            for (name, parameter), (other_name, other) in zip(cached.named_parameters(), reference.named_parameters()):
                self.assertEqual(name, other_name)
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-7,
                                           msg=lambda message: f"{name}: {message}")

    def test_frame_input_gradients_are_not_merged(self):
        torch.set_num_threads(1)
        inputs = self.data.build_model_input("4")
        frames = tuple(inputs.temporal_edge_features[0].clone().requires_grad_() for _ in range(3))
        inputs = replace(inputs, temporal_edge_features=frames)
        model = DecisionPreferenceModel(**self.data.model_configuration())
        with patch.object(model.backbone, "_spatial_snapshot", wraps=model.backbone._spatial_snapshot) as spatial:
            costs, logits = model(inputs)
        self.assertEqual(spatial.call_count, 3)
        (costs.mean() + logits.mean()).backward()
        for frame in frames:
            self.assertIsNotNone(frame.grad)
            self.assertTrue(torch.isfinite(frame.grad).all())

    def test_spatial_reuse_does_not_survive_forward_calls(self):
        torch.set_num_threads(1)
        inputs = self.data.build_model_input("4")
        inputs = replace(inputs, temporal_edge_features=tuple(inputs.temporal_edge_features[0].clone()
                                                             for _ in range(6)))
        model = DecisionPreferenceModel(**self.data.model_configuration())
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with patch.object(model.backbone, "_spatial_snapshot", wraps=model.backbone._spatial_snapshot) as spatial:
            before = model(inputs)[1]
            before.square().mean().backward()
            optimizer.step()
            after = model(inputs)[1]
        self.assertEqual(spatial.call_count, 2)
        self.assertFalse(torch.equal(before, after))

    def test_masking_preserves_geometry_and_marks_hidden_speeds_unknown(self):
        inputs = self.data.build_model_input("4")
        observed = inputs.temporal_edge_features[0].clone()
        observed[:] = torch.tensor([.2, .8, 1., 1., .1])
        inputs = replace(inputs, temporal_edge_features=(observed,))
        hidden = mask_traffic_history(inputs, .999, torch.Generator().manual_seed(17))
        self.assertIs(hidden.edge_static, inputs.edge_static)
        self.assertIs(hidden.edge_index, inputs.edge_index)
        self.assertEqual(hidden.edge_ids, inputs.edge_ids)
        mask = hidden.temporal_edge_features[0][:, 3] == 0
        self.assertTrue(mask.any())
        self.assertTrue(torch.all(hidden.temporal_edge_features[0][mask] == torch.tensor([0., 0., 0., 0., 1.])))
        self.assertTrue(torch.all(inputs.temporal_edge_features[0][:, 3] == 1))

    def test_early_stopping_restores_best_validation_epoch(self):
        torch.set_num_threads(1)
        inputs = self.data.build_model_input("4")
        examples = [PreparedDecision(str(label), f"rider-{label}", label, ("2|4|0",),
            (PreparedRoadChoice("2", ("2|3|0", "2|4|0"), "2|3|0" if label else "2|4|0",
                                "2|4|0", label),), None, None, inputs) for label in (0, 1)]
        validation = [replace(examples[0], rider_group="validation-rider")]
        first_state = {}
        from stgat_lstm.modeling.train_model import _validation_choice_score
        def scores(model, *args):
            if not first_state:
                first_state.update(deepcopy(model.state_dict()))
                return {"accuracy": 1., "negative_log_likelihood": .1}
            return {"accuracy": .5, "negative_log_likelihood": .5}
        with patch("stgat_lstm.modeling.train_model._validation_choice_score", side_effect=scores):
            model, summary = train_decision_model(examples, self.data.model_configuration(), epochs=10,
                validation=validation, patience=2, validation_interval=1, weight_decay=.0001,
                label_smoothing=.05, balance_rides=True)
        self.assertEqual(summary.epochs, 3)
        self.assertEqual(summary.selected_epoch, 1)
        self.assertTrue(all(torch.equal(value, first_state[name]) for name, value in model.state_dict().items()))
        with self.assertRaisesRegex(ValueError, "separate riders"):
            train_decision_model(examples, self.data.model_configuration(), validation=examples)

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

    def test_choice_model_normalizes_accessible_outgoing_roads(self) -> None:
        inputs = self.data.build_model_input("4")
        model = DecisionPreferenceModel(**self.data.model_configuration())
        costs, logits = model(inputs)
        self.assertEqual(costs.shape, (len(self.data.edge_ids),))
        self.assertEqual(logits.shape, (len(self.data.edge_ids),))
        log_probabilities = outgoing_choice_log_probabilities(logits, inputs)
        source = inputs.edge_index[0]
        for node in torch.unique(source[inputs.routable_mask]):
            selected = (source == node) & inputs.routable_mask
            self.assertAlmostEqual(
                float(log_probabilities[selected].exp().sum().detach()), 1.0, places=5
            )
        (costs.mean() + logits.mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_disconnected_batch_matches_individual_forward_passes(self) -> None:
        inputs = self.data.build_model_input("4")
        examples = [
            PreparedDecision(str(index), str(index), index % 2, ("2|4|0",),
                (PreparedRoadChoice("2", ("2|3|0", "2|4|0"), "2|4|0", "2|4|0", 0),),
                None, None, inputs)
            for index in range(2)
        ]
        model = DecisionPreferenceModel(**self.data.model_configuration())
        model.eval()
        individual = [model(example.inputs) for example in examples]
        batched = _forward_batch(model, examples)
        for expected, actual in zip(individual, batched):
            self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-6))
            self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-6))

    def test_training_accepts_validation_selected_alternative_weight(self) -> None:
        inputs = self.data.build_model_input("4")
        examples = [
            PreparedDecision("follow", "rider-a", 0, ("2|4|0",),
                (PreparedRoadChoice("2", ("2|3|0", "2|4|0"), "2|4|0", "2|4|0", 0),),
                None, None, inputs),
            PreparedDecision("deviate", "rider-b", 1, ("2|4|0",),
                (PreparedRoadChoice("2", ("2|3|0", "2|4|0"), "2|3|0", "2|4|0", 1),),
                None, None, inputs),
        ]
        _, summary = train_decision_model(
            examples, self.data.model_configuration(), epochs=1,
            alternative_choice_weight=2.5,
        )
        self.assertEqual(summary.deviated_choice_weight, 2.5)


if __name__ == "__main__":
    unittest.main()
