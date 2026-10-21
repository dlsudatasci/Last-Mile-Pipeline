"""Tests for rider-disjoint evaluation helpers."""

from __future__ import annotations

import unittest
import tempfile
import json
import hashlib
from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock, patch

import torch

from stgat_lstm.modeling.evaluate_model import (classification_metrics, rider_disjoint_folds,
    temporal_evidence_report, latest_only_history, evaluate_rider_disjoint,
    outgoing_choice_metrics, validation_threshold, fixed_rider_assignments,
    fixed_rider_partitions, saved_fixed_rider_assignments, evaluate_fixed_rider_split, run_real_refinement,
    refinement_overfitting_checks, audit_training_choice_inputs, select_fitting_examples,
    run_fitting_diagnostics, choice_breakdown, saved_grouped_rider_folds, run_grouped_kfold,
    select_training_recipe, _model_predictions)
from stgat_lstm.modeling.train_model import PreparedDecision, PreparedRoadChoice, train_decision_model
from stgat_lstm.modeling.model import ModelInput, DecisionPreferenceModel
from graph_test_data import build_test_graph_data


class DecisionEvaluationTests(unittest.TestCase):
    TRAFFIC_SCHEMA = ("congestion_normalized", "speed_ratio_to_reference", "congestion_observed",
                      "speed_observed", "traffic_age_scaled")

    def fixed_examples(self):
        graph = build_test_graph_data()
        inputs = graph.build_model_input("4")
        examples = [PreparedDecision(f"{rider}-{label}", f"rider-{rider}", label,
            ("2|4|0",), (PreparedRoadChoice("2", ("2|4|0", "2|3|0"),
                "2|3|0" if label else "2|4|0", "2|4|0", label),),
            None, None, inputs, ride_group=f"ride-{rider}-{label}")
            for rider in range(6) for label in (0, 1)]
        return examples, graph.model_configuration()

    def test_exact_deviation_choice_is_not_binary_detection(self):
        rows = [{"label": 1, "predicted_edge_id": "wrong-alternative", "chosen_edge_id": "actual",
                 "chosen_road_probability": .2, "probability_deviated": .9},
                {"label": 0, "predicted_edge_id": "suggested", "chosen_edge_id": "suggested",
                 "chosen_road_probability": .9, "probability_deviated": .1}]
        breakdown = choice_breakdown(rows)
        self.assertEqual(breakdown["deviated"]["correct"], 0)
        self.assertEqual(breakdown["followed"]["correct"], 1)
        self.assertEqual(breakdown["mean_class_choice_accuracy"], .5)
        self.assertEqual(classification_metrics(rows)["accuracy"], 1.)
        self.assertIsNone(choice_breakdown(rows[:1])["mean_class_choice_accuracy"])

    def test_nll_checkpoint_selection_can_choose_lower_loss_despite_lower_accuracy(self):
        torch.set_num_threads(1)
        examples, configuration = self.fixed_examples()
        scores = [{"accuracy": .8, "negative_log_likelihood": 1.2},
                  {"accuracy": .6, "negative_log_likelihood": .7}]
        summaries = {}
        for criterion in ("accuracy", "nll"):
            with patch("stgat_lstm.modeling.train_model._validation_choice_score", side_effect=scores):
                _, summary = train_decision_model(examples[:8], configuration,
                    epochs=2, validation=examples[8:], validation_interval=1, checkpoint_selection=criterion)
            summaries[criterion] = summary
        self.assertEqual(summaries["accuracy"].selected_epoch, 1)
        self.assertEqual(summaries["nll"].selected_epoch, 2)
        self.assertEqual(summaries["nll"].training_options["checkpoint_selection"], "nll")
        with self.assertRaisesRegex(ValueError, "Checkpoint selection"):
            train_decision_model(examples, configuration, checkpoint_selection="test_accuracy")

    def test_reduced_deviation_weight_uses_square_root_of_training_only_weight(self):
        torch.set_num_threads(1)
        examples, configuration = self.fixed_examples()
        expanded = examples + [replace(e, example_key=f"copy-{e.example_key}",
                                        ride_group=f"copy-{e.ride_group}") for e in examples if e.label == 0]
        _, normal = train_decision_model(expanded, configuration, epochs=1, balance_rides=True)
        _, reduced = train_decision_model(expanded, configuration, epochs=1,
                                         balance_rides=True, alternative_weight_power=.5)
        self.assertAlmostEqual(normal.deviated_choice_weight, 2.)
        self.assertAlmostEqual(reduced.deviated_choice_weight, 2. ** .5)
        self.assertEqual(normal.outgoing_choices, reduced.outgoing_choices)
        with self.assertRaisesRegex(ValueError, "automatic weights"):
            train_decision_model(examples, configuration, alternative_choice_weight=3., alternative_weight_power=.5)

    def test_recipe_selection_ignores_test_metrics(self):
        trials = [
            {"name": "better_validation", "validation": {"outgoing_choice": {"negative_log_likelihood": .4},
                "choice_breakdown": {"mean_class_choice_accuracy": .6}}, "test_accuracy": 0.},
            {"name": "better_test", "validation": {"outgoing_choice": {"negative_log_likelihood": .6},
                "choice_breakdown": {"mean_class_choice_accuracy": .9}}, "test_accuracy": 1.}]
        self.assertEqual(select_training_recipe(trials)["name"], "better_validation")
        with self.assertRaisesRegex(ValueError, "No training recipes"):
            select_training_recipe([])

    def test_training_ablation_reuses_folds_and_tests_only_selected_recipe(self):
        torch.set_num_threads(1)
        examples, configuration = self.fixed_examples()
        original = {"status": "approved_for_training", "synthetic": False,
                    "examples": [{"status": "approved", "example_key": e.example_key,
                                  "rider_group": e.rider_group} for e in examples]}
        graph = build_test_graph_data()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "original.json"
            source.write_text(json.dumps(original), encoding="utf-8")
            def prepared(graphml, node_features, corpus_path, **kwargs):
                return examples, configuration, {
                    "approved_artifact_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                    "graphml_sha256": hashlib.sha256(graphml.read_bytes()).hexdigest(),
                    "node_features_sha256": hashlib.sha256(node_features.read_bytes()).hexdigest()}
            with patch("stgat_lstm.preprocessing.build_decision_candidates.refine_real_decisions",
                       return_value={**original, "refinement": {}}) as refine, \
                 patch("stgat_lstm.modeling.evaluate_model.prepare_real", side_effect=prepared), \
                 patch("stgat_lstm.modeling.evaluate_model.build_real_graph_data", return_value=graph), \
                 patch("stgat_lstm.modeling.evaluate_model.GPSTrafficArchive"), \
                 patch("stgat_lstm.modeling.evaluate_model.attach_gps_traffic", side_effect=lambda rows,*a,**k:(rows,{})), \
                 patch("stgat_lstm.modeling.evaluate_model.train_decision_model", wraps=train_decision_model) as train, \
                 patch("stgat_lstm.modeling.evaluate_model._model_predictions", wraps=_model_predictions) as predict:
                baseline_path = folder / "baseline" / "report.json"
                baseline = run_grouped_kfold(source, source, source, folder, baseline_path,
                    epochs=2, architectures=("stgat_lstm",), progress=lambda _: None)
                original_bytes = {p: p.read_bytes() for p in baseline_path.parent.rglob("*") if p.is_file()}
                train.reset_mock()
                predict.reset_mock()
                refine.reset_mock()
                output = folder / "ablation" / "report.json"
                report = run_grouped_kfold(source, source, source, folder, output,
                    epochs=2, architectures=("stgat_lstm",), ablation_reference=baseline_path, progress=lambda _: None)
                refine.assert_not_called()
                self.assertEqual(train.call_count, 12)
                self.assertEqual(predict.call_count, 15)
                self.assertEqual(report["evaluation"], "real_training_ablation_grouped_kfold")
                self.assertEqual(report["model_fits"], 12)
                self.assertEqual(report["plan"]["folds"], baseline["plan"]["folds"])
                self.assertEqual((output.parent / "rider_folds.json").read_bytes(),
                                 (baseline_path.parent / "rider_folds.json").read_bytes())
                self.assertEqual((output.parent / "approved_training_examples.json").read_bytes(),
                                 (baseline_path.parent / "approved_training_examples.json").read_bytes())
                for index, fold in enumerate(report["folds"]):
                    result = fold["models"]["stgat_lstm"]
                    self.assertEqual(len(result["selection_trials"]), 3)
                    selected = select_training_recipe(result["selection_trials"])
                    self.assertEqual(result["selected_recipe"], selected["recipe"])
                    for call in train.call_args_list[index * 4:index * 4 + 3]:
                        self.assertEqual(call.kwargs["checkpoint_selection"], "nll")
                        self.assertEqual({e.rider_group for e in call.kwargs["validation"]}, set(fold["validation"]))
                    final_call = train.call_args_list[index * 4 + 3]
                    self.assertEqual(final_call.kwargs["epochs"], selected["training_summary"]["selected_epoch"])
                    self.assertEqual(final_call.kwargs["preference_weight"], selected["recipe"]["preference_weight"])
                    self.assertEqual(final_call.kwargs["alternative_weight_power"], selected["recipe"]["alternative_weight_power"])
                    for call in predict.call_args_list[index * 5:index * 5 + 3]:
                        self.assertEqual({e.rider_group for e in call.args[1]}, set(fold["validation"]))
                    self.assertEqual({e.rider_group for e in predict.call_args_list[index * 5 + 4].args[1]}, set(fold["test"]))
                for path, before in original_bytes.items():
                    self.assertEqual(path.read_bytes(), before)
                with self.assertRaisesRegex(ValueError, "separate output folder"):
                    run_grouped_kfold(source, source, source, folder, baseline_path,
                        epochs=2, ablation_reference=baseline_path, progress=lambda _: None)
                with self.assertRaisesRegex(ValueError, "retain the reference setting"):
                    run_grouped_kfold(source, source, source, folder, folder / "changed" / "report.json",
                        epochs=3, ablation_reference=baseline_path, progress=lambda _: None)

    def test_grouped_kfold_plan_is_frozen_and_covers_each_rider_once(self):
        examples, _ = self.fixed_examples()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            folds = saved_grouped_rider_folds(path, examples, "original-digest", 3, 17)
            self.assertEqual(folds, saved_grouped_rider_folds(path, list(reversed(examples)),
                                                            "original-digest", 3, 17))
            seen = []
            for fold in folds:
                groups = [set(fold[n]) for n in ("training", "validation", "test")]
                self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
                seen.extend(fold["test"])
            self.assertEqual(sorted(seen), sorted({e.rider_group for e in examples}))
            with self.assertRaisesRegex(ValueError, "different data/settings"):
                saved_grouped_rider_folds(path, examples, "changed-digest", 3, 17)
            document = json.loads(path.read_text())
            document["folds"][0]["test"] = document["folds"][0]["training"]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without overlap"):
                saved_grouped_rider_folds(path, examples, "original-digest", 3, 17)
        with self.assertRaisesRegex(ValueError, "at least two"):
            saved_grouped_rider_folds(Path("unused.json"), examples, "digest", 1, 17)

    def test_grouped_kfold_excludes_heldout_labels_and_traffic_and_pools_once(self):
        torch.set_num_threads(1)
        examples, configuration = self.fixed_examples()
        extras = [replace(e, example_key=f"extra-{e.example_key}") for e in examples if e.label == 0]
        original = {"status": "approved_for_training", "synthetic": False,
                    "examples": [{"status": "approved", "example_key": e.example_key,
                                  "rider_group": e.rider_group} for e in examples]}
        corrected = {**original, "refinement": {}, "examples": original["examples"] + [
            {"status": "approved", "example_key": e.example_key, "rider_group": e.rider_group,
             "quality": {"additional_training_choice": True}} for e in extras]}
        graph = build_test_graph_data()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            artifact = folder / "original.json"
            artifact.write_text(json.dumps(original), encoding="utf-8")
            with patch("stgat_lstm.preprocessing.build_decision_candidates.refine_real_decisions", return_value=corrected), \
                 patch("stgat_lstm.modeling.evaluate_model.prepare_real", return_value=(examples + extras, configuration, {})), \
                 patch("stgat_lstm.modeling.evaluate_model.build_real_graph_data", return_value=graph), \
                 patch("stgat_lstm.modeling.evaluate_model.GPSTrafficArchive"), \
                 patch("stgat_lstm.modeling.evaluate_model.attach_gps_traffic", side_effect=lambda rows,*a,**k:(rows,{})) as traffic, \
                 patch("stgat_lstm.modeling.evaluate_model.train_decision_model", wraps=train_decision_model) as train:
                report = run_grouped_kfold(artifact, artifact, artifact, folder,
                    folder / "run" / "report.json", epochs=1, progress=lambda _: None)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(train.call_count, 12)
            self.assertEqual(report["model_fits"], 12)
            self.assertEqual(report["deviation_choices"], 6)
            for index, fold in enumerate(report["folds"]):
                test_riders, val_riders = set(fold["test"]), set(fold["validation"])
                self.assertEqual(traffic.call_args_list[index * 2].kwargs["excluded_riders"], test_riders | val_riders)
                self.assertEqual(traffic.call_args_list[index * 2 + 1].kwargs["excluded_riders"], test_riders)
                for selection_call, final_call in zip(train.call_args_list[index * 4:index * 4 + 4:2],
                                                     train.call_args_list[index * 4 + 1:index * 4 + 4:2]):
                    self.assertEqual({e.rider_group for e in selection_call.args[0]}, set(fold["training"]))
                    self.assertTrue(any(e.example_key.startswith("extra-") for e in selection_call.args[0]))
                    self.assertFalse(any(e.example_key.startswith("extra-") for e in selection_call.kwargs["validation"]))
                    self.assertFalse(test_riders & {e.rider_group for e in final_call.args[0]})
                    self.assertNotIn("validation", final_call.kwargs)
                    self.assertEqual(final_call.kwargs["epochs"], 1)
                for model in fold["models"].values():
                    self.assertFalse(any(p["example_key"].startswith("extra-") for p in model["test"]["predictions"]))
                    if "final_training_summary" in model:
                        for prediction in model["test"]["predictions"]:
                            self.assertEqual(prediction["chosen_exit_training_choices"],
                                             4 if prediction["label"] else 8)
                            self.assertEqual(prediction["junction_training_choices"], 12)
            for model in report["models"].values():
                self.assertEqual(model["outgoing_choice"]["choices"], 12)
                self.assertEqual(model["choice_breakdown"]["deviated"]["choices"], 6)
                self.assertEqual(len({p["example_key"] for p in model["predictions"]}), 12)
            self.assertTrue((folder / "run" / "fold_1" / "stgat_lstm" / "checkpoint.pt").exists())

    def test_input_audit_detects_exact_conflicts_without_using_labels_as_features(self):
        examples, _ = self.fixed_examples()
        audit = audit_training_choice_inputs(examples[:2])
        self.assertEqual(audit["exact_conflicting_groups"], 1)
        self.assertEqual(audit["choices_in_exact_conflicts"], 2)
        self.assertEqual(audit["optimistic_accuracy_ceiling_from_exact_conflicts"], .5)
        self.assertEqual(len(audit["same_junction_destination_groups_with_different_choices"]), 1)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            select_fitting_examples(examples[:2], audit, size=2)

    def test_input_audit_distinguishes_destination_and_local_traffic_features(self):
        examples, _ = self.fixed_examples()
        first, second = examples[:2]
        for feature in ("destination", "traffic"):
            if feature == "destination":
                destination = second.inputs.destination_features + .1
                changed = replace(second.inputs, destination_features=destination)
            else:
                frame = second.inputs.temporal_edge_features[0] + .1
                changed = replace(second.inputs, temporal_edge_features=(frame,))
            audit = audit_training_choice_inputs([first, replace(second, inputs=changed)])
            self.assertEqual(audit["exact_conflicting_groups"], 0)
            self.assertEqual(audit["distinct_exact_input_groups"], 2)
            self.assertEqual(len(audit["same_junction_destination_groups_with_different_choices"]), 1)

    def test_input_audit_ignores_disconnected_traffic_outside_gat_receptive_field(self):
        examples, configuration = self.fixed_examples()
        base = examples[0].inputs
        count = len(base.node_features)
        expanded = replace(base,
            node_features=torch.cat((base.node_features, base.node_features.new_zeros(2, base.node_features.shape[1]))),
            edge_index=torch.cat((base.edge_index, torch.tensor([[count], [count + 1]])), dim=1),
            edge_static=torch.cat((base.edge_static, base.edge_static.new_zeros(1, base.edge_static.shape[1]))),
            destination_features=torch.cat((base.destination_features, base.destination_features.new_zeros(1, 2))),
            routable_mask=torch.cat((base.routable_mask, torch.ones(1, dtype=torch.bool))),
            edge_ids=base.edge_ids + ("far-a|far-b|0",),
            temporal_edge_features=tuple(torch.cat((frame, frame.new_zeros(1, frame.shape[1])))
                                         for frame in base.temporal_edge_features))
        changed = expanded.temporal_edge_features[0].clone()
        changed[-1, 0] = .5
        first = replace(examples[0], inputs=expanded)
        second = replace(examples[1], inputs=replace(expanded, temporal_edge_features=(changed,)))
        audit = audit_training_choice_inputs([first, second])
        self.assertEqual(audit["exact_conflicting_groups"], 1)
        model = DecisionPreferenceModel(**configuration).eval()
        with torch.no_grad():
            first_logits = model(first.inputs)[1]
            second_logits = model(second.inputs)[1]
        candidate_indices = [expanded.edge_ids.index(edge) for edge in first.choice_targets[0].candidate_edge_ids]
        torch.testing.assert_close(first_logits[candidate_indices], second_logits[candidate_indices], rtol=0., atol=1e-7)

    def test_fitting_subset_is_balanced_and_does_not_copy_ranking_targets(self):
        examples, _ = self.fixed_examples()
        unique = [replace(e, preferred_edge_ids=("2|3|0",), rejected_edge_ids=("2|4|0",),
                          inputs=replace(e.inputs, destination_features=e.inputs.destination_features + index * .1))
                  for index, e in enumerate(examples)]
        audit = audit_training_choice_inputs(unique)
        selected = select_fitting_examples(unique, audit)
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(e.choice_targets[0].deviated for e in selected), 4)
        self.assertTrue(all(e.preferred_edge_ids is None and e.rejected_edge_ids is None for e in selected))
        self.assertTrue(all(e.preferred_edge_ids is not None for e in unique))

    def test_fitting_workflow_uses_only_original_training_riders(self):
        torch.set_num_threads(1)
        templates, config = self.fixed_examples()
        examples = []
        for index, template in enumerate(templates):
            for repeat in range(4):
                examples.append(replace(template, example_key=f"{index}-{repeat}",
                    inputs=replace(template.inputs, destination_features=template.inputs.destination_features + (index * 4 + repeat) * .01)))
        assignments = fixed_rider_assignments(examples)
        graph = build_test_graph_data()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            artifact = folder / "approved.json"
            artifact.write_text(json.dumps({"status": "approved_for_training", "synthetic": False,
                                            "source_sha256": {}}), encoding="utf-8")
            split = folder / "split.json"
            split.write_text(json.dumps({"seed": 17, "assignments": assignments}), encoding="utf-8")
            (folder / "experiment_plan.json").write_text(json.dumps({"assignments": assignments,
                "original_split_sha256": hashlib.sha256(split.read_bytes()).hexdigest()}), encoding="utf-8")
            with patch("stgat_lstm.modeling.evaluate_model.prepare_real", return_value=(examples, config, {})), \
                 patch("stgat_lstm.modeling.evaluate_model.build_real_graph_data", return_value=graph), \
                 patch("stgat_lstm.modeling.evaluate_model.GPSTrafficArchive"), \
                 patch("stgat_lstm.modeling.evaluate_model.attach_gps_traffic", side_effect=lambda rows,*a,**k:(rows,{})) as traffic, \
                 patch("stgat_lstm.modeling.evaluate_model.train_decision_model", wraps=train_decision_model) as train:
                report = run_fitting_diagnostics(artifact, artifact, artifact, split, folder,
                                                folder / "diagnostic.json", epochs=1, progress=lambda _:None)
            self.assertEqual(train.call_count, 1)
            self.assertTrue(all(assignments[e.rider_group] == "training" for e in train.call_args.args[0]))
            self.assertNotIn("validation", train.call_args.kwargs)
            self.assertEqual(train.call_args.kwargs["preference_weight"], 0.)
            self.assertFalse(report["heldout_predictions_computed"])
            self.assertFalse(report["official_checkpoint_modified"])
            self.assertEqual(traffic.call_args.kwargs["excluded_riders"],
                             {r for r,g in assignments.items() if g != "training"})

    def test_refinement_selects_without_test_and_refits_development_riders(self):
        torch.set_num_threads(1)
        examples, configuration = self.fixed_examples()
        assignments = fixed_rider_assignments(examples)
        original = {"status": "approved_for_training", "synthetic": False,
                    "examples": [{"status": "approved", "example_key": e.example_key,
                                  "rider_group": e.rider_group} for e in examples], "refinement": {}}
        graph = build_test_graph_data()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            artifact = folder / "original.json"
            artifact.write_text(json.dumps(original), encoding="utf-8")
            split = folder / "split.json"
            split.write_text(json.dumps({"dataset_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                         "seed": 17, "assignments": assignments}), encoding="utf-8")
            with patch("stgat_lstm.preprocessing.build_decision_candidates.refine_real_decisions", return_value=original), \
                 patch("stgat_lstm.modeling.evaluate_model.prepare_real", return_value=(examples, configuration, {})), \
                 patch("stgat_lstm.modeling.evaluate_model.build_real_graph_data", return_value=graph), \
                 patch("stgat_lstm.modeling.evaluate_model.GPSTrafficArchive") as archive, \
                 patch("stgat_lstm.modeling.evaluate_model.attach_gps_traffic", side_effect=lambda rows,*a,**k:(rows,{})) as traffic, \
                 patch("stgat_lstm.modeling.evaluate_model.train_decision_model", wraps=train_decision_model) as train:
                archive.from_exports.return_value.summary = {}
                report = run_real_refinement(artifact, artifact, artifact, split, folder,
                                            folder / "report.json", epochs=1, progress=lambda _:None)
            self.assertEqual(train.call_count, 4)
            for call in train.call_args_list[:3]:
                self.assertEqual({e.rider_group for e in call.args[0]},
                                 {r for r,g in assignments.items() if g == "training"})
                self.assertEqual({e.rider_group for e in call.kwargs["validation"]},
                                 {r for r,g in assignments.items() if g == "validation"})
            self.assertEqual({e.rider_group for e in train.call_args_list[-1].args[0]},
                             {r for r,g in assignments.items() if g != "test"})
            self.assertNotIn("validation", train.call_args_list[-1].kwargs)
            self.assertEqual(traffic.call_args_list[0].kwargs["excluded_riders"],
                             {r for r,g in assignments.items() if g != "training"})
            self.assertEqual(traffic.call_args_list[1].kwargs["excluded_riders"],
                             {r for r,g in assignments.items() if g == "test"})
            self.assertEqual(report["model_fits"], 4)
            self.assertEqual(report["counts"]["test"]["choices"], 4)
            self.assertTrue((folder / "checkpoint.pt").exists())
            self.assertEqual(report["development"]["outgoing_choice"]["choices"], 8)
            checks = refinement_overfitting_checks(report)
            train_accuracy = report["development"]["outgoing_choice"]["accuracy"]
            test_accuracy = report["test"]["outgoing_choice"]["accuracy"]
            self.assertAlmostEqual(checks["final_refit"]["development_minus_test_accuracy"],
                                   train_accuracy - test_accuracy)
            self.assertFalse(checks["validation_after_refit_is_held_out"])

    def test_locked_first_visit_comparison_only_changes_original_training_targets(self):
        torch.set_num_threads(1)
        examples, configuration = self.fixed_examples()
        assignments = fixed_rider_assignments(examples)
        repeated = []
        for example in examples:
            if example.label == 0:
                target = example.choice_targets[0]
                repeat = replace(target, chosen_edge_id="2|3|0", suggested_edge_id="2|3|0")
                example = replace(example, choice_targets=(target, repeat))
            repeated.append(example)
        original = {"status": "approved_for_training", "synthetic": False,
                    "examples": [{"status": "approved", "example_key": e.example_key,
                                  "rider_group": e.rider_group} for e in repeated], "refinement": {}}
        graph = build_test_graph_data()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            artifact = folder / "original.json"
            artifact.write_text(json.dumps(original), encoding="utf-8")
            split = folder / "split.json"
            split.write_text(json.dumps({"dataset_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                         "seed": 17, "assignments": assignments}), encoding="utf-8")
            provenance = {"approved_artifact_sha256": hashlib.sha256((json.dumps(original, indent=2) + "\n").encode()).hexdigest(),
                          "graphml_sha256": "graph", "node_features_sha256": "features"}
            previous = {"status": "completed", "evaluation": "real_refined_rider_split",
                "plan": {"assignments": assignments, "seed": 17,
                    "original_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "original_split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
                    "regularization": {"learning_rate": .001, "weight_decay": .0001,
                        "label_smoothing": .05, "traffic_mask_probability": .15,
                        "balance_rides": True, "patience": 3, "validation_interval": 5}},
                "selected": {"preference_weight": .1, "epochs": 1, "threshold": .6},
                "provenance": {**provenance, "gps_source_sha256": {},
                    "traffic_policy": {"steps": 6, "interval_s": 300, "profile_bin_s": 300}}}
            previous_path = folder / "previous.json"
            previous_path.write_text(json.dumps(previous), encoding="utf-8")
            with patch("stgat_lstm.preprocessing.build_decision_candidates.refine_real_decisions", return_value=original), \
                 patch("stgat_lstm.modeling.evaluate_model.prepare_real", return_value=(repeated, configuration, provenance)), \
                 patch("stgat_lstm.modeling.evaluate_model.build_real_graph_data", return_value=graph), \
                 patch("stgat_lstm.modeling.evaluate_model.GPSTrafficArchive") as archive, \
                 patch("stgat_lstm.modeling.evaluate_model.attach_gps_traffic", side_effect=lambda rows,*a,**k:(rows,{})), \
                 patch("stgat_lstm.modeling.evaluate_model.train_decision_model", wraps=train_decision_model) as train:
                archive.from_exports.return_value.summary = {}
                report = run_real_refinement(artifact, artifact, artifact, split, folder,
                    folder / "report.json", epochs=1, progress=lambda _:None, first_visit_only=True,
                    reuse_selection_path=previous_path)
            self.assertEqual(train.call_count, 2)
            first = train.call_args_list[0]
            self.assertTrue(all(len(e.choice_targets) == 1 for e in first.args[0]))
            self.assertEqual(sum(len(e.choice_targets) for e in first.kwargs["validation"]), 6)
            self.assertFalse(first.kwargs["select_validation_checkpoint"])
            self.assertEqual(first.kwargs["preference_weight"], .1)
            self.assertEqual(report["selected"]["epochs"], 1)
            self.assertEqual(report["selected"]["threshold"], .6)
            self.assertEqual(report["counts"]["test"]["choices"], 6)
            self.assertEqual(report["counts"]["final_development"]["choices"], 10)
            self.assertEqual(report["counts"]["final_development_original_targets"]["choices"], 12)
            self.assertEqual(len(report["supervision_adjustment"]["selection_training"]["omitted_targets"]), 2)
            self.assertFalse(report["supervision_adjustment"]["validation_test_targets_modified"])

    def test_fixed_split_is_deterministic_and_keeps_complete_riders(self):
        examples, _ = self.fixed_examples()
        assignments = fixed_rider_assignments(examples)
        self.assertEqual(assignments, fixed_rider_assignments(list(reversed(examples))))
        groups = fixed_rider_partitions(examples, assignments)
        rider_sets = [{e.rider_group for e in group} for group in groups.values()]
        self.assertTrue(all(len(group) == 2 for group in rider_sets))
        self.assertFalse(rider_sets[0] & rider_sets[1] | rider_sets[0] & rider_sets[2] | rider_sets[1] & rider_sets[2])
        for group in groups.values():
            self.assertEqual({t.deviated for e in group for t in e.choice_targets}, {0, 1})
        with self.assertRaisesRegex(ValueError, "six riders"):
            fixed_rider_assignments(examples[:8])

    def test_saved_split_refuses_changed_data_or_seed(self):
        examples, _ = self.fixed_examples()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.json"
            assignments = saved_fixed_rider_assignments(path, examples, "data-hash", 17)
            self.assertEqual(assignments, saved_fixed_rider_assignments(path, examples, "data-hash", 17))
            for digest, seed in (("changed-hash", 17), ("data-hash", 18)):
                with self.assertRaisesRegex(ValueError, "different data or seed"):
                    saved_fixed_rider_assignments(path, examples, digest, seed)

    def test_fixed_comparison_fits_five_models_with_held_out_traffic_excluded(self):
        torch.set_num_threads(1)
        examples, configuration = self.fixed_examples()
        assignments = fixed_rider_assignments(examples)
        with patch("stgat_lstm.modeling.evaluate_model.attach_gps_traffic",
                   return_value=(examples, {"test": "coverage"})) as traffic:
            with patch("stgat_lstm.modeling.evaluate_model.train_decision_model", wraps=train_decision_model) as train:
                report = evaluate_fixed_rider_split(examples, configuration, assignments=assignments,
                    epochs=1, gps_archive=Mock(), temporal_ablation=True)
        held_out = {r for r, group in assignments.items() if group != "training"}
        self.assertEqual(traffic.call_args.kwargs["excluded_riders"], held_out)
        self.assertEqual(train.call_count, 5)
        for call in train.call_args_list:
            self.assertEqual({e.rider_group for e in call.args[0]},
                             {r for r, group in assignments.items() if group == "training"})
            self.assertEqual(call.kwargs["seed"], 17)
        self.assertEqual(train.call_args_list[-1].args[1], train.call_args_list[-2].args[1])
        self.assertEqual(report["training"]["model_fits"], 5)
        for result in report["models"].values():
            self.assertEqual(result["test"]["classification"]["threshold"], .5)
            self.assertEqual({row["rider_group"] for row in result["test"]["predictions"]},
                             {r for r, group in assignments.items() if group == "test"})

    def history_examples(self):
        inputs = ModelInput(torch.zeros((1, 1)), torch.zeros((2, 1), dtype=torch.long),
            torch.ones((1, 1)), (torch.tensor([[0., .2, 0., 1., .1]]),
                               torch.tensor([[0., .8, 0., 1., 0.]])), torch.zeros((1, 1)),
            torch.ones(1, dtype=torch.bool), ("edge",))
        return [PreparedDecision(str(i), f"rider-{i}", i % 2, ("edge",),
            (PreparedRoadChoice("0", ("edge",), "edge", "edge", 0),),
            None, None, inputs)
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

    def test_outgoing_choice_metrics_use_chosen_edge(self):
        predictions = [
            {"chosen_edge_id": "a", "predicted_edge_id": "a", "chosen_road_probability": .8},
            {"chosen_edge_id": "b", "predicted_edge_id": "c", "chosen_road_probability": .2},
        ]
        metrics = outgoing_choice_metrics(predictions)
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["mean_chosen_road_probability"], .5)
        self.assertGreater(metrics["negative_log_likelihood"], 0)

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
        with patch("stgat_lstm.modeling.evaluate_model.train_decision_model",
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
        with patch("stgat_lstm.modeling.evaluate_model.train_decision_model", return_value=(fake_model, None)) as train:
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
            routable_mask=torch.ones(1, dtype=torch.bool),
            edge_ids=("edge",),
        )

        def example(key: str, rider: str, label: int) -> PreparedDecision:
            return PreparedDecision(key, rider, label, ("edge",),
                                    (PreparedRoadChoice("0", ("edge",), "edge", "edge", 0),),
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
