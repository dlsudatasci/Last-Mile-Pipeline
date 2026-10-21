"""Evaluate rider choices with fixed splits, grouped k-fold, or rider holdouts.

Held-out riders are excluded from neural training and GPS traffic context.
The real dataset remains small, so a valid experiment is not by itself proof
of dependable generalization.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import torch

from stgat_lstm.modeling.train_model import (PreparedDecision, attach_gps_traffic, prepare_real,
                          train_decision_model, _save_training, first_junction_visits)
from stgat_lstm.modeling.model import path_cost
from stgat_lstm.preprocessing.graph_data import build_real_graph_data
from stgat_lstm.preprocessing.gps_traffic import GPSTrafficArchive


ARCHITECTURES = ("gcn", "gat", "gatv2", "stgat_lstm")


def rider_disjoint_folds(
    prepared: list[PreparedDecision],
) -> tuple[tuple[str, list[PreparedDecision], list[PreparedDecision]], ...]:
    """Return deterministic leave-one-rider-out train/test folds."""
    riders = sorted({example.rider_group for example in prepared})
    if len(riders) < 2:
        raise ValueError("Rider-disjoint evaluation requires at least two riders")
    folds = []
    for rider in riders:
        train = [example for example in prepared if example.rider_group != rider]
        test = [example for example in prepared if example.rider_group == rider]
        if not test:
            raise AssertionError("A rider fold unexpectedly has no test examples")
        labels = {example.label for example in train}
        if labels != {0, 1}:
            raise ValueError(
                f"Holding out rider {rider} leaves training data without both classes; "
                "collect more independent riders before evaluating this fold"
            )
        folds.append((rider, train, test))
    return tuple(folds)


def classification_metrics(predictions: list[dict], threshold: float = 0.5) -> dict:
    """Compute dependency-free binary metrics from held-out probabilities."""
    if not predictions:
        raise ValueError("No held-out predictions")
    if any(item["label"] not in (0, 1) for item in predictions):
        raise ValueError("Binary evaluation labels must be zero or one")
    labels = [int(item["label"]) for item in predictions]
    probabilities = [float(item["probability_deviated"]) for item in predictions]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Prediction probabilities must be finite and between zero and one")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be between zero and one")
    classes = [int(probability >= threshold) for probability in probabilities]
    tp = sum(predicted == 1 and label == 1 for predicted, label in zip(classes, labels))
    tn = sum(predicted == 0 and label == 0 for predicted, label in zip(classes, labels))
    fp = sum(predicted == 1 and label == 0 for predicted, label in zip(classes, labels))
    fn = sum(predicted == 0 and label == 1 for predicted, label in zip(classes, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    true_negative_rate = tn / (tn + fp) if tn + fp else 0.0
    epsilon = 1e-7
    log_loss = -sum(
        label * math.log(min(max(probability, epsilon), 1 - epsilon))
        + (1 - label) * math.log(min(max(1 - probability, epsilon), 1 - epsilon))
        for label, probability in zip(labels, probabilities)
    ) / len(labels)
    brier = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)

    positives = sum(labels)
    negatives = len(labels) - positives
    followed_precision = tn / (tn + fn) if tn + fn else 0.0
    followed_f1 = (2 * followed_precision * true_negative_rate / (followed_precision + true_negative_rate)
                   if followed_precision + true_negative_rate else 0.0)
    # Mann-Whitney interpretation of ROC-AUC; ties receive half credit.
    roc_auc = None
    if positives and negatives:
        grouped_scores: dict[float, list[int]] = {}
        for probability, label in zip(probabilities, labels):
            grouped_scores.setdefault(probability, [0, 0])[label] += 1
        below = 0
        concordance = 0.0
        for score in sorted(grouped_scores):
            n_negative, n_positive = grouped_scores[score]
            concordance += n_positive * (below + n_negative / 2)
            below += n_negative
        roc_auc = concordance / (positives * negatives)
    average_precision = None
    if positives:
        grouped: dict[float, list[int]] = {}
        for probability, label in zip(probabilities, labels):
            grouped.setdefault(probability, []).append(label)
        seen = 0
        seen_positives = 0
        previous_recall = 0.0
        average_precision = 0.0
        for probability in sorted(grouped, reverse=True):
            group = grouped[probability]
            seen += len(group)
            seen_positives += sum(group)
            recall_at_threshold = seen_positives / positives
            precision_at_threshold = seen_positives / seen
            average_precision += (recall_at_threshold - previous_recall) * precision_at_threshold
            previous_recall = recall_at_threshold
    return {
        "examples": len(labels),
        "threshold": threshold,
        "class_support": {"followed": negatives, "deviated": positives},
        "class_prevalence_deviated": positives / len(labels),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + true_negative_rate) / 2 if positives and negatives else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + followed_f1) / 2 if positives and negatives else None,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": brier,
        "log_loss": log_loss,
        "confusion": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
        "per_class": {
            "deviated": {"precision": precision, "recall": recall, "f1": f1, "support": positives},
            "followed": {"precision": followed_precision, "recall": true_negative_rate,
                         "f1": followed_f1, "support": negatives},
        },
    }


def outgoing_choice_metrics(predictions: list[dict]) -> dict:
    """Measure whether the highest-probability outgoing road was the rider's road."""
    if not predictions:
        raise ValueError("No outgoing-road predictions")
    correct = sum(item["predicted_edge_id"] == item["chosen_edge_id"] for item in predictions)
    chosen_probabilities = [float(item["chosen_road_probability"]) for item in predictions]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in chosen_probabilities):
        raise ValueError("Chosen-road probabilities must be finite and between zero and one")
    epsilon = 1e-7
    negative_log_likelihood = -sum(
        math.log(max(value, epsilon)) for value in chosen_probabilities
    ) / len(predictions)
    return {
        "choices": len(predictions),
        "correct": correct,
        "accuracy": correct / len(predictions),
        "mean_chosen_road_probability": sum(chosen_probabilities) / len(predictions),
        "negative_log_likelihood": negative_log_likelihood,
    }


def choice_breakdown(predictions: list[dict]) -> dict:
    """Keep exact alternative-road selection separate from binary detection."""
    result = {}
    for name, label in (("deviated", 1), ("followed", 0)):
        rows = [row for row in predictions if row["label"] == label]
        result[name] = (outgoing_choice_metrics(rows) if rows else
                        {"choices": 0, "correct": 0, "accuracy": None})
    accuracies = [result[name]["accuracy"] for name in ("deviated", "followed")]
    result["mean_class_choice_accuracy"] = (sum(accuracies) / 2
                                            if all(a is not None for a in accuracies) else None)
    return result


def _balanced_rider_groups(prepared: list[PreparedDecision], sizes: tuple[int, ...],
                           seed: int) -> list[list[str]]:
    """Balance complete riders using target counts, without fitting a model."""
    riders = sorted({e.rider_group for e in prepared})
    if sum(sizes) != len(riders) or min(sizes) < 1:
        raise ValueError("Rider group sizes must be positive and cover every rider")
    stats = {r: [sum(e.rider_group == r for e in prepared),
                 sum(t.deviated == 0 for e in prepared if e.rider_group == r for t in e.choice_targets),
                 sum(t.deviated == 1 for e in prepared if e.rider_group == r for t in e.choice_targets)]
             for r in riders}
    totals = [sum(v[i] for v in stats.values()) for i in range(3)]
    if not all(totals):
        raise ValueError("Grouped evaluation requires both followed and alternative-road targets")
    rng, best = random.Random(seed), None
    for _ in range(1000):
        order = riders.copy()
        rng.shuffle(order)
        groups, offset, score = [], 0, 0.
        for size in sizes:
            group = sorted(order[offset:offset + size])
            offset += size
            counts = [sum(stats[r][i] for r in group) for i in range(3)]
            if not all(counts):
                break
            score += sum((count / total - size / len(riders)) ** 2
                         for count, total in zip(counts, totals))
            groups.append(group)
        else:
            key = (score, tuple(tuple(g) for g in groups))
            if best is None or key < best[0]:
                best = key, groups
    if best is None:
        raise ValueError("Cannot retain both choice classes in complete rider groups; use fewer folds")
    return best[1]


def saved_grouped_rider_folds(path: Path, prepared: list[PreparedDecision],
                             dataset_sha256: str, n_splits: int, seed: int) -> list[dict]:
    """Freeze outer test riders and inner validation riders before training."""
    riders = {e.rider_group for e in prepared}
    if n_splits < 2 or len(riders) - math.ceil(len(riders) / n_splits) < 4:
        raise ValueError("Grouped k-fold needs at least two folds and four development riders per fold")
    targets = sorted((e.example_key, e.rider_group, t.source_node_id, t.chosen_edge_id,
                      t.suggested_edge_id, t.candidate_edge_ids, t.deviated)
                     for e in prepared for t in e.choice_targets)
    identity = {"dataset_sha256": dataset_sha256, "n_splits": n_splits, "seed": seed,
                "targets_sha256": hashlib.sha256(json.dumps(targets).encode()).hexdigest()}
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("strategy") != "balanced rider groups with inner rider validation" or any(
                document.get(k) != v for k, v in identity.items()):
            raise ValueError("Saved k-fold plan belongs to different data/settings; use a separate file")
        folds = document["folds"]
    else:
        sizes = tuple(len(riders) // n_splits + (i < len(riders) % n_splits) for i in range(n_splits))
        tests = _balanced_rider_groups(prepared, sizes, seed)
        folds = []
        for index, test in enumerate(tests):
            development = [e for e in prepared if e.rider_group not in test]
            count = len(riders) - len(test)
            n_validation = max(2, round(count * .2))
            train, validation = _balanced_rider_groups(
                development, (count - n_validation, n_validation), seed + index + 1)
            folds.append({"fold": index + 1, "training": train,
                          "validation": validation, "test": test})
        document = {"schema_version": 1, **identity,
                    "strategy": "balanced rider groups with inner rider validation",
                    "balance_search": "1000 seeded shuffles using choice counts only",
                    "model_scores_used": False, "folds": folds}
    seen = []
    if len(folds) != n_splits or sorted(fold["fold"] for fold in folds) != list(range(1, n_splits + 1)):
        raise ValueError("Saved fold count differs from the requested count")
    for fold in folds:
        partitions = {name: set(fold[name]) for name in ("training", "validation", "test")}
        if set.union(*partitions.values()) != riders or sum(map(len, partitions.values())) != len(riders):
            raise ValueError("K-fold partitions must cover each rider once without overlap")
        fixed_rider_partitions(prepared, {r: name for name, group in partitions.items() for r in group})
        seen.extend(fold["test"])
    if Counter(seen) != Counter({r: 1 for r in riders}):
        raise ValueError("Every rider must be held out exactly once")
    if not path.exists():
        _save_json(path, document)
    return folds


def validation_threshold(predictions: list[dict]) -> float:
    """Choose a balanced-accuracy threshold without consulting test labels."""
    probabilities = sorted({float(item["probability_deviated"]) for item in predictions})
    candidates = [0.0, 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(probabilities, probabilities[1:]))
    return max(
        candidates,
        key=lambda threshold: (
            classification_metrics(predictions, threshold)["balanced_accuracy"] or 0.0,
            classification_metrics(predictions, threshold)["f1"],
            -abs(threshold - 0.5),
        ),
    )


def latest_only_history(prepared: list[PreparedDecision]) -> list[PreparedDecision]:
    """Keep LSTM size and sequence length fixed while removing past information."""
    return [replace(example, inputs=replace(example.inputs,
                temporal_edge_features=(example.inputs.temporal_edge_features[-1],)
                * len(example.inputs.temporal_edge_features))) for example in prepared]


def temporal_evidence_report(prepared: list[PreparedDecision], schema: tuple[str, ...] | None) -> dict:
    """Describe observed traffic histories; age/missingness changes alone are not traffic variation."""
    report = {"examples": len(prepared), "examples_with_multiple_steps": 0,
              "examples_with_observed_history": 0, "examples_with_observed_value_changes": 0,
              "suggested_paths_with_observed_value_changes": 0, "per_example": []}
    if schema is None:
        return {**report, "status": "not_assessed", "reason": "Dynamic feature schema not supplied"}
    fields = [(schema.index(value), schema.index(mask)) for value, mask in
              (("congestion_normalized", "congestion_observed"),
               ("speed_ratio_to_reference", "speed_observed"))]
    for example in prepared:
        frames = torch.stack(example.inputs.temporal_edge_features)
        if frames.shape[-1] != len(schema):
            raise ValueError("Traffic feature schema does not match the prepared history")
        changed = torch.zeros(frames.shape[1], dtype=torch.bool)
        covered = torch.zeros(frames.shape[:2], dtype=torch.bool)
        for value, mask in fields:
            observed = frames[:, :, mask] == 1
            covered |= observed
            count = observed.sum(dim=0)
            low = frames[:, :, value].masked_fill(~observed, float("inf")).min(dim=0).values
            high = frames[:, :, value].masked_fill(~observed, float("-inf")).max(dim=0).values
            changed |= (count >= 2) & (high - low > 1e-6)
        lookup = {edge: row for row, edge in enumerate(example.inputs.edge_ids)}
        suggested_rows = [lookup[edge] for edge in example.suggested_edge_ids]
        path_changes = int(changed[suggested_rows].sum())
        report["examples_with_multiple_steps"] += int(len(frames) > 1)
        report["examples_with_observed_history"] += int(bool(covered.any()))
        report["examples_with_observed_value_changes"] += int(bool(changed.any()))
        report["suggested_paths_with_observed_value_changes"] += int(path_changes > 0)
        report["per_example"].append({"example_key": example.example_key, "steps": len(frames),
            "observed_edges_per_step": covered.sum(dim=1).tolist(),
            "edges_with_observed_value_changes": int(changed.sum()),
            "suggested_edges_with_observed_value_changes": path_changes})
    report["status"] = ("observed_history_variation_present" if report["examples_with_observed_value_changes"]
                        else "insufficient_temporal_evidence")
    report["interpretation"] = ("Coverage is a diagnostic, not proof of temporal benefit. "
        "Value changes are counted only when the same edge/field is observed at least twice; "
        "age and missingness changes alone do not qualify.")
    return report


def lstm_comparison(models: dict, evidence: dict, requested: bool) -> dict:
    """Paired metric differences, without declaring superiority from a pilot run."""
    full = models["stgat_lstm"]["classification"]
    comparisons = {}
    for name in ("gatv2", "stgat_lstm_latest_only"):
        if name not in models:
            continue
        control = models[name]["classification"]
        comparisons[name] = {metric: (full[metric] - control[metric]
            if full[metric] is not None and control[metric] is not None else None)
            for metric in ("balanced_accuracy", "f1", "macro_f1", "average_precision",
                           "roc_auc", "brier_score", "log_loss")}
        full_choice = models["stgat_lstm"]["outgoing_choice"]["accuracy"]
        control_choice = models[name]["outgoing_choice"]["accuracy"]
        comparisons[name]["outgoing_choice_accuracy"] = full_choice - control_choice
        full_rank = models["stgat_lstm"]["held_out_preference_ranking"]["accuracy"]
        control_rank = models[name]["held_out_preference_ranking"]["accuracy"]
        comparisons[name]["preference_ranking_accuracy"] = (
            full_rank - control_rank if full_rank is not None and control_rank is not None else None)
    return {"temporal_evidence": evidence,
            "latest_only_control": "run" if "stgat_lstm_latest_only" in models else
                ("skipped_no_observed_history_variation" if requested else "not_requested"),
            "full_minus_control": comparisons,
            "interpretation": ("Positive accuracy/F1/AP/AUC differences favor full history; negative "
                "Brier/log-loss differences favor full history. GATv2 comparison changes both architecture "
                "and history; latest-only repeats the final frame with identical LSTM parameter count "
                "and sequence length. Use repeated seeds and adequate held-out riders before claiming "
                "benefit. No observed variation means this run cannot establish temporal value.")}


def _choice_records(example: PreparedDecision, edge_logits: torch.Tensor) -> list[dict]:
    lookup = {edge_id: index for index, edge_id in enumerate(example.inputs.edge_ids)}
    records = []
    for target in example.choice_targets:
        candidate_logits = edge_logits[[lookup[edge_id] for edge_id in target.candidate_edge_ids]]
        probabilities = torch.softmax(candidate_logits, dim=0)
        predicted_position = int(probabilities.argmax())
        chosen_position = target.candidate_edge_ids.index(target.chosen_edge_id)
        suggested_position = target.candidate_edge_ids.index(target.suggested_edge_id)
        suggested_probability = float(probabilities[suggested_position])
        records.append({
            "rider_group": example.rider_group,
            "example_key": example.example_key,
            "source_node_id": target.source_node_id,
            "candidate_edge_ids": list(target.candidate_edge_ids),
            "chosen_edge_id": target.chosen_edge_id,
            "suggested_edge_id": target.suggested_edge_id,
            "predicted_edge_id": target.candidate_edge_ids[predicted_position],
            "chosen_road_probability": float(probabilities[chosen_position]),
            "suggested_road_probability": suggested_probability,
            "label": target.deviated,
            "probability_deviated": 1.0 - suggested_probability,
        })
    return records


def _model_predictions(model, examples: list[PreparedDecision]) -> tuple[list[dict], dict]:
    predictions = []
    ranking = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            costs, edge_logits = model(example.inputs)
            predictions.extend(_choice_records(example, edge_logits))
            if example.preferred_edge_ids is not None and example.rejected_edge_ids is not None:
                preferred = path_cost(costs, example.preferred_edge_ids, example.inputs.edge_ids)
                rejected = path_cost(costs, example.rejected_edge_ids, example.inputs.edge_ids)
                ranking.append(bool(preferred < rejected))
    return predictions, {
        "pairs": len(ranking),
        "correct": sum(ranking),
        "accuracy": sum(ranking) / len(ranking) if ranking else None,
    }


def fixed_rider_assignments(prepared: list[PreparedDecision], seed: int = 17) -> dict[str, str]:
    """Choose one group-disjoint 60/20/20 split using class balance, not scores.

    Riders, rather than junction rows, are the split unit. The fixed search
    balances label support and example counts before any models are fitted.
    See https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data
    """
    riders = sorted({example.rider_group for example in prepared})
    if len(riders) < 6:
        raise ValueError("A real train/validation/test rider split requires at least six riders")
    n_validation = max(2, round(len(riders) * 0.2))
    n_test = n_validation
    n_train = len(riders) - n_validation - n_test
    totals = [len(prepared),
              sum(target.deviated == 0 for e in prepared for target in e.choice_targets),
              sum(target.deviated == 1 for e in prepared for target in e.choice_targets)]
    if not all(totals):
        raise ValueError("Fixed split needs both followed and alternative-road targets")
    stats = {rider: [sum(e.rider_group == rider for e in prepared),
                    sum(t.deviated == 0 for e in prepared if e.rider_group == rider for t in e.choice_targets),
                    sum(t.deviated == 1 for e in prepared if e.rider_group == rider for t in e.choice_targets)]
             for rider in riders}
    rng = random.Random(seed)
    best = None
    for _ in range(1000):
        order = riders.copy()
        rng.shuffle(order)
        partitions = {"training": order[:n_train],
                      "validation": order[n_train:n_train + n_validation],
                      "test": order[n_train + n_validation:]}
        score = 0.0
        for group in partitions.values():
            counts = [sum(stats[rider][column] for rider in group) for column in range(3)]
            if not all(counts):
                break
            fraction = len(group) / len(riders)
            score += sum((count / total - fraction) ** 2 for count, total in zip(counts, totals))
        else:
            assignment = {rider: name for name, group in partitions.items() for rider in group}
            key = (score, tuple(assignment[rider] for rider in riders))
            if best is None or key < best[0]:
                best = (key, assignment)
    if best is None:
        raise ValueError("Cannot split complete riders while retaining both choice classes in each partition")
    return best[1]


def fixed_rider_partitions(prepared: list[PreparedDecision], assignments: dict[str, str]) -> dict:
    if set(assignments) != {e.rider_group for e in prepared}:
        raise ValueError("Fixed assignments must cover exactly the approved riders")
    if set(assignments.values()) != {"training", "validation", "test"}:
        raise ValueError("Fixed assignments require training, validation and test groups")
    partitions = {name: [e for e in prepared if assignments[e.rider_group] == name]
                  for name in ("training", "validation", "test")}
    for name, examples in partitions.items():
        if {t.deviated for e in examples for t in e.choice_targets} != {0, 1}:
            raise ValueError(f"Fixed {name} partition must contain both road-choice classes")
    return partitions


def saved_fixed_rider_assignments(path: Path, prepared: list[PreparedDecision],
                                 dataset_sha256: str, seed: int) -> dict[str, str]:
    """Freeze assignments before fitting; changed data require a separate split."""
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("dataset_sha256") != dataset_sha256 or document.get("seed") != seed:
            raise ValueError("Saved split belongs to different data or seed; use a separate split file")
        assignments = document["assignments"]
    else:
        assignments = fixed_rider_assignments(prepared, seed)
        fixed_rider_partitions(prepared, assignments)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": 1, "dataset_sha256": dataset_sha256, "seed": seed,
                    "strategy": "1000 seeded group shuffles; balance class support and examples only",
                    "model_scores_used": False, "assignments": assignments}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    fixed_rider_partitions(prepared, assignments)
    return assignments


def _suggestion_baseline(examples: list[PreparedDecision], prevalence: float) -> list[dict]:
    records = []
    for example in examples:
        for target in example.choice_targets:
            followed_probability = 1.0 - prevalence
            chosen_probability = (followed_probability if target.chosen_edge_id == target.suggested_edge_id
                                  else prevalence / (len(target.candidate_edge_ids) - 1))
            records.append({"rider_group": example.rider_group, "example_key": example.example_key,
                            "source_node_id": target.source_node_id,
                            "candidate_edge_ids": list(target.candidate_edge_ids),
                            "chosen_edge_id": target.chosen_edge_id,
                            "suggested_edge_id": target.suggested_edge_id,
                            "predicted_edge_id": target.suggested_edge_id,
                            "chosen_road_probability": chosen_probability,
                            "suggested_road_probability": followed_probability,
                            "probability_deviated": prevalence, "label": target.deviated})
    return records


def _fixed_prediction_report(predictions: list[dict], ranking: dict, threshold: float) -> dict:
    per_rider = []
    for rider in sorted({r["rider_group"] for r in predictions}):
        rows = [r for r in predictions if r["rider_group"] == rider]
        per_rider.append({"rider_group": rider, "outgoing_choice": outgoing_choice_metrics(rows),
                          "classification": classification_metrics(rows)})
    return {"outgoing_choice": outgoing_choice_metrics(predictions),
            "classification": classification_metrics(predictions),
            "classification_at_validation_threshold": classification_metrics(predictions, threshold),
            "preference_ranking": ranking, "predictions": predictions, "per_rider": per_rider,
            "macro_rider_choice_accuracy": sum(r["outgoing_choice"]["accuracy"] for r in per_rider) / len(per_rider)}


def evaluate_fixed_rider_split(
    prepared: list[PreparedDecision], model_configuration: dict, *, assignments: dict[str, str],
    epochs: int = 100, learning_rate: float = 0.005, preference_weight: float = 1.0,
    seed: int = 17, batch_size: int = 8, temporal_ablation: bool = True,
    traffic_schema: tuple[str, ...] | None = None, gps_archive: GPSTrafficArchive | None = None,
    history_steps: int = 6, history_interval_s: int = 300, gps_profile_bin_s: int = 300,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Fit each variant once, with frozen settings and independently held-out riders."""
    partitions = fixed_rider_partitions(prepared, assignments)
    excluded_riders = {rider for rider, name in assignments.items() if name != "training"}
    traffic_reports = None
    if gps_archive is not None:
        prepared, traffic_reports = attach_gps_traffic(
            prepared, gps_archive, excluded_riders=excluded_riders, history_steps=history_steps,
            history_interval_s=history_interval_s, gps_profile_bin_s=gps_profile_bin_s)
        partitions = fixed_rider_partitions(prepared, assignments)
    split = {}
    for name, examples in partitions.items():
        split[name] = {"riders": sorted({e.rider_group for e in examples}),
                       "examples": len(examples), "rides": len({e.ride_group for e in examples}),
                       "outgoing_choices": sum(len(e.choice_targets) for e in examples),
                       "followed_choices": sum(t.deviated == 0 for e in examples for t in e.choice_targets),
                       "alternative_choices": sum(t.deviated == 1 for e in examples for t in e.choice_targets),
                       "ranking_pairs": sum(e.preferred_edge_ids is not None for e in examples)}
    if progress:
        progress("Fixed rider split | " + " | ".join(
            f"{name}: {len(s['riders'])} riders, {s['examples']} examples, {s['outgoing_choices']} choices"
            for name, s in split.items()))
    prevalence = (split["training"]["alternative_choices"] / split["training"]["outgoing_choices"])
    models = {"suggestion_baseline": {}}
    for name, examples in partitions.items():
        models["suggestion_baseline"][name] = _fixed_prediction_report(
            _suggestion_baseline(examples, prevalence), {"pairs": 0, "correct": 0, "accuracy": None}, 0.5)
    evidence = {name: temporal_evidence_report(examples, traffic_schema)
                for name, examples in partitions.items()}
    architectures = list(ARCHITECTURES) + (["stgat_lstm_latest_only"] if temporal_ablation else [])
    for architecture in architectures:
        if progress:
            progress(f"Fixed rider split | training {architecture} once for {epochs} epochs")
        controlled = architecture == "stgat_lstm_latest_only"
        groups = {name: latest_only_history(examples) if controlled else examples
                  for name, examples in partitions.items()}
        configuration = {**model_configuration, "architecture": "stgat_lstm" if controlled else architecture}
        model, summary = train_decision_model(
            groups["training"], configuration, epochs=epochs, learning_rate=learning_rate,
            preference_weight=preference_weight, seed=seed, batch_size=batch_size,
            progress=(lambda epoch, loss, name=architecture: progress(
                f"  {name} epoch {epoch}/{epochs}: loss {loss:.4f}")
                if epoch == 1 or epoch == epochs or epoch % max(epochs // 4, 1) == 0 else None) if progress else None)
        validation_predictions, validation_ranking = _model_predictions(model, groups["validation"])
        threshold = validation_threshold(validation_predictions)
        result = {"training_summary": asdict(summary), "model_configuration": configuration,
                  "threshold_selection": {"source": "validation riders only", "threshold": threshold,
                                          "test_labels_used": False, "primary_threshold": 0.5}}
        result["validation"] = _fixed_prediction_report(validation_predictions, validation_ranking, threshold)
        for name in ("training", "test"):
            predictions, ranking = _model_predictions(model, groups[name])
            result[name] = _fixed_prediction_report(predictions, ranking, threshold)
        models[architecture] = result
    return {"schema_version": 5, "evaluation": "real_fixed_rider_split",
            "examples": len(prepared), "outgoing_choices": sum(len(e.choice_targets) for e in prepared),
            "riders": len(assignments), "split": split, "assignments": assignments,
            "training": {"epochs": epochs, "learning_rate": learning_rate, "preference_weight": preference_weight,
                         "seed": seed, "batch_size": batch_size, "alternative_choice_weight": "training inverse frequency",
                         "model_fits": len(architectures), "hyperparameter_tuning": False},
            "models": models, "temporal_evidence": evidence, "traffic_reports": traffic_reports,
            "metric_protocol": {"primary": "test outgoing-road argmax accuracy and negative log likelihood",
                                "binary_primary_threshold": 0.5,
                                "secondary_threshold": "selected using validation riders only",
                                "aggregation": "pooled junction counts and separate macro rider accuracy",
                                "test_used_for_training_or_tuning": False},
            "traffic_input_protocol": "All validation/test riders, target ride, and future records excluded from GPS histories",
            "interpretation": "One fixed real rider-held-out pilot. Test riders are unseen during fitting, but roads can overlap. "
                              "Small test support and a single seed limit reliability claims. Do not repeatedly tune against these test scores."}


def evaluate_synthetic_fixed_split(
    prepared: list[PreparedDecision],
    model_configuration: dict,
    *,
    epochs: int = 10,
    learning_rate: float = 0.005,
    preference_weight: float = 1.0,
    seed: int = 17,
    batch_size: int = 8,
    traffic_schema: tuple[str, ...] | None = None,
    gps_archive: GPSTrafficArchive | None = None,
    history_steps: int = 6,
    history_interval_s: int = 60,
    gps_profile_bin_s: int = 60,
    split_assignments: dict[str, str] | None = None,
    tune_hyperparameters: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Capacity check with rider-disjoint validation tuning and untouched testing."""
    by_rider: dict[str, list[PreparedDecision]] = {}
    for example in prepared:
        by_rider.setdefault(example.rider_group, []).append(example)
    class_bearing_riders = sorted(rider for rider, examples in by_rider.items()
                                  if {example.label for example in examples} == {0, 1})
    if len(class_bearing_riders) < 6:
        raise ValueError(
            "Synthetic fixed split requires at least six riders with both followed and deviated examples"
        )
    if split_assignments:
        missing = set(by_rider) - set(split_assignments)
        invalid = set(split_assignments.values()) - {"training", "validation", "test"}
        if missing or invalid:
            raise ValueError("Synthetic split assignments are incomplete or invalid")
        training_riders = {rider for rider in by_rider if split_assignments[rider] == "training"}
        validation_riders = {rider for rider in by_rider if split_assignments[rider] == "validation"}
        test_riders = {rider for rider in by_rider if split_assignments[rider] == "test"}
    else:
        validation_riders = set(class_bearing_riders[-4:-2])
        test_riders = set(class_bearing_riders[-2:])
        training_riders = set(by_rider) - validation_riders - test_riders
    if not training_riders or not validation_riders or not test_riders:
        raise ValueError("Synthetic split requires nonempty training, validation, and test rider groups")
    held_out_riders = validation_riders | test_riders

    traffic_reports = None
    if gps_archive is not None:
        prepared, traffic_reports = attach_gps_traffic(
            prepared,
            gps_archive,
            excluded_riders=held_out_riders,
            history_steps=history_steps,
            history_interval_s=history_interval_s,
            gps_profile_bin_s=gps_profile_bin_s,
        )
    train = [example for example in prepared if example.rider_group in training_riders]
    validation = [example for example in prepared if example.rider_group in validation_riders]
    test = [example for example in prepared if example.rider_group in test_riders]
    for name, examples in (("training", train), ("validation", validation), ("test", test)):
        if {example.label for example in examples} != {0, 1}:
            raise ValueError(f"Synthetic {name} split must contain both route-level classes")

    training_targets = [target.deviated for example in train for target in example.choice_targets]
    prevalence = sum(training_targets) / len(training_targets)

    def prevalence_predictions(examples: list[PreparedDecision]) -> list[dict]:
        predictions = []
        for example in examples:
            for target in example.choice_targets:
                predictions.append({
                    "rider_group": example.rider_group,
                    "example_key": example.example_key,
                    "source_node_id": target.source_node_id,
                    "candidate_edge_ids": list(target.candidate_edge_ids),
                    "chosen_edge_id": target.chosen_edge_id,
                    "suggested_edge_id": target.suggested_edge_id,
                    "predicted_edge_id": target.suggested_edge_id,
                    "chosen_road_probability": 1.0 / len(target.candidate_edge_ids),
                    "suggested_road_probability": 1.0 - prevalence,
                    "label": target.deviated,
                    "probability_deviated": prevalence,
                })
        return predictions

    models = {
        "prevalence": {
            "validation": {"classification": classification_metrics(prevalence_predictions(validation)),
                           "outgoing_choice": outgoing_choice_metrics(prevalence_predictions(validation))},
            "test": {"classification": classification_metrics(prevalence_predictions(test)),
                     "outgoing_choice": outgoing_choice_metrics(prevalence_predictions(test))},
        }
    }
    evidence = temporal_evidence_report(test, traffic_schema)

    def train_architecture(architecture: str, alternative_weight: float | None):
        controlled = architecture == "stgat_lstm_latest_only"
        configuration = {
            **model_configuration,
            "architecture": "stgat_lstm" if controlled else architecture,
        }
        training_examples = latest_only_history(train) if controlled else train
        validation_examples = latest_only_history(validation) if controlled else validation
        test_examples = latest_only_history(test) if controlled else test
        model, summary = train_decision_model(
            training_examples,
            configuration,
            epochs=epochs,
            learning_rate=learning_rate,
            preference_weight=preference_weight,
            alternative_choice_weight=alternative_weight,
            seed=seed,
            batch_size=batch_size,
        )
        validation_predictions, validation_ranking = _model_predictions(model, validation_examples)
        threshold = validation_threshold(validation_predictions)
        return model, summary, validation_examples, test_examples, validation_predictions, validation_ranking, threshold

    def model_report(model, validation_examples, test_examples, validation_predictions,
                     validation_ranking, threshold: float) -> dict:
        test_predictions, test_ranking = _model_predictions(model, test_examples)
        return {
            "threshold_selection": {
                "source": "validation balanced accuracy",
                "threshold": threshold,
                "test_labels_used": False,
            },
            "validation": {
                "classification": classification_metrics(validation_predictions, threshold),
                "outgoing_choice": outgoing_choice_metrics(validation_predictions),
                "preference_ranking": validation_ranking,
            },
            "test": {
                "classification": classification_metrics(test_predictions, threshold),
                "outgoing_choice": outgoing_choice_metrics(test_predictions),
                "preference_ranking": test_ranking,
                "predictions": test_predictions,
            },
        }

    tuning = {
        "enabled": tune_hyperparameters,
        "selection_data": "validation riders only",
        "test_labels_used_for_selection": False,
        "objective": "mean of validation outgoing-choice accuracy and balanced accuracy",
        "trials": [],
    }
    selected_weight: float | None = None
    learned_models = {}
    if tune_hyperparameters:
        best = None
        for requested_weight in (1.0, 2.0, 3.0, None):
            label = "automatic inverse frequency" if requested_weight is None else f"{requested_weight:.1f}"
            if progress is not None:
                progress(f"Synthetic validation tuning | stgat_lstm | alternative weight {label}")
            trained = train_architecture("stgat_lstm", requested_weight)
            model, summary, validation_examples, test_examples, predictions, ranking, threshold = trained
            choice = outgoing_choice_metrics(predictions)
            classification = classification_metrics(predictions, threshold)
            score = (choice["accuracy"] + classification["balanced_accuracy"]) / 2
            trial = {
                "requested_alternative_choice_weight": requested_weight,
                "actual_alternative_choice_weight": summary.deviated_choice_weight,
                "validation_score": score,
                "validation_outgoing_choice_accuracy": choice["accuracy"],
                "validation_balanced_accuracy": classification["balanced_accuracy"],
                "validation_deviation_f1": classification["f1"],
                "validation_choice_negative_log_likelihood": choice["negative_log_likelihood"],
                "selected": False,
            }
            tuning["trials"].append(trial)
            selection_key = (score, classification["f1"], choice["accuracy"],
                             -choice["negative_log_likelihood"])
            if best is None or selection_key > best[0]:
                best = (selection_key, trained, trial)
        assert best is not None
        _, selected_run, selected_trial = best
        selected_trial["selected"] = True
        selected_weight = selected_trial["actual_alternative_choice_weight"]
        model, _, validation_examples, test_examples, predictions, ranking, threshold = selected_run
        learned_models["stgat_lstm"] = model_report(
            model, validation_examples, test_examples, predictions, ranking, threshold
        )
        tuning["selected"] = {
            "alternative_choice_weight": selected_weight,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "preference_weight": preference_weight,
        }

    architectures = ["gatv2"]
    if not tune_hyperparameters:
        architectures.append("stgat_lstm")
    if evidence["examples_with_observed_value_changes"] > 0:
        architectures.append("stgat_lstm_latest_only")
    for architecture in architectures:
        if progress is not None:
            progress(f"Synthetic final comparison | training {architecture}")
        trained = train_architecture(architecture, selected_weight)
        model, _, validation_examples, test_examples, predictions, ranking, threshold = trained
        learned_models[architecture] = model_report(
            model, validation_examples, test_examples, predictions, ranking, threshold
        )
    for name in ("gatv2", "stgat_lstm", "stgat_lstm_latest_only"):
        if name in learned_models:
            models[name] = learned_models[name]
    return {
        "schema_version": 3,
        "evaluation": "controlled_synthetic_fixed_rider_split",
        "split": {
            "training_riders": sorted(training_riders),
            "validation_riders": sorted(validation_riders),
            "test_riders": sorted(test_riders),
            "training_examples": len(train),
            "validation_examples": len(validation),
            "test_examples": len(test),
        },
        "training": {
            "epochs": epochs, "learning_rate": learning_rate,
            "preference_weight": preference_weight, "seed": seed, "batch_size": batch_size,
            "alternative_choice_weight": selected_weight,
        },
        "hyperparameter_tuning": tuning,
        "models": models,
        "temporal_evidence": evidence,
        "traffic_reports": traffic_reports,
        "traffic_input_protocol": (
            "Historical profiles exclude all validation and test riders"
            if gps_archive is not None else "No GPS traffic archive supplied"
        ),
        "interpretation": (
            "Controlled synthetic capacity check. Test riders are absent from model training and traffic "
            "profiles, but repeated artificial patterns are not evidence of real-rider generalization."
        ),
    }


def evaluate_rider_disjoint(
    prepared: list[PreparedDecision],
    model_configuration: dict,
    *,
    epochs: int = 40,
    learning_rate: float = 0.005,
    preference_weight: float = 1.0,
    seed: int = 17,
    temporal_ablation: bool = False,
    traffic_schema: tuple[str, ...] | None = None,
    gps_archive: GPSTrafficArchive | None = None,
    history_steps: int = 6,
    history_interval_s: int = 300,
    gps_profile_bin_s: int = 300,
    batch_size: int = 8,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Compare simple and graph models on leave-one-rider-out predictions."""
    folds = rider_disjoint_folds(prepared)
    evidence_examples = prepared
    if gps_archive is not None:
        evidence_examples = []
        for example in prepared:
            attached, _ = attach_gps_traffic(
                [example], gps_archive,
                excluded_riders={example.rider_group},
                history_steps=history_steps,
                history_interval_s=history_interval_s,
                gps_profile_bin_s=gps_profile_bin_s,
            )
            evidence_examples.extend(attached)
    evidence = temporal_evidence_report(evidence_examples, traffic_schema)
    architectures = list(ARCHITECTURES)
    if temporal_ablation and evidence["examples_with_observed_value_changes"] > 0:
        architectures.append("stgat_lstm_latest_only")
    all_predictions: dict[str, list[dict]] = {"prevalence": []}
    ranking: dict[str, list[bool]] = {}
    fold_traffic = {}
    for architecture in architectures:
        all_predictions[architecture] = []
        ranking[architecture] = []

    for fold_index, (rider, train, test) in enumerate(folds):
        if gps_archive is not None:
            train, train_reports = attach_gps_traffic(
                train, gps_archive,
                excluded_riders={rider},
                history_steps=history_steps,
                history_interval_s=history_interval_s,
                gps_profile_bin_s=gps_profile_bin_s,
            )
            test, test_reports = attach_gps_traffic(
                test, gps_archive,
                excluded_riders={rider},
                history_steps=history_steps,
                history_interval_s=history_interval_s,
                gps_profile_bin_s=gps_profile_bin_s,
            )
            fold_traffic[rider] = {"training": train_reports, "test": test_reports}
        training_targets = [target.deviated for example in train for target in example.choice_targets]
        training_prevalence = sum(training_targets) / len(training_targets)
        for example in test:
            for target in example.choice_targets:
                all_predictions["prevalence"].append({
                    "rider_group": rider,
                    "example_key": example.example_key,
                    "source_node_id": target.source_node_id,
                    "candidate_edge_ids": list(target.candidate_edge_ids),
                    "chosen_edge_id": target.chosen_edge_id,
                    "suggested_edge_id": target.suggested_edge_id,
                    "predicted_edge_id": target.suggested_edge_id,
                    "chosen_road_probability": 1.0 / len(target.candidate_edge_ids),
                    "suggested_road_probability": 1.0 - training_prevalence,
                    "label": target.deviated,
                    "probability_deviated": training_prevalence,
                })

        for architecture in architectures:
            if progress is not None:
                progress(
                    f"Fold {fold_index + 1}/{len(folds)} | held-out rider {rider} | {architecture}"
                )
            controlled = architecture == "stgat_lstm_latest_only"
            configuration = {**model_configuration, "architecture": "stgat_lstm" if controlled else architecture}
            training_examples = latest_only_history(train) if controlled else train
            test_examples = latest_only_history(test) if controlled else test
            model, _ = train_decision_model(
                training_examples,
                configuration,
                epochs=epochs,
                learning_rate=learning_rate,
                preference_weight=preference_weight,
                seed=seed + fold_index,
                batch_size=batch_size,
            )
            model.eval()
            with torch.no_grad():
                for example in test_examples:
                    costs, edge_logits = model(example.inputs)
                    all_predictions[architecture].extend(_choice_records(example, edge_logits))
                    if example.preferred_edge_ids is not None and example.rejected_edge_ids is not None:
                        preferred = path_cost(costs, example.preferred_edge_ids, example.inputs.edge_ids)
                        rejected = path_cost(costs, example.rejected_edge_ids, example.inputs.edge_ids)
                        ranking[architecture].append(bool(preferred < rejected))

    models = {}
    for name, predictions in all_predictions.items():
        model_report = {
            "classification": classification_metrics(predictions),
            "outgoing_choice": outgoing_choice_metrics(predictions),
            "predictions": predictions,
        }
        if name != "prevalence":
            outcomes = ranking[name]
            model_report["held_out_preference_ranking"] = {
                "pairs": len(outcomes),
                "correct": sum(outcomes),
                "accuracy": sum(outcomes) / len(outcomes) if outcomes else None,
            }
        models[name] = model_report

    return {
        "schema_version": 4,
        "evaluation": "leave-one-rider-out",
        "examples": len(prepared),
        "outgoing_choices": sum(len(example.choice_targets) for example in prepared),
        "riders": len(folds),
        "folds": [
            {"held_out_rider": rider, "training_examples": len(train), "test_examples": len(test),
             "training_outgoing_choices": sum(len(item.choice_targets) for item in train),
             "test_outgoing_choices": sum(len(item.choice_targets) for item in test)}
            for rider, train, test in folds
        ],
        "training": {
            "epochs_per_fold": epochs,
            "learning_rate": learning_rate,
            "preference_weight": preference_weight,
            "base_seed": seed,
            "batch_size": batch_size,
        },
        "models": models,
        "metric_protocol": {"primary": "outgoing_choice_accuracy", "secondary":
            ["f1", "macro_f1", "average_precision", "roc_auc", "brier_score", "log_loss"],
            "routing_proxy": "held_out_preference_ranking", "threshold": 0.5,
            "aggregation": "pooled out-of-fold junction choices; riders with more choices contribute more",
            "zero_division": "undefined class precision/F1 is zero; two-class metrics are null if a class is absent",
            "tuning": "do not choose hyperparameters or threshold using these held-out predictions"},
        "lstm_comparison": lstm_comparison(models, evidence, temporal_ablation),
        "gps_traffic_fold_provenance": fold_traffic if gps_archive is not None else None,
        "traffic_input_protocol": (
            "GPS profiles are rebuilt per fold using only pre-decision records and excluding the held-out rider"
            if gps_archive is not None else "traffic inputs supplied during preparation"
        ),
        "interpretation": (
            "Held-out-rider experiment; assess sample size and class balance before claiming generalization. "
            "Inspect provenance traffic_history for actual pre-decision coverage. Unknown traffic, "
            "even across multiple history steps, cannot establish a temporal LSTM benefit. "
            "These same folds must not also be used for hyperparameter selection."
        ),
    }


def _save_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def refinement_overfitting_checks(report: dict) -> dict:
    """Compare clean choice metrics; optimization loss has a different scale."""
    trials = []
    for trial in report["trials"]:
        summary = trial["training_summary"]
        training_accuracy = summary["in_sample_outgoing_choice_correct"] / summary["outgoing_choices"]
        validation_accuracy = trial["validation"]["outgoing_choice"]["accuracy"]
        trials.append({"preference_weight": trial["preference_weight"],
            "selected_epoch": trial["selected_epoch"],
            "training_choice_accuracy": training_accuracy,
            "validation_choice_accuracy": validation_accuracy,
            "training_minus_validation_accuracy": training_accuracy - validation_accuracy,
            "validation_history": summary["validation_history"],
            "training_objective_history": summary["epoch_losses"]})
    development = report["development"]["outgoing_choice"]
    test = report["test"]["outgoing_choice"]
    return {"selection_trials": trials,
        "final_refit": {"development_choice_accuracy": development["accuracy"],
            "test_choice_accuracy": test["accuracy"],
            "development_minus_test_accuracy": development["accuracy"] - test["accuracy"],
            "development_clean_choice_nll": development["negative_log_likelihood"],
            "test_clean_choice_nll": test["negative_log_likelihood"],
            "test_minus_development_nll": test["negative_log_likelihood"] - development["negative_log_likelihood"]},
        "loss_comparison": "Clean unweighted choice NLL uses the same definition on development and test. "
            "The smoothed, weighted optimization loss also includes ranking and must not be compared directly with validation NLL.",
        "validation_after_refit_is_held_out": False,
        "interpretation": "Positive accuracy/NLL gaps can signal overfitting or a distribution difference. "
            "Small or negative gaps do not prove that overfitting is absent. Inspect per-rider and alternative-road performance. "
            "Three previously inspected test riders and one seed support an internal pilot, not a fresh independent validation."}


def audit_training_choice_inputs(prepared: list[PreparedDecision]) -> dict:
    """Find exact input aliases for the current two-layer GATv2 road head."""
    if not prepared:
        raise ValueError("Input audit requires training examples")
    base = prepared[0].inputs
    incoming = defaultdict(list)
    for index, (source, destination) in enumerate(base.edge_index.t().tolist()):
        incoming[destination].append((source, index))
    edge_lookup = {edge: i for i, edge in enumerate(base.edge_ids)}
    scopes, exact, coarse = {}, defaultdict(list), defaultdict(list)

    def digest_tensors(tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            value = tensor.detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    for example_index, example in enumerate(prepared):
        inputs = example.inputs
        if inputs.edge_ids != base.edge_ids or any(not torch.equal(getattr(inputs, field), getattr(base, field))
                for field in ("node_features", "edge_index", "edge_static", "routable_mask")):
            raise ValueError("Input audit expects one shared static road graph")
        for target_index, target in enumerate(example.choice_targets):
            candidates = target.candidate_edge_ids
            if candidates not in scopes:
                indices = [edge_lookup[edge] for edge in candidates]
                nodes = set(base.edge_index[:, indices].reshape(-1).tolist())
                frontier, edges = set(nodes), set(indices)
                # Each GAT layer aggregates incoming neighbors. Including their
                # incident attributes also covers the mean-filled self loops.
                for _ in range(2):
                    previous = set()
                    for node in frontier:
                        for source, edge in incoming[node]:
                            edges.add(edge)
                            previous.add(source)
                    nodes.update(previous)
                    frontier = previous
                node_rows, edge_rows = sorted(nodes), sorted(edges)
                static = digest_tensors((base.node_features[node_rows], base.edge_index[:, edge_rows],
                                         base.edge_static[edge_rows], base.routable_mask[indices]))
                scopes[candidates] = indices, edge_rows, static
            indices, edge_rows, static = scopes[candidates]
            signature = static + digest_tensors((inputs.destination_features[indices],
                *(frame[edge_rows] for frame in inputs.temporal_edge_features)))
            row = {"example_index": example_index, "target_index": target_index,
                   "example_key": example.example_key, "rider_group": example.rider_group,
                   "ride_group": example.ride_group, "source_node_id": target.source_node_id,
                   "destination_node_id": example.destination_node_id,
                   "chosen_edge_id": target.chosen_edge_id, "deviated": target.deviated}
            exact[(target.source_node_id, candidates, signature)].append(row)
            coarse[(target.source_node_id, example.destination_node_id, candidates)].append(row)

    conflicts = [rows for rows in exact.values() if len({r["chosen_edge_id"] for r in rows}) > 1]
    unanimous = [rows for rows in exact.values() if len({r["chosen_edge_id"] for r in rows}) == 1]
    coarse_conflicts = [rows for rows in coarse.values() if len({r["chosen_edge_id"] for r in rows}) > 1]
    total = sum(len(rows) for rows in exact.values())
    upper_correct = sum(max(Counter(r["chosen_edge_id"] for r in rows).values()) for rows in exact.values())
    return {"examples": len(prepared), "riders": len({e.rider_group for e in prepared}),
        "choices": total, "distinct_exact_input_groups": len(exact),
        "exact_conflicting_groups": len(conflicts), "choices_in_exact_conflicts": sum(map(len, conflicts)),
        "exact_conflict_groups": conflicts, "unanimous_input_groups": unanimous,
        "optimistic_accuracy_ceiling_from_exact_conflicts": upper_correct / total,
        "same_junction_destination_groups_with_different_choices": coarse_conflicts,
        "comparison": "Exact float values in the two incoming GAT hops of all candidate endpoints, "
            "candidate destination features, and all temporal frames. No rounding, labels, rider identity, or future metadata in signatures.",
        "interpretation": "Exact conflicts are differing valid or erroneous choices that this model input cannot distinguish. "
            "They are not automatically label errors. Coarse junction/destination conflicts can have different traffic inputs. "
            "No exact conflicts does not establish sufficient context; tiny numeric differences may carry little useful information. "
            "The ceiling concerns these observed training rows, not attainable test accuracy."}


def select_fitting_examples(prepared: list[PreparedDecision], audit: dict, size: int = 8) -> list[PreparedDecision]:
    """Select distinct unanimous training contexts with both choice types."""
    if size < 2 or size % 2:
        raise ValueError("Fitting subset size must be a positive even number")
    pool = [(group_index, row) for group_index, rows in enumerate(audit["unanimous_input_groups"]) for row in rows]
    selected, used, riders, sources = [], set(), Counter(), Counter()
    for label in (1, 0):
        for _ in range(size // 2):
            candidates = [(index, row) for index, row in pool if index not in used and row["deviated"] == label]
            if not candidates:
                raise ValueError("Not enough distinct unanimous contexts of each choice type")
            group_index, row = min(candidates, key=lambda pair: (riders[pair[1]["rider_group"]],
                sources[pair[1]["source_node_id"]], pair[1]["example_key"], pair[1]["target_index"]))
            example = prepared[row["example_index"]]
            target = example.choice_targets[row["target_index"]]
            selected.append(replace(example, choice_targets=(target,), preferred_edge_ids=None, rejected_edge_ids=None))
            used.add(group_index)
            riders[row["rider_group"]] += 1
            sources[row["source_node_id"]] += 1
    return selected


def run_fitting_diagnostics(graphml: Path, node_features: Path, approved_path: Path,
                           split_path: Path, gps_data: Path, output: Path, *, epochs=80,
                           seed=17, history_steps=6, history_interval_s=300,
                           gps_profile_bin_s=300, progress=print) -> dict:
    """Training-only conflict audit and one deliberate small-subset fit."""
    document = json.loads(approved_path.read_text(encoding="utf-8"))
    if document.get("synthetic") or document.get("status") != "approved_for_training":
        raise ValueError("Fitting diagnostics require the approved real artifact")
    frozen = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = frozen["assignments"]
    plan = json.loads(approved_path.with_name("experiment_plan.json").read_text(encoding="utf-8"))
    if (plan["original_split_sha256"] != hashlib.sha256(split_path.read_bytes()).hexdigest()
            or plan["assignments"] != assignments or frozen["seed"] != seed):
        raise ValueError("Diagnostics must reuse the refinement's original split")
    for name, expected in document["source_sha256"].items():
        if hashlib.sha256((gps_data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"GPS source differs from the approved artifact: {name}")
    prepared, configuration, provenance = prepare_real(graphml, node_features, approved_path,
        history_steps=history_steps, history_interval_s=history_interval_s)
    training = [e for e in prepared if assignments[e.rider_group] == "training"]
    data = build_real_graph_data(graphml, node_features, approved_decisions=approved_path)
    archive = GPSTrafficArchive.from_exports(data, gps_data)
    training, traffic = attach_gps_traffic(training, archive,
        excluded_riders={r for r, group in assignments.items() if group != "training"},
        history_steps=history_steps, history_interval_s=history_interval_s, gps_profile_bin_s=gps_profile_bin_s)
    audit = audit_training_choice_inputs(training)
    subset = select_fitting_examples(training, audit)
    configuration = {**configuration, "architecture": "stgat_lstm"}
    options = {"epochs": epochs, "learning_rate": .005, "preference_weight": 0.,
               "alternative_choice_weight": 1., "batch_size": len(subset), "seed": seed,
               "weight_decay": 0., "label_smoothing": 0., "traffic_mask_probability": 0., "balance_rides": False}
    report = {"schema_version": 1, "status": "fitting_in_progress", "purpose": "training capacity diagnostic only",
        "audit": audit, "options": options,
        "subset": [{"example_key": e.example_key, "rider_group": e.rider_group,
                    "source_node_id": e.choice_targets[0].source_node_id,
                    "chosen_edge_id": e.choice_targets[0].chosen_edge_id,
                    "deviated": e.choice_targets[0].deviated} for e in subset],
        "criteria_before_fitting": {"choice_accuracy": 1., "choice_nll_at_most": .1},
        "provenance": {**provenance, "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "excluded_riders": sorted(r for r, group in assignments.items() if group != "training"),
            "traffic": traffic, "source_code_sha256": {
                p.relative_to(Path(__file__).resolve().parents[1]).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))}},
        "heldout_predictions_computed": False, "official_checkpoint_modified": False,
        "interpretation": "Intentional fitting of eight real training choices; not generalization evidence, "
            "a replacement checkpoint, or an estimate of full-dataset accuracy."}
    _save_json(output, report)
    progress(f"Training audit: {audit['choices']} choices, {audit['exact_conflicting_groups']} exact conflicting groups")
    progress(f"One fitting check: {len(subset)} choices, {epochs} epochs maximum, unchanged GATv2-LSTM")
    model, summary = train_decision_model(subset, configuration, **options,
        progress=lambda epoch, loss: progress(f"  fitting epoch {epoch}: loss {loss:.4f}")
            if epoch == 1 or epoch % 10 == 0 else None)
    predictions, ranking = _model_predictions(model, subset)
    report.update({"status": "completed", "training_summary": asdict(summary),
                   "fitting_result": _fixed_prediction_report(predictions, ranking, .5)})
    choice = report["fitting_result"]["outgoing_choice"]
    report["capacity_criterion_met"] = choice["accuracy"] == 1. and choice["negative_log_likelihood"] <= .1
    _save_json(output, report)
    progress(f"Saved {output}: fitted {choice['correct']}/{choice['choices']} training roads | NLL {choice['negative_log_likelihood']:.4f}")
    return report


def select_training_recipe(trials: list[dict]) -> dict:
    """Select using inner validation only, never outer test predictions."""
    if not trials:
        raise ValueError("No training recipes were evaluated")
    return min(trials, key=lambda trial: (
        trial["validation"]["outgoing_choice"]["negative_log_likelihood"],
        -(trial["validation"]["choice_breakdown"]["mean_class_choice_accuracy"] or 0.)))


def _training_ablation_reference(reference_path: Path, output: Path, original_bytes: bytes,
                                 graphml: Path, node_features: Path, gps_data: Path,
                                 settings: dict) -> tuple[dict, dict, Path]:
    """Reuse the completed experiment's corpus and groups without modifying it."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("status") != "completed" or reference.get("evaluation") != "real_grouped_kfold":
        raise ValueError("Training ablation requires a completed original grouped k-fold report")
    if output.resolve().is_relative_to(reference_path.parent.resolve()):
        raise ValueError("Training ablation must use a separate output folder outside the reference run")
    plan, provenance = reference["plan"], reference["provenance"]
    if provenance["original_artifact_sha256"] != hashlib.sha256(original_bytes).hexdigest():
        raise ValueError("Training ablation original approved decisions differ from the reference")
    for key, path in (("graphml_sha256", graphml), ("node_features_sha256", node_features)):
        if provenance[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Training ablation input changed: {key}")
    actual_gps = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(gps_data.glob("*.csv"))}
    if actual_gps != provenance["gps_source_sha256"]:
        raise ValueError("Training ablation GPS exports differ from the reference")
    for key in ("seed", "epochs_maximum", "batch_size", "preference_weight", "regularization"):
        if plan[key] != settings[key]:
            raise ValueError(f"Training ablation must retain the reference setting: {key}")
    if len(plan["folds"]) != settings["n_splits"]:
        raise ValueError("Training ablation must retain the reference fold count")
    policy = provenance["traffic_policy"]
    if (policy["steps"], policy["interval_s"], policy["profile_bin_s"]) != settings["history"]:
        raise ValueError("Training ablation traffic history differs from the reference")
    corpus_path = reference_path.parent / "approved_training_examples.json"
    if hashlib.sha256(corpus_path.read_bytes()).hexdigest() != provenance["approved_artifact_sha256"]:
        raise ValueError("Reference prepared corpus has changed")
    split_path = reference_path.parent / "rider_folds.json"
    if json.loads(split_path.read_text(encoding="utf-8"))["folds"] != plan["folds"]:
        raise ValueError("Reference rider folds have changed")
    return reference, json.loads(corpus_path.read_text(encoding="utf-8")), split_path


def run_grouped_kfold(graphml: Path, node_features: Path, original_path: Path,
                      gps_data: Path, output: Path, *, n_splits=3, epochs=60,
                      learning_rate=.001, preference_weight=1., seed=17, batch_size=8,
                      history_steps=6, history_interval_s=300, gps_profile_bin_s=300,
                      architectures=("gatv2", "stgat_lstm"), temporal_ablation=False,
                      split_path: Path | None = None, progress=print,
                      ablation_reference: Path | None = None) -> dict:
    """Evaluate every original real choice once with nested rider separation.

    Additional verified follows are recovered by the same rule for all riders,
    but enter fitting only. Validation and test retain the original decisions.
    Ranking weight and regularization are fixed; inner validation selects epochs.
    """
    from stgat_lstm.preprocessing.build_decision_candidates import refine_real_decisions
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    if (original.get("synthetic") or original.get("status") != "approved_for_training"
            or original.get("refinement") or any(e.get("quality", {}).get("additional_training_choice")
                                                for e in original.get("examples", []))):
        raise ValueError("K-fold requires the original approved real artifact, before split-specific refinement")
    if not architectures or len(set(architectures)) != len(architectures) or set(architectures) - set(ARCHITECTURES):
        raise ValueError("Choose distinct supported architectures")
    if temporal_ablation and "stgat_lstm" not in architectures:
        raise ValueError("The latest-only control requires stgat_lstm")
    folder = output.parent
    options = {"learning_rate": learning_rate, "weight_decay": .0001, "label_smoothing": .05,
               "traffic_mask_probability": .15, "balance_rides": True,
               "patience": 3, "validation_interval": 5}
    original_keys = {e["example_key"] for e in original["examples"] if e["status"] == "approved"}
    riders = {e["rider_group"] for e in original["examples"] if e["status"] == "approved"}
    reference = None
    if ablation_reference is not None:
        if temporal_ablation:
            raise ValueError("Keep the training ablation separate from temporal-control experiments")
        reference, corrected, frozen_split = _training_ablation_reference(
            ablation_reference, output, original_bytes, graphml, node_features, gps_data,
            {"n_splits": n_splits, "seed": seed, "epochs_maximum": epochs, "batch_size": batch_size,
             "preference_weight": preference_weight, "regularization": options,
             "history": (history_steps, history_interval_s, gps_profile_bin_s)})
        if split_path is not None and split_path.resolve() != frozen_split.resolve():
            raise ValueError("Training ablation must use the completed reference's rider folds")
        split_path = frozen_split
        if not set(architectures).issubset(reference["models"]):
            raise ValueError("Requested ablation architecture is absent from the reference")
        progress("Reusing the completed real corpus and frozen rider folds; no new map matching")
    else:
        progress("Preparing real choices once for all riders; recovered follows will be training-only")
        corrected = refine_real_decisions(gps_data, graphml, original,
                                         {r: "training" for r in riders}, progress=progress)
    corrected["examples"] = [e for e in corrected["examples"] if e["example_key"] in original_keys
                             or e.get("quality", {}).get("additional_training_choice")]
    if not original_keys.issubset({e["example_key"] for e in corrected["examples"] if e["status"] == "approved"}):
        raise ValueError("Uniform preparation removed an original approved decision")
    corrected["refinement"]["additional_choices_partition"] = "eligible fitting riders only in each fold"
    corrected.setdefault("summary", {}).update({
        "candidate_examples": len(corrected["examples"]),
        "candidate_riders": len({e["rider_group"] for e in corrected["examples"]}),
        "label_counts": dict(Counter(e["target"]["label"] for e in corrected["examples"] if "target" in e))})
    corpus_path = folder / "approved_training_examples.json"
    if reference is None:
        _save_json(corpus_path, corrected)
    else:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_bytes((ablation_reference.parent / "approved_training_examples.json").read_bytes())
    prepared, configuration, provenance = prepare_real(
        graphml, node_features, corpus_path, history_steps=history_steps, history_interval_s=history_interval_s)
    evaluation_rows = [e for e in prepared if e.example_key in original_keys]
    split_path = split_path or folder / "rider_folds.json"
    folds = saved_grouped_rider_folds(split_path, evaluation_rows,
                                    hashlib.sha256(original_bytes).hexdigest(), n_splits, seed)
    if reference is not None:
        split_copy = folder / "rider_folds.json"
        split_copy.write_bytes(split_path.read_bytes())
        split_path = split_copy
    graph_data = build_real_graph_data(graphml, node_features, approved_decisions=corpus_path)
    archive = GPSTrafficArchive.from_exports(graph_data, gps_data)
    recipes = [{"name": "current_losses", "preference_weight": preference_weight,
                "alternative_weight_power": 1., "checkpoint_selection": "accuracy"}]
    if reference is not None:
        recipes = [
            {"name": "current_losses_nll", "preference_weight": preference_weight,
             "alternative_weight_power": 1., "checkpoint_selection": "nll"},
            {"name": "choice_only_nll", "preference_weight": 0.,
             "alternative_weight_power": 1., "checkpoint_selection": "nll"},
            {"name": "choice_reduced_weight_nll", "preference_weight": 0.,
             "alternative_weight_power": .5, "checkpoint_selection": "nll"}]
    variants = {name: name for name in architectures}
    if temporal_ablation:
        variants["stgat_lstm_latest_only"] = "stgat_lstm"
    provenance.update({"original_artifact_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "gps_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(gps_data.glob("*.csv"))},
        "traffic": "fold-specific pre-decision rider GPS; target ride and held-out riders excluded",
        "traffic_policy": {"steps": history_steps, "interval_s": history_interval_s,
                           "profile_bin_s": gps_profile_bin_s, "future_records_allowed": False,
                           "target_ride_allowed": False, "held_out_riders_allowed": False},
        "evaluation": "pooled original decisions; each rider held out once; recovered follows fit-only"})
    plan = {"schema_version": 1, "folds": folds, "split_file": str(split_path), "seed": seed,
            "architectures": list(variants), "epochs_maximum": epochs, "batch_size": batch_size,
            "preference_weight": preference_weight, "regularization": options,
            "checkpoint_selection": "inner-validation exact-road accuracy, then lower choice NLL",
            "first_visit_only": "fitting examples only; original validation/test targets unchanged",
            "expected_model_fits": n_splits * len(variants) * (len(recipes) + 1),
            "source_code_sha256": {p.relative_to(Path(__file__).resolve().parents[1]).as_posix():
                                   hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))},
            "provenance": provenance}
    if reference is not None:
        plan.update({"recipes": recipes,
            "checkpoint_selection": "inner-validation lower choice NLL; accuracy breaks ties",
            "recipe_selection": "lower inner-validation choice NLL; mean class exact-choice accuracy breaks ties",
            "ablation_reference": {"path": str(ablation_reference),
                "report_sha256": hashlib.sha256(ablation_reference.read_bytes()).hexdigest(),
                "split_sha256": hashlib.sha256((ablation_reference.parent / "rider_folds.json").read_bytes()).hexdigest()},
            "outer_test_used_for_selection": False})
    _save_json(folder / "experiment_plan.json", plan)
    report = {"schema_version": 1, "evaluation": "real_grouped_kfold", "status": "in_progress",
              "plan": plan, "examples": len(evaluation_rows), "riders": len(riders),
              "outgoing_choices": sum(len(e.choice_targets) for e in evaluation_rows),
              "deviation_choices": sum(t.deviated for e in evaluation_rows for t in e.choice_targets),
              "folds": [], "models": {}, "model_fits": 0, "provenance": provenance,
              "interpretation": "Exploratory internal rider-grouped cross-validation on previously inspected data. "
                                "No new independent riders or untouched external test set; no synthetic test rows."}
    if reference is not None:
        report["evaluation"] = "real_training_ablation_grouped_kfold"
    _save_json(output, report)
    progress(f"{n_splits} folds, {len(variants)} variants, {plan['expected_model_fits']} fits "
             f"({len(recipes)} inner recipe(s) + development refit); "
             f"{report['deviation_choices']} held-out deviations in total")
    history_options = dict(history_steps=history_steps, history_interval_s=history_interval_s,
                           gps_profile_bin_s=gps_profile_bin_s)
    for fold in folds:
        index = fold["fold"]
        train_riders, validation_riders, test_riders = (set(fold[n]) for n in ("training", "validation", "test"))
        training = [e for e in prepared if e.rider_group in train_riders]
        validation = [e for e in evaluation_rows if e.rider_group in validation_riders]
        test = [e for e in evaluation_rows if e.rider_group in test_riders]
        selection_rows, selection_traffic = attach_gps_traffic(
            training + validation, archive, excluded_riders=validation_riders | test_riders, **history_options)
        selection_train, selection_adjustment = first_junction_visits(
            [e for e in selection_rows if e.rider_group in train_riders])
        selection_validation = [e for e in selection_rows if e.rider_group in validation_riders]
        development = [e for e in prepared if e.rider_group not in test_riders]
        final_rows, final_traffic = attach_gps_traffic(
            development + test, archive, excluded_riders=test_riders, **history_options)
        final_train, final_adjustment = first_junction_visits(
            [e for e in final_rows if e.rider_group not in test_riders])
        final_test = [e for e in final_rows if e.rider_group in test_riders]
        fold_report = {**fold, "models": {}, "training_supervision": selection_adjustment,
                       "development_supervision": final_adjustment,
                       "selection_traffic": selection_traffic, "final_traffic": final_traffic}
        for name, architecture in variants.items():
            progress(f"Fold {index}/{n_splits} | {name} | selecting epoch on separate validation riders")
            train, val, development_rows, test_rows = selection_train, selection_validation, final_train, final_test
            if name.endswith("latest_only"):
                train, val, development_rows, test_rows = map(latest_only_history, (train, val, development_rows, test_rows))
            config = {**configuration, "architecture": architecture}
            def epoch_progress(epoch, loss):
                if epoch == 1 or epoch % 5 == 0:
                    progress(f"  epoch {epoch}: loss {loss:.4f}")
            trials = []
            fit_folder = folder / f"fold_{index}" / name
            for recipe in recipes:
                progress(f"  recipe {recipe['name']} | checkpoint selection {recipe['checkpoint_selection']}")
                recipe_options = {k: recipe[k] for k in (
                    "preference_weight", "alternative_weight_power", "checkpoint_selection")}
                selection_model, selection_summary = train_decision_model(
                    train, config, epochs=epochs, seed=seed,
                    batch_size=batch_size, validation=val, progress=epoch_progress, **options, **recipe_options)
                validation_predictions, validation_ranking = _model_predictions(selection_model, val)
                threshold = validation_threshold(validation_predictions)
                validation_report = _fixed_prediction_report(validation_predictions, validation_ranking, threshold)
                validation_report["choice_breakdown"] = choice_breakdown(validation_predictions)
                trials.append({"recipe": recipe, "training_summary": asdict(selection_summary),
                               "threshold": threshold, "validation": validation_report})
                breakdown = validation_report["choice_breakdown"]
                progress(f"  validation: NLL {validation_report['outgoing_choice']['negative_log_likelihood']:.4f} "
                         f"| deviation {breakdown['deviated']['correct']}/{breakdown['deviated']['choices']} "
                         f"| followed {breakdown['followed']['correct']}/{breakdown['followed']['choices']}")
                if reference is not None:
                    _save_json(fit_folder / "selection_report.json", {"status": "selection_in_progress",
                        "trials": trials, "outer_test_predictions_computed": False})
            selected = select_training_recipe(trials)
            selected_recipe = selected["recipe"]
            selected_epoch = selected["training_summary"]["selected_epoch"]
            threshold = selected["threshold"]
            progress(f"Fold {index}/{n_splits} | {name} | refitting development riders "
                     f"for {selected_epoch} fixed epochs | {selected_recipe['name']}")
            if reference is not None:
                _save_json(fit_folder / "selection_report.json", {"status": "completed",
                    "trials": trials, "selected_recipe": selected_recipe["name"],
                    "selected_epoch": selected_epoch, "outer_test_predictions_computed": False})
            model, summary = train_decision_model(
                development_rows, config, epochs=selected_epoch,
                seed=seed, batch_size=batch_size, progress=epoch_progress, **options,
                **{k: selected_recipe[k] for k in ("preference_weight", "alternative_weight_power", "checkpoint_selection")})
            development_predictions, development_ranking = _model_predictions(model, development_rows)
            predictions, ranking = _model_predictions(model, test_rows)
            exit_counts = Counter(t.chosen_edge_id for e in development_rows for t in e.choice_targets)
            source_counts = Counter(t.source_node_id for e in development_rows for t in e.choice_targets)
            for prediction in predictions:
                prediction["chosen_exit_training_choices"] = exit_counts[prediction["chosen_edge_id"]]
                prediction["junction_training_choices"] = source_counts[prediction["source_node_id"]]
            _save_training(model, summary, {**provenance, "fold": fold,
                "selected_epoch": selected_epoch, "binary_threshold": threshold,
                "selected_recipe": selected_recipe},
                fit_folder / "checkpoint.pt", fit_folder / "training_report.json")
            result = {"selection_training_summary": selected["training_summary"],
                      "final_training_summary": asdict(summary), "validation": selected["validation"],
                      "development": _fixed_prediction_report(development_predictions, development_ranking, threshold),
                      "test": _fixed_prediction_report(predictions, ranking, threshold)}
            if reference is not None:
                result.update({"selection_trials": trials, "selected_recipe": selected_recipe})
            result["test"]["choice_breakdown"] = choice_breakdown(predictions)
            result["development"]["choice_breakdown"] = choice_breakdown(development_predictions)
            fold_report["models"][name] = result
            breakdown = result["test"]["choice_breakdown"]
            progress(f"  held-out exact choices: deviation {breakdown['deviated']['correct']}/"
                     f"{breakdown['deviated']['choices']} | followed {breakdown['followed']['correct']}/"
                     f"{breakdown['followed']['choices']}")
            report["model_fits"] += len(recipes) + 1
            _save_json(fit_folder / "evaluation_report.json", result)
        targets = [t.deviated for e in final_train for t in e.choice_targets]
        baseline_predictions = _suggestion_baseline(final_test, sum(targets) / len(targets))
        fold_report["models"]["suggestion_baseline"] = {"test": _fixed_prediction_report(
            baseline_predictions, {"pairs": 0, "correct": 0, "accuracy": None}, .5)}
        report["folds"].append(fold_report)
        _save_json(output, report)
    expected = Counter((e.example_key, t.source_node_id, t.chosen_edge_id, t.suggested_edge_id)
                       for e in evaluation_rows for t in e.choice_targets)
    for name in (*variants, "suggestion_baseline"):
        tests = [f["models"][name]["test"] for f in report["folds"]]
        predictions = [p for test in tests for p in test["predictions"]]
        if Counter((p["example_key"], p["source_node_id"], p["chosen_edge_id"], p["suggested_edge_id"])
                   for p in predictions) != expected:
            raise AssertionError("Pooled predictions must cover every original road choice exactly once")
        pairs = sum(test["preference_ranking"]["pairs"] for test in tests)
        correct = sum(test["preference_ranking"]["correct"] for test in tests)
        pooled = _fixed_prediction_report(predictions,
            {"pairs": pairs, "correct": correct, "accuracy": correct / pairs if pairs else None}, .5)
        pooled["choice_breakdown"] = choice_breakdown(predictions)
        pooled["fold_choice_accuracies"] = [test["outgoing_choice"]["accuracy"] for test in tests]
        pooled["mean_fold_choice_accuracy"] = sum(pooled["fold_choice_accuracies"]) / n_splits
        pooled.pop("classification_at_validation_threshold")
        report["models"][name] = pooled
    if reference is not None:
        report["reference_comparison"] = {name: {
            "previous": {key: reference["models"][name][key] for key in ("outgoing_choice", "choice_breakdown")},
            "current": {key: report["models"][name][key] for key in ("outgoing_choice", "choice_breakdown")}}
            for name in variants}
    report["status"] = "completed"
    _save_json(output, report)
    progress(f"Saved {output}: pooled predictions cover all original choices once")
    progress(f"{'Model':27} {'All roads':>11} {'Deviations':>12} {'Followed':>11}")
    for name, result in report["models"].items():
        overall = result["outgoing_choice"]
        dev, follow = (result["choice_breakdown"][n] for n in ("deviated", "followed"))
        progress(f"{name:27} {overall['correct']:>4}/{overall['choices']:<6} "
                 f"{dev['correct']:>5}/{dev['choices']:<6} {follow['correct']:>4}/{follow['choices']:<6}")
    progress("Exact directed-road choices; binary deviation detection is a separate JSON metric. "
             "This is an internal cross-validation study, not new external validation.")
    return report


def run_real_refinement(graphml: Path, node_features: Path, original_path: Path,
                        split_path: Path, gps_data: Path, output: Path, *, epochs=60,
                        seed=17, batch_size=8, history_steps=6, history_interval_s=300,
                        gps_profile_bin_s=300, progress=print, first_visit_only=False,
                        reuse_selection_path: Path | None = None) -> dict:
    """Select or reuse validation settings, then refit on development riders."""
    from stgat_lstm.preprocessing.build_decision_candidates import refine_real_decisions
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    frozen = json.loads(split_path.read_text(encoding="utf-8"))
    if frozen.get("dataset_sha256") != hashlib.sha256(original_bytes).hexdigest() or frozen.get("seed") != seed:
        raise ValueError("Refinement needs the split belonging to the original approved data and seed")
    assignments = frozen["assignments"]
    previous = None
    if reuse_selection_path is not None:
        previous = json.loads(reuse_selection_path.read_text(encoding="utf-8"))
        if (not first_visit_only or previous.get("status") != "completed"
                or previous.get("evaluation") != "real_refined_rider_split"
                or previous["plan"]["assignments"] != assignments
                or previous["plan"]["original_artifact_sha256"] != hashlib.sha256(original_bytes).hexdigest()
                or previous["plan"]["original_split_sha256"] != hashlib.sha256(split_path.read_bytes()).hexdigest()
                or previous["plan"]["seed"] != seed or epochs != previous["selected"]["epochs"]):
            raise ValueError("Locked comparison requires the original completed refinement and its selected epoch count")
        policy = previous["provenance"]["traffic_policy"]
        if (batch_size != previous["plan"].get("batch_size", 8)
                or (history_steps, history_interval_s, gps_profile_bin_s) !=
                   (policy["steps"], policy["interval_s"], policy["profile_bin_s"])):
            raise ValueError("Locked comparison batch size or traffic history settings changed")
    expected_riders = {e["rider_group"] for e in original["examples"] if e["status"] == "approved"}
    if set(assignments) != expected_riders:
        raise ValueError("Original split does not cover the approved riders")
    folder = output.parent
    folder.mkdir(parents=True, exist_ok=True)
    plan = {"schema_version": 1, "seed": seed, "epochs_maximum": epochs,
            "original_artifact_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "original_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "assignments": assignments, "ranking_weights": [0., .1, 1.],
            "regularization": {"learning_rate": .001, "weight_decay": .0001,
                               "label_smoothing": .05, "traffic_mask_probability": .15,
                               "balance_rides": True, "patience": 3, "validation_interval": 5},
            "test_used_for_selection": False,
            "source_code_sha256": {
                p.relative_to(Path(__file__).resolve().parents[1]).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))}}
    plan["first_visit_only"] = first_visit_only
    plan["batch_size"] = batch_size
    if previous is not None:
        if previous["plan"]["regularization"] != plan["regularization"]:
            raise ValueError("Locked comparison requires unchanged regularization")
        plan.update({"ranking_weights": [previous["selected"]["preference_weight"]],
            "selection_mode": "fixed previous validation selection; no new weight or epoch search",
            "reuse_selection_sha256": hashlib.sha256(reuse_selection_path.read_bytes()).hexdigest()})
    _save_json(folder / "experiment_plan.json", plan)
    corrected = refine_real_decisions(gps_data, graphml, original, assignments, progress=progress)
    original_keys = {e["example_key"] for e in original["examples"] if e["status"] == "approved"}
    if not original_keys.issubset({e["example_key"] for e in corrected["examples"] if e["status"] == "approved"}):
        raise ValueError("Refinement unexpectedly removed an original approved example")
    path = folder / "approved_training_examples.json"
    _save_json(path, corrected)
    prepared, configuration, provenance = prepare_real(
        graphml, node_features, path, history_steps=history_steps,
        history_interval_s=history_interval_s)
    if previous is not None:
        for field in ("approved_artifact_sha256", "graphml_sha256", "node_features_sha256"):
            if provenance[field] != previous["provenance"][field]:
                raise ValueError(f"Locked comparison input changed: {field}")
        if {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(gps_data.glob("*.csv"))} != previous["provenance"]["gps_source_sha256"]:
            raise ValueError("Locked comparison GPS exports changed")
    configuration = {**configuration, "architecture": "stgat_lstm"}
    graph_data = build_real_graph_data(graphml, node_features, approved_decisions=path)
    archive = GPSTrafficArchive.from_exports(graph_data, gps_data)
    progress(f"Corrected graph: {len(graph_data.node_ids)} nodes, {len(graph_data.edge_ids)} edges")
    groups = fixed_rider_partitions(prepared, assignments)
    selection_examples, selection_traffic = attach_gps_traffic(
        prepared, archive, excluded_riders={r for r, g in assignments.items() if g != "training"},
        history_steps=history_steps, history_interval_s=history_interval_s, gps_profile_bin_s=gps_profile_bin_s)
    selection_groups = fixed_rider_partitions(selection_examples, assignments)
    selection_adjustment = None
    if first_visit_only:
        selection_groups["training"], selection_adjustment = first_junction_visits(selection_groups["training"])
        plan["training_supervision"] = {"scope": "original training riders only; validation/test targets remain frozen",
                                        **selection_adjustment}
        _save_json(folder / "experiment_plan.json", plan)
        progress(f"First-visit training choices: {selection_adjustment['choices_before']} -> {selection_adjustment['choices_after']}")
    progress(f"Refinement: {len(selection_groups['training'])} training examples; "
             f"{len(selection_groups['validation'])} validation; {len(selection_groups['test'])} test")
    trials, selected, selected_key = [], None, None
    for weight in plan["ranking_weights"]:
        progress(f"Validation {'check' if previous is not None else 'trial'} | ranking weight {weight} | {epochs} epochs")
        options = dict(plan["regularization"])
        model, summary = train_decision_model(
            selection_groups["training"], configuration, epochs=epochs,
            preference_weight=weight, seed=seed, batch_size=batch_size,
            validation=selection_groups["validation"], select_validation_checkpoint=previous is None,
            **options,
            progress=lambda epoch, loss: progress(f"  epoch {epoch}: loss {loss:.4f}")
                if epoch == 1 or epoch % 5 == 0 else None)
        predictions, ranking = _model_predictions(model, selection_groups["validation"])
        threshold = previous["selected"]["threshold"] if previous is not None else validation_threshold(predictions)
        validation_report = _fixed_prediction_report(predictions, ranking, threshold)
        trial = {"preference_weight": weight, "selected_epoch": summary.selected_epoch,
                 "training_summary": asdict(summary), "validation": validation_report,
                 "threshold": threshold}
        trials.append(trial)
        key = (validation_report["outgoing_choice"]["accuracy"],
               validation_report["classification"]["f1"],
               -validation_report["outgoing_choice"]["negative_log_likelihood"])
        if selected_key is None or key > selected_key:
            selected_key, selected = key, trial
        _save_json(folder / "selection_report.json", {"status": "selection_in_progress",
                   "trials": trials, "selected_preference_weight": selected["preference_weight"],
                   "test_predictions_computed": False})
    # Validation riders become training riders only after settings and epoch are fixed.
    final_examples, final_traffic = attach_gps_traffic(
        prepared, archive, excluded_riders={r for r, g in assignments.items() if g == "test"},
        history_steps=history_steps, history_interval_s=history_interval_s, gps_profile_bin_s=gps_profile_bin_s)
    final_groups = fixed_rider_partitions(final_examples, assignments)
    development_original = final_groups["training"] + final_groups["validation"]
    final_adjustment = None
    if first_visit_only:
        final_groups["training"], final_adjustment = first_junction_visits(final_groups["training"])
        if final_adjustment != selection_adjustment:
            raise ValueError("Training supervision changed between selection and final refit")
    development = final_groups["training"] + final_groups["validation"]
    progress(f"Final refit: {len({e.rider_group for e in development})} development riders, "
             f"{len(development)} examples, {selected['selected_epoch']} fixed epochs")
    final_model, summary = train_decision_model(
        development, configuration, epochs=selected["selected_epoch"],
        preference_weight=selected["preference_weight"], seed=seed, batch_size=batch_size,
        **options,
        progress=lambda epoch, loss: progress(f"  final epoch {epoch}: loss {loss:.4f}")
            if epoch == 1 or epoch % 5 == 0 else None)
    provenance.update({"traffic": "historical rider-GPS profiles; target ride and future records excluded",
        "traffic_policy": {"steps": history_steps, "interval_s": history_interval_s,
                           "profile_bin_s": gps_profile_bin_s, "future_records_allowed": False,
                           "target_ride_allowed": False, "test_riders_allowed": False},
        "gps_traffic_build": archive.summary, "gps_traffic_directory": str(gps_data),
        "gps_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(gps_data.glob("*.csv"))},
        "evaluation": "development riders used for fitting; three test riders excluded",
        "selected_settings": {"preference_weight": selected["preference_weight"],
                              "epochs": selected["selected_epoch"], **plan["regularization"]}})
    if first_visit_only:
        provenance["training_supervision"] = plan["training_supervision"]
    _save_training(final_model, summary, provenance, folder / "checkpoint.pt", folder / "training_report.json")
    predictions, ranking = _model_predictions(final_model, final_groups["test"])
    targets = [t.deviated for e in development for t in e.choice_targets]
    baseline = _fixed_prediction_report(_suggestion_baseline(final_groups["test"], sum(targets) / len(targets)),
                                        {"pairs": 0, "correct": 0, "accuracy": None}, .5)
    test = _fixed_prediction_report(predictions, ranking, selected["threshold"])
    def counts(examples):
        return {"riders": len({e.rider_group for e in examples}), "rides": len({e.ride_group for e in examples}),
                "examples": len(examples), "choices": sum(len(e.choice_targets) for e in examples),
                "alternatives": sum(t.deviated for e in examples for t in e.choice_targets)}
    report = {"schema_version": 1, "evaluation": "real_refined_rider_split", "status": "completed",
        "plan": plan, "refinement": corrected["refinement"], "trials": trials,
        "selected": {"preference_weight": selected["preference_weight"], "epochs": selected["selected_epoch"],
                     "validation": selected["validation"], "threshold": selected["threshold"]},
        "counts": {"selection_training": counts(selection_groups["training"]),
                   "selection_validation": counts(selection_groups["validation"]),
                   "final_development": counts(development), "test": counts(final_groups["test"])},
        "final_training_summary": asdict(summary), "test": test, "suggestion_baseline": baseline,
        "selection_traffic": selection_traffic, "final_traffic": final_traffic,
        "provenance": provenance, "model_fits": len(trials) + 1,
        "interpretation": "Repeated internal benchmark on the same test riders. Inputs now use consistent destination context; "
                          "additional real labels and masked traffic are training-only. No test-based settings or split selection. "
                          "A small previously inspected test set does not establish broad reliability."}
    development_predictions, development_ranking = _model_predictions(final_model, development)
    report["development"] = _fixed_prediction_report(development_predictions, development_ranking, selected["threshold"])
    report["overfitting_checks"] = refinement_overfitting_checks(report)
    if first_visit_only:
        original_predictions, original_ranking = _model_predictions(final_model, development_original)
        report["development_original_targets"] = _fixed_prediction_report(original_predictions, original_ranking, selected["threshold"])
        report["counts"]["final_development_original_targets"] = counts(development_original)
        report["supervision_adjustment"] = {"selection_training": selection_adjustment,
                                            "final_original_training_riders": final_adjustment,
                                            "validation_test_targets_modified": False}
    _save_json(output, report)
    _save_json(folder / "selection_report.json", {"status": "completed", "trials": trials,
               "selected_preference_weight": selected["preference_weight"], "test_used_for_selection": False})
    progress(f"Saved {output}: test {test['outgoing_choice']['correct']}/{test['outgoing_choice']['choices']} "
             f"roads | accuracy {test['outgoing_choice']['accuracy']:.3f} | F1 {test['classification']['f1']:.3f} "
             f"| NLL {test['outgoing_choice']['negative_log_likelihood']:.3f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("node_features", type=Path)
    parser.add_argument("approved_decisions", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float,
                        help="default .001 for grouped k-fold; .005 for other evaluation modes")
    parser.add_argument("--preference-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temporal-ablation", action="store_true",
                        help="Add latest-only LSTM; grouped k-fold always runs the requested control")
    parser.add_argument("--first-visit-only", action="store_true",
                        help="In --refine, keep the first training visit to each junction within an example")
    parser.add_argument("--reuse-selection", type=Path,
                        help="Completed refinement report whose selected settings are fixed for a first-visit comparison")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--diagnose-fit", action="store_true",
                       help="Audit training input conflicts and fit eight training choices; no held-out evaluation")
    modes.add_argument("--refine", action="store_true",
                       help="Correct real contexts, expand training choices, select on validation, and refit development riders")
    modes.add_argument("--fixed-split", action="store_true",
                       help="Fit once per architecture on a saved real rider-disjoint 60/20/20 split")
    modes.add_argument("--synthetic-fixed-split", action="store_true",
                        help="Use the faster fixed rider-disjoint split for a controlled synthetic dataset")
    modes.add_argument("--group-k-folds", type=int, metavar="K",
                       help="Real rider-grouped k-fold with inner validation and final refit per fold")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES,
                        help="Grouped k-fold variants; ablation defaults to stgat_lstm, otherwise gatv2 stgat_lstm")
    parser.add_argument("--training-ablation", type=Path, metavar="REFERENCE_REPORT",
                        help="Compare three validation-only training recipes using a completed grouped k-fold run")
    parser.add_argument("--split-file", type=Path,
                        help="Save/reuse rider assignments; grouped k-fold uses a separate fold-plan schema")
    parser.add_argument("--synthetic-tune", action="store_true",
                        help="select alternative-road loss weight using synthetic validation riders only")
    parser.add_argument("--traffic-archive", type=Path)
    parser.add_argument("--gps-traffic-data", type=Path,
                        help="rider exports used for leakage-safe GPS speed profiles")
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument("--history-interval-seconds", type=int, default=300)
    parser.add_argument("--traffic-max-age-seconds", type=int, default=900)
    parser.add_argument("--gps-profile-bin-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.learning_rate is None:
        args.learning_rate = .001 if args.group_k_folds is not None else .005
    if args.architectures is not None and args.group_k_folds is None:
        parser.error("--architectures requires --group-k-folds")
    if args.training_ablation is not None and args.group_k_folds is None:
        parser.error("--training-ablation requires --group-k-folds")
    if (args.first_visit_only or args.reuse_selection) and not args.refine:
        parser.error("--first-visit-only and --reuse-selection require --refine")
    if args.traffic_archive and args.gps_traffic_data:
        parser.error("Choose --traffic-archive or --gps-traffic-data")
    if args.synthetic_tune and not args.synthetic_fixed_split:
        parser.error("--synthetic-tune requires --synthetic-fixed-split")
    if args.split_file and not (args.fixed_split or args.refine or args.diagnose_fit or args.group_k_folds is not None):
        parser.error("--split-file requires --fixed-split, --refine, --diagnose-fit, or --group-k-folds")
    if args.group_k_folds is not None:
        if not args.gps_traffic_data or args.traffic_archive:
            parser.error("--group-k-folds requires --gps-traffic-data, without a Mapbox archive")
        if args.group_k_folds < 2:
            parser.error("--group-k-folds must be at least 2")
        run_grouped_kfold(args.graphml, args.node_features, args.approved_decisions,
            args.gps_traffic_data, args.output or Path("artifacts/experiments/real_training_ablation/evaluation_report.json"
                if args.training_ablation else "artifacts/experiments/real_group_kfold/evaluation_report.json"),
            n_splits=args.group_k_folds, epochs=args.epochs, learning_rate=args.learning_rate,
            preference_weight=args.preference_weight, seed=args.seed, batch_size=args.batch_size,
            history_steps=args.history_steps, history_interval_s=args.history_interval_seconds,
            gps_profile_bin_s=args.gps_profile_bin_seconds, architectures=tuple(args.architectures or
                (("stgat_lstm",) if args.training_ablation else ("gatv2", "stgat_lstm"))),
            temporal_ablation=args.temporal_ablation, split_path=args.split_file,
            ablation_reference=args.training_ablation)
        return
    if args.diagnose_fit:
        if not args.split_file or not args.gps_traffic_data or args.traffic_archive:
            parser.error("--diagnose-fit requires the original --split-file and --gps-traffic-data")
        run_fitting_diagnostics(args.graphml, args.node_features, args.approved_decisions,
            args.split_file, args.gps_traffic_data, args.output or Path("outputs/reports/fitting_diagnostics.json"),
            epochs=args.epochs, seed=args.seed, history_steps=args.history_steps,
            history_interval_s=args.history_interval_seconds, gps_profile_bin_s=args.gps_profile_bin_seconds)
        return
    if args.refine:
        if not args.split_file or not args.gps_traffic_data or args.traffic_archive:
            parser.error("--refine requires the original --split-file and --gps-traffic-data")
        run_real_refinement(args.graphml, args.node_features, args.approved_decisions,
            args.split_file, args.gps_traffic_data, args.output or Path("artifacts/experiments/real_refined/evaluation_report.json"),
            epochs=args.epochs, seed=args.seed, batch_size=args.batch_size, history_steps=args.history_steps,
            history_interval_s=args.history_interval_seconds, gps_profile_bin_s=args.gps_profile_bin_seconds,
            first_visit_only=args.first_visit_only, reuse_selection_path=args.reuse_selection)
        return
    if args.output is None:
        args.output = Path("outputs/reports/fixed_split_evaluation.json" if args.fixed_split
                           else "outputs/reports/rider_holdout_evaluation.json")
    prepared, configuration, provenance = prepare_real(
        args.graphml,
        args.node_features,
        args.approved_decisions,
        traffic_archive=args.traffic_archive, history_steps=args.history_steps,
        history_interval_s=args.history_interval_seconds, traffic_max_age_s=args.traffic_max_age_seconds,
    )
    gps_archive = None
    if args.gps_traffic_data:
        graph_data = build_real_graph_data(args.graphml, args.node_features,
                                          approved_decisions=args.approved_decisions)
        gps_archive = GPSTrafficArchive.from_exports(graph_data, args.gps_traffic_data)
        provenance["traffic"] = (
            "fold-specific historical rider-GPS speed profiles; held-out rider and future records excluded"
        )
        provenance["gps_traffic_build"] = gps_archive.summary
        provenance["traffic_policy"] = {
            "steps": args.history_steps,
            "interval_s": args.history_interval_seconds,
            "profile_bin_s": args.gps_profile_bin_seconds,
            "held_out_rider_allowed": False,
            "future_records_allowed": False,
        }
    if args.fixed_split:
        if provenance.get("synthetic"):
            parser.error("Use --synthetic-fixed-split for artifacts marked synthetic")
        split_path = args.split_file or args.output.with_name(args.output.stem + "_split.json")
        assignments = saved_fixed_rider_assignments(
            split_path, prepared, hashlib.sha256(args.approved_decisions.read_bytes()).hexdigest(), args.seed)
        print(f"Frozen rider assignments: {split_path}")
        report = evaluate_fixed_rider_split(
            prepared, configuration, assignments=assignments, epochs=args.epochs,
            learning_rate=args.learning_rate, preference_weight=args.preference_weight,
            seed=args.seed, batch_size=args.batch_size, temporal_ablation=args.temporal_ablation,
            traffic_schema=tuple(provenance["edge_dynamic_schema"]), gps_archive=gps_archive,
            history_steps=args.history_steps, history_interval_s=args.history_interval_seconds,
            gps_profile_bin_s=args.gps_profile_bin_seconds, progress=print)
        report["split_file"] = str(split_path)
    elif args.synthetic_fixed_split:
        if not provenance.get("synthetic"):
            parser.error("--synthetic-fixed-split requires an artifact marked synthetic")
        synthetic_document = json.loads(args.approved_decisions.read_text(encoding="utf-8"))
        split_assignments = {
            str(example["rider_group"]): str(example.get("quality", {}).get("synthetic_split"))
            for example in synthetic_document.get("examples", [])
            if example.get("quality", {}).get("synthetic_split")
        } or None
        report = evaluate_synthetic_fixed_split(
            prepared,
            configuration,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            preference_weight=args.preference_weight,
            seed=args.seed,
            traffic_schema=tuple(provenance["edge_dynamic_schema"]),
            gps_archive=gps_archive,
            history_steps=args.history_steps,
            history_interval_s=args.history_interval_seconds,
            gps_profile_bin_s=args.gps_profile_bin_seconds,
            batch_size=args.batch_size,
            split_assignments=split_assignments,
            tune_hyperparameters=args.synthetic_tune,
            progress=print,
        )
    else:
        report = evaluate_rider_disjoint(
            prepared,
            configuration,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            preference_weight=args.preference_weight,
            seed=args.seed,
            temporal_ablation=args.temporal_ablation,
            traffic_schema=tuple(provenance["edge_dynamic_schema"]),
            gps_archive=gps_archive,
            history_steps=args.history_steps,
            history_interval_s=args.history_interval_seconds,
            gps_profile_bin_s=args.gps_profile_bin_seconds,
            batch_size=args.batch_size,
            progress=print,
        )
    report["provenance"] = {
        **provenance,
        "evaluation": (
            "fixed real train/validation/test riders; validation/test riders excluded from training and GPS profiles"
            if args.fixed_split else "fixed rider-disjoint synthetic train/validation/test split"
            if args.synthetic_fixed_split
            else "every reported prediction is from a fold that excludes that rider"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    def display(value):
        return f"{value:.3f}" if value is not None else "n/a"
    if args.fixed_split:
        print(f"Saved {args.output}: {report['examples']} examples, {report['outgoing_choices']} choices, "
              f"{report['riders']} riders, {report['training']['model_fits']} model fits")
        print(f"{'Model':27} {'Train acc':>9} {'Val acc':>9} {'Test acc':>9} {'Bal acc':>9} {'Dev F1':>9} {'Rank':>7}")
        for name, result in report["models"].items():
            metrics = result["test"]["classification"]
            accuracies = [result[group]["outgoing_choice"]["accuracy"]
                          for group in ("training", "validation", "test")]
            rank = result["test"]["preference_ranking"]["accuracy"]
            print(f"{name:27} " + " ".join(f"{display(acc):>9}" for acc in accuracies)
                  + f" {display(metrics['balanced_accuracy']):>9} {display(metrics['f1']):>9} {display(rank):>7}")
        print("Binary metrics above use a fixed 0.5 threshold; validation-selected results are separate in JSON.")
        print(f"Test temporal evidence: {report['temporal_evidence']['test']['status']}")
        print("Real rider-held-out pilot; sample size and repeated validation still limit reliability claims.")
        return
    if args.synthetic_fixed_split:
        split = report["split"]
        print(f"Saved {args.output}: {split['training_examples']} train, "
              f"{split['validation_examples']} validation, {split['test_examples']} test examples")
        selected = report["hyperparameter_tuning"].get("selected")
        if selected:
            print("Validation-selected alternative-road weight: "
                  f"{selected['alternative_choice_weight']:.2f}x")
        print(f"{'Model':27} {'Choice acc':>10} {'Threshold':>9} {'Balanced acc':>12} {'F1 deviation':>12} {'Rank':>7}")
        for name, result in report["models"].items():
            metrics = result["test"]["classification"]
            choice = result["test"]["outgoing_choice"]["accuracy"]
            rank = result["test"].get("preference_ranking", {}).get("accuracy")
            print(f"{name:27} {display(choice):>10} {display(metrics['threshold']):>9} "
                  f"{display(metrics['balanced_accuracy']):>12} {display(metrics['f1']):>12} {display(rank):>7}")
        print(f"Temporal evidence: {report['temporal_evidence']['status']}")
        print("Controlled synthetic test; do not combine these metrics with official rider results.")
        return
    print(f"Saved {args.output}: {report['examples']} examples, {report['outgoing_choices']} outgoing choices, "
          f"{report['riders']} held-out rider folds")
    print(f"{'Model':27} {'Choice acc':>10} {'Balanced acc':>12} {'F1 deviation':>12} {'Rank':>7}")
    for name, result in report["models"].items():
        metrics = result["classification"]
        choice = result["outgoing_choice"]["accuracy"]
        rank = result.get("held_out_preference_ranking", {}).get("accuracy")
        print(f"{name:27} {display(choice):>10} {display(metrics['balanced_accuracy']):>12} "
              f"{display(metrics['f1']):>12} {display(rank):>7}")
    print(f"Temporal evidence: {report['lstm_comparison']['temporal_evidence']['status']}")
    print(f"Latest-only LSTM control: {report['lstm_comparison']['latest_only_control']}")
    print("Pilot results; consult report provenance and sample size before claiming generalization.")


if __name__ == "__main__":
    main()
