"""Evaluate rider-choice models with leave-one-rider-out folds.

Every prediction in this report comes from a model that was not trained on the
held-out rider. With the current eight examples this remains a preliminary,
high-variance evaluation rather than a final performance estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from .train_model import PreparedDecision, prepare_real, train_decision_model
from .model import path_cost


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


def classification_metrics(predictions: list[dict]) -> dict:
    """Compute dependency-free binary metrics from held-out probabilities."""
    if not predictions:
        raise ValueError("No held-out predictions")
    if any(item["label"] not in (0, 1) for item in predictions):
        raise ValueError("Binary evaluation labels must be zero or one")
    labels = [int(item["label"]) for item in predictions]
    probabilities = [float(item["probability_deviated"]) for item in predictions]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Prediction probabilities must be finite and between zero and one")
    classes = [int(probability >= 0.5) for probability in probabilities]
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
        "threshold": 0.5,
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
) -> dict:
    """Compare simple and graph models on leave-one-rider-out predictions."""
    folds = rider_disjoint_folds(prepared)
    evidence = temporal_evidence_report(prepared, traffic_schema)
    architectures = list(ARCHITECTURES)
    if temporal_ablation and evidence["examples_with_observed_value_changes"] > 0:
        architectures.append("stgat_lstm_latest_only")
    all_predictions: dict[str, list[dict]] = {"prevalence": []}
    ranking: dict[str, list[bool]] = {}
    for architecture in architectures:
        all_predictions[architecture] = []
        ranking[architecture] = []

    for fold_index, (rider, train, test) in enumerate(folds):
        training_targets = [label for example in train for _, label in example.edge_targets]
        training_prevalence = sum(training_targets) / len(training_targets)
        for example in test:
            for edge_id, label in example.edge_targets:
                all_predictions["prevalence"].append({
                    "rider_group": rider,
                    "example_key": example.example_key,
                    "edge_id": edge_id,
                    "label": label,
                    "probability_deviated": training_prevalence,
                })

        for architecture in architectures:
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
            )
            model.eval()
            with torch.no_grad():
                for example in test_examples:
                    costs, edge_logits = model(example.inputs)
                    lookup = {edge_id: index for index, edge_id in enumerate(example.inputs.edge_ids)}
                    for edge_id, label in example.edge_targets:
                        probability = float(torch.sigmoid(edge_logits[lookup[edge_id]]))
                        all_predictions[architecture].append({
                            "rider_group": rider,
                            "example_key": example.example_key,
                            "edge_id": edge_id,
                            "label": label,
                            "probability_deviated": probability,
                        })
                    if example.preferred_edge_ids is not None and example.rejected_edge_ids is not None:
                        preferred = path_cost(costs, example.preferred_edge_ids, example.inputs.edge_ids)
                        rejected = path_cost(costs, example.rejected_edge_ids, example.inputs.edge_ids)
                        ranking[architecture].append(bool(preferred < rejected))

    models = {}
    for name, predictions in all_predictions.items():
        model_report = {
            "classification": classification_metrics(predictions),
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
        "schema_version": 3,
        "evaluation": "leave-one-rider-out",
        "examples": len(prepared),
        "road_decisions": sum(len(example.edge_targets) for example in prepared),
        "riders": len(folds),
        "folds": [
            {"held_out_rider": rider, "training_examples": len(train), "test_examples": len(test),
             "training_road_decisions": sum(len(item.edge_targets) for item in train),
             "test_road_decisions": sum(len(item.edge_targets) for item in test)}
            for rider, train, test in folds
        ],
        "training": {
            "epochs_per_fold": epochs,
            "learning_rate": learning_rate,
            "preference_weight": preference_weight,
            "base_seed": seed,
        },
        "models": models,
        "metric_protocol": {"primary": "balanced_accuracy", "secondary":
            ["f1", "macro_f1", "average_precision", "roc_auc", "brier_score", "log_loss"],
            "routing_proxy": "held_out_preference_ranking", "threshold": 0.5,
            "aggregation": "pooled out-of-fold road decisions; riders with more choices contribute more",
            "zero_division": "undefined class precision/F1 is zero; two-class metrics are null if a class is absent",
            "tuning": "do not choose hyperparameters or threshold using these held-out predictions"},
        "lstm_comparison": lstm_comparison(models, evidence, temporal_ablation),
        "interpretation": (
            "Held-out-rider experiment; assess sample size and class balance before claiming generalization. "
            "Inspect provenance traffic_history for actual pre-decision coverage. Unknown traffic, "
            "even across multiple history steps, cannot establish a temporal LSTM benefit. "
            "These same folds must not also be used for hyperparameter selection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("node_features", type=Path)
    parser.add_argument("approved_decisions", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--preference-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temporal-ablation", action="store_true",
                        help="Add a matched LSTM latest-only control when observed histories vary")
    parser.add_argument("--traffic-archive", type=Path)
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument("--history-interval-seconds", type=int, default=300)
    parser.add_argument("--traffic-max-age-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path, default=Path("outputs/rider_holdout_evaluation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepared, configuration, provenance = prepare_real(
        args.graphml,
        args.node_features,
        args.approved_decisions,
        traffic_archive=args.traffic_archive, history_steps=args.history_steps,
        history_interval_s=args.history_interval_seconds, traffic_max_age_s=args.traffic_max_age_seconds,
    )
    report = evaluate_rider_disjoint(
        prepared,
        configuration,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        preference_weight=args.preference_weight,
        seed=args.seed,
        temporal_ablation=args.temporal_ablation,
        traffic_schema=tuple(provenance["edge_dynamic_schema"]),
    )
    report["provenance"] = {
        **provenance,
        "evaluation": "every reported prediction is from a fold that excludes that rider",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved {args.output}: {report['examples']} examples, {report['road_decisions']} road decisions, "
          f"{report['riders']} held-out rider folds")
    print(f"{'Model':27} {'Balanced acc':>12} {'F1 deviation':>12} {'AP':>7} {'ROC-AUC':>8} {'Rank':>7}")
    def display(value):
        return f"{value:.3f}" if value is not None else "n/a"
    for name, result in report["models"].items():
        metrics = result["classification"]
        rank = result.get("held_out_preference_ranking", {}).get("accuracy")
        print(f"{name:27} {display(metrics['balanced_accuracy']):>12} {display(metrics['f1']):>12} "
              f"{display(metrics['average_precision']):>7} {display(metrics['roc_auc']):>8} {display(rank):>7}")
    print(f"Temporal evidence: {report['lstm_comparison']['temporal_evidence']['status']}")
    print(f"Latest-only LSTM control: {report['lstm_comparison']['latest_only_control']}")
    print("Pilot results; consult report provenance and sample size before claiming generalization.")


if __name__ == "__main__":
    main()
