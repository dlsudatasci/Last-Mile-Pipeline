"""Train the shared road-deviation classifier and path-preference model.

Real training accepts only a finalized, validated decision artifact.
The training report contains in-sample checks. Use the `evaluate` command for
held-out-rider results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import DecisionPreferenceModel, ModelInput, path_cost, preference_loss
from .graph_data import build_real_graph_data, load_approved_decisions, path_choice_edge_ids
from .mapbox_traffic import TrafficArchive


@dataclass(frozen=True)
class PreparedDecision:
    example_key: str
    rider_group: str
    label: int
    suggested_edge_ids: tuple[str, ...]
    edge_targets: tuple[tuple[str, int], ...]
    preferred_edge_ids: tuple[str, ...] | None
    rejected_edge_ids: tuple[str, ...] | None
    inputs: ModelInput


@dataclass(frozen=True)
class DecisionTrainingSummary:
    examples: int
    riders: int
    followed: int
    deviated: int
    road_decisions: int
    followed_road_decisions: int
    deviation_road_decisions: int
    epochs: int
    initial_loss: float
    final_loss: float
    in_sample_road_classification_correct: int
    preference_pairs: int
    in_sample_ranking_correct: int
    epoch_losses: tuple[float, ...] = ()


def train_decision_model(
    prepared: list[PreparedDecision],
    model_configuration: dict,
    *,
    epochs: int = 40,
    learning_rate: float = 0.005,
    preference_weight: float = 1.0,
    seed: int = 17,
) -> tuple[DecisionPreferenceModel, DecisionTrainingSummary]:
    if epochs <= 0 or learning_rate <= 0 or preference_weight < 0:
        raise ValueError("Training hyperparameters are invalid")
    if not prepared:
        raise ValueError("No approved decision examples")
    target_rows = [target for example in prepared for target in example.edge_targets]
    if not target_rows:
        raise ValueError("No road-level choice targets")
    if any(label not in {0, 1} for _, label in target_rows):
        raise ValueError("Road-deviation labels must be zero or one")
    positives = sum(label == 1 for _, label in target_rows)
    negatives = len(target_rows) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Road-deviation training requires both classes")
    torch.manual_seed(seed)
    model = DecisionPreferenceModel(**model_configuration)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_weight = torch.tensor(float(negatives / positives))
    initial_loss = 0.0
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        classification_logits: list[Tensor] = []
        classification_labels: list[Tensor] = []
        ranking_losses: list[Tensor] = []
        for example in prepared:
            costs, edge_logits = model(example.inputs)
            lookup = {edge_id: index for index, edge_id in enumerate(example.inputs.edge_ids)}
            try:
                indices = [lookup[edge_id] for edge_id, _ in example.edge_targets]
            except KeyError as exc:
                raise ValueError(f"Road target has unavailable edge: {exc.args[0]}") from exc
            classification_logits.append(edge_logits[indices])
            classification_labels.append(torch.tensor(
                [float(label) for _, label in example.edge_targets], dtype=edge_logits.dtype
            ))
            if example.preferred_edge_ids is not None and example.rejected_edge_ids is not None:
                ranking_losses.append(preference_loss(
                    costs,
                    example.preferred_edge_ids,
                    example.rejected_edge_ids,
                    example.inputs.edge_ids,
                ))
        classification = F.binary_cross_entropy_with_logits(
            torch.cat(classification_logits), torch.cat(classification_labels), pos_weight=positive_weight
        )
        loss = classification
        if ranking_losses:
            loss = loss + preference_weight * torch.stack(ranking_losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Training produced a nonfinite loss")
        if epoch == 0:
            initial_loss = float(loss.detach())
        epoch_losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()

    model.eval()
    final_logits = []
    final_labels = []
    classification_correct = 0
    ranking_correct = 0
    preference_pairs = 0
    final_ranking_losses: list[Tensor] = []
    with torch.no_grad():
        for example in prepared:
            costs, edge_logits = model(example.inputs)
            lookup = {edge_id: index for index, edge_id in enumerate(example.inputs.edge_ids)}
            indices = [lookup[edge_id] for edge_id, _ in example.edge_targets]
            labels = torch.tensor(
                [float(label) for _, label in example.edge_targets], dtype=edge_logits.dtype
            )
            selected_logits = edge_logits[indices]
            final_logits.append(selected_logits)
            final_labels.append(labels)
            classification_correct += int(((selected_logits >= 0) == labels.bool()).sum())
            if example.preferred_edge_ids is not None and example.rejected_edge_ids is not None:
                final_ranking_losses.append(preference_loss(
                    costs,
                    example.preferred_edge_ids,
                    example.rejected_edge_ids,
                    example.inputs.edge_ids,
                ))
                preference_pairs += 1
                ranking_correct += int(
                    path_cost(costs, example.preferred_edge_ids, example.inputs.edge_ids)
                    < path_cost(costs, example.rejected_edge_ids, example.inputs.edge_ids)
                )
        final_classification = F.binary_cross_entropy_with_logits(
            torch.cat(final_logits), torch.cat(final_labels), pos_weight=positive_weight
        )
        final_loss = final_classification
        if final_ranking_losses:
            final_loss = final_loss + preference_weight * torch.stack(final_ranking_losses).mean()
    summary = DecisionTrainingSummary(
        examples=len(prepared),
        riders=len({example.rider_group for example in prepared}),
        followed=sum(example.label == 0 for example in prepared),
        deviated=sum(example.label == 1 for example in prepared),
        road_decisions=len(target_rows),
        followed_road_decisions=negatives,
        deviation_road_decisions=positives,
        epochs=epochs,
        initial_loss=initial_loss,
        final_loss=float(final_loss),
        in_sample_road_classification_correct=classification_correct,
        preference_pairs=preference_pairs,
        in_sample_ranking_correct=ranking_correct,
        epoch_losses=tuple(epoch_losses),
    )
    return model, summary


def prepare_real(
    graphml: Path,
    node_features: Path,
    approved_decisions: Path,
    *, traffic_archive: Path | None = None, history_steps: int = 6,
    history_interval_s: int = 300, traffic_max_age_s: int = 900,
) -> tuple[list[PreparedDecision], dict, dict]:
    graph_data = build_real_graph_data(graphml, node_features)
    decisions = load_approved_decisions(approved_decisions, graph_data)
    prepared = []
    archive = TrafficArchive.from_directory(graph_data, traffic_archive) if traffic_archive else None
    traffic_reports = {}
    for decision in decisions:
        temporal = None
        if archive is not None:
            temporal, traffic_reports[decision.example_key] = archive.sequence(
                decision.decision_timestamp_ms, steps=history_steps,
                interval_s=history_interval_s, max_age_s=traffic_max_age_s)
        inputs = graph_data.build_model_input(decision.destination_node_id, temporal_edge_features=temporal)
        choice_edges = path_choice_edge_ids(graph_data, decision.suggested_edge_ids)
        if decision.label == 0:
            edge_targets = tuple((edge_id, 0) for edge_id in choice_edges)
        else:
            divergence = next((index for index, pair in enumerate(
                zip(decision.suggested_edge_ids, decision.observed_edge_ids)) if pair[0] != pair[1]), None)
            if divergence is None:
                divergence = min(len(decision.suggested_edge_ids), len(decision.observed_edge_ids))
            if divergence >= len(decision.suggested_edge_ids):
                raise ValueError(f"Deviated example {decision.example_key} has no rejected suggested road")
            rejected_edge = decision.suggested_edge_ids[divergence]
            earlier_choices = [edge_id for edge_id in choice_edges
                               if decision.suggested_edge_ids.index(edge_id) < divergence]
            if rejected_edge not in choice_edges:
                raise ValueError(
                    f"Deviated example {decision.example_key} does not diverge at a branching road"
                )
            edge_targets = tuple((edge_id, 0) for edge_id in earlier_choices) + ((rejected_edge, 1),)
        if not edge_targets:
            raise ValueError(f"Example {decision.example_key} contains no road-choice target")
        prepared.append(
            PreparedDecision(
                example_key=decision.example_key,
                rider_group=decision.rider_group,
                label=decision.label,
                suggested_edge_ids=decision.suggested_edge_ids,
                edge_targets=edge_targets,
                preferred_edge_ids=decision.observed_edge_ids if decision.label == 1 else None,
                rejected_edge_ids=decision.suggested_edge_ids if decision.label == 1 else None,
                inputs=inputs,
            )
        )
    provenance = {
        "kind": "approved real decision training run",
        "approved_artifact_sha256": hashlib.sha256(approved_decisions.read_bytes()).hexdigest(),
        "traffic": "unknown for historical decisions; no current Mapbox data attached retrospectively",
        "evaluation": "in-sample only; not a rider-generalization estimate",
        "graphml_sha256": hashlib.sha256(graphml.read_bytes()).hexdigest(),
        "node_features_sha256": hashlib.sha256(node_features.read_bytes()).hexdigest(),
        "edge_dynamic_schema": list(graph_data.schema.edge_dynamic),
    }
    if archive is not None:
        provenance["traffic"] = "archived pre-decision snapshots; missing/stale/future traffic stays unknown"
        provenance["traffic_history"] = traffic_reports
        provenance["traffic_policy"] = {"steps": history_steps, "interval_s": history_interval_s,
                                        "max_age_s": traffic_max_age_s}
    return prepared, graph_data.model_configuration(), provenance


def _save_training(
    model: DecisionPreferenceModel,
    summary: DecisionTrainingSummary,
    provenance: dict,
    output: Path,
    report_output: Path | None = None,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": model.configuration(),
        "training_summary": asdict(summary),
        "provenance": provenance,
    }
    torch.save(checkpoint, output)
    report = {"checkpoint": str(output), "training_summary": asdict(summary), "provenance": provenance}
    report_path = report_output or output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    real = subparsers.add_parser("real")
    real.add_argument("graphml", type=Path)
    real.add_argument("node_features", type=Path)
    real.add_argument("approved_decisions", type=Path)
    real.add_argument("--output", type=Path, default=Path("outputs/gatv2_lstm_checkpoint.pt"))
    real.add_argument("--report-output", type=Path, default=Path("outputs/training_report.json"))
    real.add_argument("--traffic-archive", type=Path)
    real.add_argument("--history-steps", type=int, default=6)
    real.add_argument("--history-interval-seconds", type=int, default=300)
    real.add_argument("--traffic-max-age-seconds", type=int, default=900)
    real.add_argument("--epochs", type=int, default=40)
    real.add_argument("--learning-rate", type=float, default=0.005)
    real.add_argument("--preference-weight", type=float, default=1.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepared, configuration, provenance = prepare_real(
        args.graphml, args.node_features, args.approved_decisions,
        traffic_archive=args.traffic_archive, history_steps=args.history_steps,
        history_interval_s=args.history_interval_seconds, traffic_max_age_s=args.traffic_max_age_seconds)
    model, summary = train_decision_model(
        prepared,
        configuration,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        preference_weight=args.preference_weight,
    )
    provenance["training_settings"] = {"epochs": args.epochs, "learning_rate": args.learning_rate,
                                       "preference_weight": args.preference_weight, "seed": 17}
    report = _save_training(model, summary, provenance, args.output, args.report_output)
    readable = report["training_summary"]
    print(f"Checkpoint saved to {report['checkpoint']}")
    print(f"Training report saved to {args.report_output or args.output.with_suffix('.json')}")
    print(f"Examples: {readable['examples']} from {readable['riders']} riders | "
          f"followed: {readable['followed']} | deviated: {readable['deviated']}")
    print(f"Road choices: {readable['road_decisions']} | "
          f"followed: {readable['followed_road_decisions']} | "
          f"deviated: {readable['deviation_road_decisions']}")
    print(f"Loss: {readable['initial_loss']:.4f} -> {readable['final_loss']:.4f} "
          f"after {readable['epochs']} epochs")
    print(f"In-sample checks: {readable['in_sample_road_classification_correct']}/"
          f"{readable['road_decisions']} road labels, "
          f"{readable['in_sample_ranking_correct']}/{readable['preference_pairs']} preference pairs")
    print("Use `python -m stgat_lstm evaluate` for held-out-rider results.")


if __name__ == "__main__":
    main()
