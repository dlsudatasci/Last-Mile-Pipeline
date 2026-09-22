"""Train the shared follow/deviate classifier and path-preference model.

Real training accepts only a finalized, manually approved decision artifact.
Reported metrics are in-sample smoke checks until enough riders exist for a
rider-disjoint evaluation split.
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

from .features import ModelInput, build_model_input
from .fixture import prepare_choices, read_fixture
from .model import DecisionPreferenceModel, path_cost, preference_loss
from .real_data import build_real_graph_data, load_approved_decisions


@dataclass(frozen=True)
class PreparedDecision:
    example_key: str
    rider_group: str
    label: int
    suggested_edge_ids: tuple[str, ...]
    preferred_edge_ids: tuple[str, ...] | None
    rejected_edge_ids: tuple[str, ...] | None
    inputs: ModelInput


@dataclass(frozen=True)
class DecisionTrainingSummary:
    examples: int
    riders: int
    followed: int
    deviated: int
    epochs: int
    initial_loss: float
    final_loss: float
    in_sample_classification_correct: int
    preference_pairs: int
    in_sample_ranking_correct: int


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
    positives = sum(example.label == 1 for example in prepared)
    negatives = len(prepared) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Follow/deviate training requires both classes")
    torch.manual_seed(seed)
    model = DecisionPreferenceModel(**model_configuration)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_weight = torch.tensor(float(negatives / positives))
    initial_loss = 0.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        losses: list[Tensor] = []
        for example in prepared:
            costs, logit = model(example.inputs, example.suggested_edge_ids)
            classification = F.binary_cross_entropy_with_logits(
                logit.reshape(1),
                torch.tensor([float(example.label)]),
                pos_weight=positive_weight,
            )
            total = classification
            if example.preferred_edge_ids is not None and example.rejected_edge_ids is not None:
                total = total + preference_weight * preference_loss(
                    costs,
                    example.preferred_edge_ids,
                    example.rejected_edge_ids,
                    example.inputs.edge_ids,
                )
            losses.append(total)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Training produced a nonfinite loss")
        if epoch == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()

    model.eval()
    final_losses = []
    classification_correct = 0
    ranking_correct = 0
    preference_pairs = 0
    with torch.no_grad():
        for example in prepared:
            costs, logit = model(example.inputs, example.suggested_edge_ids)
            classification = F.binary_cross_entropy_with_logits(
                logit.reshape(1),
                torch.tensor([float(example.label)]),
                pos_weight=positive_weight,
            )
            total = classification
            classification_correct += int((logit >= 0).item() == bool(example.label))
            if example.preferred_edge_ids is not None and example.rejected_edge_ids is not None:
                rank_loss = preference_loss(
                    costs,
                    example.preferred_edge_ids,
                    example.rejected_edge_ids,
                    example.inputs.edge_ids,
                )
                total = total + preference_weight * rank_loss
                preference_pairs += 1
                ranking_correct += int(
                    path_cost(costs, example.preferred_edge_ids, example.inputs.edge_ids)
                    < path_cost(costs, example.rejected_edge_ids, example.inputs.edge_ids)
                )
            final_losses.append(total)
    summary = DecisionTrainingSummary(
        examples=len(prepared),
        riders=len({example.rider_group for example in prepared}),
        followed=negatives,
        deviated=positives,
        epochs=epochs,
        initial_loss=initial_loss,
        final_loss=float(torch.stack(final_losses).mean()),
        in_sample_classification_correct=classification_correct,
        preference_pairs=preference_pairs,
        in_sample_ranking_correct=ranking_correct,
    )
    return model, summary


def prepare_synthetic(fixture_path: Path) -> tuple[list[PreparedDecision], dict, dict]:
    graph, scenarios = read_fixture(fixture_path)
    choices = prepare_choices(graph, scenarios)
    prepared = []
    for choice in choices:
        if (
            choice.deviation_label is None
            or choice.suggested_edge_ids is None
            or "exclude_nonrouting_reason_from_preference_pairs" in choice.quality_flags
        ):
            continue
        prepared.append(
            PreparedDecision(
                example_key=choice.case_id,
                rider_group=choice.case_id,
                label=int(choice.deviation_label),
                suggested_edge_ids=choice.suggested_edge_ids,
                preferred_edge_ids=choice.preferred_edge_ids,
                rejected_edge_ids=choice.rejected_edge_ids,
                inputs=build_model_input(graph, choice),
            )
        )
    provenance = {
        "kind": "synthetic-only multitask training smoke test",
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "not_a_performance_claim": True,
    }
    return prepared, {}, provenance


def prepare_real(
    graphml: Path,
    node_features: Path,
    approved_decisions: Path,
) -> tuple[list[PreparedDecision], dict, dict]:
    graph_data = build_real_graph_data(graphml, node_features)
    decisions = load_approved_decisions(approved_decisions, graph_data)
    prepared = []
    for decision in decisions:
        inputs = graph_data.build_model_input(decision.destination_node_id)
        prepared.append(
            PreparedDecision(
                example_key=decision.example_key,
                rider_group=decision.rider_group,
                label=decision.label,
                suggested_edge_ids=decision.suggested_edge_ids,
                preferred_edge_ids=decision.observed_edge_ids if decision.label == 1 else None,
                rejected_edge_ids=decision.suggested_edge_ids if decision.label == 1 else None,
                inputs=inputs,
            )
        )
    provenance = {
        "kind": "approved real decision training smoke run",
        "approved_artifact_sha256": hashlib.sha256(approved_decisions.read_bytes()).hexdigest(),
        "traffic": "unknown for historical decisions; no current Mapbox data attached retrospectively",
        "evaluation": "in-sample only; not a rider-generalization estimate",
    }
    return prepared, graph_data.model_configuration(), provenance


def _save_training(
    model: DecisionPreferenceModel,
    summary: DecisionTrainingSummary,
    provenance: dict,
    output: Path,
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
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--fixture", type=Path, default=Path("data/synthetic/rider_preference_fixture.json"))
    synthetic.add_argument("--output", type=Path, default=Path("outputs/multitask_synthetic.pt"))
    real = subparsers.add_parser("real")
    real.add_argument("graphml", type=Path)
    real.add_argument("node_features", type=Path)
    real.add_argument("approved_decisions", type=Path)
    real.add_argument("--output", type=Path, default=Path("outputs/multitask_real.pt"))
    for command in (synthetic, real):
        command.add_argument("--epochs", type=int, default=40)
        command.add_argument("--learning-rate", type=float, default=0.005)
        command.add_argument("--preference-weight", type=float, default=1.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "synthetic":
        prepared, configuration, provenance = prepare_synthetic(args.fixture)
    else:
        prepared, configuration, provenance = prepare_real(args.graphml, args.node_features, args.approved_decisions)
    model, summary = train_decision_model(
        prepared,
        configuration,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        preference_weight=args.preference_weight,
    )
    print(json.dumps(_save_training(model, summary, provenance, args.output), indent=2))


if __name__ == "__main__":
    main()
