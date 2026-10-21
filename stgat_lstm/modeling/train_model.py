"""Train the shared outgoing-road choice and path-preference model.

Real training accepts only a finalized, validated decision artifact.
The training report contains in-sample checks. Use the `evaluate` command for
held-out-rider results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from stgat_lstm.modeling.model import DecisionPreferenceModel, ModelInput, path_cost, preference_loss
from stgat_lstm.preprocessing.graph_data import (RealDecisionExample, RealGraphData, accessible_outgoing_edge_ids,
                         build_real_graph_data, load_approved_decisions, path_choice_edge_ids)
from stgat_lstm.preprocessing.mapbox_traffic import TrafficArchive
from stgat_lstm.preprocessing.gps_traffic import GPSTrafficArchive


@dataclass(frozen=True)
class PreparedRoadChoice:
    source_node_id: str
    candidate_edge_ids: tuple[str, ...]
    chosen_edge_id: str
    suggested_edge_id: str
    deviated: int


@dataclass(frozen=True)
class PreparedDecision:
    example_key: str
    rider_group: str
    label: int
    suggested_edge_ids: tuple[str, ...]
    choice_targets: tuple[PreparedRoadChoice, ...]
    preferred_edge_ids: tuple[str, ...] | None
    rejected_edge_ids: tuple[str, ...] | None
    inputs: ModelInput
    ride_group: str | None = None
    decision_timestamp_ms: int | None = None
    destination_node_id: str | None = None


@dataclass(frozen=True)
class DecisionTrainingSummary:
    examples: int
    riders: int
    followed: int
    deviated: int
    outgoing_choices: int
    followed_outgoing_choices: int
    deviated_outgoing_choices: int
    deviated_choice_weight: float
    epochs: int
    initial_loss: float
    final_loss: float
    in_sample_outgoing_choice_correct: int
    preference_pairs: int
    in_sample_ranking_correct: int
    epoch_losses: tuple[float, ...] = ()
    selected_epoch: int = 0
    validation_history: tuple[dict, ...] = ()
    training_options: dict | None = None


def first_junction_visits(prepared: list[PreparedDecision]) -> tuple[list[PreparedDecision], dict]:
    """Keep one supervised visit per junction within each training example."""
    result, omitted = [], []
    for example in prepared:
        seen, targets = {}, []
        for index, target in enumerate(example.choice_targets):
            if target.source_node_id in seen:
                if target.deviated:
                    raise ValueError("A later alternative-road target needs a separate decision context; do not discard it")
                omitted.append({"example_key": example.example_key, "rider_group": example.rider_group,
                    "source_node_id": target.source_node_id, "target_index": index,
                    "kept_chosen_edge_id": seen[target.source_node_id].chosen_edge_id,
                    "omitted_chosen_edge_id": target.chosen_edge_id, "omitted_label": target.deviated})
                continue
            seen[target.source_node_id] = target
            targets.append(target)
        result.append(replace(example, choice_targets=tuple(targets)) if len(targets) != len(example.choice_targets) else example)
    return result, {"examples_before": len(prepared), "examples_after": len(result),
        "choices_before": sum(len(e.choice_targets) for e in prepared),
        "choices_after": sum(len(e.choice_targets) for e in result), "omitted_targets": omitted,
        "alternative_targets_before": sum(t.deviated for e in prepared for t in e.choice_targets),
        "alternative_targets_after": sum(t.deviated for e in result for t in e.choice_targets)}


def mask_traffic_history(inputs: ModelInput, probability: float,
                         generator: torch.Generator) -> ModelInput:
    """Hide observed speed context without altering roads or chosen labels."""
    if not 0 <= probability < 1:
        raise ValueError("Traffic masking probability must be in [0, 1)")
    if probability == 0:
        return inputs
    frames = []
    for original in inputs.temporal_edge_features:
        if original.shape[1] != 5:
            raise ValueError("Traffic masking requires the five-field GPS schema")
        frame = original.clone()
        hidden = (torch.rand(len(frame), generator=generator) < probability) & (frame[:, 3] == 1)
        frame[hidden] = frame.new_tensor([0., 0., 0., 0., 1.])
        frames.append(frame)
    return replace(inputs, temporal_edge_features=tuple(frames))


def _validation_choice_score(model, examples: list[PreparedDecision], batch_size: int) -> dict:
    losses, correct = [], 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            for example, (_, logits) in zip(batch, _forward_batch(model, batch)):
                lookup = {edge: i for i, edge in enumerate(example.inputs.edge_ids)}
                for target in example.choice_targets:
                    values = logits[[lookup[e] for e in target.candidate_edge_ids]]
                    chosen = target.candidate_edge_ids.index(target.chosen_edge_id)
                    correct += int(int(values.argmax()) == chosen)
                    losses.append(float(-F.log_softmax(values, dim=0)[chosen]))
    return {"accuracy": correct / len(losses), "negative_log_likelihood": sum(losses) / len(losses)}


def train_decision_model(
    prepared: list[PreparedDecision],
    model_configuration: dict,
    *,
    epochs: int = 40,
    learning_rate: float = 0.005,
    preference_weight: float = 1.0,
    alternative_choice_weight: float | None = None,
    seed: int = 17,
    batch_size: int = 8,
    progress: Callable[[int, float], None] | None = None,
    validation: list[PreparedDecision] | None = None,
    patience: int = 3,
    validation_interval: int = 5,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    traffic_mask_probability: float = 0.0,
    balance_rides: bool = False,
    select_validation_checkpoint: bool = True,
    checkpoint_selection: str = "accuracy",
    alternative_weight_power: float = 1.0,
) -> tuple[DecisionPreferenceModel, DecisionTrainingSummary]:
    if (epochs <= 0 or learning_rate <= 0 or preference_weight < 0 or batch_size <= 0
            or (alternative_choice_weight is not None and alternative_choice_weight <= 0)):
        raise ValueError("Training hyperparameters are invalid")
    if (patience <= 0 or validation_interval <= 0 or weight_decay < 0
            or not 0 <= label_smoothing < 1 or not 0 <= traffic_mask_probability < 1):
        raise ValueError("Regularization or validation settings are invalid")
    if checkpoint_selection not in {"accuracy", "nll"}:
        raise ValueError("Checkpoint selection must be accuracy or nll")
    if not 0 < alternative_weight_power <= 1 or (alternative_choice_weight is not None and alternative_weight_power != 1):
        raise ValueError("Alternative weight power must be in (0, 1] and applies only to automatic weights")
    if validation is not None:
        if not validation or {e.rider_group for e in prepared} & {e.rider_group for e in validation}:
            raise ValueError("Validation must contain separate riders")
    if not prepared:
        raise ValueError("No approved decision examples")
    target_rows = [target for example in prepared for target in example.choice_targets]
    if not target_rows:
        raise ValueError("No road-level choice targets")
    for target in target_rows:
        if target.deviated not in {0, 1}:
            raise ValueError("Derived deviation labels must be zero or one")
        if len(target.candidate_edge_ids) < 2:
            raise ValueError("Outgoing-road targets require at least two accessible choices")
        if target.chosen_edge_id not in target.candidate_edge_ids:
            raise ValueError("Chosen road is absent from its outgoing-road candidates")
        if target.suggested_edge_id not in target.candidate_edge_ids:
            raise ValueError("Suggested road is absent from its outgoing-road candidates")
        if target.deviated != int(target.chosen_edge_id != target.suggested_edge_id):
            raise ValueError("Derived deviation label disagrees with the chosen and suggested roads")
    positives = sum(target.deviated == 1 for target in target_rows)
    negatives = len(target_rows) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Outgoing-road training requires followed and deviated choices")
    deviated_choice_weight = (
        negatives / positives if alternative_choice_weight is None else alternative_choice_weight
    )
    torch.manual_seed(seed)
    model = DecisionPreferenceModel(**model_configuration)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                 if weight_decay else torch.optim.Adam(model.parameters(), lr=learning_rate))
    shuffle_generator = torch.Generator().manual_seed(seed)
    mask_generator = torch.Generator().manual_seed(seed + 1000)
    ride_counts = {}
    for example in prepared:
        key = example.ride_group or example.example_key
        ride_counts[key] = ride_counts.get(key, 0) + len(example.choice_targets)
    if balance_rides and alternative_choice_weight is None:
        weighted_positive = sum(t.deviated / ride_counts[e.ride_group or e.example_key]
                                for e in prepared for t in e.choice_targets)
        weighted_negative = sum((1 - t.deviated) / ride_counts[e.ride_group or e.example_key]
                                for e in prepared for t in e.choice_targets)
        deviated_choice_weight = weighted_negative / weighted_positive
    if alternative_weight_power != 1:
        deviated_choice_weight = deviated_choice_weight ** alternative_weight_power
    best_key, best_state, selected_epoch, stale = None, None, 0, 0
    validation_history = []
    initial_loss = 0.0
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        order = torch.randperm(len(prepared), generator=shuffle_generator).tolist()
        shuffled = [prepared[index] for index in order]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            if traffic_mask_probability:
                batch = [replace(e, inputs=mask_traffic_history(e.inputs, traffic_mask_probability,
                                                               mask_generator)) for e in batch]
            optimizer.zero_grad()
            outputs = _forward_batch(model, batch)
            choice_losses: list[Tensor] = []
            choice_weights: list[float] = []
            ranking_losses: list[Tensor] = []
            for example, (costs, edge_logits) in zip(batch, outputs):
                lookup = {edge_id: index for index, edge_id in enumerate(example.inputs.edge_ids)}
                for target in example.choice_targets:
                    try:
                        indices = [lookup[edge_id] for edge_id in target.candidate_edge_ids]
                        chosen = target.candidate_edge_ids.index(target.chosen_edge_id)
                    except (KeyError, ValueError) as exc:
                        raise ValueError(f"Outgoing-road target contains an unavailable edge: {exc}") from exc
                    choice_losses.append(F.cross_entropy(
                        edge_logits[indices].unsqueeze(0),
                        torch.tensor([chosen], dtype=torch.long, device=edge_logits.device),
                        label_smoothing=label_smoothing,
                    ))
                    weight = deviated_choice_weight if target.deviated else 1.0
                    if balance_rides:
                        weight /= ride_counts[example.ride_group or example.example_key]
                    choice_weights.append(weight)
                if example.preferred_edge_ids is not None and example.rejected_edge_ids is not None:
                    ranking_losses.append(preference_loss(
                        costs,
                        example.preferred_edge_ids,
                        example.rejected_edge_ids,
                        example.inputs.edge_ids,
                    ))
            losses = torch.stack(choice_losses)
            weights = losses.new_tensor(choice_weights)
            choice_loss = (losses * weights).sum() / weights.sum()
            loss = choice_loss
            if ranking_losses:
                loss = loss + preference_weight * torch.stack(ranking_losses).mean()
            if not torch.isfinite(loss):
                raise ValueError("Training produced a nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss_sum += float(loss.detach()) * len(batch)
        epoch_loss = epoch_loss_sum / len(prepared)
        if epoch == 0:
            initial_loss = epoch_loss
        epoch_losses.append(epoch_loss)
        if progress is not None:
            progress(epoch + 1, epoch_loss)
        if validation is not None and (epoch == 0 or (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs):
            score = _validation_choice_score(model, validation, batch_size)
            validation_history.append({"epoch": epoch + 1, **score})
            if not select_validation_checkpoint:
                continue
            key = ((-score["negative_log_likelihood"], score["accuracy"])
                   if checkpoint_selection == "nll" else
                   (score["accuracy"], -score["negative_log_likelihood"]))
            if best_key is None or key > best_key:
                best_key, best_state, selected_epoch, stale = key, deepcopy(model.state_dict()), epoch + 1, 0
            else:
                stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    final_choice_losses: list[Tensor] = []
    final_choice_weights: list[float] = []
    choice_correct = 0
    ranking_correct = 0
    preference_pairs = 0
    final_ranking_losses: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start + batch_size]
            for example, (costs, edge_logits) in zip(batch, _forward_batch(model, batch)):
                lookup = {edge_id: index for index, edge_id in enumerate(example.inputs.edge_ids)}
                for target in example.choice_targets:
                    indices = [lookup[edge_id] for edge_id in target.candidate_edge_ids]
                    chosen = target.candidate_edge_ids.index(target.chosen_edge_id)
                    candidate_logits = edge_logits[indices]
                    final_choice_losses.append(F.cross_entropy(
                        candidate_logits.unsqueeze(0),
                        torch.tensor([chosen], dtype=torch.long, device=edge_logits.device),
                        label_smoothing=label_smoothing,
                    ))
                    weight = deviated_choice_weight if target.deviated else 1.0
                    if balance_rides:
                        weight /= ride_counts[example.ride_group or example.example_key]
                    final_choice_weights.append(weight)
                    choice_correct += int(int(candidate_logits.argmax()) == chosen)
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
        losses = torch.stack(final_choice_losses)
        weights = losses.new_tensor(final_choice_weights)
        final_loss = (losses * weights).sum() / weights.sum()
        if final_ranking_losses:
            final_loss = final_loss + preference_weight * torch.stack(final_ranking_losses).mean()
    summary = DecisionTrainingSummary(
        examples=len(prepared),
        riders=len({example.rider_group for example in prepared}),
        followed=sum(example.label == 0 for example in prepared),
        deviated=sum(example.label == 1 for example in prepared),
        outgoing_choices=len(target_rows),
        followed_outgoing_choices=negatives,
        deviated_outgoing_choices=positives,
        deviated_choice_weight=deviated_choice_weight,
        epochs=len(epoch_losses),
        initial_loss=initial_loss,
        final_loss=float(final_loss),
        in_sample_outgoing_choice_correct=choice_correct,
        preference_pairs=preference_pairs,
        in_sample_ranking_correct=ranking_correct,
        epoch_losses=tuple(epoch_losses),
        selected_epoch=selected_epoch or len(epoch_losses),
        validation_history=tuple(validation_history),
        training_options={"weight_decay": weight_decay, "label_smoothing": label_smoothing,
                          "traffic_mask_probability": traffic_mask_probability,
                          "balance_rides": balance_rides, "validation_interval": validation_interval,
                          "select_validation_checkpoint": select_validation_checkpoint,
                          "checkpoint_selection": checkpoint_selection,
                          "alternative_weight_power": alternative_weight_power,
                          "patience": patience if validation is not None else None},
    )
    return model, summary


def _forward_batch(
    model: DecisionPreferenceModel,
    examples: list[PreparedDecision],
) -> list[tuple[Tensor, Tensor]]:
    """Run independent graph examples as one disconnected PyG graph."""
    if not examples:
        return []
    step_count = len(examples[0].inputs.temporal_edge_features)
    if any(len(example.inputs.temporal_edge_features) != step_count for example in examples):
        return [model(example.inputs) for example in examples]
    node_parts = []
    edge_index_parts = []
    edge_static_parts = []
    routable_parts = []
    destination_parts = []
    temporal_parts = [[] for _ in range(step_count)]
    edge_counts = []
    node_offset = 0
    for example in examples:
        inputs = example.inputs
        node_parts.append(inputs.node_features)
        edge_index_parts.append(inputs.edge_index + node_offset)
        edge_static_parts.append(inputs.edge_static)
        routable_parts.append(inputs.routable_mask)
        destination_parts.append(inputs.destination_features)
        for step, snapshot in enumerate(inputs.temporal_edge_features):
            temporal_parts[step].append(snapshot)
        edge_counts.append(len(inputs.edge_ids))
        node_offset += inputs.node_features.shape[0]
    merged = ModelInput(
        node_features=torch.cat(node_parts),
        edge_index=torch.cat(edge_index_parts, dim=1),
        edge_static=torch.cat(edge_static_parts),
        temporal_edge_features=tuple(torch.cat(parts) for parts in temporal_parts),
        destination_features=torch.cat(destination_parts),
        routable_mask=torch.cat(routable_parts),
        edge_ids=tuple(
            f"{example_index}:{edge_id}"
            for example_index, example in enumerate(examples)
            for edge_id in example.inputs.edge_ids
        ),
    )
    costs, logits = model(merged)
    result = []
    offset = 0
    for count in edge_counts:
        result.append((costs[offset:offset + count], logits[offset:offset + count]))
        offset += count
    return result


def decision_choice_targets(
    decision: RealDecisionExample,
    graph_data: RealGraphData,
) -> tuple[PreparedRoadChoice, ...]:
    """Convert one followed/deviated route into supervised junction choices."""
    choice_edges = path_choice_edge_ids(graph_data, decision.suggested_edge_ids)
    if decision.suggested_choice_edge_id is not None or decision.chosen_choice_edge_id is not None:
        expected, chosen = decision.suggested_choice_edge_id, decision.chosen_choice_edge_id
        if expected is None or chosen is None:
            raise ValueError("Explicit road-choice supervision requires both roads")
        source = expected.split("|", 1)[0]
        candidates = accessible_outgoing_edge_ids(graph_data, source)
        if expected not in choice_edges or chosen not in candidates or chosen.split("|", 1)[0] != source:
            raise ValueError("Explicit road-choice target is not an accessible branching decision")
        if decision.label != int(expected != chosen):
            raise ValueError("Explicit road choice disagrees with derived deviation label")
        return (PreparedRoadChoice(source, candidates, chosen, expected, decision.label),)
    if decision.label == 0:
        targets = []
        for edge_id in choice_edges:
            source = edge_id.split("|", 1)[0]
            targets.append(PreparedRoadChoice(
                source_node_id=source,
                candidate_edge_ids=accessible_outgoing_edge_ids(graph_data, source),
                chosen_edge_id=edge_id,
                suggested_edge_id=edge_id,
                deviated=0,
            ))
        return tuple(targets)

    divergence = next((index for index, pair in enumerate(
        zip(decision.suggested_edge_ids, decision.observed_edge_ids)) if pair[0] != pair[1]), None)
    if divergence is None:
        divergence = min(len(decision.suggested_edge_ids), len(decision.observed_edge_ids))
    if divergence >= len(decision.suggested_edge_ids) or divergence >= len(decision.observed_edge_ids):
        raise ValueError(f"Deviated example {decision.example_key} has no observed outgoing alternative")
    rejected_edge = decision.suggested_edge_ids[divergence]
    chosen_edge = decision.observed_edge_ids[divergence]
    rejected_source = rejected_edge.split("|", 1)[0]
    chosen_source = chosen_edge.split("|", 1)[0]
    if rejected_source != chosen_source:
        raise ValueError(f"Deviated example {decision.example_key} does not diverge at one source node")
    if rejected_edge not in choice_edges:
        raise ValueError(f"Deviated example {decision.example_key} does not diverge at a branching road")

    targets = []
    for edge_id in decision.suggested_edge_ids[:divergence]:
        if edge_id not in choice_edges:
            continue
        source = edge_id.split("|", 1)[0]
        targets.append(PreparedRoadChoice(
            source_node_id=source,
            candidate_edge_ids=accessible_outgoing_edge_ids(graph_data, source),
            chosen_edge_id=edge_id,
            suggested_edge_id=edge_id,
            deviated=0,
        ))
    candidates = accessible_outgoing_edge_ids(graph_data, rejected_source)
    if chosen_edge not in candidates:
        raise ValueError(f"Observed alternative in {decision.example_key} is not an accessible outgoing road")
    targets.append(PreparedRoadChoice(
        source_node_id=rejected_source,
        candidate_edge_ids=candidates,
        chosen_edge_id=chosen_edge,
        suggested_edge_id=rejected_edge,
        deviated=1,
    ))
    return tuple(targets)


def prepare_real(
    graphml: Path,
    node_features: Path,
    approved_decisions: Path,
    *, traffic_archive: Path | None = None, gps_traffic_data: Path | None = None,
    history_steps: int = 6, history_interval_s: int = 300,
    traffic_max_age_s: int = 900, gps_profile_bin_s: int = 300,
) -> tuple[list[PreparedDecision], dict, dict]:
    if traffic_archive is not None and gps_traffic_data is not None:
        raise ValueError("Choose Mapbox traffic_archive or gps_traffic_data, not both")
    artifact_metadata = json.loads(approved_decisions.read_text(encoding="utf-8"))
    graph_data = build_real_graph_data(graphml, node_features, approved_decisions=approved_decisions)
    decisions = load_approved_decisions(approved_decisions, graph_data)
    prepared = []
    archive = TrafficArchive.from_directory(graph_data, traffic_archive) if traffic_archive else None
    gps_archive = GPSTrafficArchive.from_exports(graph_data, gps_traffic_data) if gps_traffic_data else None
    traffic_reports = {}
    for decision in decisions:
        temporal = None
        if archive is not None:
            temporal, traffic_reports[decision.example_key] = archive.sequence(
                decision.decision_timestamp_ms, steps=history_steps,
                interval_s=history_interval_s, max_age_s=traffic_max_age_s)
        elif gps_archive is not None:
            temporal, traffic_reports[decision.example_key] = gps_archive.profile_sequence(
                decision.decision_timestamp_ms,
                steps=history_steps,
                interval_s=history_interval_s,
                profile_bin_s=gps_profile_bin_s,
                exclude_ride_groups={decision.ride_group},
            )
        inputs = graph_data.build_model_input(decision.destination_node_id, temporal_edge_features=temporal)
        choice_targets = decision_choice_targets(decision, graph_data)
        if not choice_targets:
            raise ValueError(f"Example {decision.example_key} contains no road-choice target")
        prepared.append(
            PreparedDecision(
                example_key=decision.example_key,
                rider_group=decision.rider_group,
                label=decision.label,
                suggested_edge_ids=decision.suggested_edge_ids,
                choice_targets=choice_targets,
                preferred_edge_ids=(decision.observed_edge_ids
                                    if decision.label == 1 and decision.has_route_pair else None),
                rejected_edge_ids=(decision.suggested_edge_ids
                                   if decision.label == 1 and decision.has_route_pair else None),
                inputs=inputs,
                ride_group=decision.ride_group,
                decision_timestamp_ms=decision.decision_timestamp_ms,
                destination_node_id=decision.destination_node_id,
            )
        )
    provenance = {
        "kind": (
            "controlled synthetic capacity training run"
            if artifact_metadata.get("synthetic")
            else "approved real decision training run"
        ),
        "synthetic": bool(artifact_metadata.get("synthetic", False)),
        "approved_artifact_sha256": hashlib.sha256(approved_decisions.read_bytes()).hexdigest(),
        "traffic": "no historical traffic source supplied; dynamic values remain explicitly unknown",
        "evaluation": "in-sample only; not a rider-generalization estimate",
        "graphml_sha256": hashlib.sha256(graphml.read_bytes()).hexdigest(),
        "node_features_sha256": hashlib.sha256(node_features.read_bytes()).hexdigest(),
        "model_graph_buffer_m": graph_data.graph.graph["model_corridor_buffer_m"],
        "model_graph_nodes": len(graph_data.node_ids),
        "model_graph_edges": len(graph_data.edge_ids),
        "edge_dynamic_schema": list(graph_data.schema.edge_dynamic),
        "prediction_target": (
            "preferred accessible outgoing road at each branching source; deviation probability is "
            "derived as 1 - P(suggested road)"
        ),
    }
    if archive is not None:
        provenance["traffic"] = "archived pre-decision snapshots; missing/stale/future traffic stays unknown"
        provenance["traffic_history"] = traffic_reports
        provenance["traffic_policy"] = {"steps": history_steps, "interval_s": history_interval_s,
                                        "max_age_s": traffic_max_age_s}
    elif gps_archive is not None:
        provenance["traffic"] = (
            "historical rider-GPS motorcycle probe speeds; target ride and post-decision records excluded"
        )
        provenance["gps_traffic_build"] = gps_archive.summary
        provenance["traffic_history"] = traffic_reports
        provenance["traffic_policy"] = {
            "steps": history_steps,
            "interval_s": history_interval_s,
            "profile_bin_s": gps_profile_bin_s,
            "future_records_allowed": False,
            "target_ride_allowed": False,
        }
    return prepared, graph_data.model_configuration(), provenance


def attach_gps_traffic(
    prepared: list[PreparedDecision],
    archive: GPSTrafficArchive,
    *,
    excluded_riders: set[str] | None = None,
    history_steps: int = 6,
    history_interval_s: int = 300,
    gps_profile_bin_s: int = 300,
) -> tuple[list[PreparedDecision], dict[str, dict]]:
    """Attach fold-safe GPS histories without changing labels or static inputs."""
    result = []
    reports = {}
    for example in prepared:
        if example.decision_timestamp_ms is None or example.ride_group is None:
            raise ValueError("Prepared decision lacks GPS traffic metadata")
        temporal, report = archive.profile_sequence(
            example.decision_timestamp_ms,
            steps=history_steps,
            interval_s=history_interval_s,
            profile_bin_s=gps_profile_bin_s,
            exclude_rider_groups=excluded_riders,
            exclude_ride_groups={example.ride_group},
        )
        reports[example.example_key] = report
        result.append(replace(example, inputs=replace(example.inputs, temporal_edge_features=temporal)))
    return result, reports


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
    real.add_argument("--output", type=Path, default=Path("artifacts/checkpoints/gatv2_lstm_checkpoint.pt"))
    real.add_argument("--report-output", type=Path, default=Path("outputs/reports/training_report.json"))
    real.add_argument("--traffic-archive", type=Path)
    real.add_argument("--gps-traffic-data", type=Path,
                      help="rider export directory used to derive historical motorcycle probe speeds")
    real.add_argument("--history-steps", type=int, default=6)
    real.add_argument("--history-interval-seconds", type=int, default=300)
    real.add_argument("--traffic-max-age-seconds", type=int, default=900)
    real.add_argument("--gps-profile-bin-seconds", type=int, default=300)
    real.add_argument("--epochs", type=int, default=40)
    real.add_argument("--learning-rate", type=float, default=0.005)
    real.add_argument("--preference-weight", type=float, default=1.0)
    real.add_argument(
        "--alternative-choice-weight",
        type=float,
        help="loss weight for choosing an alternative road; default uses inverse class frequency",
    )
    real.add_argument(
        "--tuning-report",
        type=Path,
        help="synthetic evaluation report whose validation-selected settings should be reused",
    )
    real.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.tuning_report:
        tuning_document = json.loads(args.tuning_report.read_text(encoding="utf-8"))
        selected = tuning_document.get("hyperparameter_tuning", {}).get("selected")
        if not selected:
            parser.error("--tuning-report does not contain validation-selected settings")
        args.epochs = int(selected["epochs"])
        args.learning_rate = float(selected["learning_rate"])
        args.preference_weight = float(selected["preference_weight"])
        args.alternative_choice_weight = float(selected["alternative_choice_weight"])
    prepared, configuration, provenance = prepare_real(
        args.graphml, args.node_features, args.approved_decisions,
        traffic_archive=args.traffic_archive, gps_traffic_data=args.gps_traffic_data,
        history_steps=args.history_steps, history_interval_s=args.history_interval_seconds,
        traffic_max_age_s=args.traffic_max_age_seconds, gps_profile_bin_s=args.gps_profile_bin_seconds)
    if args.tuning_report and not provenance.get("synthetic"):
        parser.error("--tuning-report can only be used with a synthetic approved artifact")
    model, summary = train_decision_model(
        prepared,
        configuration,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        preference_weight=args.preference_weight,
        alternative_choice_weight=args.alternative_choice_weight,
        batch_size=args.batch_size,
        progress=lambda epoch, loss: (
            print(f"Epoch {epoch:>3}/{args.epochs}: loss {loss:.4f}")
            if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 10, 1) == 0
            else None
        ),
    )
    provenance["training_settings"] = {"epochs": args.epochs, "learning_rate": args.learning_rate,
                                       "preference_weight": args.preference_weight,
                                       "alternative_choice_weight": args.alternative_choice_weight,
                                       "batch_size": args.batch_size, "seed": 17,
                                       "tuning_report": str(args.tuning_report) if args.tuning_report else None}
    report = _save_training(model, summary, provenance, args.output, args.report_output)
    readable = report["training_summary"]
    print(f"Checkpoint saved to {report['checkpoint']}")
    print(f"Training report saved to {args.report_output or args.output.with_suffix('.json')}")
    print(f"Examples: {readable['examples']} from {readable['riders']} riders | "
          f"followed: {readable['followed']} | deviated: {readable['deviated']}")
    print(f"Outgoing-road choices: {readable['outgoing_choices']} | "
          f"followed suggestion: {readable['followed_outgoing_choices']} | "
          f"chose another road: {readable['deviated_outgoing_choices']}")
    print(f"Class balance: alternative-road choices receive {readable['deviated_choice_weight']:.2f}x "
          "loss weight")
    print(f"Loss: {readable['initial_loss']:.4f} -> {readable['final_loss']:.4f} "
          f"after {readable['epochs']} epochs")
    print(f"In-sample checks: {readable['in_sample_outgoing_choice_correct']}/"
          f"{readable['outgoing_choices']} preferred roads, "
          f"{readable['in_sample_ranking_correct']}/{readable['preference_pairs']} preference pairs")
    print("Use `python -m stgat_lstm evaluate` for held-out-rider results.")


if __name__ == "__main__":
    main()
