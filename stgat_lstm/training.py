"""Pairwise preference learning for synthetic or later validated choices."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .features import build_model_input
from .fixture import ChoiceExample, RoadGraph
from .model import PreferenceModel, path_cost, preference_loss


@dataclass(frozen=True)
class TrainingSummary:
    architecture: str
    examples: int
    epochs: int
    initial_loss: float
    final_loss: float
    in_sample_correct: int


def _usable_choices(choices: list[ChoiceExample]) -> list[ChoiceExample]:
    return [choice for choice in choices if choice.preferred_edge_ids is not None and choice.rejected_edge_ids is not None]


def train_preference_model(
    graph: RoadGraph,
    choices: list[ChoiceExample],
    *,
    architecture: str = "stgat_lstm",
    epochs: int = 30,
    learning_rate: float = 0.01,
    seed: int = 7,
) -> tuple[PreferenceModel, TrainingSummary]:
    """Train on explicit, intentional path comparisons only.

    This does not create a validation split. Metrics on the hand-built fixture
    are in-sample smoke checks, never thesis performance estimates.
    """
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    examples = _usable_choices(choices)
    if not examples:
        raise ValueError("No usable intentional preference pairs")
    torch.manual_seed(seed)
    model = PreferenceModel(architecture=architecture)
    prepared = [(choice, build_model_input(graph, choice)) for choice in examples]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_loss = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        losses = []
        for choice, inputs in prepared:
            scores = model(inputs)
            losses.append(preference_loss(scores, choice.preferred_edge_ids, choice.rejected_edge_ids, inputs.edge_ids))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Training produced a nonfinite preference loss")
        if epoch == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_losses = []
        correct = 0
        for choice, inputs in prepared:
            scores = model(inputs)
            final_losses.append(
                preference_loss(scores, choice.preferred_edge_ids, choice.rejected_edge_ids, inputs.edge_ids)
            )
            preferred = path_cost(scores, choice.preferred_edge_ids, inputs.edge_ids)
            rejected = path_cost(scores, choice.rejected_edge_ids, inputs.edge_ids)
            correct += int(preferred < rejected)
        final_loss = float(torch.stack(final_losses).mean())
    return model, TrainingSummary(architecture, len(prepared), epochs, initial_loss, final_loss, correct)
