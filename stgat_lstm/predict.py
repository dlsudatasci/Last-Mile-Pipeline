"""Reload a synthetic checkpoint and route with its learned preference costs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .features import build_model_input
from .fixture import prepare_routing_request, read_fixture
from .model import PreferenceModel
from .routing import shortest_preference_path


def route_from_checkpoint(checkpoint_path: Path, fixture_path: Path, case_id: str) -> dict:
    """Reload only a checkpoint made for this exact synthetic fixture."""
    torch.set_num_threads(1)
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("provenance") != "synthetic-only":
        raise ValueError("This command accepts only an explicitly synthetic checkpoint")
    if checkpoint.get("fixture_sha256") != fixture_hash:
        raise ValueError("Checkpoint was trained with a different fixture")

    graph, scenarios = read_fixture(fixture_path)
    matches = [scenario for scenario in scenarios if scenario["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one synthetic scenario named {case_id}, found {len(matches)}")
    request = prepare_routing_request(matches[0])
    model_config = checkpoint.get("model_config") or {
        "architecture": checkpoint["architecture"],
        "hidden_channels": int(checkpoint["hidden_channels"]),
    }
    model = PreferenceModel(**model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    inputs = build_model_input(graph, request)
    with torch.no_grad():
        scores = model(inputs)
    costs = {edge_id: float(scores[index]) for index, edge_id in enumerate(inputs.edge_ids)}
    route = shortest_preference_path(graph, costs, request.origin_node_id, request.destination_node_id)
    return {
        "provenance": "SYNTHETIC ONLY - model checkpoint and toy road graph",
        "architecture": checkpoint["architecture"],
        "case_id": case_id,
        "route_edge_ids": list(route.edge_ids),
        "learned_preference_cost": route.total_preference_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("data/synthetic/rider_preference_fixture.json"))
    parser.add_argument("--case-id", default="syn_case_01")
    args = parser.parse_args()
    print(json.dumps(route_from_checkpoint(args.checkpoint, args.fixture, args.case_id), indent=2))


if __name__ == "__main__":
    main()
