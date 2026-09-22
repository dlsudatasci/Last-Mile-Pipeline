"""Reload a trained multitask checkpoint and score one approved decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .mapbox_traffic import build_edge_snapshot
from .model import DecisionPreferenceModel, path_cost
from .real_data import build_real_graph_data, load_approved_decisions
from .real_routing import shortest_real_preference_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("node_features", type=Path)
    parser.add_argument("approved_decisions", type=Path)
    parser.add_argument("--example-key")
    parser.add_argument("--traffic-observation", type=Path, action="append", default=[])
    args = parser.parse_args()

    graph_data = build_real_graph_data(args.graphml, args.node_features)
    decisions = load_approved_decisions(args.approved_decisions, graph_data)
    if args.example_key:
        matching = [example for example in decisions if example.example_key == args.example_key]
        if not matching:
            raise ValueError(f"Unknown approved example key: {args.example_key}")
        example = matching[0]
    else:
        example = decisions[0]

    temporal = None
    traffic_note = "historical traffic unknown"
    if args.traffic_observation:
        documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.traffic_observation]
        documents.sort(key=lambda document: document.get("requested_at", ""))
        temporal = tuple(build_edge_snapshot(graph_data, document) for document in documents)
        traffic_note = f"{len(temporal)} supplied Mapbox snapshot(s)"

    inputs = graph_data.build_model_input(example.destination_node_id, temporal_edge_features=temporal)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    configuration = checkpoint.get("model_config", {})
    if configuration.get("model_type") != "decision_preference_multitask":
        raise ValueError("Checkpoint is not a decision-preference multitask model")
    model = DecisionPreferenceModel(**configuration["backbone"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        costs, logit = model(inputs, example.suggested_edge_ids)
        probability = float(torch.sigmoid(logit))
        suggested_cost = float(path_cost(costs, example.suggested_edge_ids, inputs.edge_ids))
        observed_cost = float(path_cost(costs, example.observed_edge_ids, inputs.edge_ids))
        edge_costs = {edge_id: float(costs[index]) for index, edge_id in enumerate(inputs.edge_ids)}
        recommended = shortest_real_preference_path(
            graph_data,
            edge_costs,
            example.origin_node_id,
            example.destination_node_id,
        )

    road_names = []
    for edge_id in recommended.edge_ids:
        u, v, key = edge_id.split("|", 2)
        data = next(data for candidate_key, data in graph_data.graph[u][v].items() if str(candidate_key) == key)
        name = str(data.get("name") or "Unnamed road")
        if not road_names or road_names[-1] != name:
            road_names.append(name)

    print(
        json.dumps(
            {
                "example_key": example.example_key,
                "deviation_probability": probability,
                "predicted_label": "deviated" if probability >= 0.5 else "followed",
                "approved_label": "deviated" if example.label else "followed",
                "suggested_path_preference_cost": suggested_cost,
                "observed_path_preference_cost": observed_cost,
                "recommended_route": {
                    "edge_ids": list(recommended.edge_ids),
                    "road_names": road_names,
                    "total_learned_preference_cost": recommended.total_preference_cost,
                },
                "traffic": traffic_note,
                "interpretation": "in-sample checkpoint reload demonstration, not held-out accuracy",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
