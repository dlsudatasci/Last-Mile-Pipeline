"""Build model inputs using information available at the routing decision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .fixture import ChoiceExample, RoadGraph, RoutingRequest, TrafficSnapshot


CONGESTION = {"low": 0.0, "moderate": 0.33, "heavy": 0.67, "severe": 1.0}


@dataclass(frozen=True)
class ModelInput:
    node_features: Tensor
    edge_index: Tensor
    edge_static: Tensor
    temporal_edge_features: tuple[Tensor, ...]
    destination_features: Tensor
    edge_ids: tuple[str, ...]


def build_model_input(graph: RoadGraph, choice: ChoiceExample | RoutingRequest) -> ModelInput:
    """Encode only pre-decision traffic and motorcycle-accessible graph edges.

    The traffic-to-path mapping here belongs to the toy fixture. Real traffic
    needs its own spatial alignment to OSM edges before it uses this model.
    """
    nodes = {node["node_id"]: node for node in graph.nodes}
    if choice.destination_node_id not in nodes:
        raise ValueError(f"Unknown destination node {choice.destination_node_id}")
    node_ids = tuple(nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    xs = [float(node["x_m"]) for node in graph.nodes]
    ys = [float(node["y_m"]) for node in graph.nodes]
    x_min, y_min = min(xs), min(ys)
    x_scale, y_scale = max(max(xs) - x_min, 1.0), max(max(ys) - y_min, 1.0)

    node_features = torch.tensor(
        [
            [
                (float(nodes[node_id]["x_m"]) - x_min) / x_scale,
                (float(nodes[node_id]["y_m"]) - y_min) / y_scale,
                float(nodes[node_id]["traffic_signal"]),
            ]
            for node_id in node_ids
        ],
        dtype=torch.float32,
    )
    allowed = tuple(edge for edge in graph.edges if edge.motorcycle_allowed)
    if not allowed:
        raise ValueError("Graph has no motorcycle-accessible edges")
    edge_index = torch.tensor(
        [[node_index[edge.u] for edge in allowed], [node_index[edge.v] for edge in allowed]],
        dtype=torch.long,
    )
    edge_static = torch.tensor(
        [
            [edge.length_m / 100.0, edge.reference_time_s / 60.0, 1.0]
            for edge in allowed
        ],
        dtype=torch.float32,
    )
    destination = nodes[choice.destination_node_id]
    destination_features = torch.tensor(
        [
            [
                (float(destination["x_m"]) - float(nodes[edge.v]["x_m"])) / x_scale,
                (float(destination["y_m"]) - float(nodes[edge.v]["y_m"])) / y_scale,
            ]
            for edge in allowed
        ],
        dtype=torch.float32,
    )

    main_edges = set(graph.candidate_paths["main"])
    side_edges = set(graph.candidate_paths["side"])
    snapshots: tuple[TrafficSnapshot | None, ...] = choice.traffic_history or (None,)
    temporal_features: list[Tensor] = []
    for snapshot in snapshots:
        rows: list[list[float]] = []
        for edge in allowed:
            congestion = None
            if snapshot is not None:
                if edge.edge_id in main_edges:
                    congestion = snapshot.main_congestion
                elif edge.edge_id in side_edges:
                    congestion = snapshot.side_congestion
            if congestion not in CONGESTION and congestion is not None:
                raise ValueError(f"Unknown congestion category {congestion}")
            rows.append(
                [
                    CONGESTION[congestion] if congestion is not None else 0.0,
                    1.0 if congestion is not None else 0.0,
                    min(1.0, (choice.decision_time_s - snapshot.observed_at_s) / 300.0)
                    if snapshot is not None else 1.0,
                ]
            )
        temporal_features.append(torch.tensor(rows, dtype=torch.float32))

    return ModelInput(
        node_features=node_features,
        edge_index=edge_index,
        edge_static=edge_static,
        temporal_edge_features=tuple(temporal_features),
        destination_features=destination_features,
        edge_ids=tuple(edge.edge_id for edge in allowed),
    )
