"""Route on the real Taft OSM graph using positive learned edge costs."""

from __future__ import annotations

import heapq
import math
from typing import Mapping

import torch

from .model import PreferenceModel
from .real_data import RealGraphData
from .routing import RouteResult


def shortest_real_preference_path(
    graph_data: RealGraphData,
    edge_costs: Mapping[str, float],
    origin_node_id: str,
    destination_node_id: str,
) -> RouteResult:
    nodes = set(graph_data.node_ids)
    if origin_node_id not in nodes or destination_node_id not in nodes:
        raise ValueError("Route endpoint is outside the Taft graph")
    if origin_node_id == destination_node_id:
        return RouteResult((), 0.0)
    restricted_index = graph_data.schema.edge_static.index("access_restricted")
    adjacency: dict[str, list[tuple[str, str, float]]] = {node_id: [] for node_id in graph_data.node_ids}
    for index, edge_id in enumerate(graph_data.edge_ids):
        if float(graph_data.edge_static[index, restricted_index]) == 1.0:
            continue
        if edge_id not in edge_costs:
            raise ValueError(f"No learned score for routable edge {edge_id}")
        cost = float(edge_costs[edge_id])
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError(f"Edge {edge_id} has a nonpositive or nonfinite preference cost")
        u, v, _ = edge_id.split("|", 2)
        adjacency[u].append((v, edge_id, cost))

    distances = {origin_node_id: 0.0}
    previous: dict[str, tuple[str, str]] = {}
    frontier = [(0.0, origin_node_id)]
    while frontier:
        cost_so_far, node_id = heapq.heappop(frontier)
        if cost_so_far > distances[node_id]:
            continue
        if node_id == destination_node_id:
            break
        for next_node, edge_id, edge_cost in adjacency[node_id]:
            candidate = cost_so_far + edge_cost
            if candidate < distances.get(next_node, math.inf):
                distances[next_node] = candidate
                previous[next_node] = (node_id, edge_id)
                heapq.heappush(frontier, (candidate, next_node))
    if destination_node_id not in distances:
        raise ValueError(f"No routable path from {origin_node_id} to {destination_node_id}")

    reverse_edges = []
    cursor = destination_node_id
    while cursor != origin_node_id:
        prior_node, edge_id = previous[cursor]
        reverse_edges.append(edge_id)
        cursor = prior_node
    return RouteResult(tuple(reversed(reverse_edges)), distances[destination_node_id])


def score_and_route_real_graph(
    model: PreferenceModel,
    graph_data: RealGraphData,
    origin_node_id: str,
    destination_node_id: str,
) -> RouteResult:
    """Score one routing context, then run Dijkstra on its learned costs."""
    model.eval()
    inputs = graph_data.build_model_input(destination_node_id)
    with torch.no_grad():
        scores = model(inputs)
    if scores.shape != (len(graph_data.edge_ids),):
        raise ValueError("Model returned one score per OSM edge")
    costs = {edge_id: float(scores[index]) for index, edge_id in enumerate(graph_data.edge_ids)}
    return shortest_real_preference_path(
        graph_data,
        costs,
        origin_node_id,
        destination_node_id,
    )
