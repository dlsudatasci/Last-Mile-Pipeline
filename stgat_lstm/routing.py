"""Shortest path over learned, nonnegative motorcycle road costs."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from .fixture import RoadGraph


@dataclass(frozen=True)
class RouteResult:
    edge_ids: tuple[str, ...]
    total_preference_cost: float


def shortest_preference_path(
    graph: RoadGraph,
    edge_costs: Mapping[str, float],
    origin_node_id: str,
    destination_node_id: str,
) -> RouteResult:
    """Run Dijkstra on model output; costs are preferences, not travel seconds.

    The first implementation uses edge-additive scores. A later turn-aware
    model would need incoming-edge state in this search.
    """
    node_ids = {node["node_id"] for node in graph.nodes}
    if origin_node_id not in node_ids or destination_node_id not in node_ids:
        raise ValueError("Route endpoint is not in the graph")
    if origin_node_id == destination_node_id:
        return RouteResult(edge_ids=(), total_preference_cost=0.0)

    adjacency: dict[str, list[tuple[str, str, float]]] = {node_id: [] for node_id in node_ids}
    for edge in graph.edges:
        if not edge.motorcycle_allowed:
            continue
        if edge.edge_id not in edge_costs:
            raise ValueError(f"No learned score for accessible edge {edge.edge_id}")
        cost = float(edge_costs[edge.edge_id])
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError(f"Edge {edge.edge_id} has a nonpositive or nonfinite preference cost")
        adjacency[edge.u].append((edge.v, edge.edge_id, cost))

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
        raise ValueError(f"No accessible route from {origin_node_id} to {destination_node_id}")

    reverse_edges: list[str] = []
    cursor = destination_node_id
    while cursor != origin_node_id:
        prior_node, edge_id = previous[cursor]
        reverse_edges.append(edge_id)
        cursor = prior_node
    return RouteResult(tuple(reversed(reverse_edges)), distances[destination_node_id])
