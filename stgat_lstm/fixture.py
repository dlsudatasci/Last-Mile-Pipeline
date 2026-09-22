"""Normalize the explicitly synthetic fixture into preference examples.

The JSON `expected` section is a test oracle and is never passed to training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXCLUDED_REASONS = {
    "Wrong Turn",
    "Personal Stop (Meal, Restroom, Break, Refueling, etc.)",
    "Navigation Error / Map issue",
}


@dataclass(frozen=True)
class RoadEdge:
    edge_id: str
    u: str
    v: str
    key: int
    length_m: float
    reference_time_s: float
    motorcycle_allowed: bool


@dataclass(frozen=True)
class RoadGraph:
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[RoadEdge, ...]
    candidate_paths: dict[str, tuple[str, ...]]

    @property
    def edge_by_id(self) -> dict[str, RoadEdge]:
        return {edge.edge_id: edge for edge in self.edges}

    def check_path(self, edge_ids: tuple[str, ...]) -> tuple[str, str]:
        if not edge_ids:
            raise ValueError("A path must contain an edge")
        lookup = self.edge_by_id
        try:
            edges = [lookup[edge_id] for edge_id in edge_ids]
        except KeyError as exc:
            raise ValueError(f"Unknown path edge: {exc.args[0]}") from exc
        if any(not edge.motorcycle_allowed for edge in edges):
            raise ValueError("A training path uses a road forbidden to motorcycles")
        if any(first.v != second.u for first, second in zip(edges, edges[1:])):
            raise ValueError("A path has disconnected directed edges")
        return edges[0].u, edges[-1].v


@dataclass(frozen=True)
class TrafficSnapshot:
    observed_at_s: float
    available_at_s: float
    main_congestion: str
    side_congestion: str


@dataclass(frozen=True)
class ChoiceExample:
    case_id: str
    origin_node_id: str
    destination_node_id: str
    decision_time_s: float
    suggested_edge_ids: tuple[str, ...] | None
    observed_edge_ids: tuple[str, ...] | None
    deviation_label: bool | None
    preferred_edge_ids: tuple[str, ...] | None
    rejected_edge_ids: tuple[str, ...] | None
    traffic_history: tuple[TrafficSnapshot, ...]
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class RoutingRequest:
    """Only fields available before selecting a route."""

    case_id: str
    origin_node_id: str
    destination_node_id: str
    decision_time_s: float
    traffic_history: tuple[TrafficSnapshot, ...]


def _available_traffic(raw: dict[str, Any]) -> tuple[TrafficSnapshot, ...]:
    cutoff = float(raw["decision_time_s"])
    available = (
        TrafficSnapshot(
            observed_at_s=float(snapshot["observed_at_s"]),
            available_at_s=float(snapshot["available_at_s"]),
            main_congestion=snapshot["main_congestion"],
            side_congestion=snapshot["side_congestion"],
        )
        for snapshot in raw["traffic_history"]
        if float(snapshot["observed_at_s"]) <= cutoff
        and float(snapshot["available_at_s"]) <= cutoff
    )
    return tuple(sorted(available, key=lambda item: item.available_at_s))


def prepare_routing_request(scenario: dict[str, Any]) -> RoutingRequest:
    """Read only request context, never the observed path, survey or oracle."""
    if scenario.get("is_synthetic") is not True:
        raise ValueError("Routing request lacks synthetic provenance")
    raw = scenario["input"]
    return RoutingRequest(
        case_id=scenario["case_id"],
        origin_node_id=raw["origin_node_id"],
        destination_node_id=raw["destination_node_id"],
        decision_time_s=float(raw["decision_time_s"]),
        traffic_history=_available_traffic(raw),
    )


def read_fixture(path: str | Path) -> tuple[RoadGraph, list[dict[str, Any]]]:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if fixture.get("is_synthetic") is not True:
        raise ValueError("This loader only accepts explicitly synthetic fixtures")
    raw_graph = fixture["graph"]
    edges = tuple(
        RoadEdge(
            edge_id=raw["edge_id"],
            u=raw["u"],
            v=raw["v"],
            key=int(raw["key"]),
            length_m=float(raw["length_m"]),
            reference_time_s=float(raw["fictional_travel_time_s"]),
            motorcycle_allowed=raw["motorcycle_allowed"] is True,
        )
        for raw in raw_graph["edges"]
    )
    graph = RoadGraph(
        nodes=tuple(raw_graph["nodes"]),
        edges=edges,
        candidate_paths={key: tuple(value) for key, value in fixture["candidate_paths"].items()},
    )
    if len(graph.edge_by_id) != len(graph.edges):
        raise ValueError("Duplicate edge IDs in fixture")
    if len({node["node_id"] for node in graph.nodes}) != len(graph.nodes):
        raise ValueError("Duplicate node IDs in fixture")
    return graph, fixture["scenarios"]


def prepare_choices(graph: RoadGraph, scenarios: list[dict[str, Any]]) -> list[ChoiceExample]:
    choices: list[ChoiceExample] = []
    for scenario in scenarios:
        if scenario.get("is_synthetic") is not True:
            raise ValueError(f"Scenario {scenario.get('case_id')} lacks synthetic provenance")
        raw = scenario["input"]  # Deliberately never read scenario["expected"].
        cutoff = float(raw["decision_time_s"])
        history = raw["route_history"]
        prior = [route for route in history if float(route["generated_at_s"]) <= cutoff]
        if prior:
            latest_time = max(float(route["generated_at_s"]) for route in prior)
            matching = [route for route in prior if float(route["generated_at_s"]) == latest_time]
            if len(matching) != 1:
                raise ValueError(f"Ambiguous active route for {scenario['case_id']}")
            suggested = tuple(matching[0]["edge_ids"])
        else:
            suggested = None

        observed = tuple(raw["observed_edge_ids"]) if raw["observed_edge_ids"] is not None else None
        complete = raw["trajectory_complete"] is True
        flags: list[str] = []
        if raw["event"] is not None:
            reported_id = raw["event"]["reported_route_id"]
            reported = [route for route in history if route["route_id"] == reported_id]
            if len(reported) == 1 and float(reported[0]["generated_at_s"]) > cutoff:
                flags.append("reported_route_is_post_event")
        if suggested is None:
            flags.append("missing_prior_route_exclude_choice_labels")
        if not complete or observed is None:
            flags.append("incomplete_trajectory_exclude_choice_labels")

        traffic = _available_traffic(raw)
        if len(traffic) < len(raw["traffic_history"]):
            flags.append("exclude_late_arriving_traffic")
        if not traffic:
            flags.append("traffic_unknown_at_decision")

        label: bool | None = None
        preferred: tuple[str, ...] | None = None
        rejected: tuple[str, ...] | None = None
        if complete and observed is not None and suggested is not None:
            observed_endpoints = graph.check_path(observed)
            suggested_endpoints = graph.check_path(suggested)
            required_endpoints = (raw["origin_node_id"], raw["destination_node_id"])
            if observed_endpoints != required_endpoints or suggested_endpoints != required_endpoints:
                raise ValueError(f"Different observed and suggested trip endpoints in {scenario['case_id']}")
            label = observed != suggested
            if label and raw["event"] is not None:
                if raw["event"]["primary_reason"] not in EXCLUDED_REASONS:
                    preferred, rejected = observed, suggested
                else:
                    flags.append("exclude_nonrouting_reason_from_preference_pairs")

        choices.append(
            ChoiceExample(
                case_id=scenario["case_id"],
                origin_node_id=raw["origin_node_id"],
                destination_node_id=raw["destination_node_id"],
                decision_time_s=cutoff,
                suggested_edge_ids=suggested,
                observed_edge_ids=observed,
                deviation_label=label,
                preferred_edge_ids=preferred,
                rejected_edge_ids=rejected,
                traffic_history=traffic,
                quality_flags=tuple(flags),
            )
        )
    return choices
