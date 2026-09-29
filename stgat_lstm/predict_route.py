"""Route on the real Taft OSM graph using positive learned edge costs."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from pyproj import Transformer
from shapely.geometry import Point

from .mapbox_traffic import TrafficArchive, collect_live_route_observation
from .model import PreferenceModel, DecisionPreferenceModel, path_cost
from .graph_data import (
    RealGraphData,
    _edge_geometry,
    build_real_graph_data,
    load_approved_decisions,
    path_choice_edge_ids,
)


@dataclass(frozen=True)
class RouteResult:
    edge_ids: tuple[str, ...]
    total_preference_cost: float


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


def load_decision_checkpoint(path: Path, graph_data: RealGraphData) -> DecisionPreferenceModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    if config.get("model_type") != "edge_deviation_preference_multitask":
        raise ValueError("Expected a road-deviation preference checkpoint; retrain an older checkpoint")
    expected = graph_data.model_configuration()
    for field in ("node_feature_dim", "edge_static_dim", "edge_dynamic_dim", "destination_feature_dim"):
        if config["backbone"].get(field) != expected[field]:
            raise ValueError(f"Checkpoint has incompatible {field}; retrain with this graph's features")
    model = DecisionPreferenceModel(**config["backbone"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def snap_coordinate(graph_data: RealGraphData, longitude: float, latitude: float,
                    *, max_distance_m: float = 150.0) -> tuple[str, float]:
    if not math.isfinite(longitude) or not math.isfinite(latitude) or not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError("Invalid longitude or latitude")
    if max_distance_m <= 0:
        raise ValueError("Snap distance must be positive")
    transformer = Transformer.from_crs("EPSG:4326", graph_data.graph.graph["crs"], always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    restricted = graph_data.schema.edge_static.index("access_restricted")
    allowed_nodes = set()
    for row, edge_id in enumerate(graph_data.edge_ids):
        if graph_data.edge_static[row, restricted] == 0:
            u, v, _ = edge_id.split("|", 2)
            allowed_nodes.update((u, v))
    if not allowed_nodes:
        raise ValueError("Graph contains no accessible roads")
    node = min(sorted(allowed_nodes), key=lambda item: (float(graph_data.graph.nodes[item]["x"]) - x) ** 2
               + (float(graph_data.graph.nodes[item]["y"]) - y) ** 2)
    data = graph_data.graph.nodes[node]
    distance = math.hypot(float(data["x"]) - x, float(data["y"]) - y)
    if distance > max_distance_m:
        raise ValueError(f"Point is {distance:.0f} m from the nearest road node; select a point closer to the study roads")
    return node, distance


def route_geometry(graph_data: RealGraphData, edge_ids: tuple[str, ...]) -> dict:
    transformer = Transformer.from_crs(graph_data.graph.graph["crs"], "EPSG:4326", always_xy=True)
    lines, names = [], []
    distance_m = 0.0
    for edge_id in edge_ids:
        u, v, key = edge_id.split("|", 2)
        data = next(data for candidate_key, data in graph_data.graph[u][v].items() if str(candidate_key) == key)
        points = list(_edge_geometry(graph_data.graph, u, v, data).coords)
        start = graph_data.graph.nodes[u]
        if Point(points[-1]).distance(Point(float(start["x"]), float(start["y"]))) < Point(points[0]).distance(Point(float(start["x"]), float(start["y"]))):
            points.reverse()
        lines.append([list(transformer.transform(x, y)) for x, y in points])
        distance_m += float(data.get("length", _edge_geometry(graph_data.graph, u, v, data).length))
        name = str(data.get("name") or "Unnamed road")
        if not names or names[-1] != name:
            names.append(name)
    return {"edge_ids": list(edge_ids), "distance_m": distance_m, "road_names": names,
            "geometry": {"type": "MultiLineString", "coordinates": lines}}


def road_deviation_predictions(
    graph_data: RealGraphData, edge_ids: tuple[str, ...], edge_logits: torch.Tensor
) -> list[dict]:
    """Describe road-level deviation probabilities along one candidate route."""
    if edge_logits.shape != (len(graph_data.edge_ids),):
        raise ValueError("Model must return one deviation logit per OSM edge")
    lookup = {edge_id: index for index, edge_id in enumerate(graph_data.edge_ids)}
    choice_edges = set(path_choice_edge_ids(graph_data, edge_ids))
    predictions = []
    for position, edge_id in enumerate(edge_ids):
        u, v, key = edge_id.split("|", 2)
        data = next(data for candidate_key, data in graph_data.graph[u][v].items()
                    if str(candidate_key) == key)
        is_choice = edge_id in choice_edges
        predictions.append({
            "position": position,
            "edge_id": edge_id,
            "road_name": str(data.get("name") or "Unnamed road"),
            "from_node": u,
            "to_node": v,
            "is_decision_point": is_choice,
            "deviation_probability": (
                float(torch.sigmoid(edge_logits[lookup[edge_id]])) if is_choice else None
            ),
        })
    return predictions


def route_deviation_probability(predictions: list[dict]) -> float | None:
    """Combine conditional road risks into the chance of any route deviation."""
    probabilities = [item["deviation_probability"] for item in predictions
                     if item["deviation_probability"] is not None]
    if not probabilities:
        return None
    survival = math.prod(1.0 - probability for probability in probabilities)
    return 1.0 - survival


def route_request(model: DecisionPreferenceModel, graph_data: RealGraphData,
                  origin_node_id: str, destination_node_id: str,
                  temporal_edge_features: tuple[torch.Tensor, ...] | None = None) -> dict:
    """Route a new OD without GPS labels, surveys or approved-example files."""
    if origin_node_id == destination_node_id:
        raise ValueError("Start and destination snap to the same node; choose different points")
    started = time.perf_counter()
    length_column = graph_data.schema.edge_static.index("length_per_100m")
    distance_costs = {edge_id: max(float(graph_data.edge_static[row, length_column]) * 100, 1e-6)
                      for row, edge_id in enumerate(graph_data.edge_ids)}
    baseline = shortest_real_preference_path(graph_data, distance_costs, origin_node_id, destination_node_id)
    inputs = graph_data.build_model_input(destination_node_id, temporal_edge_features=temporal_edge_features)
    model.eval()
    with torch.no_grad():
        scores, edge_logits = model(inputs)
    costs = {edge_id: float(scores[row]) for row, edge_id in enumerate(graph_data.edge_ids)}
    learned = shortest_real_preference_path(graph_data, costs, origin_node_id, destination_node_id)
    recommended = route_geometry(graph_data, learned.edge_ids)
    recommended["total_learned_preference_cost"] = learned.total_preference_cost
    recommended["road_deviation_predictions"] = road_deviation_predictions(
        graph_data, learned.edge_ids, edge_logits
    )
    recommended["route_deviation_probability"] = route_deviation_probability(
        recommended["road_deviation_predictions"]
    )
    baseline_result = route_geometry(graph_data, baseline.edge_ids)
    baseline_result["road_deviation_predictions"] = road_deviation_predictions(
        graph_data, baseline.edge_ids, edge_logits
    )
    baseline_result["route_deviation_probability"] = route_deviation_probability(
        baseline_result["road_deviation_predictions"]
    )
    return {"origin_node_id": origin_node_id, "destination_node_id": destination_node_id,
            "recommended_route": recommended, "distance_baseline": baseline_result,
            "elapsed_seconds": time.perf_counter() - started,
            "interpretation": ("Prototype preference route. Each probability refers to rejecting that "
                "suggested road when its source is reached; non-branching roads are marked not applicable. "
                "Probabilities are uncalibrated until adequate held-out rider data are collected.")}


def _save_prediction(result: dict, output: Path, *, fresh_route: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Prediction saved to {output}")
    if fresh_route:
        route = result["recommended_route"]
        baseline = result["distance_baseline"]
        print(f"Route: {result['origin_node_id']} -> {result['destination_node_id']} | "
              f"learned: {route['distance_m'] / 1000:.2f} km | "
              f"distance baseline: {baseline['distance_m'] / 1000:.2f} km")
        print(f"Roads: {' -> '.join(route['road_names']) or 'unnamed roads'}")
        risk = route["route_deviation_probability"]
        print(f"Recommended-route deviation probability: "
              f"{risk:.3f}" if risk is not None else "Recommended route has no branching decision")
        for road in route["road_deviation_predictions"]:
            if road["is_decision_point"]:
                print(f"  road {road['position'] + 1}: {road['road_name']} | "
                      f"deviation {road['deviation_probability']:.3f}")
    else:
        print(f"Example: {result['example_key']} | predicted: {result['predicted_label']} "
              f"({result['reviewed_decision_road']['deviation_probability']:.3f} at reviewed road) | "
              f"approved label: {result['approved_label']}")
        for road in result["suggested_road_deviation_predictions"]:
            if road["is_decision_point"]:
                print(f"  suggested road {road['position'] + 1}: {road['road_name']} | "
                      f"deviation {road['deviation_probability']:.3f}")
        print(f"Path costs: suggested {result['suggested_path_preference_cost']:.3f} | "
              f"observed {result['observed_path_preference_cost']:.3f}")
        print(f"Recommended roads: {' -> '.join(result['recommended_route']['road_names']) or 'unnamed roads'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("node_features", type=Path)
    parser.add_argument("approved_decisions", type=Path, nargs="?")
    parser.add_argument("--example-key")
    parser.add_argument("--traffic-observation", type=Path, action="append", default=[])
    parser.add_argument("--traffic-archive", type=Path)
    parser.add_argument("--live-traffic", action="store_true",
                        help="request current Mapbox traffic before routing and save it in the archive")
    parser.add_argument("--token-env", default="MAPBOX_ACCESS_TOKEN")
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument("--history-interval-seconds", type=int, default=300)
    parser.add_argument("--traffic-max-age-seconds", type=int, default=900)
    parser.add_argument("--origin-node")
    parser.add_argument("--destination-node")
    parser.add_argument("--origin", help="longitude,latitude")
    parser.add_argument("--destination", help="longitude,latitude")
    parser.add_argument("--output", type=Path, default=Path("outputs/route_prediction.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)

    graph_data = build_real_graph_data(args.graphml, args.node_features)
    temporal = None
    traffic_report = {"status": "unknown"}
    if args.traffic_archive and args.traffic_observation:
        parser.error("Choose --traffic-archive or --traffic-observation")
    fresh = any((args.origin_node, args.destination_node, args.origin, args.destination))
    if args.live_traffic and not fresh:
        parser.error("--live-traffic is only valid for a fresh route request")
    if args.live_traffic and args.traffic_observation:
        parser.error("--live-traffic cannot be combined with --traffic-observation")
    archive = None
    if fresh:
        if args.approved_decisions or args.example_key:
            parser.error("Fresh routing accepts endpoints instead of an approved example")
        if args.origin and args.destination and not (args.origin_node or args.destination_node):
            origin, origin_distance = snap_coordinate(graph_data, *map(float, args.origin.split(",")))
            destination, destination_distance = snap_coordinate(graph_data, *map(float, args.destination.split(",")))
        elif args.origin_node and args.destination_node and not (args.origin or args.destination):
            origin, destination = args.origin_node, args.destination_node
            origin_distance = destination_distance = 0.0
        else:
            parser.error("Provide both --origin and --destination, or both node IDs")
        if args.live_traffic:
            traffic_directory = args.traffic_archive or Path("outputs/traffic_archive")
            live_path, _ = collect_live_route_observation(
                graph_data, origin, destination, traffic_directory, token_env=args.token_env
            )
            archive = TrafficArchive.from_directory(graph_data, traffic_directory)
            traffic_report["live_observation_file"] = live_path.name
        elif args.traffic_archive:
            archive = TrafficArchive.from_directory(graph_data, args.traffic_archive)
        elif args.traffic_observation:
            archive = TrafficArchive(graph_data, args.traffic_observation)
        if archive is not None:
            temporal, traffic_report = archive.sequence(
                int(time.time() * 1000),
                steps=args.history_steps,
                interval_s=args.history_interval_seconds,
                max_age_s=args.traffic_max_age_seconds,
            )
            traffic_report["status"] = "live_mapbox" if args.live_traffic else "archived_mapbox"
            if args.live_traffic:
                traffic_report["live_observation_file"] = live_path.name
        model = load_decision_checkpoint(args.checkpoint, graph_data)
        result = route_request(model, graph_data, origin, destination, temporal)
        result["traffic"] = traffic_report
        result["snap_distance_m"] = {
            "origin": origin_distance,
            "destination": destination_distance,
        }
        _save_prediction(result, args.output, fresh_route=True)
        return

    if args.approved_decisions is None:
        parser.error("Provide approved_decisions for replay, or endpoints for a fresh route")
    decisions = load_approved_decisions(args.approved_decisions, graph_data)
    if args.example_key:
        matching = [example for example in decisions if example.example_key == args.example_key]
        if not matching:
            raise ValueError(f"Unknown approved example key: {args.example_key}")
        example = matching[0]
    else:
        example = decisions[0]

    if args.traffic_archive:
        archive = TrafficArchive.from_directory(graph_data, args.traffic_archive)
    elif args.traffic_observation:
        archive = TrafficArchive(graph_data, args.traffic_observation)

    traffic_note = "historical traffic unknown"
    if archive is not None:
        cutoff = example.decision_timestamp_ms if args.traffic_archive else int(time.time() * 1000)
        temporal, traffic_report = archive.sequence(
            cutoff,
            steps=args.history_steps,
            interval_s=args.history_interval_seconds,
            max_age_s=args.traffic_max_age_seconds,
        )
        traffic_note = (
            "pre-decision archive" if args.traffic_archive
            else "current-input demonstration; not historical traffic evidence"
        )

    inputs = graph_data.build_model_input(
        example.destination_node_id, temporal_edge_features=temporal
    )
    model = load_decision_checkpoint(args.checkpoint, graph_data)
    with torch.no_grad():
        costs, edge_logits = model(inputs)
        suggested_cost = float(path_cost(costs, example.suggested_edge_ids, inputs.edge_ids))
        observed_cost = float(path_cost(costs, example.observed_edge_ids, inputs.edge_ids))
        edge_costs = {
            edge_id: float(costs[index]) for index, edge_id in enumerate(inputs.edge_ids)
        }
        recommended = shortest_real_preference_path(
            graph_data, edge_costs, example.origin_node_id, example.destination_node_id
        )

    suggested_predictions = road_deviation_predictions(
        graph_data, example.suggested_edge_ids, edge_logits
    )
    route_probability = route_deviation_probability(suggested_predictions)
    reviewed_decision_road = next(
        (item for item in suggested_predictions if item["is_decision_point"]), None
    )
    if reviewed_decision_road is None or route_probability is None:
        raise ValueError("Approved example has no branching road decision")
    probability = reviewed_decision_road["deviation_probability"]

    road_names = []
    for edge_id in recommended.edge_ids:
        u, v, key = edge_id.split("|", 2)
        data = next(
            data for candidate_key, data in graph_data.graph[u][v].items()
            if str(candidate_key) == key
        )
        name = str(data.get("name") or "Unnamed road")
        if not road_names or road_names[-1] != name:
            road_names.append(name)

    recommended_predictions = road_deviation_predictions(
        graph_data, recommended.edge_ids, edge_logits
    )
    result = {
        "example_key": example.example_key,
        "suggested_route_deviation_probability": route_probability,
        "suggested_road_deviation_predictions": suggested_predictions,
        "reviewed_decision_road": reviewed_decision_road,
        "predicted_label": "deviated" if probability >= 0.5 else "followed",
        "approved_label": "deviated" if example.label else "followed",
        "suggested_path_preference_cost": suggested_cost,
        "observed_path_preference_cost": observed_cost,
        "recommended_route": {
            "edge_ids": list(recommended.edge_ids),
            "road_names": road_names,
            "total_learned_preference_cost": recommended.total_preference_cost,
            "road_deviation_predictions": recommended_predictions,
            "route_deviation_probability": route_deviation_probability(recommended_predictions),
        },
        "traffic": traffic_note,
        "traffic_history": traffic_report,
        "interpretation": "in-sample checkpoint reload demonstration, not held-out accuracy",
    }
    _save_prediction(result, args.output, fresh_route=False)


if __name__ == "__main__":
    main()
