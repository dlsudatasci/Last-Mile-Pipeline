"""Build pseudonymous real preference-pair candidates for manual review.

This command writes no coordinates or direct user identifiers. Its output is
explicitly marked review-required and is not consumed by the training module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from pyproj import Transformer
from shapely import from_wkt
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import transform

from .geometry import GeoPoint, parse_gps_location, parse_route_points
from .map_match_audit import EXCLUDED_REASONS, _edge_overlap, _sample_points
from .map_matching import EdgeId, HiddenMarkovMatcher, TimedObservation
from .osm_audit import RoadEdgeIndex, _is_taft_edge
from .preference_pairs import DivergencePair, extract_divergence_pair


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _pseudonym(namespace: str, value: str) -> str:
    return hashlib.sha256(f"stgat-lstm:{namespace}:{value}".encode()).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edge_data(graph: nx.MultiDiGraph, edge_id: EdgeId) -> dict:
    u, v, wanted_key = edge_id
    for key, data in graph[u][v].items():
        if str(key) == wanted_key:
            return data
    raise KeyError(f"Missing graph edge {edge_id}")


def _path_payload(graph: nx.MultiDiGraph, edge_ids: tuple[EdgeId, ...]) -> dict:
    names: list[str] = []
    length_m = 0.0
    edges = []
    for u, v, key in edge_ids:
        data = _edge_data(graph, (u, v, key))
        raw_name = data.get("name")
        name = str(raw_name) if raw_name else "Unnamed road"
        if not names or name != names[-1]:
            names.append(name)
        length_m += float(data["length"])
        edges.append({"u": u, "v": v, "key": key})
    return {"edges": edges, "length_m": round(length_m, 1), "road_names": names}


def _payload_edge_ids(path: dict) -> tuple[EdgeId, ...]:
    return tuple((edge["u"], edge["v"], edge["key"]) for edge in path["edges"])


def _validate_document(document: dict, graph: nx.MultiDiGraph) -> None:
    if document["status"] != "review_required_not_training_ready":
        raise ValueError("Real candidates must remain review-required")
    event_keys = [candidate["event_key"] for candidate in document["candidates"]]
    if len(event_keys) != len(set(event_keys)):
        raise ValueError("Candidate event keys must be unique")
    for candidate in document["candidates"]:
        origin = candidate["common_od"]["origin_node_id"]
        destination = candidate["common_od"]["destination_node_id"]
        preferred = _payload_edge_ids(candidate["preferred_observed_path"])
        rejected = _payload_edge_ids(candidate["rejected_prior_suggestion"])
        if not preferred or not rejected or preferred == rejected:
            raise ValueError("Candidate paths must be nonempty and distinct")
        for role, path in (("preferred", preferred), ("rejected", rejected)):
            if path[0][0] != origin or path[-1][1] != destination:
                raise ValueError(f"{role} path does not have the declared common OD")
            if not all(left[1] == right[0] for left, right in zip(path, path[1:])):
                raise ValueError(f"{role} path is disconnected")
            for edge_id in path:
                _edge_data(graph, edge_id)


def _review_geojson(document: dict, graph: nx.MultiDiGraph) -> dict:
    inverse = Transformer.from_crs(graph.graph["crs"], "EPSG:4326", always_xy=True)
    features = []
    for candidate in document["candidates"]:
        for role, path_key in (
            ("preferred_observed", "preferred_observed_path"),
            ("rejected_suggestion", "rejected_prior_suggestion"),
        ):
            lines = []
            for edge_id in _payload_edge_ids(candidate[path_key]):
                data = _edge_data(graph, edge_id)
                raw_geometry = data.get("geometry")
                if raw_geometry:
                    geometry = from_wkt(raw_geometry) if isinstance(raw_geometry, str) else raw_geometry
                else:
                    left, right = graph.nodes[edge_id[0]], graph.nodes[edge_id[1]]
                    geometry = LineString(
                        [(float(left["x"]), float(left["y"])), (float(right["x"]), float(right["y"]))]
                    )
                if geometry.geom_type == "LineString":
                    lines.append(transform(inverse.transform, geometry))
                else:
                    lines.extend(transform(inverse.transform, geometry).geoms)
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(MultiLineString(lines)),
                    "properties": {
                        "event_key": candidate["event_key"],
                        "role": role,
                        "primary_reason": candidate["survey"]["primary_reason"],
                        "length_m": candidate[path_key]["length_m"],
                        "road_names": " > ".join(candidate[path_key]["road_names"]),
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _pair_payload(
    graph: nx.MultiDiGraph,
    pair: DivergencePair,
    deviation: dict[str, str],
    response: dict[str, str],
    ride_user_id: str,
    gps_overlap_prior: float,
    gps_overlap_new: float,
    mean_gps_distance_m: float,
) -> dict:
    document = {
        "status": "review_required",
        "event_key": _pseudonym("event", deviation["deviationId"]),
        "ride_group": _pseudonym("ride", deviation["rideId"]),
        "rider_group": _pseudonym("rider", ride_user_id),
        "decision_timestamp_ms": int(deviation["timestamp"]),
        "survey": {
            "primary_reason": response["primaryReason"],
            "traffic_severity": response.get("trafficSeverity") or None,
            "deviate_again": response.get("deviateAgain") or None,
        },
        "common_od": {
            "origin_node_id": pair.origin_node_id,
            "destination_node_id": pair.destination_node_id,
        },
        "preferred_observed_path": _path_payload(graph, pair.preferred_edge_ids),
        "rejected_prior_suggestion": _path_payload(graph, pair.rejected_edge_ids),
        "boundary_edges": {
            "shared_before": {"u": pair.shared_edge_before[0], "v": pair.shared_edge_before[1], "key": pair.shared_edge_before[2]},
            "shared_after": {"u": pair.shared_edge_after[0], "v": pair.shared_edge_after[1], "key": pair.shared_edge_after[2]},
        },
        "quality": {
            "gps_edge_overlap_with_prior": round(gps_overlap_prior, 3),
            "gps_edge_overlap_with_regenerated": round(gps_overlap_new, 3),
            "mean_gps_to_matched_edge_m": round(mean_gps_distance_m, 2),
        },
    }
    return document


def build_candidates(data_dir: Path, graphml_path: Path) -> dict:
    graph = nx.read_graphml(graphml_path, force_multigraph=True)
    edge_index = RoadEdgeIndex(graph)
    taft_index = RoadEdgeIndex(graph, _is_taft_edge)
    matcher = HiddenMarkovMatcher(
        graph,
        edge_index,
        search_radius_m=80.0,
        max_candidates=4,
        gps_sigma_m=15.0,
        transition_beta_m=50.0,
        max_speed_mps=45.0,
    )

    rides = _read_csv(data_dir / "rides_clean.csv")
    routes = _read_csv(data_dir / "generated_routes_clean.csv")
    deviations = _read_csv(data_dir / "deviations_clean.csv")
    responses = _read_csv(data_dir / "deviationResponses_clean.csv")
    points = _read_csv(data_dir / "map_points_clean.csv")
    ride_user = {row["id"]: row["userId"] for row in rides}
    route_by_id = {row["routeId"]: row for row in routes}
    response_by_deviation = {row["deviationId"]: row for row in responses}

    routes_by_ride: dict[str, list[dict[str, str]]] = defaultdict(list)
    for route in routes:
        routes_by_ride[route["rideId"]].append(route)
    events_by_ride: dict[str, list[int]] = defaultdict(list)
    for deviation in deviations:
        events_by_ride[deviation["rideId"]].append(int(deviation["timestamp"]))
    for times in events_by_ride.values():
        times.sort()
    gps_by_ride: dict[str, list[TimedObservation]] = defaultdict(list)
    for row in points:
        gps_by_ride[row["rideId"]].append(
            TimedObservation(
                GeoPoint(float(row["latitude"]), float(row["longitude"])),
                int(row["timestamp"]),
            )
        )
    for observations in gps_by_ride.values():
        observations.sort(key=lambda observation: observation.timestamp_ms or 0)

    failures: Counter[str] = Counter()
    candidates = []
    eligible_events = 0
    for deviation in deviations:
        response = response_by_deviation.get(deviation["deviationId"])
        if response is None:
            failures["missing_response"] += 1
            continue
        if response["primaryReason"] in EXCLUDED_REASONS:
            failures["excluded_reason"] += 1
            continue
        if taft_index.nearest(parse_gps_location(deviation["gpsLocation"])).distance_m > 500.0:
            failures["outside_taft_corridor"] += 1
            continue
        eligible_events += 1
        event_time = int(deviation["timestamp"])
        ride_id = deviation["rideId"]
        later_events = [time for time in events_by_ride[ride_id] if time > event_time]
        interval_end = min(event_time + 180_000, min(later_events) - 1 if later_events else event_time + 180_000)
        observations = [
            observation
            for observation in gps_by_ride[ride_id]
            if event_time - 60_000 <= (observation.timestamp_ms or 0) <= interval_end
        ]
        if len(observations) < 3:
            failures["insufficient_gps_window"] += 1
            continue
        event_state_index = next(
            (index for index, observation in enumerate(observations) if (observation.timestamp_ms or 0) >= event_time),
            -1,
        )
        if event_state_index <= 0:
            failures["no_pre_event_gps_anchor"] += 1
            continue

        prior_routes = [route for route in routes_by_ride[ride_id] if int(route["generatedAt"]) < event_time]
        referenced_route = route_by_id.get(deviation["routeId"])
        if not prior_routes or referenced_route is None:
            failures["missing_route_version"] += 1
            continue
        prior_route = max(prior_routes, key=lambda route: int(route["generatedAt"]))
        try:
            gps_match = matcher.match(observations)
            prior_match = matcher.match(_sample_points(parse_route_points(prior_route["routePoints"]), limit=50))
            new_match = matcher.match(_sample_points(parse_route_points(referenced_route["routePoints"]), limit=50))
        except ValueError as error:
            message = str(error)
            category = "no_feasible_directed_transition" if message.startswith("No feasible directed transition") else "map_match_failure"
            failures[category] += 1
            continue

        old_overlap = _edge_overlap(gps_match, prior_match)
        new_overlap = _edge_overlap(gps_match, new_match)
        if new_overlap <= old_overlap:
            failures["gps_does_not_favor_regenerated_route"] += 1
            continue
        pair = extract_divergence_pair(gps_match, prior_match, event_state_index)
        if pair is None:
            failures["no_shared_divergence_rejoin_boundaries"] += 1
            continue
        candidates.append(
            _pair_payload(
                graph,
                pair,
                deviation,
                response,
                ride_user[ride_id],
                old_overlap,
                new_overlap,
                gps_match.mean_observation_distance_m,
            )
        )

    candidate_riders = {candidate["rider_group"] for candidate in candidates}
    candidate_rides = {candidate["ride_group"] for candidate in candidates}
    document = {
        "schema_version": 1,
        "provenance": "Real rider export plus supplied OSM graph; no synthetic observations mixed in",
        "status": "review_required_not_training_ready",
        "graph": {
            "source": str(graphml_path),
            "sha256": _sha256(graphml_path),
            "created_date": graph.graph.get("created_date"),
            "crs": graph.graph.get("crs"),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "source_sha256": {
            name: _sha256(data_dir / name)
            for name in (
                "rides_clean.csv",
                "generated_routes_clean.csv",
                "deviations_clean.csv",
                "deviationResponses_clean.csv",
                "map_points_clean.csv",
            )
        },
        "rules": {
            "taft_corridor_m": 500,
            "gps_before_event_s": 60,
            "gps_after_event_s": 180,
            "stop_at_next_deviation": True,
            "require_intentional_reason": True,
            "require_gps_to_favor_regenerated_route": True,
            "require_shared_directed_edge_before_and_after": True,
        },
        "summary": {
            "reported_deviations": len(deviations),
            "eligible_intentional_taft_events": eligible_events,
            "review_candidates": len(candidates),
            "candidate_rides": len(candidate_rides),
            "candidate_riders": len(candidate_riders),
            "failure_counts": dict(sorted(failures.items())),
        },
        "candidates": candidates,
    }
    _validate_document(document, graph)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geojson-output", type=Path)
    args = parser.parse_args()
    result = build_candidates(args.data_dir, args.graphml)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    if args.geojson_output is not None:
        graph = nx.read_graphml(args.graphml, force_multigraph=True)
        geojson = _review_geojson(result, graph)
        args.geojson_output.parent.mkdir(parents=True, exist_ok=True)
        geojson_temporary = args.geojson_output.with_suffix(args.geojson_output.suffix + ".tmp")
        geojson_temporary.write_text(json.dumps(geojson), encoding="utf-8")
        geojson_temporary.replace(args.geojson_output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "geojson_output": str(args.geojson_output) if args.geojson_output else None,
                **result["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
