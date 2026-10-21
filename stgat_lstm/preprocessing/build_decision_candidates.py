"""Build conservative real follow-versus-deviate decision candidates.

Each example represents the decision presented by an initial suggested route.
Only information available at that time may become a model input. The later GPS
trajectory and deviation survey are retained as label evidence, not features.
The output is validation-required and cannot be used directly for training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import networkx as nx
from shapely import from_wkt
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import transform, unary_union

from stgat_lstm.preprocessing.build_deviation_candidates import _path_payload, _sha256
from stgat_lstm.preprocessing.geometry import GeoPoint, parse_route_points
from stgat_lstm.preprocessing.audit_map_matching import EXCLUDED_REASONS, _sample_points
from stgat_lstm.preprocessing.map_matching import HiddenMarkovMatcher, MapMatchResult, TimedObservation
from stgat_lstm.preprocessing.audit_road_network import RoadEdgeIndex
from stgat_lstm.preprocessing.coordinates import Transformer


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _pseudonym(namespace: str, value: str) -> str:
    return hashlib.sha256(f"stgat-lstm:{namespace}:{value}".encode()).hexdigest()[:16]


def _sample_gps(observations: list[TimedObservation], limit: int = 60) -> list[TimedObservation]:
    unique: list[TimedObservation] = []
    seen_timestamps: set[int] = set()
    for observation in observations:
        timestamp = observation.timestamp_ms
        if timestamp is None or timestamp in seen_timestamps:
            continue
        seen_timestamps.add(timestamp)
        unique.append(observation)
    if len(unique) <= limit:
        return unique
    indices = sorted({round(index * (len(unique) - 1) / (limit - 1)) for index in range(limit)})
    return [unique[index] for index in indices]


def _agreement(observed: MapMatchResult, suggested: MapMatchResult) -> dict[str, float]:
    observed_edges = set(observed.traversed_edge_ids)
    suggested_edges = set(suggested.traversed_edge_ids)
    intersection = observed_edges & suggested_edges
    union = observed_edges | suggested_edges
    return {
        "observed_edge_agreement": len(intersection) / len(observed_edges) if observed_edges else 0.0,
        "suggested_edge_coverage": len(intersection) / len(suggested_edges) if suggested_edges else 0.0,
        "edge_jaccard": len(intersection) / len(union) if union else 0.0,
    }


def _follow_choice_window(
    graph: nx.MultiDiGraph,
    observed: MapMatchResult,
    suggested: MapMatchResult,
    gps: list[TimedObservation],
    *,
    window_edges: int = 5,
) -> tuple[tuple[tuple[str, str, str], ...], int, str, str] | None:
    """Find one actually followed route window beginning at a branching node."""
    observed_edges = observed.traversed_edge_ids
    suggested_edges = suggested.traversed_edge_ids
    for suggested_index in range(0, max(len(suggested_edges) - 2, 0)):
        origin = suggested_edges[suggested_index][0]
        successors = {str(v) for _, v, _, data in graph.out_edges(origin, keys=True, data=True) if data.get("access") != "no"}
        if len(successors) < 2:
            continue
        segment = suggested_edges[suggested_index : suggested_index + window_edges]
        if len(segment) < 3:
            continue
        for observed_index in range(len(observed_edges) - len(segment) + 1):
            if observed_edges[observed_index : observed_index + len(segment)] != segment:
                continue
            state_index = next(
                (
                    index
                    for index, offset in enumerate(observed.state_edge_offsets)
                    if offset >= observed_index
                ),
                0,
            )
            timestamp = gps[min(state_index, len(gps) - 1)].timestamp_ms
            if timestamp is None:
                continue
            return segment, timestamp, segment[0][0], segment[-1][1]
    return None


def _load_deviation_examples(path: Path | None) -> list[dict]:
    if path is None:
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for candidate in document.get("candidates", []):
        examples.append(
            {
                "status": "review_required",
                "example_key": candidate["event_key"],
                "ride_group": candidate["ride_group"],
                "rider_group": candidate["rider_group"],
                "decision_timestamp_ms": candidate["decision_timestamp_ms"],
                "target": {
                    "label": "deviated",
                    "value": 1,
                    "basis": ("intentional surveyed outgoing-road choice inferred from recorded navigation and GPS geometry"
                              if candidate.get("road_choice", {}).get("label_method") == "recorded_navigation_and_local_geometry" else
                              "intentional surveyed outgoing-road choice with direct GPS evidence"
                              if candidate.get("road_choice") else
                              "intentional surveyed deviation with a GPS-supported divergence and rejoin"),
                },
                **({"common_od": candidate["common_od"]} if "common_od" in candidate else {}),
                **({"routing_context": candidate["routing_context"]}
                   if "routing_context" in candidate else {}),
                **({"road_choice": candidate["road_choice"]} if "road_choice" in candidate else {}),
                **({"recorded_navigation": candidate["recorded_navigation"]}
                   if "recorded_navigation" in candidate else {}),
                **({"reported_event_timestamp_ms": candidate["reported_event_timestamp_ms"]}
                   if "reported_event_timestamp_ms" in candidate else {}),
                "has_route_pair": candidate.get("has_route_pair", True),
                "suggested_path": candidate["rejected_prior_suggestion"],
                "observed_path_label_evidence": candidate["preferred_observed_path"],
                "survey_evidence": candidate["survey"],
                "quality": candidate["quality"],
                "feature_boundary": {
                    "allowed_at_inference": [
                        "suggested_path",
                        "decision_timestamp_ms",
                        "rider_group_or_generalized_rider_history",
                        "static_osm_features",
                        "traffic_available_by_decision_timestamp",
                    ],
                    "label_only_never_input": ["observed_path_label_evidence", "survey_evidence",
                                               "recorded_navigation", "road_choice.inference_evidence"],
                },
            }
        )
    return examples


def _payload_edges(path: dict) -> tuple[tuple[str, str, str], ...]:
    return tuple((str(edge["u"]), str(edge["v"]), str(edge["key"])) for edge in path["edges"])


def _validate_document(document: dict) -> None:
    if document["status"] != "review_required_not_training_ready":
        raise ValueError("Decision candidates must remain review-required")
    keys = [example["example_key"] for example in document["examples"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Decision example keys must be unique")
    for example in document["examples"]:
        label = example["target"]["label"]
        if (label, example["target"]["value"]) not in {("followed", 0), ("deviated", 1)}:
            raise ValueError("Decision target label and value disagree")
        context = example.get("routing_context", example.get("common_od"))
        origin = context["origin_node_id"]
        destination = context["destination_node_id"]
        for field in ("suggested_path", "observed_path_label_evidence"):
            edges = _payload_edges(example[field])
            pair_context = example.get("common_od", context)
            if not edges or (example.get("has_route_pair", True) and (
                edges[0][0] != pair_context["origin_node_id"]
                or edges[-1][1] != pair_context["destination_node_id"]
            )):
                raise ValueError(f"{field} does not have the declared common OD")
            if not all(left[1] == right[0] for left, right in zip(edges, edges[1:])):
                raise ValueError(f"{field} is disconnected")
        allowed = set(example["feature_boundary"]["allowed_at_inference"])
        forbidden = set(example["feature_boundary"]["label_only_never_input"])
        if allowed & forbidden:
            raise ValueError("Label evidence leaked into allowed inference fields")


def _decision_geojson(document: dict, graph: nx.MultiDiGraph) -> dict:
    inverse = Transformer.from_crs(graph.graph["crs"], "EPSG:4326", always_xy=True)
    features = []
    selected_projected_lines = []
    for example in document["examples"]:
        for role, field in (
            ("preferred_observed", "observed_path_label_evidence"),
            ("rejected_suggestion", "suggested_path"),
        ):
            lines = []
            for u, v, key in _payload_edges(example[field]):
                data = next(data for candidate_key, data in graph[u][v].items() if str(candidate_key) == key)
                raw_geometry = data.get("geometry")
                geometry = (
                    from_wkt(raw_geometry)
                    if isinstance(raw_geometry, str)
                    else raw_geometry
                    if raw_geometry is not None
                    else LineString(
                        [
                            (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
                            (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
                        ]
                    )
                )
                projected = transform(inverse.transform, geometry)
                selected_projected_lines.append(geometry)
                lines.extend(projected.geoms if projected.geom_type == "MultiLineString" else [projected])
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(MultiLineString(lines)),
                    "properties": {
                        "event_key": example["example_key"],
                        "role": role,
                        "primary_reason": f'{example["target"]["label"]}: {example["target"]["basis"]}',
                        "length_m": example[field]["length_m"],
                        "road_names": " > ".join(example[field]["road_names"]),
                    },
                }
            )
    review_area = unary_union(selected_projected_lines).buffer(250.0)
    for u, v, key, data in graph.edges(keys=True, data=True):
        raw_geometry = data.get("geometry")
        geometry = (
            from_wkt(raw_geometry)
            if isinstance(raw_geometry, str)
            else raw_geometry
            if raw_geometry is not None
            else LineString(
                [
                    (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
                    (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
                ]
            )
        )
        if geometry.is_empty or not geometry.intersects(review_area):
            continue
        projected = transform(inverse.transform, geometry)
        lines = list(projected.geoms) if projected.geom_type == "MultiLineString" else [projected]
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(MultiLineString(lines)),
                "properties": {
                    "event_key": None,
                    "role": "context_road",
                    "road_names": str(data.get("name") or "Unnamed road"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_decision_candidates(
    data_dir: Path,
    graphml_path: Path,
    deviation_candidates_path: Path | None = None,
) -> dict:
    graph = nx.read_graphml(graphml_path, force_multigraph=True)
    edge_index = RoadEdgeIndex(graph)
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

    initial_routes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for route in routes:
        if route["type"] == "Initial Route":
            initial_routes[route["rideId"]].append(route)
    deviations_by_ride: dict[str, list[dict[str, str]]] = defaultdict(list)
    for deviation in deviations:
        deviations_by_ride[deviation["rideId"]].append(deviation)
    response_by_deviation = {response["deviationId"]: response for response in responses}
    gps_by_ride: dict[str, list[TimedObservation]] = defaultdict(list)
    for point in points:
        gps_by_ride[point["rideId"]].append(
            TimedObservation(
                GeoPoint(float(point["latitude"]), float(point["longitude"])),
                int(point["timestamp"]),
            )
        )
    for observations in gps_by_ride.values():
        observations.sort(key=lambda item: item.timestamp_ms or 0)

    failures: Counter[str] = Counter()
    examples = []
    for ride in rides:
        ride_id = ride["id"]
        ride_routes = initial_routes.get(ride_id, [])
        if not ride_routes:
            failures["missing_initial_route"] += 1
            continue
        initial_route = min(ride_routes, key=lambda item: int(item["generatedAt"]))
        ride_deviations = deviations_by_ride.get(ride_id, [])
        if ride_deviations:
            failures["deviation_ride_handled_by_strict_event_pipeline"] += 1
            continue
        gps = _sample_gps(gps_by_ride.get(ride_id, []))
        if len(gps) < 4:
            failures["insufficient_unique_gps"] += 1
            continue

        try:
            observed_match = matcher.match(gps)
            suggested_match = matcher.match(_sample_points(parse_route_points(initial_route["routePoints"]), limit=50))
        except ValueError as error:
            category = (
                "no_feasible_directed_transition"
                if str(error).startswith("No feasible directed transition")
                else "map_match_failure"
            )
            failures[category] += 1
            continue

        agreement = _agreement(observed_match, suggested_match)
        if agreement["observed_edge_agreement"] < 0.8 or agreement["edge_jaccard"] < 0.7:
            failures["unreported_route_disagreement"] += 1
            continue
        choice = _follow_choice_window(graph, observed_match, suggested_match, gps)
        if choice is None:
            failures["no_verified_branching_follow_window"] += 1
            continue
        choice_edges, decision_timestamp, origin_node, destination_node = choice

        examples.append(
            {
                "status": "review_required",
                "example_key": _pseudonym("decision", ride_id),
                "ride_group": _pseudonym("ride", ride_id),
                "rider_group": _pseudonym("rider", ride["userId"]),
                "decision_timestamp_ms": decision_timestamp,
                "target": {
                    "label": "followed",
                    "value": 0,
                    "basis": "no logged deviation and GPS follows the suggestion through a branching node",
                },
                "common_od": {"origin_node_id": origin_node, "destination_node_id": destination_node},
                "routing_context": {"origin_node_id": origin_node,
                                    "destination_node_id": suggested_match.traversed_edge_ids[-1][1]},
                "suggested_path": _path_payload(graph, choice_edges),
                "observed_path_label_evidence": _path_payload(graph, choice_edges),
                "survey_evidence": {
                    "reported_deviation_count": 0,
                    "intentional_deviation_count": 0,
                    "missing_response_count": 0,
                    "primary_reasons": [],
                },
                "quality": {
                    **{key: round(value, 3) for key, value in agreement.items()},
                    "gps_mean_match_distance_m": round(observed_match.mean_observation_distance_m, 2),
                    "gps_max_match_distance_m": round(observed_match.max_observation_distance_m, 2),
                    "sampled_gps_points": len(gps),
                },
                "feature_boundary": {
                    "allowed_at_inference": [
                        "suggested_path",
                        "decision_timestamp_ms",
                        "rider_group_or_generalized_rider_history",
                        "static_osm_features",
                        "traffic_available_by_decision_timestamp",
                    ],
                    "label_only_never_input": ["observed_path_label_evidence", "survey_evidence"],
                },
            }
        )

    deviation_document = (
        json.loads(deviation_candidates_path.read_text(encoding="utf-8"))
        if deviation_candidates_path else {}
    )
    examples.extend(_load_deviation_examples(deviation_candidates_path))
    label_counts = Counter(example["target"]["label"] for example in examples)
    manifest_path = data_dir / "manifest.json"
    synthetic = bool(deviation_document.get("synthetic"))
    if manifest_path.exists():
        synthetic = synthetic or bool(
            json.loads(manifest_path.read_text(encoding="utf-8")).get("synthetic")
        )
    document = {
        "schema_version": 1,
        "provenance": (
            "Controlled synthetic export plus supplied OSM graph"
            if synthetic else "Official rider export plus supplied OSM graph"
        ),
        "synthetic": synthetic,
        "status": "review_required_not_training_ready",
        "prediction_target": "preferred accessible outgoing road at an eligible local branch",
        "decision_point": "before traversing a suggested road from a node with multiple accessible successors",
        "graph": {
            "source": str(graphml_path),
            "sha256": _sha256(graphml_path),
            "crs": graph.graph.get("crs"),
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
        "deviation_candidates_sha256": _sha256(deviation_candidates_path) if deviation_candidates_path else None,
        "rules": {
            "study_corridor_limit_m": None,
            "minimum_follow_observed_agreement": 0.8,
            "minimum_follow_edge_jaccard": 0.7,
            "excluded_deviation_reasons": sorted(EXCLUDED_REASONS),
            "observed_gps_and_surveys_are_label_evidence_only": True,
        },
        "summary": {
            "source_rides": len(rides),
            "candidate_examples": len(examples),
            "candidate_riders": len({example["rider_group"] for example in examples}),
            "label_counts": dict(sorted(label_counts.items())),
            "failure_counts": dict(sorted(failures.items())),
        },
        "examples": examples,
    }
    _validate_document(document)
    return document


def refine_real_decisions(data_dir: Path, graphml_path: Path, original: dict,
                          assignments: dict[str, str], progress=None) -> dict:
    """Correct goal context and recover additional real training-rider choices."""
    from stgat_lstm.preprocessing.review_decisions import apply_automatic_rules
    if original.get("synthetic") or original.get("status") != "approved_for_training":
        raise ValueError("Refinement requires an approved real artifact")
    for name, digest in original.get("source_sha256", {}).items():
        if _sha256(data_dir / name) != digest:
            raise ValueError(f"Source CSV changed: {name}; rebuild the dataset first")
    if original.get("graph", {}).get("sha256") != _sha256(graphml_path):
        raise ValueError("Refinement graph differs from the approved graph")
    result = deepcopy(original)
    result["status"] = "review_required_not_training_ready"
    result.pop("review", None)
    graph = nx.read_graphml(graphml_path, force_multigraph=True)
    matcher = HiddenMarkovMatcher(graph, RoadEdgeIndex(graph), search_radius_m=80,
                                 max_candidates=4, gps_sigma_m=15, transition_beta_m=50,
                                 max_speed_mps=45)
    rides = {_pseudonym("ride", r["id"]): r for r in _read_csv(data_dir / "rides_clean.csv")}
    routes = defaultdict(list)
    for route in _read_csv(data_dir / "generated_routes_clean.csv"):
        routes[route["rideId"]].append(route)
    points, events = defaultdict(list), defaultdict(list)
    for point in _read_csv(data_dir / "map_points_clean.csv"):
        points[point["rideId"]].append(TimedObservation(
            GeoPoint(float(point["latitude"]), float(point["longitude"])), int(point["timestamp"])))
    for event in _read_csv(data_dir / "deviations_clean.csv"):
        events[event["rideId"]].append(int(event["timestamp"]))
    matched_routes = {}
    def match_route(route):
        key = (route["rideId"], route["generatedAt"])
        if key not in matched_routes:
            matched_routes[key] = matcher.match(_sample_points(parse_route_points(route["routePoints"]), limit=50))
        return matched_routes[key]
    used = set()
    corrected = 0
    for example in result["examples"]:
        if example["target"]["value"] == 0:
            ride = rides[example["ride_group"]]
            initial = min((r for r in routes[ride["id"]] if r["type"] == "Initial Route"),
                          key=lambda r: int(r["generatedAt"]))
            goal = match_route(initial).traversed_edge_ids[-1][1]
            example["routing_context"] = {"origin_node_id": example["common_od"]["origin_node_id"],
                                          "destination_node_id": goal}
            corrected += int(goal != example["common_od"]["destination_node_id"])
            for edge in example["suggested_path"]["edges"]:
                used.add((example["ride_group"], str(edge["u"])))
        else:
            used.add((example["ride_group"], str(example["road_choice"]["source_node_id"])))
    added, failures = 0, Counter()
    for ride_key, ride in sorted(rides.items()):
        rider = _pseudonym("rider", ride["userId"])
        if assignments.get(rider) != "training":
            continue
        if progress:
            progress(f"Recovering real choices | training ride {ride_key}")
        periods = sorted(routes[ride["id"]], key=lambda r: int(r["generatedAt"]))
        for period_index, route in enumerate(periods):
            begin = int(route["generatedAt"])
            end = int(periods[period_index + 1]["generatedAt"]) if period_index + 1 < len(periods) else float("inf")
            gps = _sample_gps(sorted((p for p in points[ride["id"]] if begin <= p.timestamp_ms < end),
                                     key=lambda p: p.timestamp_ms), limit=120)
            if len(gps) < 4:
                continue
            try:
                observed, suggested = matcher.match(gps), match_route(route)
            except ValueError:
                failures["period_sequence_match_failed"] += 1
                continue
            for i, edge in enumerate(suggested.traversed_edge_ids[:-2]):
                if (ride_key, edge[0]) in used:
                    continue
                accessible = {str(v) for _, v, _, d in graph.out_edges(edge[0], keys=True, data=True)
                              if str(d.get("access", "")).lower() not in {"no", "private"}}
                if edge[1] not in accessible or len(accessible) < 2:
                    continue
                continuation = suggested.traversed_edge_ids[i:i + 3]
                for j in range(len(observed.traversed_edge_ids) - 2):
                    if observed.traversed_edge_ids[j:j + 3] != continuation:
                        continue
                    support = [k for k, offset in enumerate(observed.state_edge_offsets)
                               if offset == j and observed.states[k].edge_id == edge]
                    if len(support) < 2 or support[0] == 0:
                        continue
                    if not any(b == a + 1 for a, b in zip(support, support[1:])):
                        continue
                    anchor = support[0] - 1
                    last = max(k for k, offset in enumerate(observed.state_edge_offsets) if offset <= j + 2)
                    stamp, finish = gps[anchor].timestamp_ms, gps[last].timestamp_ms
                    if any(stamp <= event <= finish for event in events[ride["id"]]):
                        continue
                    distances = [observed.states[k].observation_distance_m for k in support]
                    if sum(distances) / len(distances) > 20:
                        continue
                    path = _path_payload(graph, (edge,))
                    result["examples"].append({
                        "status": "review_required", "example_key": _pseudonym("expanded-choice", f"{ride['id']}:{edge[0]}:{stamp}"),
                        "ride_group": ride_key, "rider_group": rider, "decision_timestamp_ms": stamp,
                        "target": {"label": "followed", "value": 0,
                                   "basis": "GPS follows the active suggestion at a verified additional branch"},
                        "common_od": {"origin_node_id": edge[0], "destination_node_id": edge[1]},
                        "routing_context": {"origin_node_id": edge[0],
                                            "destination_node_id": suggested.traversed_edge_ids[-1][1]},
                        "suggested_path": path, "observed_path_label_evidence": deepcopy(path),
                        "survey_evidence": {"scope": "verified_follow_window", "reported_deviation_count": 0,
                                            "source_ride_reported_deviation_count": len(events[ride["id"]])},
                        "quality": {"observed_edge_agreement": 1., "gps_mean_match_distance_m": sum(distances) / len(distances),
                                    "supporting_gps_points": len(support), "three_edge_continuation_verified": True,
                                    "active_route_generated_at_ms": begin, "label_window_end_ms": finish,
                                    "additional_training_choice": True},
                        "feature_boundary": {"allowed_at_inference": ["static_osm_features", "suggested_path",
                                               "decision_timestamp_ms", "traffic_available_by_decision_timestamp"],
                                             "label_only_never_input": ["observed_path_label_evidence", "survey_evidence"]}})
                    used.add((ride_key, edge[0])); added += 1
                    break
    result["refinement"] = {"changed_followed_destinations": corrected, "additional_real_choices": added,
                            "additional_choices_partition": "training riders only", "failure_counts": dict(failures),
                            "destination_policy": "end of the active suggested route; local review endpoints remain label geometry"}
    result["summary"].update({"candidate_examples": len(result["examples"]),
        "candidate_riders": len({e["rider_group"] for e in result["examples"]}),
        "label_counts": dict(Counter(e["target"]["label"] for e in result["examples"]))})
    _validate_document(result)
    return apply_automatic_rules(json.dumps(result).encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("--deviation-candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geojson-output", type=Path)
    args = parser.parse_args()
    result = build_decision_candidates(args.data_dir, args.graphml, args.deviation_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    if args.geojson_output is not None:
        graph = nx.read_graphml(args.graphml, force_multigraph=True)
        geojson = _decision_geojson(result, graph)
        args.geojson_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_geojson = args.geojson_output.with_suffix(args.geojson_output.suffix + ".tmp")
        temporary_geojson.write_text(json.dumps(geojson), encoding="utf-8")
        temporary_geojson.replace(args.geojson_output)
    summary = result["summary"]
    labels = summary["label_counts"]
    print(f"Decision candidates saved to {args.output}")
    if args.geojson_output:
        print(f"Review geometry saved to {args.geojson_output}")
    print(f"Candidates: {summary['candidate_examples']} from {summary['candidate_riders']} riders | "
          f"followed: {labels.get('followed', 0)} | deviated: {labels.get('deviated', 0)}")
    print("Status: automatic validation required before training")


if __name__ == "__main__":
    main()
