"""Build pseudonymous real preference-pair candidates for validation.

This command writes no coordinates or direct user identifiers. Its output is
explicitly marked validation-required and is not consumed by the training module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from shapely import from_wkt
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import transform

from stgat_lstm.preprocessing.coordinates import Transformer

from stgat_lstm.preprocessing.geometry import GeoPoint, parse_gps_location, parse_route_points
from stgat_lstm.preprocessing.audit_map_matching import EXCLUDED_REASONS, _edge_overlap, _sample_points
from stgat_lstm.preprocessing.map_matching import EdgeId, HiddenMarkovMatcher, TimedObservation
from stgat_lstm.preprocessing.audit_road_network import RoadEdgeIndex
from stgat_lstm.preprocessing.preference_pairs import (DivergencePair, extract_divergence_pair, extract_outgoing_choice,
                               infer_recorded_road_choice)
from stgat_lstm.preprocessing.graph_data import _restricted_access


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
        context = candidate.get("routing_context", candidate.get("common_od"))
        origin = context["origin_node_id"]
        destination = context["destination_node_id"]
        preferred = _payload_edge_ids(candidate["preferred_observed_path"])
        rejected = _payload_edge_ids(candidate["rejected_prior_suggestion"])
        if not preferred or not rejected or preferred == rejected:
            raise ValueError("Candidate paths must be nonempty and distinct")
        for role, path in (("preferred", preferred), ("rejected", rejected)):
            pair_context = candidate.get("common_od", context)
            if candidate.get("has_route_pair", True) and (
                path[0][0] != pair_context["origin_node_id"]
                or path[-1][1] != pair_context["destination_node_id"]
            ):
                raise ValueError(f"{role} path does not have the declared common OD")
            if not all(left[1] == right[0] for left, right in zip(path, path[1:])):
                raise ValueError(f"{role} path is disconnected")
            for edge_id in path:
                _edge_data(graph, edge_id)
        if not candidate.get("has_route_pair", True):
            if preferred[0][0] != origin or rejected[0][0] != origin:
                raise ValueError("Local choices must leave the declared source")
            if destination not in graph:
                raise ValueError("Routing destination is absent from the graph")


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


def build_candidates(
    data_dir: Path, graphml_path: Path, *,
    max_match_candidates: int = 6, backward_jitter_m: float = 8.0,
) -> dict:
    graph = nx.read_graphml(graphml_path, force_multigraph=True)
    edge_index = RoadEdgeIndex(graph)
    matcher = HiddenMarkovMatcher(
        graph,
        edge_index,
        search_radius_m=80.0,
        max_candidates=max_match_candidates,
        gps_sigma_m=15.0,
        transition_beta_m=50.0,
        max_speed_mps=45.0,
        backward_jitter_tolerance_m=backward_jitter_m,
    )
    strict_matcher = HiddenMarkovMatcher(graph, edge_index, search_radius_m=80.0, max_candidates=4)

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
    event_diagnostics = []
    eligible_events = 0
    for deviation in deviations:
        response = response_by_deviation.get(deviation["deviationId"])
        if response is None:
            failures["missing_response"] += 1
            continue
        if response["primaryReason"] in EXCLUDED_REASONS:
            failures["excluded_reason"] += 1
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
        # A rider may report the turn after the final recorded fix. Local
        # evidence, rather than GPS on both sides of the report, decides usability.
        if event_state_index < 0:
            event_state_index = len(observations) - 1

        prior_routes = [route for route in routes_by_ride[ride_id] if int(route["generatedAt"]) < event_time]
        referenced_route = route_by_id.get(deviation["routeId"])
        if not prior_routes or referenced_route is None:
            failures["missing_route_version"] += 1
            continue
        prior_route = max(prior_routes, key=lambda route: int(route["generatedAt"]))
        sequences = {
            "gps": observations,
            "prior_suggestion": _sample_points(parse_route_points(prior_route["routePoints"]), limit=50),
            "regenerated_route": _sample_points(parse_route_points(referenced_route["routePoints"]), limit=50),
        }
        diagnostic = {"event_key": _pseudonym("event", deviation["deviationId"]),
                      "strict_first_failure": None, "matching_failures": {}, "retained": False,
                      "gps_before_report": sum((point.timestamp_ms or 0) < event_time for point in observations),
                      "gps_at_or_after_report": sum((point.timestamp_ms or 0) >= event_time for point in observations)}
        for stage, sequence in sequences.items():
            try:
                strict_matcher.match(sequence)
            except ValueError as error:
                diagnostic["strict_first_failure"] = {
                    "stage": stage, "reason": str(error),
                }
                break
        matched = {}
        for stage, sequence in sequences.items():
            try:
                matched[stage] = matcher.match(sequence)
            except ValueError as error:
                diagnostic["matching_failures"][stage] = str(error)
        event_diagnostics.append(diagnostic)
        if "gps" not in matched or "prior_suggestion" not in matched:
            failures["gps_or_prior_suggestion_match_failed"] += 1
            continue
        gps_match, prior_match = matched["gps"], matched["prior_suggestion"]
        diagnostic["local_choice_evidence"] = {}
        choice = extract_outgoing_choice(gps_match, prior_match, observations, event_time,
                                         diagnostics=diagnostic["local_choice_evidence"])
        if choice is None:
            previous_events = [time for time in events_by_ride[ride_id] if time < event_time]
            active_since = max(int(prior_route["generatedAt"]),
                               max(previous_events) + 1 if previous_events else 0)
            extended = [point for point in gps_by_ride[ride_id]
                        if active_since <= (point.timestamp_ms or 0) <= interval_end]
            if extended and extended[0] != observations[0]:
                recovery = {"kind": "look back to active-route start for a late report",
                            "active_since_ms": active_since}
                diagnostic["extended_window_recovery"] = recovery
                try:
                    extended_match = matcher.match(extended)
                except ValueError as error:
                    recovery["matching_failure"] = str(error)
                else:
                    recovery["local_choice_evidence"] = {}
                    recovered = extract_outgoing_choice(
                        extended_match, prior_match, extended, event_time,
                        diagnostics=recovery["local_choice_evidence"],
                        earliest_choice_timestamp_ms=active_since,
                    )
                    if recovered is not None:
                        choice, gps_match, observations = recovered, extended_match, extended
                        event_state_index = next(
                            (index for index, point in enumerate(observations)
                             if (point.timestamp_ms or 0) >= event_time), len(observations) - 1)
                        diagnostic["recovered_with_active_route_window"] = True
        if choice is None:
            diagnostic["navigation_inference"] = {}
            original_instruction = deviation.get("originalRouteEdge") or deviation.get("generatedInstruction", "")
            chosen_instruction = deviation.get("deviatedEdge") or deviation.get("deviationInstruction", "")
            try:
                event_location = parse_gps_location(deviation.get("gpsLocation", ""))
            except ValueError:
                diagnostic["navigation_inference"]["status"] = "missing_or_invalid_event_location"
            else:
                choice = infer_recorded_road_choice(
                    graph, edge_index, prior_match, extended, event_time, event_location,
                    original_instruction=original_instruction, deviated_instruction=chosen_instruction,
                    earliest_choice_timestamp_ms=active_since,
                    diagnostics=diagnostic["navigation_inference"],
                )
                if choice is not None:
                    diagnostic["recovered_with_recorded_navigation"] = True
        if choice is None:
            failures["no_directly_supported_outgoing_choice"] += 1
            continue
        if any(_restricted_access(_edge_data(graph, edge).get("access"))
               for edge in (choice.chosen_edge_id, choice.suggested_edge_id)):
            failures["local_choice_uses_restricted_edge"] += 1
            continue
        old_overlap = _edge_overlap(gps_match, prior_match)
        new_overlap = (_edge_overlap(gps_match, matched["regenerated_route"])
                       if "regenerated_route" in matched else None)
        pair = (None if diagnostic.get("recovered_with_recorded_navigation") else
                extract_divergence_pair(gps_match, prior_match, event_state_index))
        if pair is not None:
            common_prefix = 0
            for left, right in zip(pair.preferred_edge_ids, pair.rejected_edge_ids):
                if left != right:
                    break
                common_prefix += 1
            if (common_prefix >= min(len(pair.preferred_edge_ids), len(pair.rejected_edge_ids))
                    or pair.preferred_edge_ids[common_prefix] != choice.chosen_edge_id
                    or pair.rejected_edge_ids[common_prefix] != choice.suggested_edge_id):
                pair = None
        if pair is not None:
            candidate = _pair_payload(
                graph,
                pair,
                deviation,
                response,
                ride_user[ride_id],
                old_overlap,
                new_overlap if new_overlap is not None else 0.0,
                gps_match.mean_observation_distance_m,
            )
        else:
            candidate = {
                "status": "review_required", "event_key": diagnostic["event_key"],
                "ride_group": _pseudonym("ride", ride_id),
                "rider_group": _pseudonym("rider", ride_user[ride_id]),
                "survey": {"primary_reason": response["primaryReason"],
                           "traffic_severity": response.get("trafficSeverity") or None,
                           "deviate_again": response.get("deviateAgain") or None},
                "preferred_observed_path": _path_payload(graph, choice.observed_continuation),
                "rejected_prior_suggestion": _path_payload(graph, (choice.suggested_edge_id,)),
                "quality": {},
            }
        candidate.update({
            "decision_timestamp_ms": choice.decision_timestamp_ms,
            "reported_event_timestamp_ms": event_time,
            "has_route_pair": pair is not None,
            "recorded_navigation": {
                field: deviation.get(field) or None for field in (
                    "originalRouteEdge", "deviatedEdge", "streetName", "generatedInstruction", "deviationInstruction")
            },
            "routing_context": {"origin_node_id": choice.source_node_id,
                                "destination_node_id": prior_match.traversed_edge_ids[-1][1]},
            "road_choice": {
                "source_node_id": choice.source_node_id,
                "suggested_edge_id": "|".join(choice.suggested_edge_id),
                "chosen_edge_id": "|".join(choice.chosen_edge_id),
                "supporting_gps_points": choice.supporting_gps_points,
                "supporting_edge_ids": ["|".join(edge) for edge in choice.supporting_edge_ids],
                "support_kind": ("gps_geometry_on_recorded_road" if diagnostic.get("recovered_with_recorded_navigation") else
                                 "chosen_edge_direct" if all(
                    edge == choice.chosen_edge_id for edge in choice.supporting_edge_ids
                ) else "alternative_continuation"),
                "first_chosen_timestamp_ms": choice.first_chosen_timestamp_ms,
                "label_method": ("recorded_navigation_and_local_geometry"
                                 if diagnostic.get("recovered_with_recorded_navigation") else "sequence_gps_choice"),
                **({"inference_evidence": diagnostic["navigation_inference"]}
                   if diagnostic.get("recovered_with_recorded_navigation") else {}),
            },
        })
        candidate["quality"].update({
            "gps_edge_overlap_with_prior": round(old_overlap, 3),
            "gps_edge_overlap_with_regenerated": round(new_overlap, 3) if new_overlap is not None else None,
            "mean_gps_to_matched_edge_m": round(choice.mean_local_match_distance_m, 2),
            "match_distance_scope": "last approach fix and first two alternative-continuation fixes",
        })
        diagnostic["retained"] = True
        diagnostic["has_route_pair"] = pair is not None
        candidates.append(candidate)

    candidate_riders = {candidate["rider_group"] for candidate in candidates}
    candidate_rides = {candidate["ride_group"] for candidate in candidates}
    manifest_path = data_dir / "manifest.json"
    synthetic = False
    if manifest_path.exists():
        synthetic = bool(json.loads(manifest_path.read_text(encoding="utf-8")).get("synthetic"))
    document = {
        "schema_version": 2,
        "provenance": (
            "Controlled synthetic export plus supplied OSM graph"
            if synthetic else "Official rider export plus supplied OSM graph"
        ),
        "synthetic": synthetic,
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
            "study_corridor_limit_m": None,
            "gps_before_event_s": 60,
            "late_report_recovery": "extend to latest pre-report route generation or previous deviation, whichever is later",
            "gps_after_event_s": 180,
            "stop_at_next_deviation": True,
            "require_gps_at_or_after_report": False,
            "require_intentional_reason": True,
            "require_gps_to_favor_regenerated_route": False,
            "require_shared_directed_edge_before_and_after": False,
            "route_ranking_pairs_optional": True,
            "minimum_alternative_continuation_gps_points": 2,
            "recorded_navigation_inference": {
                "enabled": True, "unique_supported_choice_required": True,
                "gps_road_distance_m": 20, "maximum_heading_error_degrees": 40,
                "minimum_forward_progress_m": 3, "maximum_gps_gap_seconds": 20,
                "event_to_chosen_road_distance_m": 80,
                "approach_to_junction_distance_m": 80, "gps_movement_speed_limit_mps": 45,
                "minimum_distance_advantage_over_suggested_road_m": 3,
                "future_maneuver_name_is_not_an_immediate_outgoing_edge": True,
                "navigation_text_is_label_evidence_only": True,
            },
            "max_match_candidates": max_match_candidates,
            "backward_jitter_tolerance_m": backward_jitter_m,
            "jitter_tolerance_scope": "per transition; source direction and speed bounds preserved",
        },
        "summary": {
            "reported_deviations": len(deviations),
            "eligible_intentional_events": eligible_events,
            "review_candidates": len(candidates),
            "candidate_rides": len(candidate_rides),
            "candidate_riders": len(candidate_riders),
            "route_ranking_pairs": sum(item["has_route_pair"] for item in candidates),
            "choice_only_candidates": sum(not item["has_route_pair"] for item in candidates),
            "recorded_navigation_recoveries": sum(bool(item.get("recovered_with_recorded_navigation"))
                                                  for item in event_diagnostics),
            "navigation_inference_outcomes": dict(Counter(
                item["navigation_inference"]["status"] for item in event_diagnostics
                if "navigation_inference" in item)),
            "strict_transition_failure_events_recovered": sum(
                item["retained"] and item["strict_first_failure"] is not None
                and item["strict_first_failure"]["reason"].startswith("No feasible directed transition")
                for item in event_diagnostics
            ),
            "failure_counts": dict(sorted(failures.items())),
        },
        "candidates": candidates,
        "event_diagnostics": event_diagnostics,
    }
    _validate_document(document, graph)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geojson-output", type=Path)
    parser.add_argument("--max-match-candidates", type=int, default=6)
    parser.add_argument("--backward-jitter-m", type=float, default=8.0)
    args = parser.parse_args()
    result = build_candidates(args.data_dir, args.graphml,
                              max_match_candidates=args.max_match_candidates,
                              backward_jitter_m=args.backward_jitter_m)
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
    summary = result["summary"]
    print(f"Deviation choices saved to {args.output}")
    if args.geojson_output:
        print(f"Review geometry saved to {args.geojson_output}")
    print(f"Reported deviations: {summary['reported_deviations']} | "
          f"eligible events: {summary['eligible_intentional_events']}")
    print(f"Review candidates: {summary['review_candidates']} from "
          f"{summary['candidate_riders']} riders")
    print(f"Local choices without a route pair: {summary['choice_only_candidates']} | "
          f"optional ranking pairs: {summary['route_ranking_pairs']} | "
          f"recovered strict transition failures: {summary['strict_transition_failure_events_recovered']}")
    print(f"Recovered using recorded navigation: {summary['recorded_navigation_recoveries']}")


if __name__ == "__main__":
    main()
