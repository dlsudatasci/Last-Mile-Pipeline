"""Run the HMM map matcher on provisional Taft Avenue deviation candidates.

The output is aggregate screening evidence. It intentionally does not write
matched rider trajectories or promote candidates to training labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import networkx as nx

from .geometry import GeoPoint, parse_gps_location, parse_route_points
from .map_matching import HiddenMarkovMatcher, MapMatchResult, TimedObservation
from .osm_audit import RoadEdgeIndex, _is_taft_edge


EXCLUDED_REASONS = {
    "Other",
    "Personal Stop (Meal, Restroom, Break, Refueling, etc.)",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _sample_points(points: tuple[GeoPoint, ...], limit: int = 20) -> list[TimedObservation]:
    if len(points) <= limit:
        selected = points
    else:
        indices = sorted({round(index * (len(points) - 1) / (limit - 1)) for index in range(limit)})
        selected = tuple(points[index] for index in indices)
    return [TimedObservation(point) for point in selected]


def _edge_overlap(observed: MapMatchResult, route: MapMatchResult) -> float:
    observed_edges = set(observed.traversed_edge_ids)
    route_edges = set(route.traversed_edge_ids)
    if not observed_edges:
        return 0.0
    return len(observed_edges & route_edges) / len(observed_edges)


def _route_similarity(left: MapMatchResult, right: MapMatchResult) -> float:
    left_edges, right_edges = set(left.traversed_edge_ids), set(right.traversed_edge_ids)
    union = left_edges | right_edges
    return len(left_edges & right_edges) / len(union) if union else 1.0


def audit_sequence_matches(data_dir: Path, graphml_path: Path) -> dict:
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

    routes = _read_csv(data_dir / "generated_routes_clean.csv")
    deviations = _read_csv(data_dir / "deviations_clean.csv")
    responses = _read_csv(data_dir / "deviationResponses_clean.csv")
    points = _read_csv(data_dir / "map_points_clean.csv")

    route_by_id = {row["routeId"]: row for row in routes}
    routes_by_ride: dict[str, list[dict[str, str]]] = defaultdict(list)
    for route in routes:
        routes_by_ride[route["rideId"]].append(route)
    response_reason = {row["deviationId"]: row["primaryReason"] for row in responses}
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

    taft_eligible = []
    for deviation in deviations:
        reason = response_reason.get(deviation["deviationId"], "Missing response")
        if reason in EXCLUDED_REASONS or reason == "Missing response":
            continue
        event_point = parse_gps_location(deviation["gpsLocation"])
        if taft_index.nearest(event_point).distance_m <= 500.0:
            taft_eligible.append(deviation)

    failures: Counter[str] = Counter()
    gps_mean_distances: list[float] = []
    route_similarities: list[float] = []
    new_overlaps: list[float] = []
    old_overlaps: list[float] = []
    successful_triplets = 0
    distinct_route_pairs = 0
    sequence_candidates = 0

    for deviation in taft_eligible:
        event_time = int(deviation["timestamp"])
        ride_id = deviation["rideId"]
        later_events = [time for time in events_by_ride[ride_id] if time > event_time]
        window_end = min(event_time + 120_000, min(later_events) - 1 if later_events else event_time + 120_000)
        observations = [
            observation
            for observation in gps_by_ride[ride_id]
            if event_time <= (observation.timestamp_ms or 0) <= window_end
        ]
        if len(observations) < 2:
            failures["fewer_than_two_post_event_gps_points"] += 1
            continue

        prior_routes = [
            route for route in routes_by_ride[ride_id] if int(route["generatedAt"]) < event_time
        ]
        if not prior_routes:
            failures["no_prior_route"] += 1
            continue
        prior_route = max(prior_routes, key=lambda route: int(route["generatedAt"]))
        referenced_route = route_by_id.get(deviation["routeId"])
        if referenced_route is None:
            failures["missing_referenced_route"] += 1
            continue

        try:
            gps_match = matcher.match(observations)
            prior_match = matcher.match(_sample_points(parse_route_points(prior_route["routePoints"])))
            new_match = matcher.match(_sample_points(parse_route_points(referenced_route["routePoints"])))
        except ValueError as error:
            message = str(error)
            if message.startswith("No feasible directed transition"):
                category = "no_feasible_directed_transition"
            elif message.startswith("No OSM edge candidate"):
                category = "no_nearby_osm_edge_candidate"
            else:
                category = message.split(":", 1)[0]
            failures[category] += 1
            continue

        successful_triplets += 1
        similarity = _route_similarity(prior_match, new_match)
        old_overlap = _edge_overlap(gps_match, prior_match)
        new_overlap = _edge_overlap(gps_match, new_match)
        gps_mean_distances.append(gps_match.mean_observation_distance_m)
        route_similarities.append(similarity)
        old_overlaps.append(old_overlap)
        new_overlaps.append(new_overlap)
        is_distinct = similarity < 0.95
        distinct_route_pairs += int(is_distinct)
        sequence_candidates += int(is_distinct and new_overlap > old_overlap)

    def rounded_median(values: list[float]) -> float | None:
        return round(median(values), 3) if values else None

    return {
        "provenance": "Aggregate HMM sequence screen of real exports; candidates are not training labels",
        "parameters": {
            "taft_corridor_m": 500,
            "post_event_window_s": 120,
            "candidate_search_radius_m": matcher.search_radius_m,
            "max_candidates_per_observation": matcher.max_candidates,
            "gps_sigma_m": matcher.gps_sigma_m,
            "transition_beta_m": matcher.transition_beta_m,
            "max_speed_mps": matcher.max_speed_mps,
            "saved_route_sample_limit": 20,
        },
        "screen": {
            "eligible_intentional_events_in_taft_corridor": len(taft_eligible),
            "successful_gps_prior_new_match_triplets": successful_triplets,
            "distinct_prior_and_new_route_pairs": distinct_route_pairs,
            "new_route_overlap_better_than_prior": sum(
                new > old for new, old in zip(new_overlaps, old_overlaps)
            ),
            "sequence_review_candidates": sequence_candidates,
            "median_gps_observation_distance_m": rounded_median(gps_mean_distances),
            "median_prior_new_edge_jaccard": rounded_median(route_similarities),
            "median_gps_edge_overlap_with_prior": rounded_median(old_overlaps),
            "median_gps_edge_overlap_with_new": rounded_median(new_overlaps),
            "failure_counts": dict(sorted(failures.items())),
        },
        "routing_preference_examples_ready": 0,
        "review_requirement": (
            "Inspect each candidate's divergence/rejoin edges, confirm a common origin-destination subpath, and exclude "
            "later reroutes or implausible matches before creating rider-choice pairs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("graphml", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_sequence_matches(args.data_dir, args.graphml), indent=2))


if __name__ == "__main__":
    main()
