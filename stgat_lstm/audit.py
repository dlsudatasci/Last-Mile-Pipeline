"""Read-only aggregate audit of the current rider exports.

This stage intentionally creates no preference labels. Route geometries and
GPS trajectories must be aligned to a fixed road graph first.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .geometry import GeoPoint, haversine_m, parse_gps_location, parse_route_points, point_to_polyline_distance_m


REQUIRED = {
    "rides_clean.csv": {"id", "userId", "startTime", "endTime"},
    "generated_routes_clean.csv": {"routeId", "rideId", "type", "generatedAt", "routePoints"},
    "deviations_clean.csv": {"deviationId", "rideId", "routeId", "timestamp", "originalRouteEdge", "deviatedEdge"},
    "deviationResponses_clean.csv": {"deviationId", "rideId", "primaryReason"},
    "map_points_clean.csv": {"rideId", "timestamp", "latitude", "longitude"},
}


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path.name}: {sorted(required - set(reader.fieldnames or []))}")
        return list(reader)


def _rounded_median(values: list[float]) -> float | None:
    return round(median(values), 1) if values else None


def audit_exports(data_dir: Path) -> dict:
    """Return counts and provenance checks; never expose rider IDs or GPS fixes."""
    tables = {name: _read_csv(data_dir / name, columns) for name, columns in REQUIRED.items()}
    rides = tables["rides_clean.csv"]
    routes = tables["generated_routes_clean.csv"]
    deviations = tables["deviations_clean.csv"]
    responses = tables["deviationResponses_clean.csv"]
    points = tables["map_points_clean.csv"]

    ride_ids = {row["id"] for row in rides}
    route_by_id = {row["routeId"]: row for row in routes}
    if len(ride_ids) != len(rides) or len(route_by_id) != len(routes):
        raise ValueError("Duplicate ride or route IDs prevent unambiguous joins")
    routes_by_ride: dict[str, list[dict[str, str]]] = defaultdict(list)
    for route in routes:
        routes_by_ride[route["rideId"]].append(route)
    referenced_future_routes = 0
    referenced_prior_routes = 0
    unknown_route_refs = 0
    temporal_prior_candidates = 0
    ambiguous_prior_timestamps = 0
    route_geometry = {}
    route_geometry_errors = 0
    for route in routes:
        try:
            route_geometry[route["routeId"]] = parse_route_points(route["routePoints"])
        except (SyntaxError, TypeError, ValueError):
            route_geometry_errors += 1

    gps_by_ride: dict[str, list[tuple[int, GeoPoint]]] = defaultdict(list)
    invalid_gps_points = 0
    for point in points:
        try:
            gps_by_ride[point["rideId"]].append(
                (int(point["timestamp"]), GeoPoint(float(point["latitude"]), float(point["longitude"])))
            )
        except ValueError:
            invalid_gps_points += 1
    for fixes in gps_by_ride.values():
        fixes.sort(key=lambda item: item[0])

    responses_by_deviation = {row["deviationId"]: row for row in responses}
    event_to_prior_distances: list[float] = []
    event_to_referenced_distances: list[float] = []
    prior_to_referenced_destination_distances: list[float] = []
    deviations_with_parseable_event = 0
    deviations_with_parseable_prior = 0
    deviations_with_parseable_referenced = 0
    deviations_with_gps_before_and_after = 0
    deviations_with_same_destination_100m = 0
    pre_map_match_candidates = 0
    excluded_reason_counts: Counter[str] = Counter()

    for deviation in deviations:
        route = route_by_id.get(deviation["routeId"])
        if route is None or route["rideId"] != deviation["rideId"]:
            unknown_route_refs += 1
            continue
        if int(route["generatedAt"]) > int(deviation["timestamp"]):
            referenced_future_routes += 1
        else:
            referenced_prior_routes += 1
        prior = [
            candidate
            for candidate in routes_by_ride[deviation["rideId"]]
            if int(candidate["generatedAt"]) < int(deviation["timestamp"])
        ]
        if prior:
            latest_time = max(int(candidate["generatedAt"]) for candidate in prior)
            latest = [candidate for candidate in prior if int(candidate["generatedAt"]) == latest_time]
            if len(latest) == 1:
                temporal_prior_candidates += 1
            else:
                ambiguous_prior_timestamps += 1

            try:
                event_point = parse_gps_location(deviation["gpsLocation"])
                deviations_with_parseable_event += 1
            except ValueError:
                continue
            if len(latest) != 1:
                continue
            prior_polyline = route_geometry.get(latest[0]["routeId"])
            referenced_polyline = route_geometry.get(deviation["routeId"])
            if prior_polyline is not None:
                deviations_with_parseable_prior += 1
                event_to_prior_distances.append(point_to_polyline_distance_m(event_point, prior_polyline))
            if referenced_polyline is not None:
                deviations_with_parseable_referenced += 1
                event_to_referenced_distances.append(point_to_polyline_distance_m(event_point, referenced_polyline))
            same_destination = False
            if prior_polyline is not None and referenced_polyline is not None:
                destination_distance = haversine_m(prior_polyline[-1], referenced_polyline[-1])
                prior_to_referenced_destination_distances.append(destination_distance)
                same_destination = destination_distance <= 100.0
                deviations_with_same_destination_100m += int(same_destination)

            event_time = int(deviation["timestamp"])
            nearby = gps_by_ride.get(deviation["rideId"], [])
            has_before = any(event_time - 60_000 <= timestamp <= event_time for timestamp, _ in nearby)
            has_after = any(event_time < timestamp <= event_time + 60_000 for timestamp, _ in nearby)
            if has_before and has_after:
                deviations_with_gps_before_and_after += 1

            response = responses_by_deviation.get(deviation["deviationId"])
            reason = response["primaryReason"] if response else "Missing response"
            reason_is_usable = reason not in {
                "Missing response",
                "Other",
                "Personal Stop (Meal, Restroom, Break, Refueling, etc.)",
            }
            if not reason_is_usable:
                excluded_reason_counts[reason] += 1
            if prior_polyline is not None and referenced_polyline is not None and same_destination and has_before and has_after and reason_is_usable:
                pre_map_match_candidates += 1

    points_by_ride: dict[str, list[int]] = defaultdict(list)
    for point in points:
        points_by_ride[point["rideId"]].append(int(point["timestamp"]))
    long_gaps = 0
    for times in points_by_ride.values():
        times.sort()
        long_gaps += sum(right - left > 60_000 for left, right in zip(times, times[1:]))

    responses_per_deviation = Counter(row["deviationId"] for row in responses)
    deviations_with_no_response = sum(responses_per_deviation[row["deviationId"]] == 0 for row in deviations)
    return {
        "provenance": "Aggregate read-only audit of exported rider data; no synthetic data mixed in",
        "counts": {
            "rides": len(rides),
            "riders_in_rides": len({row["userId"] for row in rides}),
            "generated_routes": len(routes),
            "reported_deviations": len(deviations),
            "deviation_responses": len(responses),
            "gps_points": len(points),
        },
        "route_references": {
            "refer_to_route_generated_after_event": referenced_future_routes,
            "refer_to_route_available_at_event": referenced_prior_routes,
            "missing_or_other_ride": unknown_route_refs,
            "single_latest_prior_route_by_timestamp": temporal_prior_candidates,
            "ambiguous_latest_prior_route_timestamps": ambiguous_prior_timestamps,
        },
        "quality": {
            "gps_intervals_over_60_seconds": long_gaps,
            "deviations_without_response": deviations_with_no_response,
            "deviations_with_multiple_responses": sum(responses_per_deviation[row["deviationId"]] > 1 for row in deviations),
            "orphan_route_ride_ids": sum(row["rideId"] not in ride_ids for row in routes),
            "orphan_point_ride_ids": sum(row["rideId"] not in ride_ids for row in points),
            "missing_gps_accuracy_column": "accuracy" not in points[0] if points else True,
        },
        "geometry_screen": {
            "route_polylines_parseable": len(route_geometry),
            "route_polyline_parse_errors": route_geometry_errors,
            "invalid_gps_points": invalid_gps_points,
            "deviations_with_parseable_event_location": deviations_with_parseable_event,
            "deviations_with_parseable_prior_route": deviations_with_parseable_prior,
            "deviations_with_parseable_referenced_route": deviations_with_parseable_referenced,
            "deviations_with_gps_within_60s_before_and_after": deviations_with_gps_before_and_after,
            "prior_and_referenced_routes_end_within_100m": deviations_with_same_destination_100m,
            "median_event_distance_to_prior_route_m": _rounded_median(event_to_prior_distances),
            "median_event_distance_to_referenced_route_m": _rounded_median(event_to_referenced_distances),
            "median_prior_to_referenced_destination_distance_m": _rounded_median(
                prior_to_referenced_destination_distances
            ),
            "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
            "pre_map_match_candidates": pre_map_match_candidates,
            "candidate_definition": (
                "Unique prior and referenced polylines; endpoints within 100 m; GPS within 60 s before and after; "
                "survey reason is neither missing, Other, nor Personal Stop. These remain candidates, not labels."
            ),
        },
        "routing_preference_examples_ready": 0,
        "next_data_step": "Validate prior-route geometry against GPS, then map-match GPS and suggestions to the same directed graph before labeling rider choices.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing the seven cleaned rider CSVs")
    args = parser.parse_args()
    print(json.dumps(audit_exports(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
