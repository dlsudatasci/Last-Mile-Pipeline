"""Collect timestamped Mapbox traffic and convert it to graph snapshots.

The collector uses the Directions ``driving-traffic`` profile with GeoJSON
geometry and segment annotations.  It never stores the access token.  The
result is route-scoped traffic evidence; edges not covered by the requested
route remain unknown rather than being assigned a value.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import torch

from .geometry import GeoPoint
from .osm_audit import RoadEdgeIndex
from .real_data import RealGraphData


MAPBOX_URL = "https://api.mapbox.com/directions/v5/mapbox/driving-traffic/{}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_directions_traffic(
    coordinates: list[tuple[float, float]],
    token: str,
    *,
    timeout_seconds: float = 30.0,
) -> dict:
    """Fetch one annotated route. Coordinates are ``(longitude, latitude)``."""
    if len(coordinates) < 2:
        raise ValueError("At least two coordinates are required")
    if not token:
        raise ValueError("A Mapbox access token is required")
    encoded = ";".join(f"{lon:.7f},{lat:.7f}" for lon, lat in coordinates)
    query = urllib.parse.urlencode(
        {
            "access_token": token,
            "alternatives": "false",
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
            "annotations": "congestion,congestion_numeric,duration,speed",
        }
    )
    request = urllib.request.Request(MAPBOX_URL.format(encoded) + "?" + query)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise ValueError(f"Mapbox Directions returned no usable route: {payload.get('code')}")
    return payload


def save_observation(payload: dict, output: Path, *, requested_at: str | None = None) -> dict:
    """Save a token-free, timestamped raw response for later alignment."""
    route = payload["routes"][0]
    annotation = _annotation_values(route)
    numeric = sum(value.get("congestion_numeric") is not None for value in annotation)
    speeds = sum(value.get("speed") is not None for value in annotation)
    document = {
        "schema_version": 1,
        "source": "mapbox_directions_driving_traffic",
        "requested_at": requested_at or _utc_now(),
        "traffic_coverage": {
            "annotated_segments": len(annotation),
            "numeric_congestion_segments": numeric,
            "unknown_congestion_segments": len(annotation) - numeric,
            "speed_segments": speeds,
            "congestion_fraction": numeric / len(annotation) if annotation else 0.0,
        },
        "route": route,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def _annotation_values(route: dict) -> list[dict]:
    values: list[dict] = []
    for leg in route.get("legs", []):
        annotation = leg.get("annotation", {})
        fields = {key: annotation.get(key, []) for key in ("congestion_numeric", "congestion", "duration", "speed")}
        count = max((len(value) for value in fields.values()), default=0)
        for index in range(count):
            values.append({key: value[index] if index < len(value) else None for key, value in fields.items()})
    return values


def _segment_points(route: dict) -> list[tuple[float, float]]:
    geometry = route.get("geometry", {})
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") != "LineString" or len(coordinates) < 2:
        raise ValueError("Mapbox route does not contain a GeoJSON LineString")
    return [(float(point[0]), float(point[1])) for point in coordinates]


def build_edge_snapshot(
    graph_data: RealGraphData,
    observation: dict,
    *,
    now_epoch_ms: int | None = None,
    match_radius_m: float = 75.0,
) -> torch.Tensor:
    """Align route segment annotations to graph edges.

    The returned columns match ``RealGraphData.schema.edge_dynamic``. Mapbox
    speed is represented relative to each edge's OSM reference speed, while
    congestion and speed retain separate observed masks.
    Segment midpoints are matched to directed OSM edges.  The midpoint test is
    deliberately conservative; unmatched or ambiguous segments stay unknown.
    """
    if match_radius_m <= 0:
        raise ValueError("match_radius_m must be positive")
    route = observation.get("route", observation.get("routes", [{}])[0])
    points = _segment_points(route)
    annotations = _annotation_values(route)
    required_dynamic = {
        "congestion_normalized",
        "speed_ratio_to_reference",
        "congestion_observed",
        "speed_observed",
        "traffic_age_scaled",
    }
    if not required_dynamic.issubset(graph_data.schema.edge_dynamic):
        raise ValueError("Real graph dynamic schema does not support Mapbox traffic")
    dynamic_index = {name: graph_data.schema.edge_dynamic.index(name) for name in required_dynamic}
    snapshot = torch.zeros((len(graph_data.edge_ids), len(graph_data.schema.edge_dynamic)), dtype=torch.float32)
    snapshot[:, dynamic_index["traffic_age_scaled"]] = 1.0
    edge_lookup = {edge_id: index for index, edge_id in enumerate(graph_data.edge_ids)}
    index = RoadEdgeIndex(graph_data.graph)
    now_ms = now_epoch_ms or int(time.time() * 1000)
    requested = observation.get("requested_at")
    try:
        observed_ms = int(datetime.fromisoformat(requested.replace("Z", "+00:00")).timestamp() * 1000)
    except (AttributeError, ValueError, TypeError):
        observed_ms = now_ms
    age = min(max((now_ms - observed_ms) / 3_600_000.0, 0.0), 1.0)
    matched_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for point, values in zip(zip(points, points[1:]), annotations):
        midpoint = ((point[0][0] + point[1][0]) / 2.0, (point[0][1] + point[1][1]) / 2.0)
        match = index.nearest(GeoPoint(midpoint[1], midpoint[0]))
        edge_id = f"{match.u}|{match.v}|{match.key}"
        if match.distance_m > match_radius_m or edge_id not in edge_lookup:
            continue
        edge_row = edge_lookup[edge_id]
        congestion = values.get("congestion_numeric")
        speed = values.get("speed")
        if congestion is not None:
            matched_values[edge_row]["congestion"].append(
                min(max(float(congestion) / 100.0, 0.0), 1.0)
            )
        if speed is not None:
            matched_values[edge_row]["speed_mps"].append(max(float(speed), 0.0))

    length_index = graph_data.schema.edge_static.index("length_per_100m")
    time_index = graph_data.schema.edge_static.index("reference_time_minutes")
    for edge_row, values in matched_values.items():
        congestion_values = values.get("congestion", [])
        speed_values = values.get("speed_mps", [])
        if congestion_values:
            snapshot[edge_row, dynamic_index["congestion_normalized"]] = sum(congestion_values) / len(congestion_values)
            snapshot[edge_row, dynamic_index["congestion_observed"]] = 1.0
        if speed_values:
            length_m = float(graph_data.edge_static[edge_row, length_index]) * 100.0
            reference_seconds = float(graph_data.edge_static[edge_row, time_index]) * 60.0
            reference_speed_mps = length_m / max(reference_seconds, 1e-6)
            mean_speed_mps = sum(speed_values) / len(speed_values)
            snapshot[edge_row, dynamic_index["speed_ratio_to_reference"]] = min(
                mean_speed_mps / reference_speed_mps, 2.0
            )
            snapshot[edge_row, dynamic_index["speed_observed"]] = 1.0
        if congestion_values or speed_values:
            snapshot[edge_row, dynamic_index["traffic_age_scaled"]] = age
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coordinates", help="semicolon-separated lon,lat pairs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="MAPBOX_ACCESS_TOKEN")
    args = parser.parse_args()
    coordinates = []
    for pair in args.coordinates.split(";"):
        lon, lat = pair.split(",", 1)
        coordinates.append((float(lon), float(lat)))
    payload = fetch_directions_traffic(coordinates, os.environ.get(args.token_env, ""))
    save_observation(payload, args.output)
    print(f"Saved timestamped Mapbox traffic observation to {args.output}")


if __name__ == "__main__":
    main()
