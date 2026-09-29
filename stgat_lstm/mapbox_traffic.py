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
from dataclasses import dataclass
import hashlib
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import torch
from pyproj import Transformer

from .geometry import GeoPoint
from .audit_road_network import RoadEdgeIndex, EdgeMatch
from .graph_data import RealGraphData


MAPBOX_URL = "https://api.mapbox.com/directions/v5/mapbox/driving-traffic/{}"


def timestamp_ms(value: str) -> int:
    """Parse an explicit timezone-aware timestamp; never substitute wall time."""
    if not isinstance(value, str):
        raise ValueError("Traffic timestamp must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid traffic timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("Traffic timestamps must include their timezone")
    return int(parsed.timestamp() * 1000)


@dataclass(frozen=True)
class TrafficRecord:
    observed_ms: int
    available_ms: int
    document: dict
    sha256: str


class TrafficArchive:
    """Reusable pre-decision traffic history with no future-data interpolation."""

    def __init__(self, graph_data: RealGraphData, paths: list[Path]):
        self.graph_data = graph_data
        self.records: list[TrafficRecord] = []
        self._snapshots: dict[str, torch.Tensor] = {}
        for path in sorted(paths):
            raw = path.read_bytes()
            document = json.loads(raw)
            if document.get("source") != "mapbox_directions_driving_traffic":
                raise ValueError(f"Not a saved Mapbox observation: {path.name}")
            observed = timestamp_ms(document.get("requested_at"))
            available = timestamp_ms(document.get("received_at", document.get("requested_at")))
            if available < observed:
                raise ValueError("Traffic receipt precedes its request")
            self.records.append(TrafficRecord(observed, available, document, hashlib.sha256(raw).hexdigest()))
        self.records.sort(key=lambda record: (record.observed_ms, record.available_ms, record.sha256))

    @classmethod
    def from_directory(cls, graph_data: RealGraphData, directory: Path):
        if not directory.is_dir():
            raise ValueError(f"Traffic archive directory does not exist: {directory}")
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise ValueError("Traffic archive contains no JSON observations")
        return cls(graph_data, paths)

    def sequence(self, cutoff_ms: int, *, steps: int = 6, interval_s: int = 300,
                 max_age_s: int = 900) -> tuple[tuple[torch.Tensor, ...], dict]:
        if steps <= 0 or interval_s <= 0 or max_age_s <= 0:
            raise ValueError("Traffic history settings must be positive")
        schema = self.graph_data.schema.edge_dynamic
        age_column = schema.index("traffic_age_scaled")
        congestion_mask = schema.index("congestion_observed")
        speed_mask = schema.index("speed_observed")
        pairs = ((schema.index("congestion_normalized"), congestion_mask),
                 (schema.index("speed_ratio_to_reference"), speed_mask))
        frames = []
        used = set()
        for step in range(steps):
            slot_ms = cutoff_ms - (steps - 1 - step) * interval_s * 1000
            frame = torch.zeros((len(self.graph_data.edge_ids), len(schema)), dtype=torch.float32)
            frame[:, age_column] = 1.0
            field_ages = torch.zeros((len(self.graph_data.edge_ids), len(pairs)), dtype=torch.float32)
            for record in self.records:
                if not (slot_ms - max_age_s * 1000 <= record.observed_ms <= slot_ms
                        and record.available_ms <= slot_ms):
                    continue
                if record.sha256 not in self._snapshots:
                    self._snapshots[record.sha256] = build_edge_snapshot(
                        self.graph_data, record.document, now_epoch_ms=record.observed_ms)
                snapshot = self._snapshots[record.sha256]
                observed_rows = (snapshot[:, congestion_mask] == 1) | (snapshot[:, speed_mask] == 1)
                if not observed_rows.any():
                    continue
                # Keep older observations for uncovered edges; never fill future values.
                age = min((slot_ms - record.observed_ms) / 3_600_000, 1.0)
                for field_index, (value_column, mask_column) in enumerate(pairs):
                    rows = snapshot[:, mask_column] == 1
                    frame[rows, value_column] = snapshot[rows, value_column]
                    frame[rows, mask_column] = 1.0
                    field_ages[rows, field_index] = age
                used.add(record.sha256)
            # Represent the oldest still-active field, rather than overwritten values.
            observed_rows = (frame[:, congestion_mask] == 1) | (frame[:, speed_mask] == 1)
            frame[observed_rows, age_column] = field_ages[observed_rows].max(dim=1).values
            frames.append(frame)
        coverage = [int(((frame[:, congestion_mask] == 1) | (frame[:, speed_mask] == 1)).sum())
                    for frame in frames]
        return tuple(frames), {
            "cutoff_ms": cutoff_ms, "steps": steps, "interval_seconds": interval_s,
            "max_age_seconds": max_age_s, "observed_edges_per_step": coverage,
            "used_observation_sha256": sorted(used),
            "legacy_request_only_timestamps": sum("received_at" not in record.document for record in self.records),
        }


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
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError:
        raise RuntimeError("Mapbox request failed; check connection, token and account access") from None
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
        "received_at": _utc_now(),
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


def collect_live_route_observation(
    graph_data: RealGraphData,
    origin_node_id: str,
    destination_node_id: str,
    archive_dir: Path,
    *,
    token_env: str = "MAPBOX_ACCESS_TOKEN",
) -> tuple[Path, dict]:
    """Fetch current route traffic for two graph nodes and append it to an archive."""
    if origin_node_id not in graph_data.graph or destination_node_id not in graph_data.graph:
        raise ValueError("Live-traffic endpoints must exist in the routing graph")
    transformer = Transformer.from_crs(graph_data.graph.graph["crs"], "EPSG:4326", always_xy=True)
    coordinates = []
    for node_id in (origin_node_id, destination_node_id):
        node = graph_data.graph.nodes[node_id]
        longitude, latitude = transformer.transform(float(node["x"]), float(node["y"]))
        coordinates.append((longitude, latitude))
    requested_at = _utc_now()
    payload = fetch_directions_traffic(coordinates, os.environ.get(token_env, ""))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = archive_dir / f"traffic_{stamp}.json"
    document = save_observation(payload, output, requested_at=requested_at)
    return output, document


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


def match_traffic_segment(index: RoadEdgeIndex, graph_data: RealGraphData,
                         start: tuple[float, float], end: tuple[float, float],
                         radius_m: float) -> EdgeMatch | None:
    """Use segment direction to distinguish coincident opposite OSM edges."""
    midpoint = GeoPoint((start[1] + end[1]) / 2, (start[0] + end[0]) / 2)
    projected, candidates = index.nearby(midpoint, radius_m, 32)
    left = index.project(GeoPoint(start[1], start[0]))
    right = index.project(GeoPoint(end[1], end[0]))
    dx, dy = right.x - left.x, right.y - left.y
    displacement = math.hypot(dx, dy)
    if displacement < 0.1:
        return None
    viable = []
    for row, distance in candidates:
        u, v, key, raw = index.records[row]
        geometry = index.tree.geometries[row]
        position = geometry.project(projected)
        before = geometry.interpolate(max(position - 2, 0))
        after = geometry.interpolate(min(position + 2, geometry.length))
        tx, ty = after.x - before.x, after.y - before.y
        source = graph_data.graph.nodes[u]
        source_point = (float(source["x"]), float(source["y"]))
        first, last = geometry.coords[0], geometry.coords[-1]
        if math.dist(last, source_point) < math.dist(first, source_point):
            tx, ty = -tx, -ty
        length = math.hypot(tx, ty)
        if length < 0.1 or (tx * dx + ty * dy) / (length * displacement) < 0.5:
            continue
        viable.append((distance, EdgeMatch(u, v, key, distance, str(raw.get("name") or ""))))
    viable.sort(key=lambda item: (item[0], item[1].u, item[1].v, item[1].key))
    if not viable or (len(viable) > 1 and abs(viable[0][0] - viable[1][0]) < 0.5):
        return None
    return viable[0][1]


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
    now_ms = int(time.time() * 1000) if now_epoch_ms is None else now_epoch_ms
    requested = observation.get("requested_at")
    observed_ms = timestamp_ms(requested)
    if observed_ms > now_ms:
        raise ValueError("Traffic observation is later than its input cutoff")
    if len(annotations) != len(points) - 1:
        raise ValueError("Traffic annotations and geometry segment counts differ")
    age = min(max((now_ms - observed_ms) / 3_600_000.0, 0.0), 1.0)
    matched_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for point, values in zip(zip(points, points[1:]), annotations):
        match = match_traffic_segment(index, graph_data, point[0], point[1], match_radius_m)
        if match is None:
            continue
        edge_id = f"{match.u}|{match.v}|{match.key}"
        if match.distance_m > match_radius_m or edge_id not in edge_lookup:
            continue
        edge_row = edge_lookup[edge_id]
        congestion = values.get("congestion_numeric")
        speed = values.get("speed")
        if congestion is not None:
            if not math.isfinite(float(congestion)) or not 0 <= float(congestion) <= 100:
                raise ValueError("Invalid numeric congestion annotation")
            matched_values[edge_row]["congestion"].append(
                min(max(float(congestion) / 100.0, 0.0), 1.0)
            )
        if speed is not None:
            if not math.isfinite(float(speed)) or float(speed) < 0:
                raise ValueError("Invalid speed annotation")
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
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path)
    outputs.add_argument("--archive-dir", type=Path)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--token-env", default="MAPBOX_ACCESS_TOKEN")
    args = parser.parse_args()
    coordinates = []
    for pair in args.coordinates.split(";"):
        lon, lat = pair.split(",", 1)
        coordinates.append((float(lon), float(lat)))
    if args.samples <= 0 or args.interval_seconds <= 0:
        parser.error("samples and interval-seconds must be positive")
    if args.samples > 1 and args.archive_dir is None:
        parser.error("Repeated collection requires --archive-dir so files are not overwritten")
    for sample in range(args.samples):
        requested_at = _utc_now()
        payload = fetch_directions_traffic(coordinates, os.environ.get(args.token_env, ""))
        output = args.output
        if args.archive_dir is not None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.archive_dir / f"traffic_{stamp}.json"
        save_observation(payload, output, requested_at=requested_at)
        print(f"Saved traffic observation {sample + 1}/{args.samples} to {output}", flush=True)
        if sample + 1 < args.samples:
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
