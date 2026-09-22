"""Audit OSM compatibility and spatial coverage for the rider exports.

This is a nearest-edge quality screen, not the final trajectory map matcher.
Independent nearest-edge matches can switch between parallel carriageways and
must not be converted directly into path-choice labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from pyproj import Transformer
from shapely import from_wkt
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .geometry import GeoPoint, parse_gps_location, parse_route_points


@dataclass(frozen=True)
class EdgeMatch:
    u: str
    v: str
    key: str
    distance_m: float
    name: str | None


class RoadEdgeIndex:
    """Projected OSM edge geometries with a spatial nearest-neighbour index."""

    def __init__(self, graph: nx.MultiDiGraph, edge_filter=None) -> None:
        crs = graph.graph.get("crs")
        if not crs:
            raise ValueError("Graph is missing its CRS")
        self.transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.records: list[tuple[str, str, str, dict]] = []
        geometries = []
        for u, v, key, data in graph.edges(keys=True, data=True):
            if edge_filter is not None and not edge_filter(data):
                continue
            raw_geometry = data.get("geometry")
            if raw_geometry:
                geometry = from_wkt(raw_geometry) if isinstance(raw_geometry, str) else raw_geometry
            else:
                left, right = graph.nodes[u], graph.nodes[v]
                geometry = LineString(
                    [(float(left["x"]), float(left["y"])), (float(right["x"]), float(right["y"]))]
                )
            if geometry.is_empty:
                continue
            self.records.append((str(u), str(v), str(key), data))
            geometries.append(geometry)
        if not geometries:
            raise ValueError("No usable edge geometries found")
        self.tree = STRtree(geometries)

    def project(self, point: GeoPoint) -> Point:
        x, y = self.transformer.transform(point.longitude, point.latitude)
        return Point(x, y)

    def nearest(self, point: GeoPoint) -> EdgeMatch:
        projected = self.project(point)
        index = int(self.tree.nearest(projected))
        distance = float(projected.distance(self.tree.geometries[index]))
        u, v, key, data = self.records[index]
        raw_name = data.get("name")
        return EdgeMatch(u, v, key, distance, str(raw_name) if raw_name else None)

    def nearby(self, point: GeoPoint, radius_m: float, limit: int) -> tuple[Point, list[tuple[int, float]]]:
        """Return nearby edge indices and distances ordered by observation fit."""
        if radius_m <= 0.0 or limit <= 0:
            raise ValueError("radius_m and limit must be positive")
        projected = self.project(point)
        indices = self.tree.query(projected.buffer(radius_m))
        candidates = [
            (int(index), float(projected.distance(self.tree.geometries[int(index)]))) for index in indices
        ]
        candidates = [candidate for candidate in candidates if candidate[1] <= radius_m]
        candidates.sort(key=lambda candidate: candidate[1])
        return projected, candidates[:limit]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")
        return list(reader)


def _percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    value = ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 1)


def _distance_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median_m": _percentile(values, 0.5),
        "p95_m": _percentile(values, 0.95),
        "within_10m": sum(value <= 10.0 for value in values),
        "within_20m": sum(value <= 20.0 for value in values),
        "within_35m": sum(value <= 35.0 for value in values),
    }


def _is_taft_edge(data: dict) -> bool:
    return "taft avenue" in str(data.get("name", "")).lower()


def _feature_report(path: Path | None, graph: nx.MultiDiGraph) -> dict | None:
    if path is None:
        return None
    rows = _read_csv(path)
    if not rows:
        raise ValueError("Node feature CSV is empty")
    identifiers = [row.get("osmid", "") for row in rows]
    graph_nodes = {str(node) for node in graph.nodes}
    headers = list(rows[0])
    feature_columns = [column for column in headers if column not in {"osmid", "geometry"}]
    return {
        "rows": len(rows),
        "graph_nodes": graph.number_of_nodes(),
        "duplicate_osmid_rows": len(identifiers) - len(set(identifiers)),
        "feature_ids_missing_from_graph": len(set(identifiers) - graph_nodes),
        "graph_nodes_missing_feature_row": len(graph_nodes - set(identifiers)),
        "feature_columns": feature_columns,
        "note": "These POI counts are node features; road properties and learned preference costs belong on directed edges.",
    }


def audit_osm_compatibility(data_dir: Path, graphml_path: Path, node_features: Path | None = None) -> dict:
    graph = nx.read_graphml(graphml_path, force_multigraph=True)
    if not graph.is_directed() or not graph.is_multigraph():
        raise ValueError("Routing requires a directed multigraph")
    edge_index = RoadEdgeIndex(graph)
    taft_index = RoadEdgeIndex(graph, _is_taft_edge)

    routes = _read_csv(data_dir / "generated_routes_clean.csv")
    points = _read_csv(data_dir / "map_points_clean.csv")
    deviations = _read_csv(data_dir / "deviations_clean.csv")
    responses = _read_csv(data_dir / "deviationResponses_clean.csv")

    route_match_distances: list[float] = []
    for route in routes:
        for point in parse_route_points(route["routePoints"]):
            route_match_distances.append(edge_index.nearest(point).distance_m)

    gps_match_distances: list[float] = []
    gps_to_taft_distances: list[float] = []
    ride_taft_distances: dict[str, list[float]] = defaultdict(list)
    for row in points:
        point = GeoPoint(float(row["latitude"]), float(row["longitude"]))
        gps_match_distances.append(edge_index.nearest(point).distance_m)
        taft_distance = taft_index.nearest(point).distance_m
        gps_to_taft_distances.append(taft_distance)
        ride_taft_distances[row["rideId"]].append(taft_distance)

    event_match_distances: list[float] = []
    event_taft_distances: dict[str, float] = {}
    for row in deviations:
        point = parse_gps_location(row["gpsLocation"])
        event_match_distances.append(edge_index.nearest(point).distance_m)
        event_taft_distances[row["deviationId"]] = taft_index.nearest(point).distance_m

    response_reason = {row["deviationId"]: row["primaryReason"] for row in responses}
    excluded_reasons = {
        "Other",
        "Personal Stop (Meal, Restroom, Break, Refueling, etc.)",
    }
    eligible_events = [
        row for row in deviations if response_reason.get(row["deviationId"], "Missing response") not in excluded_reasons
    ]

    edge_keys = Counter(key for _, _, _, data in graph.edges(keys=True, data=True) for key in data)
    total_edges = graph.number_of_edges()
    return {
        "provenance": "Read-only compatibility screen of real exports against the supplied OSM graph; no labels created",
        "graph": {
            "path": str(graphml_path),
            "created_date": graph.graph.get("created_date"),
            "created_with": graph.graph.get("created_with"),
            "crs": graph.graph.get("crs"),
            "directed": graph.is_directed(),
            "multigraph": graph.is_multigraph(),
            "nodes": graph.number_of_nodes(),
            "edges": total_edges,
            "taft_named_edges": sum(_is_taft_edge(data) for *_, data in graph.edges(keys=True, data=True)),
        },
        "edge_attribute_coverage": {
            key: {"count": edge_keys[key], "percent": round(100.0 * edge_keys[key] / total_edges, 1)}
            for key in ("geometry", "length", "highway", "name", "oneway", "lanes", "maxspeed", "access")
        },
        "nearest_osm_edge_distance": {
            "saved_route_polyline_points": _distance_summary(route_match_distances),
            "recorded_gps_points": _distance_summary(gps_match_distances),
            "deviation_event_points": _distance_summary(event_match_distances),
        },
        "taft_avenue_corridor": {
            "corridor_definition": "Distance to an OSM edge whose name contains 'Taft Avenue'; provisional 500 m threshold.",
            "gps_points_within_500m": sum(distance <= 500.0 for distance in gps_to_taft_distances),
            "gps_points_total": len(gps_to_taft_distances),
            "rides_with_any_point_within_500m": sum(min(distances) <= 500.0 for distances in ride_taft_distances.values()),
            "rides_fully_within_500m": sum(max(distances) <= 500.0 for distances in ride_taft_distances.values()),
            "rides_total": len(ride_taft_distances),
            "deviation_events_within_500m": sum(distance <= 500.0 for distance in event_taft_distances.values()),
            "eligible_pre_map_match_events_within_500m": sum(
                event_taft_distances[row["deviationId"]] <= 500.0 for row in eligible_events
            ),
        },
        "node_features": _feature_report(node_features, graph),
        "training_examples_ready": 0,
        "next_step": (
            "Run sequence-aware map matching on the Taft-corridor subset, reconstruct contiguous old and chosen paths, "
            "then split evaluation by rider before training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("--node-features", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_osm_compatibility(args.data_dir, args.graphml, args.node_features), indent=2))


if __name__ == "__main__":
    main()
