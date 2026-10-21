"""Small geospatial helpers for auditing exported route polylines.

The export stores coordinates as Python-literal strings rather than a stable
geospatial format.  These helpers parse that representation strictly and use
a local tangent-plane approximation for point-to-segment distances.  The
approximation is appropriate for the short Taft Avenue study corridor.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass


EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.latitude) or not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"Invalid latitude: {self.latitude!r}")
        if not math.isfinite(self.longitude) or not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"Invalid longitude: {self.longitude!r}")


def parse_gps_location(raw: str) -> GeoPoint:
    """Parse the export's ``latitude,longitude`` deviation location."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("gpsLocation must contain latitude,longitude")
    return GeoPoint(float(parts[0]), float(parts[1]))


def parse_route_points(raw: str) -> tuple[GeoPoint, ...]:
    """Parse a routePoints cell without evaluating executable code."""
    value = ast.literal_eval(raw)
    if not isinstance(value, (list, tuple)):
        raise ValueError("routePoints must be a list")
    points: list[GeoPoint] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"routePoints[{index}] must be an object")
        latitude = item.get("lat", item.get("latitude"))
        longitude = item.get("lng", item.get("longitude"))
        if latitude is None or longitude is None:
            raise ValueError(f"routePoints[{index}] lacks latitude or longitude")
        points.append(GeoPoint(float(latitude), float(longitude)))
    if not points:
        raise ValueError("routePoints must contain at least one point")
    return tuple(points)


def haversine_m(left: GeoPoint, right: GeoPoint) -> float:
    """Great-circle distance in metres."""
    lat1, lat2 = math.radians(left.latitude), math.radians(right.latitude)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(right.longitude - left.longitude)
    a = math.sin(delta_lat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def point_to_polyline_distance_m(point: GeoPoint, polyline: tuple[GeoPoint, ...]) -> float:
    """Return the shortest local distance from a point to a polyline."""
    if not polyline:
        raise ValueError("Cannot measure distance to an empty polyline")
    if len(polyline) == 1:
        return haversine_m(point, polyline[0])

    latitude_scale = EARTH_RADIUS_M * math.pi / 180.0
    longitude_scale = latitude_scale * math.cos(math.radians(point.latitude))

    def local(candidate: GeoPoint) -> tuple[float, float]:
        return (
            (candidate.longitude - point.longitude) * longitude_scale,
            (candidate.latitude - point.latitude) * latitude_scale,
        )

    best = math.inf
    for start, end in zip(polyline, polyline[1:]):
        start_x, start_y = local(start)
        end_x, end_y = local(end)
        delta_x, delta_y = end_x - start_x, end_y - start_y
        squared_length = delta_x * delta_x + delta_y * delta_y
        if squared_length == 0.0:
            distance = math.hypot(start_x, start_y)
        else:
            fraction = max(0.0, min(1.0, -(start_x * delta_x + start_y * delta_y) / squared_length))
            distance = math.hypot(start_x + fraction * delta_x, start_y + fraction * delta_y)
        best = min(best, distance)
    return best


def polyline_length_m(polyline: tuple[GeoPoint, ...]) -> float:
    if not polyline:
        raise ValueError("Cannot measure an empty polyline")
    return sum(haversine_m(left, right) for left, right in zip(polyline, polyline[1:]))
