"""Coordinate conversion used by the Manila road-network pipeline.

The official graph is WGS 84 / UTM zone 51N (EPSG:32651).  Keeping this
conversion local avoids loading PROJ's optional native network libraries,
which are blocked by Windows Application Control on some machines.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_E2 = _WGS84_F * (2.0 - _WGS84_F)
_EP2 = _E2 / (1.0 - _E2)
_K0 = 0.9996


def _epsg_code(value) -> int:
    text = str(value).strip().upper()
    if text.startswith("EPSG:"):
        text = text.split(":", 1)[1]
    try:
        code = int(text)
    except ValueError as error:
        raise ValueError(f"Unsupported CRS {value!r}; use an EPSG code") from error
    if code != 4326 and not (32601 <= code <= 32660 or 32701 <= code <= 32760):
        raise ValueError(f"Unsupported EPSG:{code}; only WGS84 and WGS84 UTM zones are supported")
    return code


def is_projected_crs(value) -> bool:
    """Return whether a supported CRS uses projected metre coordinates."""
    return _epsg_code(value) != 4326


def _utm_parameters(code: int) -> tuple[int, bool]:
    if 32601 <= code <= 32660:
        return code - 32600, False
    if 32701 <= code <= 32760:
        return code - 32700, True
    raise ValueError(f"EPSG:{code} is not a WGS84 UTM CRS")


def _wgs84_to_utm(longitude: float, latitude: float, code: int) -> tuple[float, float]:
    zone, southern = _utm_parameters(code)
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        raise ValueError("Invalid longitude or latitude")
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)
    n = _WGS84_A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = _EP2 * cos_lat * cos_lat
    a = cos_lat * (lon - lon0)
    e4 = _E2 * _E2
    e6 = e4 * _E2
    m = _WGS84_A * (
        (1.0 - _E2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * _E2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * math.sin(2.0 * lat)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * math.sin(4.0 * lat)
        - 35.0 * e6 / 3072.0 * math.sin(6.0 * lat)
    )
    easting = _K0 * n * (
        a
        + (1.0 - t + c) * a**3 / 6.0
        + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * _EP2) * a**5 / 120.0
    ) + 500_000.0
    northing = _K0 * (
        m
        + n
        * tan_lat
        * (
            a * a / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c * c) * a**4 / 24.0
            + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * _EP2) * a**6 / 720.0
        )
    )
    if southern:
        northing += 10_000_000.0
    return easting, northing


def _utm_to_wgs84(easting: float, northing: float, code: int) -> tuple[float, float]:
    zone, southern = _utm_parameters(code)
    x = easting - 500_000.0
    y = northing - (10_000_000.0 if southern else 0.0)
    e4 = _E2 * _E2
    e6 = e4 * _E2
    m = y / _K0
    mu = m / (_WGS84_A * (1.0 - _E2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0))
    root = math.sqrt(1.0 - _E2)
    e1 = (1.0 - root) / (1.0 + root)
    fp = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + 151.0 * e1**3 / 96.0 * math.sin(6.0 * mu)
        + 1097.0 * e1**4 / 512.0 * math.sin(8.0 * mu)
    )
    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    c1 = _EP2 * cos_fp * cos_fp
    t1 = tan_fp * tan_fp
    n1 = _WGS84_A / math.sqrt(1.0 - _E2 * sin_fp * sin_fp)
    r1 = _WGS84_A * (1.0 - _E2) / (1.0 - _E2 * sin_fp * sin_fp) ** 1.5
    d = x / (n1 * _K0)
    latitude = fp - (n1 * tan_fp / r1) * (
        d * d / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * _EP2) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1 - 252.0 * _EP2 - 3.0 * c1 * c1)
        * d**6
        / 720.0
    )
    longitude = math.radians((zone - 1) * 6 - 180 + 3) + (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * _EP2 + 24.0 * t1 * t1)
        * d**5
        / 120.0
    ) / cos_fp
    return math.degrees(longitude), math.degrees(latitude)


@dataclass(frozen=True)
class Transformer:
    """Small subset of ``pyproj.Transformer`` needed by this project."""

    source_code: int
    target_code: int

    @classmethod
    def from_crs(cls, source, target, *, always_xy: bool = True) -> "Transformer":
        if not always_xy:
            raise ValueError("This project requires longitude/easting first with always_xy=True")
        return cls(_epsg_code(source), _epsg_code(target))

    def _one(self, x: float, y: float, *, inverse: bool = False) -> tuple[float, float]:
        x, y = float(x), float(y)
        source_code = self.target_code if inverse else self.source_code
        target_code = self.source_code if inverse else self.target_code
        if source_code == target_code:
            return x, y
        if source_code == 4326:
            return _wgs84_to_utm(x, y, target_code)
        if target_code == 4326:
            return _utm_to_wgs84(x, y, source_code)
        longitude, latitude = _utm_to_wgs84(x, y, source_code)
        return _wgs84_to_utm(longitude, latitude, target_code)

    def transform(self, x, y, z=None, *, direction="FORWARD"):
        inverse = str(direction).upper().endswith("INVERSE")
        if not inverse and not str(direction).upper().endswith("FORWARD"):
            raise ValueError("direction must be FORWARD or INVERSE")
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            converted = [self._one(left, right, inverse=inverse) for left, right in zip(x, y)]
            first = [point[0] for point in converted]
            second = [point[1] for point in converted]
            return (first, second, list(z)) if z is not None else (first, second)
        first, second = self._one(x, y, inverse=inverse)
        return (first, second, z) if z is not None else (first, second)
