"""Tests for real-export geometry parsing and distance calculations."""

from __future__ import annotations

import unittest

from stgat_lstm.geometry import GeoPoint, haversine_m, parse_gps_location, parse_route_points, point_to_polyline_distance_m


class GeometryTests(unittest.TestCase):
    def test_export_coordinate_formats_are_parsed(self) -> None:
        self.assertEqual(parse_gps_location("14.55,120.99"), GeoPoint(14.55, 120.99))
        self.assertEqual(
            parse_route_points("[{'lat': 14.55, 'lng': 120.99}, {'latitude': 14.56, 'longitude': 121.0}]"),
            (GeoPoint(14.55, 120.99), GeoPoint(14.56, 121.0)),
        )

    def test_unsafe_or_invalid_route_values_are_rejected(self) -> None:
        with self.assertRaises((SyntaxError, ValueError)):
            parse_route_points("__import__('os').system('echo unsafe')")
        with self.assertRaisesRegex(ValueError, "lacks latitude"):
            parse_route_points("[{'lng': 120.99}]")
        with self.assertRaisesRegex(ValueError, "Invalid latitude"):
            parse_route_points("[{'lat': 100, 'lng': 120.99}]")

    def test_point_to_polyline_uses_segments(self) -> None:
        line = (GeoPoint(14.55, 120.99), GeoPoint(14.55, 121.0))
        on_segment = GeoPoint(14.55, 120.995)
        north_of_segment = GeoPoint(14.551, 120.995)
        self.assertLess(point_to_polyline_distance_m(on_segment, line), 0.01)
        self.assertAlmostEqual(point_to_polyline_distance_m(north_of_segment, line), 111.2, delta=0.5)

    def test_haversine_distance_has_metre_scale(self) -> None:
        distance = haversine_m(GeoPoint(0.0, 0.0), GeoPoint(0.001, 0.0))
        self.assertAlmostEqual(distance, 111.2, delta=0.2)


if __name__ == "__main__":
    unittest.main()
