"""Tests for the local WGS84 and UTM coordinate conversion."""

import unittest

from stgat_lstm.preprocessing.coordinates import Transformer, is_projected_crs


class CoordinateTests(unittest.TestCase):
    def test_manila_coordinate_round_trip(self) -> None:
        forward = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
        inverse = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        longitude, latitude = 120.9940, 14.5640
        x, y = forward.transform(longitude, latitude)
        restored_longitude, restored_latitude = inverse.transform(x, y)
        self.assertAlmostEqual(restored_longitude, longitude, places=7)
        self.assertAlmostEqual(restored_latitude, latitude, places=7)

    def test_supported_crs_validation(self) -> None:
        self.assertTrue(is_projected_crs("EPSG:32651"))
        self.assertFalse(is_projected_crs("EPSG:4326"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


if __name__ == "__main__":
    unittest.main()
