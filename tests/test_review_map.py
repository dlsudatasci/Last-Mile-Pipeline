"""Tests for the local candidate review page."""

from __future__ import annotations

import unittest

from stgat_lstm.review_map import render_review_html


class ReviewMapTests(unittest.TestCase):
    def test_embeds_supported_geojson_and_escapes_script_end(self) -> None:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiLineString", "coordinates": [[[120.9, 14.5], [121.0, 14.6]]]},
                    "properties": {
                        "event_key": "case-1",
                        "role": "preferred_observed",
                        "primary_reason": "</script>",
                        "length_m": 10,
                        "road_names": "Road",
                    },
                }
            ],
        }
        rendered = render_review_html(geojson)
        self.assertIn("Candidate path review", rendered)
        self.assertIn("<\\/script>", rendered)

    def test_rejects_unknown_roles(self) -> None:
        geojson = {
            "type": "FeatureCollection",
            "features": [{"properties": {"role": "raw_gps"}}],
        }
        with self.assertRaisesRegex(ValueError, "unsupported"):
            render_review_html(geojson)

    def test_accepts_offline_context_roads_without_public_tiles(self) -> None:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[120.9, 14.5], [121.0, 14.6]]},
                    "properties": {"event_key": None, "role": "context_road"},
                }
            ],
        }
        rendered = render_review_html(geojson)
        self.assertIn("context_road", rendered)
        self.assertNotIn("tile.openstreetmap.org", rendered)


if __name__ == "__main__":
    unittest.main()
