"""Tests for rider-GPS-derived temporal road-speed features."""

import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pyproj import Transformer

from graph_test_data import build_test_graph_data
from stgat_lstm.gps_traffic import GPSSpeedRecord, GPSTrafficArchive


class GPSTrafficTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = build_test_graph_data()

    def test_profile_uses_only_prior_nonexcluded_probe_records(self) -> None:
        cutoff = int(datetime(2026, 7, 27, 8, 10, tzinfo=timezone.utc).timestamp() * 1000)
        records = [
            GPSSpeedRecord(cutoff - 7 * 86_400_000 - 10_000, "1|2|0", 5.0, "rider-a", "ride-a"),
            GPSSpeedRecord(cutoff - 7 * 86_400_000 - 9_000, "1|2|0", 15.0, "held-out", "ride-b"),
            GPSSpeedRecord(cutoff + 1_000, "1|2|0", 1.0, "rider-c", "future"),
        ]
        archive = GPSTrafficArchive(self.data, records)
        frames, report = archive.profile_sequence(
            cutoff,
            steps=1,
            profile_bin_s=60,
            exclude_rider_groups={"held-out"},
        )
        speed_mask = self.data.schema.edge_dynamic.index("speed_observed")
        edge = self.data.edge_ids.index("1|2|0")
        self.assertEqual(float(frames[0][edge, speed_mask]), 1.0)
        self.assertEqual(report["observed_edges_per_step"], [1])
        self.assertEqual(report["source_riders"], 1)

    def test_export_builder_confirms_and_matches_two_second_segments(self) -> None:
        inverse = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        coordinates = [inverse.transform(x, 0.0) for x in (10.0, 30.0, 50.0)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "rides_clean.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=["id", "userId"])
                writer.writeheader()
                writer.writerow({"id": "ride-1", "userId": "rider-1"})
            with (root / "map_points_clean.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target, fieldnames=["rideId", "timestamp", "latitude", "longitude"]
                )
                writer.writeheader()
                for index, (longitude, latitude) in enumerate(coordinates):
                    writer.writerow({
                        "rideId": "ride-1",
                        "timestamp": 1_000_000 + index * 2_000,
                        "latitude": latitude,
                        "longitude": longitude,
                    })
            archive = GPSTrafficArchive.from_exports(self.data, root)
        self.assertEqual(archive.summary["median_sampling_interval_s"], 2.0)
        self.assertGreaterEqual(archive.summary["speed_records"], 1)
        self.assertTrue(all(record.edge_id == "1|2|0" for record in archive.records))


if __name__ == "__main__":
    unittest.main()
