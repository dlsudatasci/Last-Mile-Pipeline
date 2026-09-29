"""Tests for Mapbox feature conversion and historical traffic selection."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from graph_test_data import build_test_graph_data
from stgat_lstm.audit_road_network import EdgeMatch
from stgat_lstm.mapbox_traffic import (
    TrafficArchive,
    build_edge_snapshot,
    collect_live_route_observation,
    timestamp_ms,
)


class TrafficTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = build_test_graph_data()

    def test_mapbox_speed_and_sparse_congestion_are_separate_features(self) -> None:
        observation = {
            "requested_at": "1970-01-01T00:00:00Z",
            "route": {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[120.99, 14.55], [120.991, 14.551], [120.992, 14.552]],
                },
                "legs": [{"annotation": {
                    "speed": [5.0, 15.0],
                    "congestion_numeric": [None, 40],
                    "congestion": ["unknown", "low"],
                    "duration": [10.0, 10.0],
                }}],
            },
        }
        fake_index = Mock()
        fake_index.nearest.return_value = EdgeMatch("1", "2", "0", 2.0, "Taft Avenue")
        with patch("stgat_lstm.mapbox_traffic.RoadEdgeIndex", return_value=fake_index), patch(
                "stgat_lstm.mapbox_traffic.match_traffic_segment",
                return_value=EdgeMatch("1", "2", "0", 2.0, "Taft Avenue")):
            snapshot = build_edge_snapshot(self.data, observation, now_epoch_ms=1_800_000)

        row = self.data.edge_ids.index("1|2|0")
        columns = {name: index for index, name in enumerate(self.data.schema.edge_dynamic)}
        self.assertAlmostEqual(float(snapshot[row, columns["congestion_normalized"]]), 0.4)
        self.assertAlmostEqual(float(snapshot[row, columns["speed_ratio_to_reference"]]), 0.9, places=4)
        self.assertEqual(float(snapshot[row, columns["congestion_observed"]]), 1.0)
        self.assertEqual(float(snapshot[row, columns["speed_observed"]]), 1.0)
        self.assertAlmostEqual(float(snapshot[row, columns["traffic_age_scaled"]]), 0.5)
        unobserved = self.data.edge_ids.index("2|4|0")
        self.assertEqual(float(snapshot[unobserved, columns["speed_observed"]]), 0.0)
        self.assertEqual(float(snapshot[unobserved, columns["traffic_age_scaled"]]), 1.0)

    def test_archive_excludes_future_late_and_stale_observations(self) -> None:
        schema = self.data.schema.edge_dynamic
        speed = schema.index("speed_observed")
        ratio = schema.index("speed_ratio_to_reference")
        snapshot = self.data.build_model_input("4").temporal_edge_features[0].clone()
        snapshot[0, speed], snapshot[0, ratio] = 1.0, 0.5
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, observed, received in (("early", "00:04:00", "00:04:01"),
                                             ("late", "00:04:30", "00:11:00"),
                                             ("future", "00:11:00", "00:11:01")):
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps({"source": "mapbox_directions_driving_traffic",
                    "requested_at": f"1970-01-01T{observed}Z",
                    "received_at": f"1970-01-01T{received}Z"}), encoding="utf-8")
                paths.append(path)
            archive = TrafficArchive(self.data, paths)
            with patch("stgat_lstm.mapbox_traffic.build_edge_snapshot", return_value=snapshot) as convert:
                frames, report = archive.sequence(600_000, steps=3, interval_s=300, max_age_s=360)
                self.assertEqual(report["observed_edges_per_step"], [0, 1, 1])
                self.assertEqual(len(report["used_observation_sha256"]), 1)
                self.assertEqual(convert.call_count, 1)
                self.assertEqual(float(frames[-1][0, ratio]), 0.5)
                stale, report = archive.sequence(1_000_000, steps=1, max_age_s=10)
                self.assertEqual(report["observed_edges_per_step"], [0])
                self.assertTrue(torch.all(stale[0][:, speed] == 0))

    def test_archive_merges_partial_coverage_and_validates_timestamps(self) -> None:
        speed = self.data.schema.edge_dynamic.index("speed_observed")
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                (Path(directory) / f"{index}.json").write_text(json.dumps({
                    "source": "mapbox_directions_driving_traffic",
                    "requested_at": f"1970-01-01T00:04:0{index}Z",
                    "received_at": f"1970-01-01T00:04:0{index}Z",
                }), encoding="utf-8")
            archive = TrafficArchive.from_directory(self.data, Path(directory))
            snapshots = []
            for index in range(2):
                frame = self.data.build_model_input("4").temporal_edge_features[0].clone()
                frame[index, speed] = 1.0
                snapshots.append(frame)
            with patch("stgat_lstm.mapbox_traffic.build_edge_snapshot", side_effect=snapshots):
                frames, report = archive.sequence(300_000, steps=1)
                self.assertEqual(report["observed_edges_per_step"], [2])
                self.assertEqual(float(frames[0][0, speed]), 1.0)
                self.assertEqual(float(frames[0][1, speed]), 1.0)
        with self.assertRaisesRegex(ValueError, "timezone"):
            timestamp_ms("2026-09-26T08:00:00")
        with self.assertRaises(ValueError):
            timestamp_ms(None)

    def test_archive_age_tracks_active_fields_after_updates(self) -> None:
        columns = {name: index for index, name in enumerate(self.data.schema.edge_dynamic)}
        with tempfile.TemporaryDirectory() as directory:
            paths, snapshots = [], []
            for index, minutes in enumerate((1, 2, 3)):
                path = Path(directory) / f"{index}.json"
                path.write_text(json.dumps({
                    "source": "mapbox_directions_driving_traffic",
                    "requested_at": f"1970-01-01T00:0{minutes}:00Z",
                }), encoding="utf-8")
                paths.append(path)
                frame = self.data.build_model_input("4").temporal_edge_features[0].clone()
                frame[0, columns["speed_observed"]] = 1.0
                if index != 1:
                    frame[0, columns["congestion_observed"]] = 1.0
                snapshots.append(frame)
            with patch("stgat_lstm.mapbox_traffic.build_edge_snapshot", side_effect=snapshots):
                archive = TrafficArchive(self.data, paths)
                frames, _ = archive.sequence(120_000, steps=1)
                self.assertAlmostEqual(float(frames[0][0, columns["traffic_age_scaled"]]), 60 / 3600)
                frames, _ = archive.sequence(180_000, steps=1)
                self.assertEqual(float(frames[0][0, columns["traffic_age_scaled"]]), 0.0)

    def test_live_route_observation_uses_token_without_saving_it(self) -> None:
        payload = {
            "code": "Ok",
            "routes": [{
                "geometry": {"type": "LineString", "coordinates": [[120.99, 14.55], [120.991, 14.551]]},
                "legs": [{"annotation": {
                    "speed": [5.0], "duration": [10.0],
                    "congestion": ["moderate"], "congestion_numeric": [50],
                }}],
            }],
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"TEST_MAPBOX_TOKEN": "secret-token"}), patch(
                "stgat_lstm.mapbox_traffic.fetch_directions_traffic", return_value=payload) as fetch:
            path, document = collect_live_route_observation(
                self.data, "1", "4", Path(directory), token_env="TEST_MAPBOX_TOKEN"
            )
            self.assertTrue(path.is_file())
            self.assertEqual(document["source"], "mapbox_directions_driving_traffic")
            self.assertNotIn("secret-token", path.read_text(encoding="utf-8"))
            self.assertEqual(fetch.call_args.args[1], "secret-token")


if __name__ == "__main__":
    unittest.main()
