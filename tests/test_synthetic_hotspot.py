"""Tests for the separated controlled hotspot experiment."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from graph_test_data import build_test_graph_data
from stgat_lstm.preprocessing.graph_data import load_approved_decisions
from stgat_lstm.generation.synthetic_hotspot import CSV_FIELDS, generate_hotspot_experiment


def path(edges, length, names):
    return {
        "edges": [{"u": u, "v": v, "key": key} for u, v, key in edges],
        "length_m": length,
        "road_names": names,
    }


class SyntheticHotspotTests(unittest.TestCase):
    def test_generator_writes_app_shaped_data_and_trainable_ground_truth(self) -> None:
        data = build_test_graph_data()
        deviation = {
            "status": "approved", "example_key": "deviation", "ride_group": "ride",
            "rider_group": "rider", "decision_timestamp_ms": 1,
            "road_choice": {"source_node_id": "old", "chosen_edge_id": "old|edge|0"},
            "routing_context": {"origin_node_id": "old", "destination_node_id": "old-destination"},
            "has_route_pair": False,
            "recorded_navigation": {"deviatedEdge": "Old source instruction"},
            "target": {"label": "deviated", "value": 1},
            "common_od": {"origin_node_id": "2", "destination_node_id": "4"},
            "suggested_path": path((("2", "4", "0"),), 100.0, ["Main Road"]),
            "observed_path_label_evidence": path(
                (("2", "3", "0"), ("3", "4", "0")), 241.4, ["Side Road", "Return Road"]
            ),
        }
        followed = {
            "status": "approved", "example_key": "follow", "ride_group": "ride2",
            "rider_group": "rider2", "decision_timestamp_ms": 2,
            "target": {"label": "followed", "value": 0},
            "common_od": {"origin_node_id": "1", "destination_node_id": "4"},
            "suggested_path": path(
                (("1", "2", "0"), ("2", "4", "0")), 200.0, ["Taft Avenue", "Main Road"]
            ),
            "observed_path_label_evidence": path(
                (("1", "2", "0"), ("2", "4", "0")), 200.0, ["Taft Avenue", "Main Road"]
            ),
        }
        document = {
            "status": "approved_for_training", "prediction_target": "road rejection",
            "decision_point": "branch", "graph": {"crs": "EPSG:32651"},
            "examples": [deviation, followed],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = generate_hotspot_experiment(
                data, document, output, riders=6,
                hotspot_rides_per_rider=1, control_rides_per_rider=1,
            )
            self.assertTrue(manifest["synthetic"])
            self.assertEqual(manifest["summary"]["source_rides"], 24)
            self.assertEqual(manifest["summary"]["training_examples"], 12)
            self.assertTrue(manifest["summary"]["all_decision_rides_have_ground_truth_labels"])
            self.assertEqual(manifest["summary"]["traffic_probe_rides"], 12)
            regimes = manifest["experiment"]["traffic_regimes"]
            self.assertEqual(regimes["deviation_history_speed_mps"][-1], regimes["follow_history_speed_mps"][-1])
            self.assertTrue((output / "traffic_probes" / "rides_clean.csv").exists())
            self.assertTrue((output / "traffic_probes" / "map_points_clean.csv").exists())
            for name, fields in CSV_FIELDS.items():
                with (output / name).open(encoding="utf-8", newline="") as source:
                    self.assertEqual(tuple(csv.DictReader(source).fieldnames or ()), fields)
            artifact = output / "approved_training_examples.json"
            self.assertEqual(len(load_approved_decisions(artifact, data)), 12)
            document = json.loads(artifact.read_text())
            self.assertTrue(document["synthetic"])
            self.assertTrue(all("road_choice" not in example and "routing_context" not in example
                                and "recorded_navigation" not in example
                                for example in document["examples"]))
            self.assertEqual(
                {example["quality"]["synthetic_split"] for example in document["examples"]},
                {"training", "validation", "test"},
            )


if __name__ == "__main__":
    unittest.main()
