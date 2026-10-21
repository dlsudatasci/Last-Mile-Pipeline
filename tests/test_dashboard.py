"""Local dashboard HTTP behavior without external map or API requests."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import HTTPServer
from urllib.request import urlopen
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from graph_test_data import build_test_graph_data
from stgat_lstm.routing.dashboard import DashboardApplication, handler_for


class DashboardTests(unittest.TestCase):
    def test_page_network_routes_and_errors(self):
        application = Mock()
        application.network = {"nodes": [], "edges": [], "bounds": [0, 0, 10, 10]}
        application.route.return_value = {"recommended_route": {"edge_ids": ["a"]}}
        server = HTTPServer(("127.0.0.1", 0), handler_for(application))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        address = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(address, timeout=5) as response:
                self.assertIn(b"Generate route", response.read())
            with urlopen(address + "/network", timeout=5) as response:
                self.assertEqual(json.load(response), application.network)
            with urlopen(address + "/route?origin=1&destination=4", timeout=5) as response:
                self.assertEqual(json.load(response)["recommended_route"]["edge_ids"], ["a"])
            application.route.assert_called_once_with("1", "4")
            with self.assertRaises(HTTPError) as caught:
                urlopen(address + "/route?origin=1", timeout=5)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
            application.route.side_effect = ValueError("No routable path")
            with self.assertRaises(HTTPError) as caught:
                urlopen(address + "/route?origin=4&destination=1", timeout=5)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        finally:
            server.shutdown()
            worker.join(timeout=5)
            server.server_close()

    def test_live_route_collects_then_uses_archive(self):
        application = DashboardApplication.__new__(DashboardApplication)
        application.data = build_test_graph_data()
        application.model = Mock()
        application.policy = {"steps": 6, "interval_s": 300, "max_age_s": 900}
        application.live_traffic = True
        application.token_env = "TEST_MAPBOX_TOKEN"
        with tempfile.TemporaryDirectory() as directory:
            application.traffic_archive = Path(directory)
            archive = Mock()
            archive.sequence.return_value = ((Mock(),), {"observed_edges_per_step": [1]})
            with patch("stgat_lstm.routing.dashboard.collect_live_route_observation",
                       return_value=(Path(directory) / "traffic_live.json", {})) as collect, patch(
                       "stgat_lstm.routing.dashboard.TrafficArchive.from_directory", return_value=archive), patch(
                       "stgat_lstm.routing.dashboard.route_request", return_value={"recommended_route": {}}):
                result = application.route("1", "4")
            collect.assert_called_once()
            self.assertEqual(result["traffic"]["status"], "live_mapbox")
            self.assertEqual(result["traffic"]["live_observation_file"], "traffic_live.json")

    def test_gps_route_uses_fixed_traffic_cutoff(self):
        application = DashboardApplication.__new__(DashboardApplication)
        application.data = build_test_graph_data()
        application.model = Mock()
        application.policy = {"steps": 6, "interval_s": 300, "profile_bin_s": 300}
        application.traffic_archive = None
        application.gps_archive = Mock()
        application.gps_archive.profile_sequence.return_value = (
            (Mock(),), {"observed_edges_per_step": [1]}
        )
        application.live_traffic = False
        application.max_detour_ratio = 1.30
        application.traffic_cutoff_ms = 1_767_571_217_000
        with patch("stgat_lstm.routing.dashboard.route_request", return_value={"recommended_route": {}}):
            result = application.route("1", "4")
        application.gps_archive.profile_sequence.assert_called_once_with(
            1_767_571_217_000,
            steps=6,
            interval_s=300,
            profile_bin_s=300,
        )
        self.assertEqual(result["traffic"]["status"], "historical_rider_gps")
