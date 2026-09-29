"""Local road-network dashboard for origin/destination preference routing.

Run python -m stgat_lstm dashboard and open the printed localhost URL.
Uses the supplied OSM graph and can request server-side Mapbox traffic when enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from .mapbox_traffic import TrafficArchive, collect_live_route_observation
from .graph_data import build_real_graph_data, _edge_geometry
from .predict_route import load_decision_checkpoint, route_request


PROJECT = Path(__file__).resolve().parents[1]


class DashboardApplication:
    def __init__(self, checkpoint: Path, graphml: Path, node_features: Path,
                 traffic_archive: Path | None = None, *, live_traffic: bool = False,
                 token_env: str = "MAPBOX_ACCESS_TOKEN"):
        self.data = build_real_graph_data(graphml, node_features)
        self.model = load_decision_checkpoint(checkpoint, self.data)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        provenance = payload.get("provenance", {})
        for field, path in (("graphml_sha256", graphml), ("node_features_sha256", node_features)):
            recorded = provenance.get(field)
            if recorded is not None and recorded != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Graph/features differ from checkpoint provenance; retrain before using this dashboard")
        self.policy = provenance.get("traffic_policy", {"steps": 6, "interval_s": 300, "max_age_s": 900})
        self.traffic_archive = traffic_archive
        self.live_traffic = live_traffic
        self.token_env = token_env
        self.network = self._network()

    def _network(self) -> dict:
        nodes = [{"id": node_id, "x": float(self.data.graph.nodes[node_id]["x"]),
                  "y": -float(self.data.graph.nodes[node_id]["y"])} for node_id in self.data.node_ids]
        edges = []
        restricted = self.data.schema.edge_static.index("access_restricted")
        for row, edge_id in enumerate(self.data.edge_ids):
            u, v, key = edge_id.split("|", 2)
            raw = next(data for candidate_key, data in self.data.graph[u][v].items() if str(candidate_key) == key)
            geometry = _edge_geometry(self.data.graph, u, v, raw)
            edges.append({"id": edge_id, "points": [[x, -y] for x, y in geometry.coords],
                          "restricted": bool(self.data.edge_static[row, restricted] == 1)})
        xs, ys = [node["x"] for node in nodes], [node["y"] for node in nodes]
        return {"nodes": nodes, "edges": edges,
                "bounds": [min(xs) - 100, min(ys) - 100, max(xs) - min(xs) + 200, max(ys) - min(ys) + 200]}

    def route(self, origin: str, destination: str) -> dict:
        temporal = None
        traffic = {"status": "unknown", "observed_edges_per_step": []}
        live_observation = None
        if self.live_traffic:
            if self.traffic_archive is None:
                raise ValueError("Live traffic requires a traffic archive directory")
            path, _ = collect_live_route_observation(
                self.data, origin, destination, self.traffic_archive, token_env=self.token_env
            )
            live_observation = path.name
        if self.traffic_archive is not None:
            archive = TrafficArchive.from_directory(self.data, self.traffic_archive)
            temporal, traffic = archive.sequence(int(time.time() * 1000), **self.policy)
            traffic["status"] = "live_mapbox" if live_observation else "archived_mapbox"
            traffic["live_observation_file"] = live_observation
        result = route_request(self.model, self.data, origin, destination, temporal)
        result["traffic"] = traffic
        return result


def handler_for(application: DashboardApplication):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            request = urlparse(self.path)
            if request.path == "/":
                self._send(200, Path(__file__).with_name("dashboard.html").read_bytes(), "text/html; charset=utf-8")
                return
            if request.path == "/network":
                self._send(200, json.dumps(application.network).encode(), "application/json")
                return
            if request.path != "/route":
                self._send(404, b'{"error":"Not found"}', "application/json")
                return
            query = parse_qs(request.query)
            try:
                origin, destination = query.get("origin", []), query.get("destination", [])
                if len(origin) != 1 or len(destination) != 1:
                    raise ValueError("Choose one start and one destination")
                result = application.route(origin[0], destination[0])
            except (ValueError, RuntimeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            self._send(200, json.dumps(result).encode(), "application/json")
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "outputs/gatv2_lstm_checkpoint.pt")
    parser.add_argument("--graphml", type=Path, default=PROJECT / "data/real/osm/metro_manila_processed.graphml")
    parser.add_argument("--node-features", type=Path, default=PROJECT / "data/real/osm/road_node_features.csv")
    parser.add_argument("--traffic-archive", type=Path)
    parser.add_argument("--live-traffic", action="store_true",
                        help="request current Mapbox traffic for every route and append it to the archive")
    parser.add_argument("--token-env", default="MAPBOX_ACCESS_TOKEN")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    torch.set_num_threads(1)
    traffic_archive = args.traffic_archive
    if args.live_traffic and traffic_archive is None:
        traffic_archive = PROJECT / "outputs/traffic_archive"
    app = DashboardApplication(
        args.checkpoint,
        args.graphml,
        args.node_features,
        traffic_archive,
        live_traffic=args.live_traffic,
        token_env=args.token_env,
    )
    with HTTPServer(("127.0.0.1", args.port), handler_for(app)) as server:
        print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Dashboard stopped.")


if __name__ == "__main__":
    main()
