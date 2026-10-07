"""Create a controlled, structure-matched Taft preference experiment.

The generated records are synthetic and must not be merged with the official
rider dataset.  They use paths from the approved artifact and geometry from the
same OSM graph, so they exercise the production data loader, GPS traffic
builder, GATv2-LSTM model, losses, and evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyproj import CRS, Transformer
from shapely import from_wkt
from shapely.geometry import LineString
from shapely.ops import transform

from .audit_road_network import RoadEdgeIndex, _is_taft_edge
from .geometry import GeoPoint
from .graph_data import RealGraphData, build_real_graph_data


MANILA = timezone(timedelta(hours=8), name="Asia/Manila")
CSV_FIELDS = {
    "rides_clean.csv": (
        "id", "userId", "rideName", "startTime", "endTime", "duration", "distance",
        "suggestedRouteDistanceM", "suggestedRouteDurationSec", "averageSpeed", "maxSpeed",
        "elevationGain", "deviationCount", "createdAt",
    ),
    "map_points_clean.csv": ("rideId", "pointIndex", "latitude", "longitude", "timestamp", "elevation"),
    "generated_routes_clean.csv": (
        "routeId", "rideId", "type", "routePoints", "sequence", "generatedAt",
        "remainingTravelTimeOriginal", "remainingTravelTimeNew",
        "remainingDistanceOriginal", "remainingDistanceNew",
    ),
    "deviations_clean.csv": (
        "deviationId", "routeId", "rideId", "userId", "isFaster", "dateTime", "gpsLocation",
        "originalRouteEdge", "deviatedEdge", "streetName", "generatedInstruction",
        "deviationInstruction", "points", "timestamp", "createdAt",
    ),
    "deviationResponses_clean.csv": (
        "responseId", "deviationId", "rideId", "primaryReason", "primaryReasonOther",
        "trafficSeverity", "rushHourCause", "chooseDuringNonRush", "blockageReason",
        "blockageReasonOther", "personalStopReason", "personalStopOther", "stopDuration",
        "deviateAgain", "avoidRoadFrequency", "language", "submittedAt", "createdAt",
    ),
    "postTripQuestionnaire_clean.csv": (
        "rideId", "arrival", "etaRating", "stressRating", "language", "submittedAt",
    ),
}


def _pseudonym(namespace: str, value: str) -> str:
    return hashlib.sha256(f"stgat-lstm:{namespace}:{value}".encode()).hexdigest()[:16]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edge_data(graph, edge: dict) -> dict:
    u, v, key = str(edge["u"]), str(edge["v"]), str(edge["key"])
    choices = graph.get_edge_data(u, v)
    if not choices:
        raise ValueError(f"Template edge {u}|{v}|{key} is absent from the graph")
    if key in choices:
        return choices[key]
    for candidate_key, data in choices.items():
        if str(candidate_key) == key:
            return data
    raise ValueError(f"Template edge {u}|{v}|{key} is absent from the graph")


def _projected_path(graph, path: dict) -> LineString:
    pieces = []
    for edge in path["edges"]:
        u, v = str(edge["u"]), str(edge["v"])
        data = _edge_data(graph, edge)
        raw = data.get("geometry")
        geometry = from_wkt(raw) if isinstance(raw, str) else raw
        if geometry is None:
            left, right = graph.nodes[u], graph.nodes[v]
            geometry = LineString(((float(left["x"]), float(left["y"])),
                                   (float(right["x"]), float(right["y"]))))
        coordinates = list(geometry.coords)
        start = graph.nodes[u]
        start_xy = (float(start["x"]), float(start["y"]))
        if math.dist(coordinates[-1], start_xy) < math.dist(coordinates[0], start_xy):
            coordinates.reverse()
        if pieces and coordinates[0] == pieces[-1][-1]:
            coordinates = coordinates[1:]
        if coordinates:
            pieces.append(coordinates)
    flattened = [point for piece in pieces for point in piece]
    if len(flattened) < 2:
        raise ValueError("Template path has no usable geometry")
    return LineString(flattened)


def _wgs84_line(graph, path: dict) -> LineString:
    crs = graph.graph.get("crs")
    if not crs:
        raise ValueError("Graph CRS is required for synthetic GPS generation")
    transformer = Transformer.from_crs(CRS.from_user_input(crs), "EPSG:4326", always_xy=True)
    return transform(transformer.transform, _projected_path(graph, path))


def _route_points(line: LineString, limit: int = 50) -> list[dict[str, float]]:
    count = min(limit, max(2, math.ceil(line.length / 0.00012)))
    return [
        {"lat": round(line.interpolate(index / (count - 1), normalized=True).y, 7),
         "lng": round(line.interpolate(index / (count - 1), normalized=True).x, 7)}
        for index in range(count)
    ]


def _gps_rows(ride_id: str, line: LineString, start_ms: int, duration_s: int) -> list[dict]:
    rows = []
    for point_index, elapsed_s in enumerate(range(0, duration_s + 1, 2)):
        fraction = min(elapsed_s / max(duration_s, 1), 1.0)
        point = line.interpolate(fraction, normalized=True)
        rows.append({
            "rideId": ride_id,
            "pointIndex": point_index,
            "latitude": round(point.y, 7),
            "longitude": round(point.x, 7),
            "timestamp": start_ms + elapsed_s * 1000,
            "elevation": 0.0,
        })
    return rows


def _write_csv(output_dir: Path, name: str, rows: list[dict]) -> None:
    with (output_dir / name).open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS[name], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _approved_examples(document: dict, label: int) -> list[dict]:
    return [
        example for example in document.get("examples", [])
        if example.get("status") == "approved" and int(example.get("target", {}).get("value", -1)) == label
    ]


def _edge_reference(u: str, v: str, key: str) -> dict[str, str]:
    return {"u": str(u), "v": str(v), "key": str(key)}


def _path_from_edges(graph, edges: list[dict]) -> dict:
    length_m = 0.0
    road_names: list[str] = []
    for edge in edges:
        data = _edge_data(graph, edge)
        length_m += float(data.get("length", 0.0))
        raw_name = data.get("name")
        names = raw_name if isinstance(raw_name, list) else [raw_name]
        for name in names:
            if name and str(name) not in road_names:
                road_names.append(str(name))
    return {"edges": deepcopy(edges), "length_m": length_m, "road_names": road_names}


def _hotspot_raw_paths(graph, hotspot: dict) -> tuple[dict, dict, float]:
    """Add shared directed edges around the known divergence/rejoin pair."""
    observed = hotspot["observed_path_label_evidence"]
    suggested = hotspot["suggested_path"]
    origin = str(observed["edges"][0]["u"])
    destination = str(observed["edges"][-1]["v"])

    first = observed["edges"][0]
    reverse_options = graph.get_edge_data(str(first["v"]), origin) or {}
    if reverse_options:
        before_key, _ = min(reverse_options.items(), key=lambda item: float(item[1].get("length", 0.0)))
        before = _edge_reference(str(first["v"]), origin, str(before_key))
    else:
        incoming = list(graph.in_edges(origin, keys=True, data=True))
        if not incoming:
            return deepcopy(observed), deepcopy(suggested), 0.0
        u, v, key, _ = min(incoming, key=lambda item: float(item[3].get("length", 0.0)))
        before = _edge_reference(str(u), str(v), str(key))

    internal_nodes = {
        str(edge[node])
        for path in (observed, suggested)
        for edge in path["edges"]
        for node in ("u", "v")
    }
    outgoing = list(graph.out_edges(destination, keys=True, data=True))
    preferred_outgoing = [item for item in outgoing if str(item[1]) not in internal_nodes]
    choices = preferred_outgoing or outgoing
    if not choices:
        return deepcopy(observed), deepcopy(suggested), 0.0
    u, v, key, _ = min(choices, key=lambda item: float(item[3].get("length", 0.0)))
    after = _edge_reference(str(u), str(v), str(key))

    observed_full = _path_from_edges(graph, [before, *observed["edges"], after])
    suggested_full = _path_from_edges(graph, [before, *suggested["edges"], after])
    before_fraction = float(_edge_data(graph, before).get("length", 0.0)) / max(
        float(observed_full["length_m"]), 1.0
    )
    return observed_full, suggested_full, before_fraction


def generate_hotspot_experiment(
    graph_data: RealGraphData,
    approved_document: dict,
    output_dir: Path,
    *,
    riders: int = 8,
    hotspot_rides_per_rider: int = 6,
    control_rides_per_rider: int = 6,
) -> dict:
    """Write raw app-shaped CSV files and a directly trainable ground-truth artifact."""
    if riders < 2 or hotspot_rides_per_rider < 1 or control_rides_per_rider < 1:
        raise ValueError("Use at least two riders and both hotspot and control rides")
    deviation_templates = _approved_examples(approved_document, 1)
    follow_templates = _approved_examples(approved_document, 0)
    if not deviation_templates or not follow_templates:
        raise ValueError("Template artifact must contain approved examples from both classes")
    graph = graph_data.graph
    taft_index = RoadEdgeIndex(graph, _is_taft_edge)
    hotspot_paths = []
    for hotspot in deviation_templates:
        raw_observed, raw_suggested, event_fraction = _hotspot_raw_paths(graph, hotspot)
        observed_line = _wgs84_line(graph, raw_observed)
        event_point = observed_line.interpolate(event_fraction, normalized=True)
        if taft_index.nearest(GeoPoint(event_point.y, event_point.x)).distance_m <= 500.0:
            hotspot_paths.append((hotspot, raw_observed, raw_suggested, event_fraction, observed_line))
    if not hotspot_paths:
        raise ValueError("No approved deviation template lies within the 500 m Taft corridor")
    hotspots = [item[0] for item in hotspot_paths]
    controls = follow_templates
    path_payloads: dict[str, dict] = {}
    raw_path_payloads: dict[str, dict] = {}
    lines: dict[str, LineString] = {}
    hotspot_event_fractions: dict[int, float] = {}
    for index, (hotspot, raw_observed, raw_suggested, event_fraction, observed_line) in enumerate(hotspot_paths):
        observed_key, suggested_key = f"hotspot_{index}_observed", f"hotspot_{index}_suggested"
        path_payloads[observed_key] = hotspot["observed_path_label_evidence"]
        path_payloads[suggested_key] = hotspot["suggested_path"]
        raw_path_payloads[observed_key] = raw_observed
        raw_path_payloads[suggested_key] = raw_suggested
        lines[observed_key] = observed_line
        lines[suggested_key] = _wgs84_line(graph, raw_suggested)
        hotspot_event_fractions[index] = event_fraction
    for index, control in enumerate(controls):
        key = f"control_{index}"
        path_payloads[key] = control["suggested_path"]
        raw_path_payloads[key] = control["suggested_path"]
        lines[key] = _wgs84_line(graph, control["suggested_path"])
    rows = {name: [] for name in CSV_FIELDS}
    examples = []
    base = datetime(2026, 1, 5, 8, 0, tzinfo=MANILA)
    per_rider = hotspot_rides_per_rider + control_rides_per_rider

    def add_raw_ride(ride_id: str, user_id: str, start: datetime, observed_key: str,
                     suggested_key: str, deviated: bool, *, hotspot_index: int = 0,
                     speed_mps: float | None = None,
                     record_route: bool = True) -> tuple[int, int, int]:
        observed_line, suggested_line = lines[observed_key], lines[suggested_key]
        length_m = float(raw_path_payloads[observed_key]["length_m"])
        suggested_length_m = float(raw_path_payloads[suggested_key]["length_m"])
        if speed_mps is None:
            speed_mps = 10.0 + (int(hashlib.sha256(ride_id.encode()).hexdigest()[:2], 16) % 8) * 0.2
        duration_s = max(90, round(length_m / speed_mps))
        start_ms = int(start.timestamp() * 1000)
        end_ms = start_ms + duration_s * 1000
        decision_offset_s = (
            min(duration_s - 2, max(2, round(duration_s * hotspot_event_fractions[hotspot_index]) + 2))
            if deviated else 0
        )
        decision_ms = start_ms + decision_offset_s * 1000
        initial_route_id = f"route-initial-{ride_id}"
        rows["rides_clean.csv"].append({
            "id": ride_id, "userId": user_id, "rideName": "Synthetic Taft experiment",
            "startTime": start_ms, "endTime": end_ms, "duration": duration_s,
            "distance": round(length_m, 3),
            "suggestedRouteDistanceM": round(suggested_length_m, 3),
            "suggestedRouteDurationSec": duration_s, "averageSpeed": round(length_m / duration_s, 3),
            "maxSpeed": round(speed_mps, 3), "elevationGain": 0.0,
            "deviationCount": int(deviated), "createdAt": end_ms,
        })
        rows["map_points_clean.csv"].extend(_gps_rows(ride_id, observed_line, start_ms, duration_s))
        if record_route:
            rows["generated_routes_clean.csv"].append({
                "routeId": initial_route_id, "rideId": ride_id, "type": "Initial Route",
                "routePoints": repr(_route_points(suggested_line)), "sequence": 1,
                "generatedAt": start_ms - 10_000, "remainingTravelTimeOriginal": duration_s,
                "remainingTravelTimeNew": "", "remainingDistanceOriginal": round(suggested_length_m, 3),
                "remainingDistanceNew": "",
            })
        rows["postTripQuestionnaire_clean.csv"].append({
            "rideId": ride_id, "arrival": "On time", "etaRating": 4, "stressRating": 2,
            "language": "en", "submittedAt": end_ms + 5_000,
        })
        if deviated:
            hotspot = hotspots[hotspot_index]
            deviation_id = f"dev-{ride_id}"
            regenerated_route_id = f"route-regenerated-{ride_id}"
            rows["generated_routes_clean.csv"].append({
                "routeId": regenerated_route_id, "rideId": ride_id, "type": "Regenerated Route",
                "routePoints": repr(_route_points(observed_line)), "sequence": 2,
                "generatedAt": decision_ms, "remainingTravelTimeOriginal": duration_s,
                "remainingTravelTimeNew": duration_s, "remainingDistanceOriginal": round(suggested_length_m, 3),
                "remainingDistanceNew": round(length_m, 3),
            })
            event_fraction = min(max((decision_ms - start_ms) / max(end_ms - start_ms, 1), 0.0), 1.0)
            event_point = observed_line.interpolate(event_fraction, normalized=True)
            rows["deviations_clean.csv"].append({
                "deviationId": deviation_id, "routeId": regenerated_route_id, "rideId": ride_id,
                "userId": user_id, "isFaster": "True",
                "dateTime": datetime.fromtimestamp(decision_ms / 1000, tz=timezone.utc).isoformat(),
                "gpsLocation": f"{event_point.y},{event_point.x}",
                "originalRouteEdge": "Synthetic suggested road at the Taft hotspot",
                "deviatedEdge": "Synthetic familiar-road alternative",
                "streetName": "; ".join(hotspot["suggested_path"].get("road_names", [])),
                "generatedInstruction": "Continue on the suggested road",
                "deviationInstruction": "Take the familiar alternative",
                "points": repr([{"longitude": event_point.x, "latitude": event_point.y}]),
                "timestamp": decision_ms, "createdAt": end_ms,
            })
            rows["deviationResponses_clean.csv"].append({
                "responseId": f"response-{ride_id}", "deviationId": deviation_id, "rideId": ride_id,
                "primaryReason": "Shortcut/Faster Route/Personal Preference/Familiar Road",
                "primaryReasonOther": "", "trafficSeverity": "", "rushHourCause": "",
                "chooseDuringNonRush": "", "blockageReason": "", "blockageReasonOther": "",
                "personalStopReason": "", "personalStopOther": "", "stopDuration": "",
                "deviateAgain": "Often", "avoidRoadFrequency": "Often", "language": "en",
                "submittedAt": end_ms + 2_000, "createdAt": end_ms + 2_000,
            })
        return start_ms, decision_ms, end_ms

    def add_example(ride_id: str, user_id: str, decision_ms: int, template: dict,
                    observed_key: str, suggested_key: str, label: int,
                    basis: str) -> None:
        example = deepcopy(template)
        if label == 0:
            observed_path = deepcopy(path_payloads[observed_key])
            suggested_path = deepcopy(path_payloads[suggested_key])
            first_edge = suggested_path["edges"][0]
            last_edge = suggested_path["edges"][-1]
            example["common_od"] = {
                "origin_node_id": str(first_edge["u"]),
                "destination_node_id": str(last_edge["v"]),
            }
            example["suggested_path"] = suggested_path
            example["observed_path_label_evidence"] = observed_path
        example.update({
            "status": "approved",
            "example_key": _pseudonym("decision", ride_id),
            "ride_group": _pseudonym("ride", ride_id),
            "rider_group": _pseudonym("rider", user_id),
            "decision_timestamp_ms": decision_ms,
            "review_note": "controlled synthetic ground truth; no manual review required",
        })
        example["target"] = {
            "label": "deviated" if label else "followed", "value": label, "basis": basis,
        }
        example["quality"] = {
            "synthetic_ground_truth": True,
            "raw_ride_has_matching_ground_truth_example": True,
        }
        example["survey_evidence"] = ({
            "primary_reason": "Shortcut/Faster Route/Personal Preference/Familiar Road",
            "traffic_severity": None, "deviate_again": "Often",
        } if label else {
            "reported_deviation_count": 0, "intentional_deviation_count": 0,
            "missing_response_count": 0, "primary_reasons": [],
        })
        examples.append(example)

    # Independent probe rides create a reproducible historical traffic signal.
    # Morning sessions are slow; late-morning sessions are free flowing.
    for session_index in range(per_rider):
        congested = session_index < hotspot_rides_per_rider
        session_start = base + timedelta(
            weeks=session_index,
            hours=0 if congested else 3,
        )
        probe_keys = (
            [f"hotspot_{index}_suggested" for index in range(len(hotspots))]
            if congested else [f"control_{index}" for index in range(len(controls))]
        )
        for probe_index, suggested_key in enumerate(probe_keys):
            ride_id = f"synthetic-traffic-{session_index:02d}-{probe_index:02d}"
            user_id = f"synthetic-traffic-rider-{probe_index:02d}"
            add_raw_ride(
                ride_id,
                user_id,
                session_start - timedelta(minutes=20 if congested else 10),
                suggested_key,
                suggested_key,
                False,
                hotspot_index=0,
                speed_mps=3.0 if congested else 11.0,
                record_route=False,
            )

    for rider_index in range(riders):
        user_id = f"synthetic-rider-{rider_index:02d}"
        schedule = [1] * hotspot_rides_per_rider + [0] * control_rides_per_rider
        for ride_index, label in enumerate(schedule):
            ride_id = f"synthetic-{rider_index:02d}-{ride_index:02d}"
            hotspot_index = (ride_index + rider_index) % len(hotspots)
            hotspot = hotspots[hotspot_index]
            follow_index = ride_index - hotspot_rides_per_rider
            use_independent_control = not label
            if use_independent_control:
                control_index = (follow_index + rider_index) % len(controls)
                template = controls[control_index]
                observed_key = suggested_key = f"control_{control_index}"
            else:
                template = hotspot
                observed_key = (
                    f"hotspot_{hotspot_index}_observed"
                    if label else f"hotspot_{hotspot_index}_suggested"
                )
                suggested_key = f"hotspot_{hotspot_index}_suggested"
            start = base + timedelta(
                weeks=ride_index,
                hours=0 if label else 3,
                seconds=rider_index * 3,
            )
            _, decision_ms, _ = add_raw_ride(
                ride_id, user_id, start,
                observed_key,
                suggested_key,
                label == 1,
                hotspot_index=hotspot_index,
            )
            add_example(
                ride_id, user_id, decision_ms, template,
                observed_key,
                suggested_key,
                label,
                ("injected familiar-road deviation with congested history"
                 if label else "injected follow decision on the same roads with free-flow history"),
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in rows.items():
        _write_csv(output_dir, name, values)
    artifact = {
        "schema_version": 1,
        "provenance": "Controlled synthetic Taft hotspot experiment; not rider evidence",
        "status": "approved_for_training",
        "synthetic": True,
        "prediction_target": approved_document.get("prediction_target"),
        "decision_point": approved_document.get("decision_point"),
        "graph": approved_document.get("graph", {}),
        "source_sha256": {name: _sha256(output_dir / name) for name in CSV_FIELDS},
        "experiment": {
            "purpose": "capacity check across real-shaped Taft choices with controlled traffic context",
            "hotspots": [
                {
                    "origin_node_id": item["common_od"]["origin_node_id"],
                    "destination_node_id": item["common_od"]["destination_node_id"],
                    "suggested_roads": item["suggested_path"].get("road_names", []),
                    "observed_roads": item["observed_path_label_evidence"].get("road_names", []),
                }
                for item in hotspots
            ],
            "preference_pattern_riders": riders,
            "traffic_probe_riders": max(len(hotspots), len(controls)),
            "hotspot_rides_per_rider": hotspot_rides_per_rider,
            "control_rides_per_rider": control_rides_per_rider,
            "traffic_regimes": {
                "deviation_history_speed_mps": 3.0,
                "follow_history_speed_mps": 11.0,
            },
            "gps_interval_seconds": 2,
        },
        "summary": {
            "source_rides": len(rows["rides_clean.csv"]),
            "training_examples": len(examples),
            "training_riders": len({example["rider_group"] for example in examples}),
            "preference_pattern_rides": riders * per_rider,
            "traffic_probe_rides": (
                hotspot_rides_per_rider * len(hotspots)
                + control_rides_per_rider * len(controls)
            ),
            "all_decision_rides_have_ground_truth_labels": len(examples) == riders * per_rider,
            "label_counts": {
                "deviated": sum(int(example["target"]["value"]) == 1 for example in examples),
                "followed": sum(int(example["target"]["value"]) == 0 for example in examples),
            },
        },
        "examples": examples,
    }
    artifact_path = output_dir / "approved_training_examples.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "synthetic": True,
        "warning": "Use only as a controlled learning/capacity test. Do not combine its metrics with real riders.",
        "trainable_artifact": artifact_path.name,
        "files": {
            name: _sha256(output_dir / name)
            for name in (*CSV_FIELDS.keys(), artifact_path.name)
        },
        "experiment": artifact["experiment"],
        "summary": artifact["summary"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("node_features", type=Path)
    parser.add_argument("approved_decisions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_hotspot"))
    parser.add_argument("--riders", type=int, default=8)
    parser.add_argument("--hotspot-rides-per-rider", type=int, default=6)
    parser.add_argument("--control-rides-per-rider", type=int, default=6)
    args = parser.parse_args()
    graph_data = build_real_graph_data(args.graphml, args.node_features)
    approved = json.loads(args.approved_decisions.read_text(encoding="utf-8"))
    manifest = generate_hotspot_experiment(
        graph_data, approved, args.output_dir,
        riders=args.riders,
        hotspot_rides_per_rider=args.hotspot_rides_per_rider,
        control_rides_per_rider=args.control_rides_per_rider,
    )
    summary = manifest["summary"]
    print(f"Synthetic experiment saved to {args.output_dir}")
    print(
        f"Training examples: {summary['training_examples']} from {summary['training_riders']} riders | "
        f"deviated {summary['label_counts']['deviated']} | followed {summary['label_counts']['followed']}"
    )
    print("Synthetic results are a controlled capacity check, not evidence of rider generalization.")


if __name__ == "__main__":
    main()
