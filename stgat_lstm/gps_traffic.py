"""Build leakage-aware temporal road-speed features from rider GPS exports.

The spatial-then-temporal design follows the broad ST-GAT formulation in:
C. Zhang, J. J. Q. Yu, and Y. Liu, "Spatial-Temporal Graph Attention
Networks: A Deep Learning Approach for Traffic Forecasting," IEEE Access,
2019, DOI 10.1109/ACCESS.2019.2953888.

GPS fixes are motorcycle probe observations rather than direct measurements of
network-wide traffic. Consecutive fixes are direction-matched to OSM edges,
converted to speed ratios, and aggregated into historical time-of-day graph
snapshots. Raw coordinates and direct rider identifiers are never written by
this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import torch

from .audit_road_network import RoadEdgeIndex
from .geometry import GeoPoint, haversine_m
from .graph_data import RealGraphData, build_real_graph_data, canonical_edge_id
from .mapbox_traffic import match_traffic_segment


MANILA = timezone(timedelta(hours=8), name="Asia/Manila")


def _pseudonym(namespace: str, value: str) -> str:
    return hashlib.sha256(f"stgat-lstm:{namespace}:{value}".encode()).hexdigest()[:16]


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        if not required.issubset(fields):
            raise ValueError(f"{path.name} lacks columns: {sorted(required - fields)}")
        return list(reader)


def _circular_seconds(left: int, right: int) -> int:
    difference = abs(left - right)
    return min(difference, 86_400 - difference)


def _local_time(timestamp_ms: int) -> tuple[int, bool]:
    value = datetime.fromtimestamp(timestamp_ms / 1000, tz=MANILA)
    seconds = value.hour * 3600 + value.minute * 60 + value.second
    return seconds, value.weekday() >= 5


@dataclass(frozen=True)
class GPSSpeedRecord:
    timestamp_ms: int
    edge_id: str
    speed_mps: float
    rider_group: str
    ride_group: str


class GPSTrafficArchive:
    """Historical motorcycle probe speeds aligned to directed OSM edges."""

    def __init__(self, graph_data: RealGraphData, records: list[GPSSpeedRecord], summary: dict | None = None):
        self.graph_data = graph_data
        self.records = tuple(sorted(records, key=lambda item: item.timestamp_ms))
        self.summary = summary or {}
        self._edge_lookup = {edge_id: row for row, edge_id in enumerate(graph_data.edge_ids)}
        length = graph_data.schema.edge_static.index("length_per_100m")
        reference = graph_data.schema.edge_static.index("reference_time_minutes")
        self._reference_speed_mps = {
            edge_id: (float(graph_data.edge_static[row, length]) * 100.0)
            / max(float(graph_data.edge_static[row, reference]) * 60.0, 1e-6)
            for row, edge_id in enumerate(graph_data.edge_ids)
        }

    @classmethod
    def from_exports(
        cls,
        graph_data: RealGraphData,
        data_dir: Path,
        *,
        match_radius_m: float = 60.0,
        max_gap_s: float = 20.0,
        max_speed_mps: float = 35.0,
        stationary_distance_m: float = 3.0,
    ) -> "GPSTrafficArchive":
        if match_radius_m <= 0 or max_gap_s <= 0 or max_speed_mps <= 0 or stationary_distance_m < 0:
            raise ValueError("GPS traffic parameters are invalid")
        rides = _read_csv(data_dir / "rides_clean.csv", {"id", "userId"})
        points = _read_csv(
            data_dir / "map_points_clean.csv",
            {"rideId", "timestamp", "latitude", "longitude"},
        )
        ride_user = {row["id"]: row["userId"] for row in rides}
        if len(ride_user) != len(rides):
            raise ValueError("Duplicate ride IDs prevent GPS traffic construction")
        grouped: dict[str, list[tuple[int, GeoPoint]]] = defaultdict(list)
        invalid_points = 0
        for row in points:
            if row["rideId"] not in ride_user:
                invalid_points += 1
                continue
            try:
                grouped[row["rideId"]].append(
                    (int(row["timestamp"]), GeoPoint(float(row["latitude"]), float(row["longitude"])))
                )
            except ValueError:
                invalid_points += 1
        edge_index = RoadEdgeIndex(graph_data.graph)
        available = set(graph_data.edge_ids)
        records: list[GPSSpeedRecord] = []
        intervals: list[float] = []
        skipped = defaultdict(int)
        matched_rides: set[str] = set()
        for ride_id, fixes in grouped.items():
            fixes.sort(key=lambda item: item[0])
            unique: list[tuple[int, GeoPoint]] = []
            for fix in fixes:
                if unique and fix[0] == unique[-1][0]:
                    skipped["duplicate_timestamp"] += 1
                    continue
                unique.append(fix)
            last_edge_id: str | None = None
            for (left_time, left), (right_time, right) in zip(unique, unique[1:]):
                elapsed_s = (right_time - left_time) / 1000.0
                if elapsed_s <= 0:
                    skipped["nonincreasing_timestamp"] += 1
                    continue
                intervals.append(elapsed_s)
                if elapsed_s > max_gap_s:
                    skipped["gap_too_long"] += 1
                    last_edge_id = None
                    continue
                distance_m = haversine_m(left, right)
                speed_mps = distance_m / elapsed_s
                if not math.isfinite(speed_mps) or speed_mps > max_speed_mps:
                    skipped["implausible_speed"] += 1
                    last_edge_id = None
                    continue
                edge_id = None
                if distance_m <= stationary_distance_m and last_edge_id is not None:
                    edge_id = last_edge_id
                    speed_mps = 0.0
                elif distance_m > stationary_distance_m:
                    match = match_traffic_segment(
                        edge_index,
                        graph_data,
                        (left.longitude, left.latitude),
                        (right.longitude, right.latitude),
                        match_radius_m,
                    )
                    if match is not None:
                        candidate = canonical_edge_id(match.u, match.v, match.key)
                        if candidate in available:
                            edge_id = candidate
                if edge_id is None:
                    skipped["unmatched_segment"] += 1
                    continue
                last_edge_id = edge_id
                records.append(
                    GPSSpeedRecord(
                        timestamp_ms=right_time,
                        edge_id=edge_id,
                        speed_mps=speed_mps,
                        rider_group=_pseudonym("rider", ride_user[ride_id]),
                        ride_group=_pseudonym("ride", ride_id),
                    )
                )
                matched_rides.add(ride_id)
        summary = {
            "source": "rider_gps_motorcycle_probe_speeds",
            "rides": len(rides),
            "rides_with_matched_speed": len(matched_rides),
            "gps_points": len(points),
            "speed_records": len(records),
            "matched_edges": len({record.edge_id for record in records}),
            "invalid_or_orphan_points": invalid_points,
            "median_sampling_interval_s": round(median(intervals), 3) if intervals else None,
            "requested_sampling_interval_s": 2,
            "skipped_segments": dict(sorted(skipped.items())),
            "parameters": {
                "match_radius_m": match_radius_m,
                "max_gap_s": max_gap_s,
                "max_speed_mps": max_speed_mps,
                "stationary_distance_m": stationary_distance_m,
            },
        }
        return cls(graph_data, records, summary)

    def profile_sequence(
        self,
        cutoff_ms: int,
        *,
        steps: int = 6,
        interval_s: int = 300,
        profile_bin_s: int = 300,
        exclude_rider_groups: set[str] | None = None,
        exclude_ride_groups: set[str] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], dict]:
        """Return pre-cutoff historical time-of-day snapshots.

        A record must predate ``cutoff_ms``. Weekday observations are not mixed
        with weekend observations. Exclusions support leakage-safe rider folds.
        """
        if steps <= 0 or interval_s <= 0 or profile_bin_s <= 0 or profile_bin_s > 86_400:
            raise ValueError("GPS history settings must be positive and within one day")
        excluded_riders = exclude_rider_groups or set()
        excluded_rides = exclude_ride_groups or set()
        schema = self.graph_data.schema.edge_dynamic
        columns = {name: schema.index(name) for name in schema}
        required = {
            "congestion_normalized",
            "speed_ratio_to_reference",
            "congestion_observed",
            "speed_observed",
            "traffic_age_scaled",
        }
        if not required.issubset(columns):
            raise ValueError("Graph dynamic schema cannot represent GPS speed context")
        eligible = [
            record
            for record in self.records
            if record.timestamp_ms < cutoff_ms
            and record.rider_group not in excluded_riders
            and record.ride_group not in excluded_rides
        ]
        frames = []
        coverage = []
        samples_per_step = []
        source_rides: set[str] = set()
        source_riders: set[str] = set()
        half_bin = profile_bin_s / 2.0
        for step in range(steps):
            slot_ms = cutoff_ms - (steps - 1 - step) * interval_s * 1000
            slot_seconds, slot_weekend = _local_time(slot_ms)
            values: dict[str, list[GPSSpeedRecord]] = defaultdict(list)
            for record in eligible:
                record_seconds, record_weekend = _local_time(record.timestamp_ms)
                if record_weekend != slot_weekend:
                    continue
                if _circular_seconds(record_seconds, slot_seconds) <= half_bin:
                    values[record.edge_id].append(record)
            frame = torch.zeros((len(self.graph_data.edge_ids), len(schema)), dtype=torch.float32)
            frame[:, columns["traffic_age_scaled"]] = 1.0
            for edge_id, observations in values.items():
                row = self._edge_lookup[edge_id]
                ratios = [item.speed_mps / self._reference_speed_mps[edge_id] for item in observations]
                ratio = min(max(float(median(ratios)), 0.0), 2.0)
                frame[row, columns["speed_ratio_to_reference"]] = ratio
                frame[row, columns["congestion_normalized"]] = max(0.0, 1.0 - min(ratio, 1.0))
                frame[row, columns["speed_observed"]] = 1.0
                frame[row, columns["congestion_observed"]] = 1.0
                latest = max(item.timestamp_ms for item in observations)
                frame[row, columns["traffic_age_scaled"]] = min(
                    max((cutoff_ms - latest) / (30 * 86_400_000), 0.0), 1.0
                )
                source_rides.update(item.ride_group for item in observations)
                source_riders.update(item.rider_group for item in observations)
            frames.append(frame)
            coverage.append(len(values))
            samples_per_step.append(sum(len(items) for items in values.values()))
        return tuple(frames), {
            "source": "historical_rider_gps_speed_profile",
            "cutoff_ms": cutoff_ms,
            "steps": steps,
            "interval_seconds": interval_s,
            "profile_bin_seconds": profile_bin_s,
            "observed_edges_per_step": coverage,
            "speed_records_per_step": samples_per_step,
            "source_rides": len(source_rides),
            "source_riders": len(source_riders),
            "excluded_riders": len(excluded_riders),
            "excluded_rides": len(excluded_rides),
            "interpretation": "historical motorcycle probe-speed context, not direct network-wide traffic",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("node_features", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/gps_traffic_audit.json"))
    args = parser.parse_args()
    graph_data = build_real_graph_data(args.graphml, args.node_features)
    archive = GPSTrafficArchive.from_exports(graph_data, args.data_dir)
    report = {**archive.summary, "records_preview": [asdict(item) for item in archive.records[:5]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"GPS traffic audit saved to {args.output}")
    print(
        f"Matched {report['speed_records']} speed segments across {report['matched_edges']} roads "
        f"from {report['rides_with_matched_speed']} rides"
    )
    print(f"Median GPS interval: {report['median_sampling_interval_s']} seconds")


if __name__ == "__main__":
    main()
