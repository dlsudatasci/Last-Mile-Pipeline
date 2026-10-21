"""Extract supported outgoing choices and optional common-OD preference pairs.

Local choices compare the map-matched rider's outgoing road with the active
suggestion at the same source. Optional route-ranking pairs use shared edges
before and after the deviation to establish comparable path endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
import re
import unicodedata

import networkx as nx
from shapely import from_wkt
from shapely.geometry import LineString, Point

from stgat_lstm.preprocessing.map_matching import EdgeId, MapMatchResult, TimedObservation
from stgat_lstm.preprocessing.geometry import GeoPoint
from stgat_lstm.preprocessing.audit_road_network import RoadEdgeIndex
from stgat_lstm.preprocessing.graph_data import _restricted_access, _tag_values


@dataclass(frozen=True)
class DivergencePair:
    origin_node_id: str
    destination_node_id: str
    preferred_edge_ids: tuple[EdgeId, ...]
    rejected_edge_ids: tuple[EdgeId, ...]
    shared_edge_before: EdgeId
    shared_edge_after: EdgeId


@dataclass(frozen=True)
class RoadChoiceEvidence:
    source_node_id: str
    suggested_edge_id: EdgeId
    chosen_edge_id: EdgeId
    decision_timestamp_ms: int
    first_chosen_timestamp_ms: int
    supporting_gps_points: int
    mean_local_match_distance_m: float
    observed_continuation: tuple[EdgeId, ...]
    supporting_edge_ids: tuple[EdgeId, ...]


def parse_navigation_instruction(text: str) -> dict:
    """Separate a named road and compass heading from navigation prose."""
    normalized = re.sub(r"\b(north|south)[- ](east|west)\b", r"\1\2", text.lower())
    directions = {"north": 0, "northeast": 45, "east": 90, "southeast": 135,
                  "south": 180, "southwest": 225, "west": 270, "northwest": 315}
    heading = re.search(r"\b(?:drive|head|continue)\s+(northwest|northeast|southwest|southeast|north|south|east|west)\b", normalized)
    road = re.search(r"\b(?:onto|on)\s+(.+?)(?:\.\s*(?:in\b|$)|$)", text, re.IGNORECASE)
    road_name = road.group(1).strip().rstrip(".") if road else None
    if text.lower().startswith("your destination"):
        road_name = None
    distance = re.search(r"\bin\s+(\d+(?:\.\d+)?)\s*m\b", text, re.IGNORECASE)
    return {"road_name": road_name,
            "bearing_degrees": directions[heading.group(1)] if heading else None,
            "maneuver_distance_m": float(distance.group(1)) if distance else None}


def _road_aliases(name: str | None) -> set[str]:
    if not name:
        return set()
    aliases = set()
    for part in re.split(r"[/;]", name):
        plain = unicodedata.normalize("NFKD", part).encode("ascii", "ignore").decode().lower()
        words = re.findall(r"[a-z0-9]+", plain)
        expansions = {"st": "street", "ave": "avenue", "blvd": "boulevard", "rd": "road"}
        words = [expansions.get(word, word) for word in words]
        if words and not all(word.isdigit() for word in words):
            aliases.add(" ".join(words))
    return aliases


def _bearing_difference(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def infer_recorded_road_choice(
    graph: nx.MultiDiGraph, edge_index: RoadEdgeIndex, suggested: MapMatchResult,
    observations: list[TimedObservation], event_timestamp_ms: int,
    event_location: GeoPoint, *, original_instruction: str, deviated_instruction: str,
    earliest_choice_timestamp_ms: int, diagnostics: dict,
) -> RoadChoiceEvidence | None:
    """Resolve recorded navigation context against a unique GPS-supported branch.

    The expected edge comes from the active route geometry, not a potentially
    distant maneuver named in the old instruction. The chosen edge must fit
    the recorded road/heading and directed GPS progress. An exact shared HMM
    incoming edge is not required. Evidence remains label-only.
    """
    recorded = parse_navigation_instruction(deviated_instruction)
    original = parse_navigation_instruction(original_instruction)
    diagnostics.update({"method": "recorded_navigation_and_local_geometry",
                        "recorded_chosen": recorded, "recorded_original": original,
                        "expected_road_source": "latest pre-report route geometry",
                        "cutoff_source": "GPS approach fix preceding alternative-road movement",
                        "candidate_choices": [], "rejections": {}})
    if not recorded["road_name"] and recorded["bearing_degrees"] is None:
        diagnostics["status"] = "no_recorded_road_or_heading"
        return None
    route = suggested.traversed_edge_ids
    projected = [edge_index.project(point.point) for point in observations]
    event_point = edge_index.project(event_location)
    chosen_aliases = _road_aliases(recorded["road_name"])
    geometry_cache = {}

    def edge_data(edge):
        u, v, wanted = edge
        return next(data for key, data in graph[u][v].items() if str(key) == wanted)

    def geometry(edge):
        if edge not in geometry_cache:
            u, v, key = edge
            data = edge_data(edge)
            raw = data.get("geometry")
            line = from_wkt(raw) if isinstance(raw, str) else raw
            if line is None:
                line = LineString([(float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])) for n in (u, v)])
            if line.geom_type != "LineString" or line.length == 0:
                return None
            source = Point(float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"]))
            if Point(line.coords[-1]).distance(source) < Point(line.coords[0]).distance(source):
                line = LineString(list(line.coords)[::-1])
            geometry_cache[edge] = line
        return geometry_cache[edge]

    rejected = Counter()
    candidates = {}
    expected_by_source = {}
    for edge in route:
        expected_by_source.setdefault(edge[0], set()).add(edge)
    original_aliases = _road_aliases(original["road_name"])
    route_names = set()
    for u, v, key in route:
        for name in _tag_values(edge_data((u, v, key)).get("name")):
            route_names.update(_road_aliases(name))
    diagnostics["original_named_maneuver_on_active_route"] = (
        bool(original_aliases & route_names) if original_aliases else None)
    for offset, expected in enumerate(route):
        source = expected[0]
        if len(expected_by_source[source]) != 1 or _restricted_access(edge_data(expected).get("access")):
            continue
        node = Point(float(graph.nodes[source]["x"]), float(graph.nodes[source]["y"]))
        if not any(point.distance(node) <= 80.0 for point in projected):
            continue
        expected_line = geometry(expected)
        incoming_line = geometry(route[offset - 1]) if offset and route[offset - 1][1] == source else None
        if expected_line is None:
            continue
        for _, target, key, data in graph.out_edges(source, keys=True, data=True):
            chosen = (source, str(target), str(key))
            if chosen == expected or target == expected[1] or _restricted_access(data.get("access")):
                continue
            line = geometry(chosen)
            if line is None or line.distance(event_point) > 80.0:
                continue
            names = set()
            for field in ("name", "alt_name", "official_name"):
                for name in _tag_values(data.get(field)):
                    names.update(_road_aliases(name))
            if chosen_aliases and not chosen_aliases.intersection(names):
                rejected["recorded_road_name_does_not_match"] += 1
                continue
            supported = []
            for index in range(1, len(observations) - 1):
                before, left, right = projected[index - 1:index + 2]
                times = [observations[i].timestamp_ms for i in (index - 1, index, index + 1)]
                if any(stamp is None for stamp in times):
                    continue
                cutoff, first, last = times
                if not earliest_choice_timestamp_ms <= cutoff <= event_timestamp_ms or not cutoff < first < last:
                    continue
                if first > event_timestamp_ms + 15_000 or first - cutoff > 20_000 or last - first > 20_000:
                    continue
                if before.distance(node) > 80.0:
                    continue
                approach_distance = before.distance(incoming_line) if incoming_line is not None else before.distance(node)
                if approach_distance > (20.0 if incoming_line is not None else 5.0):
                    continue
                if line.project(before) > 3.0 and before.distance(line) + 3.0 < approach_distance:
                    continue
                distances = [left.distance(line), right.distance(line)]
                along = [line.project(left), line.project(right)]
                if max(distances) > 20.0 or along[0] < 3.0 or along[1] - along[0] < 3.0:
                    continue
                movement = left.distance(right)
                if movement < 3.0 or movement / ((last - first) / 1000.0) > 45.0:
                    continue
                if before.distance(left) / ((first - cutoff) / 1000.0) > 45.0:
                    continue
                heading = math.degrees(math.atan2(right.x - left.x, right.y - left.y)) % 360.0
                start, finish = line.interpolate(along[0]), line.interpolate(along[1])
                edge_heading = math.degrees(math.atan2(finish.x - start.x, finish.y - start.y)) % 360.0
                if _bearing_difference(heading, edge_heading) > 40.0:
                    continue
                if recorded["bearing_degrees"] is not None and _bearing_difference(heading, recorded["bearing_degrees"]) > 40.0:
                    continue
                if not all(point.distance(expected_line) - distance >= 3.0
                           for point, distance in zip((left, right), distances)):
                    continue
                supported.append(RoadChoiceEvidence(
                    source, expected, chosen, cutoff, first, 2,
                    (approach_distance + sum(distances)) / 3.0, (chosen,), (chosen, chosen)))
            if supported:
                candidates[(expected, chosen)] = min(supported, key=lambda item: item.first_chosen_timestamp_ms)
            else:
                rejected["no_directed_gps_departure_support"] += 1
    diagnostics["candidate_choices"] = [
        {"suggested_edge_id": "|".join(item.suggested_edge_id),
         "chosen_edge_id": "|".join(item.chosen_edge_id),
         "decision_timestamp_ms": item.decision_timestamp_ms,
         "first_chosen_timestamp_ms": item.first_chosen_timestamp_ms,
         "supporting_gps_points": item.supporting_gps_points,
         "mean_geometry_distance_m": item.mean_local_match_distance_m}
        for item in candidates.values()]
    diagnostics["rejections"] = dict(rejected)
    diagnostics["status"] = ("resolved_unique_choice" if len(candidates) == 1 else
                             "ambiguous_choices" if candidates else "no_supported_choice")
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def extract_outgoing_choice(
    observed: MapMatchResult,
    suggested: MapMatchResult,
    observations: list[TimedObservation],
    event_timestamp_ms: int,
    *,
    diagnostics: dict | None = None,
    earliest_choice_timestamp_ms: int | None = None,
) -> RoadChoiceEvidence | None:
    """Identify a GPS-supported outgoing choice without requiring a rejoin.

    The approach must agree with the prior suggestion. Two consecutive GPS
    fixes must support the alternative continuation before any rejoin. This
    allows a short first outgoing edge between two-second fixes without
    treating an unsupported connector detour as evidence. The last agreeing
    approach fix defines the input cutoff.
    """
    if len(observed.states) != len(observations):
        raise ValueError("Matched states and GPS observations must align")
    candidates = []
    rejected = Counter()
    shared_approaches = 0
    lower_bound = (event_timestamp_ms - 60_000 if earliest_choice_timestamp_ms is None
                   else earliest_choice_timestamp_ms)
    route = suggested.traversed_edge_ids
    for offset, (approach, chosen) in enumerate(zip(
        observed.traversed_edge_ids, observed.traversed_edge_ids[1:]
    )):
        alternatives = {route[i + 1] for i, edge in enumerate(route[:-1]) if edge == approach}
        if len(alternatives) != 1:
            continue
        shared_approaches += 1
        expected = next(iter(alternatives))
        if (chosen == expected or chosen[1] == expected[1]
                or chosen[0] != expected[0] or approach[1] != chosen[0]):
            rejected["no_distinct_alternative_at_shared_junction"] += 1
            continue
        before = [i for i, state_offset in enumerate(observed.state_edge_offsets)
                  if state_offset <= offset and observed.states[i].edge_id in route]
        first_rejoin = next((i for i in range(offset + 2, len(observed.traversed_edge_ids))
                             if observed.traversed_edge_ids[i] in route),
                            len(observed.traversed_edge_ids))
        after = [i for i, state_offset in enumerate(observed.state_edge_offsets)
                 if offset + 1 <= state_offset < first_rejoin
                 and observed.states[i].edge_id not in route]
        if not before:
            rejected["no_observed_approach_before_choice"] += 1
            continue
        if len(after) < 2:
            rejected["fewer_than_two_alternative_gps_fixes"] += 1
            continue
        if after[1] != after[0] + 1:
            rejected["alternative_gps_fixes_not_consecutive"] += 1
            continue
        last_before, first_after = before[-1], after[0]
        cutoff = observations[last_before].timestamp_ms
        departure = observations[first_after].timestamp_ms
        if cutoff is None or departure is None or cutoff >= departure or cutoff > event_timestamp_ms:
            rejected["choice_timestamps_do_not_establish_pre_report_approach"] += 1
            continue
        if not lower_bound <= departure <= event_timestamp_ms + 15_000:
            rejected["choice_outside_report_time_window"] += 1
            continue
        distances = [observed.states[i].observation_distance_m
                     for i in (last_before, after[0], after[1])]
        candidates.append(RoadChoiceEvidence(
            expected[0], expected, chosen, cutoff, departure, len(after),
            sum(distances) / len(distances),
            observed.traversed_edge_ids[offset + 1:observed.state_edge_offsets[after[1]] + 1],
            tuple(observed.states[i].edge_id for i in after[:2]),
        ))
    if diagnostics is not None:
        diagnostics.update({
            "shared_approach_transitions": shared_approaches,
            "transition_rejections": dict(rejected),
            "supported_choices": len(candidates),
            "observed_gps_points": len(observations),
            "gps_points_on_prior_suggestion": sum(state.edge_id in route for state in observed.states),
            "earliest_choice_timestamp_ms": lower_bound,
        })
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item.first_chosen_timestamp_ms - event_timestamp_ms))


def _is_contiguous(path: tuple[EdgeId, ...], origin: str, destination: str) -> bool:
    if not path or path[0][0] != origin or path[-1][1] != destination:
        return False
    return all(left[1] == right[0] for left, right in zip(path, path[1:]))


def extract_divergence_pair(
    observed: MapMatchResult,
    rejected: MapMatchResult,
    event_state_index: int,
) -> DivergencePair | None:
    """Find the nearest shared-before/shared-after boundaries around an event."""
    if not 0 <= event_state_index < len(observed.states):
        raise IndexError("event_state_index is outside the matched observation sequence")
    observed_edges = observed.traversed_edge_ids
    rejected_edges = rejected.traversed_edge_ids
    event_edge_offset = observed.state_edge_offsets[event_state_index]

    rejected_positions: dict[EdgeId, list[int]] = {}
    for index, edge_id in enumerate(rejected_edges):
        rejected_positions.setdefault(edge_id, []).append(index)

    for observed_before in range(event_edge_offset, -1, -1):
        before_edge = observed_edges[observed_before]
        for rejected_before in reversed(rejected_positions.get(before_edge, [])):
            first_after = max(event_edge_offset + 1, observed_before + 2)
            for observed_after in range(first_after, len(observed_edges)):
                after_edge = observed_edges[observed_after]
                later_rejected = [
                    index for index in rejected_positions.get(after_edge, []) if index >= rejected_before + 2
                ]
                if not later_rejected:
                    continue
                rejected_after = later_rejected[0]
                origin = before_edge[1]
                destination = after_edge[0]
                preferred_path = observed_edges[observed_before + 1 : observed_after]
                rejected_path = rejected_edges[rejected_before + 1 : rejected_after]
                if preferred_path == rejected_path:
                    continue
                if not _is_contiguous(preferred_path, origin, destination):
                    continue
                if not _is_contiguous(rejected_path, origin, destination):
                    continue
                return DivergencePair(
                    origin,
                    destination,
                    preferred_path,
                    rejected_path,
                    before_edge,
                    after_edge,
                )
    return None
