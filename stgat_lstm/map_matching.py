"""Sequence-aware HMM/Viterbi matching of GPS observations to OSM edges.

Algorithmic reference:
Paul Newson and John Krumm, "Hidden Markov Map Matching Through Noise and
Sparseness," ACM SIGSPATIAL GIS 2009, DOI 10.1145/1653771.1653818.
Official paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/map-matching-ACM-GIS-camera-ready.pdf

This is an independent implementation of the paper's core formulation:
Gaussian emission likelihood from GPS-to-road distance and an exponential
transition likelihood based on the difference between network and observed
displacement. It preserves directed OSM edges and uses Viterbi decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
from shapely.geometry import Point

from .geometry import GeoPoint
from .osm_audit import RoadEdgeIndex


EdgeId = tuple[str, str, str]


@dataclass(frozen=True)
class TimedObservation:
    point: GeoPoint
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class MatchState:
    edge_id: EdgeId
    observation_distance_m: float
    distance_along_edge_m: float
    edge_length_m: float


@dataclass(frozen=True)
class MapMatchResult:
    states: tuple[MatchState, ...]
    traversed_edge_ids: tuple[EdgeId, ...]
    state_edge_offsets: tuple[int, ...]
    log_probability: float
    mean_observation_distance_m: float
    max_observation_distance_m: float


@dataclass(frozen=True)
class _Candidate:
    state: MatchState
    record_index: int


class HiddenMarkovMatcher:
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        edge_index: RoadEdgeIndex,
        *,
        search_radius_m: float = 60.0,
        max_candidates: int = 6,
        gps_sigma_m: float = 15.0,
        transition_beta_m: float = 50.0,
        max_speed_mps: float = 45.0,
    ) -> None:
        if gps_sigma_m <= 0.0 or transition_beta_m <= 0.0 or max_speed_mps <= 0.0:
            raise ValueError("Matcher scale and speed parameters must be positive")
        self.graph = graph
        self.edge_index = edge_index
        self.search_radius_m = search_radius_m
        self.max_candidates = max_candidates
        self.gps_sigma_m = gps_sigma_m
        self.transition_beta_m = transition_beta_m
        self.max_speed_mps = max_speed_mps
        self._connectors: dict[tuple[str, str], tuple[float, tuple[EdgeId, ...]] | None] = {}
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            data["_match_length_m"] = float(data.get("length", 1.0))

    def _candidates(self, observation: TimedObservation) -> list[_Candidate]:
        projected, nearby = self.edge_index.nearby(
            observation.point, self.search_radius_m, self.max_candidates
        )
        candidates: list[_Candidate] = []
        for record_index, distance in nearby:
            u, v, key, data = self.edge_index.records[record_index]
            geometry = self.edge_index.tree.geometries[record_index]
            geometry_length = float(geometry.length)
            edge_length = float(data.get("length", geometry_length))
            if geometry_length <= 0.0 or edge_length <= 0.0:
                continue
            fraction = float(geometry.project(projected)) / geometry_length
            candidates.append(
                _Candidate(
                    MatchState((u, v, key), distance, fraction * edge_length, edge_length),
                    record_index,
                )
            )
        return candidates

    def _connector(self, source: str, target: str) -> tuple[float, tuple[EdgeId, ...]] | None:
        cache_key = (source, target)
        if cache_key in self._connectors:
            return self._connectors[cache_key]
        if source == target:
            result: tuple[float, tuple[EdgeId, ...]] | None = (0.0, ())
        else:
            target_data = self.graph.nodes[target]

            def heuristic(node: str, _target: str) -> float:
                node_data = self.graph.nodes[node]
                return math.hypot(
                    float(node_data["x"]) - float(target_data["x"]),
                    float(node_data["y"]) - float(target_data["y"]),
                )

            try:
                nodes = nx.astar_path(
                    self.graph,
                    source,
                    target,
                    heuristic=heuristic,
                    weight="_match_length_m",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                result = None
            else:
                edges: list[EdgeId] = []
                total = 0.0
                for left, right in zip(nodes, nodes[1:]):
                    edge_options = self.graph[left][right]
                    key, data = min(
                        edge_options.items(), key=lambda item: float(item[1]["_match_length_m"])
                    )
                    total += float(data["_match_length_m"])
                    edges.append((str(left), str(right), str(key)))
                result = (total, tuple(edges))
        self._connectors[cache_key] = result
        return result

    def _network_transition(self, left: MatchState, right: MatchState) -> tuple[float, tuple[EdgeId, ...]] | None:
        if left.edge_id == right.edge_id and right.distance_along_edge_m >= left.distance_along_edge_m:
            return right.distance_along_edge_m - left.distance_along_edge_m, ()
        connector = self._connector(left.edge_id[1], right.edge_id[0])
        if connector is None:
            return None
        connector_distance, connector_edges = connector
        distance = left.edge_length_m - left.distance_along_edge_m + connector_distance + right.distance_along_edge_m
        return distance, connector_edges

    def match(self, observations: list[TimedObservation]) -> MapMatchResult:
        if len(observations) < 2:
            raise ValueError("Map matching requires at least two observations")
        candidate_layers = [self._candidates(observation) for observation in observations]
        missing = [index for index, candidates in enumerate(candidate_layers) if not candidates]
        if missing:
            raise ValueError(f"No OSM edge candidate within {self.search_radius_m:g} m for observations {missing}")

        scores: list[list[float]] = []
        backpointers: list[list[int]] = []
        scores.append([self._emission(candidate.state) for candidate in candidate_layers[0]])
        backpointers.append([-1] * len(candidate_layers[0]))

        for layer_index in range(1, len(candidate_layers)):
            previous_observation = observations[layer_index - 1]
            observation = observations[layer_index]
            observed_distance = self.edge_index.project(previous_observation.point).distance(
                self.edge_index.project(observation.point)
            )
            elapsed_seconds = None
            if previous_observation.timestamp_ms is not None and observation.timestamp_ms is not None:
                elapsed_seconds = (observation.timestamp_ms - previous_observation.timestamp_ms) / 1000.0
                if elapsed_seconds <= 0.0:
                    raise ValueError("Observation timestamps must increase")

            layer_scores: list[float] = []
            layer_backpointers: list[int] = []
            for candidate in candidate_layers[layer_index]:
                best_score = -math.inf
                best_previous = -1
                for previous_index, previous in enumerate(candidate_layers[layer_index - 1]):
                    transition = self._network_transition(previous.state, candidate.state)
                    if transition is None:
                        continue
                    network_distance, _ = transition
                    if elapsed_seconds is not None and network_distance / elapsed_seconds > self.max_speed_mps:
                        continue
                    transition_score = -abs(network_distance - observed_distance) / self.transition_beta_m
                    score = scores[-1][previous_index] + transition_score + self._emission(candidate.state)
                    if score > best_score:
                        best_score = score
                        best_previous = previous_index
                layer_scores.append(best_score)
                layer_backpointers.append(best_previous)
            if all(not math.isfinite(score) for score in layer_scores):
                raise ValueError(f"No feasible directed transition into observation {layer_index}")
            scores.append(layer_scores)
            backpointers.append(layer_backpointers)

        state_indices = [max(range(len(scores[-1])), key=scores[-1].__getitem__)]
        for layer_index in range(len(candidate_layers) - 1, 0, -1):
            previous = backpointers[layer_index][state_indices[-1]]
            if previous < 0:
                raise RuntimeError("Viterbi backpointer is missing")
            state_indices.append(previous)
        state_indices.reverse()
        states = tuple(candidate_layers[index][state_index].state for index, state_index in enumerate(state_indices))

        traversed: list[EdgeId] = [states[0].edge_id]
        state_edge_offsets = [0]
        for left, right in zip(states, states[1:]):
            transition = self._network_transition(left, right)
            if transition is None:
                raise RuntimeError("Decoded transition became disconnected")
            _, connector_edges = transition
            for edge_id in (*connector_edges, right.edge_id):
                if edge_id != traversed[-1]:
                    traversed.append(edge_id)
            state_edge_offsets.append(len(traversed) - 1)
        distances = [state.observation_distance_m for state in states]
        return MapMatchResult(
            states,
            tuple(traversed),
            tuple(state_edge_offsets),
            max(scores[-1]),
            sum(distances) / len(distances),
            max(distances),
        )

    def _emission(self, state: MatchState) -> float:
        return -0.5 * (state.observation_distance_m / self.gps_sigma_m) ** 2
