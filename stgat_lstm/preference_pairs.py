"""Extract conservative common-OD route preference pairs.

The preferred path comes from the map-matched rider GPS trace. The rejected
path comes from the latest suggested route available before the deviation.
Both paths are bounded by a shared directed edge before the deviation and a
later shared directed edge after it, giving them the same origin and
destination nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .map_matching import EdgeId, MapMatchResult


@dataclass(frozen=True)
class DivergencePair:
    origin_node_id: str
    destination_node_id: str
    preferred_edge_ids: tuple[EdgeId, ...]
    rejected_edge_ids: tuple[EdgeId, ...]
    shared_edge_before: EdgeId
    shared_edge_after: EdgeId


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
