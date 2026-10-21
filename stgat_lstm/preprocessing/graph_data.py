"""Build model-ready tensors from the supplied OSM graph and reviewed choices.

Static OSM features are always available. When no historical GPS or archived
traffic source is supplied, the default temporal snapshot explicitly marks
traffic as unknown. Review-required candidate files are rejected by the
training loader.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
from shapely import from_wkt
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from torch import Tensor

from stgat_lstm.preprocessing.coordinates import is_projected_crs
from stgat_lstm.modeling.model import ModelInput


NODE_CONTEXT_COLUMNS = (
    "street_count",
    "commercial_hubs",
    "service_hubs",
    "financial_service",
    "civic_facilities",
    "educational_facilities",
    "medical_facilities",
    "driver_utility",
    "religious",
    "residential",
)
HIGHWAY_CATEGORIES = (
    "motorway_trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "service",
    "unclassified",
    "living_street",
    "other",
)
DEFAULT_SPEED_KPH = {
    "motorway_trunk": 60.0,
    "primary": 40.0,
    "secondary": 35.0,
    "tertiary": 30.0,
    "residential": 25.0,
    "service": 15.0,
    "unclassified": 25.0,
    "living_street": 15.0,
    "other": 20.0,
}


@dataclass(frozen=True)
class FeatureSchema:
    node: tuple[str, ...]
    edge_static: tuple[str, ...]
    edge_dynamic: tuple[str, ...]
    destination: tuple[str, ...]


@dataclass(frozen=True)
class RealPreferenceExample:
    event_key: str
    rider_group: str
    ride_group: str
    decision_timestamp_ms: int
    origin_node_id: str
    destination_node_id: str
    preferred_edge_ids: tuple[str, ...]
    rejected_edge_ids: tuple[str, ...]
    primary_reason: str


@dataclass(frozen=True)
class RealDecisionExample:
    example_key: str
    rider_group: str
    ride_group: str
    decision_timestamp_ms: int
    origin_node_id: str
    destination_node_id: str
    label: int
    suggested_edge_ids: tuple[str, ...]
    observed_edge_ids: tuple[str, ...]
    suggested_choice_edge_id: str | None = None
    chosen_choice_edge_id: str | None = None
    has_route_pair: bool = True


@dataclass(frozen=True)
class RealGraphData:
    graph: nx.MultiDiGraph
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    node_features: Tensor
    edge_index: Tensor
    edge_static: Tensor
    schema: FeatureSchema
    x_min: float
    y_min: float
    x_scale: float
    y_scale: float

    def build_model_input(
        self,
        destination_node_id: str,
        temporal_edge_features: tuple[Tensor, ...] | None = None,
    ) -> ModelInput:
        node_lookup = {node_id: index for index, node_id in enumerate(self.node_ids)}
        if destination_node_id not in node_lookup:
            raise ValueError(f"Destination {destination_node_id} is outside the Taft graph")
        destination = self.graph.nodes[destination_node_id]
        destination_features = torch.tensor(
            [
                [
                    (float(destination["x"]) - float(self.graph.nodes[edge_id.split("|", 2)[1]]["x"]))
                    / self.x_scale,
                    (float(destination["y"]) - float(self.graph.nodes[edge_id.split("|", 2)[1]]["y"]))
                    / self.y_scale,
                ]
                for edge_id in self.edge_ids
            ],
            dtype=torch.float32,
        )
        if temporal_edge_features is None:
            unknown = torch.zeros((len(self.edge_ids), len(self.schema.edge_dynamic)), dtype=torch.float32)
            age_index = self.schema.edge_dynamic.index("traffic_age_scaled")
            unknown[:, age_index] = 1.0  # Staleness is maximal when no observation exists.
            temporal_edge_features = (unknown,)
        for snapshot in temporal_edge_features:
            if snapshot.shape != (len(self.edge_ids), len(self.schema.edge_dynamic)):
                raise ValueError("Temporal edge snapshot has the wrong shape")
        return ModelInput(
            node_features=self.node_features,
            edge_index=self.edge_index,
            edge_static=self.edge_static,
            temporal_edge_features=temporal_edge_features,
            destination_features=destination_features,
            routable_mask=self.edge_static[:, self.schema.edge_static.index("access_restricted")] == 0,
            edge_ids=self.edge_ids,
        )

    def model_configuration(self, architecture: str = "stgat_lstm", hidden_channels: int = 16) -> dict:
        return {
            "architecture": architecture,
            "hidden_channels": hidden_channels,
            "node_feature_dim": len(self.schema.node),
            "edge_static_dim": len(self.schema.edge_static),
            "edge_dynamic_dim": len(self.schema.edge_dynamic),
            "destination_feature_dim": len(self.schema.destination),
        }


def canonical_edge_id(u: str, v: str, key: str) -> str:
    return f"{u}|{v}|{key}"


def _edge_geometry(graph: nx.MultiDiGraph, u: str, v: str, data: dict):
    raw = data.get("geometry")
    if raw:
        return from_wkt(raw) if isinstance(raw, str) else raw
    left, right = graph.nodes[u], graph.nodes[v]
    return LineString([(float(left["x"]), float(left["y"])), (float(right["x"]), float(right["y"]))])


def extract_taft_subgraph(graph: nx.MultiDiGraph, buffer_m: float = 1_000.0) -> nx.MultiDiGraph:
    if buffer_m <= 0:
        raise ValueError("buffer_m must be positive")
    crs = graph.graph.get("crs")
    if not crs or not is_projected_crs(crs):
        raise ValueError("Taft extraction requires a projected graph CRS")
    taft_lines = [
        _edge_geometry(graph, str(u), str(v), data)
        for u, v, _, data in graph.edges(keys=True, data=True)
        if "taft avenue" in str(data.get("name", "")).lower()
    ]
    if not taft_lines:
        raise ValueError("Graph has no edges named Taft Avenue")
    corridor = unary_union(taft_lines).buffer(buffer_m)
    selected = [
        node_id
        for node_id, data in graph.nodes(data=True)
        if corridor.covers(Point(float(data["x"]), float(data["y"])))
    ]
    subgraph = graph.subgraph(selected).copy()
    subgraph.graph["model_corridor_buffer_m"] = buffer_m
    if subgraph.number_of_edges() == 0:
        raise ValueError("Taft corridor extraction produced no road edges")
    return subgraph


def _read_node_features(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"osmid", *NODE_CONTEXT_COLUMNS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Node feature CSV lacks columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    result = {}
    for row in rows:
        node_id = row["osmid"]
        if node_id in result:
            raise ValueError(f"Duplicate node feature row: {node_id}")
        result[node_id] = {column: float(row[column]) for column in NODE_CONTEXT_COLUMNS}
    return result


def _tag_values(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw]
    text = str(raw)
    if text.startswith("["):
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            pass
        else:
            if isinstance(value, (list, tuple)):
                return [str(item) for item in value]
    return [text]


def _highway_category(raw) -> str:
    values = " ".join(_tag_values(raw)).lower()
    if "motorway" in values or "trunk" in values:
        return "motorway_trunk"
    for category in HIGHWAY_CATEGORIES[1:-1]:
        if category in values:
            return category
    return "other"


def _numeric_tag(raw) -> float | None:
    values = _tag_values(raw)
    parsed = []
    for value in values:
        numbers = re.findall(r"\d+(?:\.\d+)?", value)
        parsed.extend(float(number) * (1.609344 if "mph" in value.lower() else 1.0) for number in numbers)
    return sum(parsed) / len(parsed) if parsed else None


def _truthy_tag(raw) -> float:
    return float(any(value.lower() in {"yes", "true", "1"} for value in _tag_values(raw)))


def _restricted_access(raw) -> float:
    return float(any(value.lower() in {"no", "private"} for value in _tag_values(raw)))


def path_choice_edge_ids(graph_data: RealGraphData, path_edge_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return path edges whose source node offers at least two accessible next roads."""
    available = set(graph_data.edge_ids)
    choices = []
    for edge_id in path_edge_ids:
        if edge_id not in available:
            raise ValueError(f"Path has unavailable edge: {edge_id}")
        source, _, _ = edge_id.split("|", 2)
        successors = {
            str(destination)
            for _, destination, _, data in graph_data.graph.out_edges(source, keys=True, data=True)
            if _restricted_access(data.get("access")) == 0
        }
        if len(successors) >= 2:
            choices.append(edge_id)
    return tuple(choices)


def accessible_outgoing_edge_ids(graph_data: RealGraphData, source_node_id: str) -> tuple[str, ...]:
    """Return accessible directed edges offered at one graph node."""
    available = set(graph_data.edge_ids)
    edges = [
        canonical_edge_id(str(source), str(destination), str(key))
        for source, destination, key, data in graph_data.graph.out_edges(
            source_node_id, keys=True, data=True
        )
        if _restricted_access(data.get("access")) == 0
    ]
    edge_order = {edge_id: index for index, edge_id in enumerate(graph_data.edge_ids)}
    return tuple(sorted((edge_id for edge_id in edges if edge_id in available), key=edge_order.__getitem__))


def build_real_graph_data_from_graph(
    graph: nx.MultiDiGraph,
    node_feature_rows: dict[str, dict[str, float]],
    *,
    corridor_buffer_m: float = 1_000.0,
) -> RealGraphData:
    graph = extract_taft_subgraph(graph, corridor_buffer_m)
    node_ids = tuple(sorted((str(node_id) for node_id in graph.nodes),
                            key=lambda value: (0, int(value)) if value.isdigit() else (1, value)))
    missing = [node_id for node_id in node_ids if node_id not in node_feature_rows]
    if missing:
        raise ValueError(f"Missing node feature rows for {len(missing)} Taft nodes")
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    xs = [float(graph.nodes[node_id]["x"]) for node_id in node_ids]
    ys = [float(graph.nodes[node_id]["y"]) for node_id in node_ids]
    x_min, y_min = min(xs), min(ys)
    x_scale, y_scale = max(max(xs) - x_min, 1.0), max(max(ys) - y_min, 1.0)
    context_scales = {
        column: max(max(math.log1p(node_feature_rows[node_id][column]) for node_id in node_ids), 1.0)
        for column in NODE_CONTEXT_COLUMNS
    }
    node_features = torch.tensor(
        [
            [
                (float(graph.nodes[node_id]["x"]) - x_min) / x_scale,
                (float(graph.nodes[node_id]["y"]) - y_min) / y_scale,
                float("traffic_signals" in str(graph.nodes[node_id].get("highway", ""))),
                *[
                    math.log1p(node_feature_rows[node_id][column]) / context_scales[column]
                    for column in NODE_CONTEXT_COLUMNS
                ],
            ]
            for node_id in node_ids
        ],
        dtype=torch.float32,
    )

    ordered_edges = sorted(
        graph.edges(keys=True, data=True),
        key=lambda edge: (str(edge[0]), str(edge[1]), str(edge[2])),
    )
    edge_ids = tuple(canonical_edge_id(str(u), str(v), str(key)) for u, v, key, _ in ordered_edges)
    edge_index = torch.tensor(
        [
            [node_index[str(u)] for u, _, _, _ in ordered_edges],
            [node_index[str(v)] for _, v, _, _ in ordered_edges],
        ],
        dtype=torch.long,
    )
    static_rows = []
    for _, _, _, data in ordered_edges:
        length_m = float(data["length"])
        if not math.isfinite(length_m) or length_m <= 0:
            raise ValueError("OSM edge has invalid length")
        category = _highway_category(data.get("highway"))
        maxspeed = _numeric_tag(data.get("maxspeed"))
        reference_speed = maxspeed or DEFAULT_SPEED_KPH[category]
        lanes = _numeric_tag(data.get("lanes"))
        static_rows.append(
            [
                length_m / 100.0,
                (length_m / (reference_speed * 1000.0 / 3600.0)) / 60.0,
                _truthy_tag(data.get("oneway")),
                min((lanes or 0.0) / 6.0, 1.0),
                float(lanes is not None),
                min((maxspeed or 0.0) / 100.0, 1.5),
                float(maxspeed is not None),
                _truthy_tag(data.get("bridge")),
                _truthy_tag(data.get("tunnel")),
                _restricted_access(data.get("access")),
                *[float(category == candidate) for candidate in HIGHWAY_CATEGORIES],
            ]
        )
    edge_static = torch.tensor(static_rows, dtype=torch.float32)
    schema = FeatureSchema(
        node=("x_normalized", "y_normalized", "traffic_signal", *NODE_CONTEXT_COLUMNS),
        edge_static=(
            "length_per_100m",
            "reference_time_minutes",
            "oneway",
            "lanes_scaled",
            "lanes_known",
            "maxspeed_scaled",
            "maxspeed_known",
            "bridge",
            "tunnel",
            "access_restricted",
            *(f"highway_{category}" for category in HIGHWAY_CATEGORIES),
        ),
        edge_dynamic=(
            "congestion_normalized",
            "speed_ratio_to_reference",
            "congestion_observed",
            "speed_observed",
            "traffic_age_scaled",
        ),
        destination=("destination_dx", "destination_dy"),
    )
    return RealGraphData(
        graph,
        node_ids,
        edge_ids,
        node_features,
        edge_index,
        edge_static,
        schema,
        x_min,
        y_min,
        x_scale,
        y_scale,
    )


def build_real_graph_data(
    graphml_path: Path,
    node_features_path: Path,
    *,
    corridor_buffer_m: float = 1_000.0,
    approved_decisions: Path | None = None,
) -> RealGraphData:
    graph = nx.read_graphml(graphml_path, force_multigraph=True)
    if approved_decisions is not None:
        document = json.loads(approved_decisions.read_text(encoding="utf-8"))
        corridor_buffer_m = training_graph_buffer_m(graph, document, corridor_buffer_m)
    return build_real_graph_data_from_graph(
        graph,
        _read_node_features(node_features_path),
        corridor_buffer_m=corridor_buffer_m,
    )


def training_graph_buffer_m(graph: nx.MultiDiGraph, document: dict, minimum_m: float = 1000.0) -> float:
    """Cover retained road evidence and destinations without excluding labels.

    The Taft crop is a computational starting region. Expand it when approved
    examples need more of the supplied graph, retaining a 100 m context margin.
    The chosen scope is saved with the model for consistent inference scaling.
    """
    if document.get("status") != "approved_for_training":
        raise ValueError("Graph expansion requires an approved decision artifact")
    required_nodes = set()
    for example in document.get("examples", []):
        if example.get("status") != "approved":
            continue
        context = example.get("routing_context", example.get("common_od", {}))
        required_nodes.update(str(context[key]) for key in ("origin_node_id", "destination_node_id"))
        for field in ("suggested_path", "observed_path_label_evidence"):
            for edge in example[field]["edges"]:
                u, v, key = str(edge["u"]), str(edge["v"]), str(edge["key"])
                if not graph.has_edge(u, v) or not any(str(k) == key for k in graph[u][v]):
                    raise ValueError("Approved road evidence is absent from the supplied OSM graph")
                required_nodes.update((u, v))
    if not required_nodes.issubset(graph.nodes):
        raise ValueError("Approved routing context contains nodes absent from the supplied OSM graph")
    taft = unary_union([
        _edge_geometry(graph, str(u), str(v), data)
        for u, v, _, data in graph.edges(keys=True, data=True)
        if "taft avenue" in str(data.get("name", "")).lower()
    ])
    if taft.is_empty:
        raise ValueError("Graph has no Taft geometry for its initial computational crop")
    distance = max((taft.distance(Point(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])))
                    for node in required_nodes), default=0.0)
    return max(minimum_m, math.ceil((distance + 100.0) / 100.0) * 100.0)


def _candidate_path_ids(path: dict) -> tuple[str, ...]:
    return tuple(canonical_edge_id(str(edge["u"]), str(edge["v"]), str(edge["key"])) for edge in path["edges"])


def load_approved_preferences(path: Path, graph_data: RealGraphData) -> list[RealPreferenceExample]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "approved_for_training":
        raise ValueError("Real preference artifact has not been approved for training")
    available = set(graph_data.edge_ids)
    examples = []
    for candidate in document.get("candidates", []):
        if candidate.get("status") != "approved":
            continue
        if not candidate.get("has_route_pair", True):
            continue
        preferred = _candidate_path_ids(candidate["preferred_observed_path"])
        rejected = _candidate_path_ids(candidate["rejected_prior_suggestion"])
        missing = (set(preferred) | set(rejected)) - available
        if missing:
            raise ValueError(f"Approved choice has {len(missing)} edges outside the Taft graph")
        examples.append(
            RealPreferenceExample(
                candidate["event_key"],
                candidate["rider_group"],
                candidate["ride_group"],
                int(candidate["decision_timestamp_ms"]),
                str(candidate["common_od"]["origin_node_id"]),
                str(candidate["common_od"]["destination_node_id"]),
                preferred,
                rejected,
                candidate["survey"]["primary_reason"],
            )
        )
    if not examples:
        raise ValueError("Approved artifact contains no comparable route-ranking pairs")
    return examples


def load_approved_decisions(path: Path, graph_data: RealGraphData) -> list[RealDecisionExample]:
    """Load only validated follow/deviate examples."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("status") != "approved_for_training":
        raise ValueError("Real decision artifact has not been approved for training")
    available = set(graph_data.edge_ids)
    examples = []
    for example in document.get("examples", []):
        if example.get("status") != "approved":
            continue
        suggested = _candidate_path_ids(example["suggested_path"])
        observed = _candidate_path_ids(example["observed_path_label_evidence"])
        missing = (set(suggested) | set(observed)) - available
        if missing:
            raise ValueError(f"Approved decision has {len(missing)} edges outside the Taft graph")
        label = int(example["target"]["value"])
        if label not in {0, 1}:
            raise ValueError("Decision label must be 0 or 1")
        context = example.get("routing_context", example.get("common_od"))
        local = example.get("road_choice", {})
        if str(context["destination_node_id"]) not in graph_data.node_ids:
            raise ValueError("Approved decision destination is outside the model graph")
        if local and (local.get("suggested_edge_id") not in suggested
                      or local.get("chosen_edge_id") not in observed):
            raise ValueError("Approved local road choice disagrees with path evidence")
        examples.append(
            RealDecisionExample(
                example_key=str(example["example_key"]),
                rider_group=str(example["rider_group"]),
                ride_group=str(example["ride_group"]),
                decision_timestamp_ms=int(example["decision_timestamp_ms"]),
                origin_node_id=str(context["origin_node_id"]),
                destination_node_id=str(context["destination_node_id"]),
                label=label,
                suggested_edge_ids=suggested,
                observed_edge_ids=observed,
                suggested_choice_edge_id=local.get("suggested_edge_id"),
                chosen_choice_edge_id=local.get("chosen_edge_id"),
                has_route_pair=bool(example.get("has_route_pair", True)),
            )
        )
    if not examples:
        raise ValueError("Approved artifact contains no approved decision examples")
    return examples


def audit_feature_pipeline(graph_data: RealGraphData, candidate_path: Path) -> dict:
    document = json.loads(candidate_path.read_text(encoding="utf-8"))
    available = set(graph_data.edge_ids)
    covered = 0
    for candidate in document.get("candidates", []):
        paths = (
            _candidate_path_ids(candidate["preferred_observed_path"]),
            _candidate_path_ids(candidate["rejected_prior_suggestion"]),
        )
        covered += int(all(set(path).issubset(available) for path in paths))
    first_destination = (
        str(document["candidates"][0].get("routing_context", document["candidates"][0].get("common_od"))["destination_node_id"])
        if document.get("candidates")
        else graph_data.node_ids[0]
    )
    model_input = graph_data.build_model_input(first_destination)
    return {
        "provenance": "OSM feature readiness only; no model training performed",
        "corridor_graph": {"nodes": len(graph_data.node_ids), "edges": len(graph_data.edge_ids)},
        "feature_dimensions": {
            "node": len(graph_data.schema.node),
            "edge_static": len(graph_data.schema.edge_static),
            "edge_dynamic": len(graph_data.schema.edge_dynamic),
            "destination": len(graph_data.schema.destination),
        },
        "feature_names": {
            "node": graph_data.schema.node,
            "edge_static": graph_data.schema.edge_static,
            "edge_dynamic": graph_data.schema.edge_dynamic,
            "destination": graph_data.schema.destination,
        },
        "candidate_pairs_fully_covered": covered,
        "candidate_pairs_total": len(document.get("candidates", [])),
        "traffic_state": "unknown snapshot with explicit observation mask and staleness flag",
        "model_input_shapes": {
            "node_features": list(model_input.node_features.shape),
            "edge_index": list(model_input.edge_index.shape),
            "edge_static": list(model_input.edge_static.shape),
            "temporal_edge_features": [list(value.shape) for value in model_input.temporal_edge_features],
            "destination_features": list(model_input.destination_features.shape),
        },
        "candidate_status": document.get("status"),
        "training_allowed": document.get("status") == "approved_for_training",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphml", type=Path)
    parser.add_argument("node_features", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--corridor-buffer-m", type=float, default=1_000.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/audits/model_features_audit.json"))
    args = parser.parse_args()
    graph_data = build_real_graph_data(
        args.graphml,
        args.node_features,
        corridor_buffer_m=args.corridor_buffer_m,
    )
    report = audit_feature_pipeline(graph_data, args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    graph = report["corridor_graph"]
    print(f"Feature audit saved to {args.output}")
    print(f"Corridor graph: {graph['nodes']} nodes, {graph['edges']} directed edges")
    print(f"Candidate paths covered: {report['candidate_pairs_fully_covered']}/"
          f"{report['candidate_pairs_total']}")
    print(f"Candidate status: {report['candidate_status']} | training allowed: "
          f"{'yes' if report['training_allowed'] else 'no'}")


if __name__ == "__main__":
    main()
