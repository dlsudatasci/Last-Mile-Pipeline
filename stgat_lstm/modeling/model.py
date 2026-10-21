"""Graph-attention route preference models.

Research basis:
- Veličković et al., Graph Attention Networks, https://arxiv.org/abs/1710.10903
- Brody et al., How Attentive are Graph Attention Networks?,
  https://arxiv.org/abs/2105.14491 (GATv2 dynamic attention)
- PyTorch Geometric GATv2Conv,
  https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html
  (maintained library implementation used below)
- Kipf and Welling, Semi-Supervised Classification with Graph Convolutional
  Networks, https://arxiv.org/abs/1609.02907 (GCN baseline)
- Hochreiter and Schmidhuber, Long Short-Term Memory,
  https://doi.org/10.1162/neco.1997.9.8.1735 (standard LSTM component)
- Zhang, Yu, and Liu, Spatial-Temporal Graph Attention Networks,
  https://doi.org/10.1109/ACCESS.2019.2953888 (spatial-then-temporal precedent)

This is a task-specific composition of spatial attention, edge context and
PyTorch LSTM. It uses the library GATv2 operator rather than copying the
authors' experimental repository, and it does not reproduce a complete model
from any one paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv
from torch_geometric.utils import softmax

DEFAULT_ATTENTION_NEGATIVE_SLOPE = 0.2


@dataclass(frozen=True)
class ModelInput:
    """Tensors consumed by every spatial or spatiotemporal model."""

    node_features: Tensor
    edge_index: Tensor
    edge_static: Tensor
    temporal_edge_features: tuple[Tensor, ...]
    destination_features: Tensor
    routable_mask: Tensor
    edge_ids: tuple[str, ...]


class PreferenceModel(nn.Module):
    """Produce nonnegative, dimensionless preference costs per directed edge.

    ``stgat_lstm`` encodes each available graph snapshot with GATv2 and then
    applies an LSTM per road edge. ``gatv2``, ``gat`` and ``gcn`` are spatial
    baselines using the latest available snapshot and the same pairwise choice
    loss. The spatial-only GATv2 baseline isolates the LSTM's contribution.
    """

    def __init__(
        self,
        architecture: str = "stgat_lstm",
        hidden_channels: int = 16,
        attention_negative_slope: float = DEFAULT_ATTENTION_NEGATIVE_SLOPE,
        node_feature_dim: int = 3,
        edge_static_dim: int = 3,
        edge_dynamic_dim: int = 3,
        destination_feature_dim: int = 2,
    ):
        super().__init__()
        if architecture not in {"stgat_lstm", "gatv2", "gat", "gcn"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        if hidden_channels < 2:
            raise ValueError("hidden_channels must be at least 2")
        if attention_negative_slope <= 0:
            raise ValueError("attention_negative_slope must be positive")
        feature_dimensions = {
            "node_feature_dim": node_feature_dim,
            "edge_static_dim": edge_static_dim,
            "edge_dynamic_dim": edge_dynamic_dim,
            "destination_feature_dim": destination_feature_dim,
        }
        if any(value <= 0 for value in feature_dimensions.values()):
            raise ValueError(f"Feature dimensions must be positive: {feature_dimensions}")
        self.architecture = architecture
        self.hidden_channels = hidden_channels
        self.attention_negative_slope = attention_negative_slope
        self.node_feature_dim = node_feature_dim
        self.edge_static_dim = edge_static_dim
        self.edge_dynamic_dim = edge_dynamic_dim
        self.destination_feature_dim = destination_feature_dim
        attention_edge_dim = edge_static_dim + edge_dynamic_dim
        self.node_projection = nn.Linear(node_feature_dim, hidden_channels)
        if architecture in {"stgat_lstm", "gatv2"}:
            self.spatial_1 = GATv2Conv(
                hidden_channels,
                hidden_channels,
                heads=2,
                concat=False,
                edge_dim=attention_edge_dim,
                negative_slope=attention_negative_slope,
            )
            self.spatial_2 = GATv2Conv(
                hidden_channels,
                hidden_channels,
                heads=2,
                concat=False,
                edge_dim=attention_edge_dim,
                negative_slope=attention_negative_slope,
            )
        elif architecture == "gat":
            self.spatial_1 = GATConv(
                hidden_channels,
                hidden_channels,
                heads=2,
                concat=False,
                edge_dim=attention_edge_dim,
                negative_slope=attention_negative_slope,
            )
            self.spatial_2 = GATConv(
                hidden_channels,
                hidden_channels,
                heads=2,
                concat=False,
                edge_dim=attention_edge_dim,
                negative_slope=attention_negative_slope,
            )
        else:
            self.spatial_1 = GCNConv(hidden_channels, hidden_channels)
            self.spatial_2 = GCNConv(hidden_channels, hidden_channels)

        self.edge_projection = nn.Linear(
            2 * hidden_channels + edge_static_dim + edge_dynamic_dim + destination_feature_dim,
            hidden_channels,
        )
        self.temporal = nn.LSTM(hidden_channels, hidden_channels, batch_first=True) if architecture == "stgat_lstm" else None
        self.cost_head = nn.Linear(hidden_channels, 1)

    def _spatial_snapshot(self, inputs: ModelInput, edge_dynamic: Tensor) -> Tensor:
        if inputs.node_features.shape[1] != self.node_feature_dim:
            raise ValueError("Node feature dimension does not match the model configuration")
        if inputs.edge_static.shape[1] != self.edge_static_dim:
            raise ValueError("Static edge feature dimension does not match the model configuration")
        if edge_dynamic.shape[1] != self.edge_dynamic_dim:
            raise ValueError("Dynamic edge feature dimension does not match the model configuration")
        if inputs.destination_features.shape[1] != self.destination_feature_dim:
            raise ValueError("Destination feature dimension does not match the model configuration")
        node_embedding = F.relu(self.node_projection(inputs.node_features))
        attention_edge_attr = torch.cat((inputs.edge_static, edge_dynamic), dim=-1)
        if self.architecture == "gcn":
            node_embedding = F.relu(self.spatial_1(node_embedding, inputs.edge_index))
            node_embedding = F.relu(self.spatial_2(node_embedding, inputs.edge_index))
        else:
            node_embedding = F.elu(self.spatial_1(node_embedding, inputs.edge_index, attention_edge_attr))
            node_embedding = F.elu(self.spatial_2(node_embedding, inputs.edge_index, attention_edge_attr))
        source, destination = inputs.edge_index
        edge_embedding = torch.cat(
            (
                node_embedding[source],
                node_embedding[destination],
                inputs.edge_static,
                edge_dynamic,
                inputs.destination_features,
            ),
            dim=-1,
        )
        return F.relu(self.edge_projection(edge_embedding))

    def encode_edges(self, inputs: ModelInput) -> Tensor:
        """Return the final spatiotemporal embedding for every directed edge."""
        if len(inputs.temporal_edge_features) == 0:
            raise ValueError("At least one observed or unknown traffic snapshot is required")
        if self.temporal is not None:
            # Reuse exact repeats within this call; keep every LSTM time step.
            # Input gradients require separate graphs for separate frame tensors.
            spatial_frames: list[tuple[Tensor, Tensor]] = []
            encoded_sequence = []
            for snapshot in inputs.temporal_edge_features:
                encoded = None
                if not snapshot.requires_grad:
                    for previous, previous_encoded in spatial_frames:
                        if snapshot.dtype == previous.dtype and snapshot.device == previous.device and torch.equal(snapshot, previous):
                            encoded = previous_encoded
                            break
                if encoded is None:
                    encoded = self._spatial_snapshot(inputs, snapshot)
                    if not snapshot.requires_grad:
                        spatial_frames.append((snapshot, encoded))
                encoded_sequence.append(encoded)
            sequence = torch.stack(encoded_sequence, dim=1)
            embedding, _ = self.temporal(sequence)
            last = embedding[:, -1, :]
        else:
            last = self._spatial_snapshot(inputs, inputs.temporal_edge_features[-1])
        return last

    def costs_from_embeddings(self, edge_embeddings: Tensor, inputs: ModelInput) -> Tensor:
        # Predict preference cost per 100 m, then integrate over edge length.
        # This reduces sensitivity to how the road graph subdivides a street.
        return F.softplus(self.cost_head(edge_embeddings).squeeze(-1)) * inputs.edge_static[:, 0] + 1e-4

    def forward(self, inputs: ModelInput) -> Tensor:
        return self.costs_from_embeddings(self.encode_edges(inputs), inputs)

    def configuration(self) -> dict[str, int | float | str]:
        """Return the architecture fields required to recreate this model."""
        return {
            "architecture": self.architecture,
            "hidden_channels": self.hidden_channels,
            "attention_negative_slope": self.attention_negative_slope,
            "node_feature_dim": self.node_feature_dim,
            "edge_static_dim": self.edge_static_dim,
            "edge_dynamic_dim": self.edge_dynamic_dim,
            "destination_feature_dim": self.destination_feature_dim,
        }


def path_cost(edge_costs: Tensor, edge_ids: tuple[str, ...], available_edge_ids: tuple[str, ...]) -> Tensor:
    lookup = {edge_id: index for index, edge_id in enumerate(available_edge_ids)}
    try:
        indices = [lookup[edge_id] for edge_id in edge_ids]
    except KeyError as exc:
        raise ValueError(f"Path has unavailable edge: {exc.args[0]}") from exc
    if not indices:
        raise ValueError("Cannot score an empty path")
    return edge_costs[indices].sum()


def preference_loss(edge_costs: Tensor, preferred: tuple[str, ...], rejected: tuple[str, ...], available: tuple[str, ...]) -> Tensor:
    """Lower the cost of the observed intentional choice than its suggestion."""
    preferred_cost = path_cost(edge_costs, preferred, available)
    rejected_cost = path_cost(edge_costs, rejected, available)
    return F.softplus(preferred_cost - rejected_cost)


class DecisionPreferenceModel(nn.Module):
    """Shared STGAT-LSTM that scores each accessible outgoing road."""

    def __init__(self, **preference_configuration):
        super().__init__()
        self.backbone = PreferenceModel(**preference_configuration)
        hidden = self.backbone.hidden_channels
        # This model derives routing costs from normalized choice probabilities.
        # The standalone PreferenceModel keeps its direct cost head for legacy baselines.
        del self.backbone.cost_head
        self.edge_choice_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs: ModelInput) -> tuple[Tensor, Tensor]:
        """Return additive choice costs and utility logits for every road edge."""
        edge_embeddings = self.backbone.encode_edges(inputs)
        edge_choice_logits = self.edge_choice_head(edge_embeddings).squeeze(-1)
        edge_costs = outgoing_choice_costs(edge_choice_logits, inputs)
        return edge_costs, edge_choice_logits

    def configuration(self) -> dict:
        return {"model_type": "outgoing_road_choice_preference", "backbone": self.backbone.configuration()}


def outgoing_choice_log_probabilities(edge_logits: Tensor, inputs: ModelInput) -> Tensor:
    """Normalize edge utilities among accessible roads from the same source node."""
    if edge_logits.shape != (len(inputs.edge_ids),):
        raise ValueError("Choice head must return one logit per directed edge")
    if inputs.routable_mask.shape != edge_logits.shape or inputs.routable_mask.dtype != torch.bool:
        raise ValueError("Routable edge mask must be boolean with one value per edge")
    source_nodes = inputs.edge_index[0]
    result = torch.full_like(edge_logits, float("-inf"))
    indices = torch.nonzero(inputs.routable_mask, as_tuple=False).squeeze(-1)
    probabilities = softmax(edge_logits[indices], source_nodes[indices])
    result[indices] = torch.log(probabilities.clamp_min(torch.finfo(edge_logits.dtype).tiny))
    return result


def outgoing_choice_costs(edge_logits: Tensor, inputs: ModelInput) -> Tensor:
    """Convert local outgoing-road probabilities into positive additive costs."""
    log_probabilities = outgoing_choice_log_probabilities(edge_logits, inputs)
    costs = torch.full_like(edge_logits, 1e6)
    length_tiebreaker = inputs.edge_static[:, 0] * 1e-4 + 1e-6
    costs[inputs.routable_mask] = -log_probabilities[inputs.routable_mask] + length_tiebreaker[inputs.routable_mask]
    return costs
