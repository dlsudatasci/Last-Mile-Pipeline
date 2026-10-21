# STGAT-LSTM model explanation

Input update, 20 October 2026: the active OSM CSV/GeoJSON have cleaned amenity
counts. Feature dimensions and architecture are unchanged. All previous
performance tables below use the earlier features. The cleaned-input run is
complete under `artifacts/experiments/real_cleaned_osm_20261020`: selection
validation 12/13, fitted development 63/98, exact test choices 9/14 (64.3%),
test choice NLL 0.871, and test ranking 2/3. Always-follow scores 10/14 (71.4%).
Only 1/4 actual alternatives is chosen correctly. The validation-threshold binary
follow/deviate accuracy is 11/14 (78.6%); it is not exact road-choice accuracy.
Checkpoint reload and the two route demonstrations pass; this remains an
internal pilot on three previously inspected riders. Dashboard improvements
are deferred. Follow RUNBOOK for the retained first-visit training procedure.
To replay an older checkpoint, use the matching feature CSV
under `artifacts/inputs/osm_before_amenity_cleanup`, not the active cleaned CSV.

This is the technical reference for the implemented model from inputs to route
output. It explains the exact tensors, features, equations, architecture,
losses, training protocol, evaluation, routing behavior, and limitations. Use
`PRESENTATION_GUIDE.md` to turn this material into slides and speaker notes; use
`SYNTHETIC_RUNBOOK.md` or `RUNBOOK.md` for commands.

Folder layout updated 19 October 2026: model/training/evaluation code is in
`stgat_lstm/modeling`; graph, labels, map matching, and traffic code is in
`stgat_lstm/preprocessing`; routing/dashboard code is in `stgat_lstm/routing`;
the synthetic generator is in `stgat_lstm/generation`. Data prepared by these
tools is under `generated`, saved models/run bundles are under `artifacts`, and
inspection files are under `outputs`. This relocation does not change the model.

The current refinement is under `artifacts/experiments/real_refined_20261018`. It contains
77 approved examples from 36 rides and the same 15 riders: 60 followed examples
and the original 17 deviations. The 47 additional examples are verified real
junction choices from training riders only. Four original followed examples
now use the actual suggested-route destination instead of the local review
window endpoint. Their road labels remain unchanged. The model graph has
2,625 nodes and 6,765 directed edges; the 7,441-parameter architecture is unchanged.
The refinement uses three validation-only ranking-weight trials, regularized
training with checkpoint selection, and a final refit on 12 development riders.
See Section 15 and `RUNBOOK.md`, Section 6, for the protocol.

A separate approved comparison under `artifacts/experiments/real_first_visit_20261018` is
ready with only one supervision change: keep the earliest visit to a junction
within each original training example. Three later followed targets are omitted,
leaving 85 selection-training targets and 98 fitted development targets. All
examples, alternative labels, and the 13 validation/14 test targets stay intact.
The original 101-target development set is also scored for direct comparison.
The architecture and previous settings are fixed; this is not a new parameter
search. The user's local run has now produced its checkpoint and completed
evaluation report. Its scores await review; the older tables below remain the
last reviewed results. The file reorganization does not reinterpret those scores.

Why this matters: one decision example supplies one destination and traffic
history. If its matched path leaves a junction, returns, and takes a different
exit, the present model sees the same context for both visits. It has no trip
prefix to tell them apart. The comparison keeps the first target for fitting;
it does not declare the second observation false or erase it. Different choices
across examples or riders remain in training. This supervision policy does not
remove GPS observations used to build historical speed profiles.

The original 30-example benchmark from 18 October 2026 retains 30 approved
examples from 28 rides and 15 riders, giving 68 junction targets (51 suggested,
17 alternative) and 12 optional ranking pairs. It is saved under
`artifacts/experiments/real_navigation_20261016`. Local labels no longer require rejoining,
GPS agreement with a regenerated route, or GPS after the report. Late reports
can recover turns earlier in the same active-route period. Full-data training
has completed 100 epochs. Real evaluation now supports a saved fixed split
with nine training, three validation, and three test riders, fitting five
variants once each. Section 15 separates current training results, the fixed
evaluation protocol, and archived strict-baseline results.

Recorded navigation instructions now supply additional label evidence. A
fallback resolves compass headings and road-name aliases against the active
route's outgoing alternatives, event location, and directed raw GPS movement.
It does not require an exact shared incoming HMM edge. One unique supported
choice is required; metadata, supporting later GPS, and inferred chosen roads
are not inference features. A future maneuver named in the original instruction
does not automatically identify the immediate outgoing edge. This recovered
one additional deviation: Fidel A. Reyes Street versus Castro Street.

## 1. What the model is solving

The primary task is **outgoing-road choice**: at each eligible branching node, estimate a probability distribution over the accessible outgoing roads and identify which road a motorcycle rider would prefer. A follow/deviate value is derived by comparing the preferred road with a proposed route's road. Intentional deviations also supply an auxiliary route-ranking objective.

The neural network does not directly emit a complete route. It emits edge costs. Dijkstra's algorithm constructs a connected route by minimizing their sum.

```text
OSM road graph + destination + traffic history
                    |
                    v
         shared GATv2-LSTM backbone
                    |
                    v
         utility logit for every edge
                    |
                    v
   softmax among roads from each source
                    |
          +---------+---------+
          v                   v
 preferred outgoing road   -log(probability) cost
                                |
                                v
                       Dijkstra route search
```

The corresponding model class is in `stgat_lstm/modeling/model.py`:

```python
class DecisionPreferenceModel(nn.Module):
    """Shared STGAT-LSTM that scores each accessible outgoing road."""

    def __init__(self, **preference_configuration):
        super().__init__()
        self.backbone = PreferenceModel(**preference_configuration)
        hidden = self.backbone.hidden_channels
        del self.backbone.cost_head
        self.edge_choice_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
```

**Presentation explanation:** "The model scores every accessible road leaving a junction. Softmax turns those scores into competing probabilities that sum to one. The most probable edge is the preferred road, and the same probabilities become additive routing costs."

## 2. Current assumptions

### Study and graph assumptions

- The vehicle is a **motorcycle**.
- The model graph starts with a 1 km buffer around Taft Avenue and expands to include approved path evidence and destinations. The original real set uses 1.7 km, 2,379 nodes, and 6,098 directed edges. The refinement requires 2,625 nodes and 6,765 edges. Checkpoints save the scope used during training, and prediction/dashboard reload it.
- Candidate selection has no fixed distance-to-Taft limit. The former 500 m event filter and followed-ride corridor checks were removed on 15 October 2026. Proximity counts in the network audit are descriptive only. Graph coverage is checked separately before training.
- The road network is a directed OSM `MultiDiGraph`. Parallel directed edges are distinguished by `u|v|key`.
- OSM topology, direction, and road access tags are treated as correct enough for the prototype.
- Edges marked `access=no` or `access=private` are excluded from routing.
- Turn restrictions are not represented.

The corridor extraction is explicit:

```python
def extract_taft_subgraph(graph: nx.MultiDiGraph, buffer_m: float = 1_000.0):
    taft_lines = [
        _edge_geometry(graph, str(u), str(v), data)
        for u, v, _, data in graph.edges(keys=True, data=True)
        if "taft avenue" in str(data.get("name", "")).lower()
    ]
    corridor = unary_union(taft_lines).buffer(buffer_m)
    selected = [
        node_id
        for node_id, data in graph.nodes(data=True)
        if corridor.covers(Point(float(data["x"]), float(data["y"])))
    ]
    subgraph = graph.subgraph(selected).copy()
    subgraph.graph["model_corridor_buffer_m"] = buffer_m
    return subgraph
```

### Learning assumptions

- Every supervised target names the road the rider chose from at least two accessible outgoing roads.
- Label `0` means the chosen edge equals the suggested edge; label `1` means the rider chose another outgoing edge. These binary values are derived reporting labels, not the primary training target.
- A validated intentional deviation labels the chosen outgoing road rather than the suggested outgoing road at the same branch. It does not imply that two complete paths with different destinations are comparable. Route-ranking supervision is added only when comparable paths exist.
- Followed examples supervise outgoing-road choice. They do not currently create a route-ranking pair.
- The destination is available at prediction time. A suggested route is only
  needed when scoring a reviewed example or reporting deviation from a candidate
  route; fresh origin-destination routing does not feed a suggested route into
  the neural model.
- Later GPS observations and survey answers are label evidence; they are not inference features.
- The model is shared across riders. There is no rider ID embedding or rider-specific model.
- Preference cost is additive across road edges.
- Travel time is not the target. A faster route is only a possible side effect.

The separation between inference features and supervised evidence is essential:

| Information | Neural input | Training evidence or evaluation grouping |
|---|---:|---:|
| OSM node and road attributes | Yes | No |
| Destination displacement | Yes | No |
| Pre-decision GPS-derived traffic history | Yes | No |
| Accessible outgoing-road set | Yes, through graph topology and routable mask | Defines the alternatives |
| Road the rider actually chose | No | Primary class target |
| Suggested road | No direct feature | Used to derive follow/deviate and route pair |
| Later observed GPS route | No | Establishes the chosen road and observed path |
| Survey answer and deviation reason | No | Establishes whether deviation is intentional |
| Rider identity | No | Keeps complete riders in separate evaluation folds |

This means deviation data are included in training without being leaked into
the input. The model sees the context, makes a distribution over legal roads,
and the loss compares that distribution with the road the rider chose.

The real preparation code shows which information becomes supervision:

```python
PreparedDecision(
    example_key=decision.example_key,
    rider_group=decision.rider_group,
    label=decision.label,
    suggested_edge_ids=decision.suggested_edge_ids,
    choice_targets=choice_targets,
    preferred_edge_ids=(
        decision.observed_edge_ids if decision.label == 1 and decision.has_route_pair else None
    ),
    rejected_edge_ids=(
        decision.suggested_edge_ids if decision.label == 1 and decision.has_route_pair else None
    ),
    inputs=inputs,
)
```

Current approved examples are converted into road-choice labels:

- followed examples use five consecutive suggested edges beginning at a real branching node;
- deviation examples identify the suggested and chosen outgoing edges at the same source, supported by an approach fix and at least two consecutive fixes on the alternative continuation; rejoining is not mandatory;
- the original real preparation produces 30 approved examples and 68 supervised junction choices; the refinement adds 47 genuine followed choices from training riders, yielding 77 examples and 115 choices (98 suggested roads and the same 17 alternatives);
- 12 deviations also have comparable suggested and observed paths; five supply only the outgoing-road target;
- the model returns one utility logit for every graph edge, while training normalizes and scores only the accessible alternatives at each supervised source node.

At inference, the model reports the preferred outgoing road and the complete probability distribution at each branching node on a candidate route. Non-branching roads are marked not applicable because the rider has no outgoing road choice there. For compatibility with the thesis deviation framing, the candidate road's derived deviation probability is `1 - P(candidate road)`.

### Extending the limited route-ranking supervision

The current route-ranking objective receives direct supervision only from intentional deviations. A mature dataset could also construct comparable route alternatives for followed decisions:

```text
followed decision:
    preferred = suggested continuation actually followed
    rejected  = feasible alternative continuation(s)

deviated decision:
    preferred = observed intentional continuation
    rejected  = suggested continuation and other feasible alternatives
```

Alternatives must share the same local origin, destination or horizon and must have been legal and available at decision time. They can be generated from the directed OSM graph and reviewed or filtered for plausibility.

The existing pairwise objective can then be applied to every decision:

```python
for rejected_path in feasible_alternatives:
    ranking_loss += preference_loss(
        costs,
        chosen_path,
        rejected_path,
        inputs.edge_ids,
    )
```

This would extend direct route-cost supervision beyond the 12 comparable
intentional pairs. The current 13 followed examples update the choice head without
creating ranking pairs. Multiple decision points may be extracted from one
ride, but evaluation must keep all points from the same rider, including all
their rides, in one fold.

### Current temporal-data assumption

The app requests a GPS fix every two seconds. In the current export the measured
median interval is 2.001 seconds. Consecutive fixes are direction-matched to OSM
roads and converted to historical motorcycle speed records. For each decision,
the model builds six time-of-day frames using only records that predate the
decision and excluding the target ride. During held-out-rider evaluation, the
held-out rider is also excluded.

```python
if temporal_edge_features is None:
    unknown = torch.zeros(
        (len(self.edge_ids), len(self.schema.edge_dynamic)),
        dtype=torch.float32,
    )
    age_index = self.schema.edge_dynamic.index("traffic_age_scaled")
    unknown[:, age_index] = 1.0
    temporal_edge_features = (unknown,)
```

If no GPS observations cover a road and time slot, the unknown-frame logic
applies. On the expanded 1.7 km graph the final export archive contains 7,033
matched speed segments on 176 directed roads from 69 rides. The archived 1 km
run had 6,801 segments on 161 roads. These are coverage counts, not proof that
every target has adequate history. Held-out evaluation also excludes the test
rider and must be examined separately. The presence of frames does not itself
prove an LSTM benefit.

For a new route, eligibility also depends on the requested local time of day
and weekday/weekend class. Existing daytime GPS records do not automatically
describe a midnight request. The 18 October request had observed-road counts
`[0, 0, 0, 0, 0, 0]`; a saved daytime historical replay had
`[10, 9, 0, 19, 13, 16]`, using six source rides. The zeros count missing
observations, not roads with no congestion. In those empty frames, dynamic
observation masks are zero and age is marked unknown. The model still runs,
but that request supplies no observed traffic speeds. A historical replay
demonstrates recorded context, not present traffic conditions.

## 3. Expected final outputs

For one destination and traffic context, the neural network returns:

```python
edge_costs, edge_choice_logits = model(inputs)
log_probabilities = outgoing_choice_log_probabilities(edge_choice_logits, inputs)
```

Mathematically:

- $z_e$: raw utility logit of directed edge $e$;
- $p_e=\operatorname{softmax}_{A(u)}(z_e)$: probability of choosing edge $e$ among accessible roads leaving source $u$;
- $c_e=-\log(p_e)+\epsilon_e > 0$: additive preference cost of edge $e$;
- $P^*=\arg\min_P \sum_{e\in P}c_e$: recommended connected route.

The system-level output contains:

- origin and destination;
- recommended edge sequence and road names;
- total learned preference cost;
- shortest-distance baseline;
- preferred outgoing road and candidate probabilities at every branching source;
- derived deviation probability `1 - P(candidate road)`;
- traffic coverage information;
- route geometry for visualization.

The key distinction to state is:

> The neural network predicts a probability distribution over outgoing roads. It identifies the preferred road and converts those same probabilities into costs that Dijkstra uses to produce a connected route.

Each primary probability has a local scope: it scores one accessible outgoing road relative to the alternatives at the same source, given the graph, destination, road attributes, and available pre-decision traffic context. An optional candidate-route deviation summary is:

~~~math
p_{route\ deviation}=1-\prod_k q_k,
~~~

Here $q_k$ is the probability of choosing the candidate route's outgoing edge,
conditional on reaching choice $k$. The product is a derived summary; individual
junction distributions remain more explainable and need calibration on held-out
riders.

The checkpoint contains:

```python
checkpoint = {
    "state_dict": model.state_dict(),
    "model_config": model.configuration(),
    "training_summary": asdict(summary),
    "provenance": provenance,
}
```

## 4. Variables and tensor shapes

Let:

| Symbol | Meaning | Current shape |
|---|---|---:|
| $G=(V,E)$ | Directed road graph | 2,625 nodes; 6,765 edges in the refined real run |
| $N$ | Number of nodes | 2,625 |
| $M$ | Number of directed edges | 6,765 |
| $T$ | Number of traffic snapshots | 6 with GPS history; 1 unknown frame without it |
| $F_n$ | Node feature dimension | 13 |
| $F_s$ | Static edge feature dimension | 19 |
| $F_d$ | Dynamic edge feature dimension | 5 |
| $F_q$ | Destination feature dimension | 2 |
| $H$ | Hidden dimension | 16 |
| $X$ | Node-feature matrix | $N\times13$ |
| $I$ | Edge index | $2\times M$ |
| $S$ | Static edge features | $M\times19$ |
| $D_t$ | Dynamic edge features at time $t$ | $M\times5$ |
| $Q$ | Destination-relative edge features | $M\times2$ |
| $Z_t$ | Spatial edge embeddings | $M\times16$ |
| $H_T$ | Final LSTM edge embeddings | $M\times16$ |
| $c$ | Positive edge costs derived from choice probabilities | $M$ |
| $z$ | Edge utility logits | $M$ |

These tensors are grouped by `ModelInput`:

```python
@dataclass(frozen=True)
class ModelInput:
    node_features: Tensor
    edge_index: Tensor
    edge_static: Tensor
    temporal_edge_features: tuple[Tensor, ...]
    destination_features: Tensor
    routable_mask: Tensor
    edge_ids: tuple[str, ...]
```

The deployed configuration is:

```text
architecture:              stgat_lstm
hidden_channels:           16
attention heads:           2
attention negative slope:  0.2
node_feature_dim:          13
edge_static_dim:           19
edge_dynamic_dim:          5
destination_feature_dim:   2
trainable parameters:      7,441
```

`hidden_channels=16` is a capacity choice, not a mathematical consequence of the 13 inputs. The node projection learns a matrix $W\in\mathbb{R}^{16\times13}$ and bias $b\in\mathbb{R}^{16}$, so $XW^\top+b$ changes `[N, 13]` into `[N, 16]`. Widths 15, 17, or 18 are all valid because the two attention heads are averaged with `concat=False`; 16 is a compact power-of-two baseline. On this model, widths 15, 16, 17, and 18 contain 6,691, 7,441, 8,229, and 9,055 trainable parameters respectively. A larger final dataset should tune this width on rider-disjoint validation data rather than assuming 16 is optimal.

The shape flow is:

```text
node features                         [N, 13]
    -> node projection                [N, 16]
    -> GATv2 layer 1, two heads       [N, 16]
    -> GATv2 layer 2, two heads       [N, 16]

source node + target node
+ static edge + dynamic edge
+ destination displacement           [M, 58]
    -> edge projection per time       [M, 16]
    -> stack T snapshots              [M, T, 16]
    -> LSTM final state               [M, 16]
        -> outgoing-road choice head  [M]
        -> per-source softmax          [M]
        -> -log probability costs      [M]
```

Parameter distribution:

| Component | Parameters |
|---|---:|
| Node projection | 224 |
| Two GATv2 layers | 3,808 |
| Edge projection | 944 |
| LSTM | 2,176 |
| Outgoing-road choice head | 289 |
| **Total** | **7,441** |

## 5. Input features

### Node features: 13 per intersection

```python
node=(
    "x_normalized",
    "y_normalized",
    "traffic_signal",
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
```

Coordinates are normalized within the corridor. Context counts use `log1p` and are scaled by the maximum value for that feature. This reduces the effect of highly skewed point-of-interest counts.

### Static edge features: 19 per directed road

```python
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
    "highway_motorway_trunk",
    "highway_primary",
    "highway_secondary",
    "highway_tertiary",
    "highway_residential",
    "highway_service",
    "highway_unclassified",
    "highway_living_street",
    "highway_other",
)
```

Known-value masks for lanes and maximum speed prevent a missing OSM tag from being confused with a real zero.

### Dynamic traffic features: 5 per road and time

```python
edge_dynamic=(
    "congestion_normalized",
    "speed_ratio_to_reference",
    "congestion_observed",
    "speed_observed",
    "traffic_age_scaled",
)
```

- For rider GPS, `speed_ratio_to_reference` is median observed motorcycle speed divided by the edge's OSM reference speed and capped at 2.
- For rider GPS, `congestion_normalized` is the slowdown proxy `1 - min(speed_ratio, 1)`.
- Separate masks say whether slowdown and speed were actually observed.
- GPS observation age is scaled over 30 days and capped at 1 because the frames are historical time-of-day profiles.
- Unknown traffic has zero values, zero observation masks, and age 1.

The masks are essential. Without them, a zero could incorrectly mean free flow
when there was no rider observation. These features describe motorcycle probe
speeds and must not be called complete network-wide traffic.

The GPS conversion in `GPSTrafficArchive.from_exports` uses consecutive fixes:

~~~math
v=\frac{\operatorname{haversine}(p_t,p_{t+1})}
{(t_{t+1}-t_t)/1000}.
~~~

A segment is discarded when its timestamp does not increase, its gap exceeds
20 seconds, its implied speed exceeds 35 m/s, or it cannot be direction-matched
within 60 m of an available OSM edge. Movement of at most 3 m may remain on the
last matched edge as a stationary observation. The archive reports the measured
median interval; the current app-shaped synthetic records produce 2.0 seconds.

For a request cutoff, `profile_sequence` selects only earlier records, separates
weekdays from weekends, and aggregates records within a circular time-of-day
window around each slot. The current configuration constructs six slots five
minutes apart with a five-minute profile bin. The median speed ratio for every
observed road becomes that slot's dynamic value. Missing roads keep observation
masks at zero and age at one.

### Destination features: 2 per road

For each edge, the model receives the normalized displacement from the edge’s destination node to the trip destination:

```python
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
```

This makes edge costs destination-conditioned. The same road can receive different relevance for different destinations.

## 6. Node projection

The 13-dimensional node vector is projected to the hidden dimension:

```python
self.node_projection = nn.Linear(node_feature_dim, hidden_channels)

node_embedding = F.relu(
    self.node_projection(inputs.node_features)
)
```

Mathematically:

~~~math
h_v^{(0)}=\operatorname{ReLU}(W_0x_v+b_0),\quad h_v^{(0)}\in\mathbb{R}^{16}.
~~~


For the complete matrix:

~~~math
H^{(0)}=\operatorname{ReLU}(XW^{\top}+\mathbf{1}b^{\top}),
~~~

where $X\in\mathbb{R}^{N\times13}$, $W\in\mathbb{R}^{16\times13}$, and $b\in\mathbb{R}^{16}$. The same weights are applied to every node. Each of the 16 output channels is one learned weighted combination of the 13 original features:

~~~math
h_{ik}=\max\left(0,b_k+\sum_{j=1}^{13}W_{kj}X_{ij}\right).
~~~

This layer has $16\times13+16=224$ trainable parameters. It does not create new measured information; it expresses the 13 measurements in a 16-dimensional latent space that is useful for the later attention layers.

## 7. Two-layer GATv2 spatial encoder

The current model uses the maintained PyTorch Geometric `GATv2Conv` implementation:

```python
attention_edge_dim = edge_static_dim + edge_dynamic_dim

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
```

For snapshot $t$, the attention attribute of edge $i\rightarrow j$ is:

~~~math
a_{ij}^{(t)}=[s_{ij}\Vert d_{ij}^{(t)}]\in\mathbb{R}^{24}.
~~~


A conceptual GATv2 attention score is:

~~~math
e_{ij}^{(k,t)}
=
{a_k}^{\top}
\operatorname{LeakyReLU}
\left(
W_s^{(k)}h_i+
W_t^{(k)}h_j+
W_e^{(k)}a_{ij}^{(t)}
\right).
~~~


The score is normalized across neighboring messages:

~~~math
\alpha_{ij}^{(k,t)}
=
\frac{\exp(e_{ij}^{(k,t)})}
{\sum_{r\in\mathcal N(j)}\exp(e_{rj}^{(k,t)})}.
~~~


The layer aggregates transformed neighbor information using $\alpha$. Two heads are used and averaged because `concat=False`, so the output remains 16-dimensional.

The forward code is:

```python
attention_edge_attr = torch.cat(
    (inputs.edge_static, edge_dynamic),
    dim=-1,
)
node_embedding = F.elu(
    self.spatial_1(
        node_embedding,
        inputs.edge_index,
        attention_edge_attr,
    )
)
node_embedding = F.elu(
    self.spatial_2(
        node_embedding,
        inputs.edge_index,
        attention_edge_attr,
    )
)
```

### Where LeakyReLU is used

LeakyReLU is inside `GATv2Conv` when it computes attention coefficients. The configured negative slope is 0.2. ELU is then applied to each layer’s aggregated output.

**Presentation explanation:** “LeakyReLU belongs to the attention scoring equation implemented by the library. ELU is the activation we explicitly apply after graph aggregation.”

### Why GATv2

GATv2 provides dynamic attention: the ranking of neighboring nodes can depend jointly on the query node, neighboring node, and edge context. The model uses the library operator and builds a task-specific routing architecture around it. It is not a line-for-line replication of the GATv2 research repository.

## 8. Convert node embeddings into road-edge embeddings

Routing needs one representation per road edge, so the model concatenates:

- source-node embedding: 16;
- destination-node embedding: 16;
- static edge features: 19;
- current dynamic features: 5;
- destination-relative features: 2.

The concatenated dimension is (16+16+19+5+2=58).

```python
self.edge_projection = nn.Linear(
    2 * hidden_channels
    + edge_static_dim
    + edge_dynamic_dim
    + destination_feature_dim,
    hidden_channels,
)

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
edge_embedding = F.relu(
    self.edge_projection(edge_embedding)
)
```

For edge $e=(u,v)$:

~~~math
z_e^{(t)}
=
\operatorname{ReLU}
\left(
W_E
[h_u^{(t)}\Vert h_v^{(t)}\Vert s_e\Vert d_e^{(t)}\Vert q_e]
+b_E
\right)
\in\mathbb{R}^{16}.
~~~


## 9. LSTM temporal encoder

The spatial encoder is run once per traffic snapshot. Its outputs are stacked into:

~~~math
Z\in\mathbb{R}^{M\times T\times16}.
~~~


The code treats each road edge as one sequence in the LSTM batch. This compact
snippet shows the equivalent computation; `encode_edges` now reuses spatial
work for exactly repeated frames, while preserving every temporal position:

```python
self.temporal = nn.LSTM(
    hidden_channels,
    hidden_channels,
    batch_first=True,
)

sequence = torch.stack(
    [
        self._spatial_snapshot(inputs, snapshot)
        for snapshot in inputs.temporal_edge_features
    ],
    dim=1,
)
embedding, _ = self.temporal(sequence)
last = embedding[:, -1, :]
```

The LSTM uses the standard gates:

~~~math
i_t=\sigma(W_i z_t+U_i h_{t-1}+b_i)
~~~



~~~math
f_t=\sigma(W_f z_t+U_f h_{t-1}+b_f)
~~~



~~~math
g_t=\tanh(W_g z_t+U_g h_{t-1}+b_g)
~~~



~~~math
c_t=f_t\odot c_{t-1}+i_t\odot g_t
~~~



~~~math
o_t=\sigma(W_o z_t+U_o h_{t-1}+b_o),\qquad
h_t=o_t\odot\tanh(c_t).
~~~


The final edge state $h_e^{(T)}$ is used by the outgoing-road choice head.

The current policy uses six snapshots at five-minute intervals. Each snapshot
aggregates earlier rides near the corresponding local time of day, separating
weekdays from weekends. Future records, the target ride, and held-out riders are
excluded. The controlled synthetic experiment uses the same six five-minute
frames as the official pipeline. The design follows the broad
spatial-then-temporal order of Zhang, Yu, and Liu (2019), while the prediction
target and rider-choice losses are specific to this project.

## 10. Outgoing-road choice head and routing cost

The head assigns one utility logit to every directed edge. At a source node, only accessible outgoing edges compete with one another:

```python
edge_embeddings = self.backbone.encode_edges(inputs)
edge_choice_logits = self.edge_choice_head(edge_embeddings).squeeze(-1)
log_probabilities = outgoing_choice_log_probabilities(edge_choice_logits, inputs)
edge_costs = -log_probabilities + length_tiebreaker
return edge_costs, edge_choice_logits
```

For the accessible outgoing set $A(u)$ at source node $u$:

~~~math
P(e\mid u,\text{context})=
\frac{\exp(z_e)}{\sum_{j\in A(u)}\exp(z_j)}, \qquad e\in A(u).
~~~

The probabilities at one source sum to one. The preferred outgoing road is the edge with the greatest probability. If $e_s$ is the road in a proposed route, the displayed deviation probability is a derived quantity:

~~~math
P(\text{deviate from }e_s)=1-P(e_s\mid u,\text{context}).
~~~

The model converts a choice probability into an additive routing cost:

~~~math
c_e=-\log P(e\mid u,\text{context})
+10^{-4}\frac{\operatorname{length}_e}{100\text{ m}}+10^{-6}.
~~~

This cost is positive and dimensionless. The tiny length term only breaks ties and does not make travel time the objective. Restricted edges receive a prohibitive cost and are also excluded by the router.

## 11. Outgoing-road supervision

Each `PreparedRoadChoice` stores the source node, all accessible outgoing candidate edges, the road actually chosen by the rider, and the road that had been suggested. A followed observation has the same chosen and suggested edge. At the first supported divergence, the chosen edge is the observed alternative and the suggested edge is the rejected road.

Training uses multiclass cross-entropy within each candidate set. The original
baseline weights alternative-road targets by `followed choices / alternative
choices`. The refinement first reduces the influence of rides with many choices,
then computes that ratio from the ride-weighted counts. The unsmoothed choice
loss is:

```python
candidate_logits = edge_logits[candidate_indices]
choice_loss = F.cross_entropy(
    candidate_logits.unsqueeze(0),
    torch.tensor([chosen_position]),
)
```

For chosen edge $e^*$:

~~~math
L_{choice}=-\log P(e^*\mid u,\text{context}).
~~~

This directly teaches which outgoing road is preferred. Follow/deviate remains available for reporting: label 0 means $e^*=e_s$ and label 1 means $e^*\ne e_s$.

The refinement uses label smoothing $\epsilon=0.05$. With $K_i$ candidates,
target $y_i$, ride choice count $n_i$, and alternative-choice multiplier $a$:

~~~math
\tilde q_{ij}=(1-\epsilon)\mathbf 1[j=y_i]+\epsilon/K_i,
\qquad L_i=-\sum_j\tilde q_{ij}\log p_{ij},
\qquad w_i=\frac{a^{\mathbf 1[\text{alternative}_i]}}{n_i}.
~~~

The batch choice loss is $\sum_i w_i L_i / \sum_i w_i$. The multiplier $a$ is
computed from training targets only. GPS and survey evidence identify $y_i$;
they are not appended to the inference feature vector.

## 12. Choice and route-ranking loss

### Route-preference ranking loss

For an intentional deviation:

~~~math
C(P)=\sum_{e\in P}c_e.
~~~



~~~math
L_{rank}
=
\operatorname{softplus}
\left(
C(P_{observed})-C(P_{suggested})
\right).
~~~


The exact implementation is:

```python
def preference_loss(
    edge_costs,
    preferred,
    rejected,
    available,
):
    preferred_cost = path_cost(
        edge_costs, preferred, available
    )
    rejected_cost = path_cost(
        edge_costs, rejected, available
    )
    return F.softplus(
        preferred_cost - rejected_cost
    )
```

Minimizing this loss encourages:

~~~math
C(P_{observed})<C(P_{suggested}).
~~~


### Combined loss

For example (i):

~~~math
L_i
=
L_{choice,i}
+
\mathbb 1[\text{preference pair exists}]
\lambda L_{rank,i}.
~~~


The implementation uses $\lambda=1$ by default. The controlled synthetic
validation selected $\lambda=0.1$ together with a 1.0 alternative-choice
weight. Junction losses are weighted before averaging; route-ranking losses are
then averaged and added:

```python
losses = torch.stack(choice_losses)
weights = losses.new_tensor(choice_weights)
choice_loss = (losses * weights).sum() / weights.sum()
loss = choice_loss
if ranking_losses:
    loss = loss + preference_weight * torch.stack(ranking_losses).mean()
```

The GATv2-LSTM backbone and choice head receive gradients from both terms because the routing costs are derived from the same normalized choice probabilities. The original real artifact supplies 68 junction choices and 12 route-ranking pairs. The refined artifact has 115 choices and the same 12 pairs; added follows do not create ranking pairs. Examples without a comparable pair still train through choice cross-entropy. When the validation-selected ranking weight is zero, only the choice term trains the model.

## 13. Optimization

```python
torch.manual_seed(seed)
model = DecisionPreferenceModel(**model_configuration)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

for epoch in range(epochs):
    model.train()
    for batch in shuffled_decision_batches:
        # Disconnected graph components share no messages.
        optimizer.zero_grad()
        loss = outgoing_choice_and_ranking_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
```

Original CLI defaults and synthetic settings:

| Hyperparameter | CLI default | Controlled synthetic final checkpoint |
|---|---:|---:|
| Epochs | 40 | 10, selected through the evaluation report |
| Learning rate | 0.005 | 0.005 |
| Optimizer | Adam | Adam |
| Route-ranking weight | 1.0 | 0.1 |
| Alternative-choice weight | inverse class frequency | 1.0, selected on validation riders |
| Seed | 17 | 17 |
| Hidden channels | 16 | 16 |
| Dropout | none | none |
| Batch size | 8 disconnected graph examples | 8 |

The standalone training command fits every example in the approved artifact supplied on
the command line. The original real artifact contains 30 examples; the
controlled synthetic artifact contains 144. This command produces a deployable
checkpoint, but its in-sample summary is not held-out accuracy. Evaluation is a
separate command.

### Approved real-data refinement

`evaluate --refine` coordinates selection and training inside `evaluate_model.py`.
It calls `train_decision_model` in `train_model.py`; `model.py` defines the same
two-layer GATv2 and LSTM used by the original run. It does not use the standalone
training CLI defaults above.

| Setting | Real refinement |
|---|---|
| Optimizer / learning rate / weight decay | AdamW / 0.001 / 0.0001 |
| Choice-label smoothing | 0.05 |
| Training traffic masking | 0.15 probability per observed edge/frame |
| Ride influence | Divide each choice weight by the number of choices from its ride |
| Alternative-choice weight | Ratio of followed/alternative counts after applying ride weights |
| Ranking weights compared | 0, 0.1, 1 |
| Validation selection | Highest outgoing-choice accuracy, then lower choice NLL |
| Validation schedule | Epoch 1, every five epochs, and the final epoch |
| Early stopping | Three validation checks without improvement, at most 60 epochs |
| Final refit | All 12 development riders, at the selected fixed epoch count |
| Seed / hidden channels / batch size | 17 / 16 / 8 |

Label smoothing spreads 5% of the target distribution uniformly across available
outgoing candidates. Traffic masking hides some existing observations during
training by setting the whole dynamic row to its unknown representation. It
does not invent speeds, alter road labels, or modify validation/test inputs.
Weight decay limits parameter growth; none of these measures guarantees higher
test accuracy.

The three trials see nine training riders and three validation riders. The final
refit includes those 12 development riders, but never the three test riders.
During selection, GPS profiles exclude all validation/test riders. During the
final refit and test, they exclude all test riders. Both stages exclude the
target ride and observations after the decision time.

The approved speed optimization in `PreferenceModel.encode_edges` reuses a
spatial embedding only when two complete traffic frames are exactly equal.
Their positions in the six-step LSTM sequence are retained. The reuse is local
to one forward call, so every parameter update and new request recomputes the
required embeddings. Frames needing their own input gradients are computed
separately. Output and parameter-gradient equivalence are tested; this changes
computation time, not the model's architecture or objective.

`evaluation_report.json` also contains `development` and `overfitting_checks`.
For each retained trial, compare its training choice accuracy with validation
accuracy and inspect the saved validation history. For the final checkpoint,
compare clean development/test choice accuracy and NLL. Both NLL values use the
same unweighted, unsmoothed definition; the optimization loss includes smoothing,
class/ride weights, and possibly ranking, so it is not directly comparable with
validation NLL. A widening performance gap is evidence to investigate, not a
formal diagnosis on its own. Small gaps do not prove no overfitting. The three
validation riders are training data after the final refit. The test still has
only three previously inspected riders, so this is an internal pilot rather
than a fresh independent validation.

This refinement evaluates ST-GAT-LSTM and an always-follow baseline. The older
five-architecture table uses the original data preparation and training settings;
it is historical context, not a controlled comparison against the refined model.
These results cannot isolate the benefit of the LSTM or of any one refinement.

## 14. From learned costs to a route

For a new origin and destination, the system creates a shortest-distance baseline for comparison and builds one graph input for the requested destination:

```python
baseline = shortest_real_preference_path(
    graph_data,
    distance_costs,
    origin_node_id,
    destination_node_id,
)

inputs = graph_data.build_model_input(
    destination_node_id,
    temporal_edge_features=temporal_edge_features,
)

with torch.no_grad():
    scores, edge_choice_logits = model(inputs)
```

It then runs Dijkstra using the learned positive costs:

```python
costs = {
    edge_id: float(scores[row])
    for row, edge_id in enumerate(graph_data.edge_ids)
}

learned = shortest_real_preference_path(
    graph_data,
    costs,
    origin_node_id,
    destination_node_id,
)
```

The router explicitly removes restricted edges and validates every cost:

```python
for index, edge_id in enumerate(graph_data.edge_ids):
    if float(
        graph_data.edge_static[index, restricted_index]
    ) == 1.0:
        continue
    cost = float(edge_costs[edge_id])
    if not math.isfinite(cost) or cost <= 0.0:
        raise ValueError(
            "Edge has a nonpositive or nonfinite preference cost"
        )
```

The preferred outgoing road and the routing cost come from the same local probability distribution. The derived deviation probability only restates how much probability lies outside the candidate route's outgoing edge.

The route layer also compares the unconstrained learned route with the
shortest-distance baseline. With the default maximum detour ratio $r=1.30$:

~~~math
\text{accept learned route if }d(P_{learned})\leq r\,d(P_{shortest});
\quad\text{otherwise return }P_{shortest}.
~~~

This is a safety guard around the prototype cost model. It does not change the
neural probabilities and does not enter the training loss.

For a traffic-aware request, `--traffic-cutoff-ms` defines the information time.
`GPSTrafficArchive.profile_sequence` builds the six historical frames from
records before that time. Changing the cutoff does not retrain the checkpoint;
it changes the input context and therefore may change road probabilities,
costs, and the resulting route.

## 15. Training output versus evaluation output

### Current real refinement: completed

Artifacts are under `artifacts/experiments/real_refined_20261018`. Three validation-only trials
selected ranking weight 1.0 at epoch 45. That nine-rider checkpoint scores
12/13 validation roads correctly. The final checkpoint is a separate fit on
all 12 development riders for 45 epochs; its results are:

| Measure | Final development | Held-out test | Always-follow test baseline |
|---|---:|---:|---:|
| Riders | 12 | 3 | 3 |
| Exact outgoing roads correct | 67/101 | 9/14 | 10/14 |
| Outgoing-road accuracy | 66.3% | 64.3% | 71.4% |
| Clean choice NLL | 0.813 | 0.911 | 0.911 |
| Comparable route pairs correct | 8/9 | 2/3 | n/a |

Test binary balanced accuracy is 0.650 and deviation F1 is 0.500. Exact road
identification is eight of ten followed roads and one of four alternative
roads. Binary detection counts two true positives, but one identifies the
wrong alternative: detecting that a rider rejects a suggestion is not the same
as selecting their actual outgoing road. Test per-rider results are 1/1, 8/11,
and 0/2, giving macro rider accuracy 0.576.

The development/test accuracy gap is 2.05 percentage points; the NLL gap is
0.099. This is not a large training/test accuracy gap, but it does not prove
overfitting is absent. Low development accuracy can reflect limited fit,
objective tradeoffs, or heterogeneous choices; these results alone do not
identify a cause. The selection curve reaches 12/13 at epoch 45 and drops to
10/13 at epoch 60, illustrating why the earlier checkpoint was retained.

The roughly 80% goal is not met. Relative to the older internal model, NLL
improves from 3.373 to 0.911 and ranking from 1/3 to 2/3, while choice accuracy
drops from 10/14 to 9/14. The refined model's NLL is essentially tied with the
current baseline. These are mixed results, not a validated reliable preference
model or a controlled demonstration of LSTM superiority. The same 14 target
roads, candidate sets, and rider groups are preserved, but destination context
and development preparation changed. No settings were retuned using this final
test result.

Checkpoint reload reproduces all 14 held-out predictions with maximum
probability difference zero. All 89 tests pass. A historical fresh route from
400 to 6 is directed and connected, respects the access mask, and is 1,365.8 m
versus the 1,320.5 m distance baseline. It passes the 1.30 guard without fallback;
this is a routing-operation check, not ground-truth evidence of route quality.

`run_summary.json` gives the concise results, `evaluation_report.json` includes
the full development/test and overfitting diagnostics, and `run_manifest.json`
records frozen inputs and fitting/post-fit source hashes. The approved speed
optimization and requested diagnostic reporting were added after the running
fits loaded their code; reload equivalence is verified with the optimized model.

### Training report

The original full-data `artifacts/experiments/real_navigation_20261016/training_report.json` records:

- 30 approved examples from 28 rides and 15 riders;
- 68 junction targets: 51 suggested roads and 17 alternatives;
- 100 epochs, loss 1.8917 to 0.2863;
- 62/68 in-sample outgoing roads correct (91.2%);
- 11/12 in-sample route pairs correctly ranked;
- successful checkpoint reload, paired/inferred replays, and connected fresh routing.

These demonstrate training fit and pipeline operation. They do not measure
unseen-rider generalization. Full-data training fits one model. The optional
full held-out benchmark fits five variants in each of 15 rider folds, up to
75 fits; it was stopped before a new report was produced. The earlier archived
ten-rider benchmark used 50 fits. These fits do not create 75 deployment files.

### Archived strict-baseline training

The archived strict-baseline GPS-history complete-data training run reports:

- 16 approved examples from 10 riders;
- 54 outgoing-road choices: 51 suggested roads chosen and 3 alternatives chosen;
- loss reduced from 2.0788 to 0.2040 after 100 epochs;
- 43/54 in-sample preferred outgoing roads correct;
- 3/3 in-sample deviation pairs ranked correctly.

These numbers show that the implementation can fit the current examples. They are not evidence of unseen-rider generalization.

### Original fixed real rider split

The original baseline command uses `--fixed-split --temporal-ablation`.
`fixed_rider_assignments` groups the current 15 riders into nine training,
three validation, and three test riders. A deterministic search balances
label support and example counts before any fitting. It does not inspect
model predictions. The saved file binds assignments to the approved-data hash
and seed, refusing silent reuse for changed data.

The current split contains 18/6/6 examples and 41/13/14 junction choices in
training/validation/test respectively. All records from validation and test
riders are excluded from GPS profile construction. Every profile also excludes
the target ride and observations after its decision cutoff. The known static
OSM graph can contain roads used by any group; this is rider generalization,
not an unseen-road test.

```python
partitions = fixed_rider_partitions(prepared, assignments)
excluded_riders = {
    rider for rider, group in assignments.items()
    if group != "training"
}
prepared, traffic_reports = attach_gps_traffic(
    prepared, gps_archive, excluded_riders=excluded_riders,
    history_steps=6, history_interval_s=300, gps_profile_bin_s=300,
)
```

GCN, GAT, GATv2, ST-GAT-LSTM, and latest-only LSTM are each fitted once
from scratch using training riders only. The latest-only control repeats the
last frame six times and retains the full LSTM's parameter count and seed.
The existing full-data demonstration checkpoint is not used to score the test
set. Epochs, learning rate, and preference weight are fixed in advance, with no
test-based tuning. Any alternative-road class weight is calculated from the
training partition only.

Outgoing-road argmax accuracy and negative log-likelihood are primary. The
main binary metrics use a fixed 0.5 threshold; a separate JSON field reports
metrics at a threshold selected using validation riders only. Train,
validation, test, and per-rider scores remain separate. A correct evaluation
protocol does not guarantee a good model: three test riders and 14 correlated
choices provide preliminary evidence only.

The completed 18 October experiment is saved in
`artifacts/experiments/real_fixed_split_20261018/evaluation_report.json`; the shorter
`run_summary.json` records the interpretation. All five variants completed
100 epochs. Main binary metrics use threshold 0.5.

| Model | Training choice accuracy | Validation choice accuracy | Test choice accuracy | Test choice NLL | Test balanced accuracy | Test deviation F1 | Test ranking |
|---|---:|---:|---:|---:|---:|---:|---:|
| Suggestion baseline | 0.756 | 0.769 | 0.714 | 0.830 | 0.500 | 0.000 | n/a |
| GCN | 0.756 | 0.923 | 0.643 | 2.940 | 0.600 | 0.444 | 0.667 |
| GAT | 0.854 | 0.846 | 0.643 | 4.175 | 0.650 | 0.500 | 0.333 |
| GATv2 | 0.829 | 0.846 | 0.643 | 3.562 | 0.600 | 0.444 | 0.333 |
| ST-GAT-LSTM | 0.878 | 0.923 | 0.714 | 3.373 | 0.650 | 0.500 | 0.333 |
| LSTM latest-only | 0.854 | 0.769 | 0.643 | 3.974 | 0.525 | 0.286 | 0.333 |

ST-GAT-LSTM chooses 10/14 roads correctly: 8/10 followed roads and 2/4
alternatives. Its binary confusion has two true positives, eight true
negatives, two false positives, and two false negatives. Always following
gets all ten follows but none of the alternatives. The full model's training
choice accuracy is 36/41 (87.8%), compared with 10/14 (71.4%) on test riders.
This gap and the test errors prevent a claim that overfitting is ruled out.

Choice NLL is worse than baseline (3.373 versus 0.830; lower is better),
showing that some wrong predictions are very confident. Only one of three
test route pairs is correctly ranked. Per-rider choice results are 1/1,
9/11, and 0/2, giving macro rider accuracy 0.606. The apparent history
advantage is only one additional correct road versus latest-only; it does not
establish a stable LSTM improvement.

After the GPS exclusions, training has observed history at 16/18 examples,
validation at 1/6, and test at 5/6. Observed values change somewhere in the
graph at 11/18, 0/6, and 4/6 respectively; changes on the suggested paths occur
at only 2/18, 0/6, and 1/6. These coverage differences limit temporal claims.
The result supports an internally evaluated prototype with mixed performance,
not a reliable or state-of-the-art rider-preference model.

### What validated and reliable mean for this thesis

A valid internal evaluation keeps complete riders separate, builds inputs
without post-decision label evidence, fixes settings before testing, and reports
test results separately from training fit. This follows the principles in the
[grouped cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
and [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
The fixed split supplies this internal test protocol; the score can still be
poor, and the same dataset has already been used during project development.
It is not an independent prospective field trial.

A useful rider-preference model should choose roads accurately for held-out
riders and capture intentional alternatives beyond an always-follow baseline.
Its probability quality, missed alternatives, false alarms, and per-rider
variation matter alongside overall accuracy. With ten followed and four
alternative test choices, always following already achieves 10/14 = 71.4%.
An arbitrary target such as 80% accuracy is not sufficient by itself.

Stronger reliability evidence would include more independently held-out
riders and repeated conditions, stability across training seeds, and routes
that agree with rider preferences while staying connected and within the
distance guard. A claimed temporal benefit also requires observed relevant
histories and improvement over the matched latest-only LSTM control. A claim
that displayed probabilities are calibrated requires separate calibration
evidence. For an undergraduate thesis, report the implemented pipeline and
the limitations of its internal pilot without claiming deployment reliability.

### Archived leave-one-rider-out evaluation

Evaluation holds out each complete rider:

```python
for rider in riders:
    train = [
        example for example in prepared
        if example.rider_group != rider
    ]
    test = [
        example for example in prepared
        if example.rider_group == rider
    ]
```

The comparison includes:

- suggestion-following baseline;
- GCN;
- original GAT;
- spatial-only GATv2;
- GATv2-LSTM;
- latest-only LSTM control when actual temporal variation exists.

The final export supplied on 14 October 2026 is evaluated in
`artifacts/experiments/real_final_20261014/rider_holdout_evaluation.json`. It uses 16 approved
examples, 54 junction choices, and ten leave-one-rider-out folds. Every fold
trains each learned model for 40 epochs under the existing runbook settings.
The full-data demonstration checkpoint uses 100 epochs; its training accuracy
is not the held-out metric.

The suggestion-following baseline has 0.944 choice accuracy because 51 of 54
targets follow the suggestion, but it misses every deviation. Only three
alternative-road choices remained under the former strict reconstruction rules.
This class imbalance makes choice accuracy alone inadequate for judging the
learned model. Read its deviation F1, balanced accuracy, probability quality,
and route-ranking result alongside the exact-choice metric.

The completed archived strict-baseline held-out results are:

| Model | Choice accuracy | Choice NLL | Balanced accuracy | Deviation F1 | Route ranking |
|---|---:|---:|---:|---:|---:|
| Suggestion baseline | 0.944 | 0.940 | 0.500 | 0.000 | n/a |
| GCN | 0.593 | 0.938 | 0.784 | 0.214 | 0.333 |
| GAT | 0.574 | 0.898 | 0.794 | 0.222 | 0.333 |
| GATv2 | 0.667 | 1.177 | 0.814 | 0.240 | 0.667 |
| ST-GAT-LSTM | 0.556 | 1.084 | 0.745 | 0.188 | 1.000 |
| LSTM latest-only | 0.537 | 1.144 | 0.745 | 0.188 | 0.333 |

ST-GAT-LSTM chooses the exact outgoing road correctly on 30/54 held-out
targets. In the derived binary view it detects all three deviations but falsely
flags 26 of 51 followed choices, giving precision 0.103 and F1 0.188. Spatial
GATv2 performs better on choice accuracy, balanced accuracy, and deviation F1.
The full LSTM has better AP/AUC and route ranking than latest-only, but the same
binary confusion counts. These mixed results do not establish an overall real
LSTM advantage.

After excluding the held-out rider, 14/16 examples have some observed history,
ten have changing observations somewhere in the graph, and only two have
changing observations on their suggested path. All route-ranking scores are
based on just three pairs. Both tested fresh route requests triggered the
detour fallback; thus the successful route execution is not evidence of good
learned route quality.

The correct interpretation is:

> The pipeline is operational on the final export, but 54 correlated junction choices from 16 examples, including only three choices that reject the suggestion, provide limited evidence for real rider generalization or an LSTM benefit.

### Controlled hotspot capacity check

The separated synthetic experiment preserves the app's six CSV schemas and
two-second GPS structure. It creates 144 labeled decisions from 12 artificial
riders: 72 deviations and 72 follows at the same two graph-valid local Taft
hotspots. Another 144 probe rides provide six-frame worsening and recovering
speed histories. The two histories have the same final speed. Their separate
archive prevents decision GPS from becoming an inference feature. All 144
decision rides survive the same directed map matching, candidate construction,
and automatic validation used for official data.

The fixed split uses eight artificial riders for training, two for validation,
and two new riders for final testing. Four alternative-road loss weights are
compared using validation riders only; validation selected 1.0x. On 78
untouched test junction choices, ST-GAT-LSTM obtains 0.846 outgoing-choice
accuracy, 0.955 balanced accuracy, 0.800 deviation F1, 0.955 ROC-AUC, 0.833
average precision, and 1.000 route ranking. It detects all 12 deviations and
produces six false positives. GATv2 and latest-only each obtain 0.769 choice
accuracy, with balanced accuracies of 0.886 and 0.864. The prevalence baseline
also obtains 0.846 choice accuracy, but only by always following; its deviation
F1 is zero and its choice negative log-likelihood is 0.902 versus 0.282 for
ST-GAT-LSTM. Synthetic performance is controlled capacity evidence, not
real-rider evidence.

## 16. Current implementation limits

After reverting the recent augmentation work, the preserved real-only checkpoint
was reassessed on 20 October 2026. Fresh inference reproduces its saved 14 test
predictions and probabilities exactly. Exact choice is 9/14 (64.3%), with
1/4 alternative roads and 8/10 followed roads correct. Always-follow scores
10/14 overall, with no alternatives correct. The model's choice NLL is slightly
better, 0.871 versus 0.906. The 78.6% thresholded follow/deviate score is a
different task from selecting the correct directed outgoing road.

The final model fits 12/13 development alternatives but only 51/85 followed
choices. Its overall 63/98 fitting accuracy equals test accuracy; neither
equality nor the high alternative fitting accuracy establishes generalization.
The 12/13 validation result belongs to the nine-rider selection fit, before the
final checkpoint was refitted on twelve development riders.

The fresh input audit finds one pair of conflicting labels among 98 fitted
choices. Its optimistic observed-row ceiling is 97/98, so exact input conflicts
cannot explain all 35 fitting errors. Additional optimization, loss-balance,
input-context, or distribution limitations remain possible; this audit does
not identify a single cause or justify a test-driven parameter change.

The four actual test deviations illustrate different outcomes:

| Junction | Recorded outgoing road | Predicted outgoing road | Evidence |
|---|---|---|---|
| 5144 | San Marcelino Street | Nakpil Street | Four fitted choices at this junction all chose Nakpil; the held-out exit has no fitted choice support there |
| 149 | Taft Avenue, exit toward node 150 | Taft Avenue, exit toward node 12254 | Different directed exits, roughly opposite directions; a shared street name does not make the prediction correct |
| 5133 | Leon Guinto Street | Leon Guinto Street | Correct, with no observed local traffic in its incoming two-hop context |
| 5132 | Taft Avenue | Remedios Street | Wrong despite some fitted support for the recorded exit |

Two cases have no direct candidate-road speed observations across all six
frames; the other two have only one frame with one observed candidate road.
Nearby incoming-road observations provide limited additional context. This
coverage is inadequate to infer temporal value from the architecture alone.
Eleven of fourteen test choices belong to one rider, further limiting the
meaning of the aggregate score. A current-data spatial-only comparison remains
necessary before claiming an LSTM accuracy benefit.

State these directly if asked:

1. **Very small real set:** the refinement has 77 approved examples and 115 correlated junction choices, including only 17 alternatives. Its 47 added followed choices come from existing training riders. The fixed split still leaves only three test riders and 14 test choices. One inferred choice has recorded navigation and local geometry provenance.
2. **Only 12 real route-ranking pairs:** the auxiliary route-level objective has limited direct supervision on official data. Five other deviations train the choice objective without a route pair.
3. **Sparse real historical GPS context:** temporal frames now exist, but coverage at the approved decisions is too uneven to establish a real-data LSTM improvement.
4. **No rider embedding:** behavior is learned across riders rather than personalized to one rider.
5. **No turn restrictions:** routing is edge-additive.
6. **No calibrated real probability:** only 17 official supervised choices reject the suggested road.
7. **Small validation and test sets:** real data now have separate validation/test riders, but repeated tuning against the same test scores would invalidate the intended independent test. The synthetic experiment remains a separate capacity check.
8. **No travel-time objective:** the learned route represents preference cost. A hard 1.30-times-shortest-distance guard prevents excessive detours.
9. **Limited route context:** the road head sees graph, destination, edge, and traffic context, but it does not yet encode a rider-specific history or the complete prefix of the current trip.

## 17. Compact top-to-bottom explanation

Use this when asked “How does the whole model work?”

> We represent Taft Avenue as a directed OSM graph. Every intersection has normalized location and surrounding-context features. Every road has static OSM attributes, optional timestamped traffic attributes, and a vector toward the destination. For every traffic snapshot, two GATv2 layers aggregate neighboring context using node and edge attributes. We form a directed-road embedding from its endpoint nodes, road attributes, traffic, and destination direction, then an LSTM processes each road through time. The choice head assigns one utility logit to every edge. At a junction, softmax normalizes the accessible outgoing edges and multiclass cross-entropy teaches the road the rider actually chose. The same probability becomes an additive `-log(probability)` routing cost, and an auxiliary ranking loss favors an intentionally chosen route over the rejected suggestion. Dijkstra finds the connected route with the lowest total cost, subject to a 1.30 detour guard.

## 18. Code files to open during the presentation

| Explanation | File and function |
|---|---|
| Feature definitions and tensor construction | `graph_data.py:build_real_graph_data_from_graph` |
| Destination and unknown-traffic input | `graph_data.py:RealGraphData.build_model_input` |
| GATv2 and LSTM architecture | `model.py:PreferenceModel` |
| Choice logits and derived routing costs | `model.py:DecisionPreferenceModel` |
| Outgoing-choice and ranking loss | `train_model.py:train_decision_model` |
| Real approved-example preparation | `train_model.py:prepare_real` |
| GPS-to-road temporal profiles | `gps_traffic.py:GPSTrafficArchive` |
| Controlled hotspot capacity test | `synthetic_hotspot.py:generate_hotspot_experiment` |
| Real fixed rider assignments and metrics | `evaluate_model.py:fixed_rider_assignments`, `evaluate_fixed_rider_split` |
| Optional leave-one-rider-out metrics | `evaluate_model.py:evaluate_rider_disjoint` |
| Three-fold rider evaluation with inner validation | `evaluate_model.py:run_grouped_kfold`, `saved_grouped_rider_folds` |
| Model-to-Dijkstra connection | `predict_route.py:route_request` |
| Routing implementation | `predict_route.py:shortest_real_preference_path` |

For the complete command flow and every supporting file, use
`PRESENTATION_GUIDE.md`. For exact synthetic commands, use
`SYNTHETIC_RUNBOOK.md`.

The architecture is a task-specific composition. GATv2 comes from Brody et al.
and is used through PyTorch Geometric. The LSTM is PyTorch's standard
implementation following Hochreiter and Schmidhuber (1997). The
spatial-then-temporal order is closest to Zhang, Yu, and Liu, *Spatial-Temporal
Graph Attention Networks: A Deep Learning Approach for Traffic Forecasting*
(2019), DOI 10.1109/ACCESS.2019.2953888. The outgoing-road choice head,
destination-conditioned edge construction, GPS-derived motorcycle profiles,
pairwise preference objective, and learned-cost routing connection are
project-specific.

## 19. Next evaluation: pooled held-out rider choices

The fixed-split baseline remains 1/4 exact alternative roads and 8/10 followed
roads. Detecting deviation is a different task: a wrong alternative can reject
the suggestion but still miss the road/direction the rider took.

`evaluate --group-k-folds 3` rotates complete riders through three outer test
groups. Each fold selects epochs on an inner rider validation group, then fits
all development riders before testing. The same original decisions are tested
once each across the folds, including the current 17 deviations from nine
riders. Extra recovered real followed examples remain fitting-only. Fold-safe
GPS histories exclude held-out riders, target rides and future records.

The default GATv2/GATv2-LSTM comparison requires 12 fits. It changes evaluation,
not the architecture or losses. Model selection still uses overall exact-road
validation accuracy; a deviation-balanced selection criterion has not been
implemented. See RUNBOOK Section 10 for the single command and report fields.
Completed and reviewed on 21 October 2026: STGAT-LSTM selects 10/17 actual
alternatives and 17/51 followed roads (27/68 overall, 39.7%). GATv2 selects
5/17 alternatives and 28/51 follows (33/68 overall, 48.5%). Always-follow gets
51/68 overall but none of the alternatives. The LSTM selected epochs are
45/5/1; seven actual deviation exits have no chosen-exit development supervision
in their folds. The result is mixed and does not establish temporal benefit or
reliability. The earlier 1/4 result uses a different fixed-split protocol.
Report this as internal cross-validation on previously inspected data, not an
untouched external test.

The model scores outgoing roads at a known junction and destination. OSM road
and surrounding features and available pre-decision GPS traffic are inputs.
The actual observed exit and later survey reason supervise learning. The
approaching road, rider-specific history and full trip prefix are not explicit
features in the current architecture. Encoding them would be a separate model
change; supplying a later survey reason to predict that same turn would leak
the outcome.

## 20. Approved training ablation, 21 October 2026

The next experiment reuses the completed k-fold corpus and exact rider groups.
Only training and selection are varied: current choice-plus-ranking losses,
choice-only loss, and choice-only with square-root automatic deviation weight.
All three choose epochs by lower inner-validation choice NLL. Inner validation
then chooses one recipe; only that recipe is refitted and tested in each fold.
STGAT-LSTM alone requires 12 fits. Full real-data results are pending.

For a validation choice with candidate roads C and actual road y, the selection
loss is `-log P(y | C)`, averaged across validation targets. It is unsmoothed and
unweighted for evaluation, even though fitting retains its training weights and
label smoothing. The old ordinary commands still prioritize validation accuracy.
The new reduced recipe changes the automatic deviation multiplier from w to
sqrt(w), computed from fitting data only. Ride balancing remains enabled.

The GATv2-LSTM architecture, input features, GPS histories and routing function
are unchanged. No approaching-road or personal-history features were added.
The earlier GATv2 metrics used the old selection policy; they remain historical
references, not a matched architecture comparison with this revised policy.
Inspect both exact deviation and followed-road performance before claiming a
better model. Selecting an early epoch can still be legitimate; this experiment
does not force longer training or guarantee an accuracy increase.

Code: `train_model.py:train_decision_model` has opt-in `checkpoint_selection`
and `alternative_weight_power` arguments. `evaluate_model.py:run_grouped_kfold`
accepts the completed `ablation_reference`; `select_training_recipe` uses inner
validation metrics only. Ordinary defaults remain accuracy selection and power
1. The exact command, outputs and rollback instructions are in RUNBOOK Section 11.
