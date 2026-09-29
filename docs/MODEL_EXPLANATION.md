# STGAT-LSTM model explanation

This guide explains the implemented model from inputs to route output. Each part includes the code worth showing, the mathematical interpretation, and the point to say during a presentation.

## 1. What the model is solving

The project has two related prediction tasks:

1. **Road-deviation prediction:** at every eligible road choice, estimate whether a motorcycle rider will reject the suggested outgoing road.
2. **Route preference learning:** assign a positive preference cost to every directed road edge so an observed intentional route receives a lower total cost than the rejected suggestion.

The neural network does not directly emit a complete route. It emits edge costs. Dijkstra's algorithm constructs a connected route by minimizing their sum.

```text
OSM road graph + destination + traffic history
                    |
                    v
         shared GATv2-LSTM backbone
                    |
          +---------+---------+
          |                   |
          v                   v
probability per road    cost for every edge
                              |
                              v
                     Dijkstra route search
```

The corresponding model class is in `stgat_lstm/model.py`:

```python
class DecisionPreferenceModel(nn.Module):
    """Shared STGAT-LSTM with edge-deviation and routing-cost heads."""

    def __init__(self, **preference_configuration):
        super().__init__()
        self.backbone = PreferenceModel(**preference_configuration)
        hidden = self.backbone.hidden_channels
        self.edge_deviation_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
```

**Presentation explanation:** “One shared representation supports two tasks. The classification head predicts deviation, while the cost head learns which roads riders prefer. Sharing the backbone lets both labels update the graph representation.”

## 2. Current assumptions

### Study and graph assumptions

- The vehicle is a **motorcycle**.
- The current study graph is the Taft Avenue corridor, extracted within a 1 km buffer of roads named Taft Avenue.
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
    return graph.subgraph(selected).copy()
```

### Learning assumptions

- Label `0` means **followed**; label `1` means **deviated**.
- A road label applies to a suggested edge whose source node has at least two accessible successor roads.
- A followed segment supplies label `0` at every observed choice edge. A deviation supplies label `1` to the first rejected suggested edge; any verified earlier choice edges supply label `0`.
- A reviewed intentional deviation means the observed route is preferred over the suggested route for that event.
- Followed examples supervise the classification head. They do not currently create a route-ranking pair.
- The suggested route and destination are available at prediction time.
- Later GPS observations and survey answers are label evidence; they are not inference features.
- The model is shared across riders. There is no rider ID embedding or rider-specific model.
- Preference cost is additive across road edges.
- Travel time is not the target. A faster route is only a possible side effect.

The real preparation code shows which information becomes supervision:

```python
PreparedDecision(
    example_key=decision.example_key,
    rider_group=decision.rider_group,
    label=decision.label,
    suggested_edge_ids=decision.suggested_edge_ids,
    edge_targets=edge_targets,
    preferred_edge_ids=decision.observed_edge_ids if decision.label == 1 else None,
    rejected_edge_ids=decision.suggested_edge_ids if decision.label == 1 else None,
    inputs=inputs,
)
```

Current approved examples are converted into road-choice labels:

- followed examples use five consecutive suggested edges beginning at a real branching node;
- deviation examples use comparable suggested and observed subpaths between divergence and rejoin;
- the current eight examples produce 28 supervised road choices: 25 followed-road labels and three rejected-road labels;
- the model returns one road-deviation logit for every graph edge, while only eligible observed choice edges enter the classification loss.

At inference, the model reports the conditional deviation probability at each branching road on a candidate route. Non-branching roads are marked not applicable because the rider has no outgoing road choice there. An optional whole-route summary can be derived from the sequence of conditional road risks, but the road probabilities are the primary output.

### Removing the limited route-ranking supervision

The current edge-cost head receives direct ranking supervision only from intentional deviations. A mature dataset should create a choice set at every verified decision point:

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

This removes the present situation in which five followed examples update the classifier but provide no direct route-cost comparison. Multiple decision points may be extracted from one ride, but evaluation must keep all points from the same rider—and preferably the same ride—in one fold.

### Current temporal-data assumption

The architecture accepts multiple historical traffic snapshots, but the current eight approved examples have no contemporaneous historical traffic. Each currently uses one explicit “unknown traffic” frame.

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

Therefore:

- GATv2 and the remaining network are trained and operational.
- The LSTM is present, receives gradients, and can accept a sequence.
- Current real results do not prove that temporal history improves prediction.
- Meaningful LSTM evidence requires traffic snapshots collected before and during future rider decisions, followed by retraining and temporal ablation.

## 3. Expected final outputs

For one destination and traffic context, the neural network returns:

```python
edge_costs, edge_deviation_logits = model(inputs)
edge_deviation_probabilities = torch.sigmoid(edge_deviation_logits)
```

Mathematically:

- $c_e > 0$: learned preference cost of directed edge $e$;
- $z_e$: raw deviation logit for suggested road $e$;
- $p_e=\sigma(z_e)$: estimated probability that the rider rejects road $e$ when reaching its source;
- $P^*=\arg\min_P \sum_{e\in P}c_e$: recommended connected route.

The system-level output contains:

- origin and destination;
- recommended edge sequence and road names;
- total learned preference cost;
- shortest-distance baseline;
- deviation probability for every branching road on the baseline, suggestion, or recommended route;
- traffic coverage information;
- route geometry for visualization.

The key distinction to state is:

> The neural network predicts a cost and a deviation probability for each road. Dijkstra uses the costs to produce the final valid route.

The primary probability has a local scope: it predicts rejection of one proposed outgoing road at one decision point, given the graph, destination, road attributes, and available pre-decision traffic context. An optional route summary is:

~~~math
p_{route}=1-\prod_k(1-p_k),
~~~

Here $p_k$ is interpreted conditionally on the rider reaching choice $k$ after following the route so far. The product is a derived summary; individual road probabilities remain more explainable and need calibration on held-out riders.

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
| $G=(V,E)$ | Directed road graph | 1,368 nodes; 3,418 edges |
| $N$ | Number of nodes | 1,368 |
| $M$ | Number of directed edges | 3,418 |
| $T$ | Number of traffic snapshots | currently 1; planned default 6 |
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
| $c$ | Positive edge costs | $M$ |
| $z$ | Deviation logit | scalar |

These tensors are grouped by `ModelInput`:

```python
@dataclass(frozen=True)
class ModelInput:
    node_features: Tensor
    edge_index: Tensor
    edge_static: Tensor
    temporal_edge_features: tuple[Tensor, ...]
    destination_features: Tensor
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
trainable parameters:      7,458
```

`hidden_channels=16` is a capacity choice, not a mathematical consequence of the 13 inputs. The node projection learns a matrix $W\in\mathbb{R}^{16\times13}$ and bias $b\in\mathbb{R}^{16}$, so $XW^\top+b$ changes `[N, 13]` into `[N, 16]`. Widths 15, 17, or 18 are all valid because the two attention heads are averaged with `concat=False`; 16 is a compact power-of-two baseline. On this model, widths 15, 16, 17, and 18 contain 6,707, 7,458, 8,247, and 9,074 trainable parameters respectively. A larger final dataset should tune this width on rider-disjoint validation data rather than assuming 16 is optimal.

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
        -> cost head                  [M]
        -> road-deviation head        [M]
```

Parameter distribution:

| Component | Parameters |
|---|---:|
| Node projection | 224 |
| Two GATv2 layers | 3,808 |
| Edge projection | 944 |
| LSTM | 2,176 |
| Edge-cost head | 17 |
| Road-deviation head | 289 |
| **Total** | **7,458** |

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

- Congestion is Mapbox numeric congestion divided by 100.
- Speed is Mapbox speed divided by the edge’s OSM reference speed and capped at 2.
- Separate masks say whether congestion and speed were actually observed.
- Age is elapsed time divided by one hour and capped at 1.
- Unknown traffic has zero values, zero observation masks, and age 1.

The masks are essential. Without them, a zero could incorrectly mean “free flow” when the API returned no measurement.

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


The code treats each road edge as one sequence in the LSTM batch:

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


The final edge state $h_e^{(T)}$ is used by both output heads.

The intended traffic policy is six snapshots at five-minute intervals, with observations rejected when they are future, received too late, or stale. The current real checkpoint saw one unknown frame per decision, so its LSTM behaves as a learnable transformation rather than demonstrated temporal memory.

## 10. Positive edge-cost head

The edge-cost head produces an additive, strictly positive preference cost:

```python
self.cost_head = nn.Linear(hidden_channels, 1)

return (
    F.softplus(
        self.cost_head(edge_embeddings).squeeze(-1)
    )
    * inputs.edge_static[:, 0]
    + 1e-4
)
```

For edge $e$:

~~~math
c_e=
\operatorname{softplus}(w_c^\top h_e+b_c)
\cdot\frac{\operatorname{length}_e}{100\text{ m}}
+10^{-4}.
~~~


This design has three consequences:

1. `softplus` guarantees a nonnegative learned rate.
2. $10^{-4}$ guarantees a strictly positive cost required by routing checks.
3. Multiplying by edge length integrates a learned cost per 100 m, reducing sensitivity to how OSM splits the same street into segments.

The costs are dimensionless preference costs. They are not seconds, minutes, or predicted travel time.

## 11. Road-deviation head

The deviation head applies the same multilayer perceptron independently to every final edge embedding:

```python
edge_embeddings = self.backbone.encode_edges(inputs)
edge_costs = self.backbone.costs_from_embeddings(edge_embeddings, inputs)
edge_deviation_logits = self.edge_deviation_head(
    edge_embeddings
).squeeze(-1)
return edge_costs, edge_deviation_logits
```

For every directed road edge $e$:

~~~math
z_e=W_2\operatorname{ReLU}(W_1h_e+b_1)+b_2,
\qquad p_e=\sigma(z_e).
~~~

Both returned tensors have length $M$, the number of directed graph edges. During training, the loss selects only the positions named in `edge_targets`. During routing, the output lists every road in the route and reports $p_e$ at roads whose source is a real branching point. The probabilities are currently uncalibrated because there are only three positive road-deviation labels.

## 12. Multitask training loss

### Classification loss

All approved road choices contribute weighted binary cross-entropy:

```python
positive_weight = torch.tensor(float(negatives / positives))

classification = F.binary_cross_entropy_with_logits(
    torch.cat(classification_logits),
    torch.cat(classification_labels),
    pos_weight=positive_weight,
)
```

With the current 25 followed-road and three deviated-road labels:

~~~math
w_+=\frac{25}{3}\approx8.333.
~~~


This gives additional weight to errors on the less frequent deviation class.

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
L_{BCE,i}
+
\mathbb 1[\text{preference pair exists}]
\lambda L_{rank,i}.
~~~


The implementation uses $\lambda=1$ by default. It pools road-classification targets across the approved examples and averages the available route-ranking pairs:

```python
classification = F.binary_cross_entropy_with_logits(
    torch.cat(classification_logits),
    torch.cat(classification_labels),
    pos_weight=positive_weight,
)
loss = classification
if ranking_losses:
    loss = loss + preference_weight * torch.stack(ranking_losses).mean()
```

The shared backbone receives gradients from both tasks. The road-deviation head receives classification gradients. The cost head receives ranking gradients from the three approved deviations.

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
    optimizer.zero_grad()
    # Compute road-classification and route-ranking losses.
    loss.backward()
    optimizer.step()
```

Current settings:

| Hyperparameter | Current value |
|---|---:|
| Epochs | 100 |
| Learning rate | 0.005 |
| Optimizer | Adam |
| Preference-loss weight | 1.0 |
| Seed | 17 |
| Hidden channels | 16 |
| Dropout | none |
| Mini-batching | none; one averaged update per epoch |

The main training command fits all eight approved examples. It is used to produce the deployable checkpoint. It does not provide held-out accuracy; evaluation is a separate command.

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
    scores, edge_deviation_logits = model(inputs)
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

The deviation probability is descriptive. The learned route is selected from the edge costs; the probability does not directly choose the route.

## 15. Training output versus evaluation output

### Training report

The current complete-data training run reports:

- 8 approved examples from 6 riders;
- 28 road choices: 25 followed and 3 deviated;
- loss reduced from 1.996 to 0.129 after 100 epochs;
- 27/28 in-sample road classifications correct;
- 3/3 in-sample deviation pairs ranked correctly.

These numbers show that the implementation can fit the current examples. They are not evidence of unseen-rider generalization.

### Held-out-rider evaluation

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

- prevalence baseline;
- GCN;
- original GAT;
- spatial-only GATv2;
- GATv2-LSTM;
- latest-only LSTM control when actual temporal variation exists.

Current GATv2-LSTM held-out results are weak:

| Metric | Current result |
|---|---:|
| Balanced accuracy | 0.307 |
| Deviation F1 | 0.091 |
| ROC-AUC | 0.093 |
| Held-out preference ranking | 2/3 |

The correct interpretation is:

> The pipeline is operational, but 28 correlated road labels from eight examples, including only three deviations, are insufficient to claim reliable rider generalization or an LSTM benefit.

## 16. Current implementation limits

State these directly if asked:

1. **Very small training set:** only eight approved examples, producing 28 correlated road labels.
2. **Only three route-ranking pairs:** the edge-cost head has limited direct supervision.
3. **No useful historical traffic yet:** the current checkpoint cannot demonstrate temporal improvement.
4. **No rider embedding:** behavior is learned across riders rather than personalized to one rider.
5. **No turn restrictions:** routing is edge-additive.
6. **No calibrated probability:** only three positive road-deviation labels are available.
7. **No hyperparameter tuning set:** current held-out folds must not also be used repeatedly to choose settings.
8. **No travel-time objective:** the learned route represents preference cost.
9. **Limited route context:** the road head sees graph, destination, edge, and traffic context, but it does not yet encode a rider-specific history or the complete prefix of the current trip.

## 17. Compact top-to-bottom explanation

Use this when asked “How does the whole model work?”

> We represent Taft Avenue as a directed OSM graph. Every intersection has normalized location and surrounding-context features. Every road has static OSM attributes, optional timestamped traffic attributes, and a vector toward the destination. For every traffic snapshot, two GATv2 layers aggregate neighboring road context using dynamic attention informed by node and edge attributes. We then form a road embedding from its source node, destination node, static attributes, traffic, and destination direction. An LSTM processes each road's sequence of spatial embeddings. One head maps each final road embedding to a positive preference cost, while the other maps it to the probability that the rider rejects that road at a branch. Training combines road-level weighted binary cross-entropy with a pairwise ranking loss that encourages an intentionally chosen route to cost less than the rejected suggestion. At inference, the model reports road probabilities, scores all accessible directed roads, and Dijkstra finds the connected route with the lowest total learned preference cost.

## 18. Code files to open during the presentation

| Explanation | File and function |
|---|---|
| Feature definitions and tensor construction | `graph_data.py:build_real_graph_data_from_graph` |
| Destination and unknown-traffic input | `graph_data.py:RealGraphData.build_model_input` |
| GATv2 and LSTM architecture | `model.py:PreferenceModel` |
| Two model outputs | `model.py:DecisionPreferenceModel` |
| Classification and ranking loss | `train_model.py:train_decision_model` |
| Real approved-example preparation | `train_model.py:prepare_real` |
| Held-out-rider metrics | `evaluate_model.py:evaluate_rider_disjoint` |
| Model-to-Dijkstra connection | `predict_route.py:route_request` |
| Routing implementation | `predict_route.py:shortest_real_preference_path` |

The architecture is a task-specific composition. GATv2 comes from Brody et al. and is used through PyTorch Geometric. The LSTM is PyTorch's standard implementation. The per-road deviation head, destination-conditioned edge construction, pairwise preference objective, and learned-cost routing connection are the project-specific composition.
