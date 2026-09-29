# Demo-first presentation guide: STGAT-LSTM checkpoint

> Use `python -m stgat_lstm --help` for commands. Checkpoint prediction runs through `predict_route.py`, and training accepts only automatically validated official rider examples.

## New endpoint-routing demonstration

```bat
call .venv\Scripts\activate.bat
python -m stgat_lstm dashboard
```

Open http://127.0.0.1:8765. Choose two road points and generate a route. Explain: "The GATv2 encodes neighboring road context. The edge LSTM encodes the supplied temporal snapshots. The preference head outputs positive edge costs, and Dijkstra finds a route with the lowest sum. The green route uses those costs; the blue comparison uses distance. Neither cost nor probability is a predicted travel time."

Show `predict_route.py:route_request` for the model-to-search connection and `dashboard.py:DashboardApplication.route` for the interactive request. New endpoints do not require rider labels. The shared model has no rider-specific profile embedding. Turn restrictions are not represented, so this is a research routing prototype.

To explain traffic, show `mapbox_traffic.py:TrafficArchive.sequence` and `train_model.py:prepare_real`: saved, timestamped API responses are converted into directed-edge features for six past slots. Request and receipt times both precede each slot. Null congestion and uncovered edges remain explicitly unknown. The collector runs independently of the app; training reads saved files rather than querying current traffic for historical rides. See RUNBOOK sections 12–13 for full commands.

The current training report is `outputs\training_report.json`: 100 epochs on 28 road-choice labels reconstructed from eight approved real examples, loss 1.996 to 0.129, 27/28 training road labels and 3/3 training preference rankings. Historical traffic is marked unknown because later Mapbox captures cannot describe earlier rides. This tests software connectivity and fit to the small training set; it is not evidence that LSTM improves accuracy. The separate held-out-rider report remains the relevant generalization evidence.

The official export snapshot contains 56 rides from 12 riders, 170 generated-route records, 34 deviations with 34 deviation responses, 11,435 GPS points, and 44 post-trip questionnaires. Only 17 rides remain fully inside the provisional Taft corridor, and strict reconstruction yields five followed examples plus three deviation examples. The supplied OSM graph contains 36,375 nodes and 104,019 directed edges; all graph nodes have a matching feature row and all three deviation paths are covered by the extracted model graph.

## What the presentation should prove

Current evaluation explanation: "We hold out one entire rider at a time, train on the other riders, and pool the unseen-rider predictions. Balanced accuracy is our proposed primary metric because it weights followed and deviated classes equally. We also report class-specific precision, recall and F1, macro-F1, average precision, ROC-AUC, Brier score, log loss, and the confusion counts. Preference-ranking accuracy checks whether the model gives the observed deviation path a lower cost than the rejected suggestion. These are different from training accuracy."

For the LSTM question: "We compare spatial GATv2, full-history GATv2-LSTM, and a control with the same LSTM architecture and sequence length but only its latest snapshot repeated. The control helps separate historical information from model capacity. Our current real examples lack observed temporal traffic variation, so they cannot prove LSTM adds value. We will run the controlled experiment with contemporaneous archives and report the outcome even if history does not help." Run `python -m stgat_lstm evaluate` with `--temporal-ablation`; full commands and limitations are in RUNBOOK section 8.

Turn restrictions are outside the agreed study scope. Describe the routes as demonstrations of learned preference costs over the directed OSM graph, not a complete navigation service.

The updated forty-epoch pilot evaluation (28 road decisions from eight examples, six held-out-rider folds, seed 17) reports GATv2 balanced accuracy 0.300, deviation F1 0.000, ROC-AUC 0.000; GATv2-LSTM balanced accuracy 0.307, deviation F1 0.091, ROC-AUC 0.093. Both rank two of three held-out preference pairs correctly. These results do not demonstrate superiority or reliable generalization. No decision has observed historical traffic variation, so a temporal LSTM benefit cannot be established. The report is `outputs\rider_holdout_evaluation.json`. Do not replace these held-out results with the much higher in-sample training scores.

The presentation should demonstrate one complete path through the project:

```text
reviewed rider decision
        ↓
directed OSM graph and road features
        ↓
GATv2 spatial encoding + LSTM temporal interface
        ↓
per-road deviation probabilities + learned edge costs
        ↓
connected route from Dijkstra
```

Use this checkpoint statement:

> We have implemented and trained the complete GATv2-LSTM pipeline on approved real rider decisions. The checkpoint can be saved, reloaded, used to predict whether a rider will reject each suggested road at a branching point, and used to produce learned edge costs for routing. The current run is a software and training milestone, not a generalization result, because it contains only 28 correlated road labels reconstructed from eight examples. The preliminary rider-disjoint evaluation is weak.

The main presentation can fit in 10–12 slides. Keep detailed comparisons and limitations as backup slides.

---

## Slide 1 — What we will demonstrate

**Title:** Learning Motorcycle Route Preferences on Taft Avenue with GATv2-LSTM

**Show:**

- Vehicle: motorcycle
- Area: Taft Avenue corridor, Manila
- Prediction target: will the rider follow or intentionally deviate from the suggested route?
- Routing target: find a connected path with low learned rider-preference cost

**Say:**

> This demo starts with a reviewed rider decision, passes the Taft road graph through our trained model, predicts follow or deviate, and uses the learned edge costs to generate a route. Travel time is contextual information, not the training target.

---

## Slide 2 — Demo roadmap

```mermaid
flowchart LR
    A[Review a rider decision] --> B[Show graph inputs]
    B --> C[Show GATv2-LSTM code]
    C --> D[Show training checkpoint]
    D --> E[Run predict_route.py]
    E --> F[Explain prediction and route]
```

**Say:**

> The app is used for data collection. This Python project reconstructs labels, trains the model, and demonstrates inference and routing.

---

## Slide 3 — Demo the reviewed training evidence

Open the existing review page:

```bat
start "" "outputs\candidate_review_map.html"
```

Show one event and point out:

- green line: observed rider path
- red line: route suggested before the decision
- label: followed or intentionally deviated
- road names and path lengths
- approval means the example may be used for training

Current approved artifact:

- 8 decisions
- 6 riders
- 5 followed cases
- 3 intentionally deviated cases

**Say:**

> GPS and survey responses establish the label. The model does not receive the later observed path or survey response during prediction. At inference time it receives the suggested route, road graph, destination context, and traffic available by that time.

**Important files:**

- `stgat_lstm/build_decision_candidates.py`
- `stgat_lstm/create_review_map.py`
- `stgat_lstm/review_decisions.py`
- `outputs/approved_training_examples.json`

**Commands for this section:**

If you need to regenerate the review artifact before presenting, run these in order:

```bat
python -m stgat_lstm build-deviations "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --output outputs\deviation_candidates.json --geojson-output outputs\deviation_candidates.geojson
python -m stgat_lstm build-decisions "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --deviation-candidates outputs\deviation_candidates.json --output outputs\decision_candidates.json --geojson-output outputs\decision_candidates.geojson
python -m stgat_lstm review-map outputs\decision_candidates.geojson --output outputs\candidate_review_map.html
python -m stgat_lstm review auto outputs\decision_candidates.json --output outputs\approved_training_examples.json
```

The automatic step applies versioned survey, GPS, path-connectivity and common-endpoint rules. The HTML map remains optional visual evidence. Do not regenerate candidates during the presentation because doing so changes the artifact hashes and can make the checkpoint provenance stale.

---

## Slide 4 — What enters the model

The full supplied Metro Manila graph contains 36,375 nodes and 104,019 directed edges. The current Taft corridor extraction contains 1,368 nodes and 3,418 directed edges.

| Input | Meaning |
|---|---|
| Node features | position, intersection context, traffic signals, nearby POIs |
| Static edge features | length, lanes, speed reference, direction, access and road type |
| Dynamic edge features | speed ratio, congestion, availability masks and observation age |
| Destination features | position of an edge relative to the requested destination |
| Suggested road choices | candidate outgoing roads whose local deviation probabilities are predicted |

**Short map-matching explanation:**

> Raw GPS coordinates are not road-edge IDs. A Hidden Markov Model considers both GPS-to-road distance and whether consecutive road candidates form a plausible directed journey. Viterbi decoding returns the most likely connected road sequence.

**Important files:**

- `stgat_lstm/map_matching.py`
- `stgat_lstm/graph_data.py`

**Commands for this section:**

Run these before presenting when you want to show that the input data and graph were audited:

```bat
python -m stgat_lstm audit-data "data\real\rider_exports"
python -m stgat_lstm audit-network "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv"
python -m stgat_lstm audit-matching "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml"
```

These commands audit and summarize the inputs. They do not train the model.

`map_matching.py` and most of `graph_data.py` are library modules used by these commands; they are not normally launched as separate presentation commands.

---

## Slide 5 — The important GAT concept

Show this simplified idea before showing code:

```text
ordinary aggregation: all connected neighbors contribute similarly
graph attention:     the model learns how much each connected neighbor matters
```

For a road-network node `i`, attention assigns a learned weight to each connected neighbor `j`:

```text
new representation of i = Σ attention(i, j) × transformed features of j
```

In this implementation, road-edge attributes also participate in the attention calculation. This lets the model distinguish neighbors using properties such as road class, length, access, and available traffic.

Show the actual architecture block from `stgat_lstm/model.py`:

```python
self.spatial_1 = GATv2Conv(
    hidden_channels,
    hidden_channels,
    heads=2,
    concat=False,
    edge_dim=edge_static_dim + edge_dynamic_dim,
    negative_slope=0.2,
)
self.spatial_2 = GATv2Conv(
    hidden_channels,
    hidden_channels,
    heads=2,
    concat=False,
    edge_dim=edge_static_dim + edge_dynamic_dim,
    negative_slope=0.2,
)
```

Explain only these four points:

1. Two GATv2 layers allow information to pass through approximately two graph hops.
2. Two attention heads learn two views of neighboring road context.
3. `concat=False` averages the heads and keeps the hidden dimension manageable.
4. `negative_slope=0.2` configures the LeakyReLU used internally to calculate attention weights.

**Why GATv2:**

> The original GAT learns neighbor importance, but GATv2 changes the attention operation so the ranking can depend more expressively on the querying node. That is useful when the relevance of a neighboring road depends on the current location and context.

References:

- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [How Attentive Are Graph Attention Networks? — GATv2](https://openreview.net/pdf?id=F72ximsx7C1)
- [PyTorch Geometric GATv2Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html)

**Command for this section:**

There is no separate GAT-only command. The GATv2 layers run when either training or inference calls the model. Show the code in `stgat_lstm/model.py`, then use the training or inference command in Slides 7 and 9.

---

## Slide 6 — From GAT features to LSTM and two outputs

After GATv2 encodes each graph snapshot, the model creates an embedding for every directed road edge using:

```text
source-node embedding
+ destination-node embedding
+ static road features
+ dynamic traffic features
+ destination-relative features
```

The temporal layer is explicitly implemented as:

```python
self.temporal = nn.LSTM(
    hidden_channels,
    hidden_channels,
    batch_first=True,
)
```

For multiple timestamped snapshots, the same edge is represented through time and the final LSTM state is used.

The shared representation has two outputs:

```mermaid
flowchart LR
    A[GATv2 + LSTM edge embeddings] --> B[Road-deviation head]
    A --> C[Positive edge-cost head]
    B --> D[Probability for each road]
    C --> E[Dijkstra route]
```

Show this code:

```python
edge_embeddings = self.backbone.encode_edges(inputs)
edge_costs = self.backbone.costs_from_embeddings(edge_embeddings, inputs)
edge_deviation_logits = self.edge_deviation_head(
    edge_embeddings
).squeeze(-1)
```

**Be precise about the current LSTM status:**

> The LSTM is implemented and its weights are present in the trained checkpoint. The historical rider cases have no matching archived traffic sequence, so each currently uses one unknown traffic snapshot. We have completed the temporal architecture and interface, but have not yet demonstrated meaningful temporal traffic learning on real sequences.

**Optional Mapbox command for this section:**

Use this only to demonstrate the traffic-input interface. It collects a current observation; it does not create historical traffic for old rider decisions.

```bat
set MAPBOX_ACCESS_TOKEN=YOUR_TOKEN_HERE
python -m stgat_lstm traffic "120.9790,14.5800;120.9980,14.5400" --output outputs\traffic\current_traffic.json
```

---

## Slide 7 — Train and save the checkpoint

The command is:

```bat
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --epochs 100 --output outputs\gatv2_lstm_checkpoint.pt
```

What it does:

1. Refuses a decision artifact that has not passed the automatic validation gate.
2. Builds the Taft graph tensors for each approved decision.
3. Converts reviewed segments into road-choice targets and trains one deviation probability per road.
4. For deviation cases, teaches the chosen path to have lower cost than the rejected suggestion.
5. Saves weights, model configuration, training summary, and provenance.

Show this short loss block from `stgat_lstm/train_model.py`:

```python
classification = F.binary_cross_entropy_with_logits(
    torch.cat(classification_logits),
    torch.cat(classification_labels),
    pos_weight=positive_weight,
)
loss = classification + preference_weight * ranking_loss
```

The path-ranking loss asks for:

```text
cost(observed intentional choice) < cost(rejected suggested path)
```

---

## Slide 8 — Show the checkpoint result

Display the readable report:

```bat
type outputs\training_report.json
```

Current development-training result:

| Item | Result |
|---|---:|
| Approved decisions | 8 |
| Riders | 6 |
| Epochs | 100 |
| Initial loss | 1.1521 |
| Final loss | 0.5998 |
| Training road labels fit | 27/28 |
| Preference pairs correctly ranked | 3/3 |

**Say:**

> The falling loss and successful checkpoint reload show that the implementation trains end to end. Seven of eight is training-set fit, not 87.5 percent test accuracy. A generalization claim requires a rider-disjoint held-out evaluation.

---

## Slide 9 — Run `predict_route.py`

This is the main live inference demo:

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --example-key 9d8b7bf677b875c5
```

`predict_route.py` does not train or change the checkpoint. It:

1. rebuilds the same graph feature structure,
2. loads the saved `.pt` model configuration and weights,
3. selects one approved decision,
4. predicts a deviation probability for each branching road,
5. calculates suggested and observed path costs,
6. calculates a minimum learned-cost connected route.

If `--example-key` is omitted, it uses the first approved example. The explicit key makes the presentation repeatable.

**Important caution:** this example was part of training. The command proves checkpoint reload and inference; it does not measure unseen-rider performance.

The command requires these four inputs, in order:

```text
checkpoint → graphml → node feature CSV → approved decision JSON
```

If `--example-key` is omitted, the first approved example is selected.

---

## Slide 10 — Explain the prediction output

The expected structure is:

```json
{
  "suggested_route_deviation_probability": 0.2644,
  "suggested_road_deviation_predictions": [
    {
      "edge_id": "401|400|0",
      "road_name": "Taft Avenue",
      "is_decision_point": true,
      "deviation_probability": 0.18
    }
  ],
  "reviewed_decision_road": {
    "edge_id": "401|400|0",
    "deviation_probability": 0.18
  },
  "predicted_label": "followed",
  "approved_label": "followed",
  "suggested_path_preference_cost": 4.9508,
  "observed_path_preference_cost": 4.9508,
  "recommended_route": {
    "road_names": [
      "Taft Avenue",
      "Pablo Ocampo Street",
      "Leveriza Street"
    ],
    "total_learned_preference_cost": 3.5184
  },
  "traffic": "historical traffic unknown"
}
```

Explain each field:

| Field | Meaning |
|---|---|
| `suggested_road_deviation_predictions` | one conditional deviation probability at every branching road |
| `suggested_route_deviation_probability` | optional summary derived from the road probabilities |
| `reviewed_decision_road` | the first reviewed branching road used for the displayed event-level comparison |
| `predicted_label` | model prediction |
| `approved_label` | validated training label used for comparison |
| path preference costs | dimensionless learned preference scores; lower is preferred |
| `recommended_route` | connected path found by Dijkstra using all learned edge costs |
| `traffic` | historical traffic is explicitly marked unknown instead of fabricated |

For a followed case, the observed and suggested paths are the same, so their displayed costs are expected to be equal.

**Command for this section:**

No additional command is required. Read the JSON printed by Slide 9. The `recommended_route` object is the routing result produced after the model scores the edges.

---

## Slide 11 — Why routing is separate

Show the key relationship:

```text
neural model scores every directed edge
                 ↓
Dijkstra enforces a connected, traversable route
```

The model uses `softplus` to make every learned edge cost positive:

```python
return F.softplus(
    self.cost_head(edge_embeddings).squeeze(-1)
) * inputs.edge_static[:, 0] + 1e-4
```

**Say:**

> The neural network learns what roads appear preferable. Dijkstra performs the discrete graph search and guarantees a connected minimum-cost route under those learned positive costs. The cost is a preference score, not seconds of travel time.

**Command for this section:**

Routing runs inside the inference command from Slide 9. If you need to explain the implementation separately, open `stgat_lstm/predict_route.py`; it does not need to be run as a standalone command.

---

## Slide 12 — Honest project status

Completed at this checkpoint:

- directed OSM graph preparation
- HMM/Viterbi GPS map matching
- reviewed follow/deviate labels
- two-layer GATv2 implementation
- LSTM temporal architecture
- classification and path-ranking training
- checkpoint save and reload
- learned-cost Dijkstra routing
- Mapbox traffic collection and OSM-edge alignment interface

Still required for research validation:

- substantially more Taft decisions and riders
- timestamped traffic sequences aligned before decision time
- rider-disjoint train/validation/test splits
- GCN, GAT, GATv2 without LSTM, and STGAT-LSTM comparisons
- held-out classification, ranking, calibration, and route-quality metrics

**Closing statement:**

> The engineering milestone is complete: reviewed rider decisions can train a GATv2-LSTM checkpoint and that checkpoint can drive prediction and routing. The next milestone is data collection and controlled evaluation, especially temporal traffic sequences and unseen riders.

**Command for this section:**

Run the test suite before presenting:

```bat
python -m unittest discover -s tests -v
```

This verifies the software components. It is separate from held-out model evaluation.

---

## The specific answer to “How did you do it?”

Use this answer when the panel wants implementation detail:

> First, we converted the supplied directed OSM graph into a Taft Avenue corridor graph and joined the POI feature table by `osmid`. We map-matched rider GPS sequences to directed OSM edges using an HMM/Viterbi procedure. From the matched sequence, the suggested route, and the survey evidence, we constructed a reviewed binary decision: `0` means followed and `1` means intentionally deviated. Only approved examples enter training.
>
> For each decision, the model receives one graph input. Every graph node has 13 features, every directed edge has 19 static features and 5 dynamic traffic features, and every edge receives two destination-relative features. The graph structure is stored as `edge_index`, so message passing follows actual road connectivity. The node features are projected to 16 hidden dimensions. Two GATv2 layers aggregate neighboring node information using edge attributes such as length, road type, speed, access, and traffic availability.

If asked why 16, explain that it is a tunable hidden width. The learned node projection uses a `[16, 13]` weight matrix, so it converts each 13-feature row into 16 learned values. Fifteen, 17, or 18 would also run; 16 is a compact baseline with 7,458 total parameters and must eventually be compared on rider-disjoint validation data.
>
> We then combine the source-node embedding, destination-node embedding, static edge features, dynamic edge features, and destination features to create one embedding per directed road edge. For a sequence of traffic snapshots, an LSTM processes the embedding of each edge over time and keeps the final state. The current historical examples use one explicitly unknown snapshot because matching historical Mapbox traffic was not archived.
>
> The shared edge embeddings feed two heads. The road-deviation head produces one logit per edge, and the cost head produces one positive preference cost per edge with `softplus`. During training, binary cross-entropy selects reviewed branching roads and teaches whether each was followed or rejected. For an intentional deviation, a ranking loss also teaches the observed path to have lower total learned cost than the rejected suggested path.
>
> At inference, the model scores every routable edge. We pass those positive scores to Dijkstra, exclude edges restricted to motorcycles, and reconstruct the lowest-cost connected path. Therefore, the neural network learns preference scores while the graph-search algorithm guarantees a legal connected route.

### The one-minute implementation flow

```text
raw app exports
  → audit and filter incomplete records
  → HMM/Viterbi map matching
  → candidate follow/deviate decision
  → manual approval and hash-bound training artifact
  → OSM/POI/traffic tensors
  → node projection
  → two GATv2 layers
  → per-edge embedding
  → LSTM over timestamped snapshots
  → road-deviation head + positive cost head
  → Dijkstra over learned costs
```

### The most useful code-to-concept mapping

| Concept to explain | Implementation |
|---|---|
| Road graph | `edge_index` contains source and destination node indices for each directed edge |
| Spatial attention | `GATv2Conv(..., heads=2, edge_dim=24)` in `model.py` |
| Node-to-edge representation | concatenate source embedding, destination embedding, edge features, and destination features |
| Temporal encoding | `nn.LSTM(hidden_channels, hidden_channels, batch_first=True)` |
| Rider decision | each edge embedding is sent to `edge_deviation_head`; reviewed branch edges select the supervised logits |
| Preference learning | `softplus(observed_cost - suggested_cost)` for intentional deviations |
| Positive route cost | `softplus(cost_head(...)) * edge_length + 1e-4` |
| Final route | Dijkstra in `predict_route.py`, with restricted-access edges removed |

### What happens to one example inside the model?

For example `9d8b7bf677b875c5`:

1. The approved artifact identifies the origin, destination, suggested edge sequence, observed edge sequence, and followed label.
2. `graph_data.py` extracts the Taft corridor and creates the graph tensors for that destination.
3. `model.py` computes hidden representations for connected nodes with GATv2.
4. Each directed edge receives an embedding, a positive cost, and a deviation logit.
5. Reviewed branching roads select the logits that are trained as followed or rejected.
6. All edge costs are passed to Dijkstra.
7. The result is printed as a probability, path costs, road names, and total learned route cost.

This is why `predict_route.py` is the best live demonstration: it runs steps 2–7 using a saved checkpoint without retraining.

### What the model learns and what it does not learn

It learns statistical associations between graph context and approved rider choices, such as whether a proposed road tends to be accepted or rejected under the available features. It does not automatically discover a human-readable causal rule, and the current small checkpoint cannot support that claim. Attention weights and learned costs are model signals, not causal explanations.

It also does not currently learn a reliable rider-specific embedding. The checkpoint is a shared model across riders. Rider IDs are kept for grouping and future unseen-rider evaluation.

## Function-by-function reference

Use this section when the panel points to a function and asks what it does.

### `model.py`

| Function or method | Role |
|---|---|
| `PreferenceModel.__init__` | Validates the architecture and feature dimensions, creates the node projection, GATv2/GAT/GCN layers, edge projection, optional LSTM, and cost head. |
| `PreferenceModel._spatial_snapshot` | Processes one traffic snapshot: projects node features, performs graph message passing, combines endpoint embeddings with edge and destination features, and returns one embedding per directed edge. |
| `PreferenceModel.encode_edges` | Processes all available snapshots. It stacks per-snapshot edge embeddings and runs the LSTM when the architecture is `stgat_lstm`; otherwise it uses the latest snapshot. |
| `PreferenceModel.costs_from_embeddings` | Converts edge embeddings into positive length-scaled preference costs using `softplus`. |
| `PreferenceModel.forward` | Runs edge encoding followed by cost generation; the basic model output is one cost per graph edge. |
| `PreferenceModel.configuration` | Serializes architecture and feature dimensions so the checkpoint can recreate the same model later. |
| `path_cost` | Looks up a sequence of edge IDs and sums their predicted costs to produce a route cost. |
| `preference_loss` | Compares a preferred route with a rejected route and penalizes the model when the preferred route does not have lower cost. |
| `DecisionPreferenceModel.__init__` | Wraps the preference backbone and adds the per-road deviation head. |
| `DecisionPreferenceModel.forward` | Gets edge embeddings and returns `(edge_costs, edge_deviation_logits)`, both with one value per directed graph edge. |
| `DecisionPreferenceModel.configuration` | Stores the model type and backbone configuration in the checkpoint. |

The main model call chain is:

```text
DecisionPreferenceModel.forward
  → PreferenceModel.encode_edges
    → PreferenceModel._spatial_snapshot
      → GATv2 layers
  → PreferenceModel.costs_from_embeddings
  → edge-deviation head
```

### `train_model.py`

| Function or class | Role |
|---|---|
| `PreparedDecision` | Data container for one training example: graph input, label, suggested route, and optional preference pair. |
| `DecisionTrainingSummary` | Data container for epoch count, losses, example counts, and in-sample checks. |
| `train_decision_model` | Creates the model and Adam optimizer, loops through epochs, computes classification and ranking losses, backpropagates, updates weights, and calculates the final in-sample summary. |
| `prepare_real` | Builds the real Taft graph, loads only approved decisions, creates graph inputs, and assigns observed-versus-suggested preference pairs to deviation cases. |
| `_save_training` | Saves the PyTorch checkpoint to `.pt` and writes its readable training report to `.json`. |
| `main` | Parses the `real` training command, prepares approved examples, trains the model, and prints the saved report. |

The real training call chain is:

```text
main
  → prepare_real
    → build_real_graph_data
    → load_approved_decisions
  → train_decision_model
    → DecisionPreferenceModel.forward
    → classification loss
    → preference ranking loss for deviations
  → _save_training
```

### `predict_route.py`

| Function | Role |
|---|---|
| `main` | Parses the checkpoint and graph paths, rebuilds the graph input, selects an approved example, optionally loads Mapbox snapshots, reloads the saved weights, computes probability and path costs, runs learned-cost routing, saves the JSON result, and prints a short summary. |

The prediction call chain is:

```text
main
  → build_real_graph_data
  → load_approved_decisions
  → build_model_input
  → torch.load checkpoint
  → DecisionPreferenceModel.forward
  → path_cost
  → shortest_real_preference_path
  → saved JSON + terminal summary
```

The training file changes model weights. The prediction file only reads the saved weights and produces outputs.

---

## Presentation-day command sequence

Run from Windows `cmd`:

```bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
call .venv\Scripts\activate.bat
```

There are two different command modes:

1. **Full pipeline mode:** use this when rebuilding the artifacts from fresh app exports or after collecting new data.
2. **Presentation demo mode:** use this when `outputs\approved_training_examples.json`, `outputs\gatv2_lstm_checkpoint.pt`, and `outputs\candidate_review_map.html` already exist.

### Full pipeline before the presentation

Run these in order when you need to regenerate and retrain:

```bat
python -m unittest discover -s tests -v
python -m stgat_lstm audit-data "data\real\rider_exports"
python -m stgat_lstm audit-network "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv"
python -m stgat_lstm audit-matching "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml"
python -m stgat_lstm build-deviations "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --output outputs\deviation_candidates.json --geojson-output outputs\deviation_candidates.geojson
python -m stgat_lstm build-decisions "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --deviation-candidates outputs\deviation_candidates.json --output outputs\decision_candidates.json --geojson-output outputs\decision_candidates.geojson
python -m stgat_lstm review-map outputs\decision_candidates.geojson --output outputs\candidate_review_map.html
python -m stgat_lstm review auto outputs\decision_candidates.json --output outputs\approved_training_examples.json
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --epochs 100 --output outputs\gatv2_lstm_checkpoint.pt
```

No decision JSON needs to be edited. Do not train until the validated artifact reports `status: "approved_for_training"`. The optional review map can still be opened for explanation or quality-control sampling.

### What each command reads and produces

| Command | What it does | Main inputs it reads | Output |
|---|---|---|---|
| `unittest discover` | Runs software tests for geometry, map matching, review, graph features, model, and routing | Test files and their in-memory test graphs | Pass/fail report in the terminal |
| `python -m stgat_lstm audit-data` | Checks that exported CSVs can be joined and summarizes missing, invalid, future, or ambiguous records | `rides_clean.csv`, `generated_routes_clean.csv`, `deviations_clean.csv`, `deviationResponses_clean.csv`, `map_points_clean.csv` | `rider_data_audit.json` plus a terminal summary |
| `python -m stgat_lstm audit-network` | Checks graph coverage, edge distances, node-feature joins, and Taft corridor coverage | Clean export directory, OSM `.graphml`, POI/node feature CSV | `road_network_audit.json` plus a terminal summary |
| `python -m stgat_lstm audit-matching` | Tests whether rider GPS sequences can be matched to connected directed graph edges | Clean export directory and OSM `.graphml` | `map_matching_audit.json` plus a terminal summary |
| `python -m stgat_lstm build-deviations` | Finds strict deviation/rejoin route pairs for validation | Clean rider CSVs and OSM `.graphml` | `deviation_candidates.json` and optional GeoJSON |
| `python -m stgat_lstm build-decisions` | Adds verified follow cases and imports strict deviation candidates into one decision artifact | Clean rider CSVs, OSM `.graphml`, and `deviation_candidates.json` | `decision_candidates.json` and `decision_candidates.geojson` |
| `python -m stgat_lstm review-map` | Turns candidate GeoJSON into a browser map | Candidate GeoJSON | `candidate_review_map.html` |
| `python -m stgat_lstm review auto` | Applies versioned survey, GPS and path rules and records approval/rejection reasons | Candidate JSON | `approved_training_examples.json` |
| `python -m stgat_lstm train real` | Trains classification and path-preference heads | OSM `.graphml`, node feature CSV, approved decision JSON | `gatv2_lstm_checkpoint.pt` and `training_report.json` |
| `python -m stgat_lstm predict` | Reloads the checkpoint, scores one decision, and routes with learned edge costs | Checkpoint, OSM `.graphml`, node feature CSV, approved decision JSON, optional Mapbox JSON | `route_prediction.json` plus a terminal summary |

The audit commands do not create labels. Candidate generation also does not directly train the model. The sequence is deliberately:

```text
audit → candidate construction → automatic validation → approved artifact → training
```

### How to explain “auditing the source data”

> Auditing means checking whether the exported app records are structurally and temporally reliable enough to analyze. We verify that the required CSV files exist, their IDs can be joined, route geometries and GPS values can be parsed, routes referenced by an event were available before that event, GPS points exist around the event, and the suggested and observed routes have compatible endpoints. The audit reports possible candidates and failure reasons; it does not decide that a rider learned or preferred a road.

### How to explain “producing a candidate”

> A candidate is a possible route-choice example, not yet a training label. For a reported deviation, the pipeline selects the latest suggested route that existed before the deviation timestamp, parses the rider GPS sequence, and map-matches both the GPS and route geometry to the directed OSM graph. It keeps the case only when the observed path leaves the suggested path, later rejoins it, has the same practical origin and destination, and has usable survey evidence. Automatic validation then accepts supported intentional reasons with GPS-backed geometry and rejects personal stops, unsupported reasons, missing traffic severity, weak matches, and disconnected paths.
>
> For a followed case, the pipeline uses a ride without a logged deviation. It requires sufficient GPS coverage inside the Taft corridor, successful directed map matching, strong agreement between GPS and the suggested route, and a verified branching point where the rider continued along the suggestion. It then creates a local followed example. Both types remain training-blocked until automatic validation produces the hash-bound approved artifact.

### Current output check and interpretation

The current real-data outputs are internally consistent:

| Artifact | Current status | Interpretation |
|---|---|---|
| `decision_candidates.json` | `review_required_not_training_ready` | Expected before manual approval; this file must not be trained directly |
| `candidate_review_decisions.json` | Edited review decisions | Human approval choices corresponding to the candidate file |
| `approved_training_examples.json` | `approved_for_training`; 8 approved | Correct training input: 5 followed, 3 deviated, 6 riders |
| `gatv2_lstm_checkpoint.pt` | Present; `stgat_lstm` configuration | Trained checkpoint containing GATv2, LSTM, road-deviation head, and cost head weights |
| `training_report.json` | Present; 100 epochs | Training report: loss fell from 1.9962 to 0.1294; 27/28 in-sample road classifications and 3/3 preference rankings |
| `candidate_review_map.html` | Present | Browser evidence used to inspect candidate paths before approval |

The current inference demonstration lists a deviation probability for each branching road, derives an optional route summary, compares it with the approved example, and returns the learned route. This is an in-sample checkpoint-reload demonstration, not held-out accuracy.

The traffic experiment files are also usable for interface testing. `outputs/traffic/experiments/taft_test_02.json` has speed observations but no numeric congestion observations; `outputs/traffic/experiments/taft_test_03.json` has speed on 110 segments and numeric congestion on 12 segments. Missing congestion is represented with an availability mask. The real historical checkpoint still reports traffic as unknown because current Mapbox observations were not attached retrospectively to old rides.

### Before presenting

Confirm the code passes:

```bat
python -m unittest discover -s tests -v
```

If the full pipeline has already been completed and the outputs have not changed, you do not need to rerun audits, candidate generation, automatic validation, or training during the live presentation.

### During the presentation

Open the reviewed evidence:

```bat
start "" "outputs\candidate_review_map.html"
```

Show the saved training report:

```bat
type outputs\training_report.json
```

Run checkpoint inference and routing:

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --example-key 9d8b7bf677b875c5
```

### Optional current-traffic interface demonstration

Only use this to show that the program can accept a current Mapbox observation:

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --example-key 9d8b7bf677b875c5 --traffic-observation outputs\traffic\current_traffic.json
```

State that current traffic is an interface demonstration and is not the historical traffic for that old rider decision.

---

## Files to have open in the editor

Open only these tabs before presenting:

| File | Code to point out |
|---|---|
| `stgat_lstm/model.py` | GATv2 layers, LSTM, road-deviation head and positive cost head |
| `stgat_lstm/train_model.py` | classification plus preference-ranking loss |
| `stgat_lstm/predict_route.py` | checkpoint loading, prediction and route generation |
| `stgat_lstm/predict_route.py` | learned-cost Dijkstra |
| `outputs/training_report.json` | readable checkpoint result |

Keep `map_matching.py`, `graph_data.py`, and `mapbox_traffic.py` available as backup files if the panel asks about label construction, graph features, or traffic.

---

## Backup questions and answers

### Where did the architecture come from?

> It is not a line-for-line reproduction of one STGAT-LSTM paper. It is a task-specific composition with traceable sources. Graph neighborhood attention comes from Graph Attention Networks by Veličković et al. We use GATv2 from Brody et al. because it replaces the original GAT's static attention with more expressive query-dependent attention. Temporal sequence modeling uses the standard LSTM introduced by Hochreiter and Schmidhuber. The broader idea of combining a road graph's spatial representation with recurrent temporal modeling is established in transportation research such as DCRNN, although our implementation is not DCRNN: it uses GATv2 followed by a per-edge LSTM and predicts rider choice and preference cost rather than future traffic flow.

| Current component | Source or inspiration | Project-specific adaptation |
|---|---|---|
| Neighbor attention | Original GAT paper | Applied to the directed OSM road graph with road-edge attributes |
| Dynamic attention | GATv2 paper | Two PyTorch Geometric `GATv2Conv` layers with two heads |
| Temporal memory | Hochreiter and Schmidhuber LSTM | LSTM processes the spatial embedding of every road edge across traffic snapshots |
| Spatial + temporal transportation pattern | DCRNN family | Sequential GATv2 then LSTM; no diffusion-convolution recurrent cell or encoder-decoder forecast |
| Edge representation | Conceptual lineage from `finalGNNT3` | Concatenates source/destination embeddings with static, dynamic, and destination-relative features |
| Routing | `finalGNNT3` and standard shortest-path search | Positive learned preference costs with restricted-edge filtering and Dijkstra |
| Tacit knowledge | Project research objective and old tacit layer | Replaces hand-set survey multipliers with supervised road-deviation and path-ranking losses |

The current `model.py` explicitly cites GAT, GATv2, and GCN in its module documentation. It imports `GATv2Conv` from PyTorch Geometric and calls `nn.LSTM`; no current Python file imports `edge_gat_training.py`, `tacit_layer.py`, or another `finalGNNT3` module.

### Are we using "pure GATv2" or a GATv2 library?

> We use the maintained PyTorch Geometric implementation: `from torch_geometric.nn import GATv2Conv`. We did not rewrite the GATv2 equations by hand and we do not import the authors' experiment repository. This is still the GATv2 operator described by Brody et al.; the authors' official repository explicitly points users to PyTorch Geometric's `GATv2Conv`.

The current call sets `heads=2`, `concat=False`, `edge_dim=24`, and `negative_slope=0.2`. With the installed PyTorch Geometric implementation, the unspecified defaults are `dropout=0.0`, `add_self_loops=True`, `share_weights=False`, and `residual=False`. The `edge_dim=24` extension lets static road attributes and dynamic traffic attributes participate directly in the attention calculation.

### How do the two supplied papers and repositories map to our model?

The titles and repository links must be paired as follows:

- **How Attentive Are Graph Attention Networks?** — [tech-srl/how_attentive_are_gats](https://github.com/tech-srl/how_attentive_are_gats). This is the direct source for the GATv2 attention operator used through PyTorch Geometric.
- **Solve routing problems with a residual edge-graph attention neural network** — [Lei-Kun/DRL-and-graph-neural-network-for-routing-problems](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems). This is a related routing architecture, not the source of GATv2.

| Question | Residual E-GAT routing paper | Current project |
|---|---|---|
| Task | Construct solutions for TSP, CVRP, and MDCVRP | Predict motorcycle road-level deviation behavior and learn route-preference costs |
| Graph encoder | Custom edge-aware GAT with residual connections | Two PyTorch Geometric GATv2 layers with node and edge attributes |
| Route construction | Transformer-style attention/pointer decoder selects the next unvisited node | Dijkstra searches the legal OSM graph using positive learned edge costs |
| Training | Deep reinforcement learning with PPO or improved REINFORCE | Supervised decision classification plus pairwise path-preference ranking |
| Temporal component | No LSTM in the cited routing architecture | Per-edge LSTM over successive graph/traffic snapshots |

We should not copy the paper's complete residual E-GAT/Transformer/RL pipeline into this project. It solves a different optimization problem on generated problem instances. Its useful design idea for a later experiment is the residual connection. Modern PyTorch Geometric exposes a `residual` option for `GATv2Conv`, but our current layers leave it at the default `False`. Adding it would change the architecture and require a new checkpoint plus a controlled comparison.

The other useful experiment is attention dropout. It is currently `0.0`. Residual connections, dropout, and normalization are possible hyperparameters rather than missing requirements of GATv2. With only eight approved decisions, changing them now would not establish an improvement. The higher-priority scientific work is collecting enough rider decisions and time-aligned traffic snapshots, making rider-disjoint train/validation/test splits, and comparing GCN, original GAT, spatial-only GATv2, and GATv2-LSTM under the same split.

### What exactly came from `finalGNNT3`?

> The previous project established the domain concept and data foundation: an OSM road graph, node and edge features, a two-layer graph-attention encoder, endpoint embeddings combined into an edge prediction, traffic adjustment, tacit survey penalties, and shortest-path routing. The current implementation preserves those useful ideas and the underlying OSM/POI lineage. It replaces the old travel-time regression target and hand-selected tacit multipliers with reviewed rider-choice supervision, GATv2, an LSTM interface, a per-road deviation head, learned positive preference costs, and a ranking loss.

Evidence from the old source:

- `finalGNNT3/edge_gat_training.py` defines two original `GATConv` layers, concatenates source and destination node embeddings with edge features, and trains with MSE against `simulated_travel_time`.
- `finalGNNT3/tacit_layer.py` assigns configured penalties from survey-reason strings.
- `finalGNNT3/routing_engine.py` calculates `final_cost = traffic_adjusted_time * tacit_multiplier` and provides Dijkstra/A* routing.
- The current package contains no import of those modules. Its `model.py` and the old `edge_gat_training.py` also have different SHA-256 hashes, confirming that they are different source files rather than identical copies.

**Recommended wording:**

> Our architecture was inspired by the graph-to-edge approach of the earlier thesis prototype and by published GAT, GATv2, LSTM, and spatiotemporal traffic-modeling research. We did not copy a complete architecture from one paper. We composed these components for a new target: motorcycle rider deviation probability per branching road and learned path preference.

References:

- [Veličković et al., Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [Brody et al., How Attentive Are Graph Attention Networks?](https://openreview.net/pdf?id=F72ximsx7C1)
- [Official GATv2 implementation repository](https://github.com/tech-srl/how_attentive_are_gats)
- [Lei et al., Solve Routing Problems with a Residual Edge-Graph Attention Neural Network](https://doi.org/10.1016/j.neucom.2022.08.005)
- [Official residual E-GAT routing repository](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems)
- [Hochreiter and Schmidhuber, Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Li et al., DCRNN: Data-Driven Traffic Forecasting](https://openreview.net/pdf?id=SJiHXGWAZ)

### What do state-of-the-art spatiotemporal graph models do?

> Most state-of-the-art models in this area are built for traffic-state forecasting. They receive historical measurements such as speed, flow, or occupancy from many road sensors and predict those values at future times. Their main challenge is learning both spatial dependence between roads and temporal dependence across a long history. DCRNN models directed traffic diffusion with recurrent units; Graph WaveNet learns hidden spatial dependencies and long temporal patterns with dilated convolutions; PDFormer uses dynamic long-range attention and explicitly models propagation delay. Their usual evidence is performance on large public traffic datasets using MAE, RMSE, or MAPE.

### How is our model different from those models?

| Dimension | Traffic-forecasting state of the art | Our STGAT-LSTM checkpoint |
|---|---|---|
| Main question | What will traffic speed or flow be later? | Will a motorcycle rider reject this suggested road at this branching point? |
| Input | Long historical sensor sequences over many locations | Directed OSM graph, road/POI features, suggested path, and traffic available at the decision time |
| Output | Future speed, flow, occupancy, or congestion | Per-road deviation probabilities and learned edge-preference costs |
| Spatial modeling | Often adaptive, long-range, or delay-aware dependencies | Two local GATv2 layers over the directed OSM topology |
| Temporal modeling | Usually many historical time steps and multi-step forecasting | LSTM interface for graph snapshots; current historical cases have one unknown snapshot |
| Decision layer | Usually stops at forecasting | Uses positive learned costs with Dijkstra to produce a connected route |
| Data scale | Large sensor benchmarks with held-out time periods | 28 road labels from eight examples and six riders; held-out results are preliminary |

**The key distinction to say aloud:**

> We are not claiming that our small checkpoint outperforms DCRNN, Graph WaveNet, or PDFormer at traffic forecasting. Those models solve a different prediction problem. Our adaptation uses the graph and temporal ideas for a rider-choice problem: learning behavioral preference from followed and intentionally deviated routes, then passing the learned preference costs to a router.

### Is our architecture itself state of the art?

> We do not claim that. GATv2 and recurrent temporal encoding are established components. Our contribution at this stage is the task-specific pipeline: reconstructing leakage-controlled motorcycle road choices from GPS and surveys, representing the OSM road graph, learning road-level deviation and path preference together, and integrating the result with directed graph routing. A state-of-the-art performance claim would require larger data, rider-disjoint validation and testing, and controlled comparisons against baselines.

### What would a fair comparison require?

> We would split complete riders into train, validation, and test sets, then evaluate the road-level target with balanced accuracy, precision, recall, F1, average precision, ROC-AUC, Brier score, and calibration. For routing, we need a separate evaluation of whether the recommended route matches later rider choices and how it affects secondary outcomes such as travel time. We should compare a prevalence baseline, GCN, original GAT, GATv2 without temporal history, STGAT-LSTM, and simpler tabular or discrete-choice baselines. We should not compare our tiny pilot directly with benchmark scores from large traffic datasets.

### Is this state of the art?

> It uses GATv2, a more expressive successor to the original GAT, with an LSTM temporal interface. We do not claim state-of-the-art measured performance. Large models such as Graph WaveNet and PDFormer are mainly evaluated on traffic forecasting benchmarks, while our target is motorcycle rider road-deviation behavior and path preference. A performance comparison requires more riders and a separate held-out test set.

### Why use a graph neural network?

> A road network is defined by connectivity. GATv2 can combine information from roads and intersections that are actually connected, while learning that some neighbors matter more than others. A regular tabular network would not know the road topology unless we manually encoded it.

### Where is LeakyReLU?

> It is applied internally by `GATv2Conv` when attention logits are calculated. We explicitly set `negative_slope=0.2`. ELU is separately applied to the output of the attention layers.

### Is the LSTM implemented or pending?

> It is implemented, trained as part of the checkpoint, and can accept a sequence of graph snapshots. Meaningful real temporal learning is pending because the old decisions do not have aligned historical traffic sequences.

### Does the model optimize travel time?

> No. It learns rider behavior and rider-preference cost. Travel time can be an input or a secondary evaluation outcome, but it is not the main target.

### What does `predict_route.py` prove?

> It proves that the saved checkpoint can be reconstructed in a separate process, run on graph inputs, produce a deviation probability for each branching road, score paths, and provide costs to routing. It is an inference demo, not an evaluation script.

### Why are there only eight examples?

> The reconstruction rules are conservative. Ambiguous GPS matching, incomplete routes, cases outside the corridor, and deviations unrelated to route preference are excluded. These examples are sufficient for an end-to-end checkpoint test, not a generalization claim.

### Is the model personalized?

> The current checkpoint learns shared behavior across riders. Reliable rider embeddings would require more examples per rider. Rider IDs are retained for a future rider-disjoint evaluation so the test can measure behavior on unseen riders.

### How is data leakage prevented?

> The later observed path and survey response construct the target label but are excluded from inference features. Only information available by the decision timestamp may enter the model. Current traffic is not attached to an old ride and presented as historical traffic.

### Why use Dijkstra after the neural network?

> The neural network estimates positive edge preference costs. Dijkstra uses those costs while enforcing graph connectivity and access restrictions. This prevents the model from returning a disconnected road sequence.

### How does this compare with the old `finalGNNT3` folder?

> We retained the project concept and the underlying OSM and POI data foundation. The old model predicted simulated travel time and applied heuristic tacit penalties. The current pipeline replaces that model with reviewed rider-choice labels, directed HMM map matching, GATv2-LSTM training, checkpointed inference, and learned preference-cost routing. No old Python module is imported by the current package.

References for comparison:

- [DCRNN](https://openreview.net/pdf?id=SJiHXGWAZ)
- [Graph WaveNet](https://www.ijcai.org/proceedings/2019/264)
- [PDFormer](https://ojs.aaai.org/index.php/AAAI/article/view/25556)
- [Hidden Markov Map Matching Through Noise and Sparseness](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/map-matching-ACM-GIS-camera-ready.pdf)
