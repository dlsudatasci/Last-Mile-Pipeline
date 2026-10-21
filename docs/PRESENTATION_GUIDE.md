# Complete presentation guide: STGAT-LSTM route preference system

Approved follow-up, 21 October 2026: RUNBOOK Section 11 compares three training
recipes using NLL checkpoint selection on the same saved rider groups. It is
opt-in, changes training/selection only, and has no full real-data results yet.
The completed figures below remain the measured baseline. Both completed runs
and a source rollback snapshot are preserved. Architecture and features remain
unchanged; no new temporal-benefit or accuracy claim is supported yet.

Latest evaluation, reviewed 21 October 2026: three-fold rider-grouped comparison
completed all 12 fits. Use RUNBOOK Section 10 for the saved run and reproduction.
Two models with inner validation and development refit require 12 fits. The
pooled report separates exact alternative-road selection from followed-road
selection and binary deviation detection. It tests each original choice once
across held-out riders; recovered extra follows remain fitting-only. This
changes evaluation, not the GATv2-LSTM architecture or losses. Retain the
completed fixed-split scores below as a separate protocol. New pooled results:
STGAT-LSTM 10/17 exact deviations, 17/51 followed roads, 27/68 overall (39.7%);
GATv2 5/17, 28/51, 33/68 (48.5%); always-follow 0/17, 51/51, 51/68 (75%).
LSTM selected epochs are 45/5/1. This is a mixed internal result, not proof of
reliability, a temporal-history benefit, or improvement over the old 1/4 test.

Input update, 20 October 2026: current real-training commands now use the cleaned
amenity CSV, retain the first-visit policy, and save to
`artifacts/experiments/real_cleaned_osm_20261020`. The run through Step 8 is
complete and reviewed: 12/13 selection-validation choices; 63/98 development
choices; 9/14 (64.3%) exact held-out choices, versus 10/14 (71.4%) always-follow.
Test choice NLL is 0.871 and ranking is 2/3. Only one of four actual alternative
roads is selected correctly. Binary follow/deviate accuracy at the selected
threshold is 11/14 (78.6%), a different metric. Reload reproduces test predictions
exactly; both route demos pass. This is an internal pilot, not validated broad
reliability. Dashboard/visualization improvements are deferred. Existing older
results/slides below describe the earlier features. For replaying any
saved pre-cleanup checkpoint without retraining, substitute the feature CSV in
`artifacts/inputs/osm_before_amenity_cleanup`. RUNBOOK is the current full run
procedure; the architecture and road-choice labels have not changed.

This is the slide-by-slide guide for presenting the current project. It is
aligned with the outgoing-road model, GPS-derived traffic histories, controlled
synthetic experiment, fixed rider-disjoint evaluation, and routing dashboard.

Folder layout updated 19 October 2026. All code references and commands below
use the organized source folders and the `generated`, `artifacts`, and `outputs`
locations. The architecture and experiment results are unchanged by the move.
Archived experiment provenance retains its original paths; the location map is
`artifacts/layout_manifest.json`.

Updated 18 October 2026: the current refinement has 77 approved examples from
36 rides and all 15 riders, producing 115 junction targets (98 suggested roads
and 17 alternatives). It preserves all original road labels, corrects four
route destinations, and extracts 47 additional followed choices from training
riders only. The architecture is unchanged. Three validation-only ranking-weight
trials select a setup for a final refit on the 12 development riders; the same
three test riders remain excluded. Current artifacts are under
`artifacts/experiments/real_refined_20261018`. Slide 12 explains this protocol.

The user's local first-visit run has produced its checkpoint and report under
`artifacts/experiments/real_first_visit_20261018`. It omits three later repeated-junction
followed targets within original training examples, retaining all examples and
alternative labels. Validation/test targets and the previously selected settings
stay fixed. Its scores await review; the last reviewed refined result is
9/14 (64.3%). Do not present the small fitting check as test accuracy.

The original preparation before refinement has 30 approved examples from
28 rides and all 15 riders, 68 junction targets (51 suggested and 17 alternative
roads), and 12 optional ranking pairs. The trial is under
`artifacts/experiments/real_navigation_20261016`. Full-data training completed 100 epochs:
loss 1.8917 to 0.2863, 62/68 training choices correct, and 11/12 training pairs
correctly ranked. Checkpoint reload and connected routing passed. These are
in-sample checks. The original fixed-split baseline used nine training, three
validation, and three test riders and fitted five variants once each. The
refinement reuses these groups for three selection trials and one final refit;
the earlier 75-fit leave-one-rider-out comparison was stopped.
Six events were recovered by looking back
to the turn during the same active-route period rather than assuming reports
occur within 60 seconds of a turn. Post-report GPS is not mandatory.
Recorded navigation, heading, event location, and directed GPS geometry now
resolve one additional choice without requiring an exact incoming HMM edge.
That label is inferred, with its evidence recorded, and trains the choice
objective without a full ranking pair. Navigation text remains label-only.
The original real fixed-split results are on Slide 15 as a baseline. The separate archived
16-example strict-baseline table is historical and does not evaluate the
revised labels.

If asked why evaluation mentioned 75 models: it means 15 held-out-rider folds
times five model variants. Those are temporary training fits for comparison,
separate from the one checkpoint used for routing. The current deployment
checkpoint has finished training; this larger benchmark has not finished.
The fixed split provides a smaller independent comparison and does not load
that full-data checkpoint to score the test set.
It has now completed: ST-GAT-LSTM gets 10/14 test road choices correct, with
deviation F1 0.500. Choice accuracy ties the suggestion baseline, probability
quality is worse, and only 1/3 test route pairs is correctly ranked. Present
this as mixed internal pilot evidence, not a reliable deployed model.

Use the documents for different purposes:

- `PRESENTATION_GUIDE.md`: what to place on slides, what to say, which code to
  show, and which commands to demonstrate.
- `MODEL_EXPLANATION.md`: the detailed technical reference for tensors,
  equations, features, model layers, losses, evaluation, and limitations.
- `SYNTHETIC_RUNBOOK.md`: exact commands for reproducing the controlled
  synthetic experiment from start to finish.
- `RUNBOOK.md`: the corresponding workflow for official rider data.

## The claim you can defend now

> We implemented an end-to-end supervised route-preference pipeline. It turns
> app GPS and survey records into graph-valid outgoing-road choices, encodes the
> directed OSM network with GATv2, encodes six GPS-derived traffic snapshots
> with an edge-level LSTM, learns probabilities over accessible roads at each
> junction, converts those probabilities to additive costs, and uses Dijkstra
> to generate a connected route. A separated synthetic rider split shows that
> the implementation can learn the injected spatial and temporal rule. The real
> rider set is still too small for a final generalization claim.

Do not say that the synthetic metrics prove real rider behavior. They prove
that the complete implementation can recover a known rule on unseen synthetic
riders under controlled conditions.

## Recommended presentation structure

Use 15 main slides. Keep the code on each slide focused on one concept. The
backup section contains answers to likely panel questions.

---

## Slide 1 - Research problem and objectives

**Title:** Learning Motorcycle Riders' Route Preferences with GATv2-LSTM

**Show:**

1. Capture and contextualize driver deviation.
2. Learn tacit route preference from the captured context.
3. Generate a connected route using the learned preference.

**Say:**

> A conventional router minimizes a hand-defined quantity such as distance or
> time. Our goal is to learn which road a rider tends to choose when several
> legal roads are available, then use those learned choices as routing costs.
> Travel-time improvement is a possible side effect, not the objective.

**Current scope:** motorcycle riders, Taft Avenue corridor, directed OSM road
network, no turn-restriction model.

---

## Slide 2 - What the system predicts

**Show:**

```text
At a branching node u:

                 road e1
               /
incoming road -+-- road e2       model predicts P(e1|u), P(e2|u), P(e3|u)
               \
                 road e3
```

The primary target is the **outgoing road chosen by the rider**. A binary label
is derived for reporting:

```text
followed = chosen road equals suggested road
deviated = chosen road differs from suggested road
```

**Say:**

> We changed from predicting one whole-route deviation probability to predicting
> the preferred outgoing edge at each real decision point. This matches the
> action a rider actually takes, gives multiple supervised choices along a
> route, and connects naturally to route generation. The reported deviation
> probability is one minus the probability of the candidate route's road.

**Formula:**

$$
p_e = \operatorname{softmax}_{e\in A(u)}(z_e),
\qquad
p_{\text{deviation}} = 1-p_{\text{suggested}}.
$$

Here, $A(u)$ is the set of accessible outgoing roads at source node $u$, and
$z_e$ is the model's raw utility logit for road $e$.

---

## Slide 3 - Complete system flow

**Show:**

```text
App export                         OSM graph and node features
  |                                           |
  +--> audit and map matching <---------------+
  |              |
  |              +--> graph-valid road choices and route pairs
  |
  +--> consecutive GPS fixes --> historical road-speed frames
                                 |
                                 v
OSM graph + destination + six traffic frames + choice labels
                                 |
                                 v
                      GATv2 -> edge embeddings -> LSTM
                                 |
                                 v
                  outgoing-road logits and probabilities
                                 |
                                 v
                  -log(probability) edge costs -> Dijkstra
                                 |
                                 v
                       recommended connected route
```

**Say:**

> Data preparation establishes what the rider chose. The neural model learns
> from those labels. At inference, the later survey answer and observed route
> are absent; the checkpoint receives only the graph, destination, and traffic
> context and produces road-choice probabilities and routing costs.

---

## Slide 4 - Data, labels, and leakage boundary

**Show this table:**

| Information | During training | During new-route inference |
|---|---:|---:|
| OSM topology and node features | Yes | Yes |
| Static road attributes | Yes | Yes |
| Destination displacement | Yes | Yes |
| Pre-decision GPS speed history | Yes | Yes, when available |
| Suggested road and accessible alternatives | Used to form supervision | Candidate route is evaluated after scoring |
| Road actually chosen by rider | Label only | No |
| Later GPS path and survey answer | Label evidence only | No |
| Rider identity | Used only to group evaluation folds | No rider embedding |

**Say:**

> The rider's deviation is present in the dataset as the answer the model must
> learn, not as an input feature that reveals the answer. We also exclude the
> target ride from its own traffic history. During evaluation, all records from
> the held-out rider are excluded from training and traffic profiles.

---

## Slide 5 - Map matching and candidate construction

**Show the candidate map:**

```bat
start "" "outputs\maps\synthetic_hotspot\candidate_review_map.html"
```

- Green: observed rider path.
- Red dashed: prior suggested path.
- Gray: road-network context.
- Repeated events are intentional observations from different rides or riders.

**Say:**

> Raw latitude and longitude are not graph edges. The map matcher finds a
> directionally plausible OSM edge sequence using GPS distance and path
> continuity. Candidate construction keeps intentional, connected outgoing-road
> choices. Rejoining is needed only for the optional comparable route-ranking
> pair. Automatic validation rejects GPS
> errors, wrong turns, missing answers, and disconnected paths. Candidate
> selection has no fixed distance-to-Taft cutoff; proximity is a diagnostic.

**Code to open:**

- `stgat_lstm/preprocessing/map_matching.py`
- `stgat_lstm/preprocessing/build_deviation_candidates.py`
- `stgat_lstm/preprocessing/build_decision_candidates.py`
- `stgat_lstm/preprocessing/review_decisions.py`

**Synthetic reconstruction commands:**

```bat
python -m stgat_lstm build-deviations generated\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" --output generated\synthetic_hotspot\reconstructed_deviation_candidates.json --geojson-output generated\synthetic_hotspot\reconstructed_deviation_candidates.geojson
python -m stgat_lstm build-decisions generated\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" --deviation-candidates generated\synthetic_hotspot\reconstructed_deviation_candidates.json --output generated\synthetic_hotspot\reconstructed_decision_candidates.json --geojson-output generated\synthetic_hotspot\reconstructed_decision_candidates.geojson
python -m stgat_lstm review auto generated\synthetic_hotspot\reconstructed_decision_candidates.json --output generated\synthetic_hotspot\reconstructed_approved_training_examples.json
```

---

## Slide 6 - Controlled synthetic experiment

**Show:**

| Property | Controlled value |
|---|---:|
| Artificial decision riders | 12 |
| Labeled decision rides | 144 |
| Followed / deviated | 72 / 72 |
| Separate traffic-probe rides | 144 |
| Hotspots | 2 graph-valid Taft choices |
| GPS sampling interval | 2 seconds |
| Traffic history | 6 frames, 5 minutes apart |
| Split | 8 train / 2 validation / 2 untouched test riders |

The injected traffic histories are:

```text
worsening:  12, 10, 8, 6, 4, 8 m/s -> choose alternative
recovering:  4,  6, 8,10,12, 8 m/s -> follow suggestion
```

Both end at 8 m/s. A model that sees only the latest frame cannot distinguish
the two histories reliably.

**Say:**

> The synthetic records follow the same six CSV schemas, OSM paths, two-second
> GPS structure, map matching, validation, feature builder, model, losses, and
> routing code as the real workflow. The labels and repeated patterns are
> controlled. This makes the experiment a capacity and integration check. It
> does not reproduce the natural variation or uncertainty of real riders.

**Generation command:**

```bat
python -m stgat_lstm synthetic-hotspot "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" generated\real\approved_training_examples.json --output-dir generated\synthetic_hotspot
```

---

## Slide 7 - Model inputs and tensor shapes

**Show:**

| Tensor | Meaning | Shape |
|---|---|---:|
| `node_features` | 13 values for every node | `[N, 13]` |
| `edge_index` | source and destination index for every directed edge | `[2, M]` |
| `edge_static` | 19 static OSM road values | `[M, 19]` |
| `edge_dynamic[t]` | 5 traffic values at time frame `t` | `[M, 5]` |
| `destination_features` | 2 values from edge endpoint to destination | `[M, 2]` |
| `routable_mask` | whether each directed edge may be routed | `[M]` |

Refined real model graph: 2,625 nodes and 6,765 directed edges, covering approved
evidence and actual route destinations. With six traffic frames, `T=6`,
`N=2625`, and `M=6765`. The original real graph has 2,379 nodes and 6,098 edges.
The archived strict/synthetic graph used
1,368 nodes and 3,418 edges. Feature dimensions stay the same.

**Feature groups:**

- Nodes: normalized coordinates, signal, street count, and nearby POI context.
- Static edges: length, reference time, direction, lanes, speed, access, bridge,
  tunnel, and road-class indicators.
- Dynamic edges: slowdown, speed ratio, observed masks, and observation age.
- Destination: normalized x and y displacement from an edge's target node.

**Say:**

> `edge_index` contains connectivity, not measured attributes. The node and edge
> feature matrices hold the measurements. The graph topology determines which
> neighbors exchange information.

---

## Slide 8 - GATv2 spatial encoder

**Code from `stgat_lstm/modeling/model.py`:**

```python
self.node_projection = nn.Linear(node_feature_dim, hidden_channels)

self.spatial_1 = GATv2Conv(
    hidden_channels,
    hidden_channels,
    heads=2,
    concat=False,
    edge_dim=edge_static_dim + edge_dynamic_dim,
    negative_slope=attention_negative_slope,
)
self.spatial_2 = GATv2Conv(
    hidden_channels,
    hidden_channels,
    heads=2,
    concat=False,
    edge_dim=edge_static_dim + edge_dynamic_dim,
    negative_slope=attention_negative_slope,
)
```

**Forward code:**

```python
node_embedding = F.relu(self.node_projection(inputs.node_features))
attention_edge_attr = torch.cat((inputs.edge_static, edge_dynamic), dim=-1)
node_embedding = F.elu(
    self.spatial_1(node_embedding, inputs.edge_index, attention_edge_attr)
)
node_embedding = F.elu(
    self.spatial_2(node_embedding, inputs.edge_index, attention_edge_attr)
)
```

**Say:**

> The linear projection changes each measured 13-value node vector into a
> learned 16-value representation. GATv2 then assigns attention to connected
> neighbors using the endpoint representations and 24 road-context values: 19
> static plus 5 dynamic. LeakyReLU with slope 0.2 is inside `GATv2Conv` when it
> calculates attention scores. ELU is applied after each layer's aggregation.
> Two heads are averaged because `concat=False`, so the width remains 16.

Layer 1 combines immediate-neighbor context. Layer 2 combines already
contextualized neighbors, giving an approximate two-hop receptive field.

---

## Slide 9 - Road-edge embeddings and LSTM

**Code from `stgat_lstm/modeling/model.py`:**

```python
edge_embedding = torch.cat(
    (
        node_embedding[source],          # 16
        node_embedding[destination],     # 16
        inputs.edge_static,              # 19
        edge_dynamic,                    # 5
        inputs.destination_features,     # 2
    ),
    dim=-1,
)                                          # [M, 58]
edge_embedding = F.relu(self.edge_projection(edge_embedding))  # [M, 16]

sequence = torch.stack(
    [self._spatial_snapshot(inputs, frame)
     for frame in inputs.temporal_edge_features],
    dim=1,
)                                          # [M, 6, 16]
embedding, _ = self.temporal(sequence)
last = embedding[:, -1, :]                 # [M, 16]
```

**Say:**

> GATv2 encodes each distinct traffic frame; exactly repeated frames reuse the
> same spatial result. For every directed road, we combine
> its source-node embedding, destination-node embedding, road attributes,
> traffic values, and destination direction. A projection compresses the 58
> values to a 16-value road embedding. The six embeddings of the same road form
> a chronological sequence. The shared LSTM learns whether the road context is
> worsening, recovering, or stable, and its final hidden state summarizes that
> history. It does not create a separate prediction; it changes the final road
> representation used by the choice head.

---

## Slide 10 - How the model learns rider preference

**Choice head:**

```python
self.edge_choice_head = nn.Sequential(
    nn.Linear(hidden, hidden),
    nn.ReLU(),
    nn.Linear(hidden, 1),
)
```

**Training target and loss from `stgat_lstm/modeling/train_model.py`:**

```python
indices = [lookup[edge_id] for edge_id in target.candidate_edge_ids]
chosen = target.candidate_edge_ids.index(target.chosen_edge_id)

choice_losses.append(F.cross_entropy(
    edge_logits[indices].unsqueeze(0),
    torch.tensor([chosen], dtype=torch.long),
))
```

For an intentional deviation, the auxiliary route-ranking loss is:

$$
L_{rank}=\operatorname{softplus}(C_{observed}-C_{suggested}).
$$

The total objective is:

$$
L=L_{choice}+\lambda L_{rank}, \qquad \lambda=0.1
$$

**Say:**

> Cross-entropy raises the utility of the outgoing road the rider chose relative
> to all other accessible roads at that junction. For intentional deviations,
> ranking loss also penalizes the model when the full observed path costs more
> than the rejected suggested path. Adam updates the GATv2, edge projection,
> LSTM, and choice-head weights together through backpropagation.

This is supervised learning. Dijkstra does not participate in gradient-based
training; route pairs are scored by summing differentiable edge costs.

---

## Slide 11 - From probabilities to a connected route

**Code from `stgat_lstm/modeling/model.py`:**

```python
probabilities = softmax(edge_logits[indices], source_nodes[indices])
costs[inputs.routable_mask] = (
    -log_probabilities[inputs.routable_mask]
    + length_tiebreaker[inputs.routable_mask]
)
```

**Formula:**

$$
c_e=-\log(p_e)+\epsilon_e,
\qquad
P^*=\arg\min_P\sum_{e\in P}c_e.
$$

**Say:**

> A highly preferred road has a large probability and therefore a small cost.
> A route's cost is the sum of its road costs. Dijkstra finds the connected path
> with the smallest total learned cost. Roads tagged `access=no` or
> `access=private` are excluded. If the learned route exceeds 1.30 times the
> shortest-distance route, the prototype falls back to the distance route to
> prevent extreme loops or detours.

The distance term in the cost is only a tiny tie-breaker. The 1.30 rule is a
system-level constraint, not a learned preference.

---

## Slide 12 - Training, validation, and testing

**Official real-data version of this slide:**

```text
15 real riders, 77 approved examples, 115 junction choices
  |
  +-- 9 training riders: 65 examples, 88 choices
  +-- 3 validation riders: 6 examples, 13 choices
  +-- 3 test riders: 6 examples, 14 choices

Compare 3 ranking weights on training/validation riders
  |
  +-- Freeze selected weight and epoch count
  +-- Refit on 12 development riders: 71 examples, 101 choices
  +-- Score the original 3 test riders: 6 examples, 14 choices
```

The refinement uses AdamW (learning rate 0.001, weight decay 0.0001), label
smoothing 0.05, training-only traffic masking 0.15, and ride-balanced choice
weights. Three ranking weights (0, 0.1, 1) are compared on validation riders.
Each trial has a 60-epoch ceiling and retains its best validation checkpoint;
early stopping waits three validation checks. The final development refit uses
the selected fixed epoch count. It does not select settings against test results.
The main binary metrics use 0.5; a separate field uses a validation-selected
threshold. The test uses known roads and previously inspected test riders, so
describe this as a repeated internal unseen-rider pilot.

```bat
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" artifacts\experiments\real_fixed_split_20261018\approved_training_examples.json --refine --first-visit-only --split-file artifacts\experiments\real_fixed_split_20261018\rider_split.json --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 60 --batch-size 8 --output artifacts\experiments\real_cleaned_osm_20261020\evaluation_report.json
```

This performs four fits: three selection trials and one final development fit.
The older five-architecture fixed-split comparison is a separate baseline. During
selection, GPS profiles exclude validation/test riders. During the final refit
and test they exclude test riders. They always exclude the target ride and future
observations. Save and reuse the rider split; do not regenerate groups to improve
the score. The extra 47 labels are real followed choices, not new deviations or
independent riders.

**If asked how overfitting is checked:** show `overfitting_checks` in the saved
report. It compares each selected checkpoint's training and validation accuracy,
and the final checkpoint's clean development/test accuracy and choice NLL.
The training objective and validation NLL have different definitions; compare
their trends carefully, not their absolute values. Also show per-rider results
and alternative-road performance. Early stopping and regularization reduce
risk, but do not prove overfitting is absent. Validation riders join development
training only after selection, and their earlier scores cannot validate the
refitted checkpoint. The test riders were inspected in the prior experiment;
describe the new result as a repeated internal pilot.

**Separate controlled synthetic version:**

**Show:**

```text
12 artificial riders
  |
  +-- 8 riders: train model weights
  +-- 2 riders: choose alternative-choice weight and threshold
  +-- 2 riders: final untouched test
```

**Say:**

> We split by rider, not by individual road choice. This prevents choices from
> the same rider appearing in both training and testing. Four class-weight
> settings are compared only on validation riders. The test riders are used
> once for final metrics. After evaluation, a separate demonstration checkpoint
> is trained on all synthetic examples; its training figures must not replace
> the held-out test report.

**Evaluation command:**

```bat
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" generated\synthetic_hotspot\approved_training_examples.json --gps-traffic-data generated\synthetic_hotspot\traffic_probes --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 10 --batch-size 8 --preference-weight 0.1 --synthetic-fixed-split --synthetic-tune --output outputs\reports\synthetic_hotspot\rider_holdout_evaluation.json
```

**Models compared:**

- prevalence baseline: always follows the common class;
- spatial GATv2: graph and latest traffic frame, no LSTM history;
- full ST-GAT-LSTM: all six chronological frames;
- latest-only LSTM: same LSTM architecture, latest frame repeated six times.

---

## Slide 13 - Controlled test results

**Show this table:**

| Model | Choice accuracy | Choice NLL | Balanced accuracy | Deviation F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Prevalence | 0.846 | 0.902 | 0.500 | 0.000 | 0.500 | 0.154 |
| GATv2 | 0.769 | 0.366 | 0.886 | 0.615 | 0.858 | 0.468 |
| ST-GAT-LSTM | **0.846** | **0.282** | **0.955** | **0.800** | **0.955** | **0.833** |
| LSTM latest-only | 0.769 | 0.368 | 0.864 | 0.571 | 0.864 | 0.500 |

All learned models obtain 1.000 route-pair ranking accuracy on the controlled
test pairs.

**Say:**

> Raw choice accuracy alone is misleading because 66 of the 78 test junction
> choices follow the suggestion. The prevalence baseline reaches 84.6 percent
> by always following, but detects no deviations. ST-GAT-LSTM has the same raw
> choice accuracy while reaching 0.800 deviation F1, 0.955 balanced accuracy,
> and much lower negative log-likelihood. It also outperforms the latest-only
> control, which is evidence that the injected temporal sequence was useful in
> this controlled experiment.

Metric meanings:

- choice accuracy: fraction of junctions where the exact chosen outgoing road
  is top-ranked;
- choice NLL: how much probability the model assigns to the actual chosen road;
  lower is better;
- balanced accuracy: average recall for followed and deviated classes;
- deviation F1: balance of precision and recall for deviations;
- ROC-AUC and AP: ranking quality for derived deviation scores.

---

## Slide 14 - Demonstrate that traffic history changes routing

Train the final demonstration checkpoint if needed:

```bat
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" generated\synthetic_hotspot\approved_training_examples.json --gps-traffic-data generated\synthetic_hotspot\traffic_probes --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --batch-size 8 --tuning-report outputs\reports\synthetic_hotspot\rider_holdout_evaluation.json --output artifacts\checkpoints\synthetic_hotspot\checkpoint.pt --report-output outputs\reports\synthetic_hotspot\training_report.json
```

Replay the same origin and destination under two histories:

```bat
python -m stgat_lstm predict artifacts\checkpoints\synthetic_hotspot\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 1034 --destination-node 35633 --gps-traffic-data generated\synthetic_hotspot\traffic_probes --traffic-cutoff-ms 1767571217000 --max-detour-ratio 1.30 --output outputs\predictions\synthetic_hotspot\worsening_route_prediction.json

python -m stgat_lstm predict artifacts\checkpoints\synthetic_hotspot\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 1034 --destination-node 35633 --gps-traffic-data generated\synthetic_hotspot\traffic_probes --traffic-cutoff-ms 1771203600000 --max-detour-ratio 1.30 --output outputs\predictions\synthetic_hotspot\recovering_route_prediction.json
```

Expected road sequences:

```text
worsening:  Decena Street -> A. Arnaiz Avenue -> Taft Avenue
recovering: Decena Street -> Protacio Street -> Taft Avenue
```

**Say:**

> The checkpoint, graph, origin, and destination are identical. Only the traffic
> cutoff changes which six historical snapshots are supplied. The model weights
> stay fixed, but the edge embeddings, probabilities, costs, and chosen route
> can change with context. Morning and afternoon requests do not retrain the
> model; they build different input frames for the existing model.

**Dashboard command:**

```bat
python -m stgat_lstm dashboard --checkpoint artifacts\checkpoints\synthetic_hotspot\checkpoint.pt --graphml "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv" --gps-traffic-data generated\synthetic_hotspot\traffic_probes --traffic-cutoff-ms 1767571217000 --max-detour-ratio 1.30
```

Open `http://127.0.0.1:8765`. Use endpoint
`/route?origin=1034&destination=35633` for the exact controlled hotspot.

---

## Slide 15 - Status, contribution, and next transition

**Implemented:**

- app-shaped data audit and automatic validation;
- directed map matching and outgoing-road label construction;
- 13 node, 19 static-edge, 5 dynamic-edge, and 2 destination features;
- two-layer edge-aware GATv2;
- six-frame edge-level LSTM;
- outgoing-road softmax and derived deviation score;
- weighted cross-entropy plus route-ranking loss;
- rider-disjoint validation and test protocol;
- learned-cost Dijkstra routing and 1.30 detour guard;
- reproducible synthetic capacity experiment and interactive dashboard.

**Limits on thesis conclusions from the final real export:**

- 17 alternative-road targets survive the revised local-choice reconstruction;
- historical GPS coverage depends on road, time slot, and rider exclusions;
- the real comparison must be reported separately from synthetic results;
- few positive targets and a single training seed limit claims of model superiority;
- probabilities have not been calibrated on an adequate independent real set;
- held-out folds must not be reused to choose hyperparameters.

**Say:**

> The implementation milestone is complete at undergraduate proof-of-concept
> level. We have now also run the final real export through the same pipeline.
> The refinement has 77 approved examples and 115 junction choices, including
> the original 17 alternative-road choices. It adds genuine followed decisions
> from training riders, corrects destination context, selects settings using
> validation riders, and refits on the 12 development riders before testing the
> same three excluded riders. Show the current report's development/test gap,
> per-rider results, and probability quality alongside accuracy. This remains
> an internally evaluated prototype with a small, previously inspected test set.
> Synthetic results are a separate controlled capacity check.

### Current refined real results to show

Open `artifacts\experiments\real_refined_20261018\run_summary.json` for the short result and
`evaluation_report.json` for detailed predictions, per-rider metrics, and
`overfitting_checks`. The selected nine-rider checkpoint scores 12/13 on
validation at epoch 45 with ranking weight 1.0. The final checkpoint was then
refitted separately on the 12 development riders.

| Measure | Final development | Final test | Always-follow test baseline |
|---|---:|---:|---:|
| Preferred outgoing roads correct | 67/101 | 9/14 | 10/14 |
| Choice accuracy | 66.3% | 64.3% | 71.4% |
| Clean choice NLL | 0.813 | 0.911 | 0.911 |
| Route pairs correct | 8/9 | 2/3 | n/a |

**Say:**

> The refinement did not reach our roughly 80% target and did not beat the
> always-follow baseline on exact road choice. Test balanced accuracy is 0.650
> and deviation F1 is 0.500. It identifies eight of ten followed roads and one
> of four actual alternative roads. Binary deviation detection recognizes two
> deviations, but one chooses the wrong alternative, so these are different
> metrics. Relative to the older pilot, probability quality and route ranking
> improve, while exact choice accuracy decreases by one road. The small
> development/test gap does not prove no overfitting. This remains a small
> internal pilot on previously inspected test riders.

The accuracy gap is 2.05 percentage points and the NLL gap is 0.099. The selected
trial's validation accuracy drops from 12/13 at epoch 45 to 10/13 at epoch 60;
the code retains epoch 45. Test per-rider results are 1/1, 8/11, and 0/2.
The same 14 target roads and candidate sets are preserved. No extra settings
were selected against the new test score. The older architecture table below
uses different preparation and training, so it is not a controlled architecture
comparison with this refined checkpoint.

All 89 tests and checkpoint reload pass, including reproduction of all 14 test
predictions. A historical 400-to-6 route is connected and 1,365.8 m versus the
1,320.5 m distance baseline, passing the detour guard without fallback. These
checks establish pipeline operation, not independently validated route quality.

### Original real fixed-split baseline

The report is `artifacts\experiments\real_fixed_split_20261018\evaluation_report.json`;
`run_summary.json` is the shorter explanation. Five variants were each trained
once for 100 epochs. The test set contains six examples, 14 choices, four
alternative targets, and three comparable route pairs from three riders.
Binary metrics below use the fixed 0.5 threshold.

| Model | Test road-choice accuracy | Test choice NLL | Balanced accuracy | Deviation F1 | Route ranking |
|---|---:|---:|---:|---:|---:|
| Suggestion baseline | 0.714 | 0.830 | 0.500 | 0.000 | n/a |
| GCN | 0.643 | 2.940 | 0.600 | 0.444 | 0.667 |
| GAT | 0.643 | 4.175 | 0.650 | 0.500 | 0.333 |
| GATv2 | 0.643 | 3.562 | 0.600 | 0.444 | 0.333 |
| ST-GAT-LSTM | 0.714 | 3.373 | 0.650 | 0.500 | 0.333 |
| LSTM latest-only | 0.643 | 3.974 | 0.525 | 0.286 | 0.333 |

**Explain the limitations alongside the table:**

- ST-GAT-LSTM scores 36/41 choices on training riders versus 10/14 on test riders.
- It identifies two alternatives, misses two, and falsely flags two followed choices.
- Lower NLL is better. Its NLL is worse than the baseline because some errors are highly confident; do not present its probabilities as calibrated.
- Full history beats latest-only by one road choice; a single seed and four alternative targets cannot establish a stable temporal benefit.
- One test rider contributes 11/14 choices. Per-rider results are 1/1, 9/11, and 0/2; pooled accuracy alone hides that variation.
- Only 1/6 test examples has changing observed traffic on its suggested path. Five have some traffic observations somewhere in the graph.
- The saved groups were chosen before fitting. Do not regenerate them or repeatedly tune against these test results.

### Archived strict-baseline evidence to have available

The final dataset was supplied on 14 October 2026. The archived run is under
`artifacts\experiments\real_final_20261014`; the folder name records the export/run start date.

| Quantity | Final real export / retained examples |
|---|---:|
| Source rides / riders | 69 / 15 |
| GPS points | 7,826 |
| Reported deviations / responses | 37 / 37 |
| Post-trip questionnaires | 69 / 69 |
| Automatically approved examples | 16 from ten riders |
| Approved followed / deviated examples | 13 / 3 |
| Junction targets | 54: 51 followed, 3 alternative roads |
| Matched speed segments / roads | 6,801 / 161 |
| Median GPS interval | 2.001 seconds |
| Full-data training loss | 2.0788 to 0.2040 after 100 epochs |
| In-sample road choices / ranking pairs correct | 43/54 and 3/3 |

The real held-out report trains each architecture for 40 epochs in each of ten
rider folds. Inspect `rider_holdout_evaluation.json` for the final values rather
than substituting the synthetic table on Slide 13. Always following the
suggestion reaches 51/54 = 0.944 choice accuracy but zero deviation F1.

| Model | Real choice accuracy | Real balanced accuracy | Real deviation F1 | Real route ranking |
|---|---:|---:|---:|---:|
| Suggestion baseline | 0.944 | 0.500 | 0.000 | n/a |
| GCN | 0.593 | 0.784 | 0.214 | 0.333 |
| GAT | 0.574 | 0.794 | 0.222 | 0.333 |
| GATv2 | 0.667 | 0.814 | 0.240 | 0.667 |
| ST-GAT-LSTM | 0.556 | 0.745 | 0.188 | 1.000 |
| LSTM latest-only | 0.537 | 0.745 | 0.188 | 0.333 |

**Say about the real results:**

> The real outgoing-road results are weak. The full model detects all three
> deviations but falsely flags 26 followed choices. Spatial GATv2 performs
> better on choice accuracy and F1. Full history improves probability ranking
> and route-pair ranking over latest-only, but not deviation F1. Three positive
> targets and three route pairs are too few to establish a reliable advantage.

Use `artifacts\experiments\real_final_20261014\run_summary.json` as the readable record of
audits, retention, training, evaluation, traffic, routing, and provenance.

Both tested fresh real route requests triggered the 1.30 detour guard. This
shows that connected routing and the guard execute, but does not prove good
learned route quality. A reviewed deviation replay correctly favors its
observed route; that replay is an in-sample demonstration.

The friend's supplied notebook explores six candidates and an 8 m backward
GPS-jitter tolerance. It preserves results in memory rather than exporting
directed edge sequences into the CSVs. This archived real run uses the former
project matcher. The notebook's latter trimming and manual audit sections are
obsolete at the user's instruction and are not used.

---

## State-of-the-art and architecture slide or backup

Use this comparison if the panel asks what published work does and how this
project differs.

| Component | Published basis | What this project does |
|---|---|---|
| Graph attention | GAT, Velickovic et al. (2018) | Uses graph neighborhoods instead of a flat road table |
| Dynamic graph attention | GATv2, Brody et al. (2022) | Uses PyTorch Geometric `GATv2Conv` with node and edge context |
| Temporal memory | LSTM, Hochreiter and Schmidhuber (1997) | Encodes six road-context frames per edge |
| Spatial then temporal order | ST-GAT traffic forecasting precedent, Zhang et al. (2019) | Applies GATv2 per frame and LSTM across the resulting edge embeddings |
| Route preference | Pairwise preference/ranking idea | Learns chosen outgoing roads and ranks observed intentional paths below rejected suggestions in cost |
| Route construction | Dijkstra | Searches over learned positive additive edge costs |

**Defensible answer:**

> We use standard, maintained GATv2 and LSTM components. Our contribution is
> their task-specific integration with destination-conditioned road embeddings,
> GPS-derived motorcycle traffic histories, supervised outgoing-road choices,
> a route-ranking objective, and learned-cost routing. We should call the
> implementation state-of-the-art-informed, not claim that its current real-data
> performance is state of the art.

The model uses the **PyTorch Geometric GATv2 library operator**. It is not a
handwritten reproduction of the authors' repository and not a line-for-line
copy of a complete paper architecture.

## The six files to understand first

| Order | File | Why it matters |
|---:|---|---|
| 1 | `stgat_lstm/preprocessing/graph_data.py` | Defines features, Taft subgraph, accessible edges, and model tensors |
| 2 | `stgat_lstm/modeling/model.py` | Implements node projection, GATv2, edge projection, LSTM, choice head, and costs |
| 3 | `stgat_lstm/modeling/train_model.py` | Builds choice targets, attaches traffic, computes losses, trains, and saves checkpoint/report |
| 4 | `stgat_lstm/modeling/evaluate_model.py` | Creates rider splits, trains baselines and ablations, selects validation settings, and reports test metrics |
| 5 | `stgat_lstm/preprocessing/gps_traffic.py` | Converts consecutive GPS fixes into leakage-aware historical road-speed frames |
| 6 | `stgat_lstm/routing/predict_route.py` | Loads the checkpoint, scores roads, runs Dijkstra, applies detour guard, and formats route output |

Supporting files:

| Concern | Files |
|---|---|
| Command dispatcher | `stgat_lstm/__main__.py` |
| Input audits | `audit_rider_data.py`, `audit_road_network.py` |
| Map matching | `map_matching.py` |
| Candidate construction | `build_deviation_candidates.py`, `build_decision_candidates.py` |
| Automatic approval and map | `review_decisions.py`, `create_review_map.py` |
| Synthetic generator | `synthetic_hotspot.py` |
| Dashboard | `dashboard.py` |
| Regression checks | `tests/test_*.py` |

The files are modules called by `python -m stgat_lstm <command>`. You normally
do not run `model.py` or `train_model.py` directly. `__main__.py` dispatches the
command to the correct module, which imports and calls the architecture and
training functions.

### Complete Python module map

| File | Called by or used for | Purpose |
|---|---|---|
| `__init__.py` | Python package import | Marks `stgat_lstm` as the project package |
| `__main__.py` | Every `python -m stgat_lstm ...` command | Parses the first command and forwards remaining arguments to its module |
| `audit_rider_data.py` | `audit-data` | Counts rides, GPS fixes, questionnaires, gaps, and provisional deviations |
| `audit_road_network.py` | `audit-network`; imported by matching and traffic | Checks graph CRS, features, Taft coverage, and provides the road spatial index |
| `audit_map_matching.py` | `audit-matching` | Measures whether eligible rider events can be sequence-matched to OSM |
| `geometry.py` | Imported by audit, matching, and traffic | Haversine distance, polylines, and small geometry helpers |
| `coordinates.py` | Imported by graph and routing modules | Implements the project's WGS84/UTM coordinate conversion without an external PROJ DLL |
| `map_matching.py` | Candidate builders and map-match audit | HMM/Viterbi sequence matching from noisy GPS fixes to directed roads |
| `preference_pairs.py` | Deviation-candidate construction | Finds comparable observed and suggested subpaths with common endpoints |
| `build_deviation_candidates.py` | `build-deviations` | Extracts intentional local road choices and optional comparable route pairs |
| `build_decision_candidates.py` | `build-decisions` | Adds followed controls and converts routes into outgoing-road decisions |
| `review_decisions.py` | `review auto` | Applies deterministic validity rules and records approval/rejection reasons |
| `create_review_map.py` | `review-map` | Writes the local Leaflet map used to inspect green and red paths |
| `graph_data.py` | Training, evaluation, prediction | Extracts the Taft graph and creates all model tensors and choice targets |
| `gps_traffic.py` | `gps-traffic`; training, evaluation, prediction | Derives historical motorcycle probe-speed frames from two-second GPS fixes |
| `mapbox_traffic.py` | Optional `traffic` and legacy live-traffic flags | Compatibility path for Mapbox observations; not the main thesis traffic source |
| `model.py` | Imported by train, evaluate, and predict | Contains GCN/GAT/GATv2 baselines and the final GATv2-LSTM choice model |
| `train_model.py` | `train real` | Prepares approved examples, attaches traffic, optimizes weights, and saves checkpoint/report |
| `evaluate_model.py` | `evaluate` | Runs rider-disjoint splits, validation selection, baselines, ablations, and metrics |
| `predict_route.py` | `predict` and dashboard | Reloads a checkpoint, creates probabilities and costs, runs Dijkstra, and writes JSON |
| `synthetic_hotspot.py` | `synthetic-hotspot` | Generates app-shaped decisions and separate controlled traffic probes |
| `dashboard.py` | `dashboard` | Serves coordinate snapping, route requests, and the browser interface |
| `dashboard.html` | Loaded by `dashboard.py` | Browser-side map and route visualization |

### What the test files cover

The files under `tests/` are automated regression checks. They are run together
with `python -m unittest discover -s tests -v`; they are not separate pipeline
steps.

| Test group | Main assurance |
|---|---|
| `test_model.py`, `test_training_pipeline.py` | Tensor shapes, gradients, losses, training, checkpoint provenance |
| `test_evaluation.py` | Rider separation, controls, metrics, and synthetic tuning protocol |
| `test_routing.py`, `test_dashboard.py` | Positive-cost Dijkstra, access filtering, detour guard, API behavior |
| `test_map_matching.py`, `test_preference_pairs.py` | Directed sequence matching and common-endpoint pairs |
| `test_decision_dataset.py`, `test_review.py`, `test_review_map.py` | Candidate labels, automatic approval, and map output |
| `test_gps_traffic.py`, `test_traffic.py` | Speed derivation, temporal history, missingness, and optional traffic compatibility |
| `test_road_network.py`, `test_geometry.py`, `test_coordinates.py` | OSM coverage, distance, geometry, and coordinate conversion |
| `test_synthetic_hotspot.py` | Synthetic schemas, rider splits, traffic separation, and reproducibility |
| `graph_test_data.py` | Shared small graph fixture used by the tests; it is not training data |

## Presentation-day sequence

The full reproducible workflow is in `SYNTHETIC_RUNBOOK.md`. On presentation
day, avoid regenerating artifacts unless the panel specifically asks. Use:

```bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
call .venv\Scripts\activate.bat
python -m unittest discover -s tests -v
start "" "outputs\maps\synthetic_hotspot\candidate_review_map.html"
python -m stgat_lstm predict artifacts\checkpoints\synthetic_hotspot\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" generated\synthetic_hotspot\approved_training_examples.json --gps-traffic-data generated\synthetic_hotspot\traffic_probes --output outputs\predictions\synthetic_hotspot\reviewed_example_prediction.json
python -m stgat_lstm dashboard --checkpoint artifacts\checkpoints\synthetic_hotspot\checkpoint.pt --graphml "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv" --gps-traffic-data generated\synthetic_hotspot\traffic_probes --traffic-cutoff-ms 1767571217000 --max-detour-ratio 1.30
```

Open these saved results before the presentation:

```text
generated\synthetic_hotspot\manifest.json
outputs\audits\synthetic_hotspot\gps_traffic_audit.json
outputs\reports\synthetic_hotspot\rider_holdout_evaluation.json
outputs\reports\synthetic_hotspot\training_report.json
outputs\predictions\synthetic_hotspot\worsening_route_prediction.json
outputs\predictions\synthetic_hotspot\recovering_route_prediction.json
```

## Likely panel questions

### How does the model actually learn rider behavior?

Every supervised junction lists the legal outgoing roads and the road the rider
actually chose. Cross-entropy penalizes low probability on that chosen road.
For intentional deviations, route-ranking loss also penalizes the model when
the observed route has a higher total learned cost than the rejected suggestion.
Backpropagation changes the GATv2, edge projection, LSTM, and choice-head weights
so similar graph, destination, and traffic contexts favor similar choices.

### Is the deviation value a label or a probability?

The stored supervision is categorical: followed is 0 and deviated is 1. The
model outputs a probability distribution over outgoing roads. The candidate
road's derived deviation probability is `1 - P(candidate road)`. A numeric
probability should not be called calibrated until enough real held-out examples
exist.

### What exactly does the LSTM learn?

It learns transformations of chronological road embeddings that help predict
the outgoing road. In the controlled data, worsening and recovering speed
sequences end at the same latest speed, so sequence order contains information
that one frame lacks. The LSTM's output still goes to the same road-choice head;
it does not separately forecast traffic.

### Why are there six embeddings?

There is one GATv2-derived edge embedding per five-minute history slot. Six
frames cover five transitions across approximately 25 minutes. Six is a current
configuration, not a universal optimum; it can be changed and later tuned on
validation data.

### Does the model retrain for a different time of day?

No. The trained weights remain fixed. A request at another time builds a
different six-frame traffic tensor, which can produce different edge embeddings,
probabilities, costs, and route.

### Why does the synthetic data repeat routes?

Repeated observations create a consistent learnable signal across riders and
traffic regimes. They are intentional samples, not duplicate rows accidentally
counted twice. Rider-disjoint splitting checks whether the injected rule carries
to artificial riders absent from training.

### Why is the prevalence baseline accuracy already 0.846?

There are 66 followed and 12 deviated junction choices in the test set. Always
predicting followed is correct 66/78 times, or 0.846, but its deviation F1 is
zero and balanced accuracy is 0.5. This is why accuracy must be read with the
class-sensitive metrics and negative log-likelihood.

### Does Dijkstra train the network?

No. Training uses differentiable choice and ranking losses. Dijkstra is called
after the network produces positive edge costs to construct a valid connected
route.

### Do nodes and edges have weights?

Nodes and edges both have input features. GATv2 learns attention coefficients
on graph connections. The final scalar routing cost belongs to each directed
edge. Dijkstra sums edge costs; it does not sum node features.

### Why convert node embeddings into edge embeddings?

The decision and route concern a directed road, not only an intersection. A road
needs information from both endpoints plus its lanes, class, speed context, and
direction toward the destination. Concatenating these values gives every road a
representation that the LSTM and choice head can score.

### Is the project personalized?

It learns shared patterns across riders. Rider identity is used to prevent data
leakage in evaluation, but is not an input embedding. True individual
personalization would require enough repeated choices per rider and a separate
history or rider embedding.

### What does the 1.30 distance constraint do?

The system computes the shortest-distance route as a guard. If the unconstrained
learned route is more than 1.30 times that distance, it returns the distance
route. This prevents an immature cost model from selecting extreme detours.

### What came from the older project?

The present repository uses the earlier work as domain and data-format context.
The current GATv2-LSTM, outgoing-road supervision, temporal GPS profiles,
evaluation, and Dijkstra integration are implemented in this codebase; there
are no runtime imports from the old folder.

### Are the current results valid?

They are valid for the controlled synthetic experiment because train,
validation, and test riders are separated and the test labels are not used for
selection. They establish implementation capacity under the injected rule.
They are not an estimate of performance on real motorcycle riders.

### What happens when real data is ready?

Audit the new export, reconstruct and automatically validate candidates, build
historical GPS profiles that predate each decision, run rider-disjoint
evaluation, and train a final checkpoint using the selected settings. The main
code does not need a new architecture merely because the data source changes.
