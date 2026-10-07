# STGAT-LSTM rider preference prototype

**Interactive demonstration:** activate your environment, run `python -m stgat_lstm dashboard`, and open http://127.0.0.1:8765. Select two points to see the learned route and shortest-distance comparison on your supplied OSM network. The complete training, evaluation, and traffic commands are included below.

The repository layout separates command files, automatically imported implementation files, official real inputs, tests, and generated outputs. Required OSM and rider-export inputs are stored under `data/real/`, so normal commands do not depend on sibling project folders.

Detailed project notes are limited to the [project plan](docs/PLAN.md), [command runbook](docs/RUNBOOK.md), [presentation guide](docs/PRESENTATION_GUIDE.md), and [complete model explanation](docs/MODEL_EXPLANATION.md).

The training, inference, routing, and local visualization connections are implemented. Timestamped rider GPS is direction-matched to OSM roads and converted into historical motorcycle speed profiles for the LSTM. This does not establish temporal usefulness without broader coverage, or rider generalization without further held-out data. The router enforces graph direction/access flags; OSM turn restrictions are outside the agreed research scope and are not represented. It remains a research prototype.

Command-line tools keep the full result in JSON (or HTML for the map views) and print a shorter explanation in the terminal. The terminal is for quick checking; the saved artifact is the record used for review, training, and reporting.

Evaluation reports balanced accuracy, per-class precision/recall/F1, macro-F1, AP, ROC-AUC, Brier score, log loss, confusion counts, and held-out preference ranking. `python -m stgat_lstm evaluate --temporal-ablation` adds a matched LSTM latest-only control when histories contain observed traffic variation; otherwise it records why temporal evidence is insufficient.

This repository contains an executable rider-choice pipeline. It reconstructs reviewable road choices, trains a GATv2-LSTM road-deviation classifier and path-preference model, produces positive road costs, and passes those costs to a shortest-path router. Travel-time savings are an optional secondary outcome, not the training target. The current real checkpoint was trained on 28 supervised road choices reconstructed from eight approved examples and six riders, with six GPS-derived history frames per decision; its results are pilot evidence, not a generalization claim. A separated 96-example synthetic experiment verifies multiple injected Taft choices and controlled temporal histories on unseen synthetic riders.

## Repository layout

The modules reflect separate data, modeling, review, evaluation, and routing stages. Start with these files:

| Area | Primary files | Purpose |
|---|---|---|
| Candidate construction | `build_deviation_candidates.py`, `build_decision_candidates.py` | Reconstruct strict deviation paths, then combine followed and deviated local choices |
| Candidate validation | `create_review_map.py`, `review_decisions.py` | Apply reproducible survey/GPS rules and optionally display candidate paths |
| Model and evaluation | `model.py`, `train_model.py`, `evaluate_model.py` | Define and train the models, then compare held-out riders across baselines |
| Inference and routing | `predict_route.py` | Reload the checkpoint, predict deviation, score edges, and run Dijkstra |
| Real graph and traffic | `graph_data.py`, `gps_traffic.py` | Build OSM tensors and derive historical road-speed context from rider GPS |
| Controlled capacity test | `synthetic_hotspot.py` | Generate separated app-shaped decisions with multiple Taft patterns and controlled traffic history |

Supporting modules such as `geometry.py`, `map_matching.py`, `audit_road_network.py`, `audit_map_matching.py`, and `preference_pairs.py` are imported by those primary files. They are not duplicate entry points. `audit_rider_data.py` is an optional diagnostic. `__main__.py` provides the single `python -m stgat_lstm` command interface. Files under `tests` verify the real-data pipeline with small in-memory graphs. Files under `outputs` are generated artifacts rather than source code.

## What runs

- `stgat_lstm/model.py` contains a two-layer GATv2 graph encoder, an LSTM across graph snapshots and a learned edge preference cost. It also provides simple GAT and GCN baselines under the same preference loss. The output represents cost per road length integrated over each edge; it is dimensionless, not travel seconds.
- The same module provides `DecisionPreferenceModel`, which shares the STGAT-LSTM edge representation between a per-road deviation head and the learned edge-cost head. It returns one logit and one cost for every directed OSM edge. Suggested roads are available at inference; later observed paths and surveys are labels only.
- `stgat_lstm/train_model.py` trains both heads, saves a checkpoint and refuses real decision artifacts that have not passed hash-bound validation.
- The GAT and GATv2 operators explicitly use the standard attention `negative_slope=0.2` for their internal LeakyReLU attention calculation. The model applies ELU after attention layers; it does not apply a second LeakyReLU to their outputs.
- `stgat_lstm/audit_rider_data.py` reads the official real CSV exports and reports aggregate readiness checks. It parses route geometry, measures deviation locations against the latest pre-event route, checks route endpoints and summarizes GPS coverage around each event.
- `stgat_lstm/audit_road_network.py` checks whether the supplied projected, directed OSM graph can support map matching, measures route/GPS distance to its edges, verifies the separate node-feature join, and quantifies how much of the current collection actually falls inside the provisional Taft Avenue corridor.
- `stgat_lstm/map_matching.py` implements directed, sequence-aware HMM/Viterbi map matching. It combines GPS-to-road distance with network-versus-observed displacement and retains the complete traversed edge sequence, including connector edges between observations.
- `stgat_lstm/preference_pairs.py` finds shared directed edges before and after a deviation, then extracts different observed and suggested paths with exactly the same origin and destination.
- `stgat_lstm/build_deviation_candidates.py` creates a pseudonymous, coordinate-free review artifact from the real export. It does not feed unreviewed candidates into training.
- `stgat_lstm/build_decision_candidates.py` creates conservative local follow-versus-deviate candidates. Follow labels require GPS agreement through a real branching node; deviation labels come from the stricter divergence/rejoin pipeline. Later GPS and surveys are label evidence only.
- `stgat_lstm/graph_data.py` extracts a 1 km Taft Avenue OSM corridor, joins POI features, builds configurable model tensors, marks absent historical traffic as unknown, and refuses to load review-required choices for training.
- `stgat_lstm/predict_route.py` reloads real checkpoints, scores OSM edges and runs Dijkstra over positive directed-edge costs, excluding restricted-access edges. A configurable hard guard defaults to 1.30 times the shortest-route distance and falls back to the distance baseline when the unconstrained learned route is excessive.
- `stgat_lstm/mapbox_traffic.py` collects timestamped Mapbox `driving-traffic` annotations without storing the token, aligns observed route segments to directed OSM edges, and keeps speed and congestion availability separate.
- `stgat_lstm/gps_traffic.py` converts consecutive two-second rider GPS fixes into direction-matched motorcycle speed records. It creates pre-decision time-of-day histories, excludes the target ride and held-out riders, and never writes raw coordinates to its audit.
- `stgat_lstm/synthetic_hotspot.py` creates a separated controlled experiment using multiple verified Taft decisions, traffic-only GPS probe rides, and the same six app CSV schemas. Its output tests implementation and temporal capacity and is never treated as real rider evidence.
- `stgat_lstm/review_decisions.py` automatically accepts supported intentional reasons and strong followed examples after directed-path and GPS checks. It excludes unsupported reasons, missing traffic severity and inconsistent geometry, then creates a hash-bound training artifact. Manual templates remain available for exceptional audits.

For a matched chosen path \(P_c\) and rejected suggestion \(P_s\), the training loss is `softplus(cost(P_c) - cost(P_s))`. The model receives gradients from this loss, so rider choices change its trainable parameters. At routing time, the model scores the accessible graph and the search algorithm selects a connected path with the lowest learned cost. The model does not impose a faster-path preference.

## Run

Use a Python environment with the versions in `requirements.txt`. The code was tested on Windows with CPU PyTorch 2.13.0 and PyTorch Geometric 2.8.0.post1. From a Windows **cmd** terminal:

```bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m stgat_lstm audit-data "data\real\rider_exports"
python -m stgat_lstm audit-network "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv"
python -m stgat_lstm audit-matching "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml"
python -m stgat_lstm gps-traffic "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --output outputs\gps_traffic_audit.json
python -m stgat_lstm build-deviations "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --output outputs\deviation_candidates.json --geojson-output outputs\deviation_candidates.geojson
python -m stgat_lstm build-decisions "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --deviation-candidates outputs\deviation_candidates.json --output outputs\decision_candidates.json --geojson-output outputs\decision_candidates.geojson
python -m stgat_lstm review-map outputs\decision_candidates.geojson --output outputs\candidate_review_map.html
python -m stgat_lstm review auto outputs\decision_candidates.json --output outputs\approved_training_examples.json
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --gps-traffic-data "data\real\rider_exports" --epochs 100 --batch-size 8 --output outputs\gatv2_lstm_checkpoint.pt
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --gps-traffic-data "data\real\rider_exports" --epochs 40 --temporal-ablation --output outputs\rider_holdout_evaluation.json
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --gps-traffic-data "data\real\rider_exports" --output outputs\reviewed_example_prediction.json
python -m stgat_lstm dashboard --gps-traffic-data "data\real\rider_exports" --max-detour-ratio 1.30
```

The aggregate audits are required whenever the official rider exports or OSM inputs change.

The main traffic source is the app's GPS track. Consecutive fixes are converted to speed, direction-matched to OSM edges, filtered for long gaps and implausible speeds, and aggregated by local time of day. The five dynamic fields are slowdown relative to the OSM reference, speed ratio, two observation masks, and observation age. Missing coverage stays explicitly unknown. The current export yields 6,657 usable speed segments on 494 roads from 48 rides, with a 2.001-second median interval. In leakage-safe held-out evaluation, only two of eight examples have any observed history, only one has changing values, and none has changes on its suggested path.

Mapbox ingestion remains as optional compatibility code for earlier experiments. It is not required by the current thesis pipeline, and present Mapbox observations are never attached retrospectively to old rides.

If the `py -3.12` launcher is unavailable, use `python -m venv .venv` instead. A new cmd window must be activated again with `call .venv\\Scripts\\activate.bat`.

On the reviewed machine, the existing `C:\Users\Vince\Downloads\THESIS_APP\finalGNNT3\venv\Scripts\python.exe` already has both model libraries and can run these commands in place of `python`. The default system Python does not currently have PyTorch installed.

The audit uses only the Python standard library, so it can run with a separate Python installation even before ML dependencies are installed.

## Verified behavior and limits

The automated suite exercises prior-route reconstruction, exclusion of wrong turns and personal stops, missing and late traffic, incomplete GPS observations, local follow-choice construction, all model forward/backward paths, motorcycle access, pre-choice inference isolation, review hashes, evaluation, and learned-cost routing.

The preliminary leave-one-rider-out command compares a training-prevalence baseline, GCN, original GAT, spatial-only GATv2, and GATv2-LSTM. On the current 28 road decisions reconstructed from eight examples, held-out classification is poor and unstable. Historical GPS coverage is now represented, but it is sparse and uneven at these decision times. These counts justify further collection and do not establish model superiority or an LSTM benefit. The spatial-only GATv2 and latest-only LSTM controls remain necessary.

The official real export contains 56 rides from 12 riders, 170 saved routes, 34 reported deviations, 34 completed deviation responses, and 11,435 GPS points. All 34 deviation records reference a route generated after their event timestamp, so the pipeline reconstructs the latest route that existed before each event and then performs directed sequence map matching. The completed decision artifact still contains eight approved local examples from six riders: five followed and three intentionally deviated. They produce 28 supervised road choices: 25 followed and three rejected.

The OSM compatibility command can take several seconds because it loads the 74 MB Metro Manila graph and spatially checks every saved route and GPS point. Its independent nearest-edge results are quality diagnostics. They are not sequence-aware map matching and must not be used directly as rider-choice targets.

The map-match audit applies the HMM only to intentional deviation events inside the provisional Taft corridor. It compares the post-event GPS edge sequence with the prior and regenerated route sequences and reports candidates where the routes differ and GPS agrees more with the regenerated route. These still require divergence/rejoin review before they can become common-origin/common-destination preference pairs.

On the current export, the compatibility audit found 17 intentional deviation events inside the provisional corridor. The HMM sequence screen matched GPS, prior route and regenerated route for 11 events; 8 of those agree more strongly with the regenerated route and are retained as review candidates. This is a data-preparation result, not model accuracy.

The stricter candidate builder additionally requires shared directed edges before and after the event and emits equal-origin/equal-destination subpaths. Its JSON contains pseudonymous grouping keys and no coordinates. The optional GeoJSON contains only the selected OSM road geometry for visual review; it does not contain raw rider GPS points.

The decision-dataset builder uses one local route choice as the common prediction unit. A followed example must traverse the suggested path through an OSM node with at least two outgoing road choices. A deviated example must have an intentional survey response plus a GPS-supported divergence and rejoin. On the current export it produces 8 review-required candidates from 6 riders: 5 followed and 3 deviated. This class balance is suitable for pipeline development only, not a thesis performance claim or a stable train/test split.

The review-map command creates a local HTML page with a candidate selector, path details, embedded OSM context roads, and separate styling for the observed and suggested paths. It loads the Leaflet library from a CDN but does not request public map tiles. The embedded candidate layer contains OSM path geometry, not raw GPS.

On the current export, the strict deviation stage retains three candidates from three rides and three riders. One is a shorter familiar/shortcut path; two are longer paths associated with a reported road blockage and intersection avoidance. They feed the unified eight-example decision artifact, whose entries were reviewed and approved before real training.

The current shortest-path implementation assumes additive edge costs. It applies a configurable maximum-distance guard after learned-cost routing and returns the shortest-distance baseline when the unconstrained result exceeds that limit. Turn-dependent restrictions and costs require an augmented search state. The toy graph uses invented road access and traffic values; it makes no claim about Taft Avenue.

The model implementation draws on the [original GAT paper](https://arxiv.org/abs/1710.10903), [GATv2 paper](https://arxiv.org/abs/2105.14491), [original LSTM paper](https://doi.org/10.1162/neco.1997.9.8.1735), [spatial-temporal GAT traffic paper](https://doi.org/10.1109/ACCESS.2019.2953888), and [PyTorch Geometric operators](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html). These are component references, not claims that the current architecture reproduces a full published traffic model.
