# STGAT-LSTM rider preference prototype

This repository now contains the first executable model slice. It learns **rider route preferences** from path comparisons, produces positive road costs, and passes those costs to a shortest-path router. Travel-time savings are an optional secondary outcome, not the training target. The current trained artifacts use only a fictional six-node graph and four designed preference pairs; they are software checks, not evidence about Metro Manila riders.

## What runs

- `stgat_lstm/fixture.py` reconstructs the suggested route that existed before a choice, screens incomplete trajectories and excluded reasons, and keeps only traffic available at decision time.
- `stgat_lstm/features.py` builds graph and temporal features for the **synthetic fixture**. Its main/side traffic mapping is deliberately toy-specific. Real Mapbox traffic requires alignment to OSM edges.
- `stgat_lstm/model.py` contains a two-layer GATv2 graph encoder, an LSTM across graph snapshots and a learned edge preference cost. It also provides simple GAT and GCN baselines under the same preference loss. The output represents cost per road length integrated over each edge; it is dimensionless, not travel seconds.
- The same module provides `DecisionPreferenceModel`, which shares the STGAT-LSTM edge representation between a binary follow/deviate head and the learned edge-cost head. Suggested-route features are available at inference; later observed paths and surveys are labels only.
- `stgat_lstm/decision_training.py` trains both heads, saves a checkpoint and refuses real decision artifacts that have not passed the hash-bound manual review.
- The GAT and GATv2 operators explicitly use the standard attention `negative_slope=0.2` for their internal LeakyReLU attention calculation. The model applies ELU after attention layers; it does not apply a second LeakyReLU to their outputs.
- `stgat_lstm/training.py` trains directly on observed-versus-suggested intentional choices. Lower learned cost means preferred. Surveys determine which comparisons are suitable; a future survey answer is never required as a model input at route time.
- `stgat_lstm/routing.py` uses Dijkstra on learned positive costs and removes roads marked inaccessible to motorcycles.
- `stgat_lstm/predict.py` reloads a saved synthetic checkpoint and uses its learned scores to route in a separate process. It builds a routing request from pre-choice context and does not consume the observed path or later survey response.
- `stgat_lstm/audit.py` reads the current real CSV exports and reports aggregate readiness checks without mixing them into synthetic training. It parses route geometry, measures deviation locations against the latest pre-event route, checks route endpoints and summarizes GPS coverage around each event.
- `stgat_lstm/osm_audit.py` checks whether the supplied projected, directed OSM graph can support map matching, measures route/GPS distance to its edges, verifies the separate node-feature join, and quantifies how much of the current collection actually falls inside the provisional Taft Avenue corridor.
- `stgat_lstm/map_matching.py` implements directed, sequence-aware HMM/Viterbi map matching. It combines GPS-to-road distance with network-versus-observed displacement and retains the complete traversed edge sequence, including connector edges between observations.
- `stgat_lstm/preference_pairs.py` finds shared directed edges before and after a deviation, then extracts different observed and suggested paths with exactly the same origin and destination.
- `stgat_lstm/build_real_candidates.py` creates a pseudonymous, coordinate-free review artifact from the real export. It does not feed unreviewed candidates into training.
- `stgat_lstm/build_decision_dataset.py` creates conservative local follow-versus-deviate candidates. Follow labels require GPS agreement through a real branching node; deviation labels come from the stricter divergence/rejoin pipeline. Later GPS and surveys are label evidence only.
- `stgat_lstm/real_data.py` extracts a 1 km Taft Avenue OSM corridor, joins POI features, builds configurable model tensors, marks absent historical traffic as unknown, and refuses to load review-required choices for training.
- `stgat_lstm/real_routing.py` sends real OSM tensors through the configured preference model and runs Dijkstra over the resulting positive directed-edge costs, excluding edges explicitly tagged with restricted access.
- `stgat_lstm/mapbox_traffic.py` collects timestamped Mapbox `driving-traffic` annotations without storing the token, aligns observed route segments to directed OSM edges, and keeps speed and congestion availability separate.
- `stgat_lstm/review_decisions.py` creates a review-decision template and compiles completed approve/reject decisions into a hash-bound training artifact. Pending or stale decisions cannot enter training.

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
python -m stgat_lstm.demo --epochs 60 --output-dir outputs/synthetic_demo
python -m stgat_lstm.predict outputs/synthetic_demo/stgat_lstm_synthetic.pt --case-id syn_case_01
python -m stgat_lstm.audit "..\\ExportLMD\\clean_data"
python -m stgat_lstm.osm_audit "..\\ExportLMD\\clean_data" "..\\OSM\\data\\osm\\graphs\\metro_manila_processed.graphml" --node-features "..\\OSM\\data\\features\\road_node_features.csv"
python -m stgat_lstm.map_match_audit "..\\ExportLMD\\clean_data" "..\\OSM\\data\\osm\\graphs\\metro_manila_processed.graphml"
python -m stgat_lstm.build_real_candidates "..\\ExportLMD\\clean_data" "..\\OSM\\data\\osm\\graphs\\metro_manila_processed.graphml" --output outputs\\real_candidate_pairs.json --geojson-output outputs\\real_candidate_pairs.geojson
python -m stgat_lstm.build_decision_dataset "..\\ExportLMD\\clean_data" "..\\OSM\\data\\osm\\graphs\\metro_manila_processed.graphml" --deviation-candidates outputs\\real_candidate_pairs.json --output outputs\\real_decision_candidates.json --geojson-output outputs\\real_decision_candidates.geojson
python -m stgat_lstm.review_map outputs\\real_decision_candidates.geojson --output outputs\\real_decision_review.html
python -m stgat_lstm.review_decisions init outputs\\real_decision_candidates.json --output outputs\\real_decision_decisions.json
python -m stgat_lstm.review_map outputs\\real_candidate_pairs.geojson --output outputs\\real_candidate_review.html
python -m stgat_lstm.real_data "..\\OSM\\data\\osm\\graphs\\metro_manila_processed.graphml" "..\\OSM\\data\\features\\road_node_features.csv" outputs\\real_candidate_pairs.json
python -m stgat_lstm.review_decisions init outputs\\real_candidate_pairs.json --output outputs\\real_candidate_decisions.json
REM After reviewing the map, edit every pending decision in real_candidate_decisions.json.
python -m stgat_lstm.review_decisions finalize outputs\\real_candidate_pairs.json outputs\\real_candidate_decisions.json --output outputs\\real_approved_pairs.json
REM After reviewing all 8 local decisions, edit every pending decision in real_decision_decisions.json.
python -m stgat_lstm.review_decisions finalize outputs\\real_decision_candidates.json outputs\\real_decision_decisions.json --output outputs\\real_approved_decisions.json
python -m stgat_lstm.decision_training synthetic --epochs 200 --output outputs\\multitask_synthetic.pt
python -m stgat_lstm.decision_training real "..\\OSM\\data\\osm\\graphs\\metro_manila_processed.graphml" "..\\OSM\\data\\features\\road_node_features.csv" outputs\\real_approved_decisions.json --epochs 40 --output outputs\\multitask_real.pt
set MAPBOX_ACCESS_TOKEN=pk.your_token_here
python -m stgat_lstm.mapbox_traffic "120.9820,14.5560;120.9830,14.5570" --output outputs\\traffic\\taft_001.json
```

The traffic command uses Mapbox Directions `driving-traffic` and requests GeoJSON geometry plus congestion, speed, and duration annotations. Set `MAPBOX_ACCESS_TOKEN` only in the cmd session or a local secret manager; never commit it. Each output is timestamped and can be converted with `build_edge_snapshot(...)` in `stgat_lstm.mapbox_traffic` before calling `RealGraphData.build_model_input(..., temporal_edge_features=(snapshot,))`. A request covers only its returned route; unobserved Taft edges remain masked as unknown.

The real dynamic feature vector has five fields: normalized Mapbox congestion, Mapbox speed relative to the OSM reference speed, a congestion-observed mask, a speed-observed mask, and observation age. Speed is still usable when Mapbox returns `unknown`/`null` congestion. Missing congestion is never interpreted as free flow. Multiple Mapbox geometry segments matched to the same OSM edge are averaged. Repeated timestamped requests can form the sequence consumed by the LSTM; a single routing request can use the latest snapshot.

If the `py -3.12` launcher is unavailable, use `python -m venv .venv` instead. A new cmd window must be activated again with `call .venv\\Scripts\\activate.bat`.

On the reviewed machine, the existing `C:\Users\Vince\Downloads\THESIS_APP\finalGNNT3\venv\Scripts\python.exe` already has both model libraries and can run these commands in place of `python`. The default system Python does not currently have PyTorch installed.

The audit uses only the Python standard library, so it can run with a separate Python installation even before ML dependencies are installed. The demo saves three clearly named synthetic checkpoints and `synthetic_report.json`. The prediction command requires a checkpoint from the exact fixture version. Its ranking count is **in sample**. There is no meaningful test set in nine constructed cases.

## Verified behavior and limits

The automated suite exercises prior-route reconstruction, exclusion of wrong turns and personal stops, missing and late traffic, incomplete GPS observations, local follow-choice construction, all three model forward/backward paths, motorcycle access, pre-choice inference isolation and an end-to-end learned route choice. With 60 training epochs, the STGAT-LSTM fit the four designed preference pairs and selected the slower side route in one case and the main route in a different context. This establishes that the code and model-to-router interface work on those examples only.

The real export audit currently finds 55 rides, 167 saved routes, 34 reported deviations and 11,350 GPS points. All 34 deviation records reference a route generated after their event timestamp. Each has one candidate prior route by timestamp. The geometry screen reports how many route snapshots parse successfully, whether GPS exists immediately around the event, how far the event lies from each route, and whether the old and regenerated routes appear to share a destination. These checks produce pre-map-match candidates only. The audit therefore reports **zero real preference examples ready for training** until GPS and route versions are matched to the same directed OSM graph and the relevant path alternatives are reconstructed.

The OSM compatibility command can take several seconds because it loads the 74 MB Metro Manila graph and spatially checks every saved route and GPS point. Its independent nearest-edge results are quality diagnostics. They are not sequence-aware map matching and must not be used directly as rider-choice targets.

The map-match audit applies the HMM only to intentional deviation events inside the provisional Taft corridor. It compares the post-event GPS edge sequence with the prior and regenerated route sequences and reports candidates where the routes differ and GPS agrees more with the regenerated route. These still require divergence/rejoin review before they can become common-origin/common-destination preference pairs.

On the current export, the compatibility audit found 17 intentional deviation events inside the provisional corridor. The HMM sequence screen matched GPS, prior route and regenerated route for 11 events; 8 of those agree more strongly with the regenerated route and are retained as review candidates. This is a data-preparation result, not model accuracy.

The stricter candidate builder additionally requires shared directed edges before and after the event and emits equal-origin/equal-destination subpaths. Its JSON contains pseudonymous grouping keys and no coordinates. The optional GeoJSON contains only the selected OSM road geometry for visual review; it does not contain raw rider GPS points.

The decision-dataset builder uses one local route choice as the common prediction unit. A followed example must traverse the suggested path through an OSM node with at least two outgoing road choices. A deviated example must have an intentional survey response plus a GPS-supported divergence and rejoin. On the current export it produces 8 review-required candidates from 6 riders: 5 followed and 3 deviated. This class balance is suitable for pipeline development only, not a thesis performance claim or a stable train/test split.

The review-map command creates a local HTML page with a candidate selector, path details, and separate styling for the observed and rejected paths. It uses OpenStreetMap tiles and Leaflet from their public servers when the page is opened, so the basemap requires an internet connection. The embedded candidate layer contains OSM path geometry, not raw GPS.

On the current export, this strict stage retains 3 candidates from 3 rides and 3 riders. One is a shorter familiar/shortcut path; two are longer paths associated with a reported road blockage and intersection avoidance. Path length alone does not approve or reject them. All three remain marked `review_required` until their mapped geometry and event context are reviewed.

The current shortest-path implementation assumes additive edge costs. Turn-dependent restrictions and costs require an augmented search state. The toy graph uses invented road access and traffic values; it makes no claim about Taft Avenue. `PLAN.md`, `REVIEW_NOTES.md` and `APP_DATA_REVIEW.md` record the project scope and remaining data decisions.

The model implementation draws on the [original GAT paper](https://arxiv.org/abs/1710.10903), [GATv2 paper](https://arxiv.org/abs/2105.14491) and [PyTorch Geometric operators](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html). These are component references, not claims that the current architecture reproduces a full published traffic model.
