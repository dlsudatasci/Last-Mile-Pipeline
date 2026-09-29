# STGAT-LSTM command runbook (Windows cmd)

> The commands below use the current unified interface. Run `python -m stgat_lstm --help` to list every command.

## Start the local routing prototype

If your environment and `outputs\gatv2_lstm_checkpoint.pt` already exist, run from this project folder:

```bat
call .venv\Scripts\activate.bat
python -m stgat_lstm dashboard
```

Open **http://127.0.0.1:8765**. Click a start and destination on the supplied OSM road network, then **Generate route**. Green uses learned edge costs; blue shows the shortest-distance comparison. Scroll to zoom and drag to pan. Stop the server with Ctrl+C. No Mapbox token or internet tiles are needed for this view.

To request current Mapbox traffic whenever **Generate route** is pressed, set the token and start the dashboard with live traffic:

```bat
set MAPBOX_ACCESS_TOKEN=pk.your_new_token_here
python -m stgat_lstm dashboard --live-traffic --traffic-archive outputs\traffic_archive
```

The token stays in the terminal environment and is not written to an output file. Each request is saved in the archive, so later requests can use recent observations as an LSTM history.

The default checkpoint is trained only from the approved official rider-export dataset.

Run commands from:

```bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
```

## 1. One-time environment setup

```bat
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In every new `cmd` window:

```bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
call .venv\Scripts\activate.bat
```

## 2. Verify the code

```bat
python -m unittest discover -s tests -v
```

## 3. Audit the app export and OSM graph

```bat
python -m stgat_lstm audit-data "data\real\rider_exports"
python -m stgat_lstm audit-network "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv"
python -m stgat_lstm audit-matching "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml"
```

These commands diagnose data quality. They do not train a model.

They save `outputs\rider_data_audit.json`, `outputs\road_network_audit.json`, and `outputs\map_matching_audit.json`. The terminal prints the main counts and the saved file location. Use `--output` to choose a different JSON path.

## 4. Reconstruct strict deviation pairs

```bat
python -m stgat_lstm build-deviations "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --output outputs\deviation_candidates.json --geojson-output outputs\deviation_candidates.geojson
```

## 5. Build the unified local decision dataset

```bat
python -m stgat_lstm build-decisions "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --deviation-candidates outputs\deviation_candidates.json --output outputs\decision_candidates.json --geojson-output outputs\decision_candidates.geojson
python -m stgat_lstm review-map outputs\decision_candidates.geojson --output outputs\candidate_review_map.html
python -m stgat_lstm review auto outputs\decision_candidates.json --output outputs\approved_training_examples.json
```

Open the review map:

```bat
start "" "outputs\candidate_review_map.html"
```

The `review auto` command is the required validation gate. It approves followed examples only when there is no conflicting deviation report and GPS follows the suggested branch. It approves deviations only for supported intentional reasons—traffic, blockage/hazard, intersection avoidance, or shortcut/familiar-road preference—with connected common-endpoint paths and GPS support. A traffic deviation must include severity. Personal stops, unsupported/unknown reasons, disconnected paths, weak GPS agreement, and likely GPS errors are rejected. Rejected rows remain in the JSON audit trail but are excluded by the training loader.

The HTML map is optional quality control. It is useful when explaining examples but is no longer a manual approval requirement.

The approved artifact records the exact candidate SHA-256, automatic-rule version, per-example reason, and approval/rejection counts. Rebuilding from changed data therefore produces a new traceable artifact.

### Updating the dataset later

After rebuilding candidates from newer exports, rerun `review auto`. The same versioned rules are applied to every candidate, so no decision file needs to be edited by hand. Preserve dated copies of final artifacts if you need to compare collection rounds.

### What changes when final rider data arrives

If the CSV names and columns remain the same, no Python code or model architecture needs to change. Preserve the old exports and outputs as a dated snapshot, place the updated cleaned CSVs in the input folder, then rerun sections 3 through 8. The candidate count, rider count, labels, checkpoint, and evaluation report will update from the new data.

Also supply `--traffic-archive` during training and evaluation when timestamp-aligned observations exist. Current traffic cannot be attached to older rides. If the app export schema changes, update and test the corresponding loader before rebuilding. If the OSM graph or node-feature file changes, rebuild all candidates and retrain because node/edge identities and checkpoint provenance may change.

Freeze the final metric protocol, threshold, model settings, and rider split before inspecting final test results. Development data may be used for tuning; final held-out riders should be evaluated once.

## 6. Train the approved real-data model

```bat
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --epochs 100 --output outputs\gatv2_lstm_checkpoint.pt
```

Outputs:

- `outputs\gatv2_lstm_checkpoint.pt`: model weights, configuration, provenance, and training summary
- `outputs\training_report.json`: readable report

New training reports include `epoch_losses` (one training loss per epoch), final loss, in-sample classification/ranking, training settings, and input-file hashes. Epoch losses are measured during optimization; `final_loss` is measured after the last update. Training accuracy must not be presented as held-out accuracy.

The terminal prints the input counts, loss change, in-sample checks, and both output paths. The JSON report contains the complete epoch history and provenance.

## 7. Run the preliminary rider-disjoint comparison

```bat
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --epochs 40 --output outputs\rider_holdout_evaluation.json
```

This performs six leave-one-rider-out folds for GCN, original GAT, spatial-only GATv2, and GATv2-LSTM, plus a training-prevalence baseline. With only eight examples, treat the report as a pipeline diagnostic and not a final performance estimate.

The terminal now shows a compact comparison table. The JSON saves full predictions and metrics. Proposed primary metric: **balanced accuracy**, the average of follow recall and deviation recall. Secondary measures: deviation F1, macro-F1, average precision (AP), ROC-AUC, Brier score, and log loss. AP summarizes the precision/recall ranking; it is not trapezoidal PR-AUC. Brier/log loss evaluate probability quality; lower is better. Threshold is fixed at 0.5, and class support/confusion counts are retained. Undefined precision/F1 uses zero; two-class metrics are null when a class is absent. Results pool held-out decisions, so riders with more decisions contribute more. These choices are documented in `metric_protocol`.

Held-out preference-ranking accuracy is the fraction of reviewed deviation pairs where the observed path receives a lower learned cost than the rejected suggestion. This is a routing-related proxy, not evidence that a new route will be followed or improve travel time.

### Test whether traffic history helps the LSTM

With the current pilot:

```bat
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --epochs 40 --temporal-ablation --output outputs\rider_holdout_evaluation.json
```

With future timestamp-aligned traffic, add `--traffic-archive outputs\traffic_archive` to that command. It compares:

1. Spatial GATv2 using the latest snapshot.
2. GATv2-LSTM using the available history.
3. The same GATv2-LSTM receiving its latest snapshot repeated to the same history length. This control keeps its parameter count, folds, sequence length, training settings, and initialization seed matched to the full-history model, while removing past information.

The third model is skipped when no same-edge traffic field has two observed, different values across the history. Age changes and transitions from unknown to observed do not count as evidence of a traffic trend. Inspect `lstm_comparison.temporal_evidence` for counts and suggested-path coverage. If the control runs, `full_minus_control` records paired metric differences: positive accuracy/F1/AP/AUC differences favor history; negative Brier/log-loss differences favor history. One positive pilot difference does not establish superiority.

For the final experiment, lock hyperparameters and the threshold using separate training/validation riders, then evaluate held-out riders. Repeat with prespecified seeds (for example 17, 29, 43, 61, 89), using `--seed` and separate output filenames. Report variation across seeds and uncertainty across independent riders; multiple seeds do not create additional rider observations. The current command provides one seeded run and does not automatically compute confidence intervals or tune hyperparameters. If real history does not improve results, report that outcome rather than assuming LSTM must help.

## 8. Reload the checkpoint, predict, and route

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json
```

Select a specific approved example:

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --example-key 11930c364eaa39de
```

## 9. Collect a current Mapbox observation

Set the token in the current terminal session:

```bat
set MAPBOX_ACCESS_TOKEN=pk.your_new_token_here
```

Collect traffic using longitude,latitude order:

```bat
python -m stgat_lstm traffic "120.9790,14.5800;120.9980,14.5400" --output outputs\traffic\current_traffic.json
```

For a current inference demonstration, supply the traffic observation:

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --traffic-observation outputs\traffic\current_traffic.json
```

For a new route, fetch current traffic automatically before model inference:

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 400 --destination-node 6 --live-traffic --traffic-archive outputs\traffic_archive
```

This calls Mapbox for the requested endpoints, saves the token-free observation, combines it with recent archived observations, and feeds the resulting traffic sequence to the model. With an empty archive, only the latest slot contains observed traffic; repeated route requests or scheduled collection are still needed to demonstrate an LSTM temporal benefit.

Do not attach current traffic to an old decision and describe it as historical evidence. The option above demonstrates the online feature interface only.

## 10. Presentation-safe quick run

The interactive dashboard command at the top also works with your existing checkpoint. The commands below replay an approved example instead of selecting new endpoints.

```bat
call .venv\Scripts\activate.bat
python -m unittest discover -s tests -v
type outputs\training_report.json
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json
start "" "outputs\candidate_review_map.html"
```

## 11. Collect traffic histories for future decisions

The collector runs on your computer, independently of the collection app. After setting `MAPBOX_ACCESS_TOKEN` as above:

```bat
python -m stgat_lstm traffic "120.9790,14.5800;120.9980,14.5400" --archive-dir outputs\traffic_archive --samples 12 --interval-seconds 300
```

This saves twelve separate request/receipt-timestamped observations, roughly five minutes apart, making twelve API requests. You choose when to run it alongside future data collection. Each request covers the returned route, not every road in the study area. Additional probe routes are needed for additional coverage. Provider driving-traffic estimates are road context, not motorcycle-specific measurements.

Use the archive during training and evaluation:

```bat
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --traffic-archive outputs\traffic_archive --epochs 100 --output outputs\gatv2_lstm_checkpoint.pt
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --traffic-archive outputs\traffic_archive --epochs 40 --output outputs\rider_holdout_evaluation.json
```

Default history: six slots spaced five minutes apart, ending at each decision time. An observation must have been received before its slot and be no more than fifteen minutes old. There is no future interpolation. Uncovered roads and null congestion retain missingness masks. Direction-aware matching avoids applying opposing-direction traffic to an edge. Inspect `provenance.traffic_history.*.observed_edges_per_step` to verify actual coverage; six unknown slots do not constitute temporal evidence. Old files lacking receipt timestamps use their request times explicitly, recorded as `legacy_request_only_timestamps`.

For current routing without a new API request, run `python -m stgat_lstm dashboard --traffic-archive outputs\traffic_archive`. To call Mapbox for every route request, add `--live-traffic`. Both modes reload the archive for each request and apply the checkpoint's saved history policy, or defaults for older checkpoints. Use a model trained with meaningful historical traffic before claiming useful live traffic adaptation.

## 12. Route new endpoints without a reviewed example

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 400 --destination-node 6
```

The complete result is saved by default to `outputs\route_prediction.json`; the terminal prints the route length, roads, comparison distance, and deviation estimate. Add `--output another\path.json` to keep multiple runs.

Alternatively use `--origin "longitude,latitude" --destination "longitude,latitude"`. Coordinates snap to a graph node within 150 metres. Directed edges and encoded access restrictions are enforced; turn restrictions are not represented. The output reports an uncalibrated deviation probability at each branching road on the baseline and recommended routes. Learned costs are preference scores, not travel times.
