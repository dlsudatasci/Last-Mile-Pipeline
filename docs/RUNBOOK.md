# STGAT-LSTM runbook for Windows cmd

This is the start-to-finish procedure for a new project run. Run every command
from the repository root. The saved JSON files are the detailed records; the
terminal prints a shorter summary.

## 1. Create and activate the environment

```bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In a later cmd window, only repeat:

```bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
call .venv\Scripts\activate.bat
```

## 2. Verify the code

```bat
python -m unittest discover -s tests -v
```

These tests use small in-memory graphs. They verify the model, data boundaries,
map matching, traffic history, evaluation, routing, dashboard, and synthetic
experiment without changing the official dataset.

## 3. Audit the official rider and OSM data

```bat
python -m stgat_lstm audit-data "data\real\rider_exports"
python -m stgat_lstm audit-network "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv"
python -m stgat_lstm audit-matching "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml"
python -m stgat_lstm gps-traffic "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --output outputs\gps_traffic_audit.json
```

The first command checks CSV relationships, timing, surveys, and GPS coverage.
The second checks the projected directed OSM graph and node-feature join. The
third diagnoses sequence map matching. The fourth converts consecutive GPS
fixes into motorcycle probe-speed observations on directed OSM roads.

Default audit reports are saved as:

- `outputs\rider_data_audit.json`
- `outputs\road_network_audit.json`
- `outputs\map_matching_audit.json`
- `outputs\gps_traffic_audit.json`

The current GPS audit finds 6,657 usable speed segments on 494 directed roads
from 48 rides. Median sampling is 2.001 seconds. These speeds are historical
motorcycle observations; they are a limited traffic proxy, not network-wide
ground truth.

## 4. Rebuild training examples when rider exports change

```bat
python -m stgat_lstm build-deviations "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --output outputs\deviation_candidates.json --geojson-output outputs\deviation_candidates.geojson
python -m stgat_lstm build-decisions "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --deviation-candidates outputs\deviation_candidates.json --output outputs\decision_candidates.json --geojson-output outputs\decision_candidates.geojson
python -m stgat_lstm review-map outputs\decision_candidates.geojson --output outputs\candidate_review_map.html
python -m stgat_lstm review auto outputs\decision_candidates.json --output outputs\approved_training_examples.json
```

`build-deviations` keeps intentional events with a GPS-supported divergence and
rejoin. Wrong turns, GPS errors, and personal stops are not positive tacit-
knowledge labels. `build-decisions` adds conservative followed choices. `review
auto` applies the documented survey and geometry rules. Manual approval is no
longer required for normal data; the map remains available for auditing.

The model accepts only `approved_training_examples.json`. GPS after the choice
and the survey answer establish the label; they are not inference inputs.

If the source export changes, repeat Sections 3 and 4. The candidate count may
change. Do not edit the approved artifact to force a desired class balance.

## 5. Train the official ST-GAT-LSTM

```bat
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --gps-traffic-data "data\real\rider_exports" --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 100 --batch-size 8 --output outputs\gatv2_lstm_checkpoint.pt --report-output outputs\training_report.json
```

Training loads the OSM graph and approved choices, derives only historical GPS
speed profiles that predate each decision, excludes the target ride, performs
GATv2 followed by LSTM, and optimizes two objectives:

1. weighted binary cross-entropy for each eligible suggested road; and
2. pairwise loss that lowers the observed intentional route cost relative to
   its rejected suggestion.

The checkpoint stores learned parameters, architecture settings, training
counts, hashes, traffic policy, and provenance. `training_report.json` stores
the same readable report plus epoch losses. Training uses shuffled mini-batches,
and every batch performs an optimizer update. Graphs inside a batch are joined
as disconnected components and cannot exchange messages.

## 6. Evaluate held-out riders

```bat
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --gps-traffic-data "data\real\rider_exports" --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 40 --temporal-ablation --output outputs\rider_holdout_evaluation.json
```

Each fold trains without one rider and tests on that rider. The report compares
prevalence, GCN, GAT, spatial-only GATv2, and ST-GAT-LSTM. When observed history
actually varies, `--temporal-ablation` also runs an LSTM with only the latest
frame repeated. GPS histories are rebuilt in each fold and exclude the held-out
rider, target ride, and future records.

Read balanced accuracy, deviation F1, average precision, ROC-AUC, Brier score,
log loss, confusion counts, and held-out route-ranking accuracy. With only eight
approved real examples and three positive road rejections, current metrics are
pilot evidence and cannot establish generalization or superiority.

## 7. Predict a reviewed example

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --gps-traffic-data "data\real\rider_exports" --output outputs\reviewed_example_prediction.json
```

This replays an approved decision, reports per-road deviation probabilities,
compares observed and suggested learned costs, and recommends a connected route.
For replay, historical GPS profiles exclude the example's rider and ride.

## 8. Generate a new route

```bat
python -m stgat_lstm predict outputs\gatv2_lstm_checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 400 --destination-node 6 --gps-traffic-data "data\real\rider_exports" --max-detour-ratio 1.30 --output outputs\new_route_prediction.json
```

The origin and destination are OSM node IDs. At a new request time, the code
builds a historical time-of-day GPS speed profile, scores every accessible
directed road, and runs Dijkstra on the positive learned costs. It also reports
the shortest-distance baseline. A hard guard accepts the learned route only when
its distance is at most 1.30 times the shortest-distance route; otherwise it
returns the distance baseline and records the fallback in JSON. The route uses
historical rider-derived traffic context; it does not call Mapbox.

## 9. Open the routing dashboard

```bat
python -m stgat_lstm dashboard --gps-traffic-data "data\real\rider_exports" --max-detour-ratio 1.30
```

Open **http://127.0.0.1:8765**, click a start and destination, and select
**Generate route**. Green is the learned route and blue is the distance
baseline. The view uses embedded local OSM geometry and does not need public map
tiles. Stop it with Ctrl+C.

## 10. Controlled synthetic hotspot experiment

This experiment asks a narrow question: can the implemented model learn a
repeated road preference when a known pattern is injected? It never replaces
the real evaluation.

```bat
python -m stgat_lstm synthetic-hotspot "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\approved_training_examples.json --output-dir outputs\synthetic_hotspot
python -m stgat_lstm build-deviations outputs\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" --output outputs\synthetic_hotspot\reconstructed_deviation_candidates.json --geojson-output outputs\synthetic_hotspot\reconstructed_deviation_candidates.geojson
python -m stgat_lstm build-decisions outputs\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" --deviation-candidates outputs\synthetic_hotspot\reconstructed_deviation_candidates.json --output outputs\synthetic_hotspot\reconstructed_decision_candidates.json --geojson-output outputs\synthetic_hotspot\reconstructed_decision_candidates.geojson
python -m stgat_lstm review auto outputs\synthetic_hotspot\reconstructed_decision_candidates.json --output outputs\synthetic_hotspot\reconstructed_approved_training_examples.json
python -m stgat_lstm gps-traffic outputs\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --output outputs\synthetic_hotspot\gps_traffic_audit.json
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\synthetic_hotspot\reconstructed_approved_training_examples.json --gps-traffic-data outputs\synthetic_hotspot --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 20 --batch-size 8 --output outputs\synthetic_hotspot\checkpoint.pt --report-output outputs\synthetic_hotspot\training_report.json
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" outputs\synthetic_hotspot\reconstructed_approved_training_examples.json --gps-traffic-data outputs\synthetic_hotspot --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 20 --batch-size 16 --synthetic-fixed-split --output outputs\synthetic_hotspot\rider_holdout_evaluation.json
```

The generator writes the same six CSV schemas as the app export, a manifest,
and a trainable ground-truth artifact. It creates 96 labeled choices from eight
synthetic riders: 48 intentional deviations across two approved in-corridor
Taft choice patterns and 48 follows across five approved follow-route
structures. Forty-two separate traffic-only motorcycle probe rides create slow
morning history on deviation roads and free-flow late-morning history on follow
roads. The probes intentionally have no initial navigation route, so they feed
GPS traffic without becoming decision labels.

All 96 decision rides survive the production map-matching, candidate, and
automatic-approval pipeline, with no rejected decisions. The fixed
rider-disjoint split contains 48 training, 24 validation, and 24 test examples.
Thresholds are selected only from validation balanced accuracy and frozen before
testing. On unseen synthetic test riders, GATv2 and full-history ST-GAT-LSTM
reach 1.000 balanced accuracy, deviation F1, AP, ROC-AUC, and route-ranking
accuracy. The latest-only LSTM reaches 0.983 balanced accuracy, 0.923 F1, 0.929
AP, 0.983 ROC-AUC, and 1.000 ranking accuracy. Full-history ST-GAT-LSTM has a
0.424 positive/negative probability gap, compared with 0.002 in the earlier
one-hotspot experiment. Report this as controlled implementation and temporal
capacity evidence, never as real-rider evidence.

## Optional Mapbox compatibility

`mapbox_traffic.py`, `--traffic-archive`, and `--live-traffic` remain available
for prior experiments. They are no longer the main thesis traffic design. Do
not combine Mapbox and GPS traffic flags in one command, and do not attach a
current Mapbox observation to a past rider decision.

## What changes when new official data arrive

Replace the CSV files under `data\real\rider_exports` with the new consistent
export, retaining the same filenames and columns. Then repeat Sections 3 through
6. No model source code or dimensions need to change if the export schema and
OSM graph stay the same. New labels, GPS histories, checkpoints, and evaluation
reports are regenerated from the new inputs.
