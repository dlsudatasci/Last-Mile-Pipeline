# Controlled synthetic STGAT-LSTM runbook for Windows cmd

Input update, 20 October 2026: following this guide from generation through
training uses the active cleaned amenity CSV. Existing saved synthetic models
and reported metrics used the old features. To replay those models without
regenerating/retraining, use the CSV under
`artifacts/inputs/osm_before_amenity_cleanup`. This guide remains a separate
controlled experiment.

This is the complete procedure for regenerating, auditing, evaluating, training,
and demonstrating the controlled synthetic Taft experiment. Run every command
from the repository root. Generated rider CSVs and labels stay under
`generated\synthetic_hotspot`. Checkpoints go under
`artifacts\checkpoints\synthetic_hotspot`; audits, reports, predictions, and
maps go into their respective `outputs` subfolders, each with a
`synthetic_hotspot` subfolder. They must not be combined with official rider
metrics. The 19 October reorganization changes paths, not the generator or model.

The synthetic and real workflows use the same OSM graph, feature contract,
map-matching code, GATv2-LSTM architecture, losses, and routing algorithm. They
are not different only by file path. The synthetic workflow first generates
app-shaped records, uses a fixed train/validation/test rider split, keeps traffic
probe rides separate from decision rides, tunes only on validation riders, and
uses fixed timestamps to replay its controlled traffic histories.

## 1. Activate the existing environment

```bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
call .venv\Scripts\activate.bat
```

If `.venv` does not exist, create it once:

```bat
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Verify the code

```bat
python -m unittest discover -s tests -v
```

These tests use small in-memory fixtures and do not change the synthetic or real
datasets.

## 3. Confirm the required source files

The experiment uses the real OSM network and node features:

```text
data\real\osm\metro_manila_processed.graphml
data\real\osm\road_node_features.csv
```

It also uses this existing reviewed artifact as a source of two graph-valid Taft
path templates:

```text
generated\real\approved_training_examples.json
```

That template supplies valid OSM path geometry. Its rider labels are not merged
into the synthetic evaluation. The generator creates new artificial riders,
timestamps, GPS tracks, follow choices, deviations, and traffic probes. If the
template is missing, complete Sections 3 and 4 of `RUNBOOK.md` once.

## 4. Generate the synthetic experiment

```bat
python -m stgat_lstm synthetic-hotspot "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" generated\real\approved_training_examples.json --output-dir generated\synthetic_hotspot
```

This regenerates the app-shaped CSV files, exact ground-truth training artifact,
traffic-probe directory, and manifest. Expected summary:

```text
Training examples: 144 from 12 riders
Deviated: 72
Followed: 72
Traffic-probe rides: 144
```

The six traffic speeds for an injected deviation are
`12, 10, 8, 6, 4, 8 m/s`. The follow control uses
`4, 6, 8, 10, 12, 8 m/s`. Both end at 8 m/s so the latest frame alone cannot
identify the injected rule.

## 5. Audit the generated CSV and GPS data

```bat
python -m stgat_lstm audit-data generated\synthetic_hotspot --output outputs\audits\synthetic_hotspot\rider_data_audit.json
python -m stgat_lstm audit-network generated\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv" --output outputs\audits\synthetic_hotspot\road_network_audit.json
python -m stgat_lstm audit-matching generated\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" --output outputs\audits\synthetic_hotspot\map_matching_audit.json
python -m stgat_lstm gps-traffic generated\synthetic_hotspot\traffic_probes "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --output outputs\audits\synthetic_hotspot\gps_traffic_audit.json
```

The root synthetic export contains 288 rides: 144 labeled decision rides and
144 traffic-only probe rides. Expected checks include no GPS gaps over 60
seconds, 288/288 questionnaires, 72 reported deviations, successful sequence
matching for all 72 eligible deviations, and a 2.0-second median interval in
the isolated traffic probes. `gps-traffic` reads only `traffic_probes`;
decision-trip GPS is deliberately excluded from traffic input to prevent the
answer from leaking into the model.

## 6. Reconstruct and automatically validate the decisions

```bat
python -m stgat_lstm build-deviations generated\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" --output generated\synthetic_hotspot\reconstructed_deviation_candidates.json --geojson-output generated\synthetic_hotspot\reconstructed_deviation_candidates.geojson
python -m stgat_lstm build-decisions generated\synthetic_hotspot "data\real\osm\metro_manila_processed.graphml" --deviation-candidates generated\synthetic_hotspot\reconstructed_deviation_candidates.json --output generated\synthetic_hotspot\reconstructed_decision_candidates.json --geojson-output generated\synthetic_hotspot\reconstructed_decision_candidates.geojson
python -m stgat_lstm review auto generated\synthetic_hotspot\reconstructed_decision_candidates.json --output generated\synthetic_hotspot\reconstructed_approved_training_examples.json
python -m stgat_lstm review-map generated\synthetic_hotspot\reconstructed_decision_candidates.geojson --output outputs\maps\synthetic_hotspot\candidate_review_map.html
```

Expected result: 144 decision candidates from 12 riders and 144 automatic
approvals. This stage checks whether the production reconstruction pipeline can
recover the known synthetic choices. Model evaluation uses the generator's exact
ground-truth artifact, `generated\synthetic_hotspot\approved_training_examples.json`,
so followed and deviated pairs retain their designed origin and destination.

## 7. Run the fixed rider-disjoint evaluation

Run evaluation before final training because validation selects the
alternative-road loss weight that the final checkpoint will reuse.

```bat
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" generated\synthetic_hotspot\approved_training_examples.json --gps-traffic-data generated\synthetic_hotspot\traffic_probes --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 10 --batch-size 8 --preference-weight 0.1 --synthetic-fixed-split --synthetic-tune --output outputs\reports\synthetic_hotspot\rider_holdout_evaluation.json
```

The fixed split uses:

- 96 examples from eight training riders;
- 24 examples from two validation riders; and
- 24 examples from two untouched test riders.

The evaluator compares a prevalence baseline, spatial-only GATv2, full-history
ST-GAT-LSTM, and an LSTM latest-only control. Read outgoing-road accuracy and
negative log-likelihood first, followed by balanced accuracy, deviation F1,
ROC-AUC, average precision, and route-ranking accuracy.

The currently verified full-history test results are 0.846 outgoing-choice
accuracy, 0.955 balanced accuracy, 0.800 deviation F1, 0.955 ROC-AUC, 0.833
average precision, and 1.000 route-ranking accuracy. Treat new terminal output
and the saved JSON as authoritative if code or data change.

## 8. Train the final synthetic demonstration checkpoint

```bat
python -m stgat_lstm train real "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" generated\synthetic_hotspot\approved_training_examples.json --gps-traffic-data generated\synthetic_hotspot\traffic_probes --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --batch-size 8 --tuning-report outputs\reports\synthetic_hotspot\rider_holdout_evaluation.json --output artifacts\checkpoints\synthetic_hotspot\checkpoint.pt --report-output outputs\reports\synthetic_hotspot\training_report.json
```

The word `real` is the training command's subcommand name; it does not convert
the synthetic artifact into real rider evidence. The command reads the
validation-selected settings from the evaluation report and trains a final
demonstration checkpoint on all 144 synthetic examples. Its in-sample training
figures are not held-out test results.

## 9. Replay one labeled synthetic example

```bat
python -m stgat_lstm predict artifacts\checkpoints\synthetic_hotspot\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" generated\synthetic_hotspot\approved_training_examples.json --gps-traffic-data generated\synthetic_hotspot\traffic_probes --output outputs\predictions\synthetic_hotspot\reviewed_example_prediction.json
```

This reloads the final checkpoint, reconstructs the selected example's
pre-decision traffic history, reports outgoing-road probabilities, compares the
observed and suggested path costs, and generates a connected recommendation.
It is a checkpoint demonstration, not held-out accuracy; Section 7 contains the
held-out evidence.

## 10. Generate fresh routes for both synthetic traffic regimes

Worsening-history request:

```bat
python -m stgat_lstm predict artifacts\checkpoints\synthetic_hotspot\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 1034 --destination-node 35633 --gps-traffic-data generated\synthetic_hotspot\traffic_probes --traffic-cutoff-ms 1767571217000 --max-detour-ratio 1.30 --output outputs\predictions\synthetic_hotspot\worsening_route_prediction.json
```

Recovering-history request:

```bat
python -m stgat_lstm predict artifacts\checkpoints\synthetic_hotspot\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 1034 --destination-node 35633 --gps-traffic-data generated\synthetic_hotspot\traffic_probes --traffic-cutoff-ms 1771203600000 --max-detour-ratio 1.30 --output outputs\predictions\synthetic_hotspot\recovering_route_prediction.json
```

Both commands use the same model, OSM graph, origin, and destination. The fixed
cutoff changes only the historical traffic context. The currently verified
worsening route uses Decena Street, A. Arnaiz Avenue, and Taft Avenue; the
recovering route uses Decena Street, Protacio Street, and Taft Avenue.

## 11. Open the synthetic routing dashboard

Worsening traffic:

```bat
python -m stgat_lstm dashboard --checkpoint artifacts\checkpoints\synthetic_hotspot\checkpoint.pt --graphml "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv" --gps-traffic-data generated\synthetic_hotspot\traffic_probes --traffic-cutoff-ms 1767571217000 --max-detour-ratio 1.30
```

Open **http://127.0.0.1:8765**, select a start and destination around hotspot
nodes 1034 and 35633, and generate the route. Stop the server with Ctrl+C.

To demonstrate recovering traffic, run the same command again with:

```text
--traffic-cutoff-ms 1771203600000
```

The direct JSON endpoint for the exact hotspot is:

```text
http://127.0.0.1:8765/route?origin=1034&destination=35633
```

## 12. Main files to inspect after the run

```text
generated\synthetic_hotspot\manifest.json
outputs\audits\synthetic_hotspot\gps_traffic_audit.json
generated\synthetic_hotspot\reconstructed_approved_training_examples.json
outputs\reports\synthetic_hotspot\rider_holdout_evaluation.json
outputs\reports\synthetic_hotspot\training_report.json
artifacts\checkpoints\synthetic_hotspot\checkpoint.pt
outputs\predictions\synthetic_hotspot\reviewed_example_prediction.json
outputs\predictions\synthetic_hotspot\worsening_route_prediction.json
outputs\predictions\synthetic_hotspot\recovering_route_prediction.json
outputs\maps\synthetic_hotspot\candidate_review_map.html
```

The held-out evaluation report is the evidence that the controlled model learned
the injected rule. The final checkpoint and dashboard demonstrate how that rule
is converted into route costs and connected routes. Neither artifact establishes
performance on real motorcycle riders.
