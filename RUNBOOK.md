# STGAT-LSTM command runbook (Windows cmd)

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
python -m stgat_lstm.audit "..\ExportLMD\clean_data"
python -m stgat_lstm.osm_audit "..\ExportLMD\clean_data" "..\OSM\data\osm\graphs\metro_manila_processed.graphml" --node-features "..\OSM\data\features\road_node_features.csv"
python -m stgat_lstm.map_match_audit "..\ExportLMD\clean_data" "..\OSM\data\osm\graphs\metro_manila_processed.graphml"
```

These commands diagnose data quality. They do not train a model.

## 4. Reconstruct strict deviation pairs

```bat
python -m stgat_lstm.build_real_candidates "..\ExportLMD\clean_data" "..\OSM\data\osm\graphs\metro_manila_processed.graphml" --output outputs\real_candidate_pairs.json --geojson-output outputs\real_candidate_pairs.geojson
```

## 5. Build the unified local decision dataset

```bat
python -m stgat_lstm.build_decision_dataset "..\ExportLMD\clean_data" "..\OSM\data\osm\graphs\metro_manila_processed.graphml" --deviation-candidates outputs\real_candidate_pairs.json --output outputs\real_decision_candidates.json --geojson-output outputs\real_decision_candidates.geojson
python -m stgat_lstm.review_map outputs\real_decision_candidates.geojson --output outputs\real_decision_review.html
python -m stgat_lstm.review_decisions init outputs\real_decision_candidates.json --output outputs\real_decision_decisions.json
```

Open the review map:

```bat
start "" "outputs\real_decision_review.html"
```

Edit every `decision` in `outputs\real_decision_decisions.json` to `approve` or `reject`, then finalize:

```bat
python -m stgat_lstm.review_decisions finalize outputs\real_decision_candidates.json outputs\real_decision_decisions.json --output outputs\real_approved_decisions.json
```

Do not skip manual review. Rebuilding `real_decision_candidates.json` can invalidate an older review because decisions are bound to the candidate file’s SHA-256 hash.

## 6. Optional synthetic training check

```bat
python -m stgat_lstm.decision_training synthetic --epochs 200 --output outputs\multitask_synthetic.pt
```

This verifies mechanics only and makes no real-world claim.

## 7. Train the approved real-data model

```bat
python -m stgat_lstm.decision_training real "..\OSM\data\osm\graphs\metro_manila_processed.graphml" "..\OSM\data\features\road_node_features.csv" outputs\real_approved_decisions.json --epochs 40 --output outputs\multitask_real.pt
```

Outputs:

- `outputs\multitask_real.pt`: model weights, configuration, provenance, and training summary
- `outputs\multitask_real.json`: readable report

## 8. Reload the checkpoint, predict, and route

```bat
python -m stgat_lstm.decision_predict outputs\multitask_real.pt "..\OSM\data\osm\graphs\metro_manila_processed.graphml" "..\OSM\data\features\road_node_features.csv" outputs\real_approved_decisions.json
```

Select a specific approved example:

```bat
python -m stgat_lstm.decision_predict outputs\multitask_real.pt "..\OSM\data\osm\graphs\metro_manila_processed.graphml" "..\OSM\data\features\road_node_features.csv" outputs\real_approved_decisions.json --example-key 11930c364eaa39de
```

## 9. Collect a current Mapbox observation

Set the token in the current terminal session:

```bat
set MAPBOX_ACCESS_TOKEN=YOUR_TOKEN_HERE
```

Collect traffic using longitude,latitude order:

```bat
python -m stgat_lstm.mapbox_traffic "120.9790,14.5800;120.9980,14.5400" --output outputs\traffic\taft_current.json
```

For a current inference demonstration, supply the traffic observation:

```bat
python -m stgat_lstm.decision_predict outputs\multitask_real.pt "..\OSM\data\osm\graphs\metro_manila_processed.graphml" "..\OSM\data\features\road_node_features.csv" outputs\real_approved_decisions.json --traffic-observation outputs\traffic\taft_current.json
```

Do not attach current traffic to an old decision and describe it as historical evidence. The option above demonstrates the online feature interface only.

## 10. Presentation-safe quick run

```bat
call .venv\Scripts\activate.bat
python -m unittest discover -s tests -v
type outputs\multitask_real.json
python -m stgat_lstm.decision_predict outputs\multitask_real.pt "..\OSM\data\osm\graphs\metro_manila_processed.graphml" "..\OSM\data\features\road_node_features.csv" outputs\real_approved_decisions.json
start "" "outputs\real_decision_review.html"
```

