# STGAT-LSTM rider preference prototype

Repository policy, 21 October 2026: real rider exports, private experiment
bundles, generated outputs, weights and rollback snapshots stay local and are
ignored by Git. Source, tests, guides and public OSM inputs remain versioned.
Existing commit history is intentionally retained; this exclusion applies to
future commits and does not remove rider data from older commits.
After cloning, restore the private inputs described in RUNBOOK Section 1 from
your authorized local backup. A clone alone does not include rider datasets or
the frozen experiment inputs referenced by the reproduction commands.

Next approved experiment: a controlled training comparison on the same frozen
rider folds. Follow [RUNBOOK Section 11](docs/RUNBOOK.md#11-controlled-training-comparison-on-the-same-rider-folds).
It tests three validation-selected loss/weight recipes with NLL checkpoint
selection. STGAT-LSTM-only takes 12 fits. Full results are pending. Architecture
and input features are unchanged; both completed runs and a pre-change code
snapshot are preserved for comparison or rollback.

The latest completed real-data experiment is three-fold rider-grouped evaluation, documented
in [RUNBOOK Section 10](docs/RUNBOOK.md#10-next-experiment-three-fold-rider-grouped-evaluation).
It pools held-out predictions across the original decisions, reports exact
deviation-road selection separately, and compares GATv2 with GATv2-LSTM.
It uses inner rider validation and 12 fits, without changing the architecture,
adding synthetic test cases, or overwriting the completed fixed-split baseline.
Completed 21 October 2026: STGAT-LSTM selects 10/17 alternative roads (58.8%),
17/51 followed roads (33.3%), and 27/68 overall (39.7%). GATv2 selects 5/17,
28/51, and 33/68 (48.5%). Always-follow selects 51/68 (75%) but no deviations.
This is a mixed internal result, not validated reliability or evidence that
traffic history helps. LSTM epochs selected by the three inner validations
are 45, 5 and 1. The existing fixed-split figures below use a different protocol.

Updated 20 October 2026: the active CSV/GeoJSON contain cleaned amenity counts.
Follow [RUNBOOK.md](docs/RUNBOOK.md) to train and evaluate the new real-only
baseline under `artifacts/experiments/real_cleaned_osm_20261020`. The user's run
through Step 8 is complete and reviewed: exact test-road accuracy is 9/14
(64.3%), versus 10/14 (71.4%) for always following the suggestion. Test choice
NLL is 0.871, versus 0.909 in the previous first-visit run; ranking is 2/3.
The validation-selected binary follow/deviate threshold gives 11/14 (78.6%),
which is a different metric from exact outgoing-road accuracy. Reload reproduces
all test probabilities exactly; both saved routes pass connectivity, access,
and detour checks. See `run_summary.json` in the new folder. Dashboard work is
deferred. The older metric tables below use the old features. Old input copies
are under `artifacts/inputs/osm_before_amenity_cleanup`; stale standalone outputs
are under `artifacts/archive/pre_cleaned_osm_20261020`. Dated experiment bundles
and all real road-choice labels are preserved. The current recipe retains the
first-visit policy and uses the existing three validation trials plus final
refit; it does not use the old hash-locked `--reuse-selection` option.

**Interactive demonstration after completing the refinement command below:** activate your environment, run `python -m stgat_lstm dashboard --checkpoint artifacts\experiments\real_cleaned_osm_20261020\checkpoint.pt --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --traffic-cutoff-ms 1791360905826`, and open http://127.0.0.1:8765. Select two points to see the recommended route and shortest-distance comparison on your supplied OSM network with fixed historical traffic context. Inspect the detour-fallback flag before calling the recommendation a learned route. The complete training, evaluation, and traffic commands are included below.

The repository layout separates command files, automatically imported implementation files, official real inputs, tests, and generated outputs. Required OSM and rider-export inputs are stored under `data/real/`, so normal commands do not depend on sibling project folders.

Detailed project notes are limited to the [project plan](docs/PLAN.md), [real-data runbook](docs/RUNBOOK.md), [synthetic runbook](docs/SYNTHETIC_RUNBOOK.md), [presentation guide](docs/PRESENTATION_GUIDE.md), and [complete model explanation](docs/MODEL_EXPLANATION.md).

The training, inference, routing, and local visualization connections are implemented. Timestamped rider GPS is direction-matched to OSM roads and converted into historical motorcycle speed profiles for the LSTM. This does not establish temporal usefulness without broader coverage, or rider generalization without further held-out data. The router enforces graph direction/access flags; OSM turn restrictions are outside the agreed research scope and are not represented. It remains a research prototype.

Command-line tools keep the full result in JSON (or HTML for the map views) and print a shorter explanation in the terminal. The terminal is for quick checking; the saved artifact is the record used for review, training, and reporting.

Evaluation reports preferred outgoing-road accuracy and negative log-likelihood as the primary choice measures. It also reports balanced accuracy, F1, AP, ROC-AUC, Brier score, and log loss for the derived follow/deviate view, plus held-out route ranking. Add `--fixed-split --temporal-ablation` to the complete evaluation command below to use separate real training/validation/test riders and a matched latest-only LSTM control. The report records whether observed traffic histories actually vary; running an LSTM does not itself prove a temporal benefit.

This repository contains an executable rider-choice pipeline. It reconstructs reviewable road choices and trains a GATv2-LSTM to choose among the accessible outgoing roads at each branching node. The model converts each local choice probability into a positive additive cost, and Dijkstra uses those costs to build a connected route. Travel-time savings are an optional secondary outcome, not the training target.

The current approved refinement is under `artifacts/experiments/real_refined_20261018`: 77 approved examples from 36 rides and the same 15 riders. It preserves the original 17 deviations and recovers 47 additional real followed choices from training riders. Four followed examples now use the active route destination consistently. Selection tests three ranking weights on validation riders, with AdamW, label smoothing, traffic masking, and early stopping; the selected setup is refitted on all 12 development riders before testing the same three excluded riders. The GATv2-LSTM architecture remains unchanged. See [RUNBOOK Section 6](docs/RUNBOOK.md) for the complete command.

The refinement completed with ranking weight 1.0 and 45 final epochs. Development accuracy is 67/101 (66.3%); test accuracy is 9/14 (64.3%), below the always-follow baseline's 10/14 (71.4%). It chooses eight of ten followed roads and one of four alternative roads correctly. Test choice NLL is 0.911, essentially tied with the current baseline and much lower than the older model's 3.373; route ranking improves to 2/3 pairs. The 80% target was not reached. The small development/test gap does not prove no overfitting, and these previously inspected test riders support only an internal pilot. All 89 tests, checkpoint reload, and connected routing checks pass. Read `artifacts/experiments/real_refined_20261018/run_summary.json` for the concise result.

The user's local first-visit run has produced a checkpoint and completed evaluation report under `artifacts/experiments/real_first_visit_20261018`; its scores await review. Within each original training example, it supervises only the earliest visit to a junction, omitting three later followed targets. It retains every example and alternative-road target, all original validation/test targets, the same architecture, and the previous settings. The two fixed 45-epoch fits do not search for new settings. See the first-visit subsection in [RUNBOOK Section 6](docs/RUNBOOK.md). The older scores above remain the last reviewed results.

The original preparation before refinement yields 68 supervised junction choices from 30 approved examples covering 28 rides and all 15 riders, including 17 alternative-road targets and 12 optional ranking pairs. Its artifacts are under `artifacts/experiments/real_navigation_20261016`. Full-data training has completed 100 epochs, with loss 1.8917 to 0.2863, 62/68 in-sample road choices correct, and 11/12 training pairs correctly ranked. Reload, rider-choice replays, and connected routing passed.

The completed real fixed-split pilot is under `artifacts/experiments/real_fixed_split_20261018`: nine training, three validation, and three test riders; five fresh 100-epoch model fits. ST-GAT-LSTM obtains 10/14 test road choices correct (0.714), matching the suggestion baseline, with deviation F1 0.500 versus baseline 0.000. Its test choice NLL is worse (3.373 versus 0.830), and only 1/3 test route pairs is ranked correctly. This is mixed internal pilot evidence, not a reliable rider-preference or calibrated-probability claim.

The earlier 16-example strict baseline is archived under `artifacts/experiments/real_final_20261014`.

A separated 144-example synthetic experiment provides a controlled implementation-capacity check with multiple injected Taft patterns and validation-only class-weight selection. On its untouched two-rider test split, ST-GAT-LSTM reaches 0.846 outgoing-choice accuracy, 0.955 balanced accuracy, and 0.800 deviation F1.

The recommended `--refine` evaluation reuses the saved 9/3/3 rider split, compares three ST-GAT-LSTM ranking weights on validation riders, and refits the selected setup on the 12 development riders. The original five-architecture `--fixed-split` comparison remains available as a separate baseline. Validation/test riders are excluded from selection traffic; test riders are excluded from final-refit traffic. Target rides and future observations are always excluded. The optional leave-one-rider-out benchmark can require 75 fits. The six rider CSVs used in these runs are preserved under `artifacts/experiments/real_navigation_20261016/source_snapshot/rider_exports`.

## Repository layout

```text
STGAT-LSTM/
  README.md             Project overview
  requirements.txt      Python dependencies
  .gitignore            Local files excluded from Git
  stgat_lstm/           Python source; imported through the single CLI
    __main__.py         Dispatches python -m stgat_lstm commands
    preprocessing/      Audits, map matching, labels, graph features, traffic
    modeling/           model.py, train_model.py, evaluate_model.py
    routing/            Prediction, Dijkstra, dashboard.py and dashboard.html
    generation/         Controlled synthetic-data generator
  data/real/            Official rider exports and OSM inputs
  generated/            Prepared data that the pipeline can regenerate
    real/               Candidates, geometry, approved training examples
    synthetic_hotspot/  Generated rider CSVs, labels, traffic probe rides
    traffic/            Collected traffic samples and archive
  artifacts/            Saved models and reproducible experiment records
    checkpoints/        Standalone real/test/synthetic checkpoints
    experiments/        Dated real runs, including model, reports, input snapshot
    layout_manifest.json  Old/new locations and preservation checks
  outputs/              Files produced for inspection
    audits/             Data, graph, matching, and traffic checks
    reports/            Standalone training/evaluation reports
    predictions/        Route and road-choice prediction JSON
    maps/               Generated candidate-review HTML
  docs/                 Runbooks, model explanation, presentation, plans
  tests/                Automated checks
```

The local `.venv/` holds installed dependencies and `.git/` holds version
history. The app source clone under `docs/reference_code/` is kept for reference,
is ignored by Git, and is not required to run the model.

Folder layout updated 19 October 2026. A dated experiment is a complete saved
record: its checkpoint, reports, plan, approved examples, and any input snapshot
remain together under `artifacts/experiments`. Standalone inspection files go
under `outputs`. Historical JSON/checkpoint provenance keeps the paths used at
training time; `artifacts/layout_manifest.json` maps every relocated file to its
new location. Archived files were not rewritten to change their recorded paths.
The command names and model architecture are unchanged. Use the updated runbook
paths when copying commands from an older terminal session.

The modules reflect separate data, modeling, review, evaluation, and routing stages. Start with these files:

| Area | Primary files | Purpose |
|---|---|---|
| Candidate construction | `stgat_lstm/preprocessing/build_deviation_candidates.py`, `build_decision_candidates.py` | Reconstruct strict deviation paths, then combine followed and deviated local choices |
| Candidate validation | `stgat_lstm/preprocessing/create_review_map.py`, `review_decisions.py` | Apply reproducible survey/GPS rules and optionally display candidate paths |
| Model and evaluation | `stgat_lstm/modeling/model.py`, `train_model.py`, `evaluate_model.py` | Define and train the models, then compare held-out riders across baselines |
| Inference and routing | `stgat_lstm/routing/predict_route.py` | Reload the checkpoint, predict preferred outgoing roads, score edges, and run Dijkstra |
| Real graph and traffic | `stgat_lstm/preprocessing/graph_data.py`, `gps_traffic.py` | Build OSM tensors and derive historical road-speed context from rider GPS |
| Controlled capacity test | `stgat_lstm/generation/synthetic_hotspot.py` | Generate separated app-shaped decisions with multiple Taft patterns and controlled traffic history |

Supporting modules such as `geometry.py`, `coordinates.py`, `map_matching.py`, `audit_road_network.py`, `audit_map_matching.py`, and `preference_pairs.py` are imported by those primary files. They are not duplicate entry points. `coordinates.py` performs the WGS84/UTM conversion required by the official EPSG:32651 Manila graph without an external PROJ DLL. `audit_rider_data.py` is an optional diagnostic. `__main__.py` provides the single `python -m stgat_lstm` command interface. Files under `tests` verify the real-data pipeline with small in-memory graphs. Files under `outputs` are generated artifacts rather than source code.

## What runs

- `stgat_lstm/modeling/model.py` contains a two-layer GATv2 graph encoder and an LSTM across graph snapshots. The choice head returns one utility logit per directed edge. A softmax over accessible roads with the same source node turns those logits into outgoing-road probabilities.
- `DecisionPreferenceModel` converts each accessible edge probability to `-log(probability)` plus a very small positive length tie-breaker. These dimensionless additive values are the routing costs; they are not travel seconds.
- `stgat_lstm/modeling/train_model.py` trains the preferred-road head with class-balanced multiclass cross-entropy, adds route-ranking supervision when comparable intentional paths exist, saves a checkpoint, and refuses real decision artifacts that have not passed hash-bound validation.
- The GAT and GATv2 operators explicitly use the standard attention `negative_slope=0.2` for their internal LeakyReLU attention calculation. The model applies ELU after attention layers; it does not apply a second LeakyReLU to their outputs.
- `stgat_lstm/preprocessing/audit_rider_data.py` reads the official real CSV exports and reports aggregate readiness checks. It parses route geometry, measures deviation locations against the latest pre-event route, checks route endpoints and summarizes GPS coverage around each event.
- `stgat_lstm/preprocessing/audit_road_network.py` checks whether the supplied projected, directed OSM graph can support map matching, measures route/GPS distance to its edges, verifies the separate node-feature join, and quantifies how much of the current collection actually falls inside the provisional Taft Avenue corridor.
- `stgat_lstm/preprocessing/map_matching.py` implements directed, sequence-aware HMM/Viterbi map matching. It combines GPS-to-road distance with network-versus-observed displacement and retains the complete traversed edge sequence, including connector edges between observations.
- `stgat_lstm/preprocessing/preference_pairs.py` extracts GPS-supported local outgoing-road choices without requiring a rejoin. When necessary, it combines recorded navigation, road-name aliases, compass heading, event location, active-route geometry, and directed GPS progress to infer one supported choice. It also extracts optional observed/suggested paths with shared endpoints for route-ranking supervision. Recorded navigation is label evidence, not an inference feature.
- `stgat_lstm/preprocessing/build_deviation_candidates.py` creates a pseudonymous, coordinate-free review artifact from the real export. It does not feed unreviewed candidates into training.
- `stgat_lstm/preprocessing/build_decision_candidates.py` combines supported local deviation choices with conservative followed examples. Follow labels require GPS agreement through a real branching node. Later GPS and surveys are label evidence only.
- `stgat_lstm/preprocessing/graph_data.py` starts with a 1 km Taft Avenue OSM buffer and expands it to cover approved evidence and destinations, joins POI features, builds configurable model tensors, marks absent historical traffic as unknown, and refuses to load review-required choices for training. The original real graph has 2,379 nodes and 6,098 edges; the refinement covers 2,625 nodes and 6,765 edges. Inference reloads the saved checkpoint scope.
- `stgat_lstm/routing/predict_route.py` reloads real checkpoints, scores OSM edges and runs Dijkstra over positive directed-edge costs, excluding restricted-access edges. A configurable hard guard defaults to 1.30 times the shortest-route distance and falls back to the distance baseline when the unconstrained learned route is excessive.
- `stgat_lstm/preprocessing/mapbox_traffic.py` collects timestamped Mapbox `driving-traffic` annotations without storing the token, aligns observed route segments to directed OSM edges, and keeps speed and congestion availability separate.
- `stgat_lstm/preprocessing/gps_traffic.py` converts consecutive two-second rider GPS fixes into direction-matched motorcycle speed records. It creates pre-decision time-of-day histories, excludes the target ride and held-out riders, and never writes raw coordinates to its audit.
- `stgat_lstm/generation/synthetic_hotspot.py` creates a separated controlled experiment using multiple verified Taft decisions, traffic-only GPS probe rides, and the same six app CSV schemas. Its output tests implementation and temporal capacity and is never treated as real rider evidence.
- `stgat_lstm/preprocessing/review_decisions.py` automatically accepts supported intentional reasons and strong followed examples after directed-path and GPS checks. It excludes unsupported reasons, missing traffic severity and inconsistent geometry, then creates a hash-bound training artifact. Manual templates remain available for exceptional audits.

At each supervised junction, multiclass cross-entropy raises the probability of the outgoing road actually chosen by the rider relative to the other accessible alternatives. Because alternative-road choices are rare, their loss weight is set to `followed choices / alternative choices`, giving the two target groups equal aggregate influence. For a matched chosen path \(P_c\) and rejected suggestion \(P_s\), the auxiliary ranking loss is `softplus(cost(P_c) - cost(P_s))`. At routing time, Dijkstra selects a connected path with the lowest sum of `-log P(edge | source, context)` costs. The model does not impose a faster-path preference.

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
python -m stgat_lstm gps-traffic "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --output outputs\audits\gps_traffic_audit.json
python -m stgat_lstm build-deviations "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --output generated\real\deviation_candidates.json --geojson-output generated\real\deviation_candidates.geojson
python -m stgat_lstm build-decisions "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --deviation-candidates generated\real\deviation_candidates.json --output generated\real\decision_candidates.json --geojson-output generated\real\decision_candidates.geojson
python -m stgat_lstm review-map generated\real\decision_candidates.geojson --output outputs\maps\candidate_review_map.html
python -m stgat_lstm review auto generated\real\decision_candidates.json --output generated\real\approved_training_examples.json
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" artifacts\experiments\real_fixed_split_20261018\approved_training_examples.json --refine --first-visit-only --split-file artifacts\experiments\real_fixed_split_20261018\rider_split.json --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 60 --batch-size 8 --output artifacts\experiments\real_cleaned_osm_20261020\evaluation_report.json
python -m stgat_lstm predict artifacts\experiments\real_cleaned_osm_20261020\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" artifacts\experiments\real_cleaned_osm_20261020\approved_training_examples.json --gps-traffic-data "data\real\rider_exports" --output outputs\predictions\cleaned_osm_reviewed_example.json
python -m stgat_lstm dashboard --checkpoint artifacts\experiments\real_cleaned_osm_20261020\checkpoint.pt --gps-traffic-data "data\real\rider_exports" --max-detour-ratio 1.30
```

The refinement command above reproduces the accepted 69-ride dataset using the frozen original artifact and rider assignments. It performs three validation fits and one final development fit, and saves its own checkpoint; a separate `train real` command is not needed for this workflow. See the runbook before replacing these inputs with a later dataset. The aggregate audits are required whenever the official rider exports or OSM inputs change.

Candidate selection no longer enforces a 500 m Taft corridor or a required
within-corridor GPS fraction. Network proximity counts remain descriptive.
The archived `artifacts/experiments/real_final_20261014` results use the former policy;
rebuilt candidates after removing it are under `artifacts/experiments/real_no_corridor_20261015`.
Model graph coverage is checked separately before those examples are trained.

The main traffic source is the app's GPS track. Consecutive fixes are converted to speed, direction-matched to OSM edges, filtered for long gaps and implausible speeds, and aggregated by local time of day. The five dynamic fields are slowdown relative to the OSM reference, speed ratio, two observation masks, and observation age. Missing coverage stays explicitly unknown. The final export yields 6,801 usable speed segments on 161 roads from 69 rides, with a 2.001-second median interval. Training and held-out evaluation have different rider exclusions; inspect the final report's temporal evidence and fold coverage before claiming a history benefit.

Mapbox ingestion remains as optional compatibility code for earlier experiments. It is not required by the current thesis pipeline, and present Mapbox observations are never attached retrospectively to old rides.

If the `py -3.12` launcher is unavailable, use `python -m venv .venv` instead. A new cmd window must be activated again with `call .venv\\Scripts\\activate.bat`.

On the reviewed machine, this project's `.venv\Scripts\python.exe` has the model libraries installed. Activate that environment before running the commands.

The audit uses only the Python standard library, so it can run with a separate Python installation even before ML dependencies are installed.

## Verified behavior and limits

The automated suite exercises prior-route reconstruction, exclusion of wrong turns and personal stops, missing and late traffic, incomplete GPS observations, local follow-choice construction, all model forward/backward paths, motorcycle access, pre-choice inference isolation, review hashes, evaluation, and learned-cost routing.

The leave-one-rider-out command compares a suggestion-following baseline, GCN, original GAT, spatial-only GATv2, and GATv2-LSTM, with a latest-only LSTM control when observed histories vary. The archived strict-baseline report is `artifacts/experiments/real_final_20261014/rider_holdout_evaluation.json`. The suggestion baseline reaches 0.944 choice accuracy because 51/54 choices are follows, but its deviation F1 is zero. These metrics describe the former three-positive preparation, not the revised 17-positive preparation.

The completed archived strict baseline gives ST-GAT-LSTM 0.556 held-out choice accuracy,
0.745 balanced accuracy, and 0.188 deviation F1. It detects all three deviations
but falsely flags 26 followed choices. Spatial GATv2 performs better on choice
accuracy (0.667) and F1 (0.240). The full model ranks all three held-out route
pairs correctly, but both tested fresh routes require the detour fallback.
These are weak real choice results, not a successful generalization claim.
See `artifacts/experiments/real_final_20261014/run_summary.json` for the complete run record.

The final official export contains 69 rides from 15 riders, 124 saved routes, 37 reported deviations, 37 completed deviation responses, 7,826 GPS points, and 69 post-trip questionnaires. All 37 deviation records reference a route generated after their event timestamp, so the pipeline reconstructs the latest route that existed before each event and then performs directed sequence map matching. Revised automatic validation retains 30 examples from 28 rides and all 15 riders: 13 followed and 17 intentionally deviated. They produce 68 supervised outgoing-road choices: 51 suggested roads and 17 alternatives. Missing post-report GPS alone is no longer a rejection. Looking back within the same active-route period recovers six turns reported 70 to 309 seconds after the first alternative fix. Recorded navigation and local geometry recover a further choice of Castro Street over Fidel A. Reyes Street. The three original GPS-sequence failures remain excluded.

The OSM compatibility command can take several seconds because it loads the 74 MB Metro Manila graph and spatially checks every saved route and GPS point. Its independent nearest-edge results are quality diagnostics. They are not sequence-aware map matching and must not be used directly as rider-choice targets.

The map-match audit applies the HMM to eligible deviation events across the
supplied graph, without a distance-to-Taft eligibility filter. It compares the
post-event GPS edge sequence with the prior and regenerated route sequences and
reports candidates where the routes differ and GPS agrees more with the
regenerated route. This audit is a legacy triplet diagnostic, not the current
local-label selection rule. Common-endpoint route-ranking pairs remain optional.

On the final export, the compatibility audit found 35 eligible deviation events inside the provisional corridor. The preliminary HMM sequence screen matched GPS, prior route and regenerated route for 23 events; 12 of those agree more strongly with the regenerated route. The stricter builder uses a different GPS window and additional pair requirements, and retains three pairs. These are data-preparation results, not model accuracy.

The current candidate builder requires an identifiable suggested road and GPS-supported alternative at the same branching source, with at least two consecutive fixes on the alternative continuation. It does not require following the regenerated route or rejoining the old route. Where shared endpoints exist, it additionally emits a comparable ranking pair. Its JSON contains pseudonymous grouping keys and no coordinates. The optional GeoJSON contains selected OSM road geometry, not raw rider GPS points.

The decision-dataset builder uses local road choices as the prediction unit. A followed example must traverse the suggested path through a branching node. A deviated example needs an intentional survey response and evidence of the outgoing alternative. It produces 31 candidates; automatic validation rejects one followed candidate whose mean matching distance exceeds 20 m, leaving 30 approved examples. Twelve deviations also supply comparable ranking paths; five train the choice objective alone. Navigation-based labels additionally require unique inference evidence matching their edge IDs and pre-choice cutoff.

The review-map command creates a local HTML page with a candidate selector, path details, embedded OSM context roads, and separate styling for the observed and suggested paths. It loads the Leaflet library from a CDN but does not request public map tiles. The embedded candidate layer contains OSM path geometry, not raw GPS.

The latest deviation stage retains 17 events from 15 rides and nine riders. The supplied notebook motivates six matching candidates and an 8 m same-edge backward-jitter tolerance per transition. The source CSVs are unchanged, and all 29 examples from the preceding salvage trial preserve their targets, paths, and cutoffs. Per-event diagnostics and navigation recovery details are in `artifacts/experiments/real_navigation_20261016/navigation_report.json`. The notebook's obsolete trimming section is not used.

The current shortest-path implementation assumes additive edge costs. It applies a configurable maximum-distance guard after learned-cost routing and returns the shortest-distance baseline when the unconstrained result exceeds that limit. Turn-dependent restrictions and costs require an augmented search state. The toy graph uses invented road access and traffic values; it makes no claim about Taft Avenue.

The model implementation draws on the [original GAT paper](https://arxiv.org/abs/1710.10903), [GATv2 paper](https://arxiv.org/abs/2105.14491), [original LSTM paper](https://doi.org/10.1162/neco.1997.9.8.1735), [spatial-temporal GAT traffic paper](https://doi.org/10.1109/ACCESS.2019.2953888), and [PyTorch Geometric operators](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html). These are component references, not claims that the current architecture reproduces a full published traffic model.
