# Real-data STGAT-LSTM runbook for Windows cmd

Updated 20 October 2026 for the cleaned OSM amenity features.

Run commands from the project root. This trains a new real-only baseline with
the cleaned CSV under `data\real\osm`. The road network and rider exports are
unchanged. It does not generate simulated rider decisions.

For an existing installation: activate the environment in Section 1, run
Sections 2 and 3, then Sections 5 through 9. Section 4 is optional for this
amenity-only update because the road-choice labels remain valid.
A new user may also run Section 4 to inspect the complete preparation pipeline.

New run folder: `artifacts\experiments\real_cleaned_osm_20261020`.
The user's run through Step 8 completed and was reviewed on 20 October:
12/13 validation choices before refit; 63/98 fitted development choices;
9/14 (64.3%) exact held-out road choices, versus the suggestion baseline's
10/14 (71.4%). Test choice NLL is 0.871 and route ranking is 2/3. The selected
weight/epoch are 1.0/45. The 78.6% follow/deviate accuracy at the selected binary
threshold is a separate metric. Checkpoint reload reproduces all 14 test
probabilities exactly; both Step 8 routes pass connectivity/access/detour checks.
See the saved `run_summary.json`, `checkpoint_reload_check.json`, and
`run_manifest.json`. No retraining is needed to inspect this completed run;
the commands below remain available for intentional reproduction. Section 9
and visualization improvements are deferred.

## 1. Activate the environment

~~~bat
cd /d C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\STGAT-LSTM
call .venv\Scripts\activate.bat
~~~

Only if the environment does not exist:

~~~bat
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

On another computer, replace the project path with your own.

Private-input policy, 21 October 2026: real rider exports, generated outputs,
model weights, experiment bundles and rollback snapshots are ignored by Git.
They remain on the existing machine; a fresh clone does not contain them.
Before running real-data commands on another machine, restore these folders
from your authorized private backup into the same project-relative locations:

- `data\real\rider_exports`: official CSV exports for the audit/preparation commands.
- `artifacts\experiments\real_fixed_split_20261018`: frozen original approved
  examples and rider split used by Sections 6, 10 and 11.
- `artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports`:
  frozen GPS exports used by the accepted experiment commands.
- `artifacts\experiments\real_group_kfold_20261020`: completed reference report,
  prepared corpus and rider folds needed for Section 11's locked comparison.

To replay a saved model without training, restore its full experiment folder
as well. Do not substitute newly reconstructed labels for a frozen artifact
when reproducing a hash-checked comparison. The current machine already has
these inputs, so its running experiment and commands need no changes.

## 2. Verify the code

~~~bat
python -m unittest discover -s tests -v
~~~

These unit and integration tests check code behavior using small fixtures.
They do not measure accuracy on real held-out riders.

## 3. Audit the rider exports and cleaned OSM features

~~~bat
python -m stgat_lstm audit-data "data\real\rider_exports"
python -m stgat_lstm audit-network "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv"
python -m stgat_lstm audit-matching "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml"
python -m stgat_lstm gps-traffic "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --output outputs\audits\gps_traffic_audit.json
~~~

Reports are saved under `outputs\audits` as `rider_data_audit.json`,
`road_network_audit.json`, `map_matching_audit.json`, and `gps_traffic_audit.json`.
They check export relationships, graph/feature compatibility, sequence map
matching, and historical GPS-derived motorcycle speeds.

The GPS audit initially uses the default graph buffer. Training expands the
graph to cover approved roads/destinations and builds its own traffic archive;
it does not consume the audit JSON.

## 4. Optional: reconstruct and inspect training examples

The amenity cleanup changed feature values, not roads, node IDs, GPS, or choice
labels. Existing approved examples remain usable. For a full preparation demo:

~~~bat
python -m stgat_lstm build-deviations "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --output generated\real\deviation_candidates.json --geojson-output generated\real\deviation_candidates.geojson
python -m stgat_lstm build-decisions "data\real\rider_exports" "data\real\osm\metro_manila_processed.graphml" --deviation-candidates generated\real\deviation_candidates.json --output generated\real\decision_candidates.json --geojson-output generated\real\decision_candidates.geojson
python -m stgat_lstm review-map generated\real\decision_candidates.geojson --output outputs\maps\candidate_review_map.html
python -m stgat_lstm review auto generated\real\decision_candidates.json --output generated\real\approved_training_examples.json
~~~

The first two commands reconstruct suggested/observed choices from surveys,
navigation, and directed GPS evidence. Rejoining the previous route is not
required for a local road-choice label. Comparable paths additionally supply
route-ranking supervision. The HTML is optional; automatic validation replaces
routine manual approve/reject decisions.

Later GPS and survey answers establish labels, not inference features.
Wrong turns, GPS errors, and personal stops are not positive preference labels.

Section 6 deliberately reuses the frozen original artifact and rider split.
It does not substitute this optional reconstruction for the accepted comparison
inputs, preserving the original labels and rider assignments.

## 5. Inputs and settings for the new baseline

Section 6 performs training and evaluation together. You do not need to run
a separate full-data training command first.

The comparison reuses:

- Approved source examples:
  `artifacts\experiments\real_fixed_split_20261018\approved_training_examples.json`
- The saved nine-training/three-validation/three-test rider split:
  `artifacts\experiments\real_fixed_split_20261018\rider_split.json`
- Frozen GPS exports:
  `artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports`
- The unchanged GraphML and cleaned `road_node_features.csv`.

Preparation corrects destination context and recovers additional genuine
followed choices from training riders. The approved first-visit policy remains:
only the earliest supervised junction visit within an original training example
is fitted. Alternative-road labels and validation/test targets remain intact.

Architecture, seed 17, batch size 8, six five-minute history frames, AdamW
settings, smoothing, traffic masking, and ride/class balancing are unchanged.
The existing validation procedure selects among ranking weights 0, 0.1, and 1.0.
The selected weight and epoch can differ with the cleaned inputs. This is a
newly validation-selected baseline, not a fixed-setting amenity ablation.

Do not use the old `--reuse-selection` command with the cleaned CSV. That mode
requires identical input hashes and correctly rejects the feature update.

## 6. Train and evaluate

~~~bat
python -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" artifacts\experiments\real_fixed_split_20261018\approved_training_examples.json --refine --first-visit-only --split-file artifacts\experiments\real_fixed_split_20261018\rider_split.json --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 60 --batch-size 8 --seed 17 --output artifacts\experiments\real_cleaned_osm_20261020\evaluation_report.json
~~~

This performs four fits:

1. Three selection fits on nine training riders, one per predefined ranking
   weight. Validation runs at epoch 1 and every five epochs; best choice accuracy
   wins, with lower NLL breaking ties. Patience is three validation checks and
   the maximum is 60 epochs per fit.
2. One final refit on 12 development riders at the selected weight and epoch
   count. Only afterward is the model evaluated on the same three test riders.

Expected first-visit counts are 65 selection-training examples/85 choices and
71 final-development examples/98 fitted choices. Validation/test remain 13/14
choices. These are consistency checks, not accuracy targets.

Selection traffic excludes validation/test riders. Final-refit traffic excludes
test riders. Both exclude the target ride and future records. Traffic masking
hides training observations as unknown; it invents no roads or rider choices.
Validation/test inputs are not masked.

Files saved in the new experiment folder:

| File | Contents |
|---|---|
| `experiment_plan.json` | Settings, rider groups, first-visit adjustments, source hashes |
| `approved_training_examples.json` | Refined real training examples |
| `selection_report.json` | Validation trial results |
| `checkpoint.pt` | Final development-trained model |
| `training_report.json` | Epoch losses, fitted counts, provenance |
| `evaluation_report.json` | Test metrics, baseline, development metrics, traffic, overfitting checks |

Read these evaluation fields:

- `test.outgoing_choice`: exact chosen-road accuracy and choice NLL.
- `suggestion_baseline`: always-follow comparison.
- `test.classification`: derived follow/deviate metrics at threshold 0.5.
- `test.classification_at_validation_threshold`: separate threshold metrics.
- `test.preference_ranking`: comparable-route ranking.
- `development` and `development_original_targets`: fitted 98-target versus
  original 101-target development views.
- `overfitting_checks`: validation curves and clean development/test comparisons.

Weighted training loss and clean NLL have different scales. Validation riders
join training in the final refit, so their earlier selection score is not an
independent evaluation of that model. Small gaps do not prove no overfitting.

The same test riders have been inspected before. This remains a repeated
internal pilot, with only 14 test choices. Report the measured result, even if
below 80%; do not regenerate the split to increase accuracy.

Do not rerun Section 6 just to open a prediction or dashboard. It overwrites
its selected output folder. Use a different folder for any intentional repeat.

## 7. Predict a reviewed example

Run after Section 6 finishes:

~~~bat
python -m stgat_lstm predict artifacts\experiments\real_cleaned_osm_20261020\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" artifacts\experiments\real_cleaned_osm_20261020\approved_training_examples.json --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --output outputs\predictions\cleaned_osm_reviewed_example.json
~~~

This reloads the checkpoint, reports outgoing-road probabilities, compares path
costs where applicable, and generates a connected route. Replay traffic excludes
the example's rider and ride.

## 8. Generate a route

Using today's request time:

~~~bat
python -m stgat_lstm predict artifacts\experiments\real_cleaned_osm_20261020\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 400 --destination-node 6 --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --max-detour-ratio 1.30 --output outputs\predictions\cleaned_osm_current_route.json
~~~

For a historical daytime demonstration:

~~~bat
python -m stgat_lstm predict artifacts\experiments\real_cleaned_osm_20261020\checkpoint.pt "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" --origin-node 400 --destination-node 6 --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --traffic-cutoff-ms 1791360905826 --max-detour-ratio 1.30 --output outputs\predictions\cleaned_osm_historical_route.json
~~~

Dijkstra searches positive learned costs on accessible directed roads. If the
learned route exceeds 1.30 times the shortest-distance route, it returns the
distance baseline and records the fallback. This is a distance guard, not
ground-truth route-quality validation.

Missing historical observations stay unknown, not free-flow. A current-time
request can have little traffic coverage. These commands do not call Mapbox.

## 9. Open the dashboard

~~~bat
python -m stgat_lstm dashboard --checkpoint artifacts\experiments\real_cleaned_osm_20261020\checkpoint.pt --graphml "data\real\osm\metro_manila_processed.graphml" --node-features "data\real\osm\road_node_features.csv" --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --traffic-cutoff-ms 1791360905826 --max-detour-ratio 1.30
~~~

Open **http://127.0.0.1:8765**, select two points, and generate a route.
Green is the learned route; blue is the distance baseline. Local OSM geometry
is embedded; no public tile server is needed. Stop with Ctrl+C.

The fixed cutoff is a historical replay. Omit `--traffic-cutoff-ms` to use the
current request time and whatever historical coverage is available.

## 10. Next experiment: three-fold rider-grouped evaluation

Completed and reviewed on 21 October 2026: all 12 fits finished. The report has
status `completed`; no rerun is needed to inspect these results.

| Model | Exact deviation road | Exact followed road | All exact roads |
|---|---|---|---|
| GATv2 | 5/17 (29.4%) | 28/51 (54.9%) | 33/68 (48.5%) |
| STGAT-LSTM | 10/17 (58.8%) | 17/51 (33.3%) | 27/68 (39.7%) |
| Always follow suggestion | 0/17 | 51/51 | 51/68 (75.0%) |

STGAT-LSTM's selected epochs are 45, 5 and 1; GATv2's are 60, 1 and 1.
Early selection on only 9-10 validation choices is a concern to investigate,
not proof that longer training would improve held-out results. Seven of 17
deviation exits have no chosen-exit supervision in their fold's development
data. STGAT-LSTM's pooled choice NLL is 0.988, versus 0.943 for assigning equal
probability to every accessible candidate (calculated using the saved candidate
sets). More exact deviation hits come with substantially poorer followed-road
performance. Do not claim a reliable model or a temporal-history benefit from
this run; there is no latest-only control or repeated-seed comparison here.
These scores are not a direct improvement comparison with the earlier 1/4
fixed-split result because the training and evaluation partitions differ.

Run this directly after activating the environment; do not rerun Sections 3-9
first. It uses the same official rider data, cleaned OSM features, architecture,
losses, and regularization. It changes the evaluation protocol. The completed
baseline in `real_cleaned_osm_20261020` remains intact.

~~~bat
python -u -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" artifacts\experiments\real_fixed_split_20261018\approved_training_examples.json --group-k-folds 3 --architectures gatv2 stgat_lstm --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 60 --learning-rate 0.001 --preference-weight 1.0 --batch-size 8 --seed 17 --output artifacts\experiments\real_group_kfold_20261020\evaluation_report.json
~~~

Use the original artifact shown above, not the refined 77-example artifact.
Preparation corrects destination context and recovers followed choices once
using the same rules for all riders. Those additional examples enter fitting
only when their rider belongs to that fold's fitting group. They never enlarge
validation or test. There are no synthetic rider decisions in this experiment.

The 15 riders form three outer test groups of five. For each fold:

1. Eight riders train the selection model; two other riders provide inner
   validation. Epoch selection uses exact-road accuracy, with NLL breaking ties.
2. A fresh model fits all ten development riders for that selected epoch count.
3. Five held-out riders are tested only after selection and refit.

GATv2 and GATv2-LSTM each perform selection plus refit in each fold: **12 fits**
in total. There is no ranking-weight search. Maximum selection training is 60
epochs with the existing early-stopping policy. Optional `--temporal-ablation`
adds the same-size latest-only LSTM control, bringing the total to 18 fits.
This control is required before attributing an improvement specifically to
history rather than simply to adding the LSTM architecture.

Each original eligible road choice is tested once across all folds. The
current evaluation corpus contains 30 original examples, 68 junction choices,
and 17 alternative-road decisions from nine riders; pooled
evaluation can therefore assess these rather than only the four in the fixed
test split. It does not create new independent cases. Fold assignments balance
target counts before fitting and are frozen in `rider_folds.json`.

The saved seed-17 plan has 22 test choices/5 deviations in Fold 1, 23/6 in
Fold 2, and 23/6 in Fold 3. The 68 pooled choices exclude the refinement's
extra followed examples, which remain available to the appropriate fitting
riders. This is why the evaluation total differs from the 115-target refined
corpus; all 17 original deviation choices are retained.

Selection GPS profiles exclude validation and test riders. Refit/test profiles
exclude test riders. Target rides and records at or after the decision remain
excluded. Missing traffic stays unknown. The known OSM graph is shared; riders
are held out, not the road network. Training first-visit filtering remains
enabled; original validation/test road targets stay unchanged.

Files are saved under `artifacts\experiments\real_group_kfold_20261020`:

- `rider_folds.json`: frozen outer and inner rider groups, dataset/target hashes.
- `experiment_plan.json`: fixed settings and source hashes before fitting.
- `approved_training_examples.json`: uniformly prepared real corpus. Its extra
  followed examples are a fitting pool, not additional test cases.
- `fold_1`, `fold_2`, `fold_3`: each model's checkpoint, training report, validation
  history, development results, held-out predictions, and traffic diagnostics.
- `evaluation_report.json`: pooled results and all fold reports. During fitting
  its status is `in_progress`; use final metrics only when it says `completed`.

The terminal prints exact correct/total counts for **deviations**, followed
roads, and all choices. For STGAT-LSTM, inspect:

- `models.stgat_lstm.choice_breakdown.deviated`: correct alternative road and
  direction, which is the main concern in this experiment.
- `models.stgat_lstm.choice_breakdown.followed`: correct suggested road.
- `models.stgat_lstm.outgoing_choice`: pooled accuracy and NLL.
- `models.stgat_lstm.per_rider` and `fold_choice_accuracies`: variation across
  people and folds; pooled rows are not independent riders.
- `folds[*].models.stgat_lstm`: validation curves and development/test results
  for assessing fit. Validation becomes part of the refit, so its earlier score
  is not an independent evaluation of the final fold checkpoint.

Each neural test prediction also records `chosen_exit_training_choices` and
`junction_training_choices`. These count fitting targets, not independent
riders. A zero chosen-exit count identifies an exit never selected in that
fold's development supervision. These are post-prediction diagnostics, never
input features, and do not prove the cause of a wrong prediction.

Binary deviation detection remains a separate metric in `classification`.
Choosing an incorrect alternative road can count as correct binary detection
but incorrect exact-road selection. Pooled binary metrics use threshold 0.5;
fold-specific validation thresholds are recorded separately within each fold.

This is exploratory internal cross-validation on already inspected riders.
Do not choose folds based on scores, treat folds as new participants, or call
the result an untouched external test. Changing input labels or seed requires
a separate fold-plan file. A repeat run deliberately overwrites this new run
folder; use another output folder to retain separate experiments.

## 11. Controlled training comparison on the same rider folds

This is the next run approved on 21 October 2026. Its full real-data results
are pending. Activate the environment and run this command directly; do not
repeat audits, candidate generation or the completed Section 10 experiment.

~~~bat
python -u -m stgat_lstm evaluate "data\real\osm\metro_manila_processed.graphml" "data\real\osm\road_node_features.csv" artifacts\experiments\real_fixed_split_20261018\approved_training_examples.json --group-k-folds 3 --architectures stgat_lstm --training-ablation artifacts\experiments\real_group_kfold_20261020\evaluation_report.json --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --history-steps 6 --history-interval-seconds 300 --gps-profile-bin-seconds 300 --epochs 60 --learning-rate 0.001 --preference-weight 1.0 --batch-size 8 --seed 17 --output artifacts\experiments\real_training_ablation_20261021\evaluation_report.json
~~~

The reference argument reuses the completed run's exact prepared real corpus
and frozen outer/inner rider groups. It checks the original artifact, OSM,
features, GPS CSVs, history settings, seed, epoch ceiling, batch size and
regularization against that run. A changed input or setting is rejected. It
copies the rider plan into the new folder and does not run map matching again.

This command focuses on STGAT-LSTM and requires **12 fits**: three selection
recipes plus one selected development refit per fold. It does not retrain GATv2.
The earlier GATv2 result remains available, but it used the earlier training
selection policy and is not a matched architecture comparison with this new
policy. Including both architectures in this command would require 24 fits.

| Recipe | Route-ranking weight | Deviation weight | Epoch selection |
|---|---|---|---|
| `current_losses_nll` | 1.0 | Current automatic weight | Lower validation choice NLL |
| `choice_only_nll` | 0 | Current automatic weight | Lower validation choice NLL |
| `choice_reduced_weight_nll` | 0 | Square root of automatic weight | Lower validation choice NLL |

The automatic weight is computed using fitting targets and the existing ride
balancing. For example, a weight of 3.24 becomes 1.8 in the reduced recipe.
It is recomputed on development data for refit. Validation/test labels never
set this weight. Ride balancing, AdamW, smoothing, masking and first-visit
filtering remain the same as the reference. Zero ranking weight removes its
contribution to the objective, not the original path evidence or diagnostics.

Each recipe selects its checkpoint using lower unsmoothed validation choice
NLL; exact-road accuracy breaks ties. Patience remains three validation checks,
with checks at epoch 1, every five epochs, and the maximum epoch. There is no
forced minimum epoch. This changes the selection signal, not an assumption that
more training must help. The reference used accuracy first; its saved curves
provide the earlier selection-policy comparison.

Within each fold, lower inner-validation NLL selects one of the three recipes;
mean exact-choice accuracy across deviated/followed cases breaks ties. Only
that recipe is refitted on ten development riders and evaluated on the five
outer test riders. The other recipes have validation results only. Do not
choose among recipes by outer test score. NLL selection does not guarantee the
best deviation hit count, so inspect deviation and followed-road results together.

All 68 original evaluation choices and 17 deviations are unchanged. GPS
exclusions are unchanged. Model architecture, features and routing are
unchanged; this introduces no synthetic rider records, SMOTE or incoming-road
features. It remains an exploratory internal study on already inspected riders.

Outputs are under `artifacts\experiments\real_training_ablation_20261021`:

- `experiment_plan.json`: predefined recipes, reference report/fold hashes,
  unchanged settings and expected fit count.
- `rider_folds.json`: a copy of the completed reference's assignments.
- `approved_training_examples.json`: the reused real corpus.
- `fold_*/stgat_lstm/selection_report.json`: all three validation trials,
  selected recipe/epoch, and confirmation that selection precedes test prediction.
- Each fold's checkpoint, training report and evaluation report: the selected
  development refit, validation curves, fitted/test metrics and support counts.
- `evaluation_report.json`: pooled selected-recipe predictions, per-fold
  results, and `reference_comparison.stgat_lstm` with the completed baseline.

Wait for report status `completed`. Compare exact deviation choices, followed
choices, overall accuracy and NLL. A higher deviation count coupled with poor
followed-road performance is still a mixed result; implementation tests do
not demonstrate a performance improvement. A repeat overwrites this new
folder, so use a separate output folder to preserve intentional repeats.

### Reverting or returning to an earlier stage

The completed fixed-split run (`real_cleaned_osm_20261020`) and grouped run
(`real_group_kfold_20261020`) remain intact. Their checkpoints and commands in
Sections 6 and 10 remain usable. Ordinary commands retain accuracy-based
checkpoint selection and the original automatic deviation weight.

Before these edits, nine code/test/document files were saved under
`artifacts\rollback\before_training_ablation_20261021\source_snapshot`.
`rollback_manifest.json` lists their original hashes, hashes of both completed
runs, model architecture and original PLAN, and the experimental stages.
`restore_instructions.txt` describes restoring only those saved relative paths.
The source snapshot restores the completed k-fold implementation. The original
pre-k-fold fixed-split workflow is preserved through Section 6 and its saved
artifacts; the snapshot is not a claim of a separate pre-k-fold source backup.
No git reset or experiment deletion is needed to return to a saved model.

## Previous runs and cleanup

Dated experiments under `artifacts\experiments` remain intact. Their metrics
used the previous amenity counts; they are not cleaned-OSM results.

The old CSV/GeoJSON moved out of the active input directory to
`artifacts\inputs\osm_before_amenity_cleanup`. Use the archived CSV when
replaying an older checkpoint:

~~~bat
python -m stgat_lstm dashboard --checkpoint artifacts\experiments\real_refined_20261018\checkpoint.pt --graphml "data\real\osm\metro_manila_processed.graphml" --node-features "artifacts\inputs\osm_before_amenity_cleanup\road_node_features.csv" --gps-traffic-data artifacts\experiments\real_navigation_20261016\source_snapshot\rider_exports --traffic-cutoff-ms 1791360905826 --max-detour-ratio 1.30
~~~

Stale standalone checkpoints, reports, predictions, feature-dependent audits,
and the former manual-review file moved to
`artifacts\archive\pre_cleaned_osm_20261020`.
Only disposable Python cache directories were deleted. Code, tests, original
plan, real labels, and dated experiment results remain.

`artifacts\layout_manifest.json` records archive paths and hashes. Historical
provenance is unchanged and still records the original training-time paths.

## Synthetic experiments and future data changes

[SYNTHETIC_RUNBOOK.md](SYNTHETIC_RUNBOOK.md) covers the separate controlled
synthetic experiment. The real-data workflow uses the real-only model above.

For future rider-export or topology changes, revisit label preparation and
comparison inputs. Section 6 deliberately reads the frozen 69-ride dataset:
rerunning Section 4 alone does not replace those frozen experiment inputs.
