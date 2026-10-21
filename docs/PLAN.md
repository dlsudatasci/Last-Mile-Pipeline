# Confirmed Scope and Current Stage

## Restored real-only model performance reassessed (2026-10-20)

- Reloaded the preserved cleaned-OSM checkpoint using the restored pipeline, rebuilt held-out-rider-excluded GPS profiles, and reproduced all 14 test predictions/probabilities and 2/3 route ranking exactly. Real data, model weights, original results, and experiment hashes are unchanged. No training or model change was performed.
- Exact test choice remains 9/14 (64.3%), below always-follow's 10/14 (71.4%). Actual deviation roads are 1/4 correct versus baseline 0/4; followed roads are 8/10 versus baseline 10/10. Choice NLL is 0.871 versus baseline 0.906. The separate binary follow/deviate score is 78.6% at the validation-selected threshold; it is not exact outgoing-road accuracy.
- Final development fitting scores 63/98 (64.3%): 12/13 alternatives and 51/85 followed choices. Alternative-road behavior fits those observed development cases but transfers poorly to held-out riders. Equal overall development/test accuracy does not establish no overfitting or adequate fit. Selection-validation 12/13 is from the earlier nine-rider selection fit, before the final twelve-rider refit.
- A fresh fitted-input audit found 89 distinct exact contexts among 98 choices and one conflicting pair of labels. This accounts for at least one unavoidable observed-row error, not the 35 current fitting errors. Its 97/98 optimistic ceiling concerns the observed fitting inputs only and is not a test-accuracy target.
- Specific deviation cases: at node 5144, the model selects Nakpil Street instead of San Marcelino Street; its four fitted choices there all selected Nakpil. At node 149, the recorded and predicted road names are both Taft Avenue, but the directed exits go in opposite directions, so the prediction is correctly counted wrong. At node 5133, Leon Guinto Street is correct. At node 5132, the model selects Remedios Street instead of the recorded Taft exit despite some fitted support for that choice.
- Direct outgoing-road speed observations are absent in all six frames for two of the four test deviations; the other two have only one frame with one observed candidate road. Nearby incoming roads add sparse context in three cases, while the correctly predicted Leon Guinto case has no observed two-hop local traffic. These facts do not prove LSTM usefulness or complete traffic irrelevance.
- Eleven of the fourteen test choices come from one rider. The three test riders contribute 1, 11, and 2 choices, with 1, 8, and 0 correct. Overall accuracy is therefore dominated by one rider; four actual deviation choices remain too few for broad reliability claims.
- Assessment: functioning undergraduate research prototype with weak evidence of held-out rider-preference prediction. The current-data GATv2-only versus ST-GAT-LSTM comparison remains missing; archived architecture scores use older inputs/preparation. No augmentation has been reintroduced.

## Recent augmentation work reverted (2026-10-20)

- At the user's request, removed the recent real-data training augmentation and synthetic robustness workflows, their generated cases, augmented checkpoint/results, and added tests. Restored the real-only training/evaluation interface and simplified the guides.
- The active completed model is `artifacts/experiments/real_cleaned_osm_20261020/checkpoint.pt`: 9/14 exact test roads (64.3%), including 1/4 deviation roads. No augmented result is an active project result.
- Preserved official rider/OSM data, the frozen rider split, real-only artifacts, model architecture, older experiments, and ORIGPLAN. No training was performed during rollback. All 98 original tests pass.

## Cleaned-OSM run completed and reviewed (2026-10-20)

- User completed RUNBOOK through Step 8; dashboard and visualization improvements are explicitly deferred. Review performed no training or model/source changes. Files are under `artifacts/experiments/real_cleaned_osm_20261020`, with route/replay outputs under `outputs/predictions`.
- Four fits completed. Validation selection retained ranking weight 1.0, epoch 45, and binary threshold 0.8295008391141891. Its validation score is 12/13; final development refit scores 63/98 fitted targets (64.3%), or 65/101 original targets (64.4%). Final test is 9/14 exact roads (64.3%), below always-follow's 10/14 (71.4%). Exact outcomes are 8/10 followed roads and 1/4 alternatives. Test ranking is 2/3 and binary F1 at 0.5 is 0.500. Thresholded follow/deviate accuracy is 11/14 (78.6%), not exact road-choice accuracy.
- Test choice NLL improves from the previous first-visit run's 0.9093 to 0.8710; exact accuracy is unchanged. Development NLL is 0.8482, with zero accuracy gap and a 0.0228 test-minus-development NLL gap. These gaps do not prove absence of overfitting or adequate fit. There are still only three previously inspected test riders and fourteen test choices.
- Current cleaned CSV, graph, approved examples, all six frozen/current source CSVs, fitting source hashes, original rider assignments, and all test targets/candidate sets are verified. Architecture remains 7,441 parameters. Reload reproduces all fourteen predicted roads and probabilities exactly, and ranking matches. No software changes required another test run; the unchanged source previously passed all 98 tests.
- Both Step 8 routes reproduce from the checkpoint and profiles, use connected accessible edges, and measure 1,365.75 m versus 1,320.53 m shortest distance: 3.43% longer, within the 1.30 guard without fallback. Current/historical traffic frames have observed graph-edge counts 0/14/22/23/35/5 and 10/9/0/19/16/19, respectively; direct coverage on the recommended route remains sparse. Route demos use all available profiles and are operational checks, separate from test-rider-excluded evaluation. Near-one whole-route deviation aggregates are uncalibrated, length-sensitive summaries, not reliable actual-deviation probabilities.
- Saved `run_summary.json`, `checkpoint_reload_check.json`, and `run_manifest.json` for the reviewed run. Raw reports/checkpoint are unchanged. Older entries below record the earlier pre-run state. Real-plus-simulated augmentation remains unimplemented.

## Cleaned OSM features and working-file cleanup (2026-10-20)

- The new CSV/GeoJSON retain all 36,375 graph nodes, schemas, and coordinates. Amenity values decrease at 5,034 nodes (5,609 cells); graph topology and street counts remain unchanged. Both formats agree, and model tensors remain finite with 13 node features. Existing checkpoints match the old CSV, not the cleaned one. No training has been run using the cleaned features.
- User will run training/evaluation. RUNBOOK now targets `artifacts/experiments/real_cleaned_osm_20261020`, keeping the saved rider split, frozen GPS exports, approved labels, first-visit correction, architecture, and existing regularization/search procedure. It uses three validation trials and a final refit. The old two-fit `--reuse-selection` mode requires identical hashes and must not be used for changed node features. Selected weight/epoch may change, so this is a newly validation-selected baseline rather than a fixed-setting amenity ablation. No compatibility guard was weakened.
- Archived the two old OSM feature files under `artifacts/inputs/osm_before_amenity_cleanup`. Archived 13 stale standalone checkpoints/reports/predictions, feature-dependent audits, and the former manual-review file under `artifacts/archive/pre_cleaned_osm_20261020`. All 15 moved files retain their contents. Deleted only the disposable `stgat_lstm/__pycache__` and `tests/__pycache__` directories. Source code, tests, approved labels, synthetic records, dated experiment bundles, and ORIGPLAN remain.
- `artifacts/layout_manifest.json` records this follow-up cleanup and protected-file hashes. Current code/guide paths were updated; old experiment provenance remains historical. Real-plus-simulated augmentation is still a proposal and has not been implemented.

## Folder organization (2026-10-19)

- User requested clearer separation of Python code, generated files, outputs, and artifacts. Python modules now live in `stgat_lstm/preprocessing`, `modeling`, `routing`, and `generation`. The `python -m stgat_lstm` command names are unchanged; imports, tests, source-hash collection, and dashboard template/root resolution use the new locations.
- `data/real` is unchanged. Prepared real/synthetic data and collected traffic are under `generated`. Standalone checkpoints are under `artifacts/checkpoints`. Dated real experiment bundles moved intact to `artifacts/experiments`; their content hashes, historical paths, manifests, and input snapshots are preserved. Routine audits, reports, predictions, and maps are grouped under `outputs`.
- `artifacts/layout_manifest.json` records each old/new file location and preservation hashes. Working guides and command paths are updated; `ORIGPLAN.md` remains the original copy. No model architecture, labels, training settings, or metrics were changed for this organization task.
- The user's local first-visit run has produced its checkpoint and completed evaluation report. Those files moved intact to `artifacts/experiments/real_first_visit_20261018`; this organization task does not interpret or promote its scores. Older entries below record the previous agent interruption and results known at that time.
- Verification completed: 98 tests pass; all 14 CLI commands import and show help; the dashboard resolves its project root and HTML template; relocated real/refined, first-visit, and synthetic checkpoints load strictly. A saved real model generates a route on the official OSM graph from its new location. All moved non-source files, official inputs, original plan, and archived manifest artifact hashes are unchanged. `model.py` itself is byte-for-byte identical to its pre-move version.

## Approved first-visit supervision comparison (2026-10-18)

- The user approved the proposed training-only correction. `first_junction_visits` in `train_model.py` retains the earliest supervised visit to a source junction within each decision example. It keeps cross-example/rider preferences separate and refuses to silently drop a later intentional alternative. It does not mutate raw exports, approved artifacts, graph features, ranking paths, or validation/test target sets.
- `evaluate --refine --first-visit-only --reuse-selection` supports the controlled comparison in the existing evaluation module. Only the original nine training riders receive this adjustment, both during the validation check and during the final development refit. Selection-training targets change from 88 to 85; all 65 examples and ten alternatives remain. Final fitted development targets change from 101 to 98 across the same 71 examples and 12 riders. Validation/test remain 13/14 targets.
- Reuse the prior validation-selected ranking weight 1.0, epoch count 45, batch size 8, seed 17, six five-minute frames, learning rate 0.001, AdamW weight decay 0.0001, smoothing 0.05, observation masking 0.15, and ride/class weights. No new parameter search is performed. The first fit records validation curves but does not restore an earlier checkpoint or stop early; the second fit uses all 12 development riders for the same 45 epochs. The previous binary threshold is also fixed.
- Save the new experiment under `artifacts/experiments/real_first_visit_20261018`. The original `artifacts/experiments/real_refined_20261018` artifacts stay frozen. Compare both the fitted 98-target development set and the original 101-target development set, so omitting three targets cannot be mistaken for an accuracy improvement on an identical training set. Held-out comparison uses exactly the same 14 targets and candidate sets.
- All 98 tests pass, including first-visit retention, no loss of alternative targets, validation monitoring without checkpoint selection, and an end-to-end locked two-fit comparison that leaves validation/test repeats intact. At the user's request, the agent stopped the fit and waiting verification process so the user can run it in their own terminal. The first fit last reported epoch 20/45; no replacement checkpoint or evaluation report was produced. Restart the complete command in RUNBOOK Section 6's first-visit subsection; preprocessing audits do not need repeating for these frozen inputs. The last completed held-out score remains 9/14 (64.3%). Do not add other accuracy changes without user approval.

## Training-input audit and fitting check (2026-10-18)

- The user approved the recommended training-data conflict audit and small fitting check. No further production tuning, feature changes, or label-policy changes were applied. Official refined artifacts and all raw CSVs remain unchanged; the held-out accuracy is still 9/14 (64.3%).
- Added `evaluate --diagnose-fit` in the existing evaluation module, with five meaningful audit/workflow tests. All 94 tests pass. This mode uses only the original nine training riders and excludes all six validation/test riders, the target ride, and future GPS records from traffic inputs. It does not compute held-out predictions or save a replacement checkpoint.
- The audit compares exact relevant tensor values for the current two-layer GATv2 receptive field of all candidate endpoints, each candidate's destination features, and all six temporal frames. Changes on disconnected roads are excluded from the comparison and tested against actual model outputs. Repeated junction/destination metadata alone is not treated as an exact conflict.
- The 65 training examples contain 88 choices in 76 distinct exact-input groups. Four groups contain opposing chosen roads (eight targets). Three are repeated visits within three original followed examples: `467d9c2669f96dad`, `daf42f82e79cc3c4`, and `83ff78a1461545eb`. Their matched windows contain an out-and-back sequence followed by a different exit. The fourth is across a deviation and followed example from two riders at source `5133`, with destination `5203`. Matched backtracking alone does not establish a real rider U-turn or a false report.
- These exact conflicts imply an optimistic observed training ceiling of 84/88 (95.45%) for the present inputs. This is neither a generalization estimate nor an explanation for the entire 66.3% development fit. Valid preference differences are retained; no conflicting row was automatically rejected.
- A deterministic balanced subset uses eight distinct unanimous training contexts from eight riders, four follows and four alternatives. One 80-epoch fit uses unchanged GATv2-LSTM, six frames, Adam learning rate 0.005, choice loss only, uniform target weights, and no smoothing/masking/weight decay. It learns 8/8, with clean choice NLL 0.000155. This deliberately memorized training subset supports basic fitting capacity and does not prove generalization, absence of overfitting, or a better final model.
- The report is `artifacts/experiments/real_fit_diagnostics_20261018/report.json`, including the exact conflicts, selected subset, configuration, source hashes, fit results, and the proposal as recorded before approval. The user subsequently approved keeping only the earliest supervised junction visit within a training example: three later duplicate-context targets are omitted (88 to 85 choices), while all 65 examples, ten alternatives, and 47 added followed examples remain. Cross-example preferences and validation/test targets stay intact. Implementation and the pending comparison are recorded above; the archived diagnostic report is unchanged.

## Approved real-data refinement (2026-10-18)

- User authorized the proposed destination correction, additional genuine training choices, validation checkpoint selection, modest regularization, conservative augmentation, and final refit on all 12 development riders. Additional improvements require their approval first.
- `refine_real_decisions` preserves all 30 original approved example keys and road labels. Four followed examples have a different actual route destination from their local review endpoint. It recovers 47 additional followed choices from training riders only, yielding 77 approved examples from 36 rides and the same 15 riders (60 follows, 17 deviations). Source CSVs are unchanged; no intentional deviations are fabricated.
- Original rider assignments are reused exactly. Selection has 65 training examples from 25 rides, six validation examples from five rides, and six test examples from six rides. Only training records receive additional labels; validation/test destinations are corrected consistently while keys and road labels stay fixed.
- Each extra choice requires a matched three-edge continuation, direct consecutive GPS support for the chosen edge, a prior GPS anchor, mean local match error at most 20 m, no reported event in the local label window, and a known active route. Repeated ride/junction choices are skipped. Twenty-six active-route periods did not sequence-match and produced no extra labels.
- Three predefined ST-GAT-LSTM trials use ranking weights 0/0.1/1, AdamW learning rate 0.001 and weight decay 0.0001, label smoothing 0.05, training-only observation masking 0.15, and reduced influence for rides with many labels. Class weights account for ride weights. Validation checks occur every five epochs; patience is three checks, with a 60-epoch ceiling. The highest choice accuracy is retained, with lower NLL breaking epoch ties.
- Trial selection uses validation choice accuracy, deviation F1, then NLL. A final refit merges the 12 development riders at the selected fixed epoch count; the original three test riders stay excluded. Histories exclude all validation/test riders during selection and only test riders after development refit, always excluding target ride/future observations.
- The refinement fits started with backbone source checksum `f0bbd452e340d39f08cc539c52c074fb4a242484a73347ecb68c01250ec3de50`. No new architecture, traffic-window policy, or invented speed values were added. All 86 tests passed before fitting.
- The user subsequently approved reusing exactly identical spatial frames within one forward call. `model.py` now computes each distinct traffic frame once, while retaining all six LSTM steps. The cache never persists between calls and skips frames that require input gradients. Training/evaluation outputs and parameter gradients match the original computation within floating-point tolerance. All 89 tests pass. This is a speed optimization, not an accuracy improvement; the already-running process retains its original loaded implementation.
- Completed artifacts: `artifacts/experiments/real_refined_20261018`. Do not replace the older fixed-split report with these results; goal contexts and development training counts differ.
- At the user's request for proper ML/overfitting checks, reporting includes retained-trial training/validation accuracy and validation curves, plus clean final development/test accuracy and NLL. Validation riders are explicitly training data after refit. The fits were unchanged; a post-fit audit added these fields and verified that checkpoint reload reproduces all 14 held-out predictions with zero probability difference. Weighted/smoothed optimization loss is not directly compared with clean NLL. All 89 tests pass, including the reporting assertions. This remains one seed and three previously inspected test riders, not proof that overfitting is absent.
- Three selection fits executed 40/45/60 epochs and retained epochs 25/30/45. Ranking weight 1.0 won with 12/13 validation roads correct. The final refit uses 71 examples, 101 choices, and 12 development riders for 45 epochs. It scores 67/101 (66.3%) on clean development and 9/14 (64.3%) on test, below the always-follow baseline's 10/14 (71.4%). Only one of four actual alternative roads is chosen correctly; binary F1 0.500 counts two detected deviations, including one wrong alternative.
- Test NLL is 0.911 versus the current baseline's 0.911 and the older model's 3.373; test ranking improves from 1/3 to 2/3. Development/test accuracy gap is 2.05 percentage points and NLL gap is 0.099. Per-rider test results are 1/1, 8/11, 0/2. The 80% target is not met; low development accuracy and a small gap do not establish a cause or rule out overfitting. No test-based follow-up tuning was performed. The same 14 test targets/candidate sets and all six official CSV hashes are verified unchanged.
- Checkpoint reload, finite parameters, access flags, and connected routing pass. The historical 400-to-6 learned route is 1,365.8 m versus 1,320.5 m shortest distance, within the 1.30 guard without fallback. This verifies operation, not ground-truth route quality. `run_summary.json` and `run_manifest.json` record results and input/fitting/post-fit source hashes. No new architecture comparison was run under the refined preparation.

## Real fixed rider split (2026-10-18)

- Added a real-data `--fixed-split` option to `evaluate_model.py`; the older real workflow only supported leave-one-rider-out, while fixed splits were synthetic-only. The recommended runbook now uses the smaller real fixed split.
- Freeze seed 17 and approved-data hash before fitting: nine training riders, three validation riders, three test riders; 18/6/6 examples and 41/13/14 junction choices. Assignments are saved in `artifacts/experiments/real_fixed_split_20261018/rider_split.json` and must not be regenerated to improve test scores.
- Fit GCN, GAT, GATv2, ST-GAT-LSTM, and latest-only LSTM once each, 100 epochs, learning rate 0.005, batch 8, preference weight 1.0. Exclude all validation/test riders, target ride, and future records from historical GPS features. Keep the full-data deployment checkpoint separate.
- The approved artifact is frozen in the experiment directory. Reuse the six existing frozen CSVs under `artifacts/experiments/real_navigation_20261016/source_snapshot/rider_exports`; all six hashes match the current official exports. The new manifest records these inputs and source hashes. All 81 tests passed before the comparison.
- A new current-time route had no eligible observations in any of its six historical profile slots. This is unknown traffic, not free-flow traffic. The saved daytime historical replay has observed-road counts 10/9/0/19/13/16, demonstrating that profile coverage varies with request time.
- Completed all five fits. ST-GAT-LSTM training/validation/test choice accuracy is 0.878/0.923/0.714; test balanced accuracy 0.650, deviation F1 0.500. It gets 8/10 follows and 2/4 alternative roads correct. The always-follow baseline ties at 10/14 choices but misses all alternatives. Latest-only gets 9/14 and F1 0.286; the one-choice improvement does not establish a stable temporal benefit.
- Probability quality and route ranking remain weak: ST-GAT-LSTM test choice NLL 3.373 versus baseline 0.830, indicating confident errors; only 1/3 test ranking pairs is correct. Per-rider choice counts are 1/1, 9/11, and 0/2. No broad reliability claim is supported. Do not tune settings or regenerate the split against these test scores.
- Observed history is available at 16/18 training, 1/6 validation, and 5/6 test examples; changing values occur on suggested paths at only 2/18, 0/6, and 1/6. The full report, concise summary, saved split, approved snapshot, and completed manifest are under `artifacts/experiments/real_fixed_split_20261018`. Frozen input/source hashes, group separation, exclusions, finite values, and counts were verified after the run; all 81 tests passed before fitting.

## Current-data checkpoint completed (2026-10-16)

- The user accepted the current dataset for this experiment while allowing future updates. Keep the 30-example navigation-assisted artifact fixed for this run; future exports require rebuilding labels and a separate comparison.
- Six rider CSVs were copied to `artifacts/experiments/real_navigation_20261016/source_snapshot/rider_exports`. The run manifest records their hashes, row counts, the approved artifact, OSM/feature hashes, source-code hashes, and training/evaluation settings. Both training and evaluation use the frozen GPS exports.
- Full-data ST-GAT-LSTM training: 100 epochs, batch size 8, learning rate 0.005, preference weight 1.0, inverse-frequency alternative-choice weight, seed 17, six five-minute historical frames.
- Held-out comparison: 40 epochs per model per leave-one-rider-out fold, 15 riders, GCN/GAT/GATv2/ST-GAT-LSTM and a latest-only control where histories vary. These comparative results do not directly evaluate the full-data 100-epoch checkpoint. No settings are selected using the held-out results.
- Full-data training completed 100 epochs: loss 1.8917 to 0.2863, 62/68 in-sample road choices correct (91.2%), and 11/12 in-sample ranking pairs correct. These are training-fit results, not unseen-rider accuracy. The 7,441-parameter checkpoint is `artifacts/experiments/real_navigation_20261016/gatv2_lstm_checkpoint.pt`.
- Checkpoint reload passed three prediction commands. A paired replay favors Nakpil Street and the observed path (cost 5.557 versus 23.660); the inferred replay favors Castro Street. A fresh historical-context request from 400 to 6 produces a connected 1,503.1 m learned route versus a 1,320.5 m distance baseline, within the 1.30 guard without fallback. Saturated probabilities and the whole-route aggregate score are uncalibrated and do not establish route quality.
- The user questioned the comparison workload, so the 75-fit evaluation was stopped before completion; no new held-out report was produced. Explain and agree on evaluation scope before restarting it. Earlier archived evaluation used ten riders and five variants (50 fits); 75 is 15 riders times five variants, not 75 deployed architectures. This benchmark is separate from training the single deployment checkpoint.
- Source code and all frozen input hashes remained unchanged during training. `run_manifest.json` records the state and `run_summary.json` contains training, replays, routing, and the incomplete evaluation status.

## Recorded-navigation inference implemented (2026-10-16)

- Current artifact: `artifacts/experiments/real_navigation_20261016/approved_training_examples.json`. It contains 30 approved examples from 28 rides and all 15 riders: 13 followed examples and 17 deviations. The deviations cover 15 rides and nine riders. Training preparation produces 68 junction targets (51 followed and 17 alternative roads); 12 deviations provide ranking pairs and five train choice only.
- The builder preserves the recorded instruction fields on every deviation candidate. After sequence/late-report extraction fails, it uses road-name aliases or compass heading, event location, active-route outgoing edges, and consecutive raw GPS movement to resolve a unique alternative. This does not require an exact incoming HMM edge. Upcoming original maneuver names are not substituted for immediate expected outgoing edges; geometry determines those.
- Navigation recovery adds one event: Fidel A. Reyes Street was suggested and Castro Street chosen. It uses a pre-turn cutoff and is tagged `recorded_navigation_and_local_geometry`, with inference evidence and source strings retained. All 29 previously approved examples preserve their target, road paths, and cutoff.
- The fallback requires directed GPS support, accessible roads at one branching source, spatial separation from the expected road, and one supported candidate. Heading alone does not override contradictory geometry or resolve two matching roads by iteration order. The three original GPS-sequence failures remain excluded. Of 37 reported deviations, 17 are retained, 16 lack a resolved choice after both methods, three fail original matching, and one is a personal stop.
- Automatic review checks inference provenance, unique resolution, matching edge IDs and cutoff; its rules version is 3. Recorded navigation and later observations are label evidence only, excluded from inference features. Synthetic templates remove copied real navigation context and continue generating their own records.
- All 78 tests pass, including aliases, heading-only resolution, ambiguous alternatives, conflicts, reversed movement, late-report recovery, integer GraphML keys, provenance validation, and end-to-end inferred-label approval. All 30 examples prepare for training with GPS histories. Source CSV hashes are unchanged. `navigation_report.json` records this label-preparation stage; the later checkpoint training is recorded above. Earlier stages below remain historical records.

## Deviation-schema clarification (2026-10-16)

- The user correctly pointed out existing deviation road-context fields. Direct inspection confirms all 37 source events populate `originalRouteEdge`, `deviatedEdge`, `generatedInstruction`, and `deviationInstruction`; 35 populate `streetName`. Before this correction, the real extractor did not read these text fields when identifying the chosen outgoing road. Earlier claims about missing evidence described the GPS/route-only extractor, not an exhaustive use of the dataset.
- All 37 `originalRouteEdge` values equal `generatedInstruction`, and all 37 `deviatedEdge` values equal `deviationInstruction`. They are navigation strings, not OSM `u|v|key` identifiers. Examples include a turn onto Quirino Avenue in 330 m versus driving on Nakpil Street; other records contain destination or heading-only messages. A future maneuver's named road is not automatically the rejected immediate outgoing edge at the observed branch.
- Corrective direction: preserve and use these recorded fields as label evidence alongside event location, route versions, geometry, GPS, and survey intent. Resolve which directed OSM roads they identify rather than treating exact shared-approach GPS as the only evidence. Do not declare the remaining events unsalvageable or approve every text pair without resolving its meaning. No label counts or trained metrics were changed by this schema inspection.

## Late-report salvage (2026-10-15, supersedes the first local-choice trial)

- Current training artifact: `artifacts/experiments/real_salvage_20261015/approved_training_examples.json`. It has 29 approved examples from 27 rides and all 15 source riders: 13 followed examples and 16 deviations. These produce 67 junction targets (51 suggested and 16 alternative roads), with 12 optional route-ranking pairs and four choice-only deviations. The deviations cover 14 rides and nine riders. All ten previously retained positive events remain present.
- Missing GPS at/after the report no longer rejects an event automatically. The two such source events were examined; relaxing that requirement alone did not produce additional supported choices.
- The first window still starts 60 seconds before the report. If it contains no supported choice, recovery looks back to the latest prior route's generation time or the previous deviation, whichever is later. This recovered six turns reported 70.183 to 309.324 seconds after their first alternative GPS fix. It preserves the cutoff before the observed turn and does not treat report time as turn time.
- Source CSVs remain unchanged. Three original GPS-sequence failures stay excluded. The remaining 17 local-evidence failures comprise: two longer-track matching failures, five cases with no shared approach identifying the departure junction, six with no distinct alternative departure identifiable at shared approaches, and four whose candidate turn approach occurs after the report. These are diagnostic outcomes under current assumptions, not proof of false reports. Counts include the two events formerly excluded for missing post-report GPS.
- All 73 tests pass, including late-report retention, active-route time bounds, and preservation of pre-choice cutoff. Every approved example loads and prepares with historical GPS traffic. Full training and held-out evaluation have not been rerun. See `artifacts/experiments/real_salvage_20261015/salvage_report.json`; older counts below are archived stages.
- Explanation of the remaining groups: the five no-shared-approach cases do have GPS. In their selected recovery windows, four have zero fixes matched to the selected prior suggestion and one has four such fixes, but none establishes the required common incoming-road transition. This is a comparison/matching/route-association outcome, not a finding that the trips have no GPS.
- Of the six no-distinct-alternative cases, three windows match the selected prior suggestion at every GPS fix (23/23, 17/17, and 72/72). The other three have 52/64, 51/52, and 74/90 fixes on that suggestion, but the extractor cannot link the off-suggestion fixes to an eligible alternative departure at a shared approach. These diagnostic patterns do not establish why the source app reported a deviation; route association and matching remain possible explanations.
- The four timing cases fail the pre-report approach ordering check. Their grouping identifies a timing inconsistency in the candidate choice, not proof that the rider made no deviation. A subsequent candidate turn may need comparison against a regenerated active route rather than the original event's suggestion. Do not force that later turn into the earlier event label.

## Local-choice recovery trial (2026-10-15)

- Current approved set: 23 examples from 13 riders, with 13 followed and ten deviated examples. They produce 61 junction targets (51 suggested and ten alternative roads) and seven optional ranking pairs.
- A local label requires GPS evidence of the chosen outgoing road at the same source as the suggestion. Rejoining and matching the regenerated route are not mandatory. Three deviations train through the choice objective alone.
- Fifteen of 18 strict-transition failures rematch using six candidates and 8 m same-edge backward jitter per transition; one of those yields a supported local label. Other recovered labels came from removing regenerated-route/rejoin gates. The source CSVs are unchanged.
- The model graph expands to a 1.7 km Taft buffer to cover approved roads and destinations. Checkpoints save that scope. All 71 tests pass; a three-epoch training and reload check passed.
- Artifacts: `artifacts/experiments/real_local_choices_20261015`. Full training and rider-held-out evaluation of these revised labels remain pending. Counts and metrics below dated before this revision are archived baseline evidence.
- Counting units: the source export has 15 riders, 69 rides, 37 deviation events across 25 rides, 124 route versions, 7,826 GPS points, 37 deviation responses, and 69 post-trip questionnaires. The 23 approved examples cover 22 unique rides and 13 riders. One ride contributes two different deviation-event examples; this is not duplication of one event. The 13 followed examples cover 13 rides and seven riders; the ten deviation examples cover nine rides and seven riders, with one rider shared between groups. These yield 51 followed junction targets plus ten alternative-road targets. Targets can share roads and rides and are not independent samples.
- Local-choice evidence rules currently require an observed approach consistent with the latest pre-event suggestion; a distinct outgoing alternative from the same branching source; at least two consecutive GPS fixes on the alternative continuation before any rejoin; and a first alternative fix between 60 seconds before and 15 seconds after the reported event. The approach fix defines the input cutoff. These are implementation choices, not proof that excluded reported deviations are false.
- The 21-event `no_directly_supported_outgoing_choice` outcome combines failures of those local rules. Current saved diagnostics do not identify the individual failed rule for each of those 21 events. Do not attribute all of them to bad GPS or claim that none can be recovered. Two other events fail the before/after GPS-window anchor check; three fail GPS/prior-route sequence matching, and one is excluded for a personal stop.
- Detailed follow-up: the two events currently labeled `no_pre_event_gps_anchor` actually have 23 and 17 GPS points before their report, respectively, but no GPS point at or after it anywhere in that ride. Their nearest preceding fixes are 6.261 s and 15.3 s before the report. The bucket name is misleading: these are missing post-report GPS, not missing pre-event GPS. A late report may still have pre-report evidence of an actual turn, so their usability needs examination rather than automatic dismissal. All three remaining sequence-match failures are GPS transitions (observations 13, 1, and 2); the prior suggested routes match successfully. Neither finding proves physically bad riding or invalid source rides.

## Windows coordinate-library compatibility (2026-10-08)

- Windows Application Control began rejecting the unsigned `libcurl` DLL bundled with PyProj, which prevented all graph-dependent commands from importing.
- The project now uses a local WGS84/UTM transformer for the official EPSG:32651 Manila graph, and PyProj has been removed from `requirements.txt`.
- Map matching, GPS traffic construction, GeoJSON conversion, coordinate snapping, and OSM audits use the same internal transformer.
- The complete suite passes 59 tests, including Manila coordinate round-trip tests, and all three official data audits run successfully.

## Outgoing-road choice revision (2026-10-08)

- The primary prediction changed from an independent binary deviation score to the preferred accessible outgoing road at each branching source.
- `PreparedRoadChoice` records the candidate outgoing edges, rider-chosen edge, and suggested edge. Follow/deviate is derived from whether the chosen and suggested edges match.
- Training uses class-balanced multiclass cross-entropy over each junction's candidates. Alternative-road targets receive `followed choices / alternative choices` loss weight. The intentional route-ranking loss remains as auxiliary supervision.
- GATv2-LSTM choice logits are normalized per source node. Routing costs are `-log P(edge | source, context)` plus a tiny positive length tie-breaker, so explanation and routing use the same learned output.
- Prediction JSON and terminal output name the preferred outgoing road and list all candidate probabilities. Deviation remains available as `1 - P(suggested road)`.
- Evaluation treats outgoing-road accuracy and negative log-likelihood as the primary metrics. Binary follow/deviate measures remain secondary derived metrics.
- The checkpoint schema is now `outgoing_road_choice_preference`; older checkpoints must be retrained.

- Primary objective: train STGAT-LSTM to learn shared motorcycle-rider route preferences/tacit knowledge, then let a routing algorithm use the trained outputs to recommend routes.
- Improving travel time is not the goal; any time saving is a bonus. Evaluate learned rider preferences and feasible routing first. Travel time may be reported as a secondary outcome, not a required optimization target or prediction head.
- Working collection area: Taft Avenue and surrounding roads. On 15 October 2026 the user removed the 500 m eligibility limit. Candidate builders now consider all records in the supplied OSM graph. The model graph starts with a 1 km Taft extraction and expands to cover approved paths and destinations before training.
- Rebuilding without the corridor cutoff screens 36 eligible events, retains three strict deviation pairs, and produces 17 decision candidates with 16 automatic approvals. The previously outside event fails the rejoin check instead. No additional approved road choices are recovered by removing geography alone. Results are under `artifacts/experiments/real_no_corridor_20261015`.
- The mobile app is for collection. Riders currently use an older build; its precise version and whether every rider used the same build still need to be confirmed.
- Preserve original observations. Mark unknown or inconsistent fields and any reconstructed values; do not present dummy values as real rider evidence.
- Current stage: the executable STGAT-LSTM, GAT/GCN baselines, learned-cost routing, official real-export audit, route/GPS geometry screening, OSM compatibility/coverage audit, directed HMM/Viterbi map matching, conservative common-OD candidate extraction, automatic survey/GPS candidate validation, the OSM-to-model feature contract, GPS-derived temporal road-speed profiles, a controlled synthetic Taft hotspot experiment, and the learned-cost routing adapter are implemented.
- See the root `README.md` for the project overview, `RUNBOOK.md` for real-data commands, `SYNTHETIC_RUNBOOK.md` for the controlled experiment, `MODEL_EXPLANATION.md` for the technical walkthrough, and `PRESENTATION_GUIDE.md` for the defense/demo material.
- These confirmed points supersede tentative objectives or outputs suggested below.

## Official dataset snapshot and archived strict-baseline results

- The user designated the export supplied on 14 October 2026 as final. It contains 69 rides from 15 riders, 124 saved routes, 37 reported deviations with responses, 7,826 GPS points, and 69 completed post-trip questionnaires.
- Thirty-five events pass the provisional corridor/reason screen. The preliminary sequence audit matches 23 triplets and screens 12 candidates. The stricter event builder uses a different GPS window and retains three graph-valid divergence/rejoin pairs.
- The builder produces 17 examples. Automatic validation approves 16 local route-choice examples from ten riders: 13 followed and three deviated. These yield 54 supervised outgoing-road choices: 51 suggested roads chosen and three observed alternatives chosen. One followed example is rejected because its mean GPS-to-road matching distance exceeds 20 m.
- This is enough to demonstrate the complete pipeline and in-sample learning. It is not enough to support a reliable rider-generalization or state-of-the-art performance claim. The rider-disjoint results remain preliminary and weak.
- The GPS track has a 2.001-second median sampling interval. The final-export pipeline derives 6,801 usable motorcycle speed segments on 161 directed OSM roads from all 69 rides. Complete-data training has some observed history at all 16 approved examples; the held-out evaluator rebuilds profiles with additional rider exclusions. Final-run artifacts are stored under `artifacts/experiments/real_final_20261014`.
- The supplied matching notebook uses six candidates and an 8 m backward-jitter tolerance and saves matches in memory. Its later trimming and manual dispositions are obsolete at the user's instruction and are not applied. The final-export baseline run uses the existing project matcher; the notebook's matching results are not embedded in the CSVs.
- The completed ten-rider held-out evaluation gives ST-GAT-LSTM 0.556 outgoing-choice accuracy, 0.745 balanced accuracy, 0.188 deviation F1, and 3/3 route pairs ranked correctly. It detects all three deviations but gives 26 false positives. Spatial GATv2 has 0.667 choice accuracy and 0.240 deviation F1. These results are weak; an overall real-data LSTM benefit is not established.
- Full-data training completes 100 epochs with loss 2.0788 to 0.2040, 43/54 in-sample road choices correct, and 3/3 training rankings correct. Checkpoint reload and reviewed-example prediction succeed. Both tested fresh routes trigger the 1.30 detour fallback, so learned route quality remains a limitation. The complete run record is `artifacts/experiments/real_final_20261014/run_summary.json`.
- The controlled synthetic experiment contains 144 labeled decisions from 12 artificial riders: 72 deviations and 72 follows at two graph-valid local Taft hotspots. Another 144 traffic-only probe rides provide six complete five-minute frames. Worsening and recovering histories end at the same latest speed, so the full-history model cannot solve the temporal control from the latest frame alone. All app-shaped decisions go through production reconstruction and automatic validation, while controlled training uses the exact synthetic ground-truth artifact to keep followed and deviated origin-destination pairs identical.
- The fixed rider-disjoint synthetic split uses eight riders for training, two for validation-only class-weight selection, and two new riders for one final test. Validation selected a 1.0x alternative-choice weight. On the untouched test riders, ST-GAT-LSTM obtains 0.846 outgoing-choice accuracy, 0.955 balanced accuracy, 0.800 deviation F1, 0.955 ROC-AUC, 0.833 average precision, and 1.000 route-ranking accuracy. It outperforms GATv2 and latest-only, which each obtain 0.769 choice accuracy. The 0.846 prevalence accuracy comes from always following and has zero deviation F1. This is controlled implementation evidence, not real-rider evidence.

## Collection recommendations for future extensions

The present export is final for this experiment. These recommendations describe
how a future collection could address its limitations; they are not additional
collection steps required to run the final dataset.

1. Keep complete trips inside the agreed Taft corridor and repeat origin-destination patterns that contain real alternative branches.
2. Record the exact route ID or directed edge sequence shown immediately before each road decision. The current export links deviation records to routes generated after the event, so the pipeline reconstructs the latest pre-event route.
3. Keep continuous timestamped GPS near decisions, include GPS accuracy, and investigate gaps longer than 60 seconds.
4. Require an immediate deviation response for each event and either a completed post-trip questionnaire or an explicit skip state for each ride.
5. Keep wrong turns, personal stops, navigation errors, and intentional preferred-road deviations as separate purposes. Do not reward every deviation as rider knowledge.
6. Collect more independent intentional deviations from more riders and conditions. Repeated follow labels from the same riders cannot replace positive deviation choices.
7. Stop collection based on rider-disjoint learning curves and uncertainty, rather than a universal raw-row target.

Still to clarify before the final experiment freeze: the exact deployed app build; the final corridor boundary and allowed excursions; the target rider/ride count and collection duration; the final GPS temporal-history window after coverage analysis; whether the default 1.30 maximum-detour ratio should be changed; thesis success criteria; participant consent, retention, and sharing rules; and whether the final study requires a live rider trial.

If collection ends with only a few independent positive deviations, report STGAT-LSTM as a pipeline proof of concept, disclose the rider-disjoint results, and treat the strongest simpler baseline as the defensible measured model. Class weights, extra epochs, or duplicated records cannot substitute for independent rider decisions.

# Code Implementation
- Using Python. Tech Stack: Whatever the standard is, plus anything else required by the project.
- Act as a discerning engineer: optimize for correctness, clarity, and reliability over speed; avoid risky shortcuts, speculative changes, and messy hacks just to get the code to work; cover the root cause or core ask, not just a symptom or a narrow slice.
- Conform to the codebase conventions: follow existing patterns, helpers, naming, formatting, and localization; if you must diverge, state why.
- Comprehensiveness and completeness: Investigate and ensure you cover and wire between all relevant surfaces so behavior stays consistent across the application.
- Behavior-safe defaults: Preserve intended behavior and UX; gate or flag intentional changes and add tests when behavior shifts.
- Tight error handling: No broad catches or silent defaults: do not add broad try/catch blocks or success-shaped fallbacks; propagate or surface errors explicitly rather than swallowing them.
- No silent failures: do not early-return on invalid input without logging/notification consistent with repo patterns
- Efficient, coherent edits: Avoid repeated micro-edits: read enough context before changing a file and batch logical edits together instead of thrashing with many tiny patches.
- Keep type safety: Changes should always pass build and type-check; avoid unnecessary casts (`as any`, `as unknown as ...`); prefer proper types and guards, and reuse existing helpers (e.g., normalizing identifiers) instead of type-asserting.
- Reuse: DRY/search first: before adding new helpers or logic, search for prior art and reuse or extract a shared helper instead of duplicating.

# Architecture, Boundaries & Execution Workflow
- Do not stop at a partial fix; carry changes through implementation and verification.
- Every code change must run matching tests via `exec_command`.
- Goal condition: 100% pass rate on affected test suites.
- If an algorithm is being closely adapted from a specific research paper or open-source module, include a clean markdown comment block at the top of the newly planned file referencing the source DOI or URL for engineering transparency.

# Overview Context
- Focusing on Metro Manila
- We are developing a spatiotemporal GAT-LSTM neural network integrated with a dynamic routing algorithm. We need to use state-of-the-art models and routing algorithms, and piggyback off them further.
- Basically, we have a mobile app that collects driver deviation data from a generated route. These data include, but are not limited to, where they deviated, the time, and the reason for the deviation. They will be asked to complete a short survey. If traffic was the reason, what was the severity, etc? Our high-level goal is for GAT to learn about riders' tacit knowledge, or street smarts, which we collect from them. Afterward, use/create a routing algorithm.
- `data/real/rider_exports/` contains the official final rider export designated by the user on 14 October 2026.
- OSM Road Network and data that is not limited to the type of amenities and their counts across nodes.Not limited to the mentioned, but other details a route-generating app should collect.
- `data/real/osm/` contains the project-local OSM GraphML and node-feature table used directly by the model pipeline. Their compatibility and feature join are checked by the network audit command.
- I believe what we are trying to do must include both node- and edge-level predictions, right? Which then will have Sub-graph and Structural embeddings, and Graph Level as well? You may clarify with me. What properties would the nodes and edges have?
- There are a lot of nitty-gritty details for GAT, so please do not hesitate to consult with me and suggest as well for the specific details

# Git publishing policy, 21 October 2026

- User chose to exclude real rider data from future commits only. Removed
  rider export CSVs and previously tracked generated outputs from the index
  using `git rm --cached`; no local dataset files were deleted. Earlier Git
  history is intentionally unchanged; no history rewrite or force-push.
- Ignore `data/real/rider_exports`, private `artifacts/experiments` (including
  frozen CSV copies and derived per-rider records), generated outputs, model
  binaries, archives, rollback copies, environments, credentials and caches.
  Source, tests, guides and public OSM inputs remain eligible for versioning.
- Verified private paths are absent from the index and untracked additions,
  while active source/docs/OSM remain eligible. Official source CSV hashes
  are unchanged. Fresh clones require restoring private local inputs; see
  RUNBOOK Section 1. No training/model changes were made for this Git cleanup.
- Confirmed current label yield: 37 logged deviations, one personal stop
  excluded, three GPS/prior-route match failures, 16 without a sufficiently
  supported outgoing choice, 17 usable deviation labels from nine riders.

# Training comparison approved 21 October 2026

- Added an opt-in `--training-ablation REFERENCE_REPORT` mode to grouped
  evaluation. Reuse the completed k-fold corpus and exact saved outer/inner
  groups. Verify official input/history/settings hashes before fitting and
  reject output inside the completed reference folder. No repeated map matching.
- Compare current losses, choice-only loss, and choice-only with square-root
  automatic deviation weighting. All three select epochs by validation choice
  NLL. Choose a recipe using inner validation only, then refit and test that
  recipe. The STGAT-LSTM-only command takes 12 fits; RUNBOOK Section 11.
- Architecture, features, existing records, GPS exclusions, regularization and
  first-visit policy stay fixed. No SMOTE, synthetic records, forced minimum
  training epoch, incoming-road encoding or personal-history inputs were added.
  Ordinary commands retain their previous training and checkpoint-selection
  defaults. New experiment results are pending; no accuracy improvement is claimed.
- Before edits, saved nine source/test/document files and original hashes under
  `artifacts/rollback/before_training_ablation_20261021`. The manifest records
  fixed-split and completed k-fold stages, unchanged artifact hashes and restore
  instructions. The source rollback point is completed k-fold, while the earlier
  fixed-split workflow and checkpoint remain available separately.
- The new output folder is `artifacts/experiments/real_training_ablation_20261021`.
  Keep prior results, fold assignments, official data, model.py and ORIGPLAN.md.
- Verification: all 105 unit/integration tests pass. On a small deterministic
  fixture the unchanged default training reproduces every parameter tensor,
  epoch loss and selected epoch from the saved pre-change source. Actual
  reference hashes/settings and preserved experiment hashes were checked.
  Full real-data ablation fitting has not been started; the user will run it.

# Rider-grouped evaluation update, 20 October 2026

- Run completed and reviewed 21 October: all 12 fits and all 68 original
  held-out predictions per model are present. STGAT-LSTM gets 10/17 alternative
  roads, 17/51 follows, 27/68 overall (39.7%). GATv2 gets 5/17, 28/51, 33/68
  (48.5%). Always-follow gets 51/68 but no alternatives. The LSTM's selected
  epochs are 45/5/1; early selection and poor followed-road performance need
  investigation on training/validation before any changes. Seven deviation
  exits have no exact chosen-exit development supervision. Group coverage,
  original-example membership, saved reports, input hashes, checkpoint presence
  and traffic exclusion metadata were checked. No new model changes or fits
  were made during this review. Full results supersede the pending status below.
- Added `evaluate --group-k-folds 3` in the existing evaluation module. The
  default comparison is GATv2-only and GATv2-LSTM. Each fold selects epochs on
  separate validation riders, then refits development riders before testing.
  Three folds and two models require 12 fits; optional latest-only LSTM takes 18.
- Original approved decisions are the evaluation corpus. Additional genuine
  follows are recovered uniformly and used only for fitting riders within each
  fold. No synthetic choices or synthetic tests are introduced. Original
  validation/test targets are not filtered by the training first-visit policy.
- Freeze rider assignments and target hashes before fitting; exclude held-out
  riders, the target ride and future observations from applicable GPS profiles.
  Report pooled and per-fold exact alternative-road/followed-road counts, NLL,
  per-rider variation, development fit and validation curves. The current 17
  deviation decisions come from nine riders, not 17 independent participants.
- Architecture, losses, and current checkpoint-selection criterion are
  unchanged. A deviation-balanced checkpoint criterion and explicit incoming
  road/trip-prefix context remain proposals, not implemented changes. Survey
  responses made after a deviation remain label evidence, not inference input.
- Existing real baseline and original split are preserved. This is an internal
  study of already inspected data. Implementation checks pass; the user will
  run the full comparison. Exact commands are in RUNBOOK Section 10.
- Verified the actual original evaluation corpus: 30 examples, 68 choices and
  17 deviations. Saved the seed-17 plan in
  `artifacts/experiments/real_group_kfold_20261020/rider_folds.json` before any
  full-data fitting. The three test folds contain 22/23/23 choices and 5/6/6
  deviations respectively. All 101 implementation tests pass. Real-data model
  fits and performance results remain pending.

# Extra Context
- I have a previous model that doesn't actually learn tacit knowledge; instead, it adds a layer that multiplies travel time. The panel wants the model itself to learn tacit knowledge, not just be a layer multiplier.
- That project is located at C:\Users\Vince\Downloads\THESIS_APP\finalGNNT3
- You may or may not choose to use this as a reference if parts are appropriate, but I am not very familiar with it.

# References
- For research papers, I do not want you to copy them if they are not appropriate for our requirements. But you may definitely use some parts from one paper and some parts from another. However, if one paper is already viable, we can opt for that.
If ever you do use their concepts and/or code, please include the paper in a comment on their code.
- Reference papers are in: C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\1 NEW THESIS PAPERS
- https://github.com/ebrahimpichka/GAT-pt.git (this is a good basic example, however, it is not thesis level for us, but we may piggyback and add/improve it.)
- Take note that, with regard to code, follow the state of the art as much as possible.
- You may go beyond the references we have, but please cross-check with me.

# Goal & Expected Output (We can have milestones)
- MAIN GOAL AND FIRST PRIORITY IS TO DO STGAT.
- Establish 2 appropriate models, such as a simple GAT and/or GCN, to serve as baselines for our own.
- Accomplish a fully fledged STGAT
- Implement and Utilize LSTM
- Create an appropriate routing algorithm and utilize it accordingly
- Visualization of the outputs that have adjustable parameters (Website version)
- Complete Validation Pipeline (Prioritize the GAT + also the final results)
- I need to understand your implementation and all the code. Do what you have to do, however complex it is, but let me understand it, as I would need the knowledge to present and defend.
- For the specific stuff about the GAT, about what it should predict, node, edge, graph, I am not sure, so communicate with me please and help me. An idea was: Node Level - Delay Penalty/Priority Score (How likely to cause a deviation), Edge Level - Travel Time & Affinity Score (0 or 1 likelihood of rider chooses to go one point to the other); Attention Weights

Or you may check:
Phase 1 — Baselines: Implement a simple GCN and a standard GAT on the road network graph. These are baselines for comparison only.
Phase 2 — STGAT: Build the spatiotemporal GAT with node-level and edge-level prediction heads.
Phase 3 — LSTM Integration: Integrate LSTM for temporal sequence modeling of rider deviation patterns.
Phase 4 — Routing Algorithm: Implement a routing algorithm that consumes the model's outputs (edge affinity scores, delay penalties).
Phase 5 — Validation Pipeline and Metrics: End-to-end evaluation with metrics appropriate for each prediction head.
Phase 6 — Visualization: Web-based visualization with adjustable parameters.

# Disclaimer
- Development should be reviewed from the perspective of an experienced machine-learning engineer.
- Another important reminder: if you need any clarification on anything, please tell me or ask me. Let me know every step of the way. Perhaps you may ask if I have a certain resource already. Ask me, please.
- Traffic context is now derived from timestamped rider GPS as historical motorcycle probe speeds. Mapbox remains optional compatibility code rather than the primary thesis design.
- The revised real decision artifact uses local outgoing-road choices and contains 13 conservative followed examples plus ten deviated examples from 13 riders. Automatic validation requires GPS evidence of the chosen alternative; neither following the regenerated route nor rejoining is mandatory. Seven events also have comparable paths for optional route-ranking loss.
- A shared STGAT-LSTM model trains an outgoing-road choice head. At each eligible branch, softmax compares accessible roads from the same source. Positive additive routing costs are derived as negative log choice probabilities, while intentional deviations also supply route-ranking loss.
- The revised 23 approved examples produce 61 outgoing-road choices: 51 suggested roads chosen and ten observed alternatives chosen. The original 16-example evaluation is archived under artifacts/experiments/real_final_20261014 and does not evaluate these revised labels.
- Recovery trial on 15 October: 15 of the 18 strict-transition failures can now be matched with six candidates and 8 m backward jitter; one yields a supported local target. Removing regenerated-route/rejoin gates recovers additional events, giving ten deviations overall. Source CSVs remain unchanged.
- The model graph expands from its initial 1 km Taft buffer to 1.7 km to cover approved evidence and destinations, and checkpoints save that scope. All 71 tests pass. A three-epoch training and checkpoint-reload check passed; full training and rider-held-out evaluation of the revised set remain pending.
