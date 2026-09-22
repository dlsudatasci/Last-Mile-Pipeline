# Technical review and clarification register

Reviewed: 2026-09-21. Status: requirements review and first executable model slice; real-data training and application integration have not started.

This is a companion to `PLAN.md`. Confirmed answers below take precedence over earlier suggestions. Unanswered questions remain pending; recommendations are not decisions. The user may supply details later.

## Confirmed with the user

| Item | Confirmed requirement |
| --- | --- |
| Primary research objective | Train GAT on collected rider knowledge, then have a separate routing algorithm use the trained model to generate routes. Rider-choice prediction is the main learning/evaluation interest; actual route improvement requires separate evidence. |
| Travel-time objective | User explicitly clarified: improving travel time is not the goal; any improvement is a bonus. Preference learning and its use in route generation are the primary objectives. |
| Prediction clarification | The earlier answer selected follow/deviate prediction. The user subsequently clarified that the intended application is route generation using the trained GAT. Binary deviation supervision remains a candidate, but the model's routing-facing output is not finalized. |
| Vehicle | Motorcycles only. |
| Study focus | Taft Avenue, Manila; collecting more data there is ongoing. User authorized a provisional boundary assumption; see below. |
| Rider scope | User describes personalization with emphasis on generalization across riders, rather than catering to one rider. Working interpretation: a shared model; any rider-specific conditioning remains to be clarified. |
| Data collection changes | Technically possible, but preferably avoid changes to the existing app and survey. |
| Current work | Review technical requirements before model/application implementation. User additionally authorized assumptions for inconsistent old-build data and dummy data for testing. |
| Collection app | Used for data collection. Review the latest `fix/latest-issues` branch of `dlsudatasci/Last-Mile-Delivery-App`; production deployment of the GAT inside this app is not requested. |

Still required by `PLAN.md`: STGAT as the first model priority, LSTM, two model baselines, subsequent routing, validation, a web visualization, and explanations sufficient for the thesis defense. The exact architecture and measurable thesis claim need to be aligned with the confirmed prediction task.

## What was inspected

Read `PLAN.md`, the referenced OSM README and collection/feature-engineering notebooks, rider preprocessing notebook code, cleaned CSV schemas and aggregate counts, and processed GraphML metadata and node IDs. Checked the previous project's README and relevant training/tacit-layer code. Subsequently reviewed the collection app at branch `fix/latest-issues`, commit `4171a9962a06a8320fbf1aa099f047cd20faa7f9` (September 18, 2026). See [APP_DATA_REVIEW.md](APP_DATA_REVIEW.md) for findings and source links. Inventoried the supplied paper PDFs; this was not a full literature review or a complete data-quality audit. No personal participant values or credentials are reproduced here.

Paths below are relative to `C:/Users/Vince/Downloads/THESIS_APP/T3-NEW-MODEL`, unless otherwise stated.

| Available artifact | Observed facts | Implication |
| --- | --- | --- |
| `ExportLMD/clean_data/rides_clean.csv` | 55 rides from 11 distinct rider IDs; ride starts span July 24-31, 2026. | Treat this as a preliminary sample. Final collection size and coverage remain unknown. |
| Deviation and response CSVs | 34 deviations and 34 survey-response rows; all 34 deviation route references match a generated-route ID. | A useful starting point, but row counts and route references alone do not validate all joins or labels. |
| Rides without reported deviations | 35 of the 55 rides have a recorded deviation count of zero. | Potential sources of follow-route examples; verify actual adherence and logging completeness before assigning negative labels. |
| `map_points_clean.csv` | 11,350 points with ride ID, point index, latitude, longitude, timestamp and elevation; no accuracy field in this cleaned schema. | Full-trip traces appear available. Validate completeness and map-match trajectories before constructing examples. |
| GPS timing | Within-ride intervals have a median of 2.001 seconds; 39 intervals exceed 60 seconds; maximum 953.645 seconds. | Some decision windows may be unobservable and must be marked unknown or excluded. These are timestamp gaps, not a diagnosis of their cause. |
| `generated_routes_clean.csv` | 167 records: 55 initial routes, 62 regenerated routes and 50 traffic updates. | Route versions must be ordered and aligned to the route displayed before each decision. A later reroute is not a valid earlier input. |
| `originalRouteEdge`, `deviatedEdge` | Values are navigation-instruction strings, not OSM segment IDs. | A trajectory/route-to-graph mapping stage is missing. Do not join by street name or assume these are edge identifiers. |
| Survey reasons | 16 combine shortcut/faster route/personal preference/familiar road; 9 traffic; 6 avoid intersection; 3 other categories with one example each. | The combined category cannot distinguish familiarity from time saving. Very rare reasons cannot support strong per-reason conclusions from this sample. |
| Processed OSM graph | 36,375 nodes, 104,019 directed edges; simplified and intersection-consolidated; CRS `EPSG:32651`; metadata date July 21, 2026. | Usable static graph foundation, substantially larger than the current observed ride sample. Target-area clipping needs a boundary buffer for routes. |
| Node feature CSV | 36,375 unique IDs; every ID matches the processed graph, with no unmatched IDs on either side. Features include street count and nine POI categories. | Node-ID compatibility is confirmed for these files. Feature usefulness and spatial counting still require validation. |
| Previous model | `finalGNNT3/edge_gat_training.py` trains against `simulated_travel_time`; `tacit_layer.py` subsequently computes a multiplier. | It does not establish that the prediction model learns rider choices. Reuse must be selective. |

## Technical gaps that affect the research design

### 1. Define what the trained model supplies to routing

The intended system has two stages: offline learning from collected rides and surveys, followed by route generation using the trained model's outputs. The collection app does not need to run a live deviation-prediction service. The earlier discussion framed the task too narrowly around advance deviation prediction.

Proposed routing-facing output: context-dependent road/turn preference costs, or scores for candidate paths, learned from observed rider choices. Follow/deviate classification can remain an auxiliary objective or evaluation task. The final choice between local costs and whole-path scoring is pending design discussion; it must agree with the routing algorithm and the available training comparisons.

For example, a verified intentional detour can supply a comparison between the path taken and the suggested path rejected, using a shared origin/destination or divergence/rejoin points. Train the model to prefer the demonstrated choice under its recorded context. This is evidence of a conditional preference, not automatic evidence of faster travel. Wrong turns, personal stops, changed destinations and map errors need distinct treatment; do not label every deviation a desirable recommendation.

Working geographical assumption, authorized by the user's request to assume a scope: use the Taft Avenue centerline inside the City of Manila and a 500 m buffer on each side as the study corridor. Include connecting streets. Start with a 1 km centerline buffer for the computational routing graph so the study boundary does not immediately truncate alternatives. These radii are engineering starting points, not validated optimums. Keep full raw trips; flag boundary-crossing examples, and revise the boundary when actual route coverage is assessed. Exact endpoint coordinates remain deferred; a graph extract has not been created.

Define deviation distance/time thresholds, persistence, GPS-noise handling, U-turns, stops, temporary detours, route rejoining and app-triggered rerouting. Distinguish rider-initiated changes from a new recommendation displayed by the app. Missing observations are not automatically follow-route labels.

### 2. Construct both classes from comparable opportunities

Use verified follow-route and deviation decision events. Do not compare one deviation event against thousands of arbitrary GPS points, or count overlapping windows as independent observations. Document the sampling unit, exposure to the suggested route, class prevalence and exclusions. Survey absence is not evidence that no deviation happened.

The 34 reported deviations do not yet establish enough independent supervision for the proposed architecture. Plan learning curves and uncertainty estimates after the final collection scope is known; avoid inventing a universal minimum sample count.

The user authorized proceeding with explicit assumptions and synthetic examples while older-build inconsistencies remain unresolved. Preserve the raw exports. For prototypes, treat undocumented ETA fields as estimates, retain missing traffic as unknown, and recover a prior suggested route only when same-ride time ordering provides unambiguous evidence. Otherwise mark the training target unavailable. Do not invent missing real rider choices, reasons, observed travel times or historical traffic. Synthetic examples live separately under `data/synthetic/`, with expected handling for both usable and unusable records. They test data contracts and later software behavior, not thesis performance or real-world learning.

### 3. Make rider evidence participate in training

The proposed main learning objective compares observed rider choices with feasible alternatives; a binary follow/deviate loss alone is not assumed sufficient for routing. Historical rider behavior and survey-derived information may provide inputs or auxiliary supervision where justified. Survey answers submitted after an event may supervise offline learning, but cannot be inputs that the model requires before that same event. Historical aggregates must contain only information available at prediction time.

Define an operational thesis claim such as: rider-informed STGAT-LSTM predicts held-out route preferences better than models without rider-derived information, and its learned scores can generate feasible routes. This is a proposed claim, not an achieved result. Prediction success alone does not establish safer or faster routes, causal understanding of rider knowledge, or a unique explanation of why a rider deviated. If the panel still requires binary deviation prediction as the main benchmark, retain it explicitly alongside routing-facing evaluation.

### 4. Prediction heads and embeddings are optional design choices

Node, edge, subgraph and graph-level prediction heads are not all required. An embedding is an internal representation; it is not automatically an independently supervised output.

For route generation, a reasonable candidate is a graph encoder with a learned road/turn or path scoring component. If a binary auxiliary classifier is included, its output is a probability between 0 and 1; it is separate from the routing preference cost. A fixed node-only score would omit the incoming road, suggested outgoing road, destination and trip context.

Node delay and edge travel-time heads are optional and are not required by the clarified objective. Add them only if credible labels and a demonstrated benefit justify the added task. Current trip durations and estimated route durations do not automatically provide observed intersection delays or observed segment travel times. Attention weights may be visualized as model diagnostics; they are not probabilities of deviation or proof of the rider's reasoning.

The [original GAT paper](https://arxiv.org/abs/1710.10903) describes neighborhood attention. The linked [GAT-pt repository](https://github.com/ebrahimpichka/GAT-pt) targets Cora node classification, so substantial task-specific changes would be needed. Neither reference by itself specifies this project's prediction problem.

### 5. Complete the graph and feature contract

| Entity | Candidate information | Required checks |
| --- | --- | --- |
| Node/intersection | Street count, intersection type, available traffic-signal tags, nearby POI categories. | POI deduplication across source layers; missing tags versus true absence; sensitivity to the current 100 m POI radius and intersection consolidation. |
| Directed road segment | Length, geometry, road class, one-way/access tags, available lanes/maxspeed/bridge/tunnel tags. | Keep `(u, v, key)` identifiers and a fixed graph version; normalize missing/list-valued tags; verify motorcycle suitability. |
| Decision/turn | Incoming segment, recommended next segment, feasible outgoing choices, destination-relative context, time and observed prior trip history. | Recover the recommendation active before the event; respect turn/access constraints; do not infer behavior solely from a nearby node. |
| Dynamic context | Past rider observations, available traffic, observation age and missingness. | Record event time and availability time; align traffic geometry to the OSM graph; do not fabricate historical live traffic. |

The existing `drive` extraction is not a confirmed motorcycle-specific network. The supplied processed GraphML is nevertheless structurally compatible with routing: it is a projected EPSG:32651 directed multigraph with 36,375 nodes and 104,019 edges. Geometry, length, road class and one-way attributes are present for every edge; lanes cover 56.7%, maxspeed 26.1%, and access 14.4%. Turn restrictions and motorcycle-access rules still need an explicit representation and validation. Do not silently remove observed choices that fall outside the legal routing graph: record and resolve how such cases are handled. Preserve a distinction between the graph used for message passing and roads allowed for navigation.

The separate POI feature CSV has exactly one matching row for every graph node and no unmatched IDs. Its ten current numeric fields can be joined by the consolidated node ID. These are contextual node features. Road geometry, direction, access and the learned preference cost remain edge properties.

Use consistent coordinates when matching GPS to the projected graph. Consolidation changes node identities; preserve the original-ID crosswalk and graph snapshot. OSMnx represents road edges using `(u, v, key)`; dropping the key can merge distinct parallel roads. See the [OSMnx graph-conversion documentation](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.convert.graph_from_gdfs).

The implemented coordinate and OSM screens found that all 167 saved route polylines parse and all 34 deviation events have nearby GPS before and after. Median event distance is 113.4 m from the latest pre-event route and 3.3 m from the route referenced after rerouting, consistent with post-deviation route regeneration. Across the full export, 95% of GPS points are within 37.9 m of an OSM edge. The collection is geographically broader than the intended scope: only 17 of 55 rides remain fully inside a provisional 500 m Taft Avenue corridor, while 17 intentional deviation events occur inside it.

The HMM/Viterbi sequence screen is based on Newson and Krumm (2009), DOI 10.1145/1653771.1653818. It produced complete GPS/prior/regenerated route matches for 11 of the 17 corridor events. Eight have post-event GPS edge overlap that is higher for the regenerated route than for the prior route. They are review candidates only. Exact divergence/rejoin subpaths and a shared origin/destination must still be verified before these become preference pairs; six events failed a directed transition under the initial matcher parameters.

The stricter common-OD extractor now finds three candidates, each from a different ride and rider. It requires a shared directed edge before the event, a later shared directed edge after it, connected but different paths between those boundaries, and stronger GPS overlap with the regenerated route. The three survey reasons are familiar/shortcut preference, road blockage/hazard, and intersection avoidance. Their preferred/rejected path lengths are approximately 1,392/2,220 m, 445/324 m, and 387/316 m. Thus two observed choices are longer in distance, which is compatible with the confirmed preference-learning objective. This is still only a small pilot set and not sufficient for meaningful STGAT-LSTM estimation or held-out evaluation.

The real feature pipeline now extracts a provisional 1 km Taft corridor with 1,368 nodes and 3,418 directed edges. It produces 13 node features, 19 static edge features, five dynamic traffic fields and two destination-relative fields; all three candidates are covered. The five traffic fields keep normalized congestion, speed relative to the OSM reference speed, separate availability masks and observation age. Live tests returned speed on all tested segments but numeric congestion on only part of one route, so missing congestion remains unknown. Because no historical per-edge traffic observations exist in the reviewed rider export, those older choices still receive an explicitly unknown temporal snapshot and cannot be paired retrospectively with today's live traffic.

For the explicit follow-versus-deviate target, whole rides and local deviation events cannot be mixed as if they were the same decision. The current decision artifact therefore uses a local upcoming branch as its unit. Five no-deviation rides have strong GPS agreement through a real branching node and supply provisional follow cases; the three strict divergence/rejoin candidates supply deviation cases. The resulting 8 examples span 6 riders and remain review-required. This is enough to validate software interfaces, but not enough to estimate generalization or tune an STGAT-LSTM reliably.

The multitask training path shares the STGAT-LSTM encoder between a follow/deviate classification head and the existing edge-cost head. Synthetic training now produces a reloadable checkpoint and can fit all eligible constructed labels and preference pairs in sample. The real training command checks for `approved_for_training` and rejects the current pending artifact. Even after approval, results from eight examples must be labeled an in-sample smoke run rather than model accuracy.

The eight decision examples were manually approved on the generated review artifact. The real smoke checkpoint trained on 5 followed and 3 deviated examples from 6 riders, with 7/8 in-sample classification and 3/3 in-sample preference ranking. The historical inputs use unknown traffic snapshots because current Mapbox observations cannot be attached retrospectively. No rider-disjoint accuracy claim is supported by this run.

### 6. Specify what time means for STGAT-LSTM

Choose whether the LSTM processes histories of road observations available before a routing request, earlier decisions within a trip, or both. For a route generated at trip start, the rider's future movements are unavailable: any trip-history encoder needs a defined initial state. Time-of-day features alone do not specify temporal sequence modeling. Define history length, sampling interval, padding/masks, missing-data handling, sequence resets and forecast horizon if forecasting is used. Do not feed a future trip segment into an earlier prediction.

STGAT and LSTM should be designed as one coherent temporal system. The staged implementation in `PLAN.md` can remain a work sequence, but cannot substitute for defining the final architecture. The best temporal resolution depends on final observations per road and per time interval.

### 7. Establish evaluation before model training

Split by complete rides and time; prevent overlapping windows from crossing splits. Given the user's emphasis on generalization across riders, propose a rider-disjoint evaluation alongside prediction on later trips. If rider-specific conditioning is retained, define a fallback for unseen riders and evaluate its contribution separately. If claiming transfer to unseen roads, define a separate spatial evaluation. Fit preprocessing and rider-derived aggregates on permitted training history; keep validation for selection and test data for final evaluation. See [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

For preference learning, propose held-out choice log loss/ranking accuracy and route overlap or route-match measures with documented candidate sets. Evaluate the routing phase for legal connectivity, destination reachability, detour and inference latency; actual time savings need observed outcomes. If binary deviation prediction is retained, report precision-recall AUC, precision/recall/F1 at a validation-selected threshold, calibration/Brier score and class prevalence. Event warning metrics are only needed if an advance-warning task is deliberately added. Report uncertainty across independent rides/riders and multiple training seeds where feasible.

Compare GCN and standard GAT using the same labels, splits, usable information and tuning budget. Include a simple non-neural/majority benchmark and an ablation that isolates temporal modeling. To substantiate rider knowledge, compare the full model with the same model without rider-derived history/survey information. If surveys are only auxiliary targets, test that contribution separately.

### 8. Clarify live traffic access and historical availability

Mapbox Directions provides route-based annotations such as duration and congestion with `driving-traffic`; annotations are associated with route geometry and congestion can be unknown. A Mapbox account does not establish that complete historical or road-network-wide traffic data is available. See [Directions API documentation](https://docs.mapbox.com/api/navigation/directions/).

Mapbox's separate Traffic Data product requires a Traffic Data license. Confirm the product available to this project, study-area coverage, collection budget, permitted retention/training use, timestamped storage and missing/stale-data behavior. See [Mapbox Traffic Data access](https://docs.mapbox.com/data/traffic/guides/). Do not use today's traffic to represent traffic during the July trips. No account credentials are needed for this review.

### 9. Define how the later router uses learned scores

A probability of deviating does not identify which alternative road to choose or establish a travel-time saving. The proposed route/turn preference outputs address that missing interface. If routing uses learned additive costs, enforce the cost and state assumptions needed by the chosen search method; turn-dependent costs need incoming-road state. If the model scores whole candidate paths, route generation plus ranking is a different design and may miss good paths absent from the candidate set. Arbitrary model scores cannot simply be summed or treated as seconds.

Using learned preference costs in a conventional route search can meet the intent that tacit knowledge is learned inside the model: the rider evidence must change trainable parameters through the loss. A fixed hand-authored survey penalty applied after training would not demonstrate this. Ablations must show the effect of rider-derived supervision/information. Personal stops and familiar-road preferences may explain deviations without implying that the suggested road is poor.

The confirmed routing objective prioritizes learned rider preferences; travel-time improvement is a bonus, not a requirement. A slower feasible route can therefore be the preferred result. Feasibility, destination reachability and avoidance of pointless loops remain requirements. Exact detour bounds, any secondary tie-breaking rule, update frequency, route stability and fallback behavior remain engineering choices to propose. Do not silently make fastest travel time the objective. If costs depend on arrival time, validate the assumptions of the time-dependent search. Do not select a complex algorithm solely to label it state of the art. Routing evaluation must be separate from deviation-prediction evaluation.

### 10. Add practical experiment requirements

Record data/graph versions, feature definitions and units, exclusions, seeds, configurations, dependency versions and checkpoints. Establish compute availability and deployment latency before selecting model size. A project-local `.venv` is now established and runs PyTorch, PyTorch Geometric and the geospatial preparation dependencies under Python 3.14.5. The complete automated suite currently covers the model, routing, geometry, OSM indexing and map-matching core.

Use pseudonymous rider IDs in model datasets; the cleaned user table still contains identifying columns that are unnecessary for this prediction task. Confirm collection consent and research data handling. Define the dashboard's audience, required controls and uncertainty display after the model contract is agreed. Maintain an architecture explanation and source attribution for the thesis defense.

## Pending clarification register

No blank entry below should be treated as approval or as a finalized assumption.

| ID | Question | When needed | Status |
| --- | --- | --- | --- |
| Q01 | Which learned output should drive routing: road/turn preference costs or candidate-path scores, with optional deviation classification? | Before model/routing interface design | User clarified offline training followed by routing; a live warning horizon is not a prerequisite |
| Q02 | Is a shared model sufficient, or should it additionally use rider-specific history/profile when available? | Before evaluation and feature design | Generalization across riders is the stated emphasis; exact conditioning remains pending |
| Q03 | Final Taft corridor endpoints, connecting streets and allowed excursions outside the study boundary? | Before final dataset freeze | User authorized assumptions: Manila-city Taft section, 500 m study buffer and initial 1 km routing-graph buffer; validate later |
| Q04 | Expected final rider count, rides per rider and collection duration? Is the inspected sample pilot/test data? | Before feasibility and split design | More collection is ongoing; quantities, duration and sample status pending |
| Q05 | How does the app detect a deviation and trigger rerouting? Which route/version is displayed at each timestamp? | Before trusting event labels | Current branch inspected; see APP_DATA_REVIEW.md. Applied build version remains pending |
| Q06 | Are the full GPS logs and generated routes complete for all final rides? Are test rides, uploads, destination changes and missing surveys flagged? | Before dataset construction | Pending |
| Q07 | Is STGAT-LSTM fixed by the approved thesis title/panel, and are node-delay/travel-time outputs explicitly required? | Before architecture freeze | PLAN requests STGAT and LSTM; mandatory additional heads not confirmed |
| Q08 | Which historical traffic records already exist, and which Mapbox products/budget are available? | Before traffic-dependent training | Pending |
| Q09 | What measured improvement and evaluation protocol would satisfy the panel for rider-choice learning and its use in routing? Is a specific novelty claim required? | Before final experiment design | Routing purpose clarified; success criteria pending |
| Q10 | Are `isFaster`, route durations and remaining-time fields measured or estimated? | Before using these as features or secondary labels | Current branch: estimated ETA comparison, with app calibration; historical build equivalence pending |
| Q11 | What hardware, training time budget and thesis deadline are available? | Before model sizing | Pending |
| Q12 | What should the later routing phase optimize, and is a real rider trial required? | Before routing design; can clarify later | Confirmed: learned rider preferences, with travel-time improvement only a bonus. Detour bounds and rider-trial protocol remain pending |
| Q13 | Who will use the website, which controls/outputs are required, and where will inference run? | Before deployment; can clarify later | Pending |
| Q14 | What participant-consent, retention and sharing arrangements cover training and evaluation? | Before final dataset preparation | Pending |
| Q15 | Which app build/commit is installed on collecting riders' phones, and which version produced the July sample? | Before confirming historical collection semantics | Older build confirmed; exact version deferred. User authorized explicit assumptions and dummy data so prototype preparation can proceed |

Architecture width, head count, exact history length, optimizer settings, tuning ranges and framework versions are engineering decisions to propose after these requirements and data checks. The user does not need to supply every hyperparameter.

## Proposed order of work after clarification

1. Agree on the learned preference output, available-at-routing-time inputs, scope and evaluation claim.
2. Audit/map-match existing data and assess whether it supports that claim; prefer deriving missing training examples from current logs over changing the app.
3. Freeze label rules, data splits and metrics; establish simple and GCN/GAT baselines.
4. Implement the agreed STGAT-LSTM and rider-information ablations.
5. Design routing from validated outputs, then build its evaluation and the website.

No new research paper or architecture has been adopted by this review. A focused comparison of appropriate methods, including current work if useful, remains to be discussed with the user as requested in `PLAN.md`.
