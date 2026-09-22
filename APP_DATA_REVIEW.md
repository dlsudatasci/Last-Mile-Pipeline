# Collection-app review for model design

Reviewed: 2026-09-21. Scope: static source inspection and comparison with existing exported data. No application code was changed, executed or deployed.

## Version and applicability

Requested repository: `https://github.com/dlsudatasci/Last-Mile-Delivery-App.git`.

The remote `fix/latest-issues` branch resolved to commit `4171a9962a06a8320fbf1aa099f047cd20faa7f9`, dated September 18, 2026, with subject "Fix study enrollment display, pending trip review resumption, and auth flows". A separate shallow review copy is in `.review-references/Last-Mile-Delivery-App`.

The user confirmed that collecting riders use an **older build**. Its build date/commit is unknown. Therefore the source behavior below describes the requested branch, not necessarily every collected ride. July exports must not be assumed to have September semantics. Observations verified directly in the exported sample are labeled separately.

## Findings relevant to training

### 1. GPS traces and suggested-route versions are available

The reviewed source requests background GPS updates at a 2-second time interval and 2-metre distance interval. These are requested settings, not guarantees of recording frequency. It saves raw ride points, while display points are handled separately. The raw point representation includes coordinates, timestamp, elevation and heading, but does not retain the location accuracy value. [Recording and point conversion source](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/store/useRideStore.ts#L141-L149).

Route snapshots include geometry, sequence, generation time, route type and estimated remaining time/distance fields. They are saved in `generatedRoutes`, and ride points under `rides/{id}/map/points`. [Persistence source](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/firebase-crud/rides.ts#L274-L292).

The inspected export script emits latitude, longitude, timestamp and elevation, omitting heading. If the actual older build saved heading, a future export-only change could recover it without modifying the collection app. It cannot recover GPS accuracy if the original build never persisted accuracy.

Implication: map-matching the raw traces and suggested routes is a plausible preprocessing path. The saved deviation point alone is not the full detour trajectory; obtain that trajectory from the ride's raw points.

### 2. The current detector is threshold based and dependent on rerouting success

The latest source confirms deviation after three successive processed raw points at least 35 metres from the active route, subject to a 45-second reroute cooldown. Traffic refresh is considered after 180 seconds and deferred while the rider is off-route. A deviation event is appended only after a successful, still-applicable route request. [Detector and reroute source](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/store/useRideStore.ts#L604-L674).

Implications for this version: the event timestamp records threshold detection, which can occur after the actual turn. Closely spaced deviations may be suppressed by the cooldown. Failed route requests can leave real off-route behavior without a corresponding event. The detector does not explicitly gate its off-route counter on GPS accuracy. Thus event absence is not a reliable negative label without inspecting the trajectory. These specific thresholds must be checked against the older deployed build before applying them to the dataset.

### 3. A deviation's route reference points to the new route

In the latest code, `setActiveRoute` runs before `addDeviationEvent`; the latter reads the newly active generated-route ID. Survey metadata then copies that ID to `routeId`. [Update order](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/store/useRideStore.ts#L651-L666), [ID capture](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/store/useRideStore.ts#L577-L579), [survey metadata](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/app/main/%28tabs%29/record/reason-for-deviation.tsx#L57-L97).

**Confirmed in the existing cleaned export:** all 34 deviation route references resolve to a `Regenerated Route`, and all 34 referenced routes have `generatedAt` later than the deviation's `timestamp`.

Implication: do not interpret this foreign key as the rejected suggestion. Reconstruct the previous active route using the same ride's ordered snapshots, timestamps and geometry. A raw ID match alone is insufficient. Validate ambiguous timing and concurrent refresh cases rather than always subtracting one from the sequence. Do not use the post-deviation route geometry as an input for predicting the preceding rider choice.

### 4. `isFaster` compares estimates

The latest source sets `isFaster` to `newEtaSec < previousEtaSec`. The old remaining ETA uses route progress, while the new duration comes from a newly requested route. These are not two observed journey times. [Metadata calculation](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/app/main/%28tabs%29/record/reason-for-deviation.tsx#L64-L67), [remaining-ETA calculation](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/app/main/%28tabs%29/record/index.tsx#L129-L145).

The returned route duration uses `estimateMetroManilaEtaSec`, which takes the maximum of Mapbox duration and an urban-speed floor based on local hour/congestion, then rounds to a minute. The old/new fields stored in route snapshots also do not by themselves guarantee comparisons made at the same origin and time. [ETA adjustment](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/utils/directions.ts#L550-L571), [snapshot fields](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/store/useRideStore.ts#L543-L554).

Implication: label these as estimated values with documented provenance. They are not travel-time ground truth and should not be treated as proof that a deviation was beneficial. Exact historical behavior requires the deployed build.

### 5. The displayed baseline is selected by app heuristics

The latest app requests Mapbox `driving-traffic` alternatives with motorway/toll exclusions. It ranks candidates using a hand-authored combination of distance, duration, traffic delay, congestion and restricted-road exposure, then selects a route. [Selection and request source](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/utils/directions.ts#L574-L650).

Implication: the offered suggestion in this version is not simply an untouched Mapbox fastest route. Preference labels are conditional on the recommendation policy riders saw. Preserve that provenance when evaluating what the GAT learned. Do not train it merely to reproduce the app's hand-authored score and call that learned tacit knowledge.

### 6. Live traffic is requested, but detailed history is not in the inspected route export

The app requests congestion, numeric congestion, duration and distance annotations, and retains congestion segments in active state. The generated-route snapshot construction saves geometry and summary fields; it does not copy the per-segment traffic annotations or raw provider response. The cleaned export likewise has no per-segment traffic-history table. [Directions request](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/utils/directions.ts#L605-L610), [route snapshot construction](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/store/useRideStore.ts#L530-L570).

Implication: a `Traffic Update` record does not mean a full road-level traffic snapshot was archived. Separate traffic archives may exist elsewhere; that remains pending. Current traffic cannot fill missing historical live observations without changing their meaning.

### 7. Survey data needs a learning role and a completeness audit

The current survey supports traffic, intersection avoidance, hazards, personal stops, shortcut/familiar-road preferences, parking, wrong turns and navigation errors. The combined shortcut/faster/familiar category remains ambiguous about the precise motivation. [Survey options and validation](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/deviation-questionnaire.ts).

The reviewed Firestore submission creates deviation metadata when submitting review answers. Thus the `deviations` table in this path depends on submitted reviews, whereas a saved ride contains the detected deviation count. Survey submission also creates response documents with generated IDs; check repeated responses when auditing retries. Do not assume one response per event purely from the schema. [Review persistence](https://github.com/dlsudatasci/Last-Mile-Delivery-App/blob/4171a9962a06a8320fbf1aa099f047cd20faa7f9/lib/firebase-crud/reviews.ts#L33-L94).

The existing preprocessing recomputes `deviationCount` from exported deviation rows. Preserve the raw ride count in a future audit so this recomputation does not conceal detected events lacking submitted responses.

## Proposed model-to-router interface

This is a recommendation for discussion, not a finalized architecture:

1. Recover comparable decision/path examples from the recorded GPS and the suggestion active at the time. Match endpoints and confirm intentional behavior before asserting a preference.
2. Train STGAT-LSTM to score road/turn desirability or candidate paths from road structure and available temporal context. A preference loss should update the model directly from observed choices. Use survey reasons to distinguish kinds of examples or as auxiliary supervision; do not require a future survey response at inference.
3. Feed trained scores into a routing search or candidate-route ranking procedure, alongside explicit feasibility constraints. The user confirmed that rider preferences are the objective and travel-time savings are a bonus; faster travel must not be imposed as the main objective.
4. Keep follow/deviate prediction as optional auxiliary supervision or evaluation. Do not assume it alone specifies which alternative should be routed.
5. Compare against the same architecture without the rider-derived information/supervision. This tests whether the learned rider contribution adds value.

The choice of local costs versus path ranking needs to be settled before implementing the model. A path-dependent score may require candidate ranking or augmented search state; conventional shortest-path search assumes decomposable costs. Learned preference units are not automatically seconds. Cold-start routes must use only historical information available at request time, with explicit missing-data behavior.

## Remaining clarifications

- Exact older deployed build, including whether all riders use the same version; user may provide this later. Explicit assumptions and isolated synthetic data are authorized for testing in the meantime.
- Router objective is now confirmed: shared rider preferences, with travel-time savings only a bonus. Allowable detour and validation criteria remain undecided.
- Whether personal stops, wrong turns and map errors should be excluded from preferred-route demonstrations or modeled as separate purposes. Recommendation: keep them distinct rather than rewarding every deviation.
- Whether historical traffic is archived outside the inspected app/export.
- Final collection size and temporal coverage. More GPS points do not automatically create independent preference examples.

No app change is assumed necessary to begin a feasibility audit of the existing data. If that audit exposes information that cannot be reconstructed, propose the smallest collection/export change with its rationale before altering the app.
