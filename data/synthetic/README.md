# Synthetic rider-preference fixture

`rider_preference_fixture.json` is a small, hand-designed dataset for data-contract checks and later model/routing smoke tests. Every scenario is fictional and marked `is_synthetic: true`. It contains no real participant information and uses a toy local coordinate system, not Taft Avenue or an OSM extract.

This is a proposed normalized fixture, **not a drop-in Firestore CSV export or a finalized production schema**. Recorded paths are represented by known toy edge IDs to isolate preference and temporal logic. It does not test real GPS map matching, traffic access, model accuracy or real rider generalization.

## Contents

- Six nodes and seven directed edges, including one road forbidden to motorcycles.
- Two feasible paths between the same origin and destination: a main path with 60 seconds of fictional travel time and a side path with 95 seconds.
- Nine scenarios covering intentional preference for the slower path, a preference reversal under different context, adherence, a wrong turn, a personal stop, missing traffic, incomplete trajectory coverage, an unrecoverable original route, and traffic received too late to use.
- Explicit expected label/quality handling for each scenario. These are test assertions, not model inputs or learned predictions.

The slower-path preference is deliberate: the agreed objective is to learn rider preferences; travel-time savings are only a bonus. All edge times, geometry, reasons and conditions here are invented. The fixture does not encode a general rule that slower routes are better.

## Field contract

| Field | Meaning |
| --- | --- |
| `graph.nodes` | Toy IDs, local positions in metres and illustrative intersection features. |
| `graph.edges` | Directed `(u, v, key)` edges with fictional length/time and motorcycle access. |
| `candidate_paths` | Hand-specified comparison paths in this test, not evidence of what real riders considered. |
| `input.route_history` | Route IDs, edge sequences, route type and generation time, relative to the scenario start. |
| `input.decision_time_s` | Toy time at the actual choice; not a claim that real detection timestamps identify the turn precisely. |
| `input.event` | Optional deviation report referencing a route, with detection time and survey reason/submission time. |
| `input.observed_edge_ids` | Complete toy observed path, or null when observation coverage is insufficient. |
| `input.traffic_history` | Fictional observations with observation and availability times; an empty list means unknown. |
| `expected.prior_route_id` | Suggested route recoverable before the decision, or null. |
| `expected.deviation_label` | Observed follow/deviate label, or null if it cannot be established. |
| `expected.preference_pair` | Preferred and rejected candidate paths when a usable intentional choice is stipulated; otherwise null. |
| `expected.usable_traffic_count` | Records observed and available at or before the decision. |
| `expected.quality_flags` | Expected missingness/exclusion/reconstruction annotations. |

All times ending `_s` are seconds relative to each independent scenario's start. They must not be combined into one multi-trip historical time series. There are no train/validation/test splits: nine designed scenarios are not a statistical training dataset or sufficient temporal supervision for LSTM.

## Assumptions for testing inconsistent old-build data

1. A reported route ID may refer to the reroute created after the deviation. Recover the prior recommendation only from unambiguous same-ride history before the decision; retain the original ID as provenance.
2. Missing traffic is unknown, not zero congestion. Observations arriving after a decision are unavailable to the prediction at that decision.
3. Complete route adherence provides a follow label. It does not establish preference over every road the rider did not take.
4. Wrong turns and personal stops remain observable deviations, but are excluded from this fixture's ordinary destination-route preference pairs. A future purpose-specific model could handle them separately.
5. Incomplete trajectories or missing original recommendations produce unavailable labels rather than invented real-world preferences.
6. Survey reasons are offline supervision. They are not inputs required from a future rider at routing time. `expected` values, the chosen path and post-event data must not be used as pre-choice features.
7. Motorcycle-forbidden roads cannot be recommended even if their fictional travel time is shorter.

A successful integrity check means these files are internally consistent. A later smoke test could verify that a model can fit the designed examples or that a router respects supplied costs; neither result demonstrates learned real-world tacit knowledge. Real validation must use separately held-out observed rider data.

See `VALIDATION.md` for checks performed when preparing this fixture. The executable prototype and run commands are in the repository root `README.md`. Original rider exports were not modified.
