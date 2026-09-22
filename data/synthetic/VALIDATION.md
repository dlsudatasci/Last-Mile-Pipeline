# Synthetic fixture integrity check

Checked: 2026-09-21, using Python standard-library JSON parsing and explicit assertions.

Result: **PASS ? 211 integrity assertions across 9 scenarios.**

Validated:

- Synthetic provenance, unique IDs and unique directed edge keys.
- Edge endpoints, positive lengths/times and consistency with toy coordinates.
- Directed path continuity, common comparison endpoints and motorcycle access.
- Route history order and reconstruction from information preceding each decision.
- Detection/survey ordering and references to post-event reroutes.
- Traffic observation/availability cutoffs, including late-arriving data.
- Expected labels, unavailable labels and intentional-choice preference pairs.
- Wrong-turn/personal-stop exclusion from the supplied preference comparisons.
- No preference against unseen alternatives inferred from adherence alone.
- A slower-but-preferred case: side path 95 seconds versus main path 60 seconds.

These are fixture checks, not tests of a model or routing implementation. No model was trained and no real-world performance was measured. Real data exports were not edited.
