# Confirmed Scope and Current Stage
- Primary objective: train STGAT-LSTM to learn shared motorcycle-rider route preferences/tacit knowledge, then let a routing algorithm use the trained outputs to recommend routes.
- Improving travel time is not the goal; any time saving is a bonus. Evaluate learned rider preferences and feasible routing first. Travel time may be reported as a secondary outcome, not a required optimization target or prediction head.
- Working study area: Taft Avenue within the City of Manila, provisionally including a 500 m surrounding study corridor and a 1 km buffer for the routing graph. These are assumptions to revisit against collected routes.
- The mobile app is for collection. Riders currently use an older build; its precise version may be clarified later. Explicit assumptions and isolated dummy data are authorized for testing while inconsistencies are resolved.
- Preserve original observations. Mark unknown or inconsistent fields and any reconstructed values; do not present dummy values as real rider evidence.
- Current stage: coding has started. The first executable STGAT-LSTM, GAT/GCN baselines, learned-cost routing, synthetic smoke tests, aggregate real-export audit, route/GPS geometry screening, OSM compatibility/coverage audit, directed HMM/Viterbi map matching, conservative common-OD candidate extraction, the real OSM-to-model feature contract, and the real learned-cost routing adapter are implemented. Reviewing extracted paths, approving final real preference pairs, real-data training, final routing validation and the website remain future work.
- See `README.md` for commands, implemented behavior and limitations.
- See `REVIEW_NOTES.md` for decisions and pending questions, `APP_DATA_REVIEW.md` for collection-code findings, and `data/synthetic/README.md` for the dummy fixture.
- These confirmed points supersede tentative objectives or outputs suggested below.

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
- Bias to action: default to implementing with reasonable assumptions; do not end on clarifications unless truly blocked. Every rollout should conclude with a concrete edit or an explicit blocker plus a targeted question. 

# Architecture, Boundaries & Execution Workflow
- Bias to Action: Gather context, plan, implement, and verify autonomously.
- Do not stop at a partial fix; carry changes through implementation and verification.
- Every code change must run matching tests via `exec_command`.
- Goal condition: 100% pass rate on affected test suites.
- If an algorithm is being closely adapted from a specific research paper or open-source module, include a clean markdown comment block at the top of the newly planned file referencing the source DOI or URL for engineering transparency. 

# Overview Context
- Focusing on Metro Manila
- We are developing a spatiotemporal GAT-LSTM neural network integrated with a dynamic routing algorithm. We need to use state-of-the-art models and routing algorithms, and piggyback off them further.
- Basically, we have a mobile app that collects driver deviation data from a generated route. These data include, but are not limited to, where they deviated, the time, and the reason for the deviation. They will be asked to complete a short survey. If traffic was the reason, what was the severity, etc? Our high-level goal is for GAT to learn about riders' tacit knowledge, or street smarts, which we collect from them. Afterward, use/create a routing algorithm.
- C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\ExportLMD: these are the files containing the data we received from the riders for the GAT network. Note that these are not yet final, and we are targeting a specific area.
- OSM Road Network and data that is not limited to the type of amenities and their counts across nodes.Not limited to the mentioned, but other details a route-generating app should collect.
- C:\Users\Vince\Downloads\THESIS_APP\T3-NEW-MODEL\OSM: these are the files and code used to extract OSM data for the GAT network. I am not sure if this needs further processing, so please check its compatibility with the model that we have/going to make.
- I believe what we are trying to do must include both node- and edge-level predictions, right? Which then will have Sub-graph and Structural embeddings, and Graph Level as well? You may clarify with me. What properties would the nodes and edges have?
- There are a lot of nitty-gritty details for GAT, so please do not hesitate to consult with me and suggest as well for the specific details

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
- I am not a machine learning expert. So you should act as an expert AI Engineer. 
- Another important reminder: if you need any clarification on anything, please tell me or ask me. Let me know every step of the way. Perhaps you may ask if I have a certain resource already. Ask me, please.
- What I need is Real-Time traffic data as well. I’m not sure how to include and integrate this, but I do have Mapbox, if ever so help me.
- The current real decision artifact uses a common local route-choice unit and contains 5 conservative followed candidates plus 3 strict deviated candidates from 6 riders. It remains review-required and too small for meaningful rider-disjoint evaluation.
- A shared STGAT-LSTM multitask model now trains a binary follow/deviate head together with the learned edge-preference-cost head. The synthetic end-to-end checkpoint fits all 5 eligible decision labels and all 4 preference pairs in sample; this verifies training mechanics only. Real training remains gated on manual approval of the 8 reconstructed decisions.
- The eight reconstructed decisions are now approved and the real-data smoke checkpoint has been trained. It reaches 7/8 in-sample classification and 3/3 in-sample preference ranking; rider-disjoint evaluation and additional collection remain required.
