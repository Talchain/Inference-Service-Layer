# ISL Full Scientific Capability, Validity, Performance & Reliability Audit

**Lane:** Track S · **Date:** 2026-07-05 · **Mode:** read-only, evidence-led
**Repo:** `Talchain/Inference-Service-Layer` · **SHA audited:** `4f5bf770d15f0c12ac502f2a07fc4b1daad21d02` (== `staging`, local and origin)
**Method:** static trace of the live code path at the audited SHA, plus in-process empirical checks run in an isolated venv (no server, no network, no staging/production traffic, zero repo modifications — `git status` clean before and after).

**PLoT caveat (per brief clarification):** no PLoT checkout was available in this environment. Every statement about what PLoT sends or consumes is marked **[inferred from contract/docs]**, not verified against PLoT code.

---

## A. Executive verdict

**Usable with specific Tier 0 blockers.**

The live scientific path — `POST /api/v1/robustness/analyze/v2` with `response_version=2` — is a coherent, deterministic-by-design, well-guarded linear-SCM Monte Carlo engine. Its request validation, seed discipline, error shapes, warning/critique provenance, and security middleware are genuinely good for a PoC-stage service, and most of the "known issue" hypotheses in the brief turned out to be already fixed or honestly labelled in the current code.

However, the audit **verified five defects and several honesty gaps** that should be fixed or pinned before ISL outputs are treated as trustworthy inputs to coaching claims:

1. **Internal inconsistency between `probability_of_goal` and constraint probabilities** — the same question ("P(revenue ≥ x)") gets two materially different answers in one response (0.22 vs 0.13 in the reproduction) because auto-scaled noise is applied to goal samples but not to constraint samples. *(Verified, high)*
2. **`seed_used` in the V2 envelope can be wrong** — when the graph contains organisational nodes and no client seed, the router reports a seed computed from the unfiltered graph while the analyzer seeds its RNGs from the filtered graph (2714408004 vs 947429003 in the reproduction). The "reproducibility contract" field is then false. *(Verified, high for scientific credibility)*
3. **Cyclic graphs are silently analysed on the default (`response_version=1`) path** — no error, no critique, plausible-looking garbage. Cycle blocking lives only in the V2-enhanced validator. CEE is documented as still on V1. *(Verified, high)*
4. **`/analyze/unified` is broken for v2 payloads** — every v2-schema request 400s with a leaked Pydantic internal (`input_value=Query(False)`), because the endpoint calls the v2 handler as a plain function and FastAPI `Query`/`Header` default objects flow in as values. *(Verified, medium — live route, apparently unused)*
5. **Duplicate edges double-count** — two identical `x→y` edges double the effect (0.80 vs 0.40 in the reproduction); no validator rejects duplicates. *(Verified, medium)*

Plus the honesty gaps: negative EVPI emitted unlabelled (6 of 14 seeds negative, to −0.073, at 300-sample resolution where ±0.06 is noise); factor-sensitivity bootstrap count adapts on **wall-clock time**, so the determinism guarantee is latent-fragile under load; and there is **no `poetry.lock` anywhere in git history** while CI, the Dockerfile, and the security scans all assume one — builds are unpinned and the Dockerfile as committed cannot build.

None of this undermines the core engine's arithmetic (single-edge analytic checks pass exactly; same-seed runs reproduced bit-identically in this environment). The verdict is "usable" because the defects are narrow and fixable; it is "with blockers" because two of them (goal/constraint inconsistency, seed reporting) sit directly on the fields Olumi would quote to users.

---

## B. Ground truth preflight

| Item | Value |
|---|---|
| Working dir / repo root | `/home/user/Inference-Service-Layer` |
| Branch | `claude/isl-capability-audit-klrytg` @ `4f5bf77` — identical to `staging` (local and `origin/staging`) |
| Remotes | `origin` → `Talchain/Inference-Service-Layer` |
| `git status` / stash | clean, no stash — before **and after** audit tooling setup |
| Deploy relationship | `staging` is the working deploy branch (CLAUDE.md, PR CI runs on push to staging). `deploy-production.yml` triggers on `main`/tags. No `render.yaml`/`Procfile` in repo; `runtime.txt` pins Python 3.11.9; Render deployment configured outside the repo **[inferred — Render env vars like `RENDER_GIT_COMMIT` are referenced in code]** |
| Python / tooling | Python 3.11 (runtime.txt 3.11.9), Poetry (`pyproject.toml`), pytest, mypy, black, ruff. **No `poetry.lock` — never existed in git history** (`git log --all -- poetry.lock` is empty) |
| App entrypoint | `src/api/main.py` (FastAPI app; uvicorn `src.api.main:app`) |
| Route registration | `src/api/main.py:686-717`. Active: `health`, `metrics`, `robustness`. **19 routers explicitly commented out** ("Disabled for pilot — orphaned endpoints not used by PLoT integration") |
| Health endpoints | `/health`, `/ready`, `/health/services`, `/health/circuit-breakers`, `/cache/stats` (all GET, public) |
| Live analysis endpoints | `POST /api/v1/robustness/analyze`, `.../analyze/v2`, `.../analyze/unified` |
| Secrets | none committed (pattern scan of py/yml/json/md/sh clean); `.env.example` only; scripts read `ISL_API_KEY` from env |
| Generated artefacts | `openapi.json` committed — **matches** the live registered routes (no drift); `htmlcov/` generated only in my sandbox venv run, not in repo |
| CI | `.github/workflows/pr-ci.yml` (mypy + full pytest excluding `_archived`/`_quarantined`, on PRs to main/staging and pushes to staging), `security.yml` (safety, pip-audit, bandit), `deploy-production.yml` (main only, 80% coverage gate), plus openapi/config validation workflows |

Registered route inventory (empirical, from importing the app at the audited SHA):

```
GET  /openapi.json /docs /docs/oauth2-redirect /redoc          (public, docs)
GET  /health /ready /cache/stats /health/services /health/circuit-breakers   (public)
GET  /metrics                                                   (public, Prometheus)
POST /api/v1/robustness/analyze          (FACET v1)
POST /api/v1/robustness/analyze/v2       (dual uncertainty — THE live path)
POST /api/v1/robustness/analyze/unified  (schema auto-detect)
```

Commands used: `git rev-parse/branch/status/stash/log`, `Read`/`Grep` over `src/`, `tests/`, `.github/`, in-process Python checks via a scratchpad venv (`audit_checks.py`), `mypy src/`, and a targeted pytest subset (`test_determinism.py`, `test_facet_robustness.py`, `test_constraint_analysis.py`, `test_response_v2.py`, `tests/contract` → **144 passed, 2 skipped**).

---

## C. Capability inventory

Classification key: **LIVE-PLOT** = live and used by PLoT [inferred from contract/docs]; **LIVE-UNUSED** = registered but no known consumer; **DORMANT** = implemented + tested but not registered; **DEAD** = no importer; **ARCHIVED** = quarantined under `_archived`.

### C.1 Live surface

| Capability | Route | Code | Tests | Status | Assessment |
|---|---|---|---|---|---|
| Dual-uncertainty robustness (v2.2 schema) | `POST /api/v1/robustness/analyze/v2` | `src/api/robustness.py:232` → `src/services/robustness_analyzer_v2.py` | extensive (unit + integration + contract; 144 live-path tests passed in this audit) | **LIVE-PLOT** (V2 response format; V1_DEPRECATION_TIMELINE.md says PLoT migrated to V2) | Core product path. Scientifically coherent linear SCM MC; defects listed in §A/§K |
| FACET region robustness (v1 schema) | `POST /api/v1/robustness/analyze` | `src/api/robustness.py:118` → `robustness_analyzer.py` (region-based, uses `counterfactual_engine`) | `test_facet_robustness.py` | **LIVE-UNUSED** [inferred — PLoT uses v2] | Different algorithm (intervention-region exploration). Keep or retire consciously; it drags `counterfactual_engine`/`causal_validator` into the live import graph |
| Unified auto-detect endpoint | `POST /api/v1/robustness/analyze/unified` | `src/api/robustness.py:974` | thin | **LIVE-BROKEN for v2 payloads** (verified: always 400 with leaked `Query(False)` detail); v1 payloads work | Fix or remove. Currently a misleading route |
| Health/readiness/cache/circuit-breaker status | 5 GET routes | `src/api/health.py` | smoke/integration | LIVE | Public by design |
| Prometheus metrics | `GET /metrics` | `src/api/metrics.py` | unit | LIVE | Public (see §I) |

### C.2 Dormant surface (implemented, not registered — 19 routers, 48 endpoints)

All imports and `include_router` calls are commented out in `main.py` with an explicit rollback note. Per brief clarification these are classified and risk-ranked, not fully validated.

| Router (endpoints) | Backing service(s) | Tests exist | Tier suggestion | Notes |
|---|---|---|---|---|
| `causal.py` (12) | causal_discovery_engine, counterfactual/batch engines, conformal_predictor, causal_transporter, causal_representation_learner, sequential_optimizer, parameter_recommender, advanced_validation_suggester | yes (unit) | Tier 2 | The largest dormant block: discovery, counterfactuals, conformal prediction, transportability. Some (conformal, discovery) are strategically interesting; none production-validated |
| `identifiability.py` (3) | identifiability_analyzer (y0), confounding_sensitivity | yes (`test_identifiability_*`) | Tier 1–2 | Y₀-based identifiability exists **in code with tests** but is NOT live. `ENABLE_IDENTIFIABILITY_ANALYSIS=True` in settings is misleading — flag exists, route doesn't |
| `analysis.py` (2) | sensitivity_analyzer, continuous_optimizer | yes | Tier 2 | Enhanced sensitivity variants |
| `decision_robustness.py` (1) | decision_robustness_analyzer | yes | Tier 2 | Overlaps live robustness; consolidate or delete |
| `outcomes.py` (3) | outcome_logger | yes | Tier 1 | Outcome logging — needed for calibration loop later |
| `phase4.py` (4) | sequential_decision, conditional_recommender | yes | Tier 2 | Sequential decisions |
| `preferences.py` (2) | preference_elicitor, belief_updater, user_storage | yes | Tier 2 | |
| `validation.py` (4) | advanced_validator, coherence_analyzer, correlation_validator, feasibility_checker | yes | Tier 2 | |
| `teaching.py`, `team.py`, `explain.py`, `cee.py`, `batch.py`, `aggregation.py`, `dominance.py`, `risk.py`, `threshold.py`, `utility.py` (1–4 each) | one service each | mostly yes | Tier 2 / retire | Product-fit unclear; several look like earlier product directions |
| `deliberation.py` (2) | `_archived/habermas/*` | archived tests | **retire** | Router references an archived service — cannot be re-enabled as-is |

### C.3 Dead / archived code

| Item | Evidence | Recommendation |
|---|---|---|
| `sbc_validator.py`, `causal_validator_enhanced.py`, `robustness_visualizer.py` | no importer anywhere in `src/` | DEAD — delete or move to `_archived` |
| `src/services/_archived/habermas/` | archived; still type-checked — my mypy run surfaced import-not-found errors originating here | Move out of `src/` or exclude from mypy to keep the type gate clean |
| `tests/_archived` (156 tests), `tests/_quarantined` (0) | excluded from CI | Fine; prune eventually |
| `benchmarks/PERFORMANCE_REPORT.md` | dated 2025-11-20; benchmarks `/api/v1/causal/validate` — an endpoint disabled since | **Documentation drift** — stale performance evidence |
| `README.md` capability table | lists Y₀ identifiability, counterfactuals etc. as live endpoints (README.md:63,72,102) | **Documentation drift / overclaim** — describes the dormant surface as available |

**Documented-only capability:** none found beyond the README/report drift above — i.e. no claims of capabilities with zero code behind them; the failure mode here is the reverse (large implemented-but-dark surface).

---

## D. Live path trace — `POST /api/v1/robustness/analyze/v2`

### D.1 Ingress (middleware, outermost→innermost)

`TracingMiddleware` (X-Trace-Id; trace = X-Request-Id or generated) → `ObservabilityMiddleware` (service/build headers, payload hashing) → `MemoryCircuitBreaker` (reject >85% memory) → `RequestTimeoutMiddleware` (60s default) → `RequestSizeLimitMiddleware` (`MAX_REQUEST_SIZE_MB`) → security headers → GZip → CORS (explicit origins) → `RateLimitMiddleware` (Redis sliding window, in-memory fallback) → `APIKeyAuthMiddleware` (constant-time compare; startup `RuntimeError` if no keys and auth not explicitly disabled — `main.py:259-263`).

### D.2 Request parsing — `RobustnessRequestV2` (`src/models/robustness_v2.py:592`)

* `graph`: 1–50 nodes, ≤200 edges; unique node IDs; edges must reference nodes; self-loops rejected. **No cycle check, no duplicate-edge check at this layer.**
* `EdgeV2`: `from` → `from_` alias (`populate_by_name`); `exists_probability` optional → **defaults 0.8** with `EXISTS_PROBABILITY_DEFAULT` inference warning; `strength.mean` NaN/Inf rejected, clamped to [−1,1] with `STRENGTH_MEAN_CLAMPED` warning; `strength.std` must be **> 0.001** (422 otherwise); optional `edge_type: directed|bidirected`.
* `NodeV2`: `kind` is a **free-form string** (the `NodeKindV2` enum exists but is not applied); `intercept` (default 0.0), `epsilon_std` (default 0.0, ≥0), `observed_state` (finite-checked value + CEE passthrough fields), CEE passthrough `category`/`factor_type`.
* `options`: 1–10, unique IDs, `interventions: Dict[str, float]` must reference existing nodes (any kind — no kind restriction).
* `n_samples` 100–10,000 (default 1000); `seed` int|str|None (strings hashed deterministically via SHA-256); `confidence_level` 0.5–0.99; `goal_threshold` finite; `goal_constraints` ≤20, operators **`>=`/`<=` only**, `value` finite (legacy `threshold` coerced); flags `include_e_values`, `include_voi`, `include_path_decomposition`.
* **All models `extra="ignore"`** — unknown fields silently dropped (explicitly documented as the CIL cross-service contract).
* Pydantic 422s are normalised to the single `ISLV2Error422` shape for this endpoint (`main.py:497-609`).

### D.3 Handler (`robustness.py:279-971`)

1. Response version: header `X-ISL-Response-Version` beats query `response_version`; **default 1** (`DEFAULT_RESPONSE_VERSION`).
2. Request ID: body > header > generated `isl-…`; sanitised (`sanitize_request_id`, 128-char truncation).
3. **Complexity guard**: `n_samples × n_nodes × n_edges ≤ 10M` (env-overridable) → 422 with actionable suggestion.
4. **V2-enhanced path only**: `RequestValidator.validate()` — empty graph, **cycle detection (DFS)**, disconnected components (warning), |strength.mean|>3 (warning), goal exists, options non-empty, interventions non-empty + targets exist, **effective path from every option to goal** (PathValidator with 1e-6 thresholds), identical-options detection. Blockers → 422 `ISLV2Error422` with critiques.
5. `effective_seed` computed as `request.seed or compute_seed_from_graph(request.graph)` — **from the unfiltered graph** (`robustness.py:459-461`); recorded on the envelope with `seed_source`.

### D.4 Analyzer (`robustness_analyzer_v2.py:650`)

1. **Safety-net filter**: nodes of kind `decision|option|constraint` (case-insensitive) and incident edges removed; goal must survive; filtered intervention targets logged.
2. **Seed**: `request.seed if not None else compute_seed_from_graph(filtered graph)` — SHA-256 of canonical node/edge JSON. RNG streams (NumPy PCG64 via `SeededRNG`): `seed` edges, `seed+1` factors, `seed+2` auto-noise, `seed+3` epsilon noise, `seed+100/101` EVPI baseline, per-factor/per-edge SHA-256-derived sub-seeds for EVPI and marginal-switch.
3. **Parse warnings** surfaced: `STRENGTH_MEAN_CLAMPED`, `EXISTS_PROBABILITY_DEFAULT`; plus `CONSTRAINT_NODE_DEFAULT_BASE` and `ROOT_NODE_DEFAULT_VALUE` detection with critiques/warnings.
4. **MC loop** (`_run_monte_carlo`): per sample — Bernoulli edge existence, truncated-Normal strength (rejection sampling in [−1,1], fallback clamp after 100 attempts), factor sampling (normal/uniform/point_mass around `observed_state.value`, unknown distribution **raises**), SCM evaluation per option, winner tracking with split-tie counting and RNG tie-breaking. Constraint node values captured per sample when `goal_constraints` present.
5. **SCM** (`SCMEvaluatorV2.evaluate`): Kahn topological order (cycle → **warning + arbitrary order fallback**); per node: `value = base + intercept + Σ(parent_value × strength)`; base priority `factor_values > base_values > observed_state.value (roots only) > 0.0`; interventions override the whole equation; `epsilon_std>0` adds N(0,ε) **and clamps that node to [0,1]** (values are unbounded when ε=0 — an asymmetry).
6. **Auto-scaled noise** (`_apply_auto_scaled_noise`): post-MC, goal-node samples only, only if goal kind ∈ {outcome, risk}; noise std = sample std ("match unexplained to explained variance" — Neil Bramley heuristic, documented as PoC-status); `auto_noise_applied` disclosed in metadata and on the V2 envelope. **Constraint samples are not noised** (→ finding T0-1).
7. **Option results**: mean/std/median/percentile-interval (2.5/97.5 at default 0.95 — correctly labelled a prediction interval in code), raw samples retained; `probability_of_goal` = fraction ≥ `goal_threshold` (post-noise); constraint analysis per option: per-constraint P, joint P, pairwise conditionals P(Cj|Ci) — all from the **same satisfaction matrix** (same sample set), near-miss diagnostics, binding flag ([0.4,0.6]).
8. **Sensitivity** (edge-level): forced-on vs forced-off existence contrast and ±1σ magnitude contrast, 100 samples each, **computed against `options[0]` only** (order-dependent reference), elasticity vs epsilon-guarded baseline, clamped ±100.
9. **Factor sensitivity**: deterministic ±δ contrast on mean-edge-config, true elasticity with epsilon guards, `zero_reason` taxonomy, structural influence (path-strength products), bootstrap stability (10 or **20 iterations decided by wall-clock <100ms**), CV-based `attribution_stability`, `rank_flip_rate`.
10. **Robustness**: `recommendation_stability` = top win share, **× penalty `max(0.1, 1−0.05·n_defaulted_roots)`** (disclosed via `stability_penalty_factor`); `is_robust` = stability ≥ 0.7; `confidence = min(0.99, stability·(1−1/√n))` (heuristic); fragile edges = max |edge elasticity| > 0.1; alternative winners from bottom-25% strength samples; marginal switch probability via 100 isolated-edge samples.
11. **Optional**: E-value analogue (binary-search strength perturbation to flip winner; **2s wall-clock budget → whole block silently omitted when exceeded**); EVPI (see §F.7); path decomposition (top-3 signed simple paths, 20k path budget, deterministic truncation).
12. Recommended option = max win count; `recommendation_confidence` = its win share.

### D.5 Egress (V2)

Router converts V1→V2: samples cleaned via `validate_mc_samples` (non-finite → median + `NUMERICAL_INSTABILITY` critique), true p10/p50/p90 from cleaned samples (or `percentiles_source="unavailable"` + nulls), option `status` computed/partial/failed by valid ratio (0.8), degenerate-outcome critique, robustness level mapping (high/moderate/low/very_low), factor confidence = bootstrap blend (see §F.4) with `confidence_source`, provenance echo (`value_source`, `value_defaulted`), conditional winners, `factor_evpi` passthrough, `auto_noise_applied`, `seed_used`/`seed_source`, request echo (hashed goal ID), `X-Request-Id` + `X-Processing-Time-Ms` headers. `model_dump(by_alias=True, exclude_none=True)` with an `inference_warnings` always-present safety net. Unexpected exceptions → sanitised 500 envelope (`INTERNAL_ERROR` critique, no stack traces).

**Egress gaps found:** the V2 envelope (`ISLResponseV2`) has **no `sensitivity` field** (edge-level results are computed then dropped except via fragile edges/E-values) and **no `path_decomposition` field** — `include_path_decomposition=true` on the V2 path computes the decomposition and then discards it (V1 envelope carries it; V2 doesn't).

---

## E. Data model and contract alignment

| Contract expectation | ISL implementation | Evidence | Verdict | Owner | Severity |
|---|---|---|---|---|---|
| `from` → `from_` translation | Alias + `populate_by_name`; serialised back `by_alias` | robustness_v2.py:283; robustness.py:510,944 | aligned | — | — |
| Organisational nodes (`decision`/`option`/`constraint`) excluded from inference | Filtered as analyzer safety net (with warning); **not rejected**; PLoT expected to strip first [inferred] | analyzer:74,91-117 | aligned (defence-in-depth) but see seed defect | ISL | high (via T0-2) |
| Node kinds validated | `kind: str` free-form; enum unused; unknown kinds participate in inference | robustness_v2.py:360 | misaligned (loose) | ISL | low |
| `exists_probability` optional, default | default 0.8 + `EXISTS_PROBABILITY_DEFAULT` warning | robustness_v2.py:285-289,323-337 | aligned, disclosed | — | — |
| `strength.mean` ∈ [−1,1] | clamped + warning; NaN/Inf rejected | robustness_v2.py:79-122 | aligned | — | — |
| `strength.std < 0.001` handling | **rejected** (422, `gt=0.001`) — ISL-enforced; PLoT must pre-floor [inferred] | robustness_v2.py:83 | aligned (ISL side) | PLoT | — |
| Negative effects | signed means propagate; truncated-normal sampling preserves sign; verified analytically | check 10 | aligned | — | — |
| Duplicate edges | **not rejected; double-counted** (verified: 0.80 vs 0.40) | check 11; no validator | **misaligned** | ISL (validator) + PLoT (shouldn't send) | medium |
| `observed_state.value` | optional; used as base **for root nodes only**; non-root observed values ignored unless a `ParameterUncertainty` exists | analyzer:503-512 | partially aligned — silent for non-roots | ISL/PLoT contract | medium |
| `state_space.range` / normalisation | not a field; ISL assumes pre-normalised values; `raw_value`/`cap` are passthrough only | robustness_v2.py:159-160 | aligned (unit-agnostic) — scale correctness wholly PLoT-owned | PLoT | — |
| Factor categories | `category`/`factor_type` passthrough, not consumed | robustness_v2.py:381-393 | aligned | — | — |
| Categorical/boolean variables | not supported; floats only; no rejection of e.g. 0/1-encoded booleans (treated numerically) | InterventionOption; §F.9 | honest limitation, undocumented in response | ISL docs | low |
| Interventions numeric, factor-targeted | numeric enforced; **target kind unrestricted** (goal/outcome intervention accepted; path validator then judges reachability) | robustness_v2.py:507; request_validator | partially aligned | ISL | low |
| Missing intervention = baseline | yes — un-intervened nodes follow structural equations (baseline semantics implicit) | analyzer:486-533 | aligned | — | — |
| `goal_threshold` and `goal_constraints[]` both supported | yes; both computed when both present; **but from different sample sets (noised vs un-noised)** | check 3 | **misaligned internally** | ISL | **high** |
| Constraint operators | `>=`, `<=` only; `>`,`<`,`==`,`!=` rejected at 422 | robustness_v2.py:538 | aligned if contract says so; narrower than brief's list | Neil/PLoT to confirm | low |
| Constraints on unknown nodes | rejected (Pydantic) | robustness_v2.py:747-757 | aligned | — | — |
| Per-constraint + joint from same samples | yes, single satisfaction matrix; pairwise conditionals implemented | analyzer:3314-3404 | aligned | — | — |
| Temporal constraints | absent — no temporal fields exist; assumed PLoT-filtered [inferred] | — | aligned (delegated) | PLoT | — |
| Response: percentiles | true p10/p50/p90 from samples; `percentiles_source` sentinel; nulls when unavailable | robustness.py:637-667 | aligned (recent CIL fix) | — | — |
| Response: seed/request identity | `seed_used`+`seed_source`+`seed_hash_version`; request ID sanitised & echoed in header | multiple | aligned **except T0-2 divergence case** | ISL | high |
| Response: `sensitivity` (edge-level) | computed, **not on V2 envelope** | response_v2.py:612-710 | ISL sends nothing; consumer can't use | ISL contract | low-medium |
| Response: `path_decomposition` | computed when flagged, **dropped on V2 envelope** | response_v2.py; robustness.py | **ISL response-contract gap** | ISL | medium (if flag used) |
| Unknown request fields | ignored everywhere (`extra="ignore"`, deliberate CIL promise) | model_config throughout | aligned but drift-risk acknowledged | ISL+PLoT | medium (monitor) |
| Errors normalisable by PLoT | single 422 shape (`ISLV2Error422`) for both Pydantic and validator blockers; sanitised 500 envelope; Olumi Error Schema v1.0 elsewhere | main.py:497-609 | aligned | — | — |

---

## F. Scientific validity assessment

### F.1 Monte Carlo engine — **sound core, two honesty caveats**
PCG64 with strictly separated, documented seed streams; no global RNG use on the live path; seed from request or canonical graph hash. **Empirically verified:** same seed → bit-identical summary outputs across two runs; different seed → different outputs; variance behaves; single-edge graph reproduces the analytic product exactly (0.2·0.5=0.0998, 0.8·0.5=0.3991); zero-strength graph → ~0 means. Percentile/interval logic is correct and correctly renamed a *prediction interval*. Caveats: (a) bootstrap iteration count (10 vs 20) is **wall-clock-adaptive** — same seed can yield different `elasticity_std`/`confidence` under different machine load (method is at least labelled `bootstrap_10/20`); (b) E-value block silently vanishes if >2s. Both violate strict "same input+seed ⇒ same response" in the tail fields, not in the headline numbers.

### F.2 SCM semantics — **coherent linear model; root/intercept question is real but disclosed-ish**
Linear additive SCM with topological propagation, dual uncertainty sampled per edge (existence AND strength — both genuinely sampled). Roots: `observed_state.value + intercept` are **summed** (verified: 0.5+0.3→0.8). Whether that double-counts depends on how PLoT populates `intercept` vs `observed_state` **[unverifiable without PLoT — Neil/PLoT question]**. There is no guard against divergent roots (`intercept != observed_state.value` both set); no stripping either, so valid intercept-only baselines survive. Non-root noise exists via `epsilon_std` (off by default); root noise via `parameter_uncertainties`. The [0,1] clamp applies **only when epsilon noise fires** — inconsistent bounding across configurations. Cycles: blocked on V2-enhanced path; **silently tolerated on V1 path** (verified) — the Kahn fallback produces deterministic garbage. `Normal(mean,std)` for edges is properly truncated to [−1,1]; factor sampling (normal/uniform) is **unbounded** — intentional for user-scale values, but means "normalised [0,1] world" is not enforced anywhere.

### F.3 Normalisation and units — **fully PLoT-owned, ISL is honestly unit-agnostic**
No denormalisation, no range logic, no scale metadata beyond echoing `unit`/`raw_value`/`cap`. ISL cannot detect wrong-scale interventions or constraints (a `price=49` intervention in a [0,1]-normalised graph computes nonsense silently). This matches the intended ownership split, but ISL provides **no `input_scale_assumption` marker** in responses; adding one is a cheap honesty win.

### F.4 Sensitivity — **method real, reference-option caveat, confidence now honest**
Edge sensitivity = forced-existence and ±1σ magnitude contrasts (100 samples each) — a genuine perturbation method, direction preserved, epsilon-guarded, clamped for display only (raw preserved). **Both edge and factor sensitivity are computed against `options[0]` only** — sensitivity, fragile edges, and bootstrap stability all depend on request option order; nothing in the response says so. Factor confidence: **not hardcoded** — bootstrap CV → category (0.9/0.6/0.3/0.1) blended 70/30 with `1/(1+CV)`, floor-capped structural fallback, all labelled `confidence_source` and `provisional`, thresholds centralised with env overrides and version string. Rank stability exists (`rank_flip_rate`). This is a defensible heuristic honestly labelled — the remaining risk is consumers reading `confidence` as a calibrated probability.

### F.5 Robustness — **operational-defaults construct, disclosed, with one silent modifier**
`recommendation_stability` (win-share of top option) is a meaningful MC quantity; `is_robust ≥ 0.7` and the high/moderate/low/very_low mapping are operational defaults traceable to Decision Model Schema v2.6, not scientific calibration (comments say so). `confidence = min(0.99, stability·(1−1/√n))` is a made-up formula — it monotonically rewards sample count in a way that has no inferential meaning; treat as UX heuristic, should be labelled like factor confidence is. The **defaulted-root stability penalty** (−5% per defaulted root) mutates the reported stability itself; it is disclosed via `stability_penalty_factor`/`defaulted_root_node_ids`, but the field named "recommendation_stability" is then no longer the observed win-share. Fragile edges (>0.1 max elasticity) + bottom-quartile alternative-winner analysis + marginal switch probability are internally consistent and deterministic. No claim firewall exists beyond wording in `interpretation` strings.

### F.6 Constraints — **correct within scope; scope is narrow; one Tier-0 inconsistency**
Per-constraint, joint, and pairwise conditional probabilities all from one satisfaction matrix (correct — correlations between constraints are captured empirically); near-miss/binding diagnostics are sensible. Operators limited to `>=`/`<=`. Undefined conditionals (P(Ci)=0) correctly omitted. **Tier 0:** constraint samples bypass auto-noise while `probability_of_goal` includes it — verified 0.22 vs 0.13 for the identical node/threshold/option. Also `CONSTRAINT_NODE_DEFAULT_BASE` fires appropriately when a constraint targets an unmodelled non-root (defaults to 0 base).

### F.7 VOI/EVPI — **a labelled proxy, not EVPI; negative values emitted raw**
`_compute_evpi` fixes the recommended option as policy and compares P(win) (or P(joint constraints)) with a factor's uncertainty collapsed to its mean, vs baseline — at **capped 500 samples** regardless of main `n_samples` (verified: `n_evpi_samples: 500` with `n_samples=2000`). Three scientific problems: (1) **fixing the policy removes exactly the decision-switching value that defines EVPI** — this is closer to an "uncertainty contribution to win-probability" metric; (2) "perfect information" is simulated as "fix at prior mean", not expectation over revealed values; (3) at 300–500 samples the MC standard error (~±0.03–0.06 on a probability) exceeds many reported values — **verified: 6 of 14 seeds produced negative EVPI (to −0.073), emitted raw** with no below-resolution/not-significant labelling and no clamping. Good: method fields (`metric_type`, `n_evpi_samples`, `current_metric`, `perfect_metric`) are present, so the fix is labelling + naming, not rework. Consumers could currently overclaim ("knowing X is worth +1.3pp win probability") when the number is noise.

### F.8 Identifiability and causal validity — **not on the live path; no overclaim in responses; overclaim in README/settings**
The live path performs **no** identifiability, confounding, adjustment-set, collider, or transportability checking. `edge_type: bidirected` is accepted and skipped in path/E-value logic. A real y0-based identifiability analyzer + confounding sensitivity exists, tested, **dormant** (router disabled). `DiagnosticsV2.identifiability_status` is emitted as `"unknown"` — honest. README (lines 63, 102) and the `ENABLE_IDENTIFIABILITY_ANALYSIS=True` setting claim the capability — **documentation drift/overclaim**, classify identifiability as *implemented-not-registered, partially validated*.

### F.9 Categorical/ordinal/boolean — **absent, mostly safe, silently numeric**
Everything is `float`. Boolean-encoded factors (0/1) flow through linear arithmetic; the epsilon guards exist precisely because of them (constants note "binary factors 0/1"). No one-hot support, no rejection, no response marker that a factor was treated as continuous. Honest-limitation documentation is the right fix; support would be Tier 2.

### F.10 Temporal/dynamic — **absent by design**
No temporal fields, no sequential logic on the live path (`sequential_decision`/`phase4` dormant). Deadlines/temporal constraints assumed PLoT-filtered [inferred]. Classify: absent/delegated (live), dormant (phase4), future (outcome-learning — `outcomes.py` dormant).

### F.11 Calibration — **absent**
No calibration artefacts on the live path; `sbc_validator` (simulation-based calibration) is dead code; conformal prediction dormant. The auto-noise heuristic is explicitly "pending calibration against pilot outcome data". This is the single biggest gap between current state and "science-grounded coaching" — nothing validates that P(goal)=0.72 means anything empirically.

---

## G. Reliability and numerical safety

**Strong:** NaN/Inf rejected at ingress on every numeric contract field (means, stds, observed values, thresholds); JSON cannot smuggle NaN (Starlette serialises with `allow_nan=False`, so an escaped NaN would 500 rather than emit invalid JSON); non-finite MC samples cleaned with critique + validity ratio + status downgrade (computed/partial/failed at 80% valid); epsilon-guarded divisions centralised in `numerical_stability.py`; unknown factor distribution raises (fail-fast); single 422 error shape; sanitised 500s with request-ID correlation; startup fail-closed on missing keys and bad production config; timeouts, size limits, memory circuit breaker; determinism empirically verified (with §F.1 caveats).

**Weak spots (verified or code-confirmed):**
* Cycle fail-open on V1 path (§A-3) — the one true silent-wrong-answer path found.
* `mean`/`std`/`median`/CI in `OutcomeDistribution` are computed **before** sample cleaning; if non-finite samples ever occur, V1 consumers get NaN→500 while V2 consumers get cleaned percentiles alongside NaN-poisoned mean — inconsistent (currently theoretical: no producing path for non-finite samples was found).
* Non-finite sample replacement uses the **median** (spikes density at the median) rather than exclusion — acceptable at <20% but undocumented.
* Silent defaults are generally *not* silent (root-default and constraint-base warnings + penalty) — good pattern; but non-root `observed_state.value` being ignored produces **no warning** at all.
* `/analyze/unified` error path leaks internal Pydantic representation in the 400 detail (verified).
* Edge cases tested empirically: empty graph → 422 critique; unknown intervention target → 422; duplicate options → critique; zero samples impossible (min 100); duplicate edges NOT caught (§A-5).

---

## H. Performance assessment

**Committed benchmark evidence is stale** (`benchmarks/PERFORMANCE_REPORT.md`, 2025-11-20, measures now-disabled endpoints). Fresh single-call, in-process measurements at the audited SHA (this container, sandbox venv, includes sensitivity + bootstrap):

| Scenario | Wall time |
|---|---|
| 3 nodes / 3 edges, 1,000 samples, 2 options, 1 uncertainty | **40 ms** |
| same, 5,000 samples | 130 ms |
| 12 nodes / 20 edges, 5,000 samples, 3 options, 4 uncertainties | **941 ms** |
| same + `include_voi` + `include_e_values` | 1,163 ms |

* Complexity: MC loop is **pure-Python, dict-based, not vectorised** — O(n_samples × n_options × (nodes+edges)) plus sensitivity (≈400 evals/edge) and bootstrap. Fine at PoC scale; the 10M complexity guard admits requests around ~8–10s of single-threaded compute.
* **Event-loop blocking:** `analyze_robustness_v2` is `async def` but runs the entire CPU-bound analysis inline; with default `WORKERS=1`, concurrent requests serialise behind each other and health checks stall during a big analysis. This is the main scalability constraint, ahead of algorithmic cost.
* Wall-clock-adaptive features (bootstrap budget, 2s E-value budget) mean **load changes response content** — a performance/correctness coupling worth removing.
* Protections present: complexity guard, n_samples ≤10k, graph ≤50/200, request size limit, 60s timeout, memory breaker, GZip. Expensive dormant routes are fully unregistered (best protection).
* Recommended gates (no suite exists for the live v2 path): p95 < 1.5s at 12n/20e/5000s/3opt with flags off; < 3s with VOI+E-values; 5-concurrent-request test asserting no >2× latency degradation *(currently expected to fail — event loop)*; same-seed byte-equality regression test.

---

## I. Security assessment

| Area | Finding | Assessment |
|---|---|---|
| AuthN | X-API-Key, required by default, startup RuntimeError without keys unless `ISL_AUTH_DISABLED=true`; production config validation hard-fails on disabled auth, missing keys, wildcard/localhost CORS, missing Redis; constant-time comparison; hashed key prefixes in audit logs | **Good** |
| Staging/prod key separation | Keys are env-supplied per environment; no in-repo distinction visible — cannot verify separation from repo | unknown (ops) |
| CORS | explicit origins, no wildcard in prod (validated), explicit headers, credentials off by default | Good |
| Rate limiting | Redis sliding window per IP/key with in-memory fallback; prod config requires Redis | Good; fallback silently weakens under Redis outage (logged) |
| Request limits | size (MB), 60s timeout, memory circuit breaker at 85%, complexity guard, schema maxima | Good |
| Input sanitisation | request-ID sanitised (log-injection defence); node IDs pattern-restricted; goal IDs hashed in logs/echo | Good |
| Secrets | none committed (scan clean); `.env.example` placeholders; Sentry `before_send` strips auth headers; PII redaction in log formatter | Good |
| Info exposure | `/metrics`, `/health/*`, `/cache/stats`, `/docs`, `/openapi.json` **public by design** — endpoint list, latency histograms, memory stats visible unauthenticated | Medium-low; consider auth or network-level restriction for /metrics in prod |
| Error leakage | 500s sanitised; **/analyze/unified 400 leaks internal Pydantic/Query internals** (verified); v1 `/analyze` 500 detail includes `str(e)` (robustness.py:228,436 — exception text, not stack) | Low-medium; tighten both |
| Container | non-root `appuser`, slim image, healthcheck | Good — **but Dockerfile cannot build: COPYs nonexistent poetry.lock** |
| Dependency risk | **No lockfile ever committed** → CI/security scans audit a freshly-resolved set each run; builds unreproducible; `safety`+`pip-audit`+`bandit` do run in CI | **High (supply chain / reproducibility)** |
| Debug endpoints | none registered | Good |

---

## J. Maintainability assessment

**Positives:** fully typed Pydantic models with finite-value validators; mypy strict-ish (`disallow_untyped_defs`) and enforced in CI; constants centralised (`src/constants`) with rationale comments; stability thresholds centralised, versioned (`v1.0-operational-defaults`), env-overridable, explicitly `provisional: true`; critique/warning registry (`models/critique.py`) gives one place for error taxonomy; structured JSON logging with correlation IDs and PII redaction; determinism utilities centralised in `utils/rng.py`; tests are numerous (2,618 collected functions; 2,462 outside archived) and the live-path subset passes (144 passed, 2 skipped, ~20s).

**Negatives:**
* **Dormant mass dominates the repo:** 19 disabled routers + ~30 dormant/dead services vs 1 live analyzer. Every reader must re-derive "what is actually live" (this audit's main cost). No capability registry exists; the truth lives in a comment block in `main.py`.
* **Response assembly is split across three layers** (analyzer builds V1 → router converts to V2 inline over ~500 lines → ResponseBuilder assembles envelope). The V2 conversion in `robustness.py` is the most fragile file in the live path (the seed divergence and dropped `path_decomposition` both live there).
* Duplication: `evaluate`/`evaluate_multi` are near-identical 80-line twins; two robustness analyzers; `FragileEdgeEnhanced`↔`FragileEdgeV2`, `ConstraintResult`↔`ConstraintResultV2` mirror models.
* Docs drift: README capability table and PERFORMANCE_REPORT describe the pre-pilot surface; `ENABLE_*` settings exist for disabled features; `docs/_archive` + `docs/audits` accumulating.
* `src/services/_archived` is inside the package and breaks a clean `mypy src/` run (12 errors in 5 files in my environment, concentrated there).
* Test taxonomy is good (unit/integration/contract/property/smoke/perf) but: contract tests = 1 file/13 tests; property tests = 10; no test asserts two-full-response same-seed equality; no test covers `/analyze/unified` with a v2 payload (which is how §A-4 survived); cycle tests don't cover the V1 path.

**Simplification opportunities (recommend, not implement):** delete dead services; move `_archived` out of `src/`; retire or fix `/analyze/unified` and the FACET v1 route decision; extract the V1→V2 conversion into one tested module; single `SCMEvaluatorV2.evaluate(targets=[...])`; a `CAPABILITIES.md`/registry generated from the app's actual routes; make bootstrap count fixed (20) and E-value budget deterministic (count-based) to kill wall-clock coupling.

---

## K. Risk register

Severity: critical/high/medium/low · Confidence: **H**igh (verified by execution), **M**edium (code-confirmed, not executed), **L**ow (inferred). "Blocks PoC" assumes current PLoT V2 usage [inferred].

| ID | Finding | Evidence | Sev | Conf | Owner | Category | Action | Blocks PoC? | Blocks sci. credibility? | Lane |
|---|---|---|---|---|---|---|---|---|---|---|
| T0-1 | `probability_of_goal` (noised) vs constraint `prob_satisfied` (un-noised) disagree for identical node/threshold — 0.220 vs 0.134 | empirical check 3; analyzer:840-848 vs 1060-1131 | **high** | **H** | ISL | scientific validity | Decide one semantics (apply auto-noise to constraint node samples for outcome/risk nodes, or to neither) and document; regression-test equality for the identical-constraint case | yes if both fields shown | **yes** | Tier 0 |
| T0-2 | V2 envelope `seed_used` diverges from actual RNG seed whenever organisational nodes are filtered and seed is server-computed | empirical check 4 (2714408004 vs 947429003); robustness.py:459-461 vs analyzer:666-700 | **high** | **H** | ISL | contract/reproducibility | Compute the reported seed from the filtered graph (share one code path with the analyzer), add a test with a decision node in the graph | yes for reproducibility claims | **yes** | Tier 0 |
| T0-3 | Cyclic graphs silently produce results on `response_version=1` (default; CEE per docs) — no critique, no error | empirical check 5; analyzer:436-446; validator only on V2 path | **high** | **H** | ISL | correctness | Run cycle detection (cheap) in `analyze()` itself or in the legacy handler; fail closed with the existing `GRAPH_CYCLE_DETECTED` critique | yes if any V1 consumer sends cycles | yes | Tier 0 |
| T0-4 | Negative / below-resolution EVPI emitted raw (6/14 seeds negative, to −0.073 at 500-sample cap) with no resolution labelling | empirical check 6/6b; analyzer:2866-2973 | **high** | **H** | ISL | scientific validity | Add `below_resolution`/`not_significant` status when |EVPI| < ~2×MC standard error; rename or document metric as "uncertainty contribution", not EVPI | no | **yes** | Tier 0 |
| T0-5 | No `poetry.lock` in repo or history; CI/security scans/Docker assume one; Dockerfile `COPY poetry.lock` cannot succeed | git history; Dockerfile:9; pr-ci.yml cache key | **high** | **H** | ISL/ops (Paul) | security/reproducibility | Commit a lockfile; make CI use `--no-update` install; fix or delete Dockerfile | deploy-risk | indirectly (unreproducible builds) | security/ops |
| T1-1 | `/analyze/unified` broken for v2 payloads (always 400 via `Query(False)` leak) | empirical check 8 | medium | **H** | ISL | correctness/contract | Fix by calling the enhanced handler with explicit scalar args, or unregister the route | no [inferred unused] | no | Tier 1 |
| T1-2 | Duplicate `(from,to)` edges double-count effects | empirical check 11 | medium | **H** | ISL (+PLoT dedupe) | correctness | Reject duplicates in `GraphV2` validator (one-line set check) | only if PLoT can emit dupes | yes (silent wrong math) | Tier 1 |
| T1-3 | Bootstrap iteration count decided by wall-clock (<100ms → 20 else 10) — confidence fields load-dependent under fixed seed | analyzer:2122-2162 | medium | **H** (code) / not triggered in test | ISL | reliability/determinism | Fix count at 20 (costs ~ms); keep override for tests | no | yes (weakens determinism claim) | Tier 1 |
| T1-4 | E-value block silently omitted when >2s budget exceeded — response content varies with load | analyzer:2709-2761 | medium | M | ISL | reliability | Make the budget count-based, or emit `e_values_status:"budget_exceeded"` | no | partial | Tier 1 |
| T1-5 | Edge/factor sensitivity, fragile edges, bootstrap all computed against `options[0]` only; option order changes results; undisclosed | analyzer:1348,1674 | medium | M | ISL | scientific validity | Document reference-option semantics in response (e.g. `sensitivity_reference_option_id`), or compute vs recommended option | no | yes (subtle) | Tier 1 |
| T1-6 | `include_path_decomposition` computed then dropped on V2 envelope; edge-level `sensitivity` also absent from V2 | response_v2.py:612-710 | medium | **H** (model fields absent) | ISL | contract | Add fields to `ISLResponseV2` or reject the flag on V2; stop paying for dropped compute | no | no | Tier 1 |
| T1-7 | Root nodes sum `observed_state.value + intercept`; divergent-root semantics unconfirmed against PLoT's population rule | empirical check 7 | medium | **H** (behaviour) / L (whether it's wrong) | PLoT/Neil | contract | Written rule for who sets `intercept` on roots; ISL warning when both are non-zero and different | unknown | possibly | Neil question |
| T1-8 | Event-loop blocking: CPU-bound analysis inline in async handler, `WORKERS=1` — concurrency collapses, health checks stall | main.py; timings §H | medium | M | ISL/ops | performance | `run_in_executor`/anyio to-thread, or ≥2 workers + document capacity | pilot-scale OK; demo-risk under concurrent use | no | Tier 1 |
| T2-1 | Robustness `confidence = min(0.99, stability·(1−1/√n))` is an unlabelled heuristic | analyzer:2669 | medium | **H** | ISL/Neil | scientific validity | Label like factor-confidence (provisional/source) or replace with bootstrap CI on win-share | no | yes (if quoted as probability) | Tier 1/Neil |
| T2-2 | Stability penalty (−5%/defaulted root) silently alters reported win-share field | analyzer:2616-2620 | low-med | **H** | ISL | scientific validity | Report raw stability + penalty separately (fields exist; make the raw value primary) | no | partial | Tier 1 |
| T2-3 | `epsilon_std` clamp [0,1] applies only when noise fires; unbounded otherwise | analyzer:527-533 | low | **H** | ISL/Neil | scientific validity | Decide bounded-vs-unbounded world; make consistent | no | minor | Tier 2 |
| T2-4 | `kind` not enum-validated; unknown kinds silently join inference | robustness_v2.py:360 | low | **H** | ISL | contract | Validate against known kinds with warning | no | no | Tier 2 |
| T2-5 | Non-root `observed_state.value` silently ignored (no warning) | analyzer:503-512 | low-med | **H** | ISL | contract/honesty | Emit inference warning when a non-root carries an observed value that won't be used | no | partial | Tier 1 |
| S-1 | `/metrics`, `/health/*`, `/docs` public | auth.py:38-51 | low | **H** | ops | security | Restrict /metrics at network layer in prod | no | no | security |
| S-2 | Error detail leaks: unified 400 internals (verified), v1 `/analyze` 500 includes `str(e)` | robustness.py:228,1052 | low | **H** | ISL | security | Sanitise both | no | no | security |
| M-1 | README/PERFORMANCE_REPORT/`ENABLE_*` flags describe disabled capabilities | README.md:63-102; benchmarks/ | medium | **H** | ISL docs (Paul) | product/maintainability | Rewrite README around the live surface; archive stale report; delete dead flags | demo-narrative risk | yes (overclaim exposure) | Tier 0 (docs-only) |
| M-2 | Dead services (`sbc_validator`, `causal_validator_enhanced`, `robustness_visualizer`); `_archived` breaks clean mypy | imports scan; mypy run | low | **H** | ISL | maintainability | Delete/move; exclude `_archived` from mypy | no | no | Tier 2 |
| M-3 | No full-response same-seed equality test; no unified-v2 test; no V1-cycle test — the verified defects live exactly in the untested gaps | tests scan | medium | **H** | ISL | test quality | Add the regression suite in §L-1 | no | yes (evidence quality) | Tier 0/1 |

---

## L. Recommended roadmap

### Tier 0 — fix or pin before trusting current outputs
1. **Constraint/goal-noise consistency (T0-1)** — one decision + ~10-line change + regression test asserting `P(goal ≥ x) == P(constraint revenue ≥ x)` on the same response.
2. **Seed reporting (T0-2)** — compute envelope seed from the filtered graph via a single shared function; test with an organisational node present.
3. **Cycle fail-closed on V1 path (T0-3)** — call `detect_graph_cycle` in `analyze()`; test.
4. **EVPI honesty (T0-4)** — below-resolution labelling + method rename/description; do NOT clamp to zero.
5. **Lockfile + Dockerfile (T0-5)** — commit `poetry.lock`, repair the build path, re-run pip-audit against the pinned set.
6. **Docs truth pass (M-1)** — README live-surface rewrite; mark stale reports as archived. Cheap, high credibility value.
7. **Scientific regression gate (M-3)**: same-seed full-JSON equality (with `_n_bootstrap_override`); single-edge analytic fixture with exact expected means; negative-edge fixture; cycle-rejection both response versions; duplicate-edge rejection (after T1-2); unified-endpoint v2 payload; seed-with-filtered-nodes.

### Tier 1 — provenance and coaching quality once outputs are safe
Fix unified endpoint or remove it (T1-1); duplicate-edge validator (T1-2); fixed bootstrap count (T1-3); deterministic E-value budget (T1-4); disclose sensitivity reference option (T1-5); V2 envelope gaps — add or reject `path_decomposition`/edge sensitivity (T1-6); warning for ignored non-root observed values (T2-5); label robustness confidence as provisional heuristic (T2-1) and surface raw vs penalised stability (T2-2); offload CPU work from the event loop (T1-8); add `input_scale_assumption` provenance to responses (§F.3).

### Tier 2 — future scientific differentiation
Registered-and-gated identifiability endpoint (the y0 work is the strongest dormant asset); calibration loop: revive outcome logging + SBC/conformal machinery against pilot outcomes; bounded/categorical variable support with explicit encodings; per-option sensitivity; correlated factor priors (currently factors sample independently); replace heuristic confidences with bootstrap CIs; sequential/temporal modelling if the product needs it; delete the rest.

### Security/ops follow-ups
Lockfile (also Tier 0); metrics endpoint exposure decision; sanitise the two leaking error paths; verify staging vs production key separation and Render env config (outside repo); consider CI job asserting `docker build` succeeds.

### Questions for Neil
1. Root semantics: when a root has both `observed_state.value` and `intercept`, is summation correct, or should intercept be root-baseline-only? What should ISL do on divergent roots?
2. Auto-noise: should constraint-node samples receive the same auto-scaled noise as the goal (T0-1)? And is 1× model-std still the endorsed heuristic pending calibration?
3. Is the fixed-policy "EVPI" acceptable if renamed (e.g. "information value proxy"), or should true EVPI (policy re-optimisation per information state) be Tier 2?
4. Are `>=`/`<=` sufficient constraint operators for the product contract, or are `==`/`!=`/strict forms needed?
5. Endorse or replace `confidence = stability·(1−1/√n)` for robustness.

### Questions for Paul
1. Which consumers still hit `response_version=1`? (CEE per docs — confirm; the cycle hole and NaN-mean asymmetry live there.)
2. Is `/analyze/unified` (and FACET v1 `/analyze`) contractually needed by anyone? Both are candidates for removal.
3. How is Render actually building/deploying (native Python + poetry without lockfile, or the broken Dockerfile)? Determines T0-5 shape.
4. Confirm PLoT strips organisational nodes and dedupes edges before calling ISL — determines real-world exposure of T0-2/T1-2.

---

## M. Suggested next briefs (not written yet)

1. **ISL Tier-0 remediation brief** (owner: ISL/Claude Code lane): T0-1..T0-4 + regression gate M-3. Small, surgical, all changes in `robustness.py`/`robustness_analyzer_v2.py`/`robustness_v2.py` + tests.
2. **Supply-chain and build brief** (owner: Paul + ISL): lockfile, Dockerfile repair or removal, CI pin verification, Render build-path documentation.
3. **PLoT boundary verification brief** (owner: PLoT lane): read-only trace of PLoT's ISL adapter to convert every "[inferred]" in this report into verified/falsified — especially response_version used, organisational-node stripping, edge dedupe, which V2 fields PLoT consumes vs ignores.
4. **Docs truth & capability registry brief** (owner: ISL): README rewrite, stale-report archiving, generated live-route capability table.
5. **Scientific labelling brief** (owner: Neil + ISL): provenance labels (`provisional`, `method`, `reference_option`, `input_scale_assumption`) across all heuristic fields; the Neil questions above as a working session.
6. **Identifiability activation spike** (owner: Neil + ISL, Tier 2): what it would take to register the y0 identifiability route safely (perf, dependency weight, output contract).

---

## Annex — brief §13 hypotheses: verified / falsified

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | `factor_sensitivity.confidence` hardcoded 0.8 | **Falsified** (current code) | bootstrap-blend with `confidence_source`; stability_thresholds.py:140-203 |
| 2 | `epsilon_std` 0 everywhere and `auto_noise_applied` false | **Half-falsified** | `epsilon_std` defaults 0 and PLoT doesn't set it [inferred]; but `auto_noise_applied=True` for outcome/risk goals (verified empirically) |
| 3 | EVPI sample-depth mismatch | **Verified** | capped at 500 vs main n_samples (2000 in test); labelled `n_evpi_samples` |
| 4 | Negative/near-zero EVPI clamped rather than labelled | **Falsified as stated, worse in practice** | not clamped — raw negatives emitted with no below-resolution labelling (6/14 seeds) |
| 5 | Advanced services implemented but not registered | **Verified** | 19 routers / 48 endpoints commented out; §C.2 |
| 6 | Routers disabled for attack-surface reduction | **Verified (intent documented)** | main.py:696 "Disabled for pilot — orphaned endpoints not used by PLoT" |
| 7 | SCM-lite / causal-discovery dormant not live | **Verified** | causal.py + discovery/conformal/transport services all dormant |
| 8 | Identifiability claimed but not enforced | **Verified** | README + `ENABLE_IDENTIFIABILITY_ANALYSIS=True` vs disabled route; live diagnostics say `"unknown"` (honest) |
| 9 | Categorical/boolean absent or unsafe if encoded | **Verified (absent; numeric passthrough)** | §F.9 |
| 10 | Constraint correlation/conditional probabilities | **Implemented** | pairwise P(Cj|Ci) from shared satisfaction matrix; no higher-order structure |
| 11 | Root/intercept double-count in divergent cases | **Behaviour verified (summation)**; whether it double-counts depends on PLoT population rule | check 7: 0.5+0.3→0.8; Neil question |
| 12 | Seed/request-ID chain partially tested | **Mostly implemented & tested**, with one verified reporting defect (T0-2) | `test_seed_used_serialisation.py` exists; check 4 |
| 13 | Unknown fields ignored by Pydantic | **Verified — deliberate** | `extra="ignore"` documented as CIL contract everywhere |
| 14 | ISL may emit misleading success statuses | **Partially verified** | status machinery is honest (computed/partial/failed), but V1-path cycles return "success" garbage (T0-3), and load-dependent field omission (T1-4) |
| 15 | Security/auth weaker than expected | **Falsified overall** | auth/ratelimit/limits/config validation strong; residual items are the lockfile (T0-5), public /metrics, two leaky error details |

---

*Audit constraints honoured: no code edits, no commits, no branches, no pushes, no deploys, no config or dependency changes to the repo (venv lived in the session scratchpad; `git status` clean throughout), no secrets printed, no staging/production traffic, no load tests — timings above are single sequential in-process calls.*
