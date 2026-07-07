# Lane 4 evidence report — lockfile (T0-5), EVPI below-resolution labelling (T0-4), response-hash fix

- Date: 2026-07-07
- Branch: `claude-lane4/lockfile-evpi-labelling` (worktree from `origin/staging` @ `f3f5d92`)
- Doctrine authorization: provisional — all new wording/labels tagged `provisional_doctrine_v0`
- Contract status: FROZEN — no boundary field changed, renamed, or removed. All additions are
  inside an already-untyped `Dict[str, Any]` list (verified below).

## A. T0-5 — poetry.lock build integrity (commit `14117f5`)

**Problem.** `poetry.lock` was explicitly gitignored (`.gitignore:31`), so no lockfile ever
reached git, while CI cache keys (`hashFiles('**/poetry.lock')` in pr-ci / security /
deploy-production / perf-tests / openapi-validation), security scans, and both Dockerfiles
(`COPY pyproject.toml poetry.lock ./`) assumed one existed. As committed, the Docker build
could not succeed (COPY of a nonexistent file).

**Change.**
1. Removed `poetry.lock` from `.gitignore`.
2. A local `poetry.lock` existed in the main working tree (dated 2025-12-18). Verified it
   against staging's `pyproject.toml`: `poetry check --lock` → **exit 0** on Poetry 2.2.1
   (only pyproject `[tool.poetry]` deprecation warnings, no consistency errors). Committed it.
3. Additionally proven installable end-to-end: `poetry install` from this lockfile built the
   fresh worktree venv used for every test below.
4. Dockerfile: `COPY pyproject.toml poetry.lock ./` paths now line up (lockfile at repo root).
   One remaining build-integrity mismatch fixed: the lockfile is **lock-version 2.1**, which
   Poetry 1.x cannot read, while the Dockerfile pinned `poetry==1.7.1`. Bumped to
   `poetry==2.2.1` (the version the lockfile was verified with; CI already installs Poetry
   "latest" 2.x, so this aligns Docker with CI). `Dockerfile.dev` installs Poetry via the
   official installer (latest) — no change needed.

**Not done (by instruction):** no full `docker build` was attempted. The Poetry-2.x flags used
in the Dockerfile (`config virtualenvs.create false`, `install --only main --no-root`) are
valid in 2.x, but the image build itself remains unverified — flagged as follow-up.

## B. T0-4 — EVPI below-resolution labelling (commit `5bea83c`)

**Problem.** ISL emits raw negative EVPI (live logs: `fac_tech_lead` −0.004; audit example
−0.073) at the 500-sample EVPI budget cap where ±0.03–0.06 is MC noise. Audit prescription:
LABEL below-resolution values; do NOT clamp; rename nothing.

**Noise-floor derivation** (documented in `src/services/robustness_analyzer_v2.py` module
comment). `evpi = perfect_metric − baseline_metric`, a difference of two MC proportion
estimates over `n_evpi_samples` draws with **independent** seed streams (baseline: seed+100/101;
perfect: per-factor SHA-256 seeds — no CRN pairing), so variances add:

```
Var(evpi_hat) <= 2 * (0.25 / n)        # worst-case Bernoulli p = 0.5
SE_max        =  sqrt(0.5 / n)
noise_floor(n) = 1.96 * sqrt(0.5 / n)  # two-sided 95% bound
```

At the n=500 cap: floor ≈ **0.06198** — consistent with the audit's ±0.03–0.06 noise band
(1 SE ≈ 0.032, ~2 SE ≈ 0.062). The worst-case bound is used instead of the plug-in
`sqrt((p1(1−p1)+p2(1−p2))/n)` because the plug-in collapses to 0 at estimated metrics of 0/1
at small n, understating true uncertainty. Note the audit's −0.073 example is **outside** the
floor and is deliberately labelled `resolved` (not explainable as pure MC noise) — the label
distinguishes noise from genuine anomalies rather than blanket-excusing negatives.

**Fields added per `factor_evpi` entry** (additive, label-only; raw `evpi` never altered):
- `evpi_status`: `"below_resolution" | "resolved"` (|evpi| < floor → below_resolution)
- `evpi_noise_floor`: the floor for that entry's `n_evpi_samples` (rounded 6 dp)
- `evpi_noise_floor_method`: `"z95_worst_case_bernoulli_diff"`
- `evpi_labelling_doctrine`: `"provisional_doctrine_v0"`

**Frozen-contract verification (complete manifest, scope + claim type).**
Scope searched: `src/` of all five repos (`Inference-Service-Layer`, `plot-lite-service`,
`olumi-assistants-service`, `DecisionGuideAI`, `olumi-schemas`), node_modules excluded, plus
PLoT fixtures/tests. Claim type: *no strict parser of `factor_evpi` entries exists anywhere*.
- ISL: entries are `Dict[str, Any]` at every hop — `_compute_evpi` →
  `RobustnessResponseV2.factor_evpi` (`src/models/robustness_v2.py:1335`) →
  `builder.set_results` (`src/utils/response_builder.py`) → V2 envelope
  `ISLResponseV2.factor_evpi` (`src/models/response_v2.py:685`, `extra="ignore"`).
  No Pydantic model changed.
- PLoT: **zero** `factor_evpi` references in src/fixtures/tests (its VOI/EVPI comes from the
  internal heuristic + V1-shaped `value_of_information`, per known P1 finding).
- CEE: one comment mention only (`decision-review-enricher.ts:235`); enrichment passthrough is
  `z.record` (untyped).
- DGAI: debug export bundle only, typed `unknown[] | null` with `Array.isArray` guard.
- olumi-schemas: zero references.

**Tests** (`tests/unit/test_scientific_enhancements.py::TestEVPIBelowResolutionLabelling`):
formula exactness + monotonicity + n=0 degenerate; labelling fields on every entry; status⇔floor
consistency; deterministic fixture at n=100 all below-resolution; and a fixture regression
(seed=12345, n=500) producing a **raw negative EVPI (−0.001)** proven preserved un-clamped,
`evpi == perfect_metric − current_metric`, and labelled `below_resolution` — the exact live-log
pattern.

## C. Response-hash telemetry fix (commit `fc6d548`)

**Problem.** `Failed to compute response hash: 'utf-8' codec can't decode byte 0x8b` on every
compressed response. Root cause: `GZipMiddleware` was registered **first** in
`src/api/main.py`; Starlette middleware is LIFO (last added runs outermost), so GZip was the
*innermost* layer and `ObservabilityMiddleware` (`src/middleware/observability.py`) received
already-gzipped bytes (`1f 8b` magic — the `0x8b`). `json.loads` raised UnicodeDecodeError,
`x-olumi-response-hash` was never set, and cross-service reconciliation was dead for every
response large enough to compress (>1KB — i.e., every real analysis response, since httpx
callers send `Accept-Encoding: gzip`).

**Fix.** Moved the `GZipMiddleware` registration to after `ObservabilityMiddleware`, so
execution order is `… → Tracing → GZip → Observability → … → routes`: the hash is computed
over pre-compression JSON and GZip compresses afterwards. Ordering constraint documented at
the registration site. No hashing logic changed.

**RED → GREEN proof** (TestClient against the real app, `POST /api/v1/robustness/analyze/v2`,
`Accept-Encoding: gzip`, 200, body ~6KB compressed):
- RED (before): `enc=gzip`, `hash=None`, warning logged:
  `"Failed to compute response hash" error="'utf-8' codec can't decode byte 0x8b in position 1"`
- GREEN (after): `enc=gzip`, hash header present, **and** header value ==
  `canonical_json_hash(decompressed body)` — full reconciliation.

Regression tests added (`tests/integration/test_observability_contract.py::TestResponseHashOnGzippedResponses`)
pinning both the gzipped path and the small-uncompressed path. Note: committing this test file
also applied required `black` formatting to its pre-existing content (base file was
non-compliant); verified the non-append delta is formatting-only
(`head == black(base) + appended class`).

## Test evidence (all run in the fresh worktree venv, `ISL_AUTH_DISABLED=true`)

| Check | Result |
|---|---|
| `poetry check --lock` | exit 0 |
| `poetry install` from tracked lockfile | success (built the worktree venv) |
| `black --check` on all touched files | clean (after applying black) |
| `pytest tests/unit/test_scientific_enhancements.py` | **47 passed** (42 pre-existing + 5 new) |
| `pytest tests/integration/test_observability_contract.py tests/unit/test_observability_middleware.py tests/unit/test_compression.py` | **46 passed, 16 skipped** (skips = pre-existing live-server tests, skip when ISL not running) |
| `pytest tests/integration/test_tier0_regression_gate.py` | **19 passed** |
| `poetry run mypy src/` | Success: no issues found in 134 source files |

## Blockers / follow-ups

- **Follow-up (not a blocker):** full `docker build` untested by instruction; the Poetry
  1.7.1→2.2.1 bump is required by the lock-version-2.1 file but the image build should be
  exercised in CI/staging before relying on it.
- **Follow-up:** consumers may eventually want to *use* `evpi_status` (e.g. PLoT suppressing
  below-resolution EVPI chips); that is producer-side-ready but consumer adoption is
  doctrine-gated and out of this lane.
- No frozen-contract violations encountered; no items stopped.
