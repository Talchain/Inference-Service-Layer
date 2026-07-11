# Analytic flip-threshold derivation vs MC — agreement study (Track S P1 spike)

**Lane:** ISL ANALYTIC FLIP-THRESHOLD SPIKE · **Base:** origin/staging
`d773f4a` (the #71 stability-bands merge) · **Branch:**
`spike/analytic-flip-threshold` · **Date:** 2026-07-11 · **Status:** SPIKE —
experimental module + report, **zero default-path change** (module is
import-inert; no flag needed).

## Problem

Every flip threshold ISL reports is found by search: `_compute_edge_e_values`
runs a boundary check + 20-step bisection per edge per direction against the
outcome function, and the #71 stability bands re-run that same search once
per child seed (`_flip_mean_under_background`). That is up to ~44 full
option-sweep evaluations per edge on the base path and N× that for bands.
The 2026-06-10 science-performance report (§8) and ROADMAP 2.29(b) asked:
where the SCM structure permits, can the flip threshold be derived
*analytically* — and where exactly is that invalid? The honesty boundary
matters more than coverage.

## The structural finding (why closed form is EXACT here, not approximate)

Both shipped searches run on the **deterministic** `SCMEvaluatorV2`:
`analyze()` sets `evaluator._epsilon_rng = None`
(`src/services/robustness_analyzer_v2.py:1064`) before any structural
analysis, including the flip search at line 1188. The deterministic
evaluator's structural equations are linear-additive
(`SCMEvaluatorV2.evaluate`):

    node_value = base + intercept + sum(parent_value * edge_strength)

Vary ONE edge's `strength.mean` m (effective strength `m * exists_probability`)
with every other edge held fixed — expected values on the base path, a
sampled background in the band sweep — and each option's goal value is an
**exactly affine** function of m: the perturbed strength enters the recursion
exactly once, upstream node values cannot depend on it (DAG), and everything
downstream is linear in the perturbed node's value. Therefore:

- each pairwise winner-margin is affine in m → at most one crossing;
- the set of m where the baseline winner stays the winner is an interval
  (intersection of half-lines) → one flip boundary per direction;
- the bisection is spending 20 steps approximating a number that has an
  exact one-division formula: `m* = (A_w − A_r) / (B_r − B_w)`.

**Newton root-finding is unnecessary**: the mission allowed closed-form *or*
Newton on monotone paths, but on the current evaluator every reachable case
is affine, so closed form covers 100% of the production flip-search surface.
Newton would only become relevant if smooth non-linear structural equations
land — at which point the validity argument below must be re-derived, and
multiple crossings become possible (bisection is not trustworthy there
either: it assumes a single boundary).

## What shipped (all spike-scoped)

- `src/services/analytic_flip.py` — closed-form flip derivation.
  `analytic_edge_e_values` mirrors `_compute_edge_e_values` entry-for-entry
  (keys, 6-dp/4-dp rounding, e_value ≥ 1.0 floor, no-flip sentinel,
  bidirected skip, increase-direction-first);
  `analytic_flip_mean_under_background` mirrors the #71 band inner search.
  Affine coefficients come from a two-point evaluator oracle (m=0, m=+1;
  m=−1 doubles as the decrease boundary check and the affinity tripwire),
  so the module needs 4 option-sweeps per edge vs up to ~44 for bisection.
  Winner determination (baseline and boundary) uses evaluator-computed
  floats with the identical `(-value, option_id)` tie-break, so verdicts
  match the MC search bit-for-bit; only the crossing point itself is
  computed analytically.
- `tests/unit/test_analytic_flip.py` — 10 tests, RED-first (see evidence).
- `benchmarks/analytic_flip_agreement.py` — the agreement study below.
- **Import inertness pinned by test**: `TestImportInertness` scans `src/`
  and fails if any production module references the spike.

## The honesty boundary — where the analytic form is INVALID

The module **raises `AnalyticInvalidityError`** rather than ever returning a
plausible-but-wrong number:

1. **Epsilon-noised evaluator** (`_epsilon_rng` present AND some node has
   `epsilon_std > 0`): per-node noise plus the [0,1] clamp makes outcomes
   stochastic and piecewise — not affine. Rejected structurally before any
   arithmetic. Note the production flip search never sees this state
   (analyze() nulls the RNG first); the guard exists for misuse. An epsilon
   RNG with all `epsilon_std == 0` can never fire and is accepted (pinned by
   test — the guard must not over-reject).
2. **Non-affine composition** (future sigmoid/threshold/saturating structural
   equations): a three-point affinity tripwire — outcome at m=−1 must equal
   the affine prediction from m=0 and m=+1 within 1e-9 (scaled). Three
   collinear points do **not** prove affinity in general; the *proof* is the
   structural argument above, which holds for today's evaluator by
   construction. The tripwire's job is to make a future evaluator change
   fail loud here instead of silently disagreeing with MC. Pinned by a test
   that feeds a deliberately quadratic evaluator.
3. **Internal inconsistency** (boundary flips but no in-range affine
   crossing — impossible for an affine system): raises, never guesses.

Non-monotone composition on the *current* evaluator: cannot occur for a
single-edge perturbation (affine ⇒ monotone in m); opposite-sign parallel
paths through the same edge sum into one net slope, they do not create
curvature. Non-monotonicity in m becomes possible only under (1) or (2),
which is exactly where the module refuses.

## Agreement study (repo fixtures)

`poetry run python benchmarks/analytic_flip_agreement.py` — fixtures are the
three pinned `sample_variants` graphs and the 12-node/17-edge synthetic
production-shaped graph (same fixtures as the #71 budget benchmark). The
comparison targets are the **shipped wire numbers** from full `analyze()`
runs (flag off for base, `ISL_FLIP_STABILITY_BANDS=1` for bands), not a
re-derivation. Agreement tolerance 1e-5 (bisection resolution ≈ 2e-6 of the
bracket + 6-dp wire rounding).

### 1. Base path — `edge_e_values[].flip_mean` (flag off) vs analytic

| fixture | edges | flip | no-flip | agree | max Δflip_mean |
|---|---|---|---|---|---|
| sample_variants[0] | 3 | 2 | 1 | **3/3** | 0.0 |
| sample_variants[1] | 3 | 2 | 1 | **3/3** | 1.0e-06 |
| sample_variants[2] | 3 | 2 | 1 | **3/3** | 1.0e-06 |
| synthetic_12n_17e | 17 | 11 | 6 | **17/17** | 1.0e-06 |

26/26 entries agree: flip/no-flip verdict, direction, and value (max
deviation = one unit in the 6th decimal, i.e. the bisection's own error —
the analytic value is the exact crossing).

### 2. Stability bands (#71) — wire `stability.seed_flip_means` vs analytic under the same SHA-256 child-seed backgrounds

| fixture | edge×seed cells | none (no flip) | agree | max Δflip_mean |
|---|---|---|---|---|
| sample_variants[0] | 15 | 4 | **15/15** | 1.47e-06 |
| sample_variants[1] | 15 | 5 | **15/15** | 1.49e-06 |
| sample_variants[2] | 15 | 4 | **15/15** | 1.66e-06 |
| synthetic_12n_17e | 85 | 44 | **85/85** | 1.76e-06 |

130/130 cells agree, including all 57 None cells (background admits no
flip) — the None/flip verdict never diverges, and deviations sit at the
bisection-resolution level.

### 3. Speed — flip computation only (median of 21, 1 warm-up; exact evaluate() call counts)

| fixture | MC ms | analytic ms | speedup | MC evals | analytic evals | band MC ms | band analytic ms | speedup | band MC evals | band analytic evals |
|---|---|---|---|---|---|---|---|---|---|---|
| sample_variants[0] | 0.10 | 0.03 | 3.2× | 94 | 24 | 0.56 | 0.15 | 3.8× | 526 | 120 |
| sample_variants[1] | 0.10 | 0.03 | 3.4× | 94 | 24 | 0.53 | 0.14 | 3.8× | 490 | 120 |
| sample_variants[2] | 0.10 | 0.03 | 3.5× | 94 | 24 | 0.55 | 0.13 | 4.2× | 528 | 120 |
| synthetic_12n_17e | 2.53 | 0.71 | 3.6× | 753 | 204 | 10.53 | 3.40 | 3.1× | 3177 | 1020 |

Honest caveats: wall times on sub-ms cells carry process noise (a first,
noisier run measured the same synthetic cells at 9.30/3.04 and 42.79/15.75 ms
— ratios 3.1×/2.7×, consistent); the **call counts are the deterministic
cost metric** (identical across runs): 3.7× fewer evaluations on the base
path, 3.1–4.4× on the band sweep. The analytic evaluation count is exactly
`4 × options × edges` per sweep regardless of how many directions/steps the
bisection would have needed. Absolute savings today are single-digit ms —
the value is (a) exactness, (b) cost that no longer scales with bisect steps
× directions, which is what pushes large graphs toward the 2000 ms
all-or-nothing budget guards (`E_VALUE_BUDGET_MS`,
`FLIP_STABILITY_BUDGET_MS`), and (c) headroom to raise the band seed count
N cheaply.

## Claims and their evidence (commands + counts)

Precise claim types, per the evidence doctrine:

1. **Import inertness (no-load claim).** No module under `src/` imports the
   spike; scope = `src/**/*.py`; pinned by
   `TestImportInertness::test_no_production_module_imports_the_spike`
   (regex scan for `analytic_flip`, complete manifest of src tree at test
   time — re-runs on every gate, so the claim cannot silently rot).
   Claim type: no-load / no-init-exec on any default path. The benchmark
   and the test file are the only importers (both non-production).
2. **RED-first.** Commit `c010d5d` (tests only): `ISL_AUTH_DISABLED=true
   poetry run pytest tests/unit/test_analytic_flip.py -q` → 1 collection
   error, `ModuleNotFoundError: No module named 'src.services.analytic_flip'`
   (all 10 tests uncollectable). Commit `c90ec15` (implementation):
   10/10 passed. One contract refinement folded into GREEN: the two
   hand-crossing assertions compare against `round(crossing, 6)` because
   `flip_mean` carries the wire contract's 6-dp rounding (RED had `abs=1e-9`
   against the unrounded crossing, which no rounding-faithful mirror can
   satisfy).
3. **Agreement.** Tables above; command
   `poetry run python benchmarks/analytic_flip_agreement.py`; comparison
   target = wire values from full `analyze()` runs on unmodified production
   code. 26/26 base entries, 130/130 band cells, zero verdict divergences.
   Claim scope: these four fixtures, deterministic evaluator, single-edge
   perturbation — i.e. exactly the shipped flip-search surface.
4. **Exactness argument.** Structural (linearity of `SCMEvaluatorV2.evaluate`
   with `_epsilon_rng=None`, single occurrence of the perturbed strength in
   the recursion, DAG order) — *not* an empirical claim from the fixtures;
   the fixtures corroborate it. The affinity tripwire turns a future
   violation of the argument into a hard error, not a silent disagreement.
5. **Gate.** `bash scripts/pre-push-validate.sh` (mypy strict on src/ — 135
   files clean; full non-perf suite; ISL_AUTH_DISABLED=true) passed on push;
   `poetry run black --check` clean on all touched files.

## Recommendation: **ADOPT-WHERE-VALID** (follow-up lane, not this spike)

- **Validity today is total**: the deterministic linear-additive evaluator is
  the *only* evaluator the shipped flip searches run on, so the closed form
  is exact on 100% of the current surface — "where valid" = everywhere, with
  the guard + tripwire holding the boundary against future evaluators.
- Concrete adoption shape: replace the bisection inner loop of
  `_compute_edge_e_values` and `_flip_mean_under_background` with
  `analytic_flip_search`, wrapped in `try/except AnalyticInvalidityError`
  falling back to the existing bisection. Keep the budget guards (they gate
  presence, not content).
- **Wire consequence to pin before adopting**: values move by ≤ ~2e-6 (the
  bisection's error) — usually invisible under 6-dp rounding but the 6th
  decimal can and does flip (max observed Δ = 1.0e-06 on rounded wire
  values). Adoption therefore needs a golden refresh + a determinism note,
  and downstream consumers (PLoT flip probes, band-width rubric doctrine)
  should be told the values become exact rather than search-approximate.
- Also unblocks: cheaper/larger band sweeps (N no longer multiplies a
  20-step search), and a path to *distributional* flip statements (the
  crossing formula is differentiable in the background — out of scope here).
- Rejected alternative: shipping the analytic path as a parallel wire field
  (double reporting) — two nearly-identical numbers with different error
  behaviour is exactly the false-precision trap the #71 lane warned about.

## Files touched

- `src/services/analytic_flip.py` — spike module (import-inert)
- `tests/unit/test_analytic_flip.py` — 10 tests (RED-first)
- `benchmarks/analytic_flip_agreement.py` — agreement study (this doc's tables)
- `docs/lanes/2026-07-11-analytic-flip-threshold-spike.md` — this report

## Out of scope (unchanged)

- `_compute_edge_e_values`, `_flip_mean_under_background`, all wire shapes,
  all flags — **zero default-path change**; flag-off and flag-on wires are
  byte-identical to base by construction (nothing in src/ imports the spike).
- The PLoT-side probe sweeps (`flip-thresholds.ts`).
- Reserved staging scenarios (1909b083*, def3cb31*, 8e0bf73d*, 88396c52*)
  were not touched; this lane made no staging-data or live-service calls.
