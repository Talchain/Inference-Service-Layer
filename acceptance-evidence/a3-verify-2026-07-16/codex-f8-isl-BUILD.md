# Codex F8 (ISL half) — BUILD evidence

**Lane:** A3 (ISL). **Base:** `origin/staging` @ `7ca214c2` (includes #79 EVPI/path
deadline + severity). **Scope:** F8 only (NOT F15 offload — later PR). Built in a
fresh blobless clone; gate driven with the project venv + `ISL_AUTH_DISABLED=true`
(the repo convention for scripts/CI). **Author:** A3 build agent. **Date:** 2026-07-18.

## What shipped (F8)

1. **Weighted compute-admission cost model** — replaces the scalar
   `n_samples*n_nodes*n_edges` (which omitted the option multiplier, used the wrong
   `x-edges` shape, and priced no optional phase). New module-level
   `compute_weighted_cost(request) -> WeightedCost` + `get_max_cost_units()` in
   `src/services/robustness_analyzer_v2.py`, read by BOTH admission call sites AND
   `/health` (single source of truth). Formula:

   ```
   cost = S*O*W                                        (base MC, always)
        + (U+1)*min(S, EVPI_SAMPLE_CAP)*O*W            (EVPI, if include_voi & U>0)
        + W_SENS_COEF*E*min(100, S//10)*W              (edge sensitivity)
        + W_EVAL_COEF*E*O                              (e-values, if include_e_values)
        + W_BANDS_COEF*E*O                             (bands, ride on e-values)
        + W_PATH_COEF*min(MAX_DECOMPOSITION_PATHS, E*E)(path decomposition)
   ```
   S=n_samples, O=len(options), N=nodes, E=edges, W=N+E, U=unique parameter_uncertainties.
   Coefficients (provisional): base=1, W_SENS_COEF=4, W_EVAL_COEF=20, W_BANDS_COEF=200,
   W_PATH_COEF=1, EVPI_SAMPLE_CAP=2000 (reused). 422 body reports `cost_units`,
   `limit`, `dominant_term`, `cost_breakdown`, `complexity_formula_version` +
   a term-keyed `suggestion` (v2 path clean; v1 path stringified by the app's
   existing Olumi-Error-Schema HTTPException handler — pre-existing behaviour).

2. **Unique/capped parameter_uncertainties** — field `max_length=50` +
   uniqueness enforcement in `validate_parameter_uncertainties_reference_nodes`
   (typed 422 at parse, before any compute) + defensive dedup (dict-by-node_id,
   first-seen order) in `_compute_evpi`.

3. **/health `compute_admission` advertisement** — `ComputeAdmissionInfo` on
   `HealthResponse`; `build_compute_admission()` assembles `max_cost_units`
   (env-resolved), `complexity_formula_version`, `weights`, `caps` from the SAME
   module constants the gate + model enforce (no internal mirror). This is what
   PLoT reads (PLoT PR separate).

4. **Calibration harness** — `benchmarks/admission_calibration.py` (runnable;
   `--quick`/`--runs`/`--out`). Fit + table below.

## Constants single-source map (derive, don't mirror — trap #12)

- Weights + ceiling + formula version: `src/services/robustness_analyzer_v2.py`
  (next to `EVPI_SAMPLE_CAP`). Read by `src/api/robustness.py` (both call sites) and
  `src/api/health.py`.
- Request-shape caps: `src/constants/__init__.py` (`MAX_GRAPH_NODES=50`,
  `MAX_GRAPH_EDGES=200`, `MAX_OPTIONS=10`, `MAX_PARAMETER_UNCERTAINTIES=50`). Read by
  the model `Field(max_length=...)` AND `/health` caps — the advertised cap can
  never differ from the enforced cap.

## ENV-SEMANTICS HAZARD — investigated (read-only Render API), CLEAN

- **`ISL_MAX_COMPUTE_COMPLEXITY` is NOT set on isl-staging** (`srv-d4fmjpkhg0os73948t30`,
  9 env vars) **nor on isl-production** (`srv-d4sm7nggjchc738p65o0`, 6 env vars). The
  old scalar guard ran on its 30M code default.
- The new ceiling uses a **NEW** env name **`ISL_MAX_COST_UNITS`** (also NOT set on
  either service). The code does **NOT** read the old `ISL_MAX_COMPUTE_COMPLEXITY`
  for the new ceiling — so there is no silent repurposing of an old-units value.
- **No env action is REQUIRED at deploy** for correctness (nothing to remove). If an
  operator ever sets `ISL_MAX_COMPUTE_COMPLEXITY` it will be **ignored** by the new
  formula (intended). To adjust the new ceiling, set `ISL_MAX_COST_UNITS` (cost units).

## RED-first positive controls (captured on pristine base `7ca214c2`)

`scratchpad/probe_base.py` run against the untouched clone:

- **PC-1** 3 identical `demand` parameter_uncertainties → **3 factor_evpi rows,
  byte-identical** (`evpi=0.0, current=perfect=0.5, n=1000` x3) — the free-ride is
  visible. (positive control TRUE)
- **PC-2** dense-mid 40n/120e/5000s/10opt: OLD scalar = 24,000,000 ≤ 30M → **admits**.
- **PC-3** deep single 40n/120e/10000s/1opt: OLD scalar = 48,000,000 > 30M →
  **wrongly rejects** (the `x-edges` shape defect).

## AFTER-fix behaviour (provisional ceiling = 20,000,000 cost units)

| request | cost_units | dominant | verdict | pre-F8 scalar |
|---|---|---|---|---|
| Pilot 5n/8e/1000s/2opt | 67,600 | sensitivity | ADMIT | 40K admit |
| Upper-PoC 12n/100e/5000s/3opt | 3,229,200 | sensitivity | ADMIT | 6M admit |
| Dense-mid 40n/120e/10000s/4opt | 14,080,000 | sensitivity | **ADMIT** | 48M → today 422 |
| PC-3 deep single 40n/120e/10000s/1opt | 9,280,000 | sensitivity | **ADMIT** | 48M → today 422 |
| PC-2 dense-mid 40n/120e/5000s/1opt | 8,480,000 | sensitivity | ADMIT | 24M admit |
| Dense-mid 40n/120e/10000s/10opt | 23,680,000 | base_mc | REJECT | 48M reject |
| PC-2 dense-mid 40n/120e/5000s/10opt + EVPI5 | 34,880,000 | evpi | **REJECT** | 24M → today admit (free-ride) |
| Schema-max 50n/200e/10000s/10opt | 45,000,000 | base_mc | REJECT | 100M reject |

- **PC-1 after:** duplicate parameter_uncertainties → `ValidationError` "Duplicate
  parameter_uncertainties node_ids ... ['n0']" → typed 422, zero EVPI compute.
- **Boundary:** 51 uncertainties → 422 (`too_long`); exactly-50 unique → admit.

## Calibration fit (LOCAL hardware — indicative only; staging recalibration OWED)

Full table + admit/reject-at-candidate-ceilings in **`admission-calibration.md`**
(this dir). Headlines:

- `k_ms_per_unit = 7.95e-4` (through origin, 21 cells); corroborated by two prior
  runs (8.26e-4, 7.82e-4). Per-cell ratio 4.6e-4..1.22e-3 (median 7.46e-4) — the
  structural shape tracks wall-clock; ~3x spread is phase-mix (base-heavy low,
  sensitivity-heavy high).
- Aggregate 25s-target ceiling ≈ 31.5M, but sensitivity-heavy graphs hit 25s at a
  LOWER cost (`schema-max-1opt` 22.5M measured **24.7s**), so the ceiling must sit
  below the aggregate. **Shipped provisional 20M.**

## Mutation checks (throwaway copies; PYTHONPATH-loaded, verified isl-mut/isl-mut-b)

- **Mutation A** — revert admission to the pre-F8 scalar (`S*N*E`, 30M ceiling):
  **5 RED** — PC-2 reject (unit + v2 + legacy endpoints) and PC-3 admit (unit +
  endpoint) all flip; the 1-option admit control stays GREEN. The weighted
  formula + recalibrated ceiling is what bites.
- **Mutation B** — revert the uniqueness validator (existence-only): **2 RED** —
  both PC-1 duplicate-rejection tests flip; distinct-nodes, boundary, and the
  dedup control stay GREEN.
- **Mutation C** — revert the `_compute_evpi` defensive dedup (iterate the raw
  list): the defence-in-depth dedup test goes **RED** (3 rows, not 1).

## Provisional ceiling — for Paul's leniency sign-off

- **Shipped provisional `DEFAULT_MAX_COST_UNITS = 20,000,000`** (env-adjustable via
  `ISL_MAX_COST_UNITS`). **PROVISIONAL, NOT finalized** — local-hardware fit only.
- **Leniency shape:** admits every realistic user graph up to dense-mid at max
  sampling depth (`dense-mid-4opt-deep` 14.08M, measured ~9–11s local); rejects only
  the genuinely heavy combos: 10-option base at max depth (23.68M) and the
  multi-option × multi-EVPI free-ride (34.88M). Schema-max sensitivity-heavy
  (22.5M) measured **~26s local — already over the 25s target**, confirming rejection
  is correct, not over-strict.
- **Decision for Paul:** the single borderline case is `dense-mid-10opt` (23.68M,
  ~14.6s local) — rejected at 20M. Admitting it would need ~24M, which is within
  ~1M of schema-max (22.5M/26s, which must reject). Candidates: **15M** (stricter),
  **20M** (shipped), **~30M** (local-25s aggregate fit — too lenient for
  sensitivity-heavy graphs). Recommend keeping 20M provisional; finalize after
  staging recalibration.
