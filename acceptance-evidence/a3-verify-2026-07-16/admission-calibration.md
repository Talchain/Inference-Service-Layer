# F8 admission cost-model — calibration (PROVISIONAL)

Produced by `benchmarks/admission_calibration.py` (full grid, median of 3 runs,
reachable-goal graphs). **LOCAL HARDWARE — indicative only.**

> ⚠ **The ceiling is PROVISIONAL and NOT finalized.** This is a local-hardware
> fit; the Render **isl-staging** instance is a different (typically slower)
> class, so `MAX_COST_UNITS` must be **re-calibrated on staging** before it is
> locked. Env-adjustable via `ISL_MAX_COST_UNITS`. Surface the number to Paul
> before finalizing (leniency sign-off).

## Fit

- **`k_ms_per_unit = 7.95e-4 ms/unit`** (least squares through origin, 21 cells).
  Corroborated by two earlier runs: quick grid `8.26e-4`, full grid `7.82e-4`.
- Per-cell `t_ms / cost_unit` range **4.6e-4 .. 1.22e-3** (median `7.46e-4`) — a
  ~3x spread driven by phase mix (base-MC-heavy cells sit low ~5e-4/unit;
  sensitivity-heavy cells sit high ~1.1e-3/unit). The tight clustering confirms
  the **structural cost shape tracks wall-clock** (correct-by-construction).
- Aggregate suggested ceiling at `TARGET_WALL_MS=25000`: **~31.5M cost units**.
  But this aggregate is pulled UP by the cheap base-heavy cells; a
  **sensitivity-heavy** graph reaches 25s at a *lower* cost (`schema-max-1opt`
  = 22.5M units measured **24.7s**), so a single ceiling should sit **below** the
  aggregate suggestion. **Shipped (Paul-directed, provisional):
  `DEFAULT_MAX_COST_UNITS = 24,000,000`** — the more-lenient of the reviewed
  candidates (see below).
- Codex option-anchor (1opt→10opt runtime): **1.38x** here vs Codex's ~4.13x.
  Explanation (not a model error): my anchor cells are *sensitivity-dominated*
  (option-independent phase), so total runtime is not 4x from options; the base
  term IS exactly O-linear by construction (unit test `test_option_multiplier_present`
  asserts base 10opt == 10 × base 1opt). Codex's 4.13x was a base-MC-dominated
  fixture.

## Per-cell measurements (local, reachable-goal)

| cell | N | E | S | O | evpi | phases | cost_units | dominant | t_ms | ms/unit |
|---|---|---|---|---|---|---|---|---|---|---|
| pilot | 5 | 8 | 1000 | 2 | 0 | - | 67,600 | sensitivity | 41.8 | 6.19e-04 |
| small-4opt | 8 | 20 | 2000 | 4 | 0 | - | 448,000 | base_mc | 306.6 | 6.84e-04 |
| mid | 12 | 100 | 5000 | 3 | 0 | - | 3,229,200 | sensitivity | 2634.5 | 8.16e-04 |
| mid-10opt | 12 | 100 | 5000 | 10 | 0 | - | 5,959,200 | base_mc | 3278.8 | 5.50e-04 |
| dense-mid | 40 | 120 | 5000 | 4 | 0 | - | 10,880,000 | sensitivity | 7237.8 | 6.65e-04 |
| dense-mid-1opt-deep | 40 | 120 | 10000 | 1 | 0 | - | 9,280,000 | sensitivity | 10713.8 | 1.16e-03 |
| dense-mid-4opt-deep | 40 | 120 | 10000 | 4 | 0 | - | 14,080,000 | sensitivity | 12689.0 | 9.01e-04 |
| anchor-1opt | 12 | 30 | 5000 | 1 | 0 | - | 714,000 | sensitivity | 868.9 | 1.22e-03 |
| anchor-2opt | 12 | 30 | 5000 | 2 | 0 | - | 924,000 | sensitivity | 851.4 | 9.21e-04 |
| anchor-4opt | 12 | 30 | 5000 | 4 | 0 | - | 1,344,000 | base_mc | 968.0 | 7.20e-04 |
| anchor-10opt | 12 | 30 | 5000 | 10 | 0 | - | 2,604,000 | base_mc | 1197.5 | 4.60e-04 |
| evpi-3f | 12 | 30 | 5000 | 3 | 3 | - | 2,142,000 | evpi | 1429.6 | 6.67e-04 |
| evpi-5f-dense | 40 | 120 | 5000 | 4 | 5 | - | 18,560,000 | evpi | 13667.7 | 7.36e-04 |
| evalues | 12 | 40 | 3000 | 3 | 0 | ev | 1,326,400 | sensitivity | 1310.4 | 9.88e-04 |
| path | 20 | 60 | 3000 | 3 | 0 | path | 2,643,600 | sensitivity | 2558.5 | 9.68e-04 |
| upper-poc | 12 | 100 | 5000 | 3 | 0 | - | 3,229,200 | sensitivity | 3319.3 | 1.03e-03 |
| dense-mid-10opt | 40 | 120 | 10000 | 10 | 0 | - | 23,680,000 | base_mc | 12827.4 | 5.42e-04 |
| schema-max-1opt | 50 | 200 | 10000 | 1 | 0 | - | 22,500,000 | sensitivity | 24747.8 | 1.10e-03 |
| evpi-8f | 20 | 60 | 5000 | 4 | 8 | - | 9,280,000 | evpi | 5551.3 | 5.98e-04 |
| evalues+path | 20 | 60 | 5000 | 3 | 0 | ev,path | 3,163,200 | sensitivity | 2969.4 | 9.39e-04 |
| full-load | 30 | 90 | 5000 | 4 | 4 | ev,path | 11,607,300 | evpi | 8662.0 | 7.46e-04 |

## Admit/reject vs candidate ceilings (Paul reviewed → chose 24M)

Sorted by cost. Local wall-clock shown where measured. `15M` stricter · `20M`
conservative-lenient · **`24M` SHIPPED (Paul-directed, more lenient)** · `30M`
aggregate-fit.

| graph (realistic?) | cost_units | local t | 15M | 20M | **24M** | 30M |
|---|---|---|---|---|---|---|
| pilot 5n/8e/1000s/2opt (yes) | 67,600 | 0.04s | ✅ | ✅ | ✅ | ✅ |
| upper-poc 12n/100e/5000s/3opt (yes) | 3,229,200 | 3.3s | ✅ | ✅ | ✅ | ✅ |
| PC-2 dense-mid 40n/120e/5000s/1opt (yes) | 8,480,000 | ~7s | ✅ | ✅ | ✅ | ✅ |
| PC-3 dense-single 40n/120e/10000s/1opt (yes) | 9,280,000 | 10.7s | ✅ | ✅ | ✅ | ✅ |
| dense-mid 40n/120e/5000s/4opt (yes) | 10,880,000 | 7.2s | ✅ | ✅ | ✅ | ✅ |
| full-load 30n/90e/5000s/4opt+evpi4+ev+path (yes) | 11,607,300 | 8.7s | ✅ | ✅ | ✅ | ✅ |
| dense-mid-4opt-deep 40n/120e/10000s/4opt (yes) | 14,080,000 | 12.7s | ✅ | ✅ | ✅ | ✅ |
| evpi-5f-dense 40n/120e/5000s/4opt+evpi5 (yes) | 18,560,000 | 13.7s | ❌ | ✅ | ✅ | ✅ |
| schema-max-1opt 50n/200e/10000s/1opt (heavy) | 22,500,000 | **24.7s** | ❌ | ❌ | **✅** | ✅ |
| dense-mid-10opt 40n/120e/10000s/10opt (heavy) | 23,680,000 | 12.8s | ❌ | ❌ | **✅** | ✅ |
| PC-2 dense-mid 40n/120e/5000s/10opt+evpi5 (abuse) | 34,880,000 | ~28s* | ❌ | ❌ | ❌ | ❌ |
| schema-max 50n/200e/10000s/10opt (abuse) | 45,000,000 | ~30s+* | ❌ | ❌ | ❌ | ❌ |

`*` extrapolated (exceeds the internal OVERALL_REQUEST_BUDGET_MS optional-phase
degrade points).

### Reading the table (Paul's decision)

- **24M (SHIPPED, Paul-directed 2026-07-18)** — Paul reviewed the candidates and
  chose the more-lenient ceiling. At 24M every graph up to and including the two
  "heavy" cases **admits**: `dense-mid-10opt` (23.68M, 12.8s) and `schema-max-1opt`
  (22.5M, **24.7s**). The 24.7s worst-admitted case still completes **well inside
  ISL's OVERALL_REQUEST_BUDGET_MS = 50000 and PLoT's 60s timeout**, so it returns
  cleanly (the optional phases degrade-and-disclose if they run long); **F15's
  compute governor is the next line of resource defense** for concurrency.
- Only the genuinely-abusive multi-option × multi-EVPI combos reject:
  PC-2 10opt+EVPI5 (34.88M) and schema-max 10opt (45M).
- **Still PROVISIONAL** — this is a LOCAL-hardware fit; **staging recalibration is
  owed** (the Render isl-staging instance is a different, typically slower class,
  so the 24.7s local worst-case will be larger there — re-measure before locking,
  and confirm it stays inside the 50s budget on staging).
