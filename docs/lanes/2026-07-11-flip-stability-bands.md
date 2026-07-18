# Seed-sweep flip-threshold stability bands (Track S Phase 1)

**Lane:** ISL FLIP-STABILITY BANDS · **Base:** origin/staging `e029cae2d` ·
**Branch:** `claude/flip-stability-bands` · **Date:** 2026-07-11 · **Flag:**
`ISL_FLIP_STABILITY_BANDS` (default OFF)

> **STATUS UPDATE 2026-07-17 — bands are now DEFAULT-ON; the flag description
> below is historical.** Per Paul's ruling (17 Jul: core functionality, no
> flag unless genuinely needed; lenient-latency amendment: prioritise
> analysis quality), the `ISL_FLIP_STABILITY_BANDS` and
> `ISL_FLIP_STABILITY_SEEDS` env vars are **REMOVED**. Bands are computed
> whenever `edge_e_values` are; N is the code constant
> `FLIP_STABILITY_N_SEEDS = 10` (raised from 5 for a better stability
> basis; no runtime override, so the old [2, 20] clamp validation is
> deleted as dead); `FLIP_STABILITY_BUDGET_MS` is raised 2000 → **30000 ms**
> (Paul-ruled lenient default). The honest degradation MECHANICS are
> unchanged: all-or-nothing on exceed, base values untouched, disclosed via
> the `flip_stability_budget_exceeded` log event carrying `elapsed_ms`.
> Rollback is a revert commit, not a flag flip. The golden fixture is
> renamed `golden_base_v2.json` and now pins wire-modulo-`stability` ==
> pre-bands base wire; both new constants are VALUE-pinned by tests so a
> silent revert goes RED. The runtime-budget table below is the n=5 record;
> the default-on lane re-measured n=10 (see its PR). Everything below is
> the accurate record of what #71 shipped on 2026-07-11.

> **STATUS UPDATE 2026-07-18 — band budget degradation now ALSO rides the
> wire (A3 remediation).** The 07-17 note above says the honest degradation
> is "disclosed via the `flip_stability_budget_exceeded` LOG event" — that is
> now INCOMPLETE. A band budget trip (and the new governing-request-budget
> skip) additionally emits a structured `STABILITY_BANDS_UNAVAILABLE`
> `inference_warning` carrying `elapsed_ms`, so the absence is visible on the
> response wire PLoT reads — not log-only (the #226 gap for `flip_thresholds`,
> now closed for bands). The sweep also runs under a governing
> `OVERALL_REQUEST_BUDGET_MS` (50000 ms, below PLoT's 60s ISL timeout): base MC
> + core robustness are always returned; the optional phases degrade-and-
> disclose. All-or-nothing band mechanics and byte-identical determinism are
> unchanged. See `RobustnessAnalyzerV2.analyze` and
> `tests/unit/test_request_budget.py`.

## Problem

The corpus sweep found which-option flip rates of 25–75% across seeds, yet
every flip threshold ISL reports (`edge_e_values[].flip_mean`) is a single
point searched against ONE background — all other edges held at their
expected values. A consumer sees one number with no indication of how much
it moves under the graph's own stated uncertainty: false stability.

The 2026-06-10 PLoT/ISL science-performance report
(`plot-isl-science-performance-report-2026-06-10.md`) recommends: "report
flip thresholds with a **stability band from a small seed sweep** (e.g. 5
seeds …), and base 'flip confidence' on that band width, not on
recommendation margin alone" (§7; also next-steps #6 "Seed-sweep stability
band tooling", and §8's per-sample flip-point distribution framing). The
2026-07-07 science-validation report (`docs/science-validation/REPORT.md`
§1, §5.1) independently showed single-seed flip evidence is weak. ROADMAP
row 2.29(a) tracks this leg.

## What shipped

**Seed-sweep mode for the v2 flip-threshold computation** — ISL-side, in the
same code path that produces `edge_e_values`:

- `RobustnessAnalyzerV2._attach_flip_stability_bands`
  (`src/services/robustness_analyzer_v2.py`): N child seeds SHA-256-derived
  from the master request seed (`{seed}:flip_stability:{i}` — the same
  sub-seed pattern as the marginal-switch and EVPI streams). Each child seed
  samples ONE full edge configuration from the joint uncertainty (existence
  Bernoulli × truncated-normal strength — identical semantics to the main
  MC's `DualUncertaintySampler`); backgrounds are shared across edges within
  a seed (common random numbers). Each edge's flip point is then re-searched
  against that sampled background with bisection semantics identical to
  `_compute_edge_e_values` (boundary check, 20 steps, `mean ×
  exists_probability` effective value, increase direction first).
- **Wire (additive only):** each `edge_e_values[]` entry gains a
  `stability` object when the flag is on: `n_seeds`, `n_seeds_flipped`,
  `seed_flip_means` (per-seed flip mean, `null` where that background admits
  no flip), and — when at least one seed flips — `band_min`, `band_median`,
  `band_max`, `band_width`. The four `band_*` keys are **omitted** (not
  null) when nothing flips, matching the v2 wire's `exclude_none`
  serialisation so the v1 (dict passthrough) and v2
  (`FlipStabilityBandV2` on `EdgeEValueV2.stability`,
  `src/models/response_v2.py`; adapter in `src/api/robustness.py`) wires
  carry the same shape.
- **Config:** `ISL_FLIP_STABILITY_SEEDS` overrides N; default **5** per the
  06-10 report recommendation; clamped to [2, 20]; unparseable/below-minimum
  → default (the sweep must never fail a request over configuration).
- **Budget:** `FLIP_STABILITY_BUDGET_MS = 2000`, all-or-nothing on exceed —
  no partial bands (partial attachment would bias readers toward whichever
  edges computed first). Mirrors the existing `E_VALUE_BUDGET_MS` semantics:
  band *presence* is budget-gated exactly as `edge_e_values` presence
  already is; band *content* is fully deterministic.

## Claims and their evidence (commands + counts)

Precise claim types, per the evidence doctrine:

1. **Flag-off wire byte-identity (pinned).** Claim type: no-additive-fields
   + numeric identity, modulo the four pre-existing volatile fields
   (`execution_time_ms`, envelope `timestamp`, `processing_time_ms`,
   `critiques[].id`), the pre-existing `list(set(...))` ordering of
   `fragile_edges_v1`/`robust_edges`, and the environment echo `build` —
   all of which vary on BASE code already (science-validation REPORT.md §3
   catalogue). Pinned by
   `tests/unit/test_flip_stability_bands.py::TestFlagOff::test_v2_wire_matches_base_golden`
   against `tests/fixtures/flip_stability/golden_flag_off_v2.json`, which
   was captured from **unmodified base code** (commit `6dbcdb6`, before any
   src change) and still passes after the implementation.
2. **Flag-on is additive-only.**
   `TestFlagOnAdditiveOnly::test_flag_on_changes_nothing_but_stability`:
   flag-on response with every `stability` key stripped == flag-off
   response, same process, volatile fields masked. Mechanism: the sweep uses
   only fresh SHA-256-derived `SeededRNG` instances — no shared RNG stream
   is consumed, so no base number can change.
3. **Determinism.** Same request+seed → byte-identical bands across fresh
   analyzer instances (`TestDeterminism::test_same_request_same_seed_byte_identical_bands`,
   JSON-serialised equality); different master seed → different bands
   (child-seed derivation is from the master seed). Cross-process: all band
   content is round(·, 6) floats from SHA-256-seeded PCG64 streams — no
   `set()` ordering, no wall-clock content. The only nondeterminism is band
   *presence* under the wall-clock budget guard, identical in kind to the
   existing `edge_e_values` budget behaviour.
4. **RED-first.** Commit `6dbcdb6` (tests + base-captured golden): 8
   failed / 4 passed (`ISL_AUTH_DISABLED=true poetry run pytest
   tests/unit/test_flip_stability_bands.py -q` — the 4 passes are the
   flag-off pins, which must pass on base by construction). Commit
   `2dbdb5b` (implementation): 12/12 passed. One contract refinement was
   folded into GREEN: the RED contract had `band_*: null` when nothing
   flips; pydantic `exclude_none` drops None model fields on the v2 wire
   (verified empirically; None *list elements* are preserved), so nulls
   would have made the v1/v2 wire shapes diverge — the contract became
   "omit `band_*` when `n_seeds_flipped == 0`".

## Runtime budget (measured, not estimated)

`poetry run python benchmarks/flip_stability_budget.py` — median of 21
analyze() calls per cell after 1 warm-up, flag-off vs flag-on at default
N=5, Apple-silicon dev machine, two independent runs:

| Graph | off ms | on ms | delta | ratio | edges banded |
|---|---|---|---|---|---|
| sample_variants[0] (3n/3e, n=200) | 2.8 | 3.2 | +0.5 | 1.16 | 3 |
| sample_variants[1] (3n/3e, n=200) | 2.5 | 3.0 | +0.5 | 1.19 | 3 |
| sample_variants[2] (3n/3e, n=200) | 2.5 | 3.0 | +0.5 | 1.20 | 3 |
| synthetic_12n_17e (12n/17e, 3 options, n=500) | 71.6 | 82.0 | +10.4 | **1.14** | 17 |

(Second run: 1.16/1.19/1.20/1.15 — stable. A first exploratory run at 7
repetitions showed one 0.58 "ratio" on a 3 ms fixture: pure process noise —
flag-on cannot be faster; on single-digit-ms fixtures the noise is the same
order as the measurement. The production-sized synthetic row is the honest
signal.)

**Verdict: ~1.15× at default N=5 on the representative graph — well inside
the ~2× budget. No adaptive N needed.** Worst case is additionally capped by
the 2000 ms all-or-nothing guard. Honest caveat: cost scales with
N × edges × options × bisect-steps, so a 50-node/200-edge graph at N=5 will
push toward the guard — the guard, not this measurement, is the production
protection there.

## Interpretation guardrails (for future consumers)

- `band_width` is the flip-confidence input the 06-10 report recommends —
  it is NOT a confidence interval; it is the range of the flip point across
  N=5 draws of the background uncertainty.
- **`n_seeds_flipped == 1` yields `band_width == 0.0` by construction** — a
  single flipped value has zero range (`band_min == band_median == band_max`).
  A naive width rubric ("narrow band ⇒ high flip confidence") would read
  *maximal* stability from a single flipped background — the opposite of the
  truth (only 1 of N sampled backgrounds even admits a flip). Downstream
  consumers **MUST condition `band_width` on `n_seeds_flipped`**: interpret
  width only when `n_seeds_flipped >= 2`, and treat a low `n_seeds_flipped`
  as its own (weak-flip-evidence) signal. Also flagged in the
  `FlipStabilityBandV2.band_width` field description on the wire schema.
- `n_seeds_flipped < n_seeds` means some sampled backgrounds admit no flip
  at all within [-1, 1] — that asymmetry is signal, not noise.
- No user-facing copy is generated from these fields yet; the band-based
  confidence rubric for chip copy (06-10 report next-step #6, ROADMAP
  2.29(a)) is downstream consumer work (PLoT/CEE), gated on doctrine review.

## Files touched

- `src/services/robustness_analyzer_v2.py` — flag helpers, band sweep, background flip search
- `src/models/response_v2.py` — `FlipStabilityBandV2`, `EdgeEValueV2.stability`
- `src/api/robustness.py` — v1→v2 adapter passthrough
- `openapi.json` — regenerated (`poetry run python scripts/generate_openapi.py`) to
  carry `FlipStabilityBandV2` and `EdgeEValueV2.stability` — the spec is
  generated from the models, so the model change widens the file scope here
- `tests/unit/test_flip_stability_bands.py` — 15 tests (pins + contract +
  forced zero-flip omission branch)
- `tests/fixtures/flip_stability/golden_flag_off_v2.json` — base-captured flag-off golden
- `benchmarks/flip_stability_budget.py` — budget measurement (this doc's table)

## Out of scope (unchanged)

- The base `edge_e_values` single-point computation and its budget.
- PLoT's probe-based flip thresholds (`flip-thresholds.ts`) — the 06-10
  report's PLoT-side sweep (rec #1/#6) is a separate lane; this lane is the
  ISL-native band primitive.
- The analytic/semi-analytic flip-threshold spike (report §8; ROADMAP
  2.29(b)) — would supersede probe sweeps entirely; zero implementation here.
- Reserved staging scenarios (1909b083*, def3cb31*, 8e0bf73d*, 88396c52*)
  were not touched; this lane made no staging-data or live-service calls.
