# Science-validation report — robustness v2

Date: 2026-07-07.
Harness: `benchmarks/science-validation/` (this repository; no `src/` changes).
All seeds are pinned in `benchmarks/science-validation/_lib.py` (`SEEDS`).
Raw outputs with provenance: `benchmarks/science-validation/results/`.

Every claim below carries the command that reproduces it. Commands run from
the repository root under `poetry install`ed dependencies. Negative results
are stated as plainly as positive ones.

**Spec note.** The brief referenced `docs/slice3-log-and-isl-higherK-spec.md`
in the `plot-lite-service` repository. That file is not present there (checked
`main`, all remote branch names, and GitHub code search for "higherK",
2026-07-07). The experiment design below therefore follows the brief's
description alone; if the spec surfaces, its requirements should be diffed
against §1.

---

## Summary of verdicts

| Question | Verdict |
|---|---|
| Are the live uniform zeros in `marginal_switch_probability` true zeros? | **Not decidable at K=100, and in the fixtures examined they are not all true zeros.** Two fixture edges are genuinely under-resolved non-zeros (p ≈ 6.6×10⁻⁴ and 8×10⁻⁵); one edge per fixture graph is a true zero (bounded below 5.4×10⁻⁶). At K=100, single-seed zeros also occur for edges with p as high as ≈ 0.025. |
| Is the marginal-switch estimator itself sound? | **Yes.** Unbiased against closed-form truth across ten analytic cases (all exact binomial p-values ≥ 0.10); no cross-seed instability detected anywhere (all homogeneity p-values ≥ 0.01). |
| Is the EVPI noise floor conservative, tight, or leaky? | **Regime-dependent: conservative away from the decision boundary (SD 0.5–0.9× the worst-case SE), exactly tight at the knife edge, with the ≈5% false-"resolved" rate a z95 bound implies.** Not leaky beyond its stated 95% coverage. |
| Is same-seed output byte-identical? | **No, as shipped.** Four volatile fields (`metadata.execution_time_ms`, envelope `timestamp`, envelope `processing_time_ms`, `critiques[].id`); after masking them, 50/50 identical in-process and over the wire — but **cross-process, 42/50 still differ**: `fragile_edges`/`robust_edges` ordering follows the per-process hash salt and the `interpretation` string names a process-dependent subset of edges. All numeric content is cross-process deterministic (50/50 once ordering is normalised). |
| Does 72% mean 72%? | **Without auto-noise: yes, within Monte Carlo error** (unbiased vs closed-form truth, nominal coverage). **With auto-noise (outcome/risk goals): no** — reported probabilities deviate from the no-noise truth by up to ≈ 0.17 on the tested grid; the deviation is the auto-noise heuristic itself, applied as documented. |

---

## 1. Higher-K: resolution of `marginal_switch_probability`

Reproduce: `poetry run python benchmarks/science-validation/exp1_higher_k.py`
(seeds 42, 1042, 2042, 3042, 4042; K ∈ {100, 1000, 10000, 100000};
results file `results/exp1_higher_k.json`).

### Method

`_compute_marginal_switch_probability` is the unmodified production estimator;
only its `k_samples` argument is supplied explicitly by the harness (the
production call site always uses the default `MARGINAL_K_SAMPLES = 100`).
A regression guard verifies the harness override is transparent at K=100.
Because the per-edge RNG is seeded once per (seed, edge) from a SHA-256
sub-seed, the first 100 draws at K=100000 are identical to the K=100 draws —
higher K strictly extends the same stream.

Two graph families:

- **Margin family** (10 graphs): a single decisive edge whose flip probability
  is known in closed form — the winner flips exactly when the sampled strength
  is negative, so p = F_TruncNormal(0; mean, std, [−1, 1]) — spanning
  knife-edge (p ≈ 0.4) to effectively zero (p ≈ 3×10⁻⁶). Every graph also
  contains a **structural true zero**: an edge into a node that every option
  intervenes on (interventions override structural equations, so no draw of
  that edge can change any option's outcome).
- **Repo fixtures**: every edge of the three pinned graphs in
  `tests/benchmarks/sample_variants.json`.

### Estimator validation (margin family, closed-form truth)

Pooled estimates over all K and all five seeds, exact binomial test against
the analytic probability:

| Case | Analytic p | Pooled p̂ | Binomial p-value |
|---|---|---|---|
| knife_edge_p0.4 | 0.400018 | 0.399524 | 0.28 |
| moderate_p0.05 | 0.049995 | 0.049694 | 0.10 |
| moderate_p0.02 | 0.020002 | 0.019844 | 0.16 |
| floor_p0.01 | 0.010001 | 0.010002 | 0.81 |
| under_res_p0.005 | 0.005001 | 0.005066 | 0.41 |
| under_res_p0.002 | 0.002000 | 0.002020 | 0.66 |
| under_res_p0.001 | 0.001000 | 0.000992 | 0.70 |
| under_res_p0.0003 | 0.000300 | 0.000280 | 0.35 |
| under_res_p0.0001 | 0.000100 | 0.000100 | 0.89 |
| near_zero_p3e-6 | 0.000003 | 0.000002 | 1.00 |

No case rejects the analytic truth; no case shows cross-seed heterogeneity
beyond binomial noise (all Pearson χ² homogeneity p-values ≥ 0.01, most far
above). The estimator is sound; the only problem at K=100 is resolution.
The ten structural-zero edges returned 0 in every one of the 555,500 draws
each, as construction requires.

### Per-edge classification table (fixtures + margin family)

Classification rules: **TRUE ZERO** — zero flips in every draw at every K and
seed (with the pooled rule-of-three 95% upper bound reported);
**UNDER-RESOLUTION NON-ZERO** — zero at K=100 for all five seeds, but non-zero
pooled at K=100000; **UNSTABLE** — cross-seed dispersion inconsistent with
binomial sampling (χ² p < 0.01); **RESOLVED NON-ZERO** otherwise.

| Graph | Edge | Class | Pooled p̂ at K=10⁵ | Note |
|---|---|---|---|---|
| sample_variants[0] | price→demand | RESOLVED NON-ZERO | 0.0117 | zero at K=100 on some single seeds |
| sample_variants[0] | demand→revenue | **TRUE ZERO** | 0 (< 5.4×10⁻⁶) | |
| sample_variants[0] | price→revenue | RESOLVED NON-ZERO | 0.0248 | zero at K=100 on some single seeds |
| sample_variants[1] | price→demand | **UNDER-RESOLUTION NON-ZERO** | 6.6×10⁻⁴ | uniformly 0 at K=100, all 5 seeds |
| sample_variants[1] | demand→revenue | **TRUE ZERO** | 0 (< 5.4×10⁻⁶) | |
| sample_variants[1] | price→revenue | **UNDER-RESOLUTION NON-ZERO** | 8×10⁻⁵ | uniformly 0 at K=100, all 5 seeds |
| sample_variants[2] | price→demand | RESOLVED NON-ZERO | 0.2237 | |
| sample_variants[2] | demand→revenue | **TRUE ZERO** | 0 (< 5.4×10⁻⁶) | |
| sample_variants[2] | price→revenue | RESOLVED NON-ZERO | 0.2343 | |
| margin family ×10 | upstream→lever | **TRUE ZERO** (structural) | 0 | flip impossible by construction |
| margin family | lever→goal | per table above | — | matches analytic truth |

Seed-reproducibility: repeated calls with the same (seed, K) return bitwise
identical estimates (`reproducibility.identical_on_repeat = true` in the
results file); distinct seeds give independent draws whose dispersion matched
binomial expectation in every cell.

### The wire-level view (what a consumer actually sees)

Full `analyze()` under the K override, fixture graphs, seed 42
(`end_to_end` block of the results file):

| Graph, seed 42 | K=100 | K=1000 | K=10000 | K=100000 |
|---|---|---|---|---|
| variants[0] price→revenue | **0.0** | 0.018 | 0.0255 | 0.02488 |
| variants[0] price→demand | **0.0** | 0.006 | 0.0117 | 0.01194 |
| variants[0] demand→revenue | 0.0 | 0.0 | 0.0 | 0.0 |
| variants[1] price→demand | **0.0** | 0.001 | 0.0005 | 0.00055 |
| variants[1] price→revenue | **0.0** | 0.0 | 0.0001 | 0.00005 |
| variants[2] price→revenue | 0.2 | 0.216 | 0.2357 | 0.23488 |

At seed 42, `sample_variants[0]` reports **all three** fragile edges as 0.0 at
K=100 — the live "uniform zeros" phenomenon reproduced — although two of those
edges have true probabilities of ≈ 0.025 and ≈ 0.012, both at or above the
nominal 0.01 resolution floor. The mechanism: a single-seed zero at K=100
occurs with probability (1−p)¹⁰⁰ ≈ e^(−100p), i.e. ≈ 8% at p = 0.025 and
≈ 30% at p = 0.012. A zero at K=100 is therefore weak evidence even about
probabilities well above the floor, and says almost nothing about the
below-floor band.

### Runtime cost of higher K

Direct per-edge estimation on the fixture graphs averaged ≈ 1.3 ms per 1000
draws per edge (three-node graphs, two options; full sweep of exp1:
106 s). Extrapolated: K=10000 costs ≈ 13 ms per fragile edge on small graphs —
compatible with a request budget — but scales linearly in K × options × graph
size; a production K increase for 50-node/200-edge graphs needs its own
benchmark before being costed (see STATUS.md next steps).

---

## 2. EVPI noise-floor validation

Reproduce: `poetry run python benchmarks/science-validation/exp2_evpi_floor.py`
(policy seed 42; 20 replicate seeds 7000, 7013, …, 7247;
results file `results/exp2_evpi_floor.json`).

### Method

Production floor: `evpi_noise_floor(n) = 1.96·√(0.5/n)`, the two-sided 95%
bound on the worst-case (p = 0.5) standard error of a difference of two
independent proportion estimates. Since production caps EVPI at
n = min(n_samples, 500), n ∈ {2000, 10000} is unreachable through `analyze()`;
the harness replicates only the thin orchestration of `_compute_evpi`
(identical seed derivations: baseline streams seed+100/seed+101, per-factor
SHA-256 sub-seeds) and calls the unmodified production `_compute_evpi_metric`.

One graph pair — comfortable winner (P(win) = 0.864) and knife-edge
(P(win) = 0.547) — each with four uncertain factors chosen for their true
EVPI: `driver` (material), `weak` (small), `common` (exactly zero — see
structural finding below), `stranded` (no causal path to goal; exactly zero).

### Results

Empirical SD of the EVPI estimator across 20 seeds vs the floor's implied
worst-case SE (floor/1.96):

| Regime | n | SD/worst-case-SE across factors | Reading |
|---|---|---|---|
| comfortable | 500 | 0.48 – 0.85 | conservative |
| comfortable | 2000 | 0.63 – 0.99 | conservative to tight |
| comfortable | 10000 | 0.56 – 0.97 | conservative to tight |
| knife-edge | 500 | 0.72 – 0.88 | tight |
| knife-edge | 2000 | 1.09 – 1.30 | at the bound (see note) |
| knife-edge | 10000 | 0.77 – 1.26 | at the bound (see note) |

Note: ratios above 1 are consistent with estimation noise — the SD of a
20-replicate SD estimate is ≈ 16% of its value, and the variance of a
difference of two independent [0,1]-bounded means cannot mathematically exceed
0.5/n. The correct reading is that **at the knife edge the floor has no spare
margin at all**: the worst case p = 0.5 the formula assumes is the actual
operating point.

**Sign stability.** Among below-floor estimates, the sign disagreed with the
20-seed consensus in 25–61% of replicates (true-zero factors sit near 50%, as
pure noise should). A below-floor EVPI's sign carries no information —
supporting the labelling doctrine.

**False-"resolved" rate (leakage).** For the two true-zero factors
(`common`, `stranded`), 5 of 120 knife-edge estimates and 3 of 120 comfortable
estimates exceeded the floor and would be labelled `resolved` — ≈ 3–4%,
consistent with the ≈ 5% a z95 bound is designed to allow. **`resolved` is a
95% label, not a guarantee**: roughly one in twenty truly-zero factors will be
labelled resolved at any n.

**n-scaling.** Raising n from 500 to 10000 shrinks both floor and empirical SD
by the expected √20 ≈ 4.5×; the material `driver` EVPI (+0.153) is unaffected
in mean. Nothing anomalous appears at higher n; the production n = 500 cap is
a latency choice, not a statistical one.

### Structural finding: common-mode factors have exactly zero EVPI under p_win

In ISL's linear SCM, the winner is the argmax across options of outcomes that
share every non-intervened term. A factor whose causal path is not intervened
on by ANY option contributes identically to all options, cancels in the
comparison, and therefore has **true EVPI exactly zero under the
p_win_recommended metric — regardless of how strongly it drives the goal**
(demonstrated by `common`, which has a strength-0.6 edge into the goal and an
EVPI mean of ≈ 0.000 across 60 cells). Factor EVPI is non-zero only when at
least one option intervenes on the factor's path (severing its influence for
that option) — or when goal_constraints switch the metric to p_joint_goal,
which is threshold-based rather than comparative. Consequence: on graphs where
options only intervene on lever nodes off the uncertain factors' paths, the
entire `factor_evpi` block is Monte Carlo noise around zero by construction.

---

## 3. Determinism at scale

Reproduce: `poetry run python benchmarks/science-validation/exp3_determinism.py`
(graph-generator seed 20260707, request seed 42;
results file `results/exp3_determinism.json`).

### Method

50 deterministically generated diverse graphs (2–30 nodes; factor/chance/
outcome kinds; ~30% with factor uncertainties; ~40% with goal thresholds;
~20% with goal constraints; VoI / e-values / path-decomposition toggles;
n_samples ∈ {100, 300, 500}; `request_id` and `seed` pinned). Three
comparisons per graph: in-process repeat (fresh analyzer instances), wire
repeat (TestClient, `X-ISL-Response-Version: 2`), and cross-process (a child
Python interpreter regenerates the same graphs and reports response hashes).

### Results (out of 50 graphs)

| Comparison | Byte-identical |
|---|---|
| in-process, raw JSON | 11/50 |
| in-process, execution_time_ms zeroed | 37/50 |
| in-process, four volatile fields masked | **50/50** |
| wire, raw bytes | 0/50 |
| wire, four volatile fields masked | **50/50** |
| cross-process, four volatile fields masked | 8/50 |
| cross-process, additionally order-normalised | **50/50** |

### Volatile-field catalogue (complete, as observed)

1. `metadata.execution_time_ms` (in-process) / `_metadata.execution_time_ms`
   (v2 wire envelope) — wall clock.
2. Envelope `timestamp` — wall clock (differed on 50/50 wire repeats).
3. Envelope `processing_time_ms` — wall clock.
4. `critiques[].id` — **`uuid.uuid4()` per run** (`src/models/critique.py:34`).
   Fires on any graph that produces a critique (34/50 here). Same-seed
   responses can never be byte-identical while this stands; the critique
   *content* is fully deterministic.

`request_id` is also volatile when not client-pinned (analyzer:
`robustness-{uuid4}`; endpoint: `isl-{uuid4}`); pinned here by design.

### Cross-process finding: set-order leaks into the response

With the four fields above masked, same-seed responses from a *different OS
process* still differed on 42/50 graphs. Cause (localised by controlling
`PYTHONHASHSEED`): `robustness.fragile_edges` and `robustness.robust_edges`
are materialised via `list(set(...))`
(`src/services/robustness_analyzer_v2.py:2792–2805`), so their order follows
Python's per-process string-hash salt. Consequentially,
`robustness.interpretation` — which names the first three entries of that
unordered list — cites a **process-dependent arbitrary subset of the fragile
edges**: the same request produced both
"… sensitive to: n0->n4, n3->n4, n1->n7" and
"… sensitive to: n3->n4, n0->n1, n1->n5" (same seed, same numbers).

After sorting the two edge lists and masking `interpretation`, cross-process
responses are identical on **50/50** graphs — every numeric quantity
(probabilities, sensitivities, EVPI, e-values, path decompositions,
`fragile_edges_enhanced` including its order) is cross-process deterministic.
The PCG64/SHA-256 seeding design holds; the leak is purely the two
`list(set(...))` orderings and the text derived from them. Recommended
one-line fix (`sorted(...)`) recorded in §5; not made in this lane.

---

## 4. Calibration groundwork: does 72% mean 72%?

Reproduce: `poetry run python benchmarks/science-validation/exp4_calibration.py`
(seeds 42, 1042, 2042, 3042, 4042; n_samples 10000;
results file `results/exp4_calibration.json`).

### Method

Single-edge graphs: goal = x · B · T with B ~ Bernoulli(exists_probability),
T ~ TruncNormal(mean, std, [−1, 1]), and option interventions x ∈ {1.0, 0.5} —
so P(goal ≥ threshold) has a closed form (truncated-normal CDF × existence
mixture). Nine grid points over (mean, std, exists_probability, threshold),
two options, five seeds, n_samples = 10000 (the schema maximum): 90 cells per
goal kind. Goal kind `chance` leaves the Monte Carlo output untouched; goal
kind `outcome` triggers the auto-scaled noise heuristic
(N(0, std(samples)) added per sample; flagged
`provisional_pending_pilot_calibration`). For the outcome kind the harness
also computes the *noisy* truth — the analytic goal distribution convolved
with matched Gaussian noise — to separate "implementation wrong" from
"heuristic changes the target".

### Results

| Goal kind | Truth compared | Mean error | Max abs error | 95% MC-band coverage |
|---|---|---|---|---|
| chance (no auto-noise) | closed form | −0.0011 | 0.0147 | 86/90 = 95.6% |
| outcome (auto-noise) | noisy (convolved) truth | +0.0001 | 0.0108 | 87/90 = 96.7% |
| outcome (auto-noise) | clean closed form | +0.0166 | **0.1738** | — |

- **Without auto-noise, probability_of_goal is well calibrated within the
  model**: unbiased (mean error −0.001), nominal coverage (95.6% of cells
  inside the 95% Monte Carlo band, max |z| = 2.94 across 90 cells). Within
  the model's own assumptions, 72% does mean 72% ± MC error.
- **The auto-noise implementation does exactly what its documentation says**:
  against the noise-convolved truth it is unbiased with nominal coverage.
- **But auto-noise changes what the number means.** Worst grid cell
  (mean 0.7, std 0.1, exists 0.9, threshold 0.6, x = 1.0): the model's own
  probability is 0.757; ISL reports ≈ 0.590 (noisy truth 0.5895, so the
  implementation is accurate about the wrong-for-the-question quantity).
  The distortion is largest where the clean probability is far from 0.5 and
  the threshold sits inside the noise-widened distribution — i.e. precisely
  for confident answers. This is a property of the heuristic, not a bug.
- Scope: this is within-model calibration groundwork (complementing the SBC
  machinery in `src/services/sbc_validator.py`). It says nothing about
  calibration against reality — that requires pilot outcome data, as the
  provisional flag already states.

---

## 5. Doctrine-relevant findings (for Neil / Jinghui validation sweep)

Flagged for the later doctrine pass; wording below describes internal
behaviour and is not proposed user-facing language (EVPI user-facing language
remains banned pending doctrine).

1. **A zero at K=100 must not be rendered as "no sensitivity".** §1 shows
   K=100 zeros arising from edges with true switch probability up to ≈ 0.025.
   If the Slice-3 doctrine distinguishes TRUE ZERO from UNDER-RESOLUTION, the
   deciding evidence must be either a higher-K measurement or an explicit
   bound (e.g. rule-of-three: "0 flips in K draws ⇒ p < 3/K at 95%"), never
   the K=100 point estimate alone. The classification scheme in §1
   (TRUE ZERO / UNDER-RESOLUTION NON-ZERO / UNSTABLE / RESOLVED) is
   implementable as labels without changing any numeric value, matching the
   labels-over-clamps precedent.
2. **A TRUE-ZERO label is itself only a bound.** Even at K=100000 × 5 seeds,
   "true zero" means p < 5.4×10⁻⁶ at 95% — the margin family's p ≈ 3×10⁻⁶ case
   correctly lands in the zero bucket. Doctrine wording should say "below
   measurable resolution", not "cannot switch", unless the zero is structural
   (an intervened-on path, §1 method), which IS provable.
3. **`evpi_status = "resolved"` is a 95% claim.** ≈ 1 in 20 truly-zero factors
   will be labelled resolved (§2). Any downstream rendering should treat
   `resolved` as "distinguishable from noise at 95%", not "real".
4. **Common-mode factors: `factor_evpi` is structurally zero under p_win**
   (§2 structural finding). Consider labelling factors whose paths no option
   intervenes on, rather than reporting their pure-noise EVPI estimates at
   all; at minimum, doctrine should preclude narrating such values.
5. **Auto-noise changes what probability_of_goal means** (§4): with
   outcome/risk goals the reported probability is the probability under
   model-plus-matched-noise, not the model's own probability — differences
   reach ≈ 0.17 on the tested grid. The existing
   "provisional_pending_pilot_calibration" flag is scientifically accurate;
   any user-facing probability wording should not imply the model's own
   probability while auto-noise is active.
6. **The interpretation string's "sensitive to:" list is process-dependent**
   (§3): it names the first three entries of an unordered set, so two runs of
   the same request on different processes cite different edges. Independently
   of the determinism fix, doctrine should not treat the cited subset as a
   ranking — it is arbitrary. (The doctrine rule "never 'sensitive to X' for
   zero-sensitivity factors" is not violated — listed edges do exceed the
   fragile threshold — but the *selection* among them carries no meaning.)
7. **Recommended src changes (NOT made — this lane is docs/benchmarks only):**
   a. `critiques[].id` is `uuid.uuid4()` per run (`src/models/critique.py:34`),
      breaking same-seed byte-equality whenever a critique fires; derive it
      from a hash of (seed, code, affected ids) instead.
   b. `fragile_edges` / `robust_edges` are `list(set(...))`
      (`robustness_analyzer_v2.py:2792–2805`): apply `sorted(...)` so ordering
      (and the interpretation text derived from it) stops depending on the
      per-process hash salt.
   c. If Slice-3 adopts higher K: `k_samples` is already threaded through
      `_compute_marginal_switch_probability`; only the call site constant and
      a budget guard need touching. Prefix-consistency of the per-edge stream
      means a K increase refines rather than replaces K=100 results at the
      same seed.

---

## 6. Reproduction appendix

```bash
poetry install
# Everything (full grids, ~15-20 min):
poetry run python benchmarks/science-validation/run_all.py
# Smoke (~2 min):
poetry run python benchmarks/science-validation/run_all.py --quick
# Individually:
poetry run python benchmarks/science-validation/exp1_higher_k.py    # §1
poetry run python benchmarks/science-validation/exp2_evpi_floor.py  # §2
poetry run python benchmarks/science-validation/exp3_determinism.py # §3
poetry run python benchmarks/science-validation/exp4_calibration.py # §4
```

Seeds: `SEEDS` in `benchmarks/science-validation/_lib.py` — exp1 global seeds
[42, 1042, 2042, 3042, 4042]; exp2 policy seed 42, replicate seeds
7000+13i, i=0..19; exp3 generator seed 20260707, request seed 42; exp4 request
seeds 42+1000i, i=0..4. Results JSON files embed the generating command, git
SHA, Python version and UTC timestamp under `provenance`.
