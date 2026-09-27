# R3-B evaluation: does a monthly dynamic simulation materially improve Olumi's science on the real PoC journeys?

**Status:** offline evaluation prototype on branch `r3b/sim-prototype`. It is never merged and never deployed, and no production file changed.

**Inputs:**
- the 12-graph corpus, copied verbatim from `olumi-programme-docs@58ca52c` (sha256 values in `corpus/MANIFEST.json`);
- the hand-audited mapping (`MAPPING.md`, `mapping/*.json`);
- `ANALYSIS-PLAN.md`, frozen with the mapping in `FROZEN.sha256` before any simulation ran.

**Reproduce:** from `experiments/r3b_sim`, run `poetry run python -m sim.run` and then `poetry run python -m sim.report`. The MVP-B break-even is `poetry run python -m sim.breakeven` (writes `breakeven.json`). `results.json` is byte-identical across runs (seed 20260927); timings go to `runtime.json`.

## Answer
**No, not from today's model representation.** Across all 12 graphs, the strict tiers (T0/T1) admit **no** time path:
- deadline verdicts: 0;
- stock accumulations: 0;
- whole-decision rankings: 0;
- B-unique strong results (the pre-specified decision rule): **0**.

What the strict tiers can say (identity values in user units and static limit verdicts) is exactly what the same-engine H=0 control also says. So it is R3-A territory, not B.

**B's capability is real, but it appears only once assumptions are added:**
- a monthly time path changes goal verdicts relative to month 0 in 2 graphs;
- in 3 of the 4 graphs where Mode X makes the brief's own choice computable, the deadline verdict or the winner **flips** with semantics the model does not carry: net-vs-gross flows, the size of the churn response, the competitive response, and whether an uplift compounds;
- in the 4th graph, no option reaches the goal under any reading.

**The corpus does not contain enough temporal information to run B honestly.** Per the brief, that is the evaluation result.

## Corpus facts that shape the evaluation (measured)
- **`analysis_result` is `null` in all 12 corpus files.** The "current served engine" column therefore cannot be filled per graph. It is marked NOT IN CORPUS, and the current engine is described from code (ISL `3717e36`) and from served evidence already on #70.
- **The oracle graph for `+£15,102.04` (`a6ed1bff`) is not in the corpus.**
  - Its anchored-level contract is not reproduced, and the prototype is not tuned to it: this is a **semantic-model difference**.
  - Closed-form hand checks on corpus graphs replace it (see Task 5).
- **No goal carries a typed horizon.** Horizons exist only in brief text:
  - A: "within 12 months";
  - C: "within 6 months";
  - E: "by Q3", which gives no year or fiscal basis on 27 Sep 2026 → `HORIZON_AMBIGUOUS`.
- **Held levels:**
  - 44 of 67 are Olumi estimates (`cee_inference`); only 23 are brief- or user-stated.
  - Of 142 behavioural edges, 101 carry only a range-normalised strength with no user-unit `natural_effect`.
  - All 31 `natural_effect`s have a template spread (sd/mean exactly 0.25 or 0.5).
- **The two briefs conflict.** The original `R3B-CLOUD-BRIEF.md` says goals become stocks and edge mean/std are the priors. The later R3-B/Cloud-2 brief and the approved plan say neither is automatic, and they govern this evaluation.

## Method (see `ANALYSIS-PLAN.md`)
**Evidence tiers:**

| Tier | What it admits | How it counts |
|---|---|---|
| T0 | Brief/user facts only | Strict evidence |
| T1 | Adds held Olumi estimates, tagged `estimate_only` | Strict evidence |
| T2 | Adds the named defaults: level/rate persistence, onset at month 0, no lag, linear scaling, template uncertainty | Sensitivity evidence only |
| X | Adds named `prototype_assumption`s: label-derived sums, identity inversion, net/gross readings, served-frame readings of coefficients, spread sweeps | Shows potential capability only |

**Rules:**
- **Taint:** a link with no user-unit quantification whose source moves under an option withholds its target. It is never counted as a zero effect. Only conditional statements survive, e.g. "+£12,000 at unchanged subscribers".
- **Time:** `S_t` is the stock at the start of month t. Flows act on `S_t`; outputs are computed on `S_{t+1}`. Parameters are drawn once per trajectory.
- **EVPPI:** ISL's own served estimator, loaded read-only.
  - Validated first: analytic value 0.398942 vs 0.393806 estimated, and 0.197797 vs 0.198059; the zero control is below resolution. It is therefore reported as EVPPI.
  - A parameter counts as "strict resolved" only if it is resolved at 3 seeds and at 50,000 draws.

## Aggregate results

| | T0 (strict) | T1 (strict) | T2 (defaults) | X (exploratory) |
|---|---|---|---|---|
| Graphs with any computable claim | 5 / 12 | 12 / 12 | 12 / 12 | — |
| Graphs with a deadline goal result | 0 | 0 | **1** (A-183807Z, keep-current only) | 4 |
| Graphs with time accumulation | 0 | 0 | 1 | 3 (plus 1 static read of month-12 nodes) |
| Whole-decision ranking available | 0 | 0 | 0 | 0 |
| Brief decision pair comparable on the goal | 0 | 0 | 0 | 4 |
| Graphs with ≥1 resolved EVPPI: ISL status / strict | 0 / 0 | 0 / 0 | 0 / 0 | 2 / **0** |
| B-unique strong results (pre-specified rule) | 0 | 0 | n/a | n/a |

**Why whole-decision ranking is never available:** 5 of 12 graphs carry user-added options with `interventions: {}`, and in every A graph the price options are tainted by the unquantified price → churn link.

### The original brief's comparison table (aggregated)

| Measure | Current engine (served `analysis_result`) | R3-B prototype |
|---|---|---|
| % of runs with ≥1 above-resolution finding | NOT IN CORPUS for these 12 runs. Programme-wide: 13% of 114 served runs (AIC 5858555342) | Strict: 0% (not computable: no admissible uncertainty). T2: 0%. X: 17% on ISL status, **0% strict** (each "resolved" fails seed/draw stability) |
| Option separation (gap in P(goal), best vs second) | NOT IN CORPUS | Strict: not computable (0/12). X brief pair, point runs: gap of 0 or 1 depending on the assumption (see "What time adds") |
| Plausibility vs the hand-check | NOT IN CORPUS | Every identity value equals its closed form: 49×1,200 = £58,800; +£12,000 at unchanged subscribers; 2×£120k; 4×£65k. The `+£15,102.04` oracle graph is absent (semantic-model difference, not tested) |
| Deadline handled (yes/no) | No (no horizon in the engine; CODE) | Strict: no (withheld 12/12). T2: 1/12. X: 4/12 (8/12 still withheld) |
| Gaps the graph could not supply | Not reported: the engine defaults them | See the GAP frequency table and `MAPPING.md` |
| Runtime per analysis | NOT IN CORPUS (served Run 6.7–8.0 s end to end, ledger X5) | Strict T0+T1: ≈2–8 ms per graph. T2: ≈4–10 ms. X: 0.2–4.9 s per graph (every scenario × point + 2,000-draw MC + strict EVPPI reruns). Full corpus ≈14 s |

### GAP classes (graphs affected, of 12)

| GAP | Graphs | GAP | Graphs |
|---|---|---|---|
| REQUIRES_T1 (only Olumi-estimated inputs) | 12 | IDENTITY_INCOMPLETE | 2 |
| STATIC_COEFFICIENT_NO_TEMPORAL_MEANING | 8 | STOCK_LEVEL_MISSING | 2 |
| OPTION_LEVELS_MISSING | 5 | FRAME_MISSING | 2 |
| UNQUANTIFIED_NODE | 5 | ONSET_UNSPECIFIED | 2 |
| HORIZON_AMBIGUOUS | 4 | BASELINE_MISSING | 1 |
| GOAL_NOT_QUANTIFIED (E: no work-to-completion) | 4 | FLOW_NET_VS_GROSS_AMBIGUOUS | 1 |
| SUM_CARRIER_ABSENT | 4 | IDENTITY_INCONSISTENT_WITH_HELD_LEVEL | 1 |
| REQUIRES_T2 (needs a named default) | 4 | GOAL_TEMPORAL_SEMANTICS_AMBIGUOUS / HORIZON_BAKED_INTO_NODE | 1 / 1 |
| INFLOW_MISSING | 3 | INTERNAL_INCONSISTENCY / RATE_PERSISTENCE_UNSPECIFIED / OPTION_TARGET_INVALID / UNIT_UNVERIFIABLE / EFFECT_TIMING_UNSPECIFIED | 1 each |

## What time adds (the H=0 control, within the B representation; NOT B vs R3-A)
Each dynamic model is also read at month 0. **Time changes the goal verdict only in Mode X:**

- **A-183807Z** (`X_uplift_grows`): £59 moves from **£84,000 at month 0 to £100,432 at month 12**, and the goal is met.
  - Under `X_uplift_flat`, where the uplift does not compound, it reaches only £98,671 and the goal is not met.
  - The graph cannot tell the two apart. So the brief's own question ("does £59 get us to £100k in 12 months?") depends on a semantic choice the model does not make.
  - At T2, only keep-current is computable: £75,000 × 1.015¹² = **£89,671**, goal not met at months 0 and 12 alike.
- **A-180910Z** (`X_net_reading` / `X_gross_reading`):
  - Keep-current reaches **£98,520** (net additions persist) or **£95,002** (gross additions persist). Both miss £100k, so the verdict holds either way.
  - £59 + feature ranges from £122,298 (no churn response) down to £85,553 (the served-frame reading of the 0.85 coefficient, +4.25 pp churn), and to £54,303 when the served-frame competitive response is added.
  - With the served-frame competitive response (−£31,250), both feature options fall below keep-current.
  - The brief pair's winner flips between keep-current and £59 across 12 point scenarios.
- **A-181846Z** (static at month 12; the horizon is baked into nodes): keep-current is 49 × 1,360 + £26,000 = **£92,640**.
  - £59 gives £105,502, £101,550, £102,368 or £81,615, depending on two unsupported readings: the churn → subscriber effect (−25 or −159 per pp) and the price-sensitivity chain (0 or +2.125 pp).
  - The winner flips to keep-current in 1 of 4 scenarios.
- **C-183807Z** (subscribers by identity inversion, 72,000 / 49 = 1,469.4):
  - The model's own flows (20/month in, 3%/month out) imply **decline** to £65,430 by month 6.
  - No option reaches £100k (feature + £59 is the best, at £79,742–£84,454).
  - This is a coherence check a static engine cannot perform, but it rests on an inverted, non-integer starting stock.

Every one of these results carries 5–9 named defaults or assumptions (`n_defaults_and_assumptions` in `results.json`). That count is the cost.

## Task 5: the numerical oracle, used properly
- **Same semantics, closed forms reproduced exactly (tests):**
  - month-0 Pro MRR 49 × 1,200 = £58,800;
  - £59 at unchanged subscribers: +£12,000 (reported **conditional** on unchanged subscribers, because the price → churn link is unquantified);
  - A-181846Z: +£13,600 at unchanged month-12 subscribers;
  - salaries: 2 × £120k and 4 × £65k;
  - growth: 75,000 × 1.015¹²;
  - net reading: 1,200 + 12 × 40;
  - churn decay: S\* + (S₀ − S\*)(1 − c)ᵗ at H = 1 and H = 12.
- **R3-8 mutant:** a product computed on normalised operands (0.245 × 0.24) misses £58,800. The test goes RED.
- **`+£15,102.04`:** a different semantic contract (anchored levels on `a6ed1bff`) on a graph not in the corpus. It is not reproduced and not compared. The prototype was never tuned towards it.

## Scientific checks
- **Held values:** in every graph, the status quo reproduces every admissible held level (parametrised test over the 12 graphs).
- **Time coherence:**
  - exact closed forms at H = 1 and H = 12;
  - zero flows keep a stock constant;
  - parameters stay constant within a trajectory;
  - `results.json` is byte-identical across runs.
- **Withholding:**
  - T0/T1 never emit a deadline;
  - `prototype_assumption` inputs and derived sums are refused below Mode X;
  - removing the `linear_scaling` default withholds the dependent churn verdict;
  - unquantified links taint instead of counting as zero;
  - the engine itself refuses C-182848Z's declared identity (49 × 1,000 ≠ £72,000);
  - C-181846Z's missing baseline is never invented, in any mode.
- **Direction:**
  - a higher price raises Pro MRR at unchanged subscribers;
  - a larger churn response lowers the deadline outcome;
  - higher acquisition raises the stock.
- **Monte Carlo convergence** (A-180910Z `X_net_reading`, +0.5 pp churn scenario):

  | Option | P(goal by month 12) at 2k / 10k / 50k draws |
  |---|---|
  | £49 + feature | 0.684 / 0.6883 / 0.6878 |
  | £59 + feature | 1.000 / 0.9995 / 0.99962 |

  Every EVPPI is below resolution at all three draw counts.
- **EVPPI finding:** ISL's K=16 max-permutation floor collapses to **0** when the permuted fits never cross.
  - A tail artefact then reads as "resolved". In A-180910Z X, EVPPI(feature → perception) is £38.27 at seed 20260927, then £0.79, £0 and £0.18 at other seeds / 50k draws, on a £98k outcome.
  - Under the strict rule, 0 of 12 graphs have a stable resolved EVPPI at any tier.
  - Where the decision is clear-cut, EVPPI is legitimately 0 (e.g. A-183807Z: £59 beats keep-current in every draw). That is a valid result.
- **Spread sweep** (sd/mean ∈ {0.1, 0.25, 0.5, 1}, existence ∈ {1, 0.8}):
  - P(goal) for £49 + feature moves from 1.0 to 0.36–0.51, driven purely by template spreads.
  - The only "resolved" EVPPIs appear at sd/mean = 1.

## Cost of production B (what Olumi must establish to make these dynamics defensible)

| Required semantic | Graphs lacking it | Contract / repo change | Elicitation burden (who; can a typical user know it; extra turns) |
|---|---|---|---|
| Typed horizon and deadline meaning (attain-by vs at, calendar anchor) | 12 / 12 (4 ambiguous) | Goal `horizon` field in schemas; CEE extraction; PLoT forward; ISL | User; yes, briefs state it; 0–1 turns (E: "which Q3?") |
| Node roles: stock / flow / rate, gross vs net, flow → stock target | ≥6 / 12 (net/gross 1, inflow missing 3, stock level missing 2) | New typed role fields in schemas; CEE construction; ISL | Model generator; partly user (billing data); 1–2 turns |
| Start stocks as user facts | 44 of 67 held levels are Olumi estimates; 2 stocks and 1 baseline absent | CEE elicitation and provenance (exists) | User; usually yes (subscriber count, MRR); 1–2 turns |
| Rate persistence or trend over the horizon | Every T2 result depends on it | New per-rate trend field; ISL | User rarely knows; Olumi estimate with genuine uncertainty; high |
| Intervention onset and lags (release date, hiring lead time, ramp-up) | A "next release" 4/4; E lead time/ramp-up unquantified 4/4 | Option `onset` and edge `lag` fields; ISL | User; roughly (months); 1–2 turns per option |
| Behavioural effects in user units on the decisive links | 101 / 142 edges; price → churn is not fully quantified in 4/4 A graphs (A2 has a direct natural effect, but a parallel path runs through an unquantified risk node) | CEE must emit `natural_effect` (or a response family) for every option-path edge | User has limited knowledge; Olumi estimate; high |
| Genuine (not template) uncertainty | 31 / 31 natural effects are template spreads | Elicited ranges; ISL priors | High (ranges are hard to elicit) |
| `sum` carrier (e.g. MRR = Pro + non-Pro; salary = senior + junior) | 4 / 12 | CEE carrier widening (already ruled, AIQ 5859633012) and PLoT admission | None (definitional) |
| Work-to-completion for delivery goals | 4 / 4 E | New goal quantity (work remaining, velocity → completion) | User; weakly; high |
| Complete option levels | 5 / 12 have options with `{}` | CEE construction completeness | Model generator |

**Engineering for full B at human pace (ESTIMATE, not measured):** 8–13 engineer-weeks.
- schemas + adoption across 4 consumers: 1–2 weeks;
- CEE extraction and construction of roles, onsets, lags and effects: 3–5 weeks;
- PLoT forwarding: 0.5–1 week;
- an ISL time engine with deadline, per-limit-over-time and EVPPI stability: 2–3 weeks;
- UI time-path presentation: 1–2 weeks;
- re-baselining every exit row: 1 week or more.

The dominant cost is **information acquisition** (the burden column), not code. Every row marked "high" either becomes an Olumi estimate, and so stays T1/T2 with no strict gain, or becomes a new user question.

### MVP B (added after the RESULT post; see `AMENDMENTS.md`)
The 8–13 weeks above is full B at human pace. Agents compress the code. They do not compress:
- the missing semantics;
- the serial Delivery Lead review of HIGH-risk changes;
- served witnesses, which need LLM credit.

A much smaller B exists.

**Scope.** It is R3-A's planned slice 2 (stock-and-flow over the horizon), not a second engine. Journeys A and C only; E is out, because it has no work-to-completion quantity. It adds:
- one stock (paying subscribers), with a gross inflow per month and a churn % per month;
- the declared MRR = price × subscribers, plus the `sum` carrier;
- a typed horizon with a first-passage goal;
- every limit checked every month.

**Inputs no agent can supply:**
- **Two user facts:** paying subscribers today, and new subscribers per month before churn. These remove the net-vs-gross ambiguity and replace an Olumi estimate for the start stock.
- **Paul's call on two disclosed defaults:** rates persist at today's level, and an option starts at month 0 unless it is dated. Accepting them does **not** make the results strict: under this plan's tiers, MVP-B deadline verdicts are T2 (assumption-augmented) evidence. The decision is whether T2 verdicts, with their defaults stated, may be shown to users.
- **The horizon as a typed field.** CEE #2140 reportedly stores `goal_horizon_months` (Runtime, #70 5860324603). That is not verified here.

**Output: a threshold instead of an elasticity.** The price → churn response is the decisive unknown in the A graphs, and users rarely know it. Rather than assume one, MVP B can report the response at which each verdict flips.

The table comes from `sim/breakeven.py` → `breakeven.json`: Mode X, `prototype_assumption`, excluded from claims. It covers A-180910Z, £59 + feature release, competitive response 0, point values. Each verdict holds while the churn rise per +£10 is at most:

| Verdict for £59 + feature | Net reading | Gross reading |
|---|---|---|
| Reaches £100k MRR by month 12 | +2.35 pp | +2.09 pp |
| Ends month 12 above keep-current | +2.53 pp | +2.79 pp |
| Monthly churn stays ≤ 4% (user limit) | +1.5 pp | +1.5 pp |

- **The churn limit binds first.** The +1.5 pp is:
  - held churn of 3% (an Olumi estimate);
  - minus 0.5 pp from the feature release (perception 50 → 75, a CEE hypothesis, × Olumi's −0.2 pp per +10);
  - checked against the user's 4% limit.

  So it rests on T1 estimates.
- **The unknown response matters far more than the ambiguity.** Net vs gross moves the thresholds by ≤0.3 pp. The unknown response spans 0–4.25 pp in the mapping sweep.
- **With the X competitive response (−£31,250), £59 fails both MRR verdicts even at zero churn response.** It misses the goal by £8,952 (net) or £14,261 (gross), and it trails keep-current. A threshold on churn alone does not settle the decision.

**Revised ESTIMATE (not measured).**
- **Code:** ~1–2 agent-days across file-disjoint lanes:
  - ISL: port `sim/dynamic.py`'s single-stock integration, first passage, per-month limits and the threshold search, using this folder's tests as RED-first rows;
  - CEE: a CEE-local role field and the two questions;
  - PLoT: forwarding;
  - UI: one sentence in the existing result, with no time-path chart.
- **Elapsed:** ~3–5 days, set by serial DL review, served witnesses and Paul's decision, not by code.
- The remaining full-B rows stay blocked by information, not code: lags, non-template uncertainty, work-to-completion, and effects on every decisive link.

## Per-graph evaluation

<!-- BEGIN GENERATED: per-graph tables (sim/report.py) -->

### pj-20260927T180910Z-A (journey A)

> Given our goal of reaching £100k MRR within 12 months [Currently 75k] while keeping monthly churn under 4%, should we increase the Pro plan price from £49 to £59 per month with the next Pro feature release?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (3 user/brief-stated at T0; 8 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | `pro_plan_mrr` = 49_with_feature_release: £58,800 (at unchanged pro_paying_subscribers); 59_with_feature_release: £70,800 (at unchanged pro_paying_subscribers); keep_current_pricing: £58,800 | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: resolved 12 months | T2: withheld: FLOW_NET_VS_GROSS_AMBIGUOUS, SUM_CARRIER_ABSENT |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: X_gross_reading, X_net_reading |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 0/6 option-limit cells; T1: 1/6. monthly_churn keep_current_pricing: 3 (holds, T1) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 5/6 limit cells, 3/6 identity cells; top causes: OPTION_LEVELS_MISSING x6, REQUIRES_T2 x4, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING x2 | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['keep_current_pricing', '59_with_feature_release']: X computed; X goal verdict flips across assumptions for 49_with_feature_release, 59_with_feature_release |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 6, strict 0 |
| Required semantic gaps | Not reported by the engine | FLOW_NET_VS_GROSS_AMBIGUOUS, ONSET_UNSPECIFIED, OPTION_LEVELS_MISSING, REQUIRES_T1, REQUIRES_T2, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING, SUM_CARRIER_ABSENT | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: conditional:links_exist, level_persistence, linear_scaling, no_lag, onset_month_0, rate_persistence, template_uncertainty; X assumptions: X_feature_competitive_mrr, X_price_churn_direct, X_price_churn_via_sensitivity, X_reading_gross_additions_persist, X_reading_net_additions_are_net, X_sum_mrr |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 5.5 ms (T0 + T1, all options) | T2 5.7 ms; X 5.18 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

Mode X deadline outcomes (point, links exist): X_net_reading [X_feature_competitive_mrr=0, X_price_churn_direct=0, X_price_churn_via_sensitivity=0]: 49_with_feature_release £107,338 (goal met); 59_with_feature_release £122,298 (goal met); keep_current_pricing £98,520 (goal not met)<br>X_net_reading [X_feature_competitive_mrr=-31250, X_price_churn_direct=0, X_price_churn_via_sensitivity=0]: 49_with_feature_release £76,088 (goal not met); 59_with_feature_release £91,048 (goal not met); keep_current_pricing £98,520 (goal not met)<br>X_net_reading [X_feature_competitive_mrr=0, X_price_churn_direct=0.5, X_price_churn_via_sensitivity=0]: 49_with_feature_release £107,338 (goal met); 59_with_feature_release £117,090 (goal met); keep_current_pricing £98,520 (goal not met)<br>X_net_reading [X_feature_competitive_mrr=-31250, X_price_churn_direct=0.5, X_price_churn_via_sensitivity=0]: 49_with_feature_release £76,088 (goal not met); 59_with_feature_release £85,840 (goal not met); keep_current_pricing £98,520 (goal not met)<br>X_net_reading [X_feature_competitive_mrr=0, X_price_churn_direct=4.25, X_price_churn_via_sensitivity=0]: 49_with_feature_release £107,338 (goal met); 59_with_feature_release £85,553 (goal not met); keep_current_pricing £98,520 (goal not met)<br>X_net_reading [X_feature_competitive_mrr=-31250, X_price_churn_direct=4.25, X_price_churn_via_sensitivity=0]: 49_with_feature_release £76,088 (goal not met); 59_with_feature_release £54,303 (goal not met); keep_current_pricing £98,520 (goal not met)<br>X_gross_reading [X_feature_competitive_mrr=0, X_price_churn_direct=0, X_price_churn_via_sensitivity=0]: 49_with_feature_release £102,473 (goal met); 59_with_feature_release £116,989 (goal met); keep_current_pricing £95,002 (goal not met)<br>X_gross_reading [X_feature_competitive_mrr=-31250, X_price_churn_direct=0, X_price_churn_via_sensitivity=0]: 49_with_feature_release £71,223 (goal not met); 59_with_feature_release £85,739 (goal not met); keep_current_pricing £95,002 (goal not met)<br>X_gross_reading [X_feature_competitive_mrr=0, X_price_churn_direct=0.5, X_price_churn_via_sensitivity=0]: 49_with_feature_release £102,473 (goal met); 59_with_feature_release £112,590 (goal met); keep_current_pricing £95,002 (goal not met)<br>X_gross_reading [X_feature_competitive_mrr=-31250, X_price_churn_direct=0.5, X_price_churn_via_sensitivity=0]: 49_with_feature_release £71,223 (goal not met); 59_with_feature_release £81,340 (goal not met); keep_current_pricing £95,002 (goal not met)<br>X_gross_reading [X_feature_competitive_mrr=0, X_price_churn_direct=4.25, X_price_churn_via_sensitivity=0]: 49_with_feature_release £102,473 (goal met); 59_with_feature_release £85,759 (goal not met); keep_current_pricing £95,002 (goal not met)<br>X_gross_reading [X_feature_competitive_mrr=-31250, X_price_churn_direct=4.25, X_price_churn_via_sensitivity=0]: 49_with_feature_release £71,223 (goal not met); 59_with_feature_release £54,509 (goal not met); keep_current_pricing £95,002 (goal not met)

### pj-20260927T180910Z-C (journey C)

> We need to reach £100k MRR within 6 months with a £20k budget, while keeping monthly churn under 4%. Should we develop new features and increase our Pro plan price from £49 to £59 per month in the next release, or invest in additional advertising?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (2 user/brief-stated at T0; 6 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | none declared | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: resolved 6 months | T2: withheld: INFLOW_MISSING, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: none |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 1/8 option-limit cells; T1: 6/8. additional_six_month_growth_spend 9ea0857f: 30000 (holds, T1); additional_six_month_growth_spend additional_advertising: 20000 (holds, T1); additional_six_month_growth_spend features_pro_price: 20000 (holds, T1); additional_six_month_growth_spend maintain_current_plan: 0 (holds, T1); monthly_churn additional_advertising: 3 (holds, T1); monthly_churn maintain_current_plan: 3 (holds, T1) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 2/8 limit cells, 0/0 identity cells; top causes: STATIC_COEFFICIENT_NO_TEMPORAL_MEANING x2 | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['features_pro_price', 'additional_advertising']: X withheld |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | INFLOW_MISSING, ONSET_UNSPECIFIED, REQUIRES_T1, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING, UNQUANTIFIED_NODE | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: none; X assumptions: none |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 2.2 ms (T0 + T1, all options) | T2 4.9 ms; X 0.00 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

### pj-20260927T180910Z-E (journey E)

> Should we hire two senior engineers or four junior engineers to ship the new platform by Q3, while keeping annual salary spend under £400k?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (0 user/brief-stated at T0; 5 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | none declared | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: HORIZON_AMBIGUOUS | T2: withheld: GOAL_NOT_QUANTIFIED, HORIZON_AMBIGUOUS |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: none |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 0/4 option-limit cells; T1: 4/4. annual_salary_spend 400cd9c9: 250000 (holds, T1); annual_salary_spend do_not_hire_now: 0 (holds, T1); annual_salary_spend hire_2_senior_engineers: 240000 (holds, T1); annual_salary_spend hire_4_junior_engineers: 260000 (holds, T1) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 0/4 limit cells, 0/0 identity cells; top causes: — | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['hire_2_senior_engineers', 'hire_4_junior_engineers']: X withheld |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | GOAL_NOT_QUANTIFIED, HORIZON_AMBIGUOUS, INTERNAL_INCONSISTENCY, REQUIRES_T1, UNQUANTIFIED_NODE | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: none; X assumptions: none |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 1.9 ms (T0 + T1, all options) | T2 3.9 ms; X 0.00 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

### pj-20260927T181846Z-A (journey A)

> Given our goal of reaching £100k MRR within 12 months [Currently 75k] while keeping monthly churn under 4%, should we increase the Pro plan price from £49 to £59 per month with the next Pro feature release?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (2 user/brief-stated at T0; 8 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | `pro_mrr_at_month_12` = 4cbb5475: £66,640 (at unchanged pro_paying_subscribers_at_month_12); a9643e27: £66,640 (at unchanged pro_paying_subscribers_at_month_12); bb667b99: £80,240 (at unchanged pro_paying_subscribers_at_month_12); keep_pro_at_49: £66,640; raise_pro_to_54_at_release: £73,440 (at unchanged pro_paying_subscribers_at_month_12); raise_pro_to_59_at_release: £80,240 (at unchanged pro_paying_subscribers_at_month_12) | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: resolved 12 months | T2: withheld: GOAL_TEMPORAL_SEMANTICS_AMBIGUOUS, STOCK_LEVEL_MISSING, SUM_CARRIER_ABSENT |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: X_static_at_month_12 |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 0/6 option-limit cells; T1: 2/6. monthly_churn_rate 4cbb5475: 3 (holds, T1); monthly_churn_rate keep_pro_at_49: 3 (holds, T1) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 4/6 limit cells, 0/6 identity cells; top causes: STATIC_COEFFICIENT_NO_TEMPORAL_MEANING x6, FRAME_MISSING x5, REQUIRES_T2 x2 | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['keep_pro_at_49', 'raise_pro_to_59_at_release']: X computed; X goal verdict flips across assumptions for raise_pro_to_59_at_release |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | FRAME_MISSING, GOAL_TEMPORAL_SEMANTICS_AMBIGUOUS, HORIZON_BAKED_INTO_NODE, REQUIRES_T1, REQUIRES_T2, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING, STOCK_LEVEL_MISSING, SUM_CARRIER_ABSENT | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: conditional:links_exist, linear_scaling, no_lag, onset_month_0, template_uncertainty; X assumptions: X_churn_to_subscribers_m12, X_month12_nodes_are_forecast_at_H, X_price_churn_via_sensitivity, X_sum_mrr_month12 |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 3.8 ms (T0 + T1, all options) | T2 5.3 ms; X 0.19 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

Mode X deadline outcomes (point, links exist): X_static_at_month_12 [X_churn_to_subscribers_m12=-25, X_price_churn_via_sensitivity=0]: keep_pro_at_49 £92,640 (goal not met); raise_pro_to_54_at_release £99,102 (goal not met); raise_pro_to_59_at_release £105,502 (goal met)<br>X_static_at_month_12 [X_churn_to_subscribers_m12=-159, X_price_churn_via_sensitivity=0]: keep_pro_at_49 £92,640 (goal not met); raise_pro_to_54_at_release £97,294 (goal not met); raise_pro_to_59_at_release £101,550 (goal met)<br>X_static_at_month_12 [X_churn_to_subscribers_m12=-25, X_price_churn_via_sensitivity=2.125]: keep_pro_at_49 £92,640 (goal not met); raise_pro_to_54_at_release £97,668 (goal not met); raise_pro_to_59_at_release £102,368 (goal met)<br>X_static_at_month_12 [X_churn_to_subscribers_m12=-159, X_price_churn_via_sensitivity=2.125]: keep_pro_at_49 £92,640 (goal not met); raise_pro_to_54_at_release £88,171 (goal not met); raise_pro_to_59_at_release £81,615 (goal not met)

### pj-20260927T181846Z-C (journey C)

> We need to reach £100k MRR within 6 months with a £20k budget, while keeping monthly churn under 4%. Should we develop new features and increase our Pro plan price from £49 to £59 per month in the next release, or invest in additional advertising?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (2 user/brief-stated at T0; 5 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | none declared | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: resolved 6 months | T2: withheld: BASELINE_MISSING |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: none |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 3/8 option-limit cells; T1: 8/8. incremental_6_month_spend additional_advertising: 20000 (holds, T1); incremental_6_month_spend b77d5881: 30000 (holds, T1); incremental_6_month_spend carry_on_as_now: 0 (holds, T1); incremental_6_month_spend features_59_pro_price: 20000 (holds, T1); monthly_churn additional_advertising: 3 (holds, T0); monthly_churn b77d5881: 3 (holds, T0) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 0/8 limit cells, 0/0 identity cells; top causes: — | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['features_59_pro_price', 'additional_advertising']: X withheld |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | BASELINE_MISSING, REQUIRES_T1, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING, SUM_CARRIER_ABSENT | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: none; X assumptions: none |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 1.9 ms (T0 + T1, all options) | T2 5.9 ms; X 0.00 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

### pj-20260927T181846Z-E (journey E)

> Should we hire two senior engineers or four junior engineers to ship the new platform by Q3, while keeping annual salary spend under £400k?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (1 user/brief-stated at T0; 3 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | none declared | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: HORIZON_AMBIGUOUS | T2: withheld: GOAL_NOT_QUANTIFIED, HORIZON_AMBIGUOUS |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: none |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 1/4 option-limit cells; T1: 4/4. annual_salary_spend c769839e: 250000 (holds, T1); annual_salary_spend hire_four_junior_engineers: 260000 (holds, T1); annual_salary_spend hire_two_senior_engineers: 240000 (holds, T1); annual_salary_spend no_new_hires: 0 (holds, T0) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 0/4 limit cells, 0/0 identity cells; top causes: — | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['hire_two_senior_engineers', 'hire_four_junior_engineers']: X withheld |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | GOAL_NOT_QUANTIFIED, HORIZON_AMBIGUOUS, REQUIRES_T1, UNQUANTIFIED_NODE | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: none; X assumptions: none |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 1.7 ms (T0 + T1, all options) | T2 3.1 ms; X 0.00 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

### pj-20260927T182848Z-A (journey A)

> Given our goal of reaching £100k MRR within 12 months [Currently 75k] while keeping monthly churn under 4%, should we increase the Pro plan price from £49 to £59 per month with the next Pro feature release?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (2 user/brief-stated at T0; 7 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | `pro_plan_mrr` withheld (FRAME_MISSING, OPTION_LEVELS_MISSING) | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: resolved 12 months | T2: withheld: FRAME_MISSING, IDENTITY_INCOMPLETE, INFLOW_MISSING |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: none |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 0/7 option-limit cells; T1: 1/7. monthly_pro_churn carry_on_as_now: 3 (holds, T1) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 6/7 limit cells, 7/7 identity cells; top causes: FRAME_MISSING x5, OPTION_LEVELS_MISSING x4, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING x3, REQUIRES_T2 x1 | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['carry_on_as_now', '59_with_feature_release']: X withheld |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | FRAME_MISSING, IDENTITY_INCOMPLETE, INFLOW_MISSING, OPTION_LEVELS_MISSING, REQUIRES_T1, REQUIRES_T2, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: none; X assumptions: none |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 3.8 ms (T0 + T1, all options) | T2 6.0 ms; X 0.00 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

### pj-20260927T182848Z-C (journey C)

> We need to reach £100k MRR within 6 months with a £20k budget, while keeping monthly churn under 4%. Should we develop new features and increase our Pro plan price from £49 to £59 per month in the next release, or invest in additional advertising?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (4 user/brief-stated at T0; 7 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | `mrr` withheld (IDENTITY_INCONSISTENT_WITH_HELD_LEVEL) | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: resolved 6 months | T2: withheld: IDENTITY_INCONSISTENT_WITH_HELD_LEVEL, INFLOW_MISSING |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: none |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 2/8 option-limit cells; T1: 6/8. incremental_launch_budget advertising_investment: 20000 (holds, T1); incremental_launch_budget continue_as_now: 0 (holds, T1); incremental_launch_budget dffdec3f: 30000 (holds, T1); incremental_launch_budget features_59_price: 20000 (holds, T1); monthly_churn advertising_investment: 3 (holds, T0); monthly_churn continue_as_now: 3 (holds, T0) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 2/8 limit cells, 4/4 identity cells; top causes: IDENTITY_INCONSISTENT_WITH_HELD_LEVEL x4, REQUIRES_T2 x1, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING x1 | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['features_59_price', 'advertising_investment']: X withheld |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | EFFECT_TIMING_UNSPECIFIED, IDENTITY_INCONSISTENT_WITH_HELD_LEVEL, INFLOW_MISSING, REQUIRES_T1, REQUIRES_T2, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: none; X assumptions: none |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 2.1 ms (T0 + T1, all options) | T2 5.7 ms; X 0.00 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

### pj-20260927T182848Z-E (journey E)

> Should we hire two senior engineers or four junior engineers to ship the new platform by Q3, while keeping annual salary spend under £400k?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (2 user/brief-stated at T0; 4 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | `senior_annual_salary_spend` = continue_with_current_team: £0; hire_2_senior_engineers: £240,000; hire_4_junior_engineers: £0<br>`junior_annual_salary_spend` = continue_with_current_team: £0; hire_2_senior_engineers: £0; hire_4_junior_engineers: £260,000 | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: HORIZON_AMBIGUOUS | T2: withheld: GOAL_NOT_QUANTIFIED, HORIZON_AMBIGUOUS |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: none |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 0/4 option-limit cells; T1: 0/4.  | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 4/4 limit cells, 2/8 identity cells; top causes: OPTION_LEVELS_MISSING x3, UNIT_UNVERIFIABLE x3 | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['hire_2_senior_engineers', 'hire_4_junior_engineers']: X withheld |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | GOAL_NOT_QUANTIFIED, HORIZON_AMBIGUOUS, OPTION_LEVELS_MISSING, REQUIRES_T1, SUM_CARRIER_ABSENT, UNIT_UNVERIFIABLE, UNQUANTIFIED_NODE | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: none; X assumptions: none |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 2.7 ms (T0 + T1, all options) | T2 2.7 ms; X 0.00 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

### pj-20260927T183807Z-A (journey A)

> Given our goal of reaching £100k MRR within 12 months [Currently 75k] while keeping monthly churn under 4%, should we increase the Pro plan price from £49 to £59 per month with the next Pro feature release?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (2 user/brief-stated at T0; 5 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | `pro_mrr` = keep_pro_at_49: £44,100; raise_pro_to_54: £48,600 (at unchanged pro_paying_subscribers); raise_pro_to_59: £53,100 (at unchanged pro_paying_subscribers) | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: resolved 12 months | T2: T2_growth [no X effects]: keep_pro_at_49 £89,671 (goal not met) |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: yes; X models: X_uplift_flat, X_uplift_grows |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 0/6 option-limit cells; T1: 1/6. monthly_churn keep_pro_at_49: 3.2 (holds, T1) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 5/6 limit cells, 3/6 identity cells; top causes: OPTION_LEVELS_MISSING x6, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING x4 | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['keep_pro_at_49', 'raise_pro_to_59']: X computed; X goal verdict flips across assumptions for raise_pro_to_59 |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | IDENTITY_INCOMPLETE, OPTION_LEVELS_MISSING, RATE_PERSISTENCE_UNSPECIFIED, REQUIRES_T1, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: no_lag, onset_month_0, rate_persistence; X defaults: conditional:links_exist, linear_scaling, no_lag, onset_month_0, rate_persistence, template_uncertainty; X assumptions: X_growth_applies_to_uplift, X_growth_excludes_uplift, X_price_churn_direct, X_price_churn_via_sensitivity, X_pro_mrr_in_mrr |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 2.2 ms (T0 + T1, all options) | T2 8.3 ms; X 0.24 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

Mode X deadline outcomes (point, links exist): X_uplift_grows [X_price_churn_direct=0, X_price_churn_via_sensitivity=0, X_pro_mrr_in_mrr=1]: keep_pro_at_49 £89,671 (goal not met); raise_pro_to_54 £95,052 (goal not met); raise_pro_to_59 £100,432 (goal met)<br>X_uplift_grows [X_price_churn_direct=0.5, X_price_churn_via_sensitivity=0, X_pro_mrr_in_mrr=1]: keep_pro_at_49 £89,671 (goal not met); raise_pro_to_54 £94,906 (goal not met); raise_pro_to_59 £100,114 (goal met)<br>X_uplift_grows [X_price_churn_direct=4.25, X_price_churn_via_sensitivity=0, X_pro_mrr_in_mrr=1]: keep_pro_at_49 £89,671 (goal not met); raise_pro_to_54 £93,817 (goal not met); raise_pro_to_59 £97,734 (goal not met)<br>X_uplift_flat [X_price_churn_direct=0, X_price_churn_via_sensitivity=0, X_pro_mrr_in_mrr=1]: keep_pro_at_49 £89,671 (goal not met); raise_pro_to_54 £94,171 (goal not met); raise_pro_to_59 £98,671 (goal not met)<br>X_uplift_flat [X_price_churn_direct=0.5, X_price_churn_via_sensitivity=0, X_pro_mrr_in_mrr=1]: keep_pro_at_49 £89,671 (goal not met); raise_pro_to_54 £94,050 (goal not met); raise_pro_to_59 £98,406 (goal not met)<br>X_uplift_flat [X_price_churn_direct=4.25, X_price_churn_via_sensitivity=0, X_pro_mrr_in_mrr=1]: keep_pro_at_49 £89,671 (goal not met); raise_pro_to_54 £93,139 (goal not met); raise_pro_to_59 £96,415 (goal not met)

### pj-20260927T183807Z-C (journey C)

> We need to reach £100k MRR within 6 months with a £20k budget, while keeping monthly churn under 4%. Should we develop new features and increase our Pro plan price from £49 to £59 per month in the next release, or invest in additional advertising?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (2 user/brief-stated at T0; 6 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | `mrr` withheld (OPTION_LEVELS_MISSING, STOCK_LEVEL_MISSING) | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: resolved 6 months | T2: withheld: STOCK_LEVEL_MISSING |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: X_inversion |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 0/10 option-limit cells; T1: 7/10. six_month_incremental_spend advertising_investment: 20000 (holds, T1); six_month_incremental_spend continue_as_now: 0 (holds, T1); six_month_incremental_spend feature_59_price: 20000 (holds, T1); six_month_incremental_spend features_hold_49_price: 20000 (holds, T1); pro_monthly_churn advertising_investment: 3 (holds, T1); pro_monthly_churn continue_as_now: 3 (holds, T1) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 3/10 limit cells, 5/5 identity cells; top causes: STOCK_LEVEL_MISSING x4, OPTION_LEVELS_MISSING x3, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING x1 | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['feature_59_price', 'advertising_investment']: X computed |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 1, strict 0 |
| Required semantic gaps | Not reported by the engine | OPTION_LEVELS_MISSING, OPTION_TARGET_INVALID, REQUIRES_T1, STATIC_COEFFICIENT_NO_TEMPORAL_MEANING, STOCK_LEVEL_MISSING | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: conditional:links_exist, level_persistence, no_lag, onset_month_0, rate_persistence, template_uncertainty; X assumptions: X_price_churn_via_sensitivity, X_spend_total_to_acquisition, X_subscribers_by_identity_inversion |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 3.0 ms (T0 + T1, all options) | T2 6.5 ms; X 0.82 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

Mode X deadline outcomes (point, links exist): X_inversion [X_price_churn_via_sensitivity=0, X_spend_total_to_acquisition=0]: advertising_investment £73,615 (goal not met); continue_as_now £65,430 (goal not met); feature_59_price £84,454 (goal not met); features_hold_49_price £70,140 (goal not met)<br>X_inversion [X_price_churn_via_sensitivity=0.02, X_spend_total_to_acquisition=0]: advertising_investment £73,615 (goal not met); continue_as_now £65,430 (goal not met); feature_59_price £84,357 (goal not met); features_hold_49_price £70,140 (goal not met)<br>X_inversion [X_price_churn_via_sensitivity=0.4, X_spend_total_to_acquisition=0]: advertising_investment £73,615 (goal not met); continue_as_now £65,430 (goal not met); feature_59_price £82,541 (goal not met); features_hold_49_price £70,140 (goal not met)<br>X_inversion [X_price_churn_via_sensitivity=1, X_spend_total_to_acquisition=0]: advertising_investment £73,615 (goal not met); continue_as_now £65,430 (goal not met); feature_59_price £79,742 (goal not met); features_hold_49_price £70,140 (goal not met)

### pj-20260927T183807Z-E (journey E)

> Should we hire two senior engineers or four junior engineers to ship the new platform by Q3, while keeping annual salary spend under £400k?

| Measure | Current served engine | R3-B strict (T0/T1) | Notes (T2 assumption-augmented / X exploratory) |
|---|---|---|---|
| Baseline reproduced correctly | NOT IN CORPUS (`analysis_result` null). Served evidence elsewhere: keep-current £14,870 vs held £75,000 on `a6ed1bff` (B1a RED, AIQ 5859598577) | Yes: status quo equals every admissible held level (1 user/brief-stated at T0; 3 incl. Olumi estimates at T1); tested | Held levels read through the R3-8 frame reader |
| Exact identities represented | No: `nonlinear_identity` is dropped by the request model (`extra="ignore"`); propagation is additive in normalised space (CODE, ISL `3717e36`) | none declared | Declared identities only; label-derived sums are Mode X |
| Deadline represented | No horizon concept in the engine (CODE) | Withheld at T0/T1 (no admissible time path). Horizon: HORIZON_AMBIGUOUS | T2: withheld: GOAL_NOT_QUANTIFIED, HORIZON_AMBIGUOUS |
| Time accumulation represented | No (CODE) | No (needs persistence and onset defaults) | T2 time accumulation: no; X models: none |
| Supported constraints evaluated | NOT IN CORPUS; limits are scored per draw on normalised levels (CODE) | T0: 1/5 option-limit cells; T1: 5/5. annual_salary_spend carry_on_as_now: 0 (holds, T0); annual_salary_spend f90a16bd: 250000 (holds, T1); annual_salary_spend four_junior_engineers: 260000 (holds, T1); annual_salary_spend one_senior_engineer: 120000 (holds, T1); annual_salary_spend two_senior_engineers: 240000 (holds, T1) | Static (timeless) verdicts: the same the H=0 control gives |
| Unsupported claims withheld | NOT IN CORPUS; missing semantics are defaulted, not withheld (CODE) | T1 withheld: 0/5 limit cells, 0/0 identity cells; top causes: — | Taint: an unquantified link whose source moves withholds its target |
| Decision-relevant option separation | NOT IN CORPUS | Not computable at T0/T1 (goal withheld) | Whole ranking: T2 withheld, X withheld; brief pair ['two_senior_engineers', 'four_junior_engineers']: X withheld |
| EVPPI / information-value result | NOT IN CORPUS (programme-wide: 87% of 114 served runs had no above-resolution finding, AIC 5858555342) | NOT_COMPUTABLE (UNCERTAINTY_NOT_SPECIFIED: no admissible uncertainty at T0/T1) | T2 resolved: ISL-status 0, strict 0; X resolved: ISL-status 0, strict 0 |
| Required semantic gaps | Not reported by the engine | GOAL_NOT_QUANTIFIED, HORIZON_AMBIGUOUS, REQUIRES_T1, UNQUANTIFIED_NODE | From the frozen mapping plus engine withholding |
| Prototype-only assumptions | N/A | None (strict) | T2 goal-model defaults: none (no T2 goal model); X defaults: none; X assumptions: none |
| Runtime | NOT IN CORPUS (served Run 6.7-8.0 s end to end, ledger X5) | 2.5 ms (T0 + T1, all options) | T2 4.3 ms; X 0.00 s (all scenarios, point + 2,000-draw MC, strict EVPPI reruns) |

<!-- END GENERATED -->

## Decision evidence (for Paul, via the Delivery Lead)
1. **What B genuinely adds.** A monthly time path. It can turn a month-0 miss into a month-12 hit (A-183807Z £59: £84,000 → £100,432 if the uplift compounds). It can also expose flows that contradict held levels (C-183807Z's own figures imply MRR falls to £65,430). A static engine cannot state either.
2. **How much of that is usable from today's model data without invented assumptions: none.**
   - 0 of 12 graphs have a strict (T0/T1) time path.
   - Strict tiers yield only identity values and static limit verdicts, which the H=0 control reproduces.
   - T2 adds one single-option trajectory.
   - Whole-decision ranking is 0/12 at every tier.
   - In 3 of the 4 graphs where Mode X makes the brief's choice computable, the verdict flips with unsupported semantics.
3. **What production B would require.**
   - Typed horizons (12/12).
   - Stock/flow/rate roles with gross/net (≥6/12).
   - User-confirmed start stocks (44/67 levels are estimates).
   - User-unit effects on the decisive links (price → churn is not fully quantified in 4/4 A graphs).
   - Non-template uncertainty (31/31 are templates), onsets and lags, work-to-completion (4/4 E), and a `sum` carrier.
   - Full B: about 8–13 engineer-weeks at human pace (ESTIMATE). The larger cost is the elicitation burden in the table above.
   - MVP B (one stock, journeys A and C, = R3-A slice 2): ~1–2 agent-days of code and ~3–5 days elapsed (ESTIMATE). It needs 2 user facts and Paul's call on showing T2 verdicts with 2 disclosed defaults. It can report the churn response at which a verdict flips instead of assuming one (see "MVP B").
4. **Cheap wins for R3-A.**
   - The taint/withhold rule, with "at unchanged X" conditional statements.
   - The identity-versus-held-level consistency check.
   - A typed horizon with static-at-H evaluation where the model already carries month-H nodes.
   - A single-stock closed form (level × (1 + g)^H) only where a typed %/month rate exists.
   - An EVPPI seed/draw-stability gate: the served K=16 floor can collapse to 0.
5. **What the Delivery Lead should compare with the actual served R3-A result before Paul decides.**
   - The served `analysis_result` for these same 12 runs, which is absent from the corpus.
   - Whether R3-A's served identity values match these closed forms (+£12,000 at unchanged subscribers; £58,800).
   - Whether R3-A withholds price-option MRR when price → churn has no user-unit effect, or still applies the additive coefficient.
   - R3-A's post-slice-1 no-finding rate under a seed-stability gate.
   - Whether the remaining pj-x3 FAILs are dynamics (loops, lags) or, as measured here, missing semantics (horizon, flows, effects).
