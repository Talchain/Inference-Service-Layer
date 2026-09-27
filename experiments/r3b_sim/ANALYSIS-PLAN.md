# R3-B pre-specified analysis plan

**Status:** written AFTER I had inspected the corpus, and frozen BEFORE any simulation ran. This is a pre-specified analysis plan, not a preregistration.
- `FROZEN.sha256` pins this file and every `mapping/*.json`.
- `results.json` records those hashes, and a test fails if they drift.
- Any later change to a mapping is recorded in `AMENDMENTS.md` with a date and a reason, and the results are shown both before and after the change.

## Question
Does today's Olumi model contain enough temporal meaning for a monthly dynamic simulation to improve decisions honestly? This prototype is an evaluation, not a replacement engine.

## Evidence tiers
| Tier | What it admits | How it counts |
|---|---|---|
| T0 (strict) | Levels stated by the brief or the user; typed option interventions whose source is `brief_extraction` or `user_specified`; horizons resolved from the brief; **declared** `nonlinear_identity` only; the `natural_effect` mean on a **user-sourced** edge, only where its units fit, the source change equals `per_source_change` exactly, and no timing is needed | Strict evidence |
| T1 (strict) | T0 plus held `cee_inference` levels, `cee_hypothesis` interventions and Olumi-estimated `natural_effect` means (same unit and exact-change conditions), all tagged `estimate_only` | Strict evidence |
| T2 (assumption-augmented) | T1 plus the named defaults `level_persistence`, `rate_persistence`, `onset_month_0`, `no_lag`, `linear_scaling` and `template_uncertainty` | Sensitivity evidence only; never strict |
| X (exploratory) | T2 plus named `prototype_assumption`s: label-derived sums, identity inversion, net-vs-gross readings, served-frame readings of coefficients that have no `natural_effect`, and spread sweeps | Shows potential capability only; excluded from claims |

**Admitted at no tier below X:**
- a range-normalised edge strength that has no `natural_effect` (`STATIC_COEFFICIENT_NO_TEMPORAL_MEANING`);
- an unquantified node;
- a label-derived sum;
- identity inversion.

**Invented at no tier, X included:** a missing baseline, a missing flow or rate, a missing horizon, or a missing option level.

## Taint
- Suppose an edge carries no admissible quantification at the tier being evaluated, and its source changes under the option (or is itself tainted). Then its target is WITHHELD for that option, and so is everything downstream of it.
- If the source is unchanged, the edge cannot move its target under any functional form, so no taint arises.
- Where an identity has a tainted operand, the only statement that survives is the conditional one ("at unchanged <operand>").

## Discrete time (for T2 and X only)
- **State:** `S_t` is the stock at the start of month t, and `S_0` is the held level. Levels `L` are constant from month 0 onwards under `onset_month_0` and `no_lag`.
- **One step:**
  1. flows `F_t = f(S_t, L)`;
  2. `S_{t+1} = S_t + inflow_t - outflow_t`;
  3. outputs `Y_{t+1} = g(S_{t+1}, L)`.
- `Y_0 = g(S_0, L)` is the month-0 identity.
- **Parameters** are drawn once per trajectory and held fixed across months. There is no per-month resampling.

## Goal semantics
- **A and C:** "reach £100k MRR within N months" is read as *attained at some month <= H* (first passage). The value at H is reported alongside it, not ranked below it, and any disagreement between the two is reported.
- **E:** the horizon is ambiguous, so deadline verdicts are withheld.
- **Nodes versus brief:** if a graph's own nodes contradict the brief's wording, the deadline verdict is withheld.

## Option comparisons
- **Pairwise:** reported only when both options are computable at the same tier.
- **Whole-decision ranking:** reported only when **every** option, including the baseline and any option with `interventions: {}`, is computable at that tier. Otherwise it is `RANKING_WITHHELD`, and the blocking options are named.
- **The brief decision pair** (fixed in each mapping) is reported pairwise wherever both options are computable.
- **Separation:** a deterministic difference at T0/T1, or a difference in P(goal) greater than 2 Monte Carlo standard errors at T2/X.

## EVPPI
- **Estimator:** ISL `src/utils/evppi.py:factor_evppi_estimate`, loaded read-only by file path, with its sha256 recorded.
- **Validation first:**
  - a positive case, U_A = θ, U_B = 0 with θ ~ N(μ, σ²), where EVPPI = σφ(μ/σ) + μΦ(μ/σ) − max(μ, 0);
  - a zero control, where an additive θ shifts both options equally.

  If either check fails, every output calls the metric an "ISL-compatible information-value proxy".
- **Status:** uses the ISL-identical K=16 floor and the 6-dp rule. A result counts as `resolved` only if it is above the floor at 3 seeds AND at 50,000 draws.
- **Units:** £ (the outcome at H) and probability points (1{goal}).
- **`NOT_COMPUTABLE`** applies when fewer than 2 options are computable for the decision in question (the whole decision, or the brief pair), or when the tier has no admissible uncertainty (T0/T1: `UNCERTAINTY_NOT_SPECIFIED`).

## Monte Carlo
- T2 and X only; T0 and T1 are deterministic.
- Seed 20260927, 2,000 draws.
- Convergence is checked at 2,000 / 10,000 / 50,000 draws on the most computable model.

## Decision rule (fixed in advance)
- **Strong B result:** a T0/T1 result that the same-engine H=0 control cannot produce, AND that changes the goal verdict or a valid ranking.
- **Potential capability:** a result of that kind found only at T2/X. It is reported together with its count of assumptions and defaults, and that count is treated as a cost.
- **Not a success criterion:** the share of analyses with at least one resolved, decision-relevant EVPPI. It is reported as a secondary diagnostic only.
- **The H=0 control** shows what adding time does within the B representation. It is NOT B versus R3-A. R3-A is compared only against its served evidence.

## Stated prior expectation (not a result)
- T0/T1 will probably reduce to static identities and static limit verdicts, which H=0 also gives.
- If so, the number of strong B results is 0, and every dynamic value appears only at T2/X.
