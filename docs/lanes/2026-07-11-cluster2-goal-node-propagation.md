# Cluster-2 — goal-node uncertainty propagation (Track S Phase 0 credibility floor)

**Lane:** CLUSTER-2 GOAL-NODE PROPAGATION · **Base:** origin/staging `209d924` ·
**Branch:** `feat/cluster2-goal-node-propagation` · **Date:** 2026-07-11

## Problem

Goal/objective nodes have no observed value/uncertainty channel; ISL was believed
to substitute placeholder bases (non-root objective → base 0.0;
`CONSTRAINT_NODE_DEFAULT_BASE`) — the recurring goal-fit credibility gap.
Doctrine B (ratified, PLoT #204) scores goal-fit from the forward-propagated
outcome distribution vs the normalised threshold. This lane makes the goal
node's own uncertainty semantics honest and consistent with that.

## Step 1 — pinned behaviour on tip 209d924 (verified by probe, seed=42, n=200)

Exact loci (pre-change line numbers):

| Behaviour | Locus |
|---|---|
| Non-root node base: `observed_state.value` consulted ONLY if root, else base=0.0 + Σ parent×strength + intercept | `src/services/robustness_analyzer_v2.py` L573–603 (`SCMEvaluatorV2.evaluate`), L656–683 (`evaluate_multi`) |
| ParameterUncertainty base has HIGHEST priority and is **added** to parent contributions (not a pin) | same — `factor_values` branch L573–575 then L603 |
| Constraint default-base detection + two message variants (1.26b) | L827–889; `src/models/critique.py` L350–388 |
| Root-default detection (`ROOT_NODE_DEFAULT_VALUE`) | L891–926 |

Probe results (fixtures pinned in `tests/unit/test_goal_node_propagation.py::TestPinnedTipBehaviour`):

- **(a) root goal** with observed value 0.6: constant 0.6, std 0.0 for every
  option; only generic `DEGENERATE_OPTION_ZERO_VARIANCE` + `HIGH_TIE_RATE` fire.
- **(b) non-root goal** (`lever→mid→goal`, o2 intervenes lever=0.8): mean
  0.4472 ≈ 0.8×0.8×0.7 — **the goal distribution already IS the propagated
  composition of its parents.** Supplying `observed_state.value=0.6` on the
  goal produces **bit-identical samples** — the value is silently dropped.
- **(c) constraint target with defaulted base**: both 1.26b message variants
  fire; the non-objective variant claimed "may be unreliable" even when the
  target's ancestors were fully data-supported, and its suggestion recommended
  a `point_mass` PU "so its observed_state.value is used as the sampling base".
- **(e) PU on a non-root goal** (mean=observed 0.6): o2 mean 1.0425 ≈
  0.6 + 0.447 — the PU base is **additive on top of propagation**
  (double-count), i.e. the old suggestion in (c) was a trap.

## Step 2 — design

**Finding:** the propagation machinery already implements doctrine B — a
non-root goal's distribution is the forward-propagated composition of its
parents; base=0.0 is a zero *exogenous offset*, not a fabricated value. What
was dishonest was the *semantics around it*: silent drops, a "may be
unreliable" claim that misread propagation as fabrication, a suggestion that
causes double-counting, and no goal-level disclosure when the composition
itself rests on defaulted (0.0) root ancestors.

**Change set (all ISL-internal, additive, zero numeric change):**

1. **Ancestor data-support analysis** (`_defaulted_roots_reaching`): a
   defaulted root "reaches" a target iff a directed path exists that is not
   blocked by a node every option intervenes on (an all-options intervention
   overrides the structural equation, severing upstream influence).
2. **New goal disclosures** (`_build_goal_node_disclosures`):
   - `GOAL_OBSERVED_VALUE_UNUSED` — non-root goal's observed value is not a base (doctrine B); previously silently dropped.
   - `GOAL_PU_BASE_ADDITIVE` — PU on a non-root goal shifts (does not pin) the distribution.
   - `GOAL_ANCESTOR_DATA_GAP` (warning + critique) — the propagated goal distribution rests partly on placeholder-zero roots: honest "insufficient data", disclosed, never fabricated.
   - `GOAL_NODE_ROOT_STATIC` — root goal without PU/epsilon is a constant.
3. **`CONSTRAINT_NODE_DEFAULT_BASE` honesty split (non-objective variant)** —
   wording now keys off ancestor data support, not the mere absence of a PU
   entry: supported → "forward-propagated composition … model-derived, not a
   missing-data placeholder"; gap → keeps "may be unreliable" + names the
   unsupported roots. Code, emission condition, count, and severity are
   UNCHANGED (consumers key on code — 1.26b precedent, cross-repo check:
   CEE `sanitise-enrichment.ts` buckets by code with unknown→'D' fail-safe;
   PLoT `run.ts` L1798–1824 forwards warnings by code+detail.message as-is).
4. **Suggestion trap fixed** — all three variants now state that a PU base on
   a non-root node is ADDED to parent propagation and point to `intercept`
   for a fixed exogenous offset (probe (e) is the evidence).
5. **Detail fields added** to `CONSTRAINT_NODE_DEFAULT_BASE` warnings:
   `base_semantics: "zero_base_offset_plus_parent_propagation"`,
   `ancestor_data_gap: [...]`.

**Warnings/messaging made obsolete (ROADMAP 1.26b neighbour):** the generic
"defaulted to base=0.0, constraint probability may be unreliable" claim for
ancestor-supported non-objective targets, and the `point_mass`-PU suggestion
(double-count trap). Both replaced; code unchanged.

**Fields changing semantics:** none on the wire. New warning codes are
additive; `inference_warnings[].detail` gains fields (open dict).

**Blast radius:** sensitivity, EVPI, flip thresholds, win probabilities,
constraint probabilities — **numerically unchanged** (pinned by
`TestPinnedTipBehaviour`, incl. bit-identical-samples test). Downstream: CEE
unknown critique codes suppress to diagnostics (fail-safe); PLoT forwards
unknown warning codes as-is; UI unaffected.

## Step 3 — implementation

- `src/models/critique.py`: `CONSTRAINT_NODE_DEFAULT_BASE` reworded (gap
  variant, `{gap_roots}` var), `CONSTRAINT_NODE_DEFAULT_BASE_SUPPORTED` added
  (same code), objective-variant suggestion corrected (additive, not
  replacement), `GOAL_ANCESTOR_DATA_GAP` definition + registry entry.
- `src/services/robustness_analyzer_v2.py`: data-support computation hoisted
  above the constraint block (emission order unchanged: parse → constraint →
  root → goal), `_defaulted_roots_reaching` + `_build_goal_node_disclosures`
  helpers, goal critiques appended after constraint critiques.
- RED-first: commit `a177a61` (7 RED / 9 GREEN pinned), then implementation.

## PLoT follow-up (filed, not actioned here — cross-service semantics)

`plot-lite-service/src/integrations/isl/constraint-pu-injection.ts`
(`CONSTRAINT_PINNED_STD = 0.001`) injects PU for non-goal constrained nodes
believing it "pins constrained node to its observed value … prevents ISL
base=0.0 default". Probe (e) proves ISL ADDS the PU base to parent
propagation — the injection does not pin; it shifts the constrained node's
propagated distribution by its observed value (potential double-count in live
constraint probabilities). The goal node is correctly skipped
(`reason: 'goal_node'`), so the goal path is clean. Follow-up needs a doctrine
call: either PLoT stops injecting for parent-connected nodes (propagation +
disclosure is the honest default), or ISL grows a first-class *pin/anchor*
channel with replace-semantics. Until then ISL now disclosures the additive
semantics on the goal node and in every `CONSTRAINT_NODE_DEFAULT_BASE`
suggestion.

## Doctrine forks recorded

1. **PU-on-non-root additive vs pin** — kept additive (live behaviour),
   disclosed instead of changed: a numeric flip would silently move every live
   constraint probability under PLoT's injection. Evidence: probe (e),
   `constraint-pu-injection.ts` header comment.
2. **Ancestor-gap blocking granularity** — a node blocks upstream influence
   only if intervened by EVERY option (per-option precision deferred; the
   all-options rule can only under-report support, never over-claim it —
   fails toward caution).
3. **Insufficient data stays disclosed, not fabricated** — no invented priors
   for defaulted roots; `GOAL_ANCESTOR_DATA_GAP` names them and the critique
   says goal-fit reads as "insufficient data".
4. **Goal EVPI remains factor-only** — out of scope residual (EVPI gate is
   `parameter_uncertainties`-driven); deriving objective VOI from the
   propagated distribution is a separate design.

## Gates

- `poetry run mypy src/` — clean (134 files).
- `poetry run black --check src/ tests/` — clean.
- `tests/unit/test_goal_node_propagation.py` — 16/16 (7 were RED at `a177a61`).
- `tests/unit/test_constraint_analysis.py` — 37/37.
- Full `scripts/pre-push-validate.sh` — see PR description for the run output.
