"""
FACET-based robustness analyzer with dual uncertainty support (v2.3).

Implements Monte Carlo robustness analysis that samples both:
- Structural uncertainty: Edge existence (Bernoulli)
- Parametric uncertainty: Effect magnitude (Normal)

This enables answering:
- "Is my decision robust to uncertainty about whether this relationship exists?"
- "Is my decision robust to the effect being stronger/weaker than estimated?"
- "If an edge is weaker than modelled, which alternative option would win?"
"""

import hashlib
import logging
import math
import os
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Tuple, cast

import numpy as np
from pydantic import ValidationError as PydanticValidationError

from src.models.robustness_v2 import (
    BucketResult,
    ClampMetrics,
    ConditionalWinner,
    ConstraintAnalysis,
    ConstraintResult,
    EdgeV2,
    FactorSensitivityResult,
    FragileEdgeEnhanced,
    GoalConstraint,
    GraphV2,
    InferenceWarning,
    InterventionOption,
    NodeV2,
    ObjectiveRankedOption,
    ObjectiveRanking,
    OptionResult,
    OutcomeDistribution,
    ParameterUncertainty,
    PathContribution,
    PathDecomposition,
    ResponseMetadataV2,
    RobustnessRequestV2,
    RobustnessResponseV2,
    RobustnessResult,
    SensitivityResult,
    StabilityThresholdsResponse,
)
from src.models.response_v2 import (
    SUPPRESSED_ATTR_CONDITIONAL_WINNERS,
    SUPPRESSED_ATTR_FACTOR_SENSITIVITY,
    SUPPRESSED_ATTR_P_WIN_SENSITIVITY,
    SUPPRESSED_ATTR_STABILITY_THRESHOLDS,
    CorrelationModelV2,
    CorrelationProjectionV2,
    EffectiveCorrelationV2,
    ZeroSensitivityReason,
)
from src.constants import (
    ELASTICITY_CLAMP_MAX,
    FACTOR_SENSITIVITY_BASELINE_EPSILON,
    FACTOR_SENSITIVITY_VALUE_EPSILON,
    GRID_DO_EVPC_METHOD,
    MAX_CONTROL_CANDIDATES,
    MAX_CONTROL_VALUES,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_NODES,
    MAX_OPTIONS,
    MAX_PARAMETER_UNCERTAINTIES,
    NON_INFERENCE_KINDS,
    ZERO_VARIANCE_TOLERANCE,
)
from src.models.critique import (
    CONSTRAINT_NODE_DEFAULT_BASE,
    CONSTRAINT_NODE_DEFAULT_BASE_OBJECTIVE,
    CONSTRAINT_NODE_DEFAULT_BASE_SUPPORTED,
    GOAL_ANCESTOR_DATA_GAP,
    DEGENERATE_OPTION_ZERO_VARIANCE,
    HIGH_TIE_RATE,
    MARGINAL_SWITCH_TRUNCATED,
    STRUCTURAL_INFLUENCE_TRUNCATED,
)
from src.models.response_v2 import CritiqueV2
from src.services.range_fit import resolve_range_fits
from src.utils.rng import SEED_HASH_VERSION, SeededRNG, compute_seed_from_graph
from src.utils.downside import decision_evpi_from_regrets, expected_regret_per_option
from src.utils.evppi import (
    REGRESSION_EVPPI_METHOD,
    REGRESSION_EVPPI_NULL_PERMUTATIONS,
    factor_evppi_estimate,
)
from src.utils.correlation import CORRELATION_METHOD, CorrelationPlan, build_correlation_plan
from src.validation.request_validator import detect_graph_cycle
from src.__version__ import __version__
from src.models.metadata import generate_config_fingerprint
from src.config import get_settings
from src.config.stability_thresholds import (
    STABILITY_THRESHOLDS,
    classify_attribution_stability,
)

logger = logging.getLogger(__name__)

# Safety net: nodes that must not participate in inference. Defined in src.constants
# (single source of truth) so the request-model control_candidate validator and this
# filter can never fork on which kinds are non-inference (derive, don't mirror).
# Re-exported here for existing importers of the analyzer symbol.

# Path-decomposition safety budget: maximum number of simple intervention-target-to-goal
# paths to enumerate. A layered DAG valid under the 50-node/200-edge schema limits can have
# hundreds of thousands of simple paths; without this cap, enumeration would blow the
# sub-500ms budget. The bound is a path COUNT (not wall-clock) so truncation is deterministic
# — the same graph always truncates identically, preserving the determinism guarantee.
MAX_DECOMPOSITION_PATHS = 20000

# UC-2 (D-23.18) → re-fixed per Codex re-confirm N2 (D-23.19): walk-call budget
# for the structural-influence path enumerator — the previously UNCAPPED twin of
# the decomposition walker above. A call budget (not a completed-path budget) is
# deliberate: on an adversarial dense subgraph that never reaches the goal,
# completed paths stay at zero while exploration explodes, so only bounding
# recursion CALLS bounds the work absolutely. Count-based (not wall-clock) =>
# truncation is deterministic per graph.
#
# N2: the pool is REQUEST-WIDE (shared across all factors, consumed in factor
# order), NOT per-factor — a per-factor reset multiplied the worst case by U
# (13 factors x 200k measured ~2s; 50 x 200k would be ~8s of unpriced CPU, the
# original F2 class in a new phase). Measured cost ~0.77µs/call ≈ 1 admission
# unit/call at the ceiling anchor, so the pool ceiling is charged 1:1 as the
# `structural_influence` term whenever the phase can run. Worst wall ≈ 0.3-0.5s.
MAX_INFLUENCE_WALK_CALLS_TOTAL = 400_000

# Edge strength bounds from schema v2.6
EDGE_STRENGTH_MIN = -1.0
EDGE_STRENGTH_MAX = 1.0

# B3-S1: sqrt(2) for the Gaussian-copula uniform coupling Phi(y) = 0.5*erfc(-y/√2).
_SQRT2 = math.sqrt(2.0)

# B3-S1 MANDATORY tail-independence disclosure (D-23.4, research-sharpened). The
# Gaussian copula has zero tail dependence; co-shipping it with the downside/CVaR
# block makes this caveat load-bearing (the known 2008 failure mode: joint
# extremes are systematically understated under a Gaussian dependence model).
_CORRELATION_TAIL_NOTE = (
    "The Gaussian copula has zero tail dependence: it does not model factors "
    "moving to their extremes together, so joint tail (worst-case) co-movements "
    "may be understated. Downside metrics (CVaR, p05, expected_regret) can be "
    "optimistic when correlated factors are strongly dependent."
)
# Reason stamped on every suppressed independence-assuming per-factor attribution.
_CORRELATION_SUPPRESSION_REASON = "not_separable_under_correlation"

# Default samples for marginal switch probability calculation
MARGINAL_K_SAMPLES = 100

# ROADMAP 2.356 — the fragile-edge cap for the marginal-switch sweep.
#
# WHY A CAP EXISTS AT ALL. `_compute_marginal_switch_probability` spends
# MARGINAL_K_SAMPLES * O evaluate() calls PER FRAGILE EDGE, and the fragile set
# was threshold-gated but NOT count-capped, so the phase's cost was bounded only
# by the edge count — up to E * K * O * W units of work that the admission gate
# charged NOTHING for. Admission cannot price a quantity it cannot bound, and an
# unbounded phase behind a ceiling means the ceiling is not a ceiling.
#
# WHY 10, AND WHY TOP-K RATHER THAN FIRST-K. The marginal switch probability is a
# per-edge diagnostic a reader scans in rank order; the edges that matter are the
# most elastic ones, and they are already ranked by the elasticity the sensitivity
# phase computed. Selecting the top 10 by that same score keeps every edge a
# reader would actually look at and drops the tail that only ever cost compute.
# 10 matches FACTOR_FLIP_MAX_CANDIDATES, the house precedent for exactly this
# "rank, then bound" shape.
#
# WHAT IS LOST, STATED PLAINLY. Fragile edges beyond rank 10 keep their
# `switch_probability` (that is free — it partitions samples the base MC already
# drew) and lose only `marginal_switch_probability`, which is set to None and
# DISCLOSED via MARGINAL_SWITCH_TRUNCATED. A silently-omitted number would be the
# defect this lane exists to close, one level down.
MARGINAL_MAX_EDGES = 10

# --- EVPI below-resolution labelling (provisional_doctrine_v0) ------------------
# EVPI is the difference of two Monte Carlo proportion estimates
# (perfect_metric - baseline_metric), each computed over n_evpi_samples draws
# with INDEPENDENT seed streams (baseline: seed+100/101; perfect: per-factor
# hash seeds), i.e. no common-random-numbers pairing, so the variances add:
#
#   Var(evpi_hat) = Var(p_perfect_hat) + Var(p_baseline_hat)
#                 <= 2 * (0.25 / n)          [worst case p = 0.5 for a proportion]
#   SE_max(evpi_hat) = sqrt(0.5 / n)
#
# The noise floor is the two-sided 95% bound on that worst-case standard error:
#
#   noise_floor(n) = 1.96 * sqrt(0.5 / n)    (~0.031 at the n=2000 budget cap)
#
# The worst-case p=0.5 bound is used instead of a plug-in estimate
# sqrt((p1*(1-p1) + p2*(1-p2)) / n) because the plug-in collapses to 0 when an
# estimated metric hits 0 or 1 at small n, understating the true uncertainty.
# Entries with |evpi| < noise_floor are LABELLED below-resolution.
#
# F1 producer clamp (2026-07-14, ROADMAP 2.20 residual r1): a NEGATIVE raw
# difference is clamped to 0.0 on the wire (EVPI is definitionally
# non-negative; a negative estimate is pure MC noise) and flagged with
# evpi_clamped=true. Positive values are never altered; the raw components
# stay auditable via perfect_metric / current_metric. This supersedes the
# original T0-4 "labels over clamps" stance for the negative case only —
# labelling is otherwise unchanged.
EVPI_NOISE_FLOOR_Z = 1.96
EVPI_LABELLING_DOCTRINE = "provisional_doctrine_v0"

# S2 (D-23.8) HONEST RELABEL method tag for the win-probability sensitivity block
# (emitted as ``p_win_sensitivity`` — the wire block that was mislabelled
# ``factor_evpi``). The quantity is a decision-held-fixed win-probability delta at
# the factor's mean, computed with its OWN MC redraw — NOT value-of-information.
# The tag makes the method self-describing on the wire so no consumer can read the
# number as EVPI. Internal identifiers (``_compute_evpi``, ``EVPI_*`` budget/
# noise constants, log event names) retain their EVPI naming as an implementation
# detail — this tag governs what the WIRE calls the quantity.
P_WIN_SENSITIVITY_METHOD = "p_win_delta_at_mean_v1"

# S4 (D-23.8) value-of-control (EVPC) method tag. factor_evpc grids do(factor=value)
# over each control candidate's values on the retained joint CRN samples and takes
# max_x E[U|do(x)] − max_a E[U_a]; the grid is a discrete approximation, so the
# reported EVPC is a LOWER BOUND on the true (continuous) EVPC — more values tighten
# it. Imported from src.constants so the producer tag and the ISLResponseV2 validator
# share ONE source of truth (see GRID_DO_EVPC_METHOD there).

# Per-factor EVPI Monte Carlo depth cap: EVPI uses min(request.n_samples,
# EVPI_SAMPLE_CAP) draws per pass. Paul-ruled lenient defaults 2026-07-17:
# raised 500 → 2000 (EVPI was the noisiest displayed number — a hard 500-
# sample cap regardless of request depth put ±0.06 noise-floor values next
# to 4000-sample probabilities, and raising base K did nothing for it; the
# cap was also completely silent on the wire until n_evpi_samples shipped).
# 4× depth halves the noise floor (~0.062 → ~0.031). Cost is ~linear in
# cap × n_factors (~+0.3 s/factor staging) — watch many-factor graphs.
# n_evpi_samples and evpi_noise_floor on the wire derive from the capped
# value, so they adapt automatically. Value pinned by
# tests/unit/test_lenient_limits.py (silent revert goes RED).
EVPI_SAMPLE_CAP = 2000

# --- Flip-threshold stability bands (Track S Phase 1) ---------------------------
# A single flip threshold (edge_e_values[].flip_mean) is searched against ONE
# background — all other edges at expected value — so it is presented with
# false stability. The 2026-06-10 PLoT/ISL science-performance report
# recommends reporting flip thresholds with "a stability band from a small
# seed sweep (e.g. 5 seeds)" and basing flip confidence on band width.
# DEFAULT-ON, no env gating (Paul ruling 2026-07-17: core functionality —
# the former ISL_FLIP_STABILITY_BANDS / ISL_FLIP_STABILITY_SEEDS env vars
# are removed; rollback is a revert commit). Bands are computed whenever
# edge_e_values are; worst case stays inside the all-or-nothing
# FLIP_STABILITY_BUDGET_MS guard. Additivity vs the pre-bands base wire is
# pinned by tests/unit/test_flip_stability_bands.py against a base golden.
# N raised 5 -> 10 by the same ruling's lenient-latency amendment
# (prioritise analysis quality): a wider sweep is a better stability basis
# than the 06-10 report's minimum "e.g. 5 seeds" recommendation.
FLIP_STABILITY_N_SEEDS = 10

# --- Edge-sensitivity sub-sweep sizing -----------------------------------------
# Both edge-sensitivity sub-sweeps (_compute_existence_sensitivity and
# _compute_magnitude_sensitivity) redraw `min(CAP, n_samples // DIVISOR)` samples
# per sub-sweep. These were bare literals in THREE places — the two loop bodies
# and the `sensitivity` pricing term — i.e. a hand-maintained mirror inside a
# single file (programme trap 12). Naming them makes the loops and the price read
# from one source, and lets the /health advertisement carry them so a planning
# consumer can reproduce the term instead of hard-coding `min(100, S//10)` of its
# own (PLoT does exactly that today — src/config/sampling.ts).
SENSITIVITY_SUBSAMPLE_CAP = 100
SENSITIVITY_SUBSAMPLE_DIVISOR = 10


# =============================================================================
# Weighted compute-admission cost model (Codex F8)
# =============================================================================
#
# Replaces the pre-F8 scalar admission `n_samples * n_nodes * n_edges`, which
# (a) omitted the OPTION multiplier — the base MC loop evaluates every sample
#     once PER OPTION (up to MAX_OPTIONS), so a 10-option request did ~10x the
#     work the scalar priced;
# (b) used the wrong structural shape (`x n_edges`): each SCM evaluate() walks
#     n_nodes + n_edges work, not n_nodes * n_edges, so the scalar over-priced
#     edge-dense graphs (wrongly 422-ing benign deep single graphs) and
#     under-priced sample/option depth;
# (c) priced NONE of the optional phases (EVPI, sensitivity, e-values, bands,
#     path), the two heaviest of which — base MC and EVPI — are exactly the
#     uncapped, option-multiplying ones.
#
# The cost is expressed in "cost units" where 1 unit ~= one node-evaluation-
# equivalent, derived structurally from the actual loop bodies (see the phase
# inventory in the F8 design). The structural SHAPE is correct-by-construction;
# the numeric CEILING is calibrated by benchmarks/admission_calibration.py.
#
# SINGLE SOURCE OF TRUTH: these constants are read by BOTH the admission gate
# (src/api/robustness.py) AND the /health advertisement (src/api/health.py), so
# /health can never advertise a formula that differs from what admission
# enforces (derive, don't mirror — programme memory-trap #12). PLoT reads the
# /health block instead of hand-copying the numbers.

# Formula version — advertised on /health so a version-guarded consumer (PLoT)
# can fail loud on an unknown future shape rather than silently mis-plan.
# v3 (A3 Codex-fix-A, D-23.12): added the EVPC (value-of-control) and full-
# population EVPPI (S2 regression) terms — both were UNPRICED, so a request with
# max control_candidates was admitted charging ~90M actual work units against the
# 24M ceiling. The formula SHAPE changed, so the version bumps.
# v4 (OC-1, D-23.17): recalibrated the EVPPI term against MEASURED phase wall-time
# (benchmarks/oc1_evppi_recal.py, 27-cell S x U x O grid). The v3 term
# (deg+1)*U*(1+K)*O*S over-charged 42-192x: the estimator solves every option in
# ONE multi-RHS lstsq SVD (src/utils/evppi.py _inner_expected_max), so the phase
# is O-SUBLINEAR (measured O=2 -> O=10 wall ratio 1.6-2.8x across machines/
# shapes — Codex re-measured 2.8x; NOT flat, but far under the 5x charged), and the
# (deg+1) factor is second-order inside the SVD, not a multiplier. New term:
# W_EVPPI_COEF*U*(1+K)*S — still conservative (>=1.9x margin at the worst measured
# cell on loaded local hardware; more on typical shapes). Shape changed -> bump.
# v5 (ROADMAP 2.228-F3): adds the `factor_flips` term. Bumped deliberately —
# tests/unit/test_admission_calibration.py::test_formula_version_pinned exists so
# a formula change cannot land silently, and /health advertises this string.
# v6 (ROADMAP 2.356): adds `status_quo` and `alternative_winners` — two phases
# that were performing real evaluate() work the formula did not charge for, so
# the "ceiling" could be cleared by a request that then did up to ~3x the
# admitted work. Found by the evaluator-call-count ORACLE
# (tests/unit/test_admission_evaluator_oracle.py), which instruments the real
# SCMEvaluatorV2 and asserts advertised >= counted*W end to end — the check no
# self-consistency test could make, because both residuals were consistent with
# a formula that was simply missing terms.
#
# ⚠ THIS BUMP ALSO GROWS THE `weights` KEY SET, WHICH IS A BREAKING CHANGE AT THE
# PLoT SEAM BY DESIGN. PLoT couples the weights key set exactly to the version and
# treats an unexpected key as skew, so v6 must ship with the lockstep PLoT release
# that teaches it the v6 spec (see this PR's cross-reference). DEPLOY PLoT FIRST:
# PLoT keeps BOTH the v5 and v6 specs, so a v6-aware PLoT still prices a v5 ISL
# correctly, while a v6 ISL in front of a v5-only PLoT would skew every request.
COMPLEXITY_FORMULA_VERSION = "v6-status-quo-alt-winners-2026-08-03"

# Per-phase structural weights (provisional; the calibration harness is the
# source of truth for refining them — do not hand-tune without re-running it).
BASE_COST_COEF = 1  # base MC: 1 unit per sample x option x (nodes+edges) evaluate()
W_SENS_COEF = 4  # edge sensitivity: 4 sub-sweeps per edge (existence +/- , magnitude +/-)
W_EVAL_COEF = 20  # e-values: ~binary-search depth per edge (wall-clock-capped, so flat)
W_BANDS_COEF = 200  # stability bands: 10 seeds x ~20 search per edge (capped, so flat)
W_PATH_COEF = 1  # path decomposition: analytic, bounded by MAX_DECOMPOSITION_PATHS
# Factor flips (ROADMAP 2.228-F3): 1 unit per evaluate() x structural work W, the
# same convention as base_mc (S*O evaluates x W). The evaluate COUNT is bounded in
# closed form by the phase's own caps rather than estimated — see the term.
W_FACTOR_FLIP_COEF = 1

# Value-of-control (EVPC, S4) grid-do cost coefficient. _compute_factor_evpc grids
# do(factor=x) over EVERY (candidate, value) pair on EVERY retained sample, each a
# full SCM evaluate() (cost ~W = nodes+edges) — structurally identical to a base-MC
# sample-option. So the phase costs BASE_COST_COEF * S * W * sum_candidates(len(values)).
# Priced at BASE_COST_COEF so a grid point charges exactly like a base-MC unit (an
# HONEST lower bound on the true wall-clock; conservative because the do()-evaluator
# runs with epsilon noise disabled, i.e. no per-sample RNG overhead).
W_EVPC_COEF = BASE_COST_COEF  # 1 unit per (grid-point x sample x struct)

# Full-population EVPPI (S2 regression) cost coefficient. _compute_factor_evppi runs,
# per non-lever uncertain factor, (1 + REGRESSION_EVPPI_NULL_PERMUTATIONS) polynomial
# regressions (1 real + K permutation-null fits) over the FULL retained population S —
# NOT the min(S, EVPI_SAMPLE_CAP) subsample the p_win 'evpi' term prices.
#
# v4 RECALIBRATED (OC-1, D-23.17) against measured phase wall-time
# (benchmarks/oc1_evppi_recal.py). What the measurement showed:
#   * wall ~ linear in U*(1+K)*S (ms/(U*17*S) stable at ~110-280ns across the grid);
#   * O-SUBLINEAR — each fit is ONE multi-RHS lstsq SVD shared across all options
#     (src/utils/evppi.py _inner_expected_max; per-option work is only the cheap
#     back-substitution), measured O=2 -> O=10 ratio ~1.6-1.9x vs the 5x the old *O
#     factor charged;
#   * the old (deg+1)=5 multiplier double-counted work already inside the SVD.
# Net: the v3 term (deg+1)*U*(1+K)*O*S over-charged 42-192x, 422-ing legal requests
# (e.g. S=10000/U=20/O=2 charged 34M against the 24M ceiling; measured wall ~0.4s).
# At W_EVPPI_COEF=1 the charge still over-bounds the worst measured cell by >=1.9x
# on loaded local hardware (ceiling-anchored units), more on typical shapes —
# conservative direction preserved, margin lands in the 1.5-3x band on slower
# staging hardware.
W_EVPPI_COEF = 1  # unit per (factor-fit x permutation x sample); charge is O-flat,
# wall is O-sublinear (1.6-2.8x measured O=2->10) — the ~3x margin absorbs it

# ROADMAP 2.356. Per-draw STATUS-QUO reference (ROADMAP 2.286): for a level-framed
# goal_threshold, _run_monte_carlo runs one additional complete SCM evaluation per
# draw, on its own evaluator, with no interventions. That is exactly one evaluate()
# per sample — no option multiplier, because the reference is shared across options
# by construction (common random numbers) — so it costs S*W at the base_mc
# convention of 1 unit per evaluate() x W.
W_STATUS_QUO_COEF = BASE_COST_COEF

# ROADMAP 2.356. Marginal-switch sweep inside _compute_alternative_winners: one
# baseline winner determination (O evaluates, now computed ONCE for the whole
# request — see _compute_alternative_winners) plus MARGINAL_K_SAMPLES * O
# evaluates per priced fragile edge. Same evaluate()-times-W convention.
W_ALT_WINNER_COEF = BASE_COST_COEF

# PROVISIONAL admission ceiling in cost units.
#
# ⚠ PROVISIONAL, Paul-DIRECTED (2026-07-18) but still pending STAGING recalibration.
#   Starting envelope from the F8 design's worked table (12M for base+EVPI),
#   widened after including ALL phase terms; Paul then reviewed the admit/reject
#   table and chose the MORE LENIENT ceiling (20M -> 24M) to admit the
#   dense-mid-10opt case (23.7M). At 24M the heaviest ADMITTED case
#   (schema-max-1opt, ~22.5M, measured ~24.7s local) still completes well inside
#   ISL's OVERALL_REQUEST_BUDGET_MS = 50000 and PLoT's 60s timeout, so it returns
#   cleanly; F15's compute governor is the next line of resource defense. Only the
#   genuinely abusive combos (multi-option x multi-EVPI ~35M, schema-max 45M) reject.
#   Calibrated indicatively on local hardware (benchmarks/admission_calibration.py);
#   local hardware != Render isl-staging, so STAGING RECALIBRATION IS OWED before
#   this is locked. Env-adjustable via ISL_MAX_COST_UNITS (a NEW env name —
#   deliberately NOT the old ISL_MAX_COMPUTE_COMPLEXITY, whose value would be in the
#   OLD scalar units and would silently mis-bound this formula).
DEFAULT_MAX_COST_UNITS = 24_000_000

# Wall-clock target the ceiling is calibrated against (see harness).
TARGET_WALL_MS = 25000

# ---------------------------------------------------------------------------
# Phase-cost attribution registry (Codex F2 CLASS-closer, 25 Jul 2026).
#
# ROOT CAUSE of F2: `_compute_factor_evpc` shipped a new candidate x value x
# sample evaluator loop and NOBODY extended compute_weighted_cost — nothing
# failed. This registry makes that failure LOUD: a guard test
# (tests/unit/test_admission_calibration.py::TestPhasePricingInventory)
# asserts every `_compute_*` / `_run_*` method on RobustnessAnalyzerV2 appears
# here, so adding a phase without answering the pricing question breaks CI
# with an instruction, not a silent free ride. (Trap #12: where you cannot
# derive, the mirror must FAIL LOUD on drift.)
#
# ⚠ HONEST SCOPE (Codex re-confirm N3, D-23.19): this is a NAME-PREFIX
# TRIPWIRE, not a proof. A phase named outside the registered prefixes evades
# it, and `bounded:` entries are prose claims — each one needs its own
# enforcing test to be more than an assertion. Do not describe this registry
# as a class-wide guarantee; it raises the cost of the F2 mistake, it does
# not make it impossible.
#
# Value grammar:
#   "priced:<term>"     — charged by that compute_weighted_cost term (the guard
#                         also asserts the term really exists in the formula).
#   "subsumed:<method>" — runs inside another registered phase's priced loop /
#                         post-processes its outputs without new evaluate()s.
#   "bounded:<reason>"  — deliberately unpriced; the stated bound justifies it.
#                         An honest annotation of a KNOWN residual belongs here
#                         too — never launder an under-charge into "bounded"
#                         without naming it.
PHASE_COST_ATTRIBUTION: Dict[str, str] = {
    # N3 widening (_run_ prefix) immediately surfaced _run_monte_carlo — the
    # ACTUAL S*O*W evaluate() loop; option_results post-processes its outputs
    # (the earlier priced attribution on option_results was imprecise).
    # ROADMAP 2.356: _run_monte_carlo carries TWO priced loops, not one. Since
    # 2.286 it also runs the per-draw status-quo reference for a level-framed
    # goal. The single-term attribution was true when written and went false
    # underneath it — which is why the evaluator-call-count oracle exists: this
    # registry can only record an answer, never check it.
    "_run_monte_carlo": "priced:base_mc,status_quo",
    "_compute_option_results": "subsumed:_run_monte_carlo",
    "_compute_confidence_interval": "subsumed:_compute_option_results",
    "_compute_constraint_analysis": "subsumed:_compute_option_results",
    "_compute_constraint_probabilities": "subsumed:_compute_constraint_analysis",
    "_compute_conditional_probabilities": "subsumed:_compute_constraint_analysis",
    "_compute_near_miss_diagnostics": "subsumed:_compute_constraint_analysis",
    "_compute_sensitivity": "priced:sensitivity",
    "_compute_existence_sensitivity": "subsumed:_compute_sensitivity",
    "_compute_magnitude_sensitivity": "subsumed:_compute_sensitivity",
    "_compute_factor_sensitivity": (
        "bounded: 2 deterministic evaluates per uncertain factor (2*U*W <= ~25k "
        "units at caps); its _compute_structural_influence child is separately "
        "priced, see its entry"
    ),
    "_compute_conditional_winners": "bounded: partitions existing MC samples, no new evaluates",
    "_compute_bucket_result": "subsumed:_compute_conditional_winners",
    "_compute_bootstrap_stability": (
        "bounded: ADAPTIVE-COUNT, not a wall-clock cap (N3 correction — the first "
        "10 iterations always run, THEN elapsed time may admit 10 more; measured "
        "~257ms with a slow batch); iteration count is hard-capped at 20"
    ),
    "_run_bootstrap_iterations": "subsumed:_compute_bootstrap_stability",
    # Request-wide walk pool MAX_INFLUENCE_WALK_CALLS_TOTAL charged 1:1
    # (~0.77µs/call measured ≈ 1 unit/call); exact-or-null scores (N1),
    # truncation disclosed via STRUCTURAL_INFLUENCE_TRUNCATED.
    "_compute_structural_influence": "priced:structural_influence",
    "_compute_path_decomposition": "priced:path_decomposition",
    "_compute_robustness": "bounded: post-processing of existing samples; heavy child is _compute_alternative_winners",
    "_compute_edge_e_values": "priced:e_values",
    "_compute_factor_flip_values": "priced:factor_flips",
    "_compute_factor_evppi": "priced:evppi_full",
    "_compute_factor_evpc": "priced:evpc",
    "_compute_evpi": "priced:evpi",
    "_compute_evpi_metric": "subsumed:_compute_evpi",
    # ROADMAP 2.356 CLOSED the known-undercharge this entry used to confess. The
    # confession was honest and it was still not a bound: the phase is now
    # count-capped (MARGINAL_MAX_EDGES) and priced.
    "_compute_alternative_winners": "priced:alternative_winners",
    # The once-per-request expected-value baseline the sweep probes against; its
    # O evaluations are the `1 +` inside the alternative_winners term.
    "_compute_marginal_baseline": "subsumed:_compute_alternative_winners",
    "_compute_marginal_switch_probability": "subsumed:_compute_alternative_winners",
}


def get_max_cost_units() -> int:
    """Return the admission ceiling in cost units (env-resolved).

    Reads ``ISL_MAX_COST_UNITS`` (NEW env name in the new cost units) if set,
    else ``DEFAULT_MAX_COST_UNITS``. The old ``ISL_MAX_COMPUTE_COMPLEXITY`` env
    (scalar units) is intentionally NOT read here — reusing it would silently
    repurpose an old-units value against the new formula.
    """
    val = os.environ.get("ISL_MAX_COST_UNITS")
    if val is not None:
        try:
            return int(val)
        except ValueError:
            logger.warning(
                "ISL_MAX_COST_UNITS env var is not a valid integer (%s), using default %d",
                val,
                DEFAULT_MAX_COST_UNITS,
            )
    return DEFAULT_MAX_COST_UNITS


@dataclass(frozen=True)
class WeightedCost:
    """Result of compute_weighted_cost: the total plus a per-term breakdown.

    ``dominant_term`` names the single largest contributing phase, so the 422
    body can tell the caller which part of their request drove the rejection.
    """

    total: int
    terms: Dict[str, int]

    @property
    def dominant_term(self) -> str:
        if not self.terms:
            return "base_mc"
        return max(self.terms, key=lambda k: self.terms[k])


def compute_weighted_cost(request: RobustnessRequestV2) -> WeightedCost:
    """Weighted compute-admission cost for a v2 request, in cost units.

        cost = S*O*W                                       (base MC, always)
             + W_STATUS_QUO_COEF*S*W                       (status-quo reference, if level-framed goal_threshold)
             + W_ALT_WINNER_COEF*O*(1+min(E,MARGINAL_MAX_EDGES)*MARGINAL_K_SAMPLES)*W
                                                           (alternative winners, rides on sensitivity)
             + (U+1)*min(S, EVPI_SAMPLE_CAP)*O*W           (EVPI/p_win, if include_voi & U>0)
             + W_EVPPI_COEF*U*(1+K)*S                      (full-pop EVPPI, if include_voi & U>0)
             + W_EVPC_COEF*S*W*Sum_c(len(c.values))        (EVPC, if control_candidates)
             + W_SENS_COEF*E*min(SENS_CAP, S//SENS_DIV)*W  (edge sensitivity)
             + MAX_INFLUENCE_WALK_CALLS_TOTAL              (structural influence, if sensitivity & U>0)
             + W_EVAL_COEF*E*O                             (e-values, if include_e_values)
             + W_BANDS_COEF*E*O                            (bands, ride on e-values)
             + W_FACTOR_FLIP_COEF*O*W*(1 + 2N + 2*C*(O-1+B))  (factor flips, if include_factor_flips)
             + W_PATH_COEF*min(MAX_DECOMPOSITION_PATHS, E*E) (path decomp)

    where S=n_samples, O=len(options), N=n_nodes, E=n_edges, W=N+E (per-evaluate()
    structural work), U=number of UNIQUE parameter_uncertainties, K=
    REGRESSION_EVPPI_NULL_PERMUTATIONS. Every term mirrors an actual loop body in
    the analyzer (see the F8 design phase inventory). Optional-phase enable
    conditions match analyze():
      - EVPI (p_win) AND full-pop EVPPI: include_voi AND at least one parameter_uncertainty
      - EVPC (value-of-control): control_candidates present (NOT gated on include_voi —
        control is a distinct capability from information, per _compute_factor_evpc)
      - edge sensitivity: "sensitivity" in analysis_types
      - e-values / bands: request.include_e_values (bands are default-on with e-values)
      - path decomposition: request.include_path_decomposition

    D-23.12 (Codex-fix-A): the EVPC and full-population EVPPI terms were previously
    ABSENT, so a request could add up to MAX_CONTROL_CANDIDATES*MAX_CONTROL_VALUES do()
    grids (each a full S-sample SCM sweep) at ZERO admitted cost. The EVPPI term also
    corrects the sample-count: the p_win 'evpi' term caps at EVPI_SAMPLE_CAP, but the
    S2 regression runs on the FULL S, so it needs its own full-S term.
    """
    S = request.n_samples
    O = len(request.options)
    N = len(request.graph.nodes)
    E = len(request.graph.edges)
    W = N + E

    terms: Dict[str, int] = {"base_mc": BASE_COST_COEF * S * O * W}

    # Per-draw STATUS-QUO reference (ROADMAP 2.286, priced by 2.356). One extra
    # evaluate() per sample, NO option factor — the reference is shared across
    # options by construction (common random numbers).
    #
    # THE GATE IS DELIBERATELY WIDER THAN THE PHASE'S OWN CONDITION, and that is
    # the safe direction. The phase runs iff the resolver produces a level plan
    # (GoalThresholdPlan.needs_status_quo_reference), which additionally requires a
    # convertible goal — attested baseline, parents, no pinning intervention, no
    # goal-node ParameterUncertainty. Reproducing those preconditions here would
    # duplicate the resolver in the pricing path, where a drift between the two
    # copies would silently UNDER-price (trap 12). Charging on the two request
    # fields a consumer can read directly — `goal_threshold` present AND
    # `goal_threshold_frame == "level"` — over-charges only the requests the
    # resolver then refuses, and can never under-charge one it accepts.
    if request.goal_threshold is not None and request.goal_threshold_frame == "level":
        terms["status_quo"] = W_STATUS_QUO_COEF * S * W

    # EVPI (p_win sensitivity) — priced on the DEDUPLICATED factor count (uniqueness
    # is enforced at parse time, but count unique defensively so admission never
    # over-prices a duplicate that somehow reached here). Capped at EVPI_SAMPLE_CAP
    # because _compute_evpi caps its MC redraw at that sample budget.
    if request.include_voi and request.parameter_uncertainties:
        u = len({pu.node_id for pu in request.parameter_uncertainties})
        if u > 0:
            terms["evpi"] = (u + 1) * min(S, EVPI_SAMPLE_CAP) * O * W
            # Full-population EVPPI (S2 regression) — SEPARATE term because the
            # regression runs on the FULL retained population S (never the
            # EVPI_SAMPLE_CAP subsample). (1+K) fits per factor over S samples;
            # deliberately NO O factor — the estimator shares one multi-RHS SVD
            # across options (v4 recalibration, see W_EVPPI_COEF). Uses the same
            # defensive unique factor count u (an over-count vs the analyzer's
            # lever-suppressed set — conservative).
            terms["evppi_full"] = W_EVPPI_COEF * u * (1 + REGRESSION_EVPPI_NULL_PERMUTATIONS) * S

    # EVPC (S4 value-of-control) — grid do() over every (candidate, value) pair on
    # the FULL retained population, each a full SCM evaluate() (cost ~W). Gated on
    # control_candidates presence ONLY (not include_voi). The dominant free-ride the
    # v2 formula admitted before D-23.12.
    if request.control_candidates:
        grid_points = sum(len(c.values) for c in request.control_candidates)
        if grid_points > 0:
            terms["evpc"] = W_EVPC_COEF * S * W * grid_points

    # Edge sensitivity — reference option only (not multiplied by O).
    if "sensitivity" in request.analysis_types:
        terms["sensitivity"] = (
            W_SENS_COEF * E * min(SENSITIVITY_SUBSAMPLE_CAP, S // SENSITIVITY_SUBSAMPLE_DIVISOR) * W
        )
        # Structural influence (factor-sensitivity child; N2, D-23.19): charged at
        # the request-wide walk-pool ceiling (1 walk call ≈ 1 unit at the ceiling
        # anchor, measured ~0.77µs/call). Gate mirrors the phase gate
        # (uncertainties AND sensitivity requested); deliberately over-charges
        # the correlation-suppressed case — conservative, and the term is small.
        if request.parameter_uncertainties:
            terms["structural_influence"] = MAX_INFLUENCE_WALK_CALLS_TOTAL

        # Alternative winners / marginal switch (ROADMAP 2.356). Gated on the
        # SENSITIVITY phase, not on a flag of its own: the fragile set is derived
        # from the sensitivity results, so with no sensitivity phase the list is
        # empty and the sweep performs zero evaluations.
        #
        # The evaluate() count is bounded in closed form, which it was not before
        # this version:
        #     baseline winner      O          computed ONCE per request (hoisted
        #                                     out of the per-edge loop — the
        #                                     baseline config never depended on
        #                                     the edge, so this was F-fold
        #                                     duplicated work)
        #     per priced edge      K * O      K = MARGINAL_K_SAMPLES
        #     priced edges         <= min(E, MARGINAL_MAX_EDGES)
        # DERIVED, NOT MIRRORED (trap 12): both bounds are read from the constants
        # the sweep itself uses, so raising either raises the admitted price
        # automatically and the term cannot drift out of step with its loop.
        priced_edges = min(E, MARGINAL_MAX_EDGES)
        alt_evaluates = O * (1 + priced_edges * MARGINAL_K_SAMPLES)
        terms["alternative_winners"] = W_ALT_WINNER_COEF * alt_evaluates * W

    # E-values and the stability bands that ride on them (bands default-on).
    if request.include_e_values:
        terms["e_values"] = W_EVAL_COEF * E * O
        terms["bands"] = W_BANDS_COEF * E * O

    # Factor flips (ROADMAP 2.228-F3). The evaluate() count is BOUNDED IN CLOSED
    # FORM by the phase's own structure, not estimated:
    #     baseline winner          O
    #     candidate screen         2*O per eligible root factor, and eligible root
    #                              factors <= N (a factor cannot outnumber the nodes)
    #     crossing confirmations   <= 2*(O-1) probes per candidate, O evaluates each
    #     stability bands          B backgrounds x 2*O evaluates per candidate
    # with the candidate count capped at FACTOR_FLIP_MAX_CANDIDATES. Each evaluate()
    # costs ~W, matching the base_mc convention (cost = evaluates x W).
    #
    # DERIVED, NOT MIRRORED (trap 12): the cap and the seed count are read from the
    # analyzer itself, so raising either one raises the admitted price automatically
    # and this term can never drift out of step with the loop it prices.
    if request.include_factor_flips:
        candidate_cap = RobustnessAnalyzerV2.FACTOR_FLIP_MAX_CANDIDATES
        evaluates = O * (1 + 2 * N + 2 * candidate_cap * (max(O - 1, 0) + FLIP_STABILITY_N_SEEDS))
        terms["factor_flips"] = W_FACTOR_FLIP_COEF * evaluates * W

    # Path decomposition — analytic, path-count bounded.
    if request.include_path_decomposition:
        terms["path_decomposition"] = W_PATH_COEF * min(MAX_DECOMPOSITION_PATHS, E * E)

    total = sum(terms.values())
    return WeightedCost(total=total, terms=terms)


def build_compute_admission() -> Dict[str, Any]:
    """Assemble the /health `compute_admission` block from the module constants.

    Single source of truth: /health reads THIS, so the advertised ceiling,
    weights, and caps are exactly what the admission gate and the model enforce.
    ``max_cost_units`` is env-resolved (matches the live enforced ceiling).

    SUFFICIENCY CONTRACT (ROADMAP 2.260 step 3). Between them, ``weights``,
    ``formula_parameters`` and ``caps`` carry EVERY number
    ``compute_weighted_cost`` uses that a consumer cannot derive from its own
    request. Everything else is read off the caller's own request (S, O, N, E,
    W=N+E, the unique uncertainty count, the control grid size).

    The invariant: **a consumer holding this block plus its own request can
    reproduce ``compute_weighted_cost`` exactly**, for the advertised formula
    shape. Enforced mechanically, not by review —
    ``tests/unit/test_admission_calibration.py::TestAdvertisementSufficiency``
    reimplements the formula from the advertised values ALONE and asserts
    term-by-term equality across a shape grid, so a new term (or a new constant
    inside an existing term) that is not advertised breaks CI.

    ⚠ WHY ``formula_parameters`` IS A SIBLING OF ``weights``, NOT PART OF IT.
    ISL's own precedent would put these inside ``weights`` (``evpi_sample_cap``,
    ``evppi_null_permutations``, ``influence_walk_pool`` and
    ``max_decomposition_paths`` are already caps and counts living there). That
    precedent is deliberately NOT followed, for a consumer-side reason verified
    at the bytes on PLoT #302 (head 5fff2253): PLoT couples the ``weights`` KEY
    SET exactly to the formula version — it requires every expected key present
    (``compute-admission.ts:125-130``) AND treats any unexpected key as skew
    (``:141-145``, ``:175-184``). Growing ``weights`` therefore forces a
    lockstep PLoT release or PLoT degrades to its conservative fallback — the
    very 10,000 -> 4,000 depth cut this change exists to lift. Sibling keys are
    ignored by PLoT's shape check (``:110-122``), so this lands additively and a
    consumer adopts the parameters when it is ready.

    ⚠ THE COST OF THAT CHOICE, STATED PLAINLY. PLoT's unknown-weight-key guard
    is a DRIFT ALARM: a new key inside ``weights`` tells PLoT that ISL's formula
    grew something it does not price. Parameters here are OUTSIDE that alarm —
    PLoT cannot detect their addition. That protection is therefore owed
    entirely by ISL's own TestAdvertisementSufficiency (which is stronger, being
    an equality rather than a key-set comparison) plus the formula version. Do
    not add a parameter here believing a consumer will notice; it will not.
    """
    return {
        "max_cost_units": get_max_cost_units(),
        "complexity_formula_version": COMPLEXITY_FORMULA_VERSION,
        "weights": {
            "base_per_sample_per_option_per_struct": BASE_COST_COEF,
            "evpi_sample_cap": EVPI_SAMPLE_CAP,
            "evpc_coef": W_EVPC_COEF,
            "evppi_full_coef": W_EVPPI_COEF,
            "evppi_null_permutations": REGRESSION_EVPPI_NULL_PERMUTATIONS,
            "factor_flip_coef": W_FACTOR_FLIP_COEF,
            "influence_walk_pool": MAX_INFLUENCE_WALK_CALLS_TOTAL,
            "sensitivity_coef": W_SENS_COEF,
            "evalue_coef": W_EVAL_COEF,
            "bands_coef": W_BANDS_COEF,
            "path_coef": W_PATH_COEF,
            "max_decomposition_paths": MAX_DECOMPOSITION_PATHS,
            # ROADMAP 2.356 — the two v6 terms. These are the FIRST additions to
            # this key set since the sibling-vs-weights argument was settled, and
            # they go here rather than in `formula_parameters` for the reason that
            # argument turned on: they are per-phase COEFFICIENTS, not a term's own
            # loop bounds. The cost is the one the docstring names — a consumer
            # coupled to the key set sees skew — and it is paid deliberately, with
            # the version bumped to v6 and the lockstep consumer release shipped
            # alongside. Growing `weights` without bumping the version is the thing
            # that must never happen; growing it WITH a bump is the sanctioned path.
            "status_quo_coef": W_STATUS_QUO_COEF,
            "alt_winner_coef": W_ALT_WINNER_COEF,
        },
        # Per-term structural parameters (ROADMAP 2.260 step 3) — the numbers a
        # term's own loop bounds itself by, as opposed to the per-phase
        # coefficients in `weights`. Keyed BY TERM NAME (the same strings
        # WeightedCost.terms uses), so a consumer can associate each parameter
        # with the term it prices without a naming convention to remember.
        # See build_compute_admission.__doc__ for why this is a sibling of
        # `weights` rather than part of it.
        "formula_parameters": {
            # The `factor_flips` term is O*W*(1 + 2N + 2*C*(O-1+B)). C and B were
            # the two numbers a consumer could not obtain, so the term was
            # unpriceable from the advertisement and PLoT fell back conservatively
            # — the silent 10,000 -> 4,000 depth cut that PLoT #302 made loud.
            # DERIVED from the same symbols the term reads, so raising either
            # raises the advertised price with it.
            "factor_flips": {
                "max_candidates": RobustnessAnalyzerV2.FACTOR_FLIP_MAX_CANDIDATES,
                "stability_seeds": FLIP_STABILITY_N_SEEDS,
            },
            # The `sensitivity` term is W_SENS_COEF*E*min(CAP, S//DIVISOR)*W.
            # Found by the 2.260 completeness audit, NOT by the original report:
            # these were bare literals, so every consumer had to hard-code them
            # (PLoT hard-codes `Math.min(100, Math.floor(S / 10))` at
            # src/config/sampling.ts) and would silently mis-price the term if ISL
            # ever retuned the sub-sweep.
            "sensitivity": {
                "subsample_cap": SENSITIVITY_SUBSAMPLE_CAP,
                "subsample_divisor": SENSITIVITY_SUBSAMPLE_DIVISOR,
            },
            # ROADMAP 2.356. The `alternative_winners` term is
            # O*W*(1 + min(E, max_edges)*marginal_k_samples). Both numbers bound
            # the sweep's own loop, so both are parameters rather than
            # coefficients, and both are DERIVED from the constants the sweep
            # reads — raising the cap raises the advertised price with it, and a
            # consumer can never be left hard-coding a bound ISL later retunes.
            "alternative_winners": {
                "max_edges": MARGINAL_MAX_EDGES,
                "marginal_k_samples": MARGINAL_K_SAMPLES,
            },
        },
        "caps": {
            "max_options": MAX_OPTIONS,
            "max_nodes": MAX_GRAPH_NODES,
            "max_edges": MAX_GRAPH_EDGES,
            "max_parameter_uncertainties": MAX_PARAMETER_UNCERTAINTIES,
            "max_control_candidates": MAX_CONTROL_CANDIDATES,
            "max_control_values": MAX_CONTROL_VALUES,
        },
    }


def evpi_noise_floor(n_samples: int) -> float:
    """Return the MC noise floor for an EVPI estimate over n_samples draws.

    Formula: EVPI_NOISE_FLOOR_Z * sqrt(0.5 / n_samples) — the two-sided 95%
    bound on the worst-case standard error of a difference of two independent
    Bernoulli proportion estimates (see module comment above for derivation).
    """
    if n_samples <= 0:
        # Degenerate: nothing is resolvable with no samples.
        return float("inf")
    return EVPI_NOISE_FLOOR_Z * math.sqrt(0.5 / n_samples)


def filter_inference_graph(graph: GraphV2, *, log: bool = True) -> GraphV2:
    """Filter out non-inference nodes and incident edges as a safety net.

    Args:
        graph: Graph to filter.
        log: Emit the filtered-nodes warning. Pass False for auxiliary calls
            (e.g. seed derivation) so the warning fires once per request,
            from the analyzer's authoritative filter.
    """
    filtered_nodes = [node for node in graph.nodes if node.kind.lower() not in NON_INFERENCE_KINDS]
    removed_nodes = len(graph.nodes) - len(filtered_nodes)

    if removed_nodes == 0:
        return graph

    kept_node_ids = {node.id for node in filtered_nodes}
    filtered_edges = [
        edge for edge in graph.edges if edge.from_ in kept_node_ids and edge.to in kept_node_ids
    ]
    removed_edges = len(graph.edges) - len(filtered_edges)
    removed_node_ids = [node.id for node in graph.nodes if node.kind.lower() in NON_INFERENCE_KINDS]

    if log:
        logger.warning(
            "robustness_v2_filtered_non_inference_nodes",
            extra={
                "removed_node_count": removed_nodes,
                "removed_edge_count": removed_edges,
                "removed_node_ids": removed_node_ids,
                "remaining_node_count": len(filtered_nodes),
                "remaining_edge_count": len(filtered_edges),
            },
        )

    return GraphV2(nodes=filtered_nodes, edges=filtered_edges)


def compute_effective_seed(
    request: RobustnessRequestV2,
) -> Tuple[int, Literal["client_provided", "server_computed"]]:
    """Single source of truth for the analysis seed.

    A client-provided seed always wins. Otherwise the seed is derived from
    the POST-FILTER inference graph — the same graph the analyzer samples
    on — so the seed reported in responses always matches the RNG streams
    actually used. (Deriving from the raw graph would diverge whenever
    organisational decision/option/constraint nodes get filtered out.)

    Idempotent with respect to filtering: filter_inference_graph on an
    already-filtered graph is a no-op, so this may be called before or
    after the analyzer applies the filter.
    """
    if request.seed is not None:
        # Explicit None check: seed=0 is a valid explicit seed.
        return int(request.seed), "client_provided"
    # log=False: the analyzer's own filter logs the filtered-nodes warning;
    # this auxiliary filter exists only to hash the same graph.
    try:
        graph = filter_inference_graph(request.graph, log=False)
    except PydanticValidationError:
        # Filtering removed every node (all-organisational graph), so the
        # filtered GraphV2 cannot be constructed. Analysis fails closed
        # downstream anyway; hash the raw graph so seed derivation itself
        # never crashes a route outside its envelope error handling.
        graph = request.graph
    return compute_seed_from_graph(graph), "server_computed"


# =============================================================================
# Fragile Edge with Alternative Winner
# =============================================================================


@dataclass
class FragileEdge:
    """Internal representation of a fragile edge with alternative winner analysis."""

    edge_id: str  # "from->to" format
    from_id: str
    to_id: str
    alternative_winner_id: Optional[str] = None  # Option that wins when edge is weak
    switch_probability: Optional[float] = None  # P(alternative wins | edge weak)
    marginal_switch_probability: Optional[float] = None  # P(flip | only this edge varies)


# =============================================================================
# Dual Uncertainty Sampler
# =============================================================================


class DualUncertaintySampler:
    """
    Samples edge configurations with structural + parametric uncertainty.

    For each edge:
    1. Sample existence from Bernoulli(exists_probability)
    2. If exists, sample strength from Normal(mean, std)
    3. If not exists, effective_strength = 0

    This enables Monte Carlo integration over both uncertainty dimensions.
    """

    def __init__(self, edges: List[EdgeV2], rng: SeededRNG):
        """
        Initialize sampler.

        Args:
            edges: List of edges with dual uncertainty
            rng: Seeded random number generator
        """
        self.edges = edges
        self.rng = rng
        self._existence_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._sample_count = 0

    def sample_edge_configuration(self) -> Dict[Tuple[str, str], float]:
        """
        Sample one edge configuration.

        Returns:
            Dict mapping (from, to) -> effective_strength
            If edge doesn't exist in this sample, strength = 0
        """
        config = {}
        self._sample_count += 1

        for edge in self.edges:
            edge_key = (edge.from_, edge.to)

            # Structural uncertainty: does edge exist?
            if self.rng.bernoulli(edge.exists_probability):
                # Parametric uncertainty: what's the effect size?
                # Truncated normal via rejection sampling — avoids three-mode
                # artefacts (probability mass spikes at boundaries) that np.clip
                # would introduce.  Falls back to clamped mean after 100 attempts.
                strength = self.rng.truncated_normal(
                    edge.strength.mean,
                    edge.strength.std,
                    EDGE_STRENGTH_MIN,
                    EDGE_STRENGTH_MAX,
                )
                config[edge_key] = strength
                self._existence_counts[edge_key] += 1
            else:
                # Edge doesn't exist in this sample
                config[edge_key] = 0.0

        return config

    def sample_n_configurations(self, n: int) -> List[Dict[Tuple[str, str], float]]:
        """
        Sample n independent edge configurations.

        Args:
            n: Number of configurations to sample

        Returns:
            List of edge configuration dictionaries
        """
        return [self.sample_edge_configuration() for _ in range(n)]

    def get_existence_rates(self) -> Dict[str, float]:
        """
        Get actual existence rates from sampling.

        Returns:
            Dict mapping "from->to" -> observed existence rate
        """
        if self._sample_count == 0:
            return {}

        return {
            f"{edge.from_}->{edge.to}": (
                self._existence_counts[(edge.from_, edge.to)] / self._sample_count
            )
            for edge in self.edges
        }


# =============================================================================
# Factor central value (ROADMAP 2.1020) — ONE resolver, every site
# =============================================================================

# Provenance of a resolved central value. On the wire only as the derived
# `value_defaulted` flag; kept as an explicit token so no consumer has to
# re-infer "was this defaulted?" from the number itself (0.0 is a legitimate
# observed value, so the number can never carry that answer).
FACTOR_VALUE_SOURCE_OBSERVED = "observed_state"
FACTOR_VALUE_SOURCE_PRIOR_MIDPOINT = "prior_midpoint"
FACTOR_VALUE_SOURCE_DEFAULT_ZERO = "default_zero"


class ResolvedFactorValue(NamedTuple):
    """A factor's central value plus where it came from."""

    value: float
    source: str


def resolve_factor_central_value(
    node: Optional[NodeV2], uncertainty: Optional[ParameterUncertainty]
) -> ResolvedFactorValue:
    """THE single definition of "this factor's central value" (ROADMAP 2.1020).

    Defined as **the expectation of the distribution FactorSampler actually
    draws from**, so that a factor's central value can never disagree with the
    factor's own samples. Every site that needs a single number for a factor
    calls this; none derives its own (enforced by
    ``tests/unit/test_factor_central_value_resolver.py``).

    Derived from the sampler at the bytes, NOT from an opinion about what the
    fields ought to mean:

    * ``uniform``    — ``_sample_from_distribution`` draws
      ``rng.uniform(range_min, range_max)`` and IGNORES ``mean`` entirely
      (``_copula_transform`` likewise, via the Phi coupling). So the centre is
      the midpoint, and ``observed_state`` does NOT enter: the sampler never
      consulted it either. ``ParameterUncertainty``'s validator guarantees
      both bounds are present and ``range_min < range_max`` for this family.
    * ``normal``     — draws ``rng.normal(mean, std)``. E = mean.
    * ``point_mass`` — returns ``mean`` exactly. E = mean.
    * no uncertainty — the factor is never sampled at all; its value is
      ``observed_state.value`` else 0.0.

    THE DEFECT THIS CLOSES. ``_compute_factor_sensitivity`` and its bootstrap
    twin previously used ``observed_value if observed_value is not None else
    0.0`` and never consumed the draws. For a PRIOR-ONLY factor (a stated
    prior range, no observed value) that is 0.0 — so a stated
    ``Uniform[0.6, 1.0]`` was probed at -0.04 / +0.04, BOTH outside its own
    declared support, and normalised by ``max(|0.0|, 0.01) = 0.01`` instead of
    0.8. The factor's elasticity came out ~80x suppressed: it read as
    near-zero influence in "what matters most" and fell below the
    ``|elasticity| >= 0.01`` flip-candidate filter.

    NOT IN SCOPE, DELIBERATELY. ``SCMEvaluatorV2.evaluate`` seeds a ROOT
    node's BASE from ``observed_state.value`` else 0.0. That answers a
    different question — "what exogenous base does one deterministic
    evaluation use" — and carries its own published doctrine (:3615, :3770).
    Folding it in here would silently move every structural analysis, so the
    two concepts are named apart rather than having their defaults aligned
    (CLAUDE.md trap 21). It does not affect the elasticity: the SCM is affine
    in a root factor's value, so a non-probed factor's base cancels out of
    ``outcome_high - outcome_low``.
    """
    observed: Optional[float] = None
    if node is not None and node.observed_state and node.observed_state.value is not None:
        observed = node.observed_state.value

    if uncertainty is not None and uncertainty.distribution == "uniform":
        range_min = uncertainty.range_min
        range_max = uncertainty.range_max
        if range_min is not None and range_max is not None:
            # E[U(a, b)]. The sampler ignores observed_state for this family,
            # so the central value must ignore it too, or the two disagree.
            return ResolvedFactorValue(
                (range_min + range_max) / 2.0, FACTOR_VALUE_SOURCE_PRIOR_MIDPOINT
            )

    if observed is not None:
        return ResolvedFactorValue(observed, FACTOR_VALUE_SOURCE_OBSERVED)

    return ResolvedFactorValue(0.0, FACTOR_VALUE_SOURCE_DEFAULT_ZERO)


# =============================================================================
# Factor Sampler (Phase 2A Part 2)
# =============================================================================


class FactorSampler:
    """
    Samples factor node values with parameter uncertainty.

    For each factor with specified uncertainty:
    1. Get mean from node's observed_state.value
    2. Sample from distribution (normal, uniform, or point_mass)

    This enables Monte Carlo integration over factor value uncertainty,
    complementing edge uncertainty (structural + magnitude).
    """

    def __init__(
        self,
        nodes: List[NodeV2],
        uncertainties: Optional[List[ParameterUncertainty]],
        rng: SeededRNG,
        correlation_plan: Optional[CorrelationPlan] = None,
    ):
        """
        Initialize factor sampler.

        Args:
            nodes: List of graph nodes (may include observed_state)
            uncertainties: List of factor uncertainty specifications
            rng: Seeded random number generator
            correlation_plan: Optional Gaussian-copula plan (B3-S1). When None,
                every factor is drawn independently from its own marginal — the
                exact pre-B3 path, byte-identical in RNG consumption. When
                supplied, the factors in ``correlation_plan.factor_order`` are
                drawn JOINTLY via the copula and the rest stay independent.
        """
        self.rng = rng
        self._node_map: Dict[str, NodeV2] = {n.id: n for n in nodes}
        self._uncertainty_map: Dict[str, ParameterUncertainty] = {
            u.node_id: u for u in (uncertainties or [])
        }
        self._sample_count = 0
        self._value_sums: Dict[str, float] = defaultdict(float)
        # B3-S1 copula plan. `_correlated_ids` is the membership set used to route
        # factors to the joint vs independent path; empty when inert.
        self._correlation_plan = correlation_plan
        self._correlated_ids: set = (
            set(correlation_plan.factor_order) if correlation_plan else set()
        )

    def sample_factor_values(self) -> Dict[str, float]:
        """
        Sample factor values for one Monte Carlo iteration.

        Returns:
            Dict mapping node_id -> sampled value for all factor nodes
            with specified uncertainty.
        """
        self._sample_count += 1
        factor_values: Dict[str, float] = {}

        # B3-S1: when a correlation plan is present, draw its factor set JOINTLY
        # via the Gaussian copula FIRST, then fall through to the independent loop
        # for every remaining factor. When no plan is present, `_correlated_ids`
        # is empty and this whole block is skipped, so the loop below consumes the
        # RNG stream exactly as the pre-B3 path did (byte-identical when absent).
        if self._correlation_plan is not None:
            self._draw_correlated(factor_values)

        for node_id, uncertainty in self._uncertainty_map.items():
            if node_id in self._correlated_ids:
                # Already drawn jointly above — never re-draw (would double-consume
                # the RNG stream and overwrite the joint value).
                continue

            node = self._node_map.get(node_id)
            if not node:
                # Node doesn't exist - skip (should have been caught by validation)
                continue

            # Central value via the ONE resolver (2.1020). Behaviour-identical
            # to the previous inline read at this site: for normal/point_mass
            # the resolver returns exactly `observed_state.value else 0.0`, and
            # the uniform branch below ignores `mean` altogether.
            mean = resolve_factor_central_value(node, uncertainty).value

            # Sample from specified distribution
            sampled_value = self._sample_from_distribution(uncertainty, mean, node_id)
            factor_values[node_id] = sampled_value
            self._value_sums[node_id] += sampled_value

        return factor_values

    def _draw_correlated(self, factor_values: Dict[str, float]) -> None:
        """Draw the correlated factor set jointly via the Gaussian copula (B3-S1).

        Draws N iid standard normals — one scalar ``rng.normal(0, 1)`` per
        correlated factor, in the plan's canonical order (the same order and draw
        mechanism the independent path uses for a normal factor, which is what
        makes an identity/rho=0 matrix reproduce the independent normal draws
        bit-for-bit). Correlates them with the Cholesky factor (``y = L @ z``),
        then maps each correlated standard-normal through that factor's marginal.

        CRN is preserved: this fills ONE ``factor_values`` dict per MC iteration,
        shared across every option by the caller — so all options see the same
        joint draw. Only the factor stream (seed+1) is consumed; the auto-noise
        stream (seed+2) is untouched.
        """
        plan = self._correlation_plan
        assert plan is not None
        order = plan.factor_order
        z = np.array([self.rng.normal(0.0, 1.0) for _ in order], dtype=float)
        y = plan.cholesky @ z  # correlated standard normals
        for i, node_id in enumerate(order):
            uncertainty = self._uncertainty_map.get(node_id)
            node = self._node_map.get(node_id)
            if uncertainty is None or node is None:
                # Guarded upstream by validation; skip defensively.
                continue
            # Same single resolver as the independent path (2.1020), and
            # likewise behaviour-identical here: `_copula_transform` uses
            # `mean` only for the normal marginal.
            mean = resolve_factor_central_value(node, uncertainty).value
            value = self._copula_transform(uncertainty, mean, float(y[i]))
            factor_values[node_id] = value
            self._value_sums[node_id] += value

    def _copula_transform(self, uncertainty: ParameterUncertainty, mean: float, y: float) -> float:
        """Map a correlated standard-normal draw ``y`` onto a factor's marginal.

        - normal:  x = mean + std * y  (exact — no CDF round-trip, so an identity
          correlation reproduces the independent normal draw bit-for-bit).
        - uniform: x = range_min + (range_max - range_min) * Phi(y), where Phi is
          the standard-normal CDF — the Gaussian-copula uniform coupling.

        point_mass factors are rejected at request validation (zero variance), so
        they never reach here; an unexpected distribution fails closed.
        """
        dist = uncertainty.distribution
        if dist == "normal":
            std = uncertainty.std or 0.0
            return mean + std * y
        if dist == "uniform":
            range_min = uncertainty.range_min
            range_max = uncertainty.range_max
            if range_min is None or range_max is None:
                raise ValueError(
                    f"Uniform distribution for correlated factor "
                    f"'{uncertainty.node_id}' requires range_min and range_max"
                )
            u = 0.5 * math.erfc(-y / _SQRT2)  # Phi(y): standard-normal CDF
            return range_min + (range_max - range_min) * u
        raise ValueError(
            f"Correlation not supported for distribution type '{dist}' "
            f"(factor '{uncertainty.node_id}')"
        )

    def _sample_from_distribution(
        self, uncertainty: ParameterUncertainty, mean: float, node_id: str
    ) -> float:
        """
        Sample a value from the specified distribution.

        Args:
            uncertainty: Distribution specification
            mean: Mean value (from observed_state.value)
            node_id: Node ID for error messages

        Returns:
            Sampled value
        """
        dist = uncertainty.distribution

        if dist == "point_mass":
            # No sampling - use observed value directly
            return mean

        elif dist == "normal":
            # Sample from Normal(mean, std)
            std = uncertainty.std or 0.0
            return self.rng.normal(mean, std)

        elif dist == "uniform":
            # Sample uniformly from [range_min, range_max]
            range_min = uncertainty.range_min
            range_max = uncertainty.range_max
            if range_min is None or range_max is None:
                raise ValueError(
                    f"Uniform distribution for node '{node_id}' requires range_min and range_max"
                )
            return self.rng.uniform(range_min, range_max)

        else:
            # Unknown distribution - fail fast instead of silent fallback
            raise ValueError(
                f"Unknown distribution '{dist}' for node '{node_id}'. "
                f"Supported: point_mass, normal, uniform"
            )

    def get_uncertainty_map(self) -> Dict[str, "ParameterUncertainty"]:
        """Return the factor uncertainty specifications keyed by node ID."""
        return self._uncertainty_map

    def get_node(self, node_id: str) -> Optional[NodeV2]:
        """Return the node for the given ID, or None if not found."""
        return self._node_map.get(node_id)

    def has_uncertainties(self) -> bool:
        """Check if any factor uncertainties are specified."""
        return len(self._uncertainty_map) > 0

    def get_mean_sampled_values(self) -> Dict[str, float]:
        """
        Get mean values from sampling for diagnostics.

        Returns:
            Dict mapping node_id -> mean sampled value
        """
        if self._sample_count == 0:
            return {}

        return {node_id: total / self._sample_count for node_id, total in self._value_sums.items()}


# =============================================================================
# SCM Evaluator
# =============================================================================


class SCMEvaluatorV2:
    """
    Evaluates structural causal model outcomes given edge configuration.

    Implements a simplified linear SCM evaluation where:
    - Node value = sum of (parent_value * edge_strength) for all incoming edges
    - Intervention nodes have fixed values
    - Evaluation follows topological order

    For more complex models, this could integrate with a full SCM engine.

    Note: Nodes may contain `observed_state` with actual factor values from
    CEE extraction. Phase 2A Part 2 will use these for:
    - Anchoring factor distributions to observed values
    - Computing realistic outcome distributions
    - Providing value-aware robustness analysis
    """

    def __init__(self, graph: GraphV2, epsilon_rng: Optional[SeededRNG] = None):
        """
        Initialize evaluator.

        Args:
            graph: Causal graph structure (nodes may include observed_state)
            epsilon_rng: Optional dedicated RNG (seed+3) for per-node epsilon
                noise.  When provided and a node has epsilon_std > 0, adds
                N(0, epsilon_std) after computing the structural equation.
                Node values are clamped to [0, 1] after epsilon noise.
        """
        self.graph = graph
        self._epsilon_rng = epsilon_rng
        self._node_order = self._compute_topological_order()
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, List[str]] = defaultdict(list)
        # Build node lookup for quick access to observed_state
        # Phase 2A Part 2: Will use this for factor value sampling
        self._nodes_by_id: Dict[str, NodeV2] = {node.id: node for node in graph.nodes}

        for edge in graph.edges:
            self._children[edge.from_].append(edge.to)
            self._parents[edge.to].append(edge.from_)

    def _compute_topological_order(self) -> List[str]:
        """Compute topological order of nodes for evaluation."""
        # Build adjacency list
        in_degree = {node.id: 0 for node in self.graph.nodes}
        adj: Dict[str, List[str]] = defaultdict(list)

        for edge in self.graph.edges:
            adj[edge.from_].append(edge.to)
            in_degree[edge.to] += 1

        # Kahn's algorithm — use deque for O(1) popleft (list.pop(0) is O(n))
        queue: deque = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for child in adj[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.graph.nodes):
            # Defence-in-depth: this branch should NEVER execute in production.
            # Cycles are blocked upstream by RequestValidator.validate() which
            # calls detect_graph_cycle() and returns a blocker critique, preventing
            # analysis entirely.  This fallback exists purely as a safety net in
            # case the validator is bypassed (e.g. direct internal calls or future
            # refactors that skip validation).
            # If this warning fires in production, investigate why RequestValidator
            # did not catch the cycle before analysis reached this point.
            logger.warning("Graph has cycles, using arbitrary node order")
            return [n.id for n in self.graph.nodes]

        return order

    def evaluate(
        self,
        edge_strengths: Dict[Tuple[str, str], float],
        interventions: Dict[str, float],
        goal_node: str,
        base_values: Optional[Dict[str, float]] = None,
        factor_values: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Evaluate outcome under given edge configuration and interventions.

        Uses a linear additive model:
        node_value = base_value + sum(parent_value * edge_strength)

        Args:
            edge_strengths: Sampled edge strengths (0 if edge doesn't exist)
            interventions: Node interventions (do(X=x))
            goal_node: Target outcome node
            base_values: Optional base values for nodes (default: 0)
            factor_values: Optional sampled factor values (overrides observed_state.value)

        Returns:
            Value at goal_node

        Phase 2A Part 2 (ACTIVE):
            Root factor nodes use observed_state.value as their base value.
            If factor_values is provided, those take precedence (for sampling).
        """
        if base_values is None:
            base_values = {}
        if factor_values is None:
            factor_values = {}

        node_values: Dict[str, float] = {}

        for node_id in self._node_order:
            if node_id in interventions:
                # Interventional value overrides structural equations
                node_values[node_id] = interventions[node_id]
            else:
                # Get node object (used for observed_state and intercept)
                node = self._nodes_by_id.get(node_id)

                # Determine base value for this node
                # Priority: factor_values > observed_state.value > base_values > 0
                if node_id in factor_values:
                    # Sampled factor value takes highest priority
                    base = factor_values[node_id]
                elif node_id in base_values:
                    # Explicitly provided base value
                    base = base_values[node_id]
                else:
                    # Check for observed_state.value on root nodes
                    is_root = len(self._parents.get(node_id, [])) == 0
                    if (
                        is_root
                        and node
                        and node.observed_state
                        and node.observed_state.value is not None
                    ):
                        base = node.observed_state.value
                    else:
                        base = 0.0

                # Compute contribution from parents
                parents_contribution = 0.0
                for parent in self._parents[node_id]:
                    edge_key = (parent, node_id)
                    strength = edge_strengths.get(edge_key, 0.0)
                    parent_value = node_values.get(parent, 0.0)
                    parents_contribution += parent_value * strength

                # Get node intercept (default 0.0 if not set)
                intercept = getattr(node, "intercept", 0.0) if node else 0.0

                node_values[node_id] = base + intercept + parents_contribution

                # Per-node epsilon noise: unexplained variance (measurement
                # error, omitted variables).  Only applied when epsilon_rng is
                # provided and the node has epsilon_std > 0.  Clamp to [0, 1]
                # to keep normalised node values in valid range.
                if self._epsilon_rng and node and node.epsilon_std > 0:
                    node_values[node_id] += self._epsilon_rng.normal(0, node.epsilon_std)
                    node_values[node_id] = max(0.0, min(1.0, node_values[node_id]))

        return node_values.get(goal_node, 0.0)

    def evaluate_multi(
        self,
        edge_strengths: Dict[Tuple[str, str], float],
        interventions: Dict[str, float],
        target_nodes: List[str],
        base_values: Optional[Dict[str, float]] = None,
        factor_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate and return values for multiple target nodes.

        Same computational model as evaluate(), but returns a dict of values
        for the specified target nodes instead of a single goal node value.

        Args:
            edge_strengths: Sampled edge strengths (0 if edge doesn't exist)
            interventions: Node interventions (do(X=x))
            target_nodes: List of node IDs to return values for
            base_values: Optional base values for nodes (default: 0)
            factor_values: Optional sampled factor values (overrides observed_state.value)

        Returns:
            Dict mapping target_node_id -> computed value
        """
        if base_values is None:
            base_values = {}
        if factor_values is None:
            factor_values = {}

        node_values: Dict[str, float] = {}

        for node_id in self._node_order:
            if node_id in interventions:
                # Interventional value overrides structural equations
                node_values[node_id] = interventions[node_id]
            else:
                # Get node object (used for observed_state and intercept)
                node = self._nodes_by_id.get(node_id)

                # Determine base value for this node
                # Priority: factor_values > observed_state.value > base_values > 0
                if node_id in factor_values:
                    base = factor_values[node_id]
                elif node_id in base_values:
                    base = base_values[node_id]
                else:
                    is_root = len(self._parents.get(node_id, [])) == 0
                    if (
                        is_root
                        and node
                        and node.observed_state
                        and node.observed_state.value is not None
                    ):
                        base = node.observed_state.value
                    else:
                        base = 0.0

                # Compute contribution from parents
                parents_contribution = 0.0
                for parent in self._parents[node_id]:
                    edge_key = (parent, node_id)
                    strength = edge_strengths.get(edge_key, 0.0)
                    parent_value = node_values.get(parent, 0.0)
                    parents_contribution += parent_value * strength

                # Get node intercept (default 0.0 if not set)
                intercept = getattr(node, "intercept", 0.0) if node else 0.0

                node_values[node_id] = base + intercept + parents_contribution

                # Per-node epsilon noise (same logic as evaluate())
                if self._epsilon_rng and node and node.epsilon_std > 0:
                    node_values[node_id] += self._epsilon_rng.normal(0, node.epsilon_std)
                    node_values[node_id] = max(0.0, min(1.0, node_values[node_id]))

        # Return only the requested target nodes
        return {node_id: node_values.get(node_id, 0.0) for node_id in target_nodes}


class PhaseDeadline:
    """Wall-clock deadline for an optional analysis phase (Codex F7).

    Anchors a monotonic ``t0`` at construction and gates on an optional
    ``budget_ms``. Collapses the copy-pasted
    ``(time.monotonic() - t0) * 1000.0 > budget_ms`` predicate and the
    ``round(..., 1)`` elapsed recompute used by the EVPI / path-decomposition
    deadline scaffolding into one value object, so the trip condition and the
    disclosed ``elapsed_ms`` can never drift between sites.

    ``budget_ms=None`` disables the gate (``exceeded()`` is always ``False``) —
    the same all-or-nothing semantics the inline predicates had for
    direct/legacy callers that pass no budget. The monotonic clock is
    NTP-step-safe. The per-phase re-check CADENCE (EVPI_DEADLINE_CHECK_INTERVAL /
    PATH_DEADLINE_CHECK_INTERVAL modulo) stays LOCAL at each call site — this
    object only owns the trip test and the elapsed readout, never how often it is
    polled.
    """

    __slots__ = ("budget_ms", "t0")

    def __init__(self, budget_ms: Optional[float]) -> None:
        self.budget_ms = budget_ms
        self.t0 = time.monotonic()

    def exceeded(self) -> bool:
        """True iff a budget is set and the wall-clock deadline has passed."""
        return self.budget_ms is not None and (time.monotonic() - self.t0) * 1000.0 > self.budget_ms

    def elapsed_ms(self) -> float:
        """Milliseconds since ``t0``, rounded to 1 dp (disclosure/log value)."""
        return round((time.monotonic() - self.t0) * 1000.0, 1)


@dataclass(frozen=True)
class GoalThresholdPlan:
    """HOW ``probability_of_goal`` must be computed for one request (ROADMAP 2.286).

    Two modes, and the distinction is not cosmetic — it is the difference
    between a number and an inverted number:

    ``delta`` mode
        The caller ATTESTED the threshold is already in the samples' own frame.
        Compare raw samples against ``delta_threshold``. Byte-identical to the
        pre-2.286 path; ISL cannot verify a caller's provenance claim.

    ``level`` mode
        The threshold is a LEVEL of the goal quantity. Levels are recovered from
        the samples per draw, against a status-quo REFERENCE::

            level_i = goal_baseline + (option_sample_i - status_quo_sample_i)

        and compared against ``level_threshold``. Both terms come from the SAME
        Monte Carlo draw (common random numbers), so everything that is not
        caused by the option — the factors' current values, the sampled edge
        strengths, the goal's intercept — cancels exactly, leaving the option's
        causal EFFECT on the goal added to the level the goal is actually at.

    Why this replaces the static ``T - B + intercept`` conversion: that formula
    assumed the goal's samples were a CHANGE from its current level, i.e. that
    the parents contribute zero under the status quo. They do not. The evaluator
    seeds ``observed_state.value`` as the base of ROOT nodes only, so parents
    carry their ABSOLUTE current values and propagate ``parent_value *
    strength`` into a non-root goal whose own base is 0.0. The status-quo sample
    is therefore ``intercept + S_sq`` with ``S_sq != 0``, and anchoring the
    conversion at zero shifted every comparison by exactly ``S_sq``. Measured on
    the witness graph (f=0.5, strength=0.5, B=0.7, T=0.9): the status quo scores
    0.25 against a converted threshold of 0.20, so ISL reported ``100%``
    confidence in reaching a goal the status quo does not reach at all — a
    CONFIDENT INVERSION, not a rounding error.

    The ``+ intercept`` term of the old formula is gone because under a
    reference anchor it cancels on its own: it is present in both the option
    sample and the status-quo sample. It is still domain-guarded as an operand,
    because an intercept in raw user units is still evidence the whole request
    is mis-normalised.
    """

    delta_threshold: Optional[float] = None
    level_threshold: Optional[float] = None
    goal_baseline: Optional[float] = None

    @property
    def needs_status_quo_reference(self) -> bool:
        """True iff computing this plan requires the per-draw status-quo series."""
        return self.level_threshold is not None


@dataclass(frozen=True)
class ObjectivePlan:
    """Resolved comparison sense, reusing the existing target frame plan.

    Unknown objectives and unresolvable targets withhold comparison. This plan
    governs per-draw winners; unsupported ancillary metrics are gated separately.
    """

    sense: Literal["maximise", "minimise", "target", "withheld"]
    attested: bool
    # Populated only for ``sense == "target"``; carries the resolved target in
    # whichever frame the GoalThresholdPlan settled on. Exactly one of
    # target_delta / target_level is non-None on a target plan.
    target_delta: Optional[float] = None
    target_level: Optional[float] = None
    goal_baseline: Optional[float] = None
    withheld_reason: Optional[str] = None

    @property
    def needs_status_quo_reference(self) -> bool:
        """True iff scoring this objective requires the per-draw status-quo series.

        Only a level-framed target does: recovering ``level_i = baseline +
        (option_i - status_quo_i)`` needs the reference draw. A delta-framed
        target compares raw samples, and neither ``maximise`` nor ``minimise``
        needs a reference at all — so every request that does not ask for a
        level target does exactly the work it did before this field existed.
        """
        return self.sense == "target" and self.target_level is not None


# =============================================================================
# Robustness Analyzer V2
# =============================================================================


class RobustnessAnalyzerV2:
    """
    Robustness analysis with dual uncertainty (v2.2).

    For each Monte Carlo sample:
    1. Sample edge configuration (existence + strength)
    2. Evaluate each option's outcome given that configuration
    3. Track which option wins

    Aggregates across samples to compute:
    - Outcome distributions per option
    - Win probabilities
    - Sensitivity to edge existence vs magnitude
    - Overall robustness assessment
    """

    # Thresholds for robustness assessment (Decision Model Schema v2.6)
    # Level mapping: robust (>= 0.7) + confidence → high/moderate; else → low/very_low
    ROBUST_THRESHOLD = 0.7  # Win probability for "robust" recommendation
    FRAGILE_THRESHOLD = 0.1  # Elasticity threshold for fragile edges
    HIGH_SENSITIVITY_THRESHOLD = 0.2  # Elasticity for "high sensitivity"

    def __init__(self) -> None:
        """Initialize analyzer."""
        self.logger = logger
        # Test-only override: set to an int to fix bootstrap count and skip
        # adaptive timing. None = use adaptive budget (production default).
        self._n_bootstrap_override: Optional[int] = None

    # ---- Governing request budget (A3 remediation 2026-07-18) ---------------
    # An overall wall-clock budget for the whole robustness request, governing
    # the OPTIONAL sequential phases (E-values, stability bands, EVPI, path
    # decomposition). Base MC + core robustness are NEVER gated. Each optional
    # phase checks the remaining budget and DEGRADES-WITHIN / SKIPS-WITH-
    # DISCLOSURE when insufficient, so ISL always returns a clean 200-with-
    # partial instead of being aborted mid-compute and losing everything.
    #
    # Sized BELOW PLoT's ISL_TIMEOUT_MS (60000): PLoT's caller timeout sits
    # under ISL's 90s route guillotine (src/api/main.py /
    # middleware/request_limits.py), so without this an over-long stack of
    # optional phases orphans compute — the whole analysis is lost, silently.
    # 50000 leaves a ~10s margin for the base-call round trip + PLoT overhead so
    # ISL returns first (ALTITUDE hunt 1 — the inverted timeout order).
    # Value pinned by tests/unit/test_request_budget.py (silent revert -> RED).
    OVERALL_REQUEST_BUDGET_MS = 50000

    # Minimum remaining budget below which an optional phase is not even
    # attempted (avoids paying a phase's setup cost for a sweep that would trip
    # on its first internal guard check anyway).
    OPTIONAL_PHASE_MIN_BUDGET_MS = 500

    # EVPI is gated at entry (only started when at least this much of the overall
    # budget remains) AND now carries an internal wall-clock deadline (Codex F7,
    # below) so it degrades-with-disclosure instead of running unbounded past the
    # governing budget once started.
    EVPI_MIN_BUDGET_MS = 8000

    # Codex F7: EVPI and path-decomposition were wall-clock ENTRY-gated only —
    # once started they re-checked NOTHING and could run (n_uncertainties+1) full
    # MC passes / a dense-DAG path walk UNBOUNDED, cross OVERALL_REQUEST_BUDGET_MS
    # and PLoT's 60s ISL timeout, and return every row with NO disclosure. They now
    # carry an INTERNAL wall-clock budget checked mid-loop exactly like the E-value
    # and band sweeps (min(cap, remaining) measured against a monotonic phase t0),
    # degrading ALL-OR-NOTHING with disclosure on overrun.
    #
    # Both caps default to the governing OVERALL_REQUEST_BUDGET_MS, so the effective
    # per-phase bound is min(cap, remaining) == remaining: the phase is bounded ONLY
    # by the governing request deadline, never cut tighter than it (no new false
    # cuts of runs that succeed today). They exist as the phase-level knobs
    # (mirroring E_VALUE_BUDGET_MS / FLIP_STABILITY_BUDGET_MS) and as the
    # internal-trip pins tests drive to -1 (silent revert -> RED via
    # tests/unit/test_evpi_path_deadline.py).
    EVPI_BUDGET_MS = OVERALL_REQUEST_BUDGET_MS
    PATH_DECOMPOSITION_BUDGET_MS = OVERALL_REQUEST_BUDGET_MS

    # Deadline re-check granularity inside the EVPI sample loop / path-decomp walk:
    # re-check every Nth iteration, not every one (mirrors the E-value per-bisect-
    # step cadence — time.monotonic() is cheap but not free, and the RNG/compute is
    # never touched by the guard so byte-output is unchanged when not tripped).
    EVPI_DEADLINE_CHECK_INTERVAL = 64
    PATH_DEADLINE_CHECK_INTERVAL = 512

    def _optional_phase_unavailable_warning(
        self,
        code: str,
        field: str,
        reason: str,
        elapsed_ms: float,
        message: str,
    ) -> "InferenceWarning":
        """Build the wire disclosure for an optional phase skipped/tripped under
        the request budget. Mirrors the LOG-only e_value_budget_exceeded /
        flip_stability_budget_exceeded events onto inference_warnings (the
        channel PLoT reads), carrying elapsed_ms — the #226 gap for
        flip_thresholds, now closed for the whole optional-phase family.

        Codex F4: the four optional-phase degradation codes (E_VALUES_UNAVAILABLE
        / STABILITY_BANDS_UNAVAILABLE / EVPI_UNAVAILABLE /
        PATH_DECOMPOSITION_UNAVAILABLE) are the ONLY InferenceWarnings that surface
        as 'warning' — PLoT maps severity=='warning' to a shown warning and
        everything else to 'info', so stamp 'warning' HERE explicitly. The ~9
        benign input-adjustment/default diagnostics (STRENGTH_MEAN_CLAMPED,
        CONSTRAINT_NODE_DEFAULT_BASE, ROOT_NODE_DEFAULT_VALUE, ...) are built
        directly and keep InferenceWarning's quiet 'info' default."""
        return InferenceWarning(
            code=code,
            field=field,
            severity="warning",
            detail={"reason": reason, "elapsed_ms": elapsed_ms, "message": message},
        )

    def analyze(self, request: RobustnessRequestV2) -> RobustnessResponseV2:
        """
        Perform complete robustness analysis.

        Args:
            request: v2.2 robustness request

        Returns:
            Complete analysis response
        """
        start_time = time.time()
        # Monotonic anchor for the governing request budget (NTP-step-safe).
        # start_time (wall clock) is kept only for the reported execution_time.
        budget_start = time.monotonic()

        def _budget_remaining_ms() -> float:
            return self.OVERALL_REQUEST_BUDGET_MS - (time.monotonic() - budget_start) * 1000.0

        def _elapsed_ms() -> float:
            # Rounded wall-clock elapsed since the request anchor — the disclosed
            # elapsed_ms for the optional-phase *_budget_exceeded events below.
            # Mirrors PhaseDeadline.elapsed_ms() (which uses each phase's own t0);
            # here the anchor is the overall-request budget_start.
            return round((time.monotonic() - budget_start) * 1000.0, 1)

        # Generate request_id if not provided
        request_id = request.request_id or f"robustness-{uuid.uuid4().hex[:12]}"

        # Safety net: remove non-inference nodes/edges before analysis
        filtered_graph = filter_inference_graph(request.graph)
        if filtered_graph is not request.graph:
            request = request.model_copy(update={"graph": filtered_graph})

            # Post-filter validation: ensure goal node still exists
            filtered_node_ids = {node.id for node in filtered_graph.nodes}
            if request.goal_node_id not in filtered_node_ids:
                raise ValueError(
                    f"Goal node '{request.goal_node_id}' was filtered out as non-inference node. "
                    f"Goal nodes must be of kind: factor, chance, outcome, or risk."
                )

            # Post-filter validation: warn if intervention nodes were filtered
            for option in request.options:
                missing_interventions = [
                    node_id
                    for node_id in option.interventions.keys()
                    if node_id not in filtered_node_ids
                ]
                if missing_interventions:
                    self.logger.warning(
                        "robustness_v2_interventions_filtered",
                        extra={
                            "request_id": request_id,
                            "option_id": option.id,
                            "filtered_intervention_nodes": missing_interventions,
                        },
                    )

        # Fail closed on cyclic graphs on EVERY path. The V2-enhanced route
        # also blocks cycles pre-analysis via RequestValidator; this guard
        # covers the legacy/V1 route and direct analyzer calls, where a cycle
        # would otherwise fall through to an arbitrary topological order and
        # produce plausible-looking but meaningless results.
        if detect_graph_cycle([{"from_": e.from_, "to": e.to} for e in request.graph.edges]):
            raise ValueError(
                "GRAPH_CYCLE_DETECTED: graph contains a cycle - "
                "robustness analysis requires a directed acyclic graph"
            )

        # Setup - use separate RNG streams for edge and factor sampling
        # to prevent fragile determinism coupling.
        # compute_effective_seed is the single seed-derivation path shared
        # with the API layer, so the reported seed_used always matches the
        # RNG streams used here.
        seed, _ = compute_effective_seed(request)
        rng_edge = SeededRNG(seed)
        rng_factor = SeededRNG(seed + 1)
        sampler = DualUncertaintySampler(request.graph.edges, rng_edge)
        # B3-S1: build the Gaussian-copula plan when correlations are supplied.
        # Returns None (inert) otherwise — the sampler then draws every factor
        # independently, byte-identically to the pre-B3 path. `correlation_active`
        # gates both the disclosure block and the attribution suppression below.
        correlation_plan = self._build_correlation_plan(request)
        correlation_active = correlation_plan is not None
        factor_sampler = FactorSampler(
            request.graph.nodes,
            request.parameter_uncertainties,
            rng_factor,
            correlation_plan=correlation_plan,
        )
        # Dedicated RNG stream (seed+3) for per-node epsilon noise.
        # Only instantiated when at least one node has epsilon_std > 0,
        # so existing graphs with default epsilon_std=0.0 are unaffected.
        has_epsilon = any(n.epsilon_std > 0 for n in request.graph.nodes)
        rng_epsilon = SeededRNG(seed + 3) if has_epsilon else None
        evaluator = SCMEvaluatorV2(request.graph, epsilon_rng=rng_epsilon)

        self.logger.info(
            "robustness_v2_analysis_started",
            extra={
                "request_id": request_id,
                "n_samples": request.n_samples,
                "n_options": len(request.options),
                "n_edges": len(request.graph.edges),
                "n_factor_uncertainties": len(request.parameter_uncertainties or []),
                "seed": seed,
            },
        )

        # Determine constraint target nodes for multi-constraint analysis
        constraint_target_nodes: Optional[List[str]] = None
        if request.goal_constraints:
            constraint_target_nodes = sorted(set(gc.node_id for gc in request.goal_constraints))

        # Collect parse-time warnings (e.g. STRENGTH_MEAN_CLAMPED) from graph construction.
        # These are generated during Pydantic model validation of EdgeV2.strength.mean.
        inference_warnings: List[InferenceWarning] = request.graph.collect_parse_warnings()

        # ---------------------------------------------------------------
        # Node data-support analysis (Cluster-2, Track S Phase 0).
        # Computed BEFORE the constraint block so ancestor data-support
        # checks can use it; warning EMISSION order below is unchanged
        # (parse -> constraint default-base -> root default -> goal).
        # ---------------------------------------------------------------
        parent_map: dict[str, list[str]] = defaultdict(list)
        children_map: dict[str, list[str]] = defaultdict(list)
        for edge in request.graph.edges:
            parent_map[edge.to].append(edge.from_)
            children_map[edge.from_].append(edge.to)
        uncertainty_node_ids = set(u.node_id for u in (request.parameter_uncertainties or []))
        # Nodes every option intervenes on: the intervention overrides the
        # structural equation in EVERY sample, so no base default is ever
        # used there and no upstream influence passes through.
        fully_intervened_node_ids = {
            node.id
            for node in request.graph.nodes
            if all(node.id in opt.interventions for opt in request.options)
        }

        # Root nodes that will silently default to 0.0: NO observed_state.value,
        # NO ParameterUncertainty entry (which would provide sampling via
        # FactorSampler), and not intervened by every option.
        defaulted_root_node_ids: List[str] = []
        for node in request.graph.nodes:
            is_root = len(parent_map.get(node.id, [])) == 0
            if not is_root:
                continue
            has_observed_value = (
                node.observed_state is not None and node.observed_state.value is not None
            )
            if (
                not has_observed_value
                and node.id not in uncertainty_node_ids
                and node.id not in fully_intervened_node_ids
            ):
                defaulted_root_node_ids.append(node.id)

        # Detect constraint target nodes that will silently default to base=0.0
        # (non-root nodes without ParameterUncertainty, not fully covered by
        # interventions across all options)
        constraint_default_base_critiques: List[CritiqueV2] = []
        if constraint_target_nodes:
            for node_id in constraint_target_nodes:
                is_root = len(parent_map.get(node_id, [])) == 0
                has_uncertainty = node_id in uncertainty_node_ids
                # Skip warning if every option intervenes on this node
                # (intervention value overrides the base, so base=0.0 is never used)
                all_options_intervene = node_id in fully_intervened_node_ids
                if not is_root and not has_uncertainty and not all_options_intervene:
                    # Cluster-2: a non-root node's samples are the forward-
                    # propagated composition of its parents; base=0.0 is a zero
                    # exogenous OFFSET, not a fabricated value. Whether the
                    # composition is trustworthy depends on ancestor data
                    # support, so the wording keys off that — not off the mere
                    # absence of a ParameterUncertainty entry.
                    ancestor_data_gap = self._defaulted_roots_reaching(
                        node_id,
                        children_map,
                        defaulted_root_node_ids,
                        fully_intervened_node_ids,
                    )
                    # Doctrine B (post-#204): a constraint on the graph's own
                    # objective node defaulting to base=0.0 is EXPECTED — the
                    # probability is scored from the modelled outcome
                    # distribution, not left unmeasured.
                    is_objective_node = node_id == request.goal_node_id
                    if is_objective_node:
                        message = (
                            f"Node '{node_id}' is the objective node and has no "
                            f"ParameterUncertainty — defaulted to base=0.0 as "
                            f"expected for a non-root objective; its constraint "
                            f"probability is scored from the modelled outcome "
                            f"distribution, not a missing-data placeholder"
                        )
                        default_base_critique = CONSTRAINT_NODE_DEFAULT_BASE_OBJECTIVE.build(
                            node_id=node_id,
                            affected_node_ids=[node_id],
                            seed=seed,
                        )
                    elif ancestor_data_gap:
                        gap_list = ", ".join(f"'{r}'" for r in ancestor_data_gap)
                        message = (
                            f"Node '{node_id}' has no ParameterUncertainty "
                            f"— defaulted to base=0.0, and its root ancestor(s) "
                            f"{gap_list} carry no observed value or "
                            f"ParameterUncertainty (defaulted to 0.0); constraint "
                            f"probability may be unreliable"
                        )
                        default_base_critique = CONSTRAINT_NODE_DEFAULT_BASE.build(
                            node_id=node_id,
                            gap_roots=gap_list,
                            affected_node_ids=[node_id],
                            seed=seed,
                        )
                    else:
                        message = (
                            f"Node '{node_id}' has no ParameterUncertainty "
                            f"— base offset defaulted to 0.0; its samples are "
                            f"the forward-propagated composition of its parents "
                            f"(all root ancestors carry data), so the constraint "
                            f"probability is model-derived, not a missing-data "
                            f"placeholder"
                        )
                        default_base_critique = CONSTRAINT_NODE_DEFAULT_BASE_SUPPORTED.build(
                            node_id=node_id,
                            affected_node_ids=[node_id],
                            seed=seed,
                        )
                    inference_warnings.append(
                        InferenceWarning(
                            code="CONSTRAINT_NODE_DEFAULT_BASE",
                            field=f"nodes[{node_id}].base",
                            detail={
                                "node_id": node_id,
                                "defaulted_to": 0.0,
                                "reason": "no_parameter_uncertainty",
                                "base_semantics": "zero_base_offset_plus_parent_propagation",
                                "ancestor_data_gap": ancestor_data_gap,
                                "message": message,
                            },
                        )
                    )
                    constraint_default_base_critiques.append(default_base_critique)
                    self.logger.warning(
                        "isl.constraint.default_base",
                        extra={
                            "node_id": node_id,
                            "defaulted_to": 0.0,
                            "reason": "no_parameter_uncertainty",
                            "ancestor_data_gap": ancestor_data_gap,
                        },
                    )

        # Emit root-default warnings (list computed above; emission order and
        # message unchanged)
        for node_id in defaulted_root_node_ids:
            inference_warnings.append(
                InferenceWarning(
                    code="ROOT_NODE_DEFAULT_VALUE",
                    field=f"nodes[{node_id}].observed_state.value",
                    detail={
                        "node_id": node_id,
                        "defaulted_to": 0.0,
                        "message": (
                            f"No observed value provided for root node '{node_id}'; "
                            f"defaulted to 0.0. Results for downstream nodes may be "
                            f"unreliable."
                        ),
                    },
                )
            )

        # ROADMAP 2.258: resolve goal_threshold into the goal SAMPLES' frame
        # ONCE, here, before any comparison. A 'level' threshold is converted
        # using the goal's own baseline; an unattested or unconvertible one
        # resolves to None and probability_of_goal is omitted for every option,
        # with the reason disclosed on inference_warnings.
        #
        # ROADMAP 2.279: this runs BEFORE the goal-node disclosures because one
        # of them (GOAL_OBSERVED_VALUE_UNUSED) is only actionable when the
        # conversion did NOT consume the goal's observed_state. HOISTED, not
        # duplicated — the resolution still happens exactly once per analysis,
        # at one site, and cannot drift. Moving it earlier cannot change its
        # result: it is a pure static function of `request`, and `request` is
        # never mutated after the non-inference-filter rebind above. The
        # warning it produces is still appended at its original site below, so
        # the emission ORDER of inference_warnings on the wire is unchanged.
        (
            goal_threshold_plan,
            goal_threshold_frame_warning,
        ) = self._resolve_goal_threshold_in_sample_frame(request)

        # ROADMAP 2.798: the CONSTRAINT channel's plans, resolved by the same
        # rules and at the same point in the pipeline as Channel A's. Must happen
        # BEFORE the Monte Carlo, because the plans decide which target nodes need
        # a per-draw status-quo reference recorded.
        #
        # `constraint_plans is None` => at least one constraint is unresolvable =>
        # the whole constraint_analysis block will be omitted. The warnings name
        # each refused constraint by its identity.
        constraint_plans, constraint_frame_warnings = self._resolve_constraint_plans(request)
        inference_warnings.extend(constraint_frame_warnings)

        # ROADMAP 2.1192: WHAT "wins" MEANS for this request. Resolved here,
        # beside the threshold plan it reuses and BEFORE the Monte Carlo, for
        # the same reason: it decides whether the goal needs a per-draw
        # status-quo reference recorded. A pure static function of `request` and
        # the threshold plan, so it cannot drift from either.
        #
        # A `withheld` sense => the caller asked for a target-based ranking that
        # cannot be scored => NO ranking is produced. It is a PLAN rather than a
        # None precisely so that no caller downstream can quietly substitute its
        # own default and rank by `max()` after we refused to.
        objective_plan, objective_warning = self._resolve_objective_plan(
            request, goal_threshold_plan
        )
        if objective_warning is not None:
            inference_warnings.append(objective_warning)
        objective_ranking_withheld = objective_plan.sense == "withheld"

        # Cluster-2 goal-node disclosures (Track S Phase 0): make the goal
        # node's base/propagation semantics explicit — no numeric change.
        goal_disclosure_warnings, goal_disclosure_critiques = self._build_goal_node_disclosures(
            request,
            parent_map,
            children_map,
            uncertainty_node_ids,
            fully_intervened_node_ids,
            defaulted_root_node_ids,
            seed,
            self._goal_baseline_was_consumed(request, goal_threshold_plan),
        )
        inference_warnings.extend(goal_disclosure_warnings)

        # Run Monte Carlo simulation
        (
            option_outcomes,
            option_wins,
            winner_per_sample,
            edge_configs_per_sample,
            tie_count,
            constraint_node_values,
            factor_values_per_sample,
            status_quo_node_values,
        ) = self._run_monte_carlo(
            request,
            sampler,
            factor_sampler,
            evaluator,
            constraint_target_nodes,
            # The union of every node whose level conversion needs a per-draw
            # reference: the goal (Channel A) plus each level-framed constraint's
            # target (Channel B). Derived from the PLANS, so a request that asks
            # for no conversion does exactly the work it did before.
            status_quo_reference_nodes=sorted(
                (
                    {request.goal_node_id}
                    if (
                        goal_threshold_plan is not None
                        and goal_threshold_plan.needs_status_quo_reference
                    )
                    # ROADMAP 2.1192: a level-framed TARGET objective needs the
                    # same reference series, for the same reason. Unioned into
                    # the same set — one reference per node, no second dialect.
                    or objective_plan.needs_status_quo_reference
                    else set()
                )
                | set(self._constraint_status_quo_nodes(request, constraint_plans))
            )
            or None,
            objective=objective_plan,
        )
        status_quo_outcomes = status_quo_node_values.get(request.goal_node_id, [])

        objective_ranking = self._build_objective_ranking(request, objective_plan, option_wins)
        objective_ranking_withheld = objective_ranking.status == "withheld"
        if objective_ranking.withheld_reason == "no_informative_draws":
            inference_warnings.append(
                InferenceWarning(
                    code="OBJECTIVE_RANKING_WITHHELD",
                    field="goal_direction",
                    severity="warning",
                    detail={
                        "reason": "no_informative_draws",
                        "message": "No usable model comparisons were produced, so no option is recommended.",
                    },
                )
            )
        first_rank = [row for row in objective_ranking.ranked_options if row.rank == 1]
        recommended_option_id = first_rank[0].option_id if len(first_rank) == 1 else None
        recommendation_confidence = first_rank[0].win_probability if len(first_rank) == 1 else None
        maximise_metrics_available = (
            objective_ranking.status == "computed" and objective_plan.sense == "maximise"
        )
        objective_suppressed_metrics: List[str] = []

        # B2 CRN-fix (CODE-REVIEW-ISL F1): expected_regret is a JOINT Common-
        # Random-Numbers metric and MUST be computed from the PRE-noise outcomes
        # -- the exact CRN-aligned population that produced winner_per_sample /
        # win_probability just above. `_apply_auto_scaled_noise` below draws
        # INDEPENDENT per-option noise (one rng.normal(0, outcome_std_o) per
        # option), which breaks CRN alignment; computing regret from the noised
        # samples inflates it by a pure max-over-independent-noise premium (~80x
        # for near-equivalent options) and makes it disagree with win_probability.
        # We compute it here, before the noise, and thread it to the V2 emission
        # via OptionResult.pre_noise_expected_regret. cvar_10/p05 intentionally stay on the
        # NOISED samples downstream (marginal tail metrics, consistent with the
        # noised p10/p50/p90/mean).
        pre_noise_expected_regret = (
            expected_regret_per_option(option_outcomes) if maximise_metrics_available else {}
        )
        if not maximise_metrics_available:
            objective_suppressed_metrics.extend(["downside", "decision_evpi"])

        # S2 (D-23.8) factor_evppi and S4 (D-23.8) factor_evpc both need the PRE-noise
        # per-option outcomes — the same CRN-aligned joint population that produced
        # pre_noise_expected_regret and win_probability. _apply_auto_scaled_noise below
        # reassigns each option's list IN PLACE (independent per-option noise breaks CRN
        # alignment), so snapshot the pre-noise lists here. Taken when the VOI phase
        # (include_voi) OR the value-of-control phase (control_candidates) will run —
        # EVPC uses max_a E[U_a] over this population as its baseline and is NOT gated
        # on include_voi (control is a distinct capability from information).
        pre_noise_option_outcomes: Optional[Dict[str, List[float]]] = None
        if (
            request.include_voi and factor_sampler.has_uncertainties()
        ) or request.control_candidates:
            pre_noise_option_outcomes = {oid: list(vals) for oid, vals in option_outcomes.items()}

        # Disable epsilon noise for post-MC structural analyses
        # (sensitivity, counterfactual, robustness) — these compare structural
        # differences and should not include stochastic per-sample noise.
        evaluator._epsilon_rng = None

        # Compute tie rate
        tie_rate = tie_count / request.n_samples

        # Apply auto-scaled noise to outcome/risk nodes (V08 scientific accuracy)
        # Uses separate RNG stream (seed + 2) for determinism
        rng_noise = SeededRNG(seed + 2)
        option_outcomes, auto_noise_applied = self._apply_auto_scaled_noise(
            option_outcomes,
            request.goal_node_id,
            request.graph.nodes,
            rng_noise,
        )

        # Keep constraint probabilities on the same sample semantics as
        # probability_of_goal for the goal node: a constraint on the goal node
        # must be evaluated against the exact same (possibly noised) goal
        # series, otherwise "P(goal >= x)" and "P(constraint goal >= x)" give
        # two different answers in one response. Non-goal constraint nodes
        # keep model-only (un-noised) samples; when that mixes with a noised
        # goal, the mix is disclosed via an inference warning rather than
        # silently changing the noise doctrine (deferred to Lane C review).
        if constraint_node_values is not None:
            mixed_noise_nodes = self._align_goal_constraint_samples(
                constraint_node_values,
                option_outcomes,
                request.goal_node_id,
                auto_noise_applied,
            )
            if mixed_noise_nodes:
                inference_warnings.append(
                    InferenceWarning(
                        code="CONSTRAINT_SAMPLES_UNNOISED",
                        field="goal_constraints",
                        detail={
                            "node_ids": mixed_noise_nodes,
                            "message": (
                                "Auto-scaled noise was applied to the goal node "
                                "samples but not to these constraint node samples; "
                                "their probabilities reflect model-only variance."
                            ),
                        },
                    )
                )

        # Compute results (including constraint analysis if goal_constraints provided).
        # `option_outcomes` is now POST-noise; win_probability uses the pre-noise
        # `option_wins`, and the pre-noise joint regret is passed in explicitly so
        # both joint metrics ride the SAME pre-noise population (B2 CRN-fix F1).
        # ROADMAP 2.258: goal_threshold was resolved into the goal SAMPLES'
        # frame once, above (hoisted for 2.279 — see the comment there). Its
        # warning is appended HERE, at the original emission site, so the order
        # of inference_warnings on the wire is byte-identical to before.
        #
        # ROADMAP 2.286: the level-frame plan differences each option's samples
        # against the status-quo reference, but that reference is drawn PRE-noise
        # while `option_outcomes` is now POST-noise. Auto-scaled noise is drawn
        # independently per option, so it does NOT cancel in the difference — it
        # would land in the effect estimate as variance no option caused. The
        # reference cannot be re-noised coherently either (there is no
        # "status-quo option" for _apply_auto_scaled_noise to scale). So the only
        # honest move is the one the rest of this seam already makes: refuse.
        # This is unreachable at current settings (ENABLE_AUTO_SCALED_NOISE is
        # False by default) and is written as a live guard rather than an
        # assertion precisely because flipping that flag must not silently
        # re-open the defect.
        if (
            goal_threshold_plan is not None
            and goal_threshold_plan.needs_status_quo_reference
            and auto_noise_applied
        ):
            goal_threshold_plan = None
            goal_threshold_frame_warning = InferenceWarning(
                code="GOAL_THRESHOLD_NOT_CONVERTIBLE",
                field="goal_threshold_frame",
                detail={
                    "goal_node_id": request.goal_node_id,
                    "goal_threshold": request.goal_threshold,
                    "goal_threshold_frame": request.goal_threshold_frame,
                    "reason": "auto_scaled_noise_breaks_status_quo_reference",
                    "message": (
                        "Auto-scaled noise was applied to the option samples but "
                        "not to the status-quo reference they are differenced "
                        "against, so a level threshold cannot be resolved without "
                        "attributing that noise to the option. "
                        "probability_of_goal is omitted."
                    ),
                },
                severity="warning",
            )

        if goal_threshold_frame_warning is not None:
            inference_warnings.append(goal_threshold_frame_warning)

        # ROADMAP 2.798, Channel B's half of the same guard, and deliberately
        # NARROWER than Channel A's because the exposure is narrower.
        # `_apply_auto_scaled_noise` noises the GOAL series only, and
        # `_align_goal_constraint_samples` then copies that noised series into the
        # constraint values for the goal node. Every other constraint target keeps
        # un-noised model samples, so its status-quo reference stays coherent and
        # refusing on it would be over-refusal. Only a level-framed constraint
        # ON THE GOAL NODE loses its reference.
        #
        # Unreachable at current settings (ENABLE_AUTO_SCALED_NOISE is False) and
        # written as a live guard rather than an assertion precisely because
        # flipping that flag must not silently re-open the defect.
        if constraint_plans and auto_noise_applied and request.goal_constraints:
            noise_broken = [
                index
                for index, plan in constraint_plans.items()
                if plan.needs_status_quo_reference
                and request.goal_constraints[index].node_id == request.goal_node_id
            ]
            if noise_broken:
                constraint_plans = None
                for index in noise_broken:
                    constraint = request.goal_constraints[index]
                    inference_warnings.append(
                        InferenceWarning(
                            code="CONSTRAINT_NOT_CONVERTIBLE",
                            field=f"goal_constraints[{index}].value_frame",
                            detail={
                                "constraint_id": constraint.constraint_id,
                                "node_id": constraint.node_id,
                                "operator": constraint.operator,
                                "constraint_value": constraint.value,
                                "value_frame": constraint.value_frame,
                                "reason": "auto_scaled_noise_breaks_status_quo_reference",
                                "message": (
                                    "Auto-scaled noise was applied to the goal "
                                    "samples this constraint is evaluated against, "
                                    "but not to the status-quo reference they are "
                                    "differenced against, so a level value cannot "
                                    "be resolved without attributing that noise to "
                                    "the option. constraint_analysis is omitted."
                                ),
                            },
                            severity="warning",
                        )
                    )

        results = self._compute_option_results(
            option_outcomes,
            option_wins,
            request,
            goal_threshold_plan,
            constraint_node_values,
            pre_noise_expected_regret,
            status_quo_outcomes,
            constraint_plans,
            status_quo_node_values,
            objective_ranking=objective_ranking,
        )

        # Build critiques for analysis warnings
        critiques: List[CritiqueV2] = []

        # Check for zero-variance options (degenerate outcomes)
        # Use tolerance to catch near-zero values from floating point arithmetic
        option_labels = {opt.id: (opt.label or opt.id) for opt in request.options}
        for result in results:
            if result.outcome_distribution.std < ZERO_VARIANCE_TOLERANCE:
                critiques.append(
                    DEGENERATE_OPTION_ZERO_VARIANCE.build(
                        option_label=option_labels.get(result.option_id, result.option_id),
                        affected_option_ids=[result.option_id],
                        seed=seed,
                    )
                )

        # Check for high tie rate
        if tie_rate > 0.5:
            critiques.append(
                HIGH_TIE_RATE.build(
                    tie_rate_pct=int(tie_rate * 100),
                    seed=seed,
                )
            )

        # Add constraint default-base critiques (also surfaced via inference_warnings)
        critiques.extend(constraint_default_base_critiques)

        # Add Cluster-2 goal-node disclosure critiques (also surfaced via
        # inference_warnings)
        critiques.extend(goal_disclosure_critiques)

        # Compute sensitivity if requested
        sensitivity = []
        if "sensitivity" in request.analysis_types:
            sensitivity = self._compute_sensitivity(
                request, option_outcomes, sampler, rng_edge, evaluator
            )

        # B3-S1 (D-23.4) suppression RECORD (not PREDICT): each compute-gate below
        # that skips its block BECAUSE of active correlation appends to this list at
        # the skip site; `_build_correlation_disclosure` emits exactly this record, so
        # the manifest cannot drift from what the run actually withheld (altitude Q2,
        # CLAUDE.md #12). Order follows code (gate) order.
        suppressed_attributions: List[str] = []

        # Compute factor sensitivity if factor uncertainties are specified.
        # B3-S1 (D-23.4): SUPPRESSED under active correlation — per-factor OAT
        # elasticity perturbs one factor holding the others at their mean, an
        # off-manifold move that double-counts shared variance and mis-ranks
        # correlated factors. Omitted (absent, not fabricated) with the
        # correlation_model disclosure marker naming the reason.
        factor_sensitivity: List[FactorSensitivityResult] = []
        if factor_sampler.has_uncertainties() and "sensitivity" in request.analysis_types:
            if correlation_active:
                suppressed_attributions.append(SUPPRESSED_ATTR_FACTOR_SENSITIVITY)
                # stability_thresholds is a CHILD of factor_sensitivity's bootstrap
                # (which runs unconditionally whenever there are uncertainties, so it
                # is emitted iff factor_sensitivity ran). It therefore vanishes with
                # factor_sensitivity under correlation — record it at the SAME skip
                # site so the manifest names it too (hunter F-1: it previously
                # vanished silently, unnamed).
                suppressed_attributions.append(SUPPRESSED_ATTR_STABILITY_THRESHOLDS)
            else:
                factor_sensitivity = self._compute_factor_sensitivity(
                    request, option_outcomes, rng_factor, evaluator, critiques=critiques
                )

        # Compute conditional winners (factor-partitioned win probabilities).
        # B3-S1 (D-23.4): SUPPRESSED under active correlation — a single-factor
        # median split attributes a winner flip to one factor, but under
        # correlation that factor's low/high bucket also drags its correlated
        # partners, so the per-factor attribution is confounded. Omitted with the
        # disclosure marker (joint win_probability itself stays valid).
        conditional_winners = None
        if factor_sampler.has_uncertainties() and len(request.options) > 1:
            # ROADMAP 2.1192: a conditional winner is a statement about WHERE
            # THE WINNER FLIPS. Under a withheld ranking there is no winner to
            # flip, and emitting these would smuggle back through a side channel
            # exactly the ranking this response refused to state. Suppressed with
            # the same disclosure marker correlation already uses — RECORDED, so
            # the omission is visible rather than merely absent.
            if not maximise_metrics_available or recommended_option_id is None:
                objective_suppressed_metrics.append("conditional_winners")
            elif correlation_active:
                suppressed_attributions.append(SUPPRESSED_ATTR_CONDITIONAL_WINNERS)
            else:
                conditional_winners = self._compute_conditional_winners(
                    factor_values_per_sample,
                    winner_per_sample,
                    option_outcomes,
                    factor_sampler,
                    request,
                )

        # Compute robustness assessment (with alternative winner analysis)
        robustness = None
        if maximise_metrics_available and recommended_option_id is not None:
            robustness = self._compute_robustness(
                option_wins,
                winner_per_sample,
                sensitivity,
                request,
                edge_configs_per_sample,
                evaluator,
                seed,
                n_defaulted_roots=len(defaulted_root_node_ids),
                defaulted_root_node_ids=defaulted_root_node_ids,
                critiques=critiques,
            )
        else:
            objective_suppressed_metrics.append("robustness")

        # Compute E-value analogue per edge if requested. OPTIONAL phase —
        # governed by the overall request budget: skipped-with-disclosure when
        # insufficient budget remains, so the base + robustness results above
        # are never lost to a stacked-phase timeout.
        edge_e_values = None
        if request.include_e_values and (
            not maximise_metrics_available or recommended_option_id is None
        ):
            objective_suppressed_metrics.append("edge_e_values")
        elif request.include_e_values:
            remaining_ms = _budget_remaining_ms()
            if remaining_ms < self.OPTIONAL_PHASE_MIN_BUDGET_MS:
                # Not enough budget to run the E-value sweep (and the bands that
                # ride on it) — disclose both on the wire.
                elapsed_ms = _elapsed_ms()
                self.logger.info(
                    "e_value_budget_exceeded",
                    extra={"elapsed_ms": elapsed_ms, "reason": "request_budget_exhausted"},
                )
                inference_warnings.append(
                    self._optional_phase_unavailable_warning(
                        "E_VALUES_UNAVAILABLE",
                        "robustness.edge_e_values",
                        "request_budget_exhausted",
                        elapsed_ms,
                        "E-value analysis was skipped: the request budget was "
                        "exhausted before it could run. Base analysis is unaffected.",
                    )
                )
                inference_warnings.append(
                    self._optional_phase_unavailable_warning(
                        "STABILITY_BANDS_UNAVAILABLE",
                        "robustness.edge_e_values[].stability",
                        "e_values_unavailable",
                        elapsed_ms,
                        "Flip-stability bands were skipped: they ride on the "
                        "E-value sweep, which the request budget could not fund.",
                    )
                )
            else:
                edge_e_values = self._compute_edge_e_values(
                    request, evaluator, budget_ms=min(self.E_VALUE_BUDGET_MS, remaining_ms)
                )
                if edge_e_values is None:
                    # Internal E-value budget tripped mid-sweep — disclose on the
                    # wire (formerly a log-only event) and, since bands ride on
                    # E-values, disclose their absence too.
                    elapsed_ms = _elapsed_ms()
                    inference_warnings.append(
                        self._optional_phase_unavailable_warning(
                            "E_VALUES_UNAVAILABLE",
                            "robustness.edge_e_values",
                            "e_value_budget_exceeded",
                            elapsed_ms,
                            "E-value analysis exceeded its time budget and was "
                            "omitted. Base analysis is unaffected.",
                        )
                    )
                    inference_warnings.append(
                        self._optional_phase_unavailable_warning(
                            "STABILITY_BANDS_UNAVAILABLE",
                            "robustness.edge_e_values[].stability",
                            "e_values_unavailable",
                            elapsed_ms,
                            "Flip-stability bands were omitted: the E-value sweep "
                            "they ride on exceeded its time budget.",
                        )
                    )
                else:
                    # Track S Phase 1: seed-sweep flip-threshold stability bands.
                    # DEFAULT-ON (env gating removed 2026-07-17) and ADDITIVE:
                    # attaches a "stability" object to each edge_e_values entry and
                    # never mutates existing keys. Uses only fresh SHA-256-derived
                    # child RNGs — no shared RNG stream is consumed, so all other
                    # numbers are unchanged by the sweep. All-or-nothing, and
                    # OPTIONAL — gated by the remaining request budget.
                    remaining_ms = _budget_remaining_ms()
                    if remaining_ms < self.OPTIONAL_PHASE_MIN_BUDGET_MS:
                        elapsed_ms = _elapsed_ms()
                        self.logger.info(
                            "flip_stability_budget_exceeded",
                            extra={
                                "elapsed_ms": elapsed_ms,
                                "reason": "request_budget_exhausted",
                            },
                        )
                        inference_warnings.append(
                            self._optional_phase_unavailable_warning(
                                "STABILITY_BANDS_UNAVAILABLE",
                                "robustness.edge_e_values[].stability",
                                "request_budget_exhausted",
                                elapsed_ms,
                                "Flip-stability bands were skipped: the request "
                                "budget was exhausted before the sweep could run.",
                            )
                        )
                    else:
                        bands_attached = self._attach_flip_stability_bands(
                            request,
                            evaluator,
                            edge_e_values,
                            seed,
                            budget_ms=min(self.FLIP_STABILITY_BUDGET_MS, remaining_ms),
                        )
                        if not bands_attached:
                            # Internal band budget tripped — the #226 gap
                            # (log-only) now rides the wire.
                            elapsed_ms = _elapsed_ms()
                            inference_warnings.append(
                                self._optional_phase_unavailable_warning(
                                    "STABILITY_BANDS_UNAVAILABLE",
                                    "robustness.edge_e_values[].stability",
                                    "flip_stability_budget_exceeded",
                                    elapsed_ms,
                                    "Flip-stability bands exceeded their time "
                                    "budget and were omitted (all-or-nothing).",
                                )
                            )

        # ROADMAP 2.228-F3: per-root-factor flip thresholds. OPTIONAL phase,
        # request-gated by include_factor_flips and default OFF, so a consumer
        # that has not opted in sees a byte-identical response. Placed here so it
        # runs on the same epsilon-disabled evaluator the E-value sweep uses (the
        # closed form is only exact once the noise is off) and under the same
        # governing request budget, degrading with disclosure rather than
        # stacking a phase past the deadline.
        factor_flip_values = None
        if request.include_factor_flips and (
            not maximise_metrics_available or recommended_option_id is None
        ):
            objective_suppressed_metrics.append("factor_flip_values")
        elif request.include_factor_flips:
            remaining_ms = _budget_remaining_ms()
            if remaining_ms < self.OPTIONAL_PHASE_MIN_BUDGET_MS:
                elapsed_ms = _elapsed_ms()
                self.logger.info(
                    "factor_flip_budget_exceeded",
                    extra={"elapsed_ms": elapsed_ms, "reason": "request_budget_exhausted"},
                )
                inference_warnings.append(
                    self._optional_phase_unavailable_warning(
                        "FACTOR_FLIPS_UNAVAILABLE",
                        # Top-level on the V2 envelope (like path_decomposition).
                        "factor_flip_values",
                        "request_budget_exhausted",
                        elapsed_ms,
                        "Factor-flip analysis was skipped: the request budget was "
                        "exhausted before it could run. Base analysis is unaffected.",
                    )
                )
            else:
                factor_flip_values = self._compute_factor_flip_values(
                    request,
                    evaluator,
                    seed,
                    budget_ms=min(self.FACTOR_FLIP_BUDGET_MS, remaining_ms),
                )
                if factor_flip_values is None:
                    elapsed_ms = _elapsed_ms()
                    inference_warnings.append(
                        self._optional_phase_unavailable_warning(
                            "FACTOR_FLIPS_UNAVAILABLE",
                            "factor_flip_values",
                            "factor_flip_budget_exceeded",
                            elapsed_ms,
                            "Factor-flip analysis exceeded its time budget and was "
                            "omitted (all-or-nothing). Base analysis is unaffected.",
                        )
                    )

        # Compute the per-factor win-probability sensitivity if requested. OPTIONAL
        # phase — gated at entry AND (Codex F7) governed by an internal wall-clock
        # deadline once started, so it degrades-with-disclosure instead of running
        # unbounded past the budget.
        # B3-S1 (D-23.4): SUPPRESSED under active correlation — this OAT-style
        # win-probability delta fixes one factor at its mean while its correlated
        # partners stay uncertain, an off-manifold move that mis-attributes shared
        # variance. Omitted with the correlation_model disclosure marker. (Note: the
        # NEW factor_evppi below is a conditional-expectation quantity computed on
        # the joint copula samples and is EMITTED under correlation — see that block.)
        # S2 (D-23.8) HONEST RELABEL: this phase produces ``p_win_sensitivity``
        # (was mislabelled ``factor_evpi``) — a decision-held-fixed win-probability
        # delta at each factor's mean, NOT value-of-information. The variable is
        # renamed to match the wire field; the compute (_compute_evpi) and its
        # budget/deadline machinery are unchanged (internal EVPI_* naming retained
        # as an implementation detail). The degradation warning keeps its
        # operational code ("EVPI_UNAVAILABLE") — PLoT surfaces it by SEVERITY, not
        # code (see _optional_phase_unavailable_warning) — but its ``field`` now
        # points at the renamed wire field ``p_win_sensitivity``.
        p_win_sensitivity = None
        if request.include_voi and recommended_option_id is None:
            # ROADMAP 2.1192: this phase measures how much each factor moves THE
            # RECOMMENDED OPTION's win probability. Under a withheld ranking
            # there is no recommended option and no win probability, so the
            # quantity does not exist — it is not merely unavailable. Skipped
            # BEFORE it runs and recorded at the skip site.
            objective_suppressed_metrics.append("p_win_sensitivity")
        elif request.include_voi and factor_sampler.has_uncertainties() and correlation_active:
            # SUPPRESSED under active correlation — record at the skip site.
            suppressed_attributions.append(SUPPRESSED_ATTR_P_WIN_SENSITIVITY)
        elif request.include_voi and request.goal_constraints and constraint_plans is None:
            # ROADMAP 2.798: this phase's metric IS P(joint_goal) when constraints
            # exist. With no resolvable plans there is no honest metric to compute,
            # so the phase is skipped BEFORE it runs and disclosed with its real
            # reason — not silently, and not attributed to a time budget it never
            # spent.
            inference_warnings.append(
                self._optional_phase_unavailable_warning(
                    "EVPI_UNAVAILABLE",
                    "p_win_sensitivity",
                    "constraints_not_convertible",
                    _elapsed_ms(),
                    "Win-probability sensitivity (p_win_sensitivity) was skipped: "
                    "its metric is P(joint_goal) and at least one goal constraint "
                    "could not be resolved into its target's sample frame. Base "
                    "analysis is unaffected.",
                )
            )
        elif request.include_voi and factor_sampler.has_uncertainties():
            assert recommended_option_id is not None
            remaining_ms = _budget_remaining_ms()
            if remaining_ms < self.EVPI_MIN_BUDGET_MS:
                elapsed_ms = _elapsed_ms()
                self.logger.info(
                    "evpi_budget_exceeded",
                    extra={"elapsed_ms": elapsed_ms, "reason": "request_budget_exhausted"},
                )
                inference_warnings.append(
                    self._optional_phase_unavailable_warning(
                        "EVPI_UNAVAILABLE",
                        # F4: p_win_sensitivity is TOP-LEVEL on the V2 envelope, not
                        # nested under robustness.
                        "p_win_sensitivity",
                        "request_budget_exhausted",
                        elapsed_ms,
                        "Win-probability sensitivity (p_win_sensitivity) was "
                        "skipped: insufficient request budget remained. Base "
                        "analysis is unaffected.",
                    )
                )
            else:
                # F7: thread the governing request deadline into the sweep. min(cap,
                # remaining) measured against its own monotonic t0 == the
                # OVERALL_REQUEST_BUDGET_MS deadline (identical maths to the E-value
                # sweep). On overrun _compute_evpi returns None (all-or-nothing).
                p_win_sensitivity = self._compute_evpi(
                    request,
                    sampler,
                    factor_sampler,
                    evaluator,
                    seed,
                    recommended_option_id,
                    budget_ms=min(self.EVPI_BUDGET_MS, remaining_ms),
                    constraint_plans=constraint_plans,
                    objective=objective_plan,
                )
                if p_win_sensitivity is None:
                    # Reachable ONLY as a deadline trip here: the has_uncertainties()
                    # guard guarantees parameter_uncertainties is non-empty, so
                    # _compute_evpi's benign no-uncertainties None is unreachable on
                    # this path. Discard the partial phase and disclose.
                    elapsed_ms = _elapsed_ms()
                    inference_warnings.append(
                        self._optional_phase_unavailable_warning(
                            "EVPI_UNAVAILABLE",
                            "p_win_sensitivity",
                            "evpi_budget_exceeded",
                            elapsed_ms,
                            "Win-probability sensitivity (p_win_sensitivity) "
                            "exceeded its time budget and was omitted "
                            "(all-or-nothing). Base analysis is unaffected.",
                        )
                    )

        # S2 (D-23.8): per-factor EVPPI in OUTCOME units via single-loop
        # Strong-Oakley regression on the RETAINED joint CRN samples — no nested
        # MC, no new sampling. Unlike p_win_sensitivity above, factor_evppi is a
        # conditional-expectation quantity that IS honest under correlation (the
        # retained samples come from the joint copula), so it is EMITTED under
        # active correlation (not gated by `not correlation_active`). Gated only by
        # include_voi + at least one uncertainty. Degrade-with-disclosure on any
        # unexpected estimator failure (never 500s the response).
        factor_evppi = None
        if (
            request.include_voi
            and factor_sampler.has_uncertainties()
            and not maximise_metrics_available
        ):
            objective_suppressed_metrics.append("factor_evppi")
        if (
            maximise_metrics_available
            and request.include_voi
            and factor_sampler.has_uncertainties()
            and pre_noise_option_outcomes is not None
        ):
            # Per-factor EVPPI can never exceed the whole-decision EVPI (learning
            # ONE factor cannot beat learning EVERYTHING). decision_evpi on the
            # pre-noise CRN population = min_o expected_regret[o] = E[max]−max E;
            # this is the exact cap the emission clamps to (with disclosure).
            # ABSENCE, not zero: an option with no finite draw carries a `None`
            # regret and is EXCLUDED from the bound. It used to carry a fabricated
            # 0.0, which passed the `math.isfinite` filter this line used to apply
            # and collapsed the bound to 0.0 — clamping EVERY factor's EVPPI to
            # zero, i.e. reporting that no factor is worth learning about, because
            # one option's draws overflowed. The exclusion rule now lives in the
            # named helper (single statement of the rule, testable in one place).
            decision_evpi_bound = decision_evpi_from_regrets(pre_noise_expected_regret)
            # Per-factor estimator failures degrade IN-LOOP (that factor omitted,
            # computable rows kept — hunter F-4). This outer guard is a last-resort
            # for an unexpected WHOLE-call failure (not a per-factor estimator raise),
            # preserving the never-500 contract; on legal input it is not reached.
            try:
                factor_evppi = self._compute_factor_evppi(
                    request,
                    pre_noise_option_outcomes,
                    factor_values_per_sample,
                    seed,
                    decision_evpi_bound,
                    correlation_active,
                    inference_warnings=inference_warnings,
                )
            except Exception:
                self.logger.warning("factor_evppi_failed", exc_info=True)
                factor_evppi = None
                inference_warnings.append(
                    InferenceWarning(
                        code="FACTOR_EVPPI_UNAVAILABLE",
                        field="factor_evppi",
                        severity="warning",
                        detail={
                            "reason": "estimator_error",
                            "message": (
                                "Per-factor EVPPI could not be computed and was "
                                "omitted. Base analysis is unaffected."
                            ),
                        },
                    )
                )

        # S4 (D-23.8): per-lever value of control (EVPC). Grid do(factor=value) on
        # the SAME retained joint CRN samples via the SAME evaluator used for the
        # options — no nested MC, no new sampling. Request-driven: gated purely on
        # control_candidates presence (NOT include_voi — control is a distinct
        # capability from information). Emitted under active correlation (the do()
        # runs on the joint copula draws). Degrade-with-disclosure on any unexpected
        # failure (never 500s the response).
        factor_evpc = None
        if request.control_candidates and not maximise_metrics_available:
            objective_suppressed_metrics.append("factor_evpc")
        if (
            maximise_metrics_available
            and request.control_candidates
            and pre_noise_option_outcomes is not None
        ):
            try:
                factor_evpc = self._compute_factor_evpc(
                    request,
                    evaluator,
                    edge_configs_per_sample,
                    factor_values_per_sample,
                    pre_noise_option_outcomes,
                    correlation_active,
                )
            except Exception:  # pragma: no cover - defensive degrade-with-disclosure
                self.logger.warning("factor_evpc_failed", exc_info=True)
                factor_evpc = None
                inference_warnings.append(
                    InferenceWarning(
                        code="FACTOR_EVPC_UNAVAILABLE",
                        field="factor_evpc",
                        severity="warning",
                        detail={
                            "reason": "compute_error",
                            "message": (
                                "Value of control (factor_evpc) could not be computed "
                                "and was omitted. Base analysis is unaffected."
                            ),
                        },
                    )
                )

        # Compute structural pathway decomposition for the recommended option if requested.
        # Pass evaluator.graph — the post-filter graph the SCM actually computed on
        # (filter_inference_graph was applied before the evaluator was constructed), so the
        # decomposition explains exactly the structure the analysis used, not raw request.graph.
        path_decomposition = None
        if request.include_path_decomposition and recommended_option_id is None:
            objective_suppressed_metrics.append("path_decomposition")
        elif request.include_path_decomposition:
            assert recommended_option_id is not None
            remaining_ms = _budget_remaining_ms()
            if remaining_ms < self.OPTIONAL_PHASE_MIN_BUDGET_MS:
                elapsed_ms = _elapsed_ms()
                self.logger.info(
                    "path_decomposition_budget_exceeded",
                    extra={"elapsed_ms": elapsed_ms, "reason": "request_budget_exhausted"},
                )
                inference_warnings.append(
                    self._optional_phase_unavailable_warning(
                        "PATH_DECOMPOSITION_UNAVAILABLE",
                        # F4: path_decomposition is TOP-LEVEL on the V2 envelope.
                        "path_decomposition",
                        "request_budget_exhausted",
                        elapsed_ms,
                        "Path decomposition was skipped: insufficient request "
                        "budget remained. Base analysis is unaffected.",
                    )
                )
            else:
                # F7: thread the governing request deadline into path-decomposition
                # (previously path-COUNT capped via MAX_DECOMPOSITION_PATHS but never
                # wall-clock re-checked). On overrun it returns None (all-or-nothing).
                path_decomposition = self._compute_path_decomposition(
                    request,
                    recommended_option_id,
                    evaluator.graph,
                    budget_ms=min(self.PATH_DECOMPOSITION_BUDGET_MS, remaining_ms),
                )
                if path_decomposition is None:
                    # Deadline trip — discard the partial phase and disclose.
                    elapsed_ms = _elapsed_ms()
                    inference_warnings.append(
                        self._optional_phase_unavailable_warning(
                            "PATH_DECOMPOSITION_UNAVAILABLE",
                            "path_decomposition",
                            "path_decomposition_budget_exceeded",
                            elapsed_ms,
                            "Path decomposition exceeded its time budget and was "
                            "omitted (all-or-nothing). Base analysis is unaffected.",
                        )
                    )

        execution_time = int((time.time() - start_time) * 1000)

        # B3-S1: correlation disclosure block (present iff correlation active).
        # Carries the tail-independence caveat, any Higham PSD projection, and the
        # manifest of suppressed independence-assuming per-factor attributions.
        correlation_model = self._build_correlation_disclosure(
            request, correlation_plan, suppressed_attributions
        )

        # Include stability thresholds when bootstrap stability was computed
        has_bootstrap = any(fs.attribution_stability is not None for fs in factor_sensitivity)
        stability_thresholds = (
            StabilityThresholdsResponse(
                high_moderate_boundary=STABILITY_THRESHOLDS.high_moderate_boundary,
                moderate_low_boundary=STABILITY_THRESHOLDS.moderate_low_boundary,
                version=STABILITY_THRESHOLDS.version,
                provisional=STABILITY_THRESHOLDS.provisional,
            )
            if has_bootstrap
            else None
        )

        # ROADMAP 2.720 (2.521 Q1): resolve user-stated ranges to fit-or-refusal
        # disclosures. Pure and RNG-free (zero draws — tested), placed AFTER all
        # sampling so byte-identity of compute is structural as well as tested.
        # Refusal codes ride inference_warnings at severity 'warning' (the
        # degradation-disclosure class): a refusal the user never sees is a
        # silent default with extra steps.
        range_fit_disclosures, range_fit_warnings = resolve_range_fits(request.user_stated_ranges)
        inference_warnings.extend(range_fit_warnings)

        if objective_suppressed_metrics:
            inference_warnings.append(
                InferenceWarning(
                    code="OBJECTIVE_METRICS_UNAVAILABLE",
                    field="objective_ranking",
                    severity="warning",
                    detail={
                        "reason": "objective_not_supported_or_no_unique_leader",
                        "suppressed_fields": sorted(set(objective_suppressed_metrics)),
                        "message": "Some additional comparison measures are unavailable for this objective "
                        "or because there is no single leading option. No substitute recommendation was used.",
                    },
                )
            )

        response = RobustnessResponseV2(
            request_id=request_id,
            results=results,
            recommended_option_id=recommended_option_id,
            recommendation_confidence=recommendation_confidence,
            sensitivity=sensitivity,
            factor_sensitivity=factor_sensitivity,
            robustness=robustness,
            metadata=ResponseMetadataV2(
                isl_version=__version__,
                n_samples_used=request.n_samples,
                seed_used=seed,
                execution_time_ms=execution_time,
                edge_existence_rates=sampler.get_existence_rates(),
                config_fingerprint=generate_config_fingerprint(),
                tie_count=tie_count,
                tie_rate=tie_rate,
                seed_hash_version=SEED_HASH_VERSION,
                auto_noise_applied=auto_noise_applied,
                n_defaulted_root_nodes=len(defaulted_root_node_ids)
                if defaulted_root_node_ids
                else None,
            ),
            critiques=critiques,
            inference_warnings=inference_warnings,
            objective_ranking=objective_ranking,
            conditional_winners=conditional_winners,
            stability_thresholds=stability_thresholds,
            edge_e_values=edge_e_values,
            factor_flip_values=factor_flip_values,
            p_win_sensitivity=p_win_sensitivity,
            factor_evppi=factor_evppi,
            factor_evpc=factor_evpc,
            path_decomposition=path_decomposition,
            correlation_model=correlation_model,
            range_fit_disclosures=range_fit_disclosures,
        )

        self.logger.info(
            "robustness_v2_analysis_complete",
            extra={
                "request_id": request_id,
                "recommended_option": recommended_option_id,
                "recommendation_confidence": recommendation_confidence,
                "is_robust": robustness.is_robust if robustness is not None else None,
                "execution_time_ms": execution_time,
                "conditional_winners_count": len(conditional_winners) if conditional_winners else 0,
            },
        )

        return response

    @staticmethod
    def _build_correlation_plan(
        request: RobustnessRequestV2,
    ) -> Optional[CorrelationPlan]:
        """Build the Gaussian-copula plan from request.factor_correlations (B3-S1).

        Returns None when no correlations are supplied — inert-when-absent. The
        canonical factor order is first-appearance in ``parameter_uncertainties``,
        the same order ``FactorSampler``'s independent loop draws in, so an
        all-normal request whose correlated set spans the full uncertainty list
        with rho=0 reproduces the independent draws bit-for-bit. Request
        validation has already rejected every hard-invalid input (F4, D-23.13), so
        the assembled matrix is at worst NEAR-PSD (PSD-checked + Higham-projected
        inside ``build_correlation_plan``).

        The (factor_order, pairs) derivation is the request model's OWN
        ``_correlation_matrix_inputs`` — the SAME derivation the admissibility gate
        validates against, so the two can never drift (derive-don't-mirror).
        """
        correlations = request.factor_correlations
        if not correlations:
            return None
        factor_order, pairs = request._correlation_matrix_inputs()
        return build_correlation_plan(factor_order, pairs)

    @staticmethod
    def _build_correlation_disclosure(
        request: RobustnessRequestV2,
        correlation_plan: Optional[CorrelationPlan],
        suppressed_attributions: List[str],
    ) -> Optional[CorrelationModelV2]:
        """Assemble the ``correlation_model`` disclosure block (B3-S1, D-23.4).

        Returns None when correlation is inactive. When active it carries the
        method tag, the MANDATORY tail-independence caveat, any Higham PSD
        projection, and the manifest of suppressed independence-assuming per-
        factor attributions.

        RECORD, not PREDICT (altitude Q2): ``suppressed_attributions`` is the list
        the compute path APPENDED to at each skip site where a block was withheld
        because of active correlation — it is the ground truth of what the run
        actually skipped, not a second precondition-forecast that could drift from
        the gates (CLAUDE.md #12). This includes ``stability_thresholds`` (hunter
        F-1), which rides on factor_sensitivity's bootstrap and previously vanished
        unnamed.
        """
        if correlation_plan is None:
            return None

        projection = correlation_plan.projection
        psd_projection = (
            CorrelationProjectionV2(
                applied=projection.applied,
                method=projection.method,
                frobenius_distance=projection.frobenius_distance,
                max_abs_off_diagonal_adjustment=projection.max_abs_off_diagonal_adjustment,
                iterations=projection.iterations,
                # F4: disclose the EFFECTIVE adjusted correlations, not only the
                # aggregate distance, so a caller can reconstruct what drove the numbers.
                effective_correlations=[
                    EffectiveCorrelationV2(
                        factor_a=ep.factor_a,
                        factor_b=ep.factor_b,
                        requested_rho=ep.requested_rho,
                        effective_rho=ep.effective_rho,
                        adjustment=ep.adjustment,
                        stated=ep.stated,
                    )
                    for ep in projection.effective_pairs
                ],
            )
            if projection is not None
            else None
        )

        return CorrelationModelV2(
            method=CORRELATION_METHOD,
            active=True,
            correlated_factors=list(correlation_plan.factor_order),
            n_pairs=len(request.factor_correlations or []),
            tail_dependence="none",
            tail_dependence_note=_CORRELATION_TAIL_NOTE,
            psd_projection=psd_projection,
            suppressed_attributions=list(suppressed_attributions),
            suppression_reason=_CORRELATION_SUPPRESSION_REASON,
        )

    def _run_monte_carlo(
        self,
        request: RobustnessRequestV2,
        sampler: DualUncertaintySampler,
        factor_sampler: FactorSampler,
        evaluator: SCMEvaluatorV2,
        constraint_target_nodes: Optional[List[str]] = None,
        status_quo_reference_nodes: Optional[List[str]] = None,
        objective: Optional["ObjectivePlan"] = None,
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, float],
        List[Optional[str]],
        List[Dict[Tuple[str, str], float]],
        int,
        Optional[Dict[str, Dict[str, List[float]]]],
        List[Dict[str, float]],
        Dict[str, List[float]],
    ]:
        """
        Run Monte Carlo simulation with dual edge uncertainty and factor uncertainty.

        Args:
            request: The robustness analysis request
            sampler: Edge configuration sampler
            factor_sampler: Factor value sampler
            evaluator: SCM evaluator
            constraint_target_nodes: Optional list of node IDs to track for constraint analysis

        Returns:
            Tuple of:
            - option_outcomes: Dict[option_id, List[outcome_value]]
            - option_wins: Dict[option_id, win_count] (float for tie splitting)
            - winner_per_sample: List of winning option ID per sample
            - edge_configs_per_sample: Edge configurations per sample
            - tie_count: Number of samples with ties
            - constraint_node_values: Dict[option_id, Dict[node_id, List[value]]] or None
            - factor_values_per_sample: List of sampled factor value dicts per MC iteration
            - status_quo_node_values: Dict[node_id, List[value]] — per-draw node
              values with NO interventions, on the same edge/factor draws as every
              option (ROADMAP 2.286 / 2.798). Empty dict unless
              status_quo_reference_nodes is set.

              ROADMAP 2.798 widened this from the goal's single list to a
              per-node dict, because the CONSTRAINT channel needs the same
              reference for its own target nodes. One structure for one concept —
              a goal-shaped special case beside a constraint-shaped one is how
              two dialects of the same idea start.

        Note: option_wins uses float to support split-tie handling where ties are
        divided equally among tied options.
        """
        option_outcomes: Dict[str, List[float]] = {opt.id: [] for opt in request.options}
        option_wins: Dict[str, float] = {opt.id: 0.0 for opt in request.options}
        # 2.477(c): Optional — None marks a draw where no option was finite.
        winner_per_sample: List[Optional[str]] = []
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]] = []
        factor_values_per_sample: List[Dict[str, float]] = []
        tie_count = 0
        # Representative tie labels are bookkeeping, never scientific draws.
        # Their frequency depends on the objective, so consuming the edge RNG
        # here would let an objective change alter later raw model samples.
        # Streams seed..seed+3 already own edge/factor/output/epsilon sampling.
        tie_rng = SeededRNG(sampler.rng.seed + 4)
        # Epsilon is drawn independently for each evaluated option. Bind that
        # existing stream's allocation to option identity, not request order,
        # so rearranging options cannot change their samples or comparison.
        # Keep the zero-epsilon schedule untouched for historical parity.
        evaluation_options = request.options
        if evaluator._epsilon_rng is not None and any(
            node.epsilon_std > 0 for node in request.graph.nodes
        ):
            evaluation_options = sorted(request.options, key=lambda option: option.id)

        # Initialize constraint node values tracking if needed
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]] = None
        if constraint_target_nodes:
            constraint_node_values = {
                opt.id: {node_id: [] for node_id in constraint_target_nodes}
                for opt in request.options
            }

        # ROADMAP 2.286: the per-draw status-quo REFERENCE for a level-frame
        # goal_threshold. Built only when a plan actually needs it, so every
        # other request path does strictly the same work it did before.
        #
        # It gets its OWN evaluator with NO epsilon RNG, which is what keeps this
        # addition invisible to everything else: the shared `evaluator` would
        # consume draws from the epsilon stream and shift every subsequent
        # sample, silently changing results across the repo for a feature that
        # only one field asked for. Passing epsilon_rng=None cannot lose fidelity
        # HERE because the resolver refuses outright when any node that can reach
        # the goal carries epsilon_std > 0 — so on every request that gets this
        # far, the goal's value is epsilon-free by construction. The two facts
        # are load-bearing on each other; changing either without the other
        # re-opens the defect.
        #
        # ROADMAP 2.798: the reference is drawn with `evaluate_multi` over every
        # node that needs one. That is byte-safe for the goal: `evaluate` and
        # `evaluate_multi` share the same seeding, the same topological node
        # order and the same arithmetic (verified at the bytes), differing only in
        # what they RETURN — and with epsilon_rng None neither consumes a draw, so
        # widening the set cannot shift any stream.
        status_quo_node_values: Dict[str, List[float]] = {
            node_id: [] for node_id in (status_quo_reference_nodes or [])
        }
        sq_evaluator = SCMEvaluatorV2(request.graph) if status_quo_reference_nodes else None

        for _ in range(request.n_samples):
            # Sample edge configuration (structural + parametric uncertainty)
            edge_config = sampler.sample_edge_configuration()

            # Sample factor values (parameter uncertainty)
            factor_values = factor_sampler.sample_factor_values()
            factor_values_per_sample.append(factor_values)

            # The reference draw: the SAME edge strengths and factor values as
            # every option below (common random numbers), with no interventions.
            # Differencing against this is what turns an absolute propagated sum
            # back into the option's causal effect.
            if sq_evaluator is not None:
                assert status_quo_reference_nodes is not None
                reference_values = sq_evaluator.evaluate_multi(
                    edge_strengths=edge_config,
                    interventions={},
                    target_nodes=status_quo_reference_nodes,
                    factor_values=factor_values,
                )
                for node_id in status_quo_reference_nodes:
                    status_quo_node_values[node_id].append(reference_values[node_id])

            # Evaluate each option
            sample_outcomes = {}
            for option in evaluation_options:
                if constraint_target_nodes:
                    # Use evaluate_multi to get both goal and constraint node values
                    all_target_nodes = list(set([request.goal_node_id] + constraint_target_nodes))
                    node_values = evaluator.evaluate_multi(
                        edge_strengths=edge_config,
                        interventions=option.interventions,
                        target_nodes=all_target_nodes,
                        factor_values=factor_values,
                    )
                    outcome = node_values.get(request.goal_node_id, 0.0)
                    # Store constraint node values
                    assert constraint_node_values is not None
                    for node_id in constraint_target_nodes:
                        constraint_node_values[option.id][node_id].append(
                            node_values.get(node_id, 0.0)
                        )
                else:
                    # Standard evaluation for goal node only
                    outcome = evaluator.evaluate(
                        edge_strengths=edge_config,
                        interventions=option.interventions,
                        goal_node=request.goal_node_id,
                        factor_values=factor_values,
                    )

                option_outcomes[option.id].append(outcome)
                sample_outcomes[option.id] = outcome

            # Track winner with fair tie-breaking (split ties equally).
            #
            # 2.477(c): compare only the options that are FINITE at THIS sample.
            # Two defects fixed here, both executed at pristine:
            #   * a sample where EVERY option is non-finite produced
            #     ``winners == []`` (NaN != NaN, so nothing equals the max) and
            #     then ``1.0 / len(winners)`` → ZeroDivisionError → HTTP 500
            #     (witnessed at robustness_analyzer_v2.py:2923);
            #   * more insidiously, a single NaN could BEAT the field: ``max()``
            #     short-circuits on NaN comparisons, so one poisoned option could
            #     make ``max_outcome`` NaN and suppress the genuine winner of an
            #     otherwise perfectly good sample.
            # Restricting to the finite subset is the same per-index convention
            # ``expected_regret_per_option`` already uses (best_i taken over the
            # options finite at i) — alignment stays by index, nothing is
            # inpainted or reordered. A sample with no finite option is
            # UNINFORMATIVE: no option wins it, and it is not a tie either.
            finite_outcomes = {
                opt_id: val for opt_id, val in sample_outcomes.items() if math.isfinite(val)
            }

            # ROADMAP 2.1192 — the winner decision, delegated to its ONE owner.
            #
            # This block used to be an unconditional ``max()``. It is now the
            # SAME single decision, taken by ``_winners_for_draw`` and
            # parameterised by the request's attested objective sense. Nothing
            # downstream changed: win_probability, recommended_option_id,
            # recommendation_confidence, conditional_winners and
            # p_win_sensitivity all still read this one rule's output, which is
            # exactly why the fix belongs here and not in a rival scorer beside
            # it.
            #
            # Explicit maximise preserves the historical comparison. A caller
            # without an objective plan withholds; it cannot revive a default.
            #
            # The tie / finiteness / no-winner semantics below are UNTOUCHED and
            # still operate on whatever the owner returns.
            plan = objective or ObjectivePlan(
                sense="withheld", attested=False, withheld_reason="goal_direction_absent"
            )
            winners = self._winners_for_draw(
                finite_outcomes,
                plan,
                (
                    reference_values.get(request.goal_node_id)
                    if sq_evaluator is not None and plan.needs_status_quo_reference
                    else None
                ),
            )

            if not winners:
                # No option produced a finite outcome at this draw. Award no
                # win (win_probabilities then sum to the informative fraction,
                # which is the honest report) and record the absence — never a
                # fabricated winner. ``None`` rather than a sentinel string so
                # mypy forces every consumer to handle it (trap 12: the type
                # checker is the completeness check a hand-kept list is not).
                winner_per_sample.append(None)
            elif len(winners) == 1:
                # Clear winner
                option_wins[winners[0]] += 1.0
                winner_per_sample.append(winners[0])
            else:
                # Tie: split win equally among tied options
                tie_count += 1
                split_value = 1.0 / len(winners)
                for winner in winners:
                    option_wins[winner] += split_value
                # This representative is separate from exact fractional credit.
                # Stable ID order avoids request-order-dependent tie labels.
                winner_per_sample.append(str(tie_rng.choice(sorted(winners))))

            # Store edge config for alternative winner analysis
            edge_configs_per_sample.append(edge_config)

        return (
            option_outcomes,
            option_wins,
            winner_per_sample,
            edge_configs_per_sample,
            tie_count,
            constraint_node_values,
            factor_values_per_sample,
            status_quo_node_values,
        )

    @staticmethod
    def _defaulted_roots_reaching(
        target_node_id: str,
        children_map: Dict[str, List[str]],
        defaulted_root_node_ids: List[str],
        fully_intervened_node_ids: set,
    ) -> List[str]:
        """
        Root nodes that defaulted to 0.0 AND can influence the target's samples.

        A defaulted root reaches the target iff there is a directed path
        root -> ... -> target where no node after the root is intervened on by
        EVERY option (an all-options intervention overrides the structural
        equation in every sample, severing upstream influence at that node).

        Nodes intervened by only SOME options still pass influence — the
        default leaks into the remaining options' samples, so the root counts.

        Returns:
            Sorted list of defaulted root node IDs with an unblocked directed
            path to the target.
        """
        reaching: List[str] = []
        for root_id in defaulted_root_node_ids:
            if root_id in fully_intervened_node_ids:
                # Defensive: such roots are excluded from the defaulted list.
                continue
            stack = [root_id]
            seen = {root_id}
            found = False
            while stack and not found:
                current = stack.pop()
                for child in children_map.get(current, []):
                    if child in seen:
                        continue
                    seen.add(child)
                    if child in fully_intervened_node_ids:
                        # Blocked: overridden in every option's samples.
                        continue
                    if child == target_node_id:
                        found = True
                        break
                    stack.append(child)
            if found:
                reaching.append(root_id)
        return sorted(reaching)

    def _build_goal_node_disclosures(
        self,
        request: RobustnessRequestV2,
        parent_map: Dict[str, List[str]],
        children_map: Dict[str, List[str]],
        uncertainty_node_ids: set,
        fully_intervened_node_ids: set,
        defaulted_root_node_ids: List[str],
        seed: int,
        goal_baseline_consumed: bool,
    ) -> Tuple[List[InferenceWarning], List[CritiqueV2]]:
        """
        Cluster-2 (Track S Phase 0): disclose the goal node's base/propagation
        semantics. Disclosure-only — never changes any sampled value.

        Doctrine B (ratified, PLoT #204): a non-root goal node's distribution
        is the forward-propagated composition of its parents; goal-fit is
        scored from that distribution. The SCM evaluator already implements
        this (non-root base offset = 0.0 + parent contributions). What was
        previously SILENT, and is disclosed here:

        - GOAL_OBSERVED_VALUE_UNUSED: observed_state.value on a non-root goal
          is not used as a base (only root nodes consult it). SUPPRESSED when
          ``goal_baseline_consumed`` (ROADMAP 2.279): if this same run's
          goal-threshold frame conversion read the goal's
          ``observed_state.baseline``, the observed_state DID do a job and the
          warning has nothing actionable to say — a warning that fires on
          every successful analysis is a broken alarm. It keeps firing on
          every run where the conversion did not consume the baseline (no
          threshold, 'delta' frame, unstamped frame, or any convertibility
          refusal), which is precisely the case a user can act on.
        - GOAL_PU_BASE_ADDITIVE: a ParameterUncertainty entry on a non-root
          goal draws a per-sample base that is ADDED to parent propagation —
          it shifts the distribution, it does not pin the goal's value.
        - GOAL_ANCESTOR_DATA_GAP (+ critique): the propagated distribution
          rests partly on root ancestors that defaulted to 0.0 — honest
          "insufficient data", disclosed rather than fabricated.
        - GOAL_NODE_ROOT_STATIC: a root goal without ParameterUncertainty or
          epsilon noise is a constant; options cannot differ through it
          unless they intervene on it directly.
        """
        warnings: List[InferenceWarning] = []
        critiques: List[CritiqueV2] = []
        goal_id = request.goal_node_id
        goal_node = next((n for n in request.graph.nodes if n.id == goal_id), None)
        if goal_node is None:
            return warnings, critiques
        if goal_id in fully_intervened_node_ids:
            # Every option pins the goal directly; no base/propagation
            # semantics applies to any sample.
            return warnings, critiques

        goal_is_root = len(parent_map.get(goal_id, [])) == 0
        goal_has_pu = goal_id in uncertainty_node_ids
        observed_value = (
            goal_node.observed_state.value if goal_node.observed_state is not None else None
        )

        if goal_is_root:
            if not goal_has_pu and goal_node.epsilon_std == 0:
                base_value = observed_value if observed_value is not None else 0.0
                value_defaulted = observed_value is None
                warnings.append(
                    InferenceWarning(
                        code="GOAL_NODE_ROOT_STATIC",
                        field=f"nodes[{goal_id}]",
                        detail={
                            "node_id": goal_id,
                            "base_value": base_value,
                            "value_defaulted": value_defaulted,
                            "message": (
                                f"Goal node '{goal_id}' has no parents; with no "
                                f"ParameterUncertainty and epsilon_std=0 its samples "
                                f"are the constant base {base_value}"
                                + (
                                    " (defaulted to 0.0 — no observed value)"
                                    if value_defaulted
                                    else ""
                                )
                                + ". Options cannot differ through this goal unless "
                                "they intervene on it directly."
                            ),
                        },
                    )
                )
            return warnings, critiques

        # Non-root goal: distribution = forward-propagated composition of
        # parents (doctrine B).
        if goal_has_pu:
            warnings.append(
                InferenceWarning(
                    code="GOAL_PU_BASE_ADDITIVE",
                    field=f"parameter_uncertainties[{goal_id}]",
                    detail={
                        "node_id": goal_id,
                        "pu_mean_source": (
                            "observed_state.value" if observed_value is not None else "default_0.0"
                        ),
                        "message": (
                            f"Goal node '{goal_id}' is non-root and has a "
                            f"ParameterUncertainty entry: each sample draws a base "
                            f"from that distribution and the parents' propagated "
                            f"contribution is added on top. The goal's distribution "
                            f"is shifted by the sampled base — it is not pinned to it."
                        ),
                    },
                )
            )
        elif observed_value is not None and not goal_baseline_consumed:
            warnings.append(
                InferenceWarning(
                    code="GOAL_OBSERVED_VALUE_UNUSED",
                    field=f"nodes[{goal_id}].observed_state.value",
                    detail={
                        "node_id": goal_id,
                        "observed_value": observed_value,
                        "reason": "non_root_goal_forward_propagation",
                        "message": (
                            f"Goal node '{goal_id}' is non-root; its distribution is "
                            f"the forward-propagated composition of its parents "
                            f"(doctrine B). The supplied observed_state.value="
                            f"{observed_value} is not used as a base for the goal's "
                            f"samples. Use the node's intercept for a fixed exogenous "
                            f"offset."
                        ),
                    },
                )
            )

        ancestor_data_gap = self._defaulted_roots_reaching(
            goal_id, children_map, defaulted_root_node_ids, fully_intervened_node_ids
        )
        if ancestor_data_gap:
            gap_list = ", ".join(f"'{r}'" for r in ancestor_data_gap)
            warnings.append(
                InferenceWarning(
                    code="GOAL_ANCESTOR_DATA_GAP",
                    field=f"nodes[{goal_id}]",
                    detail={
                        "node_id": goal_id,
                        "unsupported_root_ancestors": ancestor_data_gap,
                        "message": (
                            f"Goal node '{goal_id}' is scored from its "
                            f"forward-propagated outcome distribution, but root "
                            f"ancestor(s) {gap_list} carry no observed value or "
                            f"ParameterUncertainty and defaulted to 0.0 — goal-level "
                            f"probabilities partially rest on placeholder zeros "
                            f"(insufficient data)."
                        ),
                    },
                )
            )
            critiques.append(
                GOAL_ANCESTOR_DATA_GAP.build(
                    node_id=goal_id,
                    gap_roots=gap_list,
                    affected_node_ids=[goal_id],
                    seed=seed,
                )
            )
        return warnings, critiques

    @staticmethod
    def _stability_confidence_figure(recommendation_stability: float, n_samples: int) -> float:
        """The figure served in the ``confidence`` slot.

        Arch step 1 (2026-07-26). It was::

            confidence = min(0.99, recommendation_stability * (1 - 1 / np.sqrt(n_samples)))

        and it was rendered to the user as "ROBUST with {confidence:.0%}
        confidence". Nothing calibrated it — no coverage study, no Brier score,
        no validation against realised outcomes. (Positive control that this
        codebase does document calibration when it has it: the admission ceiling
        at the top of this module cites ``benchmarks/admission_calibration.py``,
        and ``FactorSensitivityV2.confidence`` ships a mandatory
        ``ConfidenceProvenance`` marker with a ``calibrated`` boolean. Filtering
        this file's calibration references for any that touch THIS formula
        returns one hit, and it only warns that changing the noise constant
        would need re-validation.)

        The ``(1 - 1/sqrt(n))`` term made it worse than merely uncalibrated: it
        is a monotone function of sample COUNT alone, so the number moved with
        how long the simulation ran rather than with the estimator's actual
        sampling error, or with whether the recommendation was right. At n=1000
        it contributed a fixed 0.968 factor to every response.

        So the shrinkage and the 0.99 cap are withdrawn and the slot carries the
        stability fraction itself: the share of sampled scenarios in which the
        recommended option won. That is a real, measured quantity. The field
        NAME is kept because it is a published contract slot on two live
        response shapes (``RobustnessResult.confidence`` on the
        response_version=1 body and ``RobustnessResultV2.confidence`` on the
        ISLResponseV2 envelope); ``confidence_basis`` rides beside it so a
        consumer can branch on the semantics rather than infer them, and the
        field descriptions now deny the confidence reading outright.

        ``n_samples`` is retained in the signature: it is what a genuine
        confidence figure would need, and keeping it marks the place a
        calibrated estimator would plug in.
        """
        del n_samples  # deliberately unused — see docstring
        return recommendation_stability

    @staticmethod
    def _build_robustness_interpretation(
        is_robust: bool,
        recommendation_stability: float,
        most_frequent_winner: str,
        fragile_edges: List[str],
    ) -> str:
        """Human-readable robustness summary.

        Arch step 1 (2026-07-26): the ROBUST branches no longer open with
        "ROBUST with {confidence:.0%} confidence". They state the measured
        scenario-win share, which is what the analysis actually established.
        The three verdict bands are unchanged.
        """
        if is_robust:
            base = (
                f"Recommendation is ROBUST. {most_frequent_winner} wins in "
                f"{recommendation_stability:.0%} of sampled scenarios."
            )
            if fragile_edges:
                plural = "s" if len(fragile_edges) > 1 else ""
                return (
                    f"{base} ({len(fragile_edges)} sensitive edge{plural} identified: "
                    f"{', '.join(fragile_edges[:3])})"
                )
            return base
        if recommendation_stability >= 0.5:
            return (
                f"Recommendation is MODERATELY ROBUST. {most_frequent_winner} wins in "
                f"{recommendation_stability:.0%} of sampled scenarios, "
                f"but is sensitive to: {', '.join(fragile_edges[:3])}"
            )
        return (
            f"Recommendation is FRAGILE. No clear winner - best option wins in only "
            f"{recommendation_stability:.0%} of sampled scenarios. "
            f"High sensitivity to: {', '.join(fragile_edges[:3])}"
        )

    def _apply_auto_scaled_noise(
        self,
        option_outcomes: Dict[str, List[float]],
        goal_node_id: str,
        graph_nodes: List,
        rng: "SeededRNG",
        noise_multiplier: float = 1.0,
        enabled: Optional[bool] = None,
    ) -> Tuple[Dict[str, List[float]], bool]:
        """
        Apply auto-scaled noise to outcome/risk node samples.

        DEFAULT OFF since arch step 1 (2026-07-26) — see ``enabled`` below.

        What: Adds independent noise ~ N(0, outcome_std) to each MC sample for
        outcome and risk nodes, where outcome_std is the standard deviation of
        the model-driven samples before noise.

        Why: Represents unexplained variance not captured by the structural causal
        model (measurement error, omitted variables, unmodelled interactions).
        Without this, the outcome distributions reflect only the uncertainty from
        edges and factor priors — which underestimates real-world uncertainty.

        Impact: var(X + N) = var(X) + var(N) ≈ 2·var(X) when var(N) = var(X),
        so p10/p90 spread is approximately √2 wider than the purely model-driven
        distribution. This affects P(goal), constraint satisfaction probabilities,
        and robustness assessments.

        Status: PoC heuristic (Neil Bramley). Pending formal review and calibration
        against pilot outcome data. The noise scale (1× model std) is a deliberate
        choice — changing it affects ALL downstream percentile and confidence
        computations and requires re-validation of the calibration suite.

        Per Neil Bramley's heuristic: "Match unexplained noise to explained variance"
        - Only outcome and risk nodes receive noise
        - Noise std = std(samples) from the model
        - If std = 0, skip noise entirely (no model uncertainty)

        Args:
            option_outcomes: Dict of option_id -> list of outcome samples
            goal_node_id: The goal node being measured
            graph_nodes: List of graph nodes to check node kind
            rng: Seeded RNG for determinism
            noise_multiplier: Scale factor for noise std (default 1.0).
                0.0 disables noise entirely.  Used by calibration diagnostics
                to compare different noise levels without changing the API.
            enabled: Master switch. ``None`` (the default) reads
                ``settings.ENABLE_AUTO_SCALED_NOISE``, which is ``False``.
                Arch step 1 (2026-07-26): this heuristic is uncalibrated by its
                own admission and had no request-side switch, so every client
                received ~√2-wider intervals it could not decline. Pass
                ``enabled=True`` explicitly for calibration diagnostics.

        Returns:
            Tuple of (modified option_outcomes, noise_applied flag)
        """
        # Master switch (arch step 1): default-off unless the deployment opts in.
        if enabled is None:
            enabled = get_settings().ENABLE_AUTO_SCALED_NOISE
        if not enabled:
            return option_outcomes, False

        # Find the goal node and check its kind
        goal_node = None
        for node in graph_nodes:
            if node.id == goal_node_id:
                goal_node = node
                break

        if goal_node is None:
            return option_outcomes, False

        # Only apply noise to outcome and risk nodes
        node_kind = getattr(goal_node, "kind", "").lower()
        if node_kind not in ("outcome", "risk"):
            return option_outcomes, False

        # noise_multiplier=0 disables noise entirely
        if noise_multiplier <= 0:
            return option_outcomes, False

        # Apply noise to each option's samples
        any_noise_added = False
        for option_id, samples in option_outcomes.items():
            if not samples:
                continue

            samples_array = np.array(samples)

            # Compute the noise std over the FINITE samples only. Over the whole
            # array a single non-finite sample makes np.std → nan/inf, and
            # rng.normal(0, nan) then poisons EVERY sample — destroying the option's
            # finite majority too (all-nan → downside omitted, mean non-finite).
            # Masking (mirroring the downside percentile family, which filters to
            # finite_cleaned) keeps the std honest; non-finite entries are left
            # as-is for the downstream finite-mask machinery (n_valid_samples,
            # finite_cleaned percentiles, per-index downside regret).
            finite_mask = np.isfinite(samples_array)
            finite_samples = samples_array[finite_mask]
            if finite_samples.size == 0:
                continue  # no finite sample to scale noise from
            outcome_std = float(np.std(finite_samples))

            # Variance OVERFLOW of finite samples (hunter F-3): np.std squares the
            # values, so an all-FINITE population of magnitude >= ~1.4e154 makes
            # outcome_std inf (or nan). rng.normal(0, inf) is then nan and would
            # DESTROY every finite sample the mask exists to protect — mean nan →
            # serializer 500 on a legal, otherwise-200 request. There is no
            # representable noise scale here, so treat it exactly like the
            # no-finite-sample case: skip noise for this option (honest degrade,
            # finite samples pass through). The all-finite normal path has a finite
            # std, so this guard is a no-op there → goldens are byte-identical.
            if not math.isfinite(outcome_std):
                continue

            # If std ≈ 0, skip noise (no model uncertainty to match).
            # Tolerance handles floating-point noise from identical intervention values.
            if outcome_std <= 1e-12:
                continue

            # Deliberate modelling choice (Neil Bramley heuristic):
            # Add noise ~ N(0, outcome_std) to each sample.
            # Mathematical effect: var(X + N) = var(X) + var(N) ≈ 2·var(X) when
            # var(N) = var(X), so the p10/p90 spread is approximately √2 wider than
            # the purely model-driven distribution.
            # Rationale: the noise term represents unexplained variance not captured
            # by the structural causal model (measurement error, omitted variables, etc.).
            # This makes the outcome intervals more conservative (wider) which is the
            # correct direction under uncertainty.
            # WARNING: Changing this constant factor affects ALL downstream percentile
            # and confidence computations.  Any change requires re-validation of the
            # full calibration suite.
            noise = np.array(
                [rng.normal(0, outcome_std * noise_multiplier) for _ in range(len(samples))]
            )
            # Add noise to the FINITE samples only; leave non-finite entries as-is.
            # The RNG draw count is unchanged (len(samples)), and on the all-finite
            # path (the norm) finite_mask is all-True so this is byte-identical to
            # samples_array + noise — goldens are unmoved.
            noised = samples_array.copy()
            noised[finite_mask] = samples_array[finite_mask] + noise[finite_mask]
            option_outcomes[option_id] = noised.tolist()
            any_noise_added = True

        return option_outcomes, any_noise_added

    def _align_goal_constraint_samples(
        self,
        constraint_node_values: Dict[str, Dict[str, List[float]]],
        option_outcomes: Dict[str, List[float]],
        goal_node_id: str,
        auto_noise_applied: bool,
    ) -> List[str]:
        """
        Make goal-node constraint samples identical to the goal samples.

        The MC loop records constraint node values BEFORE auto-scaled noise is
        applied to the goal series. When the goal node itself is a constraint
        target, that leaves two different sample sets answering the same
        question: probability_of_goal (noised) vs the constraint's
        prob_satisfied (un-noised). This method replaces the goal node's
        constraint series with the exact goal series (noised or not), so both
        probabilities are computed from identical samples.

        Mutates constraint_node_values in place.

        Returns:
            Sorted list of non-goal constraint node IDs whose samples remain
            un-noised while the goal series was noised (empty when the goal
            received no noise). Callers surface this as an inference warning
            so the mixed variance semantics is disclosed, not silent.
        """
        mixed_nodes: set = set()
        for option_id, node_series in constraint_node_values.items():
            for node_id in node_series:
                if node_id == goal_node_id:
                    # Copy so later mutation of one list cannot alias the other.
                    node_series[node_id] = list(option_outcomes[option_id])
                elif auto_noise_applied:
                    mixed_nodes.add(node_id)
        return sorted(mixed_nodes)

    @staticmethod
    def _goal_baseline_was_consumed(
        request: RobustnessRequestV2,
        goal_threshold_plan: Optional["GoalThresholdPlan"],
    ) -> bool:
        """Did THIS run's frame conversion actually READ the goal's ``observed_state``?

        Derived from the resolver's OUTCOME, not re-implemented from its
        preconditions — there is no second copy of the convertibility rules here
        to drift out of sync with the resolver.

        ``_resolve_goal_threshold_in_sample_frame`` reads
        ``observed_state.baseline`` on exactly one limb: ``frame == 'level'``
        carried through to completion. Every other exit leaves the
        observed_state untouched — no threshold returns ``(None, None)``,
        ``'delta'`` returns the caller's number unconverted before the goal node
        is even looked up, an unstamped frame refuses, and each convertibility
        refusal (root goal, PU on the goal, goal pinned by an intervention,
        missing baseline, non-finite or out-of-domain operands, epsilon reaching
        the goal) returns ``None``. So ``plan is not None AND frame == 'level'``
        is exact evidence that the baseline was consumed.

        ROADMAP 2.279 uses this to suppress GOAL_OBSERVED_VALUE_UNUSED on runs
        where the observed_state did its job. The frame Literal is enumerated by
        a test, so adding a third frame value REDs rather than silently landing
        on the wrong side of this predicate.
        """
        return goal_threshold_plan is not None and (request.goal_threshold_frame == "level")

    @staticmethod
    def _build_objective_ranking(
        request: RobustnessRequestV2,
        objective: ObjectivePlan,
        option_wins: Dict[str, float],
    ) -> ObjectiveRanking:
        reason = objective.withheld_reason
        if objective.sense == "withheld":
            reason = reason or "target_not_resolvable_in_sample_frame"
        elif not any(credit > 0 for credit in option_wins.values()):
            reason = "no_informative_draws"
        if reason is not None:
            return ObjectiveRanking(
                direction=request.goal_direction,
                attested=objective.attested,
                status="withheld",
                withheld_reason=reason,
                ranked_options=[],
            )
        rows = []
        previous_credit: Optional[float] = None
        rank = 0
        for option_id, credit in sorted(option_wins.items(), key=lambda item: (-item[1], item[0])):
            if previous_credit is None or credit != previous_credit:
                rank += 1
            rows.append(
                ObjectiveRankedOption(
                    option_id=option_id,
                    rank=rank,
                    win_probability=credit / request.n_samples,
                )
            )
            previous_credit = credit
        return ObjectiveRanking(
            direction=request.goal_direction,
            attested=objective.attested,
            status="computed",
            ranked_options=rows,
        )

    @staticmethod
    def _winners_for_draw(
        finite_outcomes: Dict[str, float],
        objective: "ObjectivePlan",
        status_quo_reference: Optional[float],
    ) -> List[str]:
        """Which options win THIS draw, under the request's objective sense.

        ROADMAP 2.1192 — THE CANONICAL OWNER of "which option wins?". Every
        winner decision in this service routes through this function, and there
        is deliberately no second implementation of it.

        THAT IS THE POINT OF EXTRACTING IT. Before 2.1192 there were TWO
        unconditional ``max()`` copies of this rule: the main Monte Carlo loop
        and ``_compute_evpi_metric``'s fixed-policy replay, which feeds
        ``p_win_sensitivity``. They agreed only because both were hardcoded to
        the same sense. Parameterising one and leaving the other would have made
        them disagree the moment a user stated ``minimise`` or ``target`` — the
        ranking scored against the team's aim while the sensitivity beside it
        scored against a maximiser, under labels that both say "win". Two
        same-named-but-different code paths have cost this estate real money
        before, and the failure mode is silent: each copy looks right on its
        own.

        ``status_quo_reference`` is this draw's no-intervention goal value, and
        is required only by a level-framed target. Callers that cannot supply
        one for such a target must not call this function with a maximise plan
        instead — they must suppress their phase.

        Returns ``[]`` for an uninformative draw (no finite option, or a
        level-framed target whose reference is unusable). The callers' existing
        no-winner semantics handle that: no option is credited, and the win
        probabilities sum to the informative fraction, which is the honest
        report.
        """
        if objective.sense == "withheld":
            # Checked before everything else so no later limb can be reached by
            # a withheld plan. A withheld ranking is not a degraded ranking: it
            # is the refusal to state one, and the ONLY correct number of
            # winners is none.
            return []

        if not finite_outcomes:
            return []

        if objective.sense == "maximise":
            best = max(finite_outcomes.values())
            return [opt_id for opt_id, val in finite_outcomes.items() if val == best]

        if objective.sense == "minimise":
            best = min(finite_outcomes.values())
            return [opt_id for opt_id, val in finite_outcomes.items() if val == best]

        # ``target``: rank by distance from the stated target. Smaller is
        # better, which is what lets an option BETWEEN two extremes win — the
        # thing an argmax over a monotone SCM can never do.
        if objective.target_level is not None:
            # LEVEL frame: recover each option's level per draw against the
            # status-quo reference under common random numbers, exactly as the
            # goal-probability channel does. Everything not caused by the option
            # is present in both terms and cancels.
            if status_quo_reference is None or not math.isfinite(status_quo_reference):
                return []
            assert objective.goal_baseline is not None
            baseline = objective.goal_baseline
            distances = {
                opt_id: abs((baseline + (val - status_quo_reference)) - objective.target_level)
                for opt_id, val in finite_outcomes.items()
            }
        else:
            # DELTA frame: the caller attested the target is already in the
            # samples' own frame, so compare raw.
            assert objective.target_delta is not None
            distances = {
                opt_id: abs(val - objective.target_delta) for opt_id, val in finite_outcomes.items()
            }

        # Finite outcomes can still overflow during frame conversion or distance
        # subtraction. An unrepresentable score earns no comparison credit;
        # all-overflow draws must not fabricate an all-option tie.
        finite_distances = {
            option_id: distance
            for option_id, distance in distances.items()
            if math.isfinite(distance)
        }
        if not finite_distances:
            return []
        best_distance = min(finite_distances.values())
        return [opt_id for opt_id, dist in finite_distances.items() if dist == best_distance]

    @staticmethod
    def _resolve_objective_plan(
        request: RobustnessRequestV2,
        goal_threshold_plan: Optional["GoalThresholdPlan"],
    ) -> Tuple["ObjectivePlan", Optional[InferenceWarning]]:
        """Resolve the stated objective; absence never licenses a default winner."""
        direction = request.goal_direction

        if direction is None:
            return (
                ObjectivePlan(
                    sense="withheld", attested=False, withheld_reason="goal_direction_absent"
                ),
                InferenceWarning(
                    code="GOAL_DIRECTION_UNATTESTED",
                    field="goal_direction",
                    severity="warning",
                    detail={
                        "goal_node_id": request.goal_node_id,
                        "reason": "goal_direction_absent",
                        "message": "No option is recommended yet because the aim is unclear. "
                        "Say whether you want the goal to increase, decrease, or approach a target.",
                    },
                ),
            )

        if direction in ("maximise", "minimise"):
            return ObjectivePlan(sense=direction, attested=True), None

        # direction == "target". The parse-time validator guarantees
        # goal_threshold and goal_threshold_frame are both present, so a None
        # plan here means the frame was stated and could not be CONVERTED — the
        # resolver has already emitted its own GOAL_THRESHOLD_NOT_CONVERTIBLE
        # warning naming the reason. This warning says what that costs the
        # ranking, which the threshold channel's warning does not know.
        if goal_threshold_plan is None:
            return ObjectivePlan(
                sense="withheld",
                attested=True,
                withheld_reason="target_not_resolvable_in_sample_frame",
            ), InferenceWarning(
                code="OBJECTIVE_RANKING_WITHHELD",
                field="goal_direction",
                detail={
                    "goal_node_id": request.goal_node_id,
                    "goal_direction": direction,
                    "goal_threshold": request.goal_threshold,
                    "goal_threshold_frame": request.goal_threshold_frame,
                    "reason": "target_not_resolvable_in_sample_frame",
                    "message": (
                        "The model cannot yet compare options against this target because its "
                        "measurement frame is incomplete or unsupported. No option is recommended."
                    ),
                },
                severity="warning",
            )

        return (
            ObjectivePlan(
                sense="target",
                attested=True,
                target_delta=goal_threshold_plan.delta_threshold,
                target_level=goal_threshold_plan.level_threshold,
                goal_baseline=goal_threshold_plan.goal_baseline,
            ),
            None,
        )

    @staticmethod
    def _resolve_goal_threshold_in_sample_frame(
        request: RobustnessRequestV2,
    ) -> Tuple[Optional["GoalThresholdPlan"], Optional[InferenceWarning]]:
        """Decide HOW ``goal_threshold`` is compared, or refuse (ROADMAP 2.258 / 2.286).

        THE ARITHMETIC, derived from the evaluator rather than assumed
        (``SCMEvaluatorV2.evaluate``)::

            sample = base_goal + intercept + SUM(parent_value * strength)

        For a NON-ROOT goal ``base_goal`` is 0.0 — doctrine B, and the reason ISL
        already emits ``GOAL_OBSERVED_VALUE_UNUSED``. Writing S for the parents'
        propagated contribution::

            sample = intercept + S

        ⚠ 2.286 CORRECTION. The 2.258 derivation continued ``real_level = B + S``
        and eliminated S to get ``sample >= T - B + intercept``. **That step is
        false, and it inverted answers.** It requires S to be a CHANGE from the
        goal's current level, i.e. ``S == 0`` under the status quo. But the
        evaluator seeds ``observed_state.value`` as the base of ROOT nodes ONLY
        (see the ``is_root`` branch in ``evaluate``), and ``FactorSampler`` draws
        factor values centred on ``observed_state.value``. So parents carry their
        ABSOLUTE current values, and ``S_sq = SUM(current_parent * strength)`` is
        emphatically not zero. Anchoring at zero shifted every level comparison
        by exactly ``S_sq``.

        Measured, not argued (f=0.5, strength=0.5, B=0.7, T=0.9, status-quo
        option): status-quo sample ``0.25`` vs converted threshold ``0.20`` ->
        ISL reported ``probability_of_goal = 1.0``. The status quo leaves the
        goal at 0.7, which does not reach 0.9, so the truth is ``0.0``. A
        confident inversion rendered to a user as certainty.

        THE FIX: recover levels per draw against a status-quo REFERENCE, under
        common random numbers::

            level_i = B + (option_sample_i - status_quo_sample_i)

        S_sq cancels because it is present in both terms — and so does the
        intercept, which is why the old ``+ intercept`` term is gone rather than
        merely re-derived. What survives is the option's causal effect on the
        goal, added to the level the goal is actually at. This resolver therefore
        returns a PLAN (``GoalThresholdPlan``) rather than a scalar: the
        comparison can no longer be collapsed to one number known before the
        Monte Carlo runs.

        B is read from the goal node's ``observed_state.baseline`` — the one field
        the canonical schema (Olumi_Decision_Model_Schema_v2_6.md, B.3) defines as
        the "Reference for 'change from baseline' calculations", which is exactly
        this calculation. It is deliberately NOT defaulted from
        ``observed_state.value``: that field is documented as the *current observed
        value*, carries a live ISL warning declaring it unused for a non-root goal,
        and silently repurposing it would be a second unattested frame assumption
        of precisely the kind that caused this defect.

        FAIL CLOSED. Every path that cannot be proved returns ``(None, warning)``:
        the caller then omits ``probability_of_goal`` entirely (``exclude_none``
        drops it from the wire) and the warning names what was missing. No
        fabricated number, no clamp, no silent default.

        Returns:
            ``(plan, warning)``. A non-None plan means "safe to compute a
            probability". ``(None, None)`` means no threshold was requested at
            all — nothing to disclose.
        """
        threshold = request.goal_threshold
        if threshold is None:
            # No goal threshold requested: nothing to convert, nothing to disclose.
            return None, None

        goal_id = request.goal_node_id
        frame = request.goal_threshold_frame

        def refuse(
            reason: str, field: str, message: str, **extra: Any
        ) -> Tuple[Optional["GoalThresholdPlan"], Optional[InferenceWarning]]:
            detail: Dict[str, Any] = {
                "goal_node_id": goal_id,
                "goal_threshold": threshold,
                "goal_threshold_frame": frame,
                "reason": reason,
                "message": message,
            }
            detail.update(extra)
            return None, InferenceWarning(
                code=(
                    "GOAL_THRESHOLD_FRAME_UNSPECIFIED"
                    if reason == "frame_not_stamped"
                    else "GOAL_THRESHOLD_NOT_CONVERTIBLE"
                ),
                field=field,
                detail=detail,
                # Degradation disclosure, NOT a benign input-adjustment
                # diagnostic: PLoT hides severity=='info'. The whole point of
                # this warning is that a downstream honesty surface can say
                # "not available" WITH a reason, so it must ride as 'warning'.
                severity="warning",
            )

        # ROADMAP 2.798. The convertibility RULES below used to live inline here
        # and nowhere else. They now live in ONE place —
        # ``_resolve_threshold_in_sample_frame`` — because the constraint channel
        # needs exactly the same rules and a second copy of them would be a
        # second dialect of the same contract. Two same-named-but-different code
        # paths have cost this estate real money before (the two
        # ``generateGraphHash`` twins, one seed-bearing and one seedless,
        # conflated repo-wide), and the failure mode is silent: each copy looks
        # right on its own.
        #
        # WHAT IS SHARED is the rule set and its ORDER. WHAT IS NOT is the
        # disclosure vocabulary — reason names, warning codes, detail keys and
        # user-facing messages are this channel's, supplied below, so a consumer
        # keying on ``reason == "root_goal"`` is unaffected by the fold.
        return RobustnessAnalyzerV2._resolve_threshold_in_sample_frame(
            request,
            target_id=goal_id,
            threshold=threshold,
            frame=frame,
            frame_field="goal_threshold_frame",
            value_label="goal_threshold",
            noun="Goal node",
            omitted_field="probability_of_goal",
            reasons={
                "node_missing": "goal_node_missing",
                "pinned_by_intervention": "goal_pinned_by_intervention",
                "root_target": "root_goal",
                "parameter_uncertainty_shifts_base": ("goal_parameter_uncertainty_shifts_base"),
                "missing_baseline": "missing_goal_baseline",
                "values_outside_normalised_domain": ("goal_values_outside_normalised_domain"),
            },
            operand_names={
                "threshold": "goal_threshold",
                "baseline": "goal_baseline",
                "intercept": "goal_intercept",
            },
            refuse=refuse,
        )

    # The magnitude bound the level conversion trusts. Derived from the
    # evaluator's own [0, 1] node-value clamp, not chosen: a value orders of
    # magnitude outside it is raw user units sent where normalised values were
    # expected, and converting those silently yields a WRONG NUMBER rather than no
    # number — the one failure mode fail-closed does not otherwise cover. 1.5 is a
    # deliberate slack margin admitting legitimate overshoot (an intercept of 1.0,
    # a threshold slightly above the cap).
    NORMALISED_DOMAIN_LIMIT = 1.5

    @staticmethod
    def _resolve_threshold_in_sample_frame(
        request: RobustnessRequestV2,
        *,
        target_id: str,
        threshold: float,
        frame: Optional[str],
        frame_field: str,
        value_label: str,
        noun: str,
        omitted_field: str,
        reasons: Dict[str, str],
        operand_names: Dict[str, str],
        refuse: Callable[..., Tuple[Optional["GoalThresholdPlan"], Any]],
    ) -> Tuple[Optional["GoalThresholdPlan"], Any]:
        """THE convertibility rules — one implementation, two channels.

        Answers one question: *may this threshold be compared against this target
        node's Monte Carlo samples, and if so how?* It is shared verbatim by

        * Channel A — ``goal_threshold`` -> ``probability_of_goal`` (2.258 / 2.286)
        * Channel B — ``goal_constraints`` -> ``constraint_analysis`` (2.798)

        THE ARITHMETIC, derived from ``SCMEvaluatorV2.evaluate`` rather than
        assumed::

            sample = base + intercept + SUM(parent_value * strength)

        ``base`` is ``observed_state.value`` for ROOT nodes and 0.0 for every
        other node. So a NON-ROOT target's samples are a change measured from an
        origin of ``intercept``, not the target quantity's real level, while
        producers mint thresholds as LEVELS. Comparing the two is the category
        error that yields a structural zero.

        Levels are recovered per draw against a status-quo REFERENCE, under common
        random numbers::

            level_i = baseline + (option_sample_i - status_quo_sample_i)

        Everything not caused by the option — the factors' current values, the
        sampled edge strengths, the target's intercept — appears in BOTH terms and
        cancels exactly.

        FAIL CLOSED. Every path that cannot be PROVED returns a refusal; the
        caller then omits its field entirely and the warning names what was
        missing. No fabricated number, no clamp, no silent default.

        WHY ``refuse`` IS INJECTED. The rules are universal; the disclosure is
        not. Each channel owns its reason names, warning codes, detail keys and
        prose, because a consumer keying on ``reason == "root_goal"`` must not be
        broken by a channel it does not read. Sharing the rules while separating
        the vocabulary is what makes this a fold rather than a rename.

        Args:
            target_id: The node whose samples the threshold will be compared with.
            threshold: The number to compare.
            frame: ``'level'``, ``'delta'``, or None (NOT STAMPED -> refused).
            frame_field: Field path of the attestation, for the warning.
            value_label: How to name the threshold in prose.
            noun: How to name the target node in prose.
            omitted_field: What the caller will omit, named in prose.
            reasons: Neutral reason key -> this channel's reason name.
            operand_names: Neutral operand key -> this channel's detail key.
            refuse: ``(reason, field, message, **extra) -> (None, warning)``.

        Returns:
            ``(plan, None)`` when safe to compute, ``(None, warning)`` otherwise.
        """
        if frame is None:
            return refuse(
                "frame_not_stamped",
                frame_field,
                (
                    f"{value_label}={threshold} was supplied without "
                    f"{frame_field}, so the frame it is expressed in is unknown. "
                    f"A level threshold compared against the target's "
                    f"change-from-origin samples yields a structurally impossible "
                    f"probability, so {omitted_field} is omitted rather than "
                    f"guessed. Stamp 'level' or 'delta'."
                ),
            )

        if frame == "delta":
            # Attested to be in the samples' own frame already. This is the
            # pre-2.258 comparison, byte-identical, and it is the CALLER's
            # attestation — ISL has no way to verify a number's provenance.
            return GoalThresholdPlan(delta_threshold=threshold), None

        # frame == "level": convert into the sample frame, or refuse.
        target_node = next((n for n in request.graph.nodes if n.id == target_id), None)
        if target_node is None:
            # Unreachable via the API (the request validators reject unknown node
            # ids), kept so the helper is total for direct callers.
            return refuse(
                reasons["node_missing"],
                f"nodes[{target_id}]",
                f"{noun} '{target_id}' is not present in the graph.",
            )

        # --- convertibility preconditions -------------------------------------
        # Each of these makes `sample = intercept + S` false, so the identity the
        # conversion rests on no longer holds. Refusing is the only honest answer.
        if any(target_id in option.interventions for option in request.options):
            return refuse(
                reasons["pinned_by_intervention"],
                "options[].interventions",
                (
                    f"At least one option intervenes directly on {noun.lower()} "
                    f"'{target_id}', pinning its samples to an absolute value. "
                    f"Those samples are not change-from-origin, so a level "
                    f"threshold cannot be converted consistently across options."
                ),
            )

        if not any(edge.to == target_id for edge in request.graph.edges):
            return refuse(
                reasons["root_target"],
                f"nodes[{target_id}]",
                (
                    f"{noun} '{target_id}' has no parents. A root node takes its "
                    f"base from observed_state.value, so its samples are not in "
                    f"the non-root change-from-origin frame this conversion is "
                    f"derived for."
                ),
            )

        if any(pu.node_id == target_id for pu in (request.parameter_uncertainties or [])):
            return refuse(
                reasons["parameter_uncertainty_shifts_base"],
                f"parameter_uncertainties[{target_id}]",
                (
                    f"{noun} '{target_id}' carries a ParameterUncertainty: each "
                    f"sample draws a base that is ADDED to parent propagation, so "
                    f"the samples' origin varies per sample and a single static "
                    f"conversion is not valid."
                ),
            )

        observed = target_node.observed_state
        baseline = observed.baseline if observed is not None else None
        if baseline is None:
            return refuse(
                reasons["missing_baseline"],
                f"nodes[{target_id}].observed_state.baseline",
                (
                    f"A 'level' frame requires {noun.lower()} '{target_id}' to "
                    f"carry observed_state.baseline to convert the level into the "
                    f"samples' frame, but it carries "
                    + (
                        "no observed_state at all."
                        if observed is None
                        else "an observed_state with no baseline."
                    )
                ),
                observed_state_present=observed is not None,
            )

        intercept = target_node.intercept
        # Belt-and-braces: the field validators already reject non-finite
        # baseline/threshold, but this helper is also called directly by tests and
        # by any future non-HTTP entry point, and a non-finite threshold is exactly
        # the input that would produce a silently absurd probability.
        #
        # This check must EXIST — `abs(nan) > 1.5` is False, so the domain guard
        # below cannot catch a NaN. Its POSITION, however, is not load-bearing.
        if not all(math.isfinite(v) for v in (threshold, baseline, intercept)):
            return refuse(
                "non_finite_conversion_input",
                f"nodes[{target_id}].observed_state.baseline",
                (
                    "Conversion inputs must all be finite "
                    f"({value_label}={threshold}, baseline={baseline}, "
                    f"intercept={intercept})."
                ),
                **{
                    operand_names["baseline"]: baseline,
                    operand_names["intercept"]: intercept,
                },
            )

        # --- domain guard (Tier 2) --------------------------------------------
        # See NORMALISED_DOMAIN_LIMIT. NOTE this is Tier 2 (magnitude). Tier 1 —
        # attesting the domain properly via observed_state.value ~= raw_value / cap
        # — is deliberately NOT implemented here and is rowed separately. Neither
        # tier can see a UNIT: a count of people normalised by a percentage cap is
        # already the wrong number before any frame conversion, and it is not this
        # guard's to catch (ROADMAP 2.797, a different service).
        limit = RobustnessAnalyzerV2.NORMALISED_DOMAIN_LIMIT
        out_of_domain = {
            name: value
            for name, value in (
                (operand_names["threshold"], threshold),
                (operand_names["baseline"], baseline),
                (operand_names["intercept"], intercept),
            )
            if abs(value) > limit
        }
        if out_of_domain:
            return refuse(
                reasons["values_outside_normalised_domain"],
                f"nodes[{target_id}].observed_state.baseline",
                (
                    f"Conversion operands {sorted(out_of_domain)} exceed "
                    f"|{limit}|, so they are not in the normalised [0, 1] domain "
                    f"the evaluator assumes for node values. This usually means "
                    f"raw user units were sent where normalised values were "
                    f"expected; converting them would produce a wrong number "
                    f"rather than no number."
                ),
                out_of_domain=out_of_domain,
                domain_limit=limit,
            )

        # --- epsilon guard -----------------------------------------------------
        # The anchor needs `option_sample_i` and `status_quo_sample_i` to differ
        # ONLY by the option. Epsilon breaks that in two independent ways:
        #
        #   1. NOT CRN-MATCHABLE. `SCMEvaluatorV2.evaluate` draws epsilon inside
        #      each call, so two calls in one MC draw get two independent noise
        #      vectors. The difference would then carry ~2x epsilon variance that
        #      no option caused — manufacturing uncertainty, which is the same
        #      class of untruth as manufacturing confidence.
        #   2. THE CLAMP IS NOT ADDITIVE. A node with epsilon_std > 0 is clamped
        #      to [0, 1] after noise, so `option - status_quo` stops being the
        #      option's effect at either rail.
        #
        # Only epsilon that can actually REACH the target matters, so this walks
        # the target's ancestors rather than the whole graph: a noisy node in a
        # disconnected branch cannot perturb these samples, and refusing on it
        # would be over-refusal, which has its own cost (a user sees "not
        # available" for an answer ISL could have given honestly).
        parents_of: Dict[str, List[str]] = defaultdict(list)
        for edge in request.graph.edges:
            parents_of[edge.to].append(edge.from_)
        influencers = {target_id}
        frontier = [target_id]
        while frontier:
            for parent in parents_of[frontier.pop()]:
                if parent not in influencers:
                    influencers.add(parent)
                    frontier.append(parent)
        noisy = sorted(
            node.id
            for node in request.graph.nodes
            if node.id in influencers and node.epsilon_std > 0
        )
        if noisy:
            return refuse(
                "epsilon_breaks_status_quo_reference",
                f"nodes[{target_id}].epsilon_std",
                (
                    f"Nodes {noisy} carry epsilon_std > 0 and can influence "
                    f"'{target_id}'. A level threshold is resolved by differencing "
                    f"each option's sample against a status-quo sample from the "
                    f"same draw, but epsilon is drawn per evaluation and clamped "
                    f"to [0, 1], so that difference would carry noise no option "
                    f"caused. {omitted_field} is omitted rather than widened by "
                    f"fabricated variance."
                ),
                **{
                    operand_names["baseline"]: baseline,
                    operand_names["intercept"]: intercept,
                    "noisy_node_ids": noisy,
                },
            )

        # The domain guard bounds every operand, so both plan values are finite by
        # construction. No post-hoc finiteness branch is emitted here on purpose:
        # unreachable machinery that reads as a guarantee is exactly the defect
        # class this repo hunts.
        return GoalThresholdPlan(level_threshold=threshold, goal_baseline=baseline), None

    @staticmethod
    def _resolve_constraint_plans(
        request: RobustnessRequestV2,
    ) -> Tuple[Optional[Dict[int, "GoalThresholdPlan"]], List[InferenceWarning]]:
        """Resolve EVERY goal_constraint into a comparison plan, or refuse the block.

        ROADMAP 2.798 — Channel B's half of the fail-closed contract Channel A has
        had since 2.258 / 2.286. Runs BEFORE the Monte Carlo, because the plans
        decide which nodes need a per-draw status-quo reference recorded.

        WHY ALL-OR-NOTHING. ``joint_probability`` is P(ALL constraints satisfied):
        a conjunction is unresolvable the moment ANY conjunct is. The pairwise
        conditionals P(C_j | C_i) fail the same way. And
        ``ConstraintAnalysisV2.joint_probability`` is a REQUIRED wire field with a
        live consumer in another repo, so a partially populated block could not be
        emitted honestly even if the arithmetic allowed it — dropping the field
        would break that consumer's parse, which is the schema-skew hazard this
        estate pays for most often.

        So the refusal unit is the BLOCK. That costs nothing structural:
        ``constraint_analysis`` is already Optional and already absent on every
        request that sends no constraints, so absence is a shape every consumer
        already handles. A partial block would be a new shape, emitted exactly
        when we are least sure of ourselves.

        Returns:
            ``(plans, warnings)``. ``plans`` maps constraint INDEX -> plan when
            every constraint resolved; ``None`` means the block must be omitted.
            ``warnings`` names each unresolvable constraint by its identity.
            ``({}, [])`` when no constraints were requested.
        """
        constraints = request.goal_constraints
        if not constraints:
            return {}, []

        plans: Dict[int, "GoalThresholdPlan"] = {}
        warnings: List[InferenceWarning] = []

        for index, constraint in enumerate(constraints):

            def refuse(
                reason: str,
                field: str,
                message: str,
                _constraint: GoalConstraint = constraint,
                **extra: Any,
            ) -> Tuple[None, InferenceWarning]:
                detail: Dict[str, Any] = {
                    # Identity first: a consumer must be able to say WHICH
                    # constraint was refused without reconstructing it
                    # positionally.
                    "constraint_id": _constraint.constraint_id,
                    "node_id": _constraint.node_id,
                    "operator": _constraint.operator,
                    "constraint_value": _constraint.value,
                    "value_frame": _constraint.value_frame,
                    "reason": reason,
                    "message": message,
                }
                detail.update(extra)
                return None, InferenceWarning(
                    code=(
                        "CONSTRAINT_FRAME_UNSPECIFIED"
                        if reason == "frame_not_stamped"
                        else "CONSTRAINT_NOT_CONVERTIBLE"
                    ),
                    field=field,
                    detail=detail,
                    # Degradation disclosure, NOT a benign input-adjustment
                    # diagnostic: PLoT hides severity=='info'. A downstream
                    # honesty surface must be able to say "not available" WITH a
                    # reason, so it rides as 'warning'.
                    severity="warning",
                )

            plan, warning = RobustnessAnalyzerV2._resolve_threshold_in_sample_frame(
                request,
                target_id=constraint.node_id,
                threshold=constraint.value,
                frame=constraint.value_frame,
                frame_field=f"goal_constraints[{index}].value_frame",
                value_label="constraint value",
                noun="Constraint target node",
                omitted_field="constraint_analysis",
                reasons={
                    "node_missing": "constraint_node_missing",
                    "pinned_by_intervention": "target_pinned_by_intervention",
                    "root_target": "root_target",
                    "parameter_uncertainty_shifts_base": (
                        "target_parameter_uncertainty_shifts_base"
                    ),
                    "missing_baseline": "missing_target_baseline",
                    "values_outside_normalised_domain": (
                        "constraint_values_outside_normalised_domain"
                    ),
                },
                operand_names={
                    "threshold": "constraint_threshold",
                    "baseline": "constraint_baseline",
                    "intercept": "constraint_intercept",
                },
                refuse=refuse,
            )

            if warning is not None:
                warnings.append(warning)
            elif plan is not None:
                plans[index] = plan

        if len(plans) != len(constraints):
            return None, warnings
        return plans, warnings

    @staticmethod
    def _constraint_status_quo_nodes(
        request: RobustnessRequestV2,
        constraint_plans: Optional[Dict[int, "GoalThresholdPlan"]],
    ) -> List[str]:
        """Target nodes whose per-draw status-quo reference the plans will need.

        Derived from the PLANS, not from the constraints — a 'delta' constraint
        needs no reference, and recording one for it would make the MC do work no
        field asked for.
        """
        if not constraint_plans or not request.goal_constraints:
            return []
        return sorted(
            {
                request.goal_constraints[index].node_id
                for index, plan in constraint_plans.items()
                if plan.needs_status_quo_reference
            }
        )

    @staticmethod
    def _resolve_constraint_series(
        constraint_node_values: Dict[str, Dict[str, List[float]]],
        constraints: List[GoalConstraint],
        constraint_plans: Dict[int, "GoalThresholdPlan"],
        status_quo_node_values: Dict[str, List[float]],
        option_id: str,
    ) -> Dict[int, List[float]]:
        """Put every constraint's samples into the frame ITS threshold is stated in.

        This is the whole fix in four lines of arithmetic, and it is deliberately
        the SAME arithmetic Channel A applies to ``probability_of_goal``::

            level_i = baseline + (option_sample_i - status_quo_sample_i)

        Everything downstream — satisfaction, joint probability, conditionals,
        failure margins, near-miss fractions — then operates on a series that is
        commensurable with the threshold by construction. That is why no
        comparison site needs a frame check of its own: there is nothing left for
        one to catch.

        A 'delta' plan passes its samples through untouched, on the caller's
        attestation that they are already in the samples' own frame.

        Keyed by constraint INDEX, not node_id: two constraints may target the
        same node with different frames, and collapsing them by node would let one
        constraint's conversion answer for the other (CLAUDE.md trap 19).
        """
        option_values = constraint_node_values[option_id]
        resolved: Dict[int, List[float]] = {}

        for index, constraint in enumerate(constraints):
            plan = constraint_plans[index]
            samples = option_values[constraint.node_id]

            if plan.level_threshold is None:
                resolved[index] = samples
                continue

            reference = status_quo_node_values[constraint.node_id]
            baseline = plan.goal_baseline
            # GoalThresholdPlan sets level_threshold and goal_baseline together
            # at one site, so a level plan always carries a baseline. Asserted
            # rather than silenced: if that invariant is ever broken, this must
            # fail loudly here rather than propagate a None into the arithmetic
            # and surface as a TypeError halfway through a Monte Carlo.
            assert baseline is not None, (
                "a level plan must carry the baseline it converts against "
                f"(constraint index {index}, node {constraint.node_id})"
            )
            # strict=True is load-bearing, not lint hygiene: the whole conversion
            # rests on the option sample and the reference sample coming from the
            # SAME Monte Carlo draw. A length mismatch means that pairing has
            # broken, and a silent zip() would truncate to the shorter series and
            # produce a plausible number from misaligned draws — a fabrication of
            # exactly the kind this change exists to make impossible.
            resolved[index] = [
                baseline + (sample - reference_sample)
                for sample, reference_sample in zip(samples, reference, strict=True)
            ]

        return resolved

    def _compute_option_results(
        self,
        outcomes: Dict[str, List[float]],
        wins: Dict[str, float],
        request: RobustnessRequestV2,
        goal_threshold_plan: Optional["GoalThresholdPlan"] = None,
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]] = None,
        expected_regret: Optional[Dict[str, Optional[float]]] = None,
        status_quo_outcomes: Optional[List[float]] = None,
        constraint_plans: Optional[Dict[int, "GoalThresholdPlan"]] = None,
        status_quo_node_values: Optional[Dict[str, List[float]]] = None,
        objective_ranking: Optional[ObjectiveRanking] = None,
    ) -> List[OptionResult]:
        """Compute distribution statistics for each option.

        Args:
            outcomes: Dict[option_id, List[outcome_samples]]
            wins: Dict[option_id, win_count]
            request: The analysis request
            goal_threshold_plan: How to compare goal_threshold, already resolved by
                _resolve_goal_threshold_in_sample_frame (ROADMAP 2.258 / 2.286).
                None -> probability_of_goal is OMITTED for every option. Passed in
                rather than re-read from `request` so the frame resolution happens
                exactly once, at one site, and cannot drift per option.
            constraint_node_values: Optional dict of constraint node sample values
                for multi-constraint analysis
            expected_regret: Optional dict[option_id, pre-noise JOINT expected
                regret]. Computed by the caller from the PRE-noise CRN-aligned
                outcomes (B2 CRN-fix F1) so it rides the same population as
                win_probability. Stored on OptionResult.pre_noise_expected_regret (serialized;
                survives offload) for the V2 emission layer. None -> not threaded.
            status_quo_outcomes: Per-draw no-intervention goal values, CRN-paired
                element-wise with every option's samples. Required by, and only
                by, a level-frame plan (ROADMAP 2.286).
        """
        expected_regret = expected_regret or {}
        ranked_shares = (
            {row.option_id: row.win_probability for row in objective_ranking.ranked_options}
            if objective_ranking is not None
            else None
        )
        results = []

        for option in request.options:
            samples = outcomes[option.id]
            if not samples:
                continue

            samples_array = np.array(samples)
            ci_lower, ci_upper = self._compute_confidence_interval(
                samples_array, request.confidence_level
            )

            # Compute probability_of_goal from the resolved PLAN (ROADMAP 2.258 /
            # 2.286). This never reads request.goal_threshold: an unattested or
            # unconvertible threshold arrives here as a None plan and the field is
            # omitted, which is what stops the "< 1% chance of hitting your goal"
            # untruth in one direction and the "100% chance" untruth in the other.
            probability_of_goal = None
            if goal_threshold_plan is not None:
                if goal_threshold_plan.delta_threshold is not None:
                    # Caller attested the threshold is already in the samples' frame.
                    meets = samples_array >= goal_threshold_plan.delta_threshold
                else:
                    # Level frame: recover the goal's LEVEL per draw by adding the
                    # option's causal effect to the level the goal is actually at.
                    # Paired element-wise with the status-quo draw, so the factors'
                    # current values, the sampled strengths and the goal's intercept
                    # all cancel instead of being mistaken for progress.
                    effect = samples_array - np.array(status_quo_outcomes)
                    levels = goal_threshold_plan.goal_baseline + effect
                    meets = levels >= goal_threshold_plan.level_threshold
                probability_of_goal = int(np.sum(meets)) / len(samples)

            # Compute constraint analysis if constraints provided
            constraint_analysis_result: Optional[ConstraintAnalysis] = None
            if request.goal_constraints and constraint_node_values:
                analysis_dict = self._compute_constraint_analysis(
                    constraint_node_values,
                    request.goal_constraints,
                    option.id,
                    constraint_plans,
                    status_quo_node_values,
                )
                if analysis_dict:
                    # Convert dict to ConstraintAnalysis model
                    constraint_results = [
                        ConstraintResult(
                            # Slice 6b echo. Subscript, not .get(): the dict is
                            # built one function away in
                            # _compute_constraint_analysis, its only producer, so
                            # a missing key is a real drift and should raise here
                            # rather than silently None the identity.
                            constraint_id=c["constraint_id"],
                            node_id=c["node_id"],
                            operator=c["operator"],
                            threshold=c["threshold"],
                            label=c["label"],
                            prob_satisfied=c["prob_satisfied"],
                            failure_margin_median=c["failure_margin_median"],
                            near_miss_fraction=c["near_miss_fraction"],
                            binding=c["binding"],
                        )
                        for c in analysis_dict["constraints"]
                    ]
                    constraint_analysis_result = ConstraintAnalysis(
                        constraints=constraint_results,
                        joint_probability=analysis_dict["joint_probability"],
                        conditional_probabilities=analysis_dict["conditional_probabilities"],
                    )

            option_result = OptionResult(
                option_id=option.id,
                outcome_distribution=OutcomeDistribution(
                    mean=float(np.mean(samples_array)),
                    std=float(np.std(samples_array)),
                    median=float(np.median(samples_array)),
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    # Task 2: Store raw samples so the V2 API layer can compute
                    # actual p10/p50/p90 percentiles instead of aliasing CI bounds.
                    samples=samples,
                ),
                win_probability=(
                    ranked_shares.get(option.id)
                    if ranked_shares is not None
                    else wins[option.id] / request.n_samples
                ),
                probability_of_goal=probability_of_goal,
                constraint_analysis=constraint_analysis_result,
            )
            # B2 CRN-fix (F1): attach the PRE-noise joint regret so the V2
            # emission layer emits the CRN-aligned value instead of recomputing it
            # from the POST-noise samples above. A regular (serialized) field so it
            # survives the offload worker->endpoint dump/validate boundary.
            option_result.pre_noise_expected_regret = expected_regret.get(option.id)
            results.append(option_result)

        return results

    def _compute_confidence_interval(
        self, samples: np.ndarray, confidence_level: float
    ) -> Tuple[float, float]:
        """Compute percentile-based prediction interval from Monte Carlo samples.

        Note: Returns percentile bounds (not a frequentist confidence interval).
        For 95% level, returns 2.5th and 97.5th percentiles of the sample distribution.
        """
        alpha = 1 - confidence_level
        lower = float(np.percentile(samples, alpha / 2 * 100))
        upper = float(np.percentile(samples, (1 - alpha / 2) * 100))
        return lower, upper

    def _compute_sensitivity(
        self,
        request: RobustnessRequestV2,
        baseline_outcomes: Dict[str, List[float]],
        sampler: DualUncertaintySampler,
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
    ) -> List[SensitivityResult]:
        """
        Compute sensitivity to edge existence and magnitude.

        For each edge, measures:
        1. Existence sensitivity: Impact of forcing edge on vs off
        2. Magnitude sensitivity: Impact of varying strength mean
        """
        sensitivities: List[Dict[str, Any]] = []

        # Compute baseline mean outcome for reference option
        ref_option = request.options[0]
        baseline_mean = float(np.mean(baseline_outcomes[ref_option.id]))

        for edge in request.graph.edges:
            # Existence sensitivity
            existence_sens = self._compute_existence_sensitivity(
                request, edge, baseline_mean, rng, evaluator
            )
            sensitivities.append(
                {
                    "edge_from": edge.from_,
                    "edge_to": edge.to,
                    "sensitivity_type": "existence",
                    "elasticity": existence_sens,
                    "interpretation": self._interpret_existence_sensitivity(edge, existence_sens),
                }
            )

            # Magnitude sensitivity
            magnitude_sens = self._compute_magnitude_sensitivity(
                request, edge, baseline_mean, rng, evaluator
            )
            sensitivities.append(
                {
                    "edge_from": edge.from_,
                    "edge_to": edge.to,
                    "sensitivity_type": "magnitude",
                    "elasticity": magnitude_sens,
                    "interpretation": self._interpret_magnitude_sensitivity(edge, magnitude_sens),
                }
            )

        # Rank by absolute elasticity
        sensitivities.sort(key=lambda x: abs(float(x["elasticity"])), reverse=True)

        # Convert to SensitivityResult with ranks
        results = []
        for i, s in enumerate(sensitivities):
            results.append(
                SensitivityResult(
                    edge_from=s["edge_from"],
                    edge_to=s["edge_to"],
                    sensitivity_type=s["sensitivity_type"],
                    elasticity=s["elasticity"],
                    importance_rank=i + 1,
                    interpretation=s["interpretation"],
                )
            )

        return results

    def _compute_existence_sensitivity(
        self,
        request: RobustnessRequestV2,
        edge: EdgeV2,
        baseline_mean: float,
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
    ) -> float:
        """
        Compute sensitivity to edge existence.

        Compares outcomes when edge is forced to exist vs forced to not exist.
        """
        n_sensitivity_samples = min(
            SENSITIVITY_SUBSAMPLE_CAP, request.n_samples // SENSITIVITY_SUBSAMPLE_DIVISOR
        )
        ref_option = request.options[0]

        # Sample with edge forced to exist
        outcomes_on = []
        for _ in range(n_sensitivity_samples):
            edge_config = self._sample_with_forced_existence(
                request.graph.edges, edge, exists=True, rng=rng
            )
            outcome = evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
            )
            outcomes_on.append(outcome)

        # Sample with edge forced to not exist
        outcomes_off = []
        for _ in range(n_sensitivity_samples):
            edge_config = self._sample_with_forced_existence(
                request.graph.edges, edge, exists=False, rng=rng
            )
            outcome = evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
            )
            outcomes_off.append(outcome)

        # Compute elasticity
        mean_on = float(np.mean(outcomes_on))
        mean_off = float(np.mean(outcomes_off))
        outcome_diff = mean_on - mean_off

        # Use epsilon-stabilised denominator to handle near-zero baselines
        baseline_denom = max(abs(baseline_mean), FACTOR_SENSITIVITY_BASELINE_EPSILON)

        # Elasticity: relative change in outcome for existence change (0 -> 1)
        raw_elasticity = outcome_diff / baseline_denom
        return float(max(-ELASTICITY_CLAMP_MAX, min(ELASTICITY_CLAMP_MAX, raw_elasticity)))

    def _compute_magnitude_sensitivity(
        self,
        request: RobustnessRequestV2,
        edge: EdgeV2,
        baseline_mean: float,
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
    ) -> float:
        """
        Compute sensitivity to edge magnitude.

        Varies strength mean by ±1 std and measures outcome change.
        """
        n_sensitivity_samples = min(
            SENSITIVITY_SUBSAMPLE_CAP, request.n_samples // SENSITIVITY_SUBSAMPLE_DIVISOR
        )
        ref_option = request.options[0]

        # Sample with strength mean + std
        outcomes_high = []
        for _ in range(n_sensitivity_samples):
            edge_config = self._sample_with_shifted_mean(
                request.graph.edges, edge, shift=+edge.strength.std, rng=rng
            )
            outcome = evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
            )
            outcomes_high.append(outcome)

        # Sample with strength mean - std
        outcomes_low = []
        for _ in range(n_sensitivity_samples):
            edge_config = self._sample_with_shifted_mean(
                request.graph.edges, edge, shift=-edge.strength.std, rng=rng
            )
            outcome = evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
            )
            outcomes_low.append(outcome)

        # Compute elasticity (change per 2*std shift)
        mean_high = float(np.mean(outcomes_high))
        mean_low = float(np.mean(outcomes_low))
        outcome_diff = mean_high - mean_low

        # Use epsilon-stabilised denominator to handle near-zero baselines
        baseline_denom = max(abs(baseline_mean), FACTOR_SENSITIVITY_BASELINE_EPSILON)

        # Normalize by 2*std range
        raw_elasticity = (outcome_diff / baseline_denom) / 2.0
        return float(max(-ELASTICITY_CLAMP_MAX, min(ELASTICITY_CLAMP_MAX, raw_elasticity)))

    def _sample_with_forced_existence(
        self,
        edges: List[EdgeV2],
        target_edge: EdgeV2,
        exists: bool,
        rng: SeededRNG,
    ) -> Dict[Tuple[str, str], float]:
        """Sample edge configuration with one edge's existence forced."""
        config = {}

        for edge in edges:
            edge_key = (edge.from_, edge.to)

            if edge.from_ == target_edge.from_ and edge.to == target_edge.to:
                # Force this edge's existence
                if exists:
                    config[edge_key] = rng.truncated_normal(
                        edge.strength.mean,
                        edge.strength.std,
                        EDGE_STRENGTH_MIN,
                        EDGE_STRENGTH_MAX,
                    )
                else:
                    config[edge_key] = 0.0
            else:
                # Sample normally
                if rng.bernoulli(edge.exists_probability):
                    config[edge_key] = rng.truncated_normal(
                        edge.strength.mean,
                        edge.strength.std,
                        EDGE_STRENGTH_MIN,
                        EDGE_STRENGTH_MAX,
                    )
                else:
                    config[edge_key] = 0.0

        return config

    def _sample_with_shifted_mean(
        self,
        edges: List[EdgeV2],
        target_edge: EdgeV2,
        shift: float,
        rng: SeededRNG,
    ) -> Dict[Tuple[str, str], float]:
        """
        Sample edge configuration with one edge's mean shifted.

        CRITICAL: The target edge is FORCED to exist so we isolate
        magnitude effect from existence effect. Otherwise, magnitude
        sensitivity would be conflated with structural uncertainty.
        """
        config = {}

        for edge in edges:
            edge_key = (edge.from_, edge.to)

            if edge.from_ == target_edge.from_ and edge.to == target_edge.to:
                # TARGET EDGE: Force to exist and apply shifted mean
                # This isolates magnitude sensitivity from existence sensitivity
                config[edge_key] = rng.truncated_normal(
                    edge.strength.mean + shift,
                    edge.strength.std,
                    EDGE_STRENGTH_MIN,
                    EDGE_STRENGTH_MAX,
                )
            else:
                # OTHER EDGES: Sample normally (both existence and strength)
                if rng.bernoulli(edge.exists_probability):
                    config[edge_key] = rng.truncated_normal(
                        edge.strength.mean,
                        edge.strength.std,
                        EDGE_STRENGTH_MIN,
                        EDGE_STRENGTH_MAX,
                    )
                else:
                    config[edge_key] = 0.0

        return config

    def _interpret_existence_sensitivity(self, edge: EdgeV2, elasticity: float) -> str:
        """Generate human-readable interpretation for existence sensitivity."""
        edge_name = f"{edge.from_}->{edge.to}"

        if abs(elasticity) < 0.05:
            return f"Decision is robust to whether {edge_name} exists"
        elif abs(elasticity) < self.HIGH_SENSITIVITY_THRESHOLD:
            return f"Decision is moderately sensitive to {edge_name} existence"
        else:
            return (
                f"Decision is highly sensitive to {edge_name} existence - "
                "consider validating this relationship"
            )

    def _interpret_magnitude_sensitivity(self, edge: EdgeV2, elasticity: float) -> str:
        """Generate human-readable interpretation for magnitude sensitivity."""
        edge_name = f"{edge.from_}->{edge.to}"

        if abs(elasticity) < 0.05:
            return f"Decision is robust to effect size variation in {edge_name}"
        elif abs(elasticity) < self.HIGH_SENSITIVITY_THRESHOLD:
            return f"Decision is moderately sensitive to {edge_name} effect size"
        else:
            return (
                f"Decision is highly sensitive to {edge_name} effect size - "
                "consider narrowing uncertainty"
            )

    @staticmethod
    def _intervention_factor_union(request: RobustnessRequestV2) -> set:
        """The D-U lever set: factor IDs that ANY option intervenes on (union across
        all options). A factor in this set is a CHOICE, not information to buy — it is
        the single source of truth for lever identity, consulted by BOTH
        ``_compute_factor_sensitivity`` (INTERVENTION_OVERRIDE detection) and
        ``_compute_factor_evppi`` (lever omission). Derive-don't-mirror (CLAUDE.md
        #12): one definition, so the two sites cannot fork lever identity. Used for
        membership tests only, so element order is irrelevant to determinism.
        """
        return {
            factor_id for option in request.options for factor_id in (option.interventions or {})
        }

    @staticmethod
    def _dedup_uncertainties(request: RobustnessRequestV2) -> List:
        """Deduplicate parameter_uncertainties by node_id, first-seen order (dict
        preserves insertion order → deterministic). Parse-time validation rejects
        duplicate node_ids with a 422, but a direct internal caller could bypass it,
        so a repeated node_id can never double-count in the per-factor VOI loops."""
        return list({u.node_id: u for u in (request.parameter_uncertainties or [])}.values())

    @staticmethod
    def _per_factor_seed(seed: int, phase: str, node_id: str) -> int:
        """Deterministic per-factor sub-seed for the VOI estimators. The ``phase``
        tag ('evpi' / 'evppi') INTENTIONALLY forks the RNG stream so the two
        estimators never share permutation/redraw randomness (drift-tolerant by
        design — the tag is the only per-site variation)."""
        return int(hashlib.sha256(f"{seed}:{phase}:{node_id}".encode()).hexdigest()[:8], 16)

    def _compute_factor_sensitivity(
        self,
        request: RobustnessRequestV2,
        baseline_outcomes: Dict[str, List[float]],
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
        critiques: Optional[List[CritiqueV2]] = None,
    ) -> List[FactorSensitivityResult]:
        """
        Compute sensitivity to factor node values.

        For each factor with uncertainty specified, measures how much
        the outcome changes when the factor value is varied by ±1 std
        (or ±10% of range for uniform distributions).

        Intervention vs Non-Intervention Factor Behavior
        ------------------------------------------------
        This function computes sensitivity for NON-INTERVENTION factors only
        (contextual variables like market conditions, user attributes, etc.).

        - **Non-intervention factors**: Variables that affect the outcome but
          are not directly controlled by the decision options. Sensitivity
          reflects how much uncertainty in these factors affects the decision.

        - **Intervention factors**: Variables set by options (e.g., marketing
          spend, pricing). These are NOT included in parameter_uncertainties
          because their values are determined by the option, not estimated.

        If a factor that is also an intervention target appears in
        parameter_uncertainties, its sensitivity may be zero or near-zero
        because the intervention value overrides the perturbed value.
        This is expected behavior—use zero_reason="intervention_override"
        to diagnose such cases.

        Lever identity (D-U ruling): a factor is an intervention target if
        ANY option intervenes on it (union across options), while elasticity
        itself is measured under the reference (first) option's interventions.
        A union-lever factor whose measured elasticity under the reference
        option is nonzero is still published with that elasticity; the
        zero_reason stamp applies only when elasticity ~ 0.

        Debug Fields
        ------------
        - elasticity: Raw (unclamped) value for determinism/audit
        - elasticity_display: Clamped to [-100, 100] for UI safety
        - zero_reason: Explains why sensitivity is zero (if applicable)
        - baseline_near_zero: True if epsilon denominator was applied
        """
        # Diagnostic: log entry point
        self.logger.info(
            "factor_sensitivity_entry",
            extra={
                "has_parameter_uncertainties": bool(request.parameter_uncertainties),
                "num_uncertainties": len(request.parameter_uncertainties)
                if request.parameter_uncertainties
                else 0,
                "uncertainties": [
                    {"node_id": u.node_id, "distribution": u.distribution, "std": u.std}
                    for u in (request.parameter_uncertainties or [])
                ],
            },
        )

        if not request.parameter_uncertainties:
            return []

        sensitivities: List[Dict[str, Any]] = []
        ref_option = request.options[0]
        baseline_mean = float(np.mean(baseline_outcomes[ref_option.id]))

        # Intervention factor IDs for INTERVENTION_OVERRIDE detection. D-U ruling
        # (union-across-options): a factor ANY option intervenes on is a lever — not
        # just the reference (first) option's targets. Shared source of truth with
        # _compute_factor_evppi's lever omission (derive-don't-mirror).
        intervention_factor_ids = self._intervention_factor_union(request)

        # Diagnostic: log baseline
        self.logger.info(
            "factor_sensitivity_baseline",
            extra={
                "ref_option_id": ref_option.id,
                "baseline_mean": baseline_mean,
                "baseline_outcomes_count": len(baseline_outcomes.get(ref_option.id, [])),
                # Sorted for deterministic log output
                "intervention_factor_ids": sorted(intervention_factor_ids),
            },
        )

        # Build node map for labels
        node_map = {n.id: n for n in request.graph.nodes}

        # Sample mean edge configuration for sensitivity analysis
        # (isolate factor sensitivity from edge uncertainty)
        mean_edge_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
        }

        for uncertainty in request.parameter_uncertainties:
            node = node_map.get(uncertainty.node_id)
            if not node:
                continue

            # THE FIX (2.1020). Perturb around, and normalise by, the value the
            # SAMPLER centres this factor on — not `observed_state.value else
            # 0.0`, which put a prior-only factor's probes outside its own
            # declared support and divided by the 0.01 epsilon.
            resolved = resolve_factor_central_value(node, uncertainty)
            mean_value = resolved.value
            # `observed_value` stays strictly the OBSERVED value: it is a
            # different question, and publishing a prior midpoint in a field
            # named "observed" would be a new fabrication (trap 21).
            observed_value = (
                resolved.value if resolved.source == FACTOR_VALUE_SOURCE_OBSERVED else None
            )

            # Determine perturbation amount based on distribution
            if uncertainty.distribution == "normal":
                delta = uncertainty.std or 0.0
            elif uncertainty.distribution == "uniform":
                range_min = uncertainty.range_min or 0.0
                range_max = uncertainty.range_max or 0.0
                delta = (range_max - range_min) * 0.1  # 10% of range
            else:
                # point_mass - no sensitivity
                delta = 0.0

            if delta == 0.0:
                # No uncertainty to measure
                sensitivities.append(
                    {
                        "node_id": uncertainty.node_id,
                        "node_label": node.label,
                        "elasticity": 0.0,
                        "elasticity_display": 0.0,
                        "observed_value": observed_value,
                        "interpretation": f"Factor {node.label} has no uncertainty (point mass)",
                        "zero_reason": ZeroSensitivityReason.POINT_MASS,
                        "baseline_near_zero": False,
                    }
                )
                continue

            # Evaluate with high and low values (single evaluation - deterministic given fixed inputs)
            factor_values_high = {uncertainty.node_id: mean_value + delta}
            outcome_high = evaluator.evaluate(
                edge_strengths=mean_edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
                factor_values=factor_values_high,
            )

            factor_values_low = {uncertainty.node_id: mean_value - delta}
            outcome_low = evaluator.evaluate(
                edge_strengths=mean_edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
                factor_values=factor_values_low,
            )

            outcome_diff = outcome_high - outcome_low

            # Compute true elasticity: (%Δ outcome) / (%Δ factor)
            # Use epsilon-stabilised denominators to handle near-zero baselines
            # (e.g., binary factors 0/1 where observed_state.value = 0)
            baseline_near_zero = abs(baseline_mean) < FACTOR_SENSITIVITY_BASELINE_EPSILON
            baseline_denom = max(abs(baseline_mean), FACTOR_SENSITIVITY_BASELINE_EPSILON)
            factor_denom = max(abs(mean_value), FACTOR_SENSITIVITY_VALUE_EPSILON)

            pct_outcome_change = outcome_diff / baseline_denom
            pct_factor_change = (2 * delta) / factor_denom
            # Raw elasticity is canonical (unclamped) for determinism
            elasticity = (
                pct_outcome_change / pct_factor_change if abs(pct_factor_change) > 1e-10 else 0.0
            )
            # Display elasticity is clamped for UI safety
            elasticity_display = max(-ELASTICITY_CLAMP_MAX, min(ELASTICITY_CLAMP_MAX, elasticity))

            # Determine zero_reason if elasticity is effectively zero
            # Priority order: INTERVENTION_OVERRIDE > ZERO_DELTA > ZERO_OUTCOME_DIFF > BASELINE_NORMALISED
            # (DISCONNECTED and POINT_MASS are handled elsewhere)
            zero_reason = None
            if abs(elasticity) < 1e-10:
                # Check if factor is overridden by intervention (highest priority)
                if uncertainty.node_id in intervention_factor_ids:
                    zero_reason = ZeroSensitivityReason.INTERVENTION_OVERRIDE
                # Check if delta is too small to perturb meaningfully
                elif abs(delta) < 1e-10:
                    zero_reason = ZeroSensitivityReason.ZERO_DELTA
                # Check if perturbation didn't affect outcome
                elif abs(outcome_diff) < 1e-10:
                    zero_reason = ZeroSensitivityReason.ZERO_OUTCOME_DIFF
                # Check if epsilon denominator was applied
                elif baseline_near_zero:
                    zero_reason = ZeroSensitivityReason.BASELINE_NORMALISED
                else:
                    # Computation resulted in zero (rare edge case)
                    zero_reason = ZeroSensitivityReason.ZERO_OUTCOME_DIFF

            # Diagnostic logging for factor sensitivity computation
            self.logger.info(
                "factor_sensitivity_computation",
                extra={
                    "node_id": uncertainty.node_id,
                    "node_label": node.label,
                    "baseline_mean": baseline_mean,
                    "mean_value": mean_value,
                    "delta": delta,
                    "outcome_high": outcome_high,
                    "outcome_low": outcome_low,
                    "outcome_diff": outcome_diff,
                    "baseline_denom": baseline_denom,
                    "factor_denom": factor_denom,
                    "pct_outcome_change": pct_outcome_change,
                    "pct_factor_change": pct_factor_change,
                    "elasticity": elasticity,
                    "elasticity_display": elasticity_display,
                    "zero_reason": zero_reason.value if zero_reason else None,
                    "baseline_near_zero": baseline_near_zero,
                },
            )

            sensitivities.append(
                {
                    "node_id": uncertainty.node_id,
                    "node_label": node.label,
                    "elasticity": elasticity,
                    "elasticity_display": elasticity_display,
                    "observed_value": observed_value,
                    "interpretation": self._interpret_factor_sensitivity(node.label, elasticity),
                    "zero_reason": zero_reason,
                    "baseline_near_zero": baseline_near_zero,
                }
            )

        # Compute structural influence for all factors
        factor_node_ids: List[str] = [s["node_id"] for s in sensitivities]
        influence_scores, influence_truncated = self._compute_structural_influence(
            request.graph, factor_node_ids, request.goal_node_id
        )
        # N1 (Codex re-confirm, D-23.19): EXACT-OR-NULL. Normalized scores of a
        # truncated cohort are NOT lower bounds (the data-dependent max
        # denominator can shrink faster than a numerator — their repro inflated
        # exact 0.1 to bounded 1.0 and inverted ranks). When ANY factor
        # truncates, withhold ALL scores and ranks for the cohort and say why.
        influence_exact = not influence_truncated
        if influence_truncated and critiques is not None:
            critiques.append(
                STRUCTURAL_INFLUENCE_TRUNCATED.build(
                    factor_ids=", ".join(sorted(influence_truncated)),
                    budget=MAX_INFLUENCE_WALK_CALLS_TOTAL,
                    affected_node_ids=sorted(influence_truncated),
                    seed=rng.seed,  # resolved int (request.seed may be a str alias)
                )
            )

        # Add influence scores to sensitivities (None when the cohort truncated —
        # a normalized score is only ever published when it is exact).
        for s in sensitivities:
            s["influence_score"] = (
                influence_scores.get(str(s["node_id"]), 0.0) if influence_exact else None
            )

        # Sort by absolute elasticity for importance_rank
        sensitivities.sort(key=lambda x: abs(float(x["elasticity"])), reverse=True)

        # Compute influence_rank (sort by influence_score descending) — withheld
        # entirely when the cohort truncated (rank on withheld scores would just
        # re-publish the unsound ordering).
        if influence_exact:
            sorted_by_influence = sorted(
                sensitivities, key=lambda x: float(x["influence_score"]), reverse=True
            )
            influence_rank_map: Dict[str, Optional[int]] = {
                s["node_id"]: i + 1 for i, s in enumerate(sorted_by_influence)
            }
        else:
            influence_rank_map = {s["node_id"]: None for s in sensitivities}

        # --- Bootstrap stability analysis (3C) ---
        # Measures stability of attribution under model and sampling uncertainty:
        # how consistently each factor ranks as important when we resample edge
        # configurations. This is NOT "confidence in the causal relationship"
        # (which requires observational/experimental data we don't have).
        primary_elasticities: Dict[str, float] = {
            str(s["node_id"]): float(s["elasticity"]) for s in sensitivities
        }
        # sensitivities is already sorted by |elasticity| desc, so index+1 = importance_rank
        primary_ranks: Dict[str, int] = {
            str(s["node_id"]): i + 1 for i, s in enumerate(sensitivities)
        }
        bootstrap_stability = self._compute_bootstrap_stability(
            request,
            baseline_mean,
            ref_option,
            evaluator,
            rng,
            primary_elasticities,
            primary_ranks,
            n_bootstrap_override=self._n_bootstrap_override,
        )

        # Convert to results with ranks
        results = []
        rank = 0
        for i, s in enumerate(sensitivities):
            # 2.514(a), second half: `elasticity` is a REQUIRED float, so a factor
            # whose elasticity could not be computed (the reference option's
            # outcomes were non-finite, making the % change nan) has NO honest row
            # at all — there is no field to null. Emitting it anyway put a nan in
            # the body and the JSONResponse render killed the WHOLE 200, taking
            # every critique with it; that is why fixing `elasticity_std` alone did
            # not clear the 500 on this input. Omit the row instead. If that leaves
            # the list empty, the EXISTING response_builder derivation reports
            # `factor_sensitivity_status` honestly rather than claiming "computed".
            if not math.isfinite(float(s["elasticity"])):
                self.logger.warning(
                    "factor_sensitivity_non_finite_elasticity",
                    extra={"node_id": str(s["node_id"])},
                )
                continue
            rank += 1
            # Update zero_reason: DISCONNECTED takes priority if factor has no causal path.
            # N1 corollary: under a truncated cohort the influence score is withheld
            # (None) AND unreliable — a factor whose productive path was beyond the
            # budget would read as "no path" here, so the DISCONNECTED inference is
            # only sound when the enumeration was exact.
            zero_reason = s.get("zero_reason")  # type: ignore[assignment]
            if (
                influence_exact
                and abs(float(s["elasticity"])) < 1e-10
                and float(s["influence_score"]) < 1e-10
            ):
                # Factor is disconnected (no causal path to goal)
                # This overrides ZERO_OUTCOME_DIFF since disconnection is the root cause
                zero_reason = ZeroSensitivityReason.DISCONNECTED

            node_id = str(s["node_id"])
            bs = bootstrap_stability.get(node_id, {})

            results.append(
                FactorSensitivityResult(
                    node_id=node_id,
                    node_label=s["node_label"],
                    elasticity=s["elasticity"],
                    elasticity_display=s.get("elasticity_display"),
                    importance_rank=rank,
                    observed_value=s["observed_value"],
                    interpretation=s["interpretation"],
                    zero_reason=zero_reason,
                    baseline_near_zero=s.get("baseline_near_zero"),
                    influence_score=s["influence_score"],  # None when cohort truncated (N1)
                    influence_rank=influence_rank_map[s["node_id"]],  # None when cohort truncated
                    elasticity_std=bs.get("elasticity_std"),
                    attribution_stability=bs.get("attribution_stability"),
                    rank_flip_rate=bs.get("rank_flip_rate"),
                    stability_method=bs.get("stability_method"),
                )
            )

        return results

    def _interpret_factor_sensitivity(self, node_label: str, elasticity: float) -> str:
        """Generate human-readable interpretation for factor sensitivity."""
        if abs(elasticity) < 0.05:
            return f"Decision is robust to {node_label} value variation"
        elif abs(elasticity) < self.HIGH_SENSITIVITY_THRESHOLD:
            return f"Decision is moderately sensitive to {node_label} value"
        else:
            return (
                f"Decision is highly sensitive to {node_label} value - "
                "consider narrowing uncertainty or gathering more data"
            )

    def _compute_conditional_winners(
        self,
        factor_values_per_sample: List[Dict[str, float]],
        winner_per_sample: List[Optional[str]],
        option_outcomes: Dict[str, List[float]],
        factor_sampler: "FactorSampler",
        request: RobustnessRequestV2,
        min_bucket_size: int = 50,
    ) -> Optional[List[ConditionalWinner]]:
        """
        Compute conditional win probabilities by partitioning MC samples at
        each factor's median value.

        For each non-point-mass factor, splits samples into low (< median) and
        high (>= median) buckets and computes win probabilities within each.
        Reports only factors where the winner flips between buckets.

        Limitation: median split is simplistic. It cannot detect non-monotonic
        effects, factor interactions, or flips at extreme quantiles.

        Args:
            factor_values_per_sample: Factor values sampled per MC iteration
            winner_per_sample: Winning option ID per MC sample
            option_outcomes: Per-option outcome values (for tie-breaking by mean)
            factor_sampler: FactorSampler (for uncertainty and node lookups)
            request: The analysis request (for option labels)
            min_bucket_size: Minimum samples per bucket (skip if fewer)

        Returns:
            List of ConditionalWinner where winner_flips is True,
            or None if no flips found.
        """
        if len(request.options) <= 1:
            return None
        if not factor_values_per_sample or not factor_sampler.has_uncertainties():
            return None

        option_labels = {opt.id: (opt.label or opt.id) for opt in request.options}
        option_means = {opt_id: float(np.mean(vals)) for opt_id, vals in option_outcomes.items()}

        results: List[ConditionalWinner] = []

        for factor_id, uncertainty in factor_sampler.get_uncertainty_map().items():
            if uncertainty.distribution == "point_mass":
                continue

            # Extract factor values across all samples
            values = np.array([fv.get(factor_id, np.nan) for fv in factor_values_per_sample])
            if np.any(np.isnan(values)):
                continue

            median = float(np.median(values))

            # Partition into low/high buckets
            low_mask = values < median
            high_mask = ~low_mask

            low_count = int(np.sum(low_mask))
            high_count = int(np.sum(high_mask))

            if low_count < min_bucket_size or high_count < min_bucket_size:
                continue

            # Compute bucket winners
            low_bucket = self._compute_bucket_result(
                low_mask, winner_per_sample, option_labels, option_means
            )
            high_bucket = self._compute_bucket_result(
                high_mask, winner_per_sample, option_labels, option_means
            )

            winner_flips = low_bucket.winner_id != high_bucket.winner_id

            if not winner_flips:
                continue

            # Get node metadata
            node = factor_sampler.get_node(factor_id)
            factor_label = node.label if node else factor_id
            split_unit = (
                node.observed_state.unit
                if node and node.observed_state and node.observed_state.unit
                else None
            )

            results.append(
                ConditionalWinner(
                    factor_id=factor_id,
                    factor_label=factor_label,
                    split_value=median,
                    split_unit=split_unit,
                    low_bucket=low_bucket,
                    high_bucket=high_bucket,
                    winner_flips=True,
                )
            )

        return results if results else None

    def _compute_bucket_result(
        self,
        mask: np.ndarray,
        winner_per_sample: List[Optional[str]],
        option_labels: Dict[str, str],
        option_means: Dict[str, float],
    ) -> BucketResult:
        """Compute win probabilities within a bucket of MC samples."""
        indices = np.where(mask)[0]
        bucket_size = len(indices)

        # Count wins per option in this bucket. 2.477(c): a draw where no option
        # was finite has no winner (None) and is skipped — never counted as a
        # phantom option that could then win the bucket.
        win_counts: Dict[str, int] = {}
        for idx in indices:
            winner = winner_per_sample[idx]
            if winner is None:
                continue
            win_counts[winner] = win_counts.get(winner, 0) + 1

        # Determine bucket winner (ties broken by higher mean outcome)
        sorted_options = sorted(
            win_counts.items(),
            key=lambda x: (x[1], option_means.get(x[0], 0.0)),
            reverse=True,
        )

        winner_id = sorted_options[0][0]
        winner_prob = sorted_options[0][1] / bucket_size

        runner_up_id = None
        runner_up_prob = None
        if len(sorted_options) > 1:
            runner_up_id = sorted_options[1][0]
            runner_up_prob = sorted_options[1][1] / bucket_size

        return BucketResult(
            n_samples=bucket_size,
            winner_id=winner_id,
            winner_label=option_labels.get(winner_id, winner_id),
            winner_probability=winner_prob,
            runner_up_id=runner_up_id,
            runner_up_probability=runner_up_prob,
        )

    def _compute_bootstrap_stability(
        self,
        request: RobustnessRequestV2,
        baseline_mean: float,
        ref_option: InterventionOption,
        evaluator: SCMEvaluatorV2,
        rng: SeededRNG,
        primary_elasticities: Dict[str, float],
        primary_ranks: Dict[str, int],
        n_bootstrap_override: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute bootstrap stability metrics for factor sensitivity (3C).

        Measures stability of attribution under model and sampling uncertainty:
        how consistently each factor ranks as important when we resample edge
        configurations. This is NOT "confidence in the causal relationship"
        (which requires observational/experimental data we don't have).

        Adaptive budget (when n_bootstrap_override is None):
        - Start at N_BOOTSTRAP=10. If wall-clock < 100ms, increase to 20.
        - If > 200ms at 10, keep 10 and log budget_exceeded.
        - Jackknife fallback is not implemented: the current sensitivity code
          uses deterministic mean-edge-config evaluation (not per-MC-sample
          elasticities), so delete-d jackknife cannot recompute elasticity
          without re-running the full MC. Bootstrap at 10 iterations is
          sub-millisecond per iteration, making the fallback unnecessary.

        Args:
            request: The robustness request (graph, parameter_uncertainties, etc.)
            baseline_mean: Mean outcome from the primary MC run
            ref_option: Reference option for evaluation
            evaluator: SCM evaluator
            rng: Primary RNG (bootstrap seeds derived from rng.seed)
            primary_elasticities: Dict[node_id -> elasticity] from deterministic computation
            primary_ranks: Dict[node_id -> importance_rank] from the main sensitivity run
            n_bootstrap_override: If set, skip adaptive budget and run exactly this many
                iterations. Used by tests to ensure timing-independent determinism.

        Returns:
            Dict[node_id -> {elasticity_std, attribution_stability, rank_flip_rate, stability_method}]
        """
        if not request.parameter_uncertainties:
            return {}

        primary_seed = rng.seed
        budget_exceeded = False
        t0 = time.time()

        if n_bootstrap_override is not None:
            # Fixed count — skip adaptive budget (used by tests)
            bootstrap_elasticities = self._run_bootstrap_iterations(
                request,
                baseline_mean,
                ref_option,
                evaluator,
                primary_seed,
                n_bootstrap_override,
            )
            n_bootstrap = n_bootstrap_override
        else:
            # --- Phase 1: run 10 bootstrap iterations ---
            n_bootstrap_initial = 10
            bootstrap_elasticities = self._run_bootstrap_iterations(
                request,
                baseline_mean,
                ref_option,
                evaluator,
                primary_seed,
                n_bootstrap_initial,
            )
            t1 = time.time()
            elapsed_ms = (t1 - t0) * 1000

            # --- Phase 2: adaptive budget ---
            n_bootstrap = n_bootstrap_initial
            if elapsed_ms < 100:
                # Budget allows 20 runs; add 10 more
                extra = self._run_bootstrap_iterations(
                    request,
                    baseline_mean,
                    ref_option,
                    evaluator,
                    primary_seed + n_bootstrap_initial,
                    10,
                )
                # Merge: append elasticities for each node
                for node_id in bootstrap_elasticities:
                    bootstrap_elasticities[node_id].extend(extra.get(node_id, []))
                n_bootstrap = 20
            elif elapsed_ms > 200:
                budget_exceeded = True
                self.logger.info(
                    "factor_sensitivity_bootstrap_budget_exceeded",
                    extra={
                        "elapsed_ms": round(elapsed_ms, 1),
                        "n_bootstrap": n_bootstrap_initial,
                        "budget_exceeded": True,
                    },
                )

        stability_method = f"bootstrap_{n_bootstrap}"

        # --- Phase 3: compute stability metrics ---
        # Collect per-bootstrap rank arrays for rank_flip_rate
        n_factors = len(bootstrap_elasticities)
        node_ids = list(bootstrap_elasticities.keys())

        # Build per-bootstrap-run elasticity matrix: [run_index][factor_index]
        n_runs = n_bootstrap
        # bootstrap_elasticities[node_id] is a list of length n_runs
        elasticity_matrix = []
        for run_idx in range(n_runs):
            run_elasticities = {}
            for nid in node_ids:
                vals = bootstrap_elasticities[nid]
                run_elasticities[nid] = vals[run_idx] if run_idx < len(vals) else 0.0
            elasticity_matrix.append(run_elasticities)

        # Compute per-bootstrap importance ranks (by |elasticity|, descending)
        per_run_ranks: Dict[str, List[int]] = {nid: [] for nid in node_ids}
        for run_elasticities in elasticity_matrix:
            sorted_ids = sorted(node_ids, key=lambda nid: abs(run_elasticities[nid]), reverse=True)
            for rank_idx, nid in enumerate(sorted_ids):
                per_run_ranks[nid].append(rank_idx + 1)

        result: Dict[str, Dict[str, Any]] = {}
        for nid in node_ids:
            elasticities = np.array(bootstrap_elasticities[nid])
            e_std = float(np.std(elasticities, ddof=1)) if len(elasticities) > 1 else 0.0

            # Use primary (deterministic) elasticity for the negligible check,
            # not the bootstrap mean — the primary value is the reported number.
            primary_e = primary_elasticities.get(nid, 0.0)

            # 2.514(a): a bootstrap population containing non-finite elasticities
            # has NO honest stability summary, and reporting one was doing damage
            # in two directions at once:
            #  * `elasticity_std` is `ge=0`, so a nan raised a pydantic
            #    ValidationError and the ENTIRE analysis 500'd — taking every
            #    critique with it, including the MONTE_CARLO_FAILED that named the
            #    option responsible (the 2.477 pattern exactly);
            #  * `classify_attribution_stability` compares the CV against
            #    thresholds, and EVERY comparison against nan is False, so the
            #    factor fell through to the confident label "low". That is a
            #    fabricated classification, not a measurement — the same family as
            #    the fabricated regret 0.0 this branch fixes.
            # The whole stability sub-block is therefore ABSENT together (all three
            # fields derive from the same population; part-null would invite the
            # reader to trust the survivors). `stability_method` is kept: it
            # records which method was ATTEMPTED, which stays true.
            # Every run that could serialize before had a finite e_std by
            # construction, so this branch is never taken on those and their bytes
            # are unchanged.
            if not math.isfinite(e_std) or not math.isfinite(primary_e):
                self.logger.warning(
                    "factor_stability_non_finite_population",
                    extra={"node_id": nid},
                )
                result[nid] = {
                    "elasticity_std": None,
                    "attribution_stability": None,
                    "rank_flip_rate": None,
                    "stability_method": stability_method,
                }
                continue

            # Attribution stability from coefficient of variation
            # Thresholds are configurable via STABILITY_THRESHOLDS (provisional)
            attribution_stability = classify_attribution_stability(
                primary_e, e_std, STABILITY_THRESHOLDS
            )

            # Rank flip rate: fraction of runs where rank shifts by >= 2
            # compared to the reported primary importance_rank (the rank users see).
            base_rank = primary_ranks.get(nid, 1)
            ranks = per_run_ranks[nid]
            flips = sum(1 for r in ranks if abs(r - base_rank) >= 2)
            rank_flip_rate = flips / len(ranks) if ranks else 0.0

            result[nid] = {
                "elasticity_std": round(e_std, 8),
                "attribution_stability": attribution_stability,
                "rank_flip_rate": round(rank_flip_rate, 4),
                "stability_method": stability_method,
            }

        self.logger.info(
            "factor_sensitivity_bootstrap_complete",
            extra={
                "n_bootstrap": n_bootstrap,
                "stability_method": stability_method,
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
                "budget_exceeded": budget_exceeded,
                "n_factors": n_factors,
            },
        )

        return result

    def _run_bootstrap_iterations(
        self,
        request: RobustnessRequestV2,
        baseline_mean: float,
        ref_option: InterventionOption,
        evaluator: SCMEvaluatorV2,
        seed_offset: int,
        n_iterations: int,
    ) -> Dict[str, List[float]]:
        """
        Run N bootstrap iterations of factor sensitivity with resampled edge configs.

        Each iteration:
        1. Create a new RNG from (seed_offset + iteration_index)
        2. Sample a fresh edge configuration via DualUncertaintySampler
        3. Compute elasticity for each factor using that edge config

        Args:
            request: Robustness request
            baseline_mean: Baseline outcome mean from primary MC
            ref_option: Reference option
            evaluator: SCM evaluator
            seed_offset: Base seed for this batch of iterations
            n_iterations: Number of iterations to run

        Returns:
            Dict[node_id -> List[elasticity_values]] across iterations
        """
        node_map = {n.id: n for n in request.graph.nodes}
        param_uncertainties = request.parameter_uncertainties or []
        result: Dict[str, List[float]] = {
            u.node_id: [] for u in param_uncertainties if node_map.get(u.node_id)
        }

        for i in range(n_iterations):
            # Deterministic seed derived from primary seed + bootstrap index
            boot_rng = SeededRNG(seed_offset + i)
            boot_sampler = DualUncertaintySampler(request.graph.edges, boot_rng)
            edge_config = boot_sampler.sample_edge_configuration()

            for uncertainty in param_uncertainties:
                node = node_map.get(uncertainty.node_id)
                if not node:
                    continue

                # Same resolver as `_compute_factor_sensitivity` (2.1020) — the
                # bootstrap twin must centre and normalise identically or the
                # elasticity band would describe a different quantity from the
                # point estimate it bands.
                mean_value = resolve_factor_central_value(node, uncertainty).value

                if uncertainty.distribution == "normal":
                    delta = uncertainty.std or 0.0
                elif uncertainty.distribution == "uniform":
                    range_min = uncertainty.range_min or 0.0
                    range_max = uncertainty.range_max or 0.0
                    delta = (range_max - range_min) * 0.1
                else:
                    delta = 0.0

                if delta == 0.0:
                    result[uncertainty.node_id].append(0.0)
                    continue

                # Evaluate with this bootstrap's edge config
                factor_values_high = {uncertainty.node_id: mean_value + delta}
                outcome_high = evaluator.evaluate(
                    edge_strengths=edge_config,
                    interventions=ref_option.interventions,
                    goal_node=request.goal_node_id,
                    factor_values=factor_values_high,
                )

                factor_values_low = {uncertainty.node_id: mean_value - delta}
                outcome_low = evaluator.evaluate(
                    edge_strengths=edge_config,
                    interventions=ref_option.interventions,
                    goal_node=request.goal_node_id,
                    factor_values=factor_values_low,
                )

                outcome_diff = outcome_high - outcome_low
                baseline_denom = max(abs(baseline_mean), FACTOR_SENSITIVITY_BASELINE_EPSILON)
                factor_denom = max(abs(mean_value), FACTOR_SENSITIVITY_VALUE_EPSILON)
                pct_outcome_change = outcome_diff / baseline_denom
                pct_factor_change = (2 * delta) / factor_denom
                elasticity = (
                    pct_outcome_change / pct_factor_change
                    if abs(pct_factor_change) > 1e-10
                    else 0.0
                )

                result[uncertainty.node_id].append(elasticity)

        return result

    def _compute_structural_influence(
        self,
        graph: GraphV2,
        factor_node_ids: List[str],
        goal_node_id: str,
        max_walk_calls_total: Optional[int] = None,
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Compute structural influence score for each factor based on causal path strengths.

        Algorithm:
        1. For each factor, find all paths to goal_node_id
        2. For each path, compute path_strength = product of edge.strength.mean * exists_probability
        3. Factor influence = sum of absolute path strengths (multiple paths add)
        4. Normalize to 0-1 scale across all factors

        UC-2 (D-23.18, re-fixed per Codex N1/N2, D-23.19): enumeration is bounded
        by a REQUEST-WIDE walk-CALL pool (``max_walk_calls_total``) shared across
        all factors in factor order — a per-factor reset multiplied worst-case
        work by U (the original F2 class). The pool ceiling is priced 1:1 in
        compute_weighted_cost (`structural_influence` term).

        ⚠ N1 (P0, Codex): a truncated factor's RAW path sum is a lower bound, but
        the NORMALIZED score is NOT — the data-dependent max-denominator can
        shrink faster than a numerator, inflating other factors' normalized
        scores and inverting ranks (their repro: exact 0.1 → bounded 1.0). The
        CALLER must therefore treat any non-empty ``truncated_factor_ids`` as
        exact-or-null: withhold ALL influence scores/ranks for the cohort and
        disclose via STRUCTURAL_INFLUENCE_TRUNCATED. Never publish the
        normalized values of a truncated cohort as bounds of anything.

        Args:
            graph: Causal graph with edges
            factor_node_ids: List of factor node IDs to compute influence for
            goal_node_id: Target goal node ID
            max_walk_calls_total: request-wide recursion-call pool (bounds
                dead-branch exploration as well as completed paths)

        Returns:
            (influences, truncated_factor_ids): node_id -> influence_score (0-1,
            normalized; only meaningful when truncated_factor_ids is empty),
            plus the factors whose enumeration exhausted the shared pool.
        """
        # Late-bind the module constant (a def-time default would freeze it,
        # silently no-opping test overrides and any future env tuning).
        if max_walk_calls_total is None:
            max_walk_calls_total = MAX_INFLUENCE_WALK_CALLS_TOTAL

        # Build adjacency list for path finding
        adjacency: Dict[str, List[Tuple[str, float]]] = {}
        for edge in graph.edges:
            from_node = edge.from_
            to_node = edge.to
            # Effective strength = mean * exists_probability
            effective_strength = edge.strength.mean * edge.exists_probability
            if from_node not in adjacency:
                adjacency[from_node] = []
            adjacency[from_node].append((to_node, effective_strength))

        # REQUEST-WIDE walk pool (N2): consumed across all factors in factor
        # order, never reset — the per-factor reset was the unpriced U-multiplier.
        calls_left = max_walk_calls_total
        budget_hit = False

        def find_all_paths_strengths(
            start: str,
            end: str,
            visited: set,
        ) -> List[float]:
            """
            Find all paths from start to end and return list of path strengths.
            Each path strength is the product of edge strengths along the path.
            Stops (returning what it has) once the walk-call budget is exhausted.
            """
            nonlocal calls_left, budget_hit
            if calls_left <= 0:
                budget_hit = True
                return []
            calls_left -= 1

            if start == end:
                return [1.0]  # Base case: path of strength 1

            if start in visited:
                return []  # Cycle detection

            if start not in adjacency:
                return []  # No outgoing edges

            visited.add(start)
            path_strengths = []

            for next_node, edge_strength in adjacency[start]:
                sub_paths = find_all_paths_strengths(next_node, end, visited.copy())
                for sub_strength in sub_paths:
                    path_strengths.append(edge_strength * sub_strength)

            return path_strengths

        # Compute raw influence for each factor, draining the SHARED pool.
        # A factor that starts (or continues) after exhaustion is truncated too —
        # its enumeration is incomplete by construction.
        raw_influences: Dict[str, float] = {}
        truncated_factors: List[str] = []
        for node_id in factor_node_ids:
            budget_hit = False
            path_strengths = find_all_paths_strengths(node_id, goal_node_id, set())
            # Sum of absolute path strengths (multiple paths add)
            raw_influences[node_id] = sum(abs(s) for s in path_strengths)
            # An exhausted pool trips budget_hit on the factor's first walk call,
            # so factors that start after exhaustion are truncated too.
            if budget_hit:
                truncated_factors.append(str(node_id))

        # Normalize to 0-1 scale
        max_influence = max(raw_influences.values()) if raw_influences else 0.0
        if max_influence < 1e-10:
            # All factors have zero influence
            return {node_id: 0.0 for node_id in factor_node_ids}, truncated_factors

        return {
            node_id: raw_influences[node_id] / max_influence for node_id in factor_node_ids
        }, truncated_factors

    @staticmethod
    def _format_path_mechanism(
        status: str,
        chain: str,
        path_effect: float,
        total_effect: float,
        share: Optional[float],
    ) -> str:
        """
        Build the user-facing mechanism string for a path contribution.

        Uses "modelled pathway contribution" framing; never asserts real-world
        causation and never expresses a percentage.
        """
        if status == "computed" and share is not None:
            return (
                f"Modelled pathway contribution along {chain}: signed path coefficient "
                f"{path_effect:+.3f} of total {total_effect:+.3f} (relative share {share:+.2f})."
            )
        return (
            f"Modelled pathway contribution along {chain}: signed path coefficient "
            f"{path_effect:+.3f}. Net modelled effect is near zero ({total_effect:+.3g}); "
            f"relative share is not defined."
        )

    def _compute_path_decomposition(
        self,
        request: RobustnessRequestV2,
        recommended_option_id: str,
        graph: GraphV2,
        budget_ms: Optional[float] = None,
    ) -> Optional[PathDecomposition]:
        """
        Structural pathway decomposition for the recommended option's retained
        intervention targets (analytic path tracing).

        Decomposes the modelled structural effect on the goal into the top-3 simple
        directed paths.  Per-edge coefficient and path strength match
        ``_compute_structural_influence`` exactly: ``strength.mean * exists_probability``
        (signed) multiplied along the path.  Unlike that function, the totals here stay
        SIGNED (no abs, no normalization) so opposing paths can cancel.  Path effects are
        NOT scaled by intervention magnitude — this is structural, not an option-level
        effect-size estimate.

        ``graph`` must be the post-filter graph the SCM computed on (pass
        ``evaluator.graph``): decision/option/constraint nodes and bidirected edges are
        therefore already absent from the interior; bidirected edges are skipped
        defensively as well.  The recommended option is carried as context/metadata;
        computed paths start at the retained intervention target/factor nodes.

        Path enumeration is bounded by ``MAX_DECOMPOSITION_PATHS``: a layered DAG valid
        under the schema's node/edge limits can have hundreds of thousands of simple paths,
        so if the budget is exceeded the result is returned with ``truncated=True`` and no
        ranked paths.  The bound is a path count (not wall-clock), so truncation is
        deterministic for a given graph.

        ``budget_ms`` (Codex F7) additionally bounds wall-clock: the path-count cap is
        deterministic but a dense DAG can spend real time before hitting it, so the walk
        re-checks a monotonic deadline mid-enumeration and returns ``None`` (all-or-nothing,
        distinct from ``truncated``) when it overruns, letting analyze() disclose
        PATH_DECOMPOSITION_UNAVAILABLE.  ``None`` disables the guard — direct/legacy callers
        (which pass no budget) are unaffected and always receive a ``PathDecomposition``.
        """
        option = next((o for o in request.options if o.id == recommended_option_id), None)
        if option is None:
            # Defensive: recommended_option_id always names a real option in practice.
            return PathDecomposition(
                recommended_option_id=recommended_option_id, entry_nodes=[], paths=[]
            )

        node_ids = {node.id for node in graph.nodes}
        node_label = {node.id: (node.label or node.id) for node in graph.nodes}
        node_kind = {node.id: node.kind.lower() for node in graph.nodes}
        goal = request.goal_node_id

        # Retained intervention targets the paths start from (sorted for stable output).
        entry_nodes = sorted(n for n in option.interventions if n in node_ids)
        if not entry_nodes:
            # Every intervention target was filtered out — the model computed no effect
            # through them; report the option as context with no paths (honest signal).
            return PathDecomposition(
                recommended_option_id=recommended_option_id, entry_nodes=[], paths=[]
            )

        # Build adjacency exactly like _compute_structural_influence (signed coeff,
        # list-valued to preserve parallel edges), skipping bidirected/confounding edges.
        adjacency: Dict[str, List[Tuple[str, float]]] = {}
        for edge in graph.edges:
            if getattr(edge, "edge_type", None) == "bidirected":
                continue
            coeff = edge.strength.mean * edge.exists_probability
            adjacency.setdefault(edge.from_, []).append((edge.to, coeff))

        # Enumerate simple intervention-target-to-goal paths, bounded by a path-count
        # budget (MAX_DECOMPOSITION_PATHS). Accumulate top-down so enumeration can stop
        # early and total work is capped; this yields the same per-edge products as the
        # recursive enumerator in _compute_structural_influence. The visited set excludes
        # cycles and parallel edges branch via the adjacency list.
        #
        # Truncate only once a path BEYOND the budget is discovered (len > cap, i.e. at
        # most cap+1 paths enumerated). A graph with exactly MAX_DECOMPOSITION_PATHS paths
        # is therefore fully ranked, not truncated — matching the "exceeded the budget"
        # contract on PathDecomposition.truncated.
        # F7: internal wall-clock deadline. deadline anchors t0 at the enumeration
        # phase; budget_ms is the remaining governing request budget at entry
        # (monotonic — NTP-step-safe). The PATH_DEADLINE_CHECK_INTERVAL cadence
        # stays local at the walk() re-check below.
        deadline = PhaseDeadline(budget_ms)
        deadline_hit = False
        walk_calls = 0

        all_paths: List[Tuple[List[str], float]] = []
        truncated = False

        def walk(node: str, effect_so_far: float, path_so_far: List[str], visited: set) -> None:
            nonlocal truncated, deadline_hit, walk_calls
            if truncated or deadline_hit:
                return
            # F7: periodic wall-clock deadline re-check inside the recursion (the
            # MAX_DECOMPOSITION_PATHS count cap is deterministic but not wall-clock).
            # Unwinds like `truncated`; the caller discards the phase (all-or-nothing).
            walk_calls += 1
            if walk_calls % self.PATH_DEADLINE_CHECK_INTERVAL == 0 and deadline.exceeded():
                deadline_hit = True
                return
            if node == goal:
                all_paths.append((path_so_far, effect_so_far))
                if len(all_paths) > MAX_DECOMPOSITION_PATHS:
                    truncated = True
                return
            if node in visited or node not in adjacency:
                return
            visited = visited | {node}
            for next_node, coeff in adjacency[node]:
                if truncated:
                    return
                # Never route THROUGH an organisational decision/option/constraint node
                # in the interior (already removed from the post-filter graph; this keeps
                # the guarantee explicit and robust to direct calls). Entry nodes may be
                # any kind; the goal is never non-inference.
                if next_node != goal and node_kind.get(next_node) in NON_INFERENCE_KINDS:
                    continue
                walk(next_node, effect_so_far * coeff, path_so_far + [next_node], visited)

        # F7: bail before enumerating if the budget is already spent at phase entry.
        if deadline.exceeded():
            self.logger.info(
                "path_decomposition_budget_exceeded",
                extra={
                    "elapsed_ms": deadline.elapsed_ms(),
                    "phase": "pre_enumeration",
                },
            )
            return None

        for entry in entry_nodes:
            if truncated or deadline_hit:
                break
            if entry == goal:
                # Skip the trivial zero-length path (an intervention target that is the
                # goal contributes no pathway structure).
                continue
            walk(entry, 1.0, [entry], set())

        if deadline_hit:
            # Wall-clock deadline tripped mid-enumeration — discard the whole phase
            # (all-or-nothing, no partial ranking) and let analyze() disclose.
            self.logger.info(
                "path_decomposition_budget_exceeded",
                extra={
                    "elapsed_ms": deadline.elapsed_ms(),
                    "paths_found": len(all_paths),
                },
            )
            return None

        if truncated:
            # More simple paths than the budget allows (a path beyond the cap was found).
            # Degrade gracefully (no ranked paths) rather than spend unbounded time; report
            # the cap as path_count — the true count is higher. truncated=True is
            # deterministic and distinct from the no-reachable-path case below.
            return PathDecomposition(
                recommended_option_id=recommended_option_id,
                entry_nodes=entry_nodes,
                truncated=True,
                path_count=MAX_DECOMPOSITION_PATHS,
                paths=[],
            )

        path_count = len(all_paths)  # exact; 0 <= path_count <= MAX_DECOMPOSITION_PATHS

        if not all_paths:
            return PathDecomposition(
                recommended_option_id=recommended_option_id,
                entry_nodes=entry_nodes,
                truncated=False,
                path_count=0,
                paths=[],
            )

        total_effect = sum(effect for _, effect in all_paths)
        # Near-zero guard epsilon: matches the structural-influence near-zero guard above
        # (line ~2376) and the elasticity/E-value guards in this module — the established
        # "structurally negligible" threshold.
        epsilon = 1e-10
        indeterminate = abs(total_effect) < epsilon

        # Deterministic order: largest absolute effect first; full node tuple and signed
        # effect break ties (incl. parallel edges sharing a node sequence).
        all_paths.sort(key=lambda pe: (-abs(pe[1]), tuple(pe[0]), pe[1]))

        contributions: List[PathContribution] = []
        for path, effect in all_paths[:3]:
            if indeterminate:
                status = "indeterminate"
                share: Optional[float] = None
            else:
                status = "computed"
                share = effect / total_effect
            chain = " → ".join(node_label.get(n, n) for n in path)
            contributions.append(
                PathContribution(
                    path=path,
                    path_effect=effect,
                    total_effect=total_effect,
                    signed_contribution=share,
                    status=status,
                    mechanism=self._format_path_mechanism(
                        status, chain, effect, total_effect, share
                    ),
                )
            )

        return PathDecomposition(
            recommended_option_id=recommended_option_id,
            entry_nodes=entry_nodes,
            truncated=False,
            path_count=path_count,
            paths=contributions,
        )

    def _compute_robustness(
        self,
        option_wins: Dict[str, float],
        winner_per_sample: List[Optional[str]],
        sensitivity: List[SensitivityResult],
        request: RobustnessRequestV2,
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]],
        evaluator: SCMEvaluatorV2,
        global_seed: int,
        n_defaulted_roots: int = 0,
        defaulted_root_node_ids: Optional[List[str]] = None,
        critiques: Optional[List[CritiqueV2]] = None,
    ) -> RobustnessResult:
        """Compute overall robustness assessment with alternative winner analysis."""
        # Recommendation stability: fraction of samples with same winner
        n_samples = request.n_samples
        most_frequent_winner = max(option_wins, key=lambda k: option_wins[k])
        recommendation_stability = option_wins[most_frequent_winner] / n_samples

        # Trust downgrade: penalise stability when root nodes defaulted to 0.0,
        # since the model is running with missing inputs.
        stability_penalty_factor = None
        if n_defaulted_roots > 0:
            stability_penalty_factor = max(0.1, 1.0 - 0.05 * n_defaulted_roots)
            recommendation_stability *= stability_penalty_factor

        # Identify fragile and robust edges (by edge_id string)
        # IMPORTANT: Aggregate sensitivities per edge BEFORE categorization
        # Each edge may have multiple sensitivity entries (existence + magnitude)
        # Use max(abs(elasticity)) to determine the edge's sensitivity level
        edge_max_elasticity: Dict[str, float] = {}
        edge_info: Dict[str, Tuple[str, str]] = {}  # edge_id -> (from_id, to_id)

        for sens in sensitivity:
            edge_id = f"{sens.edge_from}->{sens.edge_to}"
            current_max = edge_max_elasticity.get(edge_id, 0.0)
            edge_max_elasticity[edge_id] = max(current_max, abs(sens.elasticity))
            edge_info[edge_id] = (sens.edge_from, sens.edge_to)

        # Now categorize edges based on their max elasticity
        # Thresholds: fragile > 0.1, robust < 0.05, moderate = [0.05, 0.1]
        fragile_edge_ids = set()
        robust_edge_ids = set()
        fragile_edge_info: Dict[str, Tuple[str, str]] = {}

        for edge_id, max_elasticity in edge_max_elasticity.items():
            if max_elasticity > self.FRAGILE_THRESHOLD:
                fragile_edge_ids.add(edge_id)
                fragile_edge_info[edge_id] = edge_info[edge_id]
            elif max_elasticity < 0.05:
                robust_edge_ids.add(edge_id)
            # Edges with 0.05 <= elasticity <= 0.1 are implicitly "moderate" (uncategorized)

        # Canonical (sorted) order: set iteration order follows the per-process
        # string-hash salt, so list(...) here leaked process identity into the
        # response — and into the interpretation string's "sensitive to:" list
        # (science-validation report §3 cross-process finding, fix §5.7b).
        fragile_edges = sorted(fragile_edge_ids)
        robust_edges = sorted(robust_edge_ids)

        # Compute alternative winners for fragile edges (includes marginal calculation)
        fragile_edges_enhanced = self._compute_alternative_winners(
            fragile_edge_info,
            edge_configs_per_sample,
            winner_per_sample,
            most_frequent_winner,
            request,
            evaluator,
            global_seed,
            edge_max_elasticity=edge_max_elasticity,
            critiques=critiques,
        )

        # Overall robustness
        # Per Decision Model Schema v2.6: is_robust = recommendation_stability >= 0.7
        # fragile_edges is a separate indicator of edge-level sensitivity
        is_robust = recommendation_stability >= self.ROBUST_THRESHOLD

        # Arch step 1 (2026-07-26): the `confidence` slot now carries the
        # recommendation-stability fraction, unmodified. See
        # _stability_confidence_figure for why the old formula was withdrawn.
        confidence = self._stability_confidence_figure(recommendation_stability, n_samples)

        interpretation = self._build_robustness_interpretation(
            is_robust=is_robust,
            recommendation_stability=recommendation_stability,
            most_frequent_winner=most_frequent_winner,
            fragile_edges=fragile_edges,
        )

        return RobustnessResult(
            is_robust=is_robust,
            confidence=confidence,
            confidence_basis="recommendation_stability_uncalibrated",
            fragile_edges=fragile_edges,
            fragile_edges_enhanced=fragile_edges_enhanced,
            robust_edges=robust_edges,
            recommendation_stability=recommendation_stability,
            interpretation=interpretation,
            stability_penalty_factor=stability_penalty_factor,
            defaulted_root_node_ids=defaulted_root_node_ids if defaulted_root_node_ids else None,
        )

    # E-value budget: max wall-clock time for the full E-value sweep.
    # Paul-ruled lenient defaults 2026-07-17: raised 2000 → 8000. On budget
    # exceed the WHOLE edge_e_values field is omitted from the response
    # (disclosed only via the e_value_budget_exceeded log event), and bands
    # vanish with it — 2000 ms risked that silent loss on large graphs on
    # staging hardware. Value pinned by tests/unit/test_lenient_limits.py
    # (silent revert goes RED).
    E_VALUE_BUDGET_MS = 8000
    E_VALUE_BISECT_STEPS = 20  # binary search precision: 2^-20 ≈ 1e-6

    def _compute_edge_e_values(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        budget_ms: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Compute E-value analogue for each edge: minimum strength perturbation to flip winner.

        Uses binary search on strength.mean for each edge while holding all other edges
        at expected values. Returns None if computation exceeds budget.

        Args:
            request: The robustness request.
            evaluator: SCM evaluator instance.
            budget_ms: Effective wall-clock budget for this sweep. Defaults to
                E_VALUE_BUDGET_MS; the analyze() orchestrator passes
                min(E_VALUE_BUDGET_MS, remaining request budget) so the sweep
                also respects the governing overall-request deadline.

        Returns:
            List of dicts with e_value info per edge, or None if budget exceeded.
        """
        budget = budget_ms if budget_ms is not None else self.E_VALUE_BUDGET_MS
        # Monotonic clock: an NTP step must not corrupt the elapsed guard.
        t0 = time.monotonic()

        # Build expected-value baseline config
        baseline_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
        }

        # Determine baseline winner
        baseline_outcomes = {}
        for option in request.options:
            baseline_outcomes[option.id] = evaluator.evaluate(
                edge_strengths=baseline_config,
                interventions=option.interventions,
                goal_node=request.goal_node_id,
            )
        sorted_baseline = sorted(baseline_outcomes.items(), key=lambda x: (-x[1], x[0]))
        baseline_winner = sorted_baseline[0][0]

        results: List[Dict[str, Any]] = []
        for edge in request.graph.edges:
            # Skip non-causal (bidirected/confounder) edges
            if getattr(edge, "edge_type", None) == "bidirected":
                continue

            # Budget check per edge
            elapsed_ms = (time.monotonic() - t0) * 1000
            if elapsed_ms > budget:
                self.logger.info(
                    "e_value_budget_exceeded",
                    extra={"elapsed_ms": round(elapsed_ms, 1), "edges_completed": len(results)},
                )
                return None  # Budget exceeded — omit from response

            edge_key = (edge.from_, edge.to)
            current_mean = edge.strength.mean
            ep = edge.exists_probability

            # Try both directions: increase and decrease
            flip_found = False
            for direction in ("increase", "decrease"):
                if direction == "increase":
                    lo, hi = current_mean, EDGE_STRENGTH_MAX
                else:
                    lo, hi = EDGE_STRENGTH_MIN, current_mean

                # Quick check: does the extreme boundary flip the winner?
                test_config = baseline_config.copy()
                test_config[edge_key] = hi * ep if direction == "increase" else lo * ep
                test_outcomes = {}
                for option in request.options:
                    test_outcomes[option.id] = evaluator.evaluate(
                        edge_strengths=test_config,
                        interventions=option.interventions,
                        goal_node=request.goal_node_id,
                    )
                sorted_test = sorted(test_outcomes.items(), key=lambda x: (-x[1], x[0]))
                if sorted_test[0][0] == baseline_winner:
                    continue  # This direction cannot flip — skip

                # ROADMAP 2.228-F3: retain the argmax on the FLIPPED side, which
                # this search already computes and used to discard.
                #
                # Why tracking the most recent flipping evaluation is EXACT rather
                # than approximate: below, `hi` is assigned ONLY inside the
                # flip branch when direction == "increase", and `lo` ONLY inside it
                # when direction == "decrease". The reported flip_mean is exactly
                # that endpoint. So the argmax recorded at the last such assignment
                # IS the argmax at flip_mean. If no bisection step ever flipped, the
                # endpoint is still the boundary probed just above, whose argmax is
                # what we seed here. Zero extra evaluations either way.
                alternative_winner = sorted_test[0][0]

                # Binary search for the flip point
                for _ in range(self.E_VALUE_BISECT_STEPS):
                    # Inner budget check — abort if time exceeded mid-search
                    if (time.monotonic() - t0) * 1000 > budget:
                        self.logger.info(
                            "e_value_budget_exceeded",
                            extra={
                                "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                                "edges_completed": len(results),
                            },
                        )
                        return None
                    mid = (lo + hi) / 2
                    test_config = baseline_config.copy()
                    test_config[edge_key] = mid * ep
                    test_outcomes = {}
                    for option in request.options:
                        test_outcomes[option.id] = evaluator.evaluate(
                            edge_strengths=test_config,
                            interventions=option.interventions,
                            goal_node=request.goal_node_id,
                        )
                    sorted_test = sorted(test_outcomes.items(), key=lambda x: (-x[1], x[0]))
                    if sorted_test[0][0] != baseline_winner:
                        # Flip happened — narrow toward current_mean
                        alternative_winner = sorted_test[0][0]
                        if direction == "increase":
                            hi = mid
                        else:
                            lo = mid
                    else:
                        # No flip — narrow away from current_mean
                        if direction == "increase":
                            lo = mid
                        else:
                            hi = mid

                flip_mean = hi if direction == "increase" else lo

                # E-value: ratio of perturbation to current value
                if abs(current_mean) > 1e-10:
                    e_value = abs(flip_mean / current_mean)
                else:
                    # current_mean ≈ 0 — any nonzero flip_mean is infinite leverage
                    e_value = abs(flip_mean) / 1e-6 if abs(flip_mean) > 1e-10 else 1.0

                e_value = max(1.0, e_value)  # E-value is always >= 1.0

                results.append(
                    {
                        "edge_id": f"{edge.from_}->{edge.to}",
                        "from_id": edge.from_,
                        "to_id": edge.to,
                        "e_value": round(e_value, 4),
                        "flip_direction": direction,
                        "current_mean": current_mean,
                        "flip_mean": round(flip_mean, 6),
                        "alternative_winner_id": alternative_winner,
                        "baseline_winner_id": baseline_winner,
                    }
                )
                flip_found = True
                break  # Take first direction that flips

            if not flip_found:
                # Edge cannot flip the recommendation in either direction
                results.append(
                    {
                        "edge_id": f"{edge.from_}->{edge.to}",
                        "from_id": edge.from_,
                        "to_id": edge.to,
                        "e_value": float("inf"),
                        "flip_direction": "increase",
                        "current_mean": current_mean,
                        "flip_mean": current_mean,
                        # No flip exists, so there is no alternative winner to
                        # name. Emitting the runner-up here would manufacture a
                        # claim the search explicitly disproved.
                        "alternative_winner_id": None,
                        "baseline_winner_id": baseline_winner,
                    }
                )

        return results

    # Flip-stability sweep budget: max wall-clock for the full band sweep.
    # All-or-nothing on exceed: NO bands are attached (partial bands would
    # bias readers toward whichever edges happened to be computed first) and
    # the base edge_e_values are never affected. Mirrors E_VALUE_BUDGET_MS
    # semantics — band *presence* is budget-gated exactly as edge_e_values
    # presence already is; band *content* is fully deterministic.
    # 30000 ms is the Paul-ruled LENIENT default (17 Jul lenient-latency
    # amendment, raised from 2000: prioritise analysis quality; when the
    # budget does trip, the flip_stability_budget_exceeded event disclosing
    # elapsed_ms is the find-out-it-was-slow signal — never a silent cut).
    FLIP_STABILITY_BUDGET_MS = 30000

    def _sample_flip_backgrounds(
        self,
        request: RobustnessRequestV2,
        master_seed: int,
        n_seeds: int,
        tag: str,
    ) -> List[Dict[Tuple[str, str], float]]:
        """N sampled edge backgrounds for a stability sweep, one per child seed.

        Child seeds are SHA-256-derived (process-safe, NOT Python ``hash()``) from
        ``f"{master_seed}:{tag}:{i}"`` — the same sub-seed pattern as the per-edge
        marginal-switch and per-factor EVPI streams. Deriving them never consumes
        any existing RNG stream, so every other number in the response is
        unchanged by the sweep.

        ``tag`` NAMESPACES the sweep. Two sweeps sharing a tag would draw
        identical backgrounds, silently correlating what are meant to be
        independent stability statements; each caller therefore passes its own
        (edge bands: "flip_stability"; factor bands: "factor_flip_stability").
        Extracted rather than duplicated so the two tags are derived from one
        place and a test can observe both call sites.
        """
        backgrounds: List[Dict[Tuple[str, str], float]] = []
        for i in range(n_seeds):
            child_seed = int(
                hashlib.sha256(f"{master_seed}:{tag}:{i}".encode()).hexdigest()[:8], 16
            )
            sweep_sampler = DualUncertaintySampler(request.graph.edges, SeededRNG(child_seed))
            backgrounds.append(sweep_sampler.sample_edge_configuration())
        return backgrounds

    def _attach_flip_stability_bands(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        edge_e_values: List[Dict[str, Any]],
        master_seed: int,
        budget_ms: Optional[float] = None,
    ) -> bool:
        """Attach a seed-sweep stability band to each edge_e_values entry.

        Returns True when the full sweep completed and bands were attached;
        False when the wall-clock budget tripped (all-or-nothing: NOTHING is
        attached, and the caller discloses STABILITY_BANDS_UNAVAILABLE on the
        wire). ``budget_ms`` defaults to FLIP_STABILITY_BUDGET_MS; the analyze()
        orchestrator passes min(FLIP_STABILITY_BUDGET_MS, remaining request
        budget) so the sweep also respects the governing overall deadline.

        Track S Phase 1. DEFAULT-ON: computed whenever edge_e_values are
        (env gating removed 2026-07-17 per Paul's ruling — core
        functionality, no flag; rollback is a revert commit).

        Why: the single-point flip threshold (flip_mean) is searched against
        ONE background — every other edge held at its expected value — so a
        consumer sees one number with no indication of how much it moves
        under the graph's own stated uncertainty. The 2026-06-10 science-
        performance report recommends a stability band from a small seed
        sweep, with flip confidence based on band width.

        Method: N child seeds are SHA-256-derived from the master (request)
        seed — same request+seed therefore yields byte-identical bands, and
        the derivation never consumes any existing RNG stream. Each child
        seed samples ONE full edge configuration from the joint uncertainty
        (existence Bernoulli x truncated-normal strength — identical
        semantics to the main MC's DualUncertaintySampler). Each edge's flip
        point is then re-searched with the other edges held at that sampled
        background. Backgrounds are shared across edges within a seed
        (common random numbers), so bands are comparable across edges.

        Mutates each entry dict by adding a "stability" key:
          n_seeds, n_seeds_flipped, seed_flip_means (per-seed flip mean or
          None when that background admits no flip), and — when at least one
          seed flips — band_min / band_median / band_max / band_width.
          The band_* keys are OMITTED (not null) when nothing flips, matching
          the v2 wire's exclude_none serialisation so v1 (dict passthrough)
          and v2 (model) wires carry the same shape.

        BAND MEMBERSHIP SEMANTICS (bytes-checked 2026-07-17, live-proof
        follow-up): the base flip_mean is NOT a member of this sweep. The
        sweep's backgrounds are drawn from child seeds
        sha256(f"{master_seed}:flip_stability:{i}"), i = 0..n_seeds-1 — the
        master seed itself is never a background — and, stronger, the base
        flip_mean is not computed under ANY sampled background: it is
        searched against the expected-value baseline (_compute_edge_e_values
        holds every other edge at mean × exists_probability). The base point
        therefore MAY legitimately lie outside [band_min, band_max]
        (observed live: flip_mean −0.5534 vs band [−0.135, 0.4232] on a
        4/10-flip edge — expected behaviour, and itself a signal that the
        flip estimate is background-sensitive). Consumers must NOT assume
        flip_mean ∈ band; the mirrored consumer warning lives on
        FlipStabilityBandV2 in src/models/response_v2.py.
        """
        budget = budget_ms if budget_ms is not None else self.FLIP_STABILITY_BUDGET_MS
        # Monotonic clock: an NTP step must not corrupt the elapsed guard.
        t0 = time.monotonic()
        n_seeds = FLIP_STABILITY_N_SEEDS

        # One sampled background per child seed, shared across all edges.
        backgrounds = self._sample_flip_backgrounds(request, master_seed, n_seeds, "flip_stability")

        edges_by_key = {(e.from_, e.to): e for e in request.graph.edges}

        bands: List[Optional[Dict[str, Any]]] = []
        for entry in edge_e_values:
            edge = edges_by_key.get((entry["from_id"], entry["to_id"]))
            if edge is None:
                # Defensive: entries are built from the same edge list, so
                # this should be unreachable; skip rather than fail the sweep.
                bands.append(None)
                continue

            seed_flip_means: List[Optional[float]] = []
            for background in backgrounds:
                if (time.monotonic() - t0) * 1000 > budget:
                    self.logger.info(
                        "flip_stability_budget_exceeded",
                        extra={
                            "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                            "edges_completed": len(bands),
                            "n_seeds": n_seeds,
                        },
                    )
                    return False  # all-or-nothing: attach nothing
                flip_mean = self._flip_mean_under_background(request, evaluator, edge, background)
                seed_flip_means.append(round(flip_mean, 6) if flip_mean is not None else None)

            flipped = [v for v in seed_flip_means if v is not None]
            band: Dict[str, Any] = {
                "n_seeds": n_seeds,
                "n_seeds_flipped": len(flipped),
                "seed_flip_means": seed_flip_means,
            }
            if flipped:
                band_min = min(flipped)
                band_max = max(flipped)
                band["band_min"] = band_min
                band["band_median"] = round(float(statistics.median(flipped)), 6)
                band["band_max"] = band_max
                band["band_width"] = round(band_max - band_min, 6)
            bands.append(band)

        # Attach only after the full sweep completed within budget.
        for entry, computed_band in zip(edge_e_values, bands):
            if computed_band is not None:
                entry["stability"] = computed_band
        return True

    def _flip_mean_under_background(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        edge: EdgeV2,
        background: Dict[Tuple[str, str], float],
    ) -> Optional[float]:
        """Flip point of one edge with the other edges held at a sampled background.

        Identical search semantics to _compute_edge_e_values — boundary check
        then E_VALUE_BISECT_STEPS bisection on strength.mean, effective value
        mean * exists_probability, 'increase' direction tried first — differing
        ONLY in the background: the other edges sit at one sampled
        configuration instead of their expected values.

        Returns the flip mean, or None when no perturbation within
        [EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX] flips the winner under this
        background.
        """
        edge_key = (edge.from_, edge.to)
        current_mean = edge.strength.mean
        ep = edge.exists_probability

        baseline_config = dict(background)
        baseline_config[edge_key] = current_mean * ep

        def winner_under(config: Dict[Tuple[str, str], float]) -> str:
            outcomes = {}
            for option in request.options:
                outcomes[option.id] = evaluator.evaluate(
                    edge_strengths=config,
                    interventions=option.interventions,
                    goal_node=request.goal_node_id,
                )
            # Deterministic tie-breaking (sort by option_id) — same as base
            return sorted(outcomes.items(), key=lambda x: (-x[1], x[0]))[0][0]

        def winner_at(mean_value: float) -> str:
            # In-place: mutate the single edge_key, evaluate, restore in finally.
            # evaluator.evaluate() only READS edge_strengths (never mutates it),
            # so this is byte-identical to copying baseline_config every call —
            # the dict contents seen by evaluate() are the same {background,
            # edge_key: mean_value*ep} — but without rebuilding a fresh dict per
            # bisection step (~44 copies/call, E×10×44 across a sweep). The
            # finally restore keeps baseline_config a valid shared background for
            # the next step (edge_key returns to its entry value current_mean*ep).
            original = baseline_config[edge_key]
            baseline_config[edge_key] = mean_value * ep
            try:
                return winner_under(baseline_config)
            finally:
                baseline_config[edge_key] = original

        baseline_winner = winner_under(baseline_config)

        for direction in ("increase", "decrease"):
            if direction == "increase":
                lo, hi = current_mean, EDGE_STRENGTH_MAX
            else:
                lo, hi = EDGE_STRENGTH_MIN, current_mean

            boundary = hi if direction == "increase" else lo
            if winner_at(boundary) == baseline_winner:
                continue  # this direction cannot flip under this background

            for _ in range(self.E_VALUE_BISECT_STEPS):
                mid = (lo + hi) / 2
                if winner_at(mid) != baseline_winner:
                    # Flip happened — narrow toward current_mean
                    if direction == "increase":
                        hi = mid
                    else:
                        lo = mid
                else:
                    # No flip — narrow away from current_mean
                    if direction == "increase":
                        lo = mid
                    else:
                        hi = mid

            return hi if direction == "increase" else lo

        return None

    # =====================================================================
    # ROADMAP 2.228-F3 — factor-value flip thresholds
    # =====================================================================
    #
    # WHAT THIS REPLACES. PLoT used to hunt factor flip thresholds by re-running
    # a full Monte Carlo analysis per probe value (up to ~60 sequential HTTP
    # calls under a 30 s budget). The diagnosis proved with a live control that
    # the factors it selected were mathematically incapable of flipping the
    # winner: 43 rows, 16 timeout / 13 error / 14 no_effect, ZERO found. Both the
    # candidate selection and the probe mechanism are replaced here.
    #
    # WHY A CLOSED FORM IS EXACT, NOT AN APPROXIMATION. Before every post-MC
    # structural analysis the evaluator's epsilon noise is disabled
    # (``evaluator._epsilon_rng = None``), and SCMEvaluatorV2.evaluate is then
    # ``base + intercept + sum(parent_value * strength)`` — linear and additive.
    # For a ROOT factor F, ``base`` is exactly F's value, so at any FIXED edge
    # background each option's goal is exactly affine in F:
    #
    #     goal_o(F) = A_o + T_o * F
    #
    # Two evaluations per option (at F = 0 and F = 1) determine (A_o, T_o)
    # exactly, and the leader/rival crossing is algebra. There is no bisection,
    # no sampling, and therefore NO sampling error to disclose — which is why the
    # only uncertainty this phase publishes is the stability band.
    #
    # An option that intervenes on F has T_o = 0 because do(F=v) overrides the
    # structural equation entirely; an option that intervenes on a node
    # DOWNSTREAM of F severs F's path and also drives T_o toward 0. Both fall out
    # of the measurement — nothing about interventions is special-cased.

    # Wall-clock budget for the whole phase. ALL-OR-NOTHING on exceed (nothing is
    # attached and FACTOR_FLIPS_UNAVAILABLE is disclosed) — the same semantics as
    # E_VALUE_BUDGET_MS, for the same reason: a partial block would bias a reader
    # toward whichever factors happened to be computed first.
    FACTOR_FLIP_BUDGET_MS = 8000

    # At most this many candidates get the crossing + band treatment, ranked by
    # descending slope spread. Candidates below the cut are still EMITTED, with
    # flip_reason 'candidate_cap_exceeded' — a silent omission here would recreate
    # the exact defect this roadmap item exists to remove.
    FACTOR_FLIP_MAX_CANDIDATES = 10

    # Candidate rule: F can move the argmax iff the per-option transmission slopes
    # differ. The live control measured non-candidates identical to 16 significant
    # figures, so 1e-9 sits far above the observed floor of the class while still
    # being tight enough that a real difference is never rounded away.
    FACTOR_FLIP_SLOPE_EPSILON = 1e-9

    # Domain of observed_state.value on the normalised wire.
    FACTOR_VALUE_MIN = 0.0
    FACTOR_VALUE_MAX = 1.0

    # Offset placing the argmax CONFIRMATION strictly on the far side of a
    # candidate crossing. Clamped to half the distance to the next crossing, so
    # the confirmation can never step over a second crossing and attribute the
    # wrong alternative winner.
    FACTOR_FLIP_CONFIRM_EPSILON = 1e-6

    def _option_goals(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        edge_config: Dict[Tuple[str, str], float],
        factor_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Per-option goal value under one edge background and factor override."""
        return {
            option.id: evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=option.interventions,
                goal_node=request.goal_node_id,
                factor_values=factor_values,
            )
            for option in request.options
        }

    @staticmethod
    def _argmax_option(outcomes: Dict[str, float]) -> str:
        """Analyzer-wide deterministic argmax: highest outcome, then lowest id."""
        return sorted(outcomes.items(), key=lambda x: (-x[1], x[0]))[0][0]

    @classmethod
    def _affine_coefficients(
        cls, at_min: Dict[str, float], at_max: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """(intercepts A_o, slopes T_o) from goals measured at the domain ends."""
        span = cls.FACTOR_VALUE_MAX - cls.FACTOR_VALUE_MIN
        slopes = {oid: (at_max[oid] - at_min[oid]) / span for oid in at_min}
        intercepts = {oid: at_min[oid] - slopes[oid] * cls.FACTOR_VALUE_MIN for oid in at_min}
        return intercepts, slopes

    def _evaluated_argmax_probe(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        edge_config: Dict[Tuple[str, str], float],
        factor_id: str,
    ) -> Callable[[float], str]:
        """Argmax at a given value of ONE factor, via real evaluations.

        A factory rather than an inline closure: the caller builds one per factor
        inside a loop, and a closure over the loop variable is a footgun even
        where it happens to be consumed immediately.
        """

        def probe(value: float) -> str:
            return self._argmax_option(
                self._option_goals(request, evaluator, edge_config, {factor_id: value})
            )

        return probe

    @classmethod
    def _affine_argmax_probe(
        cls, intercepts: Dict[str, float], slopes: Dict[str, float]
    ) -> Callable[[float], str]:
        """Argmax of the affine family, by arithmetic only — zero evaluations.

        Exact for the same reason the crossing is exact (the SCM is affine in a
        root factor's value once epsilon noise is off), which is what keeps the
        per-background band cost at the 2 * O evaluations the budget model
        assumes.
        """

        def probe(value: float) -> str:
            return cls._argmax_option(
                {oid: intercepts[oid] + slopes[oid] * value for oid in intercepts}
            )

        return probe

    @classmethod
    def _nearest_confirmed_crossing(
        cls,
        intercepts: Dict[str, float],
        slopes: Dict[str, float],
        baseline_winner: str,
        current_value: float,
        confirm: Callable[[float], str],
    ) -> Optional[Tuple[float, str, str]]:
        """Nearest in-bounds crossing at which the ARGMAX actually changes.

        Returns ``(flip_value, direction, alternative_winner_id)`` or None.

        Every rival's crossing with the leader is enumerated in closed form,
        F* = (A_i - A_j)/(T_j - T_i), skipping parallel rivals (a degenerate
        denominator is not a near-crossing, it is NO crossing — dividing anyway
        would manufacture a huge spurious value). Each candidate crossing is then
        CONFIRMED by asking for the argmax strictly on its far side, because a
        pairwise crossing is not necessarily an argmax change when a third option
        is above both lines there (design R6). Crossings are walked outward from
        current_value, so the first confirmation on each side is the nearest real
        flip; both sides are computed and the closer one wins, with 'increase'
        breaking a tie (mirroring _compute_edge_e_values, which tries 'increase'
        first).
        """
        crossings: List[float] = []
        for oid in intercepts:
            if oid == baseline_winner:
                continue
            denom = slopes[oid] - slopes[baseline_winner]
            if abs(denom) <= cls.FACTOR_FLIP_SLOPE_EPSILON:
                continue  # parallel to the leader — they never meet
            f_star = (intercepts[baseline_winner] - intercepts[oid]) / denom
            if cls.FACTOR_VALUE_MIN <= f_star <= cls.FACTOR_VALUE_MAX:
                crossings.append(f_star)
        if not crossings:
            return None

        ordered = sorted(set(crossings))
        above = [c for c in ordered if c > current_value]
        descending_below = sorted((c for c in ordered if c < current_value), reverse=True)

        found_up: Optional[Tuple[float, str, str]] = None
        for idx, crossing in enumerate(above):
            upper = above[idx + 1] if idx + 1 < len(above) else cls.FACTOR_VALUE_MAX
            step = (
                min(cls.FACTOR_FLIP_CONFIRM_EPSILON, (upper - crossing) / 2.0)
                if upper > crossing
                else 0.0
            )
            winner = confirm(crossing + step)
            if winner != baseline_winner:
                found_up = (crossing, "increase", winner)
                break

        found_down: Optional[Tuple[float, str, str]] = None
        for idx, crossing in enumerate(descending_below):
            lower = (
                descending_below[idx + 1]
                if idx + 1 < len(descending_below)
                else cls.FACTOR_VALUE_MIN
            )
            step = (
                min(cls.FACTOR_FLIP_CONFIRM_EPSILON, (crossing - lower) / 2.0)
                if crossing > lower
                else 0.0
            )
            winner = confirm(crossing - step)
            if winner != baseline_winner:
                found_down = (crossing, "decrease", winner)
                break

        if found_up is None:
            return found_down
        if found_down is None:
            return found_up
        if abs(found_down[0] - current_value) < abs(found_up[0] - current_value):
            return found_down
        return found_up

    def _compute_factor_flip_values(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        master_seed: int,
        budget_ms: Optional[float] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Per-root-factor flip thresholds (design §2.1/§2.2).

        Returns one attested row per eligible root factor, or None when the
        wall-clock budget tripped (all-or-nothing; the caller then discloses
        FACTOR_FLIPS_UNAVAILABLE on the wire).

        Eligibility: a node with no parents (the same ``is_root`` test
        SCMEvaluatorV2.evaluate itself applies when deciding whether
        observed_state.value is a node's base), that is not the goal, and that
        carries either an observed value or a parameter_uncertainties entry.
        Non-root nodes are excluded because ``factor_values`` would ADD to their
        parent contribution rather than replace a base, so the affine reading
        would not describe the quantity a consumer thinks it does.
        """
        budget = budget_ms if budget_ms is not None else self.FACTOR_FLIP_BUDGET_MS
        # Monotonic clock: an NTP step must not corrupt the elapsed guard.
        t0 = time.monotonic()

        def _tripped(rows_done: int) -> bool:
            elapsed_ms = (time.monotonic() - t0) * 1000
            if elapsed_ms <= budget:
                return False
            self.logger.info(
                "factor_flip_budget_exceeded",
                extra={"elapsed_ms": round(elapsed_ms, 1), "factors_completed": rows_done},
            )
            return True

        baseline_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
        }
        baseline_winner = self._argmax_option(
            self._option_goals(request, evaluator, baseline_config)
        )

        uncertainty_by_id = {u.node_id: u for u in (request.parameter_uncertainties or [])}
        uncertainty_ids = set(uncertainty_by_id)
        eligible = [
            node
            for node in request.graph.nodes
            if node.id != request.goal_node_id
            and not evaluator._parents.get(node.id)
            and (
                (node.observed_state is not None and node.observed_state.value is not None)
                or node.id in uncertainty_ids
            )
        ]
        if not eligible:
            return []

        # --- Candidate screen (§2.1): 2 * O evaluations per factor, and NOTHING
        # more for a factor that fails it. This is the whole point of the rule —
        # the provably-inert class costs the screen and never a probe.
        screened: List[Dict[str, Any]] = []
        for node in eligible:
            if _tripped(len(screened)):
                return None
            at_min = self._option_goals(
                request, evaluator, baseline_config, {node.id: self.FACTOR_VALUE_MIN}
            )
            at_max = self._option_goals(
                request, evaluator, baseline_config, {node.id: self.FACTOR_VALUE_MAX}
            )
            intercepts, slopes = self._affine_coefficients(at_min, at_max)
            screened.append(
                {
                    "node": node,
                    "intercepts": intercepts,
                    "slopes": slopes,
                    "spread": max(slopes.values()) - min(slopes.values()),
                    # 2.1020: the SAME central value the sensitivity probe and
                    # the sampler use. Publishing 0.0 for a factor whose own
                    # declared prior is [0.6, 1.0] stated a current value
                    # outside its support — and would now contradict the
                    # elasticity in the same response.
                    "current_value": resolve_factor_central_value(
                        node, uncertainty_by_id.get(node.id)
                    ).value,
                }
            )

        candidates = [s for s in screened if s["spread"] > self.FACTOR_FLIP_SLOPE_EPSILON]
        # Deterministic ranking: widest slope spread first, factor id as the
        # tie-break so the same request always caps the same way.
        candidates.sort(key=lambda s: (-s["spread"], s["node"].id))
        selected_ids = {s["node"].id for s in candidates[: self.FACTOR_FLIP_MAX_CANDIDATES]}

        backgrounds: Optional[List[Dict[Tuple[str, str], float]]] = None
        if selected_ids:
            # Sampled ONLY when there is a candidate to band — an all-invariant
            # graph must not pay for a sweep whose rows would be discarded.
            backgrounds = self._sample_flip_backgrounds(
                request, master_seed, FLIP_STABILITY_N_SEEDS, "factor_flip_stability"
            )

        rows: List[Dict[str, Any]] = []
        for entry in screened:  # graph order, so the block is stable across runs
            node = entry["node"]
            row: Dict[str, Any] = {
                "factor_id": node.id,
                "current_value": entry["current_value"],
                "flip_value": None,
                "direction": None,
                "flip_reason": "structurally_invariant",
                "alternative_winner_id": None,
                "baseline_winner_id": baseline_winner,
            }

            if entry["spread"] <= self.FACTOR_FLIP_SLOPE_EPSILON:
                # Provably inert: every option transmits this factor identically,
                # so no value of it can move the argmax. Attested, not probed.
                rows.append(row)
                continue

            if node.id not in selected_ids:
                row["flip_reason"] = "candidate_cap_exceeded"
                rows.append(row)
                continue

            if _tripped(len(rows)):
                return None

            confirmed = self._nearest_confirmed_crossing(
                entry["intercepts"],
                entry["slopes"],
                baseline_winner,
                entry["current_value"],
                confirm=self._evaluated_argmax_probe(request, evaluator, baseline_config, node.id),
            )
            if confirmed is None:
                row["flip_reason"] = "no_effect_within_bounds"
            else:
                flip_value, direction, alternative = confirmed
                row["flip_value"] = round(flip_value, 6)
                row["direction"] = direction
                row["flip_reason"] = "found"
                row["alternative_winner_id"] = alternative

            band = self._factor_flip_band(
                request, evaluator, node, entry["current_value"], backgrounds or [], t0, budget
            )
            if band is None:
                return None  # budget tripped mid-sweep — all-or-nothing
            row["stability"] = band
            rows.append(row)

        return rows

    def _factor_flip_band(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        node: NodeV2,
        current_value: float,
        backgrounds: List[Dict[Tuple[str, str], float]],
        t0: float,
        budget: float,
    ) -> Optional[Dict[str, Any]]:
        """Seed-sweep stability band for one factor's flip value.

        Same pattern as the edge bands: one sampled edge background per child
        seed, SHARED across factors (common random numbers, so bands are
        comparable row to row), and the flip re-derived under each. Re-derivation
        is closed form — 2 * O evaluations per background to re-measure the
        slopes, then pure arithmetic — which is why this costs roughly a tenth of
        the edge bands' bisection sweep.

        The per-background confirmation is arithmetic on the affine family rather
        than another evaluation. That is exact for the same reason the base
        crossing is exact, and it keeps the per-background cost at the 2 * O the
        budget model assumes.

        Returns the band dict, or None when the wall-clock budget tripped.
        """
        seed_flip_values: List[Optional[float]] = []
        for background in backgrounds:
            elapsed_ms = (time.monotonic() - t0) * 1000
            if elapsed_ms > budget:
                self.logger.info(
                    "factor_flip_budget_exceeded",
                    extra={
                        "elapsed_ms": round(elapsed_ms, 1),
                        "phase": "stability_bands",
                        "factor_id": node.id,
                    },
                )
                return None
            at_min = self._option_goals(
                request, evaluator, background, {node.id: self.FACTOR_VALUE_MIN}
            )
            at_max = self._option_goals(
                request, evaluator, background, {node.id: self.FACTOR_VALUE_MAX}
            )
            intercepts, slopes = self._affine_coefficients(at_min, at_max)
            # The winner THIS background starts from — derived from the affine
            # family, not re-evaluated, and not assumed to equal the
            # expected-value baseline winner (a sampled background may well be
            # led by a different option, and pretending otherwise would report a
            # flip against a leader that background never had).
            affine_argmax = self._affine_argmax_probe(intercepts, slopes)
            background_winner = affine_argmax(current_value)
            confirmed = self._nearest_confirmed_crossing(
                intercepts,
                slopes,
                background_winner,
                current_value,
                confirm=affine_argmax,
            )
            seed_flip_values.append(round(confirmed[0], 6) if confirmed is not None else None)

        flipped = [v for v in seed_flip_values if v is not None]
        band: Dict[str, Any] = {
            "n_seeds": len(backgrounds),
            "n_seeds_flipped": len(flipped),
            "seed_flip_values": seed_flip_values,
        }
        if flipped:
            band_min = min(flipped)
            band_max = max(flipped)
            band["band_min"] = band_min
            band["band_median"] = round(float(statistics.median(flipped)), 6)
            band["band_max"] = band_max
            band["band_width"] = round(band_max - band_min, 6)
        return band

    def _compute_factor_evppi(
        self,
        request: RobustnessRequestV2,
        pre_noise_option_outcomes: Dict[str, List[float]],
        factor_values_per_sample: List[Dict[str, float]],
        seed: int,
        decision_evpi_bound: Optional[float],
        correlation_active: bool,
        inference_warnings: Optional[List[InferenceWarning]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Per-factor EVPPI (Expected Value of Partial Perfect Information) in
        OUTCOME units, via single-loop Strong-Oakley regression on the retained
        joint CRN samples (S2, D-23.8). No nested MC, no new sampling.

        For each uncertain factor that is NOT an option-controlled lever, regress
        every option's per-sample PRE-noise outcome on that factor's per-sample
        value and compute ``EVPPI_i = E[max_o E[U_o|theta_i]] − max_o E[U_o]``. The
        raw estimate is clamped to ``[0, decision_evpi]`` with disclosure:

        * ``clamped_low`` — a negative raw estimate is finite-sample noise (Howard
          non-negativity); clamped to 0.
        * ``clamped_high`` — a raw estimate above the whole-decision EVPI violates
          the per-factor ≤ total theorem (estimator noise); capped at the bound.

        D-U LEVER SUPPRESSION (binding): a factor ANY option intervenes on (union
        across options — the SAME source of truth as _compute_factor_sensitivity)
        is a CHOICE, not information to buy, so it is OMITTED entirely (absent, not
        zero — missing ≠ zero). Non-lever uncertain factors always get an entry.

        PARTIAL-DROP DISCLOSURE (Codex F7, D-23.15): a REQUESTED (non-lever) factor
        can still be dropped IN-LOOP when its per-sample values are non-finite (a
        pathological uncertainty overflow) or its estimator raises (e.g. LinAlgError
        on a goal-disconnected factor). That drop was previously SILENT — the wire
        warning only fired for a WHOLE-call failure — so ``factor_evppi`` absence was
        ambiguous (no-eligible-factor vs not-requested vs estimator-failed). When
        ``inference_warnings`` is supplied and ≥1 requested factor is dropped, a
        ``FACTOR_EVPPI_PARTIAL`` warning lists the safe + failed factor ids and each
        failure's category (ids + categories only — never a request value), so the
        absence/short-list is explicit. Computable rows are always preserved.

        Returns a list of per-factor dicts (sorted by evppi descending), or None if
        no non-lever uncertain factor produced a row.
        """
        if not request.parameter_uncertainties:
            return None

        # D-U lever identity: UNION of intervention targets across ALL options —
        # the SAME source of truth _compute_factor_sensitivity consults, via the
        # shared _intervention_factor_union helper (derive, don't mirror: a called
        # function cannot drift, a re-typed comprehension can). A factor in this set
        # is a lever; its "uncertainty" is a choice.
        intervention_factor_ids = self._intervention_factor_union(request)

        # Deduplicate uncertainties by node_id (parse-time validation already
        # rejects duplicates; defensive, first-seen order for determinism).
        unique_uncertainties = self._dedup_uncertainties(request)

        n_samples = len(factor_values_per_sample)
        results: List[Dict[str, Any]] = []
        # F7 (D-23.15): track REQUESTED (non-lever) factors that are dropped in-loop so
        # the caller can disclose the drop (safe ids kept a row; failed ids + category).
        safe_ids: List[str] = []
        failed: List[Tuple[str, str]] = []  # (factor_id, failure_category)
        for uncertainty in unique_uncertainties:
            fid = uncertainty.node_id
            # LEVER SUPPRESSION: omit option-controlled levers (missing ≠ zero). A
            # lever is an INTENTIONAL omission, NOT a failure — never recorded in
            # `failed` (it must not raise a partial-drop warning).
            if fid in intervention_factor_ids:
                continue

            # Extract this factor's per-sample values from the retained joint
            # population. Every uncertainty factor is present in every sample dict
            # (FactorSampler always writes it), so a missing key is a defect → omit
            # that factor with no fabricated value.
            try:
                theta = [factor_values_per_sample[s][fid] for s in range(n_samples)]
            except (KeyError, IndexError):
                failed.append((fid, "missing_sample_value"))
                continue

            # F7 cheap pre-regression validation: a non-finite theta (e.g. an overflow
            # from a pathological uncertainty std) would make the polynomial fit raise
            # or return NaN deep inside. Catch it here, cheaply and with a precise
            # category, before the estimator (bound/validate sampled θ pre-regression).
            if not all(math.isfinite(t) for t in theta):
                self.logger.warning(
                    "factor_evppi_non_finite_theta",
                    extra={"factor_id": fid},
                )
                failed.append((fid, "non_finite_theta"))
                continue

            # Deterministic per-factor seed for the permutation-null floor.
            floor_seed = self._per_factor_seed(seed, "evppi", fid)
            # Per-factor degrade (hunter F-4): a single factor whose estimator raises
            # (e.g. a poisoned ±inf theta on a goal-disconnected factor makes
            # Polynomial.fit raise LinAlgError) must NOT drop the WHOLE factor_evppi
            # block. Omit THIS factor (missing != zero — no fabricated value) and keep
            # every other factor's perfectly-computable row.
            try:
                est = factor_evppi_estimate(theta, pre_noise_option_outcomes, seed=floor_seed)
            except Exception:
                self.logger.warning(
                    "factor_evppi_estimator_error",
                    extra={"factor_id": fid},
                    exc_info=True,
                )
                failed.append((fid, "estimator_error"))
                continue

            # 2.514(b): a NON-FINITE estimate is not an estimate. The estimator
            # does not raise on a poisoned option matrix — it returns nan/inf
            # components (measured: an option with no finite draw makes
            # `baseline_max_expected_utility` inf and `evppi_raw` nan, because
            # `factor_evppi_estimate` does not filter non-finite option columns).
            # Emitting that did two dishonest things:
            #  * `evppi = max(0.0, nan)` returns **0.0** — every nan comparison is
            #    False, so Python keeps the first argument. On the wire that is
            #    indistinguishable from a real "learning this factor is worth
            #    nothing", and it shipped with clamped_low=False, so nothing
            #    marked it as degraded;
            #  * `round(nan, 6)` is still nan, so `evppi_raw` reached the response
            #    body and the JSONResponse render died with "Out of range float
            #    values are not JSON compliant" — a 500 for the whole analysis.
            # Drop THIS factor instead (missing != zero) and let the block's
            # EXISTING partial-drop disclosure report it, rather than inventing a
            # parallel scheme. Computable factors keep their rows.
            non_finite_components = [
                name
                for name, value in (
                    ("evppi_raw", est.evppi_raw),
                    ("baseline_max_expected_utility", est.baseline_max_expected_utility),
                    ("conditional_max_expected_utility", est.conditional_max_expected_utility),
                    ("noise_floor", est.noise_floor),
                )
                if not math.isfinite(value)
            ]
            if non_finite_components:
                self.logger.warning(
                    "factor_evppi_non_finite_estimate",
                    extra={"factor_id": fid, "non_finite_components": non_finite_components},
                )
                failed.append((fid, "non_finite_estimate"))
                continue

            # Howard non-negativity clamp — DEAD-MAN'S-SWITCH: evppi_raw is >= 0 by
            # construction for the regression estimator (LS mean-preservation +
            # Jensen), so this never fires on the live path; a clamped_low=True in
            # telemetry would mean the estimator changed. Kept as defence-in-depth.
            clamped_low = est.evppi_raw < 0.0
            if clamped_low:
                # Altitude Q5: fire the switch the instant it trips instead of relying
                # on a human spotting the bool on the wire. A negative raw means the
                # estimator lost its Howard >= 0 guarantee (e.g. a lost intercept).
                self.logger.warning(
                    "evppi_estimator_regressed",
                    extra={"factor_id": fid, "evppi_raw": est.evppi_raw},
                )
            evppi = max(0.0, est.evppi_raw)

            # Per-factor ≤ whole-decision EVPI theorem: cap at decision_evpi.
            clamped_high = False
            if decision_evpi_bound is not None and evppi > decision_evpi_bound:
                clamped_high = True
                evppi = decision_evpi_bound

            below_resolution = evppi <= est.noise_floor

            # Clamp-vs-round ordering (hunter F-2): round(.,6) can nudge a clamped
            # value UP past the raw decision_evpi bound by <=5e-7, breaking the
            # documented evppi <= decision_evpi on the wire. Re-clamp AFTER rounding
            # against the RAW bound (which is <= the wire decision_evpi, since the
            # bound is min over ALL options and the wire minimises over the
            # downside-bearing subset). Non-clamp-binding rows are unaffected —
            # round(evppi,6) < bound there, so the min is a no-op and goldens hold.
            evppi_emitted = round(evppi, 6)
            if decision_evpi_bound is not None:
                evppi_emitted = min(evppi_emitted, decision_evpi_bound)

            results.append(
                {
                    "factor_id": fid,
                    "evppi": evppi_emitted,
                    # Pre-clamp raw estimate + audit components (mirrors
                    # p_win_sensitivity's current_metric/perfect_metric auditability).
                    "evppi_raw": round(est.evppi_raw, 6),
                    "baseline_max_expected_utility": round(est.baseline_max_expected_utility, 6),
                    "conditional_max_expected_utility": round(
                        est.conditional_max_expected_utility, 6
                    ),
                    "units": "outcome",
                    "method": REGRESSION_EVPPI_METHOD,
                    "regression_degree": est.degree_used,
                    "n_samples": est.n_samples,
                    # Howard non-negativity clamp fired (raw was negative noise).
                    "clamped_low": clamped_low,
                    # Per-factor ≤ total-EVPI clamp fired (raw exceeded decision_evpi).
                    "clamped_high": clamped_high,
                    # Permutation-null overfit floor; evppi ≤ floor = below_resolution.
                    "noise_floor": round(est.noise_floor, 6),
                    "status": "below_resolution" if below_resolution else "resolved",
                    # Disclosure: under active correlation the samples are joint
                    # copula draws, so this conditional-expectation EVPPI is honest
                    # (it never assumes independence). True iff correlation active.
                    "correlation_active": correlation_active,
                }
            )
            safe_ids.append(fid)

        # F7 (D-23.15): disclose any requested factor dropped in-loop. Emitted whether
        # or not any row survived — so `factor_evppi: absent` with a dropped factor is
        # never silent (it disambiguates estimator-failed from no-eligible-factor).
        # severity='warning' so PLoT's severity=='warning' mapping surfaces it (an
        # 'info' would be hidden, re-silencing the drop). Ids + categories only — no
        # request values (mirrors the correlation/EVPC validators' disclosure discipline).
        if failed and inference_warnings is not None:
            inference_warnings.append(
                InferenceWarning(
                    code="FACTOR_EVPPI_PARTIAL",
                    field="factor_evppi",
                    severity="warning",
                    detail={
                        "reason": "per_factor_dropped",
                        "safe_factor_ids": safe_ids,
                        "failed_factor_ids": [fid for fid, _ in failed],
                        "failures": [
                            {"factor_id": fid, "category": category} for fid, category in failed
                        ],
                        "message": (
                            "Per-factor EVPPI could not be computed for "
                            f"{len(failed)} requested factor(s); the remaining "
                            f"{len(safe_ids)} were computed. Base analysis is unaffected."
                        ),
                    },
                )
            )

        if not results:
            return None

        # Sort by EVPPI descending (most valuable information first).
        results.sort(key=lambda x: float(x["evppi"]), reverse=True)
        return results

    def _compute_factor_evpc(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]],
        factor_values_per_sample: List[Dict[str, float]],
        pre_noise_option_outcomes: Dict[str, List[float]],
        correlation_active: bool,
    ) -> Optional[List[Dict[str, Any]]]:
        """Per-lever Expected Value of Control (EVPC) in OUTCOME units, via grid
        do(factor=value) on the retained joint CRN samples (S4, D-23.8). No nested
        MC, no new sampling — reuses the SAME joint draws that scored the options.

        For each control candidate the user supplied, and each candidate value ``x``,
        re-evaluate every retained sample under ``do(factor_id = x)`` on that sample's
        own (edge_config, factor_values) joint draw via the SAME ``SCMEvaluatorV2``
        used for the options. The intervention OVERRIDES the factor's drawn value
        while every other factor — including its correlated partners — keeps its joint
        draw (#100 override-after-joint-draw: partners unaffected; verified against
        ``evaluate()`` lines 909-949). Then::

            E[U | do(factor=x)]  = mean over samples of the do(x) outcome
            EVPC_raw(factor)     = max_x E[U | do(factor=x)] − max_a E[U_a]
            EVPC(factor)         = max(0, EVPC_raw)      (control cannot hurt: a lever
                                   you may pull to any candidate value is never worse
                                   than not pulling it; a negative raw is grid/finite-
                                   sample slack, clamped with disclosure)

        where ``max_a E[U_a]`` is the baseline value of the best CURRENT option on the
        pre-noise CRN population — the SAME population behind win_probability,
        decision_evpi and factor_evppi. ``best_candidate_value`` (argmax over the
        grid) is ALWAYS reported, even when every candidate underperforms the baseline
        (EVPC = 0): the honest reading is "controlling this factor to these values adds
        nothing, and this was the best of them".

        EVPC is the value of CONTROL, the mirror of EVPPI's value of INFORMATION — so,
        unlike factor_evppi, option-controlled levers are NOT suppressed here: the user
        explicitly asks what a lever is worth, and control is precisely the point. It
        is a grid approximation (a LOWER BOUND on the true continuous EVPC; more values
        tighten it) and is EMITTED under active correlation (the do() runs on the joint
        copula draws, so it is honest under correlation).

        Returns a list of per-lever dicts (sorted by EVPC descending, factor_id
        tie-break), or None when control_candidates is absent (request-driven gate) or
        no sample population exists.
        """
        if not request.control_candidates:
            return None

        n_samples = len(factor_values_per_sample)
        if n_samples == 0:
            return None

        # Baseline max_a E[U_a]: the best current option's expected outcome on the
        # pre-noise CRN population (np.mean — the engine's mean convention).
        option_means = {
            oid: float(np.mean(vals)) for oid, vals in pre_noise_option_outcomes.items() if vals
        }
        finite_option_means = {oid: m for oid, m in option_means.items() if math.isfinite(m)}
        if not finite_option_means:
            return None
        baseline_max_eu = max(finite_option_means.values())

        results: List[Dict[str, Any]] = []
        for candidate in request.control_candidates:
            fid = candidate.factor_id

            # Grid do(fid = x) over the candidate values on the retained joint draws.
            best_value: Optional[float] = None
            best_do_eu = -math.inf
            for x in candidate.values:
                do_outcomes = [
                    evaluator.evaluate(
                        edge_strengths=edge_configs_per_sample[i],
                        interventions={fid: x},
                        goal_node=request.goal_node_id,
                        factor_values=factor_values_per_sample[i],
                    )
                    for i in range(n_samples)
                ]
                do_eu = float(np.mean(do_outcomes))
                # Skip a non-finite grid point (pathological graph) rather than let
                # inf/nan reach the wire; strict '>' keeps the FIRST (request-order)
                # value on ties → deterministic argmax.
                if math.isfinite(do_eu) and do_eu > best_do_eu:
                    best_do_eu = do_eu
                    best_value = x

            if best_value is None:
                # Every grid point was non-finite (e.g. a candidate value large
                # enough that the mean over samples overflows to +/-inf). This CAN
                # happen on a 200: the options use their OWN finite interventions, so
                # option means and the JSON serializer stay finite and the response
                # ships 200 — but THIS candidate is dropped, so factor_evpc can be
                # absent (or short a lever) despite control_candidates being present.
                # We omit rather than fabricate a value. Surfacing a disclosure warning
                # for a dropped candidate is a tracked follow-up (rowed as behavior),
                # intentionally NOT implemented here.
                continue

            evpc_raw = best_do_eu - baseline_max_eu
            clamped_low = evpc_raw < 0.0
            evpc = max(0.0, evpc_raw)

            results.append(
                {
                    "factor_id": fid,
                    "evpc": round(evpc, 6),
                    # Pre-clamp raw + audit legs (mirrors factor_evppi auditability).
                    "evpc_raw": round(evpc_raw, 6),
                    "best_candidate_value": best_value,
                    "baseline_max_expected_utility": round(baseline_max_eu, 6),
                    "best_do_expected_utility": round(best_do_eu, 6),
                    "units": "outcome",
                    "method": GRID_DO_EVPC_METHOD,
                    "n_samples": n_samples,
                    "n_candidate_values": len(candidate.values),
                    # Control-cannot-hurt clamp fired (raw was negative grid slack).
                    "clamped_low": clamped_low,
                    # Disclosure: do() ran on joint copula draws → honest under corr.
                    "correlation_active": correlation_active,
                }
            )

        if not results:
            return None

        # Sort by EVPC descending (most valuable lever first), factor_id tie-break.
        results.sort(key=lambda r: (-float(r["evpc"]), str(r["factor_id"])))
        return results

    def _compute_evpi(
        self,
        request: RobustnessRequestV2,
        sampler: DualUncertaintySampler,
        factor_sampler: FactorSampler,
        evaluator: SCMEvaluatorV2,
        seed: int,
        recommended_option_id: str,
        budget_ms: Optional[float] = None,
        constraint_plans: Optional[Dict[int, "GoalThresholdPlan"]] = None,
        objective: Optional["ObjectivePlan"] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Compute Expected Value of Perfect Information (EVPI) per factor.

        For each factor with ParameterUncertainty, run MC with that factor's
        uncertainty removed (fix at mean value) and compare the metric:
        - P(joint_goal) when goal_constraints exist
        - P(win) of the recommended option otherwise

        The recommended option is held fixed across all EVPI runs to avoid
        policy-switch confounding (evaluating different decisions under
        different information states).

        EVPI = metric_with_perfect_info - metric_with_uncertainty

        Args:
            request: The robustness request.
            sampler: Edge configuration sampler.
            factor_sampler: Factor value sampler.
            evaluator: SCM evaluator.
            seed: Global seed for reproducibility.
            recommended_option_id: Fixed decision policy (from main MC run).
            budget_ms: Internal wall-clock budget for the whole EVPI sweep (Codex
                F7). None disables the guard. The analyze() orchestrator passes
                min(EVPI_BUDGET_MS, remaining request budget), so the sweep also
                respects the governing overall-request deadline. On overrun the
                WHOLE phase is discarded (return None) — ALL-OR-NOTHING, never
                partial rows — and analyze() discloses EVPI_UNAVAILABLE.

        Returns:
            List of dicts with EVPI info per factor, or None (no uncertainties, or
            the internal wall-clock budget was exceeded).
        """
        if not request.parameter_uncertainties:
            return None

        # F8 defensive dedup (defence-in-depth): the model validator already
        # rejects duplicate parameter_uncertainties node_ids with a 422 at parse
        # time, but a direct internal caller could bypass it. Dedup by node_id,
        # keeping first-seen order (dict preserves insertion order → deterministic)
        # so a repeated node_id can never re-trigger the EVPI multiplier here.
        unique_uncertainties = self._dedup_uncertainties(request)

        # F7: internal wall-clock deadline (was entry-gated only). deadline anchors
        # t0 at the EVPI phase; budget_ms is the remaining governing request budget
        # at entry, so deadline.exceeded() == the OVERALL_REQUEST_BUDGET_MS deadline
        # has passed. Monotonic — an NTP step must not corrupt the guard (mirrors
        # _compute_edge_e_values / _attach_flip_stability_bands). The same deadline
        # is threaded into _compute_evpi_metric so its sample-loop re-check shares
        # this phase t0.
        deadline = PhaseDeadline(budget_ms)

        # Budget: cap samples for EVPI to limit latency (see EVPI_SAMPLE_CAP
        # comment — Paul-ruled lenient defaults 2026-07-17, 500 → 2000).
        n_samples = min(request.n_samples, EVPI_SAMPLE_CAP)
        constraint_target_nodes = None
        if request.goal_constraints:
            constraint_target_nodes = sorted(set(gc.node_id for gc in request.goal_constraints))

        # Deadline check before the (full-MC-pass) baseline.
        if deadline.exceeded():
            self.logger.info(
                "evpi_budget_exceeded",
                extra={
                    "elapsed_ms": deadline.elapsed_ms(),
                    "phase": "pre_baseline",
                },
            )
            return None

        # Baseline: all uncertainties active
        baseline_rng_edge = SeededRNG(seed + 100)
        baseline_rng_factor = SeededRNG(seed + 101)
        baseline_sampler = DualUncertaintySampler(request.graph.edges, baseline_rng_edge)
        baseline_factor_sampler = FactorSampler(
            request.graph.nodes, unique_uncertainties, baseline_rng_factor
        )
        baseline_metric = self._compute_evpi_metric(
            request,
            baseline_sampler,
            baseline_factor_sampler,
            evaluator,
            n_samples,
            constraint_target_nodes,
            recommended_option_id,
            deadline=deadline,
            constraint_plans=constraint_plans,
            objective=objective,
        )
        if baseline_metric is None:
            # Deadline tripped inside the baseline sample loop.
            self.logger.info(
                "evpi_budget_exceeded",
                extra={
                    "elapsed_ms": deadline.elapsed_ms(),
                    "phase": "baseline_metric",
                },
            )
            return None

        results: List[Dict[str, Any]] = []
        for uncertainty in unique_uncertainties:
            # Deadline check at the top of each per-factor MC pass.
            if deadline.exceeded():
                self.logger.info(
                    "evpi_budget_exceeded",
                    extra={
                        "elapsed_ms": deadline.elapsed_ms(),
                        "factors_completed": len(results),
                    },
                )
                return None
            # Create modified uncertainty list: remove this factor's uncertainty
            modified_uncertainties = [
                u for u in unique_uncertainties if u.node_id != uncertainty.node_id
            ]

            # Deterministic seed per factor.
            factor_seed = self._per_factor_seed(seed, "evpi", uncertainty.node_id)

            perfect_rng_edge = SeededRNG(factor_seed)
            perfect_rng_factor = SeededRNG(factor_seed + 1)
            perfect_sampler = DualUncertaintySampler(request.graph.edges, perfect_rng_edge)
            perfect_factor_sampler = FactorSampler(
                request.graph.nodes,
                modified_uncertainties if modified_uncertainties else None,
                perfect_rng_factor,
            )

            perfect_metric = self._compute_evpi_metric(
                request,
                perfect_sampler,
                perfect_factor_sampler,
                evaluator,
                n_samples,
                constraint_target_nodes,
                recommended_option_id,
                deadline=deadline,
                constraint_plans=constraint_plans,
                objective=objective,
            )
            if perfect_metric is None:
                # Deadline tripped inside this factor's MC pass — discard the whole
                # phase (all-or-nothing), do not emit partial EVPI rows.
                self.logger.info(
                    "evpi_budget_exceeded",
                    extra={
                        "elapsed_ms": deadline.elapsed_ms(),
                        "factors_completed": len(results),
                    },
                )
                return None

            delta_raw = perfect_metric - baseline_metric

            # Producer clamp (F1 residual r1, defense-in-depth): this quantity
            # is definitionally non-negative — a negative difference of two MC
            # proportion estimates is estimator noise, so clamp it to 0.0 at the
            # producer rather than relying solely on the PLoT boundary guard
            # (PR #219). The raw components remain auditable via perfect_metric /
            # current_metric; ``clamped`` flags the entries where the clamp fired.
            delta_clamped = delta_raw < 0.0
            delta = 0.0 if delta_clamped else delta_raw

            # Below-resolution labelling (provisional_doctrine_v0): flag
            # estimates smaller in magnitude than the MC noise floor for this
            # sample budget. Applied to the emitted (clamped) value, so a
            # clamped-to-zero entry is always below_resolution.
            noise_floor = evpi_noise_floor(n_samples)
            below_resolution = abs(delta) < noise_floor

            # S2 (D-23.8) HONEST RELABEL: this block WAS emitted as ``factor_evpi``
            # with an ``evpi`` field, but it is NOT value-of-information. It holds
            # the decision FIXED at the recommended option and reports how much the
            # recommended option's WIN PROBABILITY moves when this factor is fixed
            # at its mean — a win-probability sensitivity in probability units, with
            # its OWN MC redraw (not the CRN joint population). It structurally
            # cannot capture option-switching (the value of information), so calling
            # it EVPI was a mislabel. Renamed to ``p_win_sensitivity`` with
            # de-EVPI'd field names + a ``method`` tag; the numbers are byte-
            # identical to the pre-S2 ``factor_evpi`` values (pure key rename, the
            # arithmetic above is untouched). The honest decision-value quantities
            # are ``decision_evpi`` (S1) and ``factor_evppi`` (S2), both in outcome
            # units on the joint CRN population.
            results.append(
                {
                    "factor_id": uncertainty.node_id,
                    "p_win_delta": round(delta, 6),
                    "p_win_delta_percentage_points": round(delta * 100, 2),
                    "current_metric": round(baseline_metric, 6),
                    "perfect_metric": round(perfect_metric, 6),
                    "metric_type": "p_joint_goal"
                    if request.goal_constraints
                    else "p_win_recommended",
                    "method": P_WIN_SENSITIVITY_METHOD,
                    "n_samples": n_samples,
                    # Additive labelling fields (provisional_doctrine_v0).
                    # Safe additive extension: p_win_sensitivity entries are
                    # Dict[str, Any] at every hop (analyzer -> V1 model ->
                    # V2 envelope) and no cross-service consumer parses
                    # these entries strictly (verified 2026-07-07: PLoT has
                    # no factor_evpi/p_win_sensitivity reference; DGAI debug
                    # export treats it as unknown[]).
                    "status": "below_resolution" if below_resolution else "resolved",
                    "clamped": delta_clamped,
                    "noise_floor": round(noise_floor, 6),
                    "noise_floor_method": "z95_worst_case_bernoulli_diff",
                    "labelling_doctrine": EVPI_LABELLING_DOCTRINE,
                }
            )

        # Sort by the win-probability delta descending (most sensitive factor
        # first) — same order as the pre-S2 factor_evpi block (sorted on ``evpi``,
        # the same value now named ``p_win_delta``).
        results.sort(key=lambda x: float(x["p_win_delta"]), reverse=True)
        return results

    def _compute_evpi_metric(
        self,
        request: RobustnessRequestV2,
        sampler: DualUncertaintySampler,
        factor_sampler: FactorSampler,
        evaluator: SCMEvaluatorV2,
        n_samples: int,
        constraint_target_nodes: Optional[List[str]],
        recommended_option_id: str,
        deadline: Optional["PhaseDeadline"] = None,
        constraint_plans: Optional[Dict[int, "GoalThresholdPlan"]] = None,
        objective: Optional["ObjectivePlan"] = None,
    ) -> Optional[float]:
        """Compute the EVPI metric for a fixed decision policy over n_samples.

        ROADMAP 2.798 — THE SECOND PRODUCER. When goal_constraints exist this
        metric IS P(joint_goal), folded from the same
        ``_compute_constraint_probabilities`` the wire block uses. Fixing only the
        visible channel would have left the identical frame collision alive here,
        feeding ``p_win_sensitivity`` — the same untruth wearing a different field
        name. So this loop records its own status-quo reference and resolves each
        constraint's samples through the same plans.

        Uses recommended_option_id (from the main MC run) as the fixed policy
        to avoid policy-switch confounding across EVPI runs.

        Codex F7: ``deadline`` (the EVPI phase's PhaseDeadline, sharing its t0)
        threads the governing request deadline into the MC sample loop. Returns
        None when the deadline is exceeded mid-loop so the caller discards the
        whole EVPI phase (all-or-nothing). ``deadline=None`` disables the guard —
        direct/legacy callers are unaffected.
        """
        option_outcomes: Dict[str, List[float]] = {opt.id: [] for opt in request.options}
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]] = None
        if constraint_target_nodes:
            constraint_node_values = {
                opt.id: {nid: [] for nid in constraint_target_nodes} for opt in request.options
            }

        # ROADMAP 2.798: the per-draw status-quo reference for any level-framed
        # constraint, on the SAME edge/factor draws as the options below (common
        # random numbers). Its evaluator carries no epsilon RNG, so recording it
        # consumes no draws and cannot shift this loop's existing output.
        sq_reference_nodes = self._constraint_status_quo_nodes(request, constraint_plans)
        # ROADMAP 2.1192: a level-framed TARGET objective needs the goal's own
        # per-draw reference here too, for the same reason the constraint
        # channel needs its targets' — a level is only recoverable against a
        # no-intervention draw under common random numbers. Added to the SAME
        # set rather than recorded separately, so there is one reference series
        # per node and no second dialect of the same idea.
        if objective is not None and objective.needs_status_quo_reference:
            sq_reference_nodes = sorted(set(sq_reference_nodes) | {request.goal_node_id})
        status_quo_node_values: Dict[str, List[float]] = {
            node_id: [] for node_id in sq_reference_nodes
        }
        sq_evaluator = SCMEvaluatorV2(request.graph) if sq_reference_nodes else None

        for i in range(n_samples):
            # F7: periodic wall-clock deadline re-check (mirrors the E-value
            # per-bisect-step cadence — NOT every evaluate() call; the guard reads
            # no RNG so byte-output is unchanged when it does not trip). On overrun
            # return None so the caller discards the whole EVPI phase. The
            # EVPI_DEADLINE_CHECK_INTERVAL cadence stays local; deadline.exceeded()
            # owns the trip test (shares the EVPI phase t0).
            if (
                deadline is not None
                and i % self.EVPI_DEADLINE_CHECK_INTERVAL == 0
                and deadline.exceeded()
            ):
                return None
            edge_config = sampler.sample_edge_configuration()
            factor_values = factor_sampler.sample_factor_values()

            if sq_evaluator is not None:
                reference_values = sq_evaluator.evaluate_multi(
                    edge_strengths=edge_config,
                    interventions={},
                    target_nodes=sq_reference_nodes,
                    factor_values=factor_values,
                )
                for node_id in sq_reference_nodes:
                    status_quo_node_values[node_id].append(reference_values[node_id])

            for option in request.options:
                if constraint_target_nodes and constraint_node_values is not None:
                    all_targets = list(set([request.goal_node_id] + constraint_target_nodes))
                    node_values = evaluator.evaluate_multi(
                        edge_strengths=edge_config,
                        interventions=option.interventions,
                        target_nodes=all_targets,
                        factor_values=factor_values,
                    )
                    outcome = node_values.get(request.goal_node_id, 0.0)
                    for nid in constraint_target_nodes:
                        constraint_node_values[option.id][nid].append(node_values.get(nid, 0.0))
                else:
                    outcome = evaluator.evaluate(
                        edge_strengths=edge_config,
                        interventions=option.interventions,
                        goal_node=request.goal_node_id,
                        factor_values=factor_values,
                    )
                option_outcomes[option.id].append(outcome)

        if request.goal_constraints and constraint_node_values is not None:
            # P(joint_goal) for the fixed recommended option.
            #
            # ROADMAP 2.798: unresolvable constraints mean there is no joint
            # probability to be had, so the WHOLE phase is discarded rather than
            # computed on a fabricated one. None is the phase's existing
            # all-or-nothing signal and analyze() discloses it.
            if constraint_plans is None:
                return None
            resolved_values = self._resolve_constraint_series(
                constraint_node_values,
                request.goal_constraints,
                constraint_plans,
                status_quo_node_values,
                recommended_option_id,
            )
            _, joint_prob, _ = self._compute_constraint_probabilities(
                resolved_values,
                request.goal_constraints,
            )
            return joint_prob
        else:
            # P(win) of the fixed recommended option.
            # Tie-breaking mirrors main MC: equal credit split among tied options
            # to avoid insertion-order bias (see _run_monte_carlo tie logic).
            # P(win) of the fixed recommended option — ROADMAP 2.1192.
            #
            # THE SECOND COPY OF THE WINNER RULE, now routed through its one
            # owner. This loop was an independent unconditional ``max()``; it
            # agreed with the main Monte Carlo only because both were hardcoded
            # to the same sense. Parameterising the ranking and leaving this
            # alone would have made ``p_win_sensitivity`` measure a maximiser's
            # win probability while the ranking beside it measured the team's
            # stated objective — two authorities, one label, no rule for which
            # one a surface should believe.
            #
            # A level-framed target needs this loop's own status-quo reference
            # (recorded above under the same common random numbers). When the
            # objective needs one and this loop has none, the phase is DISCARDED
            # — the same all-or-nothing signal an unresolvable constraint plan
            # already uses — rather than silently answering with a maximiser.
            plan = objective or ObjectivePlan(
                sense="withheld", attested=False, withheld_reason="goal_direction_absent"
            )
            goal_reference_series = status_quo_node_values.get(request.goal_node_id)
            if plan.needs_status_quo_reference and not goal_reference_series:
                return None
            win_count = 0.0
            for i in range(n_samples):
                finite_i = {
                    oid: option_outcomes[oid][i]
                    for oid in option_outcomes
                    if math.isfinite(option_outcomes[oid][i])
                }
                winners = self._winners_for_draw(
                    finite_i,
                    plan,
                    goal_reference_series[i] if goal_reference_series else None,
                )
                if recommended_option_id in winners:
                    win_count += 1.0 / len(winners)
            return win_count / n_samples

    def _compute_alternative_winners(
        self,
        fragile_edge_info: Dict[str, Tuple[str, str]],
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]],
        winner_per_sample: List[Optional[str]],
        overall_winner: str,
        request: Optional[RobustnessRequestV2] = None,
        evaluator: Optional[SCMEvaluatorV2] = None,
        global_seed: Optional[int] = None,
        edge_max_elasticity: Optional[Dict[str, float]] = None,
        critiques: Optional[List[CritiqueV2]] = None,
    ) -> List[FragileEdgeEnhanced]:
        """
        Compute alternative winners for fragile edges.

        For each fragile edge, identifies which option wins most often when
        the edge is "weak" (bottom 25% of sampled strengths). Also computes
        marginal switch probability (isolated edge contribution) when
        request, evaluator, and global_seed are provided.

        ROADMAP 2.356 — TWO CHANGES, both about the marginal sweep's cost.

        1. THE BASELINE IS COMPUTED ONCE PER REQUEST, not once per edge. The
           baseline config is `{every edge: mean * exists_probability}` and the
           baseline winner follows from it — NEITHER depends on which edge is
           being probed. Recomputing them inside the per-edge loop spent O
           evaluate() calls per fragile edge to re-derive an identical answer.
           Hoisting is exactly equivalent (same config, same deterministic
           tie-break) and removes (F-1)*O evaluations.

        2. THE SWEEP IS BOUNDED to the MARGINAL_MAX_EDGES most elastic fragile
           edges. The fragile set was threshold-gated but not count-capped, so
           the phase's evaluate() count was bounded only by the edge count and
           the compute-admission gate could not price it — the whole ceiling was
           clearable by a request that then did several times the admitted work.
           Omitted edges keep `switch_probability` (free — it partitions samples
           already drawn) and lose only `marginal_switch_probability`, which is
           set to None and disclosed via MARGINAL_SWITCH_TRUNCATED.

        Args:
            fragile_edge_info: Map of edge_id -> (from_id, to_id)
            edge_configs_per_sample: Edge strengths for each MC sample
            winner_per_sample: Winner option ID for each MC sample
            overall_winner: The overall recommended option
            request: Full robustness request with graph and options (optional)
            evaluator: SCM evaluator instance (optional)
            global_seed: Request-level seed for reproducibility (optional)
            edge_max_elasticity: edge_id -> max |elasticity|, the ranking key for
                the top-K selection. When absent (direct callers in tests), the
                selection falls back to sorted edge_id order so the cap still
                binds deterministically and the bound still holds.
            critiques: Optional sink for the truncation disclosure.

        Returns:
            List of FragileEdgeEnhanced objects with enhanced fragile edge information
        """
        # Check if marginal calculation is possible
        can_compute_marginal = (
            request is not None and evaluator is not None and global_seed is not None
        )

        # --- select the edges whose marginal sweep we will actually pay for ----
        # Rank by the elasticity the sensitivity phase already computed, so the
        # edges a reader would look at first are the ones that keep their number.
        # Ties break on edge_id, so the selection is deterministic across
        # processes (the same class of defect as the fragile-edge set ordering
        # fixed in the science-validation report §5.7b).
        all_edge_ids = sorted(fragile_edge_info)
        if edge_max_elasticity is not None:
            ranked = sorted(
                all_edge_ids,
                key=lambda eid: (-abs(edge_max_elasticity.get(eid, 0.0)), eid),
            )
        else:
            ranked = all_edge_ids
        priced_edge_ids = set(ranked[:MARGINAL_MAX_EDGES])
        omitted_count = len(all_edge_ids) - len(priced_edge_ids)

        if can_compute_marginal and omitted_count > 0 and critiques is not None:
            critiques.append(
                MARGINAL_SWITCH_TRUNCATED.build(
                    computed=len(priced_edge_ids),
                    total=len(all_edge_ids),
                    omitted=omitted_count,
                    k_samples=MARGINAL_K_SAMPLES,
                    affected_node_ids=sorted(
                        {fragile_edge_info[eid][1] for eid in ranked[MARGINAL_MAX_EDGES:]}
                    ),
                    seed=global_seed,
                )
            )

        # --- the once-per-request baseline (see change 1 above) ----------------
        marginal_baseline: Optional[Tuple[Dict[Tuple[str, str], float], str]] = None
        if can_compute_marginal and priced_edge_ids:
            assert request is not None
            assert evaluator is not None
            marginal_baseline = self._compute_marginal_baseline(request, evaluator)

        results = []

        for edge_id, (from_id, to_id) in fragile_edge_info.items():
            edge_key = (from_id, to_id)

            # Compute marginal switch probability (isolated edge contribution).
            # Only for the priced (top-K) edges, and only when all required
            # parameters are provided.
            marginal_prob: Optional[float] = None
            if can_compute_marginal and edge_id in priced_edge_ids:
                assert request is not None
                assert evaluator is not None
                assert global_seed is not None
                assert marginal_baseline is not None
                marginal_prob = self._compute_marginal_switch_probability(
                    edge_key=edge_key,
                    request=request,
                    evaluator=evaluator,
                    global_seed=global_seed,
                    baseline=marginal_baseline,
                )

            # Collect edge strengths across all samples
            strengths = [config.get(edge_key, 0.0) for config in edge_configs_per_sample]

            if not strengths:
                # No data for this edge (joint sampling unavailable)
                results.append(
                    FragileEdgeEnhanced(
                        edge_id=edge_id,
                        from_id=from_id,
                        to_id=to_id,
                        alternative_winner_id=None,
                        switch_probability=None,
                        marginal_switch_probability=marginal_prob,
                    )
                )
                continue

            # Find bottom 25% threshold (weak edge samples)
            strength_array = np.array(strengths)
            weak_threshold = np.percentile(strength_array, 25)

            # Get samples where edge is weak
            weak_sample_indices = [i for i, s in enumerate(strengths) if s <= weak_threshold]

            if not weak_sample_indices:
                results.append(
                    FragileEdgeEnhanced(
                        edge_id=edge_id,
                        from_id=from_id,
                        to_id=to_id,
                        alternative_winner_id=None,
                        switch_probability=None,
                        marginal_switch_probability=marginal_prob,
                    )
                )
                continue

            # Count winner distribution in weak-edge samples
            weak_winner_counts: Dict[str, int] = defaultdict(int)
            for idx in weak_sample_indices:
                # 2.477(c): skip draws where no option was finite — they have no
                # winner to attribute. Counting a None would invent a phantom
                # option and could make it the "weak winner".
                weak_sample_winner = winner_per_sample[idx]
                if weak_sample_winner is None:
                    continue
                weak_winner_counts[weak_sample_winner] += 1

            if not weak_winner_counts:
                # Every weak-edge draw was uninformative — no attribution to make.
                continue

            # Find most frequent winner in weak-edge samples
            weak_winner = max(weak_winner_counts, key=lambda k: weak_winner_counts[k])
            weak_winner_count = weak_winner_counts[weak_winner]
            total_weak_samples = len(weak_sample_indices)

            # Determine alternative winner and switch probability
            # The alternative is the best option OTHER than the overall winner
            # switch_probability is the probability of that alternative in weak scenarios
            if weak_winner != overall_winner:
                # Clear case: a different option wins when edge is weak
                alternative_winner_id = weak_winner
                switch_probability = weak_winner_count / total_weak_samples
            else:
                # Same option wins most often, but we want to show the risk
                # Find the best alternative (second most frequent) and its probability
                alternatives = {
                    opt: count for opt, count in weak_winner_counts.items() if opt != overall_winner
                }
                if alternatives:
                    # There's at least one alternative winner in weak scenarios
                    best_alt = max(alternatives, key=lambda k: alternatives[k])
                    alternative_winner_id = best_alt
                    switch_probability = alternatives[best_alt] / total_weak_samples
                else:
                    # Only the overall winner appeared in weak scenarios - truly stable
                    alternative_winner_id = None
                    switch_probability = 0.0

            results.append(
                FragileEdgeEnhanced(
                    edge_id=edge_id,
                    from_id=from_id,
                    to_id=to_id,
                    alternative_winner_id=alternative_winner_id,
                    switch_probability=switch_probability,
                    marginal_switch_probability=marginal_prob,
                )
            )

        return results

    def _compute_marginal_baseline(
        self,
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
    ) -> Tuple[Dict[Tuple[str, str], float], str]:
        """The expected-value config and its winner — shared by every edge probe.

        ROADMAP 2.356. Extracted verbatim from
        `_compute_marginal_switch_probability`, where it ran once PER FRAGILE
        EDGE and produced the same answer every time: neither the config
        (`mean * exists_probability` for every edge) nor the winner derived from
        it takes the probed edge as an input. Hoisting it removes (F-1)*O
        evaluate() calls with no change to any returned number.

        Its O evaluations are the `1 +` inside the `alternative_winners` term, so
        it is registered `subsumed:_compute_alternative_winners` rather than
        priced separately.

        ⚠ AN EARLIER DRAFT OF THIS DOCSTRING CLAIMED THE METHOD WAS "deliberately
        named without the `_compute_` prefix so it does not enter the phase
        inventory". It was named `_compute_marginal_baseline`, so that claim was
        false on its own line, and TestPhasePricingInventory caught it within one
        run. Recorded rather than quietly deleted: the prefix tripwire is exactly
        the assume-good mirror that trap 12 says must fail loud, and here it did.
        """
        baseline_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
        }
        baseline_outcomes = {}
        for option in request.options:
            baseline_outcomes[option.id] = evaluator.evaluate(
                edge_strengths=baseline_config,
                interventions=option.interventions,
                goal_node=request.goal_node_id,
            )
        # Deterministic tie-breaking for baseline winner
        sorted_baseline = sorted(baseline_outcomes.items(), key=lambda x: (-x[1], x[0]))
        return baseline_config, sorted_baseline[0][0]

    def _compute_marginal_switch_probability(
        self,
        edge_key: Tuple[str, str],
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        global_seed: int,
        k_samples: int = MARGINAL_K_SAMPLES,
        baseline: Optional[Tuple[Dict[Tuple[str, str], float], str]] = None,
    ) -> float:
        """Compute probability of decision flip when ONLY this edge varies.

        Samples this edge K times from its uncertainty distribution while holding
        all other edges at their expected values (mean * exists_probability).
        Returns fraction of samples where the winner changes.

        Baseline semantics:
        - Other edges: held at expected value (mean * exists_probability)
        - Target edge: samples BOTH existence (Bernoulli) AND strength (Normal)
        - Baseline winner: computed under the same baseline config (not MC overall_winner)

        This ensures the marginal calculation is self-consistent: we compare
        sampled outcomes against the winner under the same baseline assumptions.

        Args:
            edge_key: (from_id, to_id) tuple
            request: Full robustness request with graph and options
            evaluator: SCM evaluator instance
            global_seed: Request-level seed for reproducibility
            k_samples: Number of samples (default 100)

        Returns:
            Probability in [0.0, 1.0] that this edge alone flips the recommendation
        """
        from_id, to_id = edge_key

        # Deterministic seed: SHA256 (process-safe, NOT Python hash())
        edge_seed_str = f"{global_seed}:{from_id}->{to_id}"
        edge_seed = int(hashlib.sha256(edge_seed_str.encode()).hexdigest()[:8], 16)
        rng = SeededRNG(edge_seed)

        # Get target edge's parameters
        edge = next((e for e in request.graph.edges if (e.from_, e.to) == edge_key), None)
        if edge is None:
            self.logger.warning(
                "marginal_switch_edge_not_found",
                extra={"edge_key": f"{from_id}->{to_id}"},
            )
            return 0.0

        # Baseline config (all edges at expected value) and the winner under it —
        # NOT overall_winner from the MC, so the comparison is self-consistent.
        # ROADMAP 2.356: computed once per request and passed in; recomputed here
        # only for direct callers that do not supply it (tests), which keeps this
        # method's contract unchanged.
        baseline_config, marginal_baseline_winner = (
            baseline
            if baseline is not None
            else self._compute_marginal_baseline(request, evaluator)
        )

        flip_count = 0

        for _ in range(k_samples):
            # Sample existence (Bernoulli)
            exists = rng.bernoulli(edge.exists_probability)

            if not exists:
                # Edge doesn't exist in this sample → effective strength is 0
                sampled_strength = 0.0
            else:
                # Truncated normal — rejection sampling within schema bounds
                sampled_strength = rng.truncated_normal(
                    edge.strength.mean,
                    edge.strength.std,
                    EDGE_STRENGTH_MIN,
                    EDGE_STRENGTH_MAX,
                )

            # Build counterfactual config: this edge sampled, others at baseline
            counterfactual_config = baseline_config.copy()
            counterfactual_config[edge_key] = sampled_strength

            # Evaluate all options under counterfactual
            outcomes = {}
            for option in request.options:
                outcomes[option.id] = evaluator.evaluate(
                    edge_strengths=counterfactual_config,
                    interventions=option.interventions,
                    goal_node=request.goal_node_id,
                )

            # Determine winner with deterministic tie-breaking (sort by option_id)
            sorted_options = sorted(outcomes.items(), key=lambda x: (-x[1], x[0]))
            sample_winner = sorted_options[0][0]

            if sample_winner != marginal_baseline_winner:
                flip_count += 1

        return flip_count / k_samples

    # =========================================================================
    # Multi-Constraint Goal Analysis (Phase 2)
    # =========================================================================

    def _check_constraint_satisfied(
        self,
        value: float,
        constraint: GoalConstraint,
    ) -> bool:
        """
        Check if a value satisfies a constraint.

        Args:
            value: The node value to check
            constraint: The constraint to check against

        Returns:
            True if value satisfies constraint, False otherwise
        """
        if constraint.operator == ">=":
            return value >= constraint.threshold
        elif constraint.operator == "<=":
            return value <= constraint.threshold
        else:
            # This should never happen due to Pydantic validation
            raise ValueError(f"Unknown operator: {constraint.operator}")

    def _compute_constraint_probabilities(
        self,
        resolved_values: Dict[int, List[float]],
        constraints: List[GoalConstraint],
    ) -> Tuple[Dict[str, float], float, List[List[bool]]]:
        """
        Compute per-constraint and joint probabilities for an option.

        ROADMAP 2.798: takes samples ALREADY RESOLVED into each constraint's own
        frame by ``_resolve_constraint_series``, keyed by constraint INDEX. It no
        longer reaches into ``constraint_node_values`` by node_id, and that is the
        point: there is no longer a path from a raw change-frame sample to a
        comparison against a level threshold, so the category error that produced
        a structural zero cannot be reintroduced by a caller here.

        Args:
            resolved_values: Dict[constraint_index, List[sample_values]] in the
                frame that constraint's threshold is stated in.
            constraints: List of GoalConstraint objects

        Returns:
            Tuple of:
            - per_constraint_probs: Dict[constraint_index_str, prob_satisfied]
            - joint_probability: P(all constraints satisfied)
            - satisfaction_matrix: List[sample_idx][constraint_idx] -> bool (for conditional prob)
        """
        if not constraints:
            return {}, 1.0, []

        n_samples = len(next(iter(resolved_values.values())))

        # Build satisfaction matrix: [sample_idx][constraint_idx] -> bool
        satisfaction_matrix: List[List[bool]] = []
        for sample_idx in range(n_samples):
            sample_satisfactions = []
            for c_idx, constraint in enumerate(constraints):
                value = resolved_values[c_idx][sample_idx]
                satisfied = self._check_constraint_satisfied(value, constraint)
                sample_satisfactions.append(satisfied)
            satisfaction_matrix.append(sample_satisfactions)

        # Per-constraint probabilities
        per_constraint_probs = {}
        for c_idx, constraint in enumerate(constraints):
            satisfied_count = sum(1 for sample in satisfaction_matrix if sample[c_idx])
            per_constraint_probs[str(c_idx)] = satisfied_count / n_samples

        # Joint probability: all constraints satisfied
        joint_satisfied_count = sum(1 for sample in satisfaction_matrix if all(sample))
        joint_probability = joint_satisfied_count / n_samples

        return per_constraint_probs, joint_probability, satisfaction_matrix

    def _compute_conditional_probabilities(
        self,
        satisfaction_matrix: List[List[bool]],
        constraints: List[GoalConstraint],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise conditional probabilities: P(C_j | C_i).

        Args:
            satisfaction_matrix: [sample_idx][constraint_idx] -> bool
            constraints: List of GoalConstraint objects

        Returns:
            Dict[constraint_i_idx, Dict[constraint_j_idx, conditional_prob]]
            Where conditional_prob is P(C_j | C_i) = P(C_i and C_j) / P(C_i)
        """
        if len(constraints) < 2:
            return {}

        n_samples = len(satisfaction_matrix)
        n_constraints = len(constraints)

        conditional_probs: Dict[str, Dict[str, float]] = {}

        for i in range(n_constraints):
            conditional_probs[str(i)] = {}
            # Count samples where constraint i is satisfied
            count_i = sum(1 for sample in satisfaction_matrix if sample[i])

            if count_i == 0:
                # P(C_j | C_i) is undefined when P(C_i) = 0 - omit these entries
                # The dict for this constraint remains empty, indicating undefined
                pass
            else:
                for j in range(n_constraints):
                    if i != j:
                        # Count samples where both i and j are satisfied
                        count_ij = sum(
                            1 for sample in satisfaction_matrix if sample[i] and sample[j]
                        )
                        conditional_probs[str(i)][str(j)] = count_ij / count_i

        return conditional_probs

    def _compute_near_miss_diagnostics(
        self,
        resolved_values: Dict[int, List[float]],
        constraints: List[GoalConstraint],
        satisfaction_matrix: List[List[bool]],
        near_miss_fraction_threshold: float = 0.1,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute near-miss diagnostics for each constraint.

        For each constraint, computes:
        - failure_margin_median: Median distance from threshold when constraint fails
        - near_miss_fraction: Fraction of failures within near_miss_fraction_threshold of threshold
        - binding: True if prob_satisfied ∈ [0.4, 0.6] (constraint is borderline)

        ROADMAP 2.798: the margins ride the SAME resolved series as the
        probabilities. Before that they were computed on the raw change-frame
        samples, so a user whose goal was stated as a level was told a shortfall
        measured between two quantities that were not the same kind of thing —
        the "you are GBP 200k short" number in the L60 witness. A diagnostic
        derived from an unreconciled comparison is the same untruth as the
        probability derived from it, and it must not survive the fix that removed
        the probability.

        Args:
            resolved_values: Dict[constraint_index, List[sample_values]] in the
                frame that constraint's threshold is stated in.
            constraints: List of GoalConstraint objects
            satisfaction_matrix: Precomputed satisfaction matrix
            near_miss_fraction_threshold: Relative threshold for "near miss" (default 10%)

        Returns:
            Dict[constraint_idx, {failure_margin_median, near_miss_fraction, binding}]
        """
        if not constraints:
            return {}

        n_samples = len(satisfaction_matrix)

        diagnostics: Dict[int, Dict[str, Any]] = {}

        for c_idx, constraint in enumerate(constraints):
            values = resolved_values[c_idx]
            threshold = constraint.threshold

            # Get failure samples
            failure_margins = []
            near_miss_count = 0

            for sample_idx, satisfied in enumerate(sample[c_idx] for sample in satisfaction_matrix):
                if not satisfied:
                    value = values[sample_idx]
                    # Compute margin (distance from threshold)
                    if constraint.operator == ">=":
                        # For >= threshold, margin is threshold - value (positive when failing)
                        margin = threshold - value
                    else:  # <=
                        # For <= threshold, margin is value - threshold (positive when failing)
                        margin = value - threshold

                    failure_margins.append(margin)

                    # Check if near-miss (within threshold% of the threshold value)
                    threshold_abs = abs(threshold) if threshold != 0 else 1.0
                    if margin <= near_miss_fraction_threshold * threshold_abs:
                        near_miss_count += 1

            # Compute diagnostics
            n_failures = len(failure_margins)
            failure_margin_median = float(np.median(failure_margins)) if failure_margins else None
            near_miss_fraction = near_miss_count / n_failures if n_failures > 0 else None

            # Compute prob_satisfied for binding determination
            satisfied_count = sum(1 for sample in satisfaction_matrix if sample[c_idx])
            prob_satisfied = satisfied_count / n_samples
            binding = 0.4 <= prob_satisfied <= 0.6

            diagnostics[c_idx] = {
                "failure_margin_median": failure_margin_median,
                "near_miss_fraction": near_miss_fraction,
                "binding": binding,
            }

        return diagnostics

    def _compute_constraint_analysis(
        self,
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]],
        constraints: Optional[List[GoalConstraint]],
        option_id: str,
        constraint_plans: Optional[Dict[int, "GoalThresholdPlan"]] = None,
        status_quo_node_values: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute full constraint analysis for an option, or REFUSE (ROADMAP 2.798).

        THE REFUSAL POINT. ``constraint_plans is None`` means at least one
        constraint could not be proved comparable against its target's samples, so
        this returns None and the caller omits ``constraint_analysis`` entirely.
        That is the whole difference between this and the pre-2.798 behaviour: a
        block that cannot be computed honestly is now ABSENT rather than filled
        with a number whose only property is that it looks like a probability.

        Args:
            constraint_node_values: Dict[option_id, Dict[node_id, List[sample_values]]]
            constraints: List of GoalConstraint objects
            option_id: The option to compute analysis for
            constraint_plans: Resolved per-constraint comparison plans. None =>
                REFUSE. Defaulted for direct callers that pass no constraints.
            status_quo_node_values: Per-draw no-intervention series per target
                node, CRN-paired with the option samples.

        Returns:
            Dict with constraint analysis results, or None if no constraints or
            the block was refused.
        """
        if not constraints or not constraint_node_values:
            return None

        if constraint_plans is None:
            # At least one constraint is unresolvable. A joint probability over a
            # resolved conjunct AND an unresolved one is not a probability of
            # anything, so the block is omitted whole. The caller has already
            # emitted a warning naming each refused constraint.
            return None

        # Put every constraint's samples into the frame ITS threshold is stated
        # in, once, before any comparison sees them.
        resolved_values = self._resolve_constraint_series(
            constraint_node_values,
            constraints,
            constraint_plans,
            status_quo_node_values or {},
            option_id,
        )

        # T3: Per-constraint and joint probability
        (
            per_constraint_probs,
            joint_probability,
            satisfaction_matrix,
        ) = self._compute_constraint_probabilities(resolved_values, constraints)

        # T4: Pairwise conditional probabilities
        conditional_probs = self._compute_conditional_probabilities(
            satisfaction_matrix, constraints
        )

        # T5: Near-miss diagnostics — on the SAME resolved series.
        near_miss_diagnostics = self._compute_near_miss_diagnostics(
            resolved_values, constraints, satisfaction_matrix
        )

        # Build constraint results
        constraint_results = []
        for c_idx, constraint in enumerate(constraints):
            diag = near_miss_diagnostics.get(c_idx, {})
            constraint_results.append(
                {
                    # Slice 6b: carry the caller's opaque id straight through.
                    # Echo-only — it is never read by any computation here.
                    "constraint_id": constraint.constraint_id,
                    "node_id": constraint.node_id,
                    "operator": constraint.operator,
                    "threshold": constraint.threshold,
                    "label": constraint.label,
                    "prob_satisfied": per_constraint_probs.get(str(c_idx), 0.0),
                    "failure_margin_median": diag.get("failure_margin_median"),
                    "near_miss_fraction": diag.get("near_miss_fraction"),
                    "binding": diag.get("binding", False),
                }
            )

        return {
            "constraints": constraint_results,
            "joint_probability": joint_probability,
            "conditional_probabilities": conditional_probs if conditional_probs else None,
        }
