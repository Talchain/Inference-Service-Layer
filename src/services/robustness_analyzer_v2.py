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
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    CorrelationModelV2,
    CorrelationProjectionV2,
    ZeroSensitivityReason,
)
from src.constants import (
    ELASTICITY_CLAMP_MAX,
    FACTOR_SENSITIVITY_BASELINE_EPSILON,
    FACTOR_SENSITIVITY_VALUE_EPSILON,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_NODES,
    MAX_OPTIONS,
    MAX_PARAMETER_UNCERTAINTIES,
    ZERO_VARIANCE_TOLERANCE,
)
from src.models.critique import (
    CONSTRAINT_NODE_DEFAULT_BASE,
    CONSTRAINT_NODE_DEFAULT_BASE_OBJECTIVE,
    CONSTRAINT_NODE_DEFAULT_BASE_SUPPORTED,
    GOAL_ANCESTOR_DATA_GAP,
    DEGENERATE_OPTION_ZERO_VARIANCE,
    HIGH_TIE_RATE,
)
from src.models.response_v2 import CritiqueV2
from src.utils.rng import SEED_HASH_VERSION, SeededRNG, compute_seed_from_graph
from src.utils.downside import expected_regret_per_option
from src.utils.evppi import REGRESSION_EVPPI_METHOD, factor_evppi_estimate
from src.utils.correlation import CORRELATION_METHOD, CorrelationPlan, build_correlation_plan
from src.validation.request_validator import detect_graph_cycle
from src.__version__ import __version__
from src.models.metadata import generate_config_fingerprint
from src.config.stability_thresholds import (
    STABILITY_THRESHOLDS,
    classify_attribution_stability,
)

logger = logging.getLogger(__name__)

# Safety net: nodes that must not participate in inference
NON_INFERENCE_KINDS = {"decision", "option", "constraint"}

# Path-decomposition safety budget: maximum number of simple intervention-target-to-goal
# paths to enumerate. A layered DAG valid under the 50-node/200-edge schema limits can have
# hundreds of thousands of simple paths; without this cap, enumeration would blow the
# sub-500ms budget. The bound is a path COUNT (not wall-clock) so truncation is deterministic
# — the same graph always truncates identically, preserving the determinism guarantee.
MAX_DECOMPOSITION_PATHS = 20000

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
COMPLEXITY_FORMULA_VERSION = "v2-weighted-2026-07"

# Per-phase structural weights (provisional; the calibration harness is the
# source of truth for refining them — do not hand-tune without re-running it).
BASE_COST_COEF = 1  # base MC: 1 unit per sample x option x (nodes+edges) evaluate()
W_SENS_COEF = 4  # edge sensitivity: 4 sub-sweeps per edge (existence +/- , magnitude +/-)
W_EVAL_COEF = 20  # e-values: ~binary-search depth per edge (wall-clock-capped, so flat)
W_BANDS_COEF = 200  # stability bands: 10 seeds x ~20 search per edge (capped, so flat)
W_PATH_COEF = 1  # path decomposition: analytic, bounded by MAX_DECOMPOSITION_PATHS

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
             + (U+1)*min(S, EVPI_SAMPLE_CAP)*O*W           (EVPI, if include_voi & U>0)
             + W_SENS_COEF*E*min(100, S//10)*W             (edge sensitivity)
             + W_EVAL_COEF*E*O                             (e-values, if include_e_values)
             + W_BANDS_COEF*E*O                            (bands, ride on e-values)
             + W_PATH_COEF*min(MAX_DECOMPOSITION_PATHS, E*E) (path decomp)

    where S=n_samples, O=len(options), N=n_nodes, E=n_edges, W=N+E (per-evaluate()
    structural work), U=number of UNIQUE parameter_uncertainties. Every term
    mirrors an actual loop body in the analyzer (see the F8 design phase
    inventory). Optional-phase enable conditions match analyze():
      - EVPI: request.include_voi AND at least one parameter_uncertainty
      - edge sensitivity: "sensitivity" in analysis_types
      - e-values / bands: request.include_e_values (bands are default-on with e-values)
      - path decomposition: request.include_path_decomposition
    """
    S = request.n_samples
    O = len(request.options)
    N = len(request.graph.nodes)
    E = len(request.graph.edges)
    W = N + E

    terms: Dict[str, int] = {"base_mc": BASE_COST_COEF * S * O * W}

    # EVPI — priced on the DEDUPLICATED factor count (uniqueness is enforced at
    # parse time, but count unique defensively so admission never over-prices a
    # duplicate that somehow reached here).
    if request.include_voi and request.parameter_uncertainties:
        u = len({pu.node_id for pu in request.parameter_uncertainties})
        if u > 0:
            terms["evpi"] = (u + 1) * min(S, EVPI_SAMPLE_CAP) * O * W

    # Edge sensitivity — reference option only (not multiplied by O).
    if "sensitivity" in request.analysis_types:
        terms["sensitivity"] = W_SENS_COEF * E * min(100, S // 10) * W

    # E-values and the stability bands that ride on them (bands default-on).
    if request.include_e_values:
        terms["e_values"] = W_EVAL_COEF * E * O
        terms["bands"] = W_BANDS_COEF * E * O

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
    """
    return {
        "max_cost_units": get_max_cost_units(),
        "complexity_formula_version": COMPLEXITY_FORMULA_VERSION,
        "weights": {
            "base_per_sample_per_option_per_struct": BASE_COST_COEF,
            "evpi_sample_cap": EVPI_SAMPLE_CAP,
            "sensitivity_coef": W_SENS_COEF,
            "evalue_coef": W_EVAL_COEF,
            "bands_coef": W_BANDS_COEF,
            "path_coef": W_PATH_COEF,
            "max_decomposition_paths": MAX_DECOMPOSITION_PATHS,
        },
        "caps": {
            "max_options": MAX_OPTIONS,
            "max_nodes": MAX_GRAPH_NODES,
            "max_edges": MAX_GRAPH_EDGES,
            "max_parameter_uncertainties": MAX_PARAMETER_UNCERTAINTIES,
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

            # Get mean from observed_state.value, default to 0
            mean = 0.0
            if node.observed_state and node.observed_state.value is not None:
                mean = node.observed_state.value

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
            mean = 0.0
            if node.observed_state and node.observed_state.value is not None:
                mean = node.observed_state.value
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
        ) = self._run_monte_carlo(
            request, sampler, factor_sampler, evaluator, constraint_target_nodes
        )

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
        pre_noise_expected_regret = expected_regret_per_option(option_outcomes)

        # S2 (D-23.8) factor_evppi needs the PRE-noise per-option outcomes — the
        # same CRN-aligned joint population that produced pre_noise_expected_regret
        # and win_probability. _apply_auto_scaled_noise below reassigns each option's
        # list IN PLACE (independent per-option noise breaks CRN alignment), so snapshot
        # the pre-noise lists here. Only taken when the VOI phase will run.
        pre_noise_option_outcomes: Optional[Dict[str, List[float]]] = None
        if request.include_voi and factor_sampler.has_uncertainties():
            pre_noise_option_outcomes = {
                oid: list(vals) for oid, vals in option_outcomes.items()
            }

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
        results = self._compute_option_results(
            option_outcomes,
            option_wins,
            request,
            constraint_node_values,
            pre_noise_expected_regret,
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

        # Compute factor sensitivity if factor uncertainties are specified.
        # B3-S1 (D-23.4): SUPPRESSED under active correlation — per-factor OAT
        # elasticity perturbs one factor holding the others at their mean, an
        # off-manifold move that double-counts shared variance and mis-ranks
        # correlated factors. Omitted (absent, not fabricated) with the
        # correlation_model disclosure marker naming the reason.
        factor_sensitivity: List[FactorSensitivityResult] = []
        if (
            factor_sampler.has_uncertainties()
            and "sensitivity" in request.analysis_types
            and not correlation_active
        ):
            factor_sensitivity = self._compute_factor_sensitivity(
                request, option_outcomes, rng_factor, evaluator
            )

        # Compute conditional winners (factor-partitioned win probabilities).
        # B3-S1 (D-23.4): SUPPRESSED under active correlation — a single-factor
        # median split attributes a winner flip to one factor, but under
        # correlation that factor's low/high bucket also drags its correlated
        # partners, so the per-factor attribution is confounded. Omitted with the
        # disclosure marker (joint win_probability itself stays valid).
        conditional_winners = None
        if (
            factor_sampler.has_uncertainties()
            and len(request.options) > 1
            and not correlation_active
        ):
            conditional_winners = self._compute_conditional_winners(
                factor_values_per_sample,
                winner_per_sample,
                option_outcomes,
                factor_sampler,
                request,
            )

        # Compute robustness assessment (with alternative winner analysis)
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
        )

        # Compute E-value analogue per edge if requested. OPTIONAL phase —
        # governed by the overall request budget: skipped-with-disclosure when
        # insufficient budget remains, so the base + robustness results above
        # are never lost to a stacked-phase timeout.
        edge_e_values = None
        if request.include_e_values:
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

        # Find recommended option (needed before EVPI to fix decision policy)
        recommended_option_id = max(option_wins, key=lambda k: option_wins[k])
        recommendation_confidence = option_wins[recommended_option_id] / request.n_samples

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
        if request.include_voi and factor_sampler.has_uncertainties() and not correlation_active:
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
            and pre_noise_option_outcomes is not None
        ):
            # Per-factor EVPPI can never exceed the whole-decision EVPI (learning
            # ONE factor cannot beat learning EVERYTHING). decision_evpi on the
            # pre-noise CRN population = min_o expected_regret[o] = E[max]−max E;
            # this is the exact cap the emission clamps to (with disclosure).
            finite_regrets = [r for r in pre_noise_expected_regret.values() if math.isfinite(r)]
            decision_evpi_bound = min(finite_regrets) if finite_regrets else None
            try:
                factor_evppi = self._compute_factor_evppi(
                    request,
                    pre_noise_option_outcomes,
                    factor_values_per_sample,
                    seed,
                    decision_evpi_bound,
                    correlation_active,
                )
            except Exception:  # pragma: no cover - defensive degrade-with-disclosure
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

        # Compute structural pathway decomposition for the recommended option if requested.
        # Pass evaluator.graph — the post-filter graph the SCM actually computed on
        # (filter_inference_graph was applied before the evaluator was constructed), so the
        # decomposition explains exactly the structure the analysis used, not raw request.graph.
        path_decomposition = None
        if request.include_path_decomposition:
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
            request, correlation_plan, factor_sampler.has_uncertainties()
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
            conditional_winners=conditional_winners,
            stability_thresholds=stability_thresholds,
            edge_e_values=edge_e_values,
            p_win_sensitivity=p_win_sensitivity,
            factor_evppi=factor_evppi,
            path_decomposition=path_decomposition,
            correlation_model=correlation_model,
        )

        self.logger.info(
            "robustness_v2_analysis_complete",
            extra={
                "request_id": request_id,
                "recommended_option": recommended_option_id,
                "recommendation_confidence": recommendation_confidence,
                "is_robust": robustness.is_robust,
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
        validation has already rejected every hard-invalid input, so the assembled
        matrix is well-posed (PSD-checked + Higham-projected inside
        ``build_correlation_plan``).
        """
        correlations = request.factor_correlations
        if not correlations:
            return None
        correlated_ids: set = set()
        for corr in correlations:
            correlated_ids.add(corr.factor_a)
            correlated_ids.add(corr.factor_b)
        uncertainties = request.parameter_uncertainties or []
        factor_order = [u.node_id for u in uncertainties if u.node_id in correlated_ids]
        pairs = [(c.factor_a, c.factor_b, c.rho) for c in correlations]
        return build_correlation_plan(factor_order, pairs)

    @staticmethod
    def _build_correlation_disclosure(
        request: RobustnessRequestV2,
        correlation_plan: Optional[CorrelationPlan],
        has_uncertainties: bool,
    ) -> Optional[CorrelationModelV2]:
        """Assemble the ``correlation_model`` disclosure block (B3-S1, D-23.4).

        Returns None when correlation is inactive. When active it carries the
        method tag, the MANDATORY tail-independence caveat, any Higham PSD
        projection, and the manifest of suppressed independence-assuming per-
        factor attributions. Only attributions that WOULD have been computed
        (their enabling preconditions hold) are listed as suppressed — a field
        that was never going to be emitted is absent for a different reason and is
        not claimed as correlation-suppressed.
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
            )
            if projection is not None
            else None
        )

        suppressed: List[str] = []
        if has_uncertainties and "sensitivity" in request.analysis_types:
            suppressed.append("factor_sensitivity")
        if has_uncertainties and len(request.options) > 1:
            suppressed.append("conditional_winners")
        if has_uncertainties and request.include_voi:
            # S2 (D-23.8): the win-probability sensitivity block (renamed from
            # factor_evpi) stays suppressed under correlation (off-manifold OAT).
            # The NEW factor_evppi is NOT listed here — it is a conditional-
            # expectation quantity on the joint copula samples and IS emitted.
            suppressed.append("p_win_sensitivity")

        return CorrelationModelV2(
            method=CORRELATION_METHOD,
            active=True,
            correlated_factors=list(correlation_plan.factor_order),
            n_pairs=len(request.factor_correlations or []),
            tail_dependence="none",
            tail_dependence_note=_CORRELATION_TAIL_NOTE,
            psd_projection=psd_projection,
            suppressed_attributions=suppressed,
            suppression_reason=_CORRELATION_SUPPRESSION_REASON,
        )

    def _run_monte_carlo(
        self,
        request: RobustnessRequestV2,
        sampler: DualUncertaintySampler,
        factor_sampler: FactorSampler,
        evaluator: SCMEvaluatorV2,
        constraint_target_nodes: Optional[List[str]] = None,
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, float],
        List[str],
        List[Dict[Tuple[str, str], float]],
        int,
        Optional[Dict[str, Dict[str, List[float]]]],
        List[Dict[str, float]],
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

        Note: option_wins uses float to support split-tie handling where ties are
        divided equally among tied options.
        """
        option_outcomes: Dict[str, List[float]] = {opt.id: [] for opt in request.options}
        option_wins: Dict[str, float] = {opt.id: 0.0 for opt in request.options}
        winner_per_sample: List[str] = []
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]] = []
        factor_values_per_sample: List[Dict[str, float]] = []
        tie_count = 0

        # Initialize constraint node values tracking if needed
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]] = None
        if constraint_target_nodes:
            constraint_node_values = {
                opt.id: {node_id: [] for node_id in constraint_target_nodes}
                for opt in request.options
            }

        for _ in range(request.n_samples):
            # Sample edge configuration (structural + parametric uncertainty)
            edge_config = sampler.sample_edge_configuration()

            # Sample factor values (parameter uncertainty)
            factor_values = factor_sampler.sample_factor_values()
            factor_values_per_sample.append(factor_values)

            # Evaluate each option
            sample_outcomes = {}
            for option in request.options:
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

            # Track winner with fair tie-breaking (split ties equally)
            max_outcome = max(sample_outcomes.values())
            winners = [opt_id for opt_id, val in sample_outcomes.items() if val == max_outcome]

            if len(winners) == 1:
                # Clear winner
                option_wins[winners[0]] += 1.0
                winner_per_sample.append(winners[0])
            else:
                # Tie: split win equally among tied options
                tie_count += 1
                split_value = 1.0 / len(winners)
                for winner in winners:
                    option_wins[winner] += split_value
                # Use deterministic random tie-breaking via the existing sampler RNG.
                # This preserves full determinism (same seed = same tie-break result)
                # while eliminating insertion-order bias that arises from always
                # picking winners[0].  We reuse sampler.rng so the RNG stream
                # remains a single deterministic sequence — no new sub-seed needed.
                winner_per_sample.append(str(sampler.rng.choice(winners)))

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
          is not used as a base (only root nodes consult it).
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
        elif observed_value is not None:
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

    def _apply_auto_scaled_noise(
        self,
        option_outcomes: Dict[str, List[float]],
        goal_node_id: str,
        graph_nodes: List,
        rng: "SeededRNG",
        noise_multiplier: float = 1.0,
    ) -> Tuple[Dict[str, List[float]], bool]:
        """
        Apply auto-scaled noise to outcome/risk node samples.

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

        Returns:
            Tuple of (modified option_outcomes, noise_applied flag)
        """
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

    def _compute_option_results(
        self,
        outcomes: Dict[str, List[float]],
        wins: Dict[str, float],
        request: RobustnessRequestV2,
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]] = None,
        expected_regret: Optional[Dict[str, float]] = None,
    ) -> List[OptionResult]:
        """Compute distribution statistics for each option.

        Args:
            outcomes: Dict[option_id, List[outcome_samples]]
            wins: Dict[option_id, win_count]
            request: The analysis request
            constraint_node_values: Optional dict of constraint node sample values
                for multi-constraint analysis
            expected_regret: Optional dict[option_id, pre-noise JOINT expected
                regret]. Computed by the caller from the PRE-noise CRN-aligned
                outcomes (B2 CRN-fix F1) so it rides the same population as
                win_probability. Stored on OptionResult.pre_noise_expected_regret (serialized;
                survives offload) for the V2 emission layer. None -> not threaded.
        """
        expected_regret = expected_regret or {}
        results = []

        for option in request.options:
            samples = outcomes[option.id]
            if not samples:
                continue

            samples_array = np.array(samples)
            ci_lower, ci_upper = self._compute_confidence_interval(
                samples_array, request.confidence_level
            )

            # Compute probability_of_goal if threshold is provided
            probability_of_goal = None
            if request.goal_threshold is not None:
                n_meets_threshold = int(np.sum(samples_array >= request.goal_threshold))
                probability_of_goal = n_meets_threshold / len(samples)

            # Compute constraint analysis if constraints provided
            constraint_analysis_result: Optional[ConstraintAnalysis] = None
            if request.goal_constraints and constraint_node_values:
                analysis_dict = self._compute_constraint_analysis(
                    constraint_node_values,
                    request.goal_constraints,
                    option.id,
                )
                if analysis_dict:
                    # Convert dict to ConstraintAnalysis model
                    constraint_results = [
                        ConstraintResult(
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
                win_probability=wins[option.id] / request.n_samples,
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
        n_sensitivity_samples = min(100, request.n_samples // 10)
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
        n_sensitivity_samples = min(100, request.n_samples // 10)
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

    def _compute_factor_sensitivity(
        self,
        request: RobustnessRequestV2,
        baseline_outcomes: Dict[str, List[float]],
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
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

        # Build set of intervention factor IDs for INTERVENTION_OVERRIDE detection.
        # D-U ruling (union-across-options): a factor ANY option intervenes on is a
        # lever — not just the reference (first) option's targets. Previously this
        # set was built from options[0] only, so a factor pinned by a non-first
        # option was published with a non-lever zero_reason while union-side
        # consumers (CEE, PLoT coaching) suppressed it as a lever.
        # The set is used for membership tests only, so ordering cannot affect
        # determinism.
        intervention_factor_ids = {
            factor_id for option in request.options for factor_id in (option.interventions or {})
        }

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

            # Get observed value
            observed_value = None
            if node.observed_state and node.observed_state.value is not None:
                observed_value = node.observed_state.value

            mean_value = observed_value if observed_value is not None else 0.0

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
        influence_scores = self._compute_structural_influence(
            request.graph, factor_node_ids, request.goal_node_id
        )

        # Add influence scores to sensitivities
        for s in sensitivities:
            s["influence_score"] = influence_scores.get(str(s["node_id"]), 0.0)

        # Sort by absolute elasticity for importance_rank
        sensitivities.sort(key=lambda x: abs(float(x["elasticity"])), reverse=True)

        # Compute influence_rank (sort by influence_score descending)
        sorted_by_influence = sorted(
            sensitivities, key=lambda x: float(x["influence_score"]), reverse=True
        )
        influence_rank_map = {s["node_id"]: i + 1 for i, s in enumerate(sorted_by_influence)}

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
        for i, s in enumerate(sensitivities):
            # Update zero_reason: DISCONNECTED takes priority if factor has no causal path
            zero_reason = s.get("zero_reason")  # type: ignore[assignment]
            if abs(float(s["elasticity"])) < 1e-10 and float(s["influence_score"]) < 1e-10:
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
                    importance_rank=i + 1,
                    observed_value=s["observed_value"],
                    interpretation=s["interpretation"],
                    zero_reason=zero_reason,
                    baseline_near_zero=s.get("baseline_near_zero"),
                    influence_score=s["influence_score"],
                    influence_rank=influence_rank_map[s["node_id"]],
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
        winner_per_sample: List[str],
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
        winner_per_sample: List[str],
        option_labels: Dict[str, str],
        option_means: Dict[str, float],
    ) -> BucketResult:
        """Compute win probabilities within a bucket of MC samples."""
        indices = np.where(mask)[0]
        bucket_size = len(indices)

        # Count wins per option in this bucket
        win_counts: Dict[str, int] = {}
        for idx in indices:
            winner = winner_per_sample[idx]
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

                observed_value = None
                if node.observed_state and node.observed_state.value is not None:
                    observed_value = node.observed_state.value
                mean_value = observed_value if observed_value is not None else 0.0

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
    ) -> Dict[str, float]:
        """
        Compute structural influence score for each factor based on causal path strengths.

        Algorithm:
        1. For each factor, find all paths to goal_node_id
        2. For each path, compute path_strength = product of edge.strength.mean * exists_probability
        3. Factor influence = sum of absolute path strengths (multiple paths add)
        4. Normalize to 0-1 scale across all factors

        Args:
            graph: Causal graph with edges
            factor_node_ids: List of factor node IDs to compute influence for
            goal_node_id: Target goal node ID

        Returns:
            Dict mapping node_id -> influence_score (0-1, normalized)
        """
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

        def find_all_paths_strengths(
            start: str,
            end: str,
            visited: set,
        ) -> List[float]:
            """
            Find all paths from start to end and return list of path strengths.
            Each path strength is the product of edge strengths along the path.
            """
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

        # Compute raw influence for each factor
        raw_influences: Dict[str, float] = {}
        for node_id in factor_node_ids:
            path_strengths = find_all_paths_strengths(node_id, goal_node_id, set())
            # Sum of absolute path strengths (multiple paths add)
            raw_influences[node_id] = sum(abs(s) for s in path_strengths)

        # Normalize to 0-1 scale
        max_influence = max(raw_influences.values()) if raw_influences else 0.0
        if max_influence < 1e-10:
            # All factors have zero influence
            return {node_id: 0.0 for node_id in factor_node_ids}

        return {node_id: raw_influences[node_id] / max_influence for node_id in factor_node_ids}

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
        winner_per_sample: List[str],
        sensitivity: List[SensitivityResult],
        request: RobustnessRequestV2,
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]],
        evaluator: SCMEvaluatorV2,
        global_seed: int,
        n_defaulted_roots: int = 0,
        defaulted_root_node_ids: Optional[List[str]] = None,
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
        )

        # Overall robustness
        # Per Decision Model Schema v2.6: is_robust = recommendation_stability >= 0.7
        # fragile_edges is a separate indicator of edge-level sensitivity
        is_robust = recommendation_stability >= self.ROBUST_THRESHOLD

        # Confidence based on sample size and stability
        confidence = min(0.99, recommendation_stability * (1 - 1 / np.sqrt(n_samples)))

        # Interpretation
        if is_robust:
            if fragile_edges:
                interpretation = (
                    f"Recommendation is ROBUST with {confidence:.0%} confidence. "
                    f"{most_frequent_winner} wins in {recommendation_stability:.0%} of scenarios. "
                    f"({len(fragile_edges)} sensitive edge{'s' if len(fragile_edges) > 1 else ''} identified)"
                )
            else:
                interpretation = (
                    f"Recommendation is ROBUST with {confidence:.0%} confidence. "
                    f"{most_frequent_winner} wins in {recommendation_stability:.0%} of scenarios."
                )
        elif recommendation_stability >= 0.5:
            interpretation = (
                f"Recommendation is MODERATELY ROBUST. "
                f"{most_frequent_winner} wins in {recommendation_stability:.0%} of scenarios, "
                f"but is sensitive to: {', '.join(fragile_edges[:3])}"
            )
        else:
            interpretation = (
                f"Recommendation is FRAGILE. No clear winner - "
                f"best option wins in only {recommendation_stability:.0%} of scenarios. "
                f"High sensitivity to: {', '.join(fragile_edges[:3])}"
            )

        return RobustnessResult(
            is_robust=is_robust,
            confidence=confidence,
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

        # Child seeds: SHA-256-derived (process-safe, NOT Python hash()) —
        # the same sub-seed pattern as the per-edge marginal-switch and
        # per-factor EVPI streams.
        child_seeds = [
            int(hashlib.sha256(f"{master_seed}:flip_stability:{i}".encode()).hexdigest()[:8], 16)
            for i in range(n_seeds)
        ]

        # One sampled background per child seed, shared across all edges.
        backgrounds: List[Dict[Tuple[str, str], float]] = []
        for child_seed in child_seeds:
            sweep_sampler = DualUncertaintySampler(request.graph.edges, SeededRNG(child_seed))
            backgrounds.append(sweep_sampler.sample_edge_configuration())

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

    def _compute_factor_evppi(
        self,
        request: RobustnessRequestV2,
        pre_noise_option_outcomes: Dict[str, List[float]],
        factor_values_per_sample: List[Dict[str, float]],
        seed: int,
        decision_evpi_bound: Optional[float],
        correlation_active: bool,
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

        Returns a list of per-factor dicts (sorted by evppi descending), or None if
        no non-lever uncertain factor exists.
        """
        if not request.parameter_uncertainties:
            return None

        # D-U lever identity: UNION of intervention targets across ALL options
        # (reuse _compute_factor_sensitivity's exact derivation — derive, don't
        # mirror). A factor in this set is a lever; its "uncertainty" is a choice.
        intervention_factor_ids = {
            factor_id
            for option in request.options
            for factor_id in (option.interventions or {})
        }

        # Deduplicate uncertainties by node_id (parse-time validation already
        # rejects duplicates; defensive, first-seen order for determinism).
        unique_uncertainties = list(
            {u.node_id: u for u in request.parameter_uncertainties}.values()
        )

        n_samples = len(factor_values_per_sample)
        results: List[Dict[str, Any]] = []
        for uncertainty in unique_uncertainties:
            fid = uncertainty.node_id
            # LEVER SUPPRESSION: omit option-controlled levers (missing ≠ zero).
            if fid in intervention_factor_ids:
                continue

            # Extract this factor's per-sample values from the retained joint
            # population. Every uncertainty factor is present in every sample dict
            # (FactorSampler always writes it), so a missing key is a defect → omit
            # that factor with no fabricated value.
            try:
                theta = [factor_values_per_sample[s][fid] for s in range(n_samples)]
            except (KeyError, IndexError):
                continue

            # Deterministic per-factor seed for the permutation-null floor (mirrors
            # the _compute_evpi per-factor seeding pattern).
            floor_seed = int(
                hashlib.sha256(f"{seed}:evppi:{fid}".encode()).hexdigest()[:8], 16
            )
            est = factor_evppi_estimate(theta, pre_noise_option_outcomes, seed=floor_seed)

            # Howard non-negativity clamp.
            clamped_low = est.evppi_raw < 0.0
            evppi = max(0.0, est.evppi_raw)

            # Per-factor ≤ whole-decision EVPI theorem: cap at decision_evpi.
            clamped_high = False
            if decision_evpi_bound is not None and evppi > decision_evpi_bound:
                clamped_high = True
                evppi = decision_evpi_bound

            below_resolution = evppi <= est.noise_floor

            results.append(
                {
                    "factor_id": fid,
                    "evppi": round(evppi, 6),
                    # Pre-clamp raw estimate + audit components (mirrors
                    # p_win_sensitivity's current_metric/perfect_metric auditability).
                    "evppi_raw": round(est.evppi_raw, 6),
                    "baseline_max_expected_utility": round(
                        est.baseline_max_expected_utility, 6
                    ),
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

        if not results:
            return None

        # Sort by EVPPI descending (most valuable information first).
        results.sort(key=lambda x: float(x["evppi"]), reverse=True)
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
        unique_uncertainties = list(
            {u.node_id: u for u in request.parameter_uncertainties}.values()
        )

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

            # Deterministic seed per factor
            factor_seed_str = f"{seed}:evpi:{uncertainty.node_id}"
            factor_seed = int(hashlib.sha256(factor_seed_str.encode()).hexdigest()[:8], 16)

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
    ) -> Optional[float]:
        """Compute the EVPI metric for a fixed decision policy over n_samples.

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
            # P(joint_goal) for the fixed recommended option
            _, joint_prob, _ = self._compute_constraint_probabilities(
                constraint_node_values,
                request.goal_constraints,
                recommended_option_id,
            )
            return joint_prob
        else:
            # P(win) of the fixed recommended option.
            # Tie-breaking mirrors main MC: equal credit split among tied options
            # to avoid insertion-order bias (see _run_monte_carlo tie logic).
            win_count = 0.0
            for i in range(n_samples):
                max_outcome = max(option_outcomes[oid][i] for oid in option_outcomes)
                winners = [oid for oid in option_outcomes if option_outcomes[oid][i] == max_outcome]
                if recommended_option_id in winners:
                    win_count += 1.0 / len(winners)
            return win_count / n_samples

    def _compute_alternative_winners(
        self,
        fragile_edge_info: Dict[str, Tuple[str, str]],
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]],
        winner_per_sample: List[str],
        overall_winner: str,
        request: Optional[RobustnessRequestV2] = None,
        evaluator: Optional[SCMEvaluatorV2] = None,
        global_seed: Optional[int] = None,
    ) -> List[FragileEdgeEnhanced]:
        """
        Compute alternative winners for fragile edges.

        For each fragile edge, identifies which option wins most often when
        the edge is "weak" (bottom 25% of sampled strengths). Also computes
        marginal switch probability (isolated edge contribution) when
        request, evaluator, and global_seed are provided.

        Args:
            fragile_edge_info: Map of edge_id -> (from_id, to_id)
            edge_configs_per_sample: Edge strengths for each MC sample
            winner_per_sample: Winner option ID for each MC sample
            overall_winner: The overall recommended option
            request: Full robustness request with graph and options (optional)
            evaluator: SCM evaluator instance (optional)
            global_seed: Request-level seed for reproducibility (optional)

        Returns:
            List of FragileEdgeEnhanced objects with enhanced fragile edge information
        """
        # Check if marginal calculation is possible
        can_compute_marginal = (
            request is not None and evaluator is not None and global_seed is not None
        )

        results = []

        for edge_id, (from_id, to_id) in fragile_edge_info.items():
            edge_key = (from_id, to_id)

            # Compute marginal switch probability (isolated edge contribution)
            # Only computed when all required parameters are provided
            # Note: marginal computes its own baseline winner under expected-value config
            marginal_prob: Optional[float] = None
            if can_compute_marginal:
                assert request is not None
                assert evaluator is not None
                assert global_seed is not None
                marginal_prob = self._compute_marginal_switch_probability(
                    edge_key=edge_key,
                    request=request,
                    evaluator=evaluator,
                    global_seed=global_seed,
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
                weak_winner_counts[winner_per_sample[idx]] += 1

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

    def _compute_marginal_switch_probability(
        self,
        edge_key: Tuple[str, str],
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        global_seed: int,
        k_samples: int = MARGINAL_K_SAMPLES,
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

        # Build baseline config: all edges at expected value (mean * exists_probability)
        # This is consistent with how existence is a sampling gate in the rest of the system
        baseline_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
        }

        # Compute baseline winner under this config (not overall_winner from MC)
        # This ensures we compare against the correct reference point
        baseline_outcomes = {}
        for option in request.options:
            baseline_outcomes[option.id] = evaluator.evaluate(
                edge_strengths=baseline_config,
                interventions=option.interventions,
                goal_node=request.goal_node_id,
            )
        # Deterministic tie-breaking for baseline winner
        sorted_baseline = sorted(baseline_outcomes.items(), key=lambda x: (-x[1], x[0]))
        marginal_baseline_winner = sorted_baseline[0][0]

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
        constraint_node_values: Dict[str, Dict[str, List[float]]],
        constraints: List[GoalConstraint],
        option_id: str,
    ) -> Tuple[Dict[str, float], float, List[List[bool]]]:
        """
        Compute per-constraint and joint probabilities for an option.

        Args:
            constraint_node_values: Dict[option_id, Dict[node_id, List[sample_values]]]
            constraints: List of GoalConstraint objects
            option_id: The option to compute probabilities for

        Returns:
            Tuple of:
            - per_constraint_probs: Dict[constraint_index_str, prob_satisfied]
            - joint_probability: P(all constraints satisfied)
            - satisfaction_matrix: List[sample_idx][constraint_idx] -> bool (for conditional prob)
        """
        if not constraints:
            return {}, 1.0, []

        option_values = constraint_node_values[option_id]
        n_samples = len(next(iter(option_values.values())))

        # Build satisfaction matrix: [sample_idx][constraint_idx] -> bool
        satisfaction_matrix: List[List[bool]] = []
        for sample_idx in range(n_samples):
            sample_satisfactions = []
            for constraint in constraints:
                value = option_values[constraint.node_id][sample_idx]
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
        constraint_node_values: Dict[str, Dict[str, List[float]]],
        constraints: List[GoalConstraint],
        option_id: str,
        satisfaction_matrix: List[List[bool]],
        near_miss_fraction_threshold: float = 0.1,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute near-miss diagnostics for each constraint.

        For each constraint, computes:
        - failure_margin_median: Median distance from threshold when constraint fails
        - near_miss_fraction: Fraction of failures within near_miss_fraction_threshold of threshold
        - binding: True if prob_satisfied ∈ [0.4, 0.6] (constraint is borderline)

        Args:
            constraint_node_values: Dict[option_id, Dict[node_id, List[sample_values]]]
            constraints: List of GoalConstraint objects
            option_id: The option to compute diagnostics for
            satisfaction_matrix: Precomputed satisfaction matrix
            near_miss_fraction_threshold: Relative threshold for "near miss" (default 10%)

        Returns:
            Dict[constraint_idx, {failure_margin_median, near_miss_fraction, binding}]
        """
        if not constraints:
            return {}

        option_values = constraint_node_values[option_id]
        n_samples = len(satisfaction_matrix)

        diagnostics: Dict[int, Dict[str, Any]] = {}

        for c_idx, constraint in enumerate(constraints):
            values = option_values[constraint.node_id]
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
    ) -> Optional[Dict[str, Any]]:
        """
        Compute full constraint analysis for an option.

        Args:
            constraint_node_values: Dict[option_id, Dict[node_id, List[sample_values]]]
            constraints: List of GoalConstraint objects
            option_id: The option to compute analysis for

        Returns:
            Dict with constraint analysis results, or None if no constraints
        """
        if not constraints or not constraint_node_values:
            return None

        # T3: Per-constraint and joint probability
        (
            per_constraint_probs,
            joint_probability,
            satisfaction_matrix,
        ) = self._compute_constraint_probabilities(constraint_node_values, constraints, option_id)

        # T4: Pairwise conditional probabilities
        conditional_probs = self._compute_conditional_probabilities(
            satisfaction_matrix, constraints
        )

        # T5: Near-miss diagnostics
        near_miss_diagnostics = self._compute_near_miss_diagnostics(
            constraint_node_values, constraints, option_id, satisfaction_matrix
        )

        # Build constraint results
        constraint_results = []
        for c_idx, constraint in enumerate(constraints):
            diag = near_miss_diagnostics.get(c_idx, {})
            constraint_results.append(
                {
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
