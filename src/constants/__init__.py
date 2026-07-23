"""
Shared constants for ISL analysis.

These thresholds should be used consistently across all analysis components
and should align with PLoT's thresholds to prevent semantic divergence.
"""

# =============================================================================
# Path Validation Thresholds
# =============================================================================

# Canonical default for edge existence probability when not specified by the UI.
# Matches PLoT DEFAULT_EXISTS_PROBABILITY — keep in sync.
DEFAULT_EXISTS_PROBABILITY = 0.8

# Edge must have meaningful probability to be considered in structural path
DEFAULT_EXISTS_PROBABILITY_THRESHOLD = 1e-6

# Edge must have meaningful effect strength to be considered in effective path
DEFAULT_STRENGTH_THRESHOLD = 1e-6


# =============================================================================
# Identical Options Detection
# =============================================================================

# Tolerance for comparing intervention values (floating-point precision)
IDENTICAL_OPTIONS_VALUE_TOLERANCE = 1e-9


# =============================================================================
# Degenerate Outcomes Detection
# =============================================================================

# Relative spread threshold - if max-min is less than 1% of max, outcomes are degenerate
DEGENERATE_RELATIVE_THRESHOLD = 0.01  # 1%


# =============================================================================
# Partial Status Thresholds
# =============================================================================

# Minimum ratio of valid samples for "computed" status (vs "partial")
MIN_VALID_RATIO = 0.8  # 80%


# =============================================================================
# Numerical Stability
# =============================================================================

# Fraction of samples with issues that triggers warning
MAX_NUMERICAL_WARNINGS_RATIO = 0.1  # 10%


# =============================================================================
# Response Versioning
# =============================================================================

# Default response version (for backward compatibility)
DEFAULT_RESPONSE_VERSION = 1

# Current V2 schema version
RESPONSE_SCHEMA_VERSION_V2 = "2.0"


# =============================================================================
# Baseline Protection (P2-ISL-5)
# =============================================================================

# Minimum baseline magnitude for safe division (epsilon guard)
BASELINE_EPSILON = 1e-8


# =============================================================================
# Zero Variance Detection
# =============================================================================

# Values below this threshold are considered effectively zero variance
# (accounts for floating point precision in numerical computations)
ZERO_VARIANCE_TOLERANCE = 1e-10


# =============================================================================
# Factor Sensitivity Elasticity Calculation
# =============================================================================

# Epsilon values for stabilising elasticity computation when baseline or
# factor values are near zero (e.g., binary factors 0/1).
# These prevent division-by-zero while preserving meaningful sensitivity values.
# Note: Values tuned for typical factor scales; may need calibration.
FACTOR_SENSITIVITY_BASELINE_EPSILON = 0.01  # Min denominator for baseline_mean
FACTOR_SENSITIVITY_VALUE_EPSILON = 0.01  # Min denominator for factor mean_value

# Maximum elasticity magnitude for presentation (prevents extreme UX values)
# Elasticity beyond this is clamped to ±ELASTICITY_CLAMP_MAX
ELASTICITY_CLAMP_MAX = 100.0


# =============================================================================
# Request size caps (compute-admission — Codex F8)
# =============================================================================
#
# Single source of truth for the request-shape ceilings. These are enforced as
# pydantic ``max_length`` constraints on RobustnessRequestV2 (fail-closed at
# parse time) AND advertised on ``/health.compute_admission.caps`` so PLoT can
# plan against them without hand-mirroring (derive, don't mirror — see the
# programme's memory-trap #12). Because the model field and the /health
# advertisement both import THESE constants, the advertised cap can never drift
# from the enforced cap.
#
# Values match the pre-F8 literals that were inline on the model fields; F8 adds
# MAX_PARAMETER_UNCERTAINTIES (previously unbounded — the duplicate free-ride).
MAX_GRAPH_NODES = 50
MAX_GRAPH_EDGES = 200
MAX_OPTIONS = 10
MAX_PARAMETER_UNCERTAINTIES = 50
# B3-S1 correlated factors: pairwise correlation list cap. Bounded by the number
# of distinct unordered pairs over the factor-uncertainty cap — a request cannot
# express more without duplicates, which the validator rejects anyway. DERIVED from
# MAX_PARAMETER_UNCERTAINTIES (not a hand-computed literal) so the cap tracks its own
# stated derivation if MAX_PARAMETER_UNCERTAINTIES ever changes (CLAUDE.md #12).
MAX_FACTOR_CORRELATIONS = MAX_PARAMETER_UNCERTAINTIES * (MAX_PARAMETER_UNCERTAINTIES - 1) // 2

# A3 S4 (D-23.8) value-of-control (EVPC): request-side `control_candidates` caps.
# A control candidate is a factor the user could pull to a chosen value; EVPC grids
# do(factor=value) over the candidate's value list on the retained joint CRN draws.
# Bounded per D-23.8 scoping Q4 rationale: EVPC re-evaluates all retained samples
# for every (candidate, value) pair, so cost scales with
# n_samples * sum(len(values)); the caps keep the added latency comparable to the
# other budgeted enhancement phases while covering realistic lever grids. Both are
# imported by the request model so the enforced cap can never drift from the field.
MAX_CONTROL_CANDIDATES = 5
MAX_CONTROL_VALUES = 7

# EVPC wire method tag (A3 S4, D-23.8). SINGLE SOURCE OF TRUTH: the analyzer stamps
# it onto each factor_evpc entry and the ISLResponseV2 value-integrity validator
# asserts against it, so a neutral leaf module (both import here) prevents the tag
# from drifting between producer and validator (derive, don't mirror).
GRID_DO_EVPC_METHOD = "grid_do_v1"

# Node kinds that must NOT participate in inference. `filter_inference_graph`
# (src.services.robustness_analyzer_v2) drops these nodes and their incident edges
# before the evaluator is built, so a request field that names one of them (e.g. a
# control_candidate) references a node that will never reach the compute. SINGLE
# SOURCE OF TRUTH: the analyzer's filter AND the request-model validators
# (control_candidates post-filter check, D-23.14) both read THIS, so the post-filter
# node set the validator checks can never drift from the set the filter actually
# removes (derive, don't mirror — CLAUDE.md #12). Compared case-insensitively.
NON_INFERENCE_KINDS = frozenset({"decision", "option", "constraint"})
