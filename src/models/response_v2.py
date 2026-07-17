"""
V2 Response Schemas for ISL Enhanced Response Format.

Provides explicit status fields, structured critiques, and diagnostics
for improved integration with PLoT and UI components.

P2 Brief Alignment:
- Adds `version` as alias for `response_schema_version`
- Adds `timestamp` in ISO 8601 format
- Adds `seed_used` for determinism (PLoT owns response_hash)
- 422 responses use unwrapped ISLV2Error422 format
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field

from src.constants import RESPONSE_SCHEMA_VERSION_V2


class InferenceWarning(BaseModel):
    """
    Structured warning emitted during inference.

    Contract: inference_warnings is always present in the response as a list
    ([] when empty, never absent). This mirrors the PLoT sentinel pattern
    (repairs_applied: [] when empty, never absent).

    Field path convention: Use edges[{from}→{to}].field.subfield with node IDs,
    not array indices. Array indices are fragile after reordering.
    """

    code: str = Field(
        ..., description="Machine-readable warning code, e.g. 'STRENGTH_MEAN_CLAMPED'"
    )
    field: str = Field(
        ...,
        description="Stable field path, e.g. 'edges[revenue_growth→market_share].strength.mean'",
    )
    detail: Dict[str, Any] = Field(
        ..., description="Arbitrary context, e.g. {'original': 1000, 'clamped': 1.0}"
    )


class ZeroSensitivityReason(str, Enum):
    """
    Explains why a sensitivity score is zero (debug-only field).

    Used to distinguish between:
    - Legitimate zero sensitivity (factor truly has no impact)
    - Computational artifacts (near-zero values, intervention overrides)
    """

    ZERO_OUTCOME_DIFF = "zero_outcome_diff"  # Perturbation doesn't affect outcome
    ZERO_DELTA = "zero_delta"  # std/delta too small to perturb
    INTERVENTION_OVERRIDE = "intervention_override"  # Intervention dominates factor
    DISCONNECTED = "disconnected"  # No causal path to goal
    BASELINE_NORMALISED = "baseline_normalised"  # Epsilon denom applied, still zero
    POINT_MASS = "point_mass"  # Distribution has no uncertainty


# =============================================================================
# Request Echo (for debugging, no sensitive data)
# =============================================================================


class RequestEchoV2(BaseModel):
    """Echo of request parameters for debugging (no sensitive data)."""

    graph_node_count: int = Field(..., description="Number of nodes in graph")
    graph_edge_count: int = Field(..., description="Number of edges in graph")
    options_count: int = Field(..., description="Number of options provided")
    goal_node_id_hash: str = Field(..., description="SHA-256 hash of goal node ID (truncated)")
    n_samples: int = Field(..., description="Number of samples requested")
    response_version_requested: int = Field(..., description="Response version requested")
    include_diagnostics: bool = Field(..., description="Whether diagnostics were requested")

    # CIL 0.2: consistent extra='ignore' across all response models
    model_config = {"extra": "ignore"}


# =============================================================================
# Critique (structured error/warning information)
# =============================================================================


class CritiqueV2(BaseModel):
    """Structured critique for UI display."""

    id: str = Field(..., description="Unique identifier for this critique")
    code: str = Field(..., description="Machine-readable code, e.g., 'NO_PATH_TO_GOAL'")
    severity: Literal["info", "warning", "error", "blocker"] = Field(
        ..., description="Severity level"
    )
    message: str = Field(..., description="Human-readable message (sanitised)")
    source: Literal["validation", "analysis", "engine"] = Field(
        ..., description="Source of the critique (explicit, not derived)"
    )
    affected_option_ids: Optional[List[str]] = Field(
        None, description="Option IDs affected by this critique"
    )
    affected_node_ids: Optional[List[str]] = Field(
        None, description="Node IDs affected by this critique"
    )
    suggestion: Optional[str] = Field(
        None, description="Actionable suggestion to resolve the issue"
    )

    # CIL 0.2: consistent extra='ignore' across all response models
    model_config = {"extra": "ignore"}


# =============================================================================
# Diagnostics (optional detailed information)
# =============================================================================


class OptionDiagnosticV2(BaseModel):
    """Diagnostic information for a single option."""

    option_id: str = Field(..., description="Option identifier")
    intervention_count: int = Field(..., description="Number of interventions")
    has_structural_path: bool = Field(
        ...,
        description="Path exists with exists_probability >= threshold",
    )
    has_effective_path: bool = Field(
        ...,
        description="Structural path AND abs(strength.mean) >= threshold",
    )
    targets_with_effective_path_count: int = Field(
        ..., description="Number of intervention targets with effective path"
    )
    targets_without_effective_path_count: int = Field(
        ..., description="Number of intervention targets without effective path"
    )
    warnings: List[str] = Field(default_factory=list, description="Option-specific warnings")

    # CIL 0.2: consistent extra='ignore' across all response models
    model_config = {"extra": "ignore"}


class DiagnosticsV2(BaseModel):
    """Diagnostic information (only included when requested)."""

    goal_node_id_hash: str = Field(..., description="Hashed goal node ID")
    goal_node_found: bool = Field(..., description="Whether goal node exists in graph")
    option_diagnostics: List[OptionDiagnosticV2] = Field(
        default_factory=list, description="Per-option diagnostics"
    )
    n_samples_requested: int = Field(..., description="Samples requested")
    n_samples_completed: int = Field(..., description="Samples completed")
    identifiability_status: Literal["identifiable", "not_identifiable", "unknown"] = Field(
        ..., description="Causal identifiability status"
    )
    identifiability_reason: Optional[str] = Field(
        None, description="Reason for identifiability status"
    )
    path_exists_probability_threshold: float = Field(
        ..., description="Threshold used for exists_probability"
    )
    path_strength_threshold: float = Field(..., description="Threshold used for strength.mean")

    # CIL 0.2: consistent extra='ignore' across all response models
    model_config = {"extra": "ignore"}


# =============================================================================
# Outcome Distribution
# =============================================================================


class OutcomeDistributionV2(BaseModel):
    """Outcome distribution with core percentiles."""

    mean: float = Field(..., description="Mean outcome value")
    std: float = Field(..., description="Standard deviation")
    # CIL 0.2: Optional — null when true percentiles cannot be computed
    # (no samples or all non-finite). See code review C3.
    p10: Optional[float] = Field(None, description="10th percentile (null when unavailable)")
    p50: Optional[float] = Field(
        None, description="50th percentile / median (null when unavailable)"
    )
    p90: Optional[float] = Field(None, description="90th percentile (null when unavailable)")
    n_samples: int = Field(..., description="Total samples")
    n_valid_samples: int = Field(..., description="Samples without NaN/Inf")
    validity_ratio: float = Field(..., description="n_valid_samples / n_samples")
    # CIL 0.2: provenance marker for percentile values
    percentiles_source: Literal["samples", "unavailable"] = Field(
        default="samples",
        description="'samples' when p10/p50/p90 computed from actual MC samples; "
        "'unavailable' when no valid samples exist (p10/p50/p90 will be null)",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


# =============================================================================
# Option Result
# =============================================================================


class OptionResultV2(BaseModel):
    """Analysis result for a single option."""

    id: str = Field(..., description="Option identifier")
    label: Optional[str] = Field(None, description="Human-readable label")
    outcome: OutcomeDistributionV2 = Field(..., description="Outcome distribution")
    win_probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="P(this option is best) - fraction of samples where this option had highest outcome",
    )
    probability_of_goal: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="P(outcome >= goal_threshold). Only present when goal_threshold is provided in request.",
    )
    constraint_analysis: Optional["ConstraintAnalysisV2"] = Field(
        None,
        description="Multi-constraint analysis results. Only present when goal_constraints is provided in request.",
    )
    status: Literal["computed", "partial", "failed"] = Field(
        ..., description="Option-specific status"
    )
    status_reason: Optional[str] = Field(None, description="Reason for non-computed status")

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


# =============================================================================
# Sensitive Factor
# =============================================================================


class SensitiveFactorV2(BaseModel):
    """Factor sensitivity information."""

    node_id: str = Field(..., description="Factor node ID")
    sensitivity_score: float = Field(..., description="Sensitivity score")
    effect_on_ranking: Literal["none", "minor", "moderate", "major"] = Field(
        ..., description="Effect on option ranking"
    )

    # CIL 0.2: consistent extra='ignore' across all response models
    model_config = {"extra": "ignore"}


# =============================================================================
# Constraint Analysis (Multi-Constraint Goal Analysis)
# =============================================================================


class ConstraintResultV2(BaseModel):
    """Result for a single goal constraint."""

    node_id: str = Field(..., description="Node ID the constraint applies to")
    operator: Literal[">=", "<="] = Field(..., description="Comparison operator")
    threshold: float = Field(..., description="Threshold value")
    label: Optional[str] = Field(None, description="Human-readable label for coaching")
    prob_satisfied: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability that this constraint is satisfied (count / n_samples)",
    )
    failure_margin_median: Optional[float] = Field(
        None,
        description="Median distance from threshold when constraint fails (positive = failing by this amount)",
    )
    near_miss_fraction: Optional[float] = Field(
        None, ge=0, le=1, description="Fraction of failures within 10% of threshold (near-misses)"
    )
    binding: Optional[bool] = Field(
        None, description="True if constraint is borderline (prob_satisfied ∈ [0.4, 0.6])"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def value(self) -> float:
        """Contract-aligned alias for threshold (v2.7 input field name).

        Mirrors `threshold` so the JSON output includes both field names,
        letting PLoT consume either without translation.
        """
        return self.threshold

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


class ConstraintAnalysisV2(BaseModel):
    """Multi-constraint analysis results for an option.

    Note: Constraint probabilities are computed from raw Monte Carlo samples
    (before auto-scaled noise is applied to outcome nodes). This may cause
    slight differences compared to probability_of_goal which uses noised samples.
    """

    constraints: List[ConstraintResultV2] = Field(
        ..., description="Per-constraint probability results"
    )
    joint_probability: float = Field(
        ..., ge=0, le=1, description="P(all constraints satisfied simultaneously)"
    )
    conditional_probabilities: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Pairwise conditional probabilities: P(C_j | C_i). "
        "Format: {constraint_i_idx: {constraint_j_idx: P(C_j | C_i)}}. "
        "When P(C_i)=0, entries for that constraint are omitted (undefined). "
        "Indices correspond to order in goal_constraints array.",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


# =============================================================================
# Fragile Edge (V2 enhanced format)
# =============================================================================


class FragileEdgeV2(BaseModel):
    """Fragile edge with alternative winner analysis.

    Identifies edges where the recommendation is sensitive to assumption changes
    and what option would win if the edge is weaker than modelled.
    """

    edge_id: str = Field(..., description="Edge identifier in 'from->to' format")
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Target node ID")
    alternative_winner_id: Optional[str] = Field(
        None,
        description="Option that wins most often when this edge is weak (bottom quartile). "
        "Null if same option wins regardless of edge strength.",
    )
    switch_probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Proportion of MC samples where alternative wins when edge is weak. "
        "0.0 if same option wins (stable), null only if no data available.",
    )
    marginal_switch_probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Probability of decision flip when ONLY this edge varies "
        "(all other edges held at baseline). Isolates individual edge contribution.",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


class FlipStabilityBandV2(BaseModel):
    """Seed-sweep stability band for one edge's flip threshold (Track S Phase 1).

    DEFAULT-ON and ADDITIVE (env gating removed 2026-07-17): present on every
    edge_e_values entry whenever e-values are computed. Absent only when the
    all-or-nothing band budget (FLIP_STABILITY_BUDGET_MS) trips — then NO
    entry carries a band and the degradation is disclosed via the
    flip_stability_budget_exceeded structured log event.

    The single-point flip_mean is searched against ONE background (all other
    edges at expected value) and is therefore presented with false stability.
    This band shows how the flip point moves across N backgrounds sampled
    from the graph's own joint uncertainty — child seeds are SHA-256-derived
    from the request seed, so the band is deterministic per request+seed.
    Per the 2026-06-10 science-performance report, flip confidence should be
    based on band_width, not on the single-point value alone.

    MEMBERSHIP: the base flip_mean is NOT a member of the sweep (it is
    searched against the expected-value background, which is never one of
    the sampled backgrounds), so flip_mean MAY legitimately lie outside
    [band_min, band_max] — observed live on low-flip-participation edges.
    Consumers must NOT assume flip_mean ∈ band (bytes-checked 2026-07-17;
    producer-side detail on RobustnessAnalyzerV2._attach_flip_stability_bands).
    """

    n_seeds: int = Field(
        ..., ge=1, description="Number of child seeds swept (constant 10, FLIP_STABILITY_N_SEEDS)"
    )
    n_seeds_flipped: int = Field(
        ...,
        ge=0,
        description="Seeds whose sampled background admits a flip within [-1, 1]. "
        "When 0, the band_* fields are omitted (exclude_none).",
    )
    band_min: Optional[float] = Field(
        None, description="Minimum flip mean across flipped seeds. Omitted when nothing flips."
    )
    band_median: Optional[float] = Field(
        None, description="Median flip mean across flipped seeds. Omitted when nothing flips."
    )
    band_max: Optional[float] = Field(
        None, description="Maximum flip mean across flipped seeds. Omitted when nothing flips."
    )
    band_width: Optional[float] = Field(
        None,
        description="band_max - band_min. The flip-confidence input recommended by the "
        "06-10 science-performance report. Omitted when nothing flips. "
        "INTERPRETATION TRAP: when n_seeds_flipped == 1 this is 0.0 by construction "
        "(a single value has zero range) — a naive width rubric would read maximal "
        "stability from a single flipped background. Consumers MUST condition any "
        "width-based confidence rubric on n_seeds_flipped.",
    )
    seed_flip_means: List[Optional[float]] = Field(
        ...,
        description="Per-child-seed flip mean, in child-seed order; null where that "
        "seed's background admits no flip.",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    model_config = {"extra": "ignore"}


class EdgeEValueV2(BaseModel):
    """E-value analogue for an edge: how wrong must the strength be to flip the recommendation?

    Analogous to VanderWeele's E-value in observational epidemiology: larger values
    indicate the assumption is robust (needs a larger violation to overturn the finding).
    """

    edge_id: str = Field(..., description="Edge identifier in 'from->to' format")
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Target node ID")
    e_value: Optional[float] = Field(
        None,
        ge=1.0,
        description="Ratio of flip_mean to current_mean (>= 1.0). "
        "Large = robust assumption. 1.0 = already at the flip boundary. "
        "Null when edge is unflippable (see is_unflippable).",
    )
    is_unflippable: bool = Field(
        default=False,
        description="True when no perturbation within [-1, 1] can flip the recommendation. "
        "When true, e_value is null.",
    )
    flip_direction: Literal["increase", "decrease"] = Field(
        ..., description="Direction strength must move to flip the recommendation"
    )
    current_mean: float = Field(..., description="Current edge strength mean")
    flip_mean: float = Field(..., description="Minimum strength mean that flips the recommendation")
    stability: Optional[FlipStabilityBandV2] = Field(
        None,
        description="Seed-sweep stability band for this flip threshold (Track S Phase 1). "
        "Default-on, additive. Absent only when the all-or-nothing band budget "
        "(FLIP_STABILITY_BUDGET_MS) trips.",
    )

    model_config = {"extra": "ignore"}


class EdgeSensitivityV2(BaseModel):
    """Edge-level sensitivity result on the V2 wire (T1-6 wire completeness).

    V2 mirror of the internal V1 SensitivityResult (same content, V2 naming
    style: edge_id/from_id/to_id like FragileEdgeV2/EdgeEValueV2). These
    forced-existence and magnitude contrasts were always computed by the V2
    analyzer but were previously dropped from the V2 envelope (consumers saw
    EDGE_SENSITIVITY_UNAVAILABLE_V2_WIRE). Additive optional field.

    Computed against the reference option disclosed in the envelope's
    sensitivity_reference_option_id field (currently options[0]).
    """

    edge_id: str = Field(..., description="Edge identifier in 'from->to' format")
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Target node ID")
    sensitivity_type: str = Field(
        ..., description="Contrast type: 'existence' (edge forced on vs off) or 'magnitude'"
    )
    sensitivity_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Normalized sensitivity score (0-1, |elasticity| relative to the "
        "max |elasticity| in this analysis; same normalization as factor sensitivity_score)",
    )
    direction: Literal["positive", "negative"] = Field(
        ..., description="Sign of the raw elasticity"
    )
    elasticity: float = Field(
        ..., description="Raw elasticity: % change in outcome per % change in parameter"
    )
    importance_rank: int = Field(
        ..., ge=1, description="Rank by |elasticity| across all edge contrasts (1 = most important)"
    )
    interpretation: str = Field(
        ...,
        description="Human-readable explanation. Wording is provisional "
        "(provisional_doctrine_v0).",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


# =============================================================================
# Robustness Result
# =============================================================================


class RobustnessResultV2(BaseModel):
    """Robustness analysis result."""

    # V2 fields
    level: Literal["high", "moderate", "low", "very_low"] = Field(
        ..., description="Robustness level"
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence [0, 1]")
    sensitive_factors: Optional[List[SensitiveFactorV2]] = Field(
        None, description="Factor sensitivity breakdown"
    )

    # V2 enhanced fragile edges with alternative winner analysis
    fragile_edges: Optional[List[FragileEdgeV2]] = Field(
        None,
        description="Edges that could flip the decision, with alternative winner analysis",
    )

    # V1 backward-compatibility fields (for PLoT integration)
    is_robust: Optional[bool] = Field(
        None, description="Whether recommendation is robust (V1 compat)"
    )
    fragile_edges_v1: Optional[List[str]] = Field(
        None,
        description="Edges that could flip the decision (V1 compat, string format)",
    )
    robust_edges: Optional[List[str]] = Field(
        None, description="Edges that don't significantly affect decision (V1 compat)"
    )
    recommendation_stability: Optional[float] = Field(
        None, ge=0, le=1, description="P(same recommendation across samples) (V1 compat)"
    )

    # E-value analogue per edge (enhancement)
    edge_e_values: Optional[List[EdgeEValueV2]] = Field(
        None,
        description="E-value analogue per edge: how wrong must the strength be "
        "to flip the recommendation. Only included when computed within budget.",
    )

    # Edge-level sensitivity (T1-6 wire completeness — additive optional)
    edge_sensitivity: Optional[List[EdgeSensitivityV2]] = Field(
        None,
        description="Edge-level sensitivity (forced-existence and magnitude contrasts) "
        "computed against the reference option disclosed in the envelope's "
        "sensitivity_reference_option_id. Mirrors the V1 sensitivity results that were "
        "previously absent from the V2 wire.",
    )

    # Trust penalty metadata (audit trail for root node default trust downgrade)
    stability_penalty_factor: Optional[float] = Field(
        None,
        description="Multiplicative penalty applied to recommendation_stability "
        "due to missing root node values. 1.0 = no penalty. "
        "Only present when root nodes defaulted to 0.0.",
    )
    defaulted_root_node_ids: Optional[List[str]] = Field(
        None,
        description="Root node IDs that defaulted to 0.0, triggering the stability penalty.",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


# =============================================================================
# Factor Sensitivity
# =============================================================================


class FactorSensitivityV2(BaseModel):
    """Factor sensitivity for drivers analysis."""

    node_id: str = Field(..., description="Factor node ID")
    label: Optional[str] = Field(None, description="Human-readable node label")
    sensitivity_score: float = Field(
        ..., ge=0, le=1, description="Normalized sensitivity score (0-1, higher = more sensitive)"
    )
    importance_score: Optional[float] = Field(
        None, ge=0, le=1, description="Relative importance (0-1, higher = more important)"
    )
    elasticity: Optional[float] = Field(
        None, description="Raw elasticity: % change in outcome per % change in factor"
    )
    elasticity_display: Optional[float] = Field(
        None, description="UI-safe elasticity clamped to [-100, 100]"
    )
    direction: Literal["positive", "negative"] = Field(..., description="Direction of effect")
    confidence: Optional[float] = Field(
        None, ge=0, le=1, description="Confidence level (omitted when not computed)"
    )
    confidence_source: Optional[Literal["bootstrap_sampling", "graph_structural"]] = Field(
        None,
        description="Source of confidence value: 'bootstrap_sampling' (from MC resampling) "
        "or 'graph_structural' (fallback from graph path analysis when bootstrap unavailable)",
    )
    importance_rank: Optional[int] = Field(
        None, ge=1, description="Rank by importance (1 = most important)"
    )
    observed_value: Optional[float] = Field(
        None, description="Factor's current value used in calculation"
    )
    interpretation: Optional[str] = Field(
        None, description="Human-readable explanation of sensitivity"
    )
    # Provenance echo (Track S — value-origin passthrough for DecisionConfidencePanel).
    # Echo-only: ISL surfaces where the factor's value came from but does NOT consume
    # these in inference. Named value_* to avoid confusion with confidence_source
    # (which describes how the confidence figure was computed, not value origin).
    value_source: Optional[str] = Field(
        None,
        description="Echo of the factor node's observed_state.source — where the value "
        "came from (e.g. 'brief_extraction', 'user_input', 'computed'). Omitted when absent.",
    )
    value_extraction_type: Optional[str] = Field(
        None,
        description="Echo of the factor node's observed_state.extractionType "
        "(e.g. 'explicit', 'inferred'), when supplied by CEE. Omitted when absent.",
    )
    value_defaulted: Optional[bool] = Field(
        None,
        description="True when the factor's value was defaulted (no observed value was "
        "provided, so it fell back to 0.0). Omitted when an observed value was supplied. "
        "Derived from the same observed-value check as the ROOT_NODE_DEFAULT_VALUE warning.",
    )
    # Debug fields (always serialised for debugging)
    zero_reason: Optional[ZeroSensitivityReason] = Field(
        None, description="Debug: explains why sensitivity is zero (when elasticity ≈ 0)"
    )
    baseline_near_zero: Optional[bool] = Field(
        None, description="Debug: True if epsilon denominator was applied"
    )
    # Structural influence fields
    influence_score: Optional[float] = Field(
        None, ge=0, le=1, description="Structural influence from causal path strengths (0-1)"
    )
    influence_rank: Optional[int] = Field(
        None, ge=1, description="Rank by influence_score (1 = highest)"
    )
    # Bootstrap uncertainty fields (3C — factor sensitivity confidence)
    elasticity_std: Optional[float] = Field(
        None, ge=0, description="Std dev of elasticity across bootstrap runs"
    )
    attribution_stability: Optional[Literal["high", "moderate", "low", "negligible"]] = Field(
        None, description="Categorical stability: 'high', 'moderate', 'low', or 'negligible'"
    )
    rank_flip_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Fraction of bootstrap runs where rank shifts by >= 2 positions",
    )
    stability_method: Optional[str] = Field(
        None, description="Method used: 'bootstrap_20' or 'bootstrap_10'"
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


# =============================================================================
# Stability Thresholds (3C)
# =============================================================================


class StabilityThresholdsResponse(BaseModel):
    """Threshold metadata for attribution_stability classification.

    Provisional — pending scientific review. NOT included in response_hash.
    """

    high_moderate_boundary: float = Field(
        ..., description="CV boundary: CV ≤ this → 'high' stability"
    )
    moderate_low_boundary: float = Field(
        ..., description="CV boundary: CV ≤ this → 'moderate'; above → 'low'"
    )
    version: str = Field(..., description="Threshold configuration version")
    provisional: bool = Field(
        ..., description="True indicates thresholds are operational defaults pending review"
    )


class BucketResultV2(BaseModel):
    """Win probability results for one side of a factor median split (V2 response)."""

    n_samples: int = Field(..., ge=0)
    winner_id: str = Field(...)
    winner_label: str = Field(...)
    winner_probability: float = Field(..., ge=0, le=1)
    runner_up_id: Optional[str] = Field(None)
    runner_up_probability: Optional[float] = Field(None, ge=0, le=1)

    model_config = {"extra": "ignore"}


class ConditionalWinnerV2(BaseModel):
    """Factor-conditional win probability (V2). See ConditionalWinner for limitations."""

    factor_id: str = Field(...)
    factor_label: str = Field(...)
    split_value: float = Field(...)
    split_unit: Optional[str] = Field(None)
    low_bucket: BucketResultV2 = Field(...)
    high_bucket: BucketResultV2 = Field(...)
    winner_flips: bool = Field(...)

    model_config = {"extra": "ignore"}


# =============================================================================
# Path Decomposition (T1-6 wire completeness — V2 mirrors of the internal
# V1 models in robustness_v2.py; same schema, different modules, following the
# FragileEdgeV2 pattern. response_v2 must not import robustness_v2.)
# =============================================================================


class PathContributionV2(BaseModel):
    """One modelled pathway's signed structural contribution to the goal (V2 wire).

    A pathway is a simple directed sequence of node IDs from a retained
    intervention target to the goal. ``path_effect`` is the signed product of
    per-edge coefficients (``strength.mean * exists_probability``) along the
    pathway and is NOT scaled by the intervention magnitude. Structural
    decomposition of the modelled effect, not a real-world causal claim.
    """

    path: List[str] = Field(
        ...,
        description="Node IDs from the retained intervention target to the goal, "
        "in directed path order.",
    )
    path_effect: float = Field(
        ...,
        description="Signed product of per-edge coefficients (strength.mean * "
        "exists_probability) along this path. Structural only — not scaled by the "
        "intervention magnitude.",
    )
    total_effect: float = Field(
        ...,
        description="Signed sum of path_effect across all enumerated "
        "intervention-target-to-goal paths. Identical on every entry, for auditability.",
    )
    signed_contribution: Optional[float] = Field(
        None,
        description="path_effect / total_effect when the net modelled effect is "
        "non-negligible; omitted when indeterminate. May be negative or exceed 1 "
        "when paths oppose.",
    )
    status: Literal["computed", "indeterminate"] = Field(
        ...,
        description="'computed' when |total_effect| >= 1e-10; 'indeterminate' when the "
        "net modelled effect is near zero and a relative share is not well defined.",
    )
    mechanism: str = Field(
        ...,
        description="Human-readable modelled-pathway-contribution statement. Describes "
        "modelled structure only; not a real-world causal claim. Wording is provisional "
        "(provisional_doctrine_v0).",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


class PathDecompositionV2(BaseModel):
    """Structural pathway decomposition on the V2 wire (T1-6 wire completeness).

    V2 mirror of the internal V1 PathDecomposition. Request-gated by
    include_path_decomposition: previously computed on request but dropped from
    the V2 envelope. Additive optional field.
    """

    recommended_option_id: str = Field(
        ...,
        description="The recommended option this decomposition explains "
        "(context/metadata; not a path node).",
    )
    entry_nodes: List[str] = Field(
        ...,
        description="Retained intervention target node IDs the paths start from "
        "(intervention targets that survived inference-graph filtering).",
    )
    truncated: bool = Field(
        default=False,
        description="True when the number of simple paths exceeded the safety budget, so "
        "the top-3 pathway ranking was suppressed for performance and paths is empty. "
        "This does NOT mean the modelled effect is zero — only that individual pathways "
        "were too numerous to rank. Distinct from an empty result with truncated=False, "
        "which means no reachable path from the retained intervention targets.",
    )
    path_count: int = Field(
        default=0,
        description="Number of simple intervention-target-to-goal paths enumerated. When "
        "truncated is True this equals the budget cap and the true count is higher.",
    )
    paths: List[PathContributionV2] = Field(
        default_factory=list,
        description="Top-3 intervention-target-to-goal paths, ranked by absolute " "path_effect.",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore"}


# =============================================================================
# Main V2 Response
# =============================================================================


class ISLResponseV2(BaseModel):
    """Enhanced response with explicit status fields and diagnostics."""

    # Version information (P2-ISL-1: added `version` alias for PLoT compatibility)
    response_schema_version: str = Field(
        default=RESPONSE_SCHEMA_VERSION_V2,
        description="Response schema version",
        alias="version",
    )
    endpoint_version: str = Field(..., description="Endpoint version, e.g., 'analyze/v2'")
    engine_version: str = Field(..., description="ISL engine version")
    build: Optional[str] = Field(
        None, description="Git commit hash (7 chars) for build verification"
    )

    # P2-ISL-1: Timestamp in ISO 8601 format
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        description="Response timestamp in ISO 8601 format",
    )

    # Explicit status fields (CRITICAL for UI)
    analysis_status: Literal["computed", "partial", "failed"] = Field(
        ..., description="Overall analysis status"
    )
    robustness_status: Literal["computed", "skipped", "unavailable", "error"] = Field(
        ..., description="Robustness analysis status"
    )
    factor_sensitivity_status: Literal["computed", "skipped", "unavailable", "error"] = Field(
        ..., description="Factor sensitivity status"
    )

    # Reason for non-computed status (sanitised, no internal details)
    status_reason: Optional[str] = Field(None, description="Reason for non-computed status")

    # Structured critiques (for UI display)
    critiques: List[CritiqueV2] = Field(default_factory=list, description="Structured critiques")

    # Request echo (for debugging integration issues)
    request_echo: RequestEchoV2 = Field(..., description="Echo of request parameters")

    # Diagnostics (OPTIONAL - only when requested)
    diagnostics: Optional[DiagnosticsV2] = Field(
        None, description="Detailed diagnostics (when requested)"
    )

    # Analysis results (only if analysis_status in ["computed", "partial"])
    options: Optional[List[OptionResultV2]] = Field(None, description="Option results")
    robustness: Optional[RobustnessResultV2] = Field(None, description="Robustness assessment")
    factor_sensitivity: Optional[List[FactorSensitivityV2]] = Field(
        None, description="Factor sensitivity results"
    )
    conditional_winners: Optional[List[ConditionalWinnerV2]] = Field(
        None,
        description="Factors where the winning option flips depending on factor value range.",
    )

    # Inference warnings (e.g. STRENGTH_MEAN_CLAMPED, CONSTRAINT_NODE_DEFAULT_BASE).
    # Contract: always present as a list — [] when empty, never absent.
    # This field survives exclude_none=True because it has a non-None default.
    inference_warnings: List[InferenceWarning] = Field(
        default_factory=list,
        description="Structured warnings about inference conditions that may affect result reliability",
    )

    # Stability threshold metadata (3C-thresholds)
    stability_thresholds: Optional[StabilityThresholdsResponse] = Field(
        None,
        description="Thresholds used for attribution_stability classification. "
        "Provisional — pending scientific review. NOT included in response_hash.",
    )

    # EVPI results (enhancement — gated by include_voi flag)
    factor_evpi: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Expected Value of Perfect Information per factor: how much does "
        "removing this factor's uncertainty improve the decision metric?",
    )

    # Path decomposition (T1-6 wire completeness — additive optional, request-gated
    # by include_path_decomposition so payload size is opt-in; matches the V1
    # response's top-level placement)
    path_decomposition: Optional[PathDecompositionV2] = Field(
        None,
        description="Structural pathway decomposition for the recommended option's "
        "retained intervention targets: top-3 simple directed paths to the goal with "
        "signed structural contributions. Structural decomposition of the modelled "
        "effect, not a causal claim. Only present when include_path_decomposition "
        "was requested.",
    )

    # Reference-option disclosure (T1-5 — additive optional)
    sensitivity_reference_option_id: Optional[str] = Field(
        None,
        description="Option ID used as the reference/baseline for edge sensitivity, "
        "factor sensitivity, and the fragile-edge classification derived from them "
        "(currently the first option in the request). Disclosure only — consumers "
        "should surface that sensitivity results are relative to this option.",
    )

    # Auto-noise disclosure — mirrors V1 _metadata.auto_noise_applied so PLoT B3 can
    # surface the disclosure without reading the internal V1 metadata envelope.
    # None when the analyser cannot determine the flag (e.g. error or partial responses
    # where metadata was not built). False must serialise as false, not be dropped.
    auto_noise_applied: Optional[bool] = Field(
        None,
        description="Whether auto-scaled noise (√2 variance inflation) was applied to "
        "outcome distributions. Mirrors V1 _metadata.auto_noise_applied. "
        "None when the analyser cannot determine the flag.",
    )

    # Correlation
    request_id: str = Field(..., description="Request ID for correlation")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    # P2-ISL-1: Determinism (ISL only returns seed_used; response_hash is PLoT-owned)
    seed_used: Optional[str] = Field(
        None, description="RNG seed used for deterministic reproduction"
    )
    seed_source: Optional[Literal["client_provided", "server_computed"]] = Field(
        None,
        description="Origin of seed_used: 'client_provided' if seed was in the request, "
        "'server_computed' if ISL derived it from graph structure",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {
        "extra": "ignore",
        "populate_by_name": True,  # Allow both 'version' and 'response_schema_version'
        "json_schema_extra": {
            "example": {
                "version": "2.0",
                "endpoint_version": "analyze/v2",
                "engine_version": "1.0.0",
                "timestamp": "2025-01-15T10:30:00Z",
                "analysis_status": "computed",
                "robustness_status": "computed",
                "factor_sensitivity_status": "computed",
                "status_reason": None,
                "critiques": [],
                "request_echo": {
                    "graph_node_count": 5,
                    "graph_edge_count": 4,
                    "options_count": 2,
                    "goal_node_id_hash": "abc123def456",
                    "n_samples": 1000,
                    "response_version_requested": 2,
                    "include_diagnostics": False,
                },
                "options": [
                    {
                        "id": "option_a",
                        "label": "Option A",
                        "outcome": {
                            "mean": 50000.0,
                            "std": 5000.0,
                            "p10": 42000.0,
                            "p50": 50000.0,
                            "p90": 58000.0,
                            "n_samples": 1000,
                            "n_valid_samples": 1000,
                            "validity_ratio": 1.0,
                        },
                        "status": "computed",
                    }
                ],
                "robustness": {
                    "level": "high",
                    "confidence": 0.92,
                },
                "auto_noise_applied": False,
                "request_id": "req_abc123",
                "processing_time_ms": 150,
                "seed_used": "42",
                "seed_source": "client_provided",
            }
        },
    }


# =============================================================================
# 422 Error Response (P2-ISL-3)
# =============================================================================


class ISLV2Error422(BaseModel):
    """
    422 error response — MUST be returned unwrapped (no envelope).

    Per P2 brief: This is the exact shape PLoT/UI expect for validation failures.
    DO NOT wrap in {"error": {...}} or add success: false.
    """

    analysis_status: Literal["blocked"] = Field(
        default="blocked",
        description="Always 'blocked' for 422 responses",
    )
    status_reason: str = Field(..., description="Human-readable reason for blocking")
    critiques: List[CritiqueV2] = Field(
        ..., description="Structured critiques explaining the validation failure"
    )
    request_id: Optional[str] = Field(
        None, description="Request ID for correlation (echoed if available)"
    )

    # CIL: explicit extra='ignore' — consistent with all other V2 models.
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "analysis_status": "blocked",
                "status_reason": "Intervention targets non-existent node",
                "critiques": [
                    {
                        "id": "critique_a1b2c3d4",
                        "code": "INVALID_INTERVENTION_TARGET",
                        "severity": "blocker",
                        "source": "validation",
                        "message": "Intervention targets non-existent node: 'nonexistent_node'",
                        "suggestion": "Interventions must target nodes that exist in the graph",
                        "affected_node_ids": ["nonexistent_node"],
                    }
                ],
                "request_id": "isl-a1b2c3d4e5f6",
            }
        },
    }
