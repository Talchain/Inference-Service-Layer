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

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from src.config.stability_thresholds import GRAPH_STRUCTURAL_METHOD_VERSION
from src.constants import GRID_DO_EVPC_METHOD, RESPONSE_SCHEMA_VERSION_V2

# Range→distribution disclosure model (ROADMAP 2.720; pure Pydantic, no cycle)
from src.models.range_fit import RangeFitDisclosure


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
    # Codex F4 (producer half): warnings previously had NO severity, so PLoT
    # downstream defaulted them to 'info' and hid them. Carry a real, typed
    # severity. Default is the QUIET 'info' — the ~9 benign input-adjustment /
    # default diagnostics (STRENGTH_MEAN_CLAMPED, CONSTRAINT_NODE_DEFAULT_BASE,
    # ROOT_NODE_DEFAULT_VALUE, ...) are built directly and must STAY 'info' so
    # PLoT's severity=='warning' ? 'warning' : 'info' mapping keeps them quiet.
    # The four optional-phase degradation codes (E_VALUES_UNAVAILABLE /
    # STABILITY_BANDS_UNAVAILABLE / EVPI_UNAVAILABLE / PATH_DECOMPOSITION_UNAVAILABLE)
    # opt UP to 'warning', stamped explicitly by
    # RobustnessAnalyzerV2._optional_phase_unavailable_warning().
    # ROADMAP 2.258 adds two more 'warning'-severity codes, stamped by
    # RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame():
    # GOAL_THRESHOLD_FRAME_UNSPECIFIED and GOAL_THRESHOLD_NOT_CONVERTIBLE. They
    # are degradation disclosures of the same class — probability_of_goal was
    # WITHHELD — so they must not be filed under the quiet 'info' that PLoT hides;
    # the reason is the whole payload. (This comment is a hand-maintained mirror
    # of the code, trap 12: it is prose, not a guarantee. The enforcing test is
    # tests/unit/test_goal_threshold_frame.py::TestFailClosedWarnings, which
    # asserts the severity at the source.) ADDITIVE optional
    # field on the untyped-passthrough seam: a non-None default so it always rides
    # the wire under exclude_none, and older consumers ignore it (extra='ignore' on
    # every V2 model). Vocabulary is the CritiqueV2 subset (no 'blocker' — a
    # warning never blocks).
    severity: Literal["info", "warning", "error"] = Field(
        default="info",
        description="Severity for downstream routing/display. Defaults to the quiet "
        "'info'; only the optional-phase degradation codes opt up to 'warning'.",
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

    # 2.477(a): Optional — OMITTED (exclude_none, never a JSON null) when the
    # option has no finite sample population to summarise, i.e. every Monte
    # Carlo draw was non-finite. Reporting 0.0 there would fabricate a
    # measurement; the pre-2.477 behaviour was worse still — a non-finite float
    # in a required field made the WHOLE response unserializable and the run
    # 500'd, taking its own MONTE_CARLO_FAILED critique with it. Present for
    # every option with >=1 finite draw, which is every option in every run that
    # could serialize before. Travels with p10/p50/p90 and downside, which are
    # already absent in exactly this case.
    mean: Optional[float] = Field(
        None, description="Mean outcome value (omitted when no valid samples)"
    )
    std: Optional[float] = Field(
        None, description="Standard deviation (omitted when no valid samples)"
    )
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

    @model_validator(mode="after")
    def _summary_stats_absent_only_without_samples(self) -> "OutcomeDistributionV2":
        """2.477(a): guard the FABRICATION and the SILENT-LOSS directions of
        ``mean``/``std`` absence.

        Two rules, both fail-loud at construction so a mutation cannot pass
        silently (trap #13):

        (1) ``mean`` and ``std`` travel TOGETHER — either both present or both
            absent. A half-summarised distribution is not a thing, and a
            one-sided drop is the shape a careless guard would produce.
        (2) They may be absent ONLY when this option has no usable sample
            population (``percentiles_source == 'unavailable'``). Dropping the
            summary statistics of a distribution we DID sample would be silent
            data loss.

        The REVERSE of (2) is deliberately NOT enforced, for the same reason
        ``_downside_requires_samples`` enforces only one direction: an option
        can legitimately have no raw ``samples`` array (percentiles
        'unavailable') while the analyzer still computed an honest mean and std
        for it, and forcing the biconditional would break those constructors.
        """
        if (self.mean is None) != (self.std is None):
            raise ValueError(
                "2.477 invariant: outcome.mean and outcome.std must be present or "
                f"absent together; got mean={self.mean!r}, std={self.std!r}. A "
                "distribution cannot be half-summarised."
            )
        if self.mean is None and self.percentiles_source != "unavailable":
            raise ValueError(
                "2.477 invariant: outcome.mean/std are absent but "
                f"percentiles_source={self.percentiles_source!r} — summary "
                "statistics may only be omitted when the option has no usable "
                "sample population (percentiles_source == 'unavailable'). "
                "Dropping them for a sampled option is silent data loss."
            )
        return self


# =============================================================================
# Downside / Tail-risk (B2)
# =============================================================================


class DownsideV2(BaseModel):
    """Per-option DOWNSIDE / tail-risk view (B2), read from the MC outcome
    samples the v2 engine already draws — no new sampling.

    Two metric families with DIFFERENT (deliberate) sample populations:

    * ``cvar_10`` / ``p05`` — MARGINAL tail metrics of this option's own outcome
      samples, taken from the SAME post-``_apply_auto_scaled_noise`` population as
      ``outcome.p10/p50/p90/mean`` (consistent, conservative/noised distribution).
    * ``expected_regret`` — a JOINT (cross-option) metric, computed from the
      PRE-noise Common-Random-Numbers population — the same samples that produce
      ``win_probability``. Auto-scaled noise draws INDEPENDENT per-option noise,
      which breaks the CRN alignment this metric relies on, so it is NOT taken
      from the noised samples (CODE-REVIEW-ISL F1).

    Rides as a sibling to ``outcome``; when omitted it is ABSENT, never a JSON
    null. All values are in the SAME units as ``outcome.mean`` /
    ``outcome.p10`` (no normalisation).

    EMISSION RULE — ⚠ this said "present EXACTLY when
    ``outcome.percentiles_source == 'samples'``" until 2.477(h); that
    biconditional was made false by 2.475 and is now stated as the implication
    it actually is. Sample availability is NECESSARY but not SUFFICIENT:

        ``downside`` present  ⟹  ``outcome.percentiles_source == 'samples'``

    (that direction IS enforced, by ``OptionResultV2._downside_requires_samples``).
    The block is additionally omitted, on a run that still returns 200, when any
    of its three components cannot be computed honestly — the threaded pre-noise
    joint regret absent or non-finite, or ``cvar_10``/``p05`` non-finite on an
    extreme finite population. Those are runs that could not produce a response
    AT ALL before 2.475: omitting one block is strictly more information than the
    500 it replaced, and a tail-risk number from a degraded population is not
    trustworthy (absent != wrong, trap #13).
    """

    cvar_10: float = Field(
        ...,
        description="Expected shortfall: the MEAN of the worst 10% (lowest) "
        "outcome samples. Tail mass = CVAR_LEVEL (0.10), a DOCTRINE-PENDING(Neil) "
        "default. Same units as outcome.mean. Guaranteed <= outcome.p10 (mean of "
        "the worst decile cannot exceed the decile boundary). "
        "POPULATION: computed from the POST-``_apply_auto_scaled_noise`` outcome "
        "samples — the SAME (noised) population as outcome.p10/p50/p90/mean and "
        "p05, and DIFFERENT from expected_regret (which is pre-noise).",
    )
    p05: float = Field(
        ...,
        description="5th-percentile outcome — extends the p10/p50/p90 family "
        "downward, computed with the SAME percentile convention as p10. "
        "POPULATION: the POST-``_apply_auto_scaled_noise`` (noised) samples, like "
        "outcome.p10/p50/p90/mean and cvar_10 — and DIFFERENT from "
        "expected_regret (pre-noise).",
    )
    expected_regret: float = Field(
        ...,
        ge=0,
        description="Joint expected regret: mean over MC samples of "
        "(best-option outcome - this option's outcome) at the SAME underlying "
        "draw (Common Random Numbers). >= 0 by construction; ~0 for the option "
        "that wins each sample. POPULATION: the PRE-noise CRN-aligned samples "
        "(the same population as win_probability), NOT the "
        "post-auto-scaled-noise outcome samples used by cvar_10/p05 — independent "
        "per-option noise would break the CRN alignment this metric requires. "
        "(So within one downside{} object the two metric families ride DIFFERENT "
        "populations: expected_regret is pre-noise; cvar_10/p05 are post-noise.)",
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
    # B2 downside — additive, optional, sibling to `outcome`. Left None (=>
    # ABSENT under exclude_none=True on the wire, never a JSON null) when it
    # cannot be built. 2.477(h): this comment and the description below claimed
    # "EXACTLY when outcome.percentiles_source == 'samples'"; 2.475 made that
    # false — sample availability is necessary, not sufficient. The implication
    # (downside => samples) is the one that holds and the one enforced below;
    # see DownsideV2's docstring for the full emission rule.
    downside: Optional[DownsideV2] = Field(
        None,
        description="Downside / tail-risk view {cvar_10, p05, expected_regret}. "
        "Requires the option's MC samples (outcome.percentiles_source == "
        "'samples'), and is additionally omitted when the pre-noise joint regret "
        "or cvar_10/p05 cannot be computed as finite values. Omitted, never null.",
    )
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

    @model_validator(mode="after")
    def _downside_requires_samples(self) -> "OptionResultV2":
        """B2: guard the FABRICATION direction — a downside (tail-risk) object may
        ride only alongside real MC samples.

        ``downside`` present ⟹ ``outcome.percentiles_source == 'samples'``.
        Emitting cvar_10 / p05 / expected_regret when the option has no valid
        samples would invent a distribution from nothing; this fails loud at
        construction so a mutation that always emits the object cannot pass
        silently (trap-#13).

        The REVERSE (samples ⟹ downside) is deliberately NOT enforced here: it is
        a completeness property of the emission locus (src/api/robustness.py),
        asserted end-to-end in the API tests. Many legitimate constructors build
        OptionResultV2 with percentiles_source defaulting to 'samples' and no
        enrichment, and forcing the biconditional would break every one of them.
        """
        if self.downside is not None and self.outcome.percentiles_source != "samples":
            raise ValueError(
                "B2 invariant: downside is present but outcome.percentiles_source="
                f"{self.outcome.percentiles_source!r} — tail-risk metrics require "
                "valid MC samples (percentiles_source == 'samples'). The downside "
                "object must never ride without the samples it summarises."
            )
        return self


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

    # Slice 6b — the wire end of the constraint_id echo. The paired contract
    # schema (boundary.EnrichmentConstraintResultSchema) has REQUIRED this
    # property all along; ISL's non-emission was carried as an accepted omission
    # in tests/contract_drift/drift_baseline.json. Emitting it closes that
    # omission, so the baseline entry is removed in the same change.
    #
    # Optional here, not required: a request that omits constraint_id must keep
    # producing exactly the pre-adoption wire shape, and the endpoint serialises
    # with exclude_none=True, so an absent id is OMITTED rather than sent as null.
    constraint_id: Optional[str] = Field(
        None,
        description=(
            "Opaque identity echoed verbatim from the request's goal_constraints "
            "entry. Present iff the caller supplied one. Lets a consumer key "
            "results by the ratified constraint ID instead of reconstructing them "
            "from response ordinal or (node_id, operator), neither of which can "
            "distinguish two constraints on the same node with the same operator."
        ),
    )
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
    # ROADMAP 2.228-F3 (design §1.1 gap 1, §2.4). The bisection in
    # RobustnessAnalyzerV2._compute_edge_e_values already evaluated the argmax on
    # the flipped side of every bracket and DISCARDED it, so a consumer could say
    # "this edge can flip the recommendation" but never "…to which option".
    # Retained here at ZERO additional evaluations: the flipped endpoint of the
    # final bracket is only ever assigned from an evaluation that flipped, so the
    # argmax recorded at that assignment IS the argmax at flip_mean.
    alternative_winner_id: Optional[str] = Field(
        None,
        description="Option that becomes the argmax when this edge's strength mean "
        "reaches flip_mean (expected-value background, analyzer tie-break: highest "
        "outcome, then lowest option id). Null when the edge is unflippable — never "
        "invented for an edge that cannot flip.",
    )
    baseline_winner_id: Optional[str] = Field(
        None,
        description="Argmax option at the expected-value baseline this row was "
        "searched against. Emitted per-row so a consumer can fail closed when it "
        "differs from the MC-recommended option (design R3): the E-value search "
        "runs in the expected-value world, which is not guaranteed to agree with "
        "the sampled recommendation.",
    )
    stability: Optional[FlipStabilityBandV2] = Field(
        None,
        description="Seed-sweep stability band for this flip threshold (Track S Phase 1). "
        "Default-on, additive. Absent only when the all-or-nothing band budget "
        "(FLIP_STABILITY_BUDGET_MS) trips.",
    )

    model_config = {"extra": "ignore"}


class FactorFlipStabilityBandV2(BaseModel):
    """Seed-sweep stability band for one FACTOR's flip value (ROADMAP 2.228-F3).

    Same shape and semantics as FlipStabilityBandV2, with one deliberate
    difference: the list field is ``seed_flip_values`` (factor values in the
    normalised [0, 1] domain of observed_state.value), not ``seed_flip_means``
    (edge strength means). Sharing the name across two different quantities is
    exactly the kind of conflation that has cost this platform diagnoses before.

    Method: N child seeds SHA-256-derived from the request seed under the tag
    ``{seed}:factor_flip_stability:{i}`` — a DISTINCT tag from the edge band's
    ``{seed}:flip_stability:{i}``, so the factor sweep never consumes or shifts
    the edge sweep's stream. One sampled edge background per child seed, SHARED
    across factors (common random numbers), so bands are comparable across rows.
    Under each background the crossing is re-derived in closed form from freshly
    measured per-option slopes.

    MEMBERSHIP: as with edge bands, the base ``flip_value`` is NOT a member of
    this sweep — it is derived against the expected-value background, which is
    never one of the sampled backgrounds. flip_value MAY therefore lie outside
    [band_min, band_max]. Consumers must NOT assume membership.

    SCOPE OF THE UNCERTAINTY (design R7): these backgrounds vary EDGE strengths
    only; other factors stay at their observed values. That matches the edge
    bands' semantics and keeps the two comparable, but it understates uncertainty
    arising from correlated errors in other factors.
    """

    n_seeds: int = Field(..., ge=1, description="Number of child seeds swept")
    n_seeds_flipped: int = Field(
        ...,
        ge=0,
        description="Seeds whose sampled background admits a flip inside [0, 1]. "
        "When 0, the band_* fields are omitted (exclude_none).",
    )
    band_min: Optional[float] = Field(
        None, description="Minimum flip value across flipped seeds. Omitted when nothing flips."
    )
    band_median: Optional[float] = Field(
        None, description="Median flip value across flipped seeds. Omitted when nothing flips."
    )
    band_max: Optional[float] = Field(
        None, description="Maximum flip value across flipped seeds. Omitted when nothing flips."
    )
    band_width: Optional[float] = Field(
        None,
        description="band_max - band_min. Omitted when nothing flips. INTERPRETATION "
        "TRAP (identical to the edge band): when n_seeds_flipped == 1 this is 0.0 by "
        "construction, so a naive width rubric reads maximal stability from a single "
        "flipped background. Consumers MUST condition any width-based confidence "
        "rubric on n_seeds_flipped.",
    )
    seed_flip_values: List[Optional[float]] = Field(
        ...,
        description="Per-child-seed flip value, in child-seed order; null where that "
        "seed's background admits no flip inside [0, 1].",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    model_config = {"extra": "ignore"}


class FactorFlipValueV2(BaseModel):
    """Value at which changing ONE root factor changes the winning option.

    ROADMAP 2.228-F3. Request-gated by ``include_factor_flips`` (default False),
    so a consumer that does not ask sees a byte-identical response.

    WHY THIS EXISTS. The prior flip search lived in PLoT and probed factors by
    re-running a full Monte Carlo per probe value. The diagnosis
    (diagnosis-2228-enrichment-values.md §1) proved with a live control that the
    factors it chose were mathematically incapable of flipping the winner: for a
    factor no option intervenes on and that is not upstream of differential
    severing, every option's outcome moves by the IDENTICAL amount, so the argmax
    is invariant. 43 live rows, zero `found`.

    METHOD. Epsilon noise is disabled before all post-MC structural analyses, so
    the SCM is exactly affine in a ROOT factor's value: goal_o(F) = A_o + T_o*F.
    Two deterministic evaluations per option measure (A_o, T_o) exactly; the
    leader/rival crossing is then closed form, F* = (A_i - A_j)/(T_j - T_i). No
    Monte Carlo, so there is no sampling error to quote — the honest uncertainty
    statement is ``stability``, not a noise floor.

    RESOLUTION CONTRACT (design §2.3). ``flip_value`` is EXACT in the
    expected-value world. How far it moves in other worlds is ``stability`` and
    NOTHING else: this row deliberately carries no flip probability and no EVPI
    quantity, because the probability machinery available here (K=100 marginal
    switch, n=2000 p_win) sits below its own resolution.
    """

    factor_id: str = Field(..., description="Root factor node id")
    current_value: float = Field(
        ...,
        description="The factor's current value, in the NORMALISED [0, 1] domain of "
        "observed_state.value. When the factor carries only a parameter_uncertainties "
        "entry, this is the CENTRE OF ITS DECLARED PRIOR (for a uniform prior, the "
        "midpoint of [range_min, range_max]) — the same value the sampler centres it "
        "on and the same value the sensitivity probe perturbs around. It was 0.0 "
        "until ROADMAP 2.1020, which stated a current value outside the factor's own "
        "declared support. 0.0 remains the value only when there is genuinely nothing "
        "to go on: no observed value and no prior range. Denormalisation to user "
        "units is PLoT's responsibility — ISL never mixes a normalised number with a "
        "display unit.",
    )
    flip_value: Optional[float] = Field(
        None,
        description="Normalised [0, 1] value at which the winning option changes. "
        "Null whenever flip_reason is not 'found' — never a fabricated in-range "
        "number for a factor whose crossing lies outside the domain.",
    )
    direction: Optional[Literal["increase", "decrease"]] = Field(
        None,
        description="Direction the factor value must move from current_value to reach "
        "flip_value. Null when flip_value is null: a direction for a flip that does "
        "not exist would be a fabricated claim.",
    )
    flip_reason: str = Field(
        ...,
        description="'found' — a confirmed argmax change inside [0, 1]. "
        "'no_effect_within_bounds' — per-option transmission slopes genuinely differ, "
        "but no crossing lies inside [0, 1]. "
        "'structurally_invariant' — the per-option transmission slopes are identical "
        "(spread <= 1e-9), so no value of this factor can move the argmax. This is a "
        "MATHEMATICAL ATTESTATION, not a failed or timed-out probe, and it is the "
        "honest wire statement for the class the diagnosis proved unprobeable. "
        "'candidate_cap_exceeded' — a genuine candidate that ranked below "
        "FACTOR_FLIP_MAX_CANDIDATES by slope spread and was not evaluated; emitted "
        "rather than dropped so the omission is never silent. Open vocabulary.",
    )
    alternative_winner_id: Optional[str] = Field(
        None,
        description="Option that becomes the argmax just past flip_value. Null unless "
        "flip_reason is 'found'.",
    )
    baseline_winner_id: str = Field(
        ...,
        description="Argmax option at the expected-value baseline, i.e. the winner this "
        "flip is measured AGAINST. Emitted per-row so a consumer can fail closed when it "
        "disagrees with the MC-recommended option (design R3).",
    )
    stability: Optional[FactorFlipStabilityBandV2] = Field(
        None,
        description="Seed-sweep stability band. Present only for evaluated candidates: "
        "a 'structurally_invariant' row has no band because its no-flip is proven "
        "rather than sampled, and computing one would spend the compute the candidate "
        "screen exists to save.",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
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
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="NOT A CONFIDENCE LEVEL. This is the UNCALIBRATED "
        "recommendation-stability fraction — the share of sampled scenarios in "
        "which the recommended option won — served under a legacy field name. It "
        "carries no coverage guarantee and no calibration study; it does not say "
        "how likely the recommendation is to be right. Read `recommendation_stability` "
        "(the same quantity, honestly named) and branch on `confidence_basis`. "
        "Arch step 1 (2026-07-26): the previous value, "
        "min(0.99, stability*(1-1/sqrt(n_samples))), additionally moved with the "
        "sample COUNT rather than with the estimator's sampling error. The "
        "description this replaces read, in full, 'Confidence [0, 1]'.",
    )
    confidence_basis: Literal["recommendation_stability_uncalibrated"] = Field(
        default="recommendation_stability_uncalibrated",
        description="Machine-readable semantics of the `confidence` field, so a "
        "consumer does not have to infer them from prose. "
        "'recommendation_stability_uncalibrated': `confidence` is the "
        "recommendation-stability fraction with no calibration behind it. Mirrors "
        "the ConfidenceProvenance marker that rides beside "
        "FactorSensitivityV2.confidence.",
    )
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


class ConfidenceProvenance(BaseModel):
    """Honest disclosure marker for a factor's `confidence` figure (S2).

    The per-factor `confidence` is built from a PROVISIONAL stability->confidence
    mapping (STABILITY_CONFIDENCE_MAP and its blend weights) that is explicitly
    NOT research-validated — it is a Neil-gate-1 operational default pending
    scientific calibration. This marker rides alongside `confidence` so a
    consumer can tell a provisional heuristic from a calibrated probability, and
    so any recalibration is a DISCLOSED, versioned change rather than a silent
    reweighting.

    Contract:
    - `method_version` identifies the confidence method. Any change to the
      mapping constants MUST bump it (enforced by the fingerprint guard in
      tests/unit/test_confidence_provenance.py) — no silent reweighting.
    - `calibrated` stays False until a validated calibration exists; while it is
      False, `confidence` must be read as a provisional stability heuristic, not
      a calibrated probability.
    """

    method_version: str = Field(
        ...,
        description="Identifier of the confidence method. PROVISIONAL mapping "
        "(Neil gate 1) — any change to the mapping constants MUST bump this "
        "version (no silent reweighting).",
    )
    calibrated: bool = Field(
        ...,
        description="False until a validated calibration exists. While False, "
        "the confidence figure is a provisional stability heuristic, not a "
        "calibrated probability.",
    )

    model_config = {"extra": "ignore"}

    @classmethod
    def bootstrap(cls, method_version: str) -> "ConfidenceProvenance":
        """Marker for the bootstrap-CV-blend confidence method.

        `calibrated` is hardcoded False here — the SINGLE place it is set: the
        mapping is PROVISIONAL (Neil gate 1) and stays uncalibrated until a
        validated calibration exists.
        """
        return cls(method_version=method_version, calibrated=False)

    @classmethod
    def graph_structural(cls) -> "ConfidenceProvenance":
        """Marker for the graph-structural FALLBACK confidence method (F-2) — a
        DIFFERENT method that stamps its OWN version (GRAPH_STRUCTURAL_METHOD_VERSION),
        never the bootstrap blend's, so the wire never names a method that did not
        produce the number.
        """
        return cls.bootstrap(GRAPH_STRUCTURAL_METHOD_VERSION)


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
    # S2 disclosure marker — additive, optional. Populated EXACTLY when
    # `confidence` is populated; left None (=> ABSENT under exclude_none=True on
    # the wire, never a JSON null) when `confidence` is absent. Signals that the
    # confidence figure comes from a PROVISIONAL, uncalibrated mapping so any
    # recalibration is a disclosed, versioned change (Neil gate 1).
    confidence_provenance: Optional[ConfidenceProvenance] = Field(
        None,
        description="Disclosure marker for the `confidence` figure: {method_version, "
        "calibrated}. Present exactly when `confidence` is present; omitted (not null) "
        "otherwise. calibrated is False until a validated calibration exists.",
    )
    importance_rank: Optional[int] = Field(
        None, ge=1, description="Rank by importance (1 = most important)"
    )
    observed_value: Optional[float] = Field(
        None,
        description="The factor's OBSERVED value (observed_state.value), or null when the "
        "factor carries only a declared prior. This is deliberately NOT the central value "
        "the engine computed with: since ROADMAP 2.1020 a prior-only factor is centred on "
        "its prior's midpoint, and publishing that number in a field named 'observed' "
        "would present a derived value as a measured one. Use value_defaulted to tell the "
        "two apart; the centre itself is published per factor on factor_flip_values."
        "current_value.",
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
        description="True when the factor's value was genuinely defaulted — no observed "
        "value AND no declared prior range to centre on, so it fell back to 0.0. Omitted "
        "when an observed value was supplied, and (since ROADMAP 2.1020) also omitted when "
        "the factor is centred on its declared prior: that value is derived, not defaulted. "
        "Derived from the single central-value resolver, so it cannot drift from the number "
        "the engine used.",
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

    @model_validator(mode="after")
    def _confidence_provenance_iff_confidence(self) -> "FactorSensitivityV2":
        """F-3: enforce the S2 iff-invariant at the MODEL level, both directions.

        `confidence_provenance` (the disclosure marker) must be present EXACTLY when
        `confidence` is present. Without this, a mutation that ALWAYS emits the
        marker (even when confidence is None) — or never emits it — passes the suite
        silently (trap-#13: an absence assertion with no discriminating enforcement).
        Enforcing the iff here makes marker-without-confidence AND
        confidence-without-marker fail loud at construction.
        """
        confidence_present = self.confidence is not None
        provenance_present = self.confidence_provenance is not None
        if confidence_present != provenance_present:
            raise ValueError(
                "S2 iff-invariant violated: confidence_provenance must be present "
                "EXACTLY when confidence is present "
                f"(confidence={'set' if confidence_present else 'None'}, "
                f"confidence_provenance="
                f"{'set' if provenance_present else 'None'}). The disclosure marker "
                "rides exactly alongside the confidence figure — never without it, "
                "and never absent when confidence is emitted."
            )
        return self


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


class EffectiveCorrelationV2(BaseModel):
    """The EFFECTIVE (post-projection) correlation for one supplied pair (F4).

    Present inside ``CorrelationProjectionV2.effective_correlations`` when the
    supplied matrix was Higham-projected. ``effective_rho`` is the off-diagonal the
    copula ACTUALLY used; ``adjustment`` = ``effective_rho - requested_rho`` is the
    silent change the projection made. Together with the unit diagonal these entries
    reconstruct the effective correlation matrix, so a caller sees which correlations
    really drove the numbers — not only the aggregate Frobenius distance.

    ``stated`` distinguishes a pair the caller supplied (True) from an UNSTATED pair
    that defaulted to correlation 0 (assumed-independent) and was MOVED off 0 by the
    projection (False, ``requested_rho == 0.0``) — those moved zero-fill pairs are
    disclosed too so the effective matrix is complete (F-2).
    """

    factor_a: str = Field(..., description="ID of the first factor node in the pair.")
    factor_b: str = Field(..., description="ID of the second factor node in the pair.")
    requested_rho: float = Field(
        ...,
        description="The correlation the caller supplied for this pair, or 0.0 for an "
        "unstated (assumed-independent) pair.",
    )
    effective_rho: float = Field(
        ...,
        description="The off-diagonal actually used after nearest-correlation "
        "projection (what drove the copula draw).",
    )
    adjustment: float = Field(
        ...,
        description="effective_rho - requested_rho: the signed change the projection "
        "silently applied to this correlation.",
    )
    stated: bool = Field(
        ...,
        description="True if the caller supplied this pair; False if it was an unstated "
        "(assumed-zero) pair the projection moved off 0.",
    )

    model_config = {"extra": "ignore"}


class CorrelationProjectionV2(BaseModel):
    """PSD-repair disclosure for a client-supplied correlation matrix (B3-S1).

    Present inside ``correlation_model.psd_projection`` ONLY when the assembled
    correlation matrix was not positive-semidefinite and was projected to the
    nearest correlation matrix (Higham 2002). Absent (null) when the input was
    already PSD, so the presence of this block is itself the disclosure that the
    supplied correlations were adjusted.

    Hard-invalid matrices (indefinite beyond the near-PSD repair band) are rejected
    at request validation with a typed 422 BEFORE any projection (F4, D-23.13), so
    this block is only ever emitted for genuinely near-PSD inputs.
    """

    applied: bool = Field(
        ..., description="Always true when this block is present (the matrix was projected)."
    )
    method: str = Field(
        ...,
        description="Projection method — 'higham_2002_nearest_correlation' (true "
        "nearest-correlation via alternating eigenvalue projections).",
    )
    frobenius_distance: float = Field(
        ...,
        description="Frobenius norm of (projected - supplied) — how far the supplied "
        "matrix was moved to reach the nearest valid correlation matrix.",
    )
    max_abs_off_diagonal_adjustment: float = Field(
        ...,
        description="Largest absolute change to any single off-diagonal correlation "
        "entry during projection.",
    )
    iterations: int = Field(..., description="Alternating-projection iterations used to converge.")
    effective_correlations: Optional[List[EffectiveCorrelationV2]] = Field(
        None,
        description="F4: the EFFECTIVE post-projection correlation per supplied pair "
        "(requested_rho, effective_rho, adjustment). With the unit diagonal these "
        "reconstruct the effective correlation matrix that actually drove the copula, "
        "so a caller sees more than the aggregate Frobenius distance. Additive-optional.",
    )

    model_config = {"extra": "ignore"}


# Suppressed-attribution manifest tokens (B3-S1). Single source of truth shared by
# the PRODUCER (the analyzer's correlation skip-site appends) and the CONSUMER
# (response_builder's factor_sensitivity_status derivation), so the token cannot drift
# across files (CLAUDE.md #12: a bare-string literal duplicated producer↔consumer is a
# silent-drift seam — if the two spellings diverge the status quietly under-discloses).
SUPPRESSED_ATTR_FACTOR_SENSITIVITY = "factor_sensitivity"
SUPPRESSED_ATTR_STABILITY_THRESHOLDS = "stability_thresholds"
SUPPRESSED_ATTR_CONDITIONAL_WINNERS = "conditional_winners"
SUPPRESSED_ATTR_P_WIN_SENSITIVITY = "p_win_sensitivity"


class CorrelationModelV2(BaseModel):
    """Disclosure block for the active factor-correlation model (B3-S1, D-23.4).

    Present ONLY when the request supplied ``factor_correlations`` (independence
    stays the silent default). It discloses the copula method, the MANDATORY
    tail-independence caveat (load-bearing whenever the copula co-ships with the
    downside/CVaR block), any PSD projection, and which independence-assuming
    per-factor attributions were suppressed and why.
    """

    method: str = Field(
        ...,
        description="Correlation model — 'gaussian_copula_v1' (Gaussian copula over the "
        "factors' existing marginals).",
    )
    active: bool = Field(..., description="Always true when this block is present.")
    correlated_factors: List[str] = Field(
        ...,
        description="Factor node IDs drawn jointly under the copula (canonical draw order).",
    )
    n_pairs: int = Field(..., description="Number of supplied pairwise correlations.")
    tail_dependence: str = Field(
        default="none",
        description="Tail-dependence coefficient class of the copula — 'none' for the "
        "Gaussian copula (zero upper/lower tail dependence).",
    )
    tail_dependence_note: str = Field(
        ...,
        description="MANDATORY caveat: the Gaussian copula has zero tail dependence, so "
        "joint extreme co-movements may be understated and downside/CVaR can be "
        "optimistic when factors are strongly correlated.",
    )
    psd_projection: Optional[CorrelationProjectionV2] = Field(
        None,
        description="Present only when the supplied matrix was not PSD and was projected "
        "to the nearest correlation matrix (Higham 2002). Null/absent when the input was "
        "already valid.",
    )
    suppressed_attributions: List[str] = Field(
        default_factory=list,
        description="Independence-assuming per-factor attributions omitted under active "
        "correlation (e.g. factor_sensitivity, p_win_sensitivity, conditional_winners). "
        "factor_evppi is NOT listed — it is a conditional-expectation quantity on the "
        "joint copula samples and remains emitted. Absent from the response, not null — "
        "this list names what was withheld.",
    )
    suppression_reason: str = Field(
        default="not_separable_under_correlation",
        description="Why the listed attributions were suppressed: per-factor "
        "independence-assuming decompositions are not separable once factors are "
        "correlated.",
    )

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _present_iff_active(self) -> "CorrelationModelV2":
        """Sibling-presence emission-iff (altitude Q1): this disclosure block is
        emitted EXACTLY when correlation is active, so whenever it exists ``active``
        must be True. Fail loud in Pydantic if a code path ever constructs it
        inactive — the same fail-loud altitude as decision_evpi /
        confidence_provenance, converging the enforcement depth."""
        if self.active is not True:
            raise ValueError(
                "correlation_model emission-iff violated: the block is present "
                f"but active={self.active!r} (it is emitted only when correlation "
                "is active, so active must be True)."
            )
        return self


class FactorEvppiEntryV2(BaseModel):
    """One per-factor EVPPI entry (S2, D-23.8). Altitude Q1: the more
    safety-critical of the VOI blocks (it hosts the clamped_low DEAD-MAN'S-SWITCH and
    two clamp-audit booleans) was the LESS typed — a bare ``Dict[str, Any]`` — than its
    sibling ``FactorSensitivityV2``. Typed here so a producer typo, a dropped status,
    or a wrong-typed audit field fails loud in Pydantic instead of serialising clean.

    Field ORDER is load-bearing: it matches the emission dict in
    ``_compute_factor_evppi`` exactly, so serialization is byte-identical to the prior
    ``Dict[str, Any]`` payload (verified pre/post on multiple shapes incl.
    correlation-active).
    """

    factor_id: str
    evppi: float
    evppi_raw: float
    baseline_max_expected_utility: float
    conditional_max_expected_utility: float
    units: str
    method: str
    regression_degree: int
    n_samples: int
    clamped_low: bool
    clamped_high: bool
    noise_floor: float
    status: str
    correlation_active: bool


class FactorEvpcEntryV2(BaseModel):
    """One per-lever EVPC entry (S4, D-23.8). Typed to match the FactorEvppiEntryV2
    house style (#104 C3): a producer typo, a dropped field, or a wrong-typed audit
    field fails loud in Pydantic instead of serialising clean. best_candidate_value is
    a REQUIRED float, so it is always present even when evpc == 0 (all candidates
    underperform) — the missing-argmax mutation now fails at the type layer.

    Field ORDER is load-bearing: it matches the emission dict in ``_compute_factor_evpc``
    exactly, so serialization is byte-identical to the prior ``Dict[str, Any]`` payload.

    The CROSS-FIELD value-integrity guard below is the EVPC analogue of the lifted
    decision_evpi / correlation_model emission-iff validators — it keeps the clamp
    identity ON this block, NOT coupled into ISLResponseV2's own validators.
    """

    factor_id: str
    evpc: float
    evpc_raw: float
    best_candidate_value: float
    baseline_max_expected_utility: float
    best_do_expected_utility: float
    units: str
    method: str
    n_samples: int
    n_candidate_values: int
    clamped_low: bool
    correlation_active: bool

    @model_validator(mode="after")
    def _value_integrity(self) -> "FactorEvpcEntryV2":
        """Guard the EVPC value BOTH ways (fail loud on a compute/emission mutation):

        (1) FINITE: evpc, evpc_raw, best_candidate_value and the audit EUs are finite
            (a bare ``float`` still admits NaN/inf, which must never reach the wire).
        (2) CLAMP IDENTITY (both directions): evpc == max(0, evpc_raw). Catches a
            forgotten clamp (raw shipped as evpc) or a stray clamp on a positive raw.
            EVPC is non-negative by construction (control cannot hurt).
        (3) TAGS: units == 'outcome' and method == 'grid_do_v1' — the self-describing
            tags a consumer relies on to read the number in the right units/method.
        """
        for name, val in (
            ("evpc", self.evpc),
            ("evpc_raw", self.evpc_raw),
            ("best_candidate_value", self.best_candidate_value),
            ("baseline_max_expected_utility", self.baseline_max_expected_utility),
            ("best_do_expected_utility", self.best_do_expected_utility),
        ):
            if not math.isfinite(val):
                raise ValueError(
                    f"factor_evpc entry for '{self.factor_id}': {name} must be "
                    f"finite; got {val!r}."
                )

        expected_evpc = max(0.0, self.evpc_raw)
        tol = 1e-9 + 1e-9 * abs(expected_evpc)
        if self.evpc < -tol or abs(self.evpc - expected_evpc) > tol:
            raise ValueError(
                f"factor_evpc entry for '{self.factor_id}': evpc must equal "
                f"max(0, evpc_raw) = {expected_evpc!r} and be >= 0; got "
                f"evpc={self.evpc!r}, evpc_raw={self.evpc_raw!r}. EVPC is the "
                "clamped value of control."
            )
        if self.units != "outcome":
            raise ValueError(
                f"factor_evpc entry for '{self.factor_id}' must be in outcome units "
                f"(units == 'outcome'); got {self.units!r}."
            )
        if self.method != GRID_DO_EVPC_METHOD:
            raise ValueError(
                f"factor_evpc entry for '{self.factor_id}' must be method-tagged "
                f"'{GRID_DO_EVPC_METHOD}'; got {self.method!r}."
            )
        return self


class SamplePopulationProvenanceV2(BaseModel):
    """Which sample population produced each served metric (arch step 1, 2026-07-26).

    ``auto_noise_applied`` is a single boolean for a whole envelope that mixes two
    populations. The B2 CRN-fix comment in ``RobustnessAnalyzerV2.analyze``
    records why they are mixed: ``expected_regret`` is a JOINT
    Common-Random-Numbers metric and must come from the PRE-noise samples (the
    auto-noise draw is independent per option, which breaks CRN alignment and
    inflates regret by a max-over-independent-noise premium), while the marginal
    tail metrics stay on the noised samples for consistency with the noised
    percentiles. That is a defensible choice — but a consumer reading a response
    could not tell which of the numbers in front of it the noise had reached.
    This block says so per metric.
    """

    auto_scaled_noise_applied: bool = Field(
        ...,
        description="Whether the auto-scaled noise heuristic actually modified any "
        "samples in this response. Default-off since 2026-07-26 "
        "(ENABLE_AUTO_SCALED_NOISE); when false, every metric below is model-only.",
    )
    noise_scale: Optional[str] = Field(
        None,
        description="Human-readable noise scale when applied, e.g. '1.0x model std "
        "per outcome/risk sample (~sqrt(2) spread inflation)'. Null when not applied.",
    )
    calibration_status: str = Field(
        default="uncalibrated_poc_heuristic",
        description="Calibration standing of the noise heuristic. It is a PoC "
        "heuristic pending formal review and calibration against pilot outcome "
        "data — it has no coverage study behind it.",
    )
    metric_populations: Dict[str, Literal["model_only", "noise_inflated"]] = Field(
        ...,
        description="Per-metric population label. 'model_only' = computed from the "
        "structural model's own samples; 'noise_inflated' = computed from samples "
        "with the auto-scaled noise term added.",
    )
    unnoised_constraint_node_ids: List[str] = Field(
        default_factory=list,
        description="Constraint node IDs whose probabilities were computed on "
        "model-only samples while the goal node's were noised (the same mix "
        "disclosed by the CONSTRAINT_SAMPLES_UNNOISED inference warning). Empty "
        "when there is no mix.",
    )

    model_config = {"extra": "ignore"}


# =============================================================================
# Main V2 Response
# =============================================================================


class ObjectiveRankingV2(BaseModel):
    """What the ranking on this response optimised (ROADMAP 2.1192).

    ``win_probability`` is the fraction of Monte Carlo draws on which an option
    scored best under the request's objective sense. Until 2.1192 that sense was
    hardcoded to "largest goal-node value" and nothing on the wire said so, so a
    consumer rendering "which option wins" was making a claim the number did not
    support: measured at 28fe0c95, the crowned option could carry
    ``probability_of_goal = 0.0``, and supplying the user's target moved the
    ranking by exactly nothing.

    This block is the provenance of the single ranking, not a rival to it.
    """

    direction: Literal["maximise", "minimise", "target"] = Field(
        ...,
        description="The objective sense the winner rule applied. Under "
        "status='withheld' this is the sense that was REQUESTED and refused, "
        "which is what a surface needs in order to say what could not be done.",
    )
    attested: bool = Field(
        ...,
        description="True when the caller stated the objective. FALSE means "
        "'maximise' is ISL's disclosed default and the team's aim was never "
        "supplied — a ranking a surface should present as an assumption, or ask "
        "about, rather than as the answer to their goal.",
    )
    status: Literal["computed", "withheld"] = Field(
        ...,
        description="'withheld' means NO ranking exists in this response: "
        "win_probability is omitted on every option and the robustness block "
        "(whose confidence IS the recommended option's win share) is omitted "
        "too. It is the absence of a ranking, never a flat one.",
    )
    withheld_reason: Optional[str] = Field(
        None, description="Machine-readable reason when status='withheld'."
    )


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
    factor_sensitivity_status: Literal[
        "computed", "skipped", "suppressed", "unavailable", "error"
    ] = Field(
        ...,
        description=(
            "Factor sensitivity status. 'suppressed' means active factor "
            "correlation deliberately withheld the per-factor attributions "
            "(non-separable under correlation) — see correlation_model; distinct "
            "from 'skipped' (not requested / nothing to compute)."
        ),
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

    # ROADMAP 2.1192. The ranking's provenance, on the CLIENT wire.
    #
    # Optional purely so error paths (where build() runs without results) stay
    # constructible; it is populated on every analysis that produces options.
    # Without it, `win_probability` crosses the boundary meaning "largest goal
    # value" while every surface downstream renders it as "best option" — which
    # is the defect this row exists to close, and it is not closed until the
    # distinction can be READ by the consumer, not merely computed here.
    objective_ranking: Optional[ObjectiveRankingV2] = Field(
        None,
        description="What this response's ranking optimised, and whether the "
        "user's objective was actually stated. See ObjectiveRankingV2.",
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

    # Per-factor win-probability sensitivity (enhancement — gated by include_voi).
    # S2 (D-23.8) HONEST RELABEL: this wire block was ``factor_evpi`` with an
    # ``evpi`` field, but it is NOT value-of-information (it holds the decision fixed
    # and reports a win-probability delta), so it was mislabelled. Renamed to
    # ``p_win_sensitivity`` with de-EVPI'd field names + a ``method`` tag; the
    # numbers are byte-identical to the pre-S2 ``factor_evpi`` values.
    p_win_sensitivity: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Per-factor win-probability sensitivity: for each uncertain "
        "factor, how much the recommended option's win probability (or P(joint_goal) "
        "when goal_constraints are set) moves when that factor is FIXED at its mean, "
        "with the decision held fixed at the recommended option. Fields: p_win_delta "
        "(probability units), p_win_delta_percentage_points, current_metric, "
        "perfect_metric, metric_type (p_win_recommended|p_joint_goal), method "
        "('p_win_delta_at_mean_v1'), n_samples, status, clamped, noise_floor. This is "
        "NOT value-of-information: holding the decision fixed, it structurally cannot "
        "capture option-switching, and it is in probability (not outcome) units, with "
        "its OWN Monte Carlo redraw (not the CRN joint population). For decision value "
        "use decision_evpi (whole decision) and factor_evppi (per-factor), both in "
        "outcome units. Non-negative (negative estimates clamped to 0, clamped=true).",
    )

    # Decision-level EVPI (S1 — A3 VOI honesty, D-23.8). Additive-optional; a READ
    # of the already-computed pre-noise joint regret (no new sampling).
    decision_evpi: Optional[float] = Field(
        None,
        ge=0,
        description="Expected value of perfect information for the WHOLE decision, in "
        "the same OUTCOME UNITS as outcome.mean (not win-probability points). "
        "EXACT FORM: decision_evpi = E[max_o U] − max_o E[U], computed on the JOINT "
        "pre-noise Common-Random-Numbers sample population (the same population as "
        "win_probability and downside.expected_regret) as the MINIMUM per-option "
        "expected regret: min_o expected_regret[o]. The identity "
        "E[max]−max_o E[o] = min_o (E[max]−E[o]) is exact, so this is the value of "
        "resolving ALL uncertainty before choosing — it captures option-switching, "
        "unlike the per-factor win-probability sensitivity in p_win_sensitivity. Zero new "
        "sampling: a read of the regret already emitted per option. >= 0 by "
        "construction (min of non-negative regrets). Present EXACTLY when the "
        "downside/regret population is present (>=1 option carries "
        "downside.expected_regret); omitted (never a JSON null) otherwise.",
    )

    # Per-factor EVPPI (S2 — A3 VOI honesty, D-23.8). Additive-optional; regression
    # EVPPI on the retained joint CRN samples (no new sampling).
    factor_evppi: Optional[List[FactorEvppiEntryV2]] = Field(
        None,
        description="Per-factor Expected Value of Partial Perfect Information (EVPPI) "
        "in the SAME OUTCOME UNITS as outcome.mean and decision_evpi: for each "
        "uncertain factor, how much better the decision could be if that factor's "
        "true value were learned before choosing. Computed by a single-loop "
        "Strong-Oakley regression EVPPI = E[max_o E[U_o|theta_i]] − max_o E[U_o] on "
        "the retained joint pre-noise CRN samples (NO nested Monte Carlo, NO new "
        "sampling); the inner conditional expectation is a polynomial regression of "
        "each option's outcome on the factor's sampled values. Fields: factor_id, "
        "evppi (clamped to [0, decision_evpi]), evppi_raw (pre-clamp audit), "
        "baseline_max_expected_utility, conditional_max_expected_utility, units "
        "('outcome'), method ('regression_evppi_v1'), regression_degree, n_samples, "
        "clamped_low (Howard non-negativity DEAD-MAN'S-SWITCH: evppi_raw is >= 0 by "
        "construction for this estimator — least-squares mean-preservation + Jensen — "
        "so clamped_low is always false; a true value would mean the estimator itself "
        "changed, not that a real negative occurred), clamped_high (per-factor EVPPI "
        "<= whole-decision EVPI theorem — capped at decision_evpi), noise_floor "
        "(permutation-null overfit floor: the MAX of K theta-shuffled null EVPPIs, i.e. "
        "a permutation test at level ~1/(K+1) so a pure-noise factor is labelled "
        "below_resolution ~1-1/(K+1) of the time), status (below_resolution when evppi "
        "<= noise_floor), correlation_active. Unlike p_win_sensitivity, this captures "
        "option-switching "
        "and is honest under correlation (the samples are joint draws, so the "
        "regression conditions on the joint). Option-controlled levers (any option "
        "intervenes on the factor — union across options) are OMITTED (absent, not "
        "zero: their uncertainty is a choice, not information to buy).",
    )

    # Per-lever EVPC (S4 — A3 value-of-control, D-23.8). Additive-optional; grid
    # do() on the retained joint CRN samples (no new sampling). Request-driven.
    # Typed (C3 house style, #104): FactorEvpcEntryV2 fails a producer typo / dropped
    # field / wrong-typed audit field loud in Pydantic, and its own model_validator
    # holds the clamp identity (not coupled into ISLResponseV2's validators).
    factor_evpc: Optional[List[FactorEvpcEntryV2]] = Field(
        None,
        description="Per-lever Expected Value of CONTROL (EVPC) in the SAME OUTCOME "
        "UNITS as outcome.mean and decision_evpi: for each control candidate the "
        "request supplied, how much better the decision could be if that factor were "
        "PULLED to its best candidate value rather than left as-is. Computed by "
        "gridding do(factor=value) over the candidate's values on the retained joint "
        "pre-noise CRN samples (NO nested Monte Carlo, NO new sampling — the same "
        "draws and the same SCM evaluator that scored the options; the intervention "
        "overrides the factor's drawn value while its correlated partners keep their "
        "joint draws): EVPC = max_x E[U|do(factor=x)] − max_a E[U_a] (the second term "
        "is the value of the best current option). COMPARATOR SEMANTICS: each "
        "do(factor=x) is a STANDALONE control action that REPLACES the option choice "
        "for that sample — it is NOT composed on top of any option's other "
        "interventions; composed control-plus-option evaluation is out of scope for "
        "grid_do_v1. GRID-APPROXIMATION SEMANTICS: the "
        "value grid is discrete, so this is a LOWER BOUND on the true (continuous) "
        "value of control — more candidate values tighten it. Fields: factor_id, evpc "
        "(clamped to >= 0: a lever you may pull to any candidate value cannot be worth "
        "less than leaving it), evpc_raw (pre-clamp audit), best_candidate_value "
        "(argmax over the grid — ALWAYS reported, even when EVPC = 0, so 'control adds "
        "nothing' names the best value tried), baseline_max_expected_utility, "
        "best_do_expected_utility, units ('outcome'), method ('grid_do_v1'), "
        "n_samples, n_candidate_values, clamped_low (raw was negative grid/finite-"
        "sample slack), correlation_active. EVPC is the value of CONTROL — the mirror "
        "of factor_evppi's value of INFORMATION (EVPPI): control replaces a factor's "
        "value with a chosen one, information reveals its true value before choosing. "
        "Because control is the point, option-controlled levers are NOT suppressed "
        "here (unlike factor_evppi). Present EXACTLY when control_candidates was "
        "supplied; emitted (like factor_evppi) under active correlation.",
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

    # Factor-value flip thresholds (ROADMAP 2.228-F3 — additive optional,
    # request-gated by include_factor_flips). Top-level on the envelope, beside
    # the other per-factor blocks (factor_sensitivity / factor_evppi /
    # factor_evpc), not nested under robustness: it is a factor-domain quantity,
    # and PLoT maps it to the top-level enrichment.flip_thresholds[] array.
    factor_flip_values: Optional[List[FactorFlipValueV2]] = Field(
        None,
        description="Per-root-factor flip thresholds: the normalised [0, 1] value at "
        "which the winning option changes, plus an attested reason when there is no "
        "such value. Deterministic and closed-form (the SCM is exactly affine in a root "
        "factor's value once epsilon noise is disabled), so no Monte Carlo error is "
        "involved and none is quoted; cross-world robustness is carried ONLY by the "
        "per-row stability band. Present exactly when include_factor_flips was "
        "requested and the phase completed within budget — on a budget trip the whole "
        "block is omitted and FACTOR_FLIPS_UNAVAILABLE is disclosed on "
        "inference_warnings.",
    )

    # Reference-option disclosure (T1-5 — additive optional)
    sensitivity_reference_option_id: Optional[str] = Field(
        None,
        description="Option ID used as the reference/baseline for edge sensitivity, "
        "factor sensitivity, and the fragile-edge classification derived from them "
        "(currently the first option in the request). Disclosure only — consumers "
        "should surface that sensitivity results are relative to this option.",
    )

    # Correlated-factors disclosure (B3-S1 — additive optional). Present only when
    # the request supplied factor_correlations (Gaussian copula active).
    correlation_model: Optional[CorrelationModelV2] = Field(
        None,
        description="Disclosure of the active factor-correlation model (Gaussian copula): "
        "method, mandatory tail-independence caveat, any PSD projection, and which "
        "independence-assuming per-factor attributions were suppressed. Absent when "
        "correlation is inactive (the independent-factor default).",
    )

    # Range-fit disclosures (ROADMAP 2.720 — additive optional). Present only
    # when the request supplied user_stated_ranges: the raw stated bounds plus
    # EITHER the interquartile-fitted distribution (2.521 Q1: quartiles on the
    # stated bounds; beta for unit_interval, normal for unbounded) OR the typed
    # refusal, whose code also rides inference_warnings at severity 'warning'.
    # S3: echo/disclosure only — compute is byte-identical, carried not applied.
    range_fit_disclosures: Optional[List[RangeFitDisclosure]] = Field(
        None,
        description="Per-range interquartile-fit disclosures for the request's "
        "user_stated_ranges: fitted parameters for display (or a typed refusal). "
        "Absent when no ranges were stated. Carried, not applied — compute is "
        "byte-identical in S3.",
    )

    # Auto-noise disclosure — mirrors V1 _metadata.auto_noise_applied so PLoT B3 can
    # surface the disclosure without reading the internal V1 metadata envelope.
    # None when the analyser cannot determine the flag (e.g. error or partial responses
    # where metadata was not built). False must serialise as false, not be dropped.
    auto_noise_applied: Optional[bool] = Field(
        None,
        description="Whether auto-scaled noise (√2 variance inflation) was applied to "
        "outcome distributions. Mirrors V1 _metadata.auto_noise_applied. "
        "None when the analyser cannot determine the flag. Since arch step 1 "
        "(2026-07-26) the heuristic is DEFAULT OFF (ENABLE_AUTO_SCALED_NOISE), so "
        "this is normally false. For WHICH metrics each population produced, read "
        "sample_population_provenance — this boolean cannot say.",
    )

    # Arch step 1 (2026-07-26): per-metric population provenance. `auto_noise_applied`
    # says noise ran; it cannot say which of the numbers in front of you it reached.
    sample_population_provenance: Optional[SamplePopulationProvenanceV2] = Field(
        None,
        description="Which sample population each served metric was computed from. "
        "One response mixes both: expected_regret / win_probability / factor_evppi / "
        "factor_evpc are computed on the PRE-noise Common-Random-Numbers population "
        "(noise draws independent per option, which breaks CRN alignment), while "
        "p10/p50/p90/mean/cvar_10/p05 are computed on the POST-noise one. None on "
        "error/blocked paths where the analyser did not report it.",
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

    @model_validator(mode="after")
    def _decision_evpi_matches_regret_population(self) -> "ISLResponseV2":
        """S1 (A3 VOI, D-23.8): guard decision_evpi BOTH directions (S2-marker style).

        decision_evpi is the honest decision-level EVPI = min_o expected_regret[o]
        on the joint pre-noise CRN population. The per-option regrets reach the wire
        as ``options[].downside.expected_regret``, and a downside block is emitted
        exactly when the option has valid MC samples — which every regret-bearing
        (hence finite-sample) option has, including the argmax-mean (== argmin-regret)
        option. So the wire downside set contains the true minimiser, and:

        (1) EMISSION (both directions, with ONE wire-derivable exemption):
            decision_evpi is present EXACTLY when the regret population is present
            (>=1 option carries downside.expected_regret) — UNLESS that population
            is INCOMPLETE, in which case honest ABSENCE is required.
            This mirrors the per-option downside emission rule and fails loud if a
            mutation ever emits the number without the population (fabrication) or
            drops it when the population exists and is complete (silent loss).
        (2) VALUE: when present, decision_evpi EQUALS the minimum per-option expected
            regret on the wire, and is non-negative. Catches a min->max / wrong-field
            mutation (it would no longer equal the wire minimum).

        ⚠ 2.477(e) — WHY THE EXEMPTION EXISTS. The paragraph above used to argue
        that the downside-bearing set always contains the true minimiser, "so on
        every 200 the population is all-finite and this min is the exact EVPI".
        That argument rested on an INCIDENTAL serializer guard (a non-finite
        outcome.mean made the whole response 500 before the number could ship),
        and 2.475 removed it: 200s with partially-populated options now exist. In
        the residual corner — an option's downside dropped by the cvar/p05 or
        regret finiteness guards while a sibling survives — ``min`` over the
        SURVIVORS can only be >= the true ``min_o``, i.e. it OVERSTATES the value
        of perfect information. The emitter therefore omits decision_evpi
        whenever any sampled option lacks a downside, and this validator must
        permit that. The exemption is DERIVED FROM THE WIRE, not asserted: an
        incomplete population is visible as an option with
        ``percentiles_source == 'samples'`` and no ``downside`` — so absence is
        still rejected when the population is complete, and the silent-loss
        direction keeps biting. (Options with no samples at all are not part of
        the population and do not license absence: they carry no honest regret
        and never did.)
        """
        regrets = [
            o.downside.expected_regret for o in (self.options or []) if o.downside is not None
        ]
        population_incomplete = any(
            o.downside is None and o.outcome.percentiles_source == "samples"
            for o in (self.options or [])
        )
        evpi = self.decision_evpi
        present = evpi is not None
        if present and not regrets:
            raise ValueError(
                "decision_evpi emission-iff violated: decision_evpi is present but the "
                "joint regret population is absent. It is min_o expected_regret[o] "
                "and cannot be computed without >=1 option carrying "
                "downside.expected_regret."
            )
        if not present and regrets and not population_incomplete:
            raise ValueError(
                "decision_evpi emission-iff violated: decision_evpi is absent while the "
                "joint regret population is present AND COMPLETE (every sampled "
                "option carries downside.expected_regret). Absence is permitted only "
                "when some option with percentiles_source == 'samples' lacks a "
                "downside, which would make min over the survivors an OVERSTATEMENT "
                "(2.477(e)); dropping it otherwise is silent loss."
            )
        if evpi is not None:
            min_regret = min(regrets)
            tol = 1e-9 + 1e-9 * abs(min_regret)
            if evpi < -tol or abs(evpi - min_regret) > tol:
                raise ValueError(
                    "decision_evpi must equal the MINIMUM per-option expected regret "
                    f"(min_o downside.expected_regret = {min_regret!r}) and be "
                    f">= 0; got {evpi!r}. It is E[max]−max E on the "
                    "joint pre-noise population = min_o expected_regret[o]."
                )
        return self

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
