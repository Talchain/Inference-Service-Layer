"""
Configurable stability thresholds for factor sensitivity bootstrap (3C).

Operational defaults for the CV-based attribution_stability classification.
These are provisional pending scientific review and can be overridden via
environment variables for testing and recalibration.

Environment variable overrides:
    STABILITY_CV_HIGH_MODERATE  — CV boundary between "high" and "moderate" (default: 0.1)
    STABILITY_CV_MODERATE_LOW   — CV boundary between "moderate" and "low" (default: 0.3)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Literal defaults of the STANDARD attribution_stability CV boundaries. The
# STABILITY_CV_* env overrides (below) move these — a testing/recalibration lever
# that feeds a DIFFERENT `attribution_stability` and hence a DIFFERENT emitted
# `confidence`. They are part of the confidence method: they join the fingerprint
# guard and drive the `+env-override` disclosure suffix (see F-1).
DEFAULT_CV_HIGH_MODERATE_BOUNDARY = 0.1
DEFAULT_CV_MODERATE_LOW_BOUNDARY = 0.3


@dataclass(frozen=True)
class StabilityThresholds:
    """CV-based thresholds for attribution_stability classification.

    Provisional — pending scientific review.
    """

    high_moderate_boundary: float  # CV ≤ this → "high"
    moderate_low_boundary: float  # CV ≤ this → "moderate"; above → "low"
    negligible_elasticity_floor: float  # |elasticity| below this → "negligible"
    provisional: bool
    version: str

    @property
    def is_overridden(self) -> bool:
        """True when either CV boundary differs from its literal default — a
        STABILITY_CV_* env override moved it, yielding a NON-standard confidence
        method (F-1).

        This is a VALUE comparison, NOT env presence: setting an override to
        exactly the default boundary is not a recalibration, so it reads False.
        Deliberately distinct from load_stability_thresholds()'s presence-based
        `overrides_active` warning, which fires whenever the env var is SET at all.
        """
        return (
            self.high_moderate_boundary != DEFAULT_CV_HIGH_MODERATE_BOUNDARY
            or self.moderate_low_boundary != DEFAULT_CV_MODERATE_LOW_BOUNDARY
        )


def _read_env_float(name: str, default: float) -> float:
    """Read a float from an environment variable, returning default if absent or invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "stability_threshold_env_invalid",
            extra={"env_var": name, "raw_value": raw, "using_default": default},
        )
        return default


def load_stability_thresholds() -> StabilityThresholds:
    """Load stability thresholds, applying environment variable overrides if set.

    Logs a warning when any override is active.
    """
    high_mod = _read_env_float("STABILITY_CV_HIGH_MODERATE", DEFAULT_CV_HIGH_MODERATE_BOUNDARY)
    mod_low = _read_env_float("STABILITY_CV_MODERATE_LOW", DEFAULT_CV_MODERATE_LOW_BOUNDARY)

    overrides_active = (
        os.environ.get("STABILITY_CV_HIGH_MODERATE") is not None
        or os.environ.get("STABILITY_CV_MODERATE_LOW") is not None
    )

    if overrides_active:
        logger.warning(
            "stability_threshold_overrides_active",
            extra={
                "high_moderate_boundary": high_mod,
                "moderate_low_boundary": mod_low,
            },
        )

    return StabilityThresholds(
        high_moderate_boundary=high_mod,
        moderate_low_boundary=mod_low,
        negligible_elasticity_floor=1e-6,
        provisional=True,
        version="v1.0-operational-defaults",
    )


def classify_attribution_stability(
    primary_elasticity: float,
    elasticity_std: float,
    thresholds: StabilityThresholds,
) -> str:
    """Classify attribution stability from CV of bootstrap elasticities.

    Args:
        primary_elasticity: Deterministic (reported) elasticity value.
        elasticity_std: Standard deviation of bootstrap elasticities.
        thresholds: Threshold configuration.

    Returns:
        One of "high", "moderate", "low", or "negligible".
    """
    if abs(primary_elasticity) < thresholds.negligible_elasticity_floor:
        return "negligible"
    cv = elasticity_std / abs(primary_elasticity)
    if cv <= thresholds.high_moderate_boundary:
        return "high"
    if cv <= thresholds.moderate_low_boundary:
        return "moderate"
    return "low"


# Module-level singleton — loaded once at import time.
# Tests can monkeypatch or re-call load_stability_thresholds() with env vars.
STABILITY_THRESHOLDS = load_stability_thresholds()


# --- Factor Confidence Computation (Track S) ---

# Provisional stability-to-confidence mapping.
# Pending scientific calibration from pilot data (Neil gate 1).
# These values are NOT research-validated — they are operational defaults.
# provisional: true
#
# NOTE: These values (0.9/0.6/0.3/0.1) are *factor-sensitivity confidence scores*
# derived from bootstrap stability classification — NOT robustness-level thresholds.
# Robustness-level classification uses ROBUST_THRESHOLD (0.7 win probability) in
# robustness_analyzer_v2.py combined with overall confidence to produce
# high/moderate/low/very_low levels.
STABILITY_CONFIDENCE_MAP = {
    "high": 0.9,
    "moderate": 0.6,
    "low": 0.3,
    "negligible": 0.1,
}
STABILITY_CONFIDENCE_DEFAULT = 0.5  # When attribution_stability is unrecognised

# Version tag for the factor-confidence method (S2 disclosure marker).
# Emitted on the wire as FactorSensitivityV2.confidence_provenance.method_version.
# CONTRACT (Neil gate 1, no silent reweighting): the confidence mapping is
# PROVISIONAL and NOT research-validated. ANY change to the constants that
# define the mapping — STABILITY_CONFIDENCE_MAP, STABILITY_CONFIDENCE_DEFAULT,
# CONFIDENCE_CATEGORY_WEIGHT, CONFIDENCE_CV_WEIGHT, CONFIDENCE_CV_ELASTICITY_FLOOR
# — MUST bump this version so downstream can detect the recalibration. The
# fingerprint guard in tests/unit/test_confidence_provenance.py pins the pair
# (constants fingerprint <-> this version) and fails loud if the constants move
# while the version stays put. `calibrated` stays False until a validated
# calibration exists.
CONFIDENCE_METHOD_VERSION = "stability-cv-blend-v1"


def get_confidence_method_version() -> str:
    """method_version for the bootstrap-CV-blend confidence, disclosing env recalibration (F-1).

    The ``STABILITY_CV_*`` env overrides move the CV boundaries that decide
    ``attribution_stability`` — the INPUT to the confidence mapping — so an active
    override yields a DIFFERENT emitted ``confidence`` computed with a NON-standard
    boundary, while ``CONFIDENCE_METHOD_VERSION`` alone would stay put. When either
    live boundary differs from its literal default, the wire discloses that the
    standard method was NOT used by appending ``+env-override``.

    Single derivation (no hand-maintained mirror): the live thresholds are read via
    ``load_stability_thresholds()`` and compared against the default literals
    ``DEFAULT_CV_HIGH_MODERATE_BOUNDARY`` / ``DEFAULT_CV_MODERATE_LOW_BOUNDARY``.
    """
    thresholds = load_stability_thresholds()
    if thresholds.is_overridden:
        return CONFIDENCE_METHOD_VERSION + "+env-override"
    return CONFIDENCE_METHOD_VERSION


# Blend weights: category signal vs CV signal
CONFIDENCE_CATEGORY_WEIGHT = 0.7
CONFIDENCE_CV_WEIGHT = 0.3

# Minimum absolute elasticity for CV computation to be meaningful
CONFIDENCE_CV_ELASTICITY_FLOOR = 1e-6


def compute_factor_confidence(
    attribution_stability: str | None,
    elasticity: float,
    elasticity_std: float | None,
) -> float | None:
    """Compute per-factor confidence from bootstrap stability metrics.

    NOTE: This confidence value is a stability-derived heuristic, not a
    calibrated probability. It reflects how stable the sensitivity ranking
    is across bootstrap resamples. Values are provisional pending scientific
    calibration from pilot data (Neil gate 1).

    Confidence is derived from:
    1. Categorical stability signal (attribution_stability: high/moderate/low/negligible)
    2. Coefficient of variation refinement (when available)

    The categorical signal (70% weight) reflects rank consistency across bootstrap
    resamples. The CV signal (30% weight) reflects magnitude consistency using a
    smooth mapping ``cv_score = 1 / (1 + CV)`` that is monotonically decreasing,
    always in (0, 1], and never collapses to zero even for large CV values.

    Args:
        attribution_stability: Categorical stability from bootstrap
            ("high", "moderate", "low", "negligible", or None)
        elasticity: Primary (deterministic) elasticity value
        elasticity_std: Standard deviation of bootstrap elasticities (optional)

    Returns:
        Confidence in [0, 1], or None if no bootstrap data available.

    Examples:
        >>> compute_factor_confidence("high", 0.5, None)
        0.9
        >>> compute_factor_confidence("moderate", 0.5, 0.1)  # With CV refinement
        0.6694  # 0.7 * 0.6 + 0.3 * (1 / (1 + 0.2)) = 0.42 + 0.25
        >>> compute_factor_confidence(None, 0.5, None)
        None
    """
    if attribution_stability is None:
        return None

    # Step 1: Category score
    category_score = STABILITY_CONFIDENCE_MAP.get(
        attribution_stability, STABILITY_CONFIDENCE_DEFAULT
    )

    # Step 2: CV refinement (when available and elasticity is non-negligible)
    # Uses smooth mapping: cv_score = 1 / (1 + CV)
    # - Monotonically decreasing: low CV (stable) → high score
    # - Always in (0, 1]: never collapses to zero for large CV
    # - CV=0 → 1.0, CV=0.5 → 0.67, CV=1.0 → 0.5, CV=2.0 → 0.33
    cv_score = None
    if elasticity_std is not None and abs(elasticity) > CONFIDENCE_CV_ELASTICITY_FLOOR:
        cv = elasticity_std / abs(elasticity)
        cv_score = 1.0 / (1.0 + cv)

    # Step 3: Blend
    if cv_score is not None:
        confidence = CONFIDENCE_CATEGORY_WEIGHT * category_score + CONFIDENCE_CV_WEIGHT * cv_score
    else:
        confidence = category_score

    # Step 4: Clamp and round
    return round(max(0.0, min(1.0, confidence)), 4)


# Graph-structural confidence parameters (fallback when bootstrap unavailable)
GRAPH_STRUCTURAL_CONFIDENCE_SCALE = 0.5
GRAPH_STRUCTURAL_CONFIDENCE_FLOOR = 0.25

# Version tag for the graph-structural FALLBACK confidence method (F-2). This is a
# DIFFERENT method from the bootstrap-CV blend — confidence = FLOOR + SCALE * influence
# (compute_graph_structural_confidence) — so it MUST NOT stamp the bootstrap
# method_version. The branch is unreachable on today's live path (bootstrap always
# runs whenever factors are emitted) but is kept for future code paths; stamping it
# honestly now means the wrong version can never go live silently. Its two constants
# (GRAPH_STRUCTURAL_CONFIDENCE_FLOOR / GRAPH_STRUCTURAL_CONFIDENCE_SCALE) have their
# OWN fingerprint entry in tests/unit/test_confidence_provenance.py.
GRAPH_STRUCTURAL_METHOD_VERSION = "graph-structural-v1"


def compute_graph_structural_confidence(
    influence_score: float | None,
) -> float:
    """Fallback confidence from graph structure when bootstrap metrics are unavailable.

    Scales influence_score [0, 1] to a confidence range of [0.25, 0.75].
    This is capped below bootstrap-derived confidence to signal lower certainty.

    Args:
        influence_score: Structural influence from causal path strengths (0-1).

    Returns:
        Confidence in [0.25, 0.75].
    """
    score = influence_score if influence_score is not None else 0.0
    return round(
        GRAPH_STRUCTURAL_CONFIDENCE_FLOOR + score * GRAPH_STRUCTURAL_CONFIDENCE_SCALE,
        4,
    )
