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
    high_mod = _read_env_float("STABILITY_CV_HIGH_MODERATE", 0.1)
    mod_low = _read_env_float("STABILITY_CV_MODERATE_LOW", 0.3)

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
