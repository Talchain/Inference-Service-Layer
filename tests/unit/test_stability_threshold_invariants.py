"""
I.1 — Stability threshold golden-fixture invariant tests.

CV boundaries (0.1, 0.3) and factor confidence blending (70/30 category/CV)
are provisional defaults.  These tests lock the current values so that any
change produces a loud, deliberate failure rather than a silent drift.

If a value changes intentionally, update the golden fixture in this file
and note the reason in the commit message.
"""

import pytest

from src.config.stability_thresholds import (
    CONFIDENCE_CATEGORY_WEIGHT,
    CONFIDENCE_CV_WEIGHT,
    STABILITY_THRESHOLDS,
    load_stability_thresholds,
)

_CHANGE_MSG = (
    "Stability threshold changed — this requires deliberate review. "
    "Update the golden fixture in test_stability_threshold_invariants.py "
    "if the change is intentional."
)


class TestStabilityThresholdInvariants:
    """Golden-fixture tests that lock provisional threshold values."""

    def test_high_stability_cv_boundary(self) -> None:
        """CV boundary for 'high stability' is 0.1."""
        assert STABILITY_THRESHOLDS.high_moderate_boundary == 0.1, _CHANGE_MSG

    def test_low_stability_cv_boundary(self) -> None:
        """CV boundary for 'low stability' is 0.3."""
        assert STABILITY_THRESHOLDS.moderate_low_boundary == 0.3, _CHANGE_MSG

    def test_negligible_elasticity_floor(self) -> None:
        """Negligible elasticity floor is 1e-6."""
        assert STABILITY_THRESHOLDS.negligible_elasticity_floor == 1e-6, _CHANGE_MSG

    def test_category_cv_blending_weight_category(self) -> None:
        """Category blending weight is 0.7 (70%)."""
        assert CONFIDENCE_CATEGORY_WEIGHT == 0.7, _CHANGE_MSG

    def test_category_cv_blending_weight_cv(self) -> None:
        """CV blending weight is 0.3 (30%)."""
        assert CONFIDENCE_CV_WEIGHT == 0.3, _CHANGE_MSG

    def test_blending_weights_sum_to_one(self) -> None:
        """Category + CV weights must sum to 1.0."""
        assert CONFIDENCE_CATEGORY_WEIGHT + CONFIDENCE_CV_WEIGHT == pytest.approx(1.0), (
            "Blending weights must sum to 1.0 — "
            f"got {CONFIDENCE_CATEGORY_WEIGHT} + {CONFIDENCE_CV_WEIGHT}"
        )

    def test_provisional_flag_is_true(self) -> None:
        """Provisional flag must be True until scientific validation."""
        assert STABILITY_THRESHOLDS.provisional is True, _CHANGE_MSG

    def test_version_string(self) -> None:
        """Version string is locked."""
        assert STABILITY_THRESHOLDS.version == "v1.0-operational-defaults", _CHANGE_MSG

    def test_load_function_returns_same_defaults(self) -> None:
        """load_stability_thresholds() without env vars matches the singleton."""
        fresh = load_stability_thresholds()
        assert fresh.high_moderate_boundary == STABILITY_THRESHOLDS.high_moderate_boundary
        assert fresh.moderate_low_boundary == STABILITY_THRESHOLDS.moderate_low_boundary
        assert fresh.provisional == STABILITY_THRESHOLDS.provisional
