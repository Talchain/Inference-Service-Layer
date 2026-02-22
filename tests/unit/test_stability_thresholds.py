"""
Unit tests for configurable stability thresholds (3C-thresholds).

Tests:
1. CV exactly at high/moderate boundary (0.1) → "high" (inclusive)
2. CV just above boundary (0.1001) → "moderate"
3. CV exactly at moderate/low boundary (0.3) → "moderate" (inclusive)
4. CV just above (0.3001) → "low"
5. Override via env var → boundary shifts
6. Threshold metadata in ISL response
"""

import os

import pytest

from src.config.stability_thresholds import (
    StabilityThresholds,
    classify_attribution_stability,
    load_stability_thresholds,
    STABILITY_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_thresholds() -> StabilityThresholds:
    """Return default thresholds (no env overrides)."""
    return StabilityThresholds(
        high_moderate_boundary=0.1,
        moderate_low_boundary=0.3,
        negligible_elasticity_floor=1e-6,
        provisional=True,
        version="v1.0-operational-defaults",
    )


# ==========================================================================
# Test 1: CV exactly at high/moderate boundary → "high" (inclusive)
# ==========================================================================


class TestHighModerateBoundary:
    def test_cv_exactly_at_boundary_is_high(self):
        """CV = 0.1 exactly → 'high' (boundary is inclusive)."""
        thresholds = _default_thresholds()
        # elasticity=1.0, std=0.1 → CV = 0.1/1.0 = 0.1
        result = classify_attribution_stability(1.0, 0.1, thresholds)
        assert result == "high"

    def test_cv_below_boundary_is_high(self):
        """CV = 0.05 → 'high'."""
        thresholds = _default_thresholds()
        result = classify_attribution_stability(1.0, 0.05, thresholds)
        assert result == "high"


# ==========================================================================
# Test 2: CV just above high/moderate boundary → "moderate"
# ==========================================================================


class TestJustAboveHighModerate:
    def test_cv_just_above_boundary_is_moderate(self):
        """CV = 0.1001 → 'moderate'."""
        thresholds = _default_thresholds()
        # elasticity=1.0, std=0.1001 → CV = 0.1001
        result = classify_attribution_stability(1.0, 0.1001, thresholds)
        assert result == "moderate"


# ==========================================================================
# Test 3: CV exactly at moderate/low boundary → "moderate" (inclusive)
# ==========================================================================


class TestModerateLowBoundary:
    def test_cv_exactly_at_boundary_is_moderate(self):
        """CV = 0.3 exactly → 'moderate' (boundary is inclusive)."""
        thresholds = _default_thresholds()
        # elasticity=1.0, std=0.3 → CV = 0.3
        result = classify_attribution_stability(1.0, 0.3, thresholds)
        assert result == "moderate"


# ==========================================================================
# Test 4: CV just above moderate/low boundary → "low"
# ==========================================================================


class TestJustAboveModerateLow:
    def test_cv_just_above_boundary_is_low(self):
        """CV = 0.3001 → 'low'."""
        thresholds = _default_thresholds()
        result = classify_attribution_stability(1.0, 0.3001, thresholds)
        assert result == "low"


# ==========================================================================
# Test 5: Override via env var → boundary shifts
# ==========================================================================


class TestEnvVarOverride:
    def test_high_moderate_override_shifts_boundary(self, monkeypatch):
        """STABILITY_CV_HIGH_MODERATE=0.2 shifts the high/moderate boundary."""
        monkeypatch.setenv("STABILITY_CV_HIGH_MODERATE", "0.2")
        thresholds = load_stability_thresholds()

        assert thresholds.high_moderate_boundary == 0.2

        # CV=0.15 is now "high" (was "moderate" with default 0.1)
        result = classify_attribution_stability(1.0, 0.15, thresholds)
        assert result == "high"

        # CV=0.25 is now "moderate" (was already "moderate")
        result = classify_attribution_stability(1.0, 0.25, thresholds)
        assert result == "moderate"

    def test_moderate_low_override_shifts_boundary(self, monkeypatch):
        """STABILITY_CV_MODERATE_LOW=0.5 shifts the moderate/low boundary."""
        monkeypatch.setenv("STABILITY_CV_MODERATE_LOW", "0.5")
        thresholds = load_stability_thresholds()

        assert thresholds.moderate_low_boundary == 0.5

        # CV=0.4 is now "moderate" (was "low" with default 0.3)
        result = classify_attribution_stability(1.0, 0.4, thresholds)
        assert result == "moderate"

    def test_both_overrides(self, monkeypatch):
        """Both env vars set at once."""
        monkeypatch.setenv("STABILITY_CV_HIGH_MODERATE", "0.2")
        monkeypatch.setenv("STABILITY_CV_MODERATE_LOW", "0.5")
        thresholds = load_stability_thresholds()

        assert thresholds.high_moderate_boundary == 0.2
        assert thresholds.moderate_low_boundary == 0.5
        assert thresholds.version == "v1.0-operational-defaults"
        assert thresholds.provisional is True

    def test_invalid_env_var_uses_default(self, monkeypatch):
        """Non-numeric env var falls back to default."""
        monkeypatch.setenv("STABILITY_CV_HIGH_MODERATE", "not_a_number")
        thresholds = load_stability_thresholds()

        assert thresholds.high_moderate_boundary == 0.1  # default


# ==========================================================================
# Test 6: Threshold metadata in ISL response
# ==========================================================================


class TestThresholdMetadataInResponse:
    def test_response_includes_stability_thresholds(self):
        """RobustnessResponseV2 includes stability_thresholds when bootstrap active."""
        from src.models.robustness_v2 import (
            EdgeV2,
            GraphV2,
            InterventionOption,
            NodeV2,
            ObservedState,
            ParameterUncertainty,
            RobustnessRequestV2,
            StrengthDistribution,
        )
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        graph = GraphV2(
            nodes=[
                NodeV2(id="factor1", kind="factor", label="Factor 1",
                       observed_state=ObservedState(value=0.5)),
                NodeV2(id="goal", kind="outcome", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "factor1", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.8, std=0.1),
                ),
            ],
        )
        request = RobustnessRequestV2(
            request_id="threshold-meta-test",
            graph=graph,
            options=[InterventionOption(id="opt1", label="Opt 1", interventions={})],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="factor1", distribution="normal", std=0.1),
            ],
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Verify stability_thresholds is present
        assert response.stability_thresholds is not None
        st = response.stability_thresholds
        assert st.high_moderate_boundary == STABILITY_THRESHOLDS.high_moderate_boundary
        assert st.moderate_low_boundary == STABILITY_THRESHOLDS.moderate_low_boundary
        assert st.version == STABILITY_THRESHOLDS.version
        assert st.provisional is True

    def test_response_serializes_stability_thresholds(self):
        """stability_thresholds appears in JSON serialization."""
        from src.models.robustness_v2 import (
            EdgeV2,
            GraphV2,
            InterventionOption,
            NodeV2,
            ObservedState,
            ParameterUncertainty,
            RobustnessRequestV2,
            StrengthDistribution,
        )
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        graph = GraphV2(
            nodes=[
                NodeV2(id="factor1", kind="factor", label="Factor 1",
                       observed_state=ObservedState(value=0.5)),
                NodeV2(id="goal", kind="outcome", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "factor1", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.8, std=0.1),
                ),
            ],
        )
        request = RobustnessRequestV2(
            request_id="threshold-serial-test",
            graph=graph,
            options=[InterventionOption(id="opt1", label="Opt 1", interventions={})],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="factor1", distribution="normal", std=0.1),
            ],
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        data = response.model_dump(by_alias=True)

        assert "stability_thresholds" in data
        st = data["stability_thresholds"]
        assert st["high_moderate_boundary"] == STABILITY_THRESHOLDS.high_moderate_boundary
        assert st["moderate_low_boundary"] == STABILITY_THRESHOLDS.moderate_low_boundary
        assert st["version"] == "v1.0-operational-defaults"
        assert st["provisional"] is True

    def test_no_bootstrap_no_stability_thresholds(self):
        """When no parameter_uncertainties → no bootstrap → no stability_thresholds."""
        from src.models.robustness_v2 import (
            EdgeV2,
            GraphV2,
            InterventionOption,
            NodeV2,
            ObservedState,
            RobustnessRequestV2,
            StrengthDistribution,
        )
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        graph = GraphV2(
            nodes=[
                NodeV2(id="factor1", kind="factor", label="Factor 1",
                       observed_state=ObservedState(value=0.5)),
                NodeV2(id="goal", kind="outcome", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "factor1", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.8, std=0.1),
                ),
            ],
        )
        request = RobustnessRequestV2(
            request_id="no-bootstrap-test",
            graph=graph,
            options=[InterventionOption(id="opt1", label="Opt 1", interventions={})],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            # No parameter_uncertainties → no bootstrap
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # No bootstrap → stability_thresholds should be None
        assert response.stability_thresholds is None


# ==========================================================================
# Supplementary: classify_attribution_stability edge cases
# ==========================================================================


class TestClassifyEdgeCases:
    def test_negligible_elasticity(self):
        """Near-zero elasticity → 'negligible' regardless of std."""
        thresholds = _default_thresholds()
        result = classify_attribution_stability(0.0, 1.0, thresholds)
        assert result == "negligible"

    def test_negligible_very_small(self):
        """Elasticity just below floor → 'negligible'."""
        thresholds = _default_thresholds()
        result = classify_attribution_stability(5e-7, 0.1, thresholds)
        assert result == "negligible"

    def test_negative_elasticity(self):
        """Negative elasticity uses |elasticity| for CV computation."""
        thresholds = _default_thresholds()
        # elasticity=-1.0, std=0.05 → CV=0.05 → "high"
        result = classify_attribution_stability(-1.0, 0.05, thresholds)
        assert result == "high"

    def test_zero_std_is_high(self):
        """Zero std → CV=0 → "high"."""
        thresholds = _default_thresholds()
        result = classify_attribution_stability(1.0, 0.0, thresholds)
        assert result == "high"

    def test_module_singleton_matches_defaults(self):
        """Module-level STABILITY_THRESHOLDS has expected default values."""
        assert STABILITY_THRESHOLDS.high_moderate_boundary == 0.1
        assert STABILITY_THRESHOLDS.moderate_low_boundary == 0.3
        assert STABILITY_THRESHOLDS.negligible_elasticity_floor == 1e-6
        assert STABILITY_THRESHOLDS.provisional is True
        assert STABILITY_THRESHOLDS.version == "v1.0-operational-defaults"
