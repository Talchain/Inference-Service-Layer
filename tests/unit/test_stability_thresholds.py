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
    compute_factor_confidence,
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
                NodeV2(
                    id="factor1",
                    kind="factor",
                    label="Factor 1",
                    observed_state=ObservedState(value=0.5),
                ),
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
                NodeV2(
                    id="factor1",
                    kind="factor",
                    label="Factor 1",
                    observed_state=ObservedState(value=0.5),
                ),
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
                NodeV2(
                    id="factor1",
                    kind="factor",
                    label="Factor 1",
                    observed_state=ObservedState(value=0.5),
                ),
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


# ==========================================================================
# Track S: Factor Confidence Computation
# ==========================================================================


class TestComputeFactorConfidence:
    """Tests for compute_factor_confidence() (Track S)."""

    def test_high_stability_no_cv_data(self):
        """attribution_stability='high', no CV data → 0.9"""
        result = compute_factor_confidence("high", 0.5, None)
        assert result == 0.9

    def test_moderate_stability_no_cv_data(self):
        """attribution_stability='moderate', no CV data → 0.6"""
        result = compute_factor_confidence("moderate", 0.5, None)
        assert result == 0.6

    def test_low_stability_no_cv_data(self):
        """attribution_stability='low', no CV data → 0.3"""
        result = compute_factor_confidence("low", 0.5, None)
        assert result == 0.3

    def test_negligible_stability_no_cv_data(self):
        """attribution_stability='negligible', no CV data → 0.1"""
        result = compute_factor_confidence("negligible", 0.05, None)
        assert result == 0.1

    def test_none_stability_returns_none(self):
        """attribution_stability=None → None (no bootstrap data)"""
        result = compute_factor_confidence(None, 0.5, None)
        assert result is None

    def test_high_stability_with_cv_refinement(self):
        """attribution_stability='high' with CV data → blended value"""
        # elasticity=0.5, elasticity_std=0.1 → CV=0.2, cv_score=1/(1+0.2)=0.8333
        # confidence = 0.7 * 0.9 + 0.3 * 0.8333 = 0.63 + 0.25 = 0.88
        result = compute_factor_confidence("high", 0.5, 0.1)
        assert result == 0.88

    def test_moderate_stability_with_cv_refinement(self):
        """attribution_stability='moderate' with CV data → blended value"""
        # elasticity=0.5, elasticity_std=0.15 → CV=0.3, cv_score=1/(1+0.3)=0.7692
        # confidence = 0.7 * 0.6 + 0.3 * 0.7692 = 0.42 + 0.2308 = 0.6508
        result = compute_factor_confidence("moderate", 0.5, 0.15)
        assert result == 0.6508

    def test_near_zero_elasticity_skips_cv(self):
        """Near-zero elasticity → CV computation skipped, category score only"""
        # elasticity=1e-7 < CONFIDENCE_CV_ELASTICITY_FLOOR, elasticity_std=0.1
        # CV would be huge, but we skip it → use category score only
        result = compute_factor_confidence("high", 1e-7, 0.1)
        assert result == 0.9  # Category score only, no CV refinement

    def test_zero_elasticity_skips_cv(self):
        """Exactly zero elasticity → CV computation skipped"""
        result = compute_factor_confidence("low", 0.0, 0.05)
        assert result == 0.3  # Category score only

    def test_negative_elasticity_uses_abs(self):
        """Negative elasticity → |elasticity| used for CV computation"""
        # elasticity=-0.5, elasticity_std=0.1 → CV=0.1/0.5=0.2, cv_score=1/(1+0.2)=0.8333
        # confidence = 0.7 * 0.9 + 0.3 * 0.8333 = 0.88
        result = compute_factor_confidence("high", -0.5, 0.1)
        assert result == 0.88

    def test_high_cv_smooth_mapping(self):
        """Very high CV (>1) → cv_score is small but nonzero (smooth mapping)"""
        # elasticity=0.1, elasticity_std=0.5 → CV=5.0, cv_score=1/(1+5)=0.1667
        # confidence = 0.7 * 0.6 + 0.3 * 0.1667 = 0.42 + 0.05 = 0.47
        result = compute_factor_confidence("moderate", 0.1, 0.5)
        assert result == 0.47

    def test_zero_cv_gives_max_cv_score(self):
        """Zero CV → cv_score = 1/(1+0) = 1.0 (maximum possible)"""
        # elasticity=0.5, elasticity_std=0.0 → CV=0, cv_score=1.0
        # confidence = 0.7 * 0.9 + 0.3 * 1.0 = 0.63 + 0.3 = 0.93
        result = compute_factor_confidence("high", 0.5, 0.0)
        assert result == 0.93

    def test_confidence_bounds(self):
        """All computed confidence values are in [0, 1]"""
        test_cases = [
            ("high", 0.5, None),
            ("moderate", 0.5, 0.1),
            ("low", 0.1, 0.5),
            ("negligible", 0.001, 0.0001),
        ]
        for stability, elasticity, std in test_cases:
            result = compute_factor_confidence(stability, elasticity, std)
            assert result is not None
            assert (
                0.0 <= result <= 1.0
            ), f"Out of bounds: {result} for {stability}, {elasticity}, {std}"

    def test_confidence_rounded_to_4dp(self):
        """Confidence values are rounded to 4 decimal places"""
        # Test case that would produce many decimal places
        # elasticity=0.333, elasticity_std=0.111 → CV≈0.3333, cv_score=1/(1+0.3333)=0.75
        # confidence ≈ 0.7 * 0.6 + 0.3 * 0.75 = 0.42 + 0.225 = 0.645
        result = compute_factor_confidence("moderate", 0.333, 0.111)
        assert result == 0.645
        # Verify it's actually rounded (check number of decimal places)
        assert len(str(result).split(".")[-1]) <= 4

    def test_unrecognized_stability_uses_default(self):
        """Unrecognised attribution_stability value → default 0.5"""
        result = compute_factor_confidence("unknown_category", 0.5, None)
        assert result == 0.5

    def test_determinism(self):
        """Same inputs → same outputs"""
        result1 = compute_factor_confidence("moderate", 0.5, 0.1)
        result2 = compute_factor_confidence("moderate", 0.5, 0.1)
        assert result1 == result2


class TestNearTieScenario:
    """Near-tie scenario: two factors with similar sensitivity.

    When two factors have near-identical elasticity, bootstrap rank flips are
    expected, producing lower attribution_stability for both. Confidence should
    reflect this — likely "moderate" or "low" — telling the user "we can't
    confidently distinguish which factor matters more."
    """

    def test_near_tie_factors_produce_lower_confidence(self):
        """Two near-tied factors → both get moderate/low stability → lower confidence."""
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

        # Two factors with nearly identical edge strengths → near-tied sensitivity
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="factor_a",
                    kind="factor",
                    label="Factor A",
                    observed_state=ObservedState(value=0.5),
                ),
                NodeV2(
                    id="factor_b",
                    kind="factor",
                    label="Factor B",
                    observed_state=ObservedState(value=0.5),
                ),
                NodeV2(id="intervention", kind="factor", label="Intervention"),
                NodeV2(id="goal", kind="outcome", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "factor_a", "to": "goal"},
                    exists_probability=0.9,
                    strength=StrengthDistribution(mean=0.5, std=0.15),  # High variance
                ),
                EdgeV2(
                    **{"from": "factor_b", "to": "goal"},
                    exists_probability=0.9,
                    strength=StrengthDistribution(mean=0.5, std=0.15),  # Same — near-tie
                ),
                EdgeV2(
                    **{"from": "intervention", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.3, std=0.05),
                ),
            ],
        )
        request = RobustnessRequestV2(
            request_id="near-tie-confidence-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Opt 1", interventions={"intervention": 0.5}),
            ],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="factor_a", distribution="normal", std=0.1),
                ParameterUncertainty(node_id="factor_b", distribution="normal", std=0.1),
            ],
        )
        analyzer = RobustnessAnalyzerV2()
        analyzer._n_bootstrap_override = 10  # Pin for determinism (match TestBootstrapStability)
        response = analyzer.analyze(request)

        assert response.factor_sensitivity is not None
        assert len(response.factor_sensitivity) == 2

        # Both factors should have bootstrap stability computed
        for fs in response.factor_sensitivity:
            assert (
                fs.attribution_stability is not None
            ), f"Factor {fs.node_id} missing attribution_stability"

        # Near-tied factors with high edge variance should experience rank flips,
        # producing moderate or low stability (not consistently "high")
        stabilities = {fs.node_id: fs.attribution_stability for fs in response.factor_sensitivity}

        # At least one should NOT be "high" — rank flips are expected
        assert not all(
            s == "high" for s in stabilities.values()
        ), f"Both near-tied factors got 'high' stability — rank flips expected: {stabilities}"

        # Now compute confidence for both and verify it reflects the instability
        confidences = {}
        for fs in response.factor_sensitivity:
            conf = compute_factor_confidence(
                fs.attribution_stability, fs.elasticity, fs.elasticity_std
            )
            assert conf is not None
            assert 0.0 <= conf <= 1.0
            confidences[fs.node_id] = conf

        # Neither factor should have high confidence (0.9) since rank-unstable
        for node_id, conf in confidences.items():
            assert (
                conf < 0.9
            ), f"Factor {node_id} has confidence {conf} despite near-tie instability"


# ==========================================================================
# Track S: CV > 1 regression test (smooth mapping)
# ==========================================================================


class TestCVGreaterThanOne:
    """Regression tests: cv_score must never collapse to zero for large CV.

    The old formula (1 - cv) clamped to [0, 1] produced cv_score=0 for any
    CV >= 1, making the 30% CV weight dead weight. The smooth mapping
    1/(1+cv) ensures cv_score is always > 0.
    """

    def test_cv_greater_than_one_produces_nonzero_score(self):
        """CV > 1 → cv_score > 0 (the regression this fixes)."""
        # elasticity=0.54, elasticity_std=1.17 → CV ≈ 2.17
        # cv_score = 1/(1+2.17) ≈ 0.316 (was 0 with old formula)
        result = compute_factor_confidence("moderate", 0.54, 1.17)
        assert result is not None
        # With old formula: 0.7 * 0.6 + 0.3 * 0 = 0.42
        # With new formula: 0.7 * 0.6 + 0.3 * (1/(1+2.1667)) ≈ 0.42 + 0.095 ≈ 0.515
        assert (
            result > 0.42
        ), f"CV > 1 should produce higher confidence than category-only ({result} <= 0.42)"

    def test_cv_exactly_one_produces_half_score(self):
        """CV = 1.0 → cv_score = 1/(1+1) = 0.5"""
        # elasticity=0.5, elasticity_std=0.5 → CV=1.0
        # confidence = 0.7 * 0.6 + 0.3 * 0.5 = 0.42 + 0.15 = 0.57
        result = compute_factor_confidence("moderate", 0.5, 0.5)
        assert result == 0.57

    def test_very_large_cv_still_positive(self):
        """CV = 10.0 → cv_score = 1/(1+10) ≈ 0.0909, confidence still > category-only min."""
        # elasticity=0.1, elasticity_std=1.0 → CV=10.0
        # cv_score = 1/11 ≈ 0.0909
        # confidence = 0.7 * 0.3 + 0.3 * 0.0909 = 0.21 + 0.0273 ≈ 0.2373
        result = compute_factor_confidence("low", 0.1, 1.0)
        assert result is not None
        assert result > 0.21, "CV=10 should still contribute positively to confidence"

    def test_smooth_mapping_monotonically_decreasing(self):
        """Higher CV → lower confidence (all else equal)."""
        # All "moderate" stability, same elasticity, increasing std
        results = []
        for std in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            conf = compute_factor_confidence("moderate", 0.5, std)
            results.append(conf)

        for i in range(len(results) - 1):
            assert (
                results[i] >= results[i + 1]
            ), f"Confidence should decrease with higher CV: {results}"


# ==========================================================================
# Track S: Attribution stability category guard (Task 2)
# ==========================================================================


ALLOWED_STABILITY_CATEGORIES = {"high", "moderate", "low", "negligible"}


class TestAttributionStabilityAllowedCategories:
    """Guard: ISL must only emit allowed attribution_stability category strings.

    If ISL emits a different string, PLoT's mapping and Track A's Evidence
    Priority card silently break. This test catches regressions.
    """

    def test_attribution_stability_only_allowed_categories(self):
        """Every attribution_stability in a real analysis is one of the four allowed strings or None."""
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
                NodeV2(
                    id="factor1",
                    kind="factor",
                    label="Factor 1",
                    observed_state=ObservedState(value=0.5),
                ),
                NodeV2(
                    id="factor2",
                    kind="factor",
                    label="Factor 2",
                    observed_state=ObservedState(value=0.3),
                ),
                NodeV2(id="intervention", kind="factor", label="Intervention"),
                NodeV2(id="goal", kind="outcome", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "factor1", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.8, std=0.1),
                ),
                EdgeV2(
                    **{"from": "factor2", "to": "goal"},
                    exists_probability=0.9,
                    strength=StrengthDistribution(mean=0.3, std=0.2),
                ),
                EdgeV2(
                    **{"from": "intervention", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.05),
                ),
            ],
        )
        request = RobustnessRequestV2(
            request_id="stability-category-guard-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Opt 1", interventions={"intervention": 0.5}),
            ],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="factor1", distribution="normal", std=0.1),
                ParameterUncertainty(node_id="factor2", distribution="normal", std=0.1),
            ],
        )
        analyzer = RobustnessAnalyzerV2()
        analyzer._n_bootstrap_override = 10
        response = analyzer.analyze(request)

        assert response.factor_sensitivity is not None
        assert len(response.factor_sensitivity) >= 1

        for fs in response.factor_sensitivity:
            assert (
                fs.attribution_stability is None
                or fs.attribution_stability in ALLOWED_STABILITY_CATEGORIES
            ), (
                f"Factor {fs.node_id} has disallowed attribution_stability: "
                f"'{fs.attribution_stability}'. Allowed: {ALLOWED_STABILITY_CATEGORIES}"
            )

    def test_classify_only_returns_allowed_categories(self):
        """classify_attribution_stability() only returns allowed category strings."""
        thresholds = _default_thresholds()
        # Sweep CV from 0 to 10 in steps — every result must be in allowed set
        for std_mult in [0.0, 0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0, 5.0, 10.0]:
            result = classify_attribution_stability(1.0, std_mult, thresholds)
            assert (
                result in ALLOWED_STABILITY_CATEGORIES
            ), f"classify_attribution_stability returned '{result}' for std={std_mult}"

    def test_negligible_elasticity_returns_allowed_category(self):
        """Near-zero elasticity → 'negligible' (an allowed category)."""
        thresholds = _default_thresholds()
        result = classify_attribution_stability(0.0, 1.0, thresholds)
        assert result in ALLOWED_STABILITY_CATEGORIES


class TestUnknownStabilityFallback:
    """Verify compute_factor_confidence() handles unknown strings gracefully."""

    def test_unknown_string_returns_default_confidence(self):
        """Unrecognised attribution_stability → falls back to 0.5 (STABILITY_CONFIDENCE_DEFAULT)."""
        result = compute_factor_confidence("unknown_category", 0.5, None)
        assert result == 0.5

    def test_empty_string_returns_default_confidence(self):
        """Empty string attribution_stability → falls back to 0.5."""
        result = compute_factor_confidence("", 0.5, None)
        assert result == 0.5

    def test_unknown_string_with_cv_data(self):
        """Unrecognised stability with CV data → blends default 0.5 with CV score."""
        # CV=0.2, cv_score=1/(1+0.2)=0.8333
        # confidence = 0.7 * 0.5 + 0.3 * 0.8333 = 0.35 + 0.25 = 0.6
        result = compute_factor_confidence("typo_category", 0.5, 0.1)
        assert result == 0.6
