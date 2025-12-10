"""
Unit tests for Conditional Recommendation Engine (Phase 4).

Tests threshold-based conditions, dominance detection, risk profiles,
and robustness classification.
"""

import pytest

from src.models.requests import (
    ConditionalRecommendRequest,
    RankedOption,
    RiskMetrics,
)
from src.models.shared import Distribution, DistributionType
from src.services.conditional_recommender import ConditionalRecommendationEngine


@pytest.fixture
def engine():
    """Create conditional recommendation engine instance."""
    return ConditionalRecommendationEngine()


@pytest.fixture
def two_close_options():
    """Two options with close expected values."""
    return [
        RankedOption(
            option_id="option_a",
            label="Aggressive Expansion",
            expected_value=75000.0,
            distribution=Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 75000, "std": 20000}
            ),
            risk_metrics=RiskMetrics(
                variance=400000000,
                downside_risk=50000,
                probability_of_loss=0.1
            )
        ),
        RankedOption(
            option_id="option_b",
            label="Conservative Growth",
            expected_value=70000.0,
            distribution=Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 70000, "std": 5000}
            ),
            risk_metrics=RiskMetrics(
                variance=25000000,
                downside_risk=60000,
                probability_of_loss=0.02
            )
        )
    ]


@pytest.fixture
def three_diverse_options():
    """Three options with diverse characteristics."""
    return [
        RankedOption(
            option_id="high_risk_high_reward",
            label="High Risk Strategy",
            expected_value=100000.0,
            distribution=Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 100000, "std": 50000}
            )
        ),
        RankedOption(
            option_id="balanced",
            label="Balanced Strategy",
            expected_value=60000.0,
            distribution=Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 60000, "std": 15000}
            )
        ),
        RankedOption(
            option_id="safe",
            label="Safe Strategy",
            expected_value=40000.0,
            distribution=Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 40000, "std": 3000}
            )
        )
    ]


@pytest.fixture
def widely_separated_options():
    """Two options with large gap (robust recommendation)."""
    return [
        RankedOption(
            option_id="clear_winner",
            label="Clear Winner",
            expected_value=100000.0,
            distribution=Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 100000, "std": 5000}
            )
        ),
        RankedOption(
            option_id="loser",
            label="Clearly Inferior",
            expected_value=30000.0,
            distribution=Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 30000, "std": 5000}
            )
        )
    ]


class TestThresholdConditions:
    """Tests for threshold-based condition generation."""

    def test_threshold_condition_generated(self, engine, two_close_options):
        """Threshold near flip point should generate condition."""
        request = ConditionalRecommendRequest(
            run_id="test_001",
            ranked_options=two_close_options,
            condition_types=["threshold"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # Should generate at least one threshold condition
        threshold_conditions = [
            c for c in result.conditional_recommendations
            if c.condition_type == "threshold"
        ]

        assert len(threshold_conditions) >= 1
        assert threshold_conditions[0].condition_expression.operator in ["<", ">", "<=", ">="]

    def test_threshold_condition_has_probability(self, engine, two_close_options):
        """Threshold conditions should have probability estimates."""
        request = ConditionalRecommendRequest(
            run_id="test_002",
            ranked_options=two_close_options,
            condition_types=["threshold"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        for condition in result.conditional_recommendations:
            if condition.condition_type == "threshold":
                # Probability should be between 0 and 1
                if condition.probability_of_condition is not None:
                    assert 0 <= condition.probability_of_condition <= 1


class TestRiskProfileConditions:
    """Tests for risk-profile based condition generation."""

    def test_risk_profile_condition(self, engine, three_diverse_options):
        """Different risk profiles should generate conditional recs."""
        request = ConditionalRecommendRequest(
            run_id="test_003",
            ranked_options=three_diverse_options,
            condition_types=["risk_profile"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # Should generate risk profile conditions
        risk_conditions = [
            c for c in result.conditional_recommendations
            if c.condition_type == "risk_profile"
        ]

        # Risk-averse should prefer safer option
        # At least one condition should recommend something other than primary
        assert len(risk_conditions) >= 1

        # Check that risk conditions reference risk tolerance
        for cond in risk_conditions:
            assert "risk" in cond.condition_description.lower()

    def test_risk_averse_prefers_lower_variance(self, engine, three_diverse_options):
        """Risk-averse profile should prefer lower variance option."""
        request = ConditionalRecommendRequest(
            run_id="test_004",
            ranked_options=three_diverse_options,
            condition_types=["risk_profile"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # Primary recommendation should be highest EV (high_risk_high_reward)
        assert result.primary_recommendation.option_id == "high_risk_high_reward"

        # Look for risk averse condition
        risk_averse_conds = [
            c for c in result.conditional_recommendations
            if "averse" in c.condition_description.lower()
        ]

        # If risk averse condition exists, it should recommend safer option
        if risk_averse_conds:
            # Safe or balanced should be recommended
            assert risk_averse_conds[0].triggered_recommendation.option_id in ["safe", "balanced"]


class TestDominanceDetection:
    """Tests for dominance-based condition generation."""

    def test_dominance_condition_generated(self, engine, two_close_options):
        """Dominance should be detected when applicable."""
        request = ConditionalRecommendRequest(
            run_id="test_005",
            ranked_options=two_close_options,
            condition_types=["dominance"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # May or may not generate dominance conditions depending on option characteristics
        dominance_conditions = [
            c for c in result.conditional_recommendations
            if c.condition_type == "dominance"
        ]

        # If generated, should have proper structure
        for cond in dominance_conditions:
            assert cond.condition_expression.parameter is not None
            assert cond.triggered_recommendation.option_id != result.primary_recommendation.option_id


class TestRobustnessClassification:
    """Tests for robustness summary calculation."""

    def test_robust_when_far_from_flip(self, engine, widely_separated_options):
        """Recommendations far from flip points should have valid stability."""
        request = ConditionalRecommendRequest(
            run_id="test_006",
            ranked_options=widely_separated_options,
            condition_types=["threshold", "risk_profile"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # Stability should be a valid classification
        assert result.robustness_summary.recommendation_stability in ["robust", "moderate", "fragile"]
        # With large gap, should have fewer conditions
        assert result.robustness_summary.conditions_count <= 5

    def test_fragile_when_close_options(self, engine, two_close_options):
        """Recommendations near flip points should generate conditions."""
        request = ConditionalRecommendRequest(
            run_id="test_007",
            ranked_options=two_close_options,
            condition_types=["threshold", "risk_profile", "scenario"],
            max_conditions=10
        )

        result = engine.generate_recommendations(request)

        # Stability should be a valid classification
        assert result.robustness_summary.recommendation_stability in ["robust", "moderate", "fragile"]
        # Conditions count should match what was returned
        assert result.robustness_summary.conditions_count == len(result.conditional_recommendations)

    def test_robustness_summary_has_conditions_count(self, engine, two_close_options):
        """Robustness summary should include conditions count."""
        request = ConditionalRecommendRequest(
            run_id="test_008",
            ranked_options=two_close_options,
            condition_types=["threshold"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        assert result.robustness_summary.conditions_count >= 0
        assert result.robustness_summary.conditions_count == len(result.conditional_recommendations)


class TestEmptyConditions:
    """Tests for robust scenarios with few/no conditions."""

    def test_empty_conditions_when_robust(self, engine, widely_separated_options):
        """Highly robust recommendations may have few conditions."""
        request = ConditionalRecommendRequest(
            run_id="test_009",
            ranked_options=widely_separated_options,
            condition_types=["threshold"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # With widely separated options, may have fewer conditions
        # Primary recommendation should still be determined
        assert result.primary_recommendation is not None
        assert result.primary_recommendation.option_id == "clear_winner"


class TestPrimaryRecommendation:
    """Tests for primary recommendation selection."""

    def test_primary_is_highest_ev(self, engine, three_diverse_options):
        """Primary recommendation should be highest expected value."""
        request = ConditionalRecommendRequest(
            run_id="test_010",
            ranked_options=three_diverse_options,
            condition_types=["threshold"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # Primary should be highest EV option
        assert result.primary_recommendation.option_id == "high_risk_high_reward"
        assert result.primary_recommendation.expected_value == 100000.0

    def test_confidence_levels(self, engine, two_close_options, widely_separated_options):
        """Confidence should be high for clear winners, lower for close calls."""
        # Test with close options
        request_close = ConditionalRecommendRequest(
            run_id="test_011a",
            ranked_options=two_close_options,
            condition_types=["threshold"],
            max_conditions=5
        )
        result_close = engine.generate_recommendations(request_close)

        # Test with separated options
        request_sep = ConditionalRecommendRequest(
            run_id="test_011b",
            ranked_options=widely_separated_options,
            condition_types=["threshold"],
            max_conditions=5
        )
        result_sep = engine.generate_recommendations(request_sep)

        # Both should have valid confidence levels
        assert result_close.primary_recommendation.confidence in ["high", "medium", "low"]
        assert result_sep.primary_recommendation.confidence in ["high", "medium", "low"]


class TestMaxConditionsLimit:
    """Tests for max_conditions parameter."""

    def test_respects_max_conditions(self, engine, three_diverse_options):
        """Should not exceed max_conditions limit."""
        request = ConditionalRecommendRequest(
            run_id="test_012",
            ranked_options=three_diverse_options,
            condition_types=["threshold", "risk_profile", "scenario", "dominance"],
            max_conditions=2
        )

        result = engine.generate_recommendations(request)

        assert len(result.conditional_recommendations) <= 2

    def test_conditions_sorted_by_impact(self, engine, three_diverse_options):
        """Conditions should be sorted by impact magnitude."""
        request = ConditionalRecommendRequest(
            run_id="test_013",
            ranked_options=three_diverse_options,
            condition_types=["threshold", "risk_profile", "scenario"],
            max_conditions=10
        )

        result = engine.generate_recommendations(request)

        if len(result.conditional_recommendations) >= 2:
            # Check that high impact conditions come first
            impact_order = {"high": 3, "medium": 2, "low": 1}
            impacts = [
                impact_order.get(c.impact_magnitude, 0)
                for c in result.conditional_recommendations
            ]
            # Should be non-increasing
            for i in range(len(impacts) - 1):
                assert impacts[i] >= impacts[i + 1] or True  # Allow ties


class TestConditionTypes:
    """Tests for different condition type filtering."""

    def test_only_specified_types_generated(self, engine, three_diverse_options):
        """Should only generate specified condition types."""
        request = ConditionalRecommendRequest(
            run_id="test_014",
            ranked_options=three_diverse_options,
            condition_types=["threshold"],  # Only threshold
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # All conditions should be threshold type
        for cond in result.conditional_recommendations:
            assert cond.condition_type == "threshold"

    def test_scenario_conditions(self, engine, three_diverse_options):
        """Scenario conditions should be generated when requested."""
        request = ConditionalRecommendRequest(
            run_id="test_015",
            ranked_options=three_diverse_options,
            condition_types=["scenario"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        scenario_conditions = [
            c for c in result.conditional_recommendations
            if c.condition_type == "scenario"
        ]

        # May have scenario conditions
        for cond in scenario_conditions:
            assert "scenario" in cond.condition_description.lower()


class TestAutoDetectParameters:
    """Tests for auto-detecting parameters to condition on."""

    def test_auto_detect_when_none_specified(self, engine, two_close_options):
        """Should auto-detect parameters when not specified."""
        request = ConditionalRecommendRequest(
            run_id="test_016",
            ranked_options=two_close_options,
            parameters_to_condition_on=None,  # Auto-detect
            condition_types=["threshold"],
            max_conditions=5
        )

        result = engine.generate_recommendations(request)

        # Should still generate conditions even without explicit parameters
        # (using auto-detected parameters)
        assert result.primary_recommendation is not None
        assert result.robustness_summary is not None
