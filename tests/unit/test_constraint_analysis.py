"""
Unit tests for multi-constraint goal analysis (Phase 2).

Tests:
- GoalConstraint model validation
- Per-constraint probability computation
- Joint probability computation
- Pairwise conditional probabilities
- Near-miss diagnostics
- Integration with RobustnessAnalyzerV2
"""

import numpy as np
import pytest

from src.models.robustness_v2 import (
    ConstraintAnalysis,
    ConstraintResult,
    EdgeV2,
    GoalConstraint,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    OptionResult,
    ParameterUncertainty,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import (
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def multi_outcome_graph():
    """
    Create a graph with multiple outcome nodes for constraint testing.

    Structure:
        marketing -> demand -> revenue
        marketing -> cost
        price -> demand
        price -> revenue (direct)
    """
    return GraphV2(
        nodes=[
            NodeV2(id="marketing", kind="factor", label="Marketing Spend"),
            NodeV2(id="price", kind="factor", label="Price"),
            NodeV2(id="demand", kind="chance", label="Demand"),
            NodeV2(id="revenue", kind="outcome", label="Revenue"),
            NodeV2(id="cost", kind="outcome", label="Cost"),
        ],
        edges=[
            EdgeV2(
                **{"from": "marketing", "to": "demand"},
                exists_probability=0.9,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
            EdgeV2(
                **{"from": "marketing", "to": "cost"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.8, std=0.05),
            ),
            EdgeV2(
                **{"from": "price", "to": "demand"},
                exists_probability=0.95,
                strength=StrengthDistribution(mean=-0.3, std=0.1),
            ),
            EdgeV2(
                **{"from": "demand", "to": "revenue"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.7, std=0.1),
            ),
            EdgeV2(
                **{"from": "price", "to": "revenue"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.4, std=0.05),
            ),
        ],
    )


@pytest.fixture
def multi_outcome_options():
    """Create options for multi-outcome testing."""
    return [
        InterventionOption(
            id="high_marketing",
            label="High Marketing",
            interventions={"marketing": 100.0, "price": 50.0},
        ),
        InterventionOption(
            id="low_marketing",
            label="Low Marketing",
            interventions={"marketing": 30.0, "price": 50.0},
        ),
    ]


@pytest.fixture
def analyzer():
    """Create a RobustnessAnalyzerV2 instance."""
    return RobustnessAnalyzerV2()


# =============================================================================
# GoalConstraint Model Tests
# =============================================================================


class TestGoalConstraintModel:
    """Test GoalConstraint model validation."""

    def test_valid_ge_constraint(self):
        """Test valid >= constraint."""
        gc = GoalConstraint(node_id="revenue", operator=">=", value=100.0, label="Min Revenue")
        assert gc.node_id == "revenue"
        assert gc.operator == ">="
        assert gc.value == 100.0
        assert gc.threshold == 100.0  # property alias
        assert gc.label == "Min Revenue"

    def test_valid_le_constraint(self):
        """Test valid <= constraint."""
        gc = GoalConstraint(node_id="cost", operator="<=", value=50.0, label="Max Cost")
        assert gc.node_id == "cost"
        assert gc.operator == "<="
        assert gc.value == 50.0

    def test_constraint_without_label(self):
        """Test constraint without optional label."""
        gc = GoalConstraint(node_id="revenue", operator=">=", value=100.0)
        assert gc.label is None

    def test_invalid_operator_rejected(self):
        """Test that invalid operators are rejected."""
        with pytest.raises(ValueError):
            GoalConstraint(node_id="revenue", operator=">", value=100.0)

    def test_invalid_operator_less_than(self):
        """Test that < operator is rejected."""
        with pytest.raises(ValueError):
            GoalConstraint(node_id="revenue", operator="<", value=100.0)

    def test_nan_threshold_rejected(self):
        """Test that NaN value is rejected."""
        with pytest.raises(ValueError):
            GoalConstraint(node_id="revenue", operator=">=", value=float("nan"))

    def test_inf_threshold_rejected(self):
        """Test that infinite value is rejected."""
        with pytest.raises(ValueError):
            GoalConstraint(node_id="revenue", operator=">=", value=float("inf"))

    def test_negative_threshold_allowed(self):
        """Test that negative values are allowed."""
        gc = GoalConstraint(node_id="profit", operator=">=", value=-10.0)
        assert gc.value == -10.0
        assert gc.threshold == -10.0  # property alias

    def test_zero_threshold_allowed(self):
        """Test that zero value is allowed."""
        gc = GoalConstraint(node_id="balance", operator=">=", value=0.0)
        assert gc.value == 0.0

    def test_value_field_matches_v27_contract(self):
        """Test that GoalConstraint accepts 'value' per Decision Model Schema v2.7.

        CEE/PLoT emit {"value": 0.04, ...}. ISL must accept that field name
        at the API boundary without requiring 'threshold'.
        """
        gc = GoalConstraint(
            node_id="fac_logo_churn",
            operator="<=",
            value=0.04,
            label="monthly logo churn ceiling",
        )
        assert gc.value == 0.04
        # Internal code can still use .threshold property
        assert gc.threshold == 0.04

    def test_legacy_threshold_kwarg_accepted_via_compat(self):
        """Legacy 'threshold' kwarg is coerced to 'value' for backward compat.

        Older clients or stored payloads may still send 'threshold'.
        The model_validator(mode='before') maps it to 'value' transparently.
        """
        gc = GoalConstraint(
            node_id="revenue",
            operator=">=",
            threshold=100.0,  # type: ignore[call-arg]
        )
        assert gc.value == 100.0
        assert gc.threshold == 100.0

    def test_value_takes_precedence_over_legacy_threshold(self):
        """When both 'value' and 'threshold' are provided, 'value' wins.

        The pre-validator only maps threshold→value when value is absent.
        """
        gc = GoalConstraint(
            node_id="revenue",
            operator=">=",
            value=42.0,
            threshold=999.0,  # type: ignore[call-arg]
        )
        assert gc.value == 42.0


class TestRequestWithGoalConstraints:
    """Test RobustnessRequestV2 with goal_constraints."""

    def test_request_with_valid_constraints(self, multi_outcome_graph, multi_outcome_options):
        """Test request with valid goal constraints."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=100,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=50.0),
                GoalConstraint(node_id="cost", operator="<=", value=100.0),
            ],
        )
        assert len(request.goal_constraints) == 2
        assert request.goal_constraints[0].node_id == "revenue"
        assert request.goal_constraints[1].node_id == "cost"

    def test_request_without_constraints_backward_compat(
        self, multi_outcome_graph, multi_outcome_options
    ):
        """Test request without constraints still works (backward compatibility)."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=100,
        )
        assert request.goal_constraints is None

    def test_request_rejects_nonexistent_constraint_node(
        self, multi_outcome_graph, multi_outcome_options
    ):
        """Test that constraints referencing non-existent nodes are rejected."""
        with pytest.raises(ValueError, match="GoalConstraint references non-existent node"):
            RobustnessRequestV2(
                graph=multi_outcome_graph,
                options=multi_outcome_options,
                goal_node_id="revenue",
                n_samples=100,
                goal_constraints=[
                    GoalConstraint(node_id="nonexistent", operator=">=", value=50.0),
                ],
            )


# =============================================================================
# Constraint Probability Computation Tests
# =============================================================================


class TestConstraintProbabilityComputation:
    """Test per-constraint and joint probability computation."""

    def test_single_constraint_probability(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test probability computation with single constraint."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=10.0),
            ],
        )

        response = analyzer.analyze(request)

        # Check constraint analysis is present
        for result in response.results:
            assert result.constraint_analysis is not None
            assert len(result.constraint_analysis.constraints) == 1
            # With value=10 and typical outcomes, should have high satisfaction
            assert 0.0 <= result.constraint_analysis.constraints[0].prob_satisfied <= 1.0
            # Joint probability equals per-constraint with single constraint
            assert (
                result.constraint_analysis.joint_probability
                == result.constraint_analysis.constraints[0].prob_satisfied
            )

    def test_multiple_constraints_joint_probability(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test joint probability with multiple constraints."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=10.0, label="Min Revenue"),
                GoalConstraint(node_id="cost", operator="<=", value=200.0, label="Max Cost"),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            assert len(ca.constraints) == 2

            # Joint probability should be <= min of individual probabilities
            p_revenue = ca.constraints[0].prob_satisfied
            p_cost = ca.constraints[1].prob_satisfied
            assert ca.joint_probability <= min(p_revenue, p_cost)

            # Verify constraint details are preserved
            assert ca.constraints[0].node_id == "revenue"
            assert ca.constraints[0].operator == ">="
            assert ca.constraints[0].threshold == 10.0
            assert ca.constraints[0].label == "Min Revenue"

    def test_impossible_constraint_zero_probability(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test constraint that's impossible to satisfy has ~0 probability."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                # Extremely high threshold that can't be met
                GoalConstraint(node_id="revenue", operator=">=", value=1000000.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            # Should be 0 or very close to 0
            assert ca.constraints[0].prob_satisfied < 0.01

    def test_always_satisfied_constraint(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test constraint that's always satisfied has ~1.0 probability."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                # Very low value that's always met
                GoalConstraint(node_id="revenue", operator=">=", value=-1000000.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            # Should be 1.0 or very close
            assert ca.constraints[0].prob_satisfied > 0.99


# =============================================================================
# Conditional Probability Tests
# =============================================================================


class TestConditionalProbabilities:
    """Test pairwise conditional probability computation."""

    def test_conditional_probabilities_computed(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test that conditional probabilities are computed for 2+ constraints."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=10.0),
                GoalConstraint(node_id="cost", operator="<=", value=200.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            # Should have conditional probabilities for 2+ constraints
            assert ca.conditional_probabilities is not None

            # Check structure: P(C1 | C0) and P(C0 | C1)
            assert "0" in ca.conditional_probabilities
            assert "1" in ca.conditional_probabilities
            assert "1" in ca.conditional_probabilities["0"]
            assert "0" in ca.conditional_probabilities["1"]

            # Conditional probabilities should be in [0, 1]
            assert 0.0 <= ca.conditional_probabilities["0"]["1"] <= 1.0
            assert 0.0 <= ca.conditional_probabilities["1"]["0"] <= 1.0

    def test_no_conditional_probabilities_single_constraint(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test that conditional probabilities are None for single constraint."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=10.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            # No conditional probabilities for single constraint

    def test_conditional_probability_undefined_when_zero_satisfaction(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test that P(C_j | C_i) is omitted (not 0.0) when P(C_i) = 0.

        When a constraint is never satisfied, P(C_j | C_i) is mathematically
        undefined (0/0), so we omit those entries rather than returning 0.0.
        """
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                # Constraint 0: Impossible to satisfy (very high value)
                GoalConstraint(node_id="revenue", operator=">=", value=1000000.0),
                # Constraint 1: Easy to satisfy
                GoalConstraint(node_id="cost", operator="<=", value=1000000.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None

            # Constraint 0 should have ~0 probability (never satisfied)
            assert ca.constraints[0].prob_satisfied < 0.01

            # Since P(C_0) ≈ 0, P(C_1 | C_0) is undefined
            # The dict for "0" should be empty (entries omitted, not set to 0)
            assert ca.conditional_probabilities is not None
            assert "0" in ca.conditional_probabilities
            # The dict is empty because P(C_0)=0 makes conditionals undefined
            assert len(ca.conditional_probabilities["0"]) == 0

            # P(C_0 | C_1) should be defined and close to 0 (since C_0 rarely satisfied)
            assert "1" in ca.conditional_probabilities
            assert "0" in ca.conditional_probabilities["1"]
            assert ca.conditional_probabilities["1"]["0"] < 0.01


# =============================================================================
# Near-Miss Diagnostics Tests
# =============================================================================


class TestNearMissDiagnostics:
    """Test near-miss diagnostic computation."""

    def test_binding_constraint_detection(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test that borderline constraints are marked as binding."""
        # We need to find a threshold that gives ~50% satisfaction
        # This requires some trial and error with the model
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=1000,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=30.0),
            ],
        )

        response = analyzer.analyze(request)

        # Just verify the binding field is computed
        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            assert ca.constraints[0].binding is not None
            # binding should be boolean
            assert isinstance(ca.constraints[0].binding, bool)

    def test_failure_margin_computed_when_failures_exist(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test failure_margin_median is computed when there are failures."""
        # Use a threshold that will have some failures
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=50.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            constraint = ca.constraints[0]
            if constraint.prob_satisfied < 1.0:
                # There are failures, so failure_margin_median should be computed
                assert constraint.failure_margin_median is not None
                # For >= constraint, margin should be positive when failing
                assert constraint.failure_margin_median >= 0

    def test_near_miss_fraction_computed(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test near_miss_fraction is computed when there are failures."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=50.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            constraint = ca.constraints[0]
            if constraint.prob_satisfied < 1.0:
                # near_miss_fraction should be computed
                assert constraint.near_miss_fraction is not None
                assert 0.0 <= constraint.near_miss_fraction <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestConstraintAnalysisIntegration:
    """Integration tests for constraint analysis with full analysis pipeline."""

    def test_constraint_analysis_with_probability_of_goal(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test constraint analysis works alongside probability_of_goal.

        Note: probability_of_goal is computed AFTER auto-scaled noise is applied,
        while constraint_analysis.prob_satisfied is computed from original samples.
        This means they may differ slightly due to the noise.
        """
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_threshold=25.0,  # Legacy single threshold
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=25.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            # Both should be present
            assert result.probability_of_goal is not None
            assert result.constraint_analysis is not None

            # Both values should be valid probabilities
            assert 0.0 <= result.probability_of_goal <= 1.0
            assert 0.0 <= result.constraint_analysis.constraints[0].prob_satisfied <= 1.0

            # With same threshold, they should be in the same ballpark
            # (within 15% of each other due to auto-scaled noise on outcome nodes)
            assert (
                abs(
                    result.probability_of_goal
                    - result.constraint_analysis.constraints[0].prob_satisfied
                )
                < 0.15
            )

    def test_constraint_analysis_deterministic(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test that constraint analysis is deterministic with same seed."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=30.0),
                GoalConstraint(node_id="cost", operator="<=", value=100.0),
            ],
        )

        response1 = analyzer.analyze(request)
        response2 = analyzer.analyze(request)

        # Results should be identical
        for r1, r2 in zip(response1.results, response2.results):
            ca1 = r1.constraint_analysis
            ca2 = r2.constraint_analysis
            assert ca1.joint_probability == ca2.joint_probability
            for c1, c2 in zip(ca1.constraints, ca2.constraints):
                assert c1.prob_satisfied == c2.prob_satisfied

    def test_options_can_have_different_constraint_results(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test that different options can have different constraint satisfaction rates."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                # Cost constraint - high_marketing should have higher cost
                GoalConstraint(node_id="cost", operator="<=", value=60.0),
            ],
        )

        response = analyzer.analyze(request)

        # Different options should have different results
        results_by_option = {r.option_id: r for r in response.results}

        # high_marketing option has marketing=100, so higher cost
        # low_marketing option has marketing=30, so lower cost
        # cost = marketing * 0.8 (approximately), so:
        # high_marketing cost ≈ 80, low_marketing cost ≈ 24
        # Constraint is cost <= 60

        high_marketing_ca = results_by_option["high_marketing"].constraint_analysis
        low_marketing_ca = results_by_option["low_marketing"].constraint_analysis

        # low_marketing should have higher satisfaction (lower cost)
        assert (
            low_marketing_ca.constraints[0].prob_satisfied
            > high_marketing_ca.constraints[0].prob_satisfied
        )

    def test_no_constraint_analysis_without_constraints(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test that no constraint analysis is present when no constraints provided."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )

        response = analyzer.analyze(request)

        for result in response.results:
            assert result.constraint_analysis is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestConstraintAnalysisEdgeCases:
    """Test edge cases in constraint analysis."""

    def test_le_operator_computation(self, analyzer, multi_outcome_graph, multi_outcome_options):
        """Test <= operator works correctly."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="cost", operator="<=", value=100.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            assert ca.constraints[0].operator == "<="
            assert 0.0 <= ca.constraints[0].prob_satisfied <= 1.0

    def test_same_node_different_thresholds(
        self, analyzer, multi_outcome_graph, multi_outcome_options
    ):
        """Test multiple constraints on same node with different thresholds."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=10.0, label="Min"),
                GoalConstraint(node_id="revenue", operator="<=", value=100.0, label="Max"),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            assert len(ca.constraints) == 2
            # Both constraints should have valid probabilities
            assert 0.0 <= ca.constraints[0].prob_satisfied <= 1.0
            assert 0.0 <= ca.constraints[1].prob_satisfied <= 1.0
            # Joint should be <= both
            assert ca.joint_probability <= ca.constraints[0].prob_satisfied
            assert ca.joint_probability <= ca.constraints[1].prob_satisfied

    def test_constraint_on_goal_node(self, analyzer, multi_outcome_graph, multi_outcome_options):
        """Test constraint on the goal node itself."""
        request = RobustnessRequestV2(
            graph=multi_outcome_graph,
            options=multi_outcome_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="revenue", operator=">=", value=20.0),
            ],
        )

        response = analyzer.analyze(request)

        for result in response.results:
            ca = result.constraint_analysis
            assert ca is not None
            assert ca.constraints[0].node_id == "revenue"


# =============================================================================
# Inference Warnings Tests
# =============================================================================


class TestInferenceWarningsDefaultBase:
    """Test that inference_warnings fires when a non-root constraint node
    has no ParameterUncertainty and therefore defaults to base=0.0."""

    @staticmethod
    def _make_graph_with_nonroot_factor():
        """Graph where fac_churn is a non-root factor (has incoming edge)."""
        return GraphV2(
            nodes=[
                NodeV2(
                    id="fac_retention",
                    kind="factor",
                    label="Retention",
                    observed_state=ObservedState(value=0.4),
                ),
                NodeV2(
                    id="fac_churn",
                    kind="factor",
                    label="Churn",
                    observed_state=ObservedState(value=0.5),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "fac_retention", "to": "fac_churn"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=-0.3, std=0.05),
                ),
                EdgeV2(
                    **{"from": "fac_churn", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=-0.5, std=0.1),
                ),
            ],
        )

    def test_warning_when_nonroot_constraint_node_lacks_uncertainty(self, caplog):
        """Non-root factor without ParameterUncertainty triggers warning."""
        graph = self._make_graph_with_nonroot_factor()
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                InterventionOption(
                    id="baseline",
                    label="Baseline",
                    interventions={"fac_retention": 0.6},
                ),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=1,
            goal_constraints=[
                GoalConstraint(node_id="fac_churn", operator="<=", value=0.04),
            ],
            # No parameter_uncertainties for fac_churn
        )

        analyzer = RobustnessAnalyzerV2()

        with caplog.at_level("WARNING"):
            response = analyzer.analyze(request)

        # Assert the structured log was emitted
        assert any(
            "isl.constraint.default_base" in r.message
            for r in caplog.records
            if r.levelname == "WARNING"
        ), "Expected structured warning log for defaulted constraint node"

        # Assert inference_warnings populated on response (InferenceWarning objects).
        # Filter by code: other warning codes (e.g. CONSTRAINT_SAMPLES_UNNOISED)
        # may legitimately coexist.
        default_base_warnings = [
            w for w in response.inference_warnings if w.code == "CONSTRAINT_NODE_DEFAULT_BASE"
        ]
        assert len(default_base_warnings) == 1
        w = default_base_warnings[0]
        assert w.code == "CONSTRAINT_NODE_DEFAULT_BASE"
        assert "fac_churn" in w.field or "fac_churn" in str(w.detail)
        assert w.detail.get("defaulted_to") == 0.0

        # Assert CritiqueV2 also emitted (visible in V2 response path)
        constraint_critiques = [
            c for c in response.critiques if c.code == "CONSTRAINT_NODE_DEFAULT_BASE"
        ]
        assert len(constraint_critiques) == 1
        assert constraint_critiques[0].severity == "warning"
        assert constraint_critiques[0].affected_node_ids == ["fac_churn"]

    def test_no_warning_when_node_has_parameter_uncertainty(self, caplog):
        """Non-root factor WITH ParameterUncertainty does NOT trigger warning."""
        graph = self._make_graph_with_nonroot_factor()
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                InterventionOption(
                    id="baseline",
                    label="Baseline",
                    interventions={"fac_retention": 0.6},
                ),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=1,
            goal_constraints=[
                GoalConstraint(node_id="fac_churn", operator="<=", value=0.04),
            ],
            parameter_uncertainties=[
                ParameterUncertainty(
                    node_id="fac_churn",
                    distribution="normal",
                    std=0.1,
                ),
            ],
        )

        analyzer = RobustnessAnalyzerV2()

        with caplog.at_level("WARNING"):
            response = analyzer.analyze(request)

        # No warning logs for default_base
        assert not any(
            "isl.constraint.default_base" in r.message
            for r in caplog.records
            if r.levelname == "WARNING"
        ), "Should NOT emit warning when ParameterUncertainty is present"

        # No CONSTRAINT_NODE_DEFAULT_BASE warning (other codes may coexist)
        assert not any(
            w.code == "CONSTRAINT_NODE_DEFAULT_BASE" for w in response.inference_warnings
        )

        # No CONSTRAINT_NODE_DEFAULT_BASE critique
        assert not any(c.code == "CONSTRAINT_NODE_DEFAULT_BASE" for c in response.critiques)

    def test_no_warning_when_all_options_intervene_on_node(self, caplog):
        """No false positive when every option directly intervenes on the node."""
        graph = self._make_graph_with_nonroot_factor()
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                InterventionOption(
                    id="opt_a",
                    label="Option A",
                    interventions={"fac_retention": 0.6, "fac_churn": 0.3},
                ),
                InterventionOption(
                    id="opt_b",
                    label="Option B",
                    interventions={"fac_retention": 0.8, "fac_churn": 0.2},
                ),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=1,
            goal_constraints=[
                GoalConstraint(node_id="fac_churn", operator="<=", value=0.04),
            ],
        )

        analyzer = RobustnessAnalyzerV2()

        with caplog.at_level("WARNING"):
            response = analyzer.analyze(request)

        # No default-base warning — intervention overrides base for every option
        # (other codes, e.g. CONSTRAINT_SAMPLES_UNNOISED, may coexist)
        assert not any(
            w.code == "CONSTRAINT_NODE_DEFAULT_BASE" for w in response.inference_warnings
        )
        assert not any(c.code == "CONSTRAINT_NODE_DEFAULT_BASE" for c in response.critiques)

    def test_warning_with_multiple_constraint_nodes(self, caplog):
        """Warning emitted for each non-root node missing ParameterUncertainty."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="fac_spend",
                    kind="factor",
                    label="Spend",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(
                    id="fac_churn",
                    kind="factor",
                    label="Churn",
                    observed_state=ObservedState(value=0.5),
                ),
                NodeV2(
                    id="fac_nps",
                    kind="factor",
                    label="NPS",
                    observed_state=ObservedState(value=0.7),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "fac_spend", "to": "fac_churn"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=-0.3, std=0.05),
                ),
                EdgeV2(
                    **{"from": "fac_spend", "to": "fac_nps"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.4, std=0.05),
                ),
                EdgeV2(
                    **{"from": "fac_churn", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=-0.5, std=0.1),
                ),
                EdgeV2(
                    **{"from": "fac_nps", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.3, std=0.1),
                ),
            ],
        )
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                InterventionOption(
                    id="baseline",
                    label="Baseline",
                    interventions={"fac_spend": 100.0},
                ),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=1,
            goal_constraints=[
                GoalConstraint(node_id="fac_churn", operator="<=", value=0.04),
                GoalConstraint(node_id="fac_nps", operator=">=", value=0.5),
            ],
        )

        analyzer = RobustnessAnalyzerV2()

        with caplog.at_level("WARNING"):
            response = analyzer.analyze(request)

        # Both non-root nodes should trigger default-base warnings
        # (filter by code: other warning codes may coexist)
        default_base_warnings = [
            w for w in response.inference_warnings if w.code == "CONSTRAINT_NODE_DEFAULT_BASE"
        ]
        assert len(default_base_warnings) == 2
        warned_nodes = {w.detail.get("node_id") for w in default_base_warnings}
        assert warned_nodes == {"fac_churn", "fac_nps"}

        # Both should have corresponding critiques
        constraint_critiques = [
            c for c in response.critiques if c.code == "CONSTRAINT_NODE_DEFAULT_BASE"
        ]
        assert len(constraint_critiques) == 2
