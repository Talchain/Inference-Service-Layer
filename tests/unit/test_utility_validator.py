"""
Unit tests for UtilityValidator service.

Tests cover:
- Weight normalization and defaults
- Aggregation method validation
- Risk tolerance parameter checks
- Graph reference validation
- Edge cases and error conditions
"""

import pytest

from src.models.requests import (
    AggregationMethod,
    GoalSpecification,
    RiskTolerance,
    UtilityFunctionSpec,
    UtilityValidationRequest,
)
from src.models.shared import GraphNodeV1, GraphV1, NodeKind
from src.services.utility_validator import UtilityValidator


@pytest.fixture
def validator():
    """Create a UtilityValidator instance."""
    return UtilityValidator()


@pytest.fixture
def simple_goals():
    """Create a list of simple goal specifications."""
    return [
        GoalSpecification(goal_id="profit", label="Maximize Profit", direction="maximize"),
        GoalSpecification(goal_id="cost", label="Minimize Cost", direction="minimize"),
    ]


@pytest.fixture
def weighted_goals():
    """Create goals with explicit weights."""
    return [
        GoalSpecification(
            goal_id="profit", label="Maximize Profit", direction="maximize", weight=0.6
        ),
        GoalSpecification(
            goal_id="cost", label="Minimize Cost", direction="minimize", weight=0.4
        ),
    ]


@pytest.fixture
def sample_graph():
    """Create a sample graph with goal nodes."""
    return GraphV1(
        nodes=[
            GraphNodeV1(id="profit", label="Profit", kind=NodeKind.GOAL),
            GraphNodeV1(id="cost", label="Cost", kind=NodeKind.GOAL),
            GraphNodeV1(id="revenue", label="Revenue", kind=NodeKind.FACTOR),
        ],
        edges=[],
    )


class TestWeightNormalization:
    """Tests for weight normalization behaviour."""

    def test_equal_weights_when_none_specified(self, validator, simple_goals):
        """Test that equal weights are applied when no weights specified."""
        spec = UtilityFunctionSpec(goals=simple_goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        assert len(result.normalised_weights) == 2
        # Equal weighting: 1/2 = 0.5
        assert result.normalised_weights["profit"] == pytest.approx(0.5, rel=1e-4)
        assert result.normalised_weights["cost"] == pytest.approx(0.5, rel=1e-4)
        assert "Equal weighting applied" in str(result.default_behaviour_applied)

    def test_weights_summing_to_one(self, validator, weighted_goals):
        """Test that weights summing to 1.0 are preserved."""
        spec = UtilityFunctionSpec(goals=weighted_goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        assert result.normalised_weights["profit"] == pytest.approx(0.6, rel=1e-4)
        assert result.normalised_weights["cost"] == pytest.approx(0.4, rel=1e-4)
        # Should not have normalization warning since sum is exactly 1.0
        normalization_warnings = [
            w for w in result.warnings if w.code == "WEIGHTS_NORMALIZED"
        ]
        assert len(normalization_warnings) == 0

    def test_weights_normalized_when_not_summing_to_one(self, validator):
        """Test that weights are normalized when they don't sum to 1.0."""
        goals = [
            GoalSpecification(
                goal_id="profit", label="Profit", direction="maximize", weight=0.3
            ),
            GoalSpecification(
                goal_id="cost", label="Cost", direction="minimize", weight=0.3
            ),
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        # 0.3 / 0.6 = 0.5
        assert result.normalised_weights["profit"] == pytest.approx(0.5, rel=1e-4)
        assert result.normalised_weights["cost"] == pytest.approx(0.5, rel=1e-4)
        # Should have normalization warning
        normalization_warnings = [
            w for w in result.warnings if w.code == "WEIGHTS_NORMALIZED"
        ]
        assert len(normalization_warnings) == 1

    def test_partial_weights_distributed(self, validator):
        """Test that unspecified weights are distributed from remaining."""
        goals = [
            GoalSpecification(
                goal_id="profit", label="Profit", direction="maximize", weight=0.6
            ),
            GoalSpecification(goal_id="cost", label="Cost", direction="minimize"),
            # No weight specified
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        # profit keeps 0.6, cost gets remaining 0.4
        assert result.normalised_weights["profit"] == pytest.approx(0.6, rel=1e-4)
        assert result.normalised_weights["cost"] == pytest.approx(0.4, rel=1e-4)

    def test_strict_mode_missing_weights(self, validator):
        """Test strict mode with missing weights."""
        goals = [
            GoalSpecification(
                goal_id="profit", label="Profit", direction="maximize", weight=0.6
            ),
            GoalSpecification(goal_id="cost", label="Cost", direction="minimize"),
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec, strict_mode=True)

        result = validator.validate(request)

        # Still valid but has warning
        assert result.valid is True
        missing_warnings = [
            w for w in result.warnings if w.code == "MISSING_WEIGHTS"
        ]
        assert len(missing_warnings) == 1


class TestAggregationMethodValidation:
    """Tests for aggregation method validation."""

    def test_weighted_sum_default(self, validator, weighted_goals):
        """Test weighted_sum is the default method."""
        spec = UtilityFunctionSpec(goals=weighted_goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        assert result.aggregation_method == "weighted_sum"

    def test_lexicographic_without_priorities_warning(self, validator, weighted_goals):
        """Test lexicographic method without priorities generates warning."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals, aggregation_method=AggregationMethod.LEXICOGRAPHIC
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        assert result.aggregation_method == "lexicographic"
        method_warnings = [
            w for w in result.warnings if w.code == "AGGREGATION_METHOD"
        ]
        assert len(method_warnings) == 1
        assert "priority" in method_warnings[0].message.lower()

    def test_lexicographic_with_priorities(self, validator):
        """Test lexicographic method with priorities is valid."""
        goals = [
            GoalSpecification(
                goal_id="profit",
                label="Profit",
                direction="maximize",
                weight=0.5,
                priority=1,
            ),
            GoalSpecification(
                goal_id="cost",
                label="Cost",
                direction="minimize",
                weight=0.5,
                priority=2,
            ),
        ]
        spec = UtilityFunctionSpec(
            goals=goals, aggregation_method=AggregationMethod.LEXICOGRAPHIC
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        # No priority warning when priorities are provided
        priority_warnings = [
            w
            for w in result.warnings
            if w.code == "AGGREGATION_METHOD" and "priority" in w.message.lower()
        ]
        assert len(priority_warnings) == 0

    def test_weighted_product_with_minimize_warning(self, validator, weighted_goals):
        """Test weighted_product with minimize goals generates warning."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals, aggregation_method=AggregationMethod.WEIGHTED_PRODUCT
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        method_warnings = [
            w for w in result.warnings if w.code == "AGGREGATION_METHOD"
        ]
        assert len(method_warnings) == 1
        assert "minimize" in method_warnings[0].message.lower()

    def test_dominant_weight_warning(self, validator):
        """Test warning when one goal has >90% weight."""
        goals = [
            GoalSpecification(
                goal_id="profit", label="Profit", direction="maximize", weight=0.95
            ),
            GoalSpecification(
                goal_id="cost", label="Cost", direction="minimize", weight=0.05
            ),
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        method_warnings = [
            w for w in result.warnings if w.code == "AGGREGATION_METHOD"
        ]
        assert len(method_warnings) == 1
        assert "90%" in method_warnings[0].message

    def test_min_max_method(self, validator, weighted_goals):
        """Test min_max aggregation method."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals, aggregation_method=AggregationMethod.MIN_MAX
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        assert result.aggregation_method == "min_max"


class TestRiskToleranceValidation:
    """Tests for risk tolerance validation."""

    def test_risk_neutral_default(self, validator, weighted_goals):
        """Test risk_neutral is the default."""
        spec = UtilityFunctionSpec(goals=weighted_goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.risk_tolerance == "risk_neutral"

    def test_risk_averse_without_coefficient_warning(self, validator, weighted_goals):
        """Test risk_averse without coefficient generates warning."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals, risk_tolerance=RiskTolerance.RISK_AVERSE
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        risk_warnings = [w for w in result.warnings if w.code == "RISK_TOLERANCE"]
        assert len(risk_warnings) == 1
        assert "default coefficient" in risk_warnings[0].message.lower()

    def test_risk_averse_with_coefficient(self, validator, weighted_goals):
        """Test risk_averse with coefficient is valid."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals,
            risk_tolerance=RiskTolerance.RISK_AVERSE,
            risk_coefficient=2.0,
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        risk_warnings = [w for w in result.warnings if w.code == "RISK_TOLERANCE"]
        assert len(risk_warnings) == 0

    def test_risk_averse_zero_coefficient_warning(self, validator, weighted_goals):
        """Test risk_averse with coefficient=0 generates warning."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals,
            risk_tolerance=RiskTolerance.RISK_AVERSE,
            risk_coefficient=0.0,
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        risk_warnings = [w for w in result.warnings if w.code == "RISK_TOLERANCE"]
        assert len(risk_warnings) == 1
        assert "risk_neutral" in risk_warnings[0].message

    def test_risk_averse_high_coefficient_warning(self, validator, weighted_goals):
        """Test risk_averse with high coefficient generates warning."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals,
            risk_tolerance=RiskTolerance.RISK_AVERSE,
            risk_coefficient=6.0,
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        risk_warnings = [w for w in result.warnings if w.code == "RISK_TOLERANCE"]
        assert len(risk_warnings) == 1
        assert "high" in risk_warnings[0].message.lower()

    def test_risk_seeking_high_coefficient_warning(self, validator, weighted_goals):
        """Test risk_seeking with high coefficient generates warning."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals,
            risk_tolerance=RiskTolerance.RISK_SEEKING,
            risk_coefficient=3.0,
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        risk_warnings = [w for w in result.warnings if w.code == "RISK_TOLERANCE"]
        assert len(risk_warnings) == 1
        assert "high" in risk_warnings[0].message.lower()

    def test_risk_neutral_with_coefficient_warning(self, validator, weighted_goals):
        """Test risk_neutral with coefficient generates warning."""
        spec = UtilityFunctionSpec(
            goals=weighted_goals,
            risk_tolerance=RiskTolerance.RISK_NEUTRAL,
            risk_coefficient=1.0,
        )
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        risk_warnings = [w for w in result.warnings if w.code == "RISK_TOLERANCE"]
        assert len(risk_warnings) == 1
        assert "ignored" in risk_warnings[0].message.lower()


class TestGraphReferenceValidation:
    """Tests for graph reference validation."""

    def test_valid_graph_references(self, validator, weighted_goals, sample_graph):
        """Test validation with valid graph references."""
        spec = UtilityFunctionSpec(goals=weighted_goals)
        request = UtilityValidationRequest(utility_spec=spec, graph=sample_graph)

        result = validator.validate(request)

        assert result.valid is True
        # No graph reference warnings for valid goals
        graph_warnings = [w for w in result.warnings if w.code == "GRAPH_REFERENCE"]
        assert len(graph_warnings) == 0

    def test_missing_node_reference_warning(self, validator, sample_graph):
        """Test warning when goal references non-existent node."""
        goals = [
            GoalSpecification(
                goal_id="profit", label="Profit", direction="maximize", weight=0.5
            ),
            GoalSpecification(
                goal_id="missing_node", label="Missing", direction="maximize", weight=0.5
            ),
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec, graph=sample_graph)

        result = validator.validate(request)

        assert result.valid is True  # Still valid, just warning
        graph_warnings = [w for w in result.warnings if w.code == "GRAPH_REFERENCE"]
        assert len(graph_warnings) == 1
        assert "missing_node" in graph_warnings[0].message

    def test_non_goal_node_reference_warning(self, validator, sample_graph):
        """Test warning when goal references non-goal node type."""
        goals = [
            GoalSpecification(
                goal_id="profit", label="Profit", direction="maximize", weight=0.5
            ),
            GoalSpecification(
                goal_id="revenue", label="Revenue", direction="maximize", weight=0.5
            ),  # revenue is a FACTOR node
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec, graph=sample_graph)

        result = validator.validate(request)

        assert result.valid is True
        graph_warnings = [w for w in result.warnings if w.code == "GRAPH_REFERENCE"]
        assert len(graph_warnings) == 1
        assert "revenue" in graph_warnings[0].message
        assert "goal" in graph_warnings[0].message.lower()

    def test_no_graph_skips_validation(self, validator, weighted_goals):
        """Test that no graph skips graph validation."""
        spec = UtilityFunctionSpec(goals=weighted_goals)
        request = UtilityValidationRequest(utility_spec=spec, graph=None)

        result = validator.validate(request)

        assert result.valid is True
        graph_warnings = [w for w in result.warnings if w.code == "GRAPH_REFERENCE"]
        assert len(graph_warnings) == 0


class TestNormalisedGoals:
    """Tests for normalised goals output."""

    def test_normalised_goals_structure(self, validator, weighted_goals):
        """Test normalised goals contain all expected fields."""
        spec = UtilityFunctionSpec(goals=weighted_goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert len(result.normalised_goals) == 2
        profit_goal = next(g for g in result.normalised_goals if g.goal_id == "profit")
        assert profit_goal.label == "Maximize Profit"
        assert profit_goal.direction == "maximize"
        assert profit_goal.normalised_weight == pytest.approx(0.6, rel=1e-4)
        assert profit_goal.original_weight == 0.6

    def test_normalised_goals_with_priorities(self, validator):
        """Test normalised goals preserve priorities."""
        goals = [
            GoalSpecification(
                goal_id="profit",
                label="Profit",
                direction="maximize",
                weight=0.5,
                priority=1,
            ),
            GoalSpecification(
                goal_id="cost",
                label="Cost",
                direction="minimize",
                weight=0.5,
                priority=2,
            ),
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        profit_goal = next(g for g in result.normalised_goals if g.goal_id == "profit")
        cost_goal = next(g for g in result.normalised_goals if g.goal_id == "cost")
        assert profit_goal.priority == 1
        assert cost_goal.priority == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_goal(self, validator):
        """Test validation with single goal."""
        goals = [
            GoalSpecification(goal_id="profit", label="Profit", direction="maximize")
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        assert result.normalised_weights["profit"] == pytest.approx(1.0, rel=1e-4)

    def test_many_goals(self, validator):
        """Test validation with many goals."""
        goals = [
            GoalSpecification(goal_id=f"goal_{i}", label=f"Goal {i}", direction="maximize")
            for i in range(10)
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        assert len(result.normalised_weights) == 10
        # Equal weighting: 1/10 = 0.1
        for goal_id, weight in result.normalised_weights.items():
            assert weight == pytest.approx(0.1, rel=1e-4)

    def test_very_small_weights_normalized(self, validator):
        """Test normalization with very small weights."""
        goals = [
            GoalSpecification(
                goal_id="profit", label="Profit", direction="maximize", weight=0.001
            ),
            GoalSpecification(
                goal_id="cost", label="Cost", direction="minimize", weight=0.001
            ),
        ]
        spec = UtilityFunctionSpec(goals=goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.valid is True
        # Should be normalized to sum to 1.0
        total = sum(result.normalised_weights.values())
        assert total == pytest.approx(1.0, rel=1e-4)

    def test_all_aggregation_methods(self, validator, weighted_goals):
        """Test all aggregation methods are accepted."""
        methods = [
            AggregationMethod.WEIGHTED_SUM,
            AggregationMethod.WEIGHTED_PRODUCT,
            AggregationMethod.LEXICOGRAPHIC,
            AggregationMethod.MIN_MAX,
        ]

        for method in methods:
            spec = UtilityFunctionSpec(goals=weighted_goals, aggregation_method=method)
            request = UtilityValidationRequest(utility_spec=spec)
            result = validator.validate(request)
            assert result.valid is True
            assert result.aggregation_method == method.value

    def test_response_schema_version(self, validator, weighted_goals):
        """Test response includes schema version."""
        spec = UtilityFunctionSpec(goals=weighted_goals)
        request = UtilityValidationRequest(utility_spec=spec)

        result = validator.validate(request)

        assert result.schema_version == "utility.v1"
