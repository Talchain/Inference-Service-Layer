"""
Unit tests for FeasibilityChecker service.

Tests constraint validation and feasibility checking functionality.
"""

import pytest

from src.models.requests import (
    ConstraintNode,
    ConstraintRelation,
    ConstraintType,
    FeasibilityRequest,
    OptionForFeasibility,
)
from src.models.shared import GraphNodeV1, GraphV1, NodeKind
from src.services.feasibility_checker import FeasibilityChecker


@pytest.fixture
def checker():
    """Create a FeasibilityChecker instance."""
    return FeasibilityChecker()


@pytest.fixture
def simple_graph():
    """Create a simple decision graph."""
    return GraphV1(
        nodes=[
            GraphNodeV1(id="goal", kind=NodeKind.GOAL, label="Maximize Profit"),
            GraphNodeV1(id="option_a", kind=NodeKind.OPTION, label="Option A"),
            GraphNodeV1(id="option_b", kind=NodeKind.OPTION, label="Option B"),
        ],
        edges=[],
    )


@pytest.fixture
def budget_constraint():
    """Create a budget constraint."""
    return ConstraintNode(
        id="budget_limit",
        constraint_type=ConstraintType.BUDGET,
        target_variable="total_cost",
        relation=ConstraintRelation.LE,
        threshold=100000.0,
        label="Budget must not exceed $100,000",
        priority="hard",
    )


@pytest.fixture
def revenue_constraint():
    """Create a revenue requirement constraint."""
    return ConstraintNode(
        id="min_revenue",
        constraint_type=ConstraintType.THRESHOLD,
        target_variable="expected_revenue",
        relation=ConstraintRelation.GE,
        threshold=50000.0,
        label="Revenue must be at least $50,000",
        priority="hard",
    )


class TestConstraintValidation:
    """Tests for constraint validation."""

    def test_valid_constraint(self, checker, simple_graph, budget_constraint):
        """Test validation of a properly specified constraint."""
        options = [
            OptionForFeasibility(
                option_id="option_a",
                name="Option A",
                variable_values={"total_cost": 85000},
            )
        ]

        request = FeasibilityRequest(
            graph=simple_graph,
            constraints=[budget_constraint],
            options=options,
        )

        result = checker.check_feasibility(request)

        # Constraint should be valid
        assert len(result.constraint_validation) == 1
        assert result.constraint_validation[0].is_valid
        assert result.constraint_validation[0].issues == []

    def test_constraint_with_missing_variable(self, checker, simple_graph):
        """Test constraint referencing non-existent variable."""
        constraint = ConstraintNode(
            id="missing_var",
            constraint_type=ConstraintType.THRESHOLD,
            target_variable="nonexistent_variable",
            relation=ConstraintRelation.LE,
            threshold=100.0,
            priority="hard",
        )

        options = [
            OptionForFeasibility(
                option_id="option_a",
                name="Option A",
                variable_values={"total_cost": 85000},
            )
        ]

        request = FeasibilityRequest(
            graph=simple_graph,
            constraints=[constraint],
            options=options,
        )

        result = checker.check_feasibility(request)

        # Constraint should be flagged as invalid
        assert len(result.constraint_validation) == 1
        assert not result.constraint_validation[0].is_valid
        assert any("not found" in issue for issue in result.constraint_validation[0].issues)

    def test_multiple_constraints_validation(self, checker, simple_graph, budget_constraint, revenue_constraint):
        """Test validation of multiple constraints."""
        options = [
            OptionForFeasibility(
                option_id="option_a",
                name="Option A",
                variable_values={"total_cost": 85000, "expected_revenue": 120000},
            )
        ]

        request = FeasibilityRequest(
            graph=simple_graph,
            constraints=[budget_constraint, revenue_constraint],
            options=options,
        )

        result = checker.check_feasibility(request)

        assert len(result.constraint_validation) == 2
        assert all(cv.is_valid for cv in result.constraint_validation)


class TestFeasibilityCheck:
    """Tests for feasibility checking."""

    def test_feasible_option(self, checker, simple_graph, budget_constraint):
        """Test option that satisfies constraint."""
        options = [
            OptionForFeasibility(
                option_id="option_a",
                name="Option A",
                variable_values={"total_cost": 85000},
                expected_value=50000,
            )
        ]

        request = FeasibilityRequest(
            graph=simple_graph,
            constraints=[budget_constraint],
            options=options,
        )

        result = checker.check_feasibility(request)

        assert "option_a" in result.feasibility.feasible_options
        assert len(result.feasibility.infeasible_options) == 0

    def test_infeasible_option(self, checker, simple_graph, budget_constraint):
        """Test option that violates constraint."""
        options = [
            OptionForFeasibility(
                option_id="option_a",
                name="Option A",
                variable_values={"total_cost": 120000},  # Exceeds budget
                expected_value=50000,
            )
        ]

        request = FeasibilityRequest(
            graph=simple_graph,
            constraints=[budget_constraint],
            options=options,
        )

        result = checker.check_feasibility(request)

        assert len(result.feasibility.feasible_options) == 0
        assert len(result.feasibility.infeasible_options) == 1

        infeasible = result.feasibility.infeasible_options[0]
        assert infeasible.option_id == "option_a"
        assert "budget_limit" in infeasible.violated_constraints
        assert infeasible.total_violation_magnitude == 20000  # 120000 - 100000

    def test_mixed_feasibility(self, checker, simple_graph, budget_constraint):
        """Test mix of feasible and infeasible options."""
        options = [
            OptionForFeasibility(
                option_id="option_a",
                name="Option A",
                variable_values={"total_cost": 85000},
            ),
            OptionForFeasibility(
                option_id="option_b",
                name="Option B",
                variable_values={"total_cost": 120000},
            ),
            OptionForFeasibility(
                option_id="option_c",
                name="Option C",
                variable_values={"total_cost": 95000},
            ),
        ]

        request = FeasibilityRequest(
            graph=simple_graph,
            constraints=[budget_constraint],
            options=options,
        )

        result = checker.check_feasibility(request)

        assert set(result.feasibility.feasible_options) == {"option_a", "option_c"}
        assert len(result.feasibility.infeasible_options) == 1
        assert result.feasibility.infeasible_options[0].option_id == "option_b"


class TestConstraintRelations:
    """Tests for different constraint relations."""

    def test_le_constraint(self, checker, simple_graph):
        """Test less-than-or-equal constraint."""
        constraint = ConstraintNode(
            id="le_test",
            constraint_type=ConstraintType.THRESHOLD,
            target_variable="value",
            relation=ConstraintRelation.LE,
            threshold=100.0,
            priority="hard",
        )

        options = [
            OptionForFeasibility(option_id="opt1", name="Below", variable_values={"value": 90}),
            OptionForFeasibility(option_id="opt2", name="Equal", variable_values={"value": 100}),
            OptionForFeasibility(option_id="opt3", name="Above", variable_values={"value": 110}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[constraint], options=options
        )
        result = checker.check_feasibility(request)

        # opt1 and opt2 should be feasible (value <= 100)
        assert set(result.feasibility.feasible_options) == {"opt1", "opt2"}
        assert len(result.feasibility.infeasible_options) == 1

    def test_ge_constraint(self, checker, simple_graph):
        """Test greater-than-or-equal constraint."""
        constraint = ConstraintNode(
            id="ge_test",
            constraint_type=ConstraintType.THRESHOLD,
            target_variable="value",
            relation=ConstraintRelation.GE,
            threshold=100.0,
            priority="hard",
        )

        options = [
            OptionForFeasibility(option_id="opt1", name="Below", variable_values={"value": 90}),
            OptionForFeasibility(option_id="opt2", name="Equal", variable_values={"value": 100}),
            OptionForFeasibility(option_id="opt3", name="Above", variable_values={"value": 110}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[constraint], options=options
        )
        result = checker.check_feasibility(request)

        # opt2 and opt3 should be feasible (value >= 100)
        assert set(result.feasibility.feasible_options) == {"opt2", "opt3"}
        assert len(result.feasibility.infeasible_options) == 1

    def test_eq_constraint(self, checker, simple_graph):
        """Test equality constraint."""
        constraint = ConstraintNode(
            id="eq_test",
            constraint_type=ConstraintType.THRESHOLD,
            target_variable="value",
            relation=ConstraintRelation.EQ,
            threshold=100.0,
            priority="hard",
        )

        options = [
            OptionForFeasibility(option_id="opt1", name="Below", variable_values={"value": 90}),
            OptionForFeasibility(option_id="opt2", name="Equal", variable_values={"value": 100}),
            OptionForFeasibility(option_id="opt3", name="Above", variable_values={"value": 110}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[constraint], options=options
        )
        result = checker.check_feasibility(request)

        # Only opt2 should be feasible (value == 100)
        assert result.feasibility.feasible_options == ["opt2"]
        assert len(result.feasibility.infeasible_options) == 2

    def test_lt_constraint(self, checker, simple_graph):
        """Test strict less-than constraint."""
        constraint = ConstraintNode(
            id="lt_test",
            constraint_type=ConstraintType.THRESHOLD,
            target_variable="value",
            relation=ConstraintRelation.LT,
            threshold=100.0,
            priority="hard",
        )

        options = [
            OptionForFeasibility(option_id="opt1", name="Below", variable_values={"value": 90}),
            OptionForFeasibility(option_id="opt2", name="Equal", variable_values={"value": 100}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[constraint], options=options
        )
        result = checker.check_feasibility(request)

        # Only opt1 should be feasible (value < 100, not <=)
        assert result.feasibility.feasible_options == ["opt1"]

    def test_gt_constraint(self, checker, simple_graph):
        """Test strict greater-than constraint."""
        constraint = ConstraintNode(
            id="gt_test",
            constraint_type=ConstraintType.THRESHOLD,
            target_variable="value",
            relation=ConstraintRelation.GT,
            threshold=100.0,
            priority="hard",
        )

        options = [
            OptionForFeasibility(option_id="opt1", name="Equal", variable_values={"value": 100}),
            OptionForFeasibility(option_id="opt2", name="Above", variable_values={"value": 110}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[constraint], options=options
        )
        result = checker.check_feasibility(request)

        # Only opt2 should be feasible (value > 100, not >=)
        assert result.feasibility.feasible_options == ["opt2"]


class TestConstraintPriorities:
    """Tests for constraint priority handling."""

    def test_soft_constraint_excluded(self, checker, simple_graph):
        """Test that soft constraints are excluded when include_partial=False."""
        soft_constraint = ConstraintNode(
            id="soft_limit",
            constraint_type=ConstraintType.THRESHOLD,
            target_variable="value",
            relation=ConstraintRelation.LE,
            threshold=100.0,
            priority="soft",
        )

        options = [
            OptionForFeasibility(option_id="opt1", name="Violates", variable_values={"value": 120}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph,
            constraints=[soft_constraint],
            options=options,
            include_partial_violations=False,
        )
        result = checker.check_feasibility(request)

        # Option should be feasible (soft constraint not enforced)
        assert result.feasibility.feasible_options == ["opt1"]

    def test_soft_constraint_included(self, checker, simple_graph):
        """Test that soft constraints are included when include_partial=True."""
        soft_constraint = ConstraintNode(
            id="soft_limit",
            constraint_type=ConstraintType.THRESHOLD,
            target_variable="value",
            relation=ConstraintRelation.LE,
            threshold=100.0,
            priority="soft",
        )

        options = [
            OptionForFeasibility(option_id="opt1", name="Violates", variable_values={"value": 120}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph,
            constraints=[soft_constraint],
            options=options,
            include_partial_violations=True,
        )
        result = checker.check_feasibility(request)

        # Option should be infeasible (soft constraint enforced)
        assert len(result.feasibility.infeasible_options) == 1
        assert not result.feasibility.infeasible_options[0].violation_details[0].is_hard_violation


class TestMultipleConstraints:
    """Tests for multiple constraint handling."""

    def test_all_constraints_must_pass(self, checker, simple_graph):
        """Test that option must satisfy all constraints to be feasible."""
        constraints = [
            ConstraintNode(
                id="budget",
                constraint_type=ConstraintType.BUDGET,
                target_variable="cost",
                relation=ConstraintRelation.LE,
                threshold=100.0,
                priority="hard",
            ),
            ConstraintNode(
                id="quality",
                constraint_type=ConstraintType.THRESHOLD,
                target_variable="quality_score",
                relation=ConstraintRelation.GE,
                threshold=80.0,
                priority="hard",
            ),
        ]

        options = [
            # Passes both
            OptionForFeasibility(
                option_id="good",
                name="Good",
                variable_values={"cost": 90, "quality_score": 85},
            ),
            # Fails budget
            OptionForFeasibility(
                option_id="expensive",
                name="Expensive",
                variable_values={"cost": 110, "quality_score": 90},
            ),
            # Fails quality
            OptionForFeasibility(
                option_id="low_quality",
                name="Low Quality",
                variable_values={"cost": 50, "quality_score": 60},
            ),
            # Fails both
            OptionForFeasibility(
                option_id="bad",
                name="Bad",
                variable_values={"cost": 120, "quality_score": 50},
            ),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=constraints, options=options
        )
        result = checker.check_feasibility(request)

        assert result.feasibility.feasible_options == ["good"]
        assert len(result.feasibility.infeasible_options) == 3

        # Check "bad" has both violations
        bad_option = next(
            o for o in result.feasibility.infeasible_options if o.option_id == "bad"
        )
        assert len(bad_option.violated_constraints) == 2


class TestWarnings:
    """Tests for warning generation."""

    def test_no_feasible_options_warning(self, checker, simple_graph, budget_constraint):
        """Test warning when no options are feasible."""
        options = [
            OptionForFeasibility(option_id="opt1", name="Too expensive", variable_values={"total_cost": 150000}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[budget_constraint], options=options
        )
        result = checker.check_feasibility(request)

        assert any("No feasible options" in w for w in result.warnings)

    def test_all_feasible_warning(self, checker, simple_graph, budget_constraint):
        """Test warning when all options are feasible (constraints may be loose)."""
        options = [
            OptionForFeasibility(option_id="opt1", name="Opt1", variable_values={"total_cost": 50000}),
            OptionForFeasibility(option_id="opt2", name="Opt2", variable_values={"total_cost": 60000}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[budget_constraint], options=options
        )
        result = checker.check_feasibility(request)

        assert any("All options are feasible" in w for w in result.warnings)

    def test_close_violation_warning(self, checker, simple_graph, budget_constraint):
        """Test warning for options that narrowly violate constraints."""
        options = [
            OptionForFeasibility(
                option_id="opt1",
                name="Just over",
                variable_values={"total_cost": 105000},  # 5% over budget
            ),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[budget_constraint], options=options
        )
        result = checker.check_feasibility(request)

        assert any("narrowly violates" in w for w in result.warnings)


class TestViolationDetails:
    """Tests for violation detail reporting."""

    def test_violation_magnitude(self, checker, simple_graph, budget_constraint):
        """Test that violation magnitude is correctly calculated."""
        options = [
            OptionForFeasibility(
                option_id="opt1",
                name="Over budget",
                variable_values={"total_cost": 130000},  # 30000 over budget
            ),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[budget_constraint], options=options
        )
        result = checker.check_feasibility(request)

        violation = result.feasibility.infeasible_options[0].violation_details[0]
        assert violation.actual_value == 130000
        assert violation.threshold == 100000
        assert violation.violation_magnitude == 30000
        assert violation.is_hard_violation

    def test_summary_generation(self, checker, simple_graph, budget_constraint):
        """Test that summary is correctly generated."""
        options = [
            OptionForFeasibility(option_id="opt1", name="Feasible", variable_values={"total_cost": 80000}),
            OptionForFeasibility(option_id="opt2", name="Infeasible", variable_values={"total_cost": 120000}),
            OptionForFeasibility(option_id="opt3", name="Feasible", variable_values={"total_cost": 90000}),
        ]

        request = FeasibilityRequest(
            graph=simple_graph, constraints=[budget_constraint], options=options
        )
        result = checker.check_feasibility(request)

        assert result.summary == "2 of 3 options are feasible"
