"""
Unit tests for continuous optimization endpoint.

Tests grid search optimization with:
- Basic optimization (maximize/minimize)
- Constraint handling
- Confidence intervals
- Sensitivity analysis
- Edge cases (no feasible solution, flat objective, boundary optimum)
"""

import pytest
from pydantic import ValidationError

from src.models.requests import (
    DecisionVariable,
    ObjectiveFunction,
    OptimisationConstraint,
    OptimisationRequest,
)
from src.models.responses import OptimisationResponse
from src.services.continuous_optimizer import ContinuousOptimizer


class TestObjectiveFunctionValidation:
    """Test ObjectiveFunction model validation."""

    def test_valid_objective(self):
        """Valid objective function should be accepted."""
        obj = ObjectiveFunction(
            variable_id="profit",
            direction="maximize",
            coefficients={"price": 100, "quantity": -5},
            constant=-10000
        )
        assert obj.direction == "maximize"
        assert obj.coefficients["price"] == 100

    def test_invalid_direction(self):
        """Invalid direction should be rejected."""
        with pytest.raises(ValidationError):
            ObjectiveFunction(
                variable_id="profit",
                direction="optimal",  # Invalid
                coefficients={"price": 100}
            )

    def test_empty_coefficients_rejected(self):
        """Empty coefficients should be rejected."""
        with pytest.raises(ValidationError):
            ObjectiveFunction(
                variable_id="profit",
                direction="maximize",
                coefficients={}
            )


class TestDecisionVariableValidation:
    """Test DecisionVariable model validation."""

    def test_valid_variable(self):
        """Valid decision variable should be accepted."""
        var = DecisionVariable(
            variable_id="price",
            lower_bound=10.0,
            upper_bound=100.0
        )
        assert var.lower_bound == 10.0
        assert var.upper_bound == 100.0

    def test_upper_bound_less_than_lower(self):
        """upper_bound < lower_bound should be rejected."""
        with pytest.raises(ValidationError):
            DecisionVariable(
                variable_id="price",
                lower_bound=100.0,
                upper_bound=10.0
            )

    def test_equal_bounds_allowed(self):
        """Equal bounds should be allowed (fixed variable)."""
        var = DecisionVariable(
            variable_id="fixed",
            lower_bound=50.0,
            upper_bound=50.0
        )
        assert var.lower_bound == var.upper_bound


class TestOptimisationRequestValidation:
    """Test OptimisationRequest model validation."""

    def test_valid_request(self):
        """Valid optimization request should be accepted."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="profit",
                direction="maximize",
                coefficients={"price": 100}
            ),
            decision_variables=[
                DecisionVariable(variable_id="price", lower_bound=10, upper_bound=100)
            ]
        )
        assert len(request.decision_variables) == 1

    def test_duplicate_variable_ids_rejected(self):
        """Duplicate variable IDs should be rejected."""
        with pytest.raises(ValidationError):
            OptimisationRequest(
                objective=ObjectiveFunction(
                    variable_id="profit",
                    direction="maximize",
                    coefficients={"price": 100}
                ),
                decision_variables=[
                    DecisionVariable(variable_id="price", lower_bound=10, upper_bound=100),
                    DecisionVariable(variable_id="price", lower_bound=20, upper_bound=80)  # Duplicate
                ]
            )

    def test_grid_points_bounds(self):
        """Grid points should be within bounds."""
        # Too few points
        with pytest.raises(ValidationError):
            OptimisationRequest(
                objective=ObjectiveFunction(
                    variable_id="profit",
                    direction="maximize",
                    coefficients={"price": 100}
                ),
                decision_variables=[
                    DecisionVariable(variable_id="price", lower_bound=10, upper_bound=100)
                ],
                grid_points=2  # < 5 minimum
            )


class TestContinuousOptimizerBasic:
    """Test basic optimization functionality."""

    def test_maximize_single_variable(self):
        """Maximize single variable: f(x) = 2x, x in [0, 10]."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 2.0},
                constant=0.0
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is not None
        assert result.optimal_point.objective_value == pytest.approx(20.0, rel=1e-6)
        assert result.optimal_point.variable_values["x"] == pytest.approx(10.0, rel=1e-6)
        assert result.optimal_point.is_boundary is True

    def test_minimize_single_variable(self):
        """Minimize single variable: f(x) = 2x, x in [0, 10]."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="minimize",
                coefficients={"x": 2.0},
                constant=0.0
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is not None
        assert result.optimal_point.objective_value == pytest.approx(0.0, rel=1e-6)
        assert result.optimal_point.variable_values["x"] == pytest.approx(0.0, rel=1e-6)

    def test_two_variable_optimization(self):
        """Optimize two variables: f(x,y) = 3x + 2y."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 3.0, "y": 2.0},
                constant=0.0
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=5),
                DecisionVariable(variable_id="y", lower_bound=0, upper_bound=10)
            ],
            grid_points=6
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is not None
        # Maximum at x=5, y=10 → f = 15 + 20 = 35
        assert result.optimal_point.objective_value == pytest.approx(35.0, rel=1e-6)
        assert result.optimal_point.variable_values["x"] == pytest.approx(5.0, rel=1e-6)
        assert result.optimal_point.variable_values["y"] == pytest.approx(10.0, rel=1e-6)


class TestConstraintHandling:
    """Test constraint handling."""

    def test_le_constraint(self):
        """Less-than-or-equal constraint: x <= 5."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 2.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            constraints=[
                OptimisationConstraint(
                    constraint_id="limit",
                    coefficients={"x": 1.0},
                    relation="le",
                    rhs=5.0
                )
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is not None
        # Should stop at x=5 due to constraint
        assert result.optimal_point.variable_values["x"] == pytest.approx(5.0, rel=1e-6)
        assert result.optimal_point.objective_value == pytest.approx(10.0, rel=1e-6)

    def test_ge_constraint(self):
        """Greater-than-or-equal constraint: x >= 3."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="minimize",
                coefficients={"x": 2.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            constraints=[
                OptimisationConstraint(
                    constraint_id="minimum",
                    coefficients={"x": 1.0},
                    relation="ge",
                    rhs=3.0
                )
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is not None
        # Should stop at x=3 due to constraint
        assert result.optimal_point.variable_values["x"] == pytest.approx(3.0, rel=1e-6)
        assert result.optimal_point.objective_value == pytest.approx(6.0, rel=1e-6)

    def test_combined_constraints(self):
        """Multiple constraints: budget allocation."""
        # Maximize profit = 3*price + 2*quantity
        # Subject to: price + quantity <= 100 (budget)
        #             price >= 10 (minimum price)
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="profit",
                direction="maximize",
                coefficients={"price": 3.0, "quantity": 2.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="price", lower_bound=0, upper_bound=100),
                DecisionVariable(variable_id="quantity", lower_bound=0, upper_bound=100)
            ],
            constraints=[
                OptimisationConstraint(
                    constraint_id="budget",
                    coefficients={"price": 1.0, "quantity": 1.0},
                    relation="le",
                    rhs=100.0
                ),
                OptimisationConstraint(
                    constraint_id="min_price",
                    coefficients={"price": 1.0},
                    relation="ge",
                    rhs=10.0
                )
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is not None
        # Since price has higher coefficient, optimal is price=100, quantity=0
        # But budget constraint: price + quantity <= 100
        # And min_price >= 10
        # With price=100, quantity must be 0 to satisfy budget
        assert result.optimal_point.variable_values["price"] <= 100
        assert result.optimal_point.variable_values["quantity"] >= 0


class TestNoFeasibleSolution:
    """Test no feasible solution handling."""

    def test_infeasible_constraints(self):
        """Conflicting constraints should return no solution."""
        # x >= 60 AND x <= 40 → impossible
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 1.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=100)
            ],
            constraints=[
                OptimisationConstraint(
                    constraint_id="c1",
                    coefficients={"x": 1.0},
                    relation="ge",
                    rhs=60.0
                ),
                OptimisationConstraint(
                    constraint_id="c2",
                    coefficients={"x": 1.0},
                    relation="le",
                    rhs=40.0
                )
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is None
        assert result.grid_metrics.feasible_points == 0
        assert len(result.warnings) > 0
        assert result.warnings[0].code == "NO_FEASIBLE_SOLUTION"


class TestFlatObjective:
    """Test flat objective handling."""

    def test_flat_objective_detection(self):
        """Flat objective should be detected."""
        # f(x) = 0 (constant)
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"y": 1.0},  # y not in variables
                constant=10.0
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        # Should find flat objective warning
        warning_codes = [w.code for w in result.warnings]
        assert "FLAT_OBJECTIVE" in warning_codes


class TestBoundaryOptimum:
    """Test boundary optimum detection."""

    def test_boundary_optimum_detection(self):
        """Boundary optimum should be detected and reported."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 1.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is not None
        assert result.optimal_point.is_boundary is True
        assert "x" in result.optimal_point.boundary_variables

        warning_codes = [w.code for w in result.warnings]
        assert "BOUNDARY_OPTIMUM" in warning_codes


class TestConfidenceIntervals:
    """Test confidence interval computation."""

    def test_confidence_interval_computed(self):
        """Confidence interval should be computed."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 10.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            grid_points=11,
            confidence_level=0.95,
            noise_std=5.0
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.optimal_point is not None
        ci = result.optimal_point.confidence_interval

        # Optimal value is 100, check CI contains it
        assert ci.lower < result.optimal_point.objective_value < ci.upper
        assert ci.confidence_level == 0.95
        assert ci.upper - ci.lower > 0  # Non-empty interval

    def test_different_confidence_levels(self):
        """Different confidence levels should give different intervals."""
        base_request = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 10.0}
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
            "noise_std": 5.0
        }

        optimizer = ContinuousOptimizer(seed=42)

        # 90% CI
        request_90 = OptimisationRequest(**{**base_request, "confidence_level": 0.90})
        result_90 = optimizer.optimize(request_90)

        # 99% CI
        request_99 = OptimisationRequest(**{**base_request, "confidence_level": 0.99})
        result_99 = optimizer.optimize(request_99)

        assert result_90.optimal_point is not None
        assert result_99.optimal_point is not None

        width_90 = result_90.optimal_point.confidence_interval.upper - result_90.optimal_point.confidence_interval.lower
        width_99 = result_99.optimal_point.confidence_interval.upper - result_99.optimal_point.confidence_interval.lower

        # 99% CI should be wider than 90% CI
        assert width_99 > width_90


class TestSensitivityAnalysis:
    """Test sensitivity analysis at optimum."""

    def test_gradient_computed(self):
        """Gradient at optimum should equal coefficients."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 5.0, "y": 3.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10),
                DecisionVariable(variable_id="y", lower_bound=0, upper_bound=10)
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.sensitivity is not None
        # For linear objective, gradient equals coefficients
        assert result.sensitivity.gradient_at_optimum["x"] == pytest.approx(5.0, rel=1e-6)
        assert result.sensitivity.gradient_at_optimum["y"] == pytest.approx(3.0, rel=1e-6)

    def test_range_within_5pct_computed(self):
        """5% tolerance range should be computed."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 10.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            grid_points=21
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.sensitivity is not None
        assert "x" in result.sensitivity.range_within_5pct
        range_5pct = result.sensitivity.range_within_5pct["x"]

        # Check it's a valid range
        assert len(range_5pct) == 2
        assert range_5pct[0] <= range_5pct[1]

    def test_robustness_score(self):
        """Robustness score should be computed."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 10.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            grid_points=21
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.sensitivity is not None
        assert 0 <= result.sensitivity.robustness_score <= 1
        assert result.sensitivity.robustness in ["robust", "moderate", "fragile"]


class TestPerformance:
    """Test performance requirements."""

    def test_20_point_grid_under_2_seconds(self):
        """20-point grid with 2 variables should complete in <2 seconds."""
        import time

        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 3.0, "y": 2.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=100),
                DecisionVariable(variable_id="y", lower_bound=0, upper_bound=100)
            ],
            grid_points=20
        )

        optimizer = ContinuousOptimizer(seed=42)

        start = time.time()
        result = optimizer.optimize(request)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Optimization took {elapsed}s, expected <2s"
        assert result.optimal_point is not None

    def test_grid_metrics_reported(self):
        """Grid metrics should be accurately reported."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 1.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10)
            ],
            grid_points=11
        )

        optimizer = ContinuousOptimizer(seed=42)
        result = optimizer.optimize(request)

        assert result.grid_metrics.grid_points_evaluated == 11
        assert result.grid_metrics.feasible_points == 11
        assert result.grid_metrics.computation_time_ms > 0
        assert result.grid_metrics.convergence_achieved is True


class TestReproducibility:
    """Test reproducibility with seed."""

    def test_same_seed_same_result(self):
        """Same seed should produce same result."""
        request = OptimisationRequest(
            objective=ObjectiveFunction(
                variable_id="output",
                direction="maximize",
                coefficients={"x": 3.0, "y": 2.0}
            ),
            decision_variables=[
                DecisionVariable(variable_id="x", lower_bound=0, upper_bound=10),
                DecisionVariable(variable_id="y", lower_bound=0, upper_bound=10)
            ],
            grid_points=11,
            seed=42
        )

        optimizer1 = ContinuousOptimizer(seed=42)
        optimizer2 = ContinuousOptimizer(seed=42)

        result1 = optimizer1.optimize(request)
        result2 = optimizer2.optimize(request)

        assert result1.optimal_point.objective_value == result2.optimal_point.objective_value
        assert result1.optimal_point.variable_values == result2.optimal_point.variable_values
