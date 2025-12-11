"""
Continuous Optimization Service.

Provides grid search optimization for continuous decision variables
with constraint handling, confidence intervals, and sensitivity analysis.
"""

import logging
import time
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.models.requests import (
    DecisionVariable,
    ObjectiveFunction,
    OptimisationConstraint,
    OptimisationRequest,
)
from src.models.responses import (
    GridSearchMetrics,
    OptimalPoint,
    OptimisationResponse,
    OptimisationSensitivity,
    OptimisationWarning,
)
from src.models.shared import ConfidenceInterval

logger = logging.getLogger(__name__)


class ContinuousOptimizer:
    """
    Grid search optimizer for continuous decision variables.

    Features:
    - Multi-dimensional grid search
    - Linear constraint handling
    - Confidence interval computation
    - Sensitivity analysis (gradient, robustness)
    - Edge case detection (boundary, flat objective, infeasible)
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize optimizer with optional random seed."""
        self._rng = np.random.default_rng(seed)

    def optimize(self, request: OptimisationRequest) -> OptimisationResponse:
        """
        Perform grid search optimization.

        Args:
            request: Optimization request with objective, variables, constraints

        Returns:
            OptimisationResponse with optimal point, sensitivity, and warnings
        """
        start_time = time.time()
        warnings: List[OptimisationWarning] = []

        # Build grid
        grids = self._build_grids(request.decision_variables, request.grid_points)
        grid_points = list(product(*grids.values()))
        var_ids = list(grids.keys())

        # Evaluate all points
        results: List[Tuple[Dict[str, float], float, bool]] = []
        for point in grid_points:
            var_values = dict(zip(var_ids, point))
            obj_value = self._evaluate_objective(request.objective, var_values)
            feasible = self._check_constraints(request.constraints, var_values)
            results.append((var_values, obj_value, feasible))

        # Filter feasible points
        feasible_results = [(v, o) for v, o, f in results if f]
        total_points = len(results)
        feasible_points = len(feasible_results)

        computation_time_ms = (time.time() - start_time) * 1000

        # Handle no feasible solution
        if not feasible_results:
            warnings.append(OptimisationWarning(
                code="NO_FEASIBLE_SOLUTION",
                message="No feasible solution found. All grid points violate constraints.",
                affected_variables=var_ids
            ))
            return OptimisationResponse(
                optimal_point=None,
                sensitivity=None,
                grid_metrics=GridSearchMetrics(
                    grid_points_evaluated=total_points,
                    feasible_points=0,
                    computation_time_ms=computation_time_ms,
                    convergence_achieved=False
                ),
                warnings=warnings
            )

        # Find optimal point
        if request.objective.direction == "maximize":
            optimal_vars, optimal_obj = max(feasible_results, key=lambda x: x[1])
        else:
            optimal_vars, optimal_obj = min(feasible_results, key=lambda x: x[1])

        # Check for flat objective
        obj_values = [o for _, o in feasible_results]
        obj_range = max(obj_values) - min(obj_values)
        if obj_range < 1e-10:
            warnings.append(OptimisationWarning(
                code="FLAT_OBJECTIVE",
                message="Objective function is essentially flat across all feasible points.",
                affected_variables=var_ids
            ))

        # Check for multiple optima (values within 0.1% of optimum)
        tolerance = abs(optimal_obj) * 0.001 if optimal_obj != 0 else 1e-10
        near_optimal = [v for v, o in feasible_results if abs(o - optimal_obj) <= tolerance]
        if len(near_optimal) > 1:
            warnings.append(OptimisationWarning(
                code="MULTIPLE_OPTIMA",
                message=f"Found {len(near_optimal)} near-optimal points. Solution may not be unique.",
                affected_variables=None
            ))

        # Check boundary conditions
        boundary_vars = []
        for var in request.decision_variables:
            val = optimal_vars[var.variable_id]
            if abs(val - var.lower_bound) < 1e-10 or abs(val - var.upper_bound) < 1e-10:
                boundary_vars.append(var.variable_id)

        is_boundary = len(boundary_vars) > 0
        if is_boundary:
            warnings.append(OptimisationWarning(
                code="BOUNDARY_OPTIMUM",
                message=f"Optimal point is at variable bounds: {boundary_vars}. Consider expanding bounds.",
                affected_variables=boundary_vars
            ))

        # Check for active constraints
        if request.constraints:
            active_constraints = self._find_active_constraints(
                request.constraints, optimal_vars
            )
            if active_constraints:
                warnings.append(OptimisationWarning(
                    code="CONSTRAINT_ACTIVE",
                    message=f"Constraints active at optimum: {active_constraints}",
                    affected_variables=None
                ))

        # Compute confidence interval
        noise_std = request.noise_std if request.noise_std else abs(optimal_obj) * 0.05
        ci = self._compute_confidence_interval(
            optimal_obj, noise_std, request.confidence_level
        )

        # Compute sensitivity analysis
        sensitivity = self._compute_sensitivity(
            request, optimal_vars, optimal_obj, feasible_results
        )

        # Build optimal point response
        optimal_point = OptimalPoint(
            variable_values=optimal_vars,
            objective_value=optimal_obj,
            confidence_interval=ci,
            is_boundary=is_boundary,
            boundary_variables=boundary_vars if is_boundary else None,
            feasible=True
        )

        return OptimisationResponse(
            optimal_point=optimal_point,
            sensitivity=sensitivity,
            grid_metrics=GridSearchMetrics(
                grid_points_evaluated=total_points,
                feasible_points=feasible_points,
                computation_time_ms=computation_time_ms,
                convergence_achieved=True
            ),
            warnings=warnings
        )

    def _build_grids(
        self, variables: List[DecisionVariable], n_points: int
    ) -> Dict[str, np.ndarray]:
        """Build grid for each decision variable."""
        grids = {}
        for var in variables:
            if var.step_size:
                # Use specified step size
                grids[var.variable_id] = np.arange(
                    var.lower_bound, var.upper_bound + var.step_size / 2, var.step_size
                )
            else:
                # Use uniform grid
                grids[var.variable_id] = np.linspace(
                    var.lower_bound, var.upper_bound, n_points
                )
        return grids

    def _evaluate_objective(
        self, objective: ObjectiveFunction, var_values: Dict[str, float]
    ) -> float:
        """Evaluate objective function at given point."""
        result = objective.constant
        for var_id, coeff in objective.coefficients.items():
            if var_id in var_values:
                result += coeff * var_values[var_id]
        return result

    def _check_constraints(
        self,
        constraints: Optional[List[OptimisationConstraint]],
        var_values: Dict[str, float]
    ) -> bool:
        """Check if point satisfies all constraints."""
        if not constraints:
            return True

        for constraint in constraints:
            lhs = sum(
                coeff * var_values.get(var_id, 0)
                for var_id, coeff in constraint.coefficients.items()
            )

            if constraint.relation == "le":
                if lhs > constraint.rhs + 1e-10:
                    return False
            elif constraint.relation == "ge":
                if lhs < constraint.rhs - 1e-10:
                    return False
            elif constraint.relation == "eq":
                if abs(lhs - constraint.rhs) > 1e-10:
                    return False

        return True

    def _find_active_constraints(
        self,
        constraints: List[OptimisationConstraint],
        var_values: Dict[str, float]
    ) -> List[str]:
        """Find constraints that are active (binding) at the given point."""
        active = []
        for constraint in constraints:
            lhs = sum(
                coeff * var_values.get(var_id, 0)
                for var_id, coeff in constraint.coefficients.items()
            )

            # Consider constraint active if LHS is within 1% of RHS
            tolerance = abs(constraint.rhs) * 0.01 if constraint.rhs != 0 else 0.01
            if abs(lhs - constraint.rhs) <= tolerance:
                active.append(constraint.constraint_id)

        return active

    def _compute_confidence_interval(
        self, value: float, noise_std: float, confidence_level: float
    ) -> ConfidenceInterval:
        """Compute confidence interval for objective value."""
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        half_width = z_score * noise_std

        return ConfidenceInterval(
            lower=value - half_width,
            upper=value + half_width,
            confidence_level=confidence_level
        )

    def _compute_sensitivity(
        self,
        request: OptimisationRequest,
        optimal_vars: Dict[str, float],
        optimal_obj: float,
        feasible_results: List[Tuple[Dict[str, float], float]]
    ) -> OptimisationSensitivity:
        """Compute sensitivity analysis at optimal point."""
        var_ids = [v.variable_id for v in request.decision_variables]
        var_bounds = {
            v.variable_id: (v.lower_bound, v.upper_bound)
            for v in request.decision_variables
        }

        # Compute gradients (partial derivatives)
        gradients = {}
        for var_id in var_ids:
            gradients[var_id] = request.objective.coefficients.get(var_id, 0.0)

        # Compute 5% tolerance range for each variable
        tolerance = abs(optimal_obj) * 0.05 if optimal_obj != 0 else 0.05
        range_5pct: Dict[str, List[float]] = {}

        for var_id in var_ids:
            # Find all feasible points where this variable varies
            # and objective is within 5% of optimum
            lb, ub = var_bounds[var_id]
            min_val, max_val = ub, lb  # Initialize inverted

            for var_values, obj_val in feasible_results:
                if abs(obj_val - optimal_obj) <= tolerance:
                    val = var_values[var_id]
                    min_val = min(min_val, val)
                    max_val = max(max_val, val)

            # If no variation found, use optimal value
            if min_val > max_val:
                min_val = max_val = optimal_vars[var_id]

            range_5pct[var_id] = [min_val, max_val]

        # Compute robustness score
        # Based on how wide the 5% ranges are relative to variable bounds
        relative_ranges = []
        for var_id in var_ids:
            lb, ub = var_bounds[var_id]
            var_range = ub - lb
            if var_range > 0:
                tolerance_range = range_5pct[var_id][1] - range_5pct[var_id][0]
                relative_ranges.append(tolerance_range / var_range)
            else:
                relative_ranges.append(1.0)  # No variation = robust

        robustness_score = np.mean(relative_ranges) if relative_ranges else 0.5

        # Determine robustness level
        if robustness_score >= 0.7:
            robustness = "robust"
        elif robustness_score >= 0.3:
            robustness = "moderate"
        else:
            robustness = "fragile"

        # Find critical variables (highest absolute gradient relative to bounds)
        normalized_gradients = {}
        for var_id in var_ids:
            lb, ub = var_bounds[var_id]
            var_range = ub - lb
            if var_range > 0:
                normalized_gradients[var_id] = abs(gradients[var_id]) * var_range
            else:
                normalized_gradients[var_id] = 0

        # Sort by normalized gradient, take top variables
        sorted_vars = sorted(
            normalized_gradients.items(), key=lambda x: x[1], reverse=True
        )
        critical_threshold = 0.5 * max(normalized_gradients.values()) if normalized_gradients else 0
        critical_variables = [
            var_id for var_id, ng in sorted_vars if ng >= critical_threshold
        ]

        return OptimisationSensitivity(
            range_within_5pct=range_5pct,
            gradient_at_optimum=gradients,
            robustness=robustness,
            robustness_score=robustness_score,
            critical_variables=critical_variables
        )
