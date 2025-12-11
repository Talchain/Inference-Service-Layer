"""
Feasibility Checker Service.

Validates constraint specifications and checks option feasibility
against constraints.
"""

import logging
from typing import List, Tuple

from src.models.requests import (
    ConstraintNode,
    ConstraintRelation,
    FeasibilityRequest,
    OptionForFeasibility,
)
from src.models.responses import (
    ConstraintValidationResult,
    ConstraintViolation,
    FeasibilityResponse,
    FeasibilityResult,
    InfeasibleOption,
)

logger = logging.getLogger(__name__)


class FeasibilityChecker:
    """
    Service for checking constraint feasibility.

    Validates constraint specifications and evaluates which options
    satisfy or violate constraints.
    """

    def __init__(self) -> None:
        """Initialize the feasibility checker."""
        pass

    def check_feasibility(self, request: FeasibilityRequest) -> FeasibilityResponse:
        """
        Check feasibility of options against constraints.

        Args:
            request: Feasibility check request with constraints and options

        Returns:
            FeasibilityResponse with validation and feasibility results
        """
        # Step 1: Validate constraint specifications
        constraint_validation = self._validate_constraints(
            request.constraints, request.options
        )

        # Step 2: Check feasibility of each option
        feasible_options: List[str] = []
        infeasible_options: List[InfeasibleOption] = []

        for option in request.options:
            violations = self._check_option_feasibility(
                option, request.constraints, request.include_partial_violations
            )

            if violations:
                infeasible_options.append(
                    InfeasibleOption(
                        option_id=option.option_id,
                        violated_constraints=[v.constraint_id for v in violations],
                        violation_details=violations,
                        total_violation_magnitude=sum(
                            v.violation_magnitude for v in violations
                        ),
                    )
                )
            else:
                feasible_options.append(option.option_id)

        # Step 3: Generate summary and warnings
        total_options = len(request.options)
        feasible_count = len(feasible_options)
        summary = f"{feasible_count} of {total_options} options are feasible"

        warnings = self._generate_warnings(
            constraint_validation, feasible_options, infeasible_options
        )

        return FeasibilityResponse(
            constraint_validation=constraint_validation,
            feasibility=FeasibilityResult(
                feasible_options=feasible_options,
                infeasible_options=infeasible_options,
            ),
            summary=summary,
            warnings=warnings,
        )

    def _validate_constraints(
        self, constraints: List[ConstraintNode], options: List[OptionForFeasibility]
    ) -> List[ConstraintValidationResult]:
        """
        Validate constraint specifications.

        Checks:
        - Constraint nodes properly connected (target variable exists)
        - Threshold values valid (not NaN, not infinite for practical purposes)
        - Constraint types supported

        Args:
            constraints: List of constraints to validate
            options: Options to check variable references against

        Returns:
            List of validation results for each constraint
        """
        results = []

        # Collect all variable names from options
        all_variables = set()
        for option in options:
            all_variables.update(option.variable_values.keys())

        for constraint in constraints:
            issues = []

            # Check 1: Target variable exists in at least one option
            if constraint.target_variable not in all_variables:
                issues.append(
                    f"Target variable '{constraint.target_variable}' not found in any option"
                )

            # Check 2: Threshold is valid
            if not isinstance(constraint.threshold, (int, float)):
                issues.append("Threshold must be a numeric value")
            elif constraint.threshold != constraint.threshold:  # NaN check
                issues.append("Threshold cannot be NaN")
            elif abs(constraint.threshold) > 1e15:
                issues.append("Threshold value appears impractically large")

            # Check 3: Constraint type is supported
            # (Already validated by Pydantic, but double-check)
            supported_types = ["threshold", "budget", "capacity", "dependency", "exclusion", "requirement"]
            if constraint.constraint_type.value not in supported_types:
                issues.append(f"Unsupported constraint type: {constraint.constraint_type}")

            # Check 4: Relation is valid
            supported_relations = ["le", "ge", "eq", "lt", "gt"]
            if constraint.relation.value not in supported_relations:
                issues.append(f"Unsupported relation: {constraint.relation}")

            results.append(
                ConstraintValidationResult(
                    constraint_id=constraint.id,
                    is_valid=len(issues) == 0,
                    issues=issues,
                )
            )

        return results

    def _check_option_feasibility(
        self,
        option: OptionForFeasibility,
        constraints: List[ConstraintNode],
        include_partial: bool,
    ) -> List[ConstraintViolation]:
        """
        Check if an option satisfies all constraints.

        Args:
            option: Option to check
            constraints: Constraints to evaluate
            include_partial: Whether to include soft constraint violations

        Returns:
            List of constraint violations (empty if feasible)
        """
        violations = []

        for constraint in constraints:
            # Skip if target variable not in option
            if constraint.target_variable not in option.variable_values:
                continue

            actual_value = option.variable_values[constraint.target_variable]
            threshold = constraint.threshold
            is_hard = constraint.priority == "hard"

            # Skip soft constraints if not requested
            if not is_hard and not include_partial:
                continue

            # Check constraint based on relation
            violated, magnitude = self._evaluate_constraint(
                actual_value, threshold, constraint.relation
            )

            if violated:
                violations.append(
                    ConstraintViolation(
                        constraint_id=constraint.id,
                        constraint_label=constraint.label,
                        actual_value=actual_value,
                        threshold=threshold,
                        violation_magnitude=magnitude,
                        is_hard_violation=is_hard,
                    )
                )

        return violations

    def _evaluate_constraint(
        self, actual: float, threshold: float, relation: ConstraintRelation
    ) -> Tuple[bool, float]:
        """
        Evaluate a single constraint.

        Args:
            actual: Actual value
            threshold: Constraint threshold
            relation: Constraint relation operator

        Returns:
            Tuple of (violated, magnitude)
        """
        if relation == ConstraintRelation.LE:
            # actual <= threshold
            violated = actual > threshold
            magnitude = max(0, actual - threshold)
        elif relation == ConstraintRelation.GE:
            # actual >= threshold
            violated = actual < threshold
            magnitude = max(0, threshold - actual)
        elif relation == ConstraintRelation.EQ:
            # actual == threshold (with small tolerance)
            tolerance = abs(threshold) * 0.001 if threshold != 0 else 0.001
            violated = abs(actual - threshold) > tolerance
            magnitude = abs(actual - threshold)
        elif relation == ConstraintRelation.LT:
            # actual < threshold
            violated = actual >= threshold
            magnitude = max(0, actual - threshold + 0.001)
        elif relation == ConstraintRelation.GT:
            # actual > threshold
            violated = actual <= threshold
            magnitude = max(0, threshold - actual + 0.001)
        else:
            violated = False
            magnitude = 0

        return violated, magnitude

    def _generate_warnings(
        self,
        validation_results: List[ConstraintValidationResult],
        feasible: List[str],
        infeasible: List[InfeasibleOption],
    ) -> List[str]:
        """
        Generate warnings based on analysis results.

        Args:
            validation_results: Constraint validation results
            feasible: List of feasible option IDs
            infeasible: List of infeasible options

        Returns:
            List of warning messages
        """
        warnings = []

        # Warning if any constraints are invalid
        invalid_constraints = [r for r in validation_results if not r.is_valid]
        if invalid_constraints:
            warnings.append(
                f"{len(invalid_constraints)} constraint(s) have validation issues"
            )

        # Warning if no feasible options
        if not feasible:
            warnings.append("No feasible options found - all options violate at least one constraint")

        # Warning if all options are feasible (constraints may be too loose)
        if not infeasible and len(feasible) > 1:
            warnings.append(
                "All options are feasible - constraints may not be sufficiently restrictive"
            )

        # Warning for close violations (within 10% of threshold)
        for inf_opt in infeasible:
            for violation in inf_opt.violation_details:
                if violation.threshold != 0:
                    relative_violation = violation.violation_magnitude / abs(violation.threshold)
                    if relative_violation < 0.1:
                        warnings.append(
                            f"Option '{inf_opt.option_id}' narrowly violates "
                            f"constraint '{violation.constraint_id}' (by {relative_violation:.1%})"
                        )

        return warnings
