"""
Utility Function Validator Service.

Validates utility function specifications including:
- Goal weight normalization
- Aggregation method compatibility
- Risk tolerance parameters
- Graph reference validation
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from src.models.requests import (
    AggregationMethod,
    GoalSpecification,
    RiskTolerance,
    UtilityFunctionSpec,
    UtilityValidationRequest,
)
from src.models.responses import (
    NormalisedGoal,
    UtilityValidationResponse,
    UtilityValidationWarning,
)
from src.models.shared import GraphV1, NodeKind

logger = logging.getLogger(__name__)


class UtilityValidator:
    """
    Service for validating utility function specifications.

    Handles:
    - Weight normalization (defaults to equal weights if not specified)
    - Aggregation method validation
    - Risk tolerance parameter checks
    - Graph goal reference validation
    """

    # Constants for validation
    WEIGHT_SUM_TOLERANCE = 0.001
    MIN_WEIGHT = 0.001
    MAX_RISK_COEFFICIENT = 10.0

    def validate(self, request: UtilityValidationRequest) -> UtilityValidationResponse:
        """
        Validate a utility function specification.

        Args:
            request: Utility validation request

        Returns:
            UtilityValidationResponse with validation results
        """
        spec = request.utility_spec
        warnings: List[UtilityValidationWarning] = []
        errors: List[str] = []
        defaults_applied: List[str] = []

        # Step 1: Extract and validate goal IDs
        goal_ids = [g.goal_id for g in spec.goals]

        # Step 2: Check graph references if provided
        if request.graph:
            graph_issues = self._validate_graph_references(spec.goals, request.graph)
            for issue in graph_issues:
                warnings.append(
                    UtilityValidationWarning(
                        code="GRAPH_REFERENCE",
                        message=issue,
                        affected_goals=goal_ids,
                    )
                )

        # Step 3: Process weights
        weights, weight_warnings, weight_defaults = self._process_weights(
            spec.goals, request.strict_mode
        )
        warnings.extend(weight_warnings)
        defaults_applied.extend(weight_defaults)

        # Step 4: Validate aggregation method compatibility
        method_issues = self._validate_aggregation_method(spec, weights)
        for issue in method_issues:
            if issue.startswith("ERROR:"):
                errors.append(issue[6:].strip())
            else:
                warnings.append(
                    UtilityValidationWarning(
                        code="AGGREGATION_METHOD",
                        message=issue,
                        affected_goals=goal_ids,
                    )
                )

        # Step 5: Validate risk tolerance
        risk_issues = self._validate_risk_tolerance(spec)
        for issue in risk_issues:
            warnings.append(
                UtilityValidationWarning(
                    code="RISK_TOLERANCE",
                    message=issue,
                    affected_goals=None,
                )
            )

        # Step 6: Build normalised goals
        normalised_goals = self._build_normalised_goals(spec.goals, weights)

        # Determine overall validity
        is_valid = len(errors) == 0

        return UtilityValidationResponse(
            valid=is_valid,
            normalised_weights=weights,
            normalised_goals=normalised_goals,
            aggregation_method=spec.aggregation_method.value,
            risk_tolerance=spec.risk_tolerance.value,
            default_behaviour_applied=defaults_applied,
            warnings=warnings,
            errors=errors,
        )

    def _validate_graph_references(
        self, goals: List[GoalSpecification], graph: GraphV1
    ) -> List[str]:
        """
        Validate that goals reference nodes in the graph.

        Args:
            goals: List of goal specifications
            graph: Graph to validate against

        Returns:
            List of issues found
        """
        issues = []

        # Extract node IDs and goal nodes from graph
        node_ids = {node.id for node in graph.nodes}
        goal_nodes = {
            node.id for node in graph.nodes if node.kind == NodeKind.GOAL
        }

        for goal in goals:
            # Check if goal_id references a node
            if goal.goal_id not in node_ids:
                issues.append(
                    f"Goal '{goal.goal_id}' does not reference any node in the graph"
                )
            elif goal.goal_id not in goal_nodes:
                # Warning if it references a non-goal node
                issues.append(
                    f"Goal '{goal.goal_id}' references a node that is not of type 'goal'"
                )

        return issues

    def _process_weights(
        self, goals: List[GoalSpecification], strict_mode: bool
    ) -> Tuple[Dict[str, float], List[UtilityValidationWarning], List[str]]:
        """
        Process and normalize goal weights.

        Default behaviour when no weights specified:
        - Equal weighting across all goals (1/n each)

        Args:
            goals: List of goal specifications
            strict_mode: If true, require all weights to be specified

        Returns:
            Tuple of (normalized_weights, warnings, defaults_applied)
        """
        warnings = []
        defaults_applied = []
        n = len(goals)

        # Extract weights (None if not specified)
        raw_weights = {g.goal_id: g.weight for g in goals}

        # Check how many weights are specified
        specified_weights = {k: v for k, v in raw_weights.items() if v is not None}
        unspecified = [k for k, v in raw_weights.items() if v is None]

        if len(unspecified) == n:
            # No weights specified - use equal weighting
            equal_weight = 1.0 / n
            weights = {g.goal_id: equal_weight for g in goals}
            defaults_applied.append(
                f"Equal weighting applied: {equal_weight:.4f} for each of {n} goals"
            )
            warnings.append(
                UtilityValidationWarning(
                    code="DEFAULT_WEIGHTS",
                    message=f"No weights specified - using equal weighting (1/{n})",
                    affected_goals=list(raw_weights.keys()),
                )
            )
        elif len(unspecified) > 0:
            # Some weights specified, some not
            if strict_mode:
                # In strict mode, this is an error handled elsewhere
                # Assign minimum weight to unspecified for now
                for goal_id in unspecified:
                    raw_weights[goal_id] = self.MIN_WEIGHT
                warnings.append(
                    UtilityValidationWarning(
                        code="MISSING_WEIGHTS",
                        message=f"Strict mode: weights required for all goals. Missing: {unspecified}",
                        affected_goals=unspecified,
                    )
                )

            # Distribute remaining weight proportionally
            total_specified = sum(specified_weights.values())
            remaining = max(0, 1.0 - total_specified)

            if remaining > 0 and unspecified:
                per_unspecified = remaining / len(unspecified)
                for goal_id in unspecified:
                    raw_weights[goal_id] = per_unspecified
                defaults_applied.append(
                    f"Unspecified weights assigned {per_unspecified:.4f} each"
                )
            else:
                # All weight already allocated - use minimum
                for goal_id in unspecified:
                    raw_weights[goal_id] = self.MIN_WEIGHT
                defaults_applied.append(
                    f"Unspecified weights assigned minimum ({self.MIN_WEIGHT})"
                )

            weights = raw_weights
        else:
            # All weights specified
            weights = specified_weights

        # Normalize if weights don't sum to 1.0
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > self.WEIGHT_SUM_TOLERANCE:
            original_sum = weight_sum
            weights = {k: v / weight_sum for k, v in weights.items()}
            warnings.append(
                UtilityValidationWarning(
                    code="WEIGHTS_NORMALIZED",
                    message=f"Weights normalized from sum={original_sum:.4f} to sum=1.0",
                    affected_goals=list(weights.keys()),
                )
            )
            defaults_applied.append(
                f"Weight normalization applied (original sum: {original_sum:.4f})"
            )

        return weights, warnings, defaults_applied

    def _validate_aggregation_method(
        self, spec: UtilityFunctionSpec, weights: Dict[str, float]
    ) -> List[str]:
        """
        Validate aggregation method compatibility.

        Args:
            spec: Utility function specification
            weights: Normalized weights

        Returns:
            List of issues (prefixed with ERROR: for errors)
        """
        issues = []
        method = spec.aggregation_method

        # Check lexicographic requires priorities
        if method == AggregationMethod.LEXICOGRAPHIC:
            has_priority = any(g.priority is not None for g in spec.goals)
            if not has_priority:
                issues.append(
                    "Lexicographic method works best with explicit priorities. "
                    "Will use weights to infer priority order."
                )

        # Check weighted_product with negative values
        if method == AggregationMethod.WEIGHTED_PRODUCT:
            has_minimize = any(g.direction == "minimize" for g in spec.goals)
            if has_minimize:
                issues.append(
                    "Weighted product with 'minimize' goals requires value transformation. "
                    "Values will be inverted before aggregation."
                )

        # Warn if very unequal weights with certain methods
        if weights:
            max_weight = max(weights.values())
            min_weight = min(weights.values())
            if max_weight > 0.9 and method == AggregationMethod.WEIGHTED_SUM:
                dominant = [k for k, v in weights.items() if v > 0.9]
                issues.append(
                    f"Goal(s) {dominant} have >90% of total weight - "
                    "other goals have minimal influence"
                )

        return issues

    def _validate_risk_tolerance(self, spec: UtilityFunctionSpec) -> List[str]:
        """
        Validate risk tolerance parameters.

        Args:
            spec: Utility function specification

        Returns:
            List of issues
        """
        issues = []

        # Check risk coefficient for risk_averse
        if spec.risk_tolerance == RiskTolerance.RISK_AVERSE:
            if spec.risk_coefficient is None:
                issues.append(
                    "Risk averse mode specified but no risk_coefficient provided. "
                    "Using default coefficient of 1.0"
                )
            elif spec.risk_coefficient == 0:
                issues.append(
                    "Risk coefficient of 0 effectively means risk_neutral behaviour"
                )
            elif spec.risk_coefficient > 5.0:
                issues.append(
                    f"High risk coefficient ({spec.risk_coefficient}) may cause "
                    "extreme penalties for variance"
                )

        # Check risk coefficient for risk_seeking
        if spec.risk_tolerance == RiskTolerance.RISK_SEEKING:
            if spec.risk_coefficient is not None and spec.risk_coefficient > 2.0:
                issues.append(
                    f"High risk coefficient ({spec.risk_coefficient}) with risk_seeking "
                    "may cause extreme preference for variance"
                )

        # Warn if risk_neutral but coefficient provided
        if spec.risk_tolerance == RiskTolerance.RISK_NEUTRAL:
            if spec.risk_coefficient is not None and spec.risk_coefficient != 0:
                issues.append(
                    "Risk coefficient provided but risk_tolerance is 'risk_neutral' - "
                    "coefficient will be ignored"
                )

        return issues

    def _build_normalised_goals(
        self, goals: List[GoalSpecification], weights: Dict[str, float]
    ) -> List[NormalisedGoal]:
        """
        Build list of normalised goal specifications.

        Args:
            goals: Original goal specifications
            weights: Normalized weights

        Returns:
            List of NormalisedGoal objects
        """
        normalised = []
        for goal in goals:
            normalised.append(
                NormalisedGoal(
                    goal_id=goal.goal_id,
                    label=goal.label,
                    direction=goal.direction,
                    normalised_weight=weights.get(goal.goal_id, 0.0),
                    original_weight=goal.weight,
                    priority=goal.priority,
                )
            )
        return normalised
