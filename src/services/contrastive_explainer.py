"""
Contrastive Explanation Engine for minimal intervention discovery.

Finds minimal sufficient interventions to achieve target outcomes,
combining binary search with FACET robustness verification.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.models.requests import ContrastiveExplanationRequest, InterventionConstraints
from src.models.responses import (
    ConfidenceInterval,
    ContrastiveExplanationResponse,
    InterventionChange,
    InterventionComparison,
    MinimalIntervention,
)
from src.models.robustness import RobustnessRequest
from src.models.shared import (
    ExplanationMetadata,
    RobustnessLevel,
    StructuralModel,
)
from src.services.counterfactual_engine import CounterfactualEngine
from src.services.explanation_generator import ExplanationGenerator
from src.services.robustness_analyzer import RobustnessAnalyzer
from src.utils.determinism import canonical_hash, make_deterministic
from src.utils.rng import SeededRNG

logger = logging.getLogger(__name__)


class ContrastiveExplainer:
    """
    Find minimal interventions achieving target outcomes.

    Uses binary search to find minimal change magnitudes, then evaluates
    interventions for robustness using FACET analysis.
    """

    def __init__(
        self,
        cf_engine: Optional[CounterfactualEngine] = None,
        robustness_analyzer: Optional[RobustnessAnalyzer] = None,
        explanation_generator: Optional[ExplanationGenerator] = None,
    ) -> None:
        """
        Initialize contrastive explainer.

        Args:
            cf_engine: Counterfactual engine for simulations
            robustness_analyzer: FACET robustness analyzer
            explanation_generator: Explanation generator
        """
        self.cf_engine = cf_engine or CounterfactualEngine()
        self.robustness_analyzer = robustness_analyzer or RobustnessAnalyzer()
        self.explanation_generator = explanation_generator or ExplanationGenerator()

    def find_minimal_interventions(
        self,
        request: ContrastiveExplanationRequest,
        max_candidates: int = 5,
        request_id: str = "unknown",
    ) -> ContrastiveExplanationResponse:
        """
        Find minimal interventions achieving target outcome.

        Algorithm:
        1. For each feasible variable, find minimal single-variable change
        2. If max_changes > 1, find minimal multi-variable combinations
        3. Evaluate all candidates for robustness via FACET
        4. Rank by optimization criterion (cost, change_magnitude, etc.)

        Args:
            request: Contrastive explanation request
            max_candidates: Maximum interventions to return
            request_id: Request ID for tracing

        Returns:
            ContrastiveExplanationResponse with ranked interventions
        """
        # Make computation deterministic using per-request RNG
        if request.seed is not None:
            rng = SeededRNG(request.seed)
        else:
            rng = make_deterministic(request.model_dump())

        logger.info(
            "contrastive_analysis_started",
            extra={
                "request_hash": canonical_hash(request.model_dump()),
                "feasible_vars": request.constraints.feasible,
                "target_outcome": request.target_outcome,
                "seed": rng.seed,
            },
        )

        try:
            candidates = []

            # Step 1: Find single-variable interventions
            for var in request.constraints.feasible:
                intervention = self._find_minimal_single_variable_change(
                    variable=var,
                    current_state=request.current_state,
                    model=request.model,
                    target_outcome=request.target_outcome,
                    constraints=request.constraints,
                    seed=rng.seed,
                )
                if intervention is not None:
                    candidates.append(intervention)

            # Step 2: Find multi-variable interventions (if allowed)
            if request.constraints.max_changes > 1 and len(request.constraints.feasible) > 1:
                multi_interventions = self._find_minimal_multi_variable_combinations(
                    current_state=request.current_state,
                    model=request.model,
                    target_outcome=request.target_outcome,
                    constraints=request.constraints,
                    seed=rng.seed,
                )
                candidates.extend(multi_interventions)

            # Step 3: Evaluate all candidates for robustness
            evaluated = []
            for candidate in candidates:
                eval_result = self._evaluate_intervention(
                    intervention=candidate,
                    model=request.model,
                    current_state=request.current_state,
                    target_outcome=request.target_outcome,
                    constraints=request.constraints,
                    request_id=request_id,
                    seed=rng.seed,
                )
                if eval_result is not None:
                    evaluated.append(eval_result)

            # Step 4: Rank interventions by optimization criterion
            ranked = self._rank_interventions(
                interventions=evaluated,
                criterion=request.constraints.minimize,
            )

            # Limit to max_candidates
            ranked = ranked[:max_candidates]

            # Step 5: Generate comparison and explanation
            comparison = self._generate_comparison(ranked)
            explanation = self._generate_explanation(
                interventions=ranked,
                target_outcome=request.target_outcome,
                current_outcome=request.observed_outcome,
            )

            logger.info(
                "contrastive_analysis_complete",
                extra={
                    "request_id": request_id,
                    "interventions_found": len(ranked),
                    "best_intervention_rank": 1 if ranked else None,
                },
            )

            return ContrastiveExplanationResponse(
                minimal_interventions=ranked,
                comparison=comparison,
                explanation=explanation,
            )

        except Exception as e:
            logger.error("contrastive_analysis_failed", exc_info=True)
            raise

    def _find_minimal_single_variable_change(
        self,
        variable: str,
        current_state: Dict[str, float],
        model: StructuralModel,
        target_outcome: Dict[str, Tuple[float, float]],
        constraints: InterventionConstraints,
        seed: int,
    ) -> Optional[Dict[str, float]]:
        """
        Find minimal change to single variable achieving target.

        Uses binary search to find smallest change magnitude.

        Args:
            variable: Variable to change
            current_state: Current variable values
            model: Structural causal model
            target_outcome: Target outcome ranges
            constraints: Intervention constraints
            seed: Random seed

        Returns:
            Dict mapping variable to minimal value, or None if no solution
        """
        outcome_var = list(target_outcome.keys())[0]
        target_min, target_max = target_outcome[outcome_var]

        current_val = current_state.get(variable)
        if current_val is None:
            logger.warning(f"Variable {variable} not in current_state")
            return None

        # Get variable bounds
        if constraints.variable_bounds and variable in constraints.variable_bounds:
            var_min, var_max = constraints.variable_bounds[variable]
        else:
            # Default: allow ±100% change
            var_min = current_val * 0.1 if current_val > 0 else current_val - abs(current_val) * 2
            var_max = current_val * 3 if current_val > 0 else current_val + abs(current_val) * 2

        # Try both directions (increase and decrease)
        best_intervention = None
        best_distance = float('inf')

        for direction in ['increase', 'decrease']:
            if direction == 'increase':
                low, high = current_val, var_max
            else:
                low, high = var_min, current_val

            # Binary search for minimal change
            best_val = None
            iterations = 0
            max_iterations = 20

            while high - low > abs(current_val) * 0.001 and iterations < max_iterations:
                iterations += 1
                mid = (low + high) / 2

                # Test this intervention
                intervention = {variable: mid}
                outcome = self._simulate_intervention(
                    intervention=intervention,
                    current_state=current_state,
                    model=model,
                    outcome_var=outcome_var,
                    seed=rng.seed,
                )

                if outcome is not None and target_min <= outcome <= target_max:
                    # Target achieved, try smaller change
                    best_val = mid
                    if direction == 'increase':
                        high = mid
                    else:
                        low = mid
                else:
                    # Need larger change
                    if direction == 'increase':
                        low = mid
                    else:
                        high = mid

            if best_val is not None:
                distance = abs(best_val - current_val)
                if distance < best_distance:
                    best_distance = distance
                    best_intervention = {variable: best_val}

        return best_intervention

    def _find_minimal_multi_variable_combinations(
        self,
        current_state: Dict[str, float],
        model: StructuralModel,
        target_outcome: Dict[str, Tuple[float, float]],
        constraints: InterventionConstraints,
        seed: int,
    ) -> List[Dict[str, float]]:
        """
        Find minimal multi-variable interventions.

        Uses greedy search to find combinations of 2+ variables.

        Args:
            current_state: Current variable values
            model: Structural causal model
            target_outcome: Target outcome ranges
            constraints: Intervention constraints
            seed: Random seed

        Returns:
            List of multi-variable intervention dicts
        """
        outcome_var = list(target_outcome.keys())[0]
        target_min, target_max = target_outcome[outcome_var]

        interventions = []

        # Try all pairs if max_changes >= 2
        if constraints.max_changes >= 2:
            feasible = constraints.feasible
            for i in range(len(feasible)):
                for j in range(i + 1, len(feasible)):
                    var1, var2 = feasible[i], feasible[j]

                    # Grid search for minimal combination
                    best_combo = self._grid_search_two_variables(
                        var1=var1,
                        var2=var2,
                        current_state=current_state,
                        model=model,
                        outcome_var=outcome_var,
                        target_min=target_min,
                        target_max=target_max,
                        constraints=constraints,
                        seed=rng.seed,
                    )

                    if best_combo is not None:
                        interventions.append(best_combo)

        return interventions

    def _grid_search_two_variables(
        self,
        var1: str,
        var2: str,
        current_state: Dict[str, float],
        model: StructuralModel,
        outcome_var: str,
        target_min: float,
        target_max: float,
        constraints: InterventionConstraints,
        seed: int,
        grid_points: int = 10,
    ) -> Optional[Dict[str, float]]:
        """
        Grid search to find minimal two-variable intervention.

        Args:
            var1: First variable
            var2: Second variable
            current_state: Current state
            model: Structural model
            outcome_var: Outcome variable
            target_min: Target minimum
            target_max: Target maximum
            constraints: Constraints
            seed: Random seed
            grid_points: Number of grid points per dimension

        Returns:
            Dict with intervention or None
        """
        current_val1 = current_state.get(var1, 0)
        current_val2 = current_state.get(var2, 0)

        # Get bounds
        if constraints.variable_bounds and var1 in constraints.variable_bounds:
            min1, max1 = constraints.variable_bounds[var1]
        else:
            min1, max1 = current_val1 * 0.5, current_val1 * 1.5

        if constraints.variable_bounds and var2 in constraints.variable_bounds:
            min2, max2 = constraints.variable_bounds[var2]
        else:
            min2, max2 = current_val2 * 0.5, current_val2 * 1.5

        # Create grid
        vals1 = np.linspace(min1, max1, grid_points)
        vals2 = np.linspace(min2, max2, grid_points)

        best_intervention = None
        best_distance = float('inf')

        for v1 in vals1:
            for v2 in vals2:
                intervention = {var1: v1, var2: v2}
                outcome = self._simulate_intervention(
                    intervention=intervention,
                    current_state=current_state,
                    model=model,
                    outcome_var=outcome_var,
                    seed=rng.seed,
                )

                if outcome is not None and target_min <= outcome <= target_max:
                    # Compute total distance
                    distance = np.sqrt((v1 - current_val1)**2 + (v2 - current_val2)**2)
                    if distance < best_distance:
                        best_distance = distance
                        best_intervention = intervention

        return best_intervention

    def _simulate_intervention(
        self,
        intervention: Dict[str, float],
        current_state: Dict[str, float],
        model: StructuralModel,
        outcome_var: str,
        seed: int,
    ) -> Optional[float]:
        """
        Simulate intervention and return outcome.

        Args:
            intervention: Intervention values
            current_state: Current state
            model: Structural model
            outcome_var: Outcome variable
            seed: Random seed

        Returns:
            Predicted outcome value or None if simulation fails
        """
        try:
            # Create counterfactual request
            from src.models.requests import CounterfactualRequest

            cf_request = CounterfactualRequest(
                model=model,
                intervention=intervention,
                outcome=outcome_var,
                context=current_state,
            )

            # Run simulation
            result = self.cf_engine.analyze(cf_request)
            return result.prediction.point_estimate

        except Exception as e:
            logger.warning(f"Simulation failed for intervention {intervention}: {e}")
            return None

    def _evaluate_intervention(
        self,
        intervention: Dict[str, float],
        model: StructuralModel,
        current_state: Dict[str, float],
        target_outcome: Dict[str, Tuple[float, float]],
        constraints: InterventionConstraints,
        request_id: str,
        seed: int,
    ) -> Optional[MinimalIntervention]:
        """
        Evaluate intervention for robustness, feasibility, cost.

        Args:
            intervention: Intervention to evaluate
            model: Structural model
            current_state: Current state
            target_outcome: Target outcome
            constraints: Constraints
            request_id: Request ID
            seed: Random seed

        Returns:
            MinimalIntervention or None if evaluation fails
        """
        try:
            outcome_var = list(target_outcome.keys())[0]

            # Simulate to get expected outcome
            expected_outcome = self._simulate_intervention(
                intervention=intervention,
                current_state=current_state,
                model=model,
                outcome_var=outcome_var,
                seed=rng.seed,
            )

            if expected_outcome is None:
                return None

            # Run FACET robustness analysis
            robustness_result = self._analyze_intervention_robustness(
                intervention=intervention,
                model=model,
                target_outcome=target_outcome,
                request_id=request_id,
                seed=rng.seed,
            )

            # Compute feasibility score
            feasibility = self._compute_feasibility(
                intervention=intervention,
                current_state=current_state,
                constraints=constraints,
            )

            # Estimate cost
            cost_estimate = self._estimate_cost(
                intervention=intervention,
                current_state=current_state,
                constraints=constraints,
            )

            # Create intervention changes
            changes = {}
            for var, new_val in intervention.items():
                old_val = current_state.get(var, 0)
                delta = new_val - old_val
                relative_change = (delta / old_val * 100) if old_val != 0 else 0

                changes[var] = InterventionChange(
                    variable=var,
                    from_value=old_val,
                    to_value=new_val,
                    delta=delta,
                    relative_change=relative_change,
                )

            # Get confidence interval from counterfactual
            from src.models.requests import CounterfactualRequest
            cf_request = CounterfactualRequest(
                model=model,
                intervention=intervention,
                outcome=outcome_var,
                context=current_state,
            )
            cf_result = self.cf_engine.analyze(cf_request)

            return MinimalIntervention(
                rank=0,  # Will be set during ranking
                changes=changes,
                expected_outcome={outcome_var: expected_outcome},
                confidence_interval={
                    outcome_var: cf_result.prediction.confidence_interval
                },
                feasibility=feasibility,
                cost_estimate=cost_estimate,
                robustness=robustness_result['level'],
                robustness_score=robustness_result['score'],
            )

        except Exception as e:
            logger.warning(f"Evaluation failed for intervention {intervention}: {e}")
            return None

    def _analyze_intervention_robustness(
        self,
        intervention: Dict[str, float],
        model: StructuralModel,
        target_outcome: Dict[str, Tuple[float, float]],
        request_id: str,
        seed: int,
    ) -> Dict:
        """
        Analyze intervention robustness using FACET.

        Args:
            intervention: Intervention to analyze
            model: Structural model
            target_outcome: Target outcome
            request_id: Request ID
            seed: Random seed

        Returns:
            Dict with robustness level and score
        """
        try:
            # Create robustness request
            robustness_request = RobustnessRequest(
                causal_model={
                    "nodes": model.variables,
                    "edges": [],  # Will be inferred from equations
                },
                intervention_proposal=intervention,
                target_outcome=target_outcome,
                perturbation_radius=0.1,  # ±10% perturbation
                min_samples=100,
                confidence_level=0.95,
                structural_model=model.model_dump(),
            )

            # Analyze robustness
            result = self.robustness_analyzer.analyze_robustness(
                request=robustness_request,
                request_id=request_id,
            )

            # Map to robustness level
            if result.robustness_score >= 0.7:
                level = RobustnessLevel.ROBUST
            elif result.robustness_score >= 0.4:
                level = RobustnessLevel.MODERATE
            else:
                level = RobustnessLevel.FRAGILE

            return {
                'level': level,
                'score': result.robustness_score,
                'analysis': result,
            }

        except Exception as e:
            logger.warning(f"Robustness analysis failed: {e}")
            # Default to moderate robustness if analysis fails
            return {
                'level': RobustnessLevel.MODERATE,
                'score': 0.5,
                'analysis': None,
            }

    def _compute_feasibility(
        self,
        intervention: Dict[str, float],
        current_state: Dict[str, float],
        constraints: InterventionConstraints,
    ) -> float:
        """
        Compute feasibility score for intervention.

        Args:
            intervention: Intervention to evaluate
            current_state: Current state
            constraints: Constraints

        Returns:
            Feasibility score (0-1)
        """
        # Check if variables are feasible
        for var in intervention:
            if var not in constraints.feasible:
                return 0.0

        # Check if within bounds
        if constraints.variable_bounds:
            for var, val in intervention.items():
                if var in constraints.variable_bounds:
                    min_val, max_val = constraints.variable_bounds[var]
                    if val < min_val or val > max_val:
                        return 0.0

        # Compute feasibility based on change magnitude
        # Smaller changes = more feasible
        total_change = 0
        for var, new_val in intervention.items():
            old_val = current_state.get(var, 0)
            if old_val != 0:
                relative_change = abs((new_val - old_val) / old_val)
                total_change += relative_change

        # Normalize to 0-1 (assume changes > 100% are less feasible)
        avg_change = total_change / len(intervention)
        feasibility = max(0.0, min(1.0, 1.0 - (avg_change / 2)))

        return feasibility

    def _estimate_cost(
        self,
        intervention: Dict[str, float],
        current_state: Dict[str, float],
        constraints: InterventionConstraints,
    ) -> str:
        """
        Estimate implementation cost.

        Args:
            intervention: Intervention to evaluate
            current_state: Current state
            constraints: Constraints

        Returns:
            Cost estimate: "low", "medium", or "high"
        """
        # Simple heuristic based on number of changes and magnitude
        num_changes = len(intervention)

        total_relative_change = 0
        for var, new_val in intervention.items():
            old_val = current_state.get(var, 0)
            if old_val != 0:
                total_relative_change += abs((new_val - old_val) / old_val)

        avg_change = total_relative_change / num_changes if num_changes > 0 else 0

        # Cost heuristic
        if num_changes == 1 and avg_change < 0.2:
            return "low"
        elif num_changes <= 2 and avg_change < 0.5:
            return "medium"
        else:
            return "high"

    def _rank_interventions(
        self,
        interventions: List[MinimalIntervention],
        criterion: str,
    ) -> List[MinimalIntervention]:
        """
        Rank interventions by optimization criterion.

        Args:
            interventions: List of interventions to rank
            criterion: Optimization criterion (change_magnitude, cost, feasibility)

        Returns:
            Ranked list of interventions
        """
        if not interventions:
            return []

        # Sort by criterion
        if criterion == "cost":
            # Sort by cost estimate (low < medium < high)
            cost_order = {"low": 0, "medium": 1, "high": 2}
            interventions = sorted(
                interventions,
                key=lambda x: (cost_order.get(x.cost_estimate, 3), -x.robustness_score)
            )
        elif criterion == "feasibility":
            interventions = sorted(
                interventions,
                key=lambda x: (-x.feasibility, -x.robustness_score)
            )
        else:  # change_magnitude (default)
            # Compute total change magnitude for each intervention
            def change_magnitude(intervention: MinimalIntervention) -> float:
                total = 0
                for change in intervention.changes.values():
                    total += abs(change.delta)
                return total

            interventions = sorted(
                interventions,
                key=lambda x: (change_magnitude(x), -x.robustness_score)
            )

        # Assign ranks
        for i, intervention in enumerate(interventions):
            intervention.rank = i + 1

        return interventions

    def _generate_comparison(
        self,
        interventions: List[MinimalIntervention],
    ) -> InterventionComparison:
        """
        Generate comparison of interventions.

        Args:
            interventions: List of ranked interventions

        Returns:
            InterventionComparison
        """
        if not interventions:
            return InterventionComparison(
                best_by_cost=0,
                best_by_robustness=0,
                best_by_feasibility=0,
                synergies="No interventions found",
                tradeoffs="No tradeoffs to analyze",
            )

        # Find best by each criterion
        cost_order = {"low": 0, "medium": 1, "high": 2}
        best_by_cost = min(
            interventions,
            key=lambda x: cost_order.get(x.cost_estimate, 3)
        ).rank

        best_by_robustness = max(
            interventions,
            key=lambda x: x.robustness_score
        ).rank

        best_by_feasibility = max(
            interventions,
            key=lambda x: x.feasibility
        ).rank

        # Generate synergies text
        if len(interventions) == 1:
            synergies = "Single intervention sufficient for target"
        else:
            synergies = "Multiple intervention strategies available; combining may yield diminishing returns"

        # Generate tradeoffs text
        if len(interventions) == 1:
            tradeoffs = "No significant tradeoffs"
        else:
            tradeoffs = self._describe_tradeoffs(interventions)

        return InterventionComparison(
            best_by_cost=best_by_cost,
            best_by_robustness=best_by_robustness,
            best_by_feasibility=best_by_feasibility,
            synergies=synergies,
            tradeoffs=tradeoffs,
        )

    def _describe_tradeoffs(
        self,
        interventions: List[MinimalIntervention],
    ) -> str:
        """
        Describe key tradeoffs between interventions.

        Args:
            interventions: List of interventions

        Returns:
            Tradeoff description
        """
        descriptions = []

        # Compare top 2 interventions
        if len(interventions) >= 2:
            int1, int2 = interventions[0], interventions[1]

            # Cost vs robustness
            if int1.cost_estimate != int2.cost_estimate:
                descriptions.append(
                    f"Intervention {int1.rank} is {int1.cost_estimate} cost but "
                    f"{int1.robustness.value} robustness; "
                    f"Intervention {int2.rank} is {int2.cost_estimate} cost but "
                    f"{int2.robustness.value} robustness"
                )

        if descriptions:
            return "; ".join(descriptions)
        else:
            return "Interventions have similar tradeoff profiles"

    def _generate_explanation(
        self,
        interventions: List[MinimalIntervention],
        target_outcome: Dict[str, Tuple[float, float]],
        current_outcome: Dict[str, float],
    ) -> ExplanationMetadata:
        """
        Generate plain English explanation.

        Args:
            interventions: List of interventions
            target_outcome: Target outcome
            current_outcome: Current outcome

        Returns:
            ExplanationMetadata
        """
        if not interventions:
            return ExplanationMetadata(
                summary="No interventions found to achieve target outcome",
                reasoning="The target outcome may not be achievable given the constraints, "
                "or the structural model may not support the required changes.",
                technical_basis="Binary search with FACET robustness verification",
                assumptions=["Structural equations are correct", "Constraints are accurate"],
            )

        best = interventions[0]
        outcome_var = list(target_outcome.keys())[0]
        target_min, target_max = target_outcome[outcome_var]

        # Describe the best intervention
        change_descriptions = []
        for var, change in best.changes.items():
            change_descriptions.append(
                f"{var} from {change.from_value:.1f} to {change.to_value:.1f} "
                f"({change.delta:+.1f}, {change.relative_change:+.1f}%)"
            )

        changes_text = ", ".join(change_descriptions)

        summary = f"Change {changes_text} to achieve target {outcome_var}"

        expected = best.expected_outcome[outcome_var]
        current = current_outcome.get(outcome_var, 0)
        gain = expected - current

        reasoning = (
            f"The recommended intervention yields an expected {outcome_var} of {expected:.0f}, "
            f"a gain of {gain:+.0f} from current {current:.0f}. "
            f"This intervention is {best.cost_estimate} cost with {best.robustness.value} robustness."
        )

        if len(interventions) > 1:
            reasoning += f" {len(interventions)} alternative strategies are available."

        return ExplanationMetadata(
            summary=summary,
            reasoning=reasoning,
            technical_basis="Binary search for minimal change with FACET robustness verification",
            assumptions=[
                "Structural equations capture true causal relationships",
                "Variable values can be changed as specified",
                "No external confounding factors",
            ],
        )
