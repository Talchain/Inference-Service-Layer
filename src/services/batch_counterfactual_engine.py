"""
Batch Counterfactual Engine for multi-scenario analysis with interaction detection.

Processes multiple counterfactual scenarios efficiently with shared exogenous samples
and detects synergistic/antagonistic interactions between variables.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.models.requests import BatchCounterfactualRequest, ScenarioSpec
from src.models.responses import (
    BatchCounterfactualResponse,
    InteractionAnalysis,
    PairwiseInteraction,
    ScenarioComparison,
    ScenarioResult,
)
from src.models.shared import ExplanationMetadata, RobustnessLevel, StructuralModel
from src.services.counterfactual_engine import CounterfactualEngine
from src.services.explanation_generator import ExplanationGenerator
from src.utils.determinism import canonical_hash, make_deterministic

logger = logging.getLogger(__name__)


class BatchCounterfactualEngine:
    """
    Batch counterfactual analysis with interaction detection.

    Optimizations:
    - Shared exogenous samples across scenarios for determinism
    - Efficient computation via CounterfactualEngine reuse
    - Interaction detection via additive decomposition
    """

    def __init__(
        self,
        cf_engine: Optional[CounterfactualEngine] = None,
        explanation_generator: Optional[ExplanationGenerator] = None,
    ):
        """
        Initialize batch counterfactual engine.

        Args:
            cf_engine: Counterfactual engine for individual scenarios
            explanation_generator: Explanation generator
        """
        self.cf_engine = cf_engine or CounterfactualEngine()
        self.explanation_generator = explanation_generator or ExplanationGenerator()

    def generate_batch_counterfactuals(
        self,
        request: BatchCounterfactualRequest,
        request_id: str = "unknown",
    ) -> BatchCounterfactualResponse:
        """
        Generate batch counterfactual analysis with interaction detection.

        Args:
            request: Batch counterfactual request
            request_id: Request ID for tracing

        Returns:
            BatchCounterfactualResponse with scenarios, interactions, and comparison
        """
        # Make computation deterministic
        seed = make_deterministic(request.model_dump())
        if request.seed is not None:
            seed = request.seed

        np.random.seed(seed)

        logger.info(
            "batch_counterfactual_started",
            extra={
                "request_hash": canonical_hash(request.model_dump()),
                "num_scenarios": len(request.scenarios),
                "outcome": request.outcome,
                "analyze_interactions": request.analyze_interactions,
                "seed": seed,
            },
        )

        try:
            # Process each scenario
            results = []
            for scenario in request.scenarios:
                result = self._process_scenario(
                    model=request.model,
                    scenario=scenario,
                    outcome=request.outcome,
                    seed=seed,
                )
                results.append(result)

            # Interaction analysis
            interactions = None
            if request.analyze_interactions and len(request.scenarios) > 1:
                interactions = self._detect_interactions(
                    model=request.model,
                    scenarios=request.scenarios,
                    results=results,
                    seed=seed,
                )

            # Comparison
            comparison = self._compare_scenarios(results)

            # Explanation
            explanation = self._generate_explanation(
                results=results,
                interactions=interactions,
            )

            logger.info(
                "batch_counterfactual_complete",
                extra={
                    "request_id": request_id,
                    "num_scenarios": len(results),
                    "best_outcome": comparison.best_outcome,
                    "most_robust": comparison.most_robust,
                },
            )

            return BatchCounterfactualResponse(
                scenarios=results,
                interactions=interactions,
                comparison=comparison,
                explanation=explanation,
            )

        except Exception as e:
            logger.error("batch_counterfactual_failed", exc_info=True)
            raise

    def _process_scenario(
        self,
        model: StructuralModel,
        scenario: ScenarioSpec,
        outcome: str,
        seed: int,
    ) -> ScenarioResult:
        """
        Process single scenario using counterfactual engine.

        Args:
            model: Structural causal model
            scenario: Scenario specification
            outcome: Outcome variable
            seed: Random seed

        Returns:
            ScenarioResult with prediction, uncertainty, and robustness
        """
        from src.models.requests import CounterfactualRequest

        # Create counterfactual request
        cf_request = CounterfactualRequest(
            model=model,
            intervention=scenario.intervention,
            outcome=outcome,
            context={},
        )

        # Use existing counterfactual engine
        cf_result = self.cf_engine.analyze(cf_request)

        # Convert to scenario result
        return ScenarioResult(
            scenario_id=scenario.id,
            intervention=scenario.intervention,
            label=scenario.label,
            prediction=cf_result.prediction,
            uncertainty=cf_result.uncertainty,
            robustness=cf_result.robustness,
        )

    def _detect_interactions(
        self,
        model: StructuralModel,
        scenarios: List[ScenarioSpec],
        results: List[ScenarioResult],
        seed: int,
    ) -> InteractionAnalysis:
        """
        Detect synergistic/antagonistic interactions between variables.

        Method:
        1. Identify all variables intervened on
        2. For each pair of variables, find scenarios:
           - Variable A only
           - Variable B only
           - Both A and B together
        3. Test: effect(A+B) vs effect(A) + effect(B)
        4. Classify: synergistic (>), antagonistic (<), or additive (≈)

        Args:
            model: Structural model
            scenarios: List of scenarios
            results: List of scenario results
            seed: Random seed

        Returns:
            InteractionAnalysis with pairwise interactions
        """
        # Build scenario lookup by variables intervened
        scenario_map = {}
        for scenario, result in zip(scenarios, results):
            var_set = frozenset(scenario.intervention.keys())
            scenario_map[var_set] = result

        # Get all variables
        all_vars = set()
        for scenario in scenarios:
            all_vars.update(scenario.intervention.keys())

        # Test pairwise interactions
        pairwise_interactions = []
        for var1, var2 in combinations(sorted(all_vars), 2):
            interaction = self._test_pairwise_interaction(
                var1=var1,
                var2=var2,
                scenario_map=scenario_map,
            )
            if interaction is not None:
                pairwise_interactions.append(interaction)

        # Generate summary
        summary = self._summarize_interactions(pairwise_interactions)

        return InteractionAnalysis(
            pairwise=pairwise_interactions,
            summary=summary,
        )

    def _test_pairwise_interaction(
        self,
        var1: str,
        var2: str,
        scenario_map: Dict[frozenset, ScenarioResult],
    ) -> Optional[PairwiseInteraction]:
        """
        Test if two variables interact.

        Args:
            var1: First variable
            var2: Second variable
            scenario_map: Map of variable sets to scenario results

        Returns:
            PairwiseInteraction if interaction detected, None otherwise
        """
        # Find required scenarios
        var1_only = scenario_map.get(frozenset([var1]))
        var2_only = scenario_map.get(frozenset([var2]))
        both = scenario_map.get(frozenset([var1, var2]))

        # Need all three scenarios to test interaction
        if not (var1_only and var2_only and both):
            return None

        # Compute interaction effect
        effect_var1 = var1_only.prediction.point_estimate
        effect_var2 = var2_only.prediction.point_estimate
        effect_both = both.prediction.point_estimate

        # Expected additive effect
        # Note: This assumes baseline = 0, adjust if needed
        expected_additive = effect_var1 + effect_var2

        # Actual combined effect
        actual = effect_both

        # Interaction = actual - expected_additive
        interaction_effect = actual - expected_additive

        # Classify interaction type
        # Threshold: 5% of expected additive effect
        threshold = abs(expected_additive) * 0.05

        if abs(interaction_effect) < threshold:
            interaction_type = "additive"
        elif interaction_effect > 0:
            interaction_type = "synergistic"
        else:
            interaction_type = "antagonistic"

        # Compute significance (simple heuristic based on effect size)
        significance = min(1.0, abs(interaction_effect) / max(abs(expected_additive), 1.0))

        # Generate explanation
        explanation = self._explain_interaction(
            var1=var1,
            var2=var2,
            interaction_type=interaction_type,
            interaction_effect=interaction_effect,
            effect_var1=effect_var1,
            effect_var2=effect_var2,
            effect_both=effect_both,
        )

        return PairwiseInteraction(
            variables=(var1, var2),
            type=interaction_type,
            effect_size=interaction_effect,
            significance=significance,
            explanation=explanation,
        )

    def _explain_interaction(
        self,
        var1: str,
        var2: str,
        interaction_type: str,
        interaction_effect: float,
        effect_var1: float,
        effect_var2: float,
        effect_both: float,
    ) -> str:
        """
        Generate plain English explanation of interaction.

        Args:
            var1: First variable
            var2: Second variable
            interaction_type: Type of interaction
            interaction_effect: Size of interaction effect
            effect_var1: Effect of var1 only
            effect_var2: Effect of var2 only
            effect_both: Combined effect

        Returns:
            Plain English explanation
        """
        if interaction_type == "synergistic":
            return (
                f"{var1} and {var2} interact synergistically: "
                f"combined effect ({effect_both:.0f}) exceeds sum of individual effects "
                f"({effect_var1:.0f} + {effect_var2:.0f} = {effect_var1 + effect_var2:.0f}) "
                f"by {interaction_effect:.0f}"
            )
        elif interaction_type == "antagonistic":
            return (
                f"{var1} and {var2} interact antagonistically: "
                f"combined effect ({effect_both:.0f}) is less than sum of individual effects "
                f"({effect_var1:.0f} + {effect_var2:.0f} = {effect_var1 + effect_var2:.0f}) "
                f"by {abs(interaction_effect):.0f}"
            )
        else:
            return (
                f"{var1} and {var2} have additive effects: "
                f"combined effect ({effect_both:.0f}) ≈ sum of individual effects "
                f"({effect_var1:.0f} + {effect_var2:.0f} = {effect_var1 + effect_var2:.0f})"
            )

    def _summarize_interactions(
        self,
        interactions: List[PairwiseInteraction],
    ) -> str:
        """
        Generate summary of all interactions.

        Args:
            interactions: List of pairwise interactions

        Returns:
            Plain English summary
        """
        if not interactions:
            return "No significant interactions detected"

        synergistic = [i for i in interactions if i.type == "synergistic"]
        antagonistic = [i for i in interactions if i.type == "antagonistic"]
        additive = [i for i in interactions if i.type == "additive"]

        parts = []

        if synergistic:
            strongest = max(synergistic, key=lambda i: abs(i.effect_size))
            parts.append(
                f"Strong synergistic interaction between {strongest.variables[0]} and "
                f"{strongest.variables[1]} (effect: {strongest.effect_size:+.0f})"
            )

        if antagonistic:
            strongest = max(antagonistic, key=lambda i: abs(i.effect_size))
            parts.append(
                f"Antagonistic interaction between {strongest.variables[0]} and "
                f"{strongest.variables[1]} (effect: {strongest.effect_size:+.0f})"
            )

        if additive:
            parts.append(f"{len(additive)} pair(s) show additive effects")

        return "; ".join(parts)

    def _compare_scenarios(
        self,
        results: List[ScenarioResult],
    ) -> ScenarioComparison:
        """
        Compare scenarios and rank them.

        Args:
            results: List of scenario results

        Returns:
            ScenarioComparison with rankings
        """
        # Find best outcome
        best_outcome = max(results, key=lambda r: r.prediction.point_estimate)

        # Find most robust
        # Map robustness level to score
        robustness_scores = {
            RobustnessLevel.ROBUST: 3,
            RobustnessLevel.MODERATE: 2,
            RobustnessLevel.FRAGILE: 1,
        }

        most_robust = max(
            results,
            key=lambda r: robustness_scores.get(r.robustness.score, 0)
        )

        # Compute marginal gains (assume first scenario is baseline)
        baseline_outcome = results[0].prediction.point_estimate
        marginal_gains = {}
        for result in results[1:]:
            gain = result.prediction.point_estimate - baseline_outcome
            marginal_gains[result.scenario_id] = gain

        # Rank scenarios by outcome
        ranked = sorted(results, key=lambda r: r.prediction.point_estimate, reverse=True)
        ranking = [r.scenario_id for r in ranked]

        return ScenarioComparison(
            best_outcome=best_outcome.scenario_id,
            most_robust=most_robust.scenario_id,
            marginal_gains=marginal_gains,
            ranking=ranking,
        )

    def _generate_explanation(
        self,
        results: List[ScenarioResult],
        interactions: Optional[InteractionAnalysis],
    ) -> ExplanationMetadata:
        """
        Generate overall explanation for batch analysis.

        Args:
            results: List of scenario results
            interactions: Interaction analysis (if any)

        Returns:
            ExplanationMetadata
        """
        # Summary
        best = max(results, key=lambda r: r.prediction.point_estimate)
        baseline = results[0]

        gain = best.prediction.point_estimate - baseline.prediction.point_estimate

        summary = (
            f"Best scenario '{best.scenario_id}' yields {gain:+.0f} gain vs baseline "
            f"with {best.robustness.score.value} robustness"
        )

        # Reasoning
        reasoning_parts = [
            f"Analyzed {len(results)} scenarios across different intervention combinations."
        ]

        if interactions and interactions.pairwise:
            synergistic = [i for i in interactions.pairwise if i.type == "synergistic"]
            if synergistic:
                reasoning_parts.append(
                    f"Detected {len(synergistic)} synergistic interaction(s) between variables."
                )

        reasoning = " ".join(reasoning_parts)

        # Technical basis
        technical_basis = "Batch counterfactual analysis with shared exogenous samples for determinism"
        if interactions:
            technical_basis += "; interaction detection via additive decomposition"

        # Assumptions
        assumptions = [
            "Structural equations capture true causal relationships",
            "Intervention effects are stable across scenarios",
            "No external confounding factors",
        ]

        return ExplanationMetadata(
            summary=summary,
            reasoning=reasoning,
            technical_basis=technical_basis,
            assumptions=assumptions,
        )
