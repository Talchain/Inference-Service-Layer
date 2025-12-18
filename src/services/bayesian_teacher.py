"""
Bayesian Teacher service implementing optimal teaching strategies.

Selects pedagogically valuable examples that maximize learning efficiency
using information-theoretic principles from Bayesian teaching.

Based on: Optimal Bayesian Teaching (Zhu et al.)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy

from src.config import get_settings
from src.models.phase1_models import (
    DecisionContext,
    Scenario,
    TeachingExample,
    UserBeliefModel,
)
from src.models.shared import ExplanationMetadata
from src.utils.determinism import make_deterministic

logger = logging.getLogger(__name__)
settings = get_settings()


class BayesianTeacher:
    """
    Implements Bayesian teaching for efficient concept learning.

    Key insight: Select examples that maximize posterior probability
    of correct understanding given current beliefs.

    Teaching value: P(understanding | example, current_beliefs)
    """

    def __init__(self) -> None:
        """Initialize the teacher."""
        self.monte_carlo_samples = 500  # Fewer samples than elicitation

        # Concept templates
        self.concepts = {
            "confounding": {
                "description": "Understanding confounding variables and their impact",
                "key_patterns": ["hidden_cause", "spurious_correlation"],
            },
            "trade_offs": {
                "description": "Understanding trade-offs between competing objectives",
                "key_patterns": ["inverse_relationship", "pareto_optimal"],
            },
            "causal_mechanism": {
                "description": "Understanding causal mechanisms and pathways",
                "key_patterns": ["direct_effect", "mediation", "moderation"],
            },
            "uncertainty": {
                "description": "Understanding uncertainty and risk",
                "key_patterns": ["variance", "tail_risk", "probabilistic_outcomes"],
            },
            "optimization": {
                "description": "Understanding optimization and constraints",
                "key_patterns": ["constraint_satisfaction", "local_optima"],
            },
        }

    def generate_teaching_examples(
        self,
        target_concept: str,
        current_beliefs: UserBeliefModel,
        context: DecisionContext,
        max_examples: int,
    ) -> Tuple[List[TeachingExample], str, List[str], str]:
        """
        Generate pedagogically optimal teaching examples.

        Uses Bayesian teaching to select examples that maximize
        expected learning about the target concept.

        Args:
            target_concept: Concept to teach
            current_beliefs: User's current beliefs
            context: Decision context
            max_examples: Maximum number of examples

        Returns:
            Tuple of (examples, explanation, learning_objectives, estimated_time)
        """
        # Make computation deterministic
        rng = make_deterministic(
            {
                "concept": target_concept,
                "beliefs": current_beliefs.model_dump(),
                "context": context.model_dump(),
            }
        )

        logger.info(
            "teaching_generation_started",
            extra={
                "concept": target_concept,
                "num_variables": len(context.variables),
                "max_examples": max_examples,
                "seed": rng.seed,
            },
        )

        # Validate concept
        if target_concept not in self.concepts:
            logger.warning(
                "unknown_concept",
                extra={"concept": target_concept},
            )
            # Default to trade_offs
            target_concept = "trade_offs"

        # Generate candidate examples
        candidates = self._generate_candidate_examples(
            target_concept, context, current_beliefs
        )

        logger.info(
            "generated_teaching_candidates",
            extra={"num_candidates": len(candidates)},
        )

        # Rank by teaching value
        ranked_examples = self._rank_by_teaching_value(
            candidates, target_concept, current_beliefs
        )

        # Select top examples
        selected_examples = ranked_examples[:max_examples]

        # Generate explanation
        explanation = self._generate_teaching_explanation(
            target_concept, selected_examples, current_beliefs
        )

        # Define learning objectives
        learning_objectives = self._define_learning_objectives(
            target_concept, selected_examples
        )

        # Estimate learning time
        estimated_time = self._estimate_learning_time(target_concept, max_examples)

        logger.info(
            "teaching_generation_complete",
            extra={
                "concept": target_concept,
                "num_examples": len(selected_examples),
                "avg_teaching_value": np.mean(
                    [ex.information_value for ex in selected_examples]
                ),
            },
        )

        return selected_examples, explanation, learning_objectives, estimated_time

    def _generate_candidate_examples(
        self,
        concept: str,
        context: DecisionContext,
        beliefs: UserBeliefModel,
    ) -> List[TeachingExample]:
        """
        Generate candidate teaching examples for the concept.

        Args:
            concept: Target concept
            context: Decision context
            beliefs: User beliefs

        Returns:
            List of candidate teaching examples
        """
        candidates = []

        # Generate different types of examples based on concept
        if concept == "confounding":
            candidates.extend(self._generate_confounding_examples(context, beliefs))
        elif concept == "trade_offs":
            candidates.extend(self._generate_tradeoff_examples(context, beliefs))
        elif concept == "causal_mechanism":
            candidates.extend(self._generate_causal_examples(context, beliefs))
        elif concept == "uncertainty":
            candidates.extend(self._generate_uncertainty_examples(context, beliefs))
        elif concept == "optimization":
            candidates.extend(self._generate_optimization_examples(context, beliefs))

        return candidates

    def _generate_confounding_examples(
        self, context: DecisionContext, beliefs: UserBeliefModel
    ) -> List[TeachingExample]:
        """Generate examples illustrating confounding."""
        examples = []

        # Example 1: Classic confounding scenario
        if len(context.variables) >= 2:
            var1, var2 = context.variables[0], context.variables[1]
            scenario = Scenario(
                description=f"Scenario showing potential confounding between {var1} and {var2}",
                outcomes={
                    var: np.random.uniform(0, 100) for var in context.variables
                },
                trade_offs=[
                    f"Apparent correlation between {var1} and {var2}",
                    "May be driven by hidden common cause",
                ],
            )

            examples.append(
                TeachingExample(
                    scenario=scenario,
                    key_insight="Correlation doesn't imply causation - look for confounders",
                    why_this_example="Demonstrates importance of controlling for common causes",
                    information_value=0.0,  # Will be computed later
                )
            )

        # Example 2: Controlling for confounders
        if len(context.variables) >= 3:
            var1, var2, var3 = context.variables[:3]
            scenario = Scenario(
                description=f"Comparing scenarios with/without controlling for {var3}",
                outcomes={
                    var: np.random.uniform(0, 100) for var in context.variables
                },
                trade_offs=[
                    f"Relationship between {var1} and {var2} changes when considering {var3}",
                    "Illustrates Simpson's paradox",
                ],
            )

            examples.append(
                TeachingExample(
                    scenario=scenario,
                    key_insight="Controlling for confounders reveals true causal effects",
                    why_this_example="Shows how proper adjustment set changes conclusions",
                    information_value=0.0,
                )
            )

        return examples

    def _generate_tradeoff_examples(
        self, context: DecisionContext, beliefs: UserBeliefModel
    ) -> List[TeachingExample]:
        """Generate examples illustrating trade-offs."""
        examples = []

        # Example 1: Clear trade-off between two objectives
        if len(context.variables) >= 2:
            var1, var2 = context.variables[0], context.variables[1]

            # Get weights to understand user preferences
            weight1 = beliefs.value_weights.get(var1)
            weight2 = beliefs.value_weights.get(var2)

            # Create scenario with inverse relationship
            high_var1 = np.random.uniform(70, 90)
            low_var2 = np.random.uniform(10, 30)

            scenario = Scenario(
                description=f"Scenario showing trade-off: high {var1}, low {var2}",
                outcomes={
                    var1: high_var1,
                    var2: low_var2,
                    **{v: np.random.uniform(40, 60) for v in context.variables[2:]},
                },
                trade_offs=[
                    f"Optimizing {var1} reduces {var2}",
                    "Cannot maximize both simultaneously",
                ],
            )

            examples.append(
                TeachingExample(
                    scenario=scenario,
                    key_insight="Trade-offs require prioritizing between competing objectives",
                    why_this_example="Illustrates Pareto optimality - improving one worsens another",
                    information_value=0.0,
                )
            )

        # Example 2: Multi-way trade-offs
        if len(context.variables) >= 3:
            scenario = Scenario(
                description="Complex scenario with multiple trade-offs",
                outcomes={var: np.random.uniform(0, 100) for var in context.variables},
                trade_offs=[
                    f"Balancing {len(context.variables)} competing objectives",
                    "No single optimal solution exists",
                ],
            )

            examples.append(
                TeachingExample(
                    scenario=scenario,
                    key_insight="Real decisions involve multiple competing trade-offs",
                    why_this_example="Demonstrates complexity of multi-objective optimization",
                    information_value=0.0,
                )
            )

        return examples

    def _generate_causal_examples(
        self, context: DecisionContext, beliefs: UserBeliefModel
    ) -> List[TeachingExample]:
        """Generate examples illustrating causal mechanisms."""
        examples = []

        # Example: Direct vs. mediated effects
        if len(context.variables) >= 3:
            var1, var2, var3 = context.variables[:3]
            scenario = Scenario(
                description=f"{var1} affects {var3} both directly and through {var2}",
                outcomes={var: np.random.uniform(0, 100) for var in context.variables},
                trade_offs=[
                    f"Direct effect: {var1} → {var3}",
                    f"Indirect effect: {var1} → {var2} → {var3}",
                ],
            )

            examples.append(
                TeachingExample(
                    scenario=scenario,
                    key_insight="Effects can be direct, indirect, or both",
                    why_this_example="Illustrates mediation and multiple causal pathways",
                    information_value=0.0,
                )
            )

        return examples

    def _generate_uncertainty_examples(
        self, context: DecisionContext, beliefs: UserBeliefModel
    ) -> List[TeachingExample]:
        """Generate examples illustrating uncertainty."""
        examples = []

        # Example: Variance and risk
        if len(context.variables) >= 1:
            var = context.variables[0]
            scenario = Scenario(
                description=f"High expected {var} but also high variance",
                outcomes={v: np.random.uniform(0, 100) for v in context.variables},
                trade_offs=[
                    "Higher expected value",
                    "Greater uncertainty and risk",
                ],
            )

            examples.append(
                TeachingExample(
                    scenario=scenario,
                    key_insight="Expected value doesn't tell the whole story - consider variance",
                    why_this_example="Demonstrates risk-return trade-off",
                    information_value=0.0,
                )
            )

        return examples

    def _generate_optimization_examples(
        self, context: DecisionContext, beliefs: UserBeliefModel
    ) -> List[TeachingExample]:
        """Generate examples illustrating optimization."""
        examples = []

        # Example: Constrained optimization
        if context.constraints and len(context.variables) >= 1:
            scenario = Scenario(
                description="Optimal solution subject to constraints",
                outcomes={v: np.random.uniform(0, 100) for v in context.variables},
                trade_offs=[
                    "Maximizing objectives",
                    f"Subject to constraints: {list(context.constraints.keys())}",
                ],
            )

            examples.append(
                TeachingExample(
                    scenario=scenario,
                    key_insight="Real optimization must respect constraints",
                    why_this_example="Shows difference between unconstrained and constrained optima",
                    information_value=0.0,
                )
            )

        return examples

    def _rank_by_teaching_value(
        self,
        candidates: List[TeachingExample],
        concept: str,
        beliefs: UserBeliefModel,
    ) -> List[TeachingExample]:
        """
        Rank teaching examples by their pedagogical value.

        Teaching value measures how much the example helps user
        understand the concept given their current beliefs.

        Args:
            candidates: Candidate examples
            concept: Target concept
            beliefs: User beliefs

        Returns:
            Ranked examples (descending teaching value)
        """
        # Compute teaching value for each candidate
        for example in candidates:
            example.information_value = self._compute_teaching_value(
                example, concept, beliefs
            )

        # Sort by teaching value (descending)
        return sorted(candidates, key=lambda x: x.information_value, reverse=True)

    def _compute_teaching_value(
        self, example: TeachingExample, concept: str, beliefs: UserBeliefModel
    ) -> float:
        """
        Compute teaching value of an example.

        Teaching value = Expected reduction in uncertainty about concept

        Args:
            example: Teaching example
            concept: Target concept
            beliefs: User beliefs

        Returns:
            Teaching value score (higher is better)
        """
        # Base teaching value on:
        # 1. Novelty (how different from user's current understanding)
        # 2. Clarity (how clearly it illustrates the concept)
        # 3. Relevance (how relevant to user's context)

        # Novelty: Based on how example outcomes differ from user's expected values
        novelty = self._compute_novelty(example, beliefs)

        # Clarity: Higher for simpler, clearer examples
        clarity = self._compute_clarity(example, concept)

        # Relevance: Based on user's value weights
        relevance = self._compute_relevance(example, beliefs)

        # Combine into overall teaching value (weighted sum)
        teaching_value = 0.4 * novelty + 0.4 * clarity + 0.2 * relevance

        return float(teaching_value)

    def _compute_novelty(
        self, example: TeachingExample, beliefs: UserBeliefModel
    ) -> float:
        """Compute novelty of example given beliefs."""
        # Compare example outcomes to user's expected values
        differences = []
        for var, value in example.scenario.outcomes.items():
            if var in beliefs.value_weights:
                # Normalize outcome value
                norm_value = value / 100.0 if value > 1 else value
                expected = beliefs.value_weights[var].parameters["mean"]
                diff = abs(norm_value - expected)
                differences.append(diff)

        if not differences:
            return 0.5  # Default moderate novelty

        # Average difference (higher = more novel)
        return float(np.mean(differences))

    def _compute_clarity(self, example: TeachingExample, concept: str) -> float:
        """Compute clarity of example for teaching concept."""
        # Clarity is higher for:
        # - Fewer variables (simpler)
        # - Clear trade-offs
        # - Focused message

        num_variables = len(example.scenario.outcomes)
        num_trade_offs = len(example.scenario.trade_offs)

        # Simpler examples are clearer
        simplicity = 1.0 / (1.0 + num_variables / 10.0)

        # Having trade-offs described is good
        has_trade_offs = min(1.0, num_trade_offs / 2.0)

        # Combine
        clarity = 0.6 * simplicity + 0.4 * has_trade_offs

        return float(clarity)

    def _compute_relevance(
        self, example: TeachingExample, beliefs: UserBeliefModel
    ) -> float:
        """Compute relevance of example to user's concerns."""
        # Relevance based on whether example involves variables user cares about
        relevance_scores = []

        for var in example.scenario.outcomes.keys():
            if var in beliefs.value_weights:
                # User cares more about variables with higher weights
                weight = beliefs.value_weights[var].parameters["mean"]
                relevance_scores.append(weight)

        if not relevance_scores:
            return 0.5  # Default moderate relevance

        # Average weight of involved variables
        return float(np.mean(relevance_scores))

    def _generate_teaching_explanation(
        self,
        concept: str,
        examples: List[TeachingExample],
        beliefs: UserBeliefModel,
    ) -> str:
        """Generate explanation of teaching strategy."""
        concept_desc = self.concepts[concept]["description"]
        num_examples = len(examples)

        explanation = (
            f"Selected {num_examples} examples to help you understand {concept_desc}. "
            f"These examples were chosen to maximize learning efficiency by "
            f"focusing on aspects that will most improve your understanding "
            f"given your current knowledge state."
        )

        return explanation

    def _define_learning_objectives(
        self, concept: str, examples: List[TeachingExample]
    ) -> List[str]:
        """Define learning objectives for the teaching session."""
        objectives = [
            f"Understand core principles of {concept}",
            f"Recognize {concept} in real decision scenarios",
        ]

        # Add example-specific objectives
        for i, example in enumerate(examples, 1):
            objectives.append(f"Learn from example {i}: {example.key_insight}")

        return objectives

    def _estimate_learning_time(self, concept: str, num_examples: int) -> str:
        """Estimate time needed to learn the concept."""
        # Rough estimates
        minutes_per_example = 3
        base_time = 2  # Base time for introduction

        total_minutes = base_time + (num_examples * minutes_per_example)

        if total_minutes < 10:
            return f"{total_minutes} minutes"
        else:
            return f"{total_minutes // 10 * 10}-{(total_minutes // 10 + 1) * 10} minutes"
