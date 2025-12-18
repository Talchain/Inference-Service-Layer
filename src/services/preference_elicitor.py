"""
Preference Elicitor service implementing ActiVA algorithm.

Efficiently learns user preferences through strategically chosen
counterfactual queries that maximize information gain.

Based on: ActiVA - Active Value Alignment through Counterfactual Queries
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy

from src.config import get_settings
from src.models.phase1_models import (
    CounterfactualQuery,
    DecisionContext,
    QueryStrategy,
    QueryStrategyInfo,
    Scenario,
    UserBeliefModel,
)
from src.models.shared import Distribution, DistributionType
from src.services.explanation_generator import ExplanationGenerator
from src.utils.determinism import make_deterministic

logger = logging.getLogger(__name__)
settings = get_settings()


class PreferenceElicitor:
    """
    Implements ActiVA algorithm for efficient preference elicitation.

    Key insight: Select counterfactual queries that maximize
    information gain about user preferences.

    Algorithm:
    1. Generate candidate scenario pairs
    2. Compute expected information gain for each
    3. Select top N queries by information gain
    4. Format as natural language questions
    """

    def __init__(self) -> None:
        """Initialize the elicitor."""
        self.explanation_generator = ExplanationGenerator()
        self.monte_carlo_samples = 1000

    def generate_queries(
        self,
        context: DecisionContext,
        current_beliefs: Optional[UserBeliefModel],
        num_queries: int,
    ) -> Tuple[List[CounterfactualQuery], QueryStrategyInfo]:
        """
        Generate counterfactual queries that maximize information gain.

        Args:
            context: Decision context with domain and variables
            current_beliefs: Current user beliefs (None for first elicitation)
            num_queries: Number of queries to generate

        Returns:
            Tuple of (queries, strategy_info)
        """
        # Make computation deterministic
        rng = make_deterministic(
            {"context": context.model_dump(), "num_queries": num_queries}
        )

        logger.info(
            "query_generation_started",
            extra={
                "domain": context.domain,
                "num_variables": len(context.variables),
                "num_queries": num_queries,
                "seed": rng.seed,
                "has_beliefs": current_beliefs is not None,
            },
        )

        # Initialize beliefs if not provided
        if current_beliefs is None:
            current_beliefs = self._initialize_beliefs(context)
            logger.info("initialized_prior_beliefs")

        # Generate candidate query pairs
        candidates = self._generate_candidate_queries(context)
        logger.info(
            "generated_candidates",
            extra={"num_candidates": len(candidates)},
        )

        # Rank by information gain
        ranked_queries = self._rank_by_information_gain(
            candidates,
            current_beliefs,
            context,
        )

        # Select top N
        selected_queries = ranked_queries[:num_queries]

        # Determine strategy used
        strategy = self._determine_strategy(current_beliefs)

        logger.info(
            "query_generation_complete",
            extra={
                "num_queries": len(selected_queries),
                "strategy": strategy.type.value,
                "avg_info_gain": np.mean([q.information_gain for q in selected_queries]),
            },
        )

        return selected_queries, strategy

    def _initialize_beliefs(self, context: DecisionContext) -> UserBeliefModel:
        """
        Initialize prior beliefs for new user.

        Uses domain-specific priors or uniform distributions.

        Args:
            context: Decision context

        Returns:
            Initial belief model
        """
        # Initialize value weights with uniform priors
        value_weights = {}
        for var in context.variables:
            value_weights[var] = Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 1.0 / len(context.variables), "std": 0.3},
            )

        # Initialize risk tolerance (beta distribution centered at 0.5)
        risk_tolerance = Distribution(
            type=DistributionType.BETA,
            parameters={"alpha": 2, "beta": 2},
        )

        # Initialize time horizon (normal, ~1 year)
        time_horizon = Distribution(
            type=DistributionType.NORMAL,
            parameters={"mean": 12, "std": 6},
        )

        # High uncertainty initially
        uncertainty_estimates = {f"{var}_weight": 0.8 for var in context.variables}

        return UserBeliefModel(
            value_weights=value_weights,
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon,
            uncertainty_estimates=uncertainty_estimates,
        )

    def _generate_candidate_queries(
        self,
        context: DecisionContext,
    ) -> List[Tuple[Scenario, Scenario]]:
        """
        Generate diverse scenario pairs for comparison.

        Strategy:
        - Cover different trade-off dimensions
        - Include edge cases and typical cases
        - Ensure scenarios are meaningfully different

        Args:
            context: Decision context

        Returns:
            List of (scenario_a, scenario_b) tuples
        """
        candidates = []

        # Generate scenarios varying each variable systematically
        for var in context.variables:
            scenarios = self._create_scenario_pairs_for_variable(var, context)
            candidates.extend(scenarios)

        # Add multi-dimensional trade-off scenarios
        trade_off_scenarios = self._create_trade_off_scenarios(context)
        candidates.extend(trade_off_scenarios)

        return candidates

    def _create_scenario_pairs_for_variable(
        self,
        variable: str,
        context: DecisionContext,
    ) -> List[Tuple[Scenario, Scenario]]:
        """
        Create scenario pairs that isolate the effect of one variable.

        Example for pricing:
        - Scenario A: Price +10%, Revenue +5%, Churn +2%
        - Scenario B: Price +10%, Revenue +8%, Churn +5%

        Tests: How much does user value revenue vs churn?

        Args:
            variable: Variable to vary
            context: Decision context

        Returns:
            List of scenario pairs
        """
        scenarios = []

        # Generate base outcomes (domain-specific defaults)
        base_outcomes = self._generate_base_outcomes(context)

        # Create variations with different trade-off profiles
        for magnitude in [0.1, 0.2]:  # 10%, 20% changes
            # Conservative scenario: smaller secondary effects
            scenario_a = Scenario(
                description=f"{variable} increases by {magnitude*100:.0f}% (conservative approach)",
                outcomes=self._compute_outcomes_with_change(
                    base_outcomes,
                    variable,
                    magnitude,
                    trade_off_profile="conservative",
                    context=context,
                ),
                trade_offs=[f"Moderate {variable} increase", "Limited side effects"],
            )

            # Aggressive scenario: larger secondary effects
            scenario_b = Scenario(
                description=f"{variable} increases by {magnitude*100:.0f}% (aggressive approach)",
                outcomes=self._compute_outcomes_with_change(
                    base_outcomes,
                    variable,
                    magnitude,
                    trade_off_profile="aggressive",
                    context=context,
                ),
                trade_offs=[f"Significant {variable} increase", "Notable side effects"],
            )

            scenarios.append((scenario_a, scenario_b))

        return scenarios

    def _create_trade_off_scenarios(
        self,
        context: DecisionContext,
    ) -> List[Tuple[Scenario, Scenario]]:
        """
        Create scenarios highlighting multi-dimensional trade-offs.

        Args:
            context: Decision context

        Returns:
            List of scenario pairs
        """
        scenarios = []

        # Only create if we have multiple variables
        if len(context.variables) < 2:
            return scenarios

        # Create scenarios trading off different variables
        for i, var1 in enumerate(context.variables):
            for var2 in context.variables[i + 1 :]:
                # Scenario favoring var1
                outcomes_a = {var: 0.0 for var in context.variables}
                outcomes_a[var1] = 0.8
                outcomes_a[var2] = 0.2

                scenario_a = Scenario(
                    description=f"Prioritize {var1} over {var2}",
                    outcomes=outcomes_a,
                    trade_offs=[f"High {var1}", f"Low {var2}"],
                )

                # Scenario favoring var2
                outcomes_b = {var: 0.0 for var in context.variables}
                outcomes_b[var1] = 0.2
                outcomes_b[var2] = 0.8

                scenario_b = Scenario(
                    description=f"Prioritize {var2} over {var1}",
                    outcomes=outcomes_b,
                    trade_offs=[f"Low {var1}", f"High {var2}"],
                )

                scenarios.append((scenario_a, scenario_b))

        return scenarios[:3]  # Limit to top 3 trade-off pairs

    def _generate_base_outcomes(self, context: DecisionContext) -> Dict[str, float]:
        """
        Generate baseline outcome values for scenarios.

        Args:
            context: Decision context

        Returns:
            Dict of variable -> baseline value
        """
        # Domain-specific baselines
        if context.domain == "pricing":
            return {
                "revenue": 50000,
                "churn": 0.05,
                "brand": 0.0,
                "acquisition": 100,
            }
        else:
            # Generic baseline: all variables at moderate levels
            return {var: 0.5 for var in context.variables}

    def _compute_outcomes_with_change(
        self,
        base_outcomes: Dict[str, float],
        variable: str,
        magnitude: float,
        trade_off_profile: str,
        context: DecisionContext,
    ) -> Dict[str, float]:
        """
        Compute outcomes when a variable changes.

        Args:
            base_outcomes: Baseline outcomes
            variable: Variable being changed
            magnitude: Change magnitude (0-1)
            trade_off_profile: 'conservative' or 'aggressive'
            context: Decision context

        Returns:
            Dict of outcomes
        """
        outcomes = base_outcomes.copy()

        # Apply primary change
        if variable in outcomes:
            outcomes[variable] = base_outcomes[variable] * (1 + magnitude)

        # Apply secondary effects based on trade-off profile
        multiplier = 0.3 if trade_off_profile == "conservative" else 0.7

        # Domain-specific secondary effects
        if context.domain == "pricing" and variable == "price":
            # Price increase affects other metrics
            if "revenue" in outcomes:
                outcomes["revenue"] = base_outcomes["revenue"] * (1 + magnitude * 0.8)
            if "churn" in outcomes:
                outcomes["churn"] = base_outcomes["churn"] * (1 + magnitude * multiplier)
            if "brand" in outcomes:
                outcomes["brand"] = base_outcomes.get("brand", 0) - magnitude * multiplier * 0.1

        return outcomes

    def _rank_by_information_gain(
        self,
        candidates: List[Tuple[Scenario, Scenario]],
        current_beliefs: UserBeliefModel,
        context: DecisionContext,
    ) -> List[CounterfactualQuery]:
        """
        Rank candidate queries by expected information gain.

        Information gain = Current entropy - Expected posterior entropy

        Args:
            candidates: Candidate scenario pairs
            current_beliefs: Current beliefs
            context: Decision context

        Returns:
            Ranked list of queries
        """
        ranked = []

        for idx, (scenario_a, scenario_b) in enumerate(candidates):
            # Compute expected information gain
            info_gain = self._compute_expected_information_gain(
                scenario_a,
                scenario_b,
                current_beliefs,
            )

            query = CounterfactualQuery(
                id=f"query_{idx:03d}",
                question=self._format_comparison_question(
                    scenario_a,
                    scenario_b,
                    context,
                ),
                scenario_a=scenario_a,
                scenario_b=scenario_b,
                information_gain=info_gain,
            )

            ranked.append(query)

        # Sort by information gain (descending)
        ranked.sort(key=lambda q: q.information_gain, reverse=True)

        return ranked

    def _compute_expected_information_gain(
        self,
        scenario_a: Scenario,
        scenario_b: Scenario,
        beliefs: UserBeliefModel,
    ) -> float:
        """
        Compute expected information gain for this query.

        Uses Monte Carlo sampling to estimate:
        - P(user prefers A | current beliefs)
        - Posterior entropy if user chooses A
        - Posterior entropy if user chooses B
        - Expected posterior entropy = weighted average
        - Information gain = current entropy - expected posterior entropy

        Args:
            scenario_a: First scenario
            scenario_b: Second scenario
            beliefs: Current beliefs

        Returns:
            Expected information gain
        """
        # Compute current entropy
        current_entropy = self._compute_belief_entropy(beliefs)

        # Monte Carlo sampling to estimate probabilities
        preferences_a = 0
        preferences_b = 0

        for _ in range(self.monte_carlo_samples):
            # Sample user preference function from beliefs
            sampled_weights = self._sample_weights(beliefs)

            # Evaluate scenarios under sampled preferences
            utility_a = self._compute_utility(scenario_a, sampled_weights)
            utility_b = self._compute_utility(scenario_b, sampled_weights)

            if utility_a > utility_b:
                preferences_a += 1
            else:
                preferences_b += 1

        p_prefer_a = preferences_a / self.monte_carlo_samples
        p_prefer_b = preferences_b / self.monte_carlo_samples

        # Avoid division by zero
        if p_prefer_a == 0 or p_prefer_b == 0:
            return 0.1  # Small default information gain

        # Estimate posterior entropies (simplified - assume uniform reduction)
        # In full implementation, would compute actual posterior distributions
        entropy_reduction_a = current_entropy * 0.2  # 20% reduction if clear preference
        entropy_reduction_b = current_entropy * 0.2

        entropy_if_a = current_entropy - entropy_reduction_a
        entropy_if_b = current_entropy - entropy_reduction_b

        # Expected posterior entropy
        expected_posterior_entropy = p_prefer_a * entropy_if_a + p_prefer_b * entropy_if_b

        # Information gain
        information_gain = current_entropy - expected_posterior_entropy

        return max(0.0, information_gain)  # Ensure non-negative

    def _compute_belief_entropy(self, beliefs: UserBeliefModel) -> float:
        """
        Compute entropy of current belief distribution.

        Args:
            beliefs: User beliefs

        Returns:
            Entropy value
        """
        # Compute entropy from uncertainty estimates
        # Higher uncertainty = higher entropy
        avg_uncertainty = np.mean(list(beliefs.uncertainty_estimates.values()))
        # Map uncertainty (0-1) to entropy (~0-2)
        return avg_uncertainty * 2.0

    def _sample_weights(self, beliefs: UserBeliefModel) -> Dict[str, float]:
        """
        Sample a weight vector from belief distribution.

        Args:
            beliefs: User beliefs

        Returns:
            Sampled weights
        """
        weights = {}
        for var, dist in beliefs.value_weights.items():
            if dist.type == DistributionType.NORMAL:
                weights[var] = np.random.normal(
                    dist.parameters["mean"],
                    dist.parameters["std"],
                )
            elif dist.type == DistributionType.BETA:
                weights[var] = np.random.beta(
                    dist.parameters["alpha"],
                    dist.parameters["beta"],
                )
            else:
                weights[var] = dist.parameters.get("mean", 0.5)

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _compute_utility(
        self,
        scenario: Scenario,
        weights: Dict[str, float],
    ) -> float:
        """
        Compute utility of scenario under given weights.

        Args:
            scenario: Scenario to evaluate
            weights: Preference weights

        Returns:
            Utility value
        """
        utility = 0.0
        for var, outcome in scenario.outcomes.items():
            weight = weights.get(var, 0.0)
            utility += weight * outcome

        return utility

    def _format_comparison_question(
        self,
        scenario_a: Scenario,
        scenario_b: Scenario,
        context: DecisionContext,
    ) -> str:
        """
        Format scenario comparison as natural language question.

        Args:
            scenario_a: First scenario
            scenario_b: Second scenario
            context: Decision context

        Returns:
            Formatted question
        """
        question = "Which outcome would you prefer?\n\n"

        question += f"**Option A:** {scenario_a.description}\n"
        question += self._format_outcomes(scenario_a.outcomes)
        if scenario_a.trade_offs:
            question += f"\nTrade-offs: {', '.join(scenario_a.trade_offs)}\n"

        question += f"\n**Option B:** {scenario_b.description}\n"
        question += self._format_outcomes(scenario_b.outcomes)
        if scenario_b.trade_offs:
            question += f"\nTrade-offs: {', '.join(scenario_b.trade_offs)}\n"

        question += f"\nConsider the trade-offs in {context.domain} decisions."

        return question

    def _format_outcomes(self, outcomes: Dict[str, float]) -> str:
        """
        Format outcomes for display.

        Args:
            outcomes: Outcome values

        Returns:
            Formatted string
        """
        lines = []
        for var, value in outcomes.items():
            # Format based on magnitude
            if abs(value) < 1:
                formatted = f"{value:.1%}"
            elif abs(value) < 1000:
                formatted = f"{value:.0f}"
            else:
                formatted = f"{value:,.0f}"
            lines.append(f"  - {var}: {formatted}")

        return "\n".join(lines)

    def _determine_strategy(self, beliefs: UserBeliefModel) -> QueryStrategyInfo:
        """
        Determine which strategy was used for query selection.

        Args:
            beliefs: Current beliefs

        Returns:
            Strategy information
        """
        # Compute average uncertainty
        avg_uncertainty = np.mean(list(beliefs.uncertainty_estimates.values()))

        if avg_uncertainty > 0.6:
            strategy_type = QueryStrategy.UNCERTAINTY_SAMPLING
            rationale = "High uncertainty - focusing on areas where your preferences are least clear"
            focus = ["All preference dimensions"]
        elif avg_uncertainty > 0.3:
            strategy_type = QueryStrategy.EXPECTED_IMPROVEMENT
            rationale = "Moderate uncertainty - refining understanding of key trade-offs"
            # Find variables with highest uncertainty
            sorted_uncertainties = sorted(
                beliefs.uncertainty_estimates.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            focus = [var for var, _ in sorted_uncertainties[:3]]
        else:
            strategy_type = QueryStrategy.EXPLORATION
            rationale = "Low uncertainty - exploring edge cases and rare scenarios"
            focus = ["Boundary conditions", "Extreme scenarios"]

        return QueryStrategyInfo(
            type=strategy_type,
            rationale=rationale,
            focus_areas=focus,
        )
