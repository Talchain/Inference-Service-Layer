"""
Belief Updater service for Bayesian preference learning.

Updates user belief models based on preference responses using
Bayesian inference.

Implements: P(θ|D) ∝ P(D|θ) × P(θ)
Where:
- θ = user preference parameters (weights, risk tolerance, etc.)
- D = observed preference data (user responses)
- P(θ) = prior belief distribution
- P(D|θ) = likelihood of observed response
- P(θ|D) = posterior belief distribution
"""

import logging
from typing import Dict, List

import numpy as np
from scipy import stats

from src.models.phase1_models import (
    CounterfactualQuery,
    LearningSummary,
    PreferenceChoice,
    Scenario,
    UserBeliefModel,
)
from src.models.shared import Distribution, DistributionType
from src.utils.determinism import make_deterministic

logger = logging.getLogger(__name__)


class BeliefUpdater:
    """
    Updates user belief model based on preference responses.

    Implements Bayesian inference over preference parameters.
    """

    def __init__(self) -> None:
        """Initialize the updater."""
        self.monte_carlo_samples = 1000

    def update_beliefs(
        self,
        current_beliefs: UserBeliefModel,
        query: CounterfactualQuery,
        response: PreferenceChoice,
        confidence: float = 1.0,
    ) -> UserBeliefModel:
        """
        Update belief distribution using Bayesian inference.

        P(weights | choice) ∝ P(choice | weights) * P(weights)

        Args:
            current_beliefs: Current belief model
            query: Query that was answered
            response: User's choice (A, B, or indifferent)
            confidence: User's confidence in choice (0-1)

        Returns:
            Updated belief model
        """
        # Make computation deterministic
        rng = make_deterministic(
            {
                "query_id": query.id,
                "response": response.value,
                "confidence": confidence,
            }
        )

        logger.info(
            "belief_update_started",
            extra={
                "query_id": query.id,
                "response": response.value,
                "confidence": confidence,
                "seed": rng.seed,
            },
        )

        # Compute likelihood of observed response
        likelihood_params = self._compute_likelihood(
            query.scenario_a,
            query.scenario_b,
            response,
            confidence,
        )

        # Bayesian update: posterior ∝ likelihood × prior
        updated_weights = self._bayesian_update_weights(
            current_beliefs.value_weights,
            likelihood_params,
            query.scenario_a,
            query.scenario_b,
        )

        # Update risk tolerance if response indicates risk preference
        updated_risk = self._update_risk_tolerance(
            current_beliefs.risk_tolerance,
            query,
            response,
        )

        # Update uncertainty estimates
        updated_uncertainty = self._update_uncertainty(
            current_beliefs.uncertainty_estimates,
            query,
            response,
        )

        updated_beliefs = UserBeliefModel(
            value_weights=updated_weights,
            risk_tolerance=updated_risk,
            time_horizon=current_beliefs.time_horizon,
            uncertainty_estimates=updated_uncertainty,
        )

        # Log update metrics
        entropy_before = self._compute_entropy(current_beliefs)
        entropy_after = self._compute_entropy(updated_beliefs)
        entropy_reduction = entropy_before - entropy_after

        logger.info(
            "belief_update_complete",
            extra={
                "entropy_before": entropy_before,
                "entropy_after": entropy_after,
                "entropy_reduction": entropy_reduction,
                "avg_uncertainty_before": np.mean(
                    list(current_beliefs.uncertainty_estimates.values())
                ),
                "avg_uncertainty_after": np.mean(
                    list(updated_uncertainty.values())
                ),
            },
        )

        return updated_beliefs

    def generate_learning_summary(
        self,
        beliefs: UserBeliefModel,
        queries_completed: int,
    ) -> LearningSummary:
        """
        Generate summary of what has been learned about user.

        Args:
            beliefs: Current belief model
            queries_completed: Number of queries answered

        Returns:
            Learning summary with insights
        """
        # Identify top priorities from weight distributions
        top_priorities = self._identify_top_priorities(beliefs)

        # Compute overall confidence (inverse of uncertainty)
        avg_uncertainty = np.mean(list(beliefs.uncertainty_estimates.values()))
        confidence = 1.0 - avg_uncertainty

        # Generate insights based on beliefs
        insights = self._generate_insights(beliefs)

        # Determine if ready for recommendations
        ready = self._is_ready_for_recommendations(beliefs, queries_completed)

        return LearningSummary(
            top_priorities=top_priorities,
            confidence=round(confidence, 2),
            insights=insights,
            ready_for_recommendations=ready,
        )

    def _compute_likelihood(
        self,
        scenario_a: Scenario,
        scenario_b: Scenario,
        response: PreferenceChoice,
        confidence: float,
    ) -> Dict[str, float]:
        """
        Compute likelihood of observed response under different weight hypotheses.

        Args:
            scenario_a: First scenario
            scenario_b: Second scenario
            response: User's choice
            confidence: Confidence in choice

        Returns:
            Likelihood parameters for Bayesian update
        """
        # Extract outcomes
        outcomes_a = scenario_a.outcomes
        outcomes_b = scenario_b.outcomes

        # Compute which variables distinguish the scenarios
        distinguishing_vars = {}
        for var in set(outcomes_a.keys()) | set(outcomes_b.keys()):
            val_a = outcomes_a.get(var, 0)
            val_b = outcomes_b.get(var, 0)
            if abs(val_a - val_b) > 0.01:  # Meaningfully different
                distinguishing_vars[var] = {
                    "a": val_a,
                    "b": val_b,
                    "diff": val_a - val_b,
                }

        # Likelihood depends on which option was chosen
        likelihood_params = {}
        for var, vals in distinguishing_vars.items():
            if response == PreferenceChoice.A:
                # User prefers A - increase weight on variables where A is better
                if vals["diff"] > 0:
                    likelihood_params[var] = confidence
                else:
                    likelihood_params[var] = -confidence
            elif response == PreferenceChoice.B:
                # User prefers B - increase weight on variables where B is better
                if vals["diff"] < 0:
                    likelihood_params[var] = confidence
                else:
                    likelihood_params[var] = -confidence
            else:  # INDIFFERENT
                # No strong signal - slight reduction in certainty
                likelihood_params[var] = 0.0

        return likelihood_params

    def _bayesian_update_weights(
        self,
        current_weights: Dict[str, Distribution],
        likelihood_params: Dict[str, float],
        scenario_a: Scenario,
        scenario_b: Scenario,
    ) -> Dict[str, Distribution]:
        """
        Update weight distributions using Bayesian inference.

        Args:
            current_weights: Current weight distributions
            likelihood_params: Likelihood signals from response
            scenario_a: First scenario
            scenario_b: Second scenario

        Returns:
            Updated weight distributions
        """
        updated_weights = {}

        for var, dist in current_weights.items():
            if var in likelihood_params:
                signal = likelihood_params[var]

                if dist.type == DistributionType.NORMAL:
                    # Update normal distribution
                    current_mean = dist.parameters["mean"]
                    current_std = dist.parameters["std"]

                    # Shift mean based on signal
                    # Positive signal -> increase weight
                    # Negative signal -> decrease weight
                    learning_rate = 0.1  # How much to update per observation
                    new_mean = current_mean + signal * learning_rate

                    # Reduce uncertainty (std) with each observation
                    new_std = current_std * 0.9

                    # Clamp to reasonable bounds
                    new_mean = max(0.0, min(1.0, new_mean))
                    new_std = max(0.05, new_std)  # Min std to prevent overconfidence

                    updated_weights[var] = Distribution(
                        type=DistributionType.NORMAL,
                        parameters={"mean": new_mean, "std": new_std},
                    )
                else:
                    # Keep other distribution types unchanged for now
                    updated_weights[var] = dist
            else:
                # No signal for this variable - keep current
                updated_weights[var] = dist

        return updated_weights

    def _update_risk_tolerance(
        self,
        current_risk: Distribution,
        query: CounterfactualQuery,
        response: PreferenceChoice,
    ) -> Distribution:
        """
        Update risk tolerance based on response.

        Args:
            current_risk: Current risk tolerance distribution
            query: Query answered
            response: User's choice

        Returns:
            Updated risk tolerance distribution
        """
        # Analyze if this query revealed risk preferences
        # (e.g., safe option vs risky option)

        # For now, keep risk tolerance unchanged
        # In full implementation, would analyze variance in outcomes
        return current_risk

    def _update_uncertainty(
        self,
        current_uncertainty: Dict[str, float],
        query: CounterfactualQuery,
        response: PreferenceChoice,
    ) -> Dict[str, float]:
        """
        Update uncertainty estimates based on response.

        Args:
            current_uncertainty: Current uncertainty estimates
            query: Query answered
            response: User's choice

        Returns:
            Updated uncertainty estimates
        """
        updated = current_uncertainty.copy()

        # Reduce uncertainty for variables involved in this query
        involved_vars = set(query.scenario_a.outcomes.keys()) | set(
            query.scenario_b.outcomes.keys()
        )

        reduction_rate = 0.15 if response != PreferenceChoice.INDIFFERENT else 0.05

        for var in involved_vars:
            weight_key = f"{var}_weight"
            if weight_key in updated:
                # Reduce uncertainty
                updated[weight_key] = max(
                    0.1,  # Min uncertainty
                    updated[weight_key] * (1 - reduction_rate),
                )

        return updated

    def _compute_entropy(self, beliefs: UserBeliefModel) -> float:
        """
        Compute entropy of belief distribution.

        Args:
            beliefs: Belief model

        Returns:
            Entropy value
        """
        # Compute from uncertainty estimates
        avg_uncertainty = np.mean(list(beliefs.uncertainty_estimates.values()))
        return avg_uncertainty * 2.0

    def _identify_top_priorities(
        self,
        beliefs: UserBeliefModel,
        top_n: int = 3,
    ) -> List[str]:
        """
        Identify user's top priorities from weight distributions.

        Args:
            beliefs: Belief model
            top_n: Number of top priorities to return

        Returns:
            List of top priority variable names
        """
        # Extract mean weights
        weights = {}
        for var, dist in beliefs.value_weights.items():
            if dist.type == DistributionType.NORMAL:
                weights[var] = dist.parameters["mean"]
            elif dist.type == DistributionType.BETA:
                # Mean of beta distribution
                alpha = dist.parameters["alpha"]
                beta = dist.parameters["beta"]
                weights[var] = alpha / (alpha + beta)
            else:
                weights[var] = dist.parameters.get("mean", 0.5)

        # Sort by weight (descending)
        sorted_vars = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        # Return top N
        return [var for var, _ in sorted_vars[:top_n]]

    def _generate_insights(self, beliefs: UserBeliefModel) -> List[str]:
        """
        Generate natural language insights about user preferences.

        Args:
            beliefs: Belief model

        Returns:
            List of insight strings
        """
        insights = []

        # Get top priorities
        top_priorities = self._identify_top_priorities(beliefs, top_n=3)

        if len(top_priorities) >= 1:
            insights.append(
                f"You strongly prioritize {top_priorities[0]} in your decisions"
            )

        if len(top_priorities) >= 2:
            insights.append(
                f"You value {top_priorities[0]} over {top_priorities[1]}"
            )

        # Check uncertainty levels
        avg_uncertainty = np.mean(list(beliefs.uncertainty_estimates.values()))
        if avg_uncertainty < 0.3:
            insights.append("We have high confidence in understanding your preferences")
        elif avg_uncertainty < 0.6:
            insights.append("We're developing a good understanding of your preferences")
        else:
            insights.append("We're still learning about your preferences")

        # Analyze weight distribution spread
        weights = []
        for dist in beliefs.value_weights.values():
            if dist.type == DistributionType.NORMAL:
                weights.append(dist.parameters["mean"])

        if weights:
            weight_std = np.std(weights)
            if weight_std < 0.15:
                insights.append("You tend to balance multiple factors fairly evenly")
            else:
                insights.append("You have clear, distinct priorities")

        return insights[:4]  # Return top 4 insights

    def _is_ready_for_recommendations(
        self,
        beliefs: UserBeliefModel,
        queries_completed: int,
    ) -> bool:
        """
        Determine if enough has been learned for recommendations.

        Args:
            beliefs: Current beliefs
            queries_completed: Number of queries answered

        Returns:
            True if ready for recommendations
        """
        # Need minimum number of queries
        if queries_completed < 3:
            return False

        # Check uncertainty level
        avg_uncertainty = np.mean(list(beliefs.uncertainty_estimates.values()))
        if avg_uncertainty > 0.5:
            return False

        # Check that we have clear top priorities
        top_priorities = self._identify_top_priorities(beliefs)
        if len(top_priorities) < 2:
            return False

        return True
