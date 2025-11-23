"""
Sequential Optimization using Thompson Sampling.

Recommends next experiments to maximize learning about
causal effects while optimizing outcomes.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Constants for Thompson sampling
N_THOMPSON_SAMPLES = 100  # Number of posterior samples for Thompson sampling
MAX_CANDIDATE_ACTIONS = 10  # Maximum number of candidate actions to evaluate
DISTANCE_NORMALIZATION = 20.0  # Normalization factor for action distance
INFO_GAIN_NORMALIZATION = 10.0  # Normalization factor for information gain


class BeliefState:
    """Represents current beliefs about model parameters."""

    def __init__(self, parameter_distributions: Dict[str, Dict]):
        """
        Initialize belief state.

        Args:
            parameter_distributions: Dict mapping parameter names to distributions
                Format: {"param1": {"type": "normal", "mean": 0, "std": 1}}
        """
        self.parameter_distributions = parameter_distributions


class OptimizationObjective:
    """Optimization objective specification."""

    def __init__(
        self, target_variable: str, goal: str, target_value: Optional[float] = None
    ):
        """
        Initialize optimization objective.

        Args:
            target_variable: Variable to optimize
            goal: "maximize", "minimize", or "target"
            target_value: Target value (for goal="target")
        """
        self.target_variable = target_variable
        self.goal = goal
        self.target_value = target_value


class ExperimentConstraints:
    """Constraints on experiments."""

    def __init__(
        self,
        budget: float,
        time_horizon: int,
        feasible_interventions: Dict[str, Tuple[float, float]],
    ):
        """
        Initialize experiment constraints.

        Args:
            budget: Total budget
            time_horizon: Number of experiments remaining
            feasible_interventions: Dict mapping variables to (min, max) bounds
        """
        self.budget = budget
        self.time_horizon = time_horizon
        self.feasible_interventions = feasible_interventions


class RecommendedExperiment:
    """Recommended experiment specification."""

    def __init__(
        self,
        intervention: Dict[str, float],
        expected_outcome: Dict[str, float],
        expected_information_gain: float,
        cost_estimate: float,
        rationale: str,
        exploration_vs_exploitation: float,
    ):
        """
        Initialize recommended experiment.

        Args:
            intervention: Intervention values
            expected_outcome: Expected outcome values
            expected_information_gain: Expected information gain
            cost_estimate: Estimated cost
            rationale: Explanation
            exploration_vs_exploitation: 0=exploit, 1=explore
        """
        self.intervention = intervention
        self.expected_outcome = expected_outcome
        self.expected_information_gain = expected_information_gain
        self.cost_estimate = cost_estimate
        self.rationale = rationale
        self.exploration_vs_exploitation = exploration_vs_exploitation


class SequentialOptimizer:
    """
    Thompson sampling for sequential experiment design.

    Balances exploration (learning parameters) with
    exploitation (optimizing outcome).
    """

    def __init__(self):
        """Initialize sequential optimizer."""
        pass

    def recommend_next_experiment(
        self,
        beliefs: BeliefState,
        objective: OptimizationObjective,
        constraints: ExperimentConstraints,
        history: List[Dict] = None,
        seed: Optional[int] = None,
    ) -> RecommendedExperiment:
        """
        Recommend next experiment using Thompson sampling.

        Args:
            beliefs: Current parameter beliefs
            objective: Optimization objective
            constraints: Experiment constraints
            history: List of previous experiments
            seed: Random seed for reproducibility

        Returns:
            RecommendedExperiment
        """
        # Use isolated random state instead of global seed
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

        # Thompson sampling: sample from posterior beliefs
        action_scores = {}

        logger.info(
            "thompson_sampling_start",
            extra={
                "n_samples": N_THOMPSON_SAMPLES,
                "n_beliefs": len(beliefs.parameter_distributions),
                "n_history": len(history) if history else 0,
            }
        )

        # Generate candidate actions
        candidate_actions = self._generate_candidate_actions(
            constraints.feasible_interventions
        )

        for action in candidate_actions:
            # Sample parameters from beliefs
            total_score = 0.0

            for _ in range(N_THOMPSON_SAMPLES):
                params = self._sample_parameters(beliefs, rng)

                # Evaluate action under sampled parameters
                value = self._evaluate_action(action, params, objective)
                total_score += value

            # Average score across samples
            action_scores[self._action_to_key(action)] = total_score / N_THOMPSON_SAMPLES

        # Select best action (highest expected value)
        best_action_key = max(action_scores.items(), key=lambda x: x[1])[0]
        best_action = self._key_to_action(
            best_action_key, list(constraints.feasible_interventions.keys())
        )

        # Compute expected outcome and information gain
        expected_outcome = self._predict_outcome(best_action, beliefs)
        information_gain = self._estimate_information_gain(
            best_action, beliefs, history or []
        )

        # Compute exploration vs exploitation score
        exploration_score = self._compute_exploration_score(best_action, history or [])

        return RecommendedExperiment(
            intervention=best_action,
            expected_outcome=expected_outcome,
            expected_information_gain=information_gain,
            cost_estimate=self._estimate_cost(best_action),
            rationale=self._generate_rationale(
                best_action, expected_outcome, information_gain
            ),
            exploration_vs_exploitation=exploration_score,
        )

    def _generate_candidate_actions(
        self, feasible_interventions: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, float]]:
        """
        Generate candidate intervention actions.

        Args:
            feasible_interventions: Feasible ranges for each variable

        Returns:
            List of candidate actions
        """
        candidates = []

        # For each variable, try low, medium, high values
        for var, (low, high) in feasible_interventions.items():
            mid = (low + high) / 2
            for value in [low, mid, high]:
                candidates.append({var: value})

        return candidates[:MAX_CANDIDATE_ACTIONS]

    def _sample_parameters(
        self, beliefs: BeliefState, rng: np.random.RandomState
    ) -> Dict[str, float]:
        """
        Sample parameters from belief distributions.

        Args:
            beliefs: Current beliefs
            rng: Random state for isolated sampling

        Returns:
            Dict of sampled parameter values
        """
        samples = {}

        for param, dist_spec in beliefs.parameter_distributions.items():
            if dist_spec.get("type") == "normal":
                mean = dist_spec.get("mean", 0)
                std = dist_spec.get("std", 1)
                samples[param] = rng.normal(mean, std)
            elif dist_spec.get("type") == "uniform":
                low = dist_spec.get("low", 0)
                high = dist_spec.get("high", 1)
                samples[param] = rng.uniform(low, high)

        return samples

    def _evaluate_action(
        self, action: Dict[str, float], params: Dict[str, float], objective: OptimizationObjective
    ) -> float:
        """
        Evaluate action under sampled parameters.

        Args:
            action: Intervention action
            params: Sampled parameters
            objective: Optimization objective

        Returns:
            Value of action
        """
        # Simplified: assume linear model
        # value = sum of (action[var] * params[var])
        value = 0.0

        for var, val in action.items():
            param_key = f"effect_{var}"
            if param_key in params:
                value += val * params[param_key]

        # Adjust based on objective
        if objective.goal == "minimize":
            value = -value

        return value

    def _predict_outcome(
        self, action: Dict[str, float], beliefs: BeliefState
    ) -> Dict[str, float]:
        """
        Predict outcome for action using current beliefs.

        Args:
            action: Intervention action
            beliefs: Current beliefs

        Returns:
            Dict of predicted outcomes
        """
        # Use mean of belief distributions
        predicted = {}

        outcome_value = 0.0
        for var, val in action.items():
            param_key = f"effect_{var}"
            if param_key in beliefs.parameter_distributions:
                dist = beliefs.parameter_distributions[param_key]
                mean_effect = dist.get("mean", 0)
                outcome_value += val * mean_effect

        predicted["outcome"] = outcome_value

        return predicted

    def _estimate_information_gain(
        self, action: Dict[str, float], beliefs: BeliefState, history: List[Dict]
    ) -> float:
        """
        Estimate expected information gain.

        Args:
            action: Proposed action
            beliefs: Current beliefs
            history: Experiment history

        Returns:
            Estimated information gain
        """
        # Simplified: higher gain for less-explored actions
        # Check how similar this action is to historical actions

        if not history:
            return 1.0  # Maximum gain if no history

        # Compute distance to nearest historical action
        min_distance = float("inf")

        for past_exp in history:
            past_action = past_exp.get("intervention", {})
            distance = self._action_distance(action, past_action)
            min_distance = min(min_distance, distance)

        # Normalize to 0-1 range
        information_gain = min(min_distance / INFO_GAIN_NORMALIZATION, 1.0)

        logger.debug(
            "information_gain_estimated",
            extra={
                "min_distance": min_distance,
                "information_gain": information_gain,
            }
        )

        return information_gain

    def _action_distance(self, action1: Dict[str, float], action2: Dict[str, float]) -> float:
        """
        Compute distance between two actions.

        Args:
            action1: First action
            action2: Second action

        Returns:
            Distance (Euclidean in action space)
        """
        # Get all variables
        all_vars = set(action1.keys()) | set(action2.keys())

        distance_sq = 0.0
        for var in all_vars:
            v1 = action1.get(var, 0)
            v2 = action2.get(var, 0)
            distance_sq += (v1 - v2) ** 2

        return np.sqrt(distance_sq)

    def _compute_exploration_score(
        self, action: Dict[str, float], history: List[Dict]
    ) -> float:
        """
        Compute exploration vs exploitation score.

        Args:
            action: Proposed action
            history: Experiment history

        Returns:
            Score 0-1 (0=pure exploit, 1=pure explore)
        """
        if not history:
            return 1.0

        # If action is similar to past actions, it's exploitation
        # If action is different, it's exploration

        min_distance = float("inf")
        for past_exp in history:
            past_action = past_exp.get("intervention", {})
            distance = self._action_distance(action, past_action)
            min_distance = min(min_distance, distance)

        # Normalize
        exploration_score = min(min_distance / DISTANCE_NORMALIZATION, 1.0)

        return exploration_score

    def _estimate_cost(self, action: Dict[str, float]) -> float:
        """
        Estimate cost of experiment.

        Args:
            action: Intervention action

        Returns:
            Estimated cost
        """
        # Simplified: cost proportional to magnitude of interventions
        cost = sum(abs(v) for v in action.values())
        return cost

    def _generate_rationale(
        self,
        action: Dict[str, float],
        expected_outcome: Dict[str, float],
        information_gain: float,
    ) -> str:
        """
        Generate rationale for recommendation.

        Args:
            action: Recommended action
            expected_outcome: Expected outcome
            information_gain: Information gain

        Returns:
            Plain English rationale
        """
        action_str = ", ".join(f"{k}={v:.1f}" for k, v in action.items())
        outcome_str = ", ".join(f"{k}={v:.1f}" for k, v in expected_outcome.items())

        if information_gain > 0.5:
            rationale = f"Explore: Test {action_str} to learn more (high information gain: {information_gain:.2f})"
        else:
            rationale = f"Exploit: Test {action_str} for expected outcome {outcome_str}"

        return rationale

    def _action_to_key(self, action: Dict[str, float]) -> str:
        """Convert action dict to string key."""
        return ",".join(f"{k}:{v:.2f}" for k, v in sorted(action.items()))

    def _key_to_action(self, key: str, var_names: List[str]) -> Dict[str, float]:
        """Convert string key back to action dict."""
        action = {}
        if key:
            for part in key.split(","):
                if ":" in part:
                    var, val = part.split(":")
                    action[var] = float(val)
        return action
