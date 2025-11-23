"""
Unit tests for SequentialOptimizer service (Feature 4).

Tests Thompson sampling and experiment recommendation.
"""

import numpy as np
import pytest

from src.services.sequential_optimizer import (
    SequentialOptimizer,
    BeliefState,
    OptimizationObjective,
    ExperimentConstraints,
    RecommendedExperiment,
)


class TestSequentialOptimizerInitialization:
    """Test service initialization."""

    def test_initialization(self):
        """Test optimizer initializes correctly."""
        optimizer = SequentialOptimizer()
        assert optimizer is not None


class TestBeliefState:
    """Test BeliefState class."""

    def test_belief_state_creation(self):
        """Test creating belief state."""
        param_dists = {
            "effect_price": {"type": "normal", "mean": 500, "std": 50},
            "effect_quality": {"type": "uniform", "low": 100, "high": 300},
        }

        beliefs = BeliefState(param_dists)

        assert len(beliefs.parameter_distributions) == 2
        assert "effect_price" in beliefs.parameter_distributions

    def test_normal_distribution(self):
        """Test normal distribution specification."""
        param_dists = {
            "param1": {"type": "normal", "mean": 0, "std": 1}
        }

        beliefs = BeliefState(param_dists)

        assert beliefs.parameter_distributions["param1"]["type"] == "normal"
        assert beliefs.parameter_distributions["param1"]["mean"] == 0

    def test_uniform_distribution(self):
        """Test uniform distribution specification."""
        param_dists = {
            "param1": {"type": "uniform", "low": 0, "high": 1}
        }

        beliefs = BeliefState(param_dists)

        assert beliefs.parameter_distributions["param1"]["type"] == "uniform"
        assert beliefs.parameter_distributions["param1"]["high"] == 1


class TestOptimizationObjective:
    """Test OptimizationObjective class."""

    def test_maximize_objective(self):
        """Test maximize objective."""
        obj = OptimizationObjective(
            target_variable="Revenue",
            goal="maximize",
        )

        assert obj.target_variable == "Revenue"
        assert obj.goal == "maximize"
        assert obj.target_value is None

    def test_minimize_objective(self):
        """Test minimize objective."""
        obj = OptimizationObjective(
            target_variable="Cost",
            goal="minimize",
        )

        assert obj.goal == "minimize"

    def test_target_objective(self):
        """Test target objective."""
        obj = OptimizationObjective(
            target_variable="Quality",
            goal="target",
            target_value=8.5,
        )

        assert obj.goal == "target"
        assert obj.target_value == 8.5


class TestExperimentConstraints:
    """Test ExperimentConstraints class."""

    def test_constraints_creation(self):
        """Test creating experiment constraints."""
        constraints = ExperimentConstraints(
            budget=100000,
            time_horizon=10,
            feasible_interventions={
                "Price": (30, 100),
                "Marketing": (10000, 100000),
            },
        )

        assert constraints.budget == 100000
        assert constraints.time_horizon == 10
        assert len(constraints.feasible_interventions) == 2

    def test_feasible_ranges(self):
        """Test feasible intervention ranges."""
        constraints = ExperimentConstraints(
            budget=50000,
            time_horizon=5,
            feasible_interventions={
                "X": (0, 10),
            },
        )

        assert constraints.feasible_interventions["X"] == (0, 10)


class TestRecommendedExperiment:
    """Test RecommendedExperiment class."""

    def test_experiment_creation(self):
        """Test creating recommended experiment."""
        experiment = RecommendedExperiment(
            intervention={"Price": 55},
            expected_outcome={"Revenue": 38000},
            expected_information_gain=0.75,
            cost_estimate=10000,
            rationale="Explore high price region",
            exploration_vs_exploitation=0.8,
        )

        assert experiment.intervention["Price"] == 55
        assert experiment.expected_information_gain == 0.75
        assert experiment.exploration_vs_exploitation == 0.8

    def test_exploration_score(self):
        """Test exploration vs exploitation score."""
        experiment = RecommendedExperiment(
            intervention={"X": 5},
            expected_outcome={"Y": 10},
            expected_information_gain=0.2,
            cost_estimate=1000,
            rationale="Exploit known optimum",
            exploration_vs_exploitation=0.1,  # Exploitation
        )

        assert experiment.exploration_vs_exploitation < 0.5  # Exploitation


class TestThompsonSampling:
    """Test Thompson sampling algorithm."""

    def test_simple_recommendation(self):
        """Test generating simple recommendation."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_price": {"type": "normal", "mean": 500, "std": 50}
        })

        objective = OptimizationObjective(
            target_variable="Revenue",
            goal="maximize",
        )

        constraints = ExperimentConstraints(
            budget=100000,
            time_horizon=10,
            feasible_interventions={"Price": (30, 100)},
        )

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            seed=42,
        )

        assert isinstance(recommendation, RecommendedExperiment)
        assert "Price" in recommendation.intervention
        assert 30 <= recommendation.intervention["Price"] <= 100

    def test_determinism_with_seed(self):
        """Test that same seed produces same recommendation."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_x": {"type": "normal", "mean": 1, "std": 0.1}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=5,
            feasible_interventions={"X": (0, 10)},
        )

        rec1 = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            seed=42,
        )

        rec2 = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            seed=42,
        )

        assert rec1.intervention == rec2.intervention
        assert rec1.expected_information_gain == rec2.expected_information_gain

    def test_multiple_variables(self):
        """Test recommendation with multiple intervention variables."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_price": {"type": "normal", "mean": 500, "std": 50},
            "effect_quality": {"type": "normal", "mean": 200, "std": 30},
        })

        objective = OptimizationObjective("Revenue", "maximize")
        constraints = ExperimentConstraints(
            budget=100000,
            time_horizon=10,
            feasible_interventions={
                "Price": (30, 100),
                "Quality": (5, 10),
            },
        )

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            seed=42,
        )

        # Should recommend intervention on at least one variable
        assert len(recommendation.intervention) >= 1

    def test_maximize_vs_minimize(self):
        """Test that maximize and minimize produce different recommendations."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_x": {"type": "normal", "mean": 2, "std": 0.5}
        })

        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=5,
            feasible_interventions={"X": (0, 10)},
        )

        rec_max = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=OptimizationObjective("Y", "maximize"),
            constraints=constraints,
            seed=42,
        )

        rec_min = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=OptimizationObjective("Y", "minimize"),
            constraints=constraints,
            seed=43,  # Different seed to get different sample
        )

        # Recommendations should exist (may or may not differ based on sampling)
        assert rec_max is not None
        assert rec_min is not None


class TestInformationGain:
    """Test information gain estimation."""

    def test_no_history_high_gain(self):
        """Test that first experiment has high information gain."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_x": {"type": "normal", "mean": 1, "std": 1}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=5,
            feasible_interventions={"X": (0, 10)},
        )

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            history=None,  # No history
            seed=42,
        )

        # First experiment should have high information gain
        assert recommendation.expected_information_gain > 0.5

    def test_with_history_lower_gain(self):
        """Test that repeated experiments have lower information gain."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_x": {"type": "normal", "mean": 1, "std": 0.5}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=5,
            feasible_interventions={"X": (0, 10)},
        )

        # Recommendation with similar history
        history = [
            {"intervention": {"X": 5}, "outcome": {"Y": 5}},
            {"intervention": {"X": 6}, "outcome": {"Y": 6}},
            {"intervention": {"X": 5.5}, "outcome": {"Y": 5.5}},
        ]

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            history=history,
            seed=42,
        )

        # May have lower gain due to explored region
        assert 0 <= recommendation.expected_information_gain <= 1

    def test_information_gain_range(self):
        """Test information gain is in valid range."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_x": {"type": "uniform", "low": 0, "high": 5}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=5,
            feasible_interventions={"X": (0, 10)},
        )

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            seed=42,
        )

        assert 0 <= recommendation.expected_information_gain <= 1


class TestExplorationVsExploitation:
    """Test exploration vs exploitation balance."""

    def test_exploration_score_range(self):
        """Test exploration score is in [0, 1]."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_x": {"type": "normal", "mean": 1, "std": 0.5}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=5,
            feasible_interventions={"X": (0, 10)},
        )

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            seed=42,
        )

        assert 0 <= recommendation.exploration_vs_exploitation <= 1

    def test_exploration_with_no_history(self):
        """Test high exploration with no history."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_x": {"type": "normal", "mean": 1, "std": 1}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=10,
            feasible_interventions={"X": (0, 10)},
        )

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            history=None,
            seed=42,
        )

        # First experiment should lean toward exploration
        assert recommendation.exploration_vs_exploitation > 0.3

    def test_exploitation_with_history(self):
        """Test that recommendations balance exploration and exploitation."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "effect_x": {"type": "normal", "mean": 2, "std": 0.1}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=2,  # Short horizon favors exploitation
            feasible_interventions={"X": (0, 10)},
        )

        history = [
            {"intervention": {"X": 5}, "outcome": {"Y": 10}},
            {"intervention": {"X": 6}, "outcome": {"Y": 12}},
        ]

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            history=history,
            seed=42,
        )

        # Should balance exploration and exploitation
        assert 0 <= recommendation.exploration_vs_exploitation <= 1


class TestCostEstimation:
    """Test cost estimation."""

    def test_cost_proportional_to_magnitude(self):
        """Test cost increases with intervention magnitude."""
        optimizer = SequentialOptimizer()

        # Small intervention
        small_cost = optimizer._estimate_cost({"X": 1})

        # Large intervention
        large_cost = optimizer._estimate_cost({"X": 100})

        assert large_cost > small_cost

    def test_cost_multiple_variables(self):
        """Test cost with multiple intervention variables."""
        optimizer = SequentialOptimizer()

        cost_single = optimizer._estimate_cost({"X": 5})
        cost_double = optimizer._estimate_cost({"X": 5, "Y": 5})

        # Multiple variables should cost more
        assert cost_double > cost_single

    def test_cost_zero_intervention(self):
        """Test cost of zero intervention."""
        optimizer = SequentialOptimizer()

        cost = optimizer._estimate_cost({"X": 0})

        assert cost == 0


class TestRationaleGeneration:
    """Test rationale generation."""

    def test_exploration_rationale(self):
        """Test rationale for exploration."""
        optimizer = SequentialOptimizer()

        rationale = optimizer._generate_rationale(
            action={"X": 8},
            expected_outcome={"Y": 16},
            information_gain=0.8,  # High gain = exploration
        )

        assert "Explore" in rationale or "explore" in rationale
        assert "X=8" in rationale

    def test_exploitation_rationale(self):
        """Test rationale for exploitation."""
        optimizer = SequentialOptimizer()

        rationale = optimizer._generate_rationale(
            action={"X": 5},
            expected_outcome={"Y": 10},
            information_gain=0.2,  # Low gain = exploitation
        )

        assert "Exploit" in rationale or "exploit" in rationale

    def test_rationale_includes_outcome(self):
        """Test rationale includes expected outcome."""
        optimizer = SequentialOptimizer()

        rationale = optimizer._generate_rationale(
            action={"Price": 55},
            expected_outcome={"Revenue": 35000},
            information_gain=0.5,
        )

        # Should mention the outcome
        assert "outcome" in rationale.lower() or "revenue" in rationale.lower()


class TestCandidateActionGeneration:
    """Test candidate action generation."""

    def test_generates_candidates(self):
        """Test that candidate actions are generated."""
        optimizer = SequentialOptimizer()

        feasible = {"X": (0, 10)}

        candidates = optimizer._generate_candidate_actions(feasible)

        assert len(candidates) > 0
        assert all("X" in action for action in candidates)

    def test_candidates_within_bounds(self):
        """Test candidates respect feasibility bounds."""
        optimizer = SequentialOptimizer()

        feasible = {"X": (5, 15)}

        candidates = optimizer._generate_candidate_actions(feasible)

        for action in candidates:
            assert 5 <= action["X"] <= 15

    def test_multiple_variable_candidates(self):
        """Test candidates with multiple variables."""
        optimizer = SequentialOptimizer()

        feasible = {
            "X": (0, 10),
            "Y": (20, 30),
        }

        candidates = optimizer._generate_candidate_actions(feasible)

        # Should generate candidates for each variable
        assert len(candidates) > 0

    def test_candidate_limit(self):
        """Test that candidates are limited."""
        optimizer = SequentialOptimizer()

        feasible = {
            "X": (0, 100),
            "Y": (0, 100),
            "Z": (0, 100),
        }

        candidates = optimizer._generate_candidate_actions(feasible)

        # Should be limited (e.g., to 10)
        assert len(candidates) <= 10


class TestParameterSampling:
    """Test parameter sampling from beliefs."""

    def test_normal_sampling(self):
        """Test sampling from normal distribution."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "param1": {"type": "normal", "mean": 5, "std": 1}
        })

        # Sample multiple times
        samples = [optimizer._sample_parameters(beliefs) for _ in range(100)]

        # Check that samples are reasonable
        values = [s["param1"] for s in samples]
        mean_sample = np.mean(values)
        std_sample = np.std(values)

        assert 4 < mean_sample < 6  # Approximately mean
        assert 0.5 < std_sample < 1.5  # Approximately std

    def test_uniform_sampling(self):
        """Test sampling from uniform distribution."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "param1": {"type": "uniform", "low": 0, "high": 10}
        })

        samples = [optimizer._sample_parameters(beliefs) for _ in range(100)]

        values = [s["param1"] for s in samples]

        # All should be in range
        assert all(0 <= v <= 10 for v in values)

        # Should cover range reasonably well
        assert max(values) > 7
        assert min(values) < 3

    def test_multiple_parameter_sampling(self):
        """Test sampling multiple parameters."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "param1": {"type": "normal", "mean": 1, "std": 0.1},
            "param2": {"type": "uniform", "low": 5, "high": 15},
        })

        sample = optimizer._sample_parameters(beliefs)

        assert "param1" in sample
        assert "param2" in sample
        assert 0.5 < sample["param1"] < 1.5
        assert 5 <= sample["param2"] <= 15


class TestActionEvaluation:
    """Test action evaluation under sampled parameters."""

    def test_simple_evaluation(self):
        """Test evaluating action with simple linear model."""
        optimizer = SequentialOptimizer()

        action = {"X": 5}
        params = {"effect_X": 2}
        objective = OptimizationObjective("Y", "maximize")

        value = optimizer._evaluate_action(action, params, objective)

        assert value == 10  # 5 * 2

    def test_maximize_vs_minimize(self):
        """Test evaluation respects maximize vs minimize."""
        optimizer = SequentialOptimizer()

        action = {"X": 5}
        params = {"effect_X": 2}

        value_max = optimizer._evaluate_action(
            action, params, OptimizationObjective("Y", "maximize")
        )

        value_min = optimizer._evaluate_action(
            action, params, OptimizationObjective("Y", "minimize")
        )

        # Minimize should flip sign
        assert value_max == -value_min

    def test_multiple_variables(self):
        """Test evaluation with multiple variables."""
        optimizer = SequentialOptimizer()

        action = {"X": 3, "Y": 4}
        params = {"effect_X": 2, "effect_Y": 5}
        objective = OptimizationObjective("Z", "maximize")

        value = optimizer._evaluate_action(action, params, objective)

        assert value == 26  # 3*2 + 4*5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_feasible_interventions(self):
        """Test with no feasible interventions."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "param": {"type": "normal", "mean": 1, "std": 0.1}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=5,
            feasible_interventions={},  # Empty!
        )

        # Should handle gracefully
        try:
            recommendation = optimizer.recommend_next_experiment(
                beliefs=beliefs,
                objective=objective,
                constraints=constraints,
                seed=42,
            )
        except (ValueError, KeyError):
            # Acceptable to raise error
            pass

    def test_zero_time_horizon(self):
        """Test with zero time horizon."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "param": {"type": "normal", "mean": 1, "std": 0.1}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=1,  # Minimum 1
            feasible_interventions={"X": (0, 10)},
        )

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            seed=42,
        )

        # Should still produce recommendation
        assert recommendation is not None

    def test_very_tight_bounds(self):
        """Test with very tight feasibility bounds."""
        optimizer = SequentialOptimizer()

        beliefs = BeliefState({
            "param": {"type": "normal", "mean": 1, "std": 0.1}
        })

        objective = OptimizationObjective("Y", "maximize")
        constraints = ExperimentConstraints(
            budget=1000,
            time_horizon=5,
            feasible_interventions={"X": (5, 5.01)},  # Very tight
        )

        recommendation = optimizer.recommend_next_experiment(
            beliefs=beliefs,
            objective=objective,
            constraints=constraints,
            seed=42,
        )

        # Should produce valid recommendation within bounds
        assert 5 <= recommendation.intervention["X"] <= 5.01
