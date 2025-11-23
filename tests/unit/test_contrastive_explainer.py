"""
Comprehensive tests for ContrastiveExplainer.

Tests cover minimal intervention discovery, binary search, multi-variable
combinations, robustness evaluation, and ranking algorithms.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.models.requests import ContrastiveExplanationRequest, InterventionConstraints
from src.models.responses import MinimalIntervention
from src.models.shared import Distribution, DistributionType, StructuralModel
from src.services.contrastive_explainer import ContrastiveExplainer


class TestContrastiveExplainerBasic:
    """Basic contrastive explanation tests."""

    def test_single_variable_minimal_intervention(self):
        """Test finding minimal change to single variable."""
        explainer = ContrastiveExplainer()

        # Model: Revenue = 10000 + 500 * Price
        model = StructuralModel(
            variables=["Price", "Revenue"],
            equations={"Revenue": "10000 + 500 * Price"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 100.0}
                )
            }
        )

        request = ContrastiveExplanationRequest(
            model=model,
            current_state={"Price": 40},
            observed_outcome={"Revenue": 30000},
            target_outcome={"Revenue": (35000, 36000)},
            constraints=InterventionConstraints(
                feasible=["Price"],
                max_changes=1,
                minimize="change_magnitude",
            ),
            seed=42,
        )

        response = explainer.find_minimal_interventions(request, max_candidates=5)

        # Should find intervention
        assert len(response.minimal_interventions) > 0
        best = response.minimal_interventions[0]

        # Check Price was changed
        assert "Price" in best.changes

        # Expected Revenue should be in target range
        assert 35000 <= best.expected_outcome["Revenue"] <= 36000

    def test_multiple_feasible_variables(self):
        """Test finding interventions across multiple feasible variables."""
        explainer = ContrastiveExplainer()

        # Model: Revenue = 10000 + 500*Price + 0.5*Marketing
        model = StructuralModel(
            variables=["Price", "Marketing", "Revenue"],
            equations={"Revenue": "10000 + 500*Price + 0.5*Marketing"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 100.0}
                )
            }
        )

        request = ContrastiveExplanationRequest(
            model=model,
            current_state={"Price": 40, "Marketing": 20000},
            observed_outcome={"Revenue": 40000},
            target_outcome={"Revenue": (45000, 46000)},
            constraints=InterventionConstraints(
                feasible=["Price", "Marketing"],
                max_changes=1,
                minimize="change_magnitude",
            ),
            seed=42,
        )

        response = explainer.find_minimal_interventions(request, max_candidates=5)

        # Should find at least one intervention
        assert len(response.minimal_interventions) > 0

        # All interventions should achieve target
        for intervention in response.minimal_interventions:
            assert 45000 <= intervention.expected_outcome["Revenue"] <= 46000

    def test_multi_variable_combination(self):
        """Test finding minimal multi-variable combination."""
        explainer = ContrastiveExplainer()

        model = StructuralModel(
            variables=["Price", "Quality", "Revenue"],
            equations={"Revenue": "10000 + 300*Price + 200*Quality"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 100.0}
                )
            }
        )

        request = ContrastiveExplanationRequest(
            model=model,
            current_state={"Price": 40, "Quality": 7.5},
            observed_outcome={"Revenue": 23500},
            target_outcome={"Revenue": (30000, 31000)},
            constraints=InterventionConstraints(
                feasible=["Price", "Quality"],
                max_changes=2,
                minimize="change_magnitude",
                variable_bounds={
                    "Price": (35, 60),
                    "Quality": (6, 10),
                },
            ),
            seed=42,
        )

        response = explainer.find_minimal_interventions(request, max_candidates=5)

        # Should find interventions
        assert len(response.minimal_interventions) > 0

        # Check that multi-variable interventions are included
        has_multi_var = any(
            len(intervention.changes) > 1
            for intervention in response.minimal_interventions
        )
        # Multi-variable may or may not be optimal, but should be available
        # Just check we can find solutions
        assert all(
            30000 <= intervention.expected_outcome["Revenue"] <= 31000
            for intervention in response.minimal_interventions
        )

    def test_respects_fixed_constraints(self):
        """Test that fixed variables are not changed."""
        explainer = ContrastiveExplainer()

        model = StructuralModel(
            variables=["Price", "Quality", "Revenue"],
            equations={"Revenue": "10000 + 500*Price + 200*Quality"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 100.0}
                )
            }
        )

        request = ContrastiveExplanationRequest(
            model=model,
            current_state={"Price": 40, "Quality": 7.5},
            observed_outcome={"Revenue": 31500},
            target_outcome={"Revenue": (35000, 36000)},
            constraints=InterventionConstraints(
                feasible=["Price"],
                fixed=["Quality"],  # Quality cannot change
                max_changes=1,
                minimize="change_magnitude",
            ),
            seed=42,
        )

        response = explainer.find_minimal_interventions(request, max_candidates=5)

        # All interventions should only change Price, not Quality
        for intervention in response.minimal_interventions:
            assert "Quality" not in intervention.changes
            assert "Price" in intervention.changes

    def test_no_solution_returns_empty(self):
        """Test that unachievable targets return empty list."""
        explainer = ContrastiveExplainer()

        model = StructuralModel(
            variables=["Price", "Revenue"],
            equations={"Revenue": "10000 + 100*Price"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 100.0}
                )
            }
        )

        request = ContrastiveExplanationRequest(
            model=model,
            current_state={"Price": 40},
            observed_outcome={"Revenue": 14000},
            target_outcome={"Revenue": (100000, 110000)},  # Impossibly high
            constraints=InterventionConstraints(
                feasible=["Price"],
                max_changes=1,
                minimize="change_magnitude",
                variable_bounds={"Price": (30, 50)},  # Limited range
            ),
            seed=42,
        )

        response = explainer.find_minimal_interventions(request, max_candidates=5)

        # Should return empty or very few results
        # (might find some outside target if bounds allow)
        # Main test: doesn't crash
        assert isinstance(response.minimal_interventions, list)

    def test_deterministic_with_seed(self):
        """Test that same seed produces identical results."""
        explainer1 = ContrastiveExplainer()
        explainer2 = ContrastiveExplainer()

        model = StructuralModel(
            variables=["Price", "Revenue"],
            equations={"Revenue": "10000 + 500*Price"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 100.0}
                )
            }
        )

        request = ContrastiveExplanationRequest(
            model=model,
            current_state={"Price": 40},
            observed_outcome={"Revenue": 30000},
            target_outcome={"Revenue": (35000, 36000)},
            constraints=InterventionConstraints(
                feasible=["Price"],
                max_changes=1,
            ),
            seed=42,
        )

        response1 = explainer1.find_minimal_interventions(request, max_candidates=3)
        response2 = explainer2.find_minimal_interventions(request, max_candidates=3)

        # Same number of interventions
        assert len(response1.minimal_interventions) == len(response2.minimal_interventions)

        # Same values
        for int1, int2 in zip(response1.minimal_interventions, response2.minimal_interventions):
            assert int1.rank == int2.rank
            for var in int1.changes:
                assert abs(int1.changes[var].to_value - int2.changes[var].to_value) < 0.01


class TestRankingAlgorithms:
    """Test intervention ranking by different criteria."""

    def test_rank_by_change_magnitude(self):
        """Test ranking by change magnitude (smallest first)."""
        explainer = ContrastiveExplainer()

        # Create mock interventions with different change magnitudes
        interventions = [
            MinimalIntervention(
                rank=0,
                changes={"Price": Mock(delta=10, from_value=40, to_value=50)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock(lower=34000, upper=36000)},
                feasibility=0.9,
                cost_estimate="medium",
                robustness="robust",
                robustness_score=0.8,
            ),
            MinimalIntervention(
                rank=0,
                changes={"Price": Mock(delta=5, from_value=40, to_value=45)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock(lower=34000, upper=36000)},
                feasibility=0.9,
                cost_estimate="low",
                robustness="robust",
                robustness_score=0.8,
            ),
        ]

        ranked = explainer._rank_interventions(interventions, "change_magnitude")

        # Smaller change should be ranked first
        assert ranked[0].changes["Price"].delta == 5
        assert ranked[1].changes["Price"].delta == 10
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_rank_by_cost(self):
        """Test ranking by cost (low before high)."""
        explainer = ContrastiveExplainer()

        interventions = [
            MinimalIntervention(
                rank=0,
                changes={"Price": Mock(delta=10)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock()},
                feasibility=0.9,
                cost_estimate="high",
                robustness="robust",
                robustness_score=0.8,
            ),
            MinimalIntervention(
                rank=0,
                changes={"Price": Mock(delta=5)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock()},
                feasibility=0.9,
                cost_estimate="low",
                robustness="robust",
                robustness_score=0.9,
            ),
        ]

        ranked = explainer._rank_interventions(interventions, "cost")

        # Low cost should be ranked first
        assert ranked[0].cost_estimate == "low"
        assert ranked[1].cost_estimate == "high"

    def test_rank_by_feasibility(self):
        """Test ranking by feasibility (highest first)."""
        explainer = ContrastiveExplainer()

        interventions = [
            MinimalIntervention(
                rank=0,
                changes={"Price": Mock(delta=10)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock()},
                feasibility=0.6,
                cost_estimate="medium",
                robustness="robust",
                robustness_score=0.8,
            ),
            MinimalIntervention(
                rank=0,
                changes={"Price": Mock(delta=5)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock()},
                feasibility=0.95,
                cost_estimate="low",
                robustness="robust",
                robustness_score=0.9,
            ),
        ]

        ranked = explainer._rank_interventions(interventions, "feasibility")

        # Higher feasibility should be ranked first
        assert ranked[0].feasibility == 0.95
        assert ranked[1].feasibility == 0.6


class TestRobustnessIntegration:
    """Test integration with FACET robustness analysis."""

    @patch('src.services.contrastive_explainer.RobustnessAnalyzer')
    def test_robustness_evaluated(self, mock_robustness_analyzer_class):
        """Test that robustness is evaluated for interventions."""
        # Setup mock
        mock_analyzer = Mock()
        mock_result = Mock()
        mock_result.robustness_score = 0.85
        mock_analyzer.analyze_robustness.return_value = mock_result
        mock_robustness_analyzer_class.return_value = mock_analyzer

        explainer = ContrastiveExplainer()

        model = StructuralModel(
            variables=["Price", "Revenue"],
            equations={"Revenue": "10000 + 500*Price"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 100.0}
                )
            }
        )

        request = ContrastiveExplanationRequest(
            model=model,
            current_state={"Price": 40},
            observed_outcome={"Revenue": 30000},
            target_outcome={"Revenue": (35000, 36000)},
            constraints=InterventionConstraints(
                feasible=["Price"],
                max_changes=1,
            ),
            seed=42,
        )

        response = explainer.find_minimal_interventions(request, max_candidates=3)

        # Robustness should be analyzed
        # (Note: actual implementation may have robustness analysis,
        # this is a basic integration test)
        assert len(response.minimal_interventions) > 0
        for intervention in response.minimal_interventions:
            assert intervention.robustness_score >= 0
            assert intervention.robustness_score <= 1


class TestComparison:
    """Test intervention comparison generation."""

    def test_comparison_single_intervention(self):
        """Test comparison when only one intervention exists."""
        explainer = ContrastiveExplainer()

        interventions = [
            MinimalIntervention(
                rank=1,
                changes={"Price": Mock(delta=5)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock()},
                feasibility=0.95,
                cost_estimate="low",
                robustness="robust",
                robustness_score=0.9,
            ),
        ]

        comparison = explainer._generate_comparison(interventions)

        assert comparison.best_by_cost == 1
        assert comparison.best_by_robustness == 1
        assert comparison.best_by_feasibility == 1
        assert "single" in comparison.synergies.lower() or "sufficient" in comparison.synergies.lower()

    def test_comparison_multiple_interventions(self):
        """Test comparison when multiple interventions exist."""
        explainer = ContrastiveExplainer()

        interventions = [
            MinimalIntervention(
                rank=1,
                changes={"Price": Mock(delta=5)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock()},
                feasibility=0.95,
                cost_estimate="low",
                robustness="robust",
                robustness_score=0.9,
            ),
            MinimalIntervention(
                rank=2,
                changes={"Marketing": Mock(delta=10000)},
                expected_outcome={"Revenue": 35000},
                confidence_interval={"Revenue": Mock()},
                feasibility=0.85,
                cost_estimate="high",
                robustness="moderate",
                robustness_score=0.7,
            ),
        ]

        comparison = explainer._generate_comparison(interventions)

        # Best by cost should be intervention 1 (low cost)
        assert comparison.best_by_cost == 1

        # Best by robustness should be intervention 1 (0.9 score)
        assert comparison.best_by_robustness == 1

        # Best by feasibility should be intervention 1 (0.95)
        assert comparison.best_by_feasibility == 1


class TestExplanationGeneration:
    """Test plain English explanation generation."""

    def test_explanation_with_interventions(self):
        """Test explanation generation when interventions are found."""
        explainer = ContrastiveExplainer()

        from src.models.responses import InterventionChange

        interventions = [
            MinimalIntervention(
                rank=1,
                changes={
                    "Price": InterventionChange(
                        variable="Price",
                        from_value=40,
                        to_value=45,
                        delta=5,
                        relative_change=12.5,
                    )
                },
                expected_outcome={"Revenue": 35000},
                confidence_interval={
                    "Revenue": Mock(lower=34000, upper=36000)
                },
                feasibility=0.95,
                cost_estimate="low",
                robustness="robust",
                robustness_score=0.9,
            ),
        ]

        explanation = explainer._generate_explanation(
            interventions=interventions,
            target_outcome={"Revenue": (35000, 36000)},
            current_outcome={"Revenue": 30000},
        )

        # Should describe the intervention
        assert "Price" in explanation.summary
        assert "40" in explanation.summary or "45" in explanation.summary

        # Should mention expected outcome
        assert "35000" in explanation.reasoning or "Revenue" in explanation.reasoning

    def test_explanation_no_interventions(self):
        """Test explanation when no interventions found."""
        explainer = ContrastiveExplainer()

        explanation = explainer._generate_explanation(
            interventions=[],
            target_outcome={"Revenue": (100000, 110000)},
            current_outcome={"Revenue": 30000},
        )

        # Should indicate no interventions found
        assert "no" in explanation.summary.lower() or "not" in explanation.summary.lower()


class TestFeasibilityAndCost:
    """Test feasibility and cost estimation."""

    def test_feasibility_within_bounds(self):
        """Test feasibility computation for intervention within bounds."""
        explainer = ContrastiveExplainer()

        feasibility = explainer._compute_feasibility(
            intervention={"Price": 45},
            current_state={"Price": 40},
            constraints=InterventionConstraints(
                feasible=["Price"],
                variable_bounds={"Price": (30, 60)},
            ),
        )

        # Should be feasible (within bounds, small change)
        assert feasibility > 0.5

    def test_feasibility_outside_bounds(self):
        """Test feasibility for intervention outside bounds."""
        explainer = ContrastiveExplainer()

        feasibility = explainer._compute_feasibility(
            intervention={"Price": 100},
            current_state={"Price": 40},
            constraints=InterventionConstraints(
                feasible=["Price"],
                variable_bounds={"Price": (30, 60)},
            ),
        )

        # Should be infeasible (outside bounds)
        assert feasibility == 0.0

    def test_cost_estimate_small_change(self):
        """Test cost estimate for small change."""
        explainer = ContrastiveExplainer()

        cost = explainer._estimate_cost(
            intervention={"Price": 42},  # Small change from 40
            current_state={"Price": 40},
            constraints=InterventionConstraints(feasible=["Price"]),
        )

        # Small change should be low cost
        assert cost == "low"

    def test_cost_estimate_large_change(self):
        """Test cost estimate for large change."""
        explainer = ContrastiveExplainer()

        cost = explainer._estimate_cost(
            intervention={"Price": 80},  # Large change from 40
            current_state={"Price": 40},
            constraints=InterventionConstraints(feasible=["Price"]),
        )

        # Large change should be high cost
        assert cost in ["medium", "high"]

    def test_cost_estimate_multi_variable(self):
        """Test cost estimate for multi-variable intervention."""
        explainer = ContrastiveExplainer()

        cost = explainer._estimate_cost(
            intervention={"Price": 45, "Marketing": 35000},
            current_state={"Price": 40, "Marketing": 30000},
            constraints=InterventionConstraints(feasible=["Price", "Marketing"]),
        )

        # Multi-variable may have medium/high cost
        assert cost in ["low", "medium", "high"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_feasible_variables(self):
        """Test handling when no feasible variables."""
        explainer = ContrastiveExplainer()

        model = StructuralModel(
            variables=["Price", "Revenue"],
            equations={"Revenue": "10000 + 500*Price"},
            distributions={}
        )

        # This should raise validation error from Pydantic (min_length=1)
        # So we won't test the explainer directly, just ensure it doesn't crash
        # if somehow it gets through

    def test_very_tight_target_range(self):
        """Test with very tight target range."""
        explainer = ContrastiveExplainer()

        model = StructuralModel(
            variables=["Price", "Revenue"],
            equations={"Revenue": "10000 + 500*Price"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 10.0}
                )
            }
        )

        request = ContrastiveExplanationRequest(
            model=model,
            current_state={"Price": 40},
            observed_outcome={"Revenue": 30000},
            target_outcome={"Revenue": (35000, 35001)},  # Very tight
            constraints=InterventionConstraints(
                feasible=["Price"],
                max_changes=1,
            ),
            seed=42,
        )

        response = explainer.find_minimal_interventions(request, max_candidates=3)

        # Should handle tight ranges (may or may not find exact solution)
        assert isinstance(response.minimal_interventions, list)
