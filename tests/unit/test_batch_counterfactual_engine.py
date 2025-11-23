"""
Comprehensive tests for Batch Counterfactual Engine.

Tests cover batch processing, interaction detection, scenario comparison,
and deterministic behavior.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.models.requests import BatchCounterfactualRequest, ScenarioSpec
from src.models.shared import Distribution, DistributionType, StructuralModel
from src.services.batch_counterfactual_engine import BatchCounterfactualEngine


class TestBatchCounterfactualBasic:
    """Basic batch counterfactual analysis tests."""

    def test_batch_processing_multiple_scenarios(self):
        """Test processing multiple scenarios."""
        engine = BatchCounterfactualEngine()

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

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="baseline", intervention={"Price": 40}),
                ScenarioSpec(id="increase", intervention={"Price": 50}),
                ScenarioSpec(id="aggressive", intervention={"Price": 60}),
            ],
            outcome="Revenue",
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Should process all scenarios
        assert len(response.scenarios) == 3
        assert response.scenarios[0].scenario_id == "baseline"
        assert response.scenarios[1].scenario_id == "increase"
        assert response.scenarios[2].scenario_id == "aggressive"

        # Results should be ordered
        baseline_rev = response.scenarios[0].prediction.point_estimate
        increase_rev = response.scenarios[1].prediction.point_estimate
        aggressive_rev = response.scenarios[2].prediction.point_estimate

        assert increase_rev > baseline_rev
        assert aggressive_rev > increase_rev

    def test_deterministic_with_seed(self):
        """Test that same seed produces identical results."""
        engine1 = BatchCounterfactualEngine()
        engine2 = BatchCounterfactualEngine()

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

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="baseline", intervention={"Price": 40}),
                ScenarioSpec(id="increase", intervention={"Price": 50}),
            ],
            outcome="Revenue",
            seed=42,
        )

        response1 = engine1.generate_batch_counterfactuals(request)
        response2 = engine2.generate_batch_counterfactuals(request)

        # Same results
        for s1, s2 in zip(response1.scenarios, response2.scenarios):
            assert abs(s1.prediction.point_estimate - s2.prediction.point_estimate) < 0.01

    def test_scenario_comparison(self):
        """Test scenario comparison and ranking."""
        engine = BatchCounterfactualEngine()

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

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="baseline", intervention={"Price": 40}),
                ScenarioSpec(id="increase", intervention={"Price": 50}),
                ScenarioSpec(id="aggressive", intervention={"Price": 60}),
            ],
            outcome="Revenue",
            analyze_interactions=False,  # Skip interactions for this test
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Check comparison
        assert response.comparison.best_outcome == "aggressive"
        assert len(response.comparison.ranking) == 3
        assert response.comparison.ranking[0] == "aggressive"

        # Check marginal gains
        assert "increase" in response.comparison.marginal_gains
        assert "aggressive" in response.comparison.marginal_gains


class TestInteractionDetection:
    """Test interaction detection algorithm."""

    def test_synergistic_interaction(self):
        """Test detection of synergistic interactions."""
        engine = BatchCounterfactualEngine()

        # Model with synergistic interaction:
        # Revenue = 10000 + 500*Price + 200*Quality + 100*Price*Quality
        # (Interaction term makes combined effect > sum of individual)
        model = StructuralModel(
            variables=["Price", "Quality", "Revenue"],
            equations={"Revenue": "10000 + 500*Price + 200*Quality + 100*Price*Quality"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 10.0}
                )
            }
        )

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="price_only", intervention={"Price": 2}),
                ScenarioSpec(id="quality_only", intervention={"Quality": 3}),
                ScenarioSpec(id="both", intervention={"Price": 2, "Quality": 3}),
            ],
            outcome="Revenue",
            analyze_interactions=True,
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Should detect interaction
        assert response.interactions is not None
        assert len(response.interactions.pairwise) > 0

        # Find Price-Quality interaction
        price_quality = next(
            (i for i in response.interactions.pairwise
             if set(i.variables) == {"Price", "Quality"}),
            None
        )

        assert price_quality is not None
        assert price_quality.type == "synergistic"
        assert price_quality.effect_size > 0

    def test_antagonistic_interaction(self):
        """Test detection of antagonistic interactions."""
        engine = BatchCounterfactualEngine()

        # Model with antagonistic interaction:
        # Revenue = 10000 + 500*Price + 200*Quality - 100*Price*Quality
        # (Negative interaction makes combined effect < sum)
        model = StructuralModel(
            variables=["Price", "Quality", "Revenue"],
            equations={"Revenue": "10000 + 500*Price + 200*Quality - 100*Price*Quality"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 10.0}
                )
            }
        )

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="price_only", intervention={"Price": 2}),
                ScenarioSpec(id="quality_only", intervention={"Quality": 3}),
                ScenarioSpec(id="both", intervention={"Price": 2, "Quality": 3}),
            ],
            outcome="Revenue",
            analyze_interactions=True,
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Should detect interaction
        assert response.interactions is not None
        assert len(response.interactions.pairwise) > 0

        # Find Price-Quality interaction
        price_quality = next(
            (i for i in response.interactions.pairwise
             if set(i.variables) == {"Price", "Quality"}),
            None
        )

        assert price_quality is not None
        assert price_quality.type == "antagonistic"
        assert price_quality.effect_size < 0

    def test_additive_effects(self):
        """Test detection of additive (non-interacting) effects."""
        engine = BatchCounterfactualEngine()

        # Model with purely additive effects (no interaction term)
        model = StructuralModel(
            variables=["Price", "Quality", "Revenue"],
            equations={"Revenue": "10000 + 500*Price + 200*Quality"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 10.0}
                )
            }
        )

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="price_only", intervention={"Price": 10}),
                ScenarioSpec(id="quality_only", intervention={"Quality": 5}),
                ScenarioSpec(id="both", intervention={"Price": 10, "Quality": 5}),
            ],
            outcome="Revenue",
            analyze_interactions=True,
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Should detect interaction
        assert response.interactions is not None

        # Find Price-Quality interaction
        price_quality = next(
            (i for i in response.interactions.pairwise
             if set(i.variables) == {"Price", "Quality"}),
            None
        )

        assert price_quality is not None
        assert price_quality.type == "additive"

    def test_no_interactions_when_disabled(self):
        """Test that interactions are not computed when disabled."""
        engine = BatchCounterfactualEngine()

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

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="baseline", intervention={"Price": 40}),
                ScenarioSpec(id="increase", intervention={"Price": 50}),
            ],
            outcome="Revenue",
            analyze_interactions=False,
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Interactions should be None when disabled
        assert response.interactions is None

    def test_interaction_missing_scenarios(self):
        """Test that interactions require all relevant scenarios."""
        engine = BatchCounterfactualEngine()

        model = StructuralModel(
            variables=["Price", "Quality", "Revenue"],
            equations={"Revenue": "10000 + 500*Price + 200*Quality"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 10.0}
                )
            }
        )

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="price_only", intervention={"Price": 10}),
                # Missing quality_only scenario
                ScenarioSpec(id="both", intervention={"Price": 10, "Quality": 5}),
            ],
            outcome="Revenue",
            analyze_interactions=True,
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Interaction analysis should not find Price-Quality (missing scenario)
        if response.interactions:
            price_quality = next(
                (i for i in response.interactions.pairwise
                 if set(i.variables) == {"Price", "Quality"}),
                None
            )
            assert price_quality is None


class TestScenarioLabels:
    """Test scenario labeling and metadata."""

    def test_scenario_labels_preserved(self):
        """Test that scenario labels are preserved in results."""
        engine = BatchCounterfactualEngine()

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

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(
                    id="baseline",
                    intervention={"Price": 40},
                    label="Current pricing strategy"
                ),
                ScenarioSpec(
                    id="increase",
                    intervention={"Price": 50},
                    label="10% price increase"
                ),
            ],
            outcome="Revenue",
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Labels should be preserved
        assert response.scenarios[0].label == "Current pricing strategy"
        assert response.scenarios[1].label == "10% price increase"


class TestExplanationGeneration:
    """Test explanation generation for batch analysis."""

    def test_explanation_includes_best_scenario(self):
        """Test that explanation mentions best scenario."""
        engine = BatchCounterfactualEngine()

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

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="baseline", intervention={"Price": 40}),
                ScenarioSpec(id="best", intervention={"Price": 60}),
            ],
            outcome="Revenue",
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Explanation should mention best scenario
        assert "best" in response.explanation.summary.lower()

    def test_explanation_mentions_interactions(self):
        """Test that explanation mentions detected interactions."""
        engine = BatchCounterfactualEngine()

        model = StructuralModel(
            variables=["Price", "Quality", "Revenue"],
            equations={"Revenue": "10000 + 500*Price + 200*Quality + 100*Price*Quality"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 10.0}
                )
            }
        )

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="price_only", intervention={"Price": 2}),
                ScenarioSpec(id="quality_only", intervention={"Quality": 3}),
                ScenarioSpec(id="both", intervention={"Price": 2, "Quality": 3}),
            ],
            outcome="Revenue",
            analyze_interactions=True,
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Explanation should mention interactions
        assert "interaction" in response.explanation.reasoning.lower() or \
               "interaction" in response.explanation.summary.lower()


class TestMarginalGains:
    """Test marginal gains computation."""

    def test_marginal_gains_computed(self):
        """Test that marginal gains are computed correctly."""
        engine = BatchCounterfactualEngine()

        model = StructuralModel(
            variables=["Price", "Revenue"],
            equations={"Revenue": "10000 + 1000*Price"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 10.0}
                )
            }
        )

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="baseline", intervention={"Price": 10}),  # Rev = 20000
                ScenarioSpec(id="increase", intervention={"Price": 15}),  # Rev = 25000
            ],
            outcome="Revenue",
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Marginal gain should be approximately 5000
        assert "increase" in response.comparison.marginal_gains
        marginal = response.comparison.marginal_gains["increase"]
        assert abs(marginal - 5000) < 100  # Allow small numerical error


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_variable_multiple_values(self):
        """Test multiple scenarios with same variable, different values."""
        engine = BatchCounterfactualEngine()

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

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="low", intervention={"Price": 30}),
                ScenarioSpec(id="medium", intervention={"Price": 40}),
                ScenarioSpec(id="high", intervention={"Price": 50}),
            ],
            outcome="Revenue",
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Should process all scenarios
        assert len(response.scenarios) == 3

        # Revenue should increase with price
        low_rev = response.scenarios[0].prediction.point_estimate
        medium_rev = response.scenarios[1].prediction.point_estimate
        high_rev = response.scenarios[2].prediction.point_estimate

        assert medium_rev > low_rev
        assert high_rev > medium_rev

    def test_complex_multi_variable_scenarios(self):
        """Test scenarios with multiple variables."""
        engine = BatchCounterfactualEngine()

        model = StructuralModel(
            variables=["Price", "Quality", "Marketing", "Revenue"],
            equations={"Revenue": "10000 + 500*Price + 200*Quality + 0.5*Marketing"},
            distributions={
                "noise": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 100.0}
                )
            }
        )

        request = BatchCounterfactualRequest(
            model=model,
            scenarios=[
                ScenarioSpec(id="baseline", intervention={"Price": 40}),
                ScenarioSpec(id="premium", intervention={"Price": 60, "Quality": 9}),
                ScenarioSpec(id="aggressive", intervention={"Price": 50, "Marketing": 100000}),
            ],
            outcome="Revenue",
            seed=42,
        )

        response = engine.generate_batch_counterfactuals(request)

        # Should process all scenarios
        assert len(response.scenarios) == 3
