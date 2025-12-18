"""
Comprehensive tests for CounterfactualEngine.

Tests cover Monte Carlo simulation, equation evaluation, uncertainty analysis,
robustness testing, and error handling.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.models.requests import CounterfactualRequest
from src.models.shared import Distribution, DistributionType, StructuralModel
from src.services.counterfactual_engine import CounterfactualEngine
from src.utils.rng import SeededRNG


class TestCounterfactualEngineBasic:
    """Basic counterfactual analysis tests."""

    def test_simple_linear_model(self):
        """Test simple linear counterfactual: Y = 10 + 2*X."""
        engine = CounterfactualEngine()

        model = StructuralModel(
            variables=["X", "Y"],
            equations={"Y": "10 + 2*X"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 5.0, "std": 1.0}
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={"X": 10.0},
            outcome="Y",
            context={}
        )

        response = engine.analyze(request)

        # Y = 10 + 2*10 = 30 (deterministic since X is fixed by intervention)
        assert abs(response.prediction.point_estimate - 30.0) < 0.1
        # With fixed intervention, CI should be exactly 30
        assert abs(response.prediction.confidence_interval.lower - 30.0) < 0.1
        assert abs(response.prediction.confidence_interval.upper - 30.0) < 0.1
        assert response.uncertainty is not None
        assert response.robustness is not None

    def test_multivariate_model(self):
        """Test multivariate model: Y = a + b*X + c*Z."""
        engine = CounterfactualEngine()

        model = StructuralModel(
            variables=["X", "Z", "Y"],
            equations={"Y": "5 + 2*X + 3*Z"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 1.0}
                ),
                "Z": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 1.0}
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={"X": 2.0, "Z": 3.0},
            outcome="Y",
            context={}
        )

        response = engine.analyze(request)

        # Y = 5 + 2*2 + 3*3 = 5 + 4 + 9 = 18
        assert abs(response.prediction.point_estimate - 18.0) < 0.5

    def test_with_context(self):
        """Test counterfactual with observed context."""
        engine = CounterfactualEngine()

        model = StructuralModel(
            variables=["X", "Y", "Z"],
            equations={"Z": "X + Y"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 1.0}
                ),
                "Y": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 1.0}
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={"X": 5.0},
            outcome="Z",
            context={"Y": 3.0}  # Fix Y to observed value
        )

        response = engine.analyze(request)

        # Z = X + Y = 5 + 3 = 8
        assert abs(response.prediction.point_estimate - 8.0) < 0.5


class TestTopologicalSorting:
    """Test topological sorting of structural equations."""

    def test_simple_chain(self):
        """Test sorting simple dependency chain: X -> Y -> Z."""
        engine = CounterfactualEngine()

        equations = {
            "Z": "2 * Y",
            "Y": "X + 1",
            "X": "5"
        }

        sorted_eqs = engine._topological_sort_equations(equations)
        sorted_vars = [var for var, _ in sorted_eqs]

        # X should come before Y, Y before Z
        assert sorted_vars.index("X") < sorted_vars.index("Y")
        assert sorted_vars.index("Y") < sorted_vars.index("Z")

    def test_independent_variables(self):
        """Test sorting independent variables."""
        engine = CounterfactualEngine()

        equations = {
            "A": "1",
            "B": "2",
            "C": "3"
        }

        sorted_eqs = engine._topological_sort_equations(equations)

        # All variables should be sorted (order doesn't matter for independent vars)
        assert len(sorted_eqs) == 3
        sorted_vars = [var for var, _ in sorted_eqs]
        assert set(sorted_vars) == {"A", "B", "C"}

    def test_complex_dependencies(self):
        """Test complex dependency graph."""
        engine = CounterfactualEngine()

        equations = {
            "D": "B + C",
            "C": "A",
            "B": "A",
            "A": "1"
        }

        sorted_eqs = engine._topological_sort_equations(equations)
        sorted_vars = [var for var, _ in sorted_eqs]

        # A should be first
        assert sorted_vars[0] == "A"
        # B and C depend on A, should come before D
        assert sorted_vars.index("B") < sorted_vars.index("D")
        assert sorted_vars.index("C") < sorted_vars.index("D")

    def test_circular_dependency_raises_error(self):
        """Test that circular dependencies are detected."""
        engine = CounterfactualEngine()

        equations = {
            "A": "B + 1",
            "B": "A + 1"
        }

        with pytest.raises(ValueError, match="Circular dependencies"):
            engine._topological_sort_equations(equations)

    def test_self_dependency_raises_error(self):
        """Test that self-dependencies are detected."""
        engine = CounterfactualEngine()

        equations = {
            "A": "A + 1"
        }

        with pytest.raises(ValueError, match="Circular dependencies"):
            engine._topological_sort_equations(equations)


class TestDistributionSampling:
    """Test sampling from different distributions."""

    def test_sample_normal_distribution(self):
        """Test sampling from normal distribution."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        samples = engine._sample_distribution(
            "normal",
            {"mean": 10.0, "std": 2.0},
            1000,
            rng
        )

        assert len(samples) == 1000
        assert 8.0 < np.mean(samples) < 12.0  # Should be close to mean=10
        assert 1.5 < np.std(samples) < 2.5   # Should be close to std=2

    def test_sample_uniform_distribution(self):
        """Test sampling from uniform distribution."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        samples = engine._sample_distribution(
            "uniform",
            {"min": 0.0, "max": 10.0},
            1000,
            rng
        )

        assert len(samples) == 1000
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 10.0)
        assert 4.0 < np.mean(samples) < 6.0  # Should be close to midpoint=5

    def test_sample_beta_distribution(self):
        """Test sampling from beta distribution."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        samples = engine._sample_distribution(
            "beta",
            {"alpha": 2.0, "beta": 5.0},
            1000,
            rng
        )

        assert len(samples) == 1000
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_sample_exponential_distribution(self):
        """Test sampling from exponential distribution."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        samples = engine._sample_distribution(
            "exponential",
            {"scale": 2.0},
            1000,
            rng
        )

        assert len(samples) == 1000
        assert np.all(samples >= 0.0)
        assert 1.5 < np.mean(samples) < 2.5  # Mean should be close to scale

    def test_unknown_distribution_raises_error(self):
        """Test that unknown distribution type raises error."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        with pytest.raises(ValueError, match="Unknown distribution type"):
            engine._sample_distribution(
                "unknown_distribution",
                {"param": 1.0},
                100,
                rng
            )


class TestEquationEvaluation:
    """Test structural equation evaluation."""

    def test_evaluate_simple_equation(self):
        """Test evaluating simple arithmetic equation."""
        engine = CounterfactualEngine()

        samples = {
            "X": np.array([1.0, 2.0, 3.0])
        }

        result = engine._evaluate_equation("2 * X + 5", samples)

        expected = np.array([7.0, 9.0, 11.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_with_multiple_variables(self):
        """Test equation with multiple variables."""
        engine = CounterfactualEngine()

        samples = {
            "X": np.array([1.0, 2.0, 3.0]),
            "Y": np.array([10.0, 20.0, 30.0])
        }

        result = engine._evaluate_equation("X + Y", samples)

        expected = np.array([11.0, 22.0, 33.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_with_math_functions(self):
        """Test equation with mathematical functions."""
        engine = CounterfactualEngine()

        samples = {
            "X": np.array([1.0, 4.0, 9.0])
        }

        result = engine._evaluate_equation("sqrt(X)", samples)

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_exp_function(self):
        """Test equation with exponential function."""
        engine = CounterfactualEngine()

        samples = {
            "X": np.array([0.0, 1.0, 2.0])
        }

        result = engine._evaluate_equation("exp(X)", samples)

        expected = np.array([1.0, np.e, np.e**2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_log_function(self):
        """Test equation with logarithm function."""
        engine = CounterfactualEngine()

        samples = {
            "X": np.array([1.0, np.e, np.e**2])
        }

        result = engine._evaluate_equation("log(X)", samples)

        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_equation_raises_error(self):
        """Test that invalid equations raise errors."""
        engine = CounterfactualEngine()

        samples = {"X": np.array([1.0, 2.0, 3.0])}

        with pytest.raises(ValueError, match="Invalid equation"):
            engine._evaluate_equation("X + undefined_var", samples)


class TestPredictionComputation:
    """Test prediction statistics computation."""

    def test_compute_prediction_normal_distribution(self):
        """Test prediction computation from normal distribution."""
        engine = CounterfactualEngine()

        # Create normally distributed samples around mean=50
        np.random.seed(42)
        samples = {
            "Y": np.random.normal(50, 5, 1000)
        }

        prediction = engine._compute_prediction(samples, "Y")

        # Point estimate should be close to 50
        assert 45 < prediction.point_estimate < 55

        # Confidence interval should roughly contain the mean
        assert prediction.confidence_interval.lower < 55
        assert prediction.confidence_interval.upper > 45

        # Sensitivity range ordering (10th/90th percentiles)
        assert prediction.sensitivity_range.pessimistic < prediction.point_estimate
        assert prediction.sensitivity_range.optimistic > prediction.point_estimate

    def test_prediction_uses_median(self):
        """Test that point estimate uses median (robust to outliers)."""
        engine = CounterfactualEngine()

        # Create samples with outliers
        samples = {
            "Y": np.array([1, 2, 3, 4, 5, 100, 200])  # Last two are outliers
        }

        prediction = engine._compute_prediction(samples, "Y")

        # Median should be 4, not affected by outliers
        assert abs(prediction.point_estimate - 4.0) < 0.1


class TestUncertaintyAnalysis:
    """Test uncertainty breakdown analysis."""

    def test_analyze_uncertainty_single_source(self):
        """Test uncertainty analysis with single exogenous variable."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        model = StructuralModel(
            variables=["X", "Y"],
            equations={"Y": "2 * X"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 10.0, "std": 2.0}
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={},
            outcome="Y",
            context={}
        )

        # Generate samples
        samples = engine._run_monte_carlo(request, rng)

        uncertainty = engine._analyze_uncertainty(request, samples)

        assert uncertainty.overall in ["low", "medium", "high"]
        assert len(uncertainty.sources) >= 1
        assert uncertainty.sources[0].factor is not None

    def test_uncertainty_level_classification(self):
        """Test classification of uncertainty levels."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        model = StructuralModel(
            variables=["X", "Y"],
            equations={"Y": "X"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 100.0, "std": 5.0}  # Low CV = 0.05
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={},
            outcome="Y",
            context={}
        )

        samples = engine._run_monte_carlo(request, rng)
        uncertainty = engine._analyze_uncertainty(request, samples)

        # Low coefficient of variation should give LOW uncertainty
        assert uncertainty.overall == "low"


class TestRobustnessAnalysis:
    """Test robustness analysis."""

    def test_analyze_robustness(self):
        """Test robustness analysis generates critical assumptions."""
        engine = CounterfactualEngine()

        model = StructuralModel(
            variables=["X", "Y"],
            equations={"Y": "2 * X + 5"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 10.0, "std": 1.0}
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={"X": 10.0},
            outcome="Y",
            context={}
        )

        baseline_result = 25.0  # 2*10 + 5
        robustness = engine._analyze_robustness(request, baseline_result)

        assert robustness.score in ["robust", "moderate", "fragile"]
        assert len(robustness.critical_assumptions) > 0
        assert robustness.critical_assumptions[0].assumption is not None
        assert robustness.critical_assumptions[0].impact >= 0


class TestDistributionConfidence:
    """Test distribution confidence assessment."""

    def test_high_confidence_low_cv(self):
        """Test high confidence for low coefficient of variation."""
        engine = CounterfactualEngine()

        # CV = 1/100 = 0.01 < 0.1 → HIGH
        params = {"mean": 100.0, "std": 1.0}
        confidence = engine._assess_distribution_confidence(params)

        assert confidence == "high"

    def test_medium_confidence_moderate_cv(self):
        """Test medium confidence for moderate CV."""
        engine = CounterfactualEngine()

        # CV = 20/100 = 0.2 (between 0.1 and 0.3) → MEDIUM
        params = {"mean": 100.0, "std": 20.0}
        confidence = engine._assess_distribution_confidence(params)

        assert confidence == "medium"

    def test_low_confidence_high_cv(self):
        """Test low confidence for high CV."""
        engine = CounterfactualEngine()

        # CV = 50/100 = 0.5 > 0.3 → LOW
        params = {"mean": 100.0, "std": 50.0}
        confidence = engine._assess_distribution_confidence(params)

        assert confidence == "low"

    def test_default_confidence_non_normal(self):
        """Test default confidence for non-normal distributions."""
        engine = CounterfactualEngine()

        # Uniform distribution parameters
        params = {"min": 0.0, "max": 10.0}
        confidence = engine._assess_distribution_confidence(params)

        assert confidence == "medium"


class TestFactorNameFormatting:
    """Test factor name formatting."""

    def test_format_snake_case(self):
        """Test formatting snake_case variable names."""
        engine = CounterfactualEngine()

        result = engine._format_factor_name("market_demand")
        assert result == "Market Demand"

    def test_format_camel_case(self):
        """Test formatting camelCase variable names."""
        engine = CounterfactualEngine()

        result = engine._format_factor_name("marketDemand")
        assert result == "Market Demand"

    def test_format_simple_name(self):
        """Test formatting simple variable names."""
        engine = CounterfactualEngine()

        result = engine._format_factor_name("price")
        assert result == "Price"


class TestErrorHandling:
    """Test error handling in counterfactual analysis."""

    def test_missing_outcome_variable(self):
        """Test error when outcome variable not in samples."""
        engine = CounterfactualEngine()

        model = StructuralModel(
            variables=["X", "Y"],
            equations={},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 1.0}
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={},
            outcome="Z",  # Z doesn't exist
            context={}
        )

        with pytest.raises(Exception):
            engine.analyze(request)

    def test_invalid_equation_in_model(self):
        """Test error handling for invalid equations."""
        engine = CounterfactualEngine()

        model = StructuralModel(
            variables=["X", "Y"],
            equations={"Y": "invalid_function(X)"},  # Invalid function
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 5.0, "std": 1.0}
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={},
            outcome="Y",
            context={}
        )

        # Should raise ValueError for invalid function
        with pytest.raises(ValueError, match="Invalid equation"):
            engine.analyze(request)


class TestMonteCarloIntegration:
    """Integration tests for Monte Carlo simulation."""

    def test_monte_carlo_respects_intervention(self):
        """Test that Monte Carlo respects intervention values."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        model = StructuralModel(
            variables=["X", "Y"],
            equations={"Y": "X"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 10.0}  # High variance
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={"X": 42.0},  # Fix X to 42
            outcome="Y",
            context={}
        )

        samples = engine._run_monte_carlo(request, rng)

        # All X samples should be exactly 42
        assert np.all(samples["X"] == 42.0)
        # All Y samples should also be 42 (since Y=X)
        assert np.all(samples["Y"] == 42.0)

    def test_monte_carlo_respects_context(self):
        """Test that Monte Carlo respects context values."""
        engine = CounterfactualEngine()
        rng = SeededRNG(42)

        model = StructuralModel(
            variables=["X", "Y", "Z"],
            equations={"Z": "X + Y"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 1.0}
                ),
                "Y": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 0.0, "std": 1.0}
                )
            }
        )

        request = CounterfactualRequest(
            model=model,
            intervention={"X": 5.0},
            outcome="Z",
            context={"Y": 10.0}  # Observed value
        )

        samples = engine._run_monte_carlo(request, rng)

        # X should be 5, Y should be 10
        assert np.all(samples["X"] == 5.0)
        assert np.all(samples["Y"] == 10.0)
        # Z should be 15 (5 + 10)
        assert np.all(samples["Z"] == 15.0)
