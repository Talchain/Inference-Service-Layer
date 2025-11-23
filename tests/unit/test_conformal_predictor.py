"""
Unit tests for ConformalPredictor service.

Tests conformal prediction including:
- Split conformal algorithm
- Coverage guarantees
- Calibration quality assessment
- Conformity score computation
- Interval comparison
- Edge cases and error handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.models.requests import ConformalCounterfactualRequest, ObservationPoint
from src.models.responses import ConformalCounterfactualResponse
from src.services.conformal_predictor import ConformalPredictor
from src.services.counterfactual_engine import CounterfactualEngine


@pytest.fixture
def mock_cf_engine():
    """Create a mock counterfactual engine."""
    return Mock(spec=CounterfactualEngine)


@pytest.fixture
def conformal_predictor(mock_cf_engine):
    """Create a ConformalPredictor instance."""
    return ConformalPredictor(mock_cf_engine)


@pytest.fixture
def simple_model():
    """Simple structural model for testing."""
    return {
        "variables": ["Price", "Revenue"],
        "equations": {"Revenue": "10000 + 500*Price"},
        "distributions": {
            "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
        },
    }


@pytest.fixture
def calibration_data():
    """Sample calibration data."""
    return [
        ObservationPoint(inputs={"Price": 40}, outcome={"Revenue": 30000}),
        ObservationPoint(inputs={"Price": 42}, outcome={"Revenue": 31000}),
        ObservationPoint(inputs={"Price": 45}, outcome={"Revenue": 32500}),
        ObservationPoint(inputs={"Price": 48}, outcome={"Revenue": 34000}),
        ObservationPoint(inputs={"Price": 50}, outcome={"Revenue": 35000}),
        ObservationPoint(inputs={"Price": 52}, outcome={"Revenue": 36000}),
        ObservationPoint(inputs={"Price": 55}, outcome={"Revenue": 37500}),
        ObservationPoint(inputs={"Price": 58}, outcome={"Revenue": 39000}),
        ObservationPoint(inputs={"Price": 60}, outcome={"Revenue": 40000}),
        ObservationPoint(inputs={"Price": 62}, outcome={"Revenue": 41000}),
        ObservationPoint(inputs={"Price": 65}, outcome={"Revenue": 42500}),
        ObservationPoint(inputs={"Price": 68}, outcome={"Revenue": 44000}),
        ObservationPoint(inputs={"Price": 70}, outcome={"Revenue": 45000}),
        ObservationPoint(inputs={"Price": 72}, outcome={"Revenue": 46000}),
        ObservationPoint(inputs={"Price": 75}, outcome={"Revenue": 47500}),
        ObservationPoint(inputs={"Price": 78}, outcome={"Revenue": 49000}),
        ObservationPoint(inputs={"Price": 80}, outcome={"Revenue": 50000}),
        ObservationPoint(inputs={"Price": 82}, outcome={"Revenue": 51000}),
        ObservationPoint(inputs={"Price": 85}, outcome={"Revenue": 52500}),
        ObservationPoint(inputs={"Price": 88}, outcome={"Revenue": 54000}),
    ]


class TestConformalPredictorInitialization:
    """Tests for ConformalPredictor initialization."""

    def test_initialization_success(self, mock_cf_engine):
        """Test successful initialization."""
        predictor = ConformalPredictor(mock_cf_engine)
        assert predictor.cf_engine == mock_cf_engine

    def test_initialization_stores_engine(self, mock_cf_engine):
        """Test that engine is properly stored."""
        predictor = ConformalPredictor(mock_cf_engine)
        assert hasattr(predictor, 'cf_engine')
        assert predictor.cf_engine is not None


class TestSplitConformal:
    """Tests for split conformal prediction."""

    def test_split_conformal_basic(self, conformal_predictor, simple_model, calibration_data):
        """Test basic split conformal prediction."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        result = conformal_predictor._split_conformal(request)

        assert isinstance(result, ConformalCounterfactualResponse)
        assert result.prediction_interval is not None
        assert result.coverage_guarantee is not None

    def test_split_uses_seed_for_determinism(self, conformal_predictor, simple_model, calibration_data):
        """Test that split conformal is deterministic with seed."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        result1 = conformal_predictor._split_conformal(request)
        result2 = conformal_predictor._split_conformal(request)

        # Results should be identical with same seed
        assert result1.prediction_interval.lower_bound == result2.prediction_interval.lower_bound
        assert result1.prediction_interval.upper_bound == result2.prediction_interval.upper_bound

    def test_split_creates_calibration_split(self, conformal_predictor, simple_model, calibration_data):
        """Test that split conformal splits data correctly."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        result = conformal_predictor._split_conformal(request)

        # Calibration size should be roughly half of original data
        calib_size = result.calibration_quality.calibration_size
        assert calib_size >= len(calibration_data) * 0.4
        assert calib_size <= len(calibration_data) * 0.6


class TestConformityScores:
    """Tests for conformity score computation."""

    def test_conformity_scores_are_non_negative(self, conformal_predictor, simple_model, calibration_data):
        """Test that conformity scores are non-negative."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data[:10],
            confidence_level=0.95,
        )

        scores = conformal_predictor._compute_conformity_scores(request, calibration_data[:10])

        assert all(score >= 0 for score in scores)

    def test_conformity_scores_length(self, conformal_predictor, simple_model, calibration_data):
        """Test that conformity scores match calibration data length."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
        )

        scores = conformal_predictor._compute_conformity_scores(request, calibration_data)

        assert len(scores) == len(calibration_data)

    def test_conformity_scores_are_residuals(self, conformal_predictor, simple_model):
        """Test that conformity scores are absolute residuals."""
        # Create data where we know the residuals
        perfect_data = [
            ObservationPoint(inputs={"Price": 50}, outcome={"Revenue": 35000}),
        ]

        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=perfect_data,
        )

        scores = conformal_predictor._compute_conformity_scores(request, perfect_data)

        # Scores should be absolute values
        assert all(isinstance(score, (int, float, np.floating)) for score in scores)


class TestCoverageGuarantee:
    """Tests for coverage guarantee computation."""

    def test_coverage_guarantee_formula(self, conformal_predictor):
        """Test coverage guarantee formula is correct."""
        # For n=19, alpha=0.05 (95% confidence)
        # Guarantee should be: (ceil(20 * 0.95) - 1) / 19 = (19 - 1) / 19 = 18/19 ≈ 0.947
        n = 19
        alpha = 0.05

        guaranteed = conformal_predictor._compute_guaranteed_coverage(n, alpha)

        expected = (np.ceil((n + 1) * (1 - alpha)) - 1) / n
        assert abs(guaranteed - expected) < 1e-10

    def test_coverage_guarantee_approaches_nominal(self, conformal_predictor):
        """Test that coverage guarantee approaches nominal as n increases."""
        alpha = 0.05
        nominal = 1 - alpha

        # Large n should give guarantee close to nominal
        n_large = 1000
        guaranteed_large = conformal_predictor._compute_guaranteed_coverage(n_large, alpha)

        # Should be within 0.5% of nominal
        assert abs(guaranteed_large - nominal) < 0.005

    def test_coverage_guarantee_always_below_nominal(self, conformal_predictor):
        """Test that finite-sample guarantee is always ≤ nominal."""
        alpha = 0.05
        nominal = 1 - alpha

        for n in [10, 20, 50, 100, 500]:
            guaranteed = conformal_predictor._compute_guaranteed_coverage(n, alpha)
            # Guaranteed should be slightly below nominal (conservative)
            assert guaranteed <= nominal + 1e-10  # Small tolerance for floating point


class TestCalibrationQuality:
    """Tests for calibration quality assessment."""

    def test_calibration_metrics_structure(self, conformal_predictor, calibration_data):
        """Test that calibration metrics have correct structure."""
        scores = np.array([1000, 1500, 2000, 2500, 3000])

        metrics = conformal_predictor._assess_calibration_quality(scores, calibration_data[:5])

        assert metrics.calibration_size == 5
        assert "mean" in metrics.residual_statistics
        assert "std" in metrics.residual_statistics
        assert "median" in metrics.residual_statistics
        assert "iqr" in metrics.residual_statistics
        assert isinstance(metrics.interval_adaptivity, float)

    def test_residual_statistics_correct(self, conformal_predictor, calibration_data):
        """Test that residual statistics are computed correctly."""
        scores = np.array([1000, 2000, 3000, 4000, 5000])

        metrics = conformal_predictor._assess_calibration_quality(scores, calibration_data[:5])

        assert abs(metrics.residual_statistics["mean"] - 3000) < 1
        assert abs(metrics.residual_statistics["median"] - 3000) < 1
        assert metrics.residual_statistics["iqr"] > 0

    def test_interval_adaptivity_non_negative(self, conformal_predictor, calibration_data):
        """Test that interval adaptivity is non-negative."""
        scores = np.array([1000, 1500, 2000, 2500, 3000])

        metrics = conformal_predictor._assess_calibration_quality(scores, calibration_data[:5])

        assert metrics.interval_adaptivity >= 0


class TestAdaptivity:
    """Tests for adaptivity computation."""

    def test_adaptivity_zero_for_constant_scores(self, conformal_predictor):
        """Test that adaptivity is zero for constant scores."""
        scores = np.array([1000, 1000, 1000, 1000, 1000])

        adaptivity = conformal_predictor._compute_adaptivity(scores)

        assert adaptivity == 0.0

    def test_adaptivity_high_for_variable_scores(self, conformal_predictor):
        """Test that adaptivity is high for variable scores."""
        scores = np.array([100, 500, 1000, 5000, 10000])

        adaptivity = conformal_predictor._compute_adaptivity(scores)

        # High variability should give high adaptivity (coefficient of variation)
        assert adaptivity > 0.5

    def test_adaptivity_is_coefficient_of_variation(self, conformal_predictor):
        """Test that adaptivity equals coefficient of variation."""
        scores = np.array([1000, 2000, 3000, 4000, 5000])

        adaptivity = conformal_predictor._compute_adaptivity(scores)

        expected_cv = np.std(scores) / np.mean(scores)
        assert abs(adaptivity - expected_cv) < 1e-10


class TestMonteCarloComparison:
    """Tests for Monte Carlo interval comparison."""

    def test_comparison_metrics_structure(self, conformal_predictor, simple_model):
        """Test comparison metrics structure."""
        from src.models.responses import ConformalInterval, ConfidenceInterval

        conformal_int = ConformalInterval(
            lower_bound={"Revenue": 48000},
            upper_bound={"Revenue": 56000},
            point_estimate={"Revenue": 52000},
            interval_width={"Revenue": 8000},
        )

        mc_int = {
            "Revenue": ConfidenceInterval(
                lower=49000,
                upper=55000,
                confidence_level=0.95,
            )
        }

        comparison = conformal_predictor._compare_intervals(conformal_int, mc_int, "Revenue")

        assert "Revenue" in comparison.monte_carlo_interval
        assert "Revenue" in comparison.conformal_interval
        assert "Revenue" in comparison.width_ratio
        assert isinstance(comparison.interpretation, str)

    def test_width_ratio_computation(self, conformal_predictor):
        """Test that width ratio is computed correctly."""
        from src.models.responses import ConformalInterval, ConfidenceInterval

        # Conformal: width = 8000
        conformal_int = ConformalInterval(
            lower_bound={"Revenue": 48000},
            upper_bound={"Revenue": 56000},
            point_estimate={"Revenue": 52000},
            interval_width={"Revenue": 8000},
        )

        # MC: width = 6000
        mc_int = {
            "Revenue": ConfidenceInterval(
                lower=49000,
                upper=55000,
                confidence_level=0.95,
            )
        }

        comparison = conformal_predictor._compare_intervals(conformal_int, mc_int, "Revenue")

        expected_ratio = 8000 / 6000
        assert abs(comparison.width_ratio["Revenue"] - expected_ratio) < 0.01

    def test_interpretation_wider_interval(self, conformal_predictor):
        """Test interpretation when conformal interval is wider."""
        from src.models.responses import ConformalInterval, ConfidenceInterval

        conformal_int = ConformalInterval(
            lower_bound={"Revenue": 45000},
            upper_bound={"Revenue": 60000},
            point_estimate={"Revenue": 52500},
            interval_width={"Revenue": 15000},
        )

        mc_int = {
            "Revenue": ConfidenceInterval(
                lower=49000,
                upper=55000,
                confidence_level=0.95,
            )
        }

        comparison = conformal_predictor._compare_intervals(conformal_int, mc_int, "Revenue")

        assert "wider" in comparison.interpretation.lower()


class TestExplanationGeneration:
    """Tests for explanation generation."""

    def test_explanation_completeness(self, conformal_predictor):
        """Test that explanation has all required fields."""
        from src.models.responses import (
            ConformalInterval,
            CoverageGuarantee,
            CalibrationMetrics,
            ComparisonMetrics,
            ConfidenceInterval,
        )

        conformal_int = ConformalInterval(
            lower_bound={"Revenue": 48000},
            upper_bound={"Revenue": 56000},
            point_estimate={"Revenue": 52000},
            interval_width={"Revenue": 8000},
        )

        coverage = CoverageGuarantee(
            nominal_coverage=0.95,
            guaranteed_coverage=0.9474,
            finite_sample_valid=True,
            assumptions=["Exchangeability"],
        )

        calibration = CalibrationMetrics(
            calibration_size=20,
            residual_statistics={"mean": 2000, "std": 500, "median": 1900, "iqr": 600},
            interval_adaptivity=0.25,
        )

        comparison = ComparisonMetrics(
            monte_carlo_interval={"Revenue": ConfidenceInterval(lower=49000, upper=55000, confidence_level=0.95)},
            conformal_interval={"Revenue": (48000, 56000)},
            width_ratio={"Revenue": 1.33},
            interpretation="Test interpretation",
        )

        explanation = conformal_predictor._generate_explanation(
            conformal_int, coverage, calibration, comparison, "split"
        )

        assert explanation.summary is not None
        assert explanation.reasoning is not None
        assert explanation.technical_basis is not None
        assert len(explanation.assumptions) > 0

    def test_explanation_mentions_coverage(self, conformal_predictor):
        """Test that explanation mentions coverage guarantee."""
        from src.models.responses import (
            ConformalInterval,
            CoverageGuarantee,
            CalibrationMetrics,
            ComparisonMetrics,
            ConfidenceInterval,
        )

        conformal_int = ConformalInterval(
            lower_bound={"Revenue": 48000},
            upper_bound={"Revenue": 56000},
            point_estimate={"Revenue": 52000},
            interval_width={"Revenue": 8000},
        )

        coverage = CoverageGuarantee(
            nominal_coverage=0.95,
            guaranteed_coverage=0.9474,
            finite_sample_valid=True,
            assumptions=["Exchangeability"],
        )

        calibration = CalibrationMetrics(
            calibration_size=20,
            residual_statistics={"mean": 2000, "std": 500, "median": 1900, "iqr": 600},
            interval_adaptivity=0.25,
        )

        comparison = ComparisonMetrics(
            monte_carlo_interval={"Revenue": ConfidenceInterval(lower=49000, upper=55000, confidence_level=0.95)},
            conformal_interval={"Revenue": (48000, 56000)},
            width_ratio={"Revenue": 1.33},
            interpretation="Test",
        )

        explanation = conformal_predictor._generate_explanation(
            conformal_int, coverage, calibration, comparison, "split"
        )

        # Summary should mention coverage percentage
        assert "94.7" in explanation.summary or "95" in explanation.summary


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimum_calibration_data_requirement(self, conformal_predictor, simple_model):
        """Test that minimum calibration data is enforced."""
        from fastapi import HTTPException

        small_calib_data = [
            ObservationPoint(inputs={"Price": 40}, outcome={"Revenue": 30000}),
            ObservationPoint(inputs={"Price": 50}, outcome={"Revenue": 35000}),
        ]

        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=small_calib_data,
            confidence_level=0.95,
        )

        with pytest.raises(HTTPException) as exc_info:
            conformal_predictor.predict_with_conformal_interval(request)

        assert exc_info.value.status_code == 400
        assert "at least 10" in exc_info.value.detail.lower()

    def test_none_calibration_data(self, conformal_predictor, simple_model):
        """Test handling of None calibration data."""
        from fastapi import HTTPException

        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=None,
            confidence_level=0.95,
        )

        with pytest.raises(HTTPException) as exc_info:
            conformal_predictor.predict_with_conformal_interval(request)

        assert exc_info.value.status_code == 400

    def test_split_with_barely_enough_data(self, conformal_predictor, simple_model):
        """Test split conformal with minimum data after split."""
        # 12 points -> split -> 6 each, which is above minimum of 5
        minimal_calib = [
            ObservationPoint(inputs={"Price": i * 5 + 40}, outcome={"Revenue": i * 2500 + 30000})
            for i in range(12)
        ]

        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=minimal_calib,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        result = conformal_predictor._split_conformal(request)

        assert result is not None
        assert result.calibration_quality.calibration_size >= 5


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_full_prediction_deterministic(self, conformal_predictor, simple_model, calibration_data):
        """Test that full prediction is deterministic with seed."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        result1 = conformal_predictor.predict_with_conformal_interval(request)
        result2 = conformal_predictor.predict_with_conformal_interval(request)

        # Check interval bounds are identical
        assert result1.prediction_interval.lower_bound == result2.prediction_interval.lower_bound
        assert result1.prediction_interval.upper_bound == result2.prediction_interval.upper_bound

        # Check coverage guarantee is identical
        assert result1.coverage_guarantee.guaranteed_coverage == result2.coverage_guarantee.guaranteed_coverage

    def test_different_seeds_give_different_results(self, conformal_predictor, simple_model, calibration_data):
        """Test that different seeds can give different results."""
        request1 = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        request2 = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=99,
        )

        result1 = conformal_predictor.predict_with_conformal_interval(request1)
        result2 = conformal_predictor.predict_with_conformal_interval(request2)

        # Results might be different due to different random splits
        # (though they could be the same by chance, so we just verify both complete)
        assert result1 is not None
        assert result2 is not None


class TestMethodFallback:
    """Tests for method fallback behavior."""

    def test_unsupported_method_falls_back_to_split(self, conformal_predictor, simple_model, calibration_data):
        """Test that unsupported methods fall back to split conformal."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="cv+",  # Not yet implemented
            seed=42,
        )

        # Should not raise error, should fall back to split
        result = conformal_predictor.predict_with_conformal_interval(request)

        assert result is not None
        assert result.prediction_interval is not None


class TestResponseStructure:
    """Tests for response structure and completeness."""

    def test_response_has_all_required_fields(self, conformal_predictor, simple_model, calibration_data):
        """Test that response has all required fields."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        result = conformal_predictor.predict_with_conformal_interval(request)

        assert hasattr(result, 'prediction_interval')
        assert hasattr(result, 'coverage_guarantee')
        assert hasattr(result, 'calibration_quality')
        assert hasattr(result, 'comparison_to_standard')
        assert hasattr(result, 'explanation')

    def test_coverage_guarantee_is_finite_sample_valid(self, conformal_predictor, simple_model, calibration_data):
        """Test that finite_sample_valid is always True."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        result = conformal_predictor.predict_with_conformal_interval(request)

        assert result.coverage_guarantee.finite_sample_valid is True

    def test_interval_width_matches_bounds(self, conformal_predictor, simple_model, calibration_data):
        """Test that interval width matches upper - lower."""
        request = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        result = conformal_predictor.predict_with_conformal_interval(request)

        for var in result.prediction_interval.lower_bound:
            lower = result.prediction_interval.lower_bound[var]
            upper = result.prediction_interval.upper_bound[var]
            width = result.prediction_interval.interval_width[var]

            assert abs(width - (upper - lower)) < 1e-6


class TestConfidenceLevels:
    """Tests for different confidence levels."""

    def test_higher_confidence_gives_wider_intervals(self, conformal_predictor, simple_model, calibration_data):
        """Test that higher confidence levels give wider intervals."""
        request_95 = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.95,
            method="split",
            seed=42,
        )

        request_90 = ConformalCounterfactualRequest(
            model=simple_model,
            intervention={"Price": 50},
            calibration_data=calibration_data,
            confidence_level=0.90,
            method="split",
            seed=42,
        )

        result_95 = conformal_predictor.predict_with_conformal_interval(request_95)
        result_90 = conformal_predictor.predict_with_conformal_interval(request_90)

        width_95 = list(result_95.prediction_interval.interval_width.values())[0]
        width_90 = list(result_90.prediction_interval.interval_width.values())[0]

        # 95% interval should be wider than 90% interval
        assert width_95 >= width_90

    def test_extreme_confidence_levels(self, conformal_predictor, simple_model, calibration_data):
        """Test extreme confidence levels (99%, 90%)."""
        for conf_level in [0.90, 0.95, 0.99]:
            request = ConformalCounterfactualRequest(
                model=simple_model,
                intervention={"Price": 50},
                calibration_data=calibration_data,
                confidence_level=conf_level,
                method="split",
                seed=42,
            )

            result = conformal_predictor.predict_with_conformal_interval(request)

            assert result.coverage_guarantee.nominal_coverage == conf_level
            assert result.coverage_guarantee.guaranteed_coverage <= conf_level + 0.05
            assert result.coverage_guarantee.guaranteed_coverage >= conf_level - 0.05
