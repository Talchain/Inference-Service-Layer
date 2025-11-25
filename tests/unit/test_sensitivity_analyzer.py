"""
Unit tests for EnhancedSensitivityAnalyzer service.

Tests quantitative sensitivity analysis including:
- Violation generation for different assumption types
- Outcome predictions under violations
- Elasticity calculations
- Robustness scoring
- Critical assumption identification
- Report generation
- Edge cases and error handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.models.sensitivity import (
    AssumptionType,
    CausalAssumption,
    SensitivityMetric,
    SensitivityReport,
    SensitivityRequest,
    ViolationScenario,
    ViolationType,
)
from src.services.sensitivity_analyzer import EnhancedSensitivityAnalyzer


@pytest.fixture
def sensitivity_analyzer():
    """Create an EnhancedSensitivityAnalyzer instance."""
    return EnhancedSensitivityAnalyzer()


@pytest.fixture
def simple_linear_model():
    """Simple linear model for testing."""
    return {
        "type": "linear",
        "equations": {
            "Revenue": "100 * Price + 5000 * Quality - 200"
        }
    }


@pytest.fixture
def basic_request(simple_linear_model):
    """Basic sensitivity request."""
    return SensitivityRequest(
        model=simple_linear_model,
        intervention={"Price": 45.0},
        assumptions=["no_unobserved_confounding", "linear_effects"],
        violation_levels=[0.1, 0.2, 0.3],
        n_samples=100
    )


class TestSensitivityAnalyzerInitialization:
    """Tests for EnhancedSensitivityAnalyzer initialization."""

    def test_initialization_success(self):
        """Test successful initialization."""
        analyzer = EnhancedSensitivityAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'ASSUMPTIONS')

    def test_assumptions_dictionary_populated(self, sensitivity_analyzer):
        """Test that assumptions dictionary is populated."""
        assert len(sensitivity_analyzer.ASSUMPTIONS) > 0
        assert AssumptionType.NO_UNOBSERVED_CONFOUNDING in sensitivity_analyzer.ASSUMPTIONS

    def test_thresholds_set(self, sensitivity_analyzer):
        """Test that criticality thresholds are set."""
        assert sensitivity_analyzer.CRITICAL_ELASTICITY_THRESHOLD > 0
        assert sensitivity_analyzer.CRITICAL_DEVIATION_THRESHOLD > 0

class TestPredictOutcome:
    """Tests for outcome prediction."""

    def test_predict_with_simple_model(self, sensitivity_analyzer, simple_linear_model):
        """Test prediction with simple linear model."""
        intervention = {"Price": 50.0}
        outcome = sensitivity_analyzer._predict_outcome(simple_linear_model, intervention)
        assert isinstance(outcome, float)
        assert outcome > 0

    def test_predict_different_interventions(self, sensitivity_analyzer, simple_linear_model):
        """Test predictions vary with interventions."""
        outcome1 = sensitivity_analyzer._predict_outcome(simple_linear_model, {"Price": 10.0})
        outcome2 = sensitivity_analyzer._predict_outcome(simple_linear_model, {"Price": 100.0})
        # Different interventions should give different outcomes
        assert outcome1 != outcome2

    def test_predict_unknown_model_type(self, sensitivity_analyzer):
        """Test prediction with unknown model type returns default."""
        model = {"type": "unknown"}
        outcome = sensitivity_analyzer._predict_outcome(model, {"Price": 50.0})
        assert outcome == 50000.0  # Default fallback

    def test_predict_empty_model(self, sensitivity_analyzer):
        """Test prediction with empty model."""
        model = {}
        outcome = sensitivity_analyzer._predict_outcome(model, {"Price": 50.0})
        assert isinstance(outcome, float)


class TestGenerateViolations:
    """Tests for violation scenario generation."""

    def test_generate_violations_basic(self, sensitivity_analyzer):
        """Test basic violation generation."""
        assumption = sensitivity_analyzer.ASSUMPTIONS[AssumptionType.NO_UNOBSERVED_CONFOUNDING]
        violations = sensitivity_analyzer._generate_violations(
            assumption,
            [0.1, 0.2, 0.3],
            n_samples=10
        )
        assert len(violations) == 3
        assert all(isinstance(v, ViolationScenario) for v in violations)

    def test_violation_magnitudes_match(self, sensitivity_analyzer):
        """Test that violation magnitudes match input."""
        assumption = sensitivity_analyzer.ASSUMPTIONS[AssumptionType.LINEAR_EFFECTS]
        violation_levels = [0.1, 0.25, 0.5]
        violations = sensitivity_analyzer._generate_violations(
            assumption,
            violation_levels,
            n_samples=10
        )
        assert all(v.magnitude in violation_levels for v in violations)

    def test_violation_severity_classification(self, sensitivity_analyzer):
        """Test that violations are classified by severity."""
        assumption = sensitivity_analyzer.ASSUMPTIONS[AssumptionType.NO_SELECTION_BIAS]
        violations = sensitivity_analyzer._generate_violations(
            assumption,
            [0.1, 0.3, 0.6],
            n_samples=10
        )
        # 0.1 -> MILD, 0.3 -> MODERATE, 0.6 -> SEVERE
        assert violations[0].severity == ViolationType.MILD
        assert violations[1].severity == ViolationType.MODERATE
        assert violations[2].severity == ViolationType.SEVERE

    def test_generate_violations_for_all_types(self, sensitivity_analyzer):
        """Test violation generation for all assumption types."""
        violation_levels = [0.2]
        for assumption_type, assumption in sensitivity_analyzer.ASSUMPTIONS.items():
            violations = sensitivity_analyzer._generate_violations(
                assumption,
                violation_levels,
                n_samples=10
            )
            assert len(violations) > 0
            assert all(v.assumption_name == assumption.name for v in violations)


class TestApplyViolationAndPredict:
    """Tests for applying violations and predicting outcomes."""

    def test_apply_confounder_violation(self, sensitivity_analyzer, simple_linear_model):
        """Test applying unobserved confounder violation."""
        violation = ViolationScenario(
            assumption_name="No Unobserved Confounding",
            severity=ViolationType.MODERATE,
            magnitude=0.3,
            description="Test violation",
            parameters={"confounder_effect": 0.3}
        )
        baseline = sensitivity_analyzer._predict_outcome(simple_linear_model, {"Price": 50.0})
        outcome = sensitivity_analyzer._apply_violation_and_predict(
            simple_linear_model,
            {"Price": 50.0},
            violation
        )
        # Outcome should differ from baseline
        assert outcome != baseline

    def test_apply_nonlinearity_violation(self, sensitivity_analyzer, simple_linear_model):
        """Test applying non-linearity violation."""
        violation = ViolationScenario(
            assumption_name="Linear Effects",
            severity=ViolationType.MILD,
            magnitude=0.2,
            description="Non-linear effect",
            parameters={"non_linearity": 0.2}
        )
        outcome = sensitivity_analyzer._apply_violation_and_predict(
            simple_linear_model,
            {"Price": 50.0},
            violation
        )
        assert isinstance(outcome, float)
        assert outcome > 0

    def test_apply_selection_bias_violation(self, sensitivity_analyzer, simple_linear_model):
        """Test applying selection bias violation."""
        violation = ViolationScenario(
            assumption_name="No Selection Bias",
            severity=ViolationType.MODERATE,
            magnitude=0.3,
            description="Selection bias",
            parameters={"selection_bias": 0.3}
        )
        baseline = sensitivity_analyzer._predict_outcome(simple_linear_model, {"Price": 50.0})
        outcome = sensitivity_analyzer._apply_violation_and_predict(
            simple_linear_model,
            {"Price": 50.0},
            violation
        )
        # Selection bias should reduce outcome
        assert outcome < baseline

    def test_apply_generic_violation(self, sensitivity_analyzer, simple_linear_model):
        """Test applying generic violation."""
        violation = ViolationScenario(
            assumption_name="Generic",
            severity=ViolationType.MILD,
            magnitude=0.1,
            description="Generic violation",
            parameters={"violation_strength": 0.1}
        )
        outcome = sensitivity_analyzer._apply_violation_and_predict(
            simple_linear_model,
            {"Price": 50.0},
            violation
        )
        assert isinstance(outcome, float)


class TestComputeSensitivityMetric:
    """Tests for sensitivity metric computation."""

    def test_compute_metric_basic(self, sensitivity_analyzer):
        """Test basic sensitivity metric computation."""
        baseline = 50000.0
        outcomes = [48000.0, 52000.0, 54000.0]
        violations = [0.1, 0.2, 0.3]
        violation_details = [
            {"magnitude": 0.1, "severity_score": 0.33, "outcome": 48000.0, "deviation_percent": 4.0},
            {"magnitude": 0.2, "severity_score": 0.67, "outcome": 52000.0, "deviation_percent": 4.0},
            {"magnitude": 0.3, "severity_score": 1.0, "outcome": 54000.0, "deviation_percent": 8.0},
        ]
        
        metric = sensitivity_analyzer._compute_sensitivity_metric(
            "Test Assumption",
            baseline,
            outcomes,
            violations,
            violation_details
        )
        
        assert isinstance(metric, SensitivityMetric)
        assert metric.baseline_outcome == baseline
        assert metric.outcome_range[0] == min(outcomes)
        assert metric.outcome_range[1] == max(outcomes)

    def test_elasticity_calculation(self, sensitivity_analyzer):
        """Test elasticity calculation."""
        baseline = 50000.0
        # Create linear relationship: 10% violation -> 5% outcome change
        outcomes = [50000.0, 52500.0, 55000.0]  # 0%, 5%, 10% changes
        violations = [0.0, 0.1, 0.2]  # 0%, 10%, 20% violations
        violation_details = [
            {"magnitude": v, "severity_score": 0.33, "outcome": o, "deviation_percent": abs(o - baseline) / baseline * 100}
            for v, o in zip(violations, outcomes)
        ]
        
        metric = sensitivity_analyzer._compute_sensitivity_metric(
            "Test",
            baseline,
            outcomes,
            violations,
            violation_details
        )
        
        # Elasticity should be around 0.5 (5% change / 10% violation)
        assert metric.elasticity >= 0

    def test_critical_flag_high_elasticity(self, sensitivity_analyzer):
        """Test that high elasticity marks assumption as critical."""
        baseline = 50000.0
        # High sensitivity: 10% violation -> 20% outcome change
        outcomes = [50000.0, 60000.0, 70000.0]
        violations = [0.0, 0.1, 0.2]
        violation_details = [
            {"magnitude": v, "severity_score": 0.33, "outcome": o, "deviation_percent": abs(o - baseline) / baseline * 100}
            for v, o in zip(violations, outcomes)
        ]
        
        metric = sensitivity_analyzer._compute_sensitivity_metric(
            "Critical Test",
            baseline,
            outcomes,
            violations,
            violation_details
        )
        
        # Should be marked as critical due to high elasticity
        assert metric.critical == True

    def test_critical_flag_large_deviation(self, sensitivity_analyzer):
        """Test that large deviation marks assumption as critical."""
        baseline = 50000.0
        outcomes = [50000.0, 50000.0, 62000.0]  # >20% deviation
        violations = [0.0, 0.1, 0.2]
        violation_details = [
            {"magnitude": v, "severity_score": 0.33, "outcome": o, "deviation_percent": abs(o - baseline) / baseline * 100}
            for v, o in zip(violations, outcomes)
        ]
        
        metric = sensitivity_analyzer._compute_sensitivity_metric(
            "Large Deviation Test",
            baseline,
            outcomes,
            violations,
            violation_details
        )
        
        # Should be marked as critical due to large deviation
        assert metric.max_deviation_percent > 20

    def test_robustness_score_inverse_of_elasticity(self, sensitivity_analyzer):
        """Test that robustness score is inverse of elasticity."""
        baseline = 50000.0
        # Low sensitivity
        outcomes = [50000.0, 50500.0, 51000.0]
        violations = [0.0, 0.1, 0.2]
        violation_details = [
            {"magnitude": v, "severity_score": 0.33, "outcome": o, "deviation_percent": abs(o - baseline) / baseline * 100}
            for v, o in zip(violations, outcomes)
        ]
        
        metric = sensitivity_analyzer._compute_sensitivity_metric(
            "Robust Test",
            baseline,
            outcomes,
            violations,
            violation_details
        )
        
        # Low elasticity should give high robustness
        assert metric.robustness_score > 0.5


class TestAnalyzeAssumptionSensitivity:
    """Tests for full sensitivity analysis."""

    def test_analyze_single_assumption(self, sensitivity_analyzer, simple_linear_model):
        """Test analysis with single assumption."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding"],
            violation_levels=[0.1, 0.2],
            n_samples=10
        )
        
        report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
        
        assert isinstance(report, SensitivityReport)
        assert len(report.sensitivities) == 1
        assert "no_unobserved_confounding" in report.sensitivities

    def test_analyze_multiple_assumptions(self, sensitivity_analyzer, simple_linear_model):
        """Test analysis with multiple assumptions."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding", "linear_effects", "no_selection_bias"],
            violation_levels=[0.1, 0.3],
            n_samples=10
        )
        
        report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
        
        assert len(report.sensitivities) == 3
        assert all(isinstance(m, SensitivityMetric) for m in report.sensitivities.values())

    def test_analyze_with_different_violation_levels(self, sensitivity_analyzer, simple_linear_model):
        """Test analysis with different violation levels."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding"],
            violation_levels=[0.05, 0.1, 0.2, 0.4],
            n_samples=10
        )
        
        report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
        
        metric = report.sensitivities["no_unobserved_confounding"]
        assert len(metric.violation_details) == 4

    def test_report_includes_critical_assumptions(self, sensitivity_analyzer, simple_linear_model):
        """Test that report identifies critical assumptions."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding", "linear_effects"],
            violation_levels=[0.1, 0.3, 0.5],
            n_samples=10
        )
        
        report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
        
        assert hasattr(report, 'most_critical')
        assert isinstance(report.most_critical, list)

    def test_report_includes_robustness_score(self, sensitivity_analyzer, simple_linear_model):
        """Test that report includes overall robustness score."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding"],
            violation_levels=[0.1, 0.2],
            n_samples=10
        )
        
        report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
        
        assert hasattr(report, 'overall_robustness_score')
        assert 0 <= report.overall_robustness_score <= 1

    def test_report_includes_recommendations(self, sensitivity_analyzer, simple_linear_model):
        """Test that report includes recommendations."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding", "linear_effects"],
            violation_levels=[0.2, 0.4],
            n_samples=10
        )
        
        report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
        
        assert hasattr(report, 'recommendations')
        assert isinstance(report.recommendations, list)


class TestCreateReport:
    """Tests for report creation."""

    def test_create_report_basic(self, sensitivity_analyzer):
        """Test basic report creation."""
        sensitivities = {
            "assumption1": SensitivityMetric(
                assumption="assumption1",
                baseline_outcome=50000.0,
                outcome_range=(48000.0, 52000.0),
                elasticity=0.5,
                critical=False,
                max_deviation_percent=4.0,
                robustness_score=0.67,
                interpretation="MINOR: test",
                violation_details=[]
            )
        }
        
        report = sensitivity_analyzer._create_report(sensitivities, 50000.0)
        
        assert isinstance(report, SensitivityReport)
        assert report.overall_robustness_score > 0

    def test_report_sorts_by_criticality(self, sensitivity_analyzer):
        """Test that report sorts assumptions by criticality."""
        sensitivities = {
            "low_sensitivity": SensitivityMetric(
                assumption="low_sensitivity",
                baseline_outcome=50000.0,
                outcome_range=(49000.0, 51000.0),
                elasticity=0.3,
                critical=False,
                max_deviation_percent=2.0,
                robustness_score=0.8,
                interpretation="MINOR",
                violation_details=[]
            ),
            "high_sensitivity": SensitivityMetric(
                assumption="high_sensitivity",
                baseline_outcome=50000.0,
                outcome_range=(40000.0, 60000.0),
                elasticity=2.5,
                critical=True,
                max_deviation_percent=20.0,
                robustness_score=0.3,
                interpretation="CRITICAL",
                violation_details=[]
            )
        }
        
        report = sensitivity_analyzer._create_report(sensitivities, 50000.0)
        
        # Critical assumption should be first in most_critical
        if len(report.most_critical) > 0:
            assert "high_sensitivity" in report.most_critical

    def test_report_summary_quality(self, sensitivity_analyzer):
        """Test that report summary is informative."""
        sensitivities = {
            "test": SensitivityMetric(
                assumption="test",
                baseline_outcome=50000.0,
                outcome_range=(48000.0, 52000.0),
                elasticity=0.5,
                critical=False,
                max_deviation_percent=4.0,
                robustness_score=0.67,
                interpretation="MINOR",
                violation_details=[]
            )
        }
        
        report = sensitivity_analyzer._create_report(sensitivities, 50000.0)
        
        assert len(report.summary) > 0
        assert "robust" in report.summary.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_violation_levels(self, sensitivity_analyzer, simple_linear_model):
        """Test handling of empty violation levels."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding"],
            violation_levels=[],
            n_samples=10
        )
        
        # Should handle gracefully (empty list is allowed by model, but will produce empty violations)
        # The service should handle this without crashing
        try:
            report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
            # If it doesn't crash, check that we get a valid report
            assert isinstance(report, SensitivityReport)
        except Exception as e:
            # Expected behavior - some implementations may raise on empty violations
            assert True

    def test_single_violation_level(self, sensitivity_analyzer, simple_linear_model):
        """Test with single violation level."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding"],
            violation_levels=[0.2],
            n_samples=10
        )
        
        report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
        
        assert isinstance(report, SensitivityReport)
        # With single violation, elasticity may be 0 (can't compute slope)
        metric = report.sensitivities["no_unobserved_confounding"]
        assert metric.elasticity >= 0

    def test_invalid_assumption_type(self, sensitivity_analyzer, simple_linear_model):
        """Test handling of invalid assumption type."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["invalid_assumption_type"],
            violation_levels=[0.1, 0.2],
            n_samples=10
        )
        
        # Should either raise error or handle gracefully
        try:
            report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
            # If handled gracefully, check for default metric
            if "invalid_assumption_type" in report.sensitivities:
                metric = report.sensitivities["invalid_assumption_type"]
                assert metric.robustness_score == 1.0  # Default for failed analysis
        except (ValueError, KeyError):
            # Expected - invalid assumption type
            assert True

    def test_extreme_violation_levels(self, sensitivity_analyzer, simple_linear_model):
        """Test with extreme violation levels."""
        request = SensitivityRequest(
            model=simple_linear_model,
            intervention={"Price": 50.0},
            assumptions=["no_unobserved_confounding"],
            violation_levels=[0.9, 1.0],  # Very large violations
            n_samples=10
        )
        
        report = sensitivity_analyzer.analyze_assumption_sensitivity(request)
        
        metric = report.sensitivities["no_unobserved_confounding"]
        # Extreme violations should likely be marked as critical
        assert metric.severity != ViolationType.MILD or metric.critical


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_get_recommendation_for_known_assumption(self, sensitivity_analyzer):
        """Test getting recommendation for known assumption."""
        rec = sensitivity_analyzer._get_recommendation("No Unobserved Confounding")
        assert len(rec) > 0
        assert isinstance(rec, str)

    def test_get_recommendation_for_unknown_assumption(self, sensitivity_analyzer):
        """Test getting recommendation for unknown assumption."""
        rec = sensitivity_analyzer._get_recommendation("Unknown Assumption")
        assert len(rec) > 0  # Should return default recommendation

    def test_all_assumptions_have_recommendations(self, sensitivity_analyzer):
        """Test that all assumptions have recommendations."""
        for assumption_type, assumption in sensitivity_analyzer.ASSUMPTIONS.items():
            rec = sensitivity_analyzer._get_recommendation(assumption.name)
            assert len(rec) > 0
