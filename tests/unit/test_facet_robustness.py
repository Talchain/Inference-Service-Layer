"""
Unit tests for FACET robustness components.

Tests InterventionRegion, RobustnessAnalyzer, and RobustnessVisualizer.
"""

import pytest

from src.models.robustness import (
    FACETRobustnessAnalysis,
    InterventionRegion,
    OutcomeGuarantee,
    RobustnessRequest,
)
from src.models.shared import StructuralModel, Variable, Distribution, DistributionType
from src.services.robustness_analyzer import RobustnessAnalyzer
from src.services.robustness_visualizer import RobustnessVisualizer


class TestInterventionRegion:
    """Test intervention region representation and operations."""

    def test_contains_point_inside(self):
        """Test point containment for point inside region."""
        region = InterventionRegion(
            variable_ranges={
                "price": (50.0, 60.0),
                "quality": (7.0, 9.0),
            }
        )

        assert region.contains({"price": 55.0, "quality": 8.0})
        assert region.contains({"price": 50.0, "quality": 7.0})  # Boundary
        assert region.contains({"price": 60.0, "quality": 9.0})  # Boundary

    def test_contains_point_outside(self):
        """Test point containment for point outside region."""
        region = InterventionRegion(
            variable_ranges={
                "price": (50.0, 60.0),
                "quality": (7.0, 9.0),
            }
        )

        assert not region.contains({"price": 45.0, "quality": 8.0})
        assert not region.contains({"price": 55.0, "quality": 10.0})
        assert not region.contains({"price": 65.0, "quality": 8.0})

    def test_contains_missing_variable(self):
        """Test point containment when variable is missing."""
        region = InterventionRegion(
            variable_ranges={
                "price": (50.0, 60.0),
                "quality": (7.0, 9.0),
            }
        )

        # Missing quality
        assert not region.contains({"price": 55.0})
        # Extra variable is ok, missing variable is not
        assert not region.contains({"price": 55.0, "other": 100.0})

    def test_volume_computation(self):
        """Test region volume calculation."""
        region = InterventionRegion(
            variable_ranges={
                "x": (0.0, 10.0),  # 10% of [0,100]
                "y": (0.0, 20.0),  # 20% of [0,100]
            }
        )

        volume = region.volume()
        # 0.1 * 0.2 = 0.02
        assert volume == pytest.approx(0.02, rel=1e-6)

    def test_volume_single_dimension(self):
        """Test volume for single dimension."""
        region = InterventionRegion(variable_ranges={"x": (0.0, 50.0)})

        volume = region.volume()
        # 50/100 = 0.5
        assert volume == pytest.approx(0.5, rel=1e-6)

    def test_sample_random(self):
        """Test random sampling from region."""
        region = InterventionRegion(
            variable_ranges={
                "price": (50.0, 60.0),
                "quality": (7.0, 9.0),
            }
        )

        samples = region.sample_random(n=100, seed=42)

        assert len(samples) == 100

        # All samples should be within region
        for sample in samples:
            assert region.contains(sample)
            assert 50.0 <= sample["price"] <= 60.0
            assert 7.0 <= sample["quality"] <= 9.0

    def test_sample_random_deterministic(self):
        """Test that sampling with seed is deterministic."""
        region = InterventionRegion(
            variable_ranges={"price": (50.0, 60.0)}
        )

        samples1 = region.sample_random(n=10, seed=42)
        samples2 = region.sample_random(n=10, seed=42)

        assert samples1 == samples2

    def test_center_point(self):
        """Test center point calculation."""
        region = InterventionRegion(
            variable_ranges={
                "price": (50.0, 60.0),
                "quality": (7.0, 9.0),
            }
        )

        center = region.center_point()

        assert center == {"price": 55.0, "quality": 8.0}


class TestOutcomeGuarantee:
    """Test outcome guarantee model."""

    def test_satisfies_target_within_bounds(self):
        """Test target satisfaction when within bounds."""
        guarantee = OutcomeGuarantee(
            outcome_variable="revenue",
            minimum=95000.0,
            maximum=105000.0,
            confidence=0.95,
        )

        assert guarantee.satisfies_target(target_min=90000.0, target_max=110000.0)
        assert guarantee.satisfies_target(target_min=95000.0, target_max=105000.0)
        assert guarantee.satisfies_target(target_min=None, target_max=110000.0)
        assert guarantee.satisfies_target(target_min=90000.0, target_max=None)

    def test_satisfies_target_outside_bounds(self):
        """Test target satisfaction when outside bounds."""
        guarantee = OutcomeGuarantee(
            outcome_variable="revenue",
            minimum=95000.0,
            maximum=105000.0,
            confidence=0.95,
        )

        # Guarantee minimum too high
        assert not guarantee.satisfies_target(target_min=100000.0)
        # Guarantee maximum too low
        assert not guarantee.satisfies_target(target_max=100000.0)

    def test_satisfies_target_no_constraints(self):
        """Test target satisfaction with no constraints."""
        guarantee = OutcomeGuarantee(
            outcome_variable="revenue",
            minimum=95000.0,
            maximum=105000.0,
            confidence=0.95,
        )

        assert guarantee.satisfies_target()


class TestRobustnessAnalyzer:
    """Test robustness analyzer service."""

    def test_generate_candidate_regions(self):
        """Test candidate region generation."""
        analyzer = RobustnessAnalyzer()

        regions = analyzer._generate_candidate_regions(
            intervention_proposal={"price": 55.0},
            perturbation_radius=0.1,
            feasible_ranges=None,
        )

        # Should generate: center, expanded, positive, negative = 4 regions
        assert len(regions) == 4

        # Center region should be tightest
        center_region = regions[0]
        assert "price" in center_region.variable_ranges
        price_min, price_max = center_region.variable_ranges["price"]

        # Center radius is 0.05 (half of 0.1)
        # 55 * (1 - 0.05) = 52.25, 55 * (1 + 0.05) = 57.75
        assert price_min == pytest.approx(52.25, rel=1e-6)
        assert price_max == pytest.approx(57.75, rel=1e-6)

    def test_generate_candidate_regions_with_feasible_ranges(self):
        """Test candidate region generation with feasibility constraints."""
        analyzer = RobustnessAnalyzer()

        regions = analyzer._generate_candidate_regions(
            intervention_proposal={"price": 10.0},
            perturbation_radius=0.5,  # Would go to 5.0-15.0
            feasible_ranges={"price": (8.0, 12.0)},
        )

        # Check all regions respect feasible ranges
        for region in regions:
            price_min, price_max = region.variable_ranges["price"]
            assert price_min >= 8.0
            assert price_max <= 12.0

    def test_create_region_around_point(self):
        """Test region creation around a point."""
        analyzer = RobustnessAnalyzer()

        region = analyzer._create_region_around_point(
            point={"price": 100.0, "quality": 10.0},
            radius=0.1,
            feasible_ranges=None,
        )

        assert "price" in region.variable_ranges
        assert "quality" in region.variable_ranges

        # Check ranges
        price_min, price_max = region.variable_ranges["price"]
        assert price_min == pytest.approx(90.0, rel=1e-6)  # 100 * 0.9
        assert price_max == pytest.approx(110.0, rel=1e-6)  # 100 * 1.1

        quality_min, quality_max = region.variable_ranges["quality"]
        assert quality_min == pytest.approx(9.0, rel=1e-6)
        assert quality_max == pytest.approx(11.0, rel=1e-6)

    def test_compute_robustness_score_no_regions(self):
        """Test robustness score with no robust regions."""
        analyzer = RobustnessAnalyzer()

        score = analyzer._compute_robustness_score(
            robust_regions=[],
            intervention_space={"price": 55.0},
            perturbation_radius=0.1,
        )

        assert score == 0.0

    def test_compute_robustness_score_single_region(self):
        """Test robustness score with single region."""
        analyzer = RobustnessAnalyzer()

        region = InterventionRegion(
            variable_ranges={"price": (50.0, 60.0)}
        )

        score = analyzer._compute_robustness_score(
            robust_regions=[region],
            intervention_space={"price": 55.0},
            perturbation_radius=0.1,
        )

        # Should be non-zero
        assert 0.0 < score <= 1.0

    def test_detect_fragility_no_regions(self):
        """Test fragility detection with no regions."""
        analyzer = RobustnessAnalyzer()

        is_fragile, reasons = analyzer._detect_fragility(
            robust_regions=[],
            robustness_score=0.0,
            total_samples=100,
        )

        assert is_fragile
        assert len(reasons) > 0
        assert "No robust intervention regions found" in reasons[0]

    def test_detect_fragility_low_score(self):
        """Test fragility detection with low score."""
        analyzer = RobustnessAnalyzer()

        # Create a tiny region
        region = InterventionRegion(
            variable_ranges={"price": (54.9, 55.1)}
        )

        is_fragile, reasons = analyzer._detect_fragility(
            robust_regions=[region],
            robustness_score=0.05,  # Very low
            total_samples=100,
        )

        assert is_fragile
        assert len(reasons) > 0

    def test_detect_fragility_robust(self):
        """Test fragility detection for robust case."""
        analyzer = RobustnessAnalyzer()

        region = InterventionRegion(
            variable_ranges={"price": (50.0, 60.0)}
        )

        is_fragile, reasons = analyzer._detect_fragility(
            robust_regions=[region],
            robustness_score=0.8,
            total_samples=100,
        )

        assert not is_fragile
        assert len(reasons) == 0

    def test_generate_interpretation_no_regions(self):
        """Test interpretation generation for no regions."""
        analyzer = RobustnessAnalyzer()

        interpretation = analyzer._generate_interpretation(
            robust_regions=[],
            robustness_score=0.0,
            is_fragile=True,
        )

        assert "No robust intervention strategy found" in interpretation

    def test_generate_interpretation_robust(self):
        """Test interpretation generation for robust case."""
        analyzer = RobustnessAnalyzer()

        region = InterventionRegion(
            variable_ranges={"price": (50.0, 60.0)}
        )

        interpretation = analyzer._generate_interpretation(
            robust_regions=[region],
            robustness_score=0.85,
            is_fragile=False,
        )

        assert "ROBUST RECOMMENDATION" in interpretation
        assert "0.85" in interpretation

    def test_generate_interpretation_fragile(self):
        """Test interpretation generation for fragile case."""
        analyzer = RobustnessAnalyzer()

        region = InterventionRegion(
            variable_ranges={"price": (54.8, 55.2)}
        )

        interpretation = analyzer._generate_interpretation(
            robust_regions=[region],
            robustness_score=0.15,
            is_fragile=True,
        )

        assert "FRAGILE RECOMMENDATION" in interpretation
        assert "0.15" in interpretation

    def test_generate_recommendation_no_regions(self):
        """Test recommendation generation for no regions."""
        analyzer = RobustnessAnalyzer()

        recommendation = analyzer._generate_recommendation(
            robust_regions=[],
            is_fragile=True,
            original_proposal={"price": 55.0},
        )

        assert "Revise strategy" in recommendation

    def test_generate_recommendation_robust(self):
        """Test recommendation generation for robust case."""
        analyzer = RobustnessAnalyzer()

        region = InterventionRegion(
            variable_ranges={"price": (52.0, 58.0)}
        )

        recommendation = analyzer._generate_recommendation(
            robust_regions=[region],
            is_fragile=False,
            original_proposal={"price": 55.0},
        )

        assert "Proceed with confidence" in recommendation
        assert "52.0-58.0" in recommendation

    def test_build_failed_analysis(self):
        """Test failed analysis result creation."""
        analyzer = RobustnessAnalyzer()

        analysis = analyzer._build_failed_analysis(
            reason="Test failure reason",
            request_id="test_001",
        )

        assert analysis.status == "failed"
        assert analysis.robustness_score == 0.0
        assert analysis.is_fragile
        assert len(analysis.fragility_reasons) > 0
        assert "Test failure reason" in analysis.fragility_reasons[0]


class TestRobustnessVisualizer:
    """Test robustness visualization."""

    def test_ascii_plot_no_regions(self):
        """Test ASCII plot with no regions."""
        visualizer = RobustnessVisualizer()

        analysis = FACETRobustnessAnalysis(
            status="fragile",
            robust_regions=[],
            outcome_guarantees={},
            robustness_score=0.0,
            region_count=0,
            total_volume=0.0,
            is_fragile=True,
            fragility_reasons=["No regions found"],
            samples_tested=100,
            samples_successful=0,
            interpretation="Failed",
            recommendation="Revise",
        )

        plot = visualizer.generate_ascii_plot(analysis)

        assert "No robust regions found" in plot

    def test_ascii_plot_1d(self):
        """Test ASCII plot for 1D case."""
        visualizer = RobustnessVisualizer()

        region = InterventionRegion(
            variable_ranges={"price": (52.0, 58.0)}
        )

        analysis = FACETRobustnessAnalysis(
            status="robust",
            robust_regions=[region],
            outcome_guarantees={},
            robustness_score=0.75,
            region_count=1,
            total_volume=0.06,
            is_fragile=False,
            fragility_reasons=[],
            samples_tested=100,
            samples_successful=100,
            interpretation="Robust",
            recommendation="Proceed",
        )

        plot = visualizer.generate_ascii_plot(analysis)

        assert "Robust Intervals" in plot
        assert "price" in plot
        assert "52.00" in plot
        assert "58.00" in plot
        assert "Robustness Score" in plot

    def test_ascii_plot_2d(self):
        """Test ASCII plot for 2D case."""
        visualizer = RobustnessVisualizer()

        region = InterventionRegion(
            variable_ranges={
                "price": (52.0, 58.0),
                "quality": (7.5, 8.5),
            }
        )

        analysis = FACETRobustnessAnalysis(
            status="robust",
            robust_regions=[region],
            outcome_guarantees={},
            robustness_score=0.75,
            region_count=1,
            total_volume=0.06,
            is_fragile=False,
            fragility_reasons=[],
            samples_tested=100,
            samples_successful=100,
            interpretation="Robust",
            recommendation="Proceed",
        )

        plot = visualizer.generate_ascii_plot(analysis)

        assert "Robust Regions" in plot
        assert "price" in plot
        assert "quality" in plot
        assert "#" in plot  # Should show filled regions
        assert "Robustness score" in plot

    def test_summary_table_robust(self):
        """Test summary table generation for robust case."""
        visualizer = RobustnessVisualizer()

        region = InterventionRegion(
            variable_ranges={"price": (52.0, 58.0)}
        )

        guarantee = OutcomeGuarantee(
            outcome_variable="revenue",
            minimum=95000.0,
            maximum=105000.0,
            confidence=0.95,
        )

        analysis = FACETRobustnessAnalysis(
            status="robust",
            robust_regions=[region],
            outcome_guarantees={"revenue": guarantee},
            robustness_score=0.75,
            region_count=1,
            total_volume=0.06,
            is_fragile=False,
            fragility_reasons=[],
            samples_tested=100,
            samples_successful=100,
            interpretation="ROBUST RECOMMENDATION (robustness: 0.75). High confidence.",
            recommendation="Recommendation: Proceed with confidence.",
        )

        table = visualizer.generate_summary_table(analysis)

        assert "FACET ROBUSTNESS ANALYSIS SUMMARY" in table
        assert "ROBUST" in table
        assert "0.750" in table
        assert "NO ✓" in table  # Not fragile
        assert "Outcome Guarantees" in table
        assert "revenue" in table
        assert "95000" in table
        assert "Interpretation:" in table
        assert "Recommendation:" in table

    def test_summary_table_fragile(self):
        """Test summary table generation for fragile case."""
        visualizer = RobustnessVisualizer()

        analysis = FACETRobustnessAnalysis(
            status="fragile",
            robust_regions=[],
            outcome_guarantees={},
            robustness_score=0.05,
            region_count=0,
            total_volume=0.0,
            is_fragile=True,
            fragility_reasons=[
                "Very low robustness score (0.050)",
                "No robust intervention regions found",
            ],
            samples_tested=100,
            samples_successful=0,
            interpretation="FRAGILE RECOMMENDATION. Exercise caution.",
            recommendation="Revise strategy.",
        )

        table = visualizer.generate_summary_table(analysis)

        assert "FRAGILE" in table
        assert "YES ⚠️" in table
        assert "Fragility Warnings" in table
        assert "Very low robustness score" in table

    def test_wrap_text(self):
        """Test text wrapping utility."""
        visualizer = RobustnessVisualizer()

        text = "This is a very long text that needs to be wrapped into multiple lines for display purposes."
        lines = visualizer._wrap_text(text, width=30)

        # Should be multiple lines
        assert len(lines) > 1

        # Each line should be <= 30 characters
        for line in lines:
            assert len(line) <= 30

        # Joined text should match original (minus spacing)
        assert " ".join(lines) == text
