"""
Unit tests for CoherenceAnalyzer service.

Tests coherence analysis including:
- Negative expected value detection
- Close race detection
- Ranking stability under perturbations
"""

import pytest

from src.models.requests import CoherenceAnalysisRequest, RankedOption
from src.models.responses import RankingStability
from src.models.shared import GraphNodeV1, GraphV1, NodeKind
from src.services.coherence_analyzer import CoherenceAnalyzer


@pytest.fixture
def analyzer():
    """Create a CoherenceAnalyzer instance with fixed seed."""
    return CoherenceAnalyzer(seed=42)


@pytest.fixture
def simple_graph():
    """Create a simple decision graph."""
    return GraphV1(
        nodes=[
            GraphNodeV1(id="goal", kind=NodeKind.GOAL, label="Maximize Revenue"),
            GraphNodeV1(id="option_a", kind=NodeKind.OPTION, label="Option A"),
            GraphNodeV1(id="option_b", kind=NodeKind.OPTION, label="Option B"),
        ],
        edges=[],
    )


class TestBasicCoherence:
    """Tests for basic coherence metrics."""

    def test_top_option_positive(self, analyzer, simple_graph):
        """Test detection of positive expected value at top."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=30000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        assert result.coherence_analysis.top_option_positive is True
        assert not any("negative expected value" in w.lower() for w in result.coherence_analysis.warnings)

    def test_top_option_negative(self, analyzer, simple_graph):
        """Test detection of negative expected value at top."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=-10000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=-20000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        assert result.coherence_analysis.top_option_positive is False
        assert any("negative expected value" in w.lower() for w in result.coherence_analysis.warnings)

    def test_margin_to_second_calculation(self, analyzer, simple_graph):
        """Test margin to second option calculation."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=100000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=80000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        assert result.coherence_analysis.margin_to_second == 20000
        assert result.coherence_analysis.margin_to_second_pct == pytest.approx(20.0, rel=0.01)


class TestCloseRaceDetection:
    """Tests for close race detection."""

    def test_close_race_detected(self, analyzer, simple_graph):
        """Test detection of close race between options."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=48000, rank=2),  # 4% difference
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            close_race_threshold=0.05,  # 5% threshold
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        # Should detect close race (4% < 5% threshold)
        assert any("close race" in w.lower() for w in result.coherence_analysis.warnings)

    def test_no_close_race_when_clear_leader(self, analyzer, simple_graph):
        """Test no close race warning when there's a clear leader."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=100000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=50000, rank=2),  # 50% difference
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            close_race_threshold=0.05,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        # Should not detect close race
        assert not any("close race" in w.lower() for w in result.coherence_analysis.warnings)


class TestStabilityAnalysis:
    """Tests for ranking stability analysis."""

    def test_stable_rankings(self, analyzer, simple_graph):
        """Test detection of stable rankings."""
        # Large margin means stable rankings
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=100000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=10000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            perturbation_magnitude=0.1,  # 10% perturbation
            num_perturbations=100,
        )

        result = analyzer.analyze(request)

        # With 10x difference, rankings should be stable
        assert result.coherence_analysis.ranking_stability == RankingStability.STABLE
        assert result.coherence_analysis.stability_score > 0.8

    def test_unstable_rankings(self, analyzer, simple_graph):
        """Test detection of unstable rankings."""
        # Very close options with high perturbation
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=49500, rank=2),  # 1% difference
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            perturbation_magnitude=0.2,  # 20% perturbation
            num_perturbations=100,
        )

        result = analyzer.analyze(request)

        # With 1% difference and 20% perturbation, should be unstable
        assert result.coherence_analysis.ranking_stability in [
            RankingStability.SENSITIVE,
            RankingStability.UNSTABLE,
        ]
        assert result.stability_analysis.ranking_change_rate > 0.1

    def test_stability_score_bounds(self, analyzer, simple_graph):
        """Test that stability score is always between 0 and 1."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=45000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            perturbation_magnitude=0.15,
            num_perturbations=50,
        )

        result = analyzer.analyze(request)

        assert 0 <= result.coherence_analysis.stability_score <= 1

    def test_ranking_change_rate(self, analyzer, simple_graph):
        """Test ranking change rate calculation."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=48000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            perturbation_magnitude=0.1,
            num_perturbations=100,
        )

        result = analyzer.analyze(request)

        # Verify ranking change rate properties
        assert result.stability_analysis.num_perturbations == 100
        assert 0 <= result.stability_analysis.ranking_change_rate <= 1
        assert result.stability_analysis.ranking_changes == int(
            result.stability_analysis.ranking_change_rate * 100
        )


class TestPerturbationResults:
    """Tests for perturbation result reporting."""

    def test_sample_perturbations_included(self, analyzer, simple_graph):
        """Test that sample perturbations are included in results."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=48000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=100,
        )

        result = analyzer.analyze(request)

        # Should have sample perturbations (up to 10)
        assert len(result.stability_analysis.sample_perturbations) > 0
        assert len(result.stability_analysis.sample_perturbations) <= 10

    def test_perturbation_result_format(self, analyzer, simple_graph):
        """Test format of individual perturbation results."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=48000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=100,
        )

        result = analyzer.analyze(request)

        for perturb in result.stability_analysis.sample_perturbations:
            assert perturb.perturbation_id >= 1
            assert perturb.top_option_id in ["opt_a", "opt_b"]
            assert isinstance(perturb.ranking_changed, bool)
            assert isinstance(perturb.value_change_pct, float)


class TestMostFrequentAlternative:
    """Tests for most frequent alternative tracking."""

    def test_most_frequent_alternative_identified(self, analyzer, simple_graph):
        """Test identification of most frequent alternative."""
        # Options where opt_b might occasionally win
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=48000, rank=2),
            RankedOption(option_id="opt_c", name="Option C", expected_value=30000, rank=3),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            perturbation_magnitude=0.1,
            num_perturbations=100,
        )

        result = analyzer.analyze(request)

        # If rankings changed, most frequent alternative should be identified
        if result.stability_analysis.ranking_changes > 0:
            assert result.stability_analysis.most_frequent_alternative is not None
            assert result.stability_analysis.most_frequent_alternative in ["opt_b", "opt_c"]

    def test_no_alternative_when_stable(self, analyzer, simple_graph):
        """Test no alternative when rankings never change."""
        # Very large margin means no ranking changes
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=1000000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=100, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            perturbation_magnitude=0.01,  # Small perturbation
            num_perturbations=100,
        )

        result = analyzer.analyze(request)

        # With such large margin, there should be no ranking changes
        if result.stability_analysis.ranking_changes == 0:
            assert result.stability_analysis.most_frequent_alternative is None


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_negative_value_recommendation(self, analyzer, simple_graph):
        """Test recommendation for negative expected value."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=-5000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=-10000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        assert any("positive expected value" in r.lower() or "re-evaluate" in r.lower()
                   for r in result.recommendations)

    def test_close_race_recommendation(self, analyzer, simple_graph):
        """Test recommendation for close race."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=49000, rank=2),  # 2% difference
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            close_race_threshold=0.05,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        assert any("close" in r.lower() or "risk tolerance" in r.lower()
                   for r in result.recommendations)

    def test_instability_recommendation(self, analyzer, simple_graph):
        """Test recommendation for ranking instability."""
        # Very close options
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=49800, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            perturbation_magnitude=0.2,  # High perturbation
            num_perturbations=100,
        )

        result = analyzer.analyze(request)

        # If rankings are unstable, should recommend gathering more data
        if result.coherence_analysis.ranking_stability != RankingStability.STABLE:
            assert any("data" in r.lower() or "uncertainty" in r.lower() or "verify" in r.lower()
                       for r in result.recommendations)

    def test_positive_recommendation_when_stable(self, analyzer, simple_graph):
        """Test positive recommendation when analysis is clean."""
        # Clear winner with large margin
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=100000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=10000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            perturbation_magnitude=0.1,
            num_perturbations=50,
        )

        result = analyzer.analyze(request)

        # Should have positive recommendation about decision being supported
        assert any("stable" in r.lower() or "well-supported" in r.lower() or "clear" in r.lower()
                   for r in result.recommendations)


class TestConfidenceIntervals:
    """Tests for confidence interval handling."""

    def test_overlapping_confidence_intervals_warning(self, analyzer, simple_graph):
        """Test warning when confidence intervals overlap."""
        options = [
            RankedOption(
                option_id="opt_a",
                name="Option A",
                expected_value=50000,
                confidence_interval=(40000, 60000),
                rank=1,
            ),
            RankedOption(
                option_id="opt_b",
                name="Option B",
                expected_value=45000,
                confidence_interval=(35000, 55000),  # Overlaps with opt_a
                rank=2,
            ),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        # Should warn about overlapping intervals
        assert any("overlap" in r.lower() or "significantly different" in r.lower()
                   for r in result.recommendations)

    def test_non_overlapping_intervals(self, analyzer, simple_graph):
        """Test no overlap warning when intervals don't overlap."""
        options = [
            RankedOption(
                option_id="opt_a",
                name="Option A",
                expected_value=100000,
                confidence_interval=(90000, 110000),
                rank=1,
            ),
            RankedOption(
                option_id="opt_b",
                name="Option B",
                expected_value=50000,
                confidence_interval=(40000, 60000),  # No overlap
                rank=2,
            ),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        # Should not warn about overlapping intervals
        assert not any("overlap" in r.lower() for r in result.recommendations)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_two_options_minimum(self, analyzer, simple_graph):
        """Test with minimum number of options (2)."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=40000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        assert result.coherence_analysis is not None
        assert result.stability_analysis is not None

    def test_zero_expected_value(self, analyzer, simple_graph):
        """Test with zero expected value."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=0, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=-10000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        # Zero is not positive
        assert result.coherence_analysis.top_option_positive is False

    def test_equal_expected_values(self, analyzer, simple_graph):
        """Test with equal expected values (tie)."""
        options = [
            RankedOption(option_id="opt_a", name="Option A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="Option B", expected_value=50000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=simple_graph,
            options=options,
            num_perturbations=10,
        )

        result = analyzer.analyze(request)

        # Margin should be zero
        assert result.coherence_analysis.margin_to_second == 0

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with same seed."""
        analyzer1 = CoherenceAnalyzer(seed=123)
        analyzer2 = CoherenceAnalyzer(seed=123)

        graph = GraphV1(
            nodes=[
                GraphNodeV1(id="goal", kind=NodeKind.GOAL, label="Goal"),
                GraphNodeV1(id="opt_a", kind=NodeKind.OPTION, label="A"),
                GraphNodeV1(id="opt_b", kind=NodeKind.OPTION, label="B"),
            ],
            edges=[],
        )

        options = [
            RankedOption(option_id="opt_a", name="A", expected_value=50000, rank=1),
            RankedOption(option_id="opt_b", name="B", expected_value=48000, rank=2),
        ]

        request = CoherenceAnalysisRequest(
            graph=graph,
            options=options,
            num_perturbations=100,
        )

        result1 = analyzer1.analyze(request)
        result2 = analyzer2.analyze(request)

        # Results should be identical
        assert result1.stability_analysis.ranking_changes == result2.stability_analysis.ranking_changes
        assert result1.stability_analysis.ranking_change_rate == result2.stability_analysis.ranking_change_rate
