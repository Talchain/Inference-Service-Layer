"""
Unit tests for conditional win probability by factor range.

Tests the _compute_conditional_winners() method and related models.
"""

import numpy as np
import pytest

from src.models.robustness_v2 import (
    BucketResult,
    ConditionalWinner,
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
    RobustnessRequestV2,
    RobustnessResponseV2,
    StrengthDistribution,
)
from src.models.response_v2 import (
    BucketResultV2,
    ConditionalWinnerV2,
)
from src.services.robustness_analyzer_v2 import (
    FactorSampler,
    RobustnessAnalyzerV2,
)
from src.utils.rng import SeededRNG


# =============================================================================
# Helpers
# =============================================================================


def _make_flip_graph():
    """
    Build a graph where the winner flips depending on factor value.

    Structure:
      factor --[strength=1.0]--> outcome
      (no intercept on outcome, so outcome ≈ factor_value * 1.0)

    Option A intervenes on outcome with +50 (pushes outcome up by 50).
    Option B intervenes on outcome with +100 (pushes outcome up by 100).

    With factor observed_state.value = 75 and wide uniform uncertainty [0, 150]:
    - When factor < 50: factor_value * 1.0 + 50 (A) vs factor_value * 1.0 + 100 (B)
      → B always wins (higher intervention)
    - When factor > 50: same logic — B still wins

    To create a flip, we need the factor to interact differently with options.
    Better approach: use two separate paths.

    Revised structure:
      factor --[strength=1.0]--> outcome  (factor value feeds outcome)
      Option A: interventions = {"factor": 0}   → zeroes factor, outcome = 0 + intercept
      Option B: interventions = {}              → uses sampled factor value

    With outcome intercept = 50:
      Option A: outcome = 0 * 1.0 + 50 = 50 (constant)
      Option B: outcome = factor_value * 1.0 + 50

    When factor < 0 (low values): Option B outcome < 50, Option A wins
    When factor > 0 (high values): Option B outcome > 50, Option B wins

    Factor centered at 0 with uniform [-50, 50] should create a clean flip.
    """
    return GraphV2(
        nodes=[
            NodeV2(
                id="factor_a",
                kind="factor",
                label="Key Factor",
                observed_state=ObservedState(value=0.0, unit="units"),
            ),
            NodeV2(id="outcome", kind="outcome", label="Revenue", intercept=50.0),
        ],
        edges=[
            EdgeV2(
                **{"from": "factor_a", "to": "outcome"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=1.0, std=0.002),
            ),
        ],
    )


def _make_flip_request(n_samples=1000, seed=42):
    """Build a request that should produce a winner flip on factor_a."""
    return RobustnessRequestV2(
        request_id="flip-test",
        graph=_make_flip_graph(),
        options=[
            InterventionOption(
                id="opt_a",
                label="Option A",
                interventions={"factor_a": 0},  # zeroes factor
            ),
            InterventionOption(
                id="opt_b",
                label="Option B",
                interventions={},  # uses sampled factor value
            ),
        ],
        goal_node_id="outcome",
        n_samples=n_samples,
        seed=seed,
        analysis_types=["sensitivity"],
        parameter_uncertainties=[
            ParameterUncertainty(
                node_id="factor_a",
                distribution="uniform",
                range_min=-50.0,
                range_max=50.0,
            ),
        ],
    )


# =============================================================================
# Tests
# =============================================================================


class TestConditionalWinnersBasicFlip:
    """Test that a winner flip is detected when factor value changes winner."""

    def test_basic_flip(self):
        """Two options, one factor with wide uniform uncertainty → winner flips."""
        request = _make_flip_request(n_samples=1000, seed=42)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert response.conditional_winners is not None
        assert len(response.conditional_winners) >= 1

        cw = response.conditional_winners[0]
        assert cw.factor_id == "factor_a"
        assert cw.factor_label == "Key Factor"
        assert cw.winner_flips is True
        assert cw.low_bucket.n_samples > 0
        assert cw.high_bucket.n_samples > 0
        assert cw.low_bucket.winner_id != cw.high_bucket.winner_id

        # Low bucket: factor < median → factor_value < 0 → Option A wins (50 > 50 + neg)
        assert cw.low_bucket.winner_id == "opt_a"
        # High bucket: factor >= median → factor_value > 0 → Option B wins (50 + pos > 50)
        assert cw.high_bucket.winner_id == "opt_b"

    def test_split_unit_propagated(self):
        """Verify split_unit comes from observed_state.unit."""
        request = _make_flip_request(n_samples=1000, seed=42)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert response.conditional_winners is not None
        cw = response.conditional_winners[0]
        assert cw.split_unit == "units"

    def test_split_value_near_median(self):
        """Verify split_value is the factor's median sampled value."""
        request = _make_flip_request(n_samples=1000, seed=42)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert response.conditional_winners is not None
        cw = response.conditional_winners[0]
        # For uniform[-50, 50], median should be near 0
        assert abs(cw.split_value) < 10.0

    def test_runner_up_populated(self):
        """Verify runner_up fields are populated when both options win some samples."""
        request = _make_flip_request(n_samples=1000, seed=42)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert response.conditional_winners is not None
        cw = response.conditional_winners[0]

        # The high bucket should have a runner-up since the split isn't perfectly
        # at 0 — some low-factor samples end up in the high bucket
        assert cw.high_bucket.runner_up_id is not None
        assert cw.high_bucket.runner_up_probability is not None

        # Low bucket may or may not have runner-up depending on exact split point
        # Just verify the types are correct
        if cw.low_bucket.runner_up_id is not None:
            assert cw.low_bucket.runner_up_probability is not None
            assert 0.0 <= cw.low_bucket.runner_up_probability <= 1.0

    def test_probabilities_valid(self):
        """Winner probabilities should be in valid range."""
        request = _make_flip_request(n_samples=1000, seed=42)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert response.conditional_winners is not None
        cw = response.conditional_winners[0]

        # Winner probability must be in [0, 1]
        assert 0.0 <= cw.low_bucket.winner_probability <= 1.0
        assert 0.0 <= cw.high_bucket.winner_probability <= 1.0

        # When runner-up exists, sum should be ~1.0 (2-option case)
        for bucket in [cw.low_bucket, cw.high_bucket]:
            if bucket.runner_up_probability is not None:
                total = bucket.winner_probability + bucket.runner_up_probability
                assert abs(total - 1.0) < 0.01


class TestConditionalWinnersNoFlip:
    """Test that no conditional_winners are returned when no flip occurs."""

    def test_no_flip_same_winner(self):
        """Same option wins in both buckets → conditional_winners is None."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="factor_a",
                    kind="factor",
                    label="Factor",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="outcome", kind="outcome", label="Revenue", intercept=0.0),
            ],
            edges=[
                EdgeV2(
                    **{"from": "factor_a", "to": "outcome"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.002),
                ),
            ],
        )

        # Option A always adds 1000, Option B adds 0 → A always wins regardless of factor
        request = RobustnessRequestV2(
            request_id="no-flip-test",
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="Option A", interventions={"outcome": 1000}),
                InterventionOption(id="opt_b", label="Option B", interventions={}),
            ],
            goal_node_id="outcome",
            n_samples=200,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(
                    node_id="factor_a",
                    distribution="normal",
                    std=20.0,
                ),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.conditional_winners is None


class TestConditionalWinnersSkipConditions:
    """Test conditions that should cause factors to be skipped."""

    def test_skip_point_mass(self):
        """Factor with distribution='point_mass' should be excluded."""
        graph = _make_flip_graph()
        request = RobustnessRequestV2(
            request_id="point-mass-test",
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="Option A", interventions={"factor_a": 0}),
                InterventionOption(id="opt_b", label="Option B", interventions={}),
            ],
            goal_node_id="outcome",
            n_samples=200,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(
                    node_id="factor_a",
                    distribution="point_mass",
                ),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.conditional_winners is None

    def test_skip_single_option(self):
        """Single option → conditional_winners is None."""
        graph = _make_flip_graph()
        request = RobustnessRequestV2(
            request_id="single-opt-test",
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="Option A", interventions={}),
            ],
            goal_node_id="outcome",
            n_samples=200,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(
                    node_id="factor_a",
                    distribution="uniform",
                    range_min=-50.0,
                    range_max=50.0,
                ),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.conditional_winners is None

    def test_skip_small_buckets(self):
        """Low n_samples → buckets too small → factor skipped."""
        # With n_samples=100 (minimum allowed), each bucket gets ~50.
        # Use min_bucket_size parameter directly on the method.
        request = _make_flip_request(n_samples=100, seed=42)
        analyzer = RobustnessAnalyzerV2()

        # Run MC manually to get factor_values_per_sample
        from src.services.robustness_analyzer_v2 import (
            DualUncertaintySampler,
            FactorSampler,
            SCMEvaluatorV2,
        )

        rng = SeededRNG(42)
        sampler = DualUncertaintySampler(request.graph.edges, rng)
        factor_sampler = FactorSampler(request.graph.nodes, request.parameter_uncertainties, rng)
        evaluator = SCMEvaluatorV2(request.graph)

        (
            option_outcomes,
            option_wins,
            winner_per_sample,
            _,
            _,
            _,
            factor_values_per_sample,
        ) = analyzer._run_monte_carlo(request, sampler, factor_sampler, evaluator)

        # With min_bucket_size=60, each bucket of ~50 is too small
        result = analyzer._compute_conditional_winners(
            factor_values_per_sample,
            winner_per_sample,
            option_outcomes,
            factor_sampler,
            request,
            min_bucket_size=60,
        )
        assert result is None


class TestConditionalWinnersModels:
    """Test V1 and V2 model serialization."""

    def test_excluded_from_response_when_none(self):
        """When conditional_winners is None, key absent from serialized dict."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="x", kind="factor", label="X"),
                NodeV2(id="y", kind="outcome", label="Y"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "x", "to": "y"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )
        request = RobustnessRequestV2(
            request_id="none-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="y",
            n_samples=100,
            seed=42,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        dumped = response.model_dump(exclude_none=True)
        assert "conditional_winners" not in dumped

    def test_v1_model_construction(self):
        """BucketResult and ConditionalWinner can be constructed correctly."""
        low = BucketResult(
            n_samples=100,
            winner_id="opt_a",
            winner_label="Option A",
            winner_probability=0.7,
            runner_up_id="opt_b",
            runner_up_probability=0.3,
        )
        high = BucketResult(
            n_samples=100,
            winner_id="opt_b",
            winner_label="Option B",
            winner_probability=0.65,
            runner_up_id="opt_a",
            runner_up_probability=0.35,
        )
        cw = ConditionalWinner(
            factor_id="price",
            factor_label="Price",
            split_value=50.0,
            split_unit="£k",
            low_bucket=low,
            high_bucket=high,
            winner_flips=True,
        )
        assert cw.winner_flips is True
        assert cw.low_bucket.winner_id == "opt_a"
        assert cw.high_bucket.winner_id == "opt_b"
        assert cw.split_unit == "£k"

    def test_v2_model_construction(self):
        """V2 models can be constructed correctly."""
        low = BucketResultV2(
            n_samples=100,
            winner_id="opt_a",
            winner_label="Option A",
            winner_probability=0.7,
        )
        high = BucketResultV2(
            n_samples=100,
            winner_id="opt_b",
            winner_label="Option B",
            winner_probability=0.65,
        )
        cw = ConditionalWinnerV2(
            factor_id="price",
            factor_label="Price",
            split_value=50.0,
            split_unit=None,
            low_bucket=low,
            high_bucket=high,
            winner_flips=True,
        )
        assert cw.winner_flips is True
        dumped = cw.model_dump(exclude_none=True)
        assert "split_unit" not in dumped

    def test_v1_to_v2_mapping(self):
        """Test conversion from V1 ConditionalWinner to V2 ConditionalWinnerV2."""
        v1 = ConditionalWinner(
            factor_id="churn",
            factor_label="Churn Rate",
            split_value=5.0,
            split_unit="%",
            low_bucket=BucketResult(
                n_samples=500,
                winner_id="opt_a",
                winner_label="Option A",
                winner_probability=0.6,
                runner_up_id="opt_b",
                runner_up_probability=0.4,
            ),
            high_bucket=BucketResult(
                n_samples=500,
                winner_id="opt_b",
                winner_label="Option B",
                winner_probability=0.55,
                runner_up_id="opt_a",
                runner_up_probability=0.45,
            ),
            winner_flips=True,
        )

        # Simulate the API-layer conversion
        v2 = ConditionalWinnerV2(
            factor_id=v1.factor_id,
            factor_label=v1.factor_label,
            split_value=v1.split_value,
            split_unit=v1.split_unit,
            low_bucket=BucketResultV2(
                n_samples=v1.low_bucket.n_samples,
                winner_id=v1.low_bucket.winner_id,
                winner_label=v1.low_bucket.winner_label,
                winner_probability=v1.low_bucket.winner_probability,
                runner_up_id=v1.low_bucket.runner_up_id,
                runner_up_probability=v1.low_bucket.runner_up_probability,
            ),
            high_bucket=BucketResultV2(
                n_samples=v1.high_bucket.n_samples,
                winner_id=v1.high_bucket.winner_id,
                winner_label=v1.high_bucket.winner_label,
                winner_probability=v1.high_bucket.winner_probability,
                runner_up_id=v1.high_bucket.runner_up_id,
                runner_up_probability=v1.high_bucket.runner_up_probability,
            ),
            winner_flips=v1.winner_flips,
        )

        assert v2.factor_id == "churn"
        assert v2.split_unit == "%"
        assert v2.low_bucket.winner_id == "opt_a"
        assert v2.high_bucket.winner_id == "opt_b"
        assert v2.winner_flips is True


class TestConditionalWinnersReproducibility:
    """Test determinism with same seed."""

    def test_reproducibility(self):
        """Same seed produces identical conditional_winners."""
        request1 = _make_flip_request(n_samples=500, seed=123)
        request2 = _make_flip_request(n_samples=500, seed=123)

        analyzer = RobustnessAnalyzerV2()
        response1 = analyzer.analyze(request1)
        response2 = analyzer.analyze(request2)

        assert response1.conditional_winners is not None
        assert response2.conditional_winners is not None
        assert len(response1.conditional_winners) == len(response2.conditional_winners)

        cw1 = response1.conditional_winners[0]
        cw2 = response2.conditional_winners[0]
        assert cw1.split_value == cw2.split_value
        assert cw1.low_bucket.winner_id == cw2.low_bucket.winner_id
        assert cw1.high_bucket.winner_id == cw2.high_bucket.winner_id
        assert cw1.low_bucket.winner_probability == cw2.low_bucket.winner_probability


class TestComputeBucketResult:
    """Test the _compute_bucket_result helper directly."""

    def test_tie_breaking_by_mean(self):
        """When two options have equal wins, higher mean outcome wins."""
        analyzer = RobustnessAnalyzerV2()

        # 10 samples: 5 wins for each option
        winner_per_sample = [
            "opt_a",
            "opt_b",
            "opt_a",
            "opt_b",
            "opt_a",
            "opt_b",
            "opt_a",
            "opt_b",
            "opt_a",
            "opt_b",
        ]
        mask = np.ones(10, dtype=bool)  # all samples in bucket
        option_labels = {"opt_a": "Option A", "opt_b": "Option B"}
        # opt_b has higher mean outcome → should be winner
        option_means = {"opt_a": 100.0, "opt_b": 200.0}

        result = analyzer._compute_bucket_result(
            mask, winner_per_sample, option_labels, option_means
        )

        assert result.winner_id == "opt_b"
        assert result.winner_label == "Option B"
        assert result.winner_probability == 0.5
        assert result.runner_up_id == "opt_a"
        assert result.runner_up_probability == 0.5
