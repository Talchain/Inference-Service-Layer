"""
Tests for within-model SBC validation benchmark (4A).

NOT part of the standard pytest run — this is benchmark infrastructure.
Run explicitly::

    ISL_AUTH_DISABLED=true python -m pytest tests/benchmarks/test_sbc_validation.py -v
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.sbc_validator import (
    CICoverageResult,
    SBCResult,
    _build_frozen_graph,
    _compute_ground_truth,
    _sample_true_world,
    run_sbc_validation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph() -> GraphV2:
    """Build a minimal 3-node graph with moderate uncertainty."""
    return GraphV2(
        nodes=[
            NodeV2(id="price", kind="factor", label="Price",
                   observed_state=ObservedState(value=0.5)),
            NodeV2(id="demand", kind="chance", label="Demand"),
            NodeV2(id="revenue", kind="outcome", label="Revenue"),
        ],
        edges=[
            EdgeV2(
                **{"from": "price", "to": "demand"},
                exists_probability=0.95,
                strength=StrengthDistribution(mean=-0.4, std=0.15),
            ),
            EdgeV2(
                **{"from": "demand", "to": "revenue"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.7, std=0.1),
            ),
            EdgeV2(
                **{"from": "price", "to": "revenue"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
        ],
    )


OPTIONS = [
    InterventionOption(id="low_price", label="Low price", interventions={"price": 0.3}),
    InterventionOption(id="high_price", label="High price", interventions={"price": 0.7}),
]
GOAL = "revenue"
SEED = 42


# ==========================================================================
# Test 1: Calibrated pipeline — coverage within 5pp for all CI levels
# ==========================================================================


class TestCalibratedPipeline:
    def test_coverage_within_tolerance(self):
        """Run SBC with 200 trials and verify coverage deviation ≤ 5pp for all CI levels.

        Uses n_samples=500 per trial for adequate posterior resolution.
        The 5pp tolerance accounts for finite-trial sampling noise.
        """
        graph = _make_graph()
        result = run_sbc_validation(
            graph=graph,
            options=OPTIONS,
            goal_node_id=GOAL,
            n_trials=200,
            n_samples=500,
            seed=SEED,
        )

        assert result.label == "within-model"
        for cr in result.coverage_results:
            assert cr.deviation <= 0.05, (
                f"CI level {cr.nominal_level}: observed={cr.observed_coverage:.3f}, "
                f"deviation={cr.deviation:.3f} exceeds 5pp tolerance"
            )
        assert result.calibrated is True


# ==========================================================================
# Test 2: Miscalibrated detection — artificially miscalibrate
# ==========================================================================


class TestMiscalibratedDetection:
    def test_narrow_uncertainty_detected(self):
        """Graph with artificially narrow uncertainty should produce under-coverage.

        Reduce edge std to near-minimum (0.002) so posterior is too tight
        relative to the true prior used for ground-truth sampling.
        The ground truth is sampled from the *original* (wide) prior,
        but inference uses the narrow graph → coverage should drop below nominal.
        """
        wide_graph = _make_graph()
        narrow_graph = _make_graph()

        # Make the narrow graph's edge uncertainty very small
        for edge in narrow_graph.edges:
            edge.strength = StrengthDistribution(
                mean=edge.strength.mean, std=0.002,
            )

        rng = np.random.Generator(np.random.PCG64(99))

        # Run SBC manually: sample truth from wide, infer with narrow
        n_trials = 100
        n_samples = 300
        ci_90_hits = 0

        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        analyzer = RobustnessAnalyzerV2()

        for trial in range(n_trials):
            trial_seed = int(rng.integers(0, 2**31))
            true_strengths, true_existence = _sample_true_world(wide_graph, rng)
            frozen = _build_frozen_graph(wide_graph, true_strengths, true_existence)
            gt = _compute_ground_truth(frozen, OPTIONS, GOAL, "low_price", trial_seed)

            request = RobustnessRequestV2(
                request_id=f"miscal-{trial}",
                graph=copy.deepcopy(narrow_graph),
                options=OPTIONS,
                goal_node_id=GOAL,
                n_samples=n_samples,
                seed=trial_seed,
                analysis_types=["comparison"],
            )
            response = analyzer.analyze(request)
            for opt_result in response.results:
                if opt_result.option_id == "low_price":
                    samples = np.array(opt_result.outcome_distribution.samples)
                    lo = float(np.percentile(samples, 5))
                    hi = float(np.percentile(samples, 95))
                    if lo <= gt <= hi:
                        ci_90_hits += 1
                    break

        observed_coverage = ci_90_hits / n_trials
        # With artificially narrow uncertainty, 90% CI coverage should drop
        # well below 0.9.  Allow some noise but expect < 0.8.
        assert observed_coverage < 0.80, (
            f"Expected under-coverage with narrow uncertainty, "
            f"but observed {observed_coverage:.2f}"
        )


# ==========================================================================
# Test 3: Determinism — same seed → identical result
# ==========================================================================


class TestDeterminism:
    def test_same_seed_identical_results(self):
        """Same graph, options, seed → identical SBC result."""
        graph = _make_graph()

        r1 = run_sbc_validation(
            graph=copy.deepcopy(graph),
            options=OPTIONS,
            goal_node_id=GOAL,
            n_trials=20,
            n_samples=100,
            seed=77,
        )
        r2 = run_sbc_validation(
            graph=copy.deepcopy(graph),
            options=OPTIONS,
            goal_node_id=GOAL,
            n_trials=20,
            n_samples=100,
            seed=77,
        )

        assert r1.pit_histogram == r2.pit_histogram
        assert r1.pit_chi2_statistic == r2.pit_chi2_statistic
        for cr1, cr2 in zip(r1.coverage_results, r2.coverage_results):
            assert cr1.observed_coverage == cr2.observed_coverage
            assert cr1.deviation == cr2.deviation


# ==========================================================================
# Test 4: Frozen-parameter correctness
# ==========================================================================


class TestFrozenParameterCorrectness:
    def test_frozen_graph_deterministic(self):
        """Frozen graph with std=0 and exists ∈ {0,1} → same outcome every time."""
        graph = _make_graph()
        rng = np.random.Generator(np.random.PCG64(42))

        true_strengths, true_existence = _sample_true_world(graph, rng)
        frozen = _build_frozen_graph(graph, true_strengths, true_existence)

        # Run ground truth computation multiple times with different seeds
        results = []
        for seed in [1, 2, 3, 100, 999]:
            gt = _compute_ground_truth(frozen, OPTIONS, GOAL, "low_price", seed)
            results.append(gt)

        # All should be identical (frozen graph has zero variance)
        for i in range(1, len(results)):
            assert results[i] == pytest.approx(results[0], abs=1e-12), (
                f"Frozen graph produced different results: "
                f"{results[0]} vs {results[i]} (seeds 1 vs {[1,2,3,100,999][i]})"
            )

    def test_frozen_graph_edge_removal(self):
        """Frozen graph correctly removes edges with exists=False."""
        graph = _make_graph()

        # Force all edges to not exist
        all_keys = [f"{e.from_}->{e.to}" for e in graph.edges]
        true_strengths = {k: 0.5 for k in all_keys}
        true_existence = {k: False for k in all_keys}

        frozen = _build_frozen_graph(graph, true_strengths, true_existence)

        for edge in frozen.edges:
            assert edge.exists_probability == 0.0

        # Outcome should be zero (no edges active, no intercepts)
        gt = _compute_ground_truth(frozen, OPTIONS, GOAL, "low_price", 42)
        assert gt == pytest.approx(0.0, abs=1e-10)


# ==========================================================================
# Test 5: PIT uniformity
# ==========================================================================


class TestPITUniformity:
    def test_pit_passes_chi_squared(self):
        """PIT histogram should pass chi-squared uniformity test for calibrated inference."""
        graph = _make_graph()
        result = run_sbc_validation(
            graph=graph,
            options=OPTIONS,
            goal_node_id=GOAL,
            n_trials=200,
            n_samples=500,
            seed=SEED,
        )

        assert result.pit_chi2_p_value > 0.05, (
            f"PIT uniformity test failed: chi2={result.pit_chi2_statistic:.2f}, "
            f"p={result.pit_chi2_p_value:.4f}"
        )
        assert result.pit_uniform is True
        assert len(result.pit_histogram) == result.pit_n_bins
        assert sum(result.pit_histogram) == result.n_trials


# ==========================================================================
# Test 6: Edge cases
# ==========================================================================


class TestEdgeCases:
    def test_single_trial(self):
        """n_trials=1 doesn't crash and produces valid structure."""
        graph = _make_graph()
        result = run_sbc_validation(
            graph=graph,
            options=OPTIONS,
            goal_node_id=GOAL,
            n_trials=1,
            n_samples=100,
            seed=42,
        )
        assert result.n_trials == 1
        assert len(result.coverage_results) == 3  # default CI levels
        assert sum(result.pit_histogram) == 1

    def test_single_option(self):
        """Single option → valid SBC (no competition, just outcome CI check)."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="x", kind="factor", label="X",
                       observed_state=ObservedState(value=0.5)),
                NodeV2(id="y", kind="outcome", label="Y"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "x", "to": "y"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.6, std=0.15),
                ),
            ],
        )
        single_option = [
            InterventionOption(id="only", label="Only", interventions={"x": 0.5}),
        ]
        result = run_sbc_validation(
            graph=graph,
            options=single_option,
            goal_node_id="y",
            n_trials=50,
            n_samples=200,
            seed=42,
        )
        assert result.n_trials == 50
        assert result.label == "within-model"

    def test_result_serialization(self):
        """SBCResult.to_dict() produces valid JSON-serializable dict."""
        result = SBCResult(
            n_trials=10,
            ci_levels=[0.5, 0.9],
            coverage_results=[
                CICoverageResult(nominal_level=0.5, observed_coverage=0.5, deviation=0.0, flagged=False),
                CICoverageResult(nominal_level=0.9, observed_coverage=0.85, deviation=0.05, flagged=False),
            ],
            pit_histogram=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            pit_n_bins=10,
            pit_chi2_statistic=0.5,
            pit_chi2_p_value=0.99,
            pit_uniform=True,
            calibrated=True,
            elapsed_s=1.23,
            ms_per_trial=123.0,
            seed=42,
        )
        d = result.to_dict()
        assert d["n_trials"] == 10
        assert d["calibrated"] is True
        assert d["label"] == "within-model"
        assert len(d["coverage_results"]) == 2

        # Verify JSON-serializable
        import json
        json.dumps(d)  # should not raise


# ==========================================================================
# Latency budget (Task 6)
# ==========================================================================


class TestLatencyBudget:
    def test_200_trials_under_60_seconds(self):
        """200 trials with n_samples=500 should complete within 60 seconds.

        The per-trial budget is ~300ms:
        - Ground truth computation (frozen graph, n_samples=1): ~5ms
        - Inference (3-node graph, n_samples=500): ~50ms
        - Overhead: ~5ms
        Total: ~60ms/trial × 200 = ~12s (well under 60s budget).
        """
        graph = _make_graph()
        t0 = time.time()

        result = run_sbc_validation(
            graph=graph,
            options=OPTIONS,
            goal_node_id=GOAL,
            n_trials=200,
            n_samples=500,
            seed=SEED,
        )

        elapsed = time.time() - t0
        assert elapsed < 60.0, (
            f"200-trial SBC took {elapsed:.1f}s, exceeding 60s budget "
            f"({result.ms_per_trial:.0f} ms/trial)"
        )
        assert result.ms_per_trial < 300, (
            f"Per-trial latency {result.ms_per_trial:.0f}ms exceeds 300ms budget"
        )
