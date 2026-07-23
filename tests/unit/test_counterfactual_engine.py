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
        assert response.uncertainty.overall in ["low", "medium", "high"]
        # A3: the fabricated `robustness` block is omitted from the response.
        assert not hasattr(response, "robustness")

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


class TestTopologicalSortDeterminism:
    """Defect 1 (A3, 2026-07-23, HUNT-VALIDATION F-4): the topo order must be a
    canonical PURE FUNCTION of equation CONTENT so it agrees with the
    content-addressed cache key. Reverting the variable-name tie-break makes these
    RED (insertion order leaks into the order for ties)."""

    def test_tie_order_is_insertion_order_independent(self):
        """Two equally-ready variables (a Kahn tie) must be ordered by NAME, not by
        dict insertion order — otherwise `{...P,Q...}` and `{...Q,P...}` compute
        different orders while colliding on the same content-sorted cache key.

        FRESH engine per call: order1 and order2 have identical CONTENT (hence an
        identical topo cache key), so a single shared engine would serve order1's
        cached result to order2 and mask the sort entirely — the cache masking IS
        the F-4 defect. Separate engines force each order through the sort."""
        # P and Q are both roots (deps on exogenous A/B, not on each other) -> a tie.
        order1 = CounterfactualEngine()._topological_sort_equations(
            {"P": "2 * A", "Q": "3 * B", "Y": "P + Q"})
        order2 = CounterfactualEngine()._topological_sort_equations(
            {"Y": "P + Q", "Q": "3 * B", "P": "2 * A"})
        assert order1 == order2, (order1, order2)
        # Canonical: the two tied roots come out name-sorted (P before Q).
        names = [v for v, _ in order1]
        assert names.index("P") < names.index("Q")

    def test_diamond_tie_order_canonical(self):
        """A diamond A -> {B,C} -> D: B and C tie and must be name-ordered
        regardless of how the dict is built. Fresh engines per call defeat the
        content-addressed topo cache (see test above)."""
        o1 = [v for v, _ in CounterfactualEngine()._topological_sort_equations(
            {"A": "1", "B": "A", "C": "A", "D": "B + C"})]
        o2 = [v for v, _ in CounterfactualEngine()._topological_sort_equations(
            {"D": "B + C", "C": "A", "B": "A", "A": "1"})]
        assert o1 == o2 == ["A", "B", "C", "D"]


class TestConstantEquation:
    """Defect 2 (A3, 2026-07-23, HUNT-VALIDATION F-4): a constant structural
    equation (e.g. "5") is legal client input. It used to evaluate to a 0-d array
    whose `.tolist()` scalar hit `len()` in adaptive Monte Carlo -> TypeError -> 500.
    A constant IS a valid structural equation; support it. Reverting the 0-d
    broadcast makes these RED (500)."""

    def test_constant_outcome_equation_returns_constant(self):
        """RED at HEAD: `equations={"Y": "5"}` raised TypeError (500). Y=5 is a
        constant in every sample -> point_estimate 5.0, CI exactly [5, 5]."""
        engine = CounterfactualEngine()
        model = StructuralModel(
            variables=["Y", "D"], equations={"Y": "5"}, distributions={}
        )
        # do(D) is a disconnected no-op purely to satisfy the non-empty-intervention
        # route guard; it does not feed Y.
        request = CounterfactualRequest(
            model=model, intervention={"D": 1.0}, outcome="Y", context={}
        )
        response = engine.analyze(request)
        assert response.prediction.point_estimate == pytest.approx(5.0, abs=1e-9)
        assert response.prediction.confidence_interval.lower == pytest.approx(5.0, abs=1e-9)
        assert response.prediction.confidence_interval.upper == pytest.approx(5.0, abs=1e-9)

    def test_constant_chain_to_outcome_supported(self):
        """A chain of CONSTANT equations reaching the outcome keeps the outcome 0-d
        (no operand carries the sample dimension): A=3, B=4, Y=A+B -> Y=7. RED at
        HEAD (500). (A constant feeding an outcome that ALSO has a sampled/intervened
        operand already broadcast to 1-d and never crashed — this all-constant chain
        is the case the 0-d handling actually rescues.)"""
        engine = CounterfactualEngine()
        model = StructuralModel(
            variables=["A", "B", "Y", "D"], equations={"A": "3", "B": "4", "Y": "A + B"},
            distributions={},
        )
        request = CounterfactualRequest(
            model=model, intervention={"D": 1.0}, outcome="Y", context={}
        )
        response = engine.analyze(request)
        assert response.prediction.point_estimate == pytest.approx(7.0, abs=1e-9)
        assert response.prediction.confidence_interval.lower == pytest.approx(7.0, abs=1e-9)
        assert response.prediction.confidence_interval.upper == pytest.approx(7.0, abs=1e-9)


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

    def test_analyze_uncertainty_overall_only(self):
        """A3: `_analyze_uncertainty` returns only the honest overall level.

        The per-factor `sources` breakdown is OMITTED (it labeled each input's own
        variance as its outcome "impact", a leverage-blind fabrication)."""
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
        # The fabricated `sources` field no longer exists on the model.
        assert not hasattr(uncertainty, "sources")

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

    def test_undefined_cv_zero_mean_spread_is_not_low(self):
        """F3b: a zero-mean outcome WITH non-zero spread has an UNDEFINED
        coefficient of variation and must NOT be reported as LOW uncertainty
        (the old code substituted cv=0 -> LOW, claiming confidence it lacked).
        Fail safe to HIGH. Exercised directly because a zero mean with spread is
        not reachable through random MC sampling (verdict F3b nuance)."""
        engine = CounterfactualEngine()
        model = StructuralModel(
            variables=["Y"],
            equations={},
            distributions={
                "Y": Distribution(
                    type=DistributionType.NORMAL, parameters={"mean": 0.0, "std": 1.0}
                )
            },
        )
        request = CounterfactualRequest(
            model=model, intervention={}, outcome="Y", context={}
        )
        # mean exactly 0, std > 0 -> CV undefined
        samples = {"Y": np.array([-2.0, -1.0, 1.0, 2.0])}
        assert engine._analyze_uncertainty(request, samples).overall == "high"

    def test_degenerate_zero_outcome_is_low(self):
        """F3b control: a genuinely degenerate outcome (mean 0, std 0 — e.g. a
        fully-intervened deterministic model) IS zero-uncertainty -> LOW. Proves
        the mean-0 guard does not over-reject a legitimate zero."""
        engine = CounterfactualEngine()
        model = StructuralModel(
            variables=["Y"],
            equations={},
            distributions={
                "Y": Distribution(
                    type=DistributionType.NORMAL, parameters={"mean": 0.0, "std": 1.0}
                )
            },
        )
        request = CounterfactualRequest(
            model=model, intervention={}, outcome="Y", context={}
        )
        samples = {"Y": np.zeros(4)}
        assert engine._analyze_uncertainty(request, samples).overall == "low"


# A3 (2026-07-22): TestRobustnessAnalysis, TestDistributionConfidence and
# TestFactorNameFormatting removed. They exercised `_analyze_robustness`,
# `_assess_distribution_confidence` and `_format_factor_name` — all deleted with
# the fabricated robustness/uncertainty-sources block (F3a/F3c). The honest
# outputs (point estimate, CI, overall uncertainty) are covered above.


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


class TestBoundedTopoSortCache:
    """C2(cf): the topological-sort cache must be BOUNDED, not grow unbounded."""

    def test_distinct_models_do_not_grow_the_cache_unbounded(self):
        """RED at HEAD: the per-request topological-sort cache leaks unbounded.

        ``self._topo_sort_cache`` (keyed by ``json.dumps(equations, sort_keys=True)``)
        grew one entry per distinct structural model, was never evicted, and lived for
        the process lifetime on the live request path (the module-level
        ``counterfactual_engine`` singleton in ``causal.py``). With the C3 mount that is
        a monotonic memory leak keyed by client-controllable equation content — the same
        class phase4 F-4 addressed. The cache is a real read-through optimisation
        (tests/performance/test_optimization_gains.py exercises it), so it is BOUNDED
        rather than removed: at capacity the oldest entry is evicted.

        Analysing MAX + 32 DISTINCT-equation models grew the cache to MAX + 32 at HEAD
        (unbounded); with the bound it never exceeds ``_TOPO_SORT_CACHE_MAX``.
        """
        from src.services.counterfactual_engine import _TOPO_SORT_CACHE_MAX

        engine = CounterfactualEngine()

        n = _TOPO_SORT_CACHE_MAX + 32
        for i in range(n):
            model = StructuralModel(
                variables=["X", "Y"],
                equations={"Y": f"{i} + 2*X"},  # distinct equation set per i
                distributions={
                    "X": Distribution(
                        type=DistributionType.NORMAL,
                        parameters={"mean": 5.0, "std": 1.0},
                    )
                },
            )
            request = CounterfactualRequest(
                model=model,
                intervention={"X": 1.0},
                outcome="Y",
                context={},
            )
            engine.analyze(request)

        # Bounded: cache size is capped regardless of how many distinct models arrive.
        assert len(engine._topo_sort_cache) <= _TOPO_SORT_CACHE_MAX
        # Sanity: it is actually saturated to the cap (proves the eviction path ran and
        # the bound is what holds size down, not that fewer than n models were seen).
        assert len(engine._topo_sort_cache) == _TOPO_SORT_CACHE_MAX

    def test_cache_hit_returns_same_result(self):
        """A repeated identical model reuses the cached sort (read-through preserved)."""
        engine = CounterfactualEngine()
        equations = {"B": "2 * A", "C": "B + 3", "D": "C * 1.5"}
        first = engine._topological_sort_equations(equations)
        second = engine._topological_sort_equations(dict(equations))
        assert first == second
        assert len(engine._topo_sort_cache) == 1

    def test_cache_read_and_write_run_under_the_lock(self):
        """The cache read and the insert/evict both run under the instance lock.

        The engine is a module-level singleton served on a concurrent request path
        (``counterfactual_engine`` in causal.py); the FIFO evict step
        (``pop(next(iter(...)))`` racing an insert) is only concurrency-safe because
        every read/insert/evict is wrapped in ``self._topo_sort_cache_lock``.

        This is a DETERMINISTIC mechanism check (no threads): under CPython's GIL a
        thread race on this dict is not deterministically observable — the individual
        ops are GIL-atomic and the compound-op effects are benign (no wrong value),
        so a "hammer N threads, assert no exception" test would pass with OR without
        the lock (vacuous). Instead a counting proxy over the real lock proves the
        guards execute: an initial MISS enters the lock twice (read guard + insert/
        evict guard) and a subsequent HIT enters it once (read guard). Reverting
        either ``with self._topo_sort_cache_lock:`` drops the count and fails here.
        """
        engine = CounterfactualEngine()

        class _CountingLock:
            def __init__(self, inner):
                self._inner = inner
                self.acquisitions = 0

            def __enter__(self):
                self.acquisitions += 1
                return self._inner.__enter__()

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

        counting = _CountingLock(engine._topo_sort_cache_lock)
        engine._topo_sort_cache_lock = counting

        equations = {"B": "2 * A", "C": "B + 3"}
        engine._topological_sort_equations(equations)  # miss: read guard + insert/evict guard
        assert counting.acquisitions >= 2, "read and insert/evict must each acquire the lock"

        acquisitions_after_miss = counting.acquisitions
        engine._topological_sort_equations(dict(equations))  # hit: read guard only
        assert (
            counting.acquisitions == acquisitions_after_miss + 1
        ), "the cache read must acquire the lock"
