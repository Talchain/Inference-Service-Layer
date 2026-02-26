"""
Tests for the outcome stability benchmark utility (1A-ISL).

NOT part of the standard pytest run — this is benchmark infrastructure.
Run explicitly::

    ISL_AUTH_DISABLED=true python -m pytest tests/benchmarks/test_outcome_stability.py -v
"""

import copy

import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    StrengthDistribution,
)
from tests.benchmarks.outcome_stability import (
    OutcomeStabilityResult,
    measure_outcome_stability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(price_to_revenue_mean: float = 0.5, price_to_revenue_std: float = 0.1) -> GraphV2:
    """Build a minimal 3-node graph with tuneable price→revenue edge."""
    return GraphV2(
        nodes=[
            NodeV2(id="price", kind="factor", label="Price"),
            NodeV2(id="demand", kind="chance", label="Demand"),
            NodeV2(id="revenue", kind="outcome", label="Revenue"),
        ],
        edges=[
            EdgeV2(
                **{"from": "price", "to": "demand"},
                exists_probability=0.95,
                strength=StrengthDistribution(mean=-0.4, std=0.1),
            ),
            EdgeV2(
                **{"from": "demand", "to": "revenue"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.8, std=0.05),
            ),
            EdgeV2(
                **{"from": "price", "to": "revenue"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=price_to_revenue_mean, std=price_to_revenue_std),
            ),
        ],
    )


OPTIONS = [
    InterventionOption(id="low_price", label="Low price", interventions={"price": 0.3}),
    InterventionOption(id="high_price", label="High price", interventions={"price": 0.7}),
]

GOAL = "revenue"
SEED = 42
N_SAMPLES = 200  # fast enough for benchmarks, enough for stability


# ---------------------------------------------------------------------------
# Test 1: Identical graphs → stable results
# ---------------------------------------------------------------------------

class TestIdenticalGraphsStable:
    def test_identical_graphs_produce_stable_results(self):
        """Same graph 5× with same seed → no option change, perfect rank stability, CV ≈ 0."""
        graph = _make_graph()
        graphs = [copy.deepcopy(graph) for _ in range(5)]

        result = measure_outcome_stability(graphs, OPTIONS, GOAL, seed=SEED, n_samples=N_SAMPLES)

        assert result.recommended_option_changes is False
        assert result.option_rank_stability == 1.0
        assert result.outcome_mean_cv == pytest.approx(0.0, abs=1e-6)
        assert result.n_runs == 5
        assert len(set(result.per_run_winners)) == 1


# ---------------------------------------------------------------------------
# Test 2: Divergent graphs → instability signal
# ---------------------------------------------------------------------------

class TestDivergentGraphsInstability:
    def test_inverted_edge_causes_instability(self):
        """One graph with a key edge weight inverted should signal instability."""
        base_graph = _make_graph(price_to_revenue_mean=0.7)
        inverted_graph = _make_graph(price_to_revenue_mean=-0.7)

        graphs = [
            copy.deepcopy(base_graph),
            copy.deepcopy(base_graph),
            copy.deepcopy(inverted_graph),  # flipped
            copy.deepcopy(base_graph),
            copy.deepcopy(base_graph),
        ]

        result = measure_outcome_stability(graphs, OPTIONS, GOAL, seed=SEED, n_samples=N_SAMPLES)

        # At least one of these must signal instability
        assert result.recommended_option_changes is True or result.option_rank_stability < 1.0


# ---------------------------------------------------------------------------
# Test 3: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_twice_identical_output(self):
        """Exact same input → identical result."""
        graphs = [_make_graph() for _ in range(3)]

        r1 = measure_outcome_stability(
            [copy.deepcopy(g) for g in graphs], OPTIONS, GOAL, seed=SEED, n_samples=N_SAMPLES,
        )
        r2 = measure_outcome_stability(
            [copy.deepcopy(g) for g in graphs], OPTIONS, GOAL, seed=SEED, n_samples=N_SAMPLES,
        )

        assert r1.recommended_option_changes == r2.recommended_option_changes
        assert r1.outcome_mean_cv == r2.outcome_mean_cv
        assert r1.option_rank_stability == r2.option_rank_stability
        assert r1.per_run_winners == r2.per_run_winners


# ---------------------------------------------------------------------------
# Test 4: Single run edge case
# ---------------------------------------------------------------------------

class TestSingleRun:
    def test_single_run_stable(self):
        """N=1 → no changes possible, perfect stability, CV=0."""
        graph = _make_graph()

        result = measure_outcome_stability([graph], OPTIONS, GOAL, seed=SEED, n_samples=N_SAMPLES)

        assert result.recommended_option_changes is False
        assert result.option_rank_stability == 1.0
        assert result.outcome_mean_cv == 0.0
        assert result.n_runs == 1


# ---------------------------------------------------------------------------
# Test 5: CV epsilon floor
# ---------------------------------------------------------------------------

class TestCVEpsilonFloor:
    def test_near_zero_mean_uses_epsilon_floor(self):
        """When recommended option's outcome mean is near zero, CV uses epsilon floor."""
        # Graph where outcomes are near zero: very weak edges, intercepts = 0
        graph = GraphV2(
            nodes=[
                NodeV2(id="x", kind="factor", label="X"),
                NodeV2(id="y", kind="outcome", label="Y"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "x", "to": "y"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.002, std=0.002),
                ),
            ],
        )
        options = [
            InterventionOption(id="a", label="A", interventions={"x": 0.01}),
            InterventionOption(id="b", label="B", interventions={"x": 0.02}),
        ]

        # Use 5 copies with tiny perturbations to edge mean
        graphs = []
        for i in range(5):
            g = copy.deepcopy(graph)
            # Vary the edge mean slightly: 0.002, 0.003, 0.004, ...
            g.edges[0].strength = StrengthDistribution(mean=0.002 + i * 0.001, std=0.002)
            graphs.append(g)

        result = measure_outcome_stability(graphs, options, "y", seed=SEED, n_samples=N_SAMPLES)

        # CV should be finite and small (epsilon floor prevents explosion)
        assert result.outcome_mean_cv < 10.0  # not infinite
        assert result.outcome_mean_cv >= 0.0


# ---------------------------------------------------------------------------
# Test: Result dataclass serialization
# ---------------------------------------------------------------------------

class TestResultSerialization:
    def test_to_dict(self):
        """Verify to_dict produces the expected shape."""
        result = OutcomeStabilityResult(
            recommended_option_changes=True,
            outcome_mean_cv=0.123,
            option_rank_stability=0.8,
            per_run_winners=["a", "a", "b", "a", "a"],
            n_runs=5,
        )
        d = result.to_dict()
        assert d["recommended_option_changes"] is True
        assert d["outcome_mean_cv"] == 0.123
        assert d["option_rank_stability"] == 0.8
        assert d["per_run_winners"] == ["a", "a", "b", "a", "a"]
        assert d["n_runs"] == 5


# ---------------------------------------------------------------------------
# Test: Empty input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_graphs_raises(self):
        """Zero graphs → ValueError."""
        with pytest.raises(ValueError, match="At least one graph"):
            measure_outcome_stability([], OPTIONS, GOAL, seed=SEED)
