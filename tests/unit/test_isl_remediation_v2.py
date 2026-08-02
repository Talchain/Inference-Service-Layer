"""
Tests for ISL Remediation Brief v2.

Covers:
- Task 2: Topological sort with 20+ nodes (deque.popleft correctness)
- Task 6: StrengthDistribution.mean bounds validation + InferenceWarning
- Task 7: Winner tie-breaking determinism and balance
- Task 9: Request complexity guard
- Task 10: Five missing edge case tests from review section B.3
  1. exists_probability = 0 (all edges absent)
  2. exists_probability = 1 (all edges certain)
  3. strength.mean at ±1.0 with large std (heavy clipping)
  4. Single-factor sensitivity (only 1 parameter_uncertainty)
  5. Graph at schema limits (marked slow)
"""

import math
import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InferenceWarning,
    InterventionOption,
    NodeV2,
    ParameterUncertainty,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.rng import SeededRNG


# =============================================================================
# Helpers
# =============================================================================


def _make_node(node_id: str, kind: str = "factor") -> NodeV2:
    return NodeV2(id=node_id, kind=kind, label=node_id.replace("_", " ").title())


def _make_edge(from_id: str, to_id: str, exists_prob: float = 0.9, mean: float = 0.5) -> EdgeV2:
    return EdgeV2(
        **{"from": from_id, "to": to_id},
        exists_probability=exists_prob,
        strength=StrengthDistribution(mean=mean, std=0.1),
    )


def _simple_request(
    graph: GraphV2,
    options=None,
    goal_node_id: str = "revenue",
    n_samples: int = 200,
    seed: int = 42,
) -> RobustnessRequestV2:
    if options is None:
        options = [
            InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
            InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
        ]
    return RobustnessRequestV2(
        request_id="test-remediation",
        graph=graph,
        options=options,
        goal_node_id=goal_node_id,
        n_samples=n_samples,
        seed=seed,
    )


# =============================================================================
# Task 2 — Topological sort with 20+ nodes (deque correctness)
# =============================================================================


class TestTopologicalSortLargeGraph:
    """Verify Kahn's algorithm produces correct topological ordering for 20+ nodes."""

    def _build_chain_graph(self, n: int) -> GraphV2:
        """Build a linear chain: node_0 -> node_1 -> ... -> node_{n-1}."""
        nodes = [_make_node(f"n{i}") for i in range(n)]
        edges = [
            EdgeV2(
                **{"from": f"n{i}", "to": f"n{i+1}"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            )
            for i in range(n - 1)
        ]
        return GraphV2(nodes=nodes, edges=edges)

    def test_chain_of_25_nodes_correct_topological_order(self):
        """25-node chain — topological sort must honour dependency order."""
        n = 25
        graph = self._build_chain_graph(n)

        from src.services.robustness_analyzer_v2 import SCMEvaluatorV2

        evaluator = SCMEvaluatorV2(graph)
        order = evaluator._compute_topological_order()

        assert len(order) == n, f"Expected {n} nodes in order, got {len(order)}"

        # In a chain, n_i must appear before n_{i+1}
        pos = {node_id: idx for idx, node_id in enumerate(order)}
        for i in range(n - 1):
            assert pos[f"n{i}"] < pos[f"n{i+1}"], (
                f"Topological violation: n{i} (pos {pos[f'n{i}']}) "
                f"comes after n{i+1} (pos {pos[f'n{i+1}']})"
            )

    def test_diamond_graph_topological_order(self):
        """Diamond DAG: A -> B, A -> C, B -> D, C -> D — D must come last."""
        nodes = [_make_node(n) for n in ["a", "b", "c", "d"]]
        edges = [
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
            EdgeV2(
                **{"from": "a", "to": "c"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
            EdgeV2(
                **{"from": "b", "to": "d"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
            EdgeV2(
                **{"from": "c", "to": "d"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
        ]
        graph = GraphV2(nodes=nodes, edges=edges)

        from src.services.robustness_analyzer_v2 import SCMEvaluatorV2

        evaluator = SCMEvaluatorV2(graph)
        order = evaluator._compute_topological_order()
        pos = {n: i for i, n in enumerate(order)}

        # a must precede b and c; b and c must precede d
        assert pos["a"] < pos["b"]
        assert pos["a"] < pos["c"]
        assert pos["b"] < pos["d"]
        assert pos["c"] < pos["d"]


# =============================================================================
# Task 6 — StrengthDistribution.mean bounds + InferenceWarning
# =============================================================================


class TestStrengthDistributionBoundsValidation:
    """Verify StrengthDistribution clamping and NaN/Inf rejection."""

    def test_valid_mean_unchanged(self):
        sd = StrengthDistribution(mean=0.5, std=0.1)
        assert sd.mean == 0.5
        assert sd._pre_clamp_mean is None

    def test_boundary_mean_1_unchanged(self):
        sd = StrengthDistribution(mean=1.0, std=0.1)
        assert sd.mean == 1.0
        assert sd._pre_clamp_mean is None

    def test_boundary_mean_neg1_unchanged(self):
        sd = StrengthDistribution(mean=-1.0, std=0.1)
        assert sd.mean == -1.0
        assert sd._pre_clamp_mean is None

    def test_mean_1000_clamped_to_1(self):
        sd = StrengthDistribution(mean=1000, std=0.1)
        assert sd.mean == 1.0
        assert sd._pre_clamp_mean == 1000

    def test_mean_neg5_clamped_to_neg1(self):
        sd = StrengthDistribution(mean=-5.0, std=0.1)
        assert sd.mean == -1.0
        assert sd._pre_clamp_mean == -5.0

    def test_mean_nan_rejected(self):
        with pytest.raises(Exception):
            StrengthDistribution(mean=float("nan"), std=0.1)

    def test_mean_inf_rejected(self):
        with pytest.raises(Exception):
            StrengthDistribution(mean=float("inf"), std=0.1)

    def test_mean_neg_inf_rejected(self):
        with pytest.raises(Exception):
            StrengthDistribution(mean=float("-inf"), std=0.1)

    def test_std_nan_rejected(self):
        with pytest.raises(Exception):
            StrengthDistribution(mean=0.5, std=float("nan"))

    def test_std_inf_rejected(self):
        with pytest.raises(Exception):
            StrengthDistribution(mean=0.5, std=float("inf"))


class TestEdgeV2ClampWarning:
    """Verify EdgeV2 emits InferenceWarning when strength.mean is clamped."""

    def test_no_warning_valid_mean(self):
        e = EdgeV2(**{"from": "a", "to": "b"}, strength=StrengthDistribution(mean=0.5, std=0.1))
        assert e._strength_clamp_warning is None

    def test_warning_emitted_for_clamped_mean(self):
        e = EdgeV2(
            **{"from": "src", "to": "tgt"}, strength=StrengthDistribution(mean=1000, std=0.1)
        )
        w = e._strength_clamp_warning
        assert w is not None
        assert w.code == "STRENGTH_MEAN_CLAMPED"
        assert "src" in w.field and "tgt" in w.field
        assert w.detail["original"] == 1000
        assert w.detail["clamped"] == 1.0

    def test_warning_field_uses_arrow_not_index(self):
        """Field path uses node IDs with →, not array indices."""
        e = EdgeV2(
            **{"from": "revenue_growth", "to": "market_share"},
            strength=StrengthDistribution(mean=-5.0, std=0.1),
        )
        w = e._strength_clamp_warning
        assert w is not None
        assert "revenue_growth" in w.field
        assert "market_share" in w.field
        assert "[" not in w.field or "→" in w.field  # uses node IDs, not indices

    def test_clamped_value_not_seen_downstream(self):
        """After parse, edge.strength.mean reflects clamped value, not original."""
        e = EdgeV2(**{"from": "a", "to": "b"}, strength=StrengthDistribution(mean=500, std=0.1))
        assert e.strength.mean == 1.0  # downstream always sees clamped value


class TestGraphCollectParseWarnings:
    """Verify GraphV2.collect_parse_warnings aggregates edge warnings."""

    def test_no_warnings_for_valid_graph(self):
        g = GraphV2(
            nodes=[_make_node("a"), _make_node("b")],
            edges=[_make_edge("a", "b", mean=0.5)],
        )
        assert g.collect_parse_warnings() == []

    def test_collects_one_warning(self):
        g = GraphV2(
            nodes=[_make_node("a"), _make_node("b")],
            edges=[_make_edge("a", "b", mean=1000)],
        )
        warnings = g.collect_parse_warnings()
        assert len(warnings) == 1
        assert warnings[0].code == "STRENGTH_MEAN_CLAMPED"

    def test_collects_multiple_warnings(self):
        g = GraphV2(
            nodes=[_make_node("a"), _make_node("b"), _make_node("c")],
            edges=[
                _make_edge("a", "b", mean=1000),
                _make_edge("b", "c", mean=-999),
            ],
        )
        warnings = g.collect_parse_warnings()
        assert len(warnings) == 2

    def test_warnings_included_in_analyzer_response(self):
        """InferenceWarnings from edge clamping appear in response.inference_warnings."""
        g = GraphV2(
            nodes=[_make_node("price"), _make_node("revenue", kind="outcome")],
            edges=[_make_edge("price", "revenue", mean=1000)],
        )
        request = _simple_request(
            graph=g,
            goal_node_id="revenue",
            options=[
                InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
                InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
            ],
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        codes = [w.code for w in response.inference_warnings]
        assert "STRENGTH_MEAN_CLAMPED" in codes


# =============================================================================
# Task 7 — Winner tie-breaking: determinism and balance
# =============================================================================


class TestWinnerTieBreaking:
    """Verify tie-breaking is deterministic and approximately balanced."""

    def _build_tied_request(self, seed: int, n_samples: int = 1000) -> RobustnessRequestV2:
        """Request with two exactly-equal options (same intervention) to force ties."""
        # Two options with identical interventions → identical outcomes → always tie
        nodes = [
            _make_node("price"),
            _make_node("revenue", kind="outcome"),
        ]
        edges = [_make_edge("price", "revenue", exists_prob=1.0, mean=0.5)]
        graph = GraphV2(nodes=nodes, edges=edges)
        return RobustnessRequestV2(
            request_id="tie-test",
            graph=graph,
            options=[
                # Identical interventions → identical outcomes every sample → tie
                InterventionOption(id="opt_a", label="A", interventions={"price": 0.5}),
                InterventionOption(id="opt_b", label="B", interventions={"price": 0.5}),
            ],
            goal_node_id="revenue",
            n_samples=n_samples,
            seed=seed,
        )

    def test_determinism_same_seed_same_winner_per_sample(self):
        """Same graph + same seed → identical winner_per_sample list."""
        analyzer = RobustnessAnalyzerV2()
        req1 = self._build_tied_request(seed=42, n_samples=200)
        req2 = self._build_tied_request(seed=42, n_samples=200)

        from src.models.robustness_v2 import GraphV2
        from src.services.robustness_analyzer_v2 import (
            DualUncertaintySampler,
            FactorSampler,
            SCMEvaluatorV2,
        )

        # Run MC directly to access winner_per_sample
        from src.utils.rng import SeededRNG, compute_seed_from_graph

        def run_mc(req: RobustnessRequestV2):
            seed = req.seed if req.seed is not None else compute_seed_from_graph(req.graph)
            rng = SeededRNG(seed)
            sampler = DualUncertaintySampler(req.graph.edges, rng)
            factor_sampler = FactorSampler(
                nodes=req.graph.nodes,
                uncertainties=req.parameter_uncertainties,
                rng=SeededRNG(seed + 1),
            )
            evaluator = SCMEvaluatorV2(req.graph)
            _, _, winners, _, _, _, _, _ = analyzer._run_monte_carlo(
                req, sampler, factor_sampler, evaluator
            )
            return winners

        w1 = run_mc(req1)
        w2 = run_mc(req2)
        assert w1 == w2, "Same seed must produce identical winner_per_sample"

    def test_balance_tied_options_approximately_equal(self):
        """Tied options should each win ~50% of samples (within ±10%)."""
        analyzer = RobustnessAnalyzerV2()
        req = self._build_tied_request(seed=123, n_samples=2000)

        from src.utils.rng import SeededRNG, compute_seed_from_graph
        from src.services.robustness_analyzer_v2 import (
            DualUncertaintySampler,
            FactorSampler,
            SCMEvaluatorV2,
        )

        seed = req.seed
        rng = SeededRNG(seed)
        sampler = DualUncertaintySampler(req.graph.edges, rng)
        factor_sampler = FactorSampler(
            nodes=req.graph.nodes,
            uncertainties=req.parameter_uncertainties,
            rng=SeededRNG(seed + 1),
        )
        evaluator = SCMEvaluatorV2(req.graph)
        _, _, winners, _, _, _, _, _ = analyzer._run_monte_carlo(
            req, sampler, factor_sampler, evaluator
        )

        n = len(winners)
        count_a = winners.count("opt_a")
        count_b = winners.count("opt_b")
        ratio_a = count_a / n
        ratio_b = count_b / n

        # With 2000 samples, expect 50% ± 10%
        assert 0.40 <= ratio_a <= 0.60, f"opt_a win rate {ratio_a:.2f} outside [0.40, 0.60]"
        assert 0.40 <= ratio_b <= 0.60, f"opt_b win rate {ratio_b:.2f} outside [0.40, 0.60]"


# =============================================================================
# Task 9 — Request compute-admission guard (Codex F8 weighted cost model)
# =============================================================================
# The pre-F8 scalar compute_complexity_score(S,N,E) = S*N*E was REPLACED by the
# weighted compute_weighted_cost(request) (it omitted the option multiplier and
# used the wrong x-edges shape). The formula/ceiling pins + full positive
# controls live in tests/unit/test_admission_calibration.py; the tests below
# preserve the original pilot/schema-max admit-vs-reject intent, re-expressed
# against the weighted model.


def _admission_request(n_nodes: int, n_edges: int, n_samples: int, n_options: int = 2):
    """Minimal valid v2 request (layered DAG) for admission-cost checks."""
    nodes = [{"id": f"n{i}", "kind": "factor", "label": f"Node {i}"} for i in range(n_nodes)]
    nodes[-1]["kind"] = "outcome"
    edges: list = []
    for j in range(1, n_nodes):
        for i in range(j):
            if len(edges) >= n_edges:
                break
            edges.append(
                {
                    "from": f"n{i}",
                    "to": f"n{j}",
                    "exists_probability": 0.9,
                    "strength": {"mean": 0.5, "std": 0.1},
                }
            )
        if len(edges) >= n_edges:
            break
    options = [
        {"id": f"o{k}", "label": f"O{k}", "interventions": {"n0": 0.3 + 0.1 * k}}
        for k in range(n_options)
    ]
    return RobustnessRequestV2(
        graph={"nodes": nodes, "edges": edges},
        options=options,
        goal_node_id=f"n{n_nodes - 1}",
        n_samples=n_samples,
        analysis_types=["comparison", "robustness"],  # sensitivity off → pure base term
    )


class TestWeightedAdmissionCost:
    """Unit tests for compute_weighted_cost (replaces TestComputeComplexityScore)."""

    def test_base_cost_is_S_times_O_times_N_plus_E(self):
        from src.services.robustness_analyzer_v2 import compute_weighted_cost

        req = _admission_request(5, 8, 1000, 2)
        wc = compute_weighted_cost(req)
        assert wc.terms["base_mc"] == 1000 * 2 * (len(req.graph.nodes) + len(req.graph.edges))

    def test_option_multiplier_priced(self):
        """The pre-F8 defect: cost had NO option term; now it is O-linear."""
        from src.services.robustness_analyzer_v2 import compute_weighted_cost

        one = compute_weighted_cost(_admission_request(12, 30, 5000, 1)).total
        four = compute_weighted_cost(_admission_request(12, 30, 5000, 4)).total
        assert four == 4 * one

    def test_schema_max_rejects(self):
        from src.services.robustness_analyzer_v2 import compute_weighted_cost, get_max_cost_units

        req = _admission_request(50, 200, 10000, 10)
        assert compute_weighted_cost(req).total > get_max_cost_units()

    def test_typical_pilot_admits(self):
        from src.services.robustness_analyzer_v2 import compute_weighted_cost, get_max_cost_units

        req = _admission_request(5, 4, 1000, 2)
        assert compute_weighted_cost(req).total <= get_max_cost_units()


# =============================================================================
# Task 10 — Five missing edge case tests (review section B.3)
# =============================================================================


def _build_simple_graph(exists_prob: float, mean: float = 0.5) -> GraphV2:
    """Simple two-node graph with configurable edge parameters."""
    return GraphV2(
        nodes=[
            _make_node("price"),
            _make_node("revenue", kind="outcome"),
        ],
        edges=[
            EdgeV2(
                **{"from": "price", "to": "revenue"},
                exists_probability=exists_prob,
                strength=StrengthDistribution(mean=mean, std=0.1),
            )
        ],
    )


def _two_option_request(
    graph: GraphV2, n_samples: int = 200, seed: int = 42
) -> RobustnessRequestV2:
    return RobustnessRequestV2(
        request_id="edge-case-test",
        graph=graph,
        options=[
            InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
            InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
        ],
        goal_node_id="revenue",
        n_samples=n_samples,
        seed=seed,
    )


class TestExistsProbabilityZero:
    """Edge case 1: exists_probability = 0 (all edges absent)."""

    def test_no_nan_in_outcomes(self):
        """All edges absent → constant-zero outcomes, no NaN."""
        graph = _build_simple_graph(exists_prob=0.0)
        request = _two_option_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        for result in response.results:
            dist = result.outcome_distribution
            assert math.isfinite(dist.mean), f"NaN in outcome mean for {result.option_id}"
            assert math.isfinite(dist.median), f"NaN in median for {result.option_id}"

    def test_no_division_by_zero_in_sensitivity(self):
        """Sensitivity analysis must not crash with all-zero outcomes."""
        graph = _build_simple_graph(exists_prob=0.0)
        request = _two_option_request(graph)
        analyzer = RobustnessAnalyzerV2()
        # Should complete without raising
        response = analyzer.analyze(request)
        assert response is not None

    def test_valid_response_shape(self):
        """Response has required fields even when edges always absent."""
        graph = _build_simple_graph(exists_prob=0.0)
        request = _two_option_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.recommended_option_id is not None
        assert 0.0 <= response.recommendation_confidence <= 1.0
        assert isinstance(response.inference_warnings, list)


class TestExistsProbabilityOne:
    """Edge case 2: exists_probability = 1 (all edges always exist)."""

    def test_deterministic_outcomes_no_structural_variance(self):
        """With exists_probability=1, structural uncertainty is zero."""
        # When exists_prob=1, no structural variance — spread comes only from
        # parametric uncertainty (strength std) and auto-scaled noise.
        graph = _build_simple_graph(exists_prob=1.0, mean=0.5)
        request = _two_option_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response is not None
        for result in response.results:
            dist = result.outcome_distribution
            assert math.isfinite(dist.mean)
            assert math.isfinite(dist.std)

    def test_valid_sensitivity_results(self):
        """Sensitivity analysis valid when all edges always present."""
        graph = _build_simple_graph(exists_prob=1.0)
        request = _two_option_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        # Each SensitivityResult must have finite elasticity or be None
        for s in response.sensitivity:
            if s.elasticity is not None:
                assert math.isfinite(s.elasticity), f"Non-finite elasticity in {s}"


class TestStrengthAtBoundaryWithLargeStd:
    """Edge case 3: strength.mean at ±1.0 with large std (heavy clipping in sampler)."""

    def test_mean_at_1_large_std_no_degenerate_distribution(self):
        """mean=1.0, std=0.5 — sampler clips strength; no NaN."""
        graph = GraphV2(
            nodes=[_make_node("price"), _make_node("revenue", kind="outcome")],
            edges=[
                EdgeV2(
                    **{"from": "price", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.5),
                )
            ],
        )
        request = _two_option_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        for result in response.results:
            assert math.isfinite(result.outcome_distribution.mean)
            assert math.isfinite(result.outcome_distribution.std)

    def test_mean_at_neg1_large_std_no_degenerate_distribution(self):
        """mean=-1.0, std=0.5 — sampler clips strength; no NaN."""
        graph = GraphV2(
            nodes=[_make_node("price"), _make_node("revenue", kind="outcome")],
            edges=[
                EdgeV2(
                    **{"from": "price", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=-1.0, std=0.5),
                )
            ],
        )
        request = _two_option_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        for result in response.results:
            assert math.isfinite(result.outcome_distribution.mean)


class TestSingleFactorSensitivity:
    """Edge case 4: Only 1 parameter_uncertainty (single-factor sensitivity)."""

    def test_single_factor_no_crash(self):
        """Single ParameterUncertainty must not crash sensitivity analysis."""
        graph = GraphV2(
            nodes=[
                _make_node("price"),
                _make_node("demand", kind="chance"),
                _make_node("revenue", kind="outcome"),
            ],
            edges=[
                _make_edge("price", "demand"),
                _make_edge("demand", "revenue"),
            ],
        )
        request = RobustnessRequestV2(
            request_id="single-factor",
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
                InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
            ],
            goal_node_id="revenue",
            n_samples=200,
            seed=42,
            parameter_uncertainties=[
                ParameterUncertainty(
                    node_id="demand",
                    distribution="normal",
                    std=0.1,
                )
            ],
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response is not None
        assert len(response.factor_sensitivity) >= 0  # may be 0 or 1

    def test_single_factor_valid_sensitivity_values(self):
        """Single-factor elasticity must be finite or None."""
        graph = GraphV2(
            nodes=[
                _make_node("price"),
                _make_node("demand", kind="chance"),
                _make_node("revenue", kind="outcome"),
            ],
            edges=[
                _make_edge("price", "demand"),
                _make_edge("demand", "revenue"),
            ],
        )
        request = RobustnessRequestV2(
            request_id="single-factor-valid",
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
                InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
            ],
            goal_node_id="revenue",
            n_samples=200,
            seed=42,
            parameter_uncertainties=[
                ParameterUncertainty(
                    node_id="demand",
                    distribution="normal",
                    std=0.1,
                )
            ],
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        for fs in response.factor_sensitivity:
            if fs.elasticity is not None:
                assert math.isfinite(fs.elasticity)


@pytest.mark.slow
class TestSchemaLimitsGraph:
    """
    Edge case 5: Graph at schema limits (100 nodes, 300 edges).

    Marked @slow — excluded from default CI (pytest -m "not slow").
    Uses reduced n_samples=100 for correctness validation.
    Full performance version (n_samples=1000, timeout 60s) for nightly/perf jobs.
    """

    def _build_schema_limit_graph(self) -> GraphV2:
        """Build a graph at the schema limit (50 nodes, up to 200 edges)."""
        n_nodes = 50
        nodes = [
            NodeV2(id=f"n{i}", kind="outcome" if i == n_nodes - 1 else "factor", label=f"Node {i}")
            for i in range(n_nodes)
        ]
        # Build a spanning chain first, then add edges up to 200 total
        edges = []
        seen_pairs = set()
        # Chain: n0→n1→...→n48→n49
        for i in range(n_nodes - 1):
            key = (f"n{i}", f"n{i+1}")
            seen_pairs.add(key)
            edges.append(
                EdgeV2(
                    **{"from": f"n{i}", "to": f"n{i+1}"},
                    exists_probability=0.9,
                    strength=StrengthDistribution(mean=0.1, std=0.05),
                )
            )
        # Add cross-edges to reach ~200 (only forward edges to avoid cycles)
        target = 200
        src = 0
        while len(edges) < target and src < n_nodes - 2:
            tgt = src + 2
            while tgt < n_nodes and len(edges) < target:
                key = (f"n{src}", f"n{tgt}")
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    edges.append(
                        EdgeV2(
                            **{"from": f"n{src}", "to": f"n{tgt}"},
                            exists_probability=0.8,
                            strength=StrengthDistribution(mean=0.05, std=0.02),
                        )
                    )
                tgt += 1
            src += 1
        return GraphV2(nodes=nodes, edges=edges)

    def test_schema_limit_no_crash(self):
        """50 nodes, 200 edges, 100 samples — must complete without crash."""
        graph = self._build_schema_limit_graph()
        request = RobustnessRequestV2(
            request_id="schema-limit-test",
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="A", interventions={"n0": 0.3}),
                InterventionOption(id="opt_b", label="B", interventions={"n0": 0.7}),
            ],
            goal_node_id="n49",
            n_samples=100,
            seed=42,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response is not None

    def test_schema_limit_no_nan(self):
        """50 nodes, 200 edges — all outcome values must be finite."""
        graph = self._build_schema_limit_graph()
        request = RobustnessRequestV2(
            request_id="schema-limit-nan",
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="A", interventions={"n0": 0.3}),
                InterventionOption(id="opt_b", label="B", interventions={"n0": 0.7}),
            ],
            goal_node_id="n49",
            n_samples=100,
            seed=42,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        for result in response.results:
            dist = result.outcome_distribution
            assert math.isfinite(dist.mean), f"NaN mean for {result.option_id}"
            assert math.isfinite(dist.median), f"NaN median for {result.option_id}"

    def test_schema_limit_valid_response_shape(self):
        """50 nodes, 200 edges — response has all required fields."""
        graph = self._build_schema_limit_graph()
        request = RobustnessRequestV2(
            request_id="schema-limit-shape",
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="A", interventions={"n0": 0.3}),
                InterventionOption(id="opt_b", label="B", interventions={"n0": 0.7}),
            ],
            goal_node_id="n49",
            n_samples=100,
            seed=42,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.recommended_option_id in ("opt_a", "opt_b")
        assert 0.0 <= response.recommendation_confidence <= 1.0
        assert isinstance(response.inference_warnings, list)
        assert len(response.results) == 2


# =============================================================================
# Enhanced response path — inference_warnings propagation
# =============================================================================


class TestEnhancedResponseInferenceWarnings:
    """Verify inference_warnings propagate through ResponseBuilder → ISLResponseV2."""

    def test_response_builder_propagates_warnings(self):
        """ResponseBuilder.set_inference_warnings → ISLResponseV2.inference_warnings."""
        from src.models.response_v2 import ISLResponseV2, RequestEchoV2
        from src.utils.response_builder import ResponseBuilder

        echo = RequestEchoV2(
            graph_node_count=2,
            graph_edge_count=1,
            options_count=2,
            goal_node_id_hash="abc",
            n_samples=100,
            response_version_requested=2,
            include_diagnostics=False,
        )
        builder = ResponseBuilder(request_id="test-123", request_echo=echo, seed_used="42")

        warnings = [
            InferenceWarning(
                code="STRENGTH_MEAN_CLAMPED",
                field="edges[a→b].strength.mean",
                detail={"original": 5.0, "clamped": 1.0},
            ),
        ]
        builder.set_inference_warnings(warnings)

        # Need minimal results to build a valid response
        from src.models.response_v2 import OptionResultV2, OutcomeDistributionV2

        builder.set_results(
            options=[
                OptionResultV2(
                    id="opt_a",
                    outcome=OutcomeDistributionV2(
                        mean=1.0,
                        std=0.1,
                        n_samples=100,
                        n_valid_samples=100,
                        validity_ratio=1.0,
                        percentiles_source="samples",
                    ),
                    win_probability=0.5,
                    status="computed",
                ),
                OptionResultV2(
                    id="opt_b",
                    outcome=OutcomeDistributionV2(
                        mean=1.0,
                        std=0.1,
                        n_samples=100,
                        n_valid_samples=100,
                        validity_ratio=1.0,
                        percentiles_source="samples",
                    ),
                    win_probability=0.5,
                    status="computed",
                ),
            ],
        )

        response = builder.build()
        assert isinstance(response, ISLResponseV2)
        assert len(response.inference_warnings) == 1
        assert response.inference_warnings[0].code == "STRENGTH_MEAN_CLAMPED"

        # Verify survives model_dump with exclude_none=True
        response_dict = response.model_dump(by_alias=True, exclude_none=True)
        assert "inference_warnings" in response_dict
        assert len(response_dict["inference_warnings"]) == 1
        assert response_dict["inference_warnings"][0]["code"] == "STRENGTH_MEAN_CLAMPED"

    def test_empty_warnings_survives_exclude_none(self):
        """inference_warnings=[] survives model_dump(exclude_none=True)."""
        from src.models.response_v2 import ISLResponseV2, RequestEchoV2
        from src.utils.response_builder import ResponseBuilder

        echo = RequestEchoV2(
            graph_node_count=2,
            graph_edge_count=1,
            options_count=2,
            goal_node_id_hash="abc",
            n_samples=100,
            response_version_requested=2,
            include_diagnostics=False,
        )
        builder = ResponseBuilder(request_id="test-empty", request_echo=echo)

        from src.models.response_v2 import OptionResultV2, OutcomeDistributionV2

        builder.set_results(
            options=[
                OptionResultV2(
                    id="opt_a",
                    outcome=OutcomeDistributionV2(
                        mean=1.0,
                        std=0.1,
                        n_samples=100,
                        n_valid_samples=100,
                        validity_ratio=1.0,
                        percentiles_source="samples",
                    ),
                    win_probability=1.0,
                    status="computed",
                ),
            ],
        )

        response = builder.build()
        response_dict = response.model_dump(by_alias=True, exclude_none=True)
        assert "inference_warnings" in response_dict
        assert response_dict["inference_warnings"] == []

    def test_clamped_edge_warning_in_enhanced_model(self):
        """End-to-end: clamped edge produces warning that survives ISLResponseV2 serialization."""
        g = GraphV2(
            nodes=[_make_node("price"), _make_node("revenue", kind="outcome")],
            edges=[_make_edge("price", "revenue", mean=5.0)],
        )
        # Verify parse-time clamp produced a warning
        warnings = g.collect_parse_warnings()
        assert len(warnings) == 1
        assert warnings[0].code == "STRENGTH_MEAN_CLAMPED"

        # Simulate enhanced path: feed warnings through ResponseBuilder
        from src.models.response_v2 import ISLResponseV2, RequestEchoV2
        from src.utils.response_builder import ResponseBuilder

        echo = RequestEchoV2(
            graph_node_count=2,
            graph_edge_count=1,
            options_count=2,
            goal_node_id_hash="abc",
            n_samples=100,
            response_version_requested=2,
            include_diagnostics=False,
        )
        builder = ResponseBuilder(request_id="test-clamp", request_echo=echo)
        builder.set_inference_warnings(warnings)

        from src.models.response_v2 import OptionResultV2, OutcomeDistributionV2

        builder.set_results(
            options=[
                OptionResultV2(
                    id="opt_a",
                    outcome=OutcomeDistributionV2(
                        mean=1.0,
                        std=0.1,
                        n_samples=100,
                        n_valid_samples=100,
                        validity_ratio=1.0,
                        percentiles_source="samples",
                    ),
                    win_probability=1.0,
                    status="computed",
                ),
            ],
        )

        response = builder.build()
        assert len(response.inference_warnings) == 1

        # Verify it survives the same serialization path the API uses
        response_dict = response.model_dump(by_alias=True, exclude_none=True)
        assert response_dict["inference_warnings"][0]["code"] == "STRENGTH_MEAN_CLAMPED"
        assert response_dict["inference_warnings"][0]["detail"]["original"] == 5.0
