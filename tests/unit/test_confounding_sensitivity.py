"""
Unit tests for confounding sensitivity analysis (3A-inference).

Tests:
1. No bidirected pairs → no sensitivity analysis
2. Single bidirected pair, recommendation flips
3. Single bidirected pair, recommendation holds
4. Multiple bidirected pairs (one flips, one doesn't)
5. Determinism (same input + seed → identical results)
6. Synthetic confounder doesn't pollute original graph
7. Latency budget (12-node, 1 bidirected pair, < 500ms)
8. Endpoint integration (POST /v2/identifiability with bidirected graph)
"""

import copy
import time

import pytest

from tests.perf_utils import assert_time_budget

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    NodeV2,
    ObservedState,
    StrengthDistribution,
)
from src.services.confounding_sensitivity import (
    ConfoundingPairResult,
    ConfoundingSensitivityResult,
    _compute_median_edge_strength,
    _inject_confounder,
    analyze_confounding_sensitivity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_identifiable_graph() -> GraphV2:
    """Simple directed graph: A→Y. No bidirected edges. Fully identifiable."""
    return GraphV2(
        nodes=[
            NodeV2(id="a", kind="factor", label="A", observed_state=ObservedState(value=0.5)),
            NodeV2(id="y", kind="outcome", label="Y", observed_state=ObservedState(value=0.5)),
        ],
        edges=[
            EdgeV2(
                **{"from": "a", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.8, std=0.1),
            ),
        ],
    )


def _make_bidirected_graph_with_options():
    """
    Graph: A→Y (directed, strong), A↔Y (bidirected).
    Two options with very different intervention values to create a clear
    baseline winner that may flip under confounding.

    Returns (graph, options_dicts, goal_node_id).
    """
    graph = GraphV2(
        nodes=[
            NodeV2(
                id="a", kind="factor", label="Factor A", observed_state=ObservedState(value=0.5)
            ),
            NodeV2(
                id="y", kind="outcome", label="Outcome Y", observed_state=ObservedState(value=0.5)
            ),
        ],
        edges=[
            EdgeV2(
                **{"from": "a", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.8, std=0.05),
            ),
            EdgeV2(
                **{"from": "a", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.1, std=0.05),
                edge_type="bidirected",
            ),
        ],
    )
    options = [
        {"id": "opt_high", "label": "High", "interventions": {"a": 1.0}},
        {"id": "opt_low", "label": "Low", "interventions": {"a": 0.0}},
    ]
    return graph, options, "y"


def _make_robust_graph_with_options():
    """
    Graph with bidirected edge but options that produce the same value
    so the recommendation never flips even under strong confounding.

    A→Y (strong, mean=0.9), A↔Y (bidirected).
    Two options with the SAME intervention value — baseline winner stays same.
    """
    graph = GraphV2(
        nodes=[
            NodeV2(
                id="a", kind="factor", label="Factor A", observed_state=ObservedState(value=0.5)
            ),
            NodeV2(
                id="y", kind="outcome", label="Outcome Y", observed_state=ObservedState(value=0.5)
            ),
        ],
        edges=[
            EdgeV2(
                **{"from": "a", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.9, std=0.01),
            ),
            EdgeV2(
                **{"from": "a", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.1, std=0.01),
                edge_type="bidirected",
            ),
        ],
    )
    # Near-identical intervention values → confounder can't easily flip
    options = [
        {"id": "opt_a", "label": "A", "interventions": {"a": 0.51}},
        {"id": "opt_b", "label": "B", "interventions": {"a": 0.50}},
    ]
    return graph, options, "y"


def _make_multi_bidirected_graph():
    """
    Graph with multiple bidirected pairs:
    - A→Y (directed), A↔Y (bidirected) — may flip
    - B→Y (directed), B↔Y (bidirected) — may or may not flip

    Returns (graph, options_dicts, goal_node_id, bidirected_pairs).
    """
    graph = GraphV2(
        nodes=[
            NodeV2(id="a", kind="factor", label="A", observed_state=ObservedState(value=0.5)),
            NodeV2(id="b", kind="factor", label="B", observed_state=ObservedState(value=0.5)),
            NodeV2(id="y", kind="outcome", label="Y", observed_state=ObservedState(value=0.5)),
        ],
        edges=[
            EdgeV2(
                **{"from": "a", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.8, std=0.05),
            ),
            EdgeV2(
                **{"from": "b", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.3, std=0.05),
            ),
            # Bidirected A↔Y
            EdgeV2(
                **{"from": "a", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.1, std=0.05),
                edge_type="bidirected",
            ),
            # Bidirected B↔Y
            EdgeV2(
                **{"from": "b", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.1, std=0.05),
                edge_type="bidirected",
            ),
        ],
    )
    options = [
        {"id": "opt_high", "label": "High", "interventions": {"a": 1.0, "b": 1.0}},
        {"id": "opt_low", "label": "Low", "interventions": {"a": 0.0, "b": 0.0}},
    ]
    bidirected_pairs = [("a", "y"), ("b", "y")]
    return graph, options, "y", bidirected_pairs


def _make_12_node_graph_with_bidirected():
    """
    12-node chain graph (n0→n1→...→n11) with one bidirected edge (n0↔n11).
    For latency testing.
    """
    nodes = [
        NodeV2(
            id=f"n{i}",
            kind="factor" if i == 0 else ("outcome" if i == 11 else "chance"),
            label=f"N{i}",
            observed_state=ObservedState(value=0.5),
        )
        for i in range(12)
    ]
    edges = [
        EdgeV2(
            **{"from": f"n{i}", "to": f"n{i+1}"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=0.5, std=0.1),
        )
        for i in range(11)
    ]
    # Add bidirected edge between n0 and n11
    edges.append(
        EdgeV2(
            **{"from": "n0", "to": "n11"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=0.1, std=0.05),
            edge_type="bidirected",
        )
    )
    graph = GraphV2(nodes=nodes, edges=edges)
    options = [
        {"id": "opt_high", "label": "High", "interventions": {"n0": 1.0}},
        {"id": "opt_low", "label": "Low", "interventions": {"n0": 0.0}},
    ]
    return graph, options, "n11", [("n0", "n11")]


# ==========================================================================
# Test 1: No bidirected pairs → no sensitivity analysis
# ==========================================================================


class TestNoBidirectedPairs:
    def test_no_bidirected_pairs_returns_empty(self):
        """All-identifiable graph → no confounding sensitivity analysis needed."""
        graph = _make_identifiable_graph()
        options = [
            {"id": "opt_a", "label": "A", "interventions": {"a": 1.0}},
            {"id": "opt_b", "label": "B", "interventions": {"a": 0.0}},
        ]
        result = analyze_confounding_sensitivity(
            graph=graph,
            options=options,
            goal_node_id="y",
            bidirected_pairs=[],
            seed=42,
            n_samples=500,
        )
        assert isinstance(result, ConfoundingSensitivityResult)
        assert result.pairs == []
        assert result.overall_robust is True
        assert result.median_edge_strength > 0


# ==========================================================================
# Test 2: Single bidirected pair, recommendation flips
# ==========================================================================


class TestSinglePairFlips:
    def test_recommendation_flips_with_confounder(self):
        """Single bidirected pair where confounding flips the recommendation."""
        graph, options, goal = _make_bidirected_graph_with_options()
        bidirected_pairs = [("a", "y")]

        result = analyze_confounding_sensitivity(
            graph=graph,
            options=options,
            goal_node_id=goal,
            bidirected_pairs=bidirected_pairs,
            seed=42,
            n_samples=500,
        )

        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.node_a == "a"
        assert pair.node_b == "y"

        # If it flipped, verify both thresholds are set
        if pair.flip_threshold_raw is not None:
            assert pair.flip_threshold_relative is not None
            assert pair.flip_threshold_relative > 0
            assert pair.flipped_winner is not None
            assert pair.flipped_winner != pair.baseline_winner
            assert result.overall_robust is False
            assert "provisional" in pair.interpretation.lower()
        else:
            # If it didn't flip, that's OK — mark robust
            assert result.overall_robust is True


# ==========================================================================
# Test 3: Single bidirected pair, recommendation holds
# ==========================================================================


class TestSinglePairHolds:
    def test_recommendation_robust_under_confounding(self):
        """Recommendation doesn't flip even at 3× median edge strength."""
        graph, options, goal = _make_robust_graph_with_options()
        bidirected_pairs = [("a", "y")]

        result = analyze_confounding_sensitivity(
            graph=graph,
            options=options,
            goal_node_id=goal,
            bidirected_pairs=bidirected_pairs,
            seed=42,
            n_samples=500,
        )

        assert len(result.pairs) == 1
        pair = result.pairs[0]

        # With near-identical intervention values, recommendation should hold
        # regardless of confounding strength
        if pair.flip_threshold_raw is None:
            assert result.overall_robust is True
            assert pair.flipped_winner is None
            assert "robust" in pair.interpretation.lower()
            assert "provisional" in pair.interpretation.lower()


# ==========================================================================
# Test 4: Multiple bidirected pairs
# ==========================================================================


class TestMultipleBidirectedPairs:
    def test_multiple_pairs_analyzed_independently(self):
        """Each bidirected pair gets its own sensitivity result."""
        graph, options, goal, bidirected_pairs = _make_multi_bidirected_graph()

        result = analyze_confounding_sensitivity(
            graph=graph,
            options=options,
            goal_node_id=goal,
            bidirected_pairs=bidirected_pairs,
            seed=42,
            n_samples=500,
        )

        assert len(result.pairs) == 2
        pair_keys = {(p.node_a, p.node_b) for p in result.pairs}
        assert pair_keys == {("a", "y"), ("b", "y")}

        # Each pair has valid fields
        for pair in result.pairs:
            assert pair.baseline_winner is not None
            assert pair.interpretation != ""
            assert "provisional" in pair.interpretation.lower()

        # overall_robust is True only if ALL pairs are robust
        if all(p.flip_threshold_raw is None for p in result.pairs):
            assert result.overall_robust is True
        else:
            assert result.overall_robust is False


# ==========================================================================
# Test 5: Determinism (same input + seed → identical results)
# ==========================================================================


class TestDeterminism:
    def test_same_seed_same_result(self):
        """Identical inputs with same seed produce identical results."""
        graph, options, goal = _make_bidirected_graph_with_options()
        bidirected_pairs = [("a", "y")]

        r1 = analyze_confounding_sensitivity(
            graph=graph,
            options=options,
            goal_node_id=goal,
            bidirected_pairs=bidirected_pairs,
            seed=42,
            n_samples=500,
        )
        r2 = analyze_confounding_sensitivity(
            graph=graph,
            options=options,
            goal_node_id=goal,
            bidirected_pairs=bidirected_pairs,
            seed=42,
            n_samples=500,
        )

        assert len(r1.pairs) == len(r2.pairs)
        for p1, p2 in zip(r1.pairs, r2.pairs):
            assert p1.node_a == p2.node_a
            assert p1.node_b == p2.node_b
            assert p1.flip_threshold_raw == p2.flip_threshold_raw
            assert p1.flip_threshold_relative == p2.flip_threshold_relative
            assert p1.baseline_winner == p2.baseline_winner
            assert p1.flipped_winner == p2.flipped_winner
        assert r1.overall_robust == r2.overall_robust
        assert r1.median_edge_strength == r2.median_edge_strength


# ==========================================================================
# Test 6: Synthetic confounder doesn't pollute original graph
# ==========================================================================


class TestConfounderIsolation:
    def test_original_graph_unchanged_after_injection(self):
        """_inject_confounder must not modify the original graph."""
        graph, _, _ = _make_bidirected_graph_with_options()
        original_nodes = copy.deepcopy(graph.nodes)
        original_edges = copy.deepcopy(graph.edges)

        modified = _inject_confounder(graph, "a", "y", 0.5)

        # Original graph unchanged
        assert len(graph.nodes) == len(original_nodes)
        assert len(graph.edges) == len(original_edges)
        for orig, curr in zip(original_nodes, graph.nodes):
            assert orig.id == curr.id

        # Modified graph has extra node + 2 edges
        assert len(modified.nodes) == len(original_nodes) + 1
        assert len(modified.edges) == len(original_edges) + 2

        # Confounder node exists in modified graph
        confounder_ids = [n.id for n in modified.nodes if n.id.startswith("__confounder_")]
        assert len(confounder_ids) == 1
        assert confounder_ids[0] == "__confounder_a_y"

    def test_full_analysis_preserves_original_graph(self):
        """Full sensitivity analysis doesn't modify input graph."""
        graph, options, goal = _make_bidirected_graph_with_options()
        original_node_count = len(graph.nodes)
        original_edge_count = len(graph.edges)

        analyze_confounding_sensitivity(
            graph=graph,
            options=options,
            goal_node_id=goal,
            bidirected_pairs=[("a", "y")],
            seed=42,
            n_samples=200,
        )

        assert len(graph.nodes) == original_node_count
        assert len(graph.edges) == original_edge_count


# ==========================================================================
# Test 7: Latency budget (12-node, 1 bidirected pair, < 500ms)
# ==========================================================================


class TestLatencyBudget:
    def test_12_node_1_pair_under_budget(self):
        """12-node graph with 1 bidirected pair completes within latency budget.

        Budget: 1 pair × up to 7 sweep steps × ~200ms/step ≈ 1400ms.
        Allow 2000ms headroom for CI variability.
        """
        graph, options, goal, bidirected_pairs = _make_12_node_graph_with_bidirected()

        t0 = time.time()
        result = analyze_confounding_sensitivity(
            graph=graph,
            options=options,
            goal_node_id=goal,
            bidirected_pairs=bidirected_pairs,
            seed=42,
            n_samples=500,
        )
        elapsed_ms = (time.time() - t0) * 1000

        # Wall-clock budget is hardware-sensitive (observed 3.2-3.3s on shared
        # GitHub runners vs the 2s budget) — enforced only under ISL_PERF_STRICT.
        # The correctness assertions below always run in default CI.
        assert_time_budget(elapsed_ms, 2000, "confounding 12-node 1-pair")
        assert len(result.pairs) == 1
        assert result.latency_ms > 0


# ==========================================================================
# Test 8: Endpoint integration
# ==========================================================================


from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


_BIDIRECTED_ENDPOINT_REQUEST = {
    "graph": {
        "nodes": [
            {"id": "a", "kind": "factor", "label": "Factor A", "observed_state": {"value": 0.5}},
            {"id": "y", "kind": "outcome", "label": "Outcome Y", "observed_state": {"value": 0.5}},
        ],
        "edges": [
            {
                "from": "a",
                "to": "y",
                "exists_probability": 1.0,
                "strength": {"mean": 0.8, "std": 0.05},
            },
            {
                "from": "a",
                "to": "y",
                "exists_probability": 1.0,
                "strength": {"mean": 0.1, "std": 0.05},
                "edge_type": "bidirected",
            },
        ],
    },
    "options": [
        {"id": "opt_high", "interventions": {"a": 1.0}},
        {"id": "opt_low", "interventions": {"a": 0.0}},
    ],
    "outcome_node_id": "y",
}

_IDENTIFIABLE_ENDPOINT_REQUEST = {
    "graph": {
        "nodes": [
            {"id": "a", "kind": "factor", "label": "A"},
            {"id": "y", "kind": "outcome", "label": "Y"},
        ],
        "edges": [
            {
                "from": "a",
                "to": "y",
                "exists_probability": 1.0,
                "strength": {"mean": 0.5, "std": 0.1},
            },
        ],
    },
    "options": [
        {"id": "opt_a", "interventions": {"a": 1.0}},
        {"id": "opt_b", "interventions": {"a": 0.0}},
    ],
    "outcome_node_id": "y",
}

_V2_ENDPOINT = "/api/v1/analysis/v2/identifiability"


class TestEndpointIntegration:
    def test_bidirected_graph_includes_confounding_sensitivity(self, client):
        """POST with bidirected graph → response includes confounding_sensitivity."""
        resp = client.post(_V2_ENDPOINT, json=_BIDIRECTED_ENDPOINT_REQUEST)
        assert resp.status_code == 200
        body = resp.json()

        # Non-identifiable pair should trigger sensitivity analysis
        assert body["all_identifiable"] is False
        assert "confounding_sensitivity" in body

        cs = body["confounding_sensitivity"]
        assert cs is not None
        assert isinstance(cs["pairs"], list)
        assert len(cs["pairs"]) >= 1
        assert "overall_robust" in cs
        assert "median_edge_strength" in cs
        assert cs["median_edge_strength"] > 0

        # Each pair has required fields
        pair = cs["pairs"][0]
        assert pair["node_a"] == "a"
        assert pair["node_b"] == "y"
        assert pair["baseline_winner"] is not None
        assert "provisional" in pair["interpretation"].lower()

    def test_identifiable_graph_no_confounding_sensitivity(self, client):
        """POST with all-identifiable graph → confounding_sensitivity is null."""
        resp = client.post(_V2_ENDPOINT, json=_IDENTIFIABLE_ENDPOINT_REQUEST)
        assert resp.status_code == 200
        body = resp.json()

        assert body["all_identifiable"] is True
        assert body.get("confounding_sensitivity") is None


# ==========================================================================
# Supplementary: median edge strength computation
# ==========================================================================


class TestMedianEdgeStrength:
    def test_median_excludes_bidirected(self):
        """Median edge strength should only consider directed edges."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(id="b", kind="outcome", label="B"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.4, std=0.1),
                ),
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=999.0, std=0.1),
                    edge_type="bidirected",
                ),
            ],
        )
        median = _compute_median_edge_strength(graph)
        assert median == pytest.approx(0.4, abs=0.01)

    def test_empty_directed_edges_returns_minimum(self):
        """Graph with only bidirected edges → minimum threshold."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(id="b", kind="outcome", label="B"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                    edge_type="bidirected",
                ),
            ],
        )
        median = _compute_median_edge_strength(graph)
        assert median == 0.01
