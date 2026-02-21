"""
Unit tests for V2 identifiability analysis (3B-prep).

Tests:
- V2→V1 graph translation (directed only)
- V2→V1 graph translation (with bidirected edges)
- Endpoint: identifiable graph
- Endpoint: non-identifiable due to bidirected edges
- Endpoint: frontdoor-eligible structure
- Backward compatibility (V1 tests still pass)
- Latency measurements
- Missing/malformed input
- Dropped V2 fields (parameter invariance)
"""

import time

import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    NodeV2,
    StrengthDistribution,
)
from src.services.identifiability_analyzer import (
    IdentifiabilityAnalyzer,
    IdentificationMethod,
    RecommendationStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_directed_graph() -> GraphV2:
    """Simple 3-node directed graph: X→Z→Y, X→Y."""
    return GraphV2(
        nodes=[
            NodeV2(id="x", kind="factor", label="Treatment"),
            NodeV2(id="z", kind="chance", label="Mediator"),
            NodeV2(id="y", kind="outcome", label="Outcome"),
        ],
        edges=[
            EdgeV2(
                **{"from": "x", "to": "z"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
            EdgeV2(
                **{"from": "z", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.6, std=0.1),
            ),
            EdgeV2(
                **{"from": "x", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.3, std=0.1),
            ),
        ],
    )


def _make_bidirected_graph() -> GraphV2:
    """Graph with directed X→Y and bidirected X↔Y (unmeasured confounder)."""
    return GraphV2(
        nodes=[
            NodeV2(id="x", kind="factor", label="Treatment"),
            NodeV2(id="y", kind="outcome", label="Outcome"),
        ],
        edges=[
            EdgeV2(
                **{"from": "x", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
            EdgeV2(
                **{"from": "x", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.1, std=0.1),
                edge_type="bidirected",
            ),
        ],
    )


def _make_frontdoor_graph() -> GraphV2:
    """Classic frontdoor: X→M→Y, with X↔Y bidirected (unmeasured confounder)."""
    return GraphV2(
        nodes=[
            NodeV2(id="x", kind="factor", label="Treatment"),
            NodeV2(id="m", kind="chance", label="Mediator"),
            NodeV2(id="y", kind="outcome", label="Outcome"),
        ],
        edges=[
            EdgeV2(
                **{"from": "x", "to": "m"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            ),
            EdgeV2(
                **{"from": "m", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.6, std=0.1),
            ),
            EdgeV2(
                **{"from": "x", "to": "y"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.1, std=0.1),
                edge_type="bidirected",
            ),
        ],
    )


@pytest.fixture
def analyzer():
    a = IdentifiabilityAnalyzer()
    a.clear_cache()
    return a


# ==========================================================================
# Test 1: V2→V1 translation — directed only
# ==========================================================================


class TestTranslationDirectedOnly:
    def test_directed_graph_translates_and_identifies(self, analyzer):
        """V2 graph with only directed edges → identifiable via backdoor."""
        graph = _make_directed_graph()
        result = analyzer.analyze_v2(graph, "x", "y")

        assert result.identifiable is True
        assert result.method in (
            IdentificationMethod.BACKDOOR,
            IdentificationMethod.DO_CALCULUS,
        )
        assert result.effect == "x → y"

    def test_all_nodes_preserved(self, analyzer):
        """All graph nodes are present in the analysis."""
        graph = _make_directed_graph()
        result = analyzer.analyze_v2(graph, "x", "y")
        # If identifiable, the adjustment set should only contain graph nodes
        if result.adjustment_set:
            node_ids = {n.id for n in graph.nodes}
            for adj in result.adjustment_set:
                assert adj in node_ids


# ==========================================================================
# Test 2: V2→V1 translation — with bidirected edges
# ==========================================================================


class TestTranslationBidirected:
    def test_bidirected_edges_become_undirected(self, analyzer):
        """Bidirected edge wired to Y₀'s undirected edge → non-identifiable."""
        graph = _make_bidirected_graph()
        result = analyzer.analyze_v2(graph, "x", "y")

        assert result.identifiable is False
        assert result.method == IdentificationMethod.NON_IDENTIFIABLE

    def test_bidirected_deduplication(self, analyzer):
        """A↔B and B↔A should be treated as the same bidirected edge."""
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
                ),
                # Two bidirected edges for the same pair — should deduplicate
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.1, std=0.1),
                    edge_type="bidirected",
                ),
                EdgeV2(
                    **{"from": "b", "to": "a"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.1, std=0.1),
                    edge_type="bidirected",
                ),
            ],
        )
        # Should not crash; should produce one result
        result = analyzer.analyze_v2(graph, "a", "b")
        assert result.identifiable is False


# ==========================================================================
# Test 3: Endpoint — identifiable graph
# ==========================================================================


class TestIdentifiableGraph:
    def test_simple_directed_identifiable(self, analyzer):
        """Simple directed graph → identifiable, backdoor method."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="price", kind="factor", label="Price"),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "price", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                ),
            ],
        )
        result = analyzer.analyze_v2(graph, "price", "revenue")

        assert result.identifiable is True
        assert result.method == IdentificationMethod.BACKDOOR
        assert result.recommendation_status == RecommendationStatus.ACTIONABLE


# ==========================================================================
# Test 4: Endpoint — non-identifiable due to bidirected edges
# ==========================================================================


class TestNonIdentifiableBidirected:
    def test_bidirected_makes_non_identifiable(self, analyzer):
        """Bidirected edge between treatment/outcome → non-identifiable."""
        graph = _make_bidirected_graph()
        result = analyzer.analyze_v2(graph, "x", "y")

        assert result.identifiable is False
        assert result.recommendation_status == RecommendationStatus.EXPLORATORY
        assert result.recommendation_caveat is not None


# ==========================================================================
# Test 5: Endpoint — frontdoor-eligible structure
# ==========================================================================


class TestFrontdoorStructure:
    def test_frontdoor_identifiable(self, analyzer):
        """Classic frontdoor (X→M→Y, X↔Y) → identifiable."""
        graph = _make_frontdoor_graph()
        result = analyzer.analyze_v2(graph, "x", "y")

        # The brief says: "Do not hard-require method: 'frontdoor'"
        # Assert identifiability and document the actual method.
        assert result.identifiable is True
        # Y₀ may report backdoor, frontdoor, or do_calculus depending on
        # its internal heuristics. Document the actual result.
        assert result.method in (
            IdentificationMethod.BACKDOOR,
            IdentificationMethod.FRONTDOOR,
            IdentificationMethod.DO_CALCULUS,
        )


# ==========================================================================
# Test 6: Backward compatibility
# ==========================================================================


class TestBackwardCompatibility:
    def test_v1_analyze_still_works(self, analyzer):
        """Existing V1 analyze() method is unaffected by V2 additions."""
        from src.models.shared import GraphEdgeV1, GraphNodeV1, GraphV1, NodeKind

        graph = GraphV1(
            nodes=[
                GraphNodeV1(id="decision", kind=NodeKind.DECISION, label="D"),
                GraphNodeV1(id="goal", kind=NodeKind.GOAL, label="G"),
            ],
            edges=[GraphEdgeV1(**{"from": "decision", "to": "goal"}, weight=0.5)],
        )
        result = analyzer.analyze(graph)

        assert result.identifiable is True
        assert result.method == IdentificationMethod.BACKDOOR

    def test_edge_type_none_behaves_as_directed(self, analyzer):
        """EdgeV2 with edge_type=None (default) behaves identically to directed."""
        graph_default = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(id="b", kind="outcome", label="B"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                    # edge_type intentionally omitted
                ),
            ],
        )
        graph_explicit = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(id="b", kind="outcome", label="B"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                    edge_type="directed",
                ),
            ],
        )
        r1 = analyzer.analyze_v2(graph_default, "a", "b")
        analyzer.clear_cache()
        r2 = analyzer.analyze_v2(graph_explicit, "a", "b")

        assert r1.identifiable == r2.identifiable
        assert r1.method == r2.method


# ==========================================================================
# Test 7: Latency
# ==========================================================================


class TestLatency:
    def _make_chain_graph(self, n_nodes: int) -> GraphV2:
        """Build a chain graph: node_0 → node_1 → ... → node_{n-1}."""
        nodes = [
            NodeV2(
                id=f"node_{i}",
                kind="factor" if i == 0 else ("outcome" if i == n_nodes - 1 else "chance"),
                label=f"Node {i}",
            )
            for i in range(n_nodes)
        ]
        edges = [
            EdgeV2(
                **{"from": f"node_{i}", "to": f"node_{i+1}"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            )
            for i in range(n_nodes - 1)
        ]
        return GraphV2(nodes=nodes, edges=edges)

    def test_latency_5_node(self, analyzer):
        graph = self._make_chain_graph(5)
        t0 = time.time()
        result = analyzer.analyze_v2(graph, "node_0", "node_4")
        elapsed_ms = (time.time() - t0) * 1000
        assert result.identifiable is True
        assert elapsed_ms < 500, f"5-node: {elapsed_ms:.1f}ms exceeds 500ms"

    def test_latency_12_node(self, analyzer):
        graph = self._make_chain_graph(12)
        t0 = time.time()
        result = analyzer.analyze_v2(graph, "node_0", "node_11")
        elapsed_ms = (time.time() - t0) * 1000
        assert result.identifiable is True
        assert elapsed_ms < 600, f"12-node: {elapsed_ms:.1f}ms exceeds 600ms"

    def test_latency_20_node(self, analyzer):
        graph = self._make_chain_graph(20)
        t0 = time.time()
        result = analyzer.analyze_v2(graph, "node_0", "node_19")
        elapsed_ms = (time.time() - t0) * 1000
        assert result.identifiable is True
        assert elapsed_ms < 1000, f"20-node: {elapsed_ms:.1f}ms exceeds 1000ms"


# ==========================================================================
# Test 8: Missing / malformed input
# ==========================================================================


class TestMalformedInput:
    def test_missing_treatment_node(self, analyzer):
        """Treatment node not in graph → no-nodes result."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="y", kind="outcome", label="Y"),
            ],
            edges=[],
        )
        result = analyzer.analyze_v2(graph, "missing_x", "y")
        assert result.identifiable is False

    def test_missing_outcome_node(self, analyzer):
        """Outcome node not in graph → no-nodes result."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="x", kind="factor", label="X"),
            ],
            edges=[],
        )
        result = analyzer.analyze_v2(graph, "x", "missing_y")
        assert result.identifiable is False

    def test_no_edges_no_path(self, analyzer):
        """Graph with nodes but no edges → no causal path."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="x", kind="factor", label="X"),
                NodeV2(id="y", kind="outcome", label="Y"),
            ],
            edges=[],
        )
        result = analyzer.analyze_v2(graph, "x", "y")
        assert result.identifiable is False
        assert result.method == IdentificationMethod.NON_IDENTIFIABLE

    def test_empty_graph_nodes(self):
        """Graph with no nodes → validation error."""
        with pytest.raises(Exception):
            GraphV2(nodes=[], edges=[])


# ==========================================================================
# Test 3b: Dropped V2 fields — parameter invariance
# ==========================================================================


class TestDroppedV2Fields:
    def test_different_parameters_same_structure_same_result(self, analyzer):
        """
        Graphs with identical structure but different parameter values
        should produce the same identifiability result.

        Intentionally dropped fields: exists_probability, strength.std,
        strength.mean, label.
        """
        graph_a = GraphV2(
            nodes=[
                NodeV2(id="x", kind="factor", label="Treatment A"),
                NodeV2(id="y", kind="outcome", label="Outcome A"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "x", "to": "y"},
                    exists_probability=0.3,
                    strength=StrengthDistribution(mean=0.1, std=0.01),
                    label="weak effect",
                ),
            ],
        )
        graph_b = GraphV2(
            nodes=[
                NodeV2(id="x", kind="factor", label="Treatment B"),
                NodeV2(id="y", kind="outcome", label="Outcome B"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "x", "to": "y"},
                    exists_probability=0.99,
                    strength=StrengthDistribution(mean=0.9, std=0.5),
                    label="strong effect",
                ),
            ],
        )

        r_a = analyzer.analyze_v2(graph_a, "x", "y")
        analyzer.clear_cache()
        r_b = analyzer.analyze_v2(graph_b, "x", "y")

        assert r_a.identifiable == r_b.identifiable
        assert r_a.method == r_b.method
        assert r_a.adjustment_set == r_b.adjustment_set


# ==========================================================================
# Endpoint-level tests via FastAPI TestClient
# ==========================================================================


from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


# JSON payloads (V2 format — dict, not Pydantic, for HTTP testing)

_SIMPLE_V2_REQUEST = {
    "graph": {
        "nodes": [
            {"id": "fac_price", "kind": "factor", "label": "Price"},
            {"id": "goal_revenue", "kind": "outcome", "label": "Revenue"},
        ],
        "edges": [
            {
                "from": "fac_price",
                "to": "goal_revenue",
                "exists_probability": 1.0,
                "strength": {"mean": 0.5, "std": 0.1},
            },
        ],
    },
    "options": [
        {"id": "opt_low", "interventions": {"fac_price": 80}},
        {"id": "opt_high", "interventions": {"fac_price": 120}},
    ],
    "outcome_node_id": "goal_revenue",
}

_BIDIRECTED_V2_REQUEST = {
    "graph": {
        "nodes": [
            {"id": "x", "kind": "factor", "label": "X"},
            {"id": "y", "kind": "outcome", "label": "Y"},
        ],
        "edges": [
            {
                "from": "x",
                "to": "y",
                "exists_probability": 1.0,
                "strength": {"mean": 0.5, "std": 0.1},
            },
            {
                "from": "x",
                "to": "y",
                "exists_probability": 1.0,
                "strength": {"mean": 0.1, "std": 0.1},
                "edge_type": "bidirected",
            },
        ],
    },
    "options": [{"id": "opt_a", "interventions": {"x": 1.0}}],
    "outcome_node_id": "y",
}

_FRONTDOOR_V2_REQUEST = {
    "graph": {
        "nodes": [
            {"id": "x", "kind": "factor", "label": "X"},
            {"id": "m", "kind": "chance", "label": "M"},
            {"id": "y", "kind": "outcome", "label": "Y"},
        ],
        "edges": [
            {
                "from": "x",
                "to": "m",
                "exists_probability": 1.0,
                "strength": {"mean": 0.5, "std": 0.1},
            },
            {
                "from": "m",
                "to": "y",
                "exists_probability": 1.0,
                "strength": {"mean": 0.6, "std": 0.1},
            },
            {
                "from": "x",
                "to": "y",
                "exists_probability": 1.0,
                "strength": {"mean": 0.1, "std": 0.1},
                "edge_type": "bidirected",
            },
        ],
    },
    "options": [{"id": "opt_a", "interventions": {"x": 1.0}}],
    "outcome_node_id": "y",
}

_V2_ENDPOINT = "/api/v1/analysis/v2/identifiability"


class TestEndpointIdentifiable:
    def test_identifiable_graph_returns_200(self, client):
        resp = client.post(_V2_ENDPOINT, json=_SIMPLE_V2_REQUEST)
        assert resp.status_code == 200
        body = resp.json()
        assert body["all_identifiable"] is True
        assert len(body["pairs"]) == 1
        pair = body["pairs"][0]
        assert pair["treatment_node_id"] == "fac_price"
        assert pair["outcome_node_id"] == "goal_revenue"
        assert pair["identifiable"] is True
        assert pair["method"] == "backdoor"

    def test_estimand_is_null_not_prose(self, client):
        """estimand should be None until symbolic estimand is threaded through."""
        resp = client.post(_V2_ENDPOINT, json=_SIMPLE_V2_REQUEST)
        assert resp.status_code == 200
        pair = resp.json()["pairs"][0]
        assert pair["estimand"] is None


class TestEndpointNonIdentifiable:
    def test_bidirected_returns_non_identifiable(self, client):
        resp = client.post(_V2_ENDPOINT, json=_BIDIRECTED_V2_REQUEST)
        assert resp.status_code == 200
        body = resp.json()
        assert body["all_identifiable"] is False
        pair = body["pairs"][0]
        assert pair["identifiable"] is False
        assert pair["method"] == "non_identifiable"


class TestEndpointFrontdoor:
    def test_frontdoor_eligible_identifiable(self, client):
        resp = client.post(_V2_ENDPOINT, json=_FRONTDOOR_V2_REQUEST)
        assert resp.status_code == 200
        body = resp.json()
        assert body["all_identifiable"] is True
        pair = body["pairs"][0]
        assert pair["identifiable"] is True
        # Do not hard-require method == "frontdoor"
        assert pair["method"] in ("backdoor", "frontdoor", "do_calculus")


class TestEndpointTreatmentExpansion:
    def test_multiple_options_deduplicate_treatments(self, client):
        """Two options sharing same treatment → one pair, not two."""
        req = {
            "graph": {
                "nodes": [
                    {"id": "price", "kind": "factor", "label": "Price"},
                    {"id": "revenue", "kind": "outcome", "label": "Revenue"},
                ],
                "edges": [
                    {
                        "from": "price",
                        "to": "revenue",
                        "exists_probability": 1.0,
                        "strength": {"mean": 0.5, "std": 0.1},
                    },
                ],
            },
            "options": [
                {"id": "low", "interventions": {"price": 80}},
                {"id": "high", "interventions": {"price": 120}},
            ],
            "outcome_node_id": "revenue",
        }
        resp = client.post(_V2_ENDPOINT, json=req)
        assert resp.status_code == 200
        # "price" appears in both options but should produce only 1 pair
        assert len(resp.json()["pairs"]) == 1

    def test_multiple_treatment_factors(self, client):
        """Two options with different treatments → two pairs."""
        req = {
            "graph": {
                "nodes": [
                    {"id": "price", "kind": "factor", "label": "Price"},
                    {"id": "quality", "kind": "factor", "label": "Quality"},
                    {"id": "revenue", "kind": "outcome", "label": "Revenue"},
                ],
                "edges": [
                    {
                        "from": "price",
                        "to": "revenue",
                        "exists_probability": 1.0,
                        "strength": {"mean": 0.5, "std": 0.1},
                    },
                    {
                        "from": "quality",
                        "to": "revenue",
                        "exists_probability": 1.0,
                        "strength": {"mean": 0.3, "std": 0.1},
                    },
                ],
            },
            "options": [
                {"id": "opt_a", "interventions": {"price": 80}},
                {"id": "opt_b", "interventions": {"quality": 0.9}},
            ],
            "outcome_node_id": "revenue",
        }
        resp = client.post(_V2_ENDPOINT, json=req)
        assert resp.status_code == 200
        pairs = resp.json()["pairs"]
        assert len(pairs) == 2
        treatment_ids = {p["treatment_node_id"] for p in pairs}
        assert treatment_ids == {"price", "quality"}


class TestEndpointMalformedInput:
    def test_missing_outcome_node_returns_400(self, client):
        req = {
            "graph": {
                "nodes": [
                    {"id": "x", "kind": "factor", "label": "X"},
                ],
                "edges": [],
            },
            "options": [{"id": "a", "interventions": {"x": 1}}],
            "outcome_node_id": "missing_y",
        }
        resp = client.post(_V2_ENDPOINT, json=req)
        assert resp.status_code == 400
        assert "missing_y" in resp.json()["message"]

    def test_missing_treatment_node_returns_400(self, client):
        req = {
            "graph": {
                "nodes": [
                    {"id": "y", "kind": "outcome", "label": "Y"},
                ],
                "edges": [],
            },
            "options": [{"id": "a", "interventions": {"missing_x": 1}}],
            "outcome_node_id": "y",
        }
        resp = client.post(_V2_ENDPOINT, json=req)
        assert resp.status_code == 400
        assert "missing_x" in resp.json()["message"]

    def test_empty_interventions_returns_400(self, client):
        req = {
            "graph": {
                "nodes": [
                    {"id": "y", "kind": "outcome", "label": "Y"},
                ],
                "edges": [],
            },
            "options": [{"id": "a", "interventions": {}}],
            "outcome_node_id": "y",
        }
        resp = client.post(_V2_ENDPOINT, json=req)
        assert resp.status_code == 400
        assert "empty interventions" in resp.json()["message"].lower()

    def test_missing_graph_returns_422(self, client):
        """Missing required field → Pydantic 422."""
        resp = client.post(_V2_ENDPOINT, json={"options": [], "outcome_node_id": "y"})
        assert resp.status_code == 422

    def test_empty_options_returns_422(self, client):
        """options must have min_length=1."""
        req = {
            "graph": {
                "nodes": [{"id": "y", "kind": "outcome", "label": "Y"}],
                "edges": [],
            },
            "options": [],
            "outcome_node_id": "y",
        }
        resp = client.post(_V2_ENDPOINT, json=req)
        assert resp.status_code == 422


class TestEndpointLatency:
    def test_endpoint_latency_12_node(self, client):
        """Full endpoint path (parse + validate + analyze) for 12-node graph."""
        nodes = [
            {"id": f"n{i}", "kind": "factor" if i == 0 else ("outcome" if i == 11 else "chance"), "label": f"N{i}"}
            for i in range(12)
        ]
        edges = [
            {"from": f"n{i}", "to": f"n{i+1}", "exists_probability": 1.0, "strength": {"mean": 0.5, "std": 0.1}}
            for i in range(11)
        ]
        req = {
            "graph": {"nodes": nodes, "edges": edges},
            "options": [{"id": "opt_a", "interventions": {"n0": 1.0}}],
            "outcome_node_id": "n11",
        }
        import time

        t0 = time.time()
        resp = client.post(_V2_ENDPOINT, json=req)
        elapsed_ms = (time.time() - t0) * 1000
        assert resp.status_code == 200
        assert elapsed_ms < 600, f"12-node endpoint: {elapsed_ms:.1f}ms exceeds 600ms"
