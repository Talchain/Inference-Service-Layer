"""
Integration tests for parameter recommendation endpoint.
"""

import pytest
from httpx import AsyncClient

from src.api.main import app
from src.models.shared import GraphV1


@pytest.fixture
def simple_graph():
    """Simple decision graph with critical path."""
    return GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch Product", "belief": 0.8},
            {"id": "n_outcome", "kind": "outcome", "label": "Market Success", "belief": 0.6},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_outcome", "weight": 2.0, "label": "Positive impact"}
        ]
    })


@pytest.fixture
def complex_graph():
    """Complex decision graph with multiple paths."""
    return GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch Product", "belief": 0.8},
            {"id": "n_mediator", "kind": "action", "label": "Marketing Campaign", "belief": 0.7},
            {"id": "n_risk", "kind": "risk", "label": "Market Competition", "belief": 0.4},
            {"id": "n_outcome", "kind": "outcome", "label": "Market Success", "belief": 0.6},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_mediator", "weight": 1.5},
            {"from": "n_mediator", "to": "n_outcome", "weight": 1.8},
            {"from": "n_risk", "to": "n_outcome", "weight": -0.5},
        ]
    })


@pytest.fixture
def risk_heavy_graph():
    """Graph with multiple risk nodes."""
    return GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Invest in R&D"},
            {"id": "n_risk1", "kind": "risk", "label": "Technology Risk"},
            {"id": "n_risk2", "kind": "risk", "label": "Market Risk"},
            {"id": "n_outcome", "kind": "outcome", "label": "Innovation Success"},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_outcome", "weight": 1.0},
            {"from": "n_risk1", "to": "n_outcome", "weight": -0.8},
            {"from": "n_risk2", "to": "n_outcome", "weight": -0.6},
        ]
    })


# ============================================================================
# Critical Path Weight Recommendations
# ============================================================================


@pytest.mark.asyncio
async def test_critical_path_edges_get_strong_weight_recommendations():
    """Critical path edges should get weight range 1.2-1.8."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch"},
            {"id": "n_outcome", "kind": "outcome", "label": "Success"},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_outcome", "weight": 0.5}  # Currently weak
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    # Find the critical edge recommendation
    weight_recs = [r for r in data["recommendations"] if r["parameter_type"] == "weight"]
    assert len(weight_recs) == 1

    rec = weight_recs[0]
    assert rec["recommended_range"][0] >= 1.2
    assert rec["recommended_range"][1] <= 1.8
    assert rec["confidence"] == "high"
    assert rec["importance"] >= 0.85


@pytest.mark.asyncio
async def test_peripheral_edges_get_moderate_weight_recommendations():
    """Non-critical edges should get lower weight ranges."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch"},
            {"id": "n_support", "kind": "action", "label": "Support Factor"},
            {"id": "n_outcome", "kind": "outcome", "label": "Success"},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_outcome", "weight": 1.5},  # Critical
            {"from": "n_support", "to": "n_outcome", "weight": 1.0},  # Peripheral
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    weight_recs = [r for r in data["recommendations"] if r["parameter_type"] == "weight"]
    assert len(weight_recs) == 2

    # Critical edge should have higher importance
    critical_rec = next(r for r in weight_recs if "n_decision_to_n_outcome" in r["parameter"])
    peripheral_rec = next(r for r in weight_recs if "n_support_to_n_outcome" in r["parameter"])

    assert critical_rec["importance"] > peripheral_rec["importance"]
    assert critical_rec["recommended_range"][0] > peripheral_rec["recommended_range"][0]


# ============================================================================
# Node Belief Recommendations
# ============================================================================


@pytest.mark.asyncio
async def test_treatment_nodes_get_high_certainty_recommendations():
    """Treatment/decision nodes should get 0.75-0.95 belief range."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch", "belief": 0.5},  # Too uncertain
            {"id": "n_outcome", "kind": "outcome", "label": "Success"},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_outcome"}
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    belief_recs = [r for r in data["recommendations"] if r["parameter_type"] == "belief"]
    decision_rec = next(r for r in belief_recs if "n_decision_belief" in r["parameter"])

    assert decision_rec["recommended_range"][0] >= 0.75
    assert decision_rec["recommended_range"][1] <= 0.95
    assert decision_rec["confidence"] == "high"


@pytest.mark.asyncio
async def test_outcome_nodes_get_high_certainty_recommendations():
    """Outcome nodes should get 0.75-0.95 belief range."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch"},
            {"id": "n_outcome", "kind": "outcome", "label": "Success", "belief": 0.3},  # Too uncertain
        ],
        "edges": [
            {"from": "n_decision", "to": "n_outcome"}
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    belief_recs = [r for r in data["recommendations"] if r["parameter_type"] == "belief"]
    outcome_rec = next(r for r in belief_recs if "n_outcome_belief" in r["parameter"])

    assert outcome_rec["recommended_range"][0] >= 0.75
    assert outcome_rec["recommended_range"][1] <= 0.95


@pytest.mark.asyncio
async def test_risk_nodes_get_moderate_uncertainty_recommendations():
    """Risk nodes should get 0.3-0.6 belief range."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch"},
            {"id": "n_risk", "kind": "risk", "label": "Market Risk", "belief": 0.9},  # Too certain for risk
            {"id": "n_outcome", "kind": "outcome", "label": "Success"},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_outcome"},
            {"from": "n_risk", "to": "n_outcome"},
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    belief_recs = [r for r in data["recommendations"] if r["parameter_type"] == "belief"]
    risk_rec = next(r for r in belief_recs if "n_risk_belief" in r["parameter"])

    assert risk_rec["recommended_range"][0] >= 0.3
    assert risk_rec["recommended_range"][1] <= 0.6
    assert "Risk factor" in risk_rec["rationale"]


@pytest.mark.asyncio
async def test_mediator_nodes_get_moderate_high_certainty():
    """Mediator nodes should get 0.65-0.85 belief range."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch"},
            {"id": "n_mediator", "kind": "action", "label": "Marketing", "belief": 0.5},
            {"id": "n_outcome", "kind": "outcome", "label": "Success"},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_mediator"},
            {"from": "n_mediator", "to": "n_outcome"},
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    belief_recs = [r for r in data["recommendations"] if r["parameter_type"] == "belief"]
    mediator_rec = next(r for r in belief_recs if "n_mediator_belief" in r["parameter"])

    assert mediator_rec["recommended_range"][0] >= 0.65
    assert mediator_rec["recommended_range"][1] <= 0.85
    assert "mediator" in mediator_rec["rationale"].lower()


# ============================================================================
# Importance Ranking
# ============================================================================


@pytest.mark.asyncio
async def test_recommendations_sorted_by_importance(complex_graph):
    """Recommendations should be sorted descending by importance."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": complex_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    importances = [r["importance"] for r in data["recommendations"]]
    assert importances == sorted(importances, reverse=True), "Recommendations not sorted by importance"


@pytest.mark.asyncio
async def test_critical_edges_ranked_highest(complex_graph):
    """Critical path parameters should be ranked highest."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": complex_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    # Critical edges: n_decision -> n_mediator -> n_outcome
    critical_params = [
        "n_decision_to_n_mediator_weight",
        "n_mediator_to_n_outcome_weight"
    ]

    for param in critical_params:
        rec = next(r for r in data["recommendations"] if r["parameter"] == param)
        assert rec["importance"] >= 0.7, f"{param} should have high importance"


# ============================================================================
# Rationale Quality
# ============================================================================


@pytest.mark.asyncio
async def test_recommendations_include_human_readable_rationale(simple_graph):
    """All recommendations should have clear, actionable rationale."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": simple_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    for rec in data["recommendations"]:
        assert len(rec["rationale"]) > 20, "Rationale too short"
        assert len(rec["rationale"]) <= 500, "Rationale exceeds max length"
        assert rec["rationale"][0].isupper(), "Rationale should start with capital letter"


@pytest.mark.asyncio
async def test_rationale_includes_node_labels(simple_graph):
    """Rationale should include human-readable node labels."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": simple_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    weight_rec = next(r for r in data["recommendations"] if r["parameter_type"] == "weight")
    assert "Launch Product" in weight_rec["rationale"] or "Market Success" in weight_rec["rationale"]


# ============================================================================
# Response Structure
# ============================================================================


@pytest.mark.asyncio
async def test_response_includes_graph_characteristics(simple_graph):
    """Response should include graph metadata."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": simple_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    assert "graph_characteristics" in data
    chars = data["graph_characteristics"]

    assert "num_critical_edges" in chars
    assert "max_path_length" in chars
    assert "avg_centrality" in chars
    assert "num_nodes" in chars
    assert "num_edges" in chars


@pytest.mark.asyncio
async def test_critical_edges_count_matches_graph(complex_graph):
    """Critical edge count should match actual critical paths."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": complex_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    # Complex graph has path: n_decision -> n_mediator -> n_outcome (2 edges)
    assert data["graph_characteristics"]["num_critical_edges"] == 2


# ============================================================================
# Current Value Comparison
# ============================================================================


@pytest.mark.asyncio
async def test_includes_current_values_when_provided():
    """If current_parameters provided, should include in response."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n_decision", "kind": "decision", "label": "Launch", "belief": 0.8},
            {"id": "n_outcome", "kind": "outcome", "label": "Success"},
        ],
        "edges": [
            {"from": "n_decision", "to": "n_outcome", "weight": 2.0}
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={
                "graph": graph.model_dump(by_alias=True),
                "current_parameters": {
                    "n_decision_to_n_outcome_weight": 2.0,
                    "n_decision_belief": 0.8
                }
            }
        )

    assert response.status_code == 200
    data = response.json()

    weight_rec = next(r for r in data["recommendations"] if r["parameter_type"] == "weight")
    belief_rec = next(r for r in data["recommendations"] if r["parameter_type"] == "belief")

    assert weight_rec["current_value"] == 2.0
    assert belief_rec["current_value"] == 0.8


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_handles_disconnected_graphs():
    """Should handle graphs with disconnected components."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n1", "kind": "decision", "label": "A"},
            {"id": "n2", "kind": "outcome", "label": "B"},
            {"id": "n3", "kind": "action", "label": "C"},
            {"id": "n4", "kind": "outcome", "label": "D"},
        ],
        "edges": [
            {"from": "n1", "to": "n2"},
            {"from": "n3", "to": "n4"},  # Disconnected from n1-n2
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) > 0


@pytest.mark.asyncio
async def test_handles_single_node_graph():
    """Should handle graph with single node."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n1", "kind": "decision", "label": "Solo", "belief": 0.5}
        ],
        "edges": []
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    # Should have 1 belief recommendation, no weight recommendations
    assert len([r for r in data["recommendations"] if r["parameter_type"] == "belief"]) == 1
    assert len([r for r in data["recommendations"] if r["parameter_type"] == "weight"]) == 0


@pytest.mark.asyncio
async def test_handles_complex_multi_path_graph(risk_heavy_graph):
    """Should handle graphs with multiple paths and risk nodes."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": risk_heavy_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    # Should have recommendations for all edges and all risk nodes
    weight_recs = [r for r in data["recommendations"] if r["parameter_type"] == "weight"]
    belief_recs = [r for r in data["recommendations"] if r["parameter_type"] == "belief"]

    assert len(weight_recs) == 3  # 3 edges
    assert len([r for r in belief_recs if "risk" in r["parameter"]]) == 2  # 2 risk nodes


# ============================================================================
# Performance
# ============================================================================


@pytest.mark.asyncio
async def test_response_time_under_2_seconds(complex_graph):
    """Endpoint should respond in <2 seconds for typical graphs."""
    import time

    async with AsyncClient(app=app, base_url="http://test") as client:
        start = time.time()
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": complex_graph.model_dump(by_alias=True)}
        )
        elapsed = time.time() - start

    assert response.status_code == 200
    assert elapsed < 2.0, f"Response took {elapsed:.2f}s (target: <2s)"


# ============================================================================
# Validation
# ============================================================================


@pytest.mark.asyncio
async def test_rejects_empty_graph():
    """Should reject graph with no nodes."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": {"nodes": [], "edges": []}}
        )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_recommended_typical_is_center_of_range(simple_graph):
    """Recommended typical value should be center of range."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": simple_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    for rec in data["recommendations"]:
        expected_typical = sum(rec["recommended_range"]) / 2
        assert abs(rec["recommended_typical"] - expected_typical) < 0.01


@pytest.mark.asyncio
async def test_all_confidence_levels_are_valid(complex_graph):
    """All recommendations should have valid confidence levels."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": complex_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    valid_confidence = {"high", "medium", "low"}
    for rec in data["recommendations"]:
        assert rec["confidence"] in valid_confidence, f"Invalid confidence: {rec['confidence']}"


@pytest.mark.asyncio
async def test_weight_ranges_are_valid():
    """Weight ranges should be within valid bounds."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n1", "kind": "decision", "label": "A"},
            {"id": "n2", "kind": "outcome", "label": "B"},
        ],
        "edges": [
            {"from": "n1", "to": "n2", "weight": 1.0}
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    weight_recs = [r for r in data["recommendations"] if r["parameter_type"] == "weight"]
    for rec in weight_recs:
        assert len(rec["recommended_range"]) == 2
        assert rec["recommended_range"][0] < rec["recommended_range"][1]
        # Weights typically range from 0.3 to 1.8 based on our logic
        assert rec["recommended_range"][0] >= 0.0
        assert rec["recommended_range"][1] <= 3.0


@pytest.mark.asyncio
async def test_belief_ranges_are_probabilities():
    """Belief ranges should be valid probabilities (0-1)."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n1", "kind": "decision", "label": "A", "belief": 0.5},
            {"id": "n2", "kind": "risk", "label": "Risk", "belief": 0.7},
        ],
        "edges": []
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/causal/parameter-recommendations",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    belief_recs = [r for r in data["recommendations"] if r["parameter_type"] == "belief"]
    for rec in belief_recs:
        assert len(rec["recommended_range"]) == 2
        assert rec["recommended_range"][0] >= 0.0, "Belief range min must be >= 0"
        assert rec["recommended_range"][1] <= 1.0, "Belief range max must be <= 1"
        assert rec["recommended_range"][0] < rec["recommended_range"][1], "Min must be < max"
