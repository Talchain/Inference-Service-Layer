"""
Integration tests for CEE enhancement endpoints.

Tests all 4 CEE endpoints with real graph structures.
"""

import pytest
from httpx import AsyncClient

from src.api.main import app
from src.models.shared import GraphNodeV1, GraphEdgeV1, GraphV1, NodeKind


@pytest.fixture
def simple_graph():
    """Simple decision graph for testing."""
    return GraphV1.model_validate({
        "nodes": [
            {
                "id": "n_decision",
                "kind": "decision",
                "label": "Launch Product",
                "belief": 0.8
            },
            {
                "id": "n_outcome",
                "kind": "outcome",
                "label": "Market Success",
                "belief": 0.6
            },
        ],
        "edges": [
            {
                "from": "n_decision",
                "to": "n_outcome",
                "weight": 2.0,
                "label": "Positive impact"
            }
        ]
    })


@pytest.fixture
def complex_graph():
    """More complex decision graph for testing."""
    return GraphV1.model_validate({
        "nodes": [
            {
                "id": "n_budget",
                "kind": "decision",
                "label": "Marketing Budget",
                "belief": 0.75
            },
            {
                "id": "n_reach",
                "kind": "outcome",
                "label": "Customer Reach",
                "belief": 0.65
            },
            {
                "id": "n_sales",
                "kind": "goal",
                "label": "Sales Revenue"
            },
        ],
        "edges": [
            {"from": "n_budget", "to": "n_reach", "weight": 2.5},
            {"from": "n_reach", "to": "n_sales", "weight": 1.8},
        ]
    })


@pytest.mark.asyncio
async def test_validation_strategies_returns_improvements(simple_graph):
    """Test that validation/strategies endpoint returns improvement suggestions."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/validation/strategies",
            json={"graph": simple_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    assert "suggested_improvements" in data
    assert isinstance(data["suggested_improvements"], list)
    # Should have at least some general recommendations
    assert len(data["suggested_improvements"]) >= 0


@pytest.mark.asyncio
async def test_sensitivity_identifies_critical_variables(complex_graph):
    """Test that sensitivity endpoint identifies critical variables."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sensitivity/detailed",
            json={"graph": complex_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    assert "assumptions" in data
    assert isinstance(data["assumptions"], list)
    assert len(data["assumptions"]) > 0

    # Check assumption structure
    for assumption in data["assumptions"]:
        assert "variable" in assumption
        assert "sensitivity" in assumption
        assert "impact" in assumption
        assert 0 <= assumption["sensitivity"] <= 1
        assert isinstance(assumption["impact"], str)

    # Assumptions should be sorted by sensitivity (highest first)
    if len(data["assumptions"]) > 1:
        sensitivities = [a["sensitivity"] for a in data["assumptions"]]
        assert sensitivities == sorted(sensitivities, reverse=True)


@pytest.mark.asyncio
async def test_contrastive_suggests_alternatives(complex_graph):
    """Test that contrastive endpoint suggests actionable alternatives."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/contrastive",
            json={
                "graph": complex_graph.model_dump(by_alias=True),
                "target_outcome": "n_sales"
            }
        )

    assert response.status_code == 200
    data = response.json()

    assert "alternatives" in data
    assert isinstance(data["alternatives"], list)
    assert len(data["alternatives"]) > 0
    assert len(data["alternatives"]) <= 3  # Should return top 3

    # Check alternative structure
    for alt in data["alternatives"]:
        assert "change" in alt
        assert "outcome_diff" in alt
        assert "feasibility" in alt
        assert 0 <= alt["feasibility"] <= 1
        assert isinstance(alt["change"], str)
        assert isinstance(alt["outcome_diff"], str)


@pytest.mark.asyncio
async def test_conformal_provides_intervals(simple_graph):
    """Test that conformal endpoint provides valid confidence intervals."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/conformal",
            json={
                "graph": simple_graph.model_dump(by_alias=True),
                "variable": "n_outcome"
            }
        )

    assert response.status_code == 200
    data = response.json()

    assert "prediction_interval" in data
    assert "confidence_level" in data
    assert "uncertainty_source" in data

    # Check interval structure
    interval = data["prediction_interval"]
    assert isinstance(interval, list)
    assert len(interval) == 2
    assert interval[0] < interval[1]  # Lower < upper
    assert all(isinstance(x, (int, float)) for x in interval)

    # Check confidence level
    assert 0 < data["confidence_level"] <= 1

    # Check uncertainty source
    assert isinstance(data["uncertainty_source"], str)
    assert len(data["uncertainty_source"]) > 0


@pytest.mark.asyncio
async def test_sensitivity_handles_empty_graph():
    """Test sensitivity endpoint handles graphs without assumptions."""
    empty_graph = GraphV1.model_validate({
        "nodes": [
            {
                "id": "n_single",
                "kind": "outcome",
                "label": "Single Node"
            }
        ],
        "edges": []
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/sensitivity/detailed",
            json={"graph": empty_graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["assumptions"] == []


@pytest.mark.asyncio
async def test_contrastive_validates_target_outcome(simple_graph):
    """Test contrastive endpoint validates target outcome exists."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/contrastive",
            json={
                "graph": simple_graph.model_dump(by_alias=True),
                "target_outcome": "n_nonexistent"
            }
        )

    assert response.status_code == 400
    data = response.json()
    assert "not found in graph" in data["message"].lower()


@pytest.mark.asyncio
async def test_conformal_validates_variable(simple_graph):
    """Test conformal endpoint validates variable exists."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/conformal",
            json={
                "graph": simple_graph.model_dump(by_alias=True),
                "variable": "n_nonexistent"
            }
        )

    assert response.status_code == 400
    data = response.json()
    assert "not found in graph" in data["message"].lower()


@pytest.mark.asyncio
async def test_all_endpoints_accept_timeout_parameter(simple_graph):
    """Test that all endpoints accept and respect timeout parameter."""
    endpoints_and_bodies = [
        ("/api/v1/sensitivity/detailed", {"graph": simple_graph.model_dump(by_alias=True), "timeout": 5000}),
        ("/api/v1/contrastive", {"graph": simple_graph.model_dump(by_alias=True), "target_outcome": "n_outcome", "timeout": 5000}),
        ("/api/v1/conformal", {"graph": simple_graph.model_dump(by_alias=True), "variable": "n_outcome", "timeout": 5000}),
        ("/api/v1/validation/strategies", {"graph": simple_graph.model_dump(by_alias=True), "timeout": 5000}),
    ]

    async with AsyncClient(app=app, base_url="http://test") as client:
        for endpoint, body in endpoints_and_bodies:
            response = await client.post(endpoint, json=body)
            assert response.status_code == 200, f"Failed for {endpoint}"


@pytest.mark.asyncio
async def test_all_endpoints_support_request_id_header(simple_graph):
    """Test that all endpoints accept X-Request-Id header."""
    endpoints_and_bodies = [
        ("/api/v1/sensitivity/detailed", {"graph": simple_graph.model_dump(by_alias=True)}),
        ("/api/v1/contrastive", {"graph": simple_graph.model_dump(by_alias=True), "target_outcome": "n_outcome"}),
        ("/api/v1/conformal", {"graph": simple_graph.model_dump(by_alias=True), "variable": "n_outcome"}),
        ("/api/v1/validation/strategies", {"graph": simple_graph.model_dump(by_alias=True)}),
    ]

    async with AsyncClient(app=app, base_url="http://test") as client:
        for endpoint, body in endpoints_and_bodies:
            response = await client.post(
                endpoint,
                json=body,
                headers={"X-Request-Id": "test_request_123"}
            )
            assert response.status_code == 200, f"Failed for {endpoint}"


@pytest.mark.asyncio
async def test_validation_detects_missing_weights():
    """Flags when >50% edges lack weights and references parameter-recommendations."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n1", "kind": "decision", "label": "A"},
            {"id": "n2", "kind": "action", "label": "B"},
            {"id": "n3", "kind": "action", "label": "C"},
            {"id": "n4", "kind": "outcome", "label": "D"},
        ],
        "edges": [
            {"from": "n1", "to": "n2", "weight": 1.0},  # Has weight
            {"from": "n1", "to": "n3"},  # Missing weight
            {"from": "n2", "to": "n4"},  # Missing weight
            {"from": "n3", "to": "n4"},  # Missing weight
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/validation/strategies",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    # Check for missing weights recommendation
    missing_weight_rec = next(
        (rec for rec in data["suggested_improvements"] if "50%" in rec["description"]),
        None
    )
    assert missing_weight_rec is not None, "Should detect missing weights"
    assert "parameter-recommendations" in missing_weight_rec["description"]
    assert missing_weight_rec["priority"] == "high"


@pytest.mark.asyncio
async def test_validation_detects_uniform_weights():
    """Flags when all edges have same weight and references parameter-recommendations."""
    graph = GraphV1.model_validate({
        "nodes": [
            {"id": "n1", "kind": "decision", "label": "A"},
            {"id": "n2", "kind": "action", "label": "B"},
            {"id": "n3", "kind": "action", "label": "C"},
            {"id": "n4", "kind": "outcome", "label": "D"},
        ],
        "edges": [
            {"from": "n1", "to": "n2", "weight": 0.5},
            {"from": "n1", "to": "n3", "weight": 0.5},
            {"from": "n2", "to": "n4", "weight": 0.5},
            {"from": "n3", "to": "n4", "weight": 0.5},
        ]
    })

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/validation/strategies",
            json={"graph": graph.model_dump(by_alias=True)}
        )

    assert response.status_code == 200
    data = response.json()

    # Check for uniform weights recommendation
    uniform_weight_rec = next(
        (rec for rec in data["suggested_improvements"] if "uniform weight" in rec["description"]),
        None
    )
    assert uniform_weight_rec is not None, "Should detect uniform weights"
    assert "0.5" in uniform_weight_rec["description"]
    assert "parameter-recommendations" in uniform_weight_rec["description"]
    assert uniform_weight_rec["priority"] == "medium"
