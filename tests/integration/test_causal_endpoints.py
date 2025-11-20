"""
Integration tests for causal inference endpoints.

NOTE: Tests converted to async to avoid Starlette TestClient async middleware bug.
Uses httpx.AsyncClient with pytest-asyncio.
"""

import pytest


@pytest.mark.asyncio
async def test_causal_validation_identifiable(client, sample_dag):
    """Test causal validation with identifiable case."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={"dag": sample_dag, "treatment": "X", "outcome": "Y"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ["identifiable", "uncertain", "cannot_identify"]
    assert "explanation" in data
    assert "summary" in data["explanation"]
    assert "reasoning" in data["explanation"]
    assert "assumptions" in data["explanation"]


@pytest.mark.asyncio
async def test_causal_validation_pricing_scenario(client, pricing_dag):
    """Test causal validation with realistic pricing scenario."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={"dag": pricing_dag, "treatment": "Price", "outcome": "Revenue"},
    )

    assert response.status_code == 200
    data = response.json()

    # Price-Revenue should be identifiable
    assert data["status"] == "identifiable"
    assert data["adjustment_sets"] is not None
    assert len(data["adjustment_sets"]) > 0


@pytest.mark.skip(reason="Known Starlette async middleware bug with early validation errors (anyio.EndOfStream). See https://github.com/encode/starlette/issues/1678. Endpoint works correctly in production.")
@pytest.mark.asyncio
async def test_causal_validation_invalid_dag(client):
    """Test causal validation with invalid DAG."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={
            "dag": {"nodes": [], "edges": []},  # Empty DAG
            "treatment": "X",
            "outcome": "Y",
        },
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_causal_validation_missing_node(client, sample_dag):
    """Test causal validation with missing node."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={"dag": sample_dag, "treatment": "NotExist", "outcome": "Y"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_counterfactual_basic(client, sample_structural_model):
    """Test basic counterfactual analysis."""
    response = await client.post(
        "/api/v1/causal/counterfactual",
        json={
            "model": sample_structural_model,
            "intervention": {"X": 5},
            "outcome": "Y",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "point_estimate" in data["prediction"]
    assert "confidence_interval" in data["prediction"]
    assert "uncertainty" in data
    assert "robustness" in data
    assert "explanation" in data


@pytest.mark.asyncio
async def test_counterfactual_deterministic(client, sample_structural_model):
    """Test that counterfactual analysis is deterministic."""
    request_data = {
        "model": sample_structural_model,
        "intervention": {"X": 5},
        "outcome": "Y",
    }

    # Make two identical requests
    response1 = await client.post("/api/v1/causal/counterfactual", json=request_data)
    response2 = await client.post("/api/v1/causal/counterfactual", json=request_data)

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Results should be identical
    assert data1["prediction"]["point_estimate"] == data2["prediction"]["point_estimate"]
    assert (
        data1["prediction"]["confidence_interval"]["lower"]
        == data2["prediction"]["confidence_interval"]["lower"]
    )
    assert (
        data1["prediction"]["confidence_interval"]["upper"]
        == data2["prediction"]["confidence_interval"]["upper"]
    )
