"""
Integration tests for batch counterfactual endpoints.

NOTE: Tests converted to async to avoid Starlette TestClient async middleware bug.
Uses httpx.AsyncClient with pytest-asyncio.
"""

import pytest


@pytest.mark.asyncio
async def test_batch_counterfactual_basic(client):
    """Test basic batch counterfactual request."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "scenarios": [
                {"id": "baseline", "intervention": {"Price": 40}, "label": "Current pricing"},
                {"id": "increase", "intervention": {"Price": 50}, "label": "10% increase"},
                {"id": "aggressive", "intervention": {"Price": 60}, "label": "20% increase"},
            ],
            "outcome": "Revenue",
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "scenarios" in data
    assert "comparison" in data
    assert "explanation" in data

    # Check scenarios
    assert len(data["scenarios"]) == 3
    for scenario in data["scenarios"]:
        assert "scenario_id" in scenario
        assert "intervention" in scenario
        assert "prediction" in scenario
        assert "uncertainty" in scenario
        assert "robustness" in scenario

    # Check predictions are ordered
    baseline_rev = data["scenarios"][0]["prediction"]["point_estimate"]
    increase_rev = data["scenarios"][1]["prediction"]["point_estimate"]
    aggressive_rev = data["scenarios"][2]["prediction"]["point_estimate"]

    assert increase_rev > baseline_rev
    assert aggressive_rev > increase_rev


@pytest.mark.asyncio
async def test_batch_counterfactual_with_interactions(client):
    """Test batch counterfactual with interaction detection."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Quality", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price + 200*Quality + 100*Price*Quality"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 10}}
                },
            },
            "scenarios": [
                {"id": "price_only", "intervention": {"Price": 2}},
                {"id": "quality_only", "intervention": {"Quality": 3}},
                {"id": "both", "intervention": {"Price": 2, "Quality": 3}},
            ],
            "outcome": "Revenue",
            "analyze_interactions": True,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check interactions
    assert "interactions" in data
    assert data["interactions"] is not None
    assert "pairwise" in data["interactions"]
    assert "summary" in data["interactions"]

    # Should detect Price-Quality interaction
    assert len(data["interactions"]["pairwise"]) > 0

    # Find Price-Quality interaction
    price_quality = next(
        (i for i in data["interactions"]["pairwise"]
         if set(i["variables"]) == {"Price", "Quality"}),
        None
    )

    assert price_quality is not None
    assert "type" in price_quality
    assert price_quality["type"] in ["synergistic", "antagonistic", "additive"]


@pytest.mark.asyncio
async def test_batch_counterfactual_deterministic(client):
    """Test that batch counterfactual is deterministic with seed."""
    request_data = {
        "model": {
            "variables": ["Price", "Revenue"],
            "equations": {"Revenue": "10000 + 500*Price"},
            "distributions": {
                "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
            },
        },
        "scenarios": [
            {"id": "baseline", "intervention": {"Price": 40}},
            {"id": "increase", "intervention": {"Price": 50}},
        ],
        "outcome": "Revenue",
        "seed": 42,
    }

    # Make two identical requests
    response1 = await client.post("/api/v1/causal/counterfactual/batch", json=request_data)
    response2 = await client.post("/api/v1/causal/counterfactual/batch", json=request_data)

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Same results
    for s1, s2 in zip(data1["scenarios"], data2["scenarios"]):
        assert abs(
            s1["prediction"]["point_estimate"] - s2["prediction"]["point_estimate"]
        ) < 0.01


@pytest.mark.asyncio
async def test_batch_counterfactual_comparison(client):
    """Test scenario comparison and ranking."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 10}}
                },
            },
            "scenarios": [
                {"id": "baseline", "intervention": {"Price": 40}},
                {"id": "increase", "intervention": {"Price": 50}},
                {"id": "aggressive", "intervention": {"Price": 60}},
            ],
            "outcome": "Revenue",
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check comparison
    comparison = data["comparison"]
    assert "best_outcome" in comparison
    assert "most_robust" in comparison
    assert "marginal_gains" in comparison
    assert "ranking" in comparison

    # Best outcome should be aggressive (highest price)
    assert comparison["best_outcome"] == "aggressive"

    # Ranking should be in descending order
    assert len(comparison["ranking"]) == 3
    assert comparison["ranking"][0] == "aggressive"

    # Marginal gains should be present for non-baseline scenarios
    assert "increase" in comparison["marginal_gains"]
    assert "aggressive" in comparison["marginal_gains"]


@pytest.mark.asyncio
async def test_batch_counterfactual_no_interactions(client):
    """Test that interactions can be disabled."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Quality", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price + 200*Quality"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 10}}
                },
            },
            "scenarios": [
                {"id": "price_only", "intervention": {"Price": 2}},
                {"id": "quality_only", "intervention": {"Quality": 3}},
                {"id": "both", "intervention": {"Price": 2, "Quality": 3}},
            ],
            "outcome": "Revenue",
            "analyze_interactions": False,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Interactions should be None when disabled
    assert data["interactions"] is None


@pytest.mark.asyncio
async def test_batch_counterfactual_with_labels(client):
    """Test that scenario labels are preserved."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 10}}
                },
            },
            "scenarios": [
                {
                    "id": "baseline",
                    "intervention": {"Price": 40},
                    "label": "Current pricing strategy"
                },
                {
                    "id": "increase",
                    "intervention": {"Price": 50},
                    "label": "10% price increase"
                },
            ],
            "outcome": "Revenue",
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Labels should be preserved
    assert data["scenarios"][0]["label"] == "Current pricing strategy"
    assert data["scenarios"][1]["label"] == "10% price increase"


@pytest.mark.asyncio
async def test_batch_counterfactual_metadata(client):
    """Test that response includes metadata."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 10}}
                },
            },
            "scenarios": [
                {"id": "baseline", "intervention": {"Price": 40}},
                {"id": "increase", "intervention": {"Price": 50}},
            ],
            "outcome": "Revenue",
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check metadata exists
    assert "_metadata" in data or "metadata" in data


@pytest.mark.asyncio
async def test_batch_counterfactual_explanation_quality(client):
    """Test that explanation is informative."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 10}}
                },
            },
            "scenarios": [
                {"id": "baseline", "intervention": {"Price": 40}},
                {"id": "best", "intervention": {"Price": 60}},
            ],
            "outcome": "Revenue",
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    explanation = data["explanation"]
    assert "summary" in explanation
    assert "reasoning" in explanation
    assert "technical_basis" in explanation
    assert "assumptions" in explanation

    # Summary should mention best scenario
    assert len(explanation["summary"]) > 0
    assert "best" in explanation["summary"].lower()


@pytest.mark.asyncio
async def test_batch_counterfactual_invalid_request(client):
    """Test handling of invalid request (Pydantic validation)."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": [],
                "equations": {},
                "distributions": {},
            },
            "scenarios": [
                # Only one scenario (should fail validation - min_length=2)
                {"id": "only_one", "intervention": {"Price": 40}},
            ],
            "outcome": "Revenue",
        },
    )

    # Should return validation error (422 for Pydantic validation)
    assert response.status_code in [400, 422]
    data = response.json()
    assert "code" in data or "detail" in data  # Error response


@pytest.mark.asyncio
async def test_batch_counterfactual_complex_model(client):
    """Test batch counterfactual with complex multi-variable model."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Quality", "Marketing", "Brand", "Revenue"],
                "equations": {
                    "Brand": "50 + 0.3*Quality - 0.1*Price",
                    "Revenue": "10000 + 800*Price + 200*Quality + 0.5*Marketing + 300*Brand",
                },
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 500}}
                },
            },
            "scenarios": [
                {
                    "id": "baseline",
                    "intervention": {"Price": 40},
                    "label": "Current state"
                },
                {
                    "id": "premium",
                    "intervention": {"Price": 60, "Quality": 9},
                    "label": "Premium positioning"
                },
                {
                    "id": "aggressive",
                    "intervention": {"Price": 50, "Marketing": 100000},
                    "label": "Aggressive marketing"
                },
            ],
            "outcome": "Revenue",
            "analyze_interactions": True,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should process all scenarios
    assert len(data["scenarios"]) == 3

    # Should have comparison
    assert "comparison" in data
    assert len(data["comparison"]["ranking"]) == 3


@pytest.mark.asyncio
async def test_batch_counterfactual_custom_samples(client):
    """Test batch counterfactual with custom sample count."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "scenarios": [
                {"id": "baseline", "intervention": {"Price": 40}},
                {"id": "increase", "intervention": {"Price": 50}},
            ],
            "outcome": "Revenue",
            "samples": 500,  # Custom sample count
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should complete successfully with custom sample count
    assert len(data["scenarios"]) == 2


@pytest.mark.asyncio
async def test_batch_counterfactual_multiple_interventions(client):
    """Test batch with varied intervention combinations."""
    response = await client.post(
        "/api/v1/causal/counterfactual/batch",
        json={
            "model": {
                "variables": ["Price", "Quality", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price + 200*Quality"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 50}}
                },
            },
            "scenarios": [
                {"id": "baseline", "intervention": {"Price": 40, "Quality": 7}},
                {"id": "price_up", "intervention": {"Price": 50, "Quality": 7}},
                {"id": "quality_up", "intervention": {"Price": 40, "Quality": 9}},
                {"id": "both_up", "intervention": {"Price": 50, "Quality": 9}},
            ],
            "outcome": "Revenue",
            "analyze_interactions": True,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should process all scenarios
    assert len(data["scenarios"]) == 4

    # Should detect Price-Quality interaction
    if data["interactions"]:
        assert len(data["interactions"]["pairwise"]) > 0
