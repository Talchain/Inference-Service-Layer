"""
Integration tests for contrastive explanation endpoints.

NOTE: Tests converted to async to avoid Starlette TestClient async middleware bug.
Uses httpx.AsyncClient with pytest-asyncio.
"""

import pytest


@pytest.mark.asyncio
async def test_contrastive_explanation_basic(client):
    """Test basic contrastive explanation request."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500 * Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "current_state": {"Price": 40},
            "observed_outcome": {"Revenue": 30000},
            "target_outcome": {"Revenue": (35000, 36000)},
            "constraints": {
                "feasible": ["Price"],
                "max_changes": 1,
                "minimize": "change_magnitude",
            },
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "minimal_interventions" in data
    assert "comparison" in data
    assert "explanation" in data

    # Check interventions
    assert len(data["minimal_interventions"]) > 0
    intervention = data["minimal_interventions"][0]

    assert "rank" in intervention
    assert "changes" in intervention
    assert "expected_outcome" in intervention
    assert "confidence_interval" in intervention
    assert "feasibility" in intervention
    assert "cost_estimate" in intervention
    assert "robustness" in intervention

    # Check that Price was changed
    assert "Price" in intervention["changes"]
    change = intervention["changes"]["Price"]
    assert "from_value" in change
    assert "to_value" in change
    assert "delta" in change
    assert "relative_change" in change

    # Check expected outcome is in target range
    assert 35000 <= intervention["expected_outcome"]["Revenue"] <= 36000


@pytest.mark.asyncio
async def test_contrastive_explanation_multiple_variables(client):
    """Test contrastive explanation with multiple feasible variables."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Marketing", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price + 0.5*Marketing"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "current_state": {"Price": 40, "Marketing": 20000},
            "observed_outcome": {"Revenue": 40000},
            "target_outcome": {"Revenue": (45000, 46000)},
            "constraints": {
                "feasible": ["Price", "Marketing"],
                "max_changes": 1,
                "minimize": "change_magnitude",
            },
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should find interventions for different variables
    assert len(data["minimal_interventions"]) > 0

    # All interventions should achieve target
    for intervention in data["minimal_interventions"]:
        assert 45000 <= intervention["expected_outcome"]["Revenue"] <= 46000


@pytest.mark.asyncio
async def test_contrastive_explanation_multi_variable_combinations(client):
    """Test finding multi-variable combinations."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Quality", "Revenue"],
                "equations": {"Revenue": "10000 + 300*Price + 200*Quality"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "current_state": {"Price": 40, "Quality": 7.5},
            "observed_outcome": {"Revenue": 23500},
            "target_outcome": {"Revenue": (30000, 31000)},
            "constraints": {
                "feasible": ["Price", "Quality"],
                "max_changes": 2,
                "minimize": "change_magnitude",
                "variable_bounds": {
                    "Price": (35, 60),
                    "Quality": (6, 10),
                },
            },
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should find interventions
    assert len(data["minimal_interventions"]) > 0

    # Check comparison
    assert "best_by_cost" in data["comparison"]
    assert "best_by_robustness" in data["comparison"]
    assert "best_by_feasibility" in data["comparison"]


@pytest.mark.asyncio
async def test_contrastive_explanation_deterministic(client):
    """Test that contrastive explanation is deterministic with seed."""
    request_data = {
        "model": {
            "variables": ["Price", "Revenue"],
            "equations": {"Revenue": "10000 + 500 * Price"},
            "distributions": {
                "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
            },
        },
        "current_state": {"Price": 40},
        "observed_outcome": {"Revenue": 30000},
        "target_outcome": {"Revenue": (35000, 36000)},
        "constraints": {
            "feasible": ["Price"],
            "max_changes": 1,
        },
        "seed": 42,
    }

    # Make two identical requests
    response1 = await client.post("/api/v1/explain/contrastive", json=request_data)
    response2 = await client.post("/api/v1/explain/contrastive", json=request_data)

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Same number of interventions
    assert len(data1["minimal_interventions"]) == len(data2["minimal_interventions"])

    # Same values (within floating point precision)
    for int1, int2 in zip(data1["minimal_interventions"], data2["minimal_interventions"]):
        assert int1["rank"] == int2["rank"]
        for var in int1["changes"]:
            assert abs(
                int1["changes"][var]["to_value"] - int2["changes"][var]["to_value"]
            ) < 0.01


@pytest.mark.asyncio
async def test_contrastive_explanation_respects_fixed_constraints(client):
    """Test that fixed variables are not changed."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Quality", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price + 200*Quality"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "current_state": {"Price": 40, "Quality": 7.5},
            "observed_outcome": {"Revenue": 31500},
            "target_outcome": {"Revenue": (35000, 36000)},
            "constraints": {
                "feasible": ["Price"],
                "fixed": ["Quality"],
                "max_changes": 1,
            },
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # All interventions should only change Price, not Quality
    for intervention in data["minimal_interventions"]:
        assert "Quality" not in intervention["changes"]
        assert "Price" in intervention["changes"]


@pytest.mark.asyncio
async def test_contrastive_explanation_ranking_by_cost(client):
    """Test ranking interventions by cost."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Marketing", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price + 0.5*Marketing"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "current_state": {"Price": 40, "Marketing": 20000},
            "observed_outcome": {"Revenue": 40000},
            "target_outcome": {"Revenue": (45000, 46000)},
            "constraints": {
                "feasible": ["Price", "Marketing"],
                "max_changes": 1,
                "minimize": "cost",
            },
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Best by cost should be rank 1
    assert data["comparison"]["best_by_cost"] == 1


@pytest.mark.asyncio
async def test_contrastive_explanation_with_bounds(client):
    """Test contrastive explanation with variable bounds."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500 * Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "current_state": {"Price": 40},
            "observed_outcome": {"Revenue": 30000},
            "target_outcome": {"Revenue": (35000, 36000)},
            "constraints": {
                "feasible": ["Price"],
                "max_changes": 1,
                "variable_bounds": {"Price": (35, 55)},
            },
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # All interventions should respect bounds
    for intervention in data["minimal_interventions"]:
        price_change = intervention["changes"]["Price"]
        assert 35 <= price_change["to_value"] <= 55


@pytest.mark.asyncio
async def test_contrastive_explanation_metadata(client):
    """Test that response includes metadata."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500 * Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "current_state": {"Price": 40},
            "observed_outcome": {"Revenue": 30000},
            "target_outcome": {"Revenue": (35000, 36000)},
            "constraints": {
                "feasible": ["Price"],
                "max_changes": 1,
            },
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Check metadata exists
    assert "_metadata" in data or "metadata" in data


@pytest.mark.asyncio
async def test_contrastive_explanation_explanation_quality(client):
    """Test that explanation is informative."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500 * Price"},
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 100}}
                },
            },
            "current_state": {"Price": 40},
            "observed_outcome": {"Revenue": 30000},
            "target_outcome": {"Revenue": (35000, 36000)},
            "constraints": {
                "feasible": ["Price"],
                "max_changes": 1,
            },
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

    # Summary should mention the intervention
    assert len(explanation["summary"]) > 0
    # Reasoning should provide context
    assert len(explanation["reasoning"]) > 0


@pytest.mark.asyncio
async def test_contrastive_explanation_invalid_request(client):
    """Test handling of invalid request (Pydantic validation)."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": [],  # Empty variables
                "equations": {},
                "distributions": {},
            },
            "current_state": {},
            "observed_outcome": {},
            "target_outcome": {},
            "constraints": {
                "feasible": [],  # Empty feasible (should fail validation)
            },
        },
    )

    # Should return validation error (422 for Pydantic validation)
    assert response.status_code in [400, 422]
    data = response.json()
    assert "code" in data or "detail" in data  # Error response


@pytest.mark.asyncio
async def test_contrastive_explanation_complex_model(client):
    """Test contrastive explanation with complex multi-variable model."""
    response = await client.post(
        "/api/v1/explain/contrastive",
        json={
            "model": {
                "variables": ["Price", "Quality", "Marketing", "Brand", "Revenue"],
                "equations": {
                    "Brand": "50 + 0.3 * Quality - 0.1 * Price",
                    "Revenue": "10000 + 800*Price + 200*Quality + 0.5*Marketing + 300*Brand",
                },
                "distributions": {
                    "noise": {"type": "normal", "parameters": {"mean": 0, "std": 500}}
                },
            },
            "current_state": {
                "Price": 40,
                "Quality": 7.5,
                "Marketing": 30000,
            },
            "observed_outcome": {"Revenue": 65000},
            "target_outcome": {"Revenue": (75000, 80000)},
            "constraints": {
                "feasible": ["Price", "Marketing"],
                "fixed": ["Quality"],
                "max_changes": 2,
                "minimize": "cost",
                "variable_bounds": {
                    "Price": (30, 100),
                    "Marketing": (10000, 100000),
                },
            },
            "seed": 42,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should find interventions
    assert len(data["minimal_interventions"]) > 0

    # Check interventions respect constraints
    for intervention in data["minimal_interventions"]:
        # Should not change Quality (fixed)
        assert "Quality" not in intervention["changes"]

        # Should only change feasible variables
        for var in intervention["changes"]:
            assert var in ["Price", "Marketing"]
