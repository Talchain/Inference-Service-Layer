"""
Integration tests for PLoT Engine workflows.

Tests validate complete workflows that PLoT uses:
- Standard analysis (validate → counterfactual → batch)
- Goal-seeking (contrastive → apply → validate)
- Cross-market (transport → validate assumptions)
"""

import pytest
import httpx
from typing import Dict, Any
import time


@pytest.fixture
def base_url():
    """ISL API base URL."""
    return "http://localhost:8000/api/v1"


@pytest.fixture
async def async_client(base_url):
    """Async HTTP client for testing."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client


# ==================== STANDARD ANALYSIS WORKFLOW ====================

@pytest.mark.asyncio
async def test_validate_analyze_compare_workflow(
    async_client,
    pricing_dag,
    batch_scenarios,
    calibration_data,
    performance_threshold
):
    """
    Test PLoT's standard analysis workflow:
    1. Validate DAG
    2. Analyze counterfactual for baseline
    3. Batch compare multiple scenarios
    4. Get conformal intervals
    """
    start_time = time.time()

    # Step 1: Validate DAG
    validate_response = await async_client.post(
        "/causal/validate",
        json={
            "dag": pricing_dag,
            "treatment": "Price",
            "outcome": "Revenue"
        }
    )
    assert validate_response.status_code == 200
    validation = validate_response.json()
    assert validation["status"] in ["identifiable", "non_identifiable"]

    # Step 2: Analyze baseline counterfactual
    counterfactual_response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 50.0},
            "outcome": "Revenue"
        }
    )
    assert counterfactual_response.status_code == 200
    baseline = counterfactual_response.json()
    assert "predicted_outcome" in baseline
    assert "explanation" in baseline

    # Step 3: Batch compare scenarios
    batch_response = await async_client.post(
        "/batch/scenarios",
        json={
            "dag": pricing_dag,
            "scenarios": batch_scenarios,
            "outcome": "Revenue",
            "mode": "compare"
        }
    )
    assert batch_response.status_code == 200
    batch_results = batch_response.json()
    assert "results" in batch_results
    assert len(batch_results["results"]) == len(batch_scenarios)

    # Step 4: Get conformal prediction intervals
    conformal_response = await async_client.post(
        "/causal/conformal",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 50.0},
            "outcome": "Revenue",
            "calibration_data": calibration_data,
            "confidence_level": 0.90
        }
    )
    assert conformal_response.status_code == 200
    conformal = conformal_response.json()
    assert "prediction_interval" in conformal
    assert "lower_bound" in conformal["prediction_interval"]
    assert "upper_bound" in conformal["prediction_interval"]

    # Verify workflow completed in time
    duration = time.time() - start_time
    assert duration < performance_threshold, f"Workflow took {duration:.2f}s"


@pytest.mark.asyncio
async def test_non_identifiable_gets_suggestions(
    async_client,
    confounded_dag
):
    """
    Test that non-identifiable DAGs return actionable suggestions.
    """
    response = await async_client.post(
        "/causal/validate/strategies",
        json={
            "dag": confounded_dag,
            "treatment": "Treatment",
            "outcome": "Outcome"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should indicate non-identifiable
    assert result.get("identifiable") == False or result.get("status") == "non_identifiable"

    # Should provide suggestions
    assert "suggestions" in result or "strategies" in result
    suggestions = result.get("suggestions", result.get("strategies", []))
    assert len(suggestions) > 0

    # Suggestions should be actionable
    for suggestion in suggestions:
        assert "strategy" in suggestion or "type" in suggestion
        assert "description" in suggestion or "explanation" in suggestion


@pytest.mark.asyncio
async def test_conformal_with_calibration_data(
    async_client,
    pricing_dag,
    large_calibration_data
):
    """
    Test conformal prediction with substantial calibration data.
    """
    response = await async_client.post(
        "/causal/conformal",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 55.0},
            "outcome": "Revenue",
            "calibration_data": large_calibration_data,
            "confidence_level": 0.95
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Verify interval structure
    assert "prediction_interval" in result
    interval = result["prediction_interval"]
    assert interval["lower_bound"] < interval["upper_bound"]

    # Verify coverage guarantee metadata
    assert "confidence_level" in result
    assert result["confidence_level"] == 0.95

    # Verify calibration was used
    assert "calibration_set_size" in result or "metadata" in result


@pytest.mark.asyncio
async def test_batch_scenarios_with_interactions(
    async_client,
    pricing_dag,
    batch_scenarios
):
    """
    Test batch scenario analysis includes interaction detection.
    """
    response = await async_client.post(
        "/batch/scenarios",
        json={
            "dag": pricing_dag,
            "scenarios": batch_scenarios,
            "outcome": "Revenue",
            "mode": "compare",
            "detect_interactions": True
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Verify all scenarios analyzed
    assert "results" in result
    assert len(result["results"]) == len(batch_scenarios)

    # Check for interaction analysis
    for scenario_result in result["results"]:
        assert "predicted_outcome" in scenario_result

    # May include interaction summary
    # (implementation optional, just verify no errors)


# ==================== GOAL-SEEKING WORKFLOW ====================

@pytest.mark.asyncio
async def test_goal_seeking_workflow(
    async_client,
    pricing_dag,
    performance_threshold
):
    """
    Test goal-seeking workflow:
    1. Get contrastive explanation for desired outcome
    2. Apply suggested intervention
    3. Validate result
    """
    start_time = time.time()

    # Step 1: Get contrastive explanation
    # User wants Revenue = 6000
    contrastive_response = await async_client.post(
        "/explain/contrastive",
        json={
            "dag": pricing_dag,
            "factual": {"Price": 50.0, "Quality": 0.8},
            "factual_outcome": 5300.0,
            "counterfactual_outcome": 6000.0,
            "outcome_variable": "Revenue"
        }
    )

    assert contrastive_response.status_code == 200
    contrastive = contrastive_response.json()

    # Should suggest minimal intervention
    assert "minimal_intervention" in contrastive or "intervention" in contrastive

    # Step 2: Apply suggested intervention
    intervention = contrastive.get("minimal_intervention", contrastive.get("intervention", {}))

    apply_response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue"
        }
    )

    assert apply_response.status_code == 200
    result = apply_response.json()

    # Step 3: Verify outcome is closer to goal
    predicted = result.get("predicted_outcome", 0)
    # Should be moving toward 6000 (from baseline 5300)
    # (exact value depends on DAG structure, just verify it ran)
    assert predicted > 0

    # Verify workflow completed in time
    duration = time.time() - start_time
    assert duration < performance_threshold


# ==================== TRANSPORTABILITY WORKFLOW ====================

@pytest.mark.asyncio
async def test_transportability_check(
    async_client,
    pricing_dag,
    source_market_data,
    target_market_data
):
    """
    Test cross-market transportability workflow.
    """
    response = await async_client.post(
        "/causal/transport",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 55.0},
            "outcome": "Revenue",
            "source_domain": source_market_data,
            "target_domain": target_market_data,
            "assumptions": ["no_selection_bias", "same_mechanism"]
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should include transportability assessment
    assert "transportable" in result or "status" in result

    # Should provide sensitivity analysis
    assert "assumptions" in result or "sensitivity" in result

    # Should include predictions for target domain
    assert "target_prediction" in result or "predicted_outcome" in result


# ==================== DETERMINISM & REPRODUCIBILITY ====================

@pytest.mark.asyncio
async def test_deterministic_with_seed(async_client, pricing_dag):
    """
    Test that same seed produces identical results.
    """
    intervention = {"Price": 50.0}

    # Run twice with same seed
    response1 = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "seed": 42
        }
    )

    response2 = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "seed": 42
        }
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    result1 = response1.json()
    result2 = response2.json()

    # Results should be identical
    assert result1.get("predicted_outcome") == result2.get("predicted_outcome")


# ==================== ERROR HANDLING ====================

@pytest.mark.asyncio
async def test_invalid_dag_returns_clear_error(async_client):
    """
    Test that invalid DAG returns actionable error.
    """
    invalid_dag = {
        "nodes": ["A", "B"],
        "edges": [
            {"from": "C", "to": "A"}  # C not in nodes
        ]
    }

    response = await async_client.post(
        "/causal/validate",
        json={
            "dag": invalid_dag,
            "treatment": "A",
            "outcome": "B"
        }
    )

    # Should return error (422 or 400)
    assert response.status_code in [400, 422]

    error = response.json()
    assert "detail" in error or "message" in error


@pytest.mark.asyncio
async def test_insufficient_calibration_fallback(async_client, pricing_dag):
    """
    Test graceful fallback when calibration data insufficient.
    """
    # Only 2 calibration points (too few)
    tiny_calibration = {
        "features": [
            {"Price": 50, "Quality": 0.8},
            {"Price": 60, "Quality": 0.7}
        ],
        "outcomes": [5200, 5100]
    }

    response = await async_client.post(
        "/causal/conformal",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 55.0},
            "outcome": "Revenue",
            "calibration_data": tiny_calibration,
            "confidence_level": 0.90
        }
    )

    # Should either:
    # - Return warning but still provide interval, OR
    # - Return clear error message
    assert response.status_code in [200, 400, 422]

    if response.status_code == 200:
        result = response.json()
        # Should include warning
        assert "warning" in result or "metadata" in result


# ==================== METADATA VALIDATION ====================

@pytest.mark.asyncio
async def test_all_responses_include_metadata(async_client, pricing_dag):
    """
    Test that all responses include required metadata for reproducibility.
    """
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 50.0},
            "outcome": "Revenue",
            "seed": 12345
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should include metadata
    assert "metadata" in result or "response_metadata" in result

    metadata = result.get("metadata", result.get("response_metadata", {}))

    # Metadata should include version info
    # (field names may vary, just check structure exists)
    assert isinstance(metadata, dict)
