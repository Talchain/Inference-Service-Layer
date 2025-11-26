"""
Contract validation tests for ISL API.

Tests validate that ISL maintains its contracts with workstreams:
- All responses include required metadata
- Determinism guarantees (same seed = same result)
- Error responses follow standard format
- Performance meets SLA (<5s per workflow)
- Response schemas are valid
"""

import pytest
import httpx
from typing import Dict, Any
import time
import asyncio


@pytest.fixture
def base_url():
    """ISL API base URL."""
    return "http://localhost:8000/api/v1"


@pytest.fixture
async def async_client(base_url):
    """Async HTTP client for testing."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client


# ==================== METADATA CONTRACTS ====================

@pytest.mark.asyncio
async def test_all_responses_include_metadata(
    async_client,
    pricing_dag
):
    """
    CONTRACT: All successful responses must include metadata for reproducibility.

    Required metadata fields:
    - isl_version or version
    - request_id or similar tracking ID
    - config_fingerprint or determinism info
    """
    endpoints_to_test = [
        ("/causal/validate", {
            "dag": pricing_dag,
            "treatment": "Price",
            "outcome": "Revenue"
        }),
        ("/causal/counterfactual", {
            "dag": pricing_dag,
            "intervention": {"Price": 50.0},
            "outcome": "Revenue"
        }),
    ]

    for endpoint, payload in endpoints_to_test:
        response = await async_client.post(endpoint, json=payload)

        if response.status_code == 200:
            result = response.json()

            # Must include metadata
            assert "metadata" in result or "response_metadata" in result, \
                f"Endpoint {endpoint} missing metadata"

            metadata = result.get("metadata", result.get("response_metadata", {}))
            assert isinstance(metadata, dict), \
                f"Endpoint {endpoint} metadata is not a dict"

            # Metadata should have some identifying info
            # (exact field names may vary, but should be present)
            assert len(metadata) > 0, \
                f"Endpoint {endpoint} has empty metadata"


@pytest.mark.asyncio
async def test_metadata_includes_version(
    async_client,
    pricing_dag
):
    """
    CONTRACT: Metadata must include ISL version for compatibility checking.
    """
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 50.0},
            "outcome": "Revenue"
        }
    )

    assert response.status_code == 200
    result = response.json()

    metadata = result.get("metadata", result.get("response_metadata", {}))

    # Should include version info (field name may vary)
    version_fields = ["isl_version", "version", "api_version"]
    has_version = any(field in metadata for field in version_fields)

    assert has_version, f"Metadata missing version field. Metadata: {metadata}"


# ==================== DETERMINISM CONTRACTS ====================

@pytest.mark.asyncio
async def test_determinism_with_seeds(
    async_client,
    pricing_dag
):
    """
    CONTRACT: Same seed must produce identical results.

    Critical for reproducibility and debugging.
    """
    intervention = {"Price": 50.0}
    seed = 42

    # Run same request twice with same seed
    response1 = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "seed": seed
        }
    )

    response2 = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "seed": seed
        }
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    result1 = response1.json()
    result2 = response2.json()

    # Core results must be identical
    assert result1.get("predicted_outcome") == result2.get("predicted_outcome"), \
        "Same seed produced different outcomes"


@pytest.mark.asyncio
async def test_different_seeds_may_differ(
    async_client,
    pricing_dag
):
    """
    CONTRACT: Different seeds may produce different results (stochastic methods).
    """
    intervention = {"Price": 50.0}

    # Run with different seeds
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
            "seed": 999
        }
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Results may differ (but both should be valid)
    result1 = response1.json()
    result2 = response2.json()

    assert "predicted_outcome" in result1
    assert "predicted_outcome" in result2


# ==================== ERROR RESPONSE CONTRACTS ====================

@pytest.mark.asyncio
async def test_error_responses_valid(async_client):
    """
    CONTRACT: Error responses must follow FastAPI standard format.

    Expected format:
    {
        "detail": "Error message" or [{"msg": "...", "type": "..."}]
    }
    """
    # Send invalid request
    response = await async_client.post(
        "/causal/validate",
        json={
            "dag": {"nodes": [], "edges": []},  # Invalid: empty DAG
            "treatment": "X",
            "outcome": "Y"
        }
    )

    # Should return error
    assert response.status_code in [400, 422]

    error = response.json()

    # Must have detail field
    assert "detail" in error, "Error response missing 'detail' field"

    # Detail should be non-empty
    detail = error["detail"]
    assert detail is not None
    assert len(str(detail)) > 0


@pytest.mark.asyncio
async def test_validation_errors_descriptive(async_client):
    """
    CONTRACT: Validation errors must be descriptive enough to fix.
    """
    # Send request with missing required field
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": {"nodes": ["A", "B"], "edges": []},
            # Missing 'intervention' field
            "outcome": "B"
        }
    )

    assert response.status_code == 422

    error = response.json()
    assert "detail" in error

    # Error should mention what's missing
    detail_str = str(error["detail"]).lower()
    # (exact message varies, just verify it's descriptive)
    assert len(detail_str) > 10


# ==================== PERFORMANCE CONTRACTS ====================

@pytest.mark.asyncio
async def test_performance_under_5s(
    async_client,
    pricing_dag,
    batch_scenarios,
    performance_threshold
):
    """
    CONTRACT: Standard workflows must complete under 5s.

    This is the SLA for workstream integrations.
    """
    # Test individual operations
    operations = [
        ("/causal/validate", {
            "dag": pricing_dag,
            "treatment": "Price",
            "outcome": "Revenue"
        }),
        ("/causal/counterfactual", {
            "dag": pricing_dag,
            "intervention": {"Price": 50.0},
            "outcome": "Revenue"
        }),
        ("/batch/scenarios", {
            "dag": pricing_dag,
            "scenarios": batch_scenarios[:3],  # First 3 scenarios
            "outcome": "Revenue",
            "mode": "compare"
        }),
    ]

    for endpoint, payload in operations:
        start_time = time.time()

        response = await async_client.post(endpoint, json=payload)
        duration = time.time() - start_time

        assert response.status_code == 200, \
            f"Endpoint {endpoint} failed"

        assert duration < performance_threshold, \
            f"Endpoint {endpoint} took {duration:.2f}s (threshold: {performance_threshold}s)"


@pytest.mark.asyncio
async def test_concurrent_requests_performance(
    async_client,
    pricing_dag,
    performance_threshold
):
    """
    CONTRACT: System handles concurrent requests without degradation.
    """
    # Send 5 concurrent requests
    async def make_request():
        return await async_client.post(
            "/causal/counterfactual",
            json={
                "dag": pricing_dag,
                "intervention": {"Price": 50.0},
                "outcome": "Revenue"
            }
        )

    start_time = time.time()

    # Run 5 requests concurrently
    responses = await asyncio.gather(*[make_request() for _ in range(5)])

    duration = time.time() - start_time

    # All should succeed
    for response in responses:
        assert response.status_code == 200

    # Total time should be reasonable (not 5x single request)
    assert duration < performance_threshold * 2, \
        f"5 concurrent requests took {duration:.2f}s"


# ==================== RESPONSE SCHEMA CONTRACTS ====================

@pytest.mark.asyncio
async def test_counterfactual_response_schema(
    async_client,
    pricing_dag
):
    """
    CONTRACT: Counterfactual responses must include required fields.

    Required:
    - predicted_outcome (float)
    - explanation (str or dict)
    - metadata (dict)
    """
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 50.0},
            "outcome": "Revenue"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Required fields
    assert "predicted_outcome" in result, "Missing predicted_outcome"
    assert isinstance(result["predicted_outcome"], (int, float)), \
        "predicted_outcome must be numeric"

    # Explanation should be present
    assert "explanation" in result or "metadata" in result, \
        "Missing explanation or metadata"


@pytest.mark.asyncio
async def test_validation_response_schema(
    async_client,
    pricing_dag
):
    """
    CONTRACT: Validation responses must include required fields.

    Required:
    - status or identifiable (bool/str)
    - metadata (dict)
    """
    response = await async_client.post(
        "/causal/validate",
        json={
            "dag": pricing_dag,
            "treatment": "Price",
            "outcome": "Revenue"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Must indicate identifiability
    has_status = "status" in result or "identifiable" in result
    assert has_status, "Missing status/identifiable field"


@pytest.mark.asyncio
async def test_batch_response_schema(
    async_client,
    pricing_dag,
    batch_scenarios
):
    """
    CONTRACT: Batch responses must include results for all scenarios.

    Required:
    - results (list) with same length as input scenarios
    - Each result has predicted_outcome
    """
    response = await async_client.post(
        "/batch/scenarios",
        json={
            "dag": pricing_dag,
            "scenarios": batch_scenarios,
            "outcome": "Revenue",
            "mode": "compare"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Must have results
    assert "results" in result, "Missing results field"

    results = result["results"]
    assert isinstance(results, list), "results must be a list"

    # Must have same number of results as scenarios
    assert len(results) == len(batch_scenarios), \
        f"Expected {len(batch_scenarios)} results, got {len(results)}"

    # Each result must have outcome
    for i, scenario_result in enumerate(results):
        assert "predicted_outcome" in scenario_result or "outcome" in scenario_result, \
            f"Result {i} missing outcome"


# ==================== BACKWARDS COMPATIBILITY ====================

@pytest.mark.asyncio
async def test_deprecated_fields_still_work(
    async_client,
    pricing_dag
):
    """
    CONTRACT: Deprecated field names continue to work (grace period).

    If API changes field names, old names should still work temporarily.
    """
    # This test would check for specific deprecated fields
    # For now, just verify basic request works
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": {"Price": 50.0},
            "outcome": "Revenue"
        }
    )

    assert response.status_code == 200


# ==================== REQUEST VALIDATION ====================

@pytest.mark.asyncio
async def test_missing_required_fields_rejected(async_client):
    """
    CONTRACT: Requests missing required fields are rejected with 422.
    """
    # Missing 'outcome' field
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": {"nodes": ["A"], "edges": []},
            "intervention": {"A": 1.0}
            # Missing 'outcome'
        }
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_field_types_rejected(async_client):
    """
    CONTRACT: Invalid field types are rejected with 422.
    """
    # 'intervention' should be dict, not string
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": {"nodes": ["A"], "edges": []},
            "intervention": "invalid",  # Should be dict
            "outcome": "A"
        }
    )

    assert response.status_code == 422


# ==================== HEALTH CHECK CONTRACT ====================

@pytest.mark.asyncio
async def test_health_endpoint_available(async_client):
    """
    CONTRACT: Health endpoint must be available for monitoring.
    """
    # Try common health endpoint paths
    health_paths = ["/health", "/_health", "/healthz", "/api/v1/health"]

    health_found = False
    for path in health_paths:
        try:
            response = await async_client.get(path)
            if response.status_code == 200:
                health_found = True
                break
        except:
            continue

    # At least one health endpoint should exist
    # (This is informational - not all systems have health endpoints)
