"""
Smoke tests for production deployment validation.

These tests verify basic functionality after deployment.
Run after every production deployment to ensure core features work.

Usage:
    pytest tests/smoke/ --base-url=https://isl.olumi.com --api-key=YOUR_KEY
"""

import pytest
import httpx
from datetime import datetime


def pytest_addoption(parser):
    """Add command-line options for smoke tests."""
    parser.addoption(
        "--base-url",
        action="store",
        default="http://localhost:8000",
        help="Base URL of ISL service",
    )
    parser.addoption(
        "--api-key",
        action="store",
        default="test_key",
        help="API key for authentication",
    )


@pytest.fixture
def base_url(request):
    """Base URL from command line."""
    return request.config.getoption("--base-url")


@pytest.fixture
def api_key(request):
    """API key from command line."""
    return request.config.getoption("--api-key")


@pytest.fixture
def client(base_url):
    """HTTP client configured for ISL."""
    return httpx.AsyncClient(base_url=base_url, timeout=30.0)


# Health Checks

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Basic health check - service is running."""
    response = await client.get("/health")

    assert response.status_code == 200, f"Health check failed: {response.text}"

    data = response.json()
    assert data["status"] == "healthy", f"Service not healthy: {data}"
    assert "uptime" in data, "Health response missing uptime"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Metrics endpoint accessible."""
    response = await client.get("/metrics")

    assert response.status_code == 200, "Metrics endpoint failed"
    assert "isl_requests_total" in response.text, "Metrics missing expected data"
    assert "isl_llm_cost_dollars" in response.text, "LLM cost metrics missing"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openapi_docs(client):
    """OpenAPI documentation available."""
    response = await client.get("/docs")

    assert response.status_code == 200, "OpenAPI docs not accessible"


# Core Functionality

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_causal_validation_basic(client, api_key):
    """Basic causal validation works."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={
            "dag": {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"]]
            },
            "treatment": "X",
            "outcome": "Y"
        },
        headers={"X-API-Key": api_key}
    )

    assert response.status_code == 200, f"Validation failed: {response.text}"

    data = response.json()
    assert data["status"] == "identifiable", "Simple causal effect not identified"
    assert "formula" in data, "Validation missing formula"
    assert "_metadata" in data, "Response missing metadata"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_counterfactual_generation_basic(client, api_key):
    """Basic counterfactual generation works."""
    response = await client.post(
        "/api/v1/causal/counterfactual",
        json={
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
                "structural_equations": {
                    "revenue": "1000 * (100 - price)"
                }
            },
            "intervention": {"price": 50},
            "outcome_variables": ["revenue"],
            "samples": 100  # Low for speed
        },
        headers={"X-API-Key": api_key}
    )

    assert response.status_code == 200, f"Counterfactual failed: {response.text}"

    data = response.json()
    assert "prediction" in data, "Counterfactual missing prediction"
    assert "point_estimate" in data["prediction"], "Missing point estimate"

    # Verify calculation correct (revenue = 1000 * (100 - 50) = 50000)
    estimate = data["prediction"]["point_estimate"]
    assert 45000 <= estimate <= 55000, f"Unexpected estimate: {estimate}"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_robustness_analysis_basic(client, api_key):
    """Basic robustness analysis works."""
    response = await client.post(
        "/api/v1/robustness/analyze",
        json={
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]]
            },
            "intervention_proposal": {"price": 50},
            "target_outcome": {"revenue": [45000, 55000]},
            "perturbation_radius": 0.1,
            "min_samples": 50  # Low for speed
        },
        headers={"X-API-Key": api_key}
    )

    assert response.status_code == 200, f"Robustness failed: {response.text}"

    data = response.json()
    assert "analysis" in data, "Robustness missing analysis"
    assert "status" in data["analysis"], "Missing robustness status"
    assert data["analysis"]["status"] in ["robust", "fragile"], \
        f"Invalid status: {data['analysis']['status']}"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_llm_endpoints_responding(client, api_key):
    """LLM-powered endpoints are responding (may be slow on cold start)."""
    response = await client.post(
        "/api/v1/deliberation/deliberate",
        json={
            "decision_context": "Smoke test decision",
            "positions": [
                {
                    "member_id": "test_user_1",
                    "position_statement": "I support this test decision for verification purposes.",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
        },
        headers={"X-API-Key": api_key},
        timeout=30.0  # LLM may be slow
    )

    # Accept both 200 (success) and 503 (cold start/overload)
    assert response.status_code in [200, 503], \
        f"Deliberation endpoint failed: {response.status_code} - {response.text}"

    if response.status_code == 200:
        data = response.json()
        assert "session_id" in data, "Deliberation missing session_id"
        assert "consensus_statement" in data, "Deliberation missing consensus"


# Error Handling

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_invalid_dag_rejected(client, api_key):
    """Invalid DAG properly rejected."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={
            "dag": {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"], ["Y", "X"]]  # Cycle!
            },
            "treatment": "X",
            "outcome": "Y"
        },
        headers={"X-API-Key": api_key}
    )

    assert response.status_code == 400, "Cyclic DAG should be rejected"

    data = response.json()
    assert "detail" in data, "Error response missing detail"
    assert "cycle" in data["detail"].lower(), "Error should mention cycle"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_missing_api_key_rejected(client):
    """Requests without API key are rejected."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={
            "dag": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "treatment": "X",
            "outcome": "Y"
        }
        # No API key header
    )

    assert response.status_code == 401, "Missing API key should be rejected"


# Performance

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_causal_validation_latency(client, api_key):
    """Causal validation completes within acceptable time."""
    import time

    start = time.time()

    response = await client.post(
        "/api/v1/causal/validate",
        json={
            "dag": {
                "nodes": ["X", "Y", "Z", "W"],
                "edges": [["X", "Y"], ["Z", "Y"], ["Z", "W"]]
            },
            "treatment": "X",
            "outcome": "Y"
        },
        headers={"X-API-Key": api_key}
    )

    duration = time.time() - start

    assert response.status_code == 200, "Validation failed"
    assert duration < 1.0, f"Validation too slow: {duration:.2f}s (target: <1s)"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_counterfactual_latency(client, api_key):
    """Counterfactual generation completes within acceptable time."""
    import time

    start = time.time()

    response = await client.post(
        "/api/v1/causal/counterfactual",
        json={
            "causal_model": {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"]],
                "structural_equations": {"Y": "2 * X + 5"}
            },
            "intervention": {"X": 10},
            "outcome_variables": ["Y"],
            "samples": 100
        },
        headers={"X-API-Key": api_key}
    )

    duration = time.time() - start

    assert response.status_code == 200, "Counterfactual failed"
    assert duration < 3.0, f"Counterfactual too slow: {duration:.2f}s (target: <3s for 100 samples)"


# Integration

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_validate_then_counterfactual_workflow(client, api_key):
    """Common workflow: validate → counterfactual."""
    # Step 1: Validate
    validation_response = await client.post(
        "/api/v1/causal/validate",
        json={
            "dag": {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"]]
            },
            "treatment": "X",
            "outcome": "Y"
        },
        headers={"X-API-Key": api_key}
    )

    assert validation_response.status_code == 200, "Validation failed"
    assert validation_response.json()["status"] == "identifiable"

    # Step 2: Generate counterfactual
    counterfactual_response = await client.post(
        "/api/v1/causal/counterfactual",
        json={
            "causal_model": {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"]],
                "structural_equations": {"Y": "5 * X"}
            },
            "intervention": {"X": 10},
            "outcome_variables": ["Y"],
            "samples": 100
        },
        headers={"X-API-Key": api_key}
    )

    assert counterfactual_response.status_code == 200, "Counterfactual failed"

    # Verify result makes sense (Y = 5 * 10 = 50)
    estimate = counterfactual_response.json()["prediction"]["point_estimate"]
    assert 45 <= estimate <= 55, f"Unexpected estimate: {estimate}"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_metadata_present_in_all_responses(client, api_key):
    """All responses include proper metadata."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={
            "dag": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "treatment": "X",
            "outcome": "Y"
        },
        headers={"X-API-Key": api_key}
    )

    assert response.status_code == 200

    data = response.json()
    assert "_metadata" in data, "Response missing metadata"

    metadata = data["_metadata"]
    assert "isl_version" in metadata, "Metadata missing version"
    assert "request_id" in metadata, "Metadata missing request_id"
    assert "config_fingerprint" in metadata, "Metadata missing config fingerprint"


# Summary

@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_test_summary(client, api_key):
    """Summary: All core endpoints functional."""
    endpoints = {
        "health": "/health",
        "metrics": "/metrics",
        "causal_validation": "/api/v1/causal/validate",
        "counterfactual": "/api/v1/causal/counterfactual",
        "robustness": "/api/v1/robustness/analyze"
    }

    results = {}

    # Health check
    response = await client.get(endpoints["health"])
    results["health"] = response.status_code == 200

    # Metrics
    response = await client.get(endpoints["metrics"])
    results["metrics"] = response.status_code == 200

    # Causal validation
    response = await client.post(
        endpoints["causal_validation"],
        json={
            "dag": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "treatment": "X",
            "outcome": "Y"
        },
        headers={"X-API-Key": api_key}
    )
    results["causal_validation"] = response.status_code == 200

    # Counterfactual
    response = await client.post(
        endpoints["counterfactual"],
        json={
            "causal_model": {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"]],
                "structural_equations": {"Y": "X"}
            },
            "intervention": {"X": 1},
            "outcome_variables": ["Y"],
            "samples": 10
        },
        headers={"X-API-Key": api_key}
    )
    results["counterfactual"] = response.status_code == 200

    # Robustness
    response = await client.post(
        endpoints["robustness"],
        json={
            "causal_model": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "intervention_proposal": {"X": 1},
            "target_outcome": {"Y": [0.9, 1.1]},
            "perturbation_radius": 0.1,
            "min_samples": 10
        },
        headers={"X-API-Key": api_key}
    )
    results["robustness"] = response.status_code == 200

    # Assert all passed
    failed = [k for k, v in results.items() if not v]
    assert len(failed) == 0, f"Failed endpoints: {failed}"

    print("\n✅ Smoke Test Summary:")
    print("=" * 50)
    for endpoint, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {endpoint}")
    print("=" * 50)
