"""
Integration tests for health endpoint.

NOTE: Tests converted to async to avoid Starlette TestClient async middleware bug.
Uses httpx.AsyncClient with pytest-asyncio.
"""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health check endpoint."""
    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_health_endpoint_fields(client):
    """Test health endpoint returns all required fields."""
    response = await client.get("/health")
    data = response.json()

    required_fields = ["status", "version", "timestamp", "build"]
    for field in required_fields:
        assert field in data, f"Health response missing field: {field}"


@pytest.mark.asyncio
async def test_health_endpoint_build_field(client):
    """Test health endpoint returns build field with Git SHA."""
    response = await client.get("/health")
    data = response.json()

    # build field should always be present
    assert "build" in data

    # build should be either "unknown" or 7-char Git SHA
    build = data["build"]
    assert build == "unknown" or len(build) == 7

    # build_full is optional (only present if Git SHA is known)
    if "build_full" in data and data["build_full"] is not None:
        assert len(data["build_full"]) == 40  # Full SHA-1 hex
