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

    required_fields = ["status", "version", "timestamp"]
    for field in required_fields:
        assert field in data, f"Health response missing field: {field}"
