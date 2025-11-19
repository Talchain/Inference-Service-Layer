"""
Integration tests for health endpoint.
"""


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    assert data["version"] == "0.1.0"


def test_health_endpoint_fields(client):
    """Test health endpoint returns all required fields."""
    response = client.get("/health")
    data = response.json()

    required_fields = ["status", "version", "timestamp"]
    for field in required_fields:
        assert field in data, f"Health response missing field: {field}"
