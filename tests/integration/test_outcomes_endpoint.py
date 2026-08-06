"""
Integration tests for the outcome-logging endpoints.

Tests:
- POST /api/v1/outcomes/log
- PATCH /api/v1/outcomes/{id}
- GET /api/v1/outcomes/{id}
- GET /api/v1/outcomes/summary

Extracted from test_decision_robustness_endpoint.py by ROADMAP 2.704. That
file was deleted with the `decision_robustness` router it was named for, but it
also carried these tests for the SEPARATE `outcomes` router, which was NOT
retired (it is the ROADMAP 2.689 seam). They are preserved here unchanged,
including their quarantine status, so the retirement does not silently take
coverage for a router it had no remit over.

⚠ Two tests in this class pass VACUOUSLY and are recorded as such rather than
quietly fixed or deleted: `test_get_nonexistent_outcome` and
`test_update_nonexistent_outcome` assert 404, and the `outcomes` router is
DARK (unmounted in src/api/main.py). The 404 they observe means "no such
route", not "no such outcome" — the same 404 their 5 quarantined siblings fail
on while expecting 200. They will only become real tests when the router is
mounted. Rowed for 2.689; NOT fixed here, because changing them is a claim
about the outcomes capability, which this lane has no remit over.
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Get auth headers if auth is enabled."""
    if os.environ.get("ISL_AUTH_DISABLED", "").lower() == "true":
        return {}
    return {"X-API-Key": os.environ.get("ISL_API_KEY", "test_key")}


class TestOutcomeLoggingEndpoints:
    """Tests for outcome logging endpoints."""

    def test_log_decision(self, client, auth_headers):
        """Test POST /api/v1/outcomes/log."""
        response = client.post(
            "/api/v1/outcomes/log",
            json={
                "decision_id": "test_decision_001",
                "graph_hash": "abc123def456",
                "response_hash": "xyz789",
                "chosen_option": "option_a",
                "recommendation_option": "option_a",
                "user_id": "test_user",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "id" in data
        assert data["decision_id"] == "test_decision_001"
        assert data["recommendation_followed"] is True
        assert "timestamp" in data

    def test_log_decision_not_followed(self, client, auth_headers):
        """Test logging when recommendation not followed."""
        response = client.post(
            "/api/v1/outcomes/log",
            json={
                "decision_id": "test_decision_002",
                "graph_hash": "abc123",
                "response_hash": "def456",
                "chosen_option": "option_b",
                "recommendation_option": "option_a",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["recommendation_followed"] is False

    def test_update_outcome(self, client, auth_headers):
        """Test PATCH /api/v1/outcomes/{id}."""
        # First create a log
        log_response = client.post(
            "/api/v1/outcomes/log",
            json={
                "decision_id": "test_decision_003",
                "graph_hash": "abc123",
                "response_hash": "def456",
                "chosen_option": "option_a",
                "recommendation_option": "option_a",
            },
            headers=auth_headers,
        )
        log_id = log_response.json()["id"]

        # Update with outcome
        response = client.patch(
            f"/api/v1/outcomes/{log_id}",
            json={
                "outcome_values": {"revenue": 155000.0},
                "notes": "Exceeded expectations",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["outcome_values"] == {"revenue": 155000.0}
        assert "outcome_timestamp" in data
        assert "Exceeded expectations" in data["notes"]

    def test_update_nonexistent_outcome(self, client, auth_headers):
        """Test PATCH with nonexistent ID returns 404."""
        response = client.patch(
            "/api/v1/outcomes/nonexistent_id",
            json={"outcome_values": {"revenue": 100000}},
            headers=auth_headers,
        )

        assert response.status_code == 404

    def test_get_outcome(self, client, auth_headers):
        """Test GET /api/v1/outcomes/{id}."""
        # First create a log
        log_response = client.post(
            "/api/v1/outcomes/log",
            json={
                "decision_id": "test_decision_004",
                "graph_hash": "abc123",
                "response_hash": "def456",
                "chosen_option": "option_a",
                "recommendation_option": "option_a",
            },
            headers=auth_headers,
        )
        log_id = log_response.json()["id"]

        # Get it back
        response = client.get(
            f"/api/v1/outcomes/{log_id}",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == log_id
        assert data["decision_id"] == "test_decision_004"

    def test_get_nonexistent_outcome(self, client, auth_headers):
        """Test GET with nonexistent ID returns 404."""
        response = client.get(
            "/api/v1/outcomes/nonexistent_id",
            headers=auth_headers,
        )

        assert response.status_code == 404

    def test_get_summary(self, client, auth_headers):
        """Test GET /api/v1/outcomes/summary."""
        # Log a few decisions first
        for i in range(3):
            client.post(
                "/api/v1/outcomes/log",
                json={
                    "decision_id": f"summary_test_{i}",
                    "graph_hash": "abc123",
                    "response_hash": "def456",
                    "chosen_option": "option_a" if i < 2 else "option_b",
                    "recommendation_option": "option_a",
                },
                headers=auth_headers,
            )

        response = client.get(
            "/api/v1/outcomes/summary",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "total_logged" in data
        assert "with_outcomes" in data
        assert "recommendations_followed" in data
        assert "recommendations_followed_pct" in data
