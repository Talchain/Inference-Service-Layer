"""
Integration tests for Decision Robustness Suite API endpoints.

Tests:
- POST /api/v1/analysis/robustness
- POST /api/v1/outcomes/log
- PATCH /api/v1/outcomes/{id}
- GET /api/v1/outcomes/{id}
- GET /api/v1/outcomes/summary
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


def make_test_graph():
    """Create a test graph payload."""
    return {
        "nodes": [
            {"id": "marketing_spend", "kind": "decision", "label": "Marketing Spend", "belief": 0.5},
            {"id": "price", "kind": "decision", "label": "Product Price", "belief": 0.6},
            {"id": "demand", "kind": "factor", "label": "Customer Demand", "belief": 0.5},
            {"id": "revenue", "kind": "goal", "label": "Revenue", "belief": 0.5},
        ],
        "edges": [
            {"from": "marketing_spend", "to": "demand", "weight": 2.0},
            {"from": "price", "to": "demand", "weight": -1.5},
            {"from": "demand", "to": "revenue", "weight": 2.5},
        ],
    }


def make_test_options():
    """Create test decision options."""
    return [
        {
            "id": "option_a",
            "label": "Aggressive Marketing",
            "interventions": {"marketing_spend": 100000, "price": 49.99},
            "is_baseline": False,
        },
        {
            "id": "option_b",
            "label": "Premium Pricing",
            "interventions": {"marketing_spend": 50000, "price": 79.99},
            "is_baseline": True,
        },
    ]


def make_robustness_request():
    """Create a complete robustness request payload."""
    return {
        "graph": make_test_graph(),
        "options": make_test_options(),
        "utility": {
            "goal_node_id": "revenue",
            "maximize": True,
        },
        "analysis_options": {
            "sensitivity_top_n": 3,
            "perturbation_range": 0.5,
            "monte_carlo_samples": 100,
            "include_pareto": False,
            "include_voi": False,
        },
    }


class TestRobustnessEndpoint:
    """Tests for POST /api/v1/analysis/robustness."""

    def test_robustness_basic(self, client, auth_headers):
        """Test basic robustness analysis."""
        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "result" in data
        result = data["result"]

        # Check required fields
        assert "option_rankings" in result
        assert "recommendation" in result
        assert "sensitivity" in result
        assert "robustness_label" in result
        assert "robustness_summary" in result
        assert "robustness_bounds" in result
        assert "narrative" in result

    def test_robustness_option_rankings(self, client, auth_headers):
        """Test option rankings are returned correctly."""
        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()["result"]

        rankings = result["option_rankings"]
        assert len(rankings) == 2

        for option in rankings:
            assert "option_id" in option
            assert "option_label" in option
            assert "expected_utility" in option
            assert "utility_distribution" in option
            assert "rank" in option

            dist = option["utility_distribution"]
            assert "p5" in dist
            assert "p50" in dist
            assert "p95" in dist

    def test_robustness_recommendation(self, client, auth_headers):
        """Test recommendation is included."""
        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()["result"]

        rec = result["recommendation"]
        assert rec["option_id"] in ["option_a", "option_b"]
        assert rec["confidence"] in ["high", "medium", "low"]
        assert rec["recommendation_status"] in ["actionable", "exploratory"]

    def test_robustness_sensitivity_analysis(self, client, auth_headers):
        """Test sensitivity analysis is included."""
        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()["result"]

        sensitivity = result["sensitivity"]
        assert isinstance(sensitivity, list)

        for param in sensitivity:
            assert "parameter_id" in param
            assert "sensitivity_score" in param
            assert 0 <= param["sensitivity_score"] <= 1
            assert "impact_direction" in param
            assert param["impact_direction"] in ["positive", "negative"]

    def test_robustness_label(self, client, auth_headers):
        """Test robustness label classification."""
        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()["result"]

        assert result["robustness_label"] in ["robust", "moderate", "fragile"]
        assert len(result["robustness_summary"]) > 0

    def test_robustness_narrative(self, client, auth_headers):
        """Test narrative is generated."""
        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()["result"]

        assert len(result["narrative"]) > 0
        # Should mention the recommendation
        rec_label = result["recommendation"]["option_label"]
        assert rec_label in result["narrative"]

    def test_robustness_with_voi(self, client, auth_headers):
        """Test robustness with Value of Information enabled."""
        request = make_robustness_request()
        request["analysis_options"]["include_voi"] = True
        request["parameter_uncertainties"] = {
            "churn_rate": {"mean": 0.15, "std": 0.05},
        }

        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()["result"]

        voi = result["value_of_information"]
        assert isinstance(voi, list)
        assert len(voi) > 0

        for item in voi:
            assert "parameter_id" in item
            assert "evpi" in item
            assert "evsi" in item
            assert item["evpi"] >= 0
            assert item["evsi"] >= 0

    def test_robustness_with_pareto(self, client, auth_headers):
        """Test robustness with Pareto frontier enabled."""
        request = make_robustness_request()

        # Add second goal
        request["graph"]["nodes"].append({
            "id": "satisfaction",
            "kind": "goal",
            "label": "Customer Satisfaction",
        })
        request["graph"]["edges"].append({
            "from": "price",
            "to": "satisfaction",
            "weight": -1.0,
        })
        request["utility"]["additional_goals"] = ["satisfaction"]
        request["analysis_options"]["include_pareto"] = True

        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()["result"]

        pareto = result["pareto"]
        assert pareto is not None
        assert "goals" in pareto
        assert len(pareto["goals"]) == 2
        assert "frontier_options" in pareto

    def test_robustness_metadata(self, client, auth_headers):
        """Test response includes metadata."""
        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Check metadata
        assert "_metadata" in data or "metadata" in data
        metadata = data.get("_metadata") or data.get("metadata")
        if metadata:
            assert "request_id" in metadata or "computation_time_ms" in metadata

    def test_robustness_invalid_goal_node(self, client, auth_headers):
        """Test error when goal node doesn't exist."""
        request = make_robustness_request()
        request["utility"]["goal_node_id"] = "nonexistent_node"

        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 400

    def test_robustness_invalid_intervention_node(self, client, auth_headers):
        """Test error when intervention node doesn't exist."""
        request = make_robustness_request()
        request["options"][0]["interventions"]["nonexistent"] = 100

        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 400

    def test_robustness_request_id_header(self, client, auth_headers):
        """Test X-Request-Id header is respected."""
        headers = {**auth_headers, "X-Request-Id": "custom-req-123"}

        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
            headers=headers,
        )

        assert response.status_code == 200


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


class TestRobustnessPerformance:
    """Tests for performance requirements."""

    def test_robustness_completes_in_time(self, client, auth_headers):
        """Test analysis completes within timeout."""
        import time

        request = make_robustness_request()
        request["analysis_options"]["monte_carlo_samples"] = 500

        start = time.time()
        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should complete in reasonable time (< 10s for tests)
        assert elapsed < 10.0

    def test_robustness_with_larger_graph(self, client, auth_headers):
        """Test analysis with a larger graph."""
        request = make_robustness_request()

        # Add more nodes
        for i in range(10):
            request["graph"]["nodes"].append({
                "id": f"factor_{i}",
                "kind": "factor",
                "label": f"Factor {i}",
            })
            request["graph"]["edges"].append({
                "from": f"factor_{i}",
                "to": "demand",
                "weight": 0.5,
            })

        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 200


class TestAuthorizationEdgeCases:
    """Tests for authorization edge cases."""

    def test_robustness_without_auth(self, client):
        """Test robustness without auth header."""
        response = client.post(
            "/api/v1/analysis/robustness",
            json=make_robustness_request(),
        )

        # Should fail if auth is enabled
        if os.environ.get("ISL_AUTH_DISABLED", "").lower() != "true":
            assert response.status_code in [401, 403]
        else:
            assert response.status_code == 200

    def test_outcomes_without_auth(self, client):
        """Test outcome logging without auth header."""
        response = client.post(
            "/api/v1/outcomes/log",
            json={
                "decision_id": "test",
                "graph_hash": "abc",
                "response_hash": "def",
                "chosen_option": "a",
                "recommendation_option": "a",
            },
        )

        # Should fail if auth is enabled
        if os.environ.get("ISL_AUTH_DISABLED", "").lower() != "true":
            assert response.status_code in [401, 403]
        else:
            assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""

    def test_robustness_empty_options(self, client, auth_headers):
        """Test error with empty options list."""
        request = make_robustness_request()
        request["options"] = []

        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 422  # Validation error

    def test_robustness_single_option(self, client, auth_headers):
        """Test error with only one option."""
        request = make_robustness_request()
        request["options"] = [request["options"][0]]

        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 422  # Need at least 2 options

    def test_robustness_empty_graph(self, client, auth_headers):
        """Test error with empty graph."""
        request = make_robustness_request()
        request["graph"]["nodes"] = []
        request["graph"]["edges"] = []

        response = client.post(
            "/api/v1/analysis/robustness",
            json=request,
            headers=auth_headers,
        )

        # Should fail validation or analysis
        assert response.status_code in [400, 422]

    def test_outcome_log_missing_fields(self, client, auth_headers):
        """Test error when required fields missing."""
        response = client.post(
            "/api/v1/outcomes/log",
            json={
                "decision_id": "test",
                # Missing required fields
            },
            headers=auth_headers,
        )

        assert response.status_code == 422
