"""
Integration tests for Y₀ Identifiability Analysis API endpoints.

Tests the /api/v1/analysis/identifiability endpoint and verifies:
- Identifiable effects return actionable status
- Non-identifiable effects return exploratory status (hard rule)
- Suggestions are provided for non-identifiable effects
- Simple DAG format endpoint works
- Error handling is correct
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Headers with API key for authenticated requests."""
    return {"X-API-Key": "test-api-key"}


# =============================================================================
# Test Data
# =============================================================================


IDENTIFIABLE_GRAPH = {
    "graph": {
        "nodes": [
            {"id": "price", "kind": "decision", "label": "Pricing Strategy"},
            {"id": "revenue", "kind": "goal", "label": "Revenue Target"},
        ],
        "edges": [
            {"from": "price", "to": "revenue", "weight": 2.0},
        ],
    }
}


BACKDOOR_IDENTIFIABLE_GRAPH = {
    "graph": {
        "nodes": [
            {"id": "price", "kind": "decision", "label": "Price"},
            {"id": "market_segment", "kind": "factor", "label": "Market Segment"},
            {"id": "revenue", "kind": "goal", "label": "Revenue"},
        ],
        "edges": [
            {"from": "price", "to": "revenue", "weight": 2.0},
            {"from": "market_segment", "to": "price", "weight": 1.5},
            {"from": "market_segment", "to": "revenue", "weight": 1.0},
        ],
    }
}


NO_PATH_GRAPH = {
    "graph": {
        "nodes": [
            {"id": "decision", "kind": "decision", "label": "Decision"},
            {"id": "other", "kind": "outcome", "label": "Other"},
            {"id": "goal", "kind": "goal", "label": "Goal"},
        ],
        "edges": [
            {"from": "decision", "to": "other", "weight": 1.0},
        ],
    }
}


MISSING_DECISION_GRAPH = {
    "graph": {
        "nodes": [
            {"id": "factor", "kind": "factor", "label": "Factor"},
            {"id": "goal", "kind": "goal", "label": "Goal"},
        ],
        "edges": [
            {"from": "factor", "to": "goal", "weight": 1.0},
        ],
    }
}


SIMPLE_DAG_REQUEST = {
    "nodes": ["X", "Y", "Z"],
    "edges": [["X", "Y"], ["Z", "X"], ["Z", "Y"]],
    "treatment": "X",
    "outcome": "Y",
}


SIMPLE_DAG_NO_PATH = {
    "nodes": ["X", "Y", "Z"],
    "edges": [["X", "Z"]],
    "treatment": "X",
    "outcome": "Y",
}


# =============================================================================
# Identifiable Effect Tests
# =============================================================================


class TestIdentifiableEffects:
    """Tests for identifiable effects."""

    def test_simple_identifiable_effect(self, client, auth_headers):
        """Simple direct effect should be identifiable with actionable status."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=IDENTIFIABLE_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["identifiability"]["identifiable"] is True
        assert data["identifiability"]["method"] == "backdoor"
        assert data["identifiability"]["adjustment_set"] == []
        assert data["recommendation_status"] == "actionable"
        assert data["recommendation_caveat"] is None
        assert "price → revenue" == data["identifiability"]["effect"]

    def test_backdoor_identifiable_effect(self, client, auth_headers):
        """Effect with confounder should be identifiable via adjustment."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=BACKDOOR_IDENTIFIABLE_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["identifiability"]["identifiable"] is True
        assert data["identifiability"]["method"] == "backdoor"
        assert "market_segment" in data["identifiability"]["adjustment_set"]
        assert data["recommendation_status"] == "actionable"

    def test_identifiable_has_high_confidence(self, client, auth_headers):
        """Identifiable effects should have high confidence."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=IDENTIFIABLE_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["identifiability"]["confidence"] == "high"


# =============================================================================
# Non-Identifiable Effect Tests (Hard Rule)
# =============================================================================


class TestNonIdentifiableEffects:
    """Tests for non-identifiable effects and hard rule enforcement."""

    def test_no_path_exploratory_status(self, client, auth_headers):
        """No causal path should result in exploratory status."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=NO_PATH_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["identifiability"]["identifiable"] is False
        # HARD RULE: Non-identifiable → exploratory
        assert data["recommendation_status"] == "exploratory"
        assert data["recommendation_caveat"] is not None

    def test_missing_decision_exploratory_status(self, client, auth_headers):
        """Missing decision node should result in exploratory status."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=MISSING_DECISION_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["identifiability"]["identifiable"] is False
        assert data["recommendation_status"] == "exploratory"

    def test_non_identifiable_has_suggestions(self, client, auth_headers):
        """Non-identifiable effects should provide suggestions."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=NO_PATH_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["suggestions"] is not None
        assert len(data["suggestions"]) > 0

        # Check suggestion structure
        for suggestion in data["suggestions"]:
            assert "description" in suggestion
            assert "priority" in suggestion

    def test_non_identifiable_caveat_contains_warning(self, client, auth_headers):
        """Caveat should warn about exploratory nature."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=NO_PATH_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        caveat = data["recommendation_caveat"].lower()
        # Should contain warning language
        assert (
            "exploratory" in caveat or
            "hypothes" in caveat or
            "cannot" in caveat or
            "confirm" in caveat or
            "causal" in caveat or  # "no causal connection"
            "meaningless" in caveat or  # for no-path case
            "connection" in caveat  # for no-path case
        )


# =============================================================================
# Simple DAG Format Tests
# =============================================================================


class TestSimpleDAGFormat:
    """Tests for /identifiability/dag endpoint."""

    def test_dag_identifiable(self, client, auth_headers):
        """DAG format should work for identifiable effects."""
        response = client.post(
            "/api/v1/analysis/identifiability/dag",
            json=SIMPLE_DAG_REQUEST,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["identifiability"]["identifiable"] is True
        assert data["identifiability"]["method"] == "backdoor"
        assert "Z" in data["identifiability"]["adjustment_set"]
        assert data["recommendation_status"] == "actionable"
        assert data["identifiability"]["effect"] == "X → Y"

    def test_dag_no_path(self, client, auth_headers):
        """DAG format with no path should return exploratory."""
        response = client.post(
            "/api/v1/analysis/identifiability/dag",
            json=SIMPLE_DAG_NO_PATH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["identifiability"]["identifiable"] is False
        assert data["recommendation_status"] == "exploratory"

    def test_dag_invalid_treatment(self, client, auth_headers):
        """Invalid treatment node should return 422."""
        invalid_request = {
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
            "treatment": "Z",  # Not in nodes
            "outcome": "Y",
        }

        response = client.post(
            "/api/v1/analysis/identifiability/dag",
            json=invalid_request,
            headers=auth_headers,
        )

        assert response.status_code == 422

    def test_dag_invalid_outcome(self, client, auth_headers):
        """Invalid outcome node should return 422."""
        invalid_request = {
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
            "treatment": "X",
            "outcome": "Z",  # Not in nodes
        }

        response = client.post(
            "/api/v1/analysis/identifiability/dag",
            json=invalid_request,
            headers=auth_headers,
        )

        assert response.status_code == 422


# =============================================================================
# Override Tests
# =============================================================================


class TestNodeOverrides:
    """Tests for decision/goal node overrides."""

    def test_override_decision_node(self, client, auth_headers):
        """Can override decision node detection."""
        request = {
            **MISSING_DECISION_GRAPH,
            "decision_node_id": "factor",
        }

        response = client.post(
            "/api/v1/analysis/identifiability",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Should now analyze factor → goal
        assert "factor → goal" == data["identifiability"]["effect"]

    def test_override_goal_node(self, client, auth_headers):
        """Can override goal node detection."""
        graph = {
            "graph": {
                "nodes": [
                    {"id": "decision", "kind": "decision", "label": "Decision"},
                    {"id": "outcome", "kind": "outcome", "label": "Outcome"},
                ],
                "edges": [
                    {"from": "decision", "to": "outcome", "weight": 1.0},
                ],
            },
            "goal_node_id": "outcome",
        }

        response = client.post(
            "/api/v1/analysis/identifiability",
            json=graph,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "decision → outcome" == data["identifiability"]["effect"]


# =============================================================================
# Metadata Tests
# =============================================================================


class TestMetadata:
    """Tests for response metadata."""

    def test_response_has_metadata(self, client, auth_headers):
        """Response should include metadata."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=IDENTIFIABLE_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "metadata" in data
        assert data["metadata"]["algorithm"] == "y0_identification"

    def test_request_id_in_metadata(self, client, auth_headers):
        """Request ID should be in metadata."""
        headers = {**auth_headers, "X-Request-Id": "test-req-123"}

        response = client.post(
            "/api/v1/analysis/identifiability",
            json=IDENTIFIABLE_GRAPH,
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["request_id"] == "test-req-123"


# =============================================================================
# Schema Version Tests
# =============================================================================


class TestSchemaVersion:
    """Tests for schema versioning."""

    def test_response_has_schema_version(self, client, auth_headers):
        """Response should include schema version."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=IDENTIFIABLE_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["schema_version"] == "identifiability.v1"


# =============================================================================
# Backdoor Paths Tests
# =============================================================================


class TestBackdoorPaths:
    """Tests for backdoor path reporting."""

    def test_backdoor_paths_reported(self, client, auth_headers):
        """Backdoor paths should be reported when present."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=BACKDOOR_IDENTIFIABLE_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Should have backdoor paths
        assert data["backdoor_paths"] is not None
        assert len(data["backdoor_paths"]) > 0

    def test_no_backdoor_paths_when_none(self, client, auth_headers):
        """No backdoor paths when none exist."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=IDENTIFIABLE_GRAPH,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Should not have backdoor paths (or empty)
        assert data["backdoor_paths"] is None or len(data["backdoor_paths"]) == 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_graph_structure(self, client, auth_headers):
        """Invalid graph structure should return 422."""
        invalid_request = {
            "graph": {
                "nodes": [],  # Empty nodes
                "edges": [],
            }
        }

        response = client.post(
            "/api/v1/analysis/identifiability",
            json=invalid_request,
            headers=auth_headers,
        )

        assert response.status_code == 422

    def test_missing_graph(self, client, auth_headers):
        """Missing graph should return 422."""
        response = client.post(
            "/api/v1/analysis/identifiability",
            json={},
            headers=auth_headers,
        )

        assert response.status_code == 422

    def test_unauthorized_request(self, client):
        """Request without API key should return 401/403 (when auth enabled) or 200 (when auth disabled)."""
        import os
        response = client.post(
            "/api/v1/analysis/identifiability",
            json=IDENTIFIABLE_GRAPH,
        )

        # When ISL_AUTH_DISABLED=true, request succeeds
        # When auth is enabled, should be 401 or 403
        auth_disabled = os.environ.get("ISL_AUTH_DISABLED", "").lower() == "true"
        if auth_disabled:
            assert response.status_code == 200
        else:
            assert response.status_code in [401, 403]


# =============================================================================
# Real-World Scenario Tests
# =============================================================================


class TestRealWorldScenarios:
    """Tests with real-world-like scenarios."""

    def test_pricing_decision_identifiable(self, client, auth_headers):
        """Pricing decision with confounders should be identifiable."""
        request = {
            "graph": {
                "nodes": [
                    {"id": "price", "kind": "decision", "label": "Product Price"},
                    {"id": "revenue", "kind": "goal", "label": "Monthly Revenue"},
                    {"id": "market_segment", "kind": "factor", "label": "Market Segment"},
                    {"id": "competition", "kind": "factor", "label": "Competitor Pricing"},
                ],
                "edges": [
                    {"from": "price", "to": "revenue", "weight": 2.0},
                    {"from": "market_segment", "to": "price", "weight": 1.0},
                    {"from": "market_segment", "to": "revenue", "weight": 1.5},
                    {"from": "competition", "to": "revenue", "weight": -0.5},
                ],
            }
        }

        response = client.post(
            "/api/v1/analysis/identifiability",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["identifiability"]["identifiable"] is True
        assert "market_segment" in data["identifiability"]["adjustment_set"]
        assert data["recommendation_status"] == "actionable"

    def test_complex_decision_graph(self, client, auth_headers):
        """Complex decision graph with multiple paths."""
        request = {
            "graph": {
                "nodes": [
                    {"id": "investment", "kind": "decision", "label": "Investment Decision"},
                    {"id": "roi", "kind": "goal", "label": "Return on Investment"},
                    {"id": "market_conditions", "kind": "factor", "label": "Market Conditions"},
                    {"id": "team_capability", "kind": "factor", "label": "Team Capability"},
                    {"id": "product_quality", "kind": "outcome", "label": "Product Quality"},
                ],
                "edges": [
                    {"from": "investment", "to": "product_quality", "weight": 2.0},
                    {"from": "product_quality", "to": "roi", "weight": 2.5},
                    {"from": "market_conditions", "to": "investment", "weight": 1.0},
                    {"from": "market_conditions", "to": "roi", "weight": 1.5},
                    {"from": "team_capability", "to": "product_quality", "weight": 1.0},
                ],
            }
        }

        response = client.post(
            "/api/v1/analysis/identifiability",
            json=request,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Should be identifiable (has clear path through product_quality)
        assert data["identifiability"]["identifiable"] is True
        assert data["recommendation_status"] == "actionable"
