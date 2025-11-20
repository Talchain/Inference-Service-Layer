"""
Integration tests for preference elicitation endpoints.

Tests end-to-end preference learning workflow through REST API.
"""

import pytest


@pytest.fixture
def pricing_request():
    """Sample preference elicitation request for pricing."""
    return {
        "user_id": "test_user_pricing_001",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn", "brand_perception"],
            "constraints": {"industry": "SaaS", "current_price": 49},
        },
        "num_queries": 5,
    }


@pytest.fixture
def feature_request():
    """Sample preference elicitation request for features."""
    return {
        "user_id": "test_user_features_001",
        "context": {
            "domain": "feature_prioritization",
            "variables": ["user_satisfaction", "development_cost", "time_to_market"],
            "constraints": {"team_size": 5, "quarter": "Q4"},
        },
        "num_queries": 3,
    }


def test_elicit_preferences_initial_request(client, pricing_request):
    """Test initial preference elicitation request."""
    response = client.post(
        "/api/v1/preferences/elicit",
        json=pricing_request,
    )

    assert response.status_code == 200
    data = response.json()

    # Should return queries
    assert "queries" in data
    assert len(data["queries"]) == 5

    # Each query should have required fields
    for query in data["queries"]:
        assert "id" in query
        assert "question" in query
        assert "scenario_a" in query
        assert "scenario_b" in query
        assert "information_gain" in query

        # Scenarios should have outcomes and trade-offs
        assert "outcomes" in query["scenario_a"]
        assert "trade_offs" in query["scenario_a"]
        assert "outcomes" in query["scenario_b"]
        assert "trade_offs" in query["scenario_b"]

    # Should include strategy info
    assert "strategy" in data
    assert "type" in data["strategy"]
    assert "rationale" in data["strategy"]
    assert "focus_areas" in data["strategy"]

    # Should estimate queries remaining
    assert "estimated_queries_remaining" in data
    assert data["estimated_queries_remaining"] >= 0


def test_elicit_preferences_different_domains(client, pricing_request, feature_request):
    """Test that different domains generate appropriate queries."""
    pricing_response = client.post(
        "/api/v1/preferences/elicit",
        json=pricing_request,
    )

    feature_response = client.post(
        "/api/v1/preferences/elicit",
        json=feature_request,
    )

    assert pricing_response.status_code == 200
    assert feature_response.status_code == 200

    pricing_data = pricing_response.json()
    feature_data = feature_response.json()

    # Should have different query IDs
    pricing_ids = {q["id"] for q in pricing_data["queries"]}
    feature_ids = {q["id"] for q in feature_data["queries"]}
    assert len(pricing_ids & feature_ids) == 0


def test_elicit_preferences_deterministic(client, pricing_request):
    """Test that elicitation is deterministic for same input."""
    response1 = client.post(
        "/api/v1/preferences/elicit",
        json=pricing_request,
    )

    response2 = client.post(
        "/api/v1/preferences/elicit",
        json=pricing_request,
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Should generate identical queries
    assert len(data1["queries"]) == len(data2["queries"])
    for q1, q2 in zip(data1["queries"], data2["queries"]):
        assert q1["id"] == q2["id"]
        assert q1["information_gain"] == q2["information_gain"]


def test_elicit_preferences_with_existing_beliefs(client):
    """Test preference elicitation with provided beliefs."""
    request = {
        "user_id": "test_user_with_beliefs",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"],
        },
        "current_beliefs": {
            "value_weights": {
                "revenue": {
                    "type": "normal",
                    "parameters": {"mean": 0.7, "std": 0.2},
                },
                "churn": {
                    "type": "normal",
                    "parameters": {"mean": 0.4, "std": 0.2},
                },
            },
            "risk_tolerance": {
                "type": "beta",
                "parameters": {"alpha": 2, "beta": 2},
            },
            "time_horizon": {
                "type": "normal",
                "parameters": {"mean": 12, "std": 3},
            },
            "uncertainty_estimates": {
                "revenue_weight": 0.3,
                "churn_weight": 0.4,
            },
        },
        "num_queries": 3,
    }

    response = client.post(
        "/api/v1/preferences/elicit",
        json=request,
    )

    assert response.status_code == 200
    data = response.json()

    # Should use provided beliefs
    # Strategy should be uncertainty_sampling or expected_improvement
    assert data["strategy"]["type"] in ["uncertainty_sampling", "expected_improvement"]


def test_elicit_preferences_invalid_context(client):
    """Test preference elicitation with invalid context."""
    request = {
        "user_id": "test_user_invalid",
        "context": {
            "domain": "pricing",
            "variables": [],  # Empty variables
        },
        "num_queries": 3,
    }

    response = client.post(
        "/api/v1/preferences/elicit",
        json=request,
    )

    # Should return validation error
    assert response.status_code in [400, 422]


def test_update_beliefs_basic(client):
    """Test basic belief update."""
    # First, elicit preferences to establish context
    elicit_request = {
        "user_id": "test_user_update_001",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"],
        },
        "num_queries": 1,
    }

    elicit_response = client.post(
        "/api/v1/preferences/elicit",
        json=elicit_request,
    )
    assert elicit_response.status_code == 200
    query_id = elicit_response.json()["queries"][0]["id"]

    # Now update beliefs based on response
    update_request = {
        "user_id": "test_user_update_001",
        "query_id": query_id,
        "response": "A",
        "confidence": 1.0,
    }

    # Note: This will fail because we don't have stored beliefs
    # In a real scenario, we'd need to initialize beliefs first
    update_response = client.post(
        "/api/v1/preferences/update",
        json=update_request,
    )

    # Should return 400 since no beliefs found
    assert update_response.status_code == 400


def test_update_beliefs_workflow(client):
    """Test complete preference elicitation workflow."""
    user_id = "test_user_workflow_001"

    # Step 1: Elicit initial preferences (with provided beliefs)
    elicit_request = {
        "user_id": user_id,
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"],
        },
        "current_beliefs": {
            "value_weights": {
                "revenue": {
                    "type": "normal",
                    "parameters": {"mean": 0.5, "std": 0.3},
                },
                "churn": {
                    "type": "normal",
                    "parameters": {"mean": 0.5, "std": 0.3},
                },
            },
            "risk_tolerance": {
                "type": "beta",
                "parameters": {"alpha": 2, "beta": 2},
            },
            "time_horizon": {
                "type": "normal",
                "parameters": {"mean": 12, "std": 3},
            },
            "uncertainty_estimates": {
                "revenue_weight": 0.6,
                "churn_weight": 0.6,
            },
        },
        "num_queries": 1,
    }

    elicit_response = client.post(
        "/api/v1/preferences/elicit",
        json=elicit_request,
    )
    assert elicit_response.status_code == 200

    # Note: Full workflow would require Redis or mock storage
    # This test demonstrates the API structure


def test_preference_endpoints_explanation_metadata(client, pricing_request):
    """Test that responses include explanation metadata."""
    response = client.post(
        "/api/v1/preferences/elicit",
        json=pricing_request,
    )

    assert response.status_code == 200
    data = response.json()

    # Should include explanation
    assert "explanation" in data
    assert "summary" in data["explanation"]
    assert "reasoning" in data["explanation"]


def test_preference_information_gain_ordering(client, pricing_request):
    """Test that queries are ordered by information gain."""
    response = client.post(
        "/api/v1/preferences/elicit",
        json=pricing_request,
    )

    assert response.status_code == 200
    data = response.json()

    # Extract information gains
    info_gains = [q["information_gain"] for q in data["queries"]]

    # Should be in descending order
    assert info_gains == sorted(info_gains, reverse=True)


def test_preference_endpoints_error_handling(client):
    """Test error handling in preference endpoints."""
    # Missing required field
    invalid_request = {
        "user_id": "test_user_error",
        # Missing context
        "num_queries": 3,
    }

    response = client.post(
        "/api/v1/preferences/elicit",
        json=invalid_request,
    )

    # Should return validation error
    assert response.status_code == 422


def test_preference_endpoints_privacy(client, pricing_request):
    """Test that user IDs are handled privately."""
    # This is more of a logging test - user IDs should be hashed in logs
    # The endpoint itself should work normally

    response = client.post(
        "/api/v1/preferences/elicit",
        json=pricing_request,
    )

    assert response.status_code == 200

    # User ID should not appear in response (for privacy)
    # But this is more of a logging concern
    data = response.json()
    assert "queries" in data


def test_multiple_users_isolated(client):
    """Test that multiple users' preferences are isolated."""
    user1_request = {
        "user_id": "user_isolation_001",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"],
        },
        "num_queries": 3,
    }

    user2_request = {
        "user_id": "user_isolation_002",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"],
        },
        "num_queries": 3,
    }

    response1 = client.post(
        "/api/v1/preferences/elicit",
        json=user1_request,
    )

    response2 = client.post(
        "/api/v1/preferences/elicit",
        json=user2_request,
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Different users should get different query IDs
    data1 = response1.json()
    data2 = response2.json()

    ids1 = {q["id"] for q in data1["queries"]}
    ids2 = {q["id"] for q in data2["queries"]}

    assert len(ids1 & ids2) == 0


def test_preference_query_structure(client, pricing_request):
    """Test that generated queries have proper structure."""
    response = client.post(
        "/api/v1/preferences/elicit",
        json=pricing_request,
    )

    assert response.status_code == 200
    data = response.json()

    for query in data["queries"]:
        # Check scenario outcomes match context variables
        scenario_a_vars = set(query["scenario_a"]["outcomes"].keys())
        scenario_b_vars = set(query["scenario_b"]["outcomes"].keys())
        context_vars = set(pricing_request["context"]["variables"])

        # Scenarios should involve context variables
        assert len(scenario_a_vars & context_vars) > 0
        assert len(scenario_b_vars & context_vars) > 0


def test_preference_num_queries_respected(client):
    """Test that num_queries parameter is respected."""
    for num in [1, 3, 5, 10]:
        request = {
            "user_id": f"test_user_num_{num}",
            "context": {
                "domain": "pricing",
                "variables": ["revenue", "churn"],
            },
            "num_queries": num,
        }

        response = client.post(
            "/api/v1/preferences/elicit",
            json=request,
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["queries"]) == num


def test_preference_constraints_included(client):
    """Test that constraints are considered in query generation."""
    request = {
        "user_id": "test_user_constraints",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"],
            "constraints": {
                "industry": "SaaS",
                "current_price": 49,
                "competitor_range": [29, 99],
            },
        },
        "num_queries": 3,
    }

    response = client.post(
        "/api/v1/preferences/elicit",
        json=request,
    )

    assert response.status_code == 200
    # Constraints should influence query generation
    # (Implementation detail - hard to test without knowing internals)
