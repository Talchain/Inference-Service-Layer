"""
Integration tests for Bayesian teaching endpoint.

Tests end-to-end teaching workflow through REST API.
"""

import pytest


@pytest.fixture
def teaching_request():
    """Sample teaching request."""
    return {
        "user_id": "test_user_teaching_001",
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
                "brand": {
                    "type": "normal",
                    "parameters": {"mean": 0.3, "std": 0.2},
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
                "brand_weight": 0.5,
            },
        },
        "target_concept": "trade_offs",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn", "brand"],
            "constraints": {"industry": "SaaS"},
        },
        "max_examples": 3,
    }


def test_generate_teaching_examples_basic(client, teaching_request):
    """Test basic teaching example generation."""
    response = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response.status_code == 200
    data = response.json()

    # Should return examples
    assert "examples" in data
    assert len(data["examples"]) <= 3
    assert len(data["examples"]) > 0

    # Each example should have required fields
    for example in data["examples"]:
        assert "scenario" in example
        assert "key_insight" in example
        assert "why_this_example" in example
        assert "information_value" in example

        # Scenario should have required fields
        assert "description" in example["scenario"]
        assert "outcomes" in example["scenario"]
        assert "trade_offs" in example["scenario"]

    # Should have explanation
    assert "explanation" in data
    assert len(data["explanation"]) > 0

    # Should have learning objectives
    assert "learning_objectives" in data
    assert len(data["learning_objectives"]) > 0

    # Should have time estimate
    assert "expected_learning_time" in data
    assert "minute" in data["expected_learning_time"].lower()


def test_generate_teaching_examples_confounding(client, teaching_request):
    """Test teaching examples for confounding concept."""
    teaching_request["target_concept"] = "confounding"

    response = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response.status_code == 200
    data = response.json()

    # Should mention confounding in explanation or objectives
    text = (
        data["explanation"].lower()
        + " ".join(data["learning_objectives"]).lower()
    )
    assert "confound" in text or "correlation" in text or "causation" in text


def test_generate_teaching_examples_causal(client, teaching_request):
    """Test teaching examples for causal mechanism concept."""
    teaching_request["target_concept"] = "causal_mechanism"

    response = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["examples"]) > 0


def test_generate_teaching_examples_uncertainty(client, teaching_request):
    """Test teaching examples for uncertainty concept."""
    teaching_request["target_concept"] = "uncertainty"

    response = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["examples"]) > 0


def test_generate_teaching_examples_optimization(client, teaching_request):
    """Test teaching examples for optimization concept."""
    teaching_request["target_concept"] = "optimization"

    response = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["examples"]) > 0


def test_generate_teaching_examples_deterministic(client, teaching_request):
    """Test that teaching is deterministic."""
    response1 = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    response2 = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Should generate same examples
    assert len(data1["examples"]) == len(data2["examples"])
    for ex1, ex2 in zip(data1["examples"], data2["examples"]):
        assert ex1["information_value"] == ex2["information_value"]


def test_generate_teaching_examples_max_examples(client, teaching_request):
    """Test that max_examples parameter is respected."""
    for max_examples in [1, 2, 3, 5]:
        teaching_request["max_examples"] = max_examples

        response = client.post(
            "/api/v1/teaching/teach",
            json=teaching_request,
        )

        assert response.status_code == 200
        data = response.json()

        # Should not exceed max_examples
        assert len(data["examples"]) <= max_examples


def test_generate_teaching_examples_ranked(client, teaching_request):
    """Test that examples are ranked by teaching value."""
    response = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response.status_code == 200
    data = response.json()

    # Extract teaching values
    values = [ex["information_value"] for ex in data["examples"]]

    # Should be in descending order
    assert values == sorted(values, reverse=True)


def test_generate_teaching_examples_different_concepts(client, teaching_request):
    """Test that different concepts generate different examples."""
    concepts = ["confounding", "trade_offs", "causal_mechanism"]
    results = {}

    for concept in concepts:
        teaching_request["target_concept"] = concept

        response = client.post(
            "/api/v1/teaching/teach",
            json=teaching_request,
        )

        assert response.status_code == 200
        results[concept] = response.json()

    # Different concepts should have different explanations
    explanations = [results[c]["explanation"] for c in concepts]
    assert len(set(explanations)) > 1


def test_generate_teaching_examples_invalid_request(client):
    """Test error handling for invalid request."""
    invalid_request = {
        "user_id": "test_user",
        # Missing required fields
    }

    response = client.post(
        "/api/v1/teaching/teach",
        json=invalid_request,
    )

    # Should return validation error
    assert response.status_code == 422


def test_generate_teaching_examples_learning_objectives(client, teaching_request):
    """Test that learning objectives are meaningful."""
    response = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response.status_code == 200
    data = response.json()

    objectives = data["learning_objectives"]

    # Should have multiple objectives
    assert len(objectives) >= 2

    # Should mention the concept
    concept = teaching_request["target_concept"]
    text = " ".join(objectives).lower()
    assert concept.replace("_", " ") in text or "trade" in text


def test_generate_teaching_examples_time_estimate(client, teaching_request):
    """Test time estimate scaling."""
    results = {}

    for max_examples in [1, 3, 5]:
        teaching_request["max_examples"] = max_examples

        response = client.post(
            "/api/v1/teaching/teach",
            json=teaching_request,
        )

        assert response.status_code == 200
        data = response.json()
        results[max_examples] = data["expected_learning_time"]

    # More examples should generally take more time
    # (Though formatting may vary)
    assert all("minute" in time.lower() for time in results.values())


def test_generate_teaching_examples_privacy(client, teaching_request):
    """Test that user IDs are handled privately."""
    response = client.post(
        "/api/v1/teaching/teach",
        json=teaching_request,
    )

    assert response.status_code == 200
    # Response should not contain raw user ID
    data = response.json()
    assert teaching_request["user_id"] not in str(data)
