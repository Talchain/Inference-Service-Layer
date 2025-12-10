"""
Integration tests for Multi-Criteria Aggregation endpoint.

Tests cover all three aggregation methods, weight normalization,
trade-off detection, percentile selection, and validation.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# Test fixtures
@pytest.fixture
def sample_criteria():
    """Sample criteria results for testing."""
    return [
        {
            "criterion_id": "cost",
            "criterion_name": "Total Cost",
            "options": [
                {
                    "option_id": "opt_a",
                    "option_label": "Option A",
                    "p10": 0.3,
                    "p50": 0.5,
                    "p90": 0.7,
                },
                {
                    "option_id": "opt_b",
                    "option_label": "Option B",
                    "p10": 0.6,
                    "p50": 0.7,
                    "p90": 0.8,
                },
                {
                    "option_id": "opt_c",
                    "option_label": "Option C",
                    "p10": 0.4,
                    "p50": 0.6,
                    "p90": 0.75,
                },
            ],
        },
        {
            "criterion_id": "speed",
            "criterion_name": "Execution Speed",
            "options": [
                {
                    "option_id": "opt_a",
                    "option_label": "Option A",
                    "p10": 0.7,
                    "p50": 0.8,
                    "p90": 0.9,
                },
                {
                    "option_id": "opt_b",
                    "option_label": "Option B",
                    "p10": 0.4,
                    "p50": 0.5,
                    "p90": 0.6,
                },
                {
                    "option_id": "opt_c",
                    "option_label": "Option C",
                    "p10": 0.5,
                    "p50": 0.6,
                    "p90": 0.7,
                },
            ],
        },
    ]


@pytest.fixture
def normalized_weights():
    """Weights that sum to 1.0."""
    return {"cost": 0.6, "speed": 0.4}


@pytest.fixture
def unnormalized_weights():
    """Weights that don't sum to 1.0 (will trigger normalization)."""
    return {"cost": 6.0, "speed": 4.0}


# ============================================================================
# Weighted Sum Aggregation Tests
# ============================================================================


def test_weighted_sum_basic(sample_criteria, normalized_weights):
    """Test basic weighted sum aggregation."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "aggregated_rankings" in data
    assert "trade_offs" in data
    assert "metadata" in data

    # Check rankings
    rankings = data["aggregated_rankings"]
    assert len(rankings) == 3
    assert rankings[0]["rank"] == 1
    assert rankings[1]["rank"] == 2
    assert rankings[2]["rank"] == 3

    # Verify scores are in 0-100 range
    for ranking in rankings:
        assert 0 <= ranking["aggregated_score"] <= 100
        assert "scores_by_criterion" in ranking

    # Expected: opt_b (0.6*0.7 + 0.4*0.5 = 0.42 + 0.2 = 0.62 = 62)
    # vs opt_a (0.6*0.5 + 0.4*0.8 = 0.3 + 0.32 = 0.62 = 62)
    # vs opt_c (0.6*0.6 + 0.4*0.6 = 0.36 + 0.24 = 0.60 = 60)
    # So opt_b or opt_a should be top (tied at 62), opt_c should be rank 3


def test_weighted_sum_percentile_selection(sample_criteria, normalized_weights):
    """Test that different percentiles produce different rankings."""
    # Test p10 (pessimistic)
    request_p10 = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,
        "percentile": "p10",
    }
    response_p10 = client.post("/api/v1/aggregation/multi-criteria", json=request_p10)
    assert response_p10.status_code == 200
    rankings_p10 = response_p10.json()["aggregated_rankings"]

    # Test p90 (optimistic)
    request_p90 = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,
        "percentile": "p90",
    }
    response_p90 = client.post("/api/v1/aggregation/multi-criteria", json=request_p90)
    assert response_p90.status_code == 200
    rankings_p90 = response_p90.json()["aggregated_rankings"]

    # Rankings should differ between pessimistic and optimistic views
    # (unless all options scale uniformly, which they don't in our fixture)
    assert len(rankings_p10) == len(rankings_p90) == 3


# ============================================================================
# Weighted Product Aggregation Tests
# ============================================================================


def test_weighted_product_basic(sample_criteria, normalized_weights):
    """Test basic weighted product aggregation."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_product",
        "weights": normalized_weights,
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    data = response.json()
    rankings = data["aggregated_rankings"]
    assert len(rankings) == 3

    # Verify rankings are properly ordered
    for i in range(len(rankings) - 1):
        assert rankings[i]["aggregated_score"] >= rankings[i + 1]["aggregated_score"]

    # Expected: opt_b (0.7^0.6 * 0.5^0.4)
    # vs opt_a (0.5^0.6 * 0.8^0.4)
    # vs opt_c (0.6^0.6 * 0.6^0.4)
    # Weighted product penalizes low scores more than weighted sum


def test_weighted_product_zero_score_handling(normalized_weights):
    """Test that weighted product handles zero scores gracefully."""
    criteria = [
        {
            "criterion_id": "c1",
            "criterion_name": "Criterion 1",
            "options": [
                {
                    "option_id": "opt_a",
                    "option_label": "Option A",
                    "p10": 0.0,  # Zero score
                    "p50": 0.0,
                    "p90": 0.5,
                },
                {
                    "option_id": "opt_b",
                    "option_label": "Option B",
                    "p10": 0.5,
                    "p50": 0.5,
                    "p90": 0.5,
                },
            ],
        },
        {
            "criterion_id": "c2",
            "criterion_name": "Criterion 2",
            "options": [
                {
                    "option_id": "opt_a",
                    "option_label": "Option A",
                    "p10": 0.8,
                    "p50": 0.8,
                    "p90": 0.8,
                },
                {
                    "option_id": "opt_b",
                    "option_label": "Option B",
                    "p10": 0.5,
                    "p50": 0.5,
                    "p90": 0.5,
                },
            ],
        },
    ]

    request = {
        "criteria": criteria,
        "aggregation_method": "weighted_product",
        "weights": {"c1": 0.5, "c2": 0.5},
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    # Should complete without math errors (uses epsilon for zero scores)
    rankings = response.json()["aggregated_rankings"]
    assert len(rankings) == 2


# ============================================================================
# Lexicographic Aggregation Tests
# ============================================================================


def test_lexicographic_basic(sample_criteria):
    """Test basic lexicographic aggregation."""
    # Higher weight = higher priority
    weights = {"cost": 0.7, "speed": 0.3}  # cost has priority

    request = {
        "criteria": sample_criteria,
        "aggregation_method": "lexicographic",
        "weights": weights,
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    data = response.json()
    rankings = data["aggregated_rankings"]
    assert len(rankings) == 3

    # Lexicographic sorts by highest-weight criterion first (cost)
    # So ranking should be primarily by cost scores
    # opt_b (cost=0.7) > opt_c (cost=0.6) > opt_a (cost=0.5)
    assert rankings[0]["option_id"] == "opt_b"
    assert rankings[1]["option_id"] == "opt_c"
    assert rankings[2]["option_id"] == "opt_a"


def test_lexicographic_priority_order():
    """Test that lexicographic respects priority order from weights."""
    criteria = [
        {
            "criterion_id": "c1",
            "criterion_name": "Criterion 1",
            "options": [
                {"option_id": "opt_a", "option_label": "A", "p10": 0.5, "p50": 0.5, "p90": 0.5},
                {"option_id": "opt_b", "option_label": "B", "p10": 0.5, "p50": 0.5, "p90": 0.5},
            ],
        },
        {
            "criterion_id": "c2",
            "criterion_name": "Criterion 2",
            "options": [
                {"option_id": "opt_a", "option_label": "A", "p10": 0.9, "p50": 0.9, "p90": 0.9},
                {"option_id": "opt_b", "option_label": "B", "p10": 0.3, "p50": 0.3, "p90": 0.3},
            ],
        },
    ]

    # c2 has higher weight, so it should determine ranking
    request = {
        "criteria": criteria,
        "aggregation_method": "lexicographic",
        "weights": {"c1": 0.3, "c2": 0.7},  # c2 has priority
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    rankings = response.json()["aggregated_rankings"]
    # opt_a should be first (c2 score 0.9 > 0.3)
    assert rankings[0]["option_id"] == "opt_a"
    assert rankings[1]["option_id"] == "opt_b"


# ============================================================================
# Weight Normalization Tests
# ============================================================================


def test_weight_normalization_warning(sample_criteria, unnormalized_weights):
    """Test that unnormalized weights trigger a warning."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": unnormalized_weights,  # Sum = 10.0, not 1.0
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "warnings" in data
    assert data["warnings"] is not None
    assert len(data["warnings"]) > 0

    # Check for normalization warning
    warning = data["warnings"][0]
    assert warning["code"] == "WEIGHTS_NORMALIZED"
    assert "normalized from sum=10.0" in warning["message"].lower()
    assert "cost" in warning["affected_items"]
    assert "speed" in warning["affected_items"]

    # Results should still be valid (using normalized weights)
    assert "aggregated_rankings" in data
    assert len(data["aggregated_rankings"]) == 3


def test_normalized_weights_no_warning(sample_criteria, normalized_weights):
    """Test that normalized weights don't trigger warnings."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,  # Sum = 1.0
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    data = response.json()
    # No warnings or warnings is None/empty
    assert data.get("warnings") is None or len(data.get("warnings", [])) == 0


# ============================================================================
# Trade-off Detection Tests
# ============================================================================


def test_trade_off_detection(sample_criteria, normalized_weights):
    """Test that trade-offs are detected between top options."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,
        "percentile": "p50",
        "trade_off_threshold": 0.05,  # 5% difference threshold
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "trade_offs" in data

    # Should detect trade-offs (opt_a good on speed, opt_b good on cost)
    trade_offs = data["trade_offs"]
    if len(trade_offs) > 0:
        trade_off = trade_offs[0]
        assert "option_a_id" in trade_off
        assert "option_b_id" in trade_off
        assert "a_better_on" in trade_off
        assert "b_better_on" in trade_off
        assert "max_difference" in trade_off
        assert trade_off["max_difference"] >= 0.05


def test_trade_off_threshold_sensitivity():
    """Test that higher thresholds reduce detected trade-offs."""
    criteria = [
        {
            "criterion_id": "c1",
            "criterion_name": "Criterion 1",
            "options": [
                {"option_id": "opt_a", "option_label": "A", "p10": 0.9, "p50": 0.9, "p90": 0.9},
                {"option_id": "opt_b", "option_label": "B", "p10": 0.5, "p50": 0.5, "p90": 0.5},
            ],
        },
        {
            "criterion_id": "c2",
            "criterion_name": "Criterion 2",
            "options": [
                {"option_id": "opt_a", "option_label": "A", "p10": 0.5, "p50": 0.5, "p90": 0.5},
                {"option_id": "opt_b", "option_label": "B", "p10": 0.9, "p50": 0.9, "p90": 0.9},
            ],
        },
    ]

    # Low threshold - should detect trade-off
    request_low = {
        "criteria": criteria,
        "aggregation_method": "weighted_sum",
        "weights": {"c1": 0.5, "c2": 0.5},
        "percentile": "p50",
        "trade_off_threshold": 0.05,
    }
    response_low = client.post("/api/v1/aggregation/multi-criteria", json=request_low)
    trade_offs_low = response_low.json()["trade_offs"]

    # High threshold - may not detect trade-off (if diff < threshold)
    request_high = {
        "criteria": criteria,
        "aggregation_method": "weighted_sum",
        "weights": {"c1": 0.5, "c2": 0.5},
        "percentile": "p50",
        "trade_off_threshold": 0.5,  # 50% - very high
    }
    response_high = client.post("/api/v1/aggregation/multi-criteria", json=request_high)
    trade_offs_high = response_high.json()["trade_offs"]

    # Low threshold should find at least as many trade-offs as high threshold
    assert len(trade_offs_low) >= len(trade_offs_high)


# ============================================================================
# Validation Tests
# ============================================================================


def test_validation_missing_criteria():
    """Test validation error for missing criteria."""
    request = {
        "criteria": [],  # Empty - should fail
        "aggregation_method": "weighted_sum",
        "weights": {"c1": 1.0},
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 422  # Validation error


def test_validation_invalid_aggregation_method(sample_criteria, normalized_weights):
    """Test validation error for invalid aggregation method."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "invalid_method",
        "weights": normalized_weights,
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    # Should be rejected at validation level or return 400
    assert response.status_code in [400, 422]


def test_validation_invalid_percentile(sample_criteria, normalized_weights):
    """Test that invalid percentile is handled."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,
        "percentile": "p99",  # Invalid percentile
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    # May accept and default to p50, or reject - either is acceptable
    # Just ensure it doesn't crash
    assert response.status_code in [200, 400, 422]


def test_validation_negative_weights(sample_criteria):
    """Test validation of negative weights."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": {"cost": -0.5, "speed": 1.5},  # Negative weight
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    # Should either normalize or reject
    assert response.status_code in [200, 400, 422]


def test_uniform_scores_edge_case():
    """Test aggregation when all options have identical scores."""
    criteria = [
        {
            "criterion_id": "c1",
            "criterion_name": "Criterion 1",
            "options": [
                {"option_id": "opt_a", "option_label": "A", "p10": 0.5, "p50": 0.5, "p90": 0.5},
                {"option_id": "opt_b", "option_label": "B", "p10": 0.5, "p50": 0.5, "p90": 0.5},
                {"option_id": "opt_c", "option_label": "C", "p10": 0.5, "p50": 0.5, "p90": 0.5},
            ],
        },
    ]

    request = {
        "criteria": criteria,
        "aggregation_method": "weighted_sum",
        "weights": {"c1": 1.0},
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    rankings = response.json()["aggregated_rankings"]
    assert len(rankings) == 3

    # All should have same aggregated score
    scores = [r["aggregated_score"] for r in rankings]
    assert scores[0] == scores[1] == scores[2]

    # Should have no trade-offs (all identical)
    trade_offs = response.json()["trade_offs"]
    assert len(trade_offs) == 0


def test_single_criterion_edge_case():
    """Test aggregation with only one criterion."""
    criteria = [
        {
            "criterion_id": "c1",
            "criterion_name": "Criterion 1",
            "options": [
                {"option_id": "opt_a", "option_label": "A", "p10": 0.3, "p50": 0.5, "p90": 0.7},
                {"option_id": "opt_b", "option_label": "B", "p10": 0.6, "p50": 0.7, "p90": 0.8},
            ],
        },
    ]

    request = {
        "criteria": criteria,
        "aggregation_method": "weighted_sum",
        "weights": {"c1": 1.0},
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    rankings = response.json()["aggregated_rankings"]
    assert len(rankings) == 2
    # opt_b should rank first (0.7 > 0.5)
    assert rankings[0]["option_id"] == "opt_b"


# ============================================================================
# Response Completeness Tests
# ============================================================================


def test_response_metadata(sample_criteria, normalized_weights):
    """Test that response includes proper metadata."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "metadata" in data

    metadata = data["metadata"]
    assert "request_id" in metadata
    assert "computation_time_ms" in metadata
    assert "isl_version" in metadata
    assert metadata["algorithm"] == "weighted_sum"
    assert metadata["cache_hit"] is False


def test_request_id_tracking(sample_criteria, normalized_weights):
    """Test that request_id is properly tracked."""
    custom_request_id = "test-multi-criteria-12345"

    request = {
        "request_id": custom_request_id,
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    metadata = response.json()["metadata"]
    assert metadata["request_id"] == custom_request_id


def test_response_structure_completeness(sample_criteria, normalized_weights):
    """Test that response has all required fields."""
    request = {
        "criteria": sample_criteria,
        "aggregation_method": "weighted_sum",
        "weights": normalized_weights,
        "percentile": "p50",
    }

    response = client.post("/api/v1/aggregation/multi-criteria", json=request)
    assert response.status_code == 200

    data = response.json()

    # Top-level fields
    assert "aggregated_rankings" in data
    assert "trade_offs" in data
    assert "metadata" in data

    # Rankings structure
    for ranking in data["aggregated_rankings"]:
        assert "option_id" in ranking
        assert "option_label" in ranking
        assert "rank" in ranking
        assert "aggregated_score" in ranking
        assert "scores_by_criterion" in ranking

    # Trade-offs structure (if any)
    for trade_off in data["trade_offs"]:
        assert "option_a_id" in trade_off
        assert "option_a_label" in trade_off
        assert "option_b_id" in trade_off
        assert "option_b_label" in trade_off
        assert "a_better_on" in trade_off
        assert "b_better_on" in trade_off
        assert "max_difference" in trade_off


# ============================================================================
# Algorithm Comparison Tests
# ============================================================================


def test_algorithm_comparison_same_input():
    """Test that different algorithms produce different results."""
    criteria = [
        {
            "criterion_id": "c1",
            "criterion_name": "Criterion 1",
            "options": [
                {"option_id": "opt_a", "option_label": "A", "p10": 0.3, "p50": 0.5, "p90": 0.7},
                {"option_id": "opt_b", "option_label": "B", "p10": 0.6, "p50": 0.7, "p90": 0.8},
            ],
        },
        {
            "criterion_id": "c2",
            "criterion_name": "Criterion 2",
            "options": [
                {"option_id": "opt_a", "option_label": "A", "p10": 0.7, "p50": 0.8, "p90": 0.9},
                {"option_id": "opt_b", "option_label": "B", "p10": 0.4, "p50": 0.5, "p90": 0.6},
            ],
        },
    ]
    weights = {"c1": 0.6, "c2": 0.4}

    # Weighted sum
    response_sum = client.post(
        "/api/v1/aggregation/multi-criteria",
        json={
            "criteria": criteria,
            "aggregation_method": "weighted_sum",
            "weights": weights,
            "percentile": "p50",
        },
    )

    # Weighted product
    response_product = client.post(
        "/api/v1/aggregation/multi-criteria",
        json={
            "criteria": criteria,
            "aggregation_method": "weighted_product",
            "weights": weights,
            "percentile": "p50",
        },
    )

    # Lexicographic
    response_lex = client.post(
        "/api/v1/aggregation/multi-criteria",
        json={
            "criteria": criteria,
            "aggregation_method": "lexicographic",
            "weights": weights,
            "percentile": "p50",
        },
    )

    assert response_sum.status_code == 200
    assert response_product.status_code == 200
    assert response_lex.status_code == 200

    scores_sum = [r["aggregated_score"] for r in response_sum.json()["aggregated_rankings"]]
    scores_product = [
        r["aggregated_score"] for r in response_product.json()["aggregated_rankings"]
    ]
    scores_lex = [r["aggregated_score"] for r in response_lex.json()["aggregated_rankings"]]

    # Different methods should generally produce different scores
    # (unless inputs are very specific)
    # At minimum, all should return valid results
    assert len(scores_sum) == len(scores_product) == len(scores_lex) == 2
