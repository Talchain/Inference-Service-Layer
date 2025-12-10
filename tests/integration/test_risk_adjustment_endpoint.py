"""
Integration tests for Risk Adjustment endpoint.

Tests cover mean-variance risk adjustment for all three risk attitudes
(averse, neutral, seeking), both input formats (mean/std_dev and percentiles),
and edge cases.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# Test fixtures
@pytest.fixture
def mean_variance_options():
    """Sample options using mean/std_dev representation."""
    return [
        {
            "option_id": "opt_aggressive",
            "option_label": "Aggressive Growth",
            "mean": 0.80,
            "std_dev": 0.20  # High variance
        },
        {
            "option_id": "opt_conservative",
            "option_label": "Conservative Growth",
            "mean": 0.60,
            "std_dev": 0.05  # Low variance
        },
        {
            "option_id": "opt_balanced",
            "option_label": "Balanced Approach",
            "mean": 0.70,
            "std_dev": 0.15
        }
    ]


@pytest.fixture
def percentile_options():
    """Sample options using percentile representation."""
    return [
        {
            "option_id": "opt_a",
            "option_label": "Option A",
            "p10": 0.50,
            "p50": 0.80,
            "p90": 0.95  # Wide spread
        },
        {
            "option_id": "opt_b",
            "option_label": "Option B",
            "p10": 0.55,
            "p50": 0.60,
            "p90": 0.65  # Narrow spread
        }
    ]


# ============================================================================
# Risk Averse Tests
# ============================================================================


def test_risk_averse_basic(mean_variance_options):
    """Test basic risk averse adjustment."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "adjusted_scores" in data
    assert "rankings_changed" in data
    assert "risk_interpretation" in data
    assert "metadata" in data

    # Check adjusted scores
    scores = data["adjusted_scores"]
    assert len(scores) == 3

    # Risk aversion should penalize high-variance options
    # Conservative (mean=0.60, var=0.0025) should rank highest
    # Aggressive (mean=0.80, var=0.04) should be penalized heavily
    assert scores[0]["option_id"] == "opt_conservative"
    assert scores[0]["certainty_equivalent"] > scores[0]["original_score"] - 0.01  # Minimal penalty

    # Aggressive should have large negative adjustment
    aggressive = next(s for s in scores if s["option_id"] == "opt_aggressive")
    assert aggressive["adjustment"] < -0.01  # Significant penalty
    assert aggressive["variance"] == 0.04


def test_risk_averse_rankings_changed(mean_variance_options):
    """Test that risk aversion changes rankings."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    data = response.json()
    # Rankings should change (aggressive was best on mean, but penalized for variance)
    assert data["rankings_changed"] is True

    # Should have ranking change details
    assert "ranking_changes" in data
    assert data["ranking_changes"] is not None
    assert len(data["ranking_changes"]) > 0

    # Check that conservative improved
    conservative_change = next(
        (c for c in data["ranking_changes"]
         if c["option_id"] == "opt_conservative"),
        None
    )
    if conservative_change:
        assert conservative_change["rank_change"] > 0  # Improved


def test_risk_averse_high_coefficient(mean_variance_options):
    """Test that higher risk coefficient increases penalty."""
    # Low coefficient
    request_low = {
        "options": mean_variance_options,
        "risk_coefficient": 1.0,
        "risk_type": "risk_averse"
    }
    response_low = client.post("/api/v1/analysis/risk-adjust", json=request_low)
    scores_low = response_low.json()["adjusted_scores"]

    # High coefficient
    request_high = {
        "options": mean_variance_options,
        "risk_coefficient": 5.0,
        "risk_type": "risk_averse"
    }
    response_high = client.post("/api/v1/analysis/risk-adjust", json=request_high)
    scores_high = response_high.json()["adjusted_scores"]

    # Aggressive option should be penalized more with higher coefficient
    aggressive_low = next(s for s in scores_low if s["option_id"] == "opt_aggressive")
    aggressive_high = next(s for s in scores_high if s["option_id"] == "opt_aggressive")

    # Higher coefficient = more negative adjustment
    assert aggressive_high["adjustment"] < aggressive_low["adjustment"]


# ============================================================================
# Risk Neutral Tests
# ============================================================================


def test_risk_neutral_no_adjustment(mean_variance_options):
    """Test that risk neutral makes no adjustment."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 0.0,
        "risk_type": "risk_neutral"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    data = response.json()
    scores = data["adjusted_scores"]

    # All certainty equivalents should equal original scores
    for score in scores:
        assert abs(score["certainty_equivalent"] - score["original_score"]) < 1e-9
        assert abs(score["adjustment"]) < 1e-9

    # Rankings should not change
    assert data["rankings_changed"] is False
    assert data["ranking_changes"] is None


def test_risk_neutral_sorts_by_mean(mean_variance_options):
    """Test that risk neutral ranks by expected value only."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 0.0,
        "risk_type": "risk_neutral"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    scores = response.json()["adjusted_scores"]

    # Should be sorted by mean: aggressive (0.80) > balanced (0.70) > conservative (0.60)
    assert scores[0]["option_id"] == "opt_aggressive"
    assert scores[1]["option_id"] == "opt_balanced"
    assert scores[2]["option_id"] == "opt_conservative"


# ============================================================================
# Risk Seeking Tests
# ============================================================================


def test_risk_seeking_rewards_variance(mean_variance_options):
    """Test that risk seeking rewards high variance."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_seeking"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    data = response.json()
    scores = data["adjusted_scores"]

    # Aggressive (high variance) should get positive adjustment
    aggressive = next(s for s in scores if s["option_id"] == "opt_aggressive")
    assert aggressive["adjustment"] > 0.01  # Significant reward

    # Conservative (low variance) should get minimal adjustment
    conservative = next(s for s in scores if s["option_id"] == "opt_conservative")
    assert abs(conservative["adjustment"]) < 0.01


def test_risk_seeking_ranks_aggressive_highest(mean_variance_options):
    """Test that risk seeking prefers high-variance options."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_seeking"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    scores = response.json()["adjusted_scores"]

    # Aggressive should rank first (highest mean + variance reward)
    assert scores[0]["option_id"] == "opt_aggressive"


# ============================================================================
# Percentile Input Format Tests
# ============================================================================


def test_percentile_input_format(percentile_options):
    """Test that percentile input format works correctly."""
    request = {
        "options": percentile_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    data = response.json()
    scores = data["adjusted_scores"]
    assert len(scores) == 2

    # Should compute mean and variance from percentiles
    # opt_a: p50=0.80, wide spread (p90-p10=0.45)
    # opt_b: p50=0.60, narrow spread (p90-p10=0.10)

    # With risk aversion, opt_b (narrow spread) should rank higher despite lower p50
    assert scores[0]["option_id"] == "opt_b"

    # Variance should be reported
    for score in scores:
        assert "variance" in score
        if score["option_id"] == "opt_a":
            # Wide spread = higher variance
            assert score["variance"] is not None
            assert score["variance"] > 0.01


def test_mixed_input_formats():
    """Test that mixing mean/std_dev and percentile formats works."""
    options = [
        {
            "option_id": "opt_mean",
            "option_label": "Mean Format",
            "mean": 0.70,
            "std_dev": 0.10
        },
        {
            "option_id": "opt_percentile",
            "option_label": "Percentile Format",
            "p10": 0.50,
            "p50": 0.70,
            "p90": 0.90
        }
    ]

    request = {
        "options": options,
        "risk_coefficient": 1.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    scores = response.json()["adjusted_scores"]
    assert len(scores) == 2


# ============================================================================
# Validation Tests
# ============================================================================


def test_validation_requires_two_options():
    """Test that at least 2 options are required."""
    request = {
        "options": [
            {
                "option_id": "opt_single",
                "option_label": "Single Option",
                "mean": 0.70,
                "std_dev": 0.10
            }
        ],
        "risk_coefficient": 1.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 422  # Validation error


def test_validation_risk_neutral_requires_zero_coefficient():
    """Test that risk_neutral requires coefficient=0."""
    request = {
        "options": [
            {"option_id": "a", "option_label": "A", "mean": 0.5, "std_dev": 0.1},
            {"option_id": "b", "option_label": "B", "mean": 0.6, "std_dev": 0.1}
        ],
        "risk_coefficient": 2.0,  # Should be 0.0 for risk_neutral
        "risk_type": "risk_neutral"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 422


def test_validation_risk_averse_requires_positive_coefficient():
    """Test that risk_averse requires coefficient > 0."""
    request = {
        "options": [
            {"option_id": "a", "option_label": "A", "mean": 0.5, "std_dev": 0.1},
            {"option_id": "b", "option_label": "B", "mean": 0.6, "std_dev": 0.1}
        ],
        "risk_coefficient": 0.0,  # Should be > 0 for risk_averse
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 422


def test_validation_invalid_risk_type():
    """Test that invalid risk_type is rejected."""
    request = {
        "options": [
            {"option_id": "a", "option_label": "A", "mean": 0.5, "std_dev": 0.1},
            {"option_id": "b", "option_label": "B", "mean": 0.6, "std_dev": 0.1}
        ],
        "risk_coefficient": 1.0,
        "risk_type": "risk_confused"  # Invalid
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 422


def test_validation_incomplete_option():
    """Test that option must have either (mean, std_dev) or (p10, p50, p90)."""
    request = {
        "options": [
            {
                "option_id": "opt_incomplete",
                "option_label": "Incomplete",
                "mean": 0.70  # Missing std_dev
            },
            {
                "option_id": "opt_complete",
                "option_label": "Complete",
                "mean": 0.60,
                "std_dev": 0.05
            }
        ],
        "risk_coefficient": 1.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 422


def test_validation_score_bounds():
    """Test that scores must be in [0, 1] range."""
    request = {
        "options": [
            {
                "option_id": "opt_invalid",
                "option_label": "Invalid Score",
                "mean": 1.5,  # Out of range
                "std_dev": 0.1
            },
            {
                "option_id": "opt_valid",
                "option_label": "Valid Score",
                "mean": 0.6,
                "std_dev": 0.1
            }
        ],
        "risk_coefficient": 1.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 422


# ============================================================================
# Edge Cases
# ============================================================================


def test_zero_variance_options():
    """Test options with zero variance (certainty)."""
    options = [
        {
            "option_id": "opt_certain",
            "option_label": "Certain Option",
            "mean": 0.80,
            "std_dev": 0.0  # Zero variance
        },
        {
            "option_id": "opt_uncertain",
            "option_label": "Uncertain Option",
            "mean": 0.75,
            "std_dev": 0.15
        }
    ]

    request = {
        "options": options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    scores = response.json()["adjusted_scores"]

    # Certain option should have no adjustment
    certain = next(s for s in scores if s["option_id"] == "opt_certain")
    assert abs(certain["adjustment"]) < 1e-9

    # Certain option should rank higher despite lower mean
    assert scores[0]["option_id"] == "opt_certain"


def test_identical_means_different_variances(mean_variance_options):
    """Test options with same mean but different variances."""
    options = [
        {
            "option_id": "opt_low_var",
            "option_label": "Low Variance",
            "mean": 0.70,
            "std_dev": 0.05
        },
        {
            "option_id": "opt_high_var",
            "option_label": "High Variance",
            "mean": 0.70,
            "std_dev": 0.20
        }
    ]

    request = {
        "options": options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    scores = response.json()["adjusted_scores"]

    # Low variance should rank higher
    assert scores[0]["option_id"] == "opt_low_var"

    # Rankings should have changed (originally tied on mean)
    assert response.json()["rankings_changed"] is True


def test_certainty_equivalent_clamping():
    """Test that CE is clamped to [0, 1] range."""
    # Extreme case: high mean with very high variance and high risk coefficient
    options = [
        {
            "option_id": "opt_extreme",
            "option_label": "Extreme",
            "mean": 0.30,
            "std_dev": 0.40  # variance = 0.16
        },
        {
            "option_id": "opt_normal",
            "option_label": "Normal",
            "mean": 0.50,
            "std_dev": 0.10
        }
    ]

    request = {
        "options": options,
        "risk_coefficient": 10.0,  # Very high
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    scores = response.json()["adjusted_scores"]

    # All CEs should be in [0, 1]
    for score in scores:
        assert 0.0 <= score["certainty_equivalent"] <= 1.0


# ============================================================================
# Response Completeness Tests
# ============================================================================


def test_response_metadata(mean_variance_options):
    """Test that response includes proper metadata."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "metadata" in data

    metadata = data["metadata"]
    assert "request_id" in metadata
    assert "computation_time_ms" in metadata
    assert "isl_version" in metadata
    assert metadata["algorithm"] == "mean_variance_risk_averse"
    assert metadata["cache_hit"] is False


def test_request_id_tracking(mean_variance_options):
    """Test that request_id is properly tracked."""
    custom_request_id = "test-risk-12345"

    request = {
        "request_id": custom_request_id,
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    metadata = response.json()["metadata"]
    assert metadata["request_id"] == custom_request_id


def test_response_structure_completeness(mean_variance_options):
    """Test that response has all required fields."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    assert response.status_code == 200

    data = response.json()

    # Top-level fields
    assert "adjusted_scores" in data
    assert "rankings_changed" in data
    assert "risk_interpretation" in data
    assert "metadata" in data

    # Adjusted scores structure
    for score in data["adjusted_scores"]:
        assert "option_id" in score
        assert "option_label" in score
        assert "original_score" in score
        assert "certainty_equivalent" in score
        assert "adjustment" in score
        # variance is optional

    # Ranking changes (if rankings changed)
    if data["rankings_changed"]:
        assert "ranking_changes" in data
        assert data["ranking_changes"] is not None
        for change in data["ranking_changes"]:
            assert "option_id" in change
            assert "option_label" in change
            assert "original_rank" in change
            assert "adjusted_rank" in change
            assert "rank_change" in change


def test_interpretation_quality(mean_variance_options):
    """Test that interpretation provides meaningful information."""
    request = {
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    }

    response = client.post("/api/v1/analysis/risk-adjust", json=request)
    interpretation = response.json()["risk_interpretation"]

    # Should mention risk type
    assert "risk aversion" in interpretation.lower() or "averse" in interpretation.lower()

    # Should mention coefficient
    assert "2.0" in interpretation

    # Should be reasonably informative (not just a token string)
    assert len(interpretation) > 50


# ============================================================================
# Algorithm Comparison Tests
# ============================================================================


def test_risk_type_comparison_same_options(mean_variance_options):
    """Test that different risk types produce different results."""
    # Risk averse
    response_averse = client.post("/api/v1/analysis/risk-adjust", json={
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_averse"
    })

    # Risk neutral
    response_neutral = client.post("/api/v1/analysis/risk-adjust", json={
        "options": mean_variance_options,
        "risk_coefficient": 0.0,
        "risk_type": "risk_neutral"
    })

    # Risk seeking
    response_seeking = client.post("/api/v1/analysis/risk-adjust", json={
        "options": mean_variance_options,
        "risk_coefficient": 2.0,
        "risk_type": "risk_seeking"
    })

    scores_averse = response_averse.json()["adjusted_scores"]
    scores_neutral = response_neutral.json()["adjusted_scores"]
    scores_seeking = response_seeking.json()["adjusted_scores"]

    # Top-ranked option should differ
    top_averse = scores_averse[0]["option_id"]
    top_neutral = scores_neutral[0]["option_id"]
    top_seeking = scores_seeking[0]["option_id"]

    # Neutral should rank by mean (aggressive)
    assert top_neutral == "opt_aggressive"

    # Averse should rank conservative higher
    assert top_averse == "opt_conservative"

    # Seeking should rank aggressive even higher
    assert top_seeking == "opt_aggressive"
