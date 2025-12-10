"""
Integration tests for Threshold Identification endpoint.

Tests cover single/multiple thresholds, monotonic parameters, ties,
confidence thresholds, and sensitivity ranking.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# Test fixtures
@pytest.fixture
def single_threshold_sweep():
    """Parameter sweep with one ranking change."""
    return {
        "parameter_id": "price",
        "parameter_label": "Product Price",
        "values": [30.0, 40.0, 50.0, 60.0],
        "scores_by_value": {
            "30.0": {"opt_a": 0.90, "opt_b": 0.60},
            "40.0": {"opt_a": 0.85, "opt_b": 0.65},
            "50.0": {"opt_a": 0.70, "opt_b": 0.80},  # Ranking changes here
            "60.0": {"opt_a": 0.65, "opt_b": 0.85}
        }
    }


@pytest.fixture
def multiple_thresholds_sweep():
    """Parameter sweep with multiple ranking changes."""
    return {
        "parameter_id": "marketing",
        "parameter_label": "Marketing Spend",
        "values": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "scores_by_value": {
            "1000.0": {"opt_a": 0.90, "opt_b": 0.60, "opt_c": 0.50},
            "2000.0": {"opt_a": 0.80, "opt_b": 0.70, "opt_c": 0.55},  # a > b > c
            "3000.0": {"opt_a": 0.70, "opt_b": 0.85, "opt_c": 0.60},  # b > a > c (threshold 1)
            "4000.0": {"opt_a": 0.65, "opt_b": 0.90, "opt_c": 0.70},  # b > c > a (threshold 2)
            "5000.0": {"opt_a": 0.60, "opt_b": 0.95, "opt_c": 0.75}   # b > c > a
        }
    }


@pytest.fixture
def monotonic_sweep():
    """Parameter sweep with no ranking changes."""
    return {
        "parameter_id": "timeline",
        "parameter_label": "Project Timeline",
        "values": [6.0, 9.0, 12.0, 15.0],
        "scores_by_value": {
            "6.0": {"opt_a": 0.90, "opt_b": 0.70},
            "9.0": {"opt_a": 0.85, "opt_b": 0.75},
            "12.0": {"opt_a": 0.80, "opt_b": 0.78},
            "15.0": {"opt_a": 0.75, "opt_b": 0.79}
        }
    }


# ============================================================================
# Single Threshold Tests
# ============================================================================


def test_single_threshold_detection(single_threshold_sweep):
    """Test detection of a single ranking threshold."""
    request = {
        "parameter_sweeps": [single_threshold_sweep],
        "confidence_threshold": 0.05
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "thresholds" in data
    assert "sensitivity_ranking" in data
    assert "total_thresholds" in data
    assert "monotonic_parameters" in data
    assert "metadata" in data

    # Should find exactly one threshold
    thresholds = data["thresholds"]
    assert len(thresholds) == 1
    assert data["total_thresholds"] == 1

    # Check threshold details
    threshold = thresholds[0]
    assert threshold["parameter_id"] == "price"
    assert threshold["threshold_value"] == 50.0
    assert threshold["ranking_before"] == ["opt_a", "opt_b"]
    assert threshold["ranking_after"] == ["opt_b", "opt_a"]
    assert set(threshold["options_affected"]) == {"opt_a", "opt_b"}


def test_single_threshold_sensitivity(single_threshold_sweep):
    """Test sensitivity ranking for single threshold."""
    request = {
        "parameter_sweeps": [single_threshold_sweep],
        "confidence_threshold": 0.05
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    sensitivity = response.json()["sensitivity_ranking"]

    assert len(sensitivity) == 1
    param_sens = sensitivity[0]
    assert param_sens["parameter_id"] == "price"
    assert param_sens["changes_count"] == 1
    assert param_sens["most_sensitive_range"] == [50.0, 50.0]
    assert param_sens["sensitivity_score"] > 0.0


# ============================================================================
# Multiple Thresholds Tests
# ============================================================================


def test_multiple_thresholds_detection(multiple_thresholds_sweep):
    """Test detection of multiple ranking thresholds."""
    request = {
        "parameter_sweeps": [multiple_thresholds_sweep],
        "confidence_threshold": 0.05
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 200

    data = response.json()
    thresholds = data["thresholds"]

    # Should find 2 thresholds
    assert len(thresholds) >= 2
    assert data["total_thresholds"] >= 2

    # Check that thresholds are ordered by parameter value
    threshold_values = [t["threshold_value"] for t in thresholds]
    assert threshold_values == sorted(threshold_values)


def test_multiple_thresholds_affected_options(multiple_thresholds_sweep):
    """Test that affected options are correctly identified."""
    request = {
        "parameter_sweeps": [multiple_thresholds_sweep],
        "confidence_threshold": 0.05
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    thresholds = response.json()["thresholds"]

    # Each threshold should have affected options
    for threshold in thresholds:
        assert len(threshold["options_affected"]) > 0
        assert "score_gap" in threshold


# ============================================================================
# Monotonic (No Thresholds) Tests
# ============================================================================


def test_monotonic_no_thresholds(monotonic_sweep):
    """Test parameter with no ranking changes (monotonic)."""
    request = {
        "parameter_sweeps": [monotonic_sweep],
        "confidence_threshold": 0.05
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 200

    data = response.json()

    # Should find no thresholds
    assert len(data["thresholds"]) == 0
    assert data["total_thresholds"] == 0

    # Should be listed as monotonic
    assert "timeline" in data["monotonic_parameters"]

    # Sensitivity should be zero
    sensitivity = data["sensitivity_ranking"][0]
    assert sensitivity["parameter_id"] == "timeline"
    assert sensitivity["changes_count"] == 0
    assert sensitivity["most_sensitive_range"] is None
    assert sensitivity["sensitivity_score"] == 0.0


# ============================================================================
# Tie Handling Tests
# ============================================================================


def test_ties_with_confidence_threshold():
    """Test that ties are handled with confidence threshold."""
    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": [1.0, 2.0, 3.0],
        "scores_by_value": {
            "1.0": {"opt_a": 0.50, "opt_b": 0.51},  # Within 0.1 threshold
            "2.0": {"opt_a": 0.52, "opt_b": 0.53},  # Still within threshold
            "3.0": {"opt_a": 0.55, "opt_b": 0.70}   # Outside threshold
        }
    }

    request = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.1  # Large threshold - treats close scores as ties
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    thresholds = response.json()["thresholds"]

    # With large confidence threshold, should find fewer (or no) thresholds
    # because small differences are ignored
    assert len(thresholds) <= 1


def test_exact_ties_alphabetical_ordering():
    """Test that exact ties are ordered alphabetically."""
    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": [1.0, 2.0],
        "scores_by_value": {
            "1.0": {"opt_a": 0.80, "opt_b": 0.80, "opt_c": 0.80},  # All tied
            "2.0": {"opt_a": 0.75, "opt_b": 0.75, "opt_c": 0.75}   # All tied
        }
    }

    request = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.0
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    thresholds = response.json()["thresholds"]

    # No thresholds because ranking never changes (all tied)
    assert len(thresholds) == 0


# ============================================================================
# Multiple Parameters Tests
# ============================================================================


def test_multiple_parameters_sensitivity_ranking(
    single_threshold_sweep,
    multiple_thresholds_sweep,
    monotonic_sweep
):
    """Test sensitivity ranking across multiple parameters."""
    request = {
        "parameter_sweeps": [
            single_threshold_sweep,
            multiple_thresholds_sweep,
            monotonic_sweep
        ],
        "confidence_threshold": 0.05
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 200

    data = response.json()
    sensitivity = data["sensitivity_ranking"]

    # Should have 3 parameters
    assert len(sensitivity) == 3

    # Should be sorted by changes_count (descending)
    changes_counts = [s["changes_count"] for s in sensitivity]
    assert changes_counts == sorted(changes_counts, reverse=True)

    # Most sensitive should be marketing (multiple thresholds)
    assert sensitivity[0]["parameter_id"] == "marketing"
    assert sensitivity[0]["changes_count"] >= 2

    # Least sensitive should be timeline (monotonic)
    assert sensitivity[-1]["parameter_id"] == "timeline"
    assert sensitivity[-1]["changes_count"] == 0


def test_multiple_parameters_threshold_aggregation(
    single_threshold_sweep,
    multiple_thresholds_sweep
):
    """Test that thresholds from multiple parameters are aggregated."""
    request = {
        "parameter_sweeps": [
            single_threshold_sweep,
            multiple_thresholds_sweep
        ],
        "confidence_threshold": 0.05
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    data = response.json()

    # Total thresholds should be sum from both parameters
    total = data["total_thresholds"]
    assert total >= 3  # At least 1 from price + 2 from marketing


# ============================================================================
# Confidence Threshold Tests
# ============================================================================


def test_low_confidence_threshold_finds_more():
    """Test that lower confidence threshold finds more thresholds."""
    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": [1.0, 2.0, 3.0, 4.0],
        "scores_by_value": {
            "1.0": {"opt_a": 0.50, "opt_b": 0.48},
            "2.0": {"opt_a": 0.49, "opt_b": 0.52},  # Small swap
            "3.0": {"opt_a": 0.48, "opt_b": 0.53},
            "4.0": {"opt_a": 0.45, "opt_b": 0.60}   # Large difference
        }
    }

    # Low threshold - treats small differences as meaningful
    request_low = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.01
    }

    response_low = client.post("/api/v1/analysis/thresholds", json=request_low)
    thresholds_low = response_low.json()["thresholds"]

    # High threshold - ignores small differences
    request_high = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.2
    }

    response_high = client.post("/api/v1/analysis/thresholds", json=request_high)
    thresholds_high = response_high.json()["thresholds"]

    # Low threshold should find at least as many thresholds
    assert len(thresholds_low) >= len(thresholds_high)


# ============================================================================
# Validation Tests
# ============================================================================


def test_validation_requires_at_least_one_sweep():
    """Test that at least one parameter sweep is required."""
    request = {
        "parameter_sweeps": [],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 422


def test_validation_requires_at_least_two_values():
    """Test that parameter sweep needs at least 2 values."""
    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": [1.0],  # Only one value
        "scores_by_value": {
            "1.0": {"opt_a": 0.5, "opt_b": 0.6}
        }
    }

    request = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 422


def test_validation_scores_must_match_values():
    """Test that scores_by_value must have entries for all values."""
    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": [1.0, 2.0, 3.0],
        "scores_by_value": {
            "1.0": {"opt_a": 0.5, "opt_b": 0.6},
            "2.0": {"opt_a": 0.55, "opt_b": 0.65}
            # Missing 3.0
        }
    }

    request = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 422


def test_validation_consistent_options():
    """Test that all values must have scores for same options."""
    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": [1.0, 2.0],
        "scores_by_value": {
            "1.0": {"opt_a": 0.5, "opt_b": 0.6},
            "2.0": {"opt_a": 0.55, "opt_c": 0.65}  # Different option (c instead of b)
        }
    }

    request = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 422


# ============================================================================
# Response Completeness Tests
# ============================================================================


def test_response_metadata(single_threshold_sweep):
    """Test that response includes proper metadata."""
    request = {
        "parameter_sweeps": [single_threshold_sweep],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 200

    data = response.json()
    assert "metadata" in data

    metadata = data["metadata"]
    assert "request_id" in metadata
    assert "computation_time_ms" in metadata
    assert "isl_version" in metadata
    assert metadata["algorithm"] == "sequential_ranking_comparison"
    assert metadata["cache_hit"] is False


def test_request_id_tracking(single_threshold_sweep):
    """Test that request_id is properly tracked."""
    custom_request_id = "test-threshold-12345"

    request = {
        "request_id": custom_request_id,
        "parameter_sweeps": [single_threshold_sweep],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 200

    metadata = response.json()["metadata"]
    assert metadata["request_id"] == custom_request_id


def test_response_structure_completeness(single_threshold_sweep):
    """Test that response has all required fields."""
    request = {
        "parameter_sweeps": [single_threshold_sweep],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 200

    data = response.json()

    # Top-level fields
    assert "thresholds" in data
    assert "sensitivity_ranking" in data
    assert "total_thresholds" in data
    assert "monotonic_parameters" in data
    assert "metadata" in data

    # Threshold structure (if any thresholds)
    if data["thresholds"]:
        for threshold in data["thresholds"]:
            assert "parameter_id" in threshold
            assert "parameter_label" in threshold
            assert "threshold_value" in threshold
            assert "ranking_before" in threshold
            assert "ranking_after" in threshold
            assert "options_affected" in threshold

    # Sensitivity structure
    for sensitivity in data["sensitivity_ranking"]:
        assert "parameter_id" in sensitivity
        assert "parameter_label" in sensitivity
        assert "changes_count" in sensitivity
        assert "sensitivity_score" in sensitivity


# ============================================================================
# Edge Cases
# ============================================================================


def test_all_options_always_tied():
    """Test sweep where all options always have identical scores."""
    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": [1.0, 2.0, 3.0],
        "scores_by_value": {
            "1.0": {"opt_a": 0.5, "opt_b": 0.5, "opt_c": 0.5},
            "2.0": {"opt_a": 0.6, "opt_b": 0.6, "opt_c": 0.6},
            "3.0": {"opt_a": 0.7, "opt_b": 0.7, "opt_c": 0.7}
        }
    }

    request = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.0
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    data = response.json()

    # No thresholds because ranking never changes (always tied)
    assert len(data["thresholds"]) == 0
    assert data["total_thresholds"] == 0


def test_single_option():
    """Test sweep with only one option (trivial case)."""
    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": [1.0, 2.0, 3.0],
        "scores_by_value": {
            "1.0": {"opt_a": 0.5},
            "2.0": {"opt_a": 0.6},
            "3.0": {"opt_a": 0.7}
        }
    }

    request = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    data = response.json()

    # No thresholds possible with only one option
    assert len(data["thresholds"]) == 0


def test_large_parameter_sweep():
    """Test sweep with many values."""
    values = [float(i) for i in range(1, 101)]  # 100 values
    scores_by_value = {}

    for i, val in enumerate(values):
        # Create gradually changing scores that cause multiple thresholds
        score_a = 0.9 - (i * 0.008)  # Decreases
        score_b = 0.5 + (i * 0.004)  # Increases
        scores_by_value[str(val)] = {
            "opt_a": max(0.1, min(1.0, score_a)),
            "opt_b": max(0.1, min(1.0, score_b))
        }

    sweep = {
        "parameter_id": "param",
        "parameter_label": "Parameter",
        "values": values,
        "scores_by_value": scores_by_value
    }

    request = {
        "parameter_sweeps": [sweep],
        "confidence_threshold": 0.05
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 200

    # Should handle large sweep without issues
    data = response.json()
    assert "thresholds" in data
    assert "sensitivity_ranking" in data
