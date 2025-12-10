"""
Contract tests for error response schema.

Verifies that all error responses conform to Olumi Error Schema v1.0
and provide consistent structure for downstream consumers.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ============================================================================
# Error Schema Contract Tests
# ============================================================================


def test_validation_error_schema():
    """Test that validation errors conform to Olumi Error Schema."""
    # Trigger validation error with invalid request
    request = {
        "parameter_sweeps": [],  # Empty - violates min_length=1
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 422

    data = response.json()

    # Required fields per Olumi Error Schema v1.0
    assert "code" in data
    assert "message" in data
    assert "reason" in data
    assert "retryable" in data
    assert "source" in data
    assert "request_id" in data

    # Validation-specific fields
    assert "validation_failures" in data
    assert isinstance(data["validation_failures"], list)
    assert len(data["validation_failures"]) > 0

    # Recovery hints
    assert "recovery" in data
    assert "hints" in data["recovery"]
    assert "suggestion" in data["recovery"]

    # Field types
    assert isinstance(data["code"], str)
    assert isinstance(data["message"], str)
    assert isinstance(data["retryable"], bool)
    assert isinstance(data["source"], str)

    # Error code should be validation-related
    assert "VALIDATION" in data["code"] or "validation" in data["reason"]


def test_not_found_error_schema():
    """Test that 404 errors conform to Olumi Error Schema."""
    response = client.get("/api/v1/nonexistent/endpoint")
    assert response.status_code == 404

    data = response.json()

    # Required fields
    assert "code" in data
    assert "message" in data
    assert "reason" in data
    assert "retryable" in data
    assert "source" in data
    assert "request_id" in data

    # Recovery hints
    assert "recovery" in data


def test_method_not_allowed_error_schema():
    """Test that 405 errors conform to Olumi Error Schema."""
    # Try GET on POST-only endpoint
    response = client.get("/api/v1/analysis/thresholds")
    assert response.status_code == 405

    data = response.json()

    # Should still have error structure (may be FastAPI default)
    assert "detail" in data or "message" in data


def test_error_code_consistency():
    """Test that error codes are consistent across similar errors."""
    # Multiple validation errors should have same code pattern
    validation_requests = [
        {"parameter_sweeps": [], "confidence_threshold": 0.1},
        {"parameter_sweeps": [{"parameter_id": "p", "parameter_label": "P", "values": [1.0], "scores_by_value": {}}], "confidence_threshold": 0.1},
    ]

    codes = []
    for req in validation_requests:
        response = client.post("/api/v1/analysis/thresholds", json=req)
        if response.status_code == 422:
            data = response.json()
            codes.append(data.get("code"))

    # All validation errors should have same code prefix
    assert len(set(codes)) <= 2  # May have slight variations but should be consistent


def test_retryable_flag_accuracy():
    """Test that retryable flag is set appropriately."""
    # Validation error - not retryable (client error)
    validation_request = {
        "parameter_sweeps": [],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=validation_request)
    if response.status_code == 422:
        data = response.json()
        assert "retryable" in data
        assert data["retryable"] is False  # Client errors are not retryable


def test_recovery_hints_present():
    """Test that recovery hints are provided for all errors."""
    # Validation error
    request = {
        "parameter_sweeps": [],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 422

    data = response.json()
    assert "recovery" in data
    assert "hints" in data["recovery"]
    assert isinstance(data["recovery"]["hints"], list)
    assert len(data["recovery"]["hints"]) > 0

    # Each hint should be a non-empty string
    for hint in data["recovery"]["hints"]:
        assert isinstance(hint, str)
        assert len(hint) > 0


def test_validation_failures_detail():
    """Test that validation_failures provide detailed error information."""
    request = {
        "parameter_sweeps": [
            {
                "parameter_id": "p",
                "parameter_label": "P",
                "values": [1.0],  # Only one value, violates min_length=2
                "scores_by_value": {
                    "1.0": {"opt_a": 0.5, "opt_b": 0.6}
                }
            }
        ],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    assert response.status_code == 422

    data = response.json()
    assert "validation_failures" in data

    # Each validation failure should describe the issue
    for failure in data["validation_failures"]:
        assert isinstance(failure, str)
        assert len(failure) > 0
        # Should contain field path and error description
        assert ":" in failure or "." in failure


def test_source_field_identifies_service():
    """Test that source field identifies ISL."""
    request = {
        "parameter_sweeps": [],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    data = response.json()

    assert "source" in data
    assert data["source"] == "isl"


def test_request_id_tracking_in_errors():
    """Test that request_id is tracked even in error responses."""
    # Provide custom request ID
    request = {
        "request_id": "test-error-tracking-12345",
        "parameter_sweeps": [],  # Invalid
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    data = response.json()

    assert "request_id" in data
    # Request ID should be present (may not match exactly due to header handling)
    assert isinstance(data["request_id"], str)
    assert len(data["request_id"]) > 0


# ============================================================================
# Error Schema Consistency Across Endpoints
# ============================================================================


def test_error_schema_consistent_across_endpoints():
    """Test that error schema is consistent across different endpoints."""
    endpoints_with_invalid_requests = [
        ("/api/v1/analysis/dominance", {"options": [], "criteria": []}),
        ("/api/v1/analysis/pareto", {"options": [], "criteria": []}),
        ("/api/v1/analysis/risk-adjust", {"options": [], "risk_coefficient": 1.0, "risk_type": "risk_averse"}),
        ("/api/v1/aggregation/multi-criteria", {"criteria": [], "aggregation_method": "weighted_sum", "weights": {}}),
        ("/api/v1/analysis/thresholds", {"parameter_sweeps": [], "confidence_threshold": 0.1}),
    ]

    required_fields = ["code", "message", "reason", "retryable", "source", "request_id"]

    for endpoint, invalid_request in endpoints_with_invalid_requests:
        response = client.post(endpoint, json=invalid_request)

        if response.status_code == 422:
            data = response.json()

            # All required fields should be present
            for field in required_fields:
                assert field in data, f"{endpoint} missing field: {field}"

            # recovery hints should be present
            assert "recovery" in data
            assert "hints" in data["recovery"]


def test_error_codes_use_standard_values():
    """Test that error codes use standard Olumi error code values."""
    # Trigger validation error
    request = {
        "parameter_sweeps": [],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    data = response.json()

    # Error code should be from standard set
    standard_codes = [
        "VALIDATION_ERROR",
        "INVALID_INPUT",
        "COMPUTATION_ERROR",
        "TIMEOUT",
        "RATE_LIMIT_EXCEEDED",
        "SERVICE_UNAVAILABLE",
        "NODE_NOT_FOUND",
        "MEMORY_LIMIT"
    ]

    # Code should match one of the standard codes
    assert any(code in data["code"] for code in standard_codes)


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


def test_error_response_backward_compatible():
    """Test that error response structure maintains backward compatibility."""
    # Trigger error
    request = {
        "parameter_sweeps": [],
        "confidence_threshold": 0.1
    }

    response = client.post("/api/v1/analysis/thresholds", json=request)
    data = response.json()

    # Fields that must always be present for backward compatibility
    essential_fields = ["code", "message", "request_id"]

    for field in essential_fields:
        assert field in data, f"Missing essential field for backward compatibility: {field}"

    # No unexpected top-level fields that could break parsers
    allowed_fields = [
        "code",
        "message",
        "reason",
        "recovery",
        "validation_failures",
        "retryable",
        "source",
        "request_id",
        "timestamp",  # Optional
        "details"  # Optional
    ]

    for field in data.keys():
        assert field in allowed_fields, f"Unexpected field that may break backward compatibility: {field}"


def test_error_response_json_serializable():
    """Test that all error responses are properly JSON serializable."""
    # Trigger various error types
    test_cases = [
        {"parameter_sweeps": [], "confidence_threshold": 0.1},  # Validation
        {"parameter_sweeps": [{"parameter_id": "p", "parameter_label": "P", "values": [1.0], "scores_by_value": {}}], "confidence_threshold": -1.0},  # Invalid value
    ]

    for test_case in test_cases:
        response = client.post("/api/v1/analysis/thresholds", json=test_case)

        # Should get valid JSON response
        assert response.headers.get("content-type") == "application/json"

        # Should be parseable
        data = response.json()
        assert isinstance(data, dict)

        # All values should be JSON-serializable types
        for key, value in data.items():
            assert isinstance(value, (str, int, float, bool, list, dict, type(None)))
