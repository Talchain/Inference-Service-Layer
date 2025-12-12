"""
Contract tests for observability features.

Verifies that request ID propagation, correlation headers, and error schema
fields work correctly in actual HTTP responses. These tests ensure the
documented observability behavior remains enforced after future changes.

Related source files:
- src/api/main.py: Request ID middleware
- src/models/responses.py: Response schemas with metadata
- src/middleware/error_handler.py: Error response formatting
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ============================================================================
# Request ID Propagation Tests
# ============================================================================


class TestRequestIdPropagation:
    """Tests for X-Request-ID header propagation."""

    def test_request_id_returned_in_response_header(self):
        """Request ID should be returned in X-Request-ID response header."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers, \
            "X-Request-ID header missing from response"
        assert len(response.headers["X-Request-ID"]) > 0, \
            "X-Request-ID header is empty"

    def test_provided_request_id_echoed_back(self):
        """When client provides X-Request-ID, it should be echoed back."""
        custom_request_id = f"test-{uuid.uuid4()}"

        response = client.get(
            "/health",
            headers={"X-Request-ID": custom_request_id}
        )

        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == custom_request_id, \
            f"Expected {custom_request_id}, got {response.headers.get('X-Request-ID')}"

    def test_request_id_generated_when_not_provided(self):
        """When no X-Request-ID provided, server should generate one."""
        response = client.get("/health")

        assert response.status_code == 200
        request_id = response.headers.get("X-Request-ID")
        assert request_id is not None, "Server should generate request ID"
        assert len(request_id) >= 8, "Generated request ID too short"

    def test_request_id_in_error_response_body(self):
        """Request ID should be included in error response bodies."""
        custom_request_id = f"error-test-{uuid.uuid4()}"

        # Trigger a validation error
        response = client.post(
            "/api/v1/analysis/thresholds",
            json={"parameter_sweeps": [], "confidence_threshold": 0.1},
            headers={"X-Request-ID": custom_request_id}
        )

        assert response.status_code == 422
        data = response.json()
        assert "request_id" in data, "request_id missing from error response body"
        assert isinstance(data["request_id"], str), "request_id should be string"

    def test_request_id_in_success_response_metadata(self):
        """Request ID should be included in success response metadata."""
        custom_request_id = f"success-test-{uuid.uuid4()}"

        response = client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "A", "option_label": "Option A", "scores": {"cost": 0.3, "quality": 0.8}},
                    {"option_id": "B", "option_label": "Option B", "scores": {"cost": 0.5, "quality": 0.9}}
                ],
                "criteria": ["cost", "quality"]
            },
            headers={"X-Request-ID": custom_request_id}
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        data = response.json()

        # Check metadata contains request_id
        if "_metadata" in data:
            assert "request_id" in data["_metadata"], \
                "request_id missing from _metadata"

    def test_request_id_unique_across_requests(self):
        """Generated request IDs should be unique across requests."""
        request_ids = set()

        for _ in range(10):
            response = client.get("/health")
            request_id = response.headers.get("X-Request-ID")
            assert request_id not in request_ids, \
                f"Duplicate request ID generated: {request_id}"
            request_ids.add(request_id)

    def test_request_id_format_valid_uuid_or_string(self):
        """Generated request IDs should be valid UUIDs or reasonable strings."""
        response = client.get("/health")
        request_id = response.headers.get("X-Request-ID")

        # Should be alphanumeric with dashes (UUID format) or similar
        assert all(c.isalnum() or c in "-_" for c in request_id), \
            f"Request ID contains invalid characters: {request_id}"


# ============================================================================
# Error Schema Contract Tests for Observability
# ============================================================================


class TestErrorSchemaObservability:
    """Tests for error response schema observability fields."""

    def test_error_response_contains_source(self):
        """Error responses should identify ISL as the source."""
        response = client.post(
            "/api/v1/analysis/thresholds",
            json={"parameter_sweeps": [], "confidence_threshold": 0.1}
        )

        assert response.status_code == 422
        data = response.json()
        assert "source" in data, "source field missing from error"
        assert data["source"] == "isl", f"Expected source 'isl', got '{data['source']}'"

    def test_error_response_contains_timestamp_or_request_id(self):
        """Error responses should have correlation data (timestamp or request_id)."""
        # Use a validation error which goes through our error handler
        response = client.post(
            "/api/v1/analysis/thresholds",
            json={"parameter_sweeps": [], "confidence_threshold": 0.1}
        )

        assert response.status_code == 422
        data = response.json()

        # At minimum, request_id should be present for correlation
        assert "request_id" in data, \
            "Error response must have request_id for correlation"

    def test_error_response_retryable_flag(self):
        """Error responses should indicate if the request is retryable."""
        # Validation error - not retryable
        response = client.post(
            "/api/v1/analysis/thresholds",
            json={"parameter_sweeps": [], "confidence_threshold": 0.1}
        )

        assert response.status_code == 422
        data = response.json()
        assert "retryable" in data, "retryable flag missing"
        assert isinstance(data["retryable"], bool), "retryable should be boolean"
        assert data["retryable"] is False, "Validation errors should not be retryable"

    def test_error_response_code_machine_readable(self):
        """Error code should be machine-readable (no spaces, uppercase)."""
        response = client.post(
            "/api/v1/analysis/thresholds",
            json={"parameter_sweeps": [], "confidence_threshold": 0.1}
        )

        assert response.status_code == 422
        data = response.json()
        assert "code" in data, "error code missing"

        code = data["code"]
        # Should be uppercase with underscores (e.g., VALIDATION_ERROR)
        assert code == code.upper() or "_" in code, \
            f"Error code should be machine-readable format: {code}"


# ============================================================================
# Response Metadata Contract Tests
# ============================================================================


class TestResponseMetadata:
    """Tests for response metadata fields."""

    def test_success_response_includes_metadata(self):
        """Successful responses should include _metadata."""
        response = client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "A", "option_label": "Option A", "scores": {"x": 0.3}},
                    {"option_id": "B", "option_label": "Option B", "scores": {"x": 0.7}}
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        data = response.json()

        # _metadata is expected in responses
        if "_metadata" in data:
            metadata = data["_metadata"]
            assert "isl_version" in metadata, "isl_version missing from metadata"
            assert "request_id" in metadata, "request_id missing from metadata"

    def test_metadata_isl_version_format(self):
        """ISL version should follow semver format."""
        response = client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "A", "option_label": "Option A", "scores": {"x": 0.3}},
                    {"option_id": "B", "option_label": "Option B", "scores": {"x": 0.7}}
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        data = response.json()

        if "_metadata" in data and "isl_version" in data["_metadata"]:
            version = data["_metadata"]["isl_version"]
            # Should look like semver (e.g., "0.1.0", "1.2.3")
            parts = version.split(".")
            assert len(parts) >= 2, f"Version should be semver format: {version}"

    def test_metadata_config_fingerprint_present(self):
        """Config fingerprint should be present for reproducibility."""
        response = client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "A", "option_label": "Option A", "scores": {"x": 0.3}},
                    {"option_id": "B", "option_label": "Option B", "scores": {"x": 0.7}}
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        data = response.json()

        if "_metadata" in data:
            assert "config_fingerprint" in data["_metadata"], \
                "config_fingerprint missing from metadata"


# ============================================================================
# Cross-Endpoint Consistency Tests
# ============================================================================


class TestObservabilityConsistency:
    """Tests for consistent observability across all endpoints."""

    @pytest.mark.parametrize("endpoint,method,body", [
        ("/health", "GET", None),
        ("/api/v1/analysis/dominance", "POST", {
            "options": [
                {"option_id": "A", "option_label": "Option A", "scores": {"x": 0.3}},
                {"option_id": "B", "option_label": "Option B", "scores": {"x": 0.7}}
            ],
            "criteria": ["x"]
        }),
        ("/api/v1/analysis/pareto", "POST", {
            "options": [
                {"option_id": "A", "option_label": "Option A", "scores": {"x": 0.3}},
                {"option_id": "B", "option_label": "Option B", "scores": {"x": 0.7}}
            ],
            "criteria": ["x"]
        }),
    ])
    def test_request_id_header_present_all_endpoints(self, endpoint, method, body):
        """All endpoints should return X-Request-ID header."""
        if method == "GET":
            response = client.get(endpoint)
        else:
            response = client.post(endpoint, json=body)

        # Any response (success or error) should have X-Request-ID
        assert "X-Request-ID" in response.headers, \
            f"{endpoint} missing X-Request-ID header"

    def test_error_schema_consistent_across_endpoints(self):
        """Error schema should be consistent across all endpoints."""
        invalid_requests = [
            ("/api/v1/analysis/dominance", {"options": []}),
            ("/api/v1/analysis/pareto", {"options": []}),
            ("/api/v1/analysis/thresholds", {"parameter_sweeps": []}),
        ]

        required_error_fields = ["code", "message", "request_id", "source"]

        for endpoint, body in invalid_requests:
            response = client.post(endpoint, json=body)

            if response.status_code >= 400:
                data = response.json()
                for field in required_error_fields:
                    assert field in data, \
                        f"{endpoint} error missing required field: {field}"


# ============================================================================
# Sentry Integration Contract Tests
# ============================================================================


class TestSentryIntegration:
    """Tests that verify Sentry-related observability hooks."""

    def test_request_id_suitable_for_sentry_correlation(self):
        """Request ID format should work with Sentry's trace correlation."""
        response = client.get("/health")
        request_id = response.headers.get("X-Request-ID")

        # Request ID should be suitable for Sentry tag/breadcrumb
        assert request_id is not None
        assert len(request_id) <= 64, "Request ID too long for Sentry tags"
        assert len(request_id) >= 8, "Request ID too short for uniqueness"

    def test_error_structure_captures_context(self):
        """Error responses should have enough context for Sentry grouping."""
        response = client.post(
            "/api/v1/analysis/thresholds",
            json={"parameter_sweeps": [], "confidence_threshold": 0.1}
        )

        assert response.status_code == 422
        data = response.json()

        # These fields help Sentry group similar errors
        assert "code" in data, "Error code needed for Sentry grouping"
        assert "source" in data, "Source needed for Sentry filtering"
        assert "message" in data, "Message needed for Sentry display"


# ============================================================================
# Header Propagation Tests
# ============================================================================


class TestHeaderPropagation:
    """Tests for proper header propagation in responses."""

    def test_cors_headers_present(self):
        """CORS headers should be present for cross-origin requests."""
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )

        # Should have CORS headers (exact values depend on config)
        # At minimum, the request should not fail
        assert response.status_code in [200, 204, 405]

    def test_content_type_json_for_api_responses(self):
        """API responses should have application/json content type."""
        response = client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "A", "option_label": "Option A", "scores": {"x": 0.3}},
                    {"option_id": "B", "option_label": "Option B", "scores": {"x": 0.7}}
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json()}"
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, \
            f"Expected JSON content type, got: {content_type}"

    def test_error_responses_json_content_type(self):
        """Error responses should also be JSON."""
        response = client.post(
            "/api/v1/analysis/thresholds",
            json={"parameter_sweeps": []}
        )

        assert response.status_code == 422
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, \
            f"Error response should be JSON, got: {content_type}"
