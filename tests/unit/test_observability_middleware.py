"""
Unit tests for observability middleware.

Tests service headers, payload hashing, and boundary logging.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import logging

from src.middleware.observability import (
    ObservabilityMiddleware,
    SERVICE_NAME,
)
from src.config import GIT_COMMIT_SHORT


class TestObservabilityMiddleware:
    """Test cases for ObservabilityMiddleware."""

    def test_service_name_constant(self):
        """Test service name is correctly defined."""
        assert SERVICE_NAME == "isl"

    def test_exempt_paths(self):
        """Test exempt paths are configured."""
        assert "/metrics" in ObservabilityMiddleware.EXEMPT_PATHS
        assert "/docs" in ObservabilityMiddleware.EXEMPT_PATHS
        assert "/redoc" in ObservabilityMiddleware.EXEMPT_PATHS
        assert "/openapi.json" in ObservabilityMiddleware.EXEMPT_PATHS


class TestBoundaryLogging:
    """Test cases for boundary logging functionality."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance."""
        app = MagicMock()
        return ObservabilityMiddleware(app)

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.method = "POST"
        request.url.path = "/api/v1/causal/validate"
        request.client.host = "192.168.1.1"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        return request

    def test_log_boundary_request_basic(self, middleware, mock_request, caplog):
        """Test boundary.request logging with basic info."""
        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash=None,
                )

        # Check log was emitted
        assert len(caplog.records) == 1
        record = caplog.records[0]

        # Check extra fields (canonical field names)
        assert record.__dict__.get("event") == "boundary.request"
        assert record.__dict__.get("timestamp") is not None  # ISO format timestamp
        assert record.__dict__.get("request_id") == "req_abc123"
        assert record.__dict__.get("service") == "isl"
        assert record.__dict__.get("endpoint") == "/api/v1/causal/validate"
        assert record.__dict__.get("method") == "POST"
        assert record.__dict__.get("client") == "192.168.1.1"

    def test_log_boundary_request_with_payload_hash(self, middleware, mock_request, caplog):
        """Test boundary.request includes payload hash when provided."""
        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash="abc123def456",
                )

        record = caplog.records[0]
        assert record.__dict__.get("payload_hash") == "abc123def456"

    def test_log_boundary_request_with_content_length(self, middleware, mock_request, caplog):
        """Test boundary.request includes content length when present."""
        mock_request.headers.get = MagicMock(side_effect=lambda h: {
            "content-length": "1024",
        }.get(h))

        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash=None,
                )

        record = caplog.records[0]
        assert record.__dict__.get("content_length") == 1024

    def test_log_boundary_response_basic(self, middleware, mock_request, caplog):
        """Test boundary.response logging with basic info."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with caplog.at_level(logging.INFO):
            middleware._log_boundary_response(
                request=mock_request,
                response=mock_response,
                request_id="req_abc123",
                elapsed_ms=45.67,
                incoming_payload_hash=None,
                response_hash=None,
            )

        record = caplog.records[0]
        # Check canonical field names
        assert record.__dict__.get("event") == "boundary.response"
        assert record.__dict__.get("timestamp") is not None  # ISO format timestamp
        assert record.__dict__.get("request_id") == "req_abc123"
        assert record.__dict__.get("service") == "isl"
        assert record.__dict__.get("endpoint") == "/api/v1/causal/validate"
        assert record.__dict__.get("method") == "POST"
        assert record.__dict__.get("status") == 200
        assert record.__dict__.get("elapsed_ms") == 45.67

    def test_log_boundary_response_with_hashes(self, middleware, mock_request, caplog):
        """Test boundary.response includes hashes when provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with caplog.at_level(logging.INFO):
            middleware._log_boundary_response(
                request=mock_request,
                response=mock_response,
                request_id="req_abc123",
                elapsed_ms=45.67,
                incoming_payload_hash="request_hash_123",
                response_hash="response_hash_456",
            )

        record = caplog.records[0]
        assert record.__dict__.get("request_payload_hash") == "request_hash_123"
        assert record.__dict__.get("response_hash") == "response_hash_456"


class TestResponseHashGeneration:
    """Test cases for response hash generation."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance."""
        app = MagicMock()
        return ObservabilityMiddleware(app)

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.url.path = "/api/v1/causal/validate"
        return request

    def test_should_hash_json_response(self, middleware, mock_request):
        """Test _should_hash_response returns True for JSON responses."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}

        result = middleware._should_hash_response(mock_request, mock_response)
        assert result is True

    def test_should_not_hash_exempt_paths(self, middleware):
        """Test _should_hash_response returns False for exempt paths."""
        mock_request = MagicMock()
        mock_request.url.path = "/metrics"

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}

        result = middleware._should_hash_response(mock_request, mock_response)
        assert result is False

    def test_should_not_hash_non_json(self, middleware, mock_request):
        """Test _should_hash_response returns False for non-JSON content."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}

        result = middleware._should_hash_response(mock_request, mock_response)
        assert result is False

    def test_compute_response_hash(self, middleware, mock_request):
        """Test _compute_response_hash computes canonical hash."""
        body = b'{"result": "success", "count": 42}'
        result = middleware._compute_response_hash(body, mock_request)

        assert result is not None
        assert len(result) == 64  # SHA-256 hex

        # Verify canonical hashing
        from src.utils.canonical_hash import canonical_json_hash
        expected_hash = canonical_json_hash({"result": "success", "count": 42})
        assert result == expected_hash

    def test_compute_response_hash_empty_body(self, middleware, mock_request):
        """Test _compute_response_hash returns None for empty body."""
        result = middleware._compute_response_hash(b"", mock_request)
        assert result is None

    def test_compute_response_hash_invalid_json(self, middleware, mock_request):
        """Test _compute_response_hash returns None for invalid JSON."""
        result = middleware._compute_response_hash(b"not json", mock_request)
        assert result is None


class TestMiddlewareIntegration:
    """Integration tests for middleware dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_adds_service_headers(self):
        """Test dispatch adds x-olumi-service and x-olumi-service-build headers."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/health"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(return_value=None)

        # Use a dict subclass that supports assignment like Starlette's MutableHeaders
        class MockHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = MockHeaders()
        mock_response.body = None

        async def mock_call_next(req):
            return mock_response

        with patch("src.middleware.observability.get_trace_id", return_value="req_test123"):
            with patch("src.middleware.observability.get_client_ip", return_value="127.0.0.1"):
                result = await middleware.dispatch(mock_request, mock_call_next)

        assert result.headers["x-olumi-service"] == "isl"
        assert result.headers["x-olumi-service-build"] == GIT_COMMIT_SHORT

    @pytest.mark.asyncio
    async def test_dispatch_logs_payload_hash_header(self, caplog):
        """Test dispatch logs incoming x-olumi-payload-hash header."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/causal/validate"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: {"x-olumi-payload-hash": "incoming_hash_123"}.get(h)
        )

        # Use a dict subclass that supports assignment
        class MockHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = MockHeaders()
        mock_response.body = None

        async def mock_call_next(req):
            return mock_response

        with patch("src.middleware.observability.get_trace_id", return_value="req_test123"):
            with patch("src.middleware.observability.get_client_ip", return_value="127.0.0.1"):
                with caplog.at_level(logging.INFO):
                    await middleware.dispatch(mock_request, mock_call_next)

        # Check that payload hash was logged
        boundary_request_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.request"
        ]
        assert len(boundary_request_logs) >= 1
        assert boundary_request_logs[0].__dict__.get("payload_hash") == "incoming_hash_123"


class TestServiceIdentification:
    """Tests for service identification headers."""

    def test_service_name_is_isl(self):
        """Test service name constant."""
        assert SERVICE_NAME == "isl"

    def test_git_commit_short_format(self):
        """Test Git commit short format."""
        # Should be either "unknown" or 7 characters
        assert GIT_COMMIT_SHORT == "unknown" or len(GIT_COMMIT_SHORT) == 7
