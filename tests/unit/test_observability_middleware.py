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
from src.utils.tracing import generate_trace_id, trace_id_ctx


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


class TestTraceEcho:
    """Tests for trace echo functionality."""

    @pytest.mark.asyncio
    async def test_dispatch_adds_trace_received_header(self):
        """Test dispatch adds x-olumi-trace-received header."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/causal/validate"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: {
                "X-Request-Id": "7792611a-5e97-40c6-9293-d238f6e65268",
                "x-olumi-payload-hash": "def456789012",
            }.get(h)
        )

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

        # Check trace-received header
        assert "x-olumi-trace-received" in result.headers
        assert result.headers["x-olumi-trace-received"] == "7792611a-5e97-40c6-9293-d238f6e65268:def456789012"

    @pytest.mark.asyncio
    async def test_dispatch_trace_received_with_missing_headers(self):
        """Test trace-received header handles missing values gracefully."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/health"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(return_value=None)

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

        # Check trace-received header shows "none" for missing values
        assert result.headers["x-olumi-trace-received"] == "none:none"

    @pytest.mark.asyncio
    async def test_dispatch_adds_downstream_calls_header(self):
        """Test dispatch adds empty x-olumi-downstream-calls header."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/health"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(return_value=None)

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

        # ISL is a leaf service - no downstream calls
        assert result.headers["x-olumi-downstream-calls"] == ""


class TestReceivedFromHeaderLogging:
    """Tests for received_from_header field in boundary logs."""

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

    def test_received_from_header_true_when_hash_provided(self, middleware, mock_request, caplog):
        """Test received_from_header is True when payload hash comes from header."""
        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash="def456789012",
                    received_from_header=True,
                )

        record = caplog.records[0]
        assert record.__dict__.get("payload_hash") == "def456789012"
        assert record.__dict__.get("received_from_header") is True

    def test_received_from_header_not_present_when_no_hash(self, middleware, mock_request, caplog):
        """Test received_from_header is not logged when no payload hash."""
        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash=None,
                    received_from_header=False,
                )

        record = caplog.records[0]
        assert record.__dict__.get("payload_hash") is None
        assert record.__dict__.get("received_from_header") is None

    def test_caller_service_logged_when_provided(self, middleware, mock_request, caplog):
        """Test caller_service is logged when provided."""
        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash="def456789012",
                    received_from_header=True,
                    caller_service="plot",
                )

        record = caplog.records[0]
        assert record.__dict__.get("caller_service") == "plot"

    @pytest.mark.asyncio
    async def test_dispatch_logs_received_from_header(self, caplog):
        """Test dispatch logs received_from_header field."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/causal/validate"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: {
                "X-Request-Id": "test-request-id",
                "x-olumi-payload-hash": "incoming_hash_123",
                "x-olumi-caller-service": "cee",
            }.get(h)
        )

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

        # Check that received_from_header and caller_service were logged
        boundary_request_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.request"
        ]
        assert len(boundary_request_logs) >= 1
        log_record = boundary_request_logs[0]
        assert log_record.__dict__.get("payload_hash") == "incoming_hash_123"
        assert log_record.__dict__.get("received_from_header") is True
        assert log_record.__dict__.get("caller_service") == "cee"


class TestRequestIdUniqueness:
    """Tests for request ID uniqueness per request."""

    def test_generate_trace_id_unique(self):
        """Test generate_trace_id produces unique IDs."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100, "All generated IDs should be unique"

    def test_generated_id_format(self):
        """Test generated ID follows req_{uuid16} format."""
        trace_id = generate_trace_id()
        assert trace_id.startswith("req_")
        assert len(trace_id) == 20  # "req_" (4) + 16 hex chars

    @pytest.mark.asyncio
    async def test_consecutive_requests_have_different_ids(self, caplog):
        """Test that consecutive requests get different request IDs."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        class MockHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        async def make_request():
            mock_request = MagicMock()
            mock_request.method = "GET"
            mock_request.url.path = "/api/v1/robustness/analyze/v2"
            mock_request.client.host = "127.0.0.1"
            mock_request.headers = MagicMock()
            mock_request.headers.get = MagicMock(return_value=None)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = MockHeaders()
            mock_response.body = None

            async def mock_call_next(req):
                return mock_response

            # Reset context to simulate new request
            trace_id_ctx.set(None)

            with patch("src.middleware.observability.get_client_ip", return_value="127.0.0.1"):
                with caplog.at_level(logging.INFO):
                    await middleware.dispatch(mock_request, mock_call_next)

        # Make two requests
        await make_request()
        await make_request()

        # Extract request IDs from logs
        boundary_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.request"
        ]
        assert len(boundary_logs) == 2

        id1 = boundary_logs[0].__dict__.get("request_id")
        id2 = boundary_logs[1].__dict__.get("request_id")

        assert id1 != id2, "Consecutive requests should have different request IDs"


class TestCorrelationId:
    """Tests for correlation_id from X-Request-Id header."""

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
        request.url.path = "/api/v1/robustness/analyze/v2"
        request.client.host = "192.168.1.1"
        request.headers = MagicMock()
        request.headers.get = MagicMock(return_value=None)
        return request

    def test_correlation_id_logged_when_header_provided(self, middleware, mock_request, caplog):
        """Test correlation_id is logged when X-Request-Id header is provided."""
        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash=None,
                    correlation_id="caller-provided-id-123",
                )

        record = caplog.records[0]
        assert record.__dict__.get("correlation_id") == "caller-provided-id-123"

    def test_correlation_id_not_logged_when_no_header(self, middleware, mock_request, caplog):
        """Test correlation_id is not logged when X-Request-Id header is missing."""
        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash=None,
                    correlation_id=None,
                )

        record = caplog.records[0]
        assert record.__dict__.get("correlation_id") is None

    @pytest.mark.asyncio
    async def test_dispatch_logs_correlation_id(self, caplog):
        """Test dispatch logs correlation_id from X-Request-Id header."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/robustness/analyze/v2"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: {
                "X-Request-Id": "verify-logging-1736945012",
            }.get(h)
        )

        class MockHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = MockHeaders()
        mock_response.body = None

        async def mock_call_next(req):
            return mock_response

        with patch("src.middleware.observability.get_trace_id", return_value="req_internal123"):
            with patch("src.middleware.observability.get_client_ip", return_value="127.0.0.1"):
                with caplog.at_level(logging.INFO):
                    await middleware.dispatch(mock_request, mock_call_next)

        # Check boundary.request log has correlation_id
        request_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.request"
        ]
        assert len(request_logs) == 1
        assert request_logs[0].__dict__.get("correlation_id") == "verify-logging-1736945012"

        # Check boundary.response log also has correlation_id
        response_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.response"
        ]
        assert len(response_logs) == 1
        assert response_logs[0].__dict__.get("correlation_id") == "verify-logging-1736945012"


class TestSensitiveDataNotLogged:
    """Tests ensuring sensitive data is never logged."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance."""
        app = MagicMock()
        return ObservabilityMiddleware(app)

    def test_api_key_not_logged(self, middleware, caplog):
        """Test X-API-Key header is NOT logged."""
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/robustness/analyze/v2"
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: {
                "X-API-Key": "secret_api_key_123",
                "content-length": "1024",
            }.get(h)
        )

        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash=None,
                )

        record = caplog.records[0]
        log_dict = record.__dict__

        # Verify API key is NOT in any log field
        assert "api_key" not in log_dict
        assert "X-API-Key" not in log_dict
        for value in log_dict.values():
            if isinstance(value, str):
                assert "secret_api_key" not in value

    def test_authorization_header_not_logged(self, middleware, caplog):
        """Test Authorization header is NOT logged."""
        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/robustness/analyze/v2"
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(
            side_effect=lambda h: {
                "Authorization": "Bearer secret_token_xyz",
                "content-length": "1024",
            }.get(h)
        )

        with patch("src.middleware.observability.get_client_ip", return_value="192.168.1.1"):
            with caplog.at_level(logging.INFO):
                middleware._log_boundary_request(
                    request=mock_request,
                    request_id="req_abc123",
                    payload_hash=None,
                )

        record = caplog.records[0]
        log_dict = record.__dict__

        # Verify Authorization is NOT in any log field
        assert "authorization" not in str(log_dict).lower()
        assert "Bearer" not in str(log_dict)
        for value in log_dict.values():
            if isinstance(value, str):
                assert "secret_token" not in value


class TestExactlyOneLogPerRequest:
    """Tests ensuring exactly one boundary.request and one boundary.response per request."""

    @pytest.mark.asyncio
    async def test_single_boundary_request_log(self, caplog):
        """Test exactly one boundary.request log is emitted per request."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/robustness/analyze/v2"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(return_value=None)

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

        # Count boundary logs
        request_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.request"
        ]
        response_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.response"
        ]

        assert len(request_logs) == 1, "Exactly one boundary.request should be logged"
        assert len(response_logs) == 1, "Exactly one boundary.response should be logged"


class TestRobustnessRoutesCoverage:
    """Tests for middleware coverage of /robustness routes."""

    @pytest.mark.asyncio
    async def test_robustness_analyze_v2_logs_boundary(self, caplog):
        """Test /api/v1/robustness/analyze/v2 route emits boundary logs."""
        app = MagicMock()
        middleware = ObservabilityMiddleware(app)

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/robustness/analyze/v2"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = MagicMock()
        mock_request.headers.get = MagicMock(return_value=None)

        class MockHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = MockHeaders()
        mock_response.body = None

        async def mock_call_next(req):
            return mock_response

        with patch("src.middleware.observability.get_trace_id", return_value="req_robustness123"):
            with patch("src.middleware.observability.get_client_ip", return_value="127.0.0.1"):
                with caplog.at_level(logging.INFO):
                    await middleware.dispatch(mock_request, mock_call_next)

        # Verify boundary.request log
        request_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.request"
        ]
        assert len(request_logs) == 1
        assert request_logs[0].__dict__.get("endpoint") == "/api/v1/robustness/analyze/v2"
        assert request_logs[0].__dict__.get("method") == "POST"
        assert request_logs[0].__dict__.get("request_id") == "req_robustness123"

        # Verify boundary.response log
        response_logs = [
            r for r in caplog.records
            if r.__dict__.get("event") == "boundary.response"
        ]
        assert len(response_logs) == 1
        assert response_logs[0].__dict__.get("endpoint") == "/api/v1/robustness/analyze/v2"
        assert response_logs[0].__dict__.get("status") == 200
