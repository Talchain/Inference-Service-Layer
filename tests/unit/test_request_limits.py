"""
Unit tests for request limits middleware.

Tests request size limits and timeout protection.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.middleware.request_limits import (
    RequestSizeLimitMiddleware,
    RequestTimeoutMiddleware,
)


class TestRequestSizeLimitMiddleware:
    """Test cases for RequestSizeLimitMiddleware."""

    def test_default_size_limit(self):
        """Test default size limit from settings."""
        with patch("src.middleware.request_limits.get_settings") as mock_settings:
            mock_settings.return_value.MAX_REQUEST_SIZE_MB = 10
            app = MagicMock()
            middleware = RequestSizeLimitMiddleware(app)

            # 10 MB = 10 * 1024 * 1024 bytes
            assert middleware.max_size_bytes == 10 * 1024 * 1024

    def test_custom_size_limit(self):
        """Test custom size limit from constructor."""
        with patch("src.middleware.request_limits.get_settings") as mock_settings:
            mock_settings.return_value.MAX_REQUEST_SIZE_MB = 10
            app = MagicMock()
            middleware = RequestSizeLimitMiddleware(app, max_size_mb=5)

            # 5 MB = 5 * 1024 * 1024 bytes
            assert middleware.max_size_bytes == 5 * 1024 * 1024

    def test_exempt_paths_exist(self):
        """Test that exempt paths set exists."""
        assert hasattr(RequestSizeLimitMiddleware, "EXEMPT_PATHS")
        assert isinstance(RequestSizeLimitMiddleware.EXEMPT_PATHS, set)


class TestRequestTimeoutMiddleware:
    """Test cases for RequestTimeoutMiddleware."""

    def test_default_timeout(self):
        """Test default timeout from settings."""
        with patch("src.middleware.request_limits.get_settings") as mock_settings:
            mock_settings.return_value.REQUEST_TIMEOUT_SECONDS = 60
            app = MagicMock()
            middleware = RequestTimeoutMiddleware(app)

            assert middleware.default_timeout == 60

    def test_custom_timeout(self):
        """Test custom timeout from constructor."""
        with patch("src.middleware.request_limits.get_settings") as mock_settings:
            mock_settings.return_value.REQUEST_TIMEOUT_SECONDS = 60
            app = MagicMock()
            middleware = RequestTimeoutMiddleware(app, timeout_seconds=30)

            assert middleware.default_timeout == 30

    def test_exempt_paths(self):
        """Test health/metrics endpoints are exempt from timeout."""
        assert "/health" in RequestTimeoutMiddleware.EXEMPT_PATHS
        assert "/ready" in RequestTimeoutMiddleware.EXEMPT_PATHS
        assert "/metrics" in RequestTimeoutMiddleware.EXEMPT_PATHS

    def test_endpoint_specific_timeouts(self):
        """Test endpoint-specific timeout configuration."""
        with patch("src.middleware.request_limits.get_settings") as mock_settings:
            mock_settings.return_value.REQUEST_TIMEOUT_SECONDS = 60
            app = MagicMock()
            middleware = RequestTimeoutMiddleware(app)

            # Validation should have shorter timeout
            assert middleware._get_timeout_for_path("/api/v1/validation/assumptions") == 30

            # Counterfactual should have longer timeout
            assert middleware._get_timeout_for_path("/api/v1/counterfactual/generate") == 120

            # Unknown path should use default
            assert middleware._get_timeout_for_path("/unknown/path") == 60


class TestRequestLimitsIntegration:
    """Integration tests for request limits."""

    @pytest.mark.asyncio
    async def test_size_limit_check_with_content_length(self):
        """Test size limit enforcement via Content-Length header."""
        with patch("src.middleware.request_limits.get_settings") as mock_settings:
            mock_settings.return_value.MAX_REQUEST_SIZE_MB = 1  # 1MB limit
            app = MagicMock()
            middleware = RequestSizeLimitMiddleware(app)

            # Create mock request with large Content-Length
            request = MagicMock()
            request.url.path = "/api/v1/causal/validate"
            request.headers.get.return_value = str(2 * 1024 * 1024)  # 2MB

            call_next = MagicMock()

            response = await middleware.dispatch(request, call_next)

            # Should return 413 error
            assert response.status_code == 413
            call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_size_limit_passes_valid_request(self):
        """Test that valid-sized requests pass through."""
        with patch("src.middleware.request_limits.get_settings") as mock_settings:
            mock_settings.return_value.MAX_REQUEST_SIZE_MB = 10
            app = MagicMock()
            middleware = RequestSizeLimitMiddleware(app)

            request = MagicMock()
            request.url.path = "/api/v1/causal/validate"
            request.headers.get.return_value = str(1024)  # 1KB - well under limit

            expected_response = MagicMock()

            async def mock_call_next(req):
                return expected_response

            response = await middleware.dispatch(request, mock_call_next)

            assert response == expected_response

    @pytest.mark.asyncio
    async def test_size_limit_skipped_for_no_content_length(self):
        """Test that requests without Content-Length pass through."""
        with patch("src.middleware.request_limits.get_settings") as mock_settings:
            mock_settings.return_value.MAX_REQUEST_SIZE_MB = 1
            app = MagicMock()
            middleware = RequestSizeLimitMiddleware(app)

            request = MagicMock()
            request.url.path = "/api/v1/causal/validate"
            request.headers.get.return_value = None  # No Content-Length

            expected_response = MagicMock()

            async def mock_call_next(req):
                return expected_response

            response = await middleware.dispatch(request, mock_call_next)

            assert response == expected_response
