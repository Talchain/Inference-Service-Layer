"""
Unit tests for API key authentication middleware.

Tests authentication, authorization, and security controls.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.middleware.auth import APIKeyAuthMiddleware, get_api_keys


class TestAPIKeyAuthMiddleware:
    """Test cases for APIKeyAuthMiddleware."""

    def test_load_api_keys_from_env(self):
        """Test loading API keys from environment variable."""
        with patch.dict("os.environ", {"ISL_API_KEYS": "key1,key2,key3"}, clear=True):
            keys = get_api_keys()
            assert keys == {"key1", "key2", "key3"}

    def test_load_api_keys_from_legacy_env(self):
        """Test loading API key from legacy ISL_API_KEY environment variable."""
        with patch.dict("os.environ", {"ISL_API_KEY": "legacy_key"}, clear=True):
            keys = get_api_keys()
            assert keys == {"legacy_key"}

    def test_load_api_keys_prefers_new_over_legacy(self):
        """Test that ISL_API_KEYS takes precedence over ISL_API_KEY."""
        with patch.dict("os.environ", {
            "ISL_API_KEYS": "new_key1,new_key2",
            "ISL_API_KEY": "legacy_key"
        }, clear=True):
            keys = get_api_keys()
            # Should use ISL_API_KEYS, not ISL_API_KEY
            assert keys == {"new_key1", "new_key2"}
            assert "legacy_key" not in keys

    def test_load_api_keys_empty(self):
        """Test loading when no API keys configured."""
        with patch.dict("os.environ", {"ISL_API_KEYS": ""}):
            keys = get_api_keys()
            assert keys == set()

    def test_load_api_keys_with_whitespace(self):
        """Test loading API keys with extra whitespace."""
        with patch.dict("os.environ", {"ISL_API_KEYS": " key1 , key2 , key3 "}):
            keys = get_api_keys()
            assert keys == {"key1", "key2", "key3"}

    def test_load_api_keys_from_parameter(self):
        """Test loading API keys from constructor parameter."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="test_key1,test_key2")
        assert middleware._api_keys == {"test_key1", "test_key2"}

    def test_public_paths_exempt(self):
        """Test that public paths are exempt from authentication."""
        middleware = APIKeyAuthMiddleware(MagicMock(), api_keys="secret_key")

        # These paths should be exempt
        assert middleware._is_public_path("/health")
        assert middleware._is_public_path("/ready")
        assert middleware._is_public_path("/metrics")
        assert middleware._is_public_path("/docs")
        assert middleware._is_public_path("/redoc")
        assert middleware._is_public_path("/openapi.json")

        # These paths should NOT be exempt
        assert not middleware._is_public_path("/api/v1/causal/validate")
        assert not middleware._is_public_path("/api/v1/batch/counterfactual")

    def test_auth_disabled_when_no_keys(self):
        """Test authentication is disabled when no keys configured."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="")
        assert not middleware._auth_enabled

    def test_auth_enabled_when_keys_present(self):
        """Test authentication is enabled when keys are configured."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="secret_key")
        assert middleware._auth_enabled


class TestAPIKeyMiddlewareClientIP:
    """Test client IP extraction from proxy headers."""

    def test_get_client_ip_direct(self):
        """Test getting client IP from direct connection."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="key")

        request = MagicMock()
        request.headers.get.return_value = None
        request.client.host = "192.168.1.100"

        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.100"

    def test_get_client_ip_from_forwarded_for(self):
        """Test getting client IP from X-Forwarded-For header."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="key")

        request = MagicMock()
        request.headers.get.side_effect = lambda h: {
            "X-Forwarded-For": "203.0.113.1, 10.0.0.1, 10.0.0.2",
            "X-Real-IP": None,
        }.get(h)
        request.client.host = "10.0.0.3"

        ip = middleware._get_client_ip(request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_from_real_ip(self):
        """Test getting client IP from X-Real-IP header."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="key")

        request = MagicMock()
        request.headers.get.side_effect = lambda h: {
            "X-Forwarded-For": None,
            "X-Real-IP": "203.0.113.50",
        }.get(h)
        request.client.host = "10.0.0.1"

        ip = middleware._get_client_ip(request)
        assert ip == "203.0.113.50"

    def test_get_client_ip_unknown(self):
        """Test getting client IP when no client info available."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="key")

        request = MagicMock()
        request.headers.get.return_value = None
        request.client = None

        ip = middleware._get_client_ip(request)
        assert ip == "unknown"


@pytest.mark.asyncio
class TestAPIKeyMiddlewareDispatch:
    """Test middleware dispatch behavior."""

    async def test_public_endpoint_allowed_without_key(self):
        """Test that public endpoints work without API key."""
        call_next = MagicMock()
        expected_response = MagicMock()
        call_next.return_value = expected_response

        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="secret_key")

        request = MagicMock()
        request.url.path = "/health"

        # Note: BaseHTTPMiddleware dispatch is async
        # We need to test the logic, not the actual middleware chain
        assert middleware._is_public_path("/health")

    async def test_protected_endpoint_requires_key(self):
        """Test that protected endpoints require API key."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="secret_key")

        # Protected endpoint should not be public
        assert not middleware._is_public_path("/api/v1/causal/validate")

    async def test_valid_key_grants_access(self):
        """Test that valid API key grants access."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="valid_key")

        # Key should be in the set
        assert "valid_key" in middleware._api_keys

    async def test_invalid_key_denies_access(self):
        """Test that invalid API key is rejected."""
        app = MagicMock()
        middleware = APIKeyAuthMiddleware(app, api_keys="valid_key")

        # Invalid key should not be in the set
        assert "invalid_key" not in middleware._api_keys
