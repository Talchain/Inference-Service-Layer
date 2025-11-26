"""
API Key Authentication Middleware.

Provides X-API-Key header validation for protected endpoints.
"""

import logging
import os
from typing import Optional, Set

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.secure_logging import get_security_audit_logger

logger = logging.getLogger(__name__)
security_audit = get_security_audit_logger()


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate X-API-Key header against configured API keys.

    Public endpoints are exempt from authentication.
    Supports multiple API keys via comma-separated ISL_API_KEYS environment variable.
    """

    # Endpoints that don't require authentication
    PUBLIC_PATHS: Set[str] = {
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    # Prefixes that don't require authentication
    PUBLIC_PREFIXES: tuple = (
        "/docs",
        "/redoc",
    )

    def __init__(self, app, api_keys: Optional[str] = None):
        """
        Initialize the middleware.

        Args:
            app: The FastAPI application
            api_keys: Comma-separated list of valid API keys.
                     If None, reads from ISL_API_KEYS environment variable.
        """
        super().__init__(app)
        self._api_keys = self._load_api_keys(api_keys)
        self._auth_enabled = len(self._api_keys) > 0

        if self._auth_enabled:
            logger.info(
                "API key authentication enabled",
                extra={"num_keys": len(self._api_keys)},
            )
        else:
            logger.warning(
                "API key authentication DISABLED - no ISL_API_KEYS configured"
            )

    def _load_api_keys(self, api_keys: Optional[str]) -> Set[str]:
        """
        Load API keys from parameter or environment variable.

        Args:
            api_keys: Comma-separated API keys or None to read from env

        Returns:
            Set of valid API keys
        """
        keys_str = api_keys or os.getenv("ISL_API_KEYS", "")
        if not keys_str:
            return set()

        # Split by comma, strip whitespace, filter empty strings
        keys = {k.strip() for k in keys_str.split(",") if k.strip()}
        return keys

    def _is_public_path(self, path: str) -> bool:
        """
        Check if the path is a public endpoint.

        Args:
            path: The request path

        Returns:
            True if the path is public, False otherwise
        """
        # Exact match for public paths
        if path in self.PUBLIC_PATHS:
            return True

        # Prefix match for documentation paths
        if path.startswith(self.PUBLIC_PREFIXES):
            return True

        return False

    async def dispatch(self, request: Request, call_next):
        """
        Process the request with API key authentication.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            Response from next handler or 401 error
        """
        # Skip authentication for public endpoints
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Skip authentication if no API keys configured (development mode)
        if not self._auth_enabled:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")

        client_ip = self._get_client_ip(request)

        # Validate API key
        if not api_key:
            # Log authentication failure via security audit logger
            security_audit.log_authentication_attempt(
                success=False,
                client_ip=client_ip,
                reason="missing_api_key",
                path=request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "schema": "error.v1",
                    "code": "UNAUTHORIZED",
                    "message": "Missing API key. Provide X-API-Key header.",
                    "retryable": False,
                    "suggested_action": "provide_api_key",
                },
            )

        if api_key not in self._api_keys:
            # Log authentication failure via security audit logger
            security_audit.log_authentication_attempt(
                success=False,
                client_ip=client_ip,
                api_key_prefix=api_key[:8] if len(api_key) >= 8 else api_key,
                reason="invalid_api_key",
                path=request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "schema": "error.v1",
                    "code": "UNAUTHORIZED",
                    "message": "Invalid API key.",
                    "retryable": False,
                    "suggested_action": "check_api_key",
                },
            )

        # API key is valid - log successful authentication
        security_audit.log_authentication_attempt(
            success=True,
            client_ip=client_ip,
            api_key_prefix=api_key[:8] if len(api_key) >= 8 else api_key,
            path=request.url.path,
        )

        # Continue processing
        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address, respecting proxy headers.

        Args:
            request: The incoming request

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header first (set by proxies/load balancers)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2
            # The first IP is the original client
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header (set by nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"


def get_api_keys() -> Set[str]:
    """
    Get the configured API keys.

    Returns:
        Set of valid API keys from ISL_API_KEYS environment variable
    """
    keys_str = os.getenv("ISL_API_KEYS", "")
    if not keys_str:
        return set()
    return {k.strip() for k in keys_str.split(",") if k.strip()}
