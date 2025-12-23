"""
API Key Authentication Middleware.

Provides X-API-Key header validation for protected endpoints.
"""

import hashlib
import logging
import os
import secrets
from typing import Optional, Set

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.models.responses import ErrorCode, ErrorResponse, RecoveryHints
from src.utils.ip_extraction import get_client_ip
from src.utils.secure_logging import get_security_audit_logger
from src.utils.tracing import get_trace_id

logger = logging.getLogger(__name__)
security_audit = get_security_audit_logger()


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate X-API-Key header against configured API keys.

    Public endpoints are exempt from authentication.
    Supports multiple API keys via comma-separated environment variable.
    Checks ISL_API_KEYS first, then falls back to ISL_API_KEY for backward compatibility.

    SECURITY: Authentication is enabled by default. Explicit opt-out via ISL_AUTH_DISABLED=true.
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

    def __init__(self, app, api_keys: Optional[str] = None, auth_disabled: bool = False):
        """
        Initialize the middleware.

        Args:
            app: The FastAPI application
            api_keys: Comma-separated list of valid API keys.
                     If None, reads from ISL_API_KEYS environment variable.
            auth_disabled: Explicitly disable authentication (for local development only)
        """
        super().__init__(app)
        self._auth_disabled = auth_disabled or os.getenv("ISL_AUTH_DISABLED", "false").lower() == "true"
        self._api_keys = self._load_api_keys(api_keys)
        self._auth_enabled = len(self._api_keys) > 0 and not self._auth_disabled

        if self._auth_disabled:
            logger.warning(
                "⚠️  API key authentication DISABLED via ISL_AUTH_DISABLED=true - "
                "NOT FOR PRODUCTION USE"
            )
        elif self._auth_enabled:
            logger.info(
                "API key authentication enabled",
                extra={"num_keys": len(self._api_keys)},
            )
        # Note: The case of "no keys + auth not disabled" is now handled at startup
        # in main.py with a RuntimeError, so middleware init only sees valid states.

    def _load_api_keys(self, api_keys: Optional[str]) -> Set[str]:
        """
        Load API keys from parameter or environment variable.

        Args:
            api_keys: Comma-separated API keys or None to read from env

        Returns:
            Set of valid API keys
        """
        # Check ISL_API_KEYS first, then fall back to ISL_API_KEY for backward compatibility
        keys_str = api_keys or os.getenv("ISL_API_KEYS") or os.getenv("ISL_API_KEY", "")
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

        # Skip authentication if explicitly disabled
        if self._auth_disabled:
            return await call_next(request)

        # Skip authentication if no API keys configured (development mode)
        if not self._auth_enabled:
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")

        client_ip = get_client_ip(request)

        # Get request ID for correlation
        request_id = request.headers.get("X-Request-Id") or request.headers.get("X-Trace-Id") or get_trace_id()

        # Validate API key
        if not api_key:
            # Log authentication failure via security audit logger
            security_audit.log_authentication_attempt(
                success=False,
                client_ip=client_ip,
                reason="missing_api_key",
                path=request.url.path,
            )
            error_response = ErrorResponse(
                code=ErrorCode.UNAUTHORIZED.value,
                message="Missing API key. Provide X-API-Key header.",
                reason="missing_api_key",
                recovery=RecoveryHints(
                    hints=[
                        "Include X-API-Key header in your request",
                        "Ensure API key is properly configured"
                    ],
                    suggestion="Provide X-API-Key header",
                ),
                retryable=False,
                source="isl",
                request_id=request_id,
            )
            return JSONResponse(
                status_code=401,
                content=error_response.model_dump(exclude_none=True),
            )

        # Use constant-time comparison to prevent timing attacks
        if not any(secrets.compare_digest(api_key, valid_key) for valid_key in self._api_keys):
            # Log authentication failure via security audit logger
            # Use hash prefix instead of raw key prefix for privacy
            key_hash_prefix = hashlib.sha256(api_key.encode()).hexdigest()[:8]
            security_audit.log_authentication_attempt(
                success=False,
                client_ip=client_ip,
                api_key_prefix=key_hash_prefix,
                reason="invalid_api_key",
                path=request.url.path,
            )
            error_response = ErrorResponse(
                code=ErrorCode.UNAUTHORIZED.value,
                message="Invalid API key.",
                reason="invalid_api_key",
                recovery=RecoveryHints(
                    hints=[
                        "Verify your API key is correct",
                        "Check if your API key has expired"
                    ],
                    suggestion="Check your API key",
                ),
                retryable=False,
                source="isl",
                request_id=request_id,
            )
            return JSONResponse(
                status_code=401,
                content=error_response.model_dump(exclude_none=True),
            )

        # API key is valid - log successful authentication
        # Use hash prefix instead of raw key prefix for privacy
        key_hash_prefix = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        security_audit.log_authentication_attempt(
            success=True,
            client_ip=client_ip,
            api_key_prefix=key_hash_prefix,
            path=request.url.path,
        )

        # Continue processing
        return await call_next(request)


def get_api_keys() -> Set[str]:
    """
    Get the configured API keys.

    Returns:
        Set of valid API keys from ISL_API_KEYS or ISL_API_KEY environment variable
    """
    # Check ISL_API_KEYS first, then fall back to ISL_API_KEY for backward compatibility
    keys_str = os.getenv("ISL_API_KEYS") or os.getenv("ISL_API_KEY", "")
    if not keys_str:
        return set()
    return {k.strip() for k in keys_str.split(",") if k.strip()}
