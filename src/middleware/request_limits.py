"""
Request size limit and timeout middleware.

Provides protection against:
- Oversized request bodies (DoS via large payloads)
- Long-running requests (resource exhaustion)

Supports per-endpoint timeout configuration for computation-heavy operations.
"""

import asyncio
import logging
import re
from typing import Dict, Optional, Set

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings

logger = logging.getLogger(__name__)


# Per-endpoint timeout configuration (in seconds)
# Computation-heavy endpoints get longer timeouts
ENDPOINT_TIMEOUTS: Dict[str, int] = {
    # Validation endpoints - fast
    "/api/v1/validation/": 30,

    # Counterfactual endpoints - can be slow due to Monte Carlo
    "/api/v1/counterfactual/": 120,
    "/api/v1/counterfactual/batch": 180,

    # Sensitivity analysis - computation heavy
    "/api/v1/sensitivity/": 120,

    # Discovery endpoints - may involve LLM calls
    "/api/v1/discovery/": 90,

    # Explanations - moderate
    "/api/v1/explanations/": 60,
}


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.

    Rejects requests that exceed the configured maximum size.
    """

    # Endpoints exempt from size limits (typically file uploads, if any)
    EXEMPT_PATHS: Set[str] = set()

    def __init__(self, app, max_size_mb: int = None):
        """
        Initialize the middleware.

        Args:
            app: The FastAPI application
            max_size_mb: Maximum request body size in megabytes.
                        If None, reads from settings.
        """
        super().__init__(app)
        settings = get_settings()
        self.max_size_bytes = (max_size_mb or settings.MAX_REQUEST_SIZE_MB) * 1024 * 1024

        logger.info(
            "Request size limit configured",
            extra={"max_size_mb": self.max_size_bytes / (1024 * 1024)}
        )

    async def dispatch(self, request: Request, call_next):
        """
        Process request with size checking.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response (normal or 413 if too large)
        """
        # Skip check for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Check Content-Length header if present
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                length = int(content_length)
                if length > self.max_size_bytes:
                    logger.warning(
                        "Request too large (Content-Length)",
                        extra={
                            "path": request.url.path,
                            "content_length": length,
                            "max_size": self.max_size_bytes,
                        }
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "schema": "error.v1",
                            "code": "REQUEST_TOO_LARGE",
                            "message": f"Request body too large. Maximum size is {self.max_size_bytes / (1024 * 1024):.0f}MB.",
                            "retryable": False,
                            "suggested_action": "reduce_payload_size",
                        }
                    )
            except ValueError:
                pass  # Invalid Content-Length, let it through for validation

        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request timeout.

    Cancels requests that take longer than the configured timeout.
    Supports per-endpoint timeout configuration for computation-heavy operations.
    """

    # Endpoints exempt from timeout (for long-running operations)
    EXEMPT_PATHS: Set[str] = {"/health", "/ready", "/metrics"}

    def __init__(self, app, timeout_seconds: int = None, endpoint_timeouts: Dict[str, int] = None):
        """
        Initialize the middleware.

        Args:
            app: The FastAPI application
            timeout_seconds: Default maximum request processing time in seconds.
                           If None, reads from settings.
            endpoint_timeouts: Optional dict of endpoint patterns to timeout values.
                             Uses ENDPOINT_TIMEOUTS by default.
        """
        super().__init__(app)
        settings = get_settings()
        self.default_timeout = timeout_seconds or settings.REQUEST_TIMEOUT_SECONDS
        self.endpoint_timeouts = endpoint_timeouts or ENDPOINT_TIMEOUTS

        logger.info(
            "Request timeout configured",
            extra={
                "default_timeout_seconds": self.default_timeout,
                "endpoint_specific_timeouts": len(self.endpoint_timeouts),
            }
        )

    def _get_timeout_for_path(self, path: str) -> int:
        """
        Get the timeout for a specific path.

        Checks endpoint-specific timeouts first (longest prefix match),
        then falls back to default timeout.

        Args:
            path: Request path

        Returns:
            Timeout in seconds
        """
        # Find the longest matching prefix
        best_match = None
        best_match_len = 0

        for pattern, timeout in self.endpoint_timeouts.items():
            if path.startswith(pattern) and len(pattern) > best_match_len:
                best_match = timeout
                best_match_len = len(pattern)

        return best_match if best_match is not None else self.default_timeout

    async def dispatch(self, request: Request, call_next):
        """
        Process request with timeout enforcement.

        Uses per-endpoint timeouts for computation-heavy operations.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response (normal or 504 if timeout)
        """
        # Skip timeout for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Get endpoint-specific timeout
        timeout = self._get_timeout_for_path(request.url.path)

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                "Request timeout exceeded",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "timeout_seconds": timeout,
                    "is_endpoint_specific": timeout != self.default_timeout,
                }
            )
            return JSONResponse(
                status_code=504,
                content={
                    "schema": "error.v1",
                    "code": "REQUEST_TIMEOUT",
                    "message": f"Request processing timed out after {timeout} seconds.",
                    "retryable": True,
                    "suggested_action": "retry_with_simpler_input",
                }
            )
