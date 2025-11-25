"""
Request size limit and timeout middleware.

Provides protection against:
- Oversized request bodies (DoS via large payloads)
- Long-running requests (resource exhaustion)
"""

import asyncio
import logging
from typing import Set

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings

logger = logging.getLogger(__name__)


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
    """

    # Endpoints exempt from timeout (for long-running operations)
    EXEMPT_PATHS: Set[str] = {"/health", "/ready", "/metrics"}

    def __init__(self, app, timeout_seconds: int = None):
        """
        Initialize the middleware.

        Args:
            app: The FastAPI application
            timeout_seconds: Maximum request processing time in seconds.
                           If None, reads from settings.
        """
        super().__init__(app)
        settings = get_settings()
        self.timeout_seconds = timeout_seconds or settings.REQUEST_TIMEOUT_SECONDS

        logger.info(
            "Request timeout configured",
            extra={"timeout_seconds": self.timeout_seconds}
        )

    async def dispatch(self, request: Request, call_next):
        """
        Process request with timeout enforcement.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response (normal or 504 if timeout)
        """
        # Skip timeout for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(
                "Request timeout exceeded",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "timeout_seconds": self.timeout_seconds,
                }
            )
            return JSONResponse(
                status_code=504,
                content={
                    "schema": "error.v1",
                    "code": "REQUEST_TIMEOUT",
                    "message": f"Request processing timed out after {self.timeout_seconds} seconds.",
                    "retryable": True,
                    "suggested_action": "retry_with_simpler_input",
                }
            )
