"""
API Key Authentication Middleware.

Validates X-API-Key header against configured API keys from environment.
Supports multiple comma-separated API keys for different clients.

Example: ISL_API_KEY=plot_key,cee_key,ui_key
"""

import os
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API key authentication.

    Checks for X-API-Key header and validates against ISL_API_KEY environment variable.
    Supports comma-separated API keys for multiple clients (e.g., ISL_API_KEY=plot_key,cee_key).
    Public endpoints (/health, /ready, /metrics, /docs, /redoc, /openapi.json) are exempt.
    """

    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    def __init__(self, app):
        super().__init__(app)
        api_key_env = os.getenv("ISL_API_KEY")

        if not api_key_env:
            raise ValueError(
                "ISL_API_KEY environment variable must be set for API key authentication"
            )

        # Support comma-separated API keys for multiple clients
        # Example: ISL_API_KEY=plot_key,cee_key,ui_key
        self.valid_api_keys = {key.strip() for key in api_key_env.split(",") if key.strip()}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Validate API key for protected endpoints.

        Args:
            request: The incoming request
            call_next: The next middleware/route handler

        Returns:
            Response from the next handler or 401/403 error
        """
        # Allow public endpoints without authentication
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)

        # Allow OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Extract API key from header
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error_code": "AUTHENTICATION_REQUIRED",
                    "message": "Missing X-API-Key header",
                    "retryable": False,
                    "suggested_action": "include_api_key_header",
                },
            )

        # Validate API key (check if it's in the set of valid keys)
        if api_key not in self.valid_api_keys:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error_code": "INVALID_API_KEY",
                    "message": "Invalid API key",
                    "retryable": False,
                    "suggested_action": "use_valid_api_key",
                },
            )

        # API key is valid, proceed to next handler
        response = await call_next(request)
        return response
