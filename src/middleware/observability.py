"""
Observability middleware for cross-service tracing and response verification.

Provides:
- Service identification headers (x-olumi-service, x-olumi-service-build)
- Request payload hash logging (x-olumi-payload-hash)
- Response hash generation (x-olumi-response-hash)
- Boundary logging for request/response tracing
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from src.config import GIT_COMMIT_SHORT
from src.utils.canonical_hash import canonical_json_hash
from src.utils.ip_extraction import get_client_ip
from src.utils.tracing import get_trace_id

logger = logging.getLogger(__name__)

# Service identification constants
SERVICE_NAME = "isl"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for observability headers and boundary logging.

    Adds headers:
    - x-olumi-service: Service name (e.g., "isl")
    - x-olumi-service-build: Git commit SHA (short form)
    - x-olumi-response-hash: SHA-256 of response body (JSON endpoints only)
    - x-olumi-trace-received: Echo of received request_id:payload_hash (for debugging)
    - x-olumi-downstream-calls: Empty (ISL is a leaf service)

    Logs:
    - boundary.request: Incoming request with payload hash and received_from_header flag
    - boundary.response: Outgoing response with response hash
    """

    # Paths exempt from response hashing (streaming, large responses, etc.)
    EXEMPT_PATHS = {
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with observability enhancements."""
        start_time = time.time()
        request_id = get_trace_id()

        # Extract received trace info from headers (for echo/debugging)
        received_request_id = request.headers.get("X-Request-Id")
        incoming_payload_hash = request.headers.get("x-olumi-payload-hash")
        caller_service = request.headers.get("x-olumi-caller-service")

        # Log boundary.request event with received_from_header flag
        self._log_boundary_request(
            request=request,
            request_id=request_id,
            payload_hash=incoming_payload_hash,
            received_from_header=incoming_payload_hash is not None,
            caller_service=caller_service,
        )

        # Process request
        response = await call_next(request)

        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000

        # Add service identification headers to all responses
        response.headers["x-olumi-service"] = SERVICE_NAME
        response.headers["x-olumi-service-build"] = GIT_COMMIT_SHORT

        # Add trace echo header for debugging (confirms what ISL received)
        trace_received = f"{received_request_id or 'none'}:{incoming_payload_hash or 'none'}"
        response.headers["x-olumi-trace-received"] = trace_received

        # Add downstream calls header (empty - ISL is a leaf service)
        response.headers["x-olumi-downstream-calls"] = ""

        # Calculate and add response hash for JSON responses
        # Must capture body from streaming response for hashing
        response_hash = None
        response_body = None

        if self._should_hash_response(request, response):
            # Consume the response body for hashing
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk

            # Compute canonical hash
            response_hash = self._compute_response_hash(response_body, request)
            if response_hash:
                response.headers["x-olumi-response-hash"] = response_hash

            # Recreate the response with the consumed body
            response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        # Log boundary.response event
        self._log_boundary_response(
            request=request,
            response=response,
            request_id=request_id,
            elapsed_ms=elapsed_ms,
            incoming_payload_hash=incoming_payload_hash,
            response_hash=response_hash,
        )

        return response

    def _should_hash_response(self, request: Request, response: Response) -> bool:
        """Check if response should be hashed."""
        # Skip exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return False

        # Only hash JSON responses
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return False

        return True

    def _compute_response_hash(
        self, body: bytes, request: Request
    ) -> Optional[str]:
        """Compute canonical hash of response body."""
        if not body:
            return None

        try:
            parsed = json.loads(body)
            return canonical_json_hash(parsed)
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse response JSON for hashing",
                extra={"error": str(e), "path": request.url.path}
            )
        except Exception as e:
            logger.warning(
                "Failed to compute response hash",
                extra={"error": str(e), "path": request.url.path}
            )
        return None

    def _log_boundary_request(
        self,
        request: Request,
        request_id: str,
        payload_hash: Optional[str],
        received_from_header: bool = False,
        caller_service: Optional[str] = None,
    ) -> None:
        """Log boundary.request event for incoming requests."""
        # Use unified IP extraction for consistency with rate limiting/auth
        client_ip = get_client_ip(request)

        log_extra = {
            "event": "boundary.request",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "request_id": request_id,
            "service": SERVICE_NAME,
            "endpoint": request.url.path,
            "method": request.method,
            "client": client_ip,
            "build": GIT_COMMIT_SHORT,
        }

        # Include payload hash if provided by caller
        if payload_hash:
            log_extra["payload_hash"] = payload_hash
            log_extra["received_from_header"] = received_from_header

        # Include caller service if provided
        if caller_service:
            log_extra["caller_service"] = caller_service

        # Include content length if present
        content_length = request.headers.get("content-length")
        if content_length:
            log_extra["content_length"] = int(content_length)

        logger.info("boundary.request", extra=log_extra)

    def _log_boundary_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        elapsed_ms: float,
        incoming_payload_hash: Optional[str],
        response_hash: Optional[str],
    ) -> None:
        """Log boundary.response event for outgoing responses."""
        log_extra = {
            "event": "boundary.response",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "request_id": request_id,
            "service": SERVICE_NAME,
            "endpoint": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
            "build": GIT_COMMIT_SHORT,
        }

        # Include hashes for verification
        if incoming_payload_hash:
            log_extra["request_payload_hash"] = incoming_payload_hash
        if response_hash:
            log_extra["response_hash"] = response_hash

        logger.info("boundary.response", extra=log_extra)
