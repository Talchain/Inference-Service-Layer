"""
Operation-level latency tracing and distributed tracing for performance analysis.

Provides:
- Operation latency tracing
- Distributed trace ID generation and propagation
- Context-local trace storage
- Enriched logging with trace context
- Request ID sanitization (security)
"""

import logging
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator, Optional, Tuple

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Request ID validation constraints
REQUEST_ID_MAX_LENGTH = 128
REQUEST_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._:-]+$')

# Context-local trace ID storage
trace_id_ctx: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


@contextmanager
def trace_operation(operation_name: str, request_id: str) -> Generator[None, None, None]:
    """
    Context manager for tracing operation latency.

    Args:
        operation_name: Name of operation being traced
        request_id: Request ID for correlation

    Example:
        >>> with trace_operation("dag_validation", "req_123"):
        ...     validate_dag(dag)
    """
    start = time.time()

    try:
        yield
    finally:
        duration_ms = (time.time() - start) * 1000

        logger.info(
            f"Operation completed: {operation_name}",
            extra={
                "request_id": request_id,
                "operation": operation_name,
                "duration_ms": duration_ms,
                "trace_id": get_trace_id()
            }
        )


# Distributed Tracing Functions

def sanitize_request_id(request_id: Optional[str]) -> Tuple[str, bool]:
    """
    Sanitize an inbound request ID for security.

    Validates:
    - Not None/empty
    - Length <= REQUEST_ID_MAX_LENGTH (128)
    - Characters match REQUEST_ID_PATTERN (alphanumeric + . _ : -)

    Args:
        request_id: Inbound request ID (potentially untrusted)

    Returns:
        Tuple of (sanitized_id, was_valid):
        - If valid: returns (original_id, True)
        - If invalid: returns (new_generated_id, False)

    Security: Prevents log injection and header abuse from malformed IDs.
    """
    if not request_id:
        return generate_trace_id(), False

    # Check length
    if len(request_id) > REQUEST_ID_MAX_LENGTH:
        logger.warning(
            "request_id_rejected",
            extra={
                "reason": "too_long",
                "length": len(request_id),
                "max_length": REQUEST_ID_MAX_LENGTH,
            }
        )
        return generate_trace_id(), False

    # Check character set
    if not REQUEST_ID_PATTERN.match(request_id):
        logger.warning(
            "request_id_rejected",
            extra={
                "reason": "invalid_characters",
                "request_id_prefix": request_id[:20] if len(request_id) > 20 else request_id,
            }
        )
        return generate_trace_id(), False

    return request_id, True


def generate_trace_id() -> str:
    """
    Generate a unique request ID.

    Format: req_{uuid16} to align with platform standard.
    """
    return f"req_{uuid.uuid4().hex[:16]}"


def get_trace_id() -> str:
    """Get the current trace ID from context."""
    trace_id = trace_id_ctx.get()
    if not trace_id:
        trace_id = generate_trace_id()
        trace_id_ctx.set(trace_id)
    return trace_id


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID for the current context."""
    trace_id_ctx.set(trace_id)


def get_user_id() -> Optional[str]:
    """Get the current user ID from context."""
    return user_id_ctx.get()


def set_user_id(user_id: str) -> None:
    """Set the user ID for the current context."""
    user_id_ctx.set(user_id)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add distributed tracing to all requests.

    Supports both X-Request-Id (platform standard) and X-Trace-Id (ISL legacy).
    Priority: X-Request-Id > X-Trace-Id > generated
    """

    async def dispatch(self, request: Request, call_next):
        """Process request with tracing."""
        # Extract request ID from headers (priority: X-Request-Id > X-Trace-Id)
        # This aligns with Olumi platform standard while maintaining backward compatibility
        inbound_id = (
            request.headers.get('X-Request-Id') or
            request.headers.get('X-Trace-Id')
        )

        # Sanitize for security (prevents log injection, header abuse)
        request_id, _ = sanitize_request_id(inbound_id)
        set_trace_id(request_id)

        # Extract user ID if present
        user_id = request.headers.get('X-User-Id')
        if user_id:
            set_user_id(user_id)

        # Process request
        response = await call_next(request)

        # Add correlation IDs to response headers
        # Include both for compatibility during migration
        response.headers['X-Request-Id'] = request_id
        response.headers['X-Trace-Id'] = request_id  # Deprecated, for backward compatibility

        return response
