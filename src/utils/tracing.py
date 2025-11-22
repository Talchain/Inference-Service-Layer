"""
Operation-level latency tracing and distributed tracing for performance analysis.

Provides:
- Operation latency tracing
- Distributed trace ID generation and propagation
- Context-local trace storage
- Enriched logging with trace context
"""

import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator, Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

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

def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return f"trace_{uuid.uuid4().hex[:16]}"


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
    """Middleware to add distributed tracing to all requests."""

    async def dispatch(self, request: Request, call_next):
        """Process request with tracing."""
        # Extract trace ID from header or generate new one
        trace_id = request.headers.get('X-Trace-Id') or generate_trace_id()
        set_trace_id(trace_id)

        # Extract user ID if present
        user_id = request.headers.get('X-User-Id')
        if user_id:
            set_user_id(user_id)

        # Process request
        response = await call_next(request)

        # Add trace ID to response headers
        response.headers['X-Trace-Id'] = trace_id

        return response
