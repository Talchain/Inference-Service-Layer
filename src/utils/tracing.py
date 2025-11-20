"""
Operation-level latency tracing for performance analysis.
"""

import time
import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


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
                "duration_ms": duration_ms
            }
        )
