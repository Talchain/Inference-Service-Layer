"""
Secure logging utilities for privacy-compliant logging.

Provides functions to sanitize sensitive data before logging.
"""

import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def hash_user_id(user_id: str) -> str:
    """
    Hash user ID for logging privacy.

    Args:
        user_id: Raw user identifier

    Returns:
        First 16 characters of SHA-256 hash

    Example:
        >>> hash_user_id("user_12345")
        "a3f2b1c4d5e6f7g8"
    """
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


def sanitize_model_for_logging(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove sensitive model details for logging.

    Args:
        model: Raw model dictionary

    Returns:
        Sanitized model summary without sensitive data

    Example:
        >>> model = {
        ...     "dag": {"nodes": ["A", "B", "C"], "edges": [["A", "B"]]},
        ...     "parameters": {"A": 1.0, "B": 2.0}
        ... }
        >>> sanitize_model_for_logging(model)
        {'node_count': 3, 'edge_count': 1, 'has_parameters': True}
    """
    dag = model.get("dag", {})
    nodes = dag.get("nodes", [])
    edges = dag.get("edges", [])

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "has_parameters": bool(model.get("parameters")),
        "has_distributions": bool(model.get("distributions"))
    }


def sanitize_request_for_logging(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize entire request for logging.

    Args:
        request_data: Raw request data

    Returns:
        Sanitized request summary

    Example:
        >>> request = {
        ...     "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
        ...     "treatment": "A",
        ...     "outcome": "B"
        ... }
        >>> sanitize_request_for_logging(request)
        {'has_dag': True, 'node_count': 2, 'has_treatment': True, 'has_outcome': True}
    """
    summary: Dict[str, Any] = {}

    # DAG summary
    if "dag" in request_data:
        dag = request_data["dag"]
        summary["has_dag"] = True
        summary["node_count"] = len(dag.get("nodes", []))
        summary["edge_count"] = len(dag.get("edges", []))

    # Model summary
    if "model" in request_data:
        summary["model"] = sanitize_model_for_logging(request_data["model"])

    # Simple flags for other fields
    for field in ["treatment", "outcome", "intervention", "context"]:
        if field in request_data:
            summary[f"has_{field}"] = True

    return summary


def log_request_safe(
    endpoint: str,
    user_id: Optional[str] = None,
    request_data: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log request without sensitive data.

    Args:
        endpoint: API endpoint path
        user_id: User identifier (will be hashed)
        request_data: Request data (will be sanitized)
        request_id: Request ID for tracing

    Example:
        >>> log_request_safe(
        ...     endpoint="/api/v1/causal/validate",
        ...     user_id="user_123",
        ...     request_data={"dag": {...}},
        ...     request_id="req_abc"
        ... )
        # Logs: {"endpoint": "/api/v1/causal/validate", "user_hash": "a3f2...", ...}
    """
    log_data: Dict[str, Any] = {
        "endpoint": endpoint,
    }

    if request_id:
        log_data["request_id"] = request_id

    if user_id:
        log_data["user_hash"] = hash_user_id(user_id)

    if request_data:
        log_data["request_summary"] = sanitize_request_for_logging(request_data)

    logger.info("Request received", extra=log_data)


def log_response_safe(
    endpoint: str,
    status_code: int,
    duration_ms: float,
    request_id: Optional[str] = None,
    error: Optional[str] = None
) -> None:
    """
    Log response without sensitive data.

    Args:
        endpoint: API endpoint path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        request_id: Request ID for tracing
        error: Error type (if failed)

    Example:
        >>> log_response_safe(
        ...     endpoint="/api/v1/causal/validate",
        ...     status_code=200,
        ...     duration_ms=45.2,
        ...     request_id="req_abc"
        ... )
    """
    log_data: Dict[str, Any] = {
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2)
    }

    if request_id:
        log_data["request_id"] = request_id

    if error:
        log_data["error_type"] = error

    level = logging.INFO if status_code < 400 else logging.ERROR
    logger.log(level, "Request completed", extra=log_data)


def sanitize_error_for_logging(error: Exception) -> Dict[str, str]:
    """
    Sanitize error for logging (remove sensitive details).

    Args:
        error: Exception object

    Returns:
        Sanitized error info

    Example:
        >>> try:
        ...     raise ValueError("DAG has 100 nodes")
        ... except ValueError as e:
        ...     sanitize_error_for_logging(e)
        {'error_type': 'ValueError', 'error_message': 'DAG has 100 nodes'}
    """
    return {
        "error_type": type(error).__name__,
        "error_message": str(error)[:200]  # Limit message length
    }


def log_error_safe(
    endpoint: str,
    error: Exception,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> None:
    """
    Log error without sensitive data.

    Args:
        endpoint: API endpoint path
        error: Exception that occurred
        request_id: Request ID for tracing
        user_id: User identifier (will be hashed)

    Example:
        >>> try:
        ...     # Some operation
        ...     pass
        ... except ValueError as e:
        ...     log_error_safe("/api/v1/causal/validate", e, "req_abc")
    """
    log_data: Dict[str, Any] = {
        "endpoint": endpoint,
        **sanitize_error_for_logging(error)
    }

    if request_id:
        log_data["request_id"] = request_id

    if user_id:
        log_data["user_hash"] = hash_user_id(user_id)

    logger.error("Request failed", extra=log_data, exc_info=True)
