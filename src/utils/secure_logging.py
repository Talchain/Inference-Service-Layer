"""
Secure logging utilities for privacy-compliant logging.

Provides:
- Functions to sanitize sensitive data before logging
- Automatic correlation ID injection from request context
- PII redaction for sensitive fields
- Security audit logging
- Structured JSON formatting
"""

import hashlib
import logging
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Set

from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)

# Import tracing context - deferred to avoid circular imports
_tracing_module = None


def _get_tracing_module():
    """Lazy import of tracing module to avoid circular imports."""
    global _tracing_module
    if _tracing_module is None:
        from . import tracing
        _tracing_module = tracing
    return _tracing_module


# Patterns for PII detection and redaction
PII_PATTERNS: List[tuple[str, Pattern]] = [
    ("api_key", re.compile(r'(?i)(api[_-]?key|apikey|x-api-key)["\s:=]+([a-zA-Z0-9_\-]{16,})')),
    ("bearer_token", re.compile(r'(?i)bearer\s+([a-zA-Z0-9_\-\.]{20,})')),
    ("password", re.compile(r'(?i)(password|passwd|pwd)["\s:=]+([^\s"\']{4,})')),
    ("credit_card", re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b')),
    ("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
]

# Fields that should be completely redacted
SENSITIVE_FIELDS: Set[str] = {
    "password",
    "passwd",
    "pwd",
    "secret",
    "api_key",
    "apikey",
    "api-key",
    "x-api-key",
    "authorization",
    "auth",
    "token",
    "bearer",
    "credential",
    "credentials",
    "private_key",
    "privatekey",
    "redis_password",
    "openai_api_key",
    "anthropic_api_key",
}

# Fields that should be partially masked (show first/last few chars)
MASKABLE_FIELDS: Set[str] = {
    "email",
    "client_ip",
    "ip",
    "ip_address",
}


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


# =============================================================================
# PII Redaction Functions
# =============================================================================

def redact_value(value: Any, field_name: str = "") -> Any:
    """
    Redact sensitive values based on field name or content.

    Args:
        value: Value to potentially redact
        field_name: Name of the field (used for field-based redaction)

    Returns:
        Redacted value or original if not sensitive
    """
    if value is None:
        return None

    field_lower = field_name.lower().replace("-", "_")

    # Completely redact sensitive fields
    if field_lower in SENSITIVE_FIELDS:
        return "[REDACTED]"

    # Partially mask certain fields
    if field_lower in MASKABLE_FIELDS:
        str_value = str(value)
        if len(str_value) > 6:
            return f"{str_value[:3]}***{str_value[-3:]}"
        return "[MASKED]"

    # For string values, check for embedded PII patterns
    if isinstance(value, str):
        return redact_string(value)

    # For dicts, recursively redact
    if isinstance(value, dict):
        return {k: redact_value(v, k) for k, v in value.items()}

    # For lists, recursively redact
    if isinstance(value, list):
        return [redact_value(item) for item in value]

    return value


def redact_string(text: str) -> str:
    """
    Redact PII patterns from a string.

    Args:
        text: String to redact

    Returns:
        String with PII patterns redacted
    """
    if not isinstance(text, str):
        return text

    result = text

    for pattern_name, pattern in PII_PATTERNS:
        if pattern_name in ("api_key", "bearer_token", "password"):
            # Replace the captured secret value
            def redact_match(m):
                if m.lastindex and m.lastindex >= 2:
                    return f"{m.group(1)}=[REDACTED]"
                return "[REDACTED]"
            result = pattern.sub(redact_match, result)
        elif pattern_name == "credit_card":
            result = pattern.sub("[CARD-REDACTED]", result)
        elif pattern_name == "ssn":
            result = pattern.sub("[SSN-REDACTED]", result)

    return result


# =============================================================================
# Correlation ID JSON Formatter
# =============================================================================

class CorrelationIDFormatter(jsonlogger.JsonFormatter):
    """
    JSON formatter that automatically injects correlation IDs and redacts PII.
    """

    def __init__(self, *args, redact_pii: bool = True, **kwargs):
        """
        Initialize the formatter.

        Args:
            redact_pii: Whether to redact PII from log messages
        """
        super().__init__(*args, **kwargs)
        self.redact_pii = redact_pii

    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict) -> None:
        """
        Add correlation IDs and apply PII redaction to log fields.
        """
        super().add_fields(log_record, record, message_dict)

        # Get tracing module
        tracing = _get_tracing_module()

        # Always add correlation ID from context
        trace_id = tracing.get_trace_id() if tracing else None
        if trace_id:
            log_record["correlation_id"] = trace_id
            log_record["trace_id"] = trace_id

        # Add user ID if available
        user_id = tracing.get_user_id() if tracing else None
        if user_id:
            log_record["user_id"] = hash_user_id(user_id) if self.redact_pii else user_id

        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Apply PII redaction to all fields
        if self.redact_pii:
            for key, value in list(log_record.items()):
                if key not in ("timestamp", "level", "logger", "correlation_id", "trace_id"):
                    log_record[key] = redact_value(value, key)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with PII redaction on the message."""
        if self.redact_pii and record.msg:
            record.msg = redact_string(str(record.msg))
        return super().format(record)


# =============================================================================
# Security Audit Logger
# =============================================================================

class SecurityAuditLogger:
    """
    Dedicated logger for security-relevant events.

    Logs authentication, authorization, and security-related events
    with additional context for audit purposes.
    """

    def __init__(self, logger_name: str = "security.audit"):
        """Initialize the security audit logger."""
        self._logger = logging.getLogger(logger_name)

    def log_authentication_attempt(
        self,
        success: bool,
        client_ip: str,
        api_key_prefix: Optional[str] = None,
        reason: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        """
        Log an authentication attempt.

        Args:
            success: Whether authentication succeeded
            client_ip: Client IP address
            api_key_prefix: First few characters of the API key (for debugging)
            reason: Reason for failure (if applicable)
            path: Request path
        """
        event = "auth_success" if success else "auth_failure"
        level = logging.INFO if success else logging.WARNING

        # Get trace ID from context
        tracing = _get_tracing_module()
        trace_id = tracing.get_trace_id() if tracing else None

        self._logger.log(
            level,
            event,
            extra={
                "event_type": "authentication",
                "success": success,
                "client_ip": client_ip,
                "api_key_prefix": api_key_prefix[:8] + "..." if api_key_prefix and len(api_key_prefix) > 8 else api_key_prefix,
                "reason": reason,
                "path": path,
                "audit": True,
                "correlation_id": trace_id,
            }
        )

    def log_authorization_check(
        self,
        success: bool,
        client_ip: str,
        resource: str,
        action: str,
        reason: Optional[str] = None,
    ) -> None:
        """
        Log an authorization check.

        Args:
            success: Whether authorization succeeded
            client_ip: Client IP address
            resource: Resource being accessed
            action: Action being performed
            reason: Reason for failure (if applicable)
        """
        event = "authz_success" if success else "authz_failure"
        level = logging.INFO if success else logging.WARNING

        tracing = _get_tracing_module()
        trace_id = tracing.get_trace_id() if tracing else None

        self._logger.log(
            level,
            event,
            extra={
                "event_type": "authorization",
                "success": success,
                "client_ip": client_ip,
                "resource": resource,
                "action": action,
                "reason": reason,
                "audit": True,
                "correlation_id": trace_id,
            }
        )

    def log_rate_limit_exceeded(
        self,
        client_ip: str,
        identifier: str,
        limit: int,
        window_seconds: int,
        path: Optional[str] = None,
    ) -> None:
        """
        Log a rate limit violation.

        Args:
            client_ip: Client IP address
            identifier: Rate limit identifier (IP or API key hash)
            limit: Rate limit threshold
            window_seconds: Rate limit window
            path: Request path
        """
        tracing = _get_tracing_module()
        trace_id = tracing.get_trace_id() if tracing else None

        self._logger.warning(
            "rate_limit_exceeded",
            extra={
                "event_type": "rate_limit",
                "client_ip": client_ip,
                "identifier": identifier,
                "limit": limit,
                "window_seconds": window_seconds,
                "path": path,
                "audit": True,
                "correlation_id": trace_id,
            }
        )

    def log_suspicious_activity(
        self,
        activity_type: str,
        client_ip: str,
        details: Dict[str, Any],
        severity: str = "medium",
    ) -> None:
        """
        Log suspicious activity for security monitoring.

        Args:
            activity_type: Type of suspicious activity
            client_ip: Client IP address
            details: Additional details about the activity
            severity: Severity level (low, medium, high, critical)
        """
        level = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(severity, logging.WARNING)

        tracing = _get_tracing_module()
        trace_id = tracing.get_trace_id() if tracing else None

        self._logger.log(
            level,
            "suspicious_activity",
            extra={
                "event_type": "security_alert",
                "activity_type": activity_type,
                "client_ip": client_ip,
                "severity": severity,
                "details": details,
                "audit": True,
                "correlation_id": trace_id,
            }
        )

    def log_config_validation_failure(
        self,
        errors: List[str],
        environment: str,
    ) -> None:
        """
        Log configuration validation failures.

        Args:
            errors: List of validation error messages
            environment: Current environment
        """
        self._logger.error(
            "config_validation_failed",
            extra={
                "event_type": "configuration",
                "errors": errors,
                "environment": environment,
                "audit": True,
            }
        )


def setup_secure_logging(
    log_level: str = "INFO",
    redact_pii: bool = True,
) -> logging.Logger:
    """
    Configure secure logging with correlation IDs and PII redaction.

    Args:
        log_level: Logging level
        redact_pii: Whether to enable PII redaction

    Returns:
        Configured root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Create stdout handler with secure JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = CorrelationIDFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
        },
        redact_pii=redact_pii,
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    return root_logger


# Singleton security audit logger
_security_audit_logger: Optional[SecurityAuditLogger] = None


def get_security_audit_logger() -> SecurityAuditLogger:
    """Get the security audit logger singleton."""
    global _security_audit_logger
    if _security_audit_logger is None:
        _security_audit_logger = SecurityAuditLogger()
    return _security_audit_logger
