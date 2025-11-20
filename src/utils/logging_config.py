"""
Structured JSON logging configuration for ISL.

Provides machine-readable logs for production observability.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON string
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint

        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = round(record.duration_ms, 2)

        if hasattr(record, "user_hash"):
            log_data["user_hash"] = record.user_hash

        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code

        if hasattr(record, "operation"):
            log_data["operation"] = record.operation

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """
    Configure structured JSON logging.

    Args:
        level: Logging level (INFO, DEBUG, WARNING, ERROR)
        json_format: Use JSON formatter (True for production)
    """
    handler = logging.StreamHandler()

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        # Human-readable format for development
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
