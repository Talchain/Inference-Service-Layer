"""
Outcome Logging Service.

Implements minimal outcome logging infrastructure for future calibration.
Brief 7, Task 8: Outcome Logging Infrastructure.

Records decisions and their outcomes for analyzing recommendation accuracy.
Phase 3 calibration will build on this stable schema.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from src.models.decision_robustness import (
    OutcomeLog,
    OutcomeLogRequest,
    OutcomeSummary,
    OutcomeUpdateRequest,
)
from src.utils.cache import get_cache

logger = logging.getLogger(__name__)

# In-memory storage (production would use database)
# Using cache with long TTL as simple storage
_outcome_store = get_cache("outcome_logs", max_size=10000, ttl=86400 * 30)  # 30 days

# Index for quick lookups
_outcome_index: Dict[str, str] = {}  # id -> cache_key


class OutcomeLogger:
    """
    Service for logging decision outcomes.

    Provides endpoints for:
    - Recording decisions
    - Updating with actual outcomes
    - Retrieving calibration statistics
    """

    def __init__(self) -> None:
        """Initialize the outcome logger."""
        self._logs: Dict[str, OutcomeLog] = {}

    def log_decision(
        self,
        request: OutcomeLogRequest,
        request_id: str,
    ) -> OutcomeLog:
        """
        Record a decision for future calibration.

        Args:
            request: Outcome log request
            request_id: Request ID for tracing

        Returns:
            Created OutcomeLog with assigned ID
        """
        log_id = f"log_{uuid4().hex[:12]}"

        outcome_log = OutcomeLog(
            id=log_id,
            decision_id=request.decision_id,
            graph_hash=request.graph_hash,
            response_hash=request.response_hash,
            chosen_option=request.chosen_option,
            recommendation_option=request.recommendation_option,
            recommendation_followed=(
                request.chosen_option == request.recommendation_option
            ),
            timestamp=datetime.utcnow(),
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            notes=request.notes,
        )

        # Store in memory and cache
        self._logs[log_id] = outcome_log
        _outcome_store.put(log_id, outcome_log.model_dump())
        _outcome_index[log_id] = log_id

        logger.info(
            "outcome_logged",
            extra={
                "request_id": request_id,
                "log_id": log_id,
                "decision_id": request.decision_id,
                "recommendation_followed": outcome_log.recommendation_followed,
            },
        )

        return outcome_log

    def update_outcome(
        self,
        log_id: str,
        request: OutcomeUpdateRequest,
        request_id: str,
    ) -> Optional[OutcomeLog]:
        """
        Update an outcome log with actual outcome values.

        Args:
            log_id: ID of the outcome log to update
            request: Update request with outcome values
            request_id: Request ID for tracing

        Returns:
            Updated OutcomeLog, or None if not found
        """
        # Try memory first, then cache
        outcome_log = self._logs.get(log_id)

        if outcome_log is None:
            # Try cache
            cached = _outcome_store.get(log_id)
            if cached:
                outcome_log = OutcomeLog(**cached)
                self._logs[log_id] = outcome_log

        if outcome_log is None:
            logger.warning(
                "outcome_log_not_found",
                extra={"request_id": request_id, "log_id": log_id},
            )
            return None

        # Update with outcome values
        outcome_log.outcome_values = request.outcome_values
        outcome_log.outcome_timestamp = datetime.utcnow()
        if request.notes:
            existing_notes = outcome_log.notes or ""
            outcome_log.notes = f"{existing_notes}\n{request.notes}".strip()

        # Update storage
        self._logs[log_id] = outcome_log
        _outcome_store.put(log_id, outcome_log.model_dump())

        logger.info(
            "outcome_updated",
            extra={
                "request_id": request_id,
                "log_id": log_id,
                "outcome_values": request.outcome_values,
            },
        )

        return outcome_log

    def get_outcome(self, log_id: str) -> Optional[OutcomeLog]:
        """
        Retrieve an outcome log by ID.

        Args:
            log_id: Outcome log ID

        Returns:
            OutcomeLog if found, None otherwise
        """
        outcome_log = self._logs.get(log_id)

        if outcome_log is None:
            cached = _outcome_store.get(log_id)
            if cached:
                outcome_log = OutcomeLog(**cached)
                self._logs[log_id] = outcome_log

        return outcome_log

    def get_summary(self, request_id: str) -> OutcomeSummary:
        """
        Get summary statistics for calibration.

        Basic stats that Phase 3 will expand into full calibration.

        Args:
            request_id: Request ID for tracing

        Returns:
            OutcomeSummary with basic statistics
        """
        all_logs = list(self._logs.values())

        # Also check cache for any not in memory
        # In production, would query database

        total = len(all_logs)
        with_outcomes = sum(1 for log in all_logs if log.outcome_values)
        followed = sum(1 for log in all_logs if log.recommendation_followed)

        # Calculate average outcomes
        outcomes_followed = [
            log.outcome_values
            for log in all_logs
            if log.recommendation_followed and log.outcome_values
        ]
        outcomes_not_followed = [
            log.outcome_values
            for log in all_logs
            if not log.recommendation_followed and log.outcome_values
        ]

        avg_followed = None
        if outcomes_followed:
            # Average the first value in each outcome dict
            values = []
            for ov in outcomes_followed:
                if ov:
                    values.append(list(ov.values())[0])
            if values:
                avg_followed = sum(values) / len(values)

        avg_not_followed = None
        if outcomes_not_followed:
            values = []
            for ov in outcomes_not_followed:
                if ov:
                    values.append(list(ov.values())[0])
            if values:
                avg_not_followed = sum(values) / len(values)

        logger.info(
            "outcome_summary_generated",
            extra={
                "request_id": request_id,
                "total_logged": total,
                "with_outcomes": with_outcomes,
            },
        )

        return OutcomeSummary(
            total_logged=total,
            with_outcomes=with_outcomes,
            recommendations_followed=followed,
            recommendations_followed_pct=(followed / total * 100) if total > 0 else 0.0,
            avg_outcome_when_followed=avg_followed,
            avg_outcome_when_not_followed=avg_not_followed,
        )

    def list_outcomes(
        self,
        limit: int = 100,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[OutcomeLog]:
        """
        List outcome logs with optional filtering.

        Args:
            limit: Maximum number to return
            tenant_id: Optional tenant filter
            user_id: Optional user filter

        Returns:
            List of matching OutcomeLogs
        """
        results = []

        for log in self._logs.values():
            if tenant_id and log.tenant_id != tenant_id:
                continue
            if user_id and log.user_id != user_id:
                continue

            results.append(log)

            if len(results) >= limit:
                break

        # Sort by timestamp descending
        results.sort(key=lambda x: x.timestamp, reverse=True)

        return results


# Singleton instance
_outcome_logger: Optional[OutcomeLogger] = None


def get_outcome_logger() -> OutcomeLogger:
    """
    Get the singleton OutcomeLogger instance.

    Returns:
        OutcomeLogger instance
    """
    global _outcome_logger
    if _outcome_logger is None:
        _outcome_logger = OutcomeLogger()
    return _outcome_logger
