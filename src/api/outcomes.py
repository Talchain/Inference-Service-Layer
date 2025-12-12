"""
Outcome Logging API endpoints.

Endpoints for recording decisions and outcomes for future calibration.
Brief 7, Task 8: Outcome Logging Infrastructure.
"""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Path

from src.config import get_settings
from src.models.decision_robustness import (
    OutcomeLog,
    OutcomeLogRequest,
    OutcomeSummary,
    OutcomeUpdateRequest,
)
from src.models.isl_metadata import create_isl_metadata
from src.models.responses import ErrorCode, ErrorResponse, RecoveryHints
from src.services.outcome_logger import get_outcome_logger

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


# NOTE: /summary must be defined BEFORE /{log_id} routes to avoid path conflict
@router.get(
    "/summary",
    response_model=OutcomeSummary,
    summary="Get Outcome Summary",
    description="""
Get summary statistics for outcome logging.

Returns basic calibration stats:
- Total decisions logged
- Decisions with recorded outcomes
- Recommendations followed rate
- Average outcomes (when followed vs not followed)

Phase 3 will expand this into full calibration analysis.
""",
    responses={
        200: {
            "description": "Summary statistics",
            "model": OutcomeSummary,
        },
    },
)
async def get_summary(
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> OutcomeSummary:
    """
    Get summary statistics for calibration.
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    outcome_logger = get_outcome_logger()
    summary = outcome_logger.get_summary(request_id)

    elapsed_ms = (time.time() - start_time) * 1000

    # Add metadata
    summary.metadata = create_isl_metadata(
        request_id=request_id,
        computation_time_ms=elapsed_ms,
        algorithm="outcome_summary",
        cache_hit=False,
    )

    return summary


@router.post(
    "/log",
    response_model=OutcomeLog,
    summary="Log a Decision",
    description="""
Record a decision for future calibration analysis.

Call this endpoint after a user makes a decision based on ISL's recommendation.
Records:
- Which option the user chose
- What ISL recommended
- Whether the recommendation was followed

Later, update the log with actual outcomes using PATCH /outcomes/{id}.
""",
    responses={
        200: {
            "description": "Decision logged successfully",
            "model": OutcomeLog,
        },
        503: {
            "description": "Outcome logging disabled",
            "model": ErrorResponse,
        },
    },
)
async def log_decision(
    request: OutcomeLogRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> OutcomeLog:
    """
    Log a decision outcome for calibration.

    Records the decision and recommendation for future analysis.
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    # Check feature flag
    if not getattr(settings, "ENABLE_OUTCOME_LOGGING", True):
        raise HTTPException(
            status_code=503,
            detail="Outcome logging is disabled",
        )

    logger.info(
        "logging_decision",
        extra={
            "request_id": request_id,
            "decision_id": request.decision_id,
            "chosen_option": request.chosen_option,
            "recommendation_option": request.recommendation_option,
        },
    )

    try:
        outcome_logger = get_outcome_logger()
        outcome_log = outcome_logger.log_decision(request, request_id)

        logger.info(
            "decision_logged",
            extra={
                "request_id": request_id,
                "log_id": outcome_log.id,
                "recommendation_followed": outcome_log.recommendation_followed,
            },
        )

        return outcome_log

    except Exception as e:
        logger.error(
            "log_decision_error",
            exc_info=True,
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code=ErrorCode.COMPUTATION_ERROR.value,
                message="Failed to log decision",
                reason="internal_error",
                recovery=RecoveryHints(
                    hints=["Retry the request"],
                    suggestion="Try again",
                ),
                retryable=True,
                source="isl",
                request_id=request_id,
            ).model_dump(),
        )


@router.patch(
    "/{log_id}",
    response_model=OutcomeLog,
    summary="Update Outcome",
    description="""
Update an outcome log with actual outcome values.

Call this endpoint when the actual outcomes of a decision become known.
This data is used for calibration analysis in Phase 3.

Provide:
- outcome_values: Actual measured outcomes (e.g., {"revenue": 155000})
- notes: Optional additional context
""",
    responses={
        200: {
            "description": "Outcome updated successfully",
            "model": OutcomeLog,
        },
        404: {
            "description": "Outcome log not found",
            "model": ErrorResponse,
        },
        503: {
            "description": "Outcome logging disabled",
            "model": ErrorResponse,
        },
    },
)
async def update_outcome(
    request: OutcomeUpdateRequest,
    log_id: str = Path(..., description="Outcome log ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> OutcomeLog:
    """
    Update an outcome log with actual outcomes.

    Adds the actual outcome values after they become known.
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    # Check feature flag
    if not getattr(settings, "ENABLE_OUTCOME_LOGGING", True):
        raise HTTPException(
            status_code=503,
            detail="Outcome logging is disabled",
        )

    logger.info(
        "updating_outcome",
        extra={
            "request_id": request_id,
            "log_id": log_id,
            "outcome_values": request.outcome_values,
        },
    )

    try:
        outcome_logger = get_outcome_logger()
        updated_log = outcome_logger.update_outcome(log_id, request, request_id)

        if updated_log is None:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    code=ErrorCode.NODE_NOT_FOUND.value,
                    message=f"Outcome log '{log_id}' not found",
                    reason="not_found",
                    recovery=RecoveryHints(
                        hints=["Check the log ID is correct"],
                        suggestion="Verify the log ID from the original log response",
                    ),
                    retryable=False,
                    source="isl",
                    request_id=request_id,
                ).model_dump(),
            )

        logger.info(
            "outcome_updated",
            extra={
                "request_id": request_id,
                "log_id": log_id,
            },
        )

        return updated_log

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "update_outcome_error",
            exc_info=True,
            extra={"request_id": request_id, "log_id": log_id},
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code=ErrorCode.COMPUTATION_ERROR.value,
                message="Failed to update outcome",
                reason="internal_error",
                recovery=RecoveryHints(
                    hints=["Retry the request"],
                    suggestion="Try again",
                ),
                retryable=True,
                source="isl",
                request_id=request_id,
            ).model_dump(),
        )


@router.get(
    "/{log_id}",
    response_model=OutcomeLog,
    summary="Get Outcome Log",
    description="Retrieve a specific outcome log by ID.",
    responses={
        200: {
            "description": "Outcome log found",
            "model": OutcomeLog,
        },
        404: {
            "description": "Outcome log not found",
            "model": ErrorResponse,
        },
    },
)
async def get_outcome(
    log_id: str = Path(..., description="Outcome log ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> OutcomeLog:
    """
    Get an outcome log by ID.
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    outcome_logger = get_outcome_logger()
    outcome_log = outcome_logger.get_outcome(log_id)

    if outcome_log is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                code=ErrorCode.NODE_NOT_FOUND.value,
                message=f"Outcome log '{log_id}' not found",
                reason="not_found",
                recovery=RecoveryHints(
                    hints=["Check the log ID is correct"],
                    suggestion="Verify the log ID from the original log response",
                ),
                retryable=False,
                source="isl",
                request_id=request_id,
            ).model_dump(),
        )

    return outcome_log
