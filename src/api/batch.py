"""
Batch processing endpoints for ISL.

Provides high-throughput batch processing for:
- Causal validation (up to 50 requests)
- Counterfactual analysis (up to 20 requests)

Uses parallel processing for 5-10x speedup vs sequential requests.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from src.models.metadata import create_response_metadata
from src.models.requests import CausalValidationRequest, CounterfactualRequest
from src.models.responses import (
    CausalValidationResponse,
    CounterfactualResponse,
    ErrorCode,
    ErrorResponse,
    ResponseMetadata,
)
from src.services.causal_validator import CausalValidator
from src.services.counterfactual_engine import CounterfactualEngine

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services (thread-safe)
causal_validator = CausalValidator()
counterfactual_engine = CounterfactualEngine()

# Batch processing limits
MAX_VALIDATION_BATCH = 50
MAX_COUNTERFACTUAL_BATCH = 20
MAX_WORKERS = 10  # Concurrent workers
BATCH_ITEM_TIMEOUT_SECONDS = 30  # Per-item timeout to prevent hanging


# === Request/Response Models ===


class BatchValidationRequest(BaseModel):
    """Batch causal validation request."""

    requests: List[CausalValidationRequest] = Field(
        ...,
        description="List of causal validation requests to process",
        min_length=1,
        max_length=MAX_VALIDATION_BATCH,
    )


class BatchValidationItem(BaseModel):
    """Single item in batch validation response."""

    index: int = Field(..., description="Index in the original request list")
    success: bool = Field(..., description="Whether this request succeeded")
    result: Optional[CausalValidationResponse] = Field(
        None, description="Validation result (if successful)"
    )
    error: Optional[str] = Field(None, description="Error message (if failed)")


class BatchValidationResponse(BaseModel):
    """Batch causal validation response."""

    results: List[BatchValidationItem] = Field(..., description="Results for each request")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    metadata: Optional[ResponseMetadata] = Field(None, description="Response metadata")


class BatchCounterfactualRequest(BaseModel):
    """Batch counterfactual analysis request."""

    requests: List[CounterfactualRequest] = Field(
        ...,
        description="List of counterfactual requests to process",
        min_length=1,
        max_length=MAX_COUNTERFACTUAL_BATCH,
    )


class BatchCounterfactualItem(BaseModel):
    """Single item in batch counterfactual response."""

    index: int = Field(..., description="Index in the original request list")
    success: bool = Field(..., description="Whether this request succeeded")
    result: Optional[CounterfactualResponse] = Field(
        None, description="Counterfactual result (if successful)"
    )
    error: Optional[str] = Field(None, description="Error message (if failed)")


class BatchCounterfactualResponse(BaseModel):
    """Batch counterfactual analysis response."""

    results: List[BatchCounterfactualItem] = Field(..., description="Results for each request")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    metadata: Optional[ResponseMetadata] = Field(None, description="Response metadata")


# === Helper Functions ===


def _process_validation_item(index: int, request: CausalValidationRequest) -> BatchValidationItem:
    """
    Process a single validation request.

    Args:
        index: Index in the batch
        request: Validation request

    Returns:
        BatchValidationItem with result or error
    """
    try:
        result = causal_validator.validate(request)
        return BatchValidationItem(index=index, success=True, result=result, error=None)
    except Exception as e:
        logger.warning(
            "batch_validation_item_failed",
            extra={"index": index, "error": str(e)},
        )
        return BatchValidationItem(index=index, success=False, result=None, error=str(e))


def _process_counterfactual_item(
    index: int, request: CounterfactualRequest
) -> BatchCounterfactualItem:
    """
    Process a single counterfactual request.

    Args:
        index: Index in the batch
        request: Counterfactual request

    Returns:
        BatchCounterfactualItem with result or error
    """
    try:
        result = counterfactual_engine.analyze(request)
        return BatchCounterfactualItem(index=index, success=True, result=result, error=None)
    except Exception as e:
        logger.warning(
            "batch_counterfactual_item_failed",
            extra={"index": index, "error": str(e)},
        )
        return BatchCounterfactualItem(index=index, success=False, result=None, error=str(e))


# === Endpoints ===


@router.post(
    "/validate",
    response_model=BatchValidationResponse,
    summary="Batch causal validation",
    description="""
    Validates multiple causal models in parallel for 5-10x speedup.

    **Limits:**
    - Max requests per batch: 50
    - Parallel workers: 10

    **Behavior:**
    - Processes requests in parallel
    - Returns partial results if some requests fail
    - Each result includes success status and error details

    **Use when:** Validating multiple causal models at once (e.g., scenario comparison).
    """,
    responses={
        200: {"description": "Batch processing completed (may include partial failures)"},
        400: {"description": "Invalid input (e.g., batch too large, empty batch)"},
        500: {"description": "Internal error prevented batch processing"},
    },
)
async def batch_validate_causal_models(
    request: BatchValidationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> BatchValidationResponse:
    """
    Batch validate causal models.

    Args:
        request: Batch validation request
        x_request_id: Optional request ID for tracing

    Returns:
        BatchValidationResponse with results for each request
    """
    request_id = x_request_id or f"batch_{uuid.uuid4().hex[:12]}"
    batch_size = len(request.requests)
    start_time = datetime.utcnow()

    logger.info(
        "batch_validation_started",
        extra={"request_id": request_id, "batch_size": batch_size},
    )

    try:
        results: List[BatchValidationItem] = []

        # Process in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_validation_item, idx, req): idx
                for idx, req in enumerate(request.requests)
            }

            # Collect results as they complete (with timeout)
            total_timeout = BATCH_ITEM_TIMEOUT_SECONDS * batch_size
            for future in as_completed(futures, timeout=total_timeout):
                try:
                    result = future.result(timeout=BATCH_ITEM_TIMEOUT_SECONDS)
                    results.append(result)
                except FutureTimeoutError:
                    # Handle timeout for individual item
                    idx = futures[future]
                    logger.warning(
                        "batch_validation_item_timeout",
                        extra={"index": idx, "timeout_seconds": BATCH_ITEM_TIMEOUT_SECONDS},
                    )
                    results.append(
                        BatchValidationItem(
                            index=idx,
                            success=False,
                            result=None,
                            error=f"Request timed out after {BATCH_ITEM_TIMEOUT_SECONDS}s",
                        )
                    )
                except Exception as e:
                    # Handle unexpected errors from executor
                    idx = futures[future]
                    logger.error(
                        "batch_validation_executor_error",
                        extra={"index": idx, "error": str(e)},
                        exc_info=True,
                    )
                    results.append(
                        BatchValidationItem(
                            index=idx,
                            success=False,
                            result=None,
                            error=f"Executor error: {str(e)}",
                        )
                    )

        # Sort results by index to preserve order
        results.sort(key=lambda x: x.index)

        # Calculate summary statistics
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        success_count = sum(1 for r in results if r.success)
        failure_count = batch_size - success_count

        summary = {
            "total_requests": batch_size,
            "successful": success_count,
            "failed": failure_count,
            "duration_ms": round(duration_ms, 2),
            "avg_duration_ms": round(duration_ms / batch_size, 2),
        }

        logger.info(
            "batch_validation_completed",
            extra={
                "request_id": request_id,
                "summary": summary,
            },
        )

        return BatchValidationResponse(
            results=results,
            summary=summary,
            metadata=create_response_metadata(request_id),
        )

    except Exception as e:
        logger.error("batch_validation_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process batch validation. Check logs for details.",
        )


@router.post(
    "/counterfactual",
    response_model=BatchCounterfactualResponse,
    summary="Batch counterfactual analysis",
    description="""
    Analyzes multiple counterfactual scenarios in parallel for 5-10x speedup.

    **Limits:**
    - Max requests per batch: 20
    - Parallel workers: 10

    **Behavior:**
    - Processes requests in parallel
    - Returns partial results if some requests fail
    - Each result includes success status and error details

    **Use when:** Evaluating multiple "what if" scenarios simultaneously.
    """,
    responses={
        200: {"description": "Batch processing completed (may include partial failures)"},
        400: {"description": "Invalid input (e.g., batch too large, empty batch)"},
        500: {"description": "Internal error prevented batch processing"},
    },
)
async def batch_analyze_counterfactuals(
    request: BatchCounterfactualRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> BatchCounterfactualResponse:
    """
    Batch analyze counterfactuals.

    Args:
        request: Batch counterfactual request
        x_request_id: Optional request ID for tracing

    Returns:
        BatchCounterfactualResponse with results for each request
    """
    request_id = x_request_id or f"batch_{uuid.uuid4().hex[:12]}"
    batch_size = len(request.requests)
    start_time = datetime.utcnow()

    logger.info(
        "batch_counterfactual_started",
        extra={"request_id": request_id, "batch_size": batch_size},
    )

    try:
        results: List[BatchCounterfactualItem] = []

        # Process in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_counterfactual_item, idx, req): idx
                for idx, req in enumerate(request.requests)
            }

            # Collect results as they complete (with timeout)
            total_timeout = BATCH_ITEM_TIMEOUT_SECONDS * batch_size
            for future in as_completed(futures, timeout=total_timeout):
                try:
                    result = future.result(timeout=BATCH_ITEM_TIMEOUT_SECONDS)
                    results.append(result)
                except FutureTimeoutError:
                    # Handle timeout for individual item
                    idx = futures[future]
                    logger.warning(
                        "batch_counterfactual_item_timeout",
                        extra={"index": idx, "timeout_seconds": BATCH_ITEM_TIMEOUT_SECONDS},
                    )
                    results.append(
                        BatchCounterfactualItem(
                            index=idx,
                            success=False,
                            result=None,
                            error=f"Request timed out after {BATCH_ITEM_TIMEOUT_SECONDS}s",
                        )
                    )
                except Exception as e:
                    # Handle unexpected errors from executor
                    idx = futures[future]
                    logger.error(
                        "batch_counterfactual_executor_error",
                        extra={"index": idx, "error": str(e)},
                        exc_info=True,
                    )
                    results.append(
                        BatchCounterfactualItem(
                            index=idx,
                            success=False,
                            result=None,
                            error=f"Executor error: {str(e)}",
                        )
                    )

        # Sort results by index to preserve order
        results.sort(key=lambda x: x.index)

        # Calculate summary statistics
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        success_count = sum(1 for r in results if r.success)
        failure_count = batch_size - success_count

        summary = {
            "total_requests": batch_size,
            "successful": success_count,
            "failed": failure_count,
            "duration_ms": round(duration_ms, 2),
            "avg_duration_ms": round(duration_ms / batch_size, 2),
        }

        logger.info(
            "batch_counterfactual_completed",
            extra={
                "request_id": request_id,
                "summary": summary,
            },
        )

        return BatchCounterfactualResponse(
            results=results,
            summary=summary,
            metadata=create_response_metadata(request_id),
        )

    except Exception as e:
        logger.error("batch_counterfactual_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process batch counterfactual analysis. Check logs for details.",
        )
