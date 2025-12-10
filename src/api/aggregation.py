"""
Multi-Criteria Aggregation API Endpoint.

Provides endpoint for aggregating option scores across multiple criteria
using various aggregation methods.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from src.api.dependencies import get_multi_criteria_aggregator
from src.models.isl_metadata import MetadataBuilder
from src.models.requests import MultiCriteriaRequest
from src.models.responses import MultiCriteriaResponse
from src.services.multi_criteria_aggregator import MultiCriteriaAggregator

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/multi-criteria",
    response_model=MultiCriteriaResponse,
    summary="Aggregate scores across multiple criteria",
    description="""
    Combines option scores across multiple criteria using configurable aggregation methods.

    **Aggregation Methods:**
    - `weighted_sum`: Compensatory, allows trade-offs (Score = Σ(weight_i × score_i) × 100)
    - `weighted_product`: Balanced, penalizes low scores (Score = ∏(score_i ^ weight_i) × 100)
    - `lexicographic`: Hierarchical, no trade-offs (Sort by highest-weight criterion first)

    **Features:**
    - Auto-normalizes weights that don't sum to 1.0 (with warning)
    - Percentile selection (p10=pessimistic, p50=expected, p90=optimistic)
    - Trade-off detection with configurable threshold
    - Validation warnings for transparency

    **Returns:**
    - Aggregated rankings sorted best to worst
    - Significant trade-offs between top options
    - Warnings about auto-corrections
    - Performance metadata

    **Use when:**
    - Combining inference results across multiple criteria
    - Comparing options with different strengths
    - Understanding trade-offs in decisions

    **Algorithm:** Depends on method (O(n) for all methods)
    **Limits:** 1-10 criteria, 2-100 options per criterion
    """,
    responses={
        200: {"description": "Aggregation completed successfully"},
        400: {"description": "Invalid input (validation error)"},
        500: {"description": "Internal computation error"},
    },
)
async def aggregate_multi_criteria(
    request: MultiCriteriaRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
    aggregator: MultiCriteriaAggregator = Depends(get_multi_criteria_aggregator)
) -> MultiCriteriaResponse:
    """
    Aggregate option scores across multiple criteria.

    Args:
        request: Multi-criteria aggregation request
        x_request_id: Optional request ID for tracing

    Returns:
        MultiCriteriaResponse: Aggregated rankings, trade-offs, warnings, metadata
    """
    # Generate request ID if not provided
    request_id = request.request_id or x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    # Initialize metadata builder
    metadata_builder = MetadataBuilder(request_id)

    try:
        logger.info(
            "aggregation_request",
            extra={
                "request_id": request_id,
                "num_criteria": len(request.criteria),
                "aggregation_method": request.aggregation_method,
                "percentile": request.percentile,
                "weights": request.weights,
            },
        )

        # Perform aggregation
        rankings, trade_offs, warnings = aggregator.aggregate(request)

        logger.info(
            "aggregation_completed",
            extra={
                "request_id": request_id,
                "num_rankings": len(rankings),
                "num_trade_offs": len(trade_offs),
                "num_warnings": len(warnings),
                "top_option": rankings[0].option_id if rankings else None,
            },
        )

        # Build response
        response = MultiCriteriaResponse(
            aggregated_rankings=rankings,
            trade_offs=trade_offs,
            warnings=warnings if warnings else None,
        )

        # Add metadata
        response.metadata = metadata_builder.build(algorithm=request.aggregation_method)

        return response

    except ValueError as e:
        logger.warning(
            "aggregation_validation_error",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {str(e)}"
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "aggregation_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform multi-criteria aggregation. Check logs for details."
        )
