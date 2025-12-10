"""
Dominance Detection API Endpoint.

Provides endpoint for detecting dominance relationships between options
and identifying the Pareto frontier.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.isl_metadata import MetadataBuilder
from src.models.requests import DominanceRequest
from src.models.responses import DominanceResponse
from src.services.dominance_analyzer import DominanceAnalyzer

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
dominance_analyzer = DominanceAnalyzer()


@router.post(
    "/dominance",
    response_model=DominanceResponse,
    summary="Detect dominance relationships between options",
    description="""
    Identifies which options are dominated by others across multiple criteria.

    **Dominance Definition:**
    Option B dominates option A if:
    - B is better or equal to A on ALL criteria AND
    - B is strictly better than A on AT LEAST ONE criterion

    **Returns:**
    - Dominated options with dominators and degree
    - Non-dominated options (Pareto frontier)
    - Performance metadata

    **Use when:**
    - Eliminating inferior options before detailed analysis
    - Identifying Pareto-optimal choices
    - Understanding trade-offs between options

    **Algorithm:** O(n²) pairwise comparison
    **Limits:** 2-100 options, 1-10 criteria
    """,
    responses={
        200: {"description": "Dominance analysis completed successfully"},
        400: {"description": "Invalid input (validation error)"},
        500: {"description": "Internal computation error"},
    },
)
async def detect_dominance(
    request: DominanceRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> DominanceResponse:
    """
    Detect dominance relationships between options.

    Args:
        request: Dominance detection request with options and criteria
        x_request_id: Optional request ID for tracing

    Returns:
        DominanceResponse: Dominated options and Pareto frontier
    """
    # Generate request ID if not provided
    request_id = request.request_id or x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    # Initialize metadata builder
    metadata_builder = MetadataBuilder(request_id)

    try:
        logger.info(
            "dominance_request",
            extra={
                "request_id": request_id,
                "num_options": len(request.options),
                "num_criteria": len(request.criteria),
                "option_ids": [opt.option_id for opt in request.options],
                "criteria": request.criteria,
            },
        )

        # Perform dominance analysis
        dominated_relations, non_dominated_ids = dominance_analyzer.analyze(
            options=request.options,
            criteria=request.criteria
        )

        # Calculate frontier statistics
        total_options = len(request.options)
        frontier_size = len(non_dominated_ids)

        logger.info(
            "dominance_completed",
            extra={
                "request_id": request_id,
                "num_dominated": len(dominated_relations),
                "frontier_size": frontier_size,
                "frontier_pct": round(100 * frontier_size / total_options, 1),
            },
        )

        # Build response
        response = DominanceResponse(
            dominated=dominated_relations,
            non_dominated_ids=non_dominated_ids,
            total_options=total_options,
            frontier_size=frontier_size,
        )

        # Add metadata
        response.metadata = metadata_builder.build(algorithm="pairwise_dominance")

        return response

    except ValueError as e:
        # Validation errors (should be caught by Pydantic, but handle just in case)
        logger.warning(
            "dominance_validation_error",
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
            "dominance_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform dominance analysis. Check logs for details."
        )
