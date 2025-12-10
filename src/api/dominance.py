"""
Dominance Detection API Endpoint.

Provides endpoint for detecting dominance relationships between options
and identifying the Pareto frontier.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from src.api.dependencies import get_dominance_analyzer
from src.models.isl_metadata import MetadataBuilder
from src.models.requests import DominanceRequest, ParetoRequest
from src.models.responses import DominanceResponse, ParetoResponse, ParetoFrontierOption
from src.services.dominance_analyzer import DominanceAnalyzer

router = APIRouter()
logger = logging.getLogger(__name__)


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
    dominance_analyzer: DominanceAnalyzer = Depends(get_dominance_analyzer)
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


@router.post(
    "/pareto",
    response_model=ParetoResponse,
    summary="Compute Pareto frontier (non-dominated options)",
    description="""
    Identifies the Pareto frontier - the set of non-dominated options.

    **Pareto Frontier Definition:**
    The Pareto frontier contains all options where no other option is
    better or equal on ALL criteria AND strictly better on AT LEAST ONE.

    **Returns:**
    - Frontier options with full scores
    - Dominated options with dominators
    - Frontier truncation flag if > max_frontier_size

    **Use when:**
    - Narrowing down to optimal choices
    - Exploring trade-offs between criteria
    - Presenting decision options to users

    **Algorithm:** O(n²) dominance detection via pairwise comparison
    **Limits:** 2-100 options, 1-10 criteria, max_frontier_size default 20
    """,
    responses={
        200: {"description": "Pareto frontier computed successfully"},
        400: {"description": "Invalid input (validation error)"},
        500: {"description": "Internal computation error"},
    },
)
async def compute_pareto_frontier(
    request: ParetoRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
    dominance_analyzer: DominanceAnalyzer = Depends(get_dominance_analyzer)
) -> ParetoResponse:
    """
    Compute Pareto frontier from options.

    Args:
        request: Pareto request with options, criteria, and max_frontier_size
        x_request_id: Optional request ID for tracing

    Returns:
        ParetoResponse: Frontier options, dominated options, and metadata
    """
    # Generate request ID if not provided
    request_id = request.request_id or x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    # Initialize metadata builder
    metadata_builder = MetadataBuilder(request_id)

    try:
        logger.info(
            "pareto_request",
            extra={
                "request_id": request_id,
                "num_options": len(request.options),
                "num_criteria": len(request.criteria),
                "max_frontier_size": request.max_frontier_size,
            },
        )

        # Perform dominance analysis (Pareto frontier = non-dominated options)
        dominated_relations, non_dominated_ids = dominance_analyzer.analyze(
            options=request.options,
            criteria=request.criteria
        )

        # Build frontier options with full details
        option_lookup = {opt.option_id: opt for opt in request.options}
        frontier_options = [
            ParetoFrontierOption(
                option_id=opt_id,
                option_label=option_lookup[opt_id].option_label,
                scores=option_lookup[opt_id].scores
            )
            for opt_id in non_dominated_ids
        ]

        # Check if frontier needs truncation
        frontier_truncated = False
        if len(frontier_options) > request.max_frontier_size:
            frontier_truncated = True
            frontier_options = frontier_options[:request.max_frontier_size]

        logger.info(
            "pareto_completed",
            extra={
                "request_id": request_id,
                "frontier_size": len(frontier_options),
                "total_options": len(request.options),
                "frontier_pct": round(100 * len(frontier_options) / len(request.options), 1),
                "truncated": frontier_truncated,
            },
        )

        # Build response
        response = ParetoResponse(
            frontier=frontier_options,
            dominated=dominated_relations,
            frontier_size=len(non_dominated_ids),  # True size before truncation
            total_options=len(request.options),
            frontier_truncated=frontier_truncated,
        )

        # Add metadata
        response.metadata = metadata_builder.build(algorithm="skyline_pareto")

        return response

    except ValueError as e:
        logger.warning(
            "pareto_validation_error",
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
            "pareto_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to compute Pareto frontier. Check logs for details."
        )
