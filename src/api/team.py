"""
Team alignment endpoints.

Provides endpoints for finding common ground and aligned options
across different team perspectives.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.metadata import create_response_metadata
from src.models.requests import TeamAlignmentRequest
from src.models.responses import TeamAlignmentResponse
from src.services.team_aligner import TeamAligner

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
team_aligner = TeamAligner()


@router.post(
    "/align",
    response_model=TeamAlignmentResponse,
    summary="Find team alignment",
    description="""
    Identifies common ground across team perspectives and recommends
    options that best satisfy all stakeholders.

    Provides:
    - Shared goals and constraints
    - Options ranked by satisfaction score
    - Identified conflicts with resolutions
    - Top recommendation with rationale

    **Use when:** Making decisions with multiple stakeholders.
    """,
    responses={
        200: {"description": "Team alignment analysis completed successfully"},
        400: {"description": "Invalid input (e.g., no perspectives provided)"},
        500: {"description": "Internal computation error"},
    },
)
async def align_team(request: TeamAlignmentRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> TeamAlignmentResponse:
    """
    Find team alignment across perspectives.

    Args:
        request: Team alignment request with perspectives and options

    Returns:
        TeamAlignmentResponse: Alignment analysis with recommendations
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "team_alignment_request",
            extra={
                "num_perspectives": len(request.perspectives),
                "num_options": len(request.options),
                "roles": [p.role for p in request.perspectives],
            },
        )

        result = team_aligner.align(request)

        logger.info(
            "team_alignment_completed",
            extra={
                "agreement_level": result.common_ground.agreement_level,
                "recommended_option": result.recommendation.top_option,
                "num_conflicts": len(result.conflicts),
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("team_alignment_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to perform team alignment. Check logs for details.",
        )
