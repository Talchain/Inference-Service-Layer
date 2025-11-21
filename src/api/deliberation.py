"""
Habermas Machine deliberation API endpoint.

Provides AI-mediated democratic team deliberation for achieving
genuine consensus through structured exploration of values and concerns.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.deliberation import DeliberationRequest, DeliberationResponse
from src.models.metadata import create_response_metadata
from src.services.deliberation_orchestrator import DeliberationOrchestrator
from src.utils.business_metrics import track_habermas_deliberation

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
deliberation_orchestrator = DeliberationOrchestrator()


@router.post(
    "/deliberate",
    response_model=DeliberationResponse,
    summary="Conduct Habermas Machine deliberation round",
    description="""
    AI-mediated democratic deliberation for team alignment.

    **Habermas Machine transforms team decision-making through:**
    - Structured value elicitation (what matters and why)
    - Automatic common ground identification
    - AI-generated consensus statements
    - Iterative refinement through team feedback
    - Convergence toward genuine alignment

    **Not:** Vote counting, majority rule, forced compromise
    **Is:** Democratic deliberation, mutual understanding, genuine consensus

    **Use this when:**
    - Making strategic team decisions
    - Need alignment across diverse perspectives
    - Want to surface and resolve disagreements productively
    - Building shared understanding, not just vote aggregation

    **Process:**
    1. Team members submit positions (values, concerns, rationales)
    2. System identifies common ground
    3. Generates consensus statement
    4. Team reviews and suggests edits
    5. Iterate until convergence

    **Returns:**
    - Common ground analysis
    - Generated consensus statement
    - Convergence assessment
    - Recommended next steps

    **Performance:** 200-800ms per round

    **Example workflow:**
    ```
    Round 1: Submit initial positions → Get consensus draft
    Round 2: Submit edit suggestions → Get refined consensus
    Round 3: Review → Converged!
    ```
    """,
    responses={
        200: {"description": "Deliberation round completed successfully"},
        400: {"description": "Invalid input (e.g., missing positions)"},
        500: {"description": "Internal computation error"},
    },
)
async def conduct_deliberation(
    request: DeliberationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> DeliberationResponse:
    """
    Conduct one round of Habermas Machine deliberation.

    Args:
        request: Deliberation request with team positions
        x_request_id: Optional request ID for tracing

    Returns:
        DeliberationResponse: Analysis, consensus, and next steps
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "deliberation_request",
            extra={
                "request_id": request_id,
                "session_id": request.session_id,
                "positions": len(request.positions),
                "has_previous_consensus": request.previous_consensus is not None,
                "edit_suggestions": (
                    len(request.edit_suggestions) if request.edit_suggestions else 0
                ),
            },
        )

        # Conduct deliberation round
        response = deliberation_orchestrator.conduct_deliberation_round(
            request=request,
            request_id=request_id,
        )

        # Track metrics
        track_habermas_deliberation(
            status=response.status,
            agreement_level=response.common_ground.agreement_level,
            round_number=response.round_number,
            converged=(response.status == "converged"),
        )

        logger.info(
            "deliberation_completed",
            extra={
                "request_id": request_id,
                "session_id": response.session_id,
                "round_number": response.round_number,
                "agreement_level": response.common_ground.agreement_level,
                "status": response.status,
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("deliberation_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to conduct deliberation: {str(e)}",
        )


@router.get(
    "/session/{session_id}",
    summary="Get deliberation session details",
    description="Retrieve complete deliberation session history and current state.",
)
async def get_deliberation_session(
    session_id: str,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    """
    Get deliberation session details.

    Args:
        session_id: Session identifier
        x_request_id: Optional request ID

    Returns:
        Session details with full history
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    logger.info(
        "session_retrieval_request",
        extra={"request_id": request_id, "session_id": session_id},
    )

    session = deliberation_orchestrator.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    return {
        "session": session.model_dump(),
        "_metadata": create_response_metadata(request_id).model_dump(),
    }
