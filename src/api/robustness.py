"""
Robustness analysis API endpoint.

Provides FACET-based robustness verification for counterfactual recommendations.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.metadata import create_response_metadata
from src.models.robustness import RobustnessRequest, RobustnessResponse
from src.services.robustness_analyzer import RobustnessAnalyzer
from src.utils.business_metrics import track_robustness_analysis

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
robustness_analyzer = RobustnessAnalyzer()


@router.post(
    "/analyze",
    response_model=RobustnessResponse,
    summary="Analyze robustness of counterfactual recommendation using FACET",
    description="""
    Verifies whether proposed intervention robustly achieves target outcomes
    by exploring intervention space and identifying robust regions.

    **FACET (Region-Based Robustness Analysis)**

    Instead of asking "What if price = £55?", FACET answers:
    - "Which price ranges reliably achieve £95k-£105k revenue?"
    - "How fragile is this recommendation to small changes?"
    - "Do I have flexibility in implementation?"

    **Use this when:**
    - Making high-stakes decisions
    - Need confidence in recommendations
    - Want to understand operating ranges
    - Assessing strategy fragility

    **Returns:**
    - Robust intervention regions (e.g., "price £52-£58 works")
    - Outcome guarantees (e.g., "revenue £95k-£105k")
    - Robustness score (0-1, higher = more robust)
    - Fragility warnings
    - Actionable recommendations

    **Performance:** 1-5 seconds depending on samples and dimensions

    **Example workflow:**
    1. Get counterfactual: "price=£55 → revenue=£100k"
    2. Check robustness: Is this recommendation stable?
    3. Receive: "Any price £52-£58 works" (robust) or "Only £54.8-£55.2 works" (fragile)
    4. Decide: Proceed with confidence or revise strategy
    """,
    responses={
        200: {"description": "Robustness analysis completed successfully"},
        400: {"description": "Invalid input (e.g., malformed model)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_robustness(
    request: RobustnessRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> RobustnessResponse:
    """
    Analyze robustness of counterfactual recommendation using FACET.

    Args:
        request: Robustness analysis request
        x_request_id: Optional request ID for tracing

    Returns:
        RobustnessResponse: Robustness analysis with regions and guarantees
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "robustness_analysis_request",
            extra={
                "request_id": request_id,
                "intervention_vars": len(request.intervention_proposal),
                "target_outcomes": len(request.target_outcome),
                "samples": request.min_samples,
                "perturbation_radius": request.perturbation_radius,
            },
        )

        # Perform FACET analysis
        analysis = robustness_analyzer.analyze_robustness(
            request=request,
            request_id=request_id,
        )

        # Track metrics
        track_robustness_analysis(
            status=analysis.status,
            robustness_score=analysis.robustness_score,
            is_fragile=analysis.is_fragile,
            regions_found=analysis.region_count,
        )

        # Create response with metadata
        response = RobustnessResponse(
            analysis=analysis,
            metadata=create_response_metadata(request_id),
        )

        logger.info(
            "robustness_analysis_completed",
            extra={
                "request_id": request_id,
                "status": analysis.status,
                "robustness_score": analysis.robustness_score,
                "regions_found": analysis.region_count,
                "is_fragile": analysis.is_fragile,
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("robustness_analysis_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform robustness analysis: {str(e)}",
        )
