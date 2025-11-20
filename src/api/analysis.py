"""
Sensitivity analysis endpoints.

Provides endpoints for testing assumption robustness and identifying
critical factors.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.metadata import create_response_metadata
from src.models.requests import SensitivityAnalysisRequest
from src.models.responses import SensitivityAnalysisResponse
from src.services.sensitivity_analyzer import SensitivityAnalyzer

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
sensitivity_analyzer = SensitivityAnalyzer()


@router.post(
    "/sensitivity",
    response_model=SensitivityAnalysisResponse,
    summary="Perform sensitivity analysis",
    description="""
    Tests how robust conclusions are to changes in assumptions.

    Provides:
    - Assumption importance ranking
    - Impact assessment for each assumption
    - Overall robustness score
    - Critical breakpoints where conclusions flip

    **Use when:** Understanding which assumptions matter most.
    """,
    responses={
        200: {"description": "Sensitivity analysis completed successfully"},
        400: {"description": "Invalid input (e.g., no assumptions provided)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_sensitivity(
    request: SensitivityAnalysisRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> SensitivityAnalysisResponse:
    """
    Perform sensitivity analysis on assumptions.

    Args:
        request: Sensitivity analysis request with model and assumptions

    Returns:
        SensitivityAnalysisResponse: Sensitivity analysis results
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "sensitivity_analysis_request",
            extra={
                "baseline_result": request.baseline_result,
                "num_assumptions": len(request.assumptions),
                "assumption_names": [a.name for a in request.assumptions],
            },
        )

        result = sensitivity_analyzer.analyze(request)

        logger.info(
            "sensitivity_analysis_completed",
            extra={
                "robustness": result.robustness.overall,
                "num_breakpoints": len(result.robustness.breakpoints),
                "critical_assumptions": [
                    a.name for a in result.assumptions if a.importance == "critical"
                ],
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("sensitivity_analysis_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to perform sensitivity analysis. Check logs for details.",
        )
