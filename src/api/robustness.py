"""
Robustness analysis API endpoint.

Provides FACET-based robustness verification for counterfactual recommendations.

Supports two schema versions:
- v1: Legacy single-uncertainty model (fixed edge weights)
- v2: Dual uncertainty model (edge existence + strength distribution)
"""

import logging
import uuid
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Header, HTTPException
from pydantic import ValidationError

from src.models.metadata import create_response_metadata
from src.models.robustness import RobustnessRequest, RobustnessResponse
from src.models.robustness_v2 import (
    RobustnessRequestV2,
    RobustnessResponseV2,
    detect_schema_version,
)
from src.services.robustness_analyzer import RobustnessAnalyzer
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.business_metrics import track_robustness_analysis

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
robustness_analyzer = RobustnessAnalyzer()
robustness_analyzer_v2 = RobustnessAnalyzerV2()


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


@router.post(
    "/analyze/v2",
    response_model=RobustnessResponseV2,
    summary="Analyze robustness with dual uncertainty (v2.2 schema)",
    description="""
    Performs robustness analysis with dual uncertainty support:
    - **Structural uncertainty**: Probability that each edge exists (Bernoulli)
    - **Parametric uncertainty**: Distribution over effect magnitude (Normal)

    This enables answering:
    - "Is my decision robust to uncertainty about whether this relationship exists?"
    - "Is my decision robust to the effect being stronger/weaker than estimated?"

    **Input schema (v2.2):**
    - `graph`: Causal graph with nodes and edges
    - `edges`: Each has `exists_probability` and `strength: {mean, std}`
    - `options`: Decision alternatives to compare
    - `goal_node_id`: Target outcome to optimize

    **Output:**
    - Outcome distributions per option
    - Win probabilities
    - Sensitivity to edge existence AND magnitude
    - Overall robustness assessment

    **Performance:** 100-2000ms depending on n_samples (default 1000)
    """,
    responses={
        200: {"description": "Robustness analysis completed successfully"},
        400: {"description": "Invalid input schema"},
        422: {"description": "Validation error"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_robustness_v2(
    request: RobustnessRequestV2,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> RobustnessResponseV2:
    """
    Analyze robustness with dual uncertainty (v2.2 schema).

    Args:
        request: V2 robustness analysis request with dual uncertainty edges
        x_request_id: Optional request ID for tracing

    Returns:
        RobustnessResponseV2: Complete robustness analysis with sensitivity
    """
    # Use request_id from request or header
    request_id = request.request_id or x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "robustness_v2_analysis_request",
            extra={
                "request_id": request_id,
                "n_options": len(request.options),
                "n_edges": len(request.graph.edges),
                "n_samples": request.n_samples,
                "goal_node": request.goal_node_id,
            },
        )

        # Perform v2 analysis
        response = robustness_analyzer_v2.analyze(request)

        # Track metrics
        track_robustness_analysis(
            status="robust" if response.robustness.is_robust else "fragile",
            robustness_score=response.recommendation_confidence,
            is_fragile=not response.robustness.is_robust,
            regions_found=0,  # v2 doesn't use regions concept
        )

        logger.info(
            "robustness_v2_analysis_completed",
            extra={
                "request_id": request_id,
                "recommended_option": response.recommended_option_id,
                "confidence": response.recommendation_confidence,
                "is_robust": response.robustness.is_robust,
                "execution_time_ms": response.metadata.execution_time_ms,
            },
        )

        return response

    except ValidationError as e:
        logger.warning(
            "robustness_v2_validation_error",
            extra={"request_id": request_id, "errors": e.errors()},
        )
        raise HTTPException(status_code=422, detail=e.errors())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("robustness_v2_analysis_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform v2 robustness analysis: {str(e)}",
        )


@router.post(
    "/analyze/unified",
    summary="Unified robustness analysis (auto-detects schema version)",
    description="""
    Unified endpoint that accepts both v1 and v2.2 request schemas.

    Automatically detects schema version and routes to appropriate analyzer:
    - If request contains `graph` + `options`: Uses v2.2 analyzer
    - If request contains `causal_model`: Uses v1 analyzer

    **Backward compatible** - existing v1 clients continue to work.
    """,
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"description": "Unknown schema format"},
        422: {"description": "Validation error"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_robustness_unified(
    request: Dict[str, Any],
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> Union[RobustnessResponse, RobustnessResponseV2]:
    """
    Unified robustness analysis endpoint supporting both v1 and v2 schemas.

    Args:
        request: Raw request dict (v1 or v2 format)
        x_request_id: Optional request ID for tracing

    Returns:
        Response in corresponding schema version
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        # Detect schema version
        schema_version = detect_schema_version(request)

        if schema_version == "v2":
            # Validate and process v2 request
            validated_request = RobustnessRequestV2(**request)
            validated_request.request_id = (
                validated_request.request_id or request_id
            )
            return await analyze_robustness_v2(
                validated_request,
                x_request_id=request_id,
            )
        else:
            # Validate and process v1 request
            validated_request = RobustnessRequest(**request)
            return await analyze_robustness(
                validated_request,
                x_request_id=request_id,
            )

    except ValueError as e:
        # Schema detection failed
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("unified_robustness_analysis_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform robustness analysis: {str(e)}",
        )
