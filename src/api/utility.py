"""
Utility Function Validation API Endpoint.

Provides endpoint for validating utility function specifications
for multi-goal aggregation.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.isl_metadata import create_isl_metadata
from src.models.requests import UtilityValidationRequest
from src.models.responses import UtilityValidationResponse
from src.services.utility_validator import UtilityValidator

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
utility_validator = UtilityValidator()


@router.post(
    "/validate",
    response_model=UtilityValidationResponse,
    summary="Validate utility function specification",
    description="""
    Validates a utility function specification for multi-goal aggregation.

    **Input Shape:**
    - `goals`: List of goals with optional weights, directions, and priorities
    - `aggregation_method`: Method for combining goal values
    - `risk_tolerance`: Risk attitude for utility computation

    **Aggregation Methods:**
    - `weighted_sum`: Linear combination U = Σ(w_i × v_i) - Compensatory, allows trade-offs
    - `weighted_product`: Geometric mean U = ∏(v_i ^ w_i) - Balanced, penalizes low scores
    - `lexicographic`: Priority-based sorting - Non-compensatory, no trade-offs
    - `min_max`: Pessimistic U = min(v_i) - Focus on worst-case goal

    **Default Behaviour (no weights specified):**
    - Equal weighting: Each goal gets weight = 1/n
    - Weights are always normalized to sum to 1.0

    **Risk Tolerance:**
    - `risk_neutral`: Use expected values only
    - `risk_averse`: Penalize variance (requires risk_coefficient)
    - `risk_seeking`: Prefer variance (optional risk_coefficient)

    **Validation Checks:**
    - Weight normalization (auto-normalizes with warning)
    - Goal ID uniqueness
    - Graph reference validation (if graph provided)
    - Aggregation method compatibility
    - Risk parameter validity

    **Returns:**
    - `valid`: Whether specification is valid for use
    - `normalised_weights`: Weights after normalization
    - `normalised_goals`: Full goal specifications after defaults applied
    - `default_behaviour_applied`: List of defaults that were applied
    - `warnings`: Non-fatal issues detected
    - `errors`: Fatal issues (if valid=false)

    **Use when:**
    - Setting up multi-goal decision problems
    - Validating utility specifications before aggregation
    - Understanding what defaults will be applied
    """,
    responses={
        200: {"description": "Validation completed (check 'valid' field for result)"},
        400: {"description": "Invalid request format"},
        500: {"description": "Internal validation error"},
    },
)
async def validate_utility(
    request: UtilityValidationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> UtilityValidationResponse:
    """
    Validate a utility function specification.

    Args:
        request: Utility validation request
        x_request_id: Optional request ID for tracing

    Returns:
        UtilityValidationResponse: Validation results
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "utility_validation_request",
            extra={
                "request_id": request_id,
                "num_goals": len(request.utility_spec.goals),
                "aggregation_method": request.utility_spec.aggregation_method.value,
                "risk_tolerance": request.utility_spec.risk_tolerance.value,
                "strict_mode": request.strict_mode,
                "has_graph": request.graph is not None,
            },
        )

        # Perform validation
        result = utility_validator.validate(request)

        logger.info(
            "utility_validation_completed",
            extra={
                "request_id": request_id,
                "valid": result.valid,
                "num_warnings": len(result.warnings),
                "num_errors": len(result.errors),
                "defaults_applied": len(result.default_behaviour_applied),
            },
        )

        # Add metadata
        result.metadata = create_isl_metadata(
            request_id=request_id,
            computation_time_ms=0.0,
            algorithm="utility_validation",
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "utility_validation_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to validate utility specification. Check logs for details.",
        )
