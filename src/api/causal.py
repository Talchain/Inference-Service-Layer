"""
Causal inference endpoints.

Provides endpoints for:
- Causal model validation (Y₀)
- Counterfactual analysis (FACET)
"""

import logging
import uuid

from fastapi import APIRouter, Header, HTTPException
from typing import Optional

from src.models.metadata import create_response_metadata
from src.models.requests import CausalValidationRequest, CounterfactualRequest
from src.models.responses import (
    CausalValidationResponse,
    CounterfactualResponse,
    ErrorCode,
    ErrorResponse,
)
from src.services.causal_validator import CausalValidator
from src.services.counterfactual_engine import CounterfactualEngine

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
causal_validator = CausalValidator()
counterfactual_engine = CounterfactualEngine()


@router.post(
    "/validate",
    response_model=CausalValidationResponse,
    summary="Validate causal model structure",
    description="""
    Validates whether a causal model (DAG) supports causal identification
    for a given treatment-outcome pair.

    Returns adjustment sets if identifiable, or specific issues if not.

    **Use when:** Building a decision model, before running scenarios.

    **Returns:**
    - `identifiable`: Valid adjustment sets provided
    - `uncertain`: Potential issues detected, clarification needed
    - `cannot_identify`: Fundamental structural problems
    """,
    responses={
        200: {"description": "Validation completed successfully"},
        400: {"description": "Invalid input (e.g., empty DAG, node not found)"},
        500: {"description": "Internal computation error"},
    },
)
async def validate_causal_model(
    request: CausalValidationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> CausalValidationResponse:
    """
    Validate causal model for identifiability.

    Args:
        request: Causal validation request with DAG and variables
        x_request_id: Optional request ID for tracing

    Returns:
        CausalValidationResponse: Validation results with adjustment sets or issues
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "causal_validation_request",
            extra={
                "request_id": request_id,
                "treatment": request.treatment,
                "outcome": request.outcome,
                "num_nodes": len(request.dag.nodes),
                "num_edges": len(request.dag.edges),
            },
        )

        result = causal_validator.validate(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "causal_validation_completed",
            extra={
                "request_id": request_id,
                "status": result.status,
                "confidence": result.confidence,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("causal_validation_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to validate causal model. Check logs for details.",
        )


@router.post(
    "/counterfactual",
    response_model=CounterfactualResponse,
    summary="Perform counterfactual analysis",
    description="""
    Analyzes what would happen under a counterfactual intervention.

    Provides:
    - Point estimates and confidence intervals
    - Uncertainty breakdown by source
    - Robustness analysis
    - Critical assumptions

    **Use when:** Evaluating "what if" scenarios for decision making.
    """,
    responses={
        200: {"description": "Counterfactual analysis completed successfully"},
        400: {"description": "Invalid input (e.g., malformed structural model)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_counterfactual(
    request: CounterfactualRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> CounterfactualResponse:
    """
    Perform counterfactual analysis.

    Args:
        request: Counterfactual request with structural model and intervention
        x_request_id: Optional request ID for tracing

    Returns:
        CounterfactualResponse: Counterfactual predictions with uncertainty
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "counterfactual_request",
            extra={
                "request_id": request_id,
                "outcome": request.outcome,
                "intervention": request.intervention,
                "num_variables": len(request.model.variables),
            },
        )

        result = counterfactual_engine.analyze(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "counterfactual_completed",
            extra={
                "request_id": request_id,
                "point_estimate": result.prediction.point_estimate,
                "uncertainty": result.uncertainty.overall,
                "robustness": result.robustness.score,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("counterfactual_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to perform counterfactual analysis. Check logs for details.",
        )
