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
from src.models.requests import (
    BatchCounterfactualRequest,
    CausalValidationRequest,
    ConformalCounterfactualRequest,
    CounterfactualRequest,
    TransportabilityRequest,
)
from src.models.responses import (
    BatchCounterfactualResponse,
    CausalValidationResponse,
    ConformalCounterfactualResponse,
    CounterfactualResponse,
    ErrorCode,
    ErrorResponse,
    TransportabilityResponse,
)
from src.services.batch_counterfactual_engine import BatchCounterfactualEngine
from src.services.causal_transporter import CausalTransporter
from src.services.causal_validator import CausalValidator
from src.services.conformal_predictor import ConformalPredictor
from src.services.counterfactual_engine import CounterfactualEngine

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
causal_validator = CausalValidator()
counterfactual_engine = CounterfactualEngine()
batch_counterfactual_engine = BatchCounterfactualEngine()
causal_transporter = CausalTransporter()
conformal_predictor = ConformalPredictor(counterfactual_engine)


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
    - `identifiable`: Valid adjustment sets provided (with method, formula, assumptions)
    - `uncertain`: Potential issues detected, clarification needed
    - `cannot_identify`: Fundamental structural problems (with reason and suggestions)
    - `degraded`: Advanced analysis failed, fallback assessment provided
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


@router.post(
    "/counterfactual/batch",
    response_model=BatchCounterfactualResponse,
    summary="Batch counterfactual analysis with interaction detection",
    description="""
    Analyzes multiple counterfactual scenarios in a single request with interaction detection.

    Provides:
    - Individual predictions for each scenario
    - Interaction analysis (synergistic/antagonistic effects)
    - Scenario comparison and ranking
    - Marginal gains analysis

    **Features:**
    - Shared exogenous samples for determinism across scenarios
    - Automatic detection of variable interactions
    - Efficient batch processing
    - Consistent uncertainty quantification

    **Use when:** Comparing multiple intervention strategies or testing parameter combinations.

    **Example Use Cases:**
    - Testing price AND quality changes together
    - Comparing aggressive vs conservative strategies
    - Detecting when interventions amplify or cancel each other
    """,
    responses={
        200: {"description": "Batch counterfactual analysis completed successfully"},
        400: {"description": "Invalid input (e.g., too few scenarios, malformed model)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_batch_counterfactual(
    request: BatchCounterfactualRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> BatchCounterfactualResponse:
    """
    Perform batch counterfactual analysis with interaction detection.

    Args:
        request: Batch counterfactual request with multiple scenarios
        x_request_id: Optional request ID for tracing

    Returns:
        BatchCounterfactualResponse: Results for all scenarios with interactions
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "batch_counterfactual_request",
            extra={
                "request_id": request_id,
                "num_scenarios": len(request.scenarios),
                "outcome": request.outcome,
                "analyze_interactions": request.analyze_interactions,
                "samples": request.samples,
            },
        )

        result = batch_counterfactual_engine.generate_batch_counterfactuals(
            request=request,
            request_id=request_id,
        )

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "batch_counterfactual_completed",
            extra={
                "request_id": request_id,
                "num_scenarios": len(result.scenarios),
                "best_outcome": result.comparison.best_outcome,
                "most_robust": result.comparison.most_robust,
                "num_interactions": len(result.interactions.pairwise) if result.interactions else 0,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "batch_counterfactual_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform batch counterfactual analysis. Check logs for details.",
        )


@router.post(
    "/counterfactual/conformal",
    response_model=ConformalCounterfactualResponse,
    summary="Conformal prediction for counterfactuals",
    description="""
    Generate counterfactual predictions with conformal prediction intervals.

    Provides finite-sample valid prediction intervals with guaranteed coverage,
    offering provable uncertainty quantification beyond standard Monte Carlo methods.

    **Key Features:**
    - Distribution-free (no parametric assumptions)
    - Finite-sample validity (not asymptotic)
    - Guaranteed coverage: P(Y ∈ interval) ≥ 1-α
    - Adaptive to local uncertainty
    - Comparison to Monte Carlo intervals

    **Use when:** You need rigorous uncertainty guarantees for counterfactual
    predictions, especially in high-stakes decisions where coverage matters.

    **Example Use Cases:**
    - Safety-critical predictions requiring provable bounds
    - Regulatory compliance needing formal guarantees
    - Scientific research requiring rigorous uncertainty quantification
    - Model validation and calibration assessment

    **Requirements:**
    - Calibration data: At least 10 historical observations (more is better)
    - Exchangeability: Calibration and test points from same distribution
    """,
    responses={
        200: {"description": "Conformal prediction completed successfully"},
        400: {"description": "Invalid input or insufficient calibration data"},
        500: {"description": "Internal computation error"},
    },
)
async def conformal_counterfactual_prediction(
    request: ConformalCounterfactualRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ConformalCounterfactualResponse:
    """
    Generate counterfactual with conformal prediction interval.

    Args:
        request: Conformal counterfactual request with calibration data
        x_request_id: Optional request ID for tracing

    Returns:
        ConformalCounterfactualResponse: Prediction with guaranteed coverage
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "conformal_counterfactual_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "confidence_level": request.confidence_level,
                "calibration_size": len(request.calibration_data) if request.calibration_data else 0,
            },
        )

        result = conformal_predictor.predict_with_conformal_interval(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "conformal_counterfactual_completed",
            extra={
                "request_id": request_id,
                "guaranteed_coverage": result.coverage_guarantee.guaranteed_coverage,
                "calibration_size": result.calibration_quality.calibration_size,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "conformal_counterfactual_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform conformal prediction. Check logs for details.",
        )


@router.post(
    "/transport",
    response_model=TransportabilityResponse,
    summary="Analyze causal effect transportability",
    description="""
    Determines whether a causal effect identified in a source domain
    can be validly transported to a target domain.

    Provides:
    - Transportability assessment (yes/no)
    - Transport formula if transportable
    - Required assumptions with testability
    - Robustness assessment
    - Suggestions if not transportable

    **Use when:** Deciding if findings from one context (e.g., UK market)
    apply to another context (e.g., Germany market).

    **Example Use Cases:**
    - Does price sensitivity learned in UK work in France?
    - Can clinical trial results from one population apply to another?
    - Will marketing effects from online channels work offline?
    """,
    responses={
        200: {"description": "Transportability analysis completed successfully"},
        400: {"description": "Invalid input (e.g., malformed DAG, missing nodes)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_transportability(
    request: TransportabilityRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> TransportabilityResponse:
    """
    Analyze transportability of causal effects between domains.

    Args:
        request: Transportability request with source/target domains
        x_request_id: Optional request ID for tracing

    Returns:
        TransportabilityResponse: Transportability analysis results
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "transportability_request",
            extra={
                "request_id": request_id,
                "source_domain": request.source_domain.name,
                "target_domain": request.target_domain.name,
                "treatment": request.treatment,
                "outcome": request.outcome,
            },
        )

        result = causal_transporter.analyze_transportability(request)

        # Inject metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "transportability_completed",
            extra={
                "request_id": request_id,
                "transportable": result.transportable,
                "method": result.method,
                "robustness": result.robustness,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "transportability_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform transportability analysis. Check logs for details.",
        )
