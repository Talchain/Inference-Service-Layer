"""
Advanced model validation endpoints.

Provides comprehensive validation for causal models including:
- Model structure validation
- Constraint feasibility checking
- Coherence analysis for inference results
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.isl_metadata import create_isl_metadata
from src.models.metadata import create_response_metadata
from src.models.phase1_models import (
    AdvancedValidationRequest,
    AdvancedValidationResponse,
)
from src.models.requests import CoherenceAnalysisRequest, FeasibilityRequest
from src.models.responses import CoherenceAnalysisResponse, FeasibilityResponse
from src.services.advanced_validator import AdvancedModelValidator
from src.services.coherence_analyzer import CoherenceAnalyzer
from src.services.feasibility_checker import FeasibilityChecker

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
advanced_validator = AdvancedModelValidator()
feasibility_checker = FeasibilityChecker()
coherence_analyzer = CoherenceAnalyzer()


@router.post(
    "/validate",
    response_model=AdvancedValidationResponse,
    summary="Perform advanced model validation",
    description="""
    Performs comprehensive validation of causal models.

    **Validation Levels:**
    - `BASIC`: Quick structural checks only
    - `STANDARD`: Structural + statistical checks (default)
    - `COMPREHENSIVE`: All checks + detailed analysis

    **Validation Types:**
    1. **Structural**: DAG properties, cycles, connectivity, size
    2. **Statistical**: Distributions, parameters, equations
    3. **Domain**: Best practices, naming conventions

    **Target**: 90%+ issue detection rate

    **Returns:**
    - Overall quality assessment (EXCELLENT/GOOD/ACCEPTABLE/POOR)
    - Numerical quality score (0-100)
    - Detailed validation results by category
    - Actionable suggestions for improvements
    - Best practice adherence checks

    **Use when:**
    - Building a new causal model
    - Before running causal inference
    - Reviewing model quality
    - Need suggestions for improvements
    """,
    responses={
        200: {"description": "Validation completed successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal computation error"},
    },
)
async def validate_model(
    request: AdvancedValidationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> AdvancedValidationResponse:
    """
    Perform advanced model validation.

    Args:
        request: Validation request with DAG and optional structural model
        x_request_id: Optional request ID for tracing

    Returns:
        AdvancedValidationResponse: Validation results and suggestions
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "validation_request",
            extra={
                "request_id": request_id,
                "num_nodes": len(request.dag.get("nodes", [])),
                "num_edges": len(request.dag.get("edges", [])),
                "level": request.validation_level.value,
                "has_structural_model": request.structural_model is not None,
            },
        )

        # Perform validation
        (
            quality_level,
            quality_score,
            validation_results,
            suggestions,
            best_practices,
        ) = advanced_validator.validate(
            dag=request.dag,
            structural_model=request.structural_model,
            context=request.context,
            validation_level=request.validation_level,
        )

        # Generate explanation
        from src.models.shared import ExplanationMetadata

        explanation = ExplanationMetadata(
            summary=f"Model quality: {quality_level.value.upper()} ({quality_score:.0f}/100)",
            reasoning=_generate_reasoning(validation_results, quality_score),
            technical_basis=f"Validation score computed from {len(validation_results.structural.checks + validation_results.statistical.checks + validation_results.domain.checks)} checks across structural, statistical, and domain categories",
            assumptions=[
                "Structural assumptions based on DAG properties",
                "Statistical assumptions based on specified distributions",
            ],
        )

        logger.info(
            "validation_completed",
            extra={
                "request_id": request_id,
                "quality_level": quality_level.value,
                "quality_score": quality_score,
                "num_suggestions": len(suggestions),
            },
        )

        response = AdvancedValidationResponse(
            overall_quality=quality_level,
            quality_score=quality_score,
            validation_results=validation_results,
            suggestions=suggestions,
            best_practices=best_practices,
            explanation=explanation,
        )

        # Inject metadata
        response.metadata = create_response_metadata(request_id)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "validation_error",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to validate model. Check logs for details.",
        )


def _generate_reasoning(validation_results, quality_score: float) -> str:
    """Generate reasoning for quality assessment."""
    all_checks = (
        validation_results.structural.checks
        + validation_results.statistical.checks
        + validation_results.domain.checks
    )

    num_pass = sum(1 for c in all_checks if c.status.value == "pass")
    num_fail = sum(1 for c in all_checks if c.status.value == "fail")
    num_warning = sum(1 for c in all_checks if c.status.value == "warning")

    reasoning = f"Performed {len(all_checks)} validation checks. "
    reasoning += f"Results: {num_pass} passed, {num_warning} warnings, {num_fail} failed. "

    if quality_score >= 90:
        reasoning += "Model follows best practices and has no critical issues."
    elif quality_score >= 75:
        reasoning += "Model is generally good with minor improvements possible."
    elif quality_score >= 50:
        reasoning += "Model is acceptable but has areas for improvement."
    else:
        reasoning += "Model has significant issues that should be addressed."

    return reasoning


@router.post(
    "/feasibility",
    response_model=FeasibilityResponse,
    summary="Check option feasibility against constraints",
    description="""
    Validates constraint specifications and checks which options satisfy
    or violate the specified constraints.

    **Constraint Types:**
    - `threshold`: Value must be above/below threshold
    - `budget`: Resource allocation constraint
    - `capacity`: Maximum capacity constraint
    - `dependency`: One option depends on another
    - `exclusion`: Mutually exclusive options
    - `requirement`: Required condition

    **Constraint Relations:**
    - `le`: Less than or equal (<=)
    - `ge`: Greater than or equal (>=)
    - `eq`: Equal (=)
    - `lt`: Less than (<)
    - `gt`: Greater than (>)

    **Constraint Priorities:**
    - `hard`: Must be satisfied (violation makes option infeasible)
    - `medium`: Should be satisfied (violation reported)
    - `soft`: Nice to have (only included if `include_partial_violations=true`)

    **Returns:**
    - Constraint validation results (are constraints properly specified?)
    - Feasibility results (which options pass/fail?)
    - Violation details (how much does each option violate?)
    - Warnings (edge cases, near-misses)

    **Use when:**
    - Checking if decision options meet business constraints
    - Filtering options before optimization
    - Understanding why certain options are infeasible
    """,
    responses={
        200: {"description": "Feasibility check completed successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal computation error"},
    },
)
async def check_feasibility(
    request: FeasibilityRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> FeasibilityResponse:
    """
    Check feasibility of options against constraints.

    Args:
        request: Feasibility request with graph, constraints, and options
        x_request_id: Optional request ID for tracing

    Returns:
        FeasibilityResponse: Feasibility check results
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "feasibility_check_request",
            extra={
                "request_id": request_id,
                "num_constraints": len(request.constraints),
                "num_options": len(request.options),
                "include_partial": request.include_partial_violations,
            },
        )

        # Perform feasibility check
        result = feasibility_checker.check_feasibility(request)

        logger.info(
            "feasibility_check_completed",
            extra={
                "request_id": request_id,
                "feasible_count": len(result.feasibility.feasible_options),
                "infeasible_count": len(result.feasibility.infeasible_options),
                "num_warnings": len(result.warnings),
            },
        )

        # Add metadata
        result.metadata = create_isl_metadata(
            request_id=request_id,
            computation_time_ms=0.0,  # Service doesn't track time internally
            algorithm="constraint_evaluation",
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "feasibility_check_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to check feasibility. Check logs for details.",
        )


@router.post(
    "/coherence",
    response_model=CoherenceAnalysisResponse,
    summary="Analyze coherence of inference results",
    description="""
    Provides coherence checks for inference results including:

    **Coherence Checks:**
    - **Negative Expected Value**: Detects when top option has negative expected value
    - **Close Races**: Detects when options are within the specified threshold
    - **Ranking Instability**: Tests if rankings change under small perturbations

    **Stability Classifications:**
    - `stable`: Rankings change in <10% of perturbations
    - `sensitive`: Rankings change in 10-30% of perturbations
    - `unstable`: Rankings change in >30% of perturbations

    **Analysis Features:**
    - Margin-to-second calculation (absolute and percentage)
    - Perturbation-based stability testing
    - Identification of most frequent alternative top option
    - Confidence interval overlap detection

    **Returns:**
    - Coherence analysis (top_option_positive, margin_to_second, ranking_stability)
    - Stability analysis (perturbation results, change rate)
    - Actionable recommendations
    - Warnings for edge cases

    **Use when:**
    - Validating inference results before presenting to users
    - Understanding decision robustness
    - Identifying close races that need more analysis
    - Integration with PLoT proxy for CEE narrative generation
    """,
    responses={
        200: {"description": "Coherence analysis completed successfully"},
        400: {"description": "Invalid input (e.g., fewer than 2 options)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_coherence(
    request: CoherenceAnalysisRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> CoherenceAnalysisResponse:
    """
    Analyze coherence of ranked options.

    Args:
        request: Coherence analysis request with options and parameters
        x_request_id: Optional request ID for tracing

    Returns:
        CoherenceAnalysisResponse: Coherence analysis results
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "coherence_analysis_request",
            extra={
                "request_id": request_id,
                "num_options": len(request.options),
                "perturbation_magnitude": request.perturbation_magnitude,
                "num_perturbations": request.num_perturbations,
                "close_race_threshold": request.close_race_threshold,
            },
        )

        # Perform coherence analysis
        result = coherence_analyzer.analyze(request)

        logger.info(
            "coherence_analysis_completed",
            extra={
                "request_id": request_id,
                "top_option_positive": result.coherence_analysis.top_option_positive,
                "ranking_stability": result.coherence_analysis.ranking_stability.value,
                "stability_score": result.coherence_analysis.stability_score,
                "num_warnings": len(result.coherence_analysis.warnings),
            },
        )

        # Add metadata
        result.metadata = create_isl_metadata(
            request_id=request_id,
            computation_time_ms=0.0,  # Service doesn't track time internally
            algorithm="perturbation_stability",
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "coherence_analysis_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze coherence. Check logs for details.",
        )
