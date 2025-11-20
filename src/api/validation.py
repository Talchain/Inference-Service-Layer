"""
Advanced model validation endpoints.

Provides comprehensive validation for causal models.
"""

import logging

from fastapi import APIRouter, HTTPException

from src.models.phase1_models import (
    AdvancedValidationRequest,
    AdvancedValidationResponse,
)
from src.services.advanced_validator import AdvancedModelValidator

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
advanced_validator = AdvancedModelValidator()


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
) -> AdvancedValidationResponse:
    """
    Perform advanced model validation.

    Args:
        request: Validation request with DAG and optional structural model

    Returns:
        AdvancedValidationResponse: Validation results and suggestions
    """
    try:
        logger.info(
            "validation_request",
            extra={
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
                "quality_level": quality_level.value,
                "quality_score": quality_score,
                "num_suggestions": len(suggestions),
            },
        )

        return AdvancedValidationResponse(
            overall_quality=quality_level,
            quality_score=quality_score,
            validation_results=validation_results,
            suggestions=suggestions,
            best_practices=best_practices,
            explanation=explanation,
        )

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
