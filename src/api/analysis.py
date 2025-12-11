"""
Sensitivity analysis and optimization endpoints.

Provides endpoints for testing assumption robustness, identifying
critical factors, and optimizing continuous decision variables.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.isl_metadata import create_isl_metadata
from src.models.metadata import create_response_metadata
from src.models.requests import OptimisationRequest, SensitivityAnalysisRequest
from src.models.responses import OptimisationResponse, SensitivityAnalysisResponse
from src.services.continuous_optimizer import ContinuousOptimizer
from src.services.sensitivity_analyzer import EnhancedSensitivityAnalyzer

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
# Using EnhancedSensitivityAnalyzer (aliased for compatibility)
sensitivity_analyzer = EnhancedSensitivityAnalyzer()
continuous_optimizer = ContinuousOptimizer()


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


@router.post(
    "/optimise",
    response_model=OptimisationResponse,
    summary="Optimize continuous decision variables",
    description="""
    Perform grid search optimization for continuous decision variables.

    Finds optimal values for decision variables that maximize (or minimize)
    a linear objective function subject to linear constraints.

    **Features:**
    - Multi-dimensional grid search
    - Linear constraint handling (<=, >=, =)
    - Confidence intervals for objective value
    - Sensitivity analysis (gradient, 5% tolerance range)
    - Edge case detection (boundary, flat objective, infeasible)

    **Algorithm:**
    1. Build uniform grid over decision variable bounds
    2. Evaluate objective at all grid points
    3. Filter feasible points (satisfy all constraints)
    4. Select point with best objective value
    5. Compute sensitivity analysis at optimum

    **Warnings:**
    - `NO_FEASIBLE_SOLUTION`: No grid points satisfy all constraints
    - `FLAT_OBJECTIVE`: Objective is constant across feasible region
    - `BOUNDARY_OPTIMUM`: Optimal point is at variable bounds
    - `MULTIPLE_OPTIMA`: Multiple near-optimal points found
    - `CONSTRAINT_ACTIVE`: Constraints are binding at optimum

    **Performance:** <2 seconds for 20-point grid with 2 variables.

    **Use when:** Finding optimal price, quantity, budget allocation.
    """,
    responses={
        200: {"description": "Optimization completed successfully"},
        400: {"description": "Invalid input (e.g., upper_bound < lower_bound)"},
        500: {"description": "Internal computation error"},
    },
)
async def optimize_continuous(
    request: OptimisationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> OptimisationResponse:
    """
    Optimize continuous decision variables using grid search.

    Args:
        request: Optimization request with objective, variables, constraints

    Returns:
        OptimisationResponse: Optimal point with sensitivity analysis
    """
    # Generate request ID if not provided
    request_id = request.request_id or x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "optimisation_request",
            extra={
                "request_id": request_id,
                "objective_direction": request.objective.direction,
                "num_variables": len(request.decision_variables),
                "variable_ids": [v.variable_id for v in request.decision_variables],
                "num_constraints": len(request.constraints) if request.constraints else 0,
                "grid_points": request.grid_points,
            },
        )

        # Create optimizer with seed if provided
        optimizer = ContinuousOptimizer(seed=request.seed) if request.seed else continuous_optimizer

        result = optimizer.optimize(request)

        # Log completion
        if result.optimal_point:
            logger.info(
                "optimisation_completed",
                extra={
                    "request_id": request_id,
                    "objective_value": result.optimal_point.objective_value,
                    "is_boundary": result.optimal_point.is_boundary,
                    "feasible_points": result.grid_metrics.feasible_points,
                    "computation_time_ms": result.grid_metrics.computation_time_ms,
                    "num_warnings": len(result.warnings),
                },
            )
        else:
            logger.warning(
                "optimisation_no_solution",
                extra={
                    "request_id": request_id,
                    "feasible_points": result.grid_metrics.feasible_points,
                    "num_warnings": len(result.warnings),
                },
            )

        # Add metadata
        result.metadata = create_isl_metadata(
            request_id=request_id,
            computation_time_ms=result.grid_metrics.computation_time_ms,
            algorithm="grid_search",
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "optimisation_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform optimization. Check logs for details.",
        )
