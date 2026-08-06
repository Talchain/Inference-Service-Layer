"""
Sensitivity analysis and optimization endpoints.

``/sensitivity`` is WITHDRAWN (ROADMAP 2.704) — see the route docstring and
``src/api/withdrawn.py``. ``/optimise`` is untouched: it is an honest linear
grid search with disclosed limitations, dark only for want of a producer.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.withdrawn import DEFAULT_OUTCOME_REASON, refuse_withdrawn
from src.models.isl_metadata import create_isl_metadata
from src.models.metadata import create_response_metadata
from src.models.requests import OptimisationRequest
from src.models.responses import OptimisationResponse
from src.services.continuous_optimizer import ContinuousOptimizer

router = APIRouter()
logger = logging.getLogger(__name__)

ANALYSIS_SENSITIVITY_ROUTE = "/api/v1/analysis/sensitivity"

# Initialize services
continuous_optimizer = ContinuousOptimizer()


@router.post(
    "/sensitivity",
    summary="[WITHDRAWN] Perform sensitivity analysis",
    description="""
    **WITHDRAWN (ROADMAP 2.704) — this route answers 501 and computes nothing.**

    It previously reported assumption importance, robustness scores and
    breakpoints. The underlying analyzer string-split equation text hunting for
    a `coef * var` pattern and returned a hardcoded **50000.0** outcome whenever
    that parsing failed, and its per-assumption error path reported analysis
    FAILURE as maximal robustness (`robustness_score=1.0`, "Analysis failed -
    assuming robust").

    The seam was broken as well: this handler called `sensitivity_analyzer
    .analyze(...)`, a method that does not exist on the analyzer, so every call
    would have 500'd even if the router were mounted.

    The live robustness surface is `/api/v1/robustness/*`.
    """,
    responses={
        501: {"description": "Capability withdrawn — output was fabricated, not computed"},
    },
)
async def analyze_sensitivity(request: Request) -> JSONResponse:
    """Refuse: sensitivity analysis defaulted to a constant and called a
    method that does not exist.

    Takes the raw request rather than a validated model on purpose — no input
    is "valid" for a capability that does not exist, and a 422 would imply a
    well-formed body would have been answered.
    """
    return refuse_withdrawn(
        route=ANALYSIS_SENSITIVITY_ROUTE,
        reason=DEFAULT_OUTCOME_REASON,
        request=request,
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
