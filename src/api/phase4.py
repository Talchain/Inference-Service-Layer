"""
Phase 4: Sequential Decisions & Conditional Recommendations API.

Provides endpoints for:
- Conditional recommendation generation
- Sequential decision analysis (backward induction)
- Policy tree retrieval
- Stage sensitivity analysis
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from src.models.metadata import create_response_metadata
from src.models.requests import (
    ConditionalRecommendRequest,
    SequentialAnalysisRequest,
    StageSensitivityRequest,
)
from src.models.responses import (
    ConditionalRecommendResponse,
    PolicyTreeResponse,
    SequentialAnalysisResponse,
    StageSensitivityResponse,
)
from src.services.conditional_recommender import ConditionalRecommendationEngine
from src.services.sequential_decision import SequentialDecisionEngine

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
conditional_engine = ConditionalRecommendationEngine()
sequential_engine = SequentialDecisionEngine()

# Cache for policy trees (in production, use Redis)
_policy_cache: dict = {}


@router.post(
    "/conditional-recommend",
    response_model=ConditionalRecommendResponse,
    summary="Generate conditional recommendations",
    description="""
    Generate actionable conditions that qualify recommendations.

    Real decisions aren't binary. Users need to know "Choose A if X, otherwise B"
    rather than just "Choose A."

    **Condition Types:**
    - **threshold**: Parameter thresholds where ranking flips (e.g., "If ROI < 0.3, choose B")
    - **dominance**: When weighting changes make one option dominate
    - **risk_profile**: How recommendations change with risk tolerance
    - **scenario**: Clustered parameter combinations (pessimistic/optimistic)

    **Robustness Summary:**
    - **robust**: Recommendation unlikely to change
    - **moderate**: Some conditions could flip recommendation
    - **fragile**: Many conditions could flip recommendation

    **Use when:** Making strategic decisions that need qualification.
    """,
    responses={
        200: {"description": "Conditional recommendations generated successfully"},
        400: {"description": "Invalid input (e.g., fewer than 2 options)"},
        500: {"description": "Internal computation error"},
    },
)
async def generate_conditional_recommendations(
    request: ConditionalRecommendRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ConditionalRecommendResponse:
    """
    Generate conditional recommendations for ranked options.

    Args:
        request: Conditional recommendation request with ranked options

    Returns:
        ConditionalRecommendResponse with primary and conditional recommendations
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "conditional_recommend_request",
            extra={
                "request_id": request_id,
                "run_id": request.run_id,
                "num_options": len(request.ranked_options),
                "condition_types": request.condition_types,
                "max_conditions": request.max_conditions,
            },
        )

        result = conditional_engine.generate_recommendations(request)

        logger.info(
            "conditional_recommend_completed",
            extra={
                "request_id": request_id,
                "primary_option": result.primary_recommendation.option_id,
                "num_conditions": len(result.conditional_recommendations),
                "robustness": result.robustness_summary.recommendation_stability,
            },
        )

        # Add metadata
        result.metadata = create_response_metadata(request_id)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "conditional_recommend_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate conditional recommendations. Check logs for details.",
        )


@router.post(
    "/sequential",
    response_model=SequentialAnalysisResponse,
    summary="Analyze sequential decision problem",
    description="""
    Solve multi-stage decision problems using backward induction.

    Many strategic decisions are sequential: "If we launch, then we'll need to
    decide on pricing. If pricing works, we'll decide on expansion."
    Optimal strategy requires reasoning backward from final outcomes.

    **Algorithm:**
    1. Start from final stage (terminal payoffs)
    2. For each state at stage T, compute optimal action and value
    3. Move backward, computing optimal action given immediate payoff + continuation value
    4. Repeat until stage 0

    **Value of Flexibility:**
    Compares optimal policy (deciding at each stage with information) vs
    committing upfront (ignoring future information).

    **Use when:** Planning multi-stage strategies with uncertainty resolution.
    """,
    responses={
        200: {"description": "Sequential analysis completed successfully"},
        400: {"description": "Invalid input (e.g., invalid graph structure)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_sequential_decision(
    request: SequentialAnalysisRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> SequentialAnalysisResponse:
    """
    Analyze a sequential decision problem.

    Args:
        request: Sequential analysis request with graph and stages

    Returns:
        SequentialAnalysisResponse with optimal policy and stage analyses
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "sequential_analysis_request",
            extra={
                "request_id": request_id,
                "num_nodes": len(request.graph.nodes),
                "num_edges": len(request.graph.edges),
                "num_stages": len(request.stages),
                "discount_factor": request.discount_factor,
                "risk_tolerance": request.risk_tolerance,
            },
        )

        result = sequential_engine.analyze(request)

        # Cache for policy-tree endpoint
        cache_key = f"seq_{request_id}"
        _policy_cache[cache_key] = {
            "request": request,
            "result": result,
        }

        logger.info(
            "sequential_analysis_completed",
            extra={
                "request_id": request_id,
                "expected_value": result.optimal_policy.expected_total_value,
                "value_of_flexibility": result.value_of_flexibility,
                "timing_sensitivity": result.sensitivity_to_timing,
            },
        )

        # Add metadata
        result.metadata = create_response_metadata(request_id)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "sequential_analysis_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze sequential decision. Check logs for details.",
        )


@router.post(
    "/policy-tree",
    response_model=PolicyTreeResponse,
    summary="Get policy as decision tree",
    description="""
    Generate a decision tree representation of the optimal policy.

    Returns a tree structure where:
    - Decision nodes show optimal action
    - Chance nodes show outcome probabilities
    - Terminal nodes show payoffs

    **Use when:** Visualizing or exporting the optimal policy.
    """,
    responses={
        200: {"description": "Policy tree generated successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal computation error"},
    },
)
async def get_policy_tree(
    request: SequentialAnalysisRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> PolicyTreeResponse:
    """
    Generate policy tree from sequential analysis.

    Args:
        request: Sequential analysis request

    Returns:
        PolicyTreeResponse with tree structure
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "policy_tree_request",
            extra={
                "request_id": request_id,
                "num_nodes": len(request.graph.nodes),
                "num_stages": len(request.stages),
            },
        )

        result = sequential_engine.get_policy_tree(request)

        logger.info(
            "policy_tree_completed",
            extra={
                "request_id": request_id,
                "total_nodes": result.total_nodes,
                "total_stages": result.total_stages,
            },
        )

        # Add metadata
        result.metadata = create_response_metadata(request_id)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "policy_tree_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate policy tree. Check logs for details.",
        )


@router.post(
    "/stage-sensitivity",
    response_model=StageSensitivityResponse,
    summary="Stage-by-stage sensitivity analysis",
    description="""
    Perform sensitivity analysis for each stage of a sequential decision.

    Tests how robust the optimal policy is to parameter changes at each stage.

    **Metrics:**
    - **parameter_sensitivities**: How much each parameter affects stage value
    - **policy_changes_at**: Parameter values where optimal action changes
    - **robustness_score**: Overall robustness (0-1)

    **Use when:** Understanding which parameters matter most at each stage.
    """,
    responses={
        200: {"description": "Stage sensitivity analysis completed successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_stage_sensitivity(
    request: StageSensitivityRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> StageSensitivityResponse:
    """
    Perform stage-by-stage sensitivity analysis.

    Args:
        request: Stage sensitivity request

    Returns:
        StageSensitivityResponse with per-stage results
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "stage_sensitivity_request",
            extra={
                "request_id": request_id,
                "num_stages": len(request.stages),
                "variation_range": request.variation_range,
            },
        )

        result = sequential_engine.stage_sensitivity(request)

        logger.info(
            "stage_sensitivity_completed",
            extra={
                "request_id": request_id,
                "overall_robustness": result.overall_robustness,
                "most_sensitive": result.most_sensitive_parameters,
            },
        )

        # Add metadata
        result.metadata = create_response_metadata(request_id)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "stage_sensitivity_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze stage sensitivity. Check logs for details.",
        )
