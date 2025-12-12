"""
Decision Robustness Suite API endpoints.

Unified robustness analysis endpoint combining sensitivity, robustness bounds,
value of information, and Pareto frontier analysis.

Brief 7: ISL — Decision Robustness Suite
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.config import get_settings
from src.models.decision_robustness import (
    RobustnessRequest,
    RobustnessResponse,
    RobustnessResult,
)
from src.models.isl_metadata import create_isl_metadata
from src.models.responses import ErrorCode, ErrorResponse, RecoveryHints
from src.services.decision_robustness_analyzer import (
    DecisionRobustnessAnalyzer,
    get_graph_hash,
)
from src.utils.business_metrics import track_business_metric

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Initialize analyzer
_analyzer = DecisionRobustnessAnalyzer()


@router.post(
    "/robustness",
    response_model=RobustnessResponse,
    summary="Unified Decision Robustness Analysis",
    description="""
Perform comprehensive robustness analysis combining:

- **Option Rankings**: All options ranked by expected utility with distributions
- **Recommendation**: Top recommendation with confidence level
- **Sensitivity Analysis**: Most sensitive parameters affecting the decision
- **Robustness Bounds**: Thresholds where parameters would flip recommendation
- **Value of Information**: EVPI/EVSI for uncertain parameters
- **Pareto Frontier**: Trade-offs for multi-goal decisions (if applicable)
- **Narrative**: Plain language summary of the analysis

## Response Schema

The response includes a unified `RobustnessResult` with all metrics:
- `option_rankings`: Ranked list with utility distributions
- `recommendation`: Top option with confidence and status
- `sensitivity`: Top sensitive parameters (configurable N)
- `robustness_label`: 'robust' | 'moderate' | 'fragile'
- `robustness_bounds`: Flip thresholds for key parameters
- `value_of_information`: EVPI/EVSI with data collection suggestions
- `pareto`: Pareto frontier for multi-goal (optional)
- `narrative`: Human-readable summary

## Robustness Classification

- **Robust**: No single parameter change within ±50% flips recommendation
- **Moderate**: Flip requires ±20-50% parameter change
- **Fragile**: Flip requires <±20% change (recommendation_status becomes 'exploratory')

## Performance

Typical analysis completes in <5 seconds for graphs with ≤50 nodes.
""",
    responses={
        200: {
            "description": "Successful robustness analysis",
            "model": RobustnessResponse,
        },
        400: {
            "description": "Invalid request",
            "model": ErrorResponse,
        },
        500: {
            "description": "Analysis error",
            "model": ErrorResponse,
        },
    },
)
async def analyze_robustness(
    request: RobustnessRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> RobustnessResponse:
    """
    Unified robustness analysis endpoint.

    Single call returns complete robustness metrics including sensitivity,
    robustness bounds, VoI, and Pareto frontier analysis.
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    # Check feature flag
    if not getattr(settings, "ENABLE_ROBUSTNESS_SUITE", True):
        raise HTTPException(
            status_code=503,
            detail="Decision Robustness Suite is disabled",
        )

    logger.info(
        "robustness_analysis_request",
        extra={
            "request_id": request_id,
            "num_options": len(request.options),
            "num_nodes": len(request.graph.nodes),
            "goal_node": request.utility.goal_node_id,
        },
    )

    try:
        # Validate request
        _validate_request(request)

        # Perform analysis
        result = _analyzer.analyze(request, request_id)

        # Compute hashes for tracking
        graph_hash = get_graph_hash(request.graph)
        response_hash = hashlib.sha256(
            json.dumps(result.model_dump(), default=str).encode()
        ).hexdigest()[:16]

        elapsed_ms = (time.time() - start_time) * 1000

        # Track metrics
        track_business_metric(
            "robustness_analysis",
            {
                "robustness_label": result.robustness_label.value,
                "num_options": len(request.options),
                "elapsed_ms": elapsed_ms,
                "recommendation_status": result.recommendation.recommendation_status.value,
            },
        )

        # Create metadata
        metadata = create_isl_metadata(
            request_id=request_id,
            computation_time_ms=elapsed_ms,
            algorithm="decision_robustness_suite_v1",
            cache_hit=False,
        )

        # Add hashes to metadata for outcome logging
        if hasattr(metadata, "extra"):
            metadata.extra = {"graph_hash": graph_hash, "response_hash": response_hash}

        logger.info(
            "robustness_analysis_completed",
            extra={
                "request_id": request_id,
                "elapsed_ms": elapsed_ms,
                "robustness_label": result.robustness_label.value,
                "top_option": result.recommendation.option_id,
            },
        )

        return RobustnessResponse(
            result=result,
            metadata=metadata,
        )

    except HTTPException:
        raise

    except ValueError as e:
        logger.warning(
            "robustness_validation_error",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code=ErrorCode.VALIDATION_ERROR.value,
                message=str(e),
                reason="validation_failed",
                recovery=RecoveryHints(
                    hints=[
                        "Check that all option IDs are unique",
                        "Ensure goal_node_id exists in the graph",
                        "Verify intervention nodes exist in the graph",
                    ],
                    suggestion="Fix validation errors and retry",
                ),
                retryable=False,
                source="isl",
                request_id=request_id,
            ).model_dump(),
        )

    except Exception as e:
        logger.error(
            "robustness_analysis_error",
            exc_info=True,
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code=ErrorCode.COMPUTATION_ERROR.value,
                message="Robustness analysis failed",
                reason="internal_error",
                recovery=RecoveryHints(
                    hints=[
                        "Simplify the graph if it has many nodes",
                        "Reduce Monte Carlo samples if timeout occurs",
                        "Check graph structure for disconnected components",
                    ],
                    suggestion="Retry with simpler configuration",
                ),
                retryable=True,
                source="isl",
                request_id=request_id,
            ).model_dump(),
        )


def _validate_request(request: RobustnessRequest) -> None:
    """
    Validate robustness request.

    Args:
        request: Request to validate

    Raises:
        ValueError: If validation fails
    """
    # Check goal node exists
    node_ids = {n.id for n in request.graph.nodes}

    if request.utility.goal_node_id not in node_ids:
        raise ValueError(
            f"Goal node '{request.utility.goal_node_id}' not found in graph"
        )

    # Check additional goals exist
    if request.utility.additional_goals:
        for goal in request.utility.additional_goals:
            if goal not in node_ids:
                raise ValueError(
                    f"Additional goal '{goal}' not found in graph"
                )

    # Check intervention nodes exist
    for option in request.options:
        for intervention_node in option.interventions.keys():
            if intervention_node not in node_ids:
                raise ValueError(
                    f"Intervention node '{intervention_node}' in option "
                    f"'{option.id}' not found in graph"
                )
