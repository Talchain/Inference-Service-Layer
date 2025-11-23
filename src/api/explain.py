"""
Contrastive explanation endpoints.

Provides endpoints for finding minimal interventions to achieve target outcomes.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.metadata import create_response_metadata
from src.models.requests import ContrastiveExplanationRequest
from src.models.responses import ContrastiveExplanationResponse
from src.services.contrastive_explainer import ContrastiveExplainer

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
contrastive_explainer = ContrastiveExplainer()


@router.post(
    "/contrastive",
    response_model=ContrastiveExplanationResponse,
    summary="Generate contrastive explanations with minimal interventions",
    description="""
    Find minimal interventions to achieve target outcomes.

    Transforms "what if X=50" predictions into actionable guidance:
    "Change X from 40 to 45 to achieve target."

    Provides:
    - Minimal single-variable interventions
    - Minimal multi-variable combinations (if allowed)
    - FACET-based robustness scoring for each intervention
    - Cost and feasibility estimates
    - Ranked recommendations by optimization criterion

    **Algorithm:**
    1. Binary search for minimal change to each feasible variable
    2. Grid search for minimal multi-variable combinations
    3. FACET robustness verification for each candidate
    4. Ranking by cost, change magnitude, or feasibility

    **Use when:** You need actionable recommendations for achieving specific outcomes,
    not just "what if" predictions.

    **Example:**
    - Current: Revenue = £40k
    - Target: Revenue = £50k
    - Result: "Increase Price from £40 to £45 (£5 increase) achieves target"
    """,
    responses={
        200: {"description": "Contrastive explanation generated successfully"},
        400: {"description": "Invalid input (e.g., no feasible variables, invalid constraints)"},
        500: {"description": "Internal computation error"},
    },
)
async def generate_contrastive_explanation(
    request: ContrastiveExplanationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ContrastiveExplanationResponse:
    """
    Generate contrastive explanation with minimal interventions.

    Args:
        request: Contrastive explanation request with model, current state, and target

    Returns:
        ContrastiveExplanationResponse: Minimal interventions achieving target
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "contrastive_explanation_request",
            extra={
                "request_id": request_id,
                "feasible_vars": request.constraints.feasible,
                "target_outcome": request.target_outcome,
                "max_changes": request.constraints.max_changes,
                "optimization_criterion": request.constraints.minimize,
            },
        )

        # Find minimal interventions
        result = contrastive_explainer.find_minimal_interventions(
            request=request,
            max_candidates=5,
            request_id=request_id,
        )

        # Add metadata
        result.metadata = create_response_metadata(request_id)

        logger.info(
            "contrastive_explanation_completed",
            extra={
                "request_id": request_id,
                "num_interventions": len(result.minimal_interventions),
                "best_intervention_cost": result.minimal_interventions[0].cost_estimate
                if result.minimal_interventions else None,
                "best_intervention_robustness": result.minimal_interventions[0].robustness.value
                if result.minimal_interventions else None,
            },
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "contrastive_explanation_error",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate contrastive explanation. Check logs for details.",
        )
