"""
Preference elicitation endpoints.

- ``/elicit`` generates counterfactual queries for preference learning
  (ActiVA). Untouched — the information-gain query ranking is real.
- ``/update`` is WITHDRAWN (ROADMAP 2.704): it fed a fabricated, scenario-less
  query to a real Bayesian updater. See the route docstring and
  ``src/api/withdrawn.py``.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.withdrawn import FABRICATED_QUERY_REASON, refuse_withdrawn
from src.models.metadata import create_response_metadata
from src.models.phase1_models import (
    PreferenceElicitationRequest,
    PreferenceElicitationResponse,
)
from src.services.preference_elicitor import PreferenceElicitor
from src.services.user_storage import UserStorage
from src.utils.business_metrics import (
    track_activa_query_generated,
    track_activa_information_gain,
)

router = APIRouter()
logger = logging.getLogger(__name__)

PREFERENCES_UPDATE_ROUTE = "/api/v1/preferences/update"

# Initialize services.
# `BeliefUpdater` was instantiated here for /update only; with that route
# withdrawn it has no caller, so the instantiation is gone. The service module
# itself is retained (its posterior machinery is real and is the substrate a
# future, honestly-fed belief-update route would use).
preference_elicitor = PreferenceElicitor()
user_storage = UserStorage()


@router.post(
    "/elicit",
    response_model=PreferenceElicitationResponse,
    summary="Generate preference elicitation queries",
    description="""
    Generates counterfactual queries to efficiently learn user preferences.

    Uses ActiVA algorithm to maximize information gain with minimal queries.
    Queries are ranked by expected reduction in uncertainty about user values.

    **Algorithm:** Information-theoretic query selection
    - Computes: H(current beliefs) - E[H(posterior beliefs)]
    - Selects queries that maximize expected information gain
    - Typically learns preferences in 5-7 questions

    **Use when:**
    - Starting preference elicitation for a new user/context
    - User has answered previous queries (provide updated beliefs)
    - Need to refine understanding of user priorities

    **Returns:**
    - Ranked counterfactual queries with scenarios
    - Query selection strategy and rationale
    - Expected information gain per query
    - Estimated remaining queries needed
    """,
    responses={
        200: {"description": "Queries generated successfully"},
        400: {"description": "Invalid input (e.g., empty context)"},
        500: {"description": "Internal computation error"},
    },
)
async def elicit_preferences(
    request: PreferenceElicitationRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> PreferenceElicitationResponse:
    """
    Generate preference elicitation queries.

    Args:
        request: Preference elicitation request with context and beliefs
        x_request_id: Optional request ID for tracing

    Returns:
        PreferenceElicitationResponse: Ranked queries with strategy info
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "preference_elicitation_request",
            extra={
                "request_id": request_id,
                "user_id": _hash_user_id(request.user_id),
                "domain": request.context.domain,
                "num_variables": len(request.context.variables),
                "num_queries": request.num_queries,
                "has_current_beliefs": request.current_beliefs is not None,
            },
        )

        # Get current beliefs from storage if not provided
        current_beliefs = request.current_beliefs
        if current_beliefs is None:
            stored_beliefs = user_storage.get_beliefs(request.user_id)
            if stored_beliefs:
                current_beliefs = stored_beliefs
                logger.info(
                    "loaded_beliefs_from_storage",
                    extra={"user_id": _hash_user_id(request.user_id)},
                )

        # Generate queries (returns tuple of queries and strategy)
        queries, strategy = preference_elicitor.generate_queries(
            context=request.context,
            current_beliefs=current_beliefs,
            num_queries=request.num_queries,
        )

        # Calculate total expected information gain
        expected_info_gain = sum(q.information_gain for q in queries)

        # Track metrics
        for query in queries:
            track_activa_query_generated()
            track_activa_information_gain(query.information_gain)

        # Estimate remaining queries
        # Rough heuristic: need ~5 queries per 0.5 uncertainty
        if current_beliefs:
            avg_uncertainty = sum(current_beliefs.uncertainty_estimates.values()) / len(
                current_beliefs.uncertainty_estimates
            )
            estimated_remaining = max(0, int(avg_uncertainty * 10))
        else:
            estimated_remaining = 5  # Default for first elicitation

        # Generate explanation
        from src.models.shared import ExplanationMetadata

        explanation = ExplanationMetadata(
            summary=f"Generated {len(queries)} queries using {strategy.type.value} strategy",
            reasoning=strategy.rationale,
            technical_basis=f"ActiVA algorithm: Information gain computed via Monte Carlo sampling (1000 samples) to estimate H(current) - E[H(posterior)]",
            assumptions=[f"Focus area: {area}" for area in strategy.focus_areas],
        )

        logger.info(
            "preference_elicitation_completed",
            extra={
                "request_id": request_id,
                "user_id": _hash_user_id(request.user_id),
                "num_queries": len(queries),
                "strategy": strategy.type.value,
                "expected_info_gain": expected_info_gain,
            },
        )

        response = PreferenceElicitationResponse(
            queries=queries,
            strategy=strategy,
            expected_information_gain=expected_info_gain,
            estimated_queries_remaining=estimated_remaining,
            explanation=explanation,
        )

        # Inject metadata
        response.metadata = create_response_metadata(request_id)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "preference_elicitation_error",
            extra={
                "user_id": _hash_user_id(request.user_id),
                "error": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate preference queries. Check logs for details.",
        )


@router.post(
    "/update",
    summary="[WITHDRAWN] Update user beliefs from response",
    description="""
    **WITHDRAWN (ROADMAP 2.704) — this route answers 501 and computes nothing.**

    The Bayesian machinery underneath (`BeliefUpdater`) is real, but this route
    fed it a **fabricated query**: it constructed a placeholder
    `CounterfactualQuery` whose `scenario_a`/`scenario_b` had EMPTY `outcomes`
    and no trade-offs ("Placeholder - in production, retrieve from storage"),
    and the likelihood `P(response | beliefs)` is computed FROM those scenarios.
    The posterior therefore could not reflect the question the user actually
    answered, while the response reported "updated beliefs", learning progress
    and readiness for recommendations.

    The missing piece is a query-storage seam that was never built. `/elicit`
    is unaffected and still generates queries.
    """,
    responses={
        501: {"description": "Capability withdrawn — output was fabricated, not computed"},
    },
)
async def update_beliefs(request: Request) -> JSONResponse:
    """Refuse: the belief update ran against a query with no scenarios.

    Takes the raw request rather than a validated model on purpose — no input
    is "valid" for a capability that does not exist, and a 422 would imply a
    well-formed body would have been answered.
    """
    return refuse_withdrawn(
        route=PREFERENCES_UPDATE_ROUTE,
        reason=FABRICATED_QUERY_REASON,
        request=request,
    )


def _hash_user_id(user_id: str) -> str:
    """
    Hash user ID for privacy in logs.

    Args:
        user_id: User identifier

    Returns:
        Hashed user ID (first 16 chars of SHA256)
    """
    import hashlib

    return hashlib.sha256(user_id.encode()).hexdigest()[:16]
