"""
Preference elicitation endpoints.

Provides endpoints for:
- Generating counterfactual queries for preference learning (ActiVA)
- Updating user belief models based on responses (Bayesian inference)
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.models.phase1_models import (
    CounterfactualQuery,
    PreferenceElicitationRequest,
    PreferenceElicitationResponse,
    PreferenceUpdateRequest,
    PreferenceUpdateResponse,
)
from src.services.belief_updater import BeliefUpdater
from src.services.preference_elicitor import PreferenceElicitor
from src.services.user_storage import UserStorage

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
preference_elicitor = PreferenceElicitor()
belief_updater = BeliefUpdater()
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
) -> PreferenceElicitationResponse:
    """
    Generate preference elicitation queries.

    Args:
        request: Preference elicitation request with context and beliefs

    Returns:
        PreferenceElicitationResponse: Ranked queries with strategy info
    """
    try:
        logger.info(
            "preference_elicitation_request",
            extra={
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
                "user_id": _hash_user_id(request.user_id),
                "num_queries": len(queries),
                "strategy": strategy.type.value,
                "expected_info_gain": expected_info_gain,
            },
        )

        return PreferenceElicitationResponse(
            queries=queries,
            strategy=strategy,
            expected_information_gain=expected_info_gain,
            estimated_queries_remaining=estimated_remaining,
            explanation=explanation,
        )

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
    response_model=PreferenceUpdateResponse,
    summary="Update user beliefs from response",
    description="""
    Updates user belief model based on their preference response.

    Uses Bayesian inference to update probability distributions over:
    - Value weights (importance of different outcomes)
    - Risk tolerance
    - Uncertainty estimates

    **Algorithm:** Bayesian belief updating
    - Computes likelihood: P(response | beliefs)
    - Updates posterior: P(beliefs | response) ∝ P(response | beliefs) × P(beliefs)
    - Reduces uncertainty with each response

    **Use when:**
    - User has answered a preference query
    - Want to refine belief model before generating next queries

    **Returns:**
    - Updated belief model (stored automatically)
    - Learning progress summary
    - Next batch of queries
    - Recommendation readiness status
    """,
    responses={
        200: {"description": "Beliefs updated successfully"},
        400: {"description": "Invalid input (e.g., unknown query_id)"},
        500: {"description": "Internal computation error"},
    },
)
async def update_beliefs(
    request: PreferenceUpdateRequest,
) -> PreferenceUpdateResponse:
    """
    Update user beliefs based on preference response.

    Args:
        request: Preference update request with query response

    Returns:
        PreferenceUpdateResponse: Updated beliefs and next queries
    """
    try:
        logger.info(
            "preference_update_request",
            extra={
                "user_id": _hash_user_id(request.user_id),
                "query_id": request.query_id,
                "response": request.response.value,
                "confidence": request.confidence,
            },
        )

        # Get current beliefs from storage
        current_beliefs = user_storage.get_beliefs(request.user_id)

        if current_beliefs is None:
            logger.error(
                "beliefs_not_found",
                extra={"user_id": _hash_user_id(request.user_id)},
            )
            raise HTTPException(
                status_code=400,
                detail=f"No beliefs found for user. Please start with /elicit endpoint.",
            )

        # Retrieve the query that was answered
        # In a full implementation, we'd store queries in Redis
        # For now, we'll work with the belief model directly
        # The query_id is used for tracking/logging purposes

        # Note: In production, you'd retrieve the full CounterfactualQuery from storage
        # using request.query_id. For this implementation, the BeliefUpdater will
        # handle belief updates based on generic response patterns.

        # For now, we'll create a minimal query placeholder
        # This should be replaced with proper query retrieval from storage
        from src.models.phase1_models import Scenario

        # Placeholder - in production, retrieve from storage
        query = CounterfactualQuery(
            id=request.query_id,
            question="Query response recorded",
            scenario_a=Scenario(
                description="Scenario A",
                outcomes={},
                trade_offs=[],
            ),
            scenario_b=Scenario(
                description="Scenario B",
                outcomes={},
                trade_offs=[],
            ),
            information_gain=0.0,
        )

        # Update beliefs
        updated_beliefs = belief_updater.update_beliefs(
            current_beliefs=current_beliefs,
            query=query,
            response=request.response,
            confidence=request.confidence,
        )

        # Store updated beliefs
        user_storage.store_beliefs(
            user_id=request.user_id,
            beliefs=updated_beliefs,
        )

        # Add query to history
        user_storage.add_query_to_history(
            user_id=request.user_id,
            query=query,
            response=request.response.value,
        )

        # Get query count
        queries_completed = user_storage.get_query_count(request.user_id)

        # Generate learning summary
        learning_summary = belief_updater.generate_learning_summary(
            beliefs=updated_beliefs,
            queries_completed=queries_completed,
        )

        # Generate next batch of queries
        # Get context from the original elicitation request
        # In production, this would be retrieved from storage
        # For now, we'll use a minimal context based on the belief model
        from src.models.phase1_models import DecisionContext

        # Extract variables from belief model
        variables = list(updated_beliefs.value_weights.keys())

        next_context = DecisionContext(
            domain="general",
            variables=variables,
            constraints=None,
        )

        next_queries, _ = preference_elicitor.generate_queries(
            context=next_context,
            current_beliefs=updated_beliefs,
            num_queries=3,  # Generate 3 next queries by default
        )

        # Estimate remaining queries
        avg_uncertainty = sum(updated_beliefs.uncertainty_estimates.values()) / len(
            updated_beliefs.uncertainty_estimates
        )
        # Rough heuristic: need ~5 queries per 0.5 uncertainty
        estimated_remaining = max(0, int(avg_uncertainty * 10) - queries_completed)

        logger.info(
            "preference_update_completed",
            extra={
                "user_id": _hash_user_id(request.user_id),
                "queries_completed": queries_completed,
                "avg_uncertainty": round(avg_uncertainty, 3),
                "estimated_remaining": estimated_remaining,
                "ready_for_recommendations": learning_summary.ready_for_recommendations,
            },
        )

        return PreferenceUpdateResponse(
            updated_beliefs=updated_beliefs,
            queries_completed=queries_completed,
            estimated_queries_remaining=estimated_remaining,
            next_queries=next_queries,
            learning_summary=learning_summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "preference_update_error",
            extra={
                "user_id": _hash_user_id(request.user_id),
                "error": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to update beliefs. Check logs for details.",
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
