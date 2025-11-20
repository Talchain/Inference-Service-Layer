"""
Bayesian teaching endpoints.

Provides endpoints for pedagogically optimized teaching examples.
"""

import logging

from fastapi import APIRouter, HTTPException

from src.models.phase1_models import (
    BayesianTeachingRequest,
    BayesianTeachingResponse,
)
from src.services.bayesian_teacher import BayesianTeacher

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
bayesian_teacher = BayesianTeacher()


@router.post(
    "/teach",
    response_model=BayesianTeachingResponse,
    summary="Generate teaching examples",
    description="""
    Generates pedagogically optimized teaching examples using Bayesian teaching.

    Selects examples that maximize learning efficiency given user's current
    understanding. Uses information theory to choose examples that most
    efficiently teach the target concept.

    **Algorithm:** Bayesian Teaching (Zhu et al.)
    - Selects examples that maximize: P(understanding | example, current_beliefs)
    - Considers novelty, clarity, and relevance
    - Adapts to user's knowledge state

    **Supported Concepts:**
    - `confounding`: Understanding confounding variables and spurious correlations
    - `trade_offs`: Understanding trade-offs between competing objectives
    - `causal_mechanism`: Understanding causal mechanisms and pathways
    - `uncertainty`: Understanding uncertainty and risk
    - `optimization`: Understanding optimization under constraints

    **Use when:**
    - User needs to learn a specific concept
    - Want to provide pedagogically optimal examples
    - Need to adapt teaching to user's current understanding

    **Returns:**
    - Ranked teaching examples with pedagogical rationale
    - Learning objectives
    - Expected learning time
    - Overall teaching strategy explanation
    """,
    responses={
        200: {"description": "Teaching examples generated successfully"},
        400: {"description": "Invalid input (e.g., unknown concept)"},
        500: {"description": "Internal computation error"},
    },
)
async def generate_teaching_examples(
    request: BayesianTeachingRequest,
) -> BayesianTeachingResponse:
    """
    Generate teaching examples using Bayesian teaching.

    Args:
        request: Teaching request with concept and beliefs

    Returns:
        BayesianTeachingResponse: Teaching examples and strategy
    """
    try:
        logger.info(
            "teaching_request",
            extra={
                "user_id": _hash_user_id(request.user_id),
                "concept": request.target_concept,
                "max_examples": request.max_examples,
                "domain": request.context.domain,
            },
        )

        # Generate teaching examples
        examples, explanation, objectives, time = bayesian_teacher.generate_teaching_examples(
            target_concept=request.target_concept,
            current_beliefs=request.current_beliefs,
            context=request.context,
            max_examples=request.max_examples,
        )

        logger.info(
            "teaching_completed",
            extra={
                "user_id": _hash_user_id(request.user_id),
                "concept": request.target_concept,
                "num_examples": len(examples),
                "avg_teaching_value": sum(ex.information_value for ex in examples)
                / len(examples)
                if examples
                else 0,
            },
        )

        return BayesianTeachingResponse(
            examples=examples,
            explanation=explanation,
            learning_objectives=objectives,
            expected_learning_time=time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "teaching_error",
            extra={
                "user_id": _hash_user_id(request.user_id),
                "concept": request.target_concept,
                "error": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate teaching examples. Check logs for details.",
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
