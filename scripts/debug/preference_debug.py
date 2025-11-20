#!/usr/bin/env python3
"""
Debug utilities for preference elicitation components.
"""

import json
import logging
from typing import Dict, List

from src.models.requests import PreferenceElicitationRequest
from src.services.preference_elicitor import PreferenceElicitor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_preference_elicitation() -> None:
    """Test preference elicitation with example context."""
    logger.info("=" * 80)
    logger.info("Testing Preference Elicitation")
    logger.info("=" * 80)

    # Create pricing decision context
    context = {
        "domain": "pricing",
        "variables": ["revenue", "churn"],
        "description": "Optimizing pricing strategy to balance revenue and customer retention",
    }

    request = PreferenceElicitationRequest(
        user_id="test_user",
        context=context,
        num_queries=3,
    )

    logger.info(f"User ID: {request.user_id}")
    logger.info(f"Domain: {context['domain']}")
    logger.info(f"Variables: {context['variables']}")
    logger.info(f"Num queries: {request.num_queries}")
    logger.info("")

    # Elicit preferences
    elicitor = PreferenceElicitor()
    result = elicitor.elicit(request)

    logger.info(f"Generated {len(result.queries)} queries")
    logger.info("")

    for i, query in enumerate(result.queries, 1):
        logger.info(f"Query {i} (ID: {query.query_id}):")
        logger.info(f"  Scenario A: {query.scenario_a.outcomes}")
        logger.info(f"  Scenario B: {query.scenario_b.outcomes}")
        logger.info(f"  Question: {query.question}")
        logger.info("")

    if result.explanation:
        logger.info(f"Explanation: {result.explanation}")

    logger.info("\nFull response:")
    logger.info(json.dumps(result.dict(), indent=2))
    logger.info("=" * 80)


def debug_preference_elicitor(
    user_id: str,
    domain: str,
    variables: List[str],
    num_queries: int = 3,
    description: str = "",
) -> None:
    """
    Debug preference elicitor with custom context.

    Args:
        user_id: User identifier
        domain: Decision domain
        variables: List of decision variables
        num_queries: Number of queries to generate
        description: Context description
    """
    logger.info("=" * 80)
    logger.info("Debugging Preference Elicitor")
    logger.info("=" * 80)

    context = {
        "domain": domain,
        "variables": variables,
        "description": description or f"Decision making in {domain} domain",
    }

    request = PreferenceElicitationRequest(
        user_id=user_id,
        context=context,
        num_queries=num_queries,
    )

    logger.info("Input:")
    logger.info(f"  User ID: {user_id}")
    logger.info(f"  Domain: {domain}")
    logger.info(f"  Variables: {variables}")
    logger.info(f"  Num queries: {num_queries}")
    logger.info("")

    elicitor = PreferenceElicitor()

    try:
        result = elicitor.elicit(request)

        logger.info("Elicitation Result:")
        logger.info(f"  Queries generated: {len(result.queries)}")

        for i, query in enumerate(result.queries, 1):
            logger.info(f"\n  Query {i}:")
            logger.info(f"    ID: {query.query_id}")
            logger.info(f"    Scenario A: {query.scenario_a.outcomes}")
            logger.info(f"    Scenario B: {query.scenario_b.outcomes}")
            logger.info(f"    Question: {query.question}")

        if result.explanation:
            logger.info(f"\n  Explanation: {result.explanation}")

        logger.info("\nFull JSON response:")
        logger.info(json.dumps(result.dict(), indent=2))

    except Exception as e:
        logger.error(f"Elicitation failed: {e}", exc_info=True)

    logger.info("=" * 80)


if __name__ == "__main__":
    test_preference_elicitation()
