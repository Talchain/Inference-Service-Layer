#!/usr/bin/env python3
"""
Debug utilities for belief updating components.
"""

import json
import logging
from typing import Dict

from src.models.shared import BeliefState, Distribution
from src.models.requests import BeliefUpdateRequest
from src.services.belief_updater import BeliefUpdater

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_belief_update() -> None:
    """Test belief update with example preference."""
    logger.info("=" * 80)
    logger.info("Testing Belief Update")
    logger.info("=" * 80)

    # Create initial beliefs (uniform prior)
    initial_beliefs = BeliefState(
        value_weights={
            "revenue": Distribution(
                type="normal",
                parameters={"mean": 0.5, "std": 0.2},
            ),
            "churn": Distribution(
                type="normal",
                parameters={"mean": 0.5, "std": 0.2},
            ),
        },
        uncertainty_estimates={
            "revenue_weight": 0.8,
            "churn_weight": 0.8,
        },
    )

    # User prefers Scenario A
    preference = {
        "query_id": "query_001",
        "scenario_a": {
            "outcomes": {"revenue": 1000, "churn": 0.05},
        },
        "scenario_b": {
            "outcomes": {"revenue": 800, "churn": 0.02},
        },
        "chosen": "A",
        "confidence": 0.8,
    }

    request = BeliefUpdateRequest(
        user_id="test_user",
        current_beliefs=initial_beliefs,
        preference=preference,
    )

    logger.info("Initial beliefs:")
    logger.info(f"  Revenue weight: mean={initial_beliefs.value_weights['revenue'].parameters['mean']:.3f}, "
                f"std={initial_beliefs.value_weights['revenue'].parameters['std']:.3f}")
    logger.info(f"  Churn weight: mean={initial_beliefs.value_weights['churn'].parameters['mean']:.3f}, "
                f"std={initial_beliefs.value_weights['churn'].parameters['std']:.3f}")
    logger.info(f"  Uncertainty: revenue={initial_beliefs.uncertainty_estimates['revenue_weight']:.3f}, "
                f"churn={initial_beliefs.uncertainty_estimates['churn_weight']:.3f}")
    logger.info("")

    logger.info("Preference:")
    logger.info(f"  Scenario A: {preference['scenario_a']['outcomes']}")
    logger.info(f"  Scenario B: {preference['scenario_b']['outcomes']}")
    logger.info(f"  Chosen: {preference['chosen']}")
    logger.info(f"  Confidence: {preference['confidence']}")
    logger.info("")

    # Update beliefs
    updater = BeliefUpdater()
    result = updater.update(request)

    logger.info("Updated beliefs:")
    logger.info(f"  Revenue weight: mean={result.updated_beliefs.value_weights['revenue'].parameters['mean']:.3f}, "
                f"std={result.updated_beliefs.value_weights['revenue'].parameters['std']:.3f}")
    logger.info(f"  Churn weight: mean={result.updated_beliefs.value_weights['churn'].parameters['mean']:.3f}, "
                f"std={result.updated_beliefs.value_weights['churn'].parameters['std']:.3f}")
    logger.info(f"  Uncertainty: revenue={result.updated_beliefs.uncertainty_estimates['revenue_weight']:.3f}, "
                f"churn={result.updated_beliefs.uncertainty_estimates['churn_weight']:.3f}")
    logger.info("")

    if result.explanation:
        logger.info(f"Explanation: {result.explanation}")

    logger.info("\nFull response:")
    logger.info(json.dumps(result.dict(), indent=2))
    logger.info("=" * 80)


def debug_belief_updater(
    value_weights: Dict[str, Dict[str, float]],
    uncertainty: Dict[str, float],
    preference: Dict,
) -> None:
    """
    Debug belief updater with custom beliefs and preference.

    Args:
        value_weights: Dict mapping variable names to {'mean': float, 'std': float}
        uncertainty: Dict mapping uncertainty keys to uncertainty values
        preference: Preference dict with query_id, scenario_a, scenario_b, chosen, confidence
    """
    logger.info("=" * 80)
    logger.info("Debugging Belief Updater")
    logger.info("=" * 80)

    # Create belief state
    beliefs = BeliefState(
        value_weights={
            var: Distribution(type="normal", parameters=params)
            for var, params in value_weights.items()
        },
        uncertainty_estimates=uncertainty,
    )

    request = BeliefUpdateRequest(
        user_id="debug_user",
        current_beliefs=beliefs,
        preference=preference,
    )

    logger.info("Input:")
    logger.info(f"  Value weights: {value_weights}")
    logger.info(f"  Uncertainty: {uncertainty}")
    logger.info(f"  Preference: {preference}")
    logger.info("")

    updater = BeliefUpdater()

    try:
        result = updater.update(request)

        logger.info("Update Result:")
        logger.info("  Updated value weights:")
        for var, dist in result.updated_beliefs.value_weights.items():
            logger.info(f"    {var}: mean={dist.parameters['mean']:.3f}, "
                       f"std={dist.parameters['std']:.3f}")

        logger.info("  Updated uncertainty:")
        for key, val in result.updated_beliefs.uncertainty_estimates.items():
            logger.info(f"    {key}: {val:.3f}")

        if result.explanation:
            logger.info(f"\n  Explanation: {result.explanation}")

        logger.info("\nFull JSON response:")
        logger.info(json.dumps(result.dict(), indent=2))

    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)

    logger.info("=" * 80)


if __name__ == "__main__":
    test_belief_update()
