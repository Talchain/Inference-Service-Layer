#!/usr/bin/env python3
"""
Debug utilities for causal inference components.
"""

import json
import logging
from typing import Dict, List

from src.models.requests import CausalValidationRequest
from src.services.causal_validator import CausalValidator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_causal_validation() -> None:
    """Test causal validation with example DAG."""
    logger.info("=" * 80)
    logger.info("Testing Causal Validation")
    logger.info("=" * 80)

    # Create simple confounded DAG: Z→X→Y, Z→Y
    dag_data = {
        "nodes": ["Z", "X", "Y"],
        "edges": [
            {"source": "Z", "target": "X"},
            {"source": "X", "target": "Y"},
            {"source": "Z", "target": "Y"},  # Confounder
        ],
    }

    request = CausalValidationRequest(
        dag=dag_data,
        treatment="X",
        outcome="Y",
    )

    logger.info(f"DAG nodes: {dag_data['nodes']}")
    logger.info(f"DAG edges: {dag_data['edges']}")
    logger.info(f"Treatment: {request.treatment}")
    logger.info(f"Outcome: {request.outcome}")
    logger.info("")

    # Validate
    validator = CausalValidator()
    result = validator.validate(request)

    logger.info(f"Validation status: {result.status}")
    logger.info(f"Confidence: {result.confidence}")
    logger.info(f"Identifiable: {result.identifiable}")
    logger.info("")

    if result.adjustment_sets:
        logger.info("Adjustment sets found:")
        for i, adj_set in enumerate(result.adjustment_sets, 1):
            logger.info(f"  {i}. {{{', '.join(sorted(adj_set))}}}")
    else:
        logger.info("No valid adjustment sets found")

    if result.issues:
        logger.info("\nIssues detected:")
        for issue in result.issues:
            logger.info(f"  - {issue}")

    logger.info("\nFull response:")
    logger.info(json.dumps(result.dict(), indent=2))
    logger.info("=" * 80)


def debug_causal_validator(
    nodes: List[str],
    edges: List[Dict[str, str]],
    treatment: str,
    outcome: str,
) -> None:
    """
    Debug causal validator with custom DAG.

    Args:
        nodes: List of node names
        edges: List of edge dicts with 'source' and 'target' keys
        treatment: Treatment variable name
        outcome: Outcome variable name
    """
    logger.info("=" * 80)
    logger.info("Debugging Causal Validator")
    logger.info("=" * 80)

    dag_data = {"nodes": nodes, "edges": edges}

    request = CausalValidationRequest(
        dag=dag_data,
        treatment=treatment,
        outcome=outcome,
    )

    logger.info("Input DAG:")
    logger.info(f"  Nodes: {nodes}")
    logger.info(f"  Edges: {edges}")
    logger.info(f"  Treatment: {treatment}")
    logger.info(f"  Outcome: {outcome}")
    logger.info("")

    validator = CausalValidator()

    try:
        result = validator.validate(request)

        logger.info("Validation Result:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Confidence: {result.confidence}")
        logger.info(f"  Identifiable: {result.identifiable}")

        if result.adjustment_sets:
            logger.info(f"  Adjustment sets: {result.adjustment_sets}")
        if result.issues:
            logger.info(f"  Issues: {result.issues}")

        logger.info("\nFull JSON response:")
        logger.info(json.dumps(result.dict(), indent=2))

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)

    logger.info("=" * 80)


if __name__ == "__main__":
    test_causal_validation()
