"""
Request validation utilities for ISL endpoints.

Provides size limits, edge case handling, and validation helpers
to prevent DoS attacks and ensure robust input validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import validator, Field


# Global limits to prevent DoS attacks
MAX_OPTIONS = 100
MAX_CRITERIA = 10
MAX_PARAMETERS = 20
MAX_SWEEP_POINTS = 200


class RequestSizeLimits:
    """Centralized request size limits for all ISL endpoints."""

    MAX_OPTIONS = 100
    MAX_CRITERIA = 10
    MAX_PARAMETERS = 20
    MAX_SWEEP_POINTS = 200
    MAX_GRAPH_NODES = 100
    MAX_GRAPH_EDGES = 500


def validate_option_count(options: List[Any], context: str = "options") -> None:
    """
    Validate that option count is within limits.

    Args:
        options: List of options to validate
        context: Description of what's being validated (for error messages)

    Raises:
        ValueError: If count exceeds maximum
    """
    if len(options) > RequestSizeLimits.MAX_OPTIONS:
        raise ValueError(
            f"Too many {context}: {len(options)} exceeds maximum of "
            f"{RequestSizeLimits.MAX_OPTIONS}"
        )


def validate_criteria_count(criteria: List[Any], context: str = "criteria") -> None:
    """
    Validate that criteria count is within limits.

    Args:
        criteria: List of criteria to validate
        context: Description of what's being validated

    Raises:
        ValueError: If count exceeds maximum
    """
    if len(criteria) > RequestSizeLimits.MAX_CRITERIA:
        raise ValueError(
            f"Too many {context}: {len(criteria)} exceeds maximum of "
            f"{RequestSizeLimits.MAX_CRITERIA}"
        )


def validate_option_scores(
    option_scores: Dict[str, Dict[str, float]],
    min_options: int = 2,
    require_complete: bool = True
) -> None:
    """
    Validate option scores dictionary for multi-criteria analysis.

    Args:
        option_scores: Dict mapping option_id → criterion_id → score
        min_options: Minimum number of options required
        require_complete: If True, all options must have scores for all criteria

    Raises:
        ValueError: If validation fails
    """
    if len(option_scores) < min_options:
        raise ValueError(
            f"At least {min_options} options required, got {len(option_scores)}"
        )

    validate_option_count(list(option_scores.keys()), "options")

    if not option_scores:
        return

    # Collect all criteria
    all_criteria = set()
    for scores in option_scores.values():
        all_criteria.update(scores.keys())

    validate_criteria_count(list(all_criteria), "criteria")

    # Check completeness if required
    if require_complete:
        for opt_id, scores in option_scores.items():
            missing_criteria = all_criteria - set(scores.keys())
            if missing_criteria:
                raise ValueError(
                    f"Option '{opt_id}' missing scores for criteria: "
                    f"{sorted(missing_criteria)}"
                )


def validate_criterion_results(
    criterion_results: List[Any],
    max_options_per_criterion: int = RequestSizeLimits.MAX_OPTIONS
) -> None:
    """
    Validate criterion results for aggregation.

    Args:
        criterion_results: List of criterion results
        max_options_per_criterion: Maximum options allowed per criterion

    Raises:
        ValueError: If validation fails
    """
    validate_criteria_count(criterion_results, "criterion results")

    for cr in criterion_results:
        if hasattr(cr, 'option_scores'):
            option_count = len(cr.option_scores)
            if option_count > max_options_per_criterion:
                raise ValueError(
                    f"Criterion '{getattr(cr, 'criterion_id', 'unknown')}' has "
                    f"{option_count} options, maximum {max_options_per_criterion} allowed"
                )


def validate_graph_size(num_nodes: int, num_edges: int) -> None:
    """
    Validate graph size is within limits.

    Args:
        num_nodes: Number of nodes in graph
        num_edges: Number of edges in graph

    Raises:
        ValueError: If size exceeds limits
    """
    if num_nodes > RequestSizeLimits.MAX_GRAPH_NODES:
        raise ValueError(
            f"Graph too large: {num_nodes} nodes exceeds maximum of "
            f"{RequestSizeLimits.MAX_GRAPH_NODES}"
        )

    if num_edges > RequestSizeLimits.MAX_GRAPH_EDGES:
        raise ValueError(
            f"Graph too large: {num_edges} edges exceeds maximum of "
            f"{RequestSizeLimits.MAX_GRAPH_EDGES}"
        )


def check_uniform_scores(
    scores: List[float],
    tolerance: float = 1e-9
) -> bool:
    """
    Check if all scores are uniform (identical).

    Args:
        scores: List of numerical scores
        tolerance: Tolerance for considering scores equal

    Returns:
        True if all scores are within tolerance of each other
    """
    if not scores:
        return True

    first = scores[0]
    return all(abs(score - first) < tolerance for score in scores)


def validate_weights_sum(
    weights: Dict[str, float],
    target: float = 1.0,
    tolerance: float = 0.001
) -> tuple[bool, float]:
    """
    Validate that weights sum to target value.

    Args:
        weights: Dictionary of weights
        target: Target sum (default 1.0)
        tolerance: Acceptable deviation from target

    Returns:
        Tuple of (is_valid, actual_sum)
    """
    weight_sum = sum(weights.values())
    is_valid = abs(weight_sum - target) <= tolerance
    return is_valid, weight_sum


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0.

    Args:
        weights: Dictionary of weights

    Returns:
        Normalized weights dictionary

    Raises:
        ValueError: If all weights are zero
    """
    total = sum(weights.values())

    if total == 0:
        raise ValueError("Cannot normalize: all weights are zero")

    return {k: v / total for k, v in weights.items()}
