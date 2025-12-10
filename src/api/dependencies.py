"""
Dependency injection providers for FastAPI endpoints.

This module provides factory functions for service dependencies using FastAPI's
Depends() mechanism. Benefits include:
- Better testability (easy to mock services)
- Cleaner separation of concerns
- Explicit dependency graph
- Lifecycle management
"""

from functools import lru_cache
from typing import Optional

from src.services.dominance_analyzer import DominanceAnalyzer
from src.services.multi_criteria_aggregator import MultiCriteriaAggregator
from src.services.risk_adjuster import RiskAdjuster
from src.services.threshold_identifier import ThresholdIdentifier


# ============================================================================
# Analysis Services
# ============================================================================

@lru_cache()
def get_dominance_analyzer() -> DominanceAnalyzer:
    """
    Get DominanceAnalyzer service instance.

    Uses LRU cache to ensure singleton behavior - same instance reused
    across all requests. Safe for stateless services.

    Returns:
        DominanceAnalyzer: Service for detecting dominance relationships
    """
    return DominanceAnalyzer()


@lru_cache()
def get_risk_adjuster() -> RiskAdjuster:
    """
    Get RiskAdjuster service instance.

    Returns:
        RiskAdjuster: Service for computing certainty equivalents
    """
    return RiskAdjuster()


@lru_cache()
def get_threshold_identifier() -> ThresholdIdentifier:
    """
    Get ThresholdIdentifier service instance.

    Returns:
        ThresholdIdentifier: Service for parameter sensitivity analysis
    """
    return ThresholdIdentifier()


# ============================================================================
# Aggregation Services
# ============================================================================

@lru_cache()
def get_multi_criteria_aggregator() -> MultiCriteriaAggregator:
    """
    Get MultiCriteriaAggregator service instance.

    Returns:
        MultiCriteriaAggregator: Service for multi-criteria scoring
    """
    return MultiCriteriaAggregator()


# ============================================================================
# Future: Additional services can be added here
# ============================================================================
# Example for services with configuration:
#
# def get_counterfactual_engine(
#     settings: Settings = Depends(get_settings)
# ) -> CounterfactualEngine:
#     return CounterfactualEngine(
#         max_iterations=settings.MAX_MONTE_CARLO_ITERATIONS
#     )
