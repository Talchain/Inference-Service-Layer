"""
Causal Discovery Engine using structure learning.

Provides data-driven and knowledge-guided DAG discovery.
Simplified implementation focusing on score-based methods.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from src.utils.cache import get_cache
from src.services.advanced_discovery_algorithms import AdvancedCausalDiscovery
from src.utils.error_recovery import (
    CircuitBreaker,
    with_fallback,
    FallbackStrategy,
    health_monitor
)

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL = 3600  # 1 hour (discoveries are expensive, cache longer)
CACHE_MAX_SIZE = 200  # Smaller cache for discovery results

# Circuit breakers for expensive operations
_notears_breaker = CircuitBreaker("notears_discovery", failure_threshold=3, timeout=120)
_pc_breaker = CircuitBreaker("pc_discovery", failure_threshold=3, timeout=120)


class CausalDiscoveryEngine:
    """
    Causal structure discovery engine.

    Supports data-driven discovery using correlation-based methods
    and knowledge-guided discovery with domain constraints.

    Features:
    - Caching of discovery results for performance
    - Deterministic results with seed parameter
    """

    def __init__(self, enable_caching: bool = True, enable_advanced: bool = False):
        """
        Initialize causal discovery engine.

        Args:
            enable_caching: Whether to enable result caching (default: True)
            enable_advanced: Whether to enable advanced algorithms (NOTEARS, PC) (default: False)
        """
        self.enable_caching = enable_caching
        self.enable_advanced = enable_advanced

        # Initialize cache for expensive discovery operations
        if enable_caching:
            self._discovery_cache = get_cache(
                "causal_discovery", max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL
            )
            logger.info("caching_enabled", extra={"service": "CausalDiscoveryEngine"})
        else:
            self._discovery_cache = None

        # Initialize advanced algorithms if enabled
        if enable_advanced:
            self._advanced_discovery = AdvancedCausalDiscovery()
            logger.info("advanced_algorithms_enabled", extra={"service": "CausalDiscoveryEngine"})
        else:
            self._advanced_discovery = None

    def _create_data_cache_key(
        self,
        data: np.ndarray,
        variable_names: List[str],
        prior_knowledge: Optional[Dict],
        threshold: float,
        seed: Optional[int],
    ) -> Dict:
        """
        Create cache key for data-driven discovery.

        Args:
            data: Data matrix
            variable_names: Variable names
            prior_knowledge: Prior knowledge constraints
            threshold: Correlation threshold
            seed: Random seed

        Returns:
            Cache key dict
        """
        # Create hash of data (more stable than storing full array)
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]

        # Create stable key
        return {
            "operation": "discover_from_data",
            "data_hash": data_hash,
            "data_shape": data.shape,
            "variable_names": sorted(variable_names),
            "prior_knowledge": prior_knowledge,
            "threshold": threshold,
            "seed": seed,
        }

    def discover_from_data(
        self,
        data: np.ndarray,
        variable_names: List[str],
        prior_knowledge: Optional[Dict] = None,
        threshold: float = 0.3,
        seed: Optional[int] = None,
    ) -> Tuple[nx.DiGraph, float]:
        """
        Discover DAG structure from data using correlation-based approach.

        Args:
            data: n×d data matrix (n samples, d variables)
            variable_names: List of variable names
            prior_knowledge: Optional prior knowledge (forbidden/required edges)
            threshold: Correlation threshold for edge creation
            seed: Random seed for reproducibility

        Returns:
            Tuple of (discovered DAG, confidence score)
        """
        # Check cache first (before validation to avoid repeated validation overhead)
        cache_key = None
        if self.enable_caching and self._discovery_cache is not None:
            cache_key = self._create_data_cache_key(
                data, variable_names, prior_knowledge, threshold, seed
            )
            cached_result = self._discovery_cache.get(cache_key)
            if cached_result is not None:
                logger.info(
                    "discovery_cache_hit",
                    extra={
                        "n_variables": len(variable_names),
                        "n_samples": data.shape[0],
                    }
                )
                return cached_result

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Validate input data
        if data.size == 0:
            raise ValueError("Input data is empty")

        if np.isnan(data).any():
            raise ValueError("Data contains NaN values. Please clean data before discovery.")

        if np.isinf(data).any():
            raise ValueError("Data contains Inf values. Please clean data before discovery.")

        n, d = data.shape

        # Validate variable names match data dimensions
        if len(variable_names) != d:
            raise ValueError(
                f"Number of variable names ({len(variable_names)}) does not match "
                f"number of data columns ({d})"
            )

        # Check minimum sample size
        if n < 10:
            logger.warning(
                f"Small sample size ({n} < 10). Results may be unreliable."
            )

        # Compute correlation matrix
        corr_matrix = np.corrcoef(data, rowvar=False)

        # Create DAG based on correlations
        dag = nx.DiGraph()
        dag.add_nodes_from(variable_names)

        # Add edges for strong correlations
        # Use a simple heuristic: if corr(X,Y) > threshold, add edge X→Y
        # Direction determined by temporal ordering or variable index
        for i in range(d):
            for j in range(i + 1, d):
                corr = abs(corr_matrix[i, j])
                if corr > threshold:
                    # Add edge from earlier variable to later
                    dag.add_edge(variable_names[i], variable_names[j])

        # Apply prior knowledge constraints
        if prior_knowledge:
            logger.info(
                "applying_prior_knowledge",
                extra={
                    "required_edges": len(prior_knowledge.get("required_edges", [])),
                    "forbidden_edges": len(prior_knowledge.get("forbidden_edges", [])),
                }
            )
            self._apply_prior_knowledge(dag, prior_knowledge)

        # Compute confidence score based on data fit
        confidence = self._compute_discovery_confidence(data, dag, variable_names)

        logger.info(
            "discovery_completed",
            extra={
                "n_nodes": len(dag.nodes()),
                "n_edges": len(dag.edges()),
                "confidence": confidence,
                "is_acyclic": nx.is_directed_acyclic_graph(dag),
            }
        )

        # Cache the result
        if self.enable_caching and self._discovery_cache is not None and cache_key is not None:
            self._discovery_cache.put(cache_key, (dag, confidence))

        return dag, confidence

    def discover_advanced(
        self,
        data: np.ndarray,
        variable_names: List[str],
        algorithm: str = "notears",
        **kwargs,
    ) -> Tuple[nx.DiGraph, float]:
        """
        Discover DAG structure using advanced algorithms (NOTEARS, PC).

        Implements graceful degradation:
        - If advanced algorithm fails, falls back to simple correlation-based discovery
        - Uses circuit breakers to prevent repeated expensive failures
        - Always returns a result (never fails completely)

        Args:
            data: n×d data matrix (n samples, d variables)
            variable_names: List of variable names
            algorithm: Algorithm to use (notears, pc, auto)
            **kwargs: Algorithm-specific parameters

        Returns:
            Tuple of (discovered DAG, score)

        Raises:
            ValueError: If advanced algorithms not enabled
        """
        if not self.enable_advanced or self._advanced_discovery is None:
            # Fall back to simple discovery
            logger.warning(
                "advanced_discovery_not_enabled_fallback",
                extra={"algorithm": algorithm}
            )
            return self.discover_from_data(data, variable_names)

        logger.info(
            "advanced_discovery_start",
            extra={
                "algorithm": algorithm,
                "n_samples": data.shape[0],
                "n_variables": data.shape[1],
            }
        )

        try:
            if algorithm == "auto":
                # Try multiple algorithms and return best
                try:
                    results = self._advanced_discovery.auto_discover(data, variable_names)
                    if results:
                        best_dag, best_score, best_algorithm = results[0]
                        logger.info(
                            "advanced_discovery_complete",
                            extra={
                                "best_algorithm": best_algorithm,
                                "n_edges": len(best_dag.edges()),
                                "score": best_score,
                            }
                        )
                        health_monitor.record_success("advanced_discovery")
                        return best_dag, best_score
                    else:
                        # All advanced algorithms failed, fall back
                        raise RuntimeError("All advanced algorithms failed")
                except Exception as e:
                    logger.warning(
                        "auto_discovery_failed_fallback",
                        extra={"error": str(e)}
                    )
                    health_monitor.record_fallback("advanced_discovery")
                    return self._fallback_to_simple_discovery(data, variable_names)

            elif algorithm == "notears":
                # Use NOTEARS with circuit breaker
                try:
                    dag, score = _notears_breaker.call(
                        self._advanced_discovery.discover,
                        data, variable_names, algorithm, **kwargs
                    )
                    logger.info(
                        "advanced_discovery_complete",
                        extra={
                            "algorithm": algorithm,
                            "n_edges": len(dag.edges()),
                            "score": score,
                        }
                    )
                    health_monitor.record_success("advanced_discovery")
                    return dag, score
                except Exception as e:
                    logger.warning(
                        "notears_failed_fallback",
                        extra={
                            "error": str(e),
                            "circuit_state": _notears_breaker.state.value
                        }
                    )
                    health_monitor.record_fallback("advanced_discovery")
                    return self._fallback_to_simple_discovery(data, variable_names)

            elif algorithm == "pc":
                # Use PC algorithm with circuit breaker
                try:
                    dag, score = _pc_breaker.call(
                        self._advanced_discovery.discover,
                        data, variable_names, algorithm, **kwargs
                    )
                    logger.info(
                        "advanced_discovery_complete",
                        extra={
                            "algorithm": algorithm,
                            "n_edges": len(dag.edges()),
                            "score": score,
                        }
                    )
                    health_monitor.record_success("advanced_discovery")
                    return dag, score
                except Exception as e:
                    logger.warning(
                        "pc_failed_fallback",
                        extra={
                            "error": str(e),
                            "circuit_state": _pc_breaker.state.value
                        }
                    )
                    health_monitor.record_fallback("advanced_discovery")
                    return self._fallback_to_simple_discovery(data, variable_names)

            else:
                # Unknown algorithm, use safe default
                logger.warning(
                    f"unknown_algorithm_fallback",
                    extra={"algorithm": algorithm}
                )
                health_monitor.record_fallback("advanced_discovery")
                return self._fallback_to_simple_discovery(data, variable_names)

        except Exception as e:
            # Ultimate fallback: Always return simple correlation-based discovery
            logger.error(
                "advanced_discovery_failed_ultimate_fallback",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            health_monitor.record_failure("advanced_discovery")
            return self._fallback_to_simple_discovery(data, variable_names)

    def _fallback_to_simple_discovery(
        self,
        data: np.ndarray,
        variable_names: List[str]
    ) -> Tuple[nx.DiGraph, float]:
        """
        Fallback to simple correlation-based discovery.

        Always succeeds with reasonable defaults.

        Args:
            data: Data matrix
            variable_names: Variable names

        Returns:
            Tuple of (DAG, confidence score)
        """
        logger.info(
            "using_simple_discovery_fallback",
            extra={
                "n_samples": data.shape[0],
                "n_variables": data.shape[1]
            }
        )

        try:
            # Use standard correlation-based discovery
            dag, confidence = self.discover_from_data(
                data, variable_names, threshold=0.3
            )
            logger.info("simple_discovery_fallback_success")
            return dag, confidence

        except Exception as e:
            # Even simple discovery failed - return minimal DAG
            logger.error(
                "simple_discovery_failed_minimal_fallback",
                extra={"error": str(e)},
                exc_info=True
            )
            return self._minimal_dag_fallback(variable_names)

    def _minimal_dag_fallback(
        self,
        variable_names: List[str]
    ) -> Tuple[nx.DiGraph, float]:
        """
        Ultimate fallback: Return minimal valid DAG when all discovery fails.

        This is the last resort that always succeeds. Returns a DAG with
        only nodes (no edges) and very low confidence score.

        Args:
            variable_names: Variable names

        Returns:
            Tuple of (minimal DAG, confidence score)
        """
        logger.warning(
            "using_minimal_dag_fallback",
            extra={
                "n_variables": len(variable_names),
                "reason": "All discovery methods failed"
            }
        )

        # Create minimal DAG with nodes only (safest option)
        dag = nx.DiGraph()
        dag.add_nodes_from(variable_names)

        # Very low confidence to indicate this is a fallback
        confidence = 0.1

        logger.info(
            "minimal_dag_created",
            extra={
                "n_nodes": len(dag.nodes()),
                "n_edges": 0,
                "confidence": confidence
            }
        )

        return dag, confidence

    def discover_from_knowledge(
        self,
        domain_description: str,
        variable_names: List[str],
        prior_knowledge: Optional[Dict] = None,
    ) -> List[Tuple[nx.DiGraph, float]]:
        """
        Discover DAG structure from domain knowledge.

        Simplified implementation returns plausible structures.

        Args:
            domain_description: Plain English description
            variable_names: List of variable names
            prior_knowledge: Optional constraints

        Returns:
            List of (DAG, confidence) tuples
        """
        # Simplified: Create a few plausible DAGs based on variable names
        # In production, would use LLM to generate structures

        dags = []

        # Strategy 1: Chain structure
        dag1 = nx.DiGraph()
        dag1.add_nodes_from(variable_names)
        for i in range(len(variable_names) - 1):
            dag1.add_edge(variable_names[i], variable_names[i + 1])
        dags.append((dag1, 0.6))

        # Strategy 2: Star structure (first variable affects all others)
        if len(variable_names) > 1:
            dag2 = nx.DiGraph()
            dag2.add_nodes_from(variable_names)
            for i in range(1, len(variable_names)):
                dag2.add_edge(variable_names[0], variable_names[i])
            dags.append((dag2, 0.5))

        return dags

    def _apply_prior_knowledge(self, dag: nx.DiGraph, prior_knowledge: Dict):
        """
        Apply prior knowledge constraints to DAG.

        Args:
            dag: NetworkX DiGraph (modified in place)
            prior_knowledge: Dict with 'forbidden_edges' and 'required_edges'
        """
        # Remove forbidden edges
        if "forbidden_edges" in prior_knowledge:
            for edge in prior_knowledge["forbidden_edges"]:
                if len(edge) == 2:
                    dag.remove_edge(*edge) if dag.has_edge(*edge) else None

        # Add required edges
        if "required_edges" in prior_knowledge:
            for edge in prior_knowledge["required_edges"]:
                if len(edge) == 2:
                    dag.add_edge(*edge)

    def _compute_discovery_confidence(
        self, data: np.ndarray, dag: nx.DiGraph, variable_names: List[str]
    ) -> float:
        """
        Compute confidence in discovered DAG.

        Args:
            data: Original data
            dag: Discovered DAG
            variable_names: Variable names

        Returns:
            Confidence score (0-1)
        """
        # Simple heuristic: based on number of edges found
        # In production, would use BIC, AIC, or cross-validation

        n_edges = len(dag.edges())
        n_possible = len(variable_names) * (len(variable_names) - 1) / 2

        # Moderate number of edges is good (not too sparse, not too dense)
        edge_ratio = n_edges / n_possible if n_possible > 0 else 0

        if 0.1 <= edge_ratio <= 0.4:
            return 0.8
        elif edge_ratio < 0.1:
            return 0.6  # Too sparse
        else:
            return 0.5  # Too dense

    def validate_discovered_dag(self, dag: nx.DiGraph) -> Dict[str, Any]:
        """
        Validate discovered DAG for acyclicity and other properties.

        Args:
            dag: Discovered DAG

        Returns:
            Dict with validation results
        """
        is_acyclic = nx.is_directed_acyclic_graph(dag)
        n_nodes = len(dag.nodes())
        n_edges = len(dag.edges())

        return {
            "is_acyclic": is_acyclic,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "valid": is_acyclic,
        }
