"""
Causal Discovery Engine using structure learning.

Provides data-driven and knowledge-guided DAG discovery.
Simplified implementation focusing on score-based methods.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class CausalDiscoveryEngine:
    """
    Causal structure discovery engine.

    Supports data-driven discovery using correlation-based methods
    and knowledge-guided discovery with domain constraints.
    """

    def __init__(self):
        """Initialize causal discovery engine."""
        pass

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
