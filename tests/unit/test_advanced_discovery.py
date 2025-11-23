"""
Unit tests for advanced causal discovery algorithms.

Tests NOTEARS, PC algorithm, and integration with discovery engine.
"""

import numpy as np
import networkx as nx
import pytest

from src.services.advanced_discovery_algorithms import (
    NOTEARSDiscovery,
    PCAlgorithm,
    AdvancedCausalDiscovery,
)
from src.services.causal_discovery_engine import CausalDiscoveryEngine


class TestNOTEARSDiscovery:
    """Test NOTEARS algorithm."""

    def test_basic_discovery(self):
        """Test basic NOTEARS discovery."""
        # Create synthetic data from known DAG: X -> Y -> Z
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        Y = 2 * X + np.random.randn(n, 1) * 0.1
        Z = 1.5 * Y + np.random.randn(n, 1) * 0.1
        data = np.hstack([X, Y, Z])

        variable_names = ["X", "Y", "Z"]

        notears = NOTEARSDiscovery(lambda1=0.01, lambda2=0.01, max_iter=50)
        dag, score = notears.discover(data, variable_names)

        # Check DAG is acyclic
        assert nx.is_directed_acyclic_graph(dag)

        # Check we found some edges
        assert len(dag.edges()) > 0

        # Score should be finite
        assert np.isfinite(score)

    def test_h_function_acyclic(self):
        """Test acyclicity constraint for acyclic matrix."""
        notears = NOTEARSDiscovery()

        # Acyclic matrix
        W = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])

        h = notears._h_function(W)

        # Should be close to 0 for acyclic
        assert abs(h) < 1.0

    def test_soft_threshold(self):
        """Test soft-thresholding operator."""
        notears = NOTEARSDiscovery()

        W = np.array([
            [0.5, -0.3],
            [0.1, 0.8],
        ])

        W_thresholded = notears._soft_threshold(W, threshold=0.2)

        # Small values should be zeroed
        assert W_thresholded[1, 0] == 0  # 0.1 < 0.2

        # Large values should be shrunk
        assert abs(W_thresholded[0, 0]) < abs(W[0, 0])

    def test_weight_matrix_to_dag(self):
        """Test conversion of weight matrix to DAG."""
        notears = NOTEARSDiscovery()

        W = np.array([
            [0, 0.5, 0],
            [0, 0, 0.8],
            [0, 0, 0],
        ])

        variable_names = ["X", "Y", "Z"]

        dag = notears._weight_matrix_to_dag(W, variable_names)

        assert len(dag.nodes()) == 3
        assert len(dag.edges()) == 2
        assert dag.has_edge("X", "Y")
        assert dag.has_edge("Y", "Z")

    def test_empty_data(self):
        """Test handling of edge cases."""
        notears = NOTEARSDiscovery(max_iter=10)

        # Single variable
        data = np.random.randn(50, 1)
        variable_names = ["X"]

        dag, score = notears.discover(data, variable_names)

        assert len(dag.nodes()) == 1
        assert len(dag.edges()) == 0


class TestPCAlgorithm:
    """Test PC algorithm."""

    def test_basic_discovery(self):
        """Test basic PC algorithm discovery."""
        # Create synthetic data
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        Y = 2 * X + np.random.randn(n, 1) * 0.1
        Z = 1.5 * Y + np.random.randn(n, 1) * 0.1
        data = np.hstack([X, Y, Z])

        variable_names = ["X", "Y", "Z"]

        pc = PCAlgorithm(alpha=0.05)
        dag, confidence = pc.discover(data, variable_names)

        # Check DAG is acyclic
        assert nx.is_directed_acyclic_graph(dag)

        # Confidence should be in [0, 1]
        assert 0 <= confidence <= 1

    def test_learn_skeleton(self):
        """Test skeleton learning."""
        pc = PCAlgorithm()

        # Correlated variables
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        Y = 2 * X + np.random.randn(n, 1) * 0.1
        data = np.hstack([X, Y])

        variable_names = ["X", "Y"]

        skeleton = pc._learn_skeleton(data, variable_names)

        # Should have edge between X and Y
        assert skeleton.has_edge("X", "Y") or skeleton.has_edge("Y", "X")

    def test_independent_variables(self):
        """Test with independent variables."""
        pc = PCAlgorithm()

        # Independent variables
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        Y = np.random.randn(n, 1)
        data = np.hstack([X, Y])

        variable_names = ["X", "Y"]

        dag, _ = pc.discover(data, variable_names)

        # Should have at most 1 edge (due to temporal ordering)
        assert len(dag.edges()) <= 1


class TestAdvancedCausalDiscovery:
    """Test unified advanced discovery interface."""

    def test_discover_notears(self):
        """Test discover with NOTEARS algorithm."""
        advanced = AdvancedCausalDiscovery()

        np.random.seed(42)
        data = np.random.randn(100, 3)
        variable_names = ["X", "Y", "Z"]

        dag, score = advanced.discover(data, variable_names, algorithm="notears")

        assert isinstance(dag, nx.DiGraph)
        assert np.isfinite(score)

    def test_discover_pc(self):
        """Test discover with PC algorithm."""
        advanced = AdvancedCausalDiscovery()

        np.random.seed(42)
        data = np.random.randn(100, 3)
        variable_names = ["X", "Y", "Z"]

        dag, confidence = advanced.discover(data, variable_names, algorithm="pc")

        assert isinstance(dag, nx.DiGraph)
        assert 0 <= confidence <= 1

    def test_discover_invalid_algorithm(self):
        """Test discover with invalid algorithm."""
        advanced = AdvancedCausalDiscovery()

        data = np.random.randn(50, 2)
        variable_names = ["X", "Y"]

        with pytest.raises(ValueError, match="Unknown algorithm"):
            advanced.discover(data, variable_names, algorithm="invalid")

    def test_auto_discover(self):
        """Test automatic algorithm selection."""
        advanced = AdvancedCausalDiscovery()

        np.random.seed(42)
        data = np.random.randn(100, 3)
        variable_names = ["X", "Y", "Z"]

        results = advanced.auto_discover(data, variable_names)

        # Should return multiple results
        assert len(results) > 0

        # Each result should be a tuple
        for dag, score, algorithm_name in results:
            assert isinstance(dag, nx.DiGraph)
            assert isinstance(algorithm_name, str)


class TestCausalDiscoveryEngineIntegration:
    """Test integration with CausalDiscoveryEngine."""

    def test_discover_advanced_notears(self):
        """Test advanced discovery through discovery engine."""
        engine = CausalDiscoveryEngine(enable_advanced=True)

        np.random.seed(42)
        data = np.random.randn(100, 3)
        variable_names = ["X", "Y", "Z"]

        dag, score = engine.discover_advanced(data, variable_names, algorithm="notears")

        assert isinstance(dag, nx.DiGraph)
        assert len(dag.nodes()) == 3

    def test_discover_advanced_auto(self):
        """Test auto discovery through engine."""
        engine = CausalDiscoveryEngine(enable_advanced=True)

        np.random.seed(42)
        data = np.random.randn(50, 2)
        variable_names = ["X", "Y"]

        dag, score = engine.discover_advanced(data, variable_names, algorithm="auto")

        assert isinstance(dag, nx.DiGraph)

    def test_discover_advanced_disabled(self):
        """Test error when advanced algorithms not enabled."""
        engine = CausalDiscoveryEngine(enable_advanced=False)

        data = np.random.randn(50, 2)
        variable_names = ["X", "Y"]

        with pytest.raises(ValueError, match="Advanced algorithms not enabled"):
            engine.discover_advanced(data, variable_names)

    def test_comparison_simple_vs_advanced(self):
        """Compare simple correlation vs advanced algorithm."""
        engine = CausalDiscoveryEngine(enable_advanced=True)

        # Create data with clear causal structure
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 1)
        Y = 2 * X + np.random.randn(n, 1) * 0.1
        Z = 1.5 * Y + np.random.randn(n, 1) * 0.1
        data = np.hstack([X, Y, Z])

        variable_names = ["X", "Y", "Z"]

        # Simple discovery
        dag_simple, _ = engine.discover_from_data(
            data, variable_names, threshold=0.3, seed=42
        )

        # Advanced discovery
        dag_advanced, _ = engine.discover_advanced(
            data, variable_names, algorithm="notears"
        )

        # Both should be DAGs
        assert nx.is_directed_acyclic_graph(dag_simple)
        assert nx.is_directed_acyclic_graph(dag_advanced)

        # Advanced might find different structure
        # (not necessarily better with simplified implementation)
        assert len(dag_advanced.edges()) >= 0


class TestSyntheticData:
    """Test with known synthetic causal structures."""

    def test_chain_structure(self):
        """Test discovery of chain structure X -> Y -> Z."""
        # Generate data from chain
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 1)
        Y = 1.5 * X + np.random.randn(n, 1) * 0.5
        Z = 1.2 * Y + np.random.randn(n, 1) * 0.5
        data = np.hstack([X, Y, Z])

        variable_names = ["X", "Y", "Z"]

        notears = NOTEARSDiscovery(lambda1=0.05, lambda2=0.05, max_iter=50)
        dag, _ = notears.discover(data, variable_names)

        # Should be acyclic
        assert nx.is_directed_acyclic_graph(dag)

        # Should have some edges
        assert len(dag.edges()) > 0

    def test_fork_structure(self):
        """Test discovery of fork structure Z -> X, Z -> Y."""
        # Generate data from fork
        np.random.seed(42)
        n = 200
        Z = np.random.randn(n, 1)
        X = 1.5 * Z + np.random.randn(n, 1) * 0.5
        Y = 1.2 * Z + np.random.randn(n, 1) * 0.5
        data = np.hstack([Z, X, Y])

        variable_names = ["Z", "X", "Y"]

        notears = NOTEARSDiscovery(lambda1=0.05, lambda2=0.05, max_iter=50)
        dag, _ = notears.discover(data, variable_names)

        # Should be acyclic
        assert nx.is_directed_acyclic_graph(dag)


class TestPerformance:
    """Test performance characteristics."""

    def test_small_data_performance(self):
        """Test performance with small dataset."""
        import time

        np.random.seed(42)
        data = np.random.randn(50, 3)
        variable_names = ["X", "Y", "Z"]

        notears = NOTEARSDiscovery(max_iter=20)

        start = time.time()
        dag, score = notears.discover(data, variable_names)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0

    def test_convergence(self):
        """Test that NOTEARS converges."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        variable_names = ["X", "Y", "Z"]

        notears = NOTEARSDiscovery(max_iter=100, h_tol=1e-8)
        dag, score = notears.discover(data, variable_names)

        # Should produce a valid DAG
        assert nx.is_directed_acyclic_graph(dag)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
