"""
Unit tests for CausalDiscoveryEngine service (Feature 3).

Tests structure learning from data and knowledge.
"""

import networkx as nx
import numpy as np
import pytest

from src.services.causal_discovery_engine import CausalDiscoveryEngine


class TestCausalDiscoveryEngineInitialization:
    """Test service initialization."""

    def test_initialization(self):
        """Test engine initializes correctly."""
        engine = CausalDiscoveryEngine()
        assert engine is not None


class TestDiscoveryFromData:
    """Test causal discovery from observational data."""

    def test_simple_correlation_discovery(self):
        """Test discovery from simple correlated data."""
        engine = CausalDiscoveryEngine()

        # Generate correlated data: X -> Y
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1
        data = np.column_stack([x, y])

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            threshold=0.3,
            seed=42,
        )

        assert isinstance(dag, nx.DiGraph)
        assert "X" in dag.nodes()
        assert "Y" in dag.nodes()
        assert confidence > 0

    def test_three_variable_discovery(self):
        """Test discovery with three variables."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 1.5 * x + np.random.randn(n) * 0.1
        z = 2 * y + np.random.randn(n) * 0.1
        data = np.column_stack([x, y, z])

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y", "Z"],
            threshold=0.5,
            seed=42,
        )

        # Should detect relationships
        assert len(dag.edges()) > 0

    def test_threshold_sensitivity(self):
        """Test that threshold affects edge detection."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 0.5 * x + np.random.randn(n)  # Moderate correlation
        data = np.column_stack([x, y])

        # High threshold - fewer edges
        dag_high, _ = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            threshold=0.8,
            seed=42,
        )

        # Low threshold - more edges
        dag_low, _ = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            threshold=0.1,
            seed=42,
        )

        # Low threshold should find more edges (or equal)
        assert len(dag_low.edges()) >= len(dag_high.edges())

    def test_independent_variables(self):
        """Test discovery with independent variables."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)  # Independent
        data = np.column_stack([x, y])

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            threshold=0.3,
            seed=42,
        )

        # Should detect no strong edges (or very few)
        assert len(dag.edges()) <= 1  # May have weak spurious correlation

    def test_prior_knowledge_required_edges(self):
        """Test that required edges are enforced."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)  # Independent
        data = np.column_stack([x, y])

        prior_knowledge = {
            "required_edges": [("X", "Y")]
        }

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            prior_knowledge=prior_knowledge,
            threshold=0.3,
            seed=42,
        )

        # Required edge should be present
        assert dag.has_edge("X", "Y")

    def test_prior_knowledge_forbidden_edges(self):
        """Test that forbidden edges are removed."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 * x + np.random.randn(n) * 0.1  # Strong correlation
        data = np.column_stack([x, y])

        prior_knowledge = {
            "forbidden_edges": [("X", "Y"), ("Y", "X")]
        }

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            prior_knowledge=prior_knowledge,
            threshold=0.3,
            seed=42,
        )

        # Forbidden edges should not be present
        assert not dag.has_edge("X", "Y")
        assert not dag.has_edge("Y", "X")

    def test_determinism_with_seed(self):
        """Test that same seed produces same results."""
        engine = CausalDiscoveryEngine()

        np.random.seed(100)
        data = np.random.randn(50, 3)

        dag1, conf1 = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y", "Z"],
            threshold=0.3,
            seed=42,
        )

        dag2, conf2 = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y", "Z"],
            threshold=0.3,
            seed=42,
        )

        # Should be identical
        assert set(dag1.edges()) == set(dag2.edges())
        assert conf1 == conf2

    def test_confidence_scoring(self):
        """Test that confidence scores are reasonable."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 3 * x + np.random.randn(n) * 0.01  # Very strong correlation
        data = np.column_stack([x, y])

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            threshold=0.3,
            seed=42,
        )

        # Strong correlation should yield high confidence
        assert confidence > 0.5
        assert confidence <= 1.0

    def test_larger_dataset(self):
        """Test with larger number of variables."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100
        k = 10
        data = np.random.randn(n, k)

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=[f"V{i}" for i in range(k)],
            threshold=0.4,
            seed=42,
        )

        # Should handle without error
        assert len(dag.nodes()) == k


class TestDiscoveryFromKnowledge:
    """Test causal discovery from domain knowledge."""

    def test_simple_knowledge_discovery(self):
        """Test discovery from simple domain description."""
        engine = CausalDiscoveryEngine()

        dags = engine.discover_from_knowledge(
            domain_description="Price affects revenue",
            variable_names=["Price", "Revenue"],
        )

        assert len(dags) > 0
        assert all(isinstance(dag, nx.DiGraph) for dag, _ in dags)
        assert all(isinstance(conf, float) for _, conf in dags)

    def test_three_variable_knowledge(self):
        """Test knowledge discovery with three variables."""
        engine = CausalDiscoveryEngine()

        dags = engine.discover_from_knowledge(
            domain_description="Quality affects price and revenue",
            variable_names=["Quality", "Price", "Revenue"],
        )

        assert len(dags) > 0

        # All DAGs should have the same nodes
        for dag, _ in dags:
            assert len(dag.nodes()) == 3

    def test_top_k_parameter(self):
        """Test that top_k limits number of results."""
        engine = CausalDiscoveryEngine()

        dags_3 = engine.discover_from_knowledge(
            domain_description="Variables interact",
            variable_names=["A", "B", "C"],
            top_k=3,
        )

        dags_1 = engine.discover_from_knowledge(
            domain_description="Variables interact",
            variable_names=["A", "B", "C"],
            top_k=1,
        )

        assert len(dags_3) <= 3
        assert len(dags_1) <= 1

    def test_prior_knowledge_in_discovery(self):
        """Test that prior knowledge is applied."""
        engine = CausalDiscoveryEngine()

        prior_knowledge = {
            "required_edges": [("Price", "Revenue")]
        }

        dags = engine.discover_from_knowledge(
            domain_description="Pricing model",
            variable_names=["Price", "Quality", "Revenue"],
            prior_knowledge=prior_knowledge,
        )

        # At least some DAGs should have the required edge
        has_required = any(dag.has_edge("Price", "Revenue") for dag, _ in dags)
        assert has_required

    def test_confidence_ranking(self):
        """Test that DAGs are ranked by confidence."""
        engine = CausalDiscoveryEngine()

        dags = engine.discover_from_knowledge(
            domain_description="Test ranking",
            variable_names=["A", "B", "C"],
            top_k=5,
        )

        if len(dags) > 1:
            # Confidence should be non-increasing
            confidences = [conf for _, conf in dags]
            assert all(confidences[i] >= confidences[i+1] for i in range(len(confidences)-1))

    def test_empty_description(self):
        """Test with minimal description."""
        engine = CausalDiscoveryEngine()

        dags = engine.discover_from_knowledge(
            domain_description="variables",
            variable_names=["X", "Y"],
        )

        # Should still generate some structures
        assert len(dags) > 0

    def test_long_description(self):
        """Test with detailed domain description."""
        engine = CausalDiscoveryEngine()

        description = """
        In an e-commerce system, product price affects customer decisions
        which affects revenue. Quality also affects both price and revenue directly.
        Marketing spend influences customer awareness and revenue.
        """

        dags = engine.discover_from_knowledge(
            domain_description=description,
            variable_names=["Price", "Quality", "Marketing", "Revenue"],
        )

        assert len(dags) > 0


class TestDAGValidation:
    """Test DAG validation functionality."""

    def test_acyclicity_check(self):
        """Test that discovered DAGs are acyclic."""
        engine = CausalDiscoveryEngine()

        # Create cyclic graph
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])  # Cycle!

        is_valid = engine.validate_discovered_dag(dag)

        # Should detect cycle
        assert not is_valid

    def test_valid_dag_check(self):
        """Test validation of valid DAG."""
        engine = CausalDiscoveryEngine()

        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])  # No cycle

        is_valid = engine.validate_discovered_dag(dag)

        assert is_valid

    def test_empty_dag_validation(self):
        """Test validation of empty DAG."""
        engine = CausalDiscoveryEngine()

        dag = nx.DiGraph()
        dag.add_nodes_from(["A", "B"])

        is_valid = engine.validate_discovered_dag(dag)

        # Empty DAG is valid (no cycles)
        assert is_valid

    def test_self_loop_detection(self):
        """Test detection of self-loops."""
        engine = CausalDiscoveryEngine()

        dag = nx.DiGraph()
        dag.add_edge("A", "A")  # Self-loop

        is_valid = engine.validate_discovered_dag(dag)

        # Should be invalid (self-loop is a cycle)
        assert not is_valid


class TestPriorKnowledgeApplication:
    """Test application of prior knowledge constraints."""

    def test_required_edges_application(self):
        """Test that required edges are added."""
        engine = CausalDiscoveryEngine()

        dag = nx.DiGraph()
        dag.add_nodes_from(["A", "B", "C"])

        prior = {
            "required_edges": [("A", "B"), ("B", "C")]
        }

        engine._apply_prior_knowledge(dag, prior)

        assert dag.has_edge("A", "B")
        assert dag.has_edge("B", "C")

    def test_forbidden_edges_removal(self):
        """Test that forbidden edges are removed."""
        engine = CausalDiscoveryEngine()

        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

        prior = {
            "forbidden_edges": [("C", "A")]
        }

        engine._apply_prior_knowledge(dag, prior)

        assert not dag.has_edge("C", "A")
        assert dag.has_edge("A", "B")  # Others preserved

    def test_both_constraints(self):
        """Test applying both required and forbidden edges."""
        engine = CausalDiscoveryEngine()

        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("C", "D")])

        prior = {
            "required_edges": [("X", "Y")],
            "forbidden_edges": [("A", "B")],
        }

        engine._apply_prior_knowledge(dag, prior)

        assert dag.has_edge("X", "Y")
        assert not dag.has_edge("A", "B")

    def test_no_prior_knowledge(self):
        """Test with no prior knowledge."""
        engine = CausalDiscoveryEngine()

        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B")])

        original_edges = set(dag.edges())

        engine._apply_prior_knowledge(dag, None)

        # Should remain unchanged
        assert set(dag.edges()) == original_edges


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_variable(self):
        """Test discovery with single variable."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        data = np.random.randn(100, 1)

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X"],
            seed=42,
        )

        # Should handle gracefully
        assert len(dag.nodes()) == 1
        assert len(dag.edges()) == 0

    def test_minimal_data(self):
        """Test with minimal data points."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        data = np.random.randn(5, 2)  # Only 5 samples

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            seed=42,
        )

        # Should handle without crashing
        assert dag is not None

    def test_perfect_correlation(self):
        """Test with perfectly correlated variables."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = x  # Perfect correlation
        data = np.column_stack([x, y])

        dag, confidence = engine.discover_from_data(
            data=data,
            variable_names=["X", "Y"],
            threshold=0.3,
            seed=42,
        )

        # Should detect strong relationship
        assert len(dag.edges()) > 0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        engine = CausalDiscoveryEngine()

        data = np.array([[1, 2], [3, np.nan], [5, 6]])

        # Should handle gracefully (implementation dependent)
        try:
            dag, confidence = engine.discover_from_data(
                data=data,
                variable_names=["X", "Y"],
                seed=42,
            )
            assert dag is not None
        except (ValueError, RuntimeError):
            # Also acceptable to raise error for NaN
            pass

    def test_variable_name_mismatch(self):
        """Test handling of variable name count mismatch."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        data = np.random.randn(100, 3)

        # Only 2 names for 3 columns
        # Implementation should handle this gracefully
        try:
            dag, confidence = engine.discover_from_data(
                data=data,
                variable_names=["X", "Y"],  # Missing one name
                seed=42,
            )
        except (ValueError, IndexError):
            # Expected error
            pass


class TestConfidenceComputation:
    """Test confidence score computation."""

    def test_confidence_range(self):
        """Test that confidence is in [0, 1]."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        for _ in range(10):
            data = np.random.randn(100, 2)
            dag, confidence = engine.discover_from_data(
                data=data,
                variable_names=["X", "Y"],
                seed=42,
            )

            assert 0 <= confidence <= 1

    def test_strong_correlation_high_confidence(self):
        """Test strong correlations yield higher confidence."""
        engine = CausalDiscoveryEngine()

        np.random.seed(42)
        n = 100

        # Strong correlation
        x_strong = np.random.randn(n)
        y_strong = 5 * x_strong + np.random.randn(n) * 0.01
        data_strong = np.column_stack([x_strong, y_strong])

        _, conf_strong = engine.discover_from_data(
            data=data_strong,
            variable_names=["X", "Y"],
            threshold=0.1,
            seed=42,
        )

        # Weak correlation
        x_weak = np.random.randn(n)
        y_weak = 0.1 * x_weak + np.random.randn(n) * 5
        data_weak = np.column_stack([x_weak, y_weak])

        _, conf_weak = engine.discover_from_data(
            data=data_weak,
            variable_names=["X", "Y"],
            threshold=0.1,
            seed=42,
        )

        # Strong correlation should have higher confidence
        assert conf_strong > conf_weak
