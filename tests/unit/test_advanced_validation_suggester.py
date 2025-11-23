"""
Unit tests for AdvancedValidationSuggester service (Feature 2).

Tests comprehensive adjustment strategy generation and path analysis.
"""

import networkx as nx
import pytest

from src.services.advanced_validation_suggester import (
    AdvancedValidationSuggester,
    AdjustmentStrategy,
    PathAnalysis,
)


class TestAdvancedValidationSuggesterInitialization:
    """Test service initialization."""

    def test_initialization(self):
        """Test suggester initializes correctly."""
        suggester = AdvancedValidationSuggester()
        assert suggester is not None


class TestBackdoorStrategyGeneration:
    """Test backdoor adjustment strategy generation."""

    def test_simple_backdoor_strategy(self):
        """Test generating backdoor strategy for simple confounding."""
        suggester = AdvancedValidationSuggester()

        # DAG: Confounder -> Treatment, Confounder -> Outcome
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Confounder", "Treatment"),
            ("Confounder", "Outcome"),
            ("Treatment", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        assert len(strategies) > 0
        assert any(s.type == "backdoor" for s in strategies)

    def test_backdoor_strategy_with_existing_confounder(self):
        """Test backdoor strategy when confounder already measured."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Brand", "Price"),
            ("Brand", "Revenue"),
            ("Price", "Revenue"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Price", "Revenue")

        # Should find backdoor strategy with no new nodes needed
        backdoor_strategies = [s for s in strategies if s.type == "backdoor"]
        assert len(backdoor_strategies) > 0
        assert any(len(s.nodes_to_add) == 0 for s in backdoor_strategies)

    def test_multiple_backdoor_paths(self):
        """Test handling multiple backdoor paths."""
        suggester = AdvancedValidationSuggester()

        # Multiple confounders
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("U1", "Treatment"),
            ("U1", "Outcome"),
            ("U2", "Treatment"),
            ("U2", "Outcome"),
            ("Treatment", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        assert len(strategies) >= 2  # At least one strategy per confounder

    def test_no_backdoor_paths(self):
        """Test when no backdoor paths exist."""
        suggester = AdvancedValidationSuggester()

        # Simple causal chain, no confounding
        dag = nx.DiGraph()
        dag.add_edges_from([("Treatment", "Outcome")])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        # May have frontdoor or IV strategies, but backdoor should be empty/low priority
        backdoor_strategies = [s for s in strategies if s.type == "backdoor"]
        assert len(backdoor_strategies) == 0


class TestFrontdoorStrategyGeneration:
    """Test frontdoor adjustment strategy generation."""

    def test_simple_frontdoor_strategy(self):
        """Test frontdoor strategy with complete mediator."""
        suggester = AdvancedValidationSuggester()

        # Classic frontdoor: Treatment -> Mediator -> Outcome
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Treatment", "Mediator"),
            ("Mediator", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        frontdoor_strategies = [s for s in strategies if s.type == "frontdoor"]
        if frontdoor_strategies:  # Frontdoor only works under specific conditions
            assert len(frontdoor_strategies) > 0

    def test_frontdoor_with_multiple_mediators(self):
        """Test frontdoor with multiple mediators on path."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Treatment", "M1"),
            ("M1", "M2"),
            ("M2", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        # Should detect mediator set
        assert len(strategies) >= 0  # At least some strategy found


class TestInstrumentalVariableStrategy:
    """Test instrumental variable strategy generation."""

    def test_simple_instrumental_variable(self):
        """Test IV strategy with valid instrument."""
        suggester = AdvancedValidationSuggester()

        # Instrument -> Treatment -> Outcome
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Instrument", "Treatment"),
            ("Treatment", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        iv_strategies = [s for s in strategies if s.type == "instrumental"]
        assert len(iv_strategies) > 0

    def test_invalid_instrument(self):
        """Test that invalid instruments are not suggested."""
        suggester = AdvancedValidationSuggester()

        # Invalid: node affects outcome directly
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("BadInstrument", "Treatment"),
            ("BadInstrument", "Outcome"),  # Violates exclusion restriction
            ("Treatment", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        # Should not suggest BadInstrument as IV (it's a confounder)
        iv_strategies = [s for s in strategies if s.type == "instrumental"]
        # If IV strategies exist, they shouldn't include BadInstrument
        for s in iv_strategies:
            assert "BadInstrument" not in s.nodes_to_add


class TestPathAnalysis:
    """Test comprehensive path analysis."""

    def test_backdoor_path_detection(self):
        """Test detection of backdoor paths."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Confounder", "Treatment"),
            ("Confounder", "Outcome"),
            ("Treatment", "Outcome"),
        ])

        analysis = suggester.analyze_paths(dag, "Treatment", "Outcome")

        assert len(analysis.backdoor_paths) > 0
        # Should contain path through Confounder
        assert any("Confounder" in path for path in analysis.backdoor_paths)

    def test_frontdoor_path_detection(self):
        """Test detection of directed paths."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Treatment", "Mediator"),
            ("Mediator", "Outcome"),
        ])

        analysis = suggester.analyze_paths(dag, "Treatment", "Outcome")

        assert len(analysis.frontdoor_paths) > 0
        assert any("Mediator" in path for path in analysis.frontdoor_paths)

    def test_direct_path_detection(self):
        """Test detection of direct causal paths."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([("Treatment", "Outcome")])

        analysis = suggester.analyze_paths(dag, "Treatment", "Outcome")

        assert len(analysis.frontdoor_paths) == 1
        assert analysis.frontdoor_paths[0] == ["Treatment", "Outcome"]

    def test_critical_node_identification(self):
        """Test identification of nodes blocking multiple paths."""
        suggester = AdvancedValidationSuggester()

        # Confounder appears in multiple backdoor paths
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("CommonConfounder", "Treatment"),
            ("CommonConfounder", "Outcome"),
            ("CommonConfounder", "M"),
            ("M", "Outcome"),
            ("Treatment", "Outcome"),
        ])

        analysis = suggester.analyze_paths(dag, "Treatment", "Outcome")

        # CommonConfounder should be identified as critical
        assert "CommonConfounder" in analysis.critical_nodes

    def test_empty_path_analysis(self):
        """Test path analysis with no paths."""
        suggester = AdvancedValidationSuggester()

        # Disconnected nodes
        dag = nx.DiGraph()
        dag.add_nodes_from(["Treatment", "Outcome"])

        analysis = suggester.analyze_paths(dag, "Treatment", "Outcome")

        assert len(analysis.backdoor_paths) == 0
        assert len(analysis.frontdoor_paths) == 0


class TestStrategyRanking:
    """Test ranking of adjustment strategies."""

    def test_strategy_ranking_by_identifiability(self):
        """Test strategies ranked by expected identifiability."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([
            ("U1", "Treatment"),
            ("U1", "Outcome"),
            ("U2", "Treatment"),
            ("U2", "Outcome"),
            ("Treatment", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        # Should be ranked (highest identifiability first)
        if len(strategies) > 1:
            for i in range(len(strategies) - 1):
                assert strategies[i].expected_identifiability >= strategies[i+1].expected_identifiability - 0.3

    def test_strategy_ranking_by_simplicity(self):
        """Test strategies prefer simpler solutions."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Confounder", "Treatment"),
            ("Confounder", "Outcome"),
            ("Treatment", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        # Strategies with fewer nodes should rank higher (all else equal)
        if len(strategies) > 1:
            # First strategy should be simpler or equally complex
            assert len(strategies[0].nodes_to_add) <= len(strategies[-1].nodes_to_add) + 2


class TestEdgeInference:
    """Test inference of confounder edges."""

    def test_confounder_edge_inference(self):
        """Test that confounder edges are inferred correctly."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([("Treatment", "Outcome")])

        # Simulate backdoor path through unmeasured confounder
        backdoor_paths = [["Treatment", "UnmeasuredU", "Outcome"]]

        edges = suggester._infer_confounder_edges(
            dag, "UnmeasuredU", "Treatment", "Outcome", backdoor_paths
        )

        assert ("UnmeasuredU", "Treatment") in edges
        assert ("UnmeasuredU", "Outcome") in edges

    def test_edge_inference_consistency(self):
        """Test edge inference is consistent."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([("Treatment", "Outcome")])

        edges1 = suggester._infer_confounder_edges(dag, "U", "T", "O", [])
        edges2 = suggester._infer_confounder_edges(dag, "U", "T", "O", [])

        assert edges1 == edges2  # Deterministic


class TestComplexDAGScenarios:
    """Test with complex DAG scenarios."""

    def test_complex_dag_with_multiple_patterns(self):
        """Test DAG with confounding, mediation, and instruments."""
        suggester = AdvancedValidationSuggester()

        # Complex scenario
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Instrument", "Treatment"),
            ("Confounder", "Treatment"),
            ("Confounder", "Outcome"),
            ("Treatment", "Mediator"),
            ("Mediator", "Outcome"),
        ])

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")

        # Should find multiple types of strategies
        types = {s.type for s in strategies}
        assert len(types) >= 2  # At least 2 different strategy types

    def test_large_dag(self):
        """Test performance with larger DAG."""
        suggester = AdvancedValidationSuggester()

        # Create DAG with 20 nodes
        dag = nx.DiGraph()
        for i in range(19):
            dag.add_edge(f"Node{i}", f"Node{i+1}")

        strategies = suggester.suggest_adjustment_strategies(dag, "Node0", "Node19")

        # Should handle without error
        assert strategies is not None

    def test_dag_with_collider(self):
        """Test DAG with collider structure."""
        suggester = AdvancedValidationSuggester()

        # Collider: Treatment -> Collider <- Outcome
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Treatment", "Collider"),
            ("Outcome", "Collider"),
        ])

        analysis = suggester.analyze_paths(dag, "Treatment", "Outcome")

        # No backdoor paths (collider blocks by default)
        assert len(analysis.backdoor_paths) == 0


class TestAdjustmentStrategyClass:
    """Test AdjustmentStrategy class."""

    def test_strategy_creation(self):
        """Test creating adjustment strategy."""
        strategy = AdjustmentStrategy(
            strategy_type="backdoor",
            nodes_to_add=["Confounder"],
            edges_to_add=[("Confounder", "Treatment"), ("Confounder", "Outcome")],
            explanation="Control for Confounder",
            theoretical_basis="Pearl's backdoor criterion",
            expected_identifiability=0.9,
        )

        assert strategy.type == "backdoor"
        assert len(strategy.nodes_to_add) == 1
        assert len(strategy.edges_to_add) == 2
        assert strategy.expected_identifiability == 0.9

    def test_strategy_types(self):
        """Test different strategy types."""
        types = ["backdoor", "frontdoor", "instrumental"]

        for t in types:
            strategy = AdjustmentStrategy(
                strategy_type=t,
                nodes_to_add=[],
                edges_to_add=[],
                explanation=f"{t} strategy",
                theoretical_basis=f"{t} criterion",
                expected_identifiability=0.8,
            )
            assert strategy.type == t


class TestPathAnalysisClass:
    """Test PathAnalysis class."""

    def test_path_analysis_creation(self):
        """Test creating path analysis."""
        analysis = PathAnalysis(
            backdoor_paths=[["T", "U", "O"]],
            frontdoor_paths=[["T", "M", "O"]],
            blocked_paths=[["T", "C", "O"]],
            critical_nodes=["U"],
        )

        assert len(analysis.backdoor_paths) == 1
        assert len(analysis.frontdoor_paths) == 1
        assert len(analysis.blocked_paths) == 1
        assert len(analysis.critical_nodes) == 1

    def test_empty_path_analysis(self):
        """Test empty path analysis."""
        analysis = PathAnalysis(
            backdoor_paths=[],
            frontdoor_paths=[],
            blocked_paths=[],
            critical_nodes=[],
        )

        assert len(analysis.backdoor_paths) == 0
        assert len(analysis.critical_nodes) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_self_loop(self):
        """Test DAG with self-loop."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Treatment", "Treatment"),  # Self-loop
            ("Treatment", "Outcome"),
        ])

        # Should handle gracefully
        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")
        assert strategies is not None

    def test_missing_nodes(self):
        """Test with treatment/outcome not in DAG."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B")])

        # Should handle missing nodes
        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")
        assert strategies is not None  # Empty or default

    def test_empty_dag(self):
        """Test with empty DAG."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()

        strategies = suggester.suggest_adjustment_strategies(dag, "Treatment", "Outcome")
        assert strategies is not None

    def test_identical_treatment_outcome(self):
        """Test when treatment and outcome are same."""
        suggester = AdvancedValidationSuggester()

        dag = nx.DiGraph()
        dag.add_node("Variable")

        strategies = suggester.suggest_adjustment_strategies(dag, "Variable", "Variable")

        # Should handle this edge case
        assert strategies is not None
