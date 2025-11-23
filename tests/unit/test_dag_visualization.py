"""
Unit tests for DAG visualization utilities.

Tests rendering, path highlighting, and export formats.
"""

import json
import networkx as nx
import pytest

from src.utils.dag_visualization import (
    DAGVisualization,
    visualize_dag,
    visualize_paths,
    visualize_strategy,
    NODE_COLORS,
    EDGE_COLORS,
)


class TestDAGVisualization:
    """Test DAGVisualization class."""

    def test_render_to_json_basic(self):
        """Test basic JSON rendering."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z")])

        result = viz.render_to_json(dag)

        assert "nodes" in result
        assert "edges" in result
        assert "metadata" in result
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2

    def test_render_to_json_with_roles(self):
        """Test JSON rendering with node roles."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("Z", "X"), ("Z", "Y"), ("X", "Y")])

        result = viz.render_to_json(
            dag,
            treatment="X",
            outcome="Y",
        )

        # Check that nodes have correct roles
        nodes_by_id = {n["id"]: n for n in result["nodes"]}

        assert nodes_by_id["X"]["role"] == "treatment"
        assert nodes_by_id["Y"]["role"] == "outcome"
        assert nodes_by_id["Z"]["role"] == "confounder"

    def test_render_to_json_with_highlighted_paths(self):
        """Test JSON rendering with path highlighting."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("Z", "X"), ("Z", "Y"), ("X", "Y")])

        highlighted_paths = [["Z", "X", "Y"]]

        result = viz.render_to_json(
            dag,
            highlighted_paths=highlighted_paths,
        )

        # Check that edges are highlighted
        highlighted_edges = [
            e for e in result["edges"] if e["type"] == "highlighted"
        ]
        assert len(highlighted_edges) == 2  # Z->X and X->Y

    def test_render_to_dot(self):
        """Test DOT format rendering."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z")])

        result = viz.render_to_dot(dag, treatment="X", outcome="Z")

        assert "digraph CausalDAG" in result
        assert '"X"' in result
        assert '"Y"' in result
        assert '"Z"' in result
        assert '->' in result

    def test_create_path_visualization_backdoor(self):
        """Test backdoor path visualization."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("Z", "X"), ("Z", "Y")])

        paths = [["Z", "X"], ["Z", "Y"]]

        result = viz.create_path_visualization(
            dag, paths, "X", "Y", path_type="backdoor"
        )

        assert "paths" in result
        assert len(result["paths"]) == 2
        assert result["paths"][0]["type"] == "backdoor"

        # Check node roles
        nodes_by_id = {n["id"]: n for n in result["nodes"]}
        assert nodes_by_id["Z"]["role"] == "confounder"

    def test_create_path_visualization_frontdoor(self):
        """Test frontdoor path visualization."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "M"), ("M", "Y")])

        paths = [["X", "M", "Y"]]

        result = viz.create_path_visualization(
            dag, paths, "X", "Y", path_type="frontdoor"
        )

        assert "paths" in result
        assert result["paths"][0]["type"] == "frontdoor"

        # Check mediator role
        nodes_by_id = {n["id"]: n for n in result["nodes"]}
        assert nodes_by_id["M"]["role"] == "mediator"

    def test_create_strategy_visualization(self):
        """Test adjustment strategy visualization."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("Z", "X"), ("Z", "Y"), ("X", "Y")])

        result = viz.create_strategy_visualization(
            dag,
            treatment="X",
            outcome="Y",
            adjustment_set=["Z"],
            strategy_type="backdoor",
        )

        assert "strategy" in result
        assert result["strategy"]["type"] == "backdoor"
        assert result["strategy"]["adjustment_set"] == ["Z"]

        # Check node roles
        nodes_by_id = {n["id"]: n for n in result["nodes"]}
        assert nodes_by_id["Z"]["role"] == "confounder"

    def test_layout_hierarchical(self):
        """Test hierarchical layout."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "M"), ("M", "Y")])

        positions = viz._compute_layout(dag, "hierarchical", "X", "Y")

        assert len(positions) == 3
        assert all(isinstance(pos, tuple) for pos in positions.values())
        assert all(len(pos) == 2 for pos in positions.values())

    def test_layout_spring(self):
        """Test spring layout."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z")])

        positions = viz._compute_layout(dag, "spring", None, None)

        assert len(positions) == 3
        assert all(isinstance(pos, tuple) for pos in positions.values())

    def test_layout_circular(self):
        """Test circular layout."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z"), ("Z", "X")])

        positions = viz._compute_layout(dag, "circular", None, None)

        assert len(positions) == 3

    def test_infer_node_roles(self):
        """Test node role inference."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("Z", "X"),
            ("Z", "Y"),
            ("X", "M"),
            ("M", "Y"),
        ])

        roles = viz._infer_node_roles(dag, "X", "Y", None)

        assert roles["X"] == "treatment"
        assert roles["Y"] == "outcome"
        assert roles["Z"] == "confounder"
        assert roles["M"] == "mediator"

    def test_get_highlighted_edges(self):
        """Test extraction of highlighted edges."""
        viz = DAGVisualization()
        paths = [
            ["X", "M", "Y"],
            ["Z", "X"],
        ]

        edges = viz._get_highlighted_edges(paths)

        assert ("X", "M") in edges
        assert ("M", "Y") in edges
        assert ("Z", "X") in edges
        assert len(edges) == 3

    def test_describe_path_backdoor(self):
        """Test backdoor path description."""
        viz = DAGVisualization()
        path = ["Z", "X"]

        description = viz._describe_path(path, "X", "Y", "backdoor")

        assert "Backdoor path" in description
        assert "Z → X" in description
        assert "confounding" in description

    def test_describe_path_frontdoor(self):
        """Test frontdoor path description."""
        viz = DAGVisualization()
        path = ["X", "M", "Y"]

        description = viz._describe_path(path, "X", "Y", "frontdoor")

        assert "Frontdoor path" in description
        assert "X → M → Y" in description

    def test_describe_path_direct(self):
        """Test direct path description."""
        viz = DAGVisualization()
        path = ["X", "Y"]

        description = viz._describe_path(path, "X", "Y", "direct")

        assert "Direct causal path" in description
        assert "X → Y" in description

    def test_describe_strategy_backdoor(self):
        """Test backdoor strategy description."""
        viz = DAGVisualization()

        description = viz._describe_strategy(
            "backdoor", ["Z"], "X", "Y"
        )

        assert "Backdoor adjustment" in description
        assert "Control for Z" in description

    def test_describe_strategy_instrumental(self):
        """Test IV strategy description."""
        viz = DAGVisualization()

        description = viz._describe_strategy(
            "instrumental", ["I"], "X", "Y"
        )

        assert "Instrumental variable" in description
        assert "Use I as instrument" in description


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_visualize_dag_json(self):
        """Test visualize_dag with JSON format."""
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z")])

        result = visualize_dag(dag, treatment="X", outcome="Z", format="json")

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 3

    def test_visualize_dag_dot(self):
        """Test visualize_dag with DOT format."""
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y")])

        result = visualize_dag(dag, format="dot")

        assert isinstance(result, str)
        assert "digraph" in result

    def test_visualize_dag_invalid_format(self):
        """Test visualize_dag with invalid format."""
        dag = nx.DiGraph()
        dag.add_node("X")

        with pytest.raises(ValueError, match="Unknown format"):
            visualize_dag(dag, format="invalid")

    def test_visualize_paths(self):
        """Test visualize_paths function."""
        dag = nx.DiGraph()
        dag.add_edges_from([("Z", "X"), ("Z", "Y")])

        paths = [["Z", "X"], ["Z", "Y"]]

        result = visualize_paths(dag, paths, "X", "Y", path_type="backdoor")

        assert "paths" in result
        assert len(result["paths"]) == 2

    def test_visualize_strategy(self):
        """Test visualize_strategy function."""
        dag = nx.DiGraph()
        dag.add_edges_from([("Z", "X"), ("Z", "Y"), ("X", "Y")])

        result = visualize_strategy(
            dag, "X", "Y", ["Z"], "backdoor"
        )

        assert "strategy" in result
        assert result["strategy"]["type"] == "backdoor"


class TestComplexDAGs:
    """Test visualization with complex DAG structures."""

    def test_large_dag(self):
        """Test visualization of large DAG."""
        viz = DAGVisualization()
        dag = nx.DiGraph()

        # Create a larger DAG
        for i in range(10):
            for j in range(i + 1, min(i + 3, 10)):
                dag.add_edge(f"X{i}", f"X{j}")

        result = viz.render_to_json(dag)

        assert len(result["nodes"]) == 10
        assert len(result["edges"]) > 0

    def test_dag_with_multiple_paths(self):
        """Test DAG with multiple paths between nodes."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("X", "M1"),
            ("M1", "Y"),
            ("X", "M2"),
            ("M2", "Y"),
            ("X", "Y"),  # Direct path
        ])

        result = viz.render_to_json(dag, treatment="X", outcome="Y")

        nodes_by_id = {n["id"]: n for n in result["nodes"]}
        assert nodes_by_id["M1"]["role"] == "mediator"
        assert nodes_by_id["M2"]["role"] == "mediator"

    def test_dag_with_colliders(self):
        """Test DAG with collider structure."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([
            ("X", "C"),
            ("Y", "C"),  # C is a collider
        ])

        result = viz.render_to_json(dag, treatment="X", outcome="Y")

        # Just verify it renders without errors
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dag(self):
        """Test visualization of empty DAG."""
        viz = DAGVisualization()
        dag = nx.DiGraph()

        result = viz.render_to_json(dag)

        assert len(result["nodes"]) == 0
        assert len(result["edges"]) == 0

    def test_single_node_dag(self):
        """Test visualization of single-node DAG."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_node("X")

        result = viz.render_to_json(dag)

        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 0

    def test_no_paths(self):
        """Test path visualization with no paths."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_nodes_from(["X", "Y", "Z"])

        result = viz.create_path_visualization(
            dag, [], "X", "Y", path_type="backdoor"
        )

        assert result["paths"] == []

    def test_missing_treatment_or_outcome(self):
        """Test visualization without treatment/outcome specified."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z")])

        result = viz.render_to_json(dag)

        # Should still work, just without role coloring
        assert len(result["nodes"]) == 3

    def test_cyclic_graph(self):
        """Test visualization of cyclic graph (not a DAG)."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z"), ("Z", "X")])

        # Should still render, just fall back to spring layout
        result = viz.render_to_json(dag, layout="hierarchical")

        assert len(result["nodes"]) == 3
        assert result["metadata"]["is_acyclic"] is False


class TestNodeColors:
    """Test node color assignment."""

    def test_treatment_color(self):
        """Test treatment node gets correct color."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y")])

        result = viz.render_to_json(dag, treatment="X")

        treatment_node = [n for n in result["nodes"] if n["id"] == "X"][0]
        assert treatment_node["color"] == NODE_COLORS["treatment"]

    def test_outcome_color(self):
        """Test outcome node gets correct color."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y")])

        result = viz.render_to_json(dag, outcome="Y")

        outcome_node = [n for n in result["nodes"] if n["id"] == "Y"][0]
        assert outcome_node["color"] == NODE_COLORS["outcome"]

    def test_custom_roles(self):
        """Test custom node role assignment."""
        viz = DAGVisualization()
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z")])

        node_roles = {"X": "instrument", "Y": "mediator", "Z": "outcome"}

        result = viz.render_to_json(dag, node_roles=node_roles)

        nodes_by_id = {n["id"]: n for n in result["nodes"]}
        assert nodes_by_id["X"]["color"] == NODE_COLORS["instrument"]
        assert nodes_by_id["Y"]["color"] == NODE_COLORS["mediator"]
        assert nodes_by_id["Z"]["color"] == NODE_COLORS["outcome"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
