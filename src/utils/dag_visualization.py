"""
DAG Visualization utilities for causal graphs.

Provides rendering capabilities for DAGs with:
- Path highlighting (backdoor, frontdoor, direct)
- Node role coloring (treatment, outcome, confounder, mediator)
- Export to multiple formats (SVG, PNG, JSON)
- Interactive visualization data for frontends
"""

import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# Node role colors
NODE_COLORS = {
    "treatment": "#3498db",  # Blue
    "outcome": "#e74c3c",    # Red
    "confounder": "#f39c12", # Orange
    "mediator": "#9b59b6",   # Purple
    "instrument": "#1abc9c", # Teal
    "default": "#95a5a6",    # Gray
}

# Edge colors
EDGE_COLORS = {
    "backdoor": "#e74c3c",   # Red (confounding)
    "direct": "#2ecc71",     # Green (causal)
    "highlighted": "#f39c12",# Orange (highlighted path)
    "default": "#7f8c8d",    # Dark gray
}


class DAGVisualization:
    """
    DAG visualization generator.

    Creates visual representations of causal DAGs with
    annotations, path highlighting, and multiple export formats.
    """

    def __init__(self):
        """Initialize DAG visualization generator."""
        pass

    def render_to_json(
        self,
        dag: nx.DiGraph,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        highlighted_paths: Optional[List[List[str]]] = None,
        node_roles: Optional[Dict[str, str]] = None,
        layout: str = "hierarchical",
    ) -> Dict[str, Any]:
        """
        Render DAG to JSON format for frontend visualization.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable (for coloring)
            outcome: Outcome variable (for coloring)
            highlighted_paths: Paths to highlight
            node_roles: Dict mapping node names to roles
            layout: Layout algorithm (hierarchical, spring, circular)

        Returns:
            Dict with nodes, edges, and layout information
        """
        # Compute node positions
        positions = self._compute_layout(dag, layout, treatment, outcome)

        # Determine node roles
        if node_roles is None:
            node_roles = self._infer_node_roles(
                dag, treatment, outcome, highlighted_paths
            )

        # Build nodes list
        nodes = []
        for node in dag.nodes():
            role = node_roles.get(node, "default")
            nodes.append({
                "id": node,
                "label": node,
                "role": role,
                "color": NODE_COLORS.get(role, NODE_COLORS["default"]),
                "x": positions[node][0],
                "y": positions[node][1],
            })

        # Build edges list
        edges = []
        highlighted_edges = self._get_highlighted_edges(highlighted_paths)

        for source, target in dag.edges():
            edge_type = "default"
            if (source, target) in highlighted_edges:
                edge_type = "highlighted"

            edges.append({
                "source": source,
                "target": target,
                "type": edge_type,
                "color": EDGE_COLORS.get(edge_type, EDGE_COLORS["default"]),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "layout": layout,
            "metadata": {
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "treatment": treatment,
                "outcome": outcome,
                "is_acyclic": nx.is_directed_acyclic_graph(dag),
            }
        }

    def render_to_dot(
        self,
        dag: nx.DiGraph,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        highlighted_paths: Optional[List[List[str]]] = None,
        node_roles: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Render DAG to Graphviz DOT format.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable
            highlighted_paths: Paths to highlight
            node_roles: Node role assignments

        Returns:
            DOT format string
        """
        if node_roles is None:
            node_roles = self._infer_node_roles(
                dag, treatment, outcome, highlighted_paths
            )

        highlighted_edges = self._get_highlighted_edges(highlighted_paths)

        # Build DOT string
        lines = ["digraph CausalDAG {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=ellipse, style=filled];")
        lines.append("")

        # Add nodes with colors
        for node in dag.nodes():
            role = node_roles.get(node, "default")
            color = NODE_COLORS.get(role, NODE_COLORS["default"])
            lines.append(f'  "{node}" [fillcolor="{color}", label="{node}"];')

        lines.append("")

        # Add edges with colors
        for source, target in dag.edges():
            if (source, target) in highlighted_edges:
                color = EDGE_COLORS["highlighted"]
                lines.append(
                    f'  "{source}" -> "{target}" [color="{color}", penwidth=2.0];'
                )
            else:
                color = EDGE_COLORS["default"]
                lines.append(f'  "{source}" -> "{target}" [color="{color}"];')

        lines.append("}")

        return "\n".join(lines)

    def create_path_visualization(
        self,
        dag: nx.DiGraph,
        paths: List[List[str]],
        treatment: str,
        outcome: str,
        path_type: str = "backdoor",
    ) -> Dict[str, Any]:
        """
        Create visualization focused on specific paths.

        Args:
            dag: NetworkX DiGraph
            paths: List of paths to visualize
            treatment: Treatment variable
            outcome: Outcome variable
            path_type: Type of paths (backdoor, frontdoor, direct)

        Returns:
            Visualization dict with path annotations
        """
        # Identify nodes involved in paths
        path_nodes = set()
        for path in paths:
            path_nodes.update(path)

        # Create node roles
        node_roles = {}
        for node in dag.nodes():
            if node == treatment:
                node_roles[node] = "treatment"
            elif node == outcome:
                node_roles[node] = "outcome"
            elif node in path_nodes:
                # Determine role based on position
                if path_type == "backdoor":
                    node_roles[node] = "confounder"
                elif path_type == "frontdoor":
                    node_roles[node] = "mediator"
                else:
                    node_roles[node] = "default"
            else:
                node_roles[node] = "default"

        # Generate visualization
        viz = self.render_to_json(
            dag,
            treatment=treatment,
            outcome=outcome,
            highlighted_paths=paths,
            node_roles=node_roles,
        )

        # Add path annotations
        viz["paths"] = [
            {
                "nodes": path,
                "type": path_type,
                "description": self._describe_path(path, treatment, outcome, path_type),
            }
            for path in paths
        ]

        return viz

    def create_strategy_visualization(
        self,
        dag: nx.DiGraph,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
        strategy_type: str,
    ) -> Dict[str, Any]:
        """
        Create visualization showing adjustment strategy.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable
            adjustment_set: Variables to adjust for
            strategy_type: Type of strategy (backdoor, frontdoor, IV)

        Returns:
            Visualization dict with strategy annotations
        """
        # Create node roles
        node_roles = {}
        for node in dag.nodes():
            if node == treatment:
                node_roles[node] = "treatment"
            elif node == outcome:
                node_roles[node] = "outcome"
            elif node in adjustment_set:
                if strategy_type == "backdoor":
                    node_roles[node] = "confounder"
                elif strategy_type == "frontdoor":
                    node_roles[node] = "mediator"
                elif strategy_type == "instrumental":
                    node_roles[node] = "instrument"
                else:
                    node_roles[node] = "default"
            else:
                node_roles[node] = "default"

        viz = self.render_to_json(
            dag,
            treatment=treatment,
            outcome=outcome,
            node_roles=node_roles,
        )

        # Add strategy annotation
        viz["strategy"] = {
            "type": strategy_type,
            "adjustment_set": adjustment_set,
            "description": self._describe_strategy(
                strategy_type, adjustment_set, treatment, outcome
            ),
        }

        return viz

    def _compute_layout(
        self,
        dag: nx.DiGraph,
        layout: str,
        treatment: Optional[str],
        outcome: Optional[str],
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute node positions for layout.

        Args:
            dag: NetworkX DiGraph
            layout: Layout algorithm name
            treatment: Treatment variable (for hierarchical layout)
            outcome: Outcome variable (for hierarchical layout)

        Returns:
            Dict mapping node names to (x, y) positions
        """
        if layout == "hierarchical":
            # Use topological ordering for y-axis
            try:
                topo_order = list(nx.topological_sort(dag))
                positions = {}

                # Assign y based on topological level
                levels = {}
                for i, node in enumerate(topo_order):
                    levels[node] = i

                # Assign x based on connectivity
                for node in dag.nodes():
                    level = levels.get(node, 0)
                    # Spread nodes horizontally within level
                    x = hash(node) % 100 / 100.0
                    y = level / (len(topo_order) + 1)
                    positions[node] = (x, y)

                return positions
            except nx.NetworkXError:
                # Fall back to spring layout if not DAG
                logger.warning("Failed hierarchical layout, using spring")
                layout = "spring"

        if layout == "circular":
            pos = nx.circular_layout(dag)
        elif layout == "spring":
            pos = nx.spring_layout(dag, seed=42)
        else:
            # Default to spring layout
            pos = nx.spring_layout(dag, seed=42)

        # Convert to dict of tuples
        return {node: (float(x), float(y)) for node, (x, y) in pos.items()}

    def _infer_node_roles(
        self,
        dag: nx.DiGraph,
        treatment: Optional[str],
        outcome: Optional[str],
        highlighted_paths: Optional[List[List[str]]],
    ) -> Dict[str, str]:
        """
        Infer node roles based on graph structure.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable
            highlighted_paths: Highlighted paths

        Returns:
            Dict mapping nodes to roles
        """
        roles = {}

        # Assign primary roles
        if treatment:
            roles[treatment] = "treatment"
        if outcome:
            roles[outcome] = "outcome"

        # Infer other roles
        if treatment and outcome:
            # Find confounders (common causes)
            treatment_ancestors = nx.ancestors(dag, treatment) if treatment in dag else set()
            outcome_ancestors = nx.ancestors(dag, outcome) if outcome in dag else set()
            confounders = treatment_ancestors & outcome_ancestors

            for node in confounders:
                if node not in roles:
                    roles[node] = "confounder"

            # Find mediators (on path from treatment to outcome)
            try:
                if treatment in dag and outcome in dag:
                    paths = nx.all_simple_paths(dag, treatment, outcome, cutoff=10)
                    for path in paths:
                        for node in path[1:-1]:  # Exclude treatment and outcome
                            if node not in roles:
                                roles[node] = "mediator"
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                pass

        # Default role for remaining nodes
        for node in dag.nodes():
            if node not in roles:
                roles[node] = "default"

        return roles

    def _get_highlighted_edges(
        self, highlighted_paths: Optional[List[List[str]]]
    ) -> Set[Tuple[str, str]]:
        """
        Extract edges from highlighted paths.

        Args:
            highlighted_paths: List of paths

        Returns:
            Set of (source, target) tuples
        """
        if not highlighted_paths:
            return set()

        edges = set()
        for path in highlighted_paths:
            for i in range(len(path) - 1):
                edges.add((path[i], path[i + 1]))

        return edges

    def _describe_path(
        self, path: List[str], treatment: str, outcome: str, path_type: str
    ) -> str:
        """
        Generate human-readable path description.

        Args:
            path: Path nodes
            treatment: Treatment variable
            outcome: Outcome variable
            path_type: Path type

        Returns:
            Description string
        """
        path_str = " → ".join(path)

        if path_type == "backdoor":
            return (
                f"Backdoor path: {path_str}. "
                f"This confounding path must be blocked by adjusting for variables on it."
            )
        elif path_type == "frontdoor":
            return (
                f"Frontdoor path: {path_str}. "
                f"Mediator-based identification through this path."
            )
        elif path_type == "direct":
            return f"Direct causal path: {path_str}. This is the causal effect of interest."
        else:
            return f"Path: {path_str}"

    def _describe_strategy(
        self, strategy_type: str, adjustment_set: List[str], treatment: str, outcome: str
    ) -> str:
        """
        Generate strategy description.

        Args:
            strategy_type: Strategy type
            adjustment_set: Variables to adjust for
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            Description string
        """
        if not adjustment_set:
            return f"{strategy_type.capitalize()} strategy: No adjustment needed."

        vars_str = ", ".join(adjustment_set)

        if strategy_type == "backdoor":
            return (
                f"Backdoor adjustment: Control for {vars_str} to identify "
                f"the causal effect of {treatment} on {outcome}."
            )
        elif strategy_type == "frontdoor":
            return (
                f"Frontdoor adjustment: Use mediators {vars_str} to identify "
                f"the causal effect when there are unobserved confounders."
            )
        elif strategy_type == "instrumental":
            return (
                f"Instrumental variable: Use {vars_str} as instrument to identify "
                f"the causal effect in the presence of confounding."
            )
        else:
            return f"{strategy_type.capitalize()} strategy using {vars_str}."


def visualize_dag(
    dag: nx.DiGraph,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    format: str = "json",
    **kwargs,
) -> Any:
    """
    Convenience function to visualize a DAG.

    Args:
        dag: NetworkX DiGraph
        treatment: Treatment variable
        outcome: Outcome variable
        format: Output format (json, dot)
        **kwargs: Additional arguments for visualization

    Returns:
        Visualization in requested format
    """
    viz = DAGVisualization()

    if format == "json":
        return viz.render_to_json(dag, treatment, outcome, **kwargs)
    elif format == "dot":
        return viz.render_to_dot(dag, treatment, outcome, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'dot'.")


def visualize_paths(
    dag: nx.DiGraph,
    paths: List[List[str]],
    treatment: str,
    outcome: str,
    path_type: str = "backdoor",
) -> Dict[str, Any]:
    """
    Convenience function to visualize paths.

    Args:
        dag: NetworkX DiGraph
        paths: Paths to visualize
        treatment: Treatment variable
        outcome: Outcome variable
        path_type: Path type

    Returns:
        Path visualization dict
    """
    viz = DAGVisualization()
    return viz.create_path_visualization(dag, paths, treatment, outcome, path_type)


def visualize_strategy(
    dag: nx.DiGraph,
    treatment: str,
    outcome: str,
    adjustment_set: List[str],
    strategy_type: str,
) -> Dict[str, Any]:
    """
    Convenience function to visualize adjustment strategy.

    Args:
        dag: NetworkX DiGraph
        treatment: Treatment variable
        outcome: Outcome variable
        adjustment_set: Adjustment set
        strategy_type: Strategy type

    Returns:
        Strategy visualization dict
    """
    viz = DAGVisualization()
    return viz.create_strategy_visualization(
        dag, treatment, outcome, adjustment_set, strategy_type
    )
