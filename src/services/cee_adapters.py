"""
CEE Adapter utilities for converting GraphV1 to ISL formats.

Provides helper functions to convert CEE's GraphV1 decision graphs
into ISL's internal representations (NetworkX DAGs, StructuralModels, etc.).
"""

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from src.models.shared import DAGStructure, Distribution, DistributionType, GraphV1, StructuralModel

logger = logging.getLogger(__name__)


def graph_v1_to_dag_structure(graph: GraphV1) -> DAGStructure:
    """
    Convert GraphV1 to ISL DAGStructure.

    Args:
        graph: CEE GraphV1 structure

    Returns:
        DAGStructure: ISL DAG structure
    """
    nodes = [node.id for node in graph.nodes]
    edges = [(edge.from_, edge.to) for edge in graph.edges]

    return DAGStructure(nodes=nodes, edges=edges)


def graph_v1_to_networkx(graph: GraphV1) -> nx.DiGraph:
    """
    Convert GraphV1 to NetworkX directed graph.

    Args:
        graph: CEE GraphV1 structure

    Returns:
        nx.DiGraph: NetworkX directed graph with node/edge attributes
    """
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in graph.nodes:
        G.add_node(
            node.id,
            kind=node.kind.value,
            label=node.label,
            body=node.body,
            belief=node.belief,
            metadata=node.metadata or {}
        )

    # Add edges with attributes
    for edge in graph.edges:
        G.add_edge(
            edge.from_,
            edge.to,
            weight=edge.weight if edge.weight is not None else 1.0,
            label=edge.label
        )

    return G


def graph_v1_to_structural_model(graph: GraphV1) -> StructuralModel:
    """
    Convert GraphV1 to ISL StructuralModel.

    Creates a simple linear structural causal model from the graph,
    using edge weights as coefficients.

    Args:
        graph: CEE GraphV1 structure

    Returns:
        StructuralModel: ISL structural model

    Raises:
        ValueError: If graph cannot be converted to valid SCM
    """
    G = graph_v1_to_networkx(graph)

    # Identify root nodes (no incoming edges)
    root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

    if not root_nodes:
        raise ValueError("Graph has no root nodes - cannot create structural model")

    variables = list(G.nodes())
    equations = {}
    distributions = {}

    # Create equations for non-root nodes
    for node in G.nodes():
        predecessors = list(G.predecessors(node))

        if not predecessors:
            # Root node - exogenous variable
            # Use belief as prior if available
            node_data = G.nodes[node]
            belief = node_data.get('belief', 0.5)

            # Create simple normal distribution centered at belief
            distributions[node] = Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": float(belief * 100), "std": 20.0}
            )
            # Root nodes have no equation (exogenous)
        else:
            # Non-root node - create linear equation from parents
            terms = []
            for parent in predecessors:
                edge_data = G.get_edge_data(parent, node)
                weight = edge_data.get('weight', 1.0)
                terms.append(f"{weight}*{parent}")

            # Add small noise term
            intercept = 10.0
            equation = f"{intercept} + " + " + ".join(terms)
            equations[node] = equation

    return StructuralModel(
        variables=variables,
        equations=equations,
        distributions=distributions
    )


def infer_treatment(graph: GraphV1) -> str:
    """
    Infer treatment variable from GraphV1.

    Looks for nodes with kind='decision' or 'option'. If multiple exist,
    returns the first one. If none exist, returns first node.

    Args:
        graph: CEE GraphV1 structure

    Returns:
        str: Node ID to use as treatment
    """
    # Look for decision or option nodes
    decision_nodes = [
        node.id for node in graph.nodes
        if node.kind.value in ['decision', 'option']
    ]

    if decision_nodes:
        return decision_nodes[0]

    # Fallback: first node
    logger.warning(
        "No decision/option nodes found in graph, using first node as treatment"
    )
    return graph.nodes[0].id


def infer_outcome(graph: GraphV1) -> str:
    """
    Infer outcome variable from GraphV1.

    Looks for nodes with kind='outcome' or 'goal'. If multiple exist,
    returns the first one. If none exist, returns last node.

    Args:
        graph: CEE GraphV1 structure

    Returns:
        str: Node ID to use as outcome
    """
    # Look for outcome or goal nodes
    outcome_nodes = [
        node.id for node in graph.nodes
        if node.kind.value in ['outcome', 'goal']
    ]

    if outcome_nodes:
        return outcome_nodes[0]

    # Fallback: last node (often the outcome in decision graphs)
    logger.warning(
        "No outcome/goal nodes found in graph, using last node as outcome"
    )
    return graph.nodes[-1].id


def extract_assumptions(graph: GraphV1) -> List[Dict[str, any]]:
    """
    Extract assumptions from GraphV1 for sensitivity analysis.

    Extracts edge weights and node beliefs as testable assumptions.

    Args:
        graph: CEE GraphV1 structure

    Returns:
        List of assumption dictionaries with name, current_value, range
    """
    assumptions = []

    # Extract beliefs from nodes
    for node in graph.nodes:
        if node.belief is not None:
            assumptions.append({
                "name": f"{node.id}_belief",
                "description": f"Belief in {node.label}",
                "current_value": node.belief,
                "range": [0.0, 1.0]
            })

    # Extract edge weights
    for edge in graph.edges:
        if edge.weight is not None:
            assumptions.append({
                "name": f"{edge.from_}_to_{edge.to}_weight",
                "description": f"Influence of {edge.from_} on {edge.to}",
                "current_value": edge.weight,
                "range": [-3.0, 3.0]
            })

    return assumptions


def calculate_graph_complexity(graph: GraphV1) -> Dict[str, any]:
    """
    Calculate complexity metrics for GraphV1.

    Args:
        graph: CEE GraphV1 structure

    Returns:
        Dict with complexity metrics
    """
    G = graph_v1_to_networkx(graph)

    metrics = {
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "avg_degree": sum(dict(G.degree()).values()) / len(G.nodes()) if G.nodes() else 0,
        "is_connected": nx.is_weakly_connected(G),
        "num_components": nx.number_weakly_connected_components(G),
    }

    # Try to compute more metrics
    try:
        # Longest path (only works for DAGs)
        if nx.is_directed_acyclic_graph(G):
            metrics["longest_path"] = len(nx.dag_longest_path(G))
        else:
            metrics["longest_path"] = None
    except:
        metrics["longest_path"] = None

    return metrics


def format_graph_summary(graph: GraphV1) -> str:
    """
    Create human-readable summary of GraphV1.

    Args:
        graph: CEE GraphV1 structure

    Returns:
        str: Summary description
    """
    node_types = {}
    for node in graph.nodes:
        kind = node.kind.value
        node_types[kind] = node_types.get(kind, 0) + 1

    type_summary = ", ".join(f"{count} {kind}" for kind, count in sorted(node_types.items()))

    return (
        f"Graph with {len(graph.nodes)} nodes ({type_summary}) "
        f"and {len(graph.edges)} edges"
    )


def find_critical_path_edges(G: nx.DiGraph, treatment: str, outcome: str) -> List[Tuple[str, str]]:
    """
    Identify edges on critical paths from treatment to outcome.

    Args:
        G: NetworkX directed graph
        treatment: Treatment/decision node ID
        outcome: Outcome node ID

    Returns:
        List of (from_node, to_node) tuples for critical path edges
    """
    critical_edges = set()

    try:
        if nx.has_path(G, treatment, outcome):
            # Find all simple paths from treatment to outcome
            paths = list(nx.all_simple_paths(G, treatment, outcome, cutoff=10))

            # Add all edges that appear in any path
            for path in paths:
                for i in range(len(path) - 1):
                    critical_edges.add((path[i], path[i + 1]))
    except (nx.NetworkXError, nx.NodeNotFound):
        # If nodes don't exist or no path, return empty set
        pass

    return list(critical_edges)


def calculate_node_centralities(G: nx.DiGraph) -> Dict[str, float]:
    """
    Calculate centrality metrics for all nodes.

    Args:
        G: NetworkX directed graph

    Returns:
        Dict mapping node ID to centrality score (0-1)
    """
    centralities = {}

    try:
        # Use betweenness centrality as primary metric
        betweenness = nx.betweenness_centrality(G)
        degree_cent = nx.degree_centrality(G)

        # Combine metrics (weighted average)
        for node in G.nodes():
            centralities[node] = 0.6 * betweenness.get(node, 0) + 0.4 * degree_cent.get(node, 0)
    except:
        # Fallback to degree centrality only
        try:
            centralities = nx.degree_centrality(G)
        except:
            # Last resort: all nodes get 0.5
            for node in G.nodes():
                centralities[node] = 0.5

    return centralities


def identify_node_role(
    node_id: str,
    graph: GraphV1,
    treatment: str,
    outcome: str
) -> str:
    """
    Identify the role of a node in the causal graph.

    Args:
        node_id: Node ID to classify
        graph: GraphV1 structure
        treatment: Treatment node ID
        outcome: Outcome node ID

    Returns:
        Role string: 'treatment', 'outcome', 'mediator', 'confounder', 'other'
    """
    if node_id == treatment:
        return 'treatment'
    elif node_id == outcome:
        return 'outcome'

    # Check if node is on path between treatment and outcome
    G = graph_v1_to_networkx(graph)

    try:
        if nx.has_path(G, treatment, outcome):
            paths = list(nx.all_simple_paths(G, treatment, outcome, cutoff=10))
            for path in paths:
                if node_id in path:
                    return 'mediator'

        # Check if node influences both treatment and outcome (confounder)
        has_path_to_treatment = nx.has_path(G, node_id, treatment)
        has_path_to_outcome = nx.has_path(G, node_id, outcome)

        if has_path_to_treatment and has_path_to_outcome:
            return 'confounder'
    except (nx.NetworkXError, nx.NodeNotFound):
        pass

    return 'other'
