"""
Graph parsing utilities for converting DAG structures.

Handles conversion between different graph representations:
- Edge list format (API input)
- NetworkX graphs
- Y₀ graph format
"""

from typing import List, Tuple

import networkx as nx
from y0.dsl import Variable
from y0.graph import NxMixedGraph


def edge_list_to_networkx(
    nodes: List[str],
    edges: List[Tuple[str, str]],
) -> nx.DiGraph:
    """
    Convert edge list representation to NetworkX DiGraph.

    Args:
        nodes: List of node names
        edges: List of (from, to) tuples

    Returns:
        nx.DiGraph: NetworkX directed graph

    Example:
        >>> graph = edge_list_to_networkx(
        ...     ["A", "B", "C"],
        ...     [("A", "B"), ("B", "C")]
        ... )
        >>> list(graph.nodes())
        ['A', 'B', 'C']
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def networkx_to_y0(graph: nx.DiGraph) -> NxMixedGraph:
    """
    Convert NetworkX graph to Y₀ NxMixedGraph format.

    Y₀ uses Variable objects and NxMixedGraph for causal inference.

    Args:
        graph: NetworkX directed graph

    Returns:
        NxMixedGraph: Y₀ compatible graph

    Example:
        >>> nx_graph = nx.DiGraph([("X", "Y"), ("Z", "Y")])
        >>> y0_graph = networkx_to_y0(nx_graph)
    """
    # Create Y₀ graph with directed edges only
    y0_graph = NxMixedGraph()

    # Add nodes as Variable objects
    for node in graph.nodes():
        y0_graph.add_node(Variable(node))

    # Add edges
    for from_node, to_node in graph.edges():
        y0_graph.add_directed_edge(
            Variable(from_node),
            Variable(to_node),
        )

    return y0_graph


def edge_list_to_y0(
    nodes: List[str],
    edges: List[Tuple[str, str]],
) -> NxMixedGraph:
    """
    Convert edge list directly to Y₀ graph format.

    Combines edge_list_to_networkx and networkx_to_y0.

    Args:
        nodes: List of node names
        edges: List of (from, to) tuples

    Returns:
        NxMixedGraph: Y₀ compatible graph

    Example:
        >>> y0_graph = edge_list_to_y0(
        ...     ["X", "Y", "Z"],
        ...     [("X", "Y"), ("Z", "Y")]
        ... )
    """
    nx_graph = edge_list_to_networkx(nodes, edges)
    return networkx_to_y0(nx_graph)


def get_parents(graph: nx.DiGraph, node: str) -> List[str]:
    """
    Get parent nodes (direct causes) of a given node.

    Args:
        graph: NetworkX directed graph
        node: Node name

    Returns:
        List[str]: List of parent node names

    Example:
        >>> graph = nx.DiGraph([("A", "C"), ("B", "C")])
        >>> get_parents(graph, "C")
        ['A', 'B']
    """
    return list(graph.predecessors(node))


def get_children(graph: nx.DiGraph, node: str) -> List[str]:
    """
    Get child nodes (direct effects) of a given node.

    Args:
        graph: NetworkX directed graph
        node: Node name

    Returns:
        List[str]: List of child node names

    Example:
        >>> graph = nx.DiGraph([("A", "B"), ("A", "C")])
        >>> get_children(graph, "A")
        ['B', 'C']
    """
    return list(graph.successors(node))


def find_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
) -> List[List[str]]:
    """
    Find all paths from source to target.

    Args:
        graph: NetworkX directed graph
        source: Source node name
        target: Target node name

    Returns:
        List[List[str]]: List of paths (each path is a list of nodes)

    Example:
        >>> graph = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
        >>> find_paths(graph, "A", "C")
        [['A', 'C'], ['A', 'B', 'C']]
    """
    try:
        return list(nx.all_simple_paths(graph, source, target))
    except nx.NetworkXNoPath:
        return []


def find_backdoor_paths(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> List[List[str]]:
    """
    Find backdoor paths between treatment and outcome.

    A backdoor path is a path from treatment to outcome that starts
    with an arrow into the treatment (confounding path).

    Args:
        graph: NetworkX directed graph
        treatment: Treatment node name
        outcome: Outcome node name

    Returns:
        List[List[str]]: List of backdoor paths

    Example:
        >>> # Graph: Z -> X -> Y, Z -> Y (Z confounds X-Y)
        >>> graph = nx.DiGraph([("Z", "X"), ("X", "Y"), ("Z", "Y")])
        >>> find_backdoor_paths(graph, "X", "Y")
        [['X', 'Z', 'Y']]  # Backdoor path through confounder Z
    """
    backdoor_paths = []

    # Get parents of treatment (potential confounders)
    parents = get_parents(graph, treatment)

    # For each parent, find paths to outcome
    for parent in parents:
        # Check if there's a path from parent to outcome
        if nx.has_path(graph, parent, outcome):
            paths = find_paths(graph, parent, outcome)
            # Prepend treatment to each path to show full backdoor path
            for path in paths:
                backdoor_paths.append([treatment] + path)

    return backdoor_paths
