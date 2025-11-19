"""
Input validation utilities.

Provides helper functions for validating API inputs beyond Pydantic's
automatic validation.
"""

from typing import Any, Dict, List, Set, Tuple

from fastapi import HTTPException


def validate_dag_structure(
    nodes: List[str],
    edges: List[Tuple[str, str]],
) -> None:
    """
    Validate that a DAG structure is well-formed.

    Checks:
    - At least one node exists
    - All edge endpoints reference existing nodes
    - No self-loops
    - Graph is acyclic (using DFS)

    Args:
        nodes: List of node names
        edges: List of (from, to) tuples

    Raises:
        HTTPException: If validation fails with 400 status

    Example:
        >>> validate_dag_structure(["A", "B"], [("A", "B")])
        >>> # Passes validation
        >>> validate_dag_structure([], [])
        HTTPException: DAG must contain at least one node
    """
    if not nodes:
        raise HTTPException(
            status_code=400,
            detail="DAG must contain at least one node",
        )

    node_set = set(nodes)

    # Check for duplicate nodes
    if len(node_set) != len(nodes):
        raise HTTPException(
            status_code=400,
            detail="DAG contains duplicate nodes",
        )

    # Validate edges
    for from_node, to_node in edges:
        if from_node not in node_set:
            raise HTTPException(
                status_code=400,
                detail=f"Edge references non-existent node: {from_node}",
            )
        if to_node not in node_set:
            raise HTTPException(
                status_code=400,
                detail=f"Edge references non-existent node: {to_node}",
            )
        if from_node == to_node:
            raise HTTPException(
                status_code=400,
                detail=f"Self-loop detected: {from_node} -> {to_node}",
            )

    # Check for cycles using DFS
    if has_cycle(nodes, edges):
        raise HTTPException(
            status_code=400,
            detail="Graph contains cycles (not a valid DAG)",
        )


def has_cycle(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    """
    Detect cycles in a directed graph using DFS.

    Args:
        nodes: List of node names
        edges: List of (from, to) tuples

    Returns:
        bool: True if graph contains a cycle
    """
    # Build adjacency list
    graph: Dict[str, List[str]] = {node: [] for node in nodes}
    for from_node, to_node in edges:
        graph[from_node].append(to_node)

    # Track visited and recursion stack
    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in nodes:
        if node not in visited:
            if dfs(node):
                return True

    return False


def validate_node_in_graph(node: str, nodes: List[str], node_type: str = "Node") -> None:
    """
    Validate that a node exists in the graph.

    Args:
        node: Node name to check
        nodes: List of valid nodes
        node_type: Type of node for error message (e.g., "Treatment", "Outcome")

    Raises:
        HTTPException: If node not found

    Example:
        >>> validate_node_in_graph("Price", ["Price", "Revenue"], "Treatment")
        >>> # Passes
        >>> validate_node_in_graph("Cost", ["Price", "Revenue"], "Treatment")
        HTTPException: Treatment node 'Cost' not found in graph
    """
    if node not in nodes:
        raise HTTPException(
            status_code=400,
            detail=f"{node_type} node '{node}' not found in graph. "
            f"Available nodes: {', '.join(nodes)}",
        )


def validate_probability(value: float, name: str = "Probability") -> None:
    """
    Validate that a value is a valid probability (0-1).

    Args:
        value: Value to check
        name: Name for error message

    Raises:
        HTTPException: If value not in [0, 1]
    """
    if not 0 <= value <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between 0 and 1, got {value}",
        )


def validate_positive(value: float, name: str = "Value") -> None:
    """
    Validate that a value is positive.

    Args:
        value: Value to check
        name: Name for error message

    Raises:
        HTTPException: If value <= 0
    """
    if value <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be positive, got {value}",
        )
