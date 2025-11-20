"""
Security-focused validators for input validation.

Provides reusable validators for preventing DoS, injection, and other attacks.
"""

import re
from typing import Any, Dict, List, Tuple

# Security limits
MAX_DAG_NODES = 50
MAX_DAG_EDGES = 200
MAX_STRING_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 10000
MAX_EQUATION_LENGTH = 1000
MAX_LIST_SIZE_SMALL = 20  # For priorities, constraints
MAX_LIST_SIZE_MEDIUM = 30  # For assumptions
MAX_LIST_SIZE_LARGE = 50  # For options
MAX_DICT_SIZE = 100
MAX_MONTE_CARLO_SAMPLES = 100000
MIN_MONTE_CARLO_SAMPLES = 1000

# Regex patterns
IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')
SAFE_EQUATION_PATTERN = re.compile(r'^[a-zA-Z0-9_+\-*/()., ]+$')


def validate_dag_size(nodes: List[str], edges: List[Tuple[str, str]]) -> None:
    """
    Validate DAG size limits.

    Args:
        nodes: List of node names
        edges: List of edges

    Raises:
        ValueError: If limits exceeded
    """
    if len(nodes) > MAX_DAG_NODES:
        raise ValueError(
            f"DAG cannot exceed {MAX_DAG_NODES} nodes (got {len(nodes)}). "
            f"Simplify your model to reduce complexity."
        )

    if len(edges) > MAX_DAG_EDGES:
        raise ValueError(
            f"DAG cannot exceed {MAX_DAG_EDGES} edges (got {len(edges)}). "
            f"Simplify your model to reduce complexity."
        )


def validate_no_self_loops(edges: List[Tuple[str, str]]) -> None:
    """
    Validate no self-loops in DAG.

    Args:
        edges: List of edges

    Raises:
        ValueError: If self-loop found
    """
    for edge in edges:
        if edge[0] == edge[1]:
            raise ValueError(
                f"Self-loops not allowed: {edge[0]} → {edge[0]}. "
                f"DAGs must be acyclic."
            )


def validate_no_duplicate_nodes(nodes: List[str]) -> None:
    """
    Validate no duplicate nodes.

    Args:
        nodes: List of node names

    Raises:
        ValueError: If duplicates found
    """
    if len(nodes) != len(set(nodes)):
        duplicates = [n for n in nodes if nodes.count(n) > 1]
        raise ValueError(
            f"Duplicate nodes found: {set(duplicates)}. "
            f"Each node must be unique."
        )


def validate_edges_reference_nodes(edges: List[Tuple[str, str]], nodes: List[str]) -> None:
    """
    Validate edges only reference existing nodes.

    Args:
        edges: List of edges
        nodes: List of node names

    Raises:
        ValueError: If edge references non-existent node
    """
    node_set = set(nodes)
    for edge in edges:
        if edge[0] not in node_set:
            raise ValueError(
                f"Edge references non-existent node: '{edge[0]}'. "
                f"Add '{edge[0]}' to nodes list."
            )
        if edge[1] not in node_set:
            raise ValueError(
                f"Edge references non-existent node: '{edge[1]}'. "
                f"Add '{edge[1]}' to nodes list."
            )


def validate_identifier(name: str, field_name: str = "identifier") -> None:
    """
    Validate identifier (variable/node name).

    Args:
        name: Identifier to validate
        field_name: Field name for error message

    Raises:
        ValueError: If invalid identifier
    """
    if not IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"{field_name} '{name}' is not a valid identifier. "
            f"Must start with letter or underscore, contain only "
            f"alphanumeric characters and underscores."
        )


def validate_node_names(nodes: List[str]) -> None:
    """
    Validate all node names are valid identifiers.

    Args:
        nodes: List of node names

    Raises:
        ValueError: If any node name invalid
    """
    for node in nodes:
        validate_identifier(node, "Node name")


def validate_user_id(user_id: str) -> None:
    """
    Validate user ID format.

    Args:
        user_id: User identifier

    Raises:
        ValueError: If invalid format
    """
    if not USER_ID_PATTERN.match(user_id):
        raise ValueError(
            f"User ID '{user_id}' contains invalid characters. "
            f"Only alphanumeric, underscore, and hyphen allowed."
        )


def validate_equations_safe(equations: Dict[str, str]) -> None:
    """
    Validate structural equations contain only safe characters.

    Prevents potential code injection.

    Args:
        equations: Dictionary of variable → equation

    Raises:
        ValueError: If unsafe characters found
    """
    for var, equation in equations.items():
        if not SAFE_EQUATION_PATTERN.match(equation):
            raise ValueError(
                f"Equation for '{var}' contains unsafe characters. "
                f"Allowed: letters, numbers, +, -, *, /, (, ), ., space"
            )

        if len(equation) > MAX_EQUATION_LENGTH:
            raise ValueError(
                f"Equation for '{var}' exceeds {MAX_EQUATION_LENGTH} characters "
                f"(got {len(equation)})"
            )


def validate_dict_size(d: Dict[str, Any], field_name: str = "dictionary") -> None:
    """
    Validate dictionary size limit.

    Args:
        d: Dictionary to validate
        field_name: Field name for error message

    Raises:
        ValueError: If too large
    """
    if len(d) > MAX_DICT_SIZE:
        raise ValueError(
            f"{field_name} cannot exceed {MAX_DICT_SIZE} entries "
            f"(got {len(d)})"
        )


def validate_monte_carlo_samples(samples: int) -> None:
    """
    Validate Monte Carlo sample count.

    Args:
        samples: Number of samples

    Raises:
        ValueError: If out of valid range
    """
    if samples < MIN_MONTE_CARLO_SAMPLES:
        raise ValueError(
            f"Monte Carlo samples must be >= {MIN_MONTE_CARLO_SAMPLES} "
            f"for reasonable accuracy (got {samples})"
        )

    if samples > MAX_MONTE_CARLO_SAMPLES:
        raise ValueError(
            f"Monte Carlo samples cannot exceed {MAX_MONTE_CARLO_SAMPLES} "
            f"(got {samples}). This would cause excessive computation time."
        )
