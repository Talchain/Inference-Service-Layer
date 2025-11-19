"""
Unit tests for validation utilities.
"""

import pytest
from fastapi import HTTPException

from src.utils.validation import (
    has_cycle,
    validate_dag_structure,
    validate_node_in_graph,
    validate_positive,
    validate_probability,
)


def test_validate_dag_structure_valid():
    """Test validation of valid DAG."""
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C")]

    # Should not raise
    validate_dag_structure(nodes, edges)


def test_validate_dag_structure_empty_nodes():
    """Test validation fails for empty nodes."""
    with pytest.raises(HTTPException) as exc_info:
        validate_dag_structure([], [])

    assert exc_info.value.status_code == 400
    assert "at least one node" in exc_info.value.detail


def test_validate_dag_structure_self_loop():
    """Test validation fails for self-loop."""
    nodes = ["A", "B"]
    edges = [("A", "A")]

    with pytest.raises(HTTPException) as exc_info:
        validate_dag_structure(nodes, edges)

    assert exc_info.value.status_code == 400
    assert "Self-loop" in exc_info.value.detail


def test_validate_dag_structure_cycle():
    """Test validation fails for cycle."""
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C"), ("C", "A")]

    with pytest.raises(HTTPException) as exc_info:
        validate_dag_structure(nodes, edges)

    assert exc_info.value.status_code == 400
    assert "cycle" in exc_info.value.detail.lower()


def test_has_cycle_no_cycle():
    """Test cycle detection with acyclic graph."""
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C")]

    assert not has_cycle(nodes, edges)


def test_has_cycle_with_cycle():
    """Test cycle detection with cyclic graph."""
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C"), ("C", "A")]

    assert has_cycle(nodes, edges)


def test_validate_node_in_graph_valid():
    """Test node validation with valid node."""
    validate_node_in_graph("A", ["A", "B", "C"], "Treatment")


def test_validate_node_in_graph_invalid():
    """Test node validation with invalid node."""
    with pytest.raises(HTTPException) as exc_info:
        validate_node_in_graph("D", ["A", "B", "C"], "Treatment")

    assert exc_info.value.status_code == 400
    assert "not found" in exc_info.value.detail


def test_validate_probability_valid():
    """Test probability validation with valid values."""
    validate_probability(0.0)
    validate_probability(0.5)
    validate_probability(1.0)


def test_validate_probability_invalid():
    """Test probability validation with invalid values."""
    with pytest.raises(HTTPException):
        validate_probability(-0.1)

    with pytest.raises(HTTPException):
        validate_probability(1.1)


def test_validate_positive_valid():
    """Test positive validation with valid values."""
    validate_positive(0.1)
    validate_positive(100)


def test_validate_positive_invalid():
    """Test positive validation with invalid values."""
    with pytest.raises(HTTPException):
        validate_positive(0)

    with pytest.raises(HTTPException):
        validate_positive(-1)
