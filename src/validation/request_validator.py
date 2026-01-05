"""
Request validation pipeline for ISL V2 response format.

Validates request structure before analysis execution.
Any blocker prevents analysis entirely.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.constants import IDENTICAL_OPTIONS_VALUE_TOLERANCE
from src.models.critique import (
    EDGE_STRENGTH_OUT_OF_RANGE,
    EMPTY_INTERVENTIONS,
    GRAPH_CYCLE_DETECTED,
    GRAPH_DISCONNECTED,
    GRAPH_EMPTY,
    IDENTICAL_OPTIONS,
    INVALID_INTERVENTION_TARGET,
    MISSING_GOAL_NODE,
    NO_EFFECTIVE_PATH_TO_GOAL,
    NO_OPTIONS,
)
from src.models.response_v2 import CritiqueV2, OptionDiagnosticV2
from src.validation.path_validator import PathValidationConfig, PathValidator


@dataclass
class ValidationResult:
    """Result of request validation."""

    is_valid: bool
    critiques: List[CritiqueV2] = field(default_factory=list)
    option_diagnostics: List[OptionDiagnosticV2] = field(default_factory=list)

    @property
    def has_blockers(self) -> bool:
        """Check if any blocker critiques exist."""
        return any(c.severity == "blocker" for c in self.critiques)


def detect_graph_cycle(edges: List[Dict[str, Any]]) -> bool:
    """
    Detect if graph contains a cycle using DFS.

    Args:
        edges: List of edge dictionaries

    Returns:
        True if cycle detected, False otherwise
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = {}
    nodes_in_edges: set = set()

    for edge in edges:
        source = (
            edge.get("from")
            or edge.get("from_")
            or edge.get("source_id")
        )
        target = edge.get("to") or edge.get("target_id")

        if source and target:
            if source not in adj:
                adj[source] = []
            adj[source].append(target)
            nodes_in_edges.add(source)
            nodes_in_edges.add(target)

    # DFS with cycle detection
    # WHITE = not visited, GRAY = in current path, BLACK = fully processed
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {node: WHITE for node in nodes_in_edges}

    def dfs(node: str) -> bool:
        """Returns True if cycle found."""
        color[node] = GRAY
        for neighbor in adj.get(node, []):
            if color.get(neighbor, WHITE) == GRAY:
                # Back edge found - cycle
                return True
            if color.get(neighbor, WHITE) == WHITE:
                if dfs(neighbor):
                    return True
        color[node] = BLACK
        return False

    for node in nodes_in_edges:
        if color[node] == WHITE:
            if dfs(node):
                return True

    return False


def canonicalise_interventions(interventions: Dict[str, float]) -> str:
    """
    Create canonical string representation for comparison.

    Args:
        interventions: Intervention dict

    Returns:
        Canonical JSON string for comparison
    """
    # Sort by key, round values to avoid floating-point issues
    pairs = sorted(
        (
            k,
            round(v / IDENTICAL_OPTIONS_VALUE_TOLERANCE)
            * IDENTICAL_OPTIONS_VALUE_TOLERANCE,
        )
        for k, v in interventions.items()
    )
    return json.dumps(pairs, sort_keys=True)


def detect_identical_options(
    options: List[Dict[str, Any]]
) -> Optional[Tuple[str, str]]:
    """
    Detect if any two options have identical interventions.

    Args:
        options: List of option dictionaries

    Returns:
        Tuple of (label_a, label_b) if identical options found, None otherwise
    """
    seen: Dict[str, Tuple[str, str]] = {}  # canonical -> (option_id, option_label)

    for option in options:
        interventions = option.get("interventions", {})

        if not interventions:
            continue  # Empty interventions handled separately

        canonical = canonicalise_interventions(interventions)

        if canonical in seen:
            prev_id, prev_label = seen[canonical]
            return (prev_label, option.get("label", option.get("id", "unknown")))

        option_id = option.get("id", "unknown")
        option_label = option.get("label", option_id)
        seen[canonical] = (option_id, option_label)

    return None


class RequestValidator:
    """
    Validates analysis requests before execution.

    Critical design decision: Block entire run on any structural/validation blocker.
    Per-option failures only occur during analysis execution, not during validation.
    """

    def __init__(
        self,
        graph: Dict[str, Any],
        options: List[Dict[str, Any]],
        goal_node_id: str,
        path_config: Optional[PathValidationConfig] = None,
    ):
        """
        Initialize request validator.

        Args:
            graph: Graph dictionary with 'nodes' and 'edges'
            options: List of option dictionaries
            goal_node_id: Goal node ID
            path_config: Optional path validation configuration
        """
        self.graph = graph
        self.nodes = graph.get("nodes", [])
        self.edges = graph.get("edges", [])
        self.options = options
        self.goal_node_id = goal_node_id

        self.nodes_by_id = {n["id"]: n for n in self.nodes}
        self.path_validator = PathValidator(
            self.nodes,
            self.edges,
            config=path_config,
        )

    def validate(self) -> ValidationResult:
        """
        Validate request structure.

        Any blocker here prevents analysis entirely.
        Per-option issues are blockers if ANY option is invalid.

        Returns:
            ValidationResult with critiques and diagnostics
        """
        critiques: List[CritiqueV2] = []
        option_diagnostics: List[OptionDiagnosticV2] = []

        # 0a. Check for empty graph
        if not self.nodes:
            critiques.append(GRAPH_EMPTY.build())
            return ValidationResult(
                is_valid=False,
                critiques=critiques,
                option_diagnostics=[],
            )

        # 0b. Check for cycles in graph (DAG required)
        if detect_graph_cycle(self.edges):
            critiques.append(GRAPH_CYCLE_DETECTED.build())
            return ValidationResult(
                is_valid=False,
                critiques=critiques,
                option_diagnostics=[],
            )

        # 0c. Check for disconnected components (warning only)
        disconnected_count = self._count_disconnected_components()
        if disconnected_count > 1:
            critiques.append(
                GRAPH_DISCONNECTED.build(count=disconnected_count)
            )

        # 0d. Check edge strength ranges (warning only)
        for edge in self.edges:
            strength = edge.get("strength", {})
            mean = strength.get("mean", 0)
            from_node = edge.get("from") or edge.get("from_")
            to_node = edge.get("to")
            if abs(mean) > 3.0:
                critiques.append(
                    EDGE_STRENGTH_OUT_OF_RANGE.build(
                        from_node=from_node,
                        to_node=to_node,
                        value=mean,
                    )
                )

        # 1. Validate goal node exists
        goal_found = self.goal_node_id in self.nodes_by_id
        if not goal_found:
            critiques.append(
                MISSING_GOAL_NODE.build(affected_node_ids=[self.goal_node_id])
            )
            # Can't validate paths without goal - return early
            return ValidationResult(
                is_valid=False,
                critiques=critiques,
                option_diagnostics=[],
            )

        # 2. Validate options exist
        if not self.options:
            critiques.append(NO_OPTIONS.build())
            return ValidationResult(
                is_valid=False,
                critiques=critiques,
                option_diagnostics=[],
            )

        # 3. Validate each option
        for option in self.options:
            option_id = option.get("id", "unknown")
            option_label = option.get("label", option_id)
            interventions = option.get("interventions", {})

            # Check interventions non-empty
            if not interventions:
                critiques.append(
                    EMPTY_INTERVENTIONS.build(
                        label=option_label,
                        affected_option_ids=[option_id],
                    )
                )
                continue  # Can't validate paths without interventions

            # Check intervention targets exist
            invalid_targets = [
                t for t in interventions.keys() if t not in self.nodes_by_id
            ]
            if invalid_targets:
                critiques.append(
                    INVALID_INTERVENTION_TARGET.build(
                        label=option_label,
                        affected_option_ids=[option_id],
                        affected_node_ids=invalid_targets,
                    )
                )
                continue  # Can't validate paths with invalid targets

            # Validate paths to goal
            diag = self.path_validator.validate_option(
                option_id=option_id,
                intervention_targets=list(interventions.keys()),
                goal_id=self.goal_node_id,
            )
            option_diagnostics.append(diag)

            # Block if no effective path
            if not diag.has_effective_path:
                critiques.append(
                    NO_EFFECTIVE_PATH_TO_GOAL.build(
                        label=option_label,
                        affected_option_ids=[option_id],
                    )
                )

        # 4. Check for identical options (only if we have valid options)
        valid_options = [
            o
            for o in self.options
            if o.get("interventions")
            and all(t in self.nodes_by_id for t in o["interventions"].keys())
        ]
        if len(valid_options) >= 2:
            identical = detect_identical_options(valid_options)
            if identical:
                critiques.append(
                    IDENTICAL_OPTIONS.build(
                        label_a=identical[0],
                        label_b=identical[1],
                    )
                )

        has_blockers = any(c.severity == "blocker" for c in critiques)

        return ValidationResult(
            is_valid=not has_blockers,
            critiques=critiques,
            option_diagnostics=option_diagnostics,
        )

    def _count_disconnected_components(self) -> int:
        """
        Count disconnected components in the graph using Union-Find.

        Returns:
            Number of connected components (1 = fully connected)
        """
        if not self.nodes:
            return 0

        # Build undirected adjacency (for connectivity, direction doesn't matter)
        node_ids = set(n["id"] for n in self.nodes)

        # Union-Find with path compression
        parent: Dict[str, str] = {nid: nid for nid in node_ids}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Union nodes connected by edges
        for edge in self.edges:
            from_node = edge.get("from") or edge.get("from_")
            to_node = edge.get("to")
            if from_node in node_ids and to_node in node_ids:
                union(from_node, to_node)

        # Count unique roots
        roots = set(find(nid) for nid in node_ids)
        return len(roots)
