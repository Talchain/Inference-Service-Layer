"""
Path-to-goal validation using BFS.

Validates that intervention targets have causal paths to the goal node.
Supports both structural paths (edge exists) and effective paths (non-zero strength).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.constants import (
    DEFAULT_EXISTS_PROBABILITY_THRESHOLD,
    DEFAULT_STRENGTH_THRESHOLD,
)
from src.models.response_v2 import OptionDiagnosticV2


@dataclass
class PathValidationConfig:
    """Configuration for path validation thresholds."""

    exists_probability_threshold: float = DEFAULT_EXISTS_PROBABILITY_THRESHOLD
    strength_threshold: float = DEFAULT_STRENGTH_THRESHOLD


class PathValidator:
    """
    Validates causal paths from intervention targets to goal.

    Uses BFS with collections.deque for O(n) time complexity.
    """

    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        config: Optional[PathValidationConfig] = None,
    ):
        """
        Initialize path validator.

        Args:
            nodes: List of node dictionaries with 'id' field
            edges: List of edge dictionaries with 'from', 'to',
                   'exists_probability', and 'strength' fields
            config: Optional configuration for thresholds
        """
        self.nodes_by_id = {n["id"]: n for n in nodes}
        self.config = config or PathValidationConfig()

        # Build two adjacency lists: structural and effective
        self.structural_adjacency = self._build_adjacency(
            edges, check_strength=False
        )
        self.effective_adjacency = self._build_adjacency(
            edges, check_strength=True
        )

    def _build_adjacency(
        self,
        edges: List[Dict[str, Any]],
        check_strength: bool,
    ) -> Dict[str, List[str]]:
        """
        Build adjacency list from edges meeting threshold criteria.

        Args:
            edges: Edge list
            check_strength: Whether to also check strength threshold

        Returns:
            Adjacency list mapping source node -> list of target nodes
        """
        adj: Dict[str, List[str]] = {}

        for edge in edges:
            # Get exists_probability (default to 1.0 if not specified)
            exists_prob = edge.get("exists_probability", 1.0)

            # Must meet exists_probability threshold
            if exists_prob < self.config.exists_probability_threshold:
                continue

            # For effective paths, also check strength
            if check_strength:
                strength = edge.get("strength", {})
                if isinstance(strength, dict):
                    strength_mean = strength.get("mean", 0.0)
                else:
                    strength_mean = 0.0

                if abs(strength_mean) < self.config.strength_threshold:
                    continue

            # Get source node (handle multiple field name conventions)
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

        return adj

    def has_path(
        self,
        source_id: str,
        goal_id: str,
        adjacency: Dict[str, List[str]],
    ) -> bool:
        """
        Check if path exists using BFS with deque (O(n)).

        Args:
            source_id: Starting node ID
            goal_id: Target node ID
            adjacency: Adjacency list to use

        Returns:
            True if path exists, False otherwise
        """
        if source_id == goal_id:
            return True

        if source_id not in self.nodes_by_id:
            return False

        visited: Set[str] = set()
        queue: deque = deque([source_id])

        while queue:
            current = queue.popleft()  # O(1) with deque

            if current == goal_id:
                return True

            if current in visited:
                continue
            visited.add(current)

            for neighbour in adjacency.get(current, []):
                if neighbour not in visited:
                    queue.append(neighbour)

        return False

    def has_structural_path(self, source_id: str, goal_id: str) -> bool:
        """
        Check if structural path exists (edge probability only).

        Args:
            source_id: Starting node ID
            goal_id: Target node ID

        Returns:
            True if structural path exists
        """
        return self.has_path(source_id, goal_id, self.structural_adjacency)

    def has_effective_path(self, source_id: str, goal_id: str) -> bool:
        """
        Check if effective path exists (probability AND non-zero strength).

        Args:
            source_id: Starting node ID
            goal_id: Target node ID

        Returns:
            True if effective path exists
        """
        return self.has_path(source_id, goal_id, self.effective_adjacency)

    def validate_option(
        self,
        option_id: str,
        intervention_targets: List[str],
        goal_id: str,
    ) -> OptionDiagnosticV2:
        """
        Validate all intervention targets for an option.

        Args:
            option_id: Option identifier
            intervention_targets: List of node IDs being intervened on
            goal_id: Goal node ID

        Returns:
            OptionDiagnosticV2 with validation results
        """
        targets_with_effective = 0
        targets_without_effective = 0
        has_any_structural = False
        has_any_effective = False
        warnings: List[str] = []

        for target in intervention_targets:
            if target not in self.nodes_by_id:
                targets_without_effective += 1
                warnings.append(f"Target '{target}' not found in graph")
                continue

            structural = self.has_structural_path(target, goal_id)
            effective = self.has_effective_path(target, goal_id)

            if structural:
                has_any_structural = True

            if effective:
                has_any_effective = True
                targets_with_effective += 1
            else:
                targets_without_effective += 1
                if structural:
                    warnings.append(
                        f"Target '{target}' has path but coefficients may be too small"
                    )

        return OptionDiagnosticV2(
            option_id=option_id,
            intervention_count=len(intervention_targets),
            has_structural_path=has_any_structural,
            has_effective_path=has_any_effective,
            targets_with_effective_path_count=targets_with_effective,
            targets_without_effective_path_count=targets_without_effective,
            warnings=warnings,
        )
