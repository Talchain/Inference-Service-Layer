"""
Enhanced Y₀ validation with complete adjustment strategies.

Provides comprehensive suggestions for making non-identifiable
DAGs identifiable, including complete adjustment strategies,
path analysis, and validation.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from src.models.shared import DAGStructure

logger = logging.getLogger(__name__)


class AdjustmentStrategy:
    """Complete adjustment strategy for identifiability."""

    def __init__(
        self,
        strategy_type: str,
        nodes_to_add: List[str],
        edges_to_add: List[Tuple[str, str]],
        explanation: str,
        theoretical_basis: str,
        expected_identifiability: float,
    ):
        """
        Initialize adjustment strategy.

        Args:
            strategy_type: Type of strategy (backdoor, frontdoor, instrumental)
            nodes_to_add: Nodes that need to be added/measured
            edges_to_add: Edges to add to the DAG
            explanation: Plain English explanation
            theoretical_basis: Theoretical justification
            expected_identifiability: Confidence in identifiability (0-1)
        """
        self.type = strategy_type
        self.nodes_to_add = nodes_to_add
        self.edges_to_add = edges_to_add
        self.explanation = explanation
        self.theoretical_basis = theoretical_basis
        self.expected_identifiability = expected_identifiability


class PathAnalysis:
    """Analysis of causal paths in a DAG."""

    def __init__(
        self,
        backdoor_paths: List[List[str]],
        frontdoor_paths: List[List[str]],
        blocked_paths: List[List[str]],
        critical_nodes: List[str],
    ):
        """
        Initialize path analysis.

        Args:
            backdoor_paths: List of backdoor paths
            frontdoor_paths: List of frontdoor/directed paths
            blocked_paths: List of already blocked paths
            critical_nodes: Nodes that block multiple paths
        """
        self.backdoor_paths = backdoor_paths
        self.frontdoor_paths = frontdoor_paths
        self.blocked_paths = blocked_paths
        self.critical_nodes = critical_nodes


class AdvancedValidationSuggester:
    """
    Enhanced causal validation suggester.

    Provides complete adjustment strategies, not just single-node suggestions.
    Includes path analysis and strategy validation.
    """

    def __init__(self):
        """Initialize advanced validation suggester."""
        pass

    def suggest_adjustment_strategies(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[AdjustmentStrategy]:
        """
        Suggest complete adjustment strategies.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of adjustment strategies ranked by expected success
        """
        strategies = []

        # 1. Backdoor adjustment strategies
        backdoor_strategies = self._find_backdoor_strategies(dag, treatment, outcome)
        strategies.extend(backdoor_strategies)

        # 2. Frontdoor adjustment (if backdoor not possible)
        if not backdoor_strategies or all(
            len(s.nodes_to_add) > 3 for s in backdoor_strategies
        ):
            frontdoor_strategy = self._find_frontdoor_strategy(dag, treatment, outcome)
            if frontdoor_strategy:
                strategies.append(frontdoor_strategy)

        # 3. Instrumental variable strategies
        iv_strategies = self._find_instrumental_strategies(dag, treatment, outcome)
        strategies.extend(iv_strategies)

        # Rank by complexity and identifiability confidence
        return self._rank_strategies(strategies)

    def analyze_paths(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> PathAnalysis:
        """
        Comprehensive path analysis.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            PathAnalysis with all relevant paths
        """
        backdoor = self._find_backdoor_paths(dag, treatment, outcome)
        frontdoor = self._find_directed_paths(dag, treatment, outcome)
        blocked = self._find_blocked_paths(dag, treatment, outcome, backdoor)
        critical = self._identify_critical_nodes(dag, backdoor, treatment, outcome)

        return PathAnalysis(
            backdoor_paths=backdoor,
            frontdoor_paths=frontdoor,
            blocked_paths=blocked,
            critical_nodes=critical,
        )

    def _find_backdoor_strategies(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[AdjustmentStrategy]:
        """
        Find backdoor adjustment strategies.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of backdoor adjustment strategies
        """
        strategies = []

        # Find backdoor paths
        backdoor_paths = self._find_backdoor_paths(dag, treatment, outcome)

        if not backdoor_paths:
            # No backdoor paths - already identifiable
            return strategies

        # Find nodes that appear in backdoor paths (potential confounders)
        confounder_candidates = set()
        for path in backdoor_paths:
            # Exclude treatment and outcome
            for node in path:
                if node not in [treatment, outcome]:
                    confounder_candidates.add(node)

        # Try different adjustment sets
        for candidate in confounder_candidates:
            # Check if this node is already in the DAG
            if candidate in dag.nodes():
                # Already measured - good candidate
                strategy = AdjustmentStrategy(
                    strategy_type="backdoor",
                    nodes_to_add=[],  # Already present
                    edges_to_add=[],
                    explanation=f"Control for existing variable {candidate} to block backdoor paths",
                    theoretical_basis="Pearl's backdoor criterion",
                    expected_identifiability=0.9,
                )
                strategies.append(strategy)
            else:
                # Need to add this node
                # Infer edges based on position in paths
                inferred_edges = self._infer_confounder_edges(
                    dag, candidate, treatment, outcome, backdoor_paths
                )

                strategy = AdjustmentStrategy(
                    strategy_type="backdoor",
                    nodes_to_add=[candidate],
                    edges_to_add=inferred_edges,
                    explanation=f"Add and measure variable {candidate}, then control for it to block backdoor paths",
                    theoretical_basis="Pearl's backdoor criterion",
                    expected_identifiability=0.7,  # Lower confidence for unmeasured variables
                )
                strategies.append(strategy)

        return strategies

    def _find_frontdoor_strategy(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> Optional[AdjustmentStrategy]:
        """
        Find frontdoor adjustment strategy.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            Frontdoor adjustment strategy if possible
        """
        # Find potential mediators (nodes on path from treatment to outcome)
        mediators = self._find_complete_mediators(dag, treatment, outcome)

        if not mediators:
            return None

        # Pick simplest mediator set
        mediator_set = min(mediators, key=len) if mediators else set()

        if not mediator_set:
            return None

        # Construct frontdoor edges
        edges = []
        for mediator in mediator_set:
            if not dag.has_edge(treatment, mediator):
                edges.append((treatment, mediator))
            if not dag.has_edge(mediator, outcome):
                edges.append((mediator, outcome))

        return AdjustmentStrategy(
            strategy_type="frontdoor",
            nodes_to_add=list(mediator_set),
            edges_to_add=edges,
            explanation=f"Use frontdoor criterion via mediators: {', '.join(mediator_set)}",
            theoretical_basis="Pearl's frontdoor criterion",
            expected_identifiability=0.8,
        )

    def _find_instrumental_strategies(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[AdjustmentStrategy]:
        """
        Find instrumental variable strategies.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of IV strategies
        """
        strategies = []

        # Look for potential instruments
        # (nodes that affect treatment but not outcome directly)
        for node in dag.nodes():
            if node in [treatment, outcome]:
                continue

            # Check if node affects treatment
            affects_treatment = nx.has_path(dag, node, treatment)

            # Check if node doesn't affect outcome directly (only via treatment)
            if affects_treatment:
                # This could be an instrument
                strategy = AdjustmentStrategy(
                    strategy_type="instrumental",
                    nodes_to_add=[node] if node not in dag.nodes() else [],
                    edges_to_add=[(node, treatment)] if not dag.has_edge(node, treatment) else [],
                    explanation=f"Use {node} as instrumental variable",
                    theoretical_basis="Instrumental variables identification",
                    expected_identifiability=0.6,  # IV often has weaker identification
                )
                strategies.append(strategy)

        return strategies[:2]  # Return top 2 IV strategies

    def _find_backdoor_paths(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[List[str]]:
        """
        Find all backdoor paths from treatment to outcome.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of backdoor paths (each path is a list of nodes)
        """
        backdoor_paths = []

        # A backdoor path:
        # 1. Starts with an arrow into treatment
        # 2. Ends at outcome
        # 3. Does not traverse an arrow out of treatment

        # Find all nodes that point to treatment
        treatment_parents = list(dag.predecessors(treatment))

        for parent in treatment_parents:
            # Find all paths from parent to outcome
            try:
                for path in nx.all_simple_paths(
                    dag.to_undirected(), parent, outcome, cutoff=10
                ):
                    # Check if this path doesn't go through treatment->X
                    # (it should only enter treatment, not exit)
                    if treatment in path:
                        treatment_idx = path.index(treatment)
                        # Check path doesn't continue from treatment
                        if treatment_idx == len(path) - 1 or path[
                            treatment_idx + 1
                        ] in dag.predecessors(treatment):
                            # This is a valid backdoor path
                            backdoor_paths.append(path)
            except nx.NodeNotFound:
                continue

        return backdoor_paths

    def _find_directed_paths(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[List[str]]:
        """
        Find all directed paths from treatment to outcome.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of directed paths
        """
        try:
            return list(nx.all_simple_paths(dag, treatment, outcome, cutoff=10))
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def _find_blocked_paths(
        self,
        dag: nx.DiGraph,
        treatment: str,
        outcome: str,
        all_backdoor_paths: List[List[str]],
    ) -> List[List[str]]:
        """
        Find paths that are already blocked.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable
            all_backdoor_paths: All backdoor paths

        Returns:
            List of blocked paths
        """
        # For now, return empty list (would need collider detection logic)
        return []

    def _identify_critical_nodes(
        self,
        dag: nx.DiGraph,
        backdoor_paths: List[List[str]],
        treatment: str,
        outcome: str,
    ) -> List[str]:
        """
        Find nodes that block multiple paths if controlled.

        Args:
            dag: NetworkX DiGraph
            backdoor_paths: List of backdoor paths
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of critical nodes ranked by impact
        """
        node_impact = defaultdict(int)

        for path in backdoor_paths:
            # Count how many paths each node appears in
            for node in path:
                if node not in [treatment, outcome]:
                    node_impact[node] += 1

        # Return nodes sorted by impact (descending)
        return sorted(node_impact.keys(), key=lambda n: node_impact[n], reverse=True)

    def _find_complete_mediators(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[Set[str]]:
        """
        Find complete mediator sets for frontdoor criterion.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of mediator sets
        """
        mediator_sets = []

        # Find nodes on directed paths from treatment to outcome
        directed_paths = self._find_directed_paths(dag, treatment, outcome)

        for path in directed_paths:
            # Nodes between treatment and outcome
            mediators = set(path[1:-1])  # Exclude treatment and outcome
            if mediators:
                mediator_sets.append(mediators)

        return mediator_sets

    def _infer_confounder_edges(
        self,
        dag: nx.DiGraph,
        confounder: str,
        treatment: str,
        outcome: str,
        backdoor_paths: List[List[str]],
    ) -> List[Tuple[str, str]]:
        """
        Infer edges for a confounder based on backdoor paths.

        Args:
            dag: NetworkX DiGraph
            confounder: Confounder variable
            treatment: Treatment variable
            outcome: Outcome variable
            backdoor_paths: Backdoor paths containing this confounder

        Returns:
            List of edges to add
        """
        edges = []

        # Typically a confounder affects both treatment and outcome
        edges.append((confounder, treatment))
        edges.append((confounder, outcome))

        return edges

    def _rank_strategies(
        self, strategies: List[AdjustmentStrategy]
    ) -> List[AdjustmentStrategy]:
        """
        Rank strategies by desirability.

        Args:
            strategies: List of adjustment strategies

        Returns:
            Ranked list of strategies
        """

        def score(s: AdjustmentStrategy) -> float:
            # Priority:
            # 1. High identifiability confidence
            # 2. Few nodes to add (simpler)
            # 3. Few edges to add
            identifiability_score = s.expected_identifiability * 0.5
            simplicity_score = 1.0 / (len(s.nodes_to_add) + 1) * 0.3
            edge_score = 1.0 / (len(s.edges_to_add) + 1) * 0.2

            return identifiability_score + simplicity_score + edge_score

        return sorted(strategies, key=score, reverse=True)
