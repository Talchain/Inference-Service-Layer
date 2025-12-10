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
from src.utils.cache import get_cache
from src.utils.error_recovery import (
    CircuitBreaker,
    health_monitor,
)

logger = logging.getLogger(__name__)

# Constants for configuration
MAX_PATH_DEPTH = 10  # Maximum depth for path search
MAX_STRATEGIES = 10  # Maximum number of strategies to return

# Cache configuration
CACHE_TTL = 1800  # 30 minutes
CACHE_MAX_SIZE = 500  # Maximum cached results

# Circuit breakers for expensive operations
_path_analysis_breaker = CircuitBreaker("path_analysis", failure_threshold=3, timeout=60)
_strategy_generation_breaker = CircuitBreaker("strategy_generation", failure_threshold=3, timeout=60)


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
    ) -> None:
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
    ) -> None:
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

    Features:
    - Caching of expensive path operations
    - Caching of strategy generation
    - Performance optimization for repeated queries
    """

    def __init__(self, enable_caching: bool = True) -> None:
        """
        Initialize advanced validation suggester.

        Args:
            enable_caching: Whether to enable result caching (default: True)
        """
        self.enable_caching = enable_caching

        # Initialize caches for expensive operations
        if enable_caching:
            self._path_cache = get_cache(
                "validation_paths", max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL
            )
            self._strategy_cache = get_cache(
                "validation_strategies", max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL
            )
            logger.info("caching_enabled", extra={"service": "AdvancedValidationSuggester"})
        else:
            self._path_cache = None
            self._strategy_cache = None

    def _create_dag_cache_key(self, dag: nx.DiGraph, treatment: str, outcome: str, operation: str) -> Dict:
        """
        Create cache key for DAG-based operations.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable
            operation: Operation name

        Returns:
            Cache key dict
        """
        # Create stable representation of DAG structure
        nodes = sorted(dag.nodes())
        edges = sorted(dag.edges())

        return {
            "operation": operation,
            "nodes": nodes,
            "edges": edges,
            "treatment": treatment,
            "outcome": outcome,
        }

    def suggest_adjustment_strategies(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[AdjustmentStrategy]:
        """
        Suggest complete adjustment strategies.

        Implements graceful degradation:
        - If complex analysis fails, falls back to simple strategies
        - Uses circuit breaker to prevent repeated expensive failures
        - Always returns at least basic strategies

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of adjustment strategies ranked by expected success
        """
        # Check cache first
        if self.enable_caching and self._strategy_cache is not None:
            cache_key = self._create_dag_cache_key(dag, treatment, outcome, "strategies")
            cached_result = self._strategy_cache.get(cache_key)
            if cached_result is not None:
                logger.info(
                    "strategy_cache_hit",
                    extra={
                        "treatment": treatment,
                        "outcome": outcome,
                        "n_strategies": len(cached_result),
                    }
                )
                return cached_result

        logger.info(
            "generating_adjustment_strategies",
            extra={
                "treatment": treatment,
                "outcome": outcome,
                "n_nodes": len(dag.nodes()),
                "n_edges": len(dag.edges()),
            }
        )

        try:
            # Use circuit breaker for expensive strategy generation
            strategies = _strategy_generation_breaker.call(
                self._generate_strategies_internal,
                dag, treatment, outcome
            )

            health_monitor.record_success("validation_suggester")

            # Cache the result
            if self.enable_caching and self._strategy_cache is not None:
                cache_key = self._create_dag_cache_key(dag, treatment, outcome, "strategies")
                self._strategy_cache.put(cache_key, strategies)

            return strategies

        except Exception as e:
            # Fall back to simple strategies
            logger.warning(
                "strategy_generation_failed_fallback",
                extra={
                    "error": str(e),
                    "circuit_state": _strategy_generation_breaker.state.value,
                },
                exc_info=True
            )
            health_monitor.record_fallback("validation_suggester")
            return self._fallback_to_simple_strategies(dag, treatment, outcome)

    def _generate_strategies_internal(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[AdjustmentStrategy]:
        """
        Internal method for strategy generation (wrapped by circuit breaker).

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of ranked adjustment strategies
        """
        strategies = []

        # 1. Backdoor adjustment strategies
        backdoor_strategies = self._find_backdoor_strategies(dag, treatment, outcome)
        logger.debug(f"Found {len(backdoor_strategies)} backdoor strategies")
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
        logger.debug(f"Found {len(iv_strategies)} instrumental variable strategies")
        strategies.extend(iv_strategies)

        # Rank by complexity and identifiability confidence
        ranked_strategies = self._rank_strategies(strategies)

        logger.info(
            "strategies_generated",
            extra={
                "n_strategies": len(ranked_strategies),
                "strategy_types": [s.type for s in ranked_strategies],
                "top_identifiability": ranked_strategies[0].expected_identifiability if ranked_strategies else 0,
            }
        )

        return ranked_strategies

    def analyze_paths(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> PathAnalysis:
        """
        Comprehensive path analysis.

        Implements graceful degradation:
        - If complex path analysis fails, falls back to simple path finding
        - Uses circuit breaker to prevent repeated expensive failures
        - Always returns at least basic path information

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            PathAnalysis with all relevant paths
        """
        try:
            # Use circuit breaker for expensive path analysis
            path_analysis = _path_analysis_breaker.call(
                self._analyze_paths_internal,
                dag, treatment, outcome
            )
            health_monitor.record_success("path_analysis")
            return path_analysis

        except Exception as e:
            # Fall back to simple path analysis
            logger.warning(
                "path_analysis_failed_fallback",
                extra={
                    "error": str(e),
                    "circuit_state": _path_analysis_breaker.state.value,
                },
                exc_info=True
            )
            health_monitor.record_fallback("path_analysis")
            return self._fallback_path_analysis(dag, treatment, outcome)

    def _analyze_paths_internal(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> PathAnalysis:
        """
        Internal method for path analysis (wrapped by circuit breaker).

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            Complete PathAnalysis
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

        A valid instrument must satisfy:
        1. Relevance: Instrument affects treatment
        2. Exclusion: Instrument affects outcome ONLY through treatment
        3. Independence: Instrument is independent of confounders (not checked here)

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of IV strategies
        """
        strategies = []

        # Look for potential instruments
        for node in dag.nodes():
            if node in [treatment, outcome]:
                continue

            # Check relevance: node affects treatment
            affects_treatment = nx.has_path(dag, node, treatment)

            if not affects_treatment:
                continue

            # Check exclusion restriction: all paths from node to outcome go through treatment
            satisfies_exclusion = self._check_exclusion_restriction(
                dag, node, treatment, outcome
            )

            if affects_treatment and satisfies_exclusion:
                # Valid instrument
                strategy = AdjustmentStrategy(
                    strategy_type="instrumental",
                    nodes_to_add=[node] if node not in dag.nodes() else [],
                    edges_to_add=[(node, treatment)] if not dag.has_edge(node, treatment) else [],
                    explanation=f"Use {node} as instrumental variable (satisfies exclusion restriction)",
                    theoretical_basis="Instrumental variables identification",
                    expected_identifiability=0.7,  # Higher confidence for valid IV
                )
                strategies.append(strategy)

        return strategies[:2]  # Return top 2 IV strategies

    def _check_exclusion_restriction(
        self, dag: nx.DiGraph, instrument: str, treatment: str, outcome: str
    ) -> bool:
        """
        Check if instrument satisfies exclusion restriction.

        The exclusion restriction requires that all paths from instrument to outcome
        pass through treatment.

        Args:
            dag: NetworkX DiGraph
            instrument: Potential instrument variable
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            True if exclusion restriction is satisfied
        """
        if not nx.has_path(dag, instrument, outcome):
            # No path to outcome - exclusion satisfied trivially
            return True

        # Find all paths from instrument to outcome
        try:
            all_paths = list(nx.all_simple_paths(dag, instrument, outcome, cutoff=MAX_PATH_DEPTH))
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return True

        # Check that all paths go through treatment
        for path in all_paths:
            if treatment not in path:
                # Found a path that doesn't go through treatment - exclusion violated
                logger.debug(
                    f"Instrument {instrument} violates exclusion: "
                    f"path {' -> '.join(path)} doesn't go through {treatment}"
                )
                return False

        return True

    def _find_backdoor_paths(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[List[str]]:
        """
        Find all backdoor paths from treatment to outcome.

        A backdoor path is a path that:
        1. Starts with an arrow INTO treatment (parent -> treatment)
        2. Reaches outcome
        3. Does NOT traverse any descendant edge OUT of treatment

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List of backdoor paths (each path is a list of nodes)
        """
        # Check cache first
        if self.enable_caching and self._path_cache is not None:
            cache_key = self._create_dag_cache_key(dag, treatment, outcome, "backdoor_paths")
            cached_result = self._path_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(
                    "backdoor_path_cache_hit",
                    extra={
                        "treatment": treatment,
                        "outcome": outcome,
                        "n_paths": len(cached_result),
                    }
                )
                return cached_result

        backdoor_paths = []

        if treatment not in dag.nodes() or outcome not in dag.nodes():
            return backdoor_paths

        # Find all nodes that point to treatment (confounders)
        treatment_parents = list(dag.predecessors(treatment))

        if not treatment_parents:
            return backdoor_paths

        # For each parent, find paths to outcome
        for parent in treatment_parents:
            # Use DFS to find paths that respect edge directions
            visited = set()
            current_path = [parent]
            self._dfs_backdoor_paths(
                dag, parent, outcome, treatment, current_path, visited, backdoor_paths
            )

        # Cache the result
        if self.enable_caching and self._path_cache is not None:
            cache_key = self._create_dag_cache_key(dag, treatment, outcome, "backdoor_paths")
            self._path_cache.put(cache_key, backdoor_paths)

        return backdoor_paths

    def _dfs_backdoor_paths(
        self,
        dag: nx.DiGraph,
        current: str,
        target: str,
        treatment: str,
        path: List[str],
        visited: Set[str],
        backdoor_paths: List[List[str]],
        max_depth: int = 10,
    ) -> None:
        """
        DFS helper to find backdoor paths.

        Args:
            dag: NetworkX DiGraph
            current: Current node
            target: Target node (outcome)
            treatment: Treatment variable (to avoid traversing out of)
            path: Current path
            visited: Visited nodes in current path
            backdoor_paths: Accumulated backdoor paths
            max_depth: Maximum path depth
        """
        if len(path) > max_depth:
            return

        if current == target:
            backdoor_paths.append(path.copy())
            return

        if current in visited:
            return

        visited.add(current)

        # Explore both directions (since backdoor paths use undirected edges)
        # But don't exit treatment via a descendant edge
        neighbors = set()

        # Add predecessors (can always traverse backwards)
        neighbors.update(dag.predecessors(current))

        # Add successors (can traverse forward unless it's from treatment)
        if current != treatment:
            neighbors.update(dag.successors(current))

        for neighbor in neighbors:
            if neighbor not in visited:
                path.append(neighbor)
                self._dfs_backdoor_paths(
                    dag, neighbor, target, treatment, path, visited, backdoor_paths, max_depth
                )
                path.pop()

        visited.remove(current)

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

    def _fallback_to_simple_strategies(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[AdjustmentStrategy]:
        """
        Fallback to simple adjustment strategies when complex analysis fails.

        Always succeeds with basic backdoor strategy based on DAG structure.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            List with at least one simple strategy
        """
        logger.info(
            "using_simple_strategy_fallback",
            extra={
                "treatment": treatment,
                "outcome": outcome,
                "reason": "Complex strategy generation failed"
            }
        )

        strategies = []

        try:
            # Simple strategy: suggest controlling for all non-treatment, non-outcome nodes
            # This is a safe (if inefficient) backdoor adjustment
            potential_confounders = [
                node for node in dag.nodes()
                if node not in [treatment, outcome]
            ]

            if potential_confounders:
                # Limit to first 5 nodes to keep it manageable
                confounders_to_control = potential_confounders[:5]

                strategy = AdjustmentStrategy(
                    strategy_type="backdoor",
                    nodes_to_add=[],
                    edges_to_add=[],
                    explanation=f"Control for variables: {', '.join(confounders_to_control)} "
                                f"(simplified backdoor adjustment)",
                    theoretical_basis="Conservative backdoor criterion (controls for all available variables)",
                    expected_identifiability=0.6,  # Lower confidence for simplified approach
                )
                strategies.append(strategy)
            else:
                # No other nodes - suggest basic data collection
                strategy = AdjustmentStrategy(
                    strategy_type="backdoor",
                    nodes_to_add=["confounder"],
                    edges_to_add=[("confounder", treatment), ("confounder", outcome)],
                    explanation="Collect data on potential confounders that affect both treatment and outcome",
                    theoretical_basis="Backdoor criterion",
                    expected_identifiability=0.5,
                )
                strategies.append(strategy)

            logger.info("simple_strategy_fallback_success", extra={"n_strategies": len(strategies)})
            return strategies

        except Exception as e:
            # Ultimate fallback: return minimal suggestion
            logger.error(
                "simple_strategy_fallback_failed",
                extra={"error": str(e)},
                exc_info=True
            )

            # Return absolute minimal strategy
            return [
                AdjustmentStrategy(
                    strategy_type="manual",
                    nodes_to_add=[],
                    edges_to_add=[],
                    explanation="Complex analysis unavailable. Manually identify and control for confounders.",
                    theoretical_basis="Manual identification required",
                    expected_identifiability=0.3,
                )
            ]

    def _fallback_path_analysis(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> PathAnalysis:
        """
        Fallback to simple path analysis when complex analysis fails.

        Always succeeds with basic path information.

        Args:
            dag: NetworkX DiGraph
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            Minimal PathAnalysis with available information
        """
        logger.info(
            "using_simple_path_analysis_fallback",
            extra={
                "treatment": treatment,
                "outcome": outcome,
                "reason": "Complex path analysis failed"
            }
        )

        try:
            # Try simple directed path finding only
            frontdoor = self._find_directed_paths(dag, treatment, outcome)

            # Return minimal analysis with just directed paths
            logger.info("simple_path_analysis_success", extra={"n_paths": len(frontdoor)})
            return PathAnalysis(
                backdoor_paths=[],
                frontdoor_paths=frontdoor,
                blocked_paths=[],
                critical_nodes=[],
            )

        except Exception as e:
            # Ultimate fallback: return empty analysis
            logger.error(
                "simple_path_analysis_failed",
                extra={"error": str(e)},
                exc_info=True
            )

            # Return minimal empty analysis (always succeeds)
            return PathAnalysis(
                backdoor_paths=[],
                frontdoor_paths=[],
                blocked_paths=[],
                critical_nodes=[],
            )

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
