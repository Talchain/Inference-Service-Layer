"""
Y₀ Identifiability Analyzer Service.

Provides identifiability analysis for causal effects in decision graphs,
implementing the hard rule for non-identifiable effects.

Key features:
- Y₀-powered identifiability checking for Decision → Goal effects
- Backdoor, frontdoor, and instrumental variable method detection
- Adjustment set computation
- Suggestions for non-identifiable cases
- Graph topology caching for efficiency
- Hard rule enforcement: non-identifiable → exploratory recommendations
"""

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from y0.algorithm.identify import identify_outcomes
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from src.models.shared import ConfidenceLevel, GraphV1, NodeKind
from src.utils.cache import TTLCache

logger = logging.getLogger(__name__)

# Cache for identifiability results (keyed by graph topology hash)
# TTL of 1 hour, max 500 entries
_identifiability_cache = TTLCache(max_size=500, ttl=3600, name="identifiability")


class IdentificationMethod(str, Enum):
    """Method used for causal identification."""

    BACKDOOR = "backdoor"
    FRONTDOOR = "frontdoor"
    INSTRUMENTAL = "instrumental"
    DO_CALCULUS = "do_calculus"
    NON_IDENTIFIABLE = "non_identifiable"


class RecommendationStatus(str, Enum):
    """Status of recommendations based on identifiability."""

    ACTIONABLE = "actionable"  # Effect is identifiable, recommendations are reliable
    EXPLORATORY = "exploratory"  # Effect is NOT identifiable, recommendations are uncertain


@dataclass
class IdentifiabilitySuggestion:
    """Suggestion for making an effect identifiable."""

    description: str
    variable_to_add: Optional[str] = None
    edges_to_add: Optional[List[Tuple[str, str]]] = None
    priority: str = "recommended"  # "critical", "recommended", "optional"


@dataclass
class IdentifiabilityResult:
    """Result of identifiability analysis."""

    effect: str  # "decision → goal"
    identifiable: bool
    method: Optional[IdentificationMethod]
    adjustment_set: Optional[List[str]]
    confidence: ConfidenceLevel
    explanation: str
    recommendation_status: RecommendationStatus
    recommendation_caveat: Optional[str]
    suggestions: Optional[List[IdentifiabilitySuggestion]]
    backdoor_paths: Optional[List[str]]


class IdentifiabilityAnalyzer:
    """
    Analyzer for causal effect identifiability using Y₀.

    Implements the hard rule: if the primary decision→goal effect
    is non-identifiable, recommendations must be marked as "exploratory".
    """

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self._cache = _identifiability_cache

    def analyze(
        self,
        graph: GraphV1,
        decision_node_id: Optional[str] = None,
        goal_node_id: Optional[str] = None,
    ) -> IdentifiabilityResult:
        """
        Analyze identifiability of the primary Decision → Goal effect.

        Args:
            graph: GraphV1 structure with nodes and edges
            decision_node_id: Optional override for decision node
            goal_node_id: Optional override for goal node

        Returns:
            IdentifiabilityResult with full analysis
        """
        # Extract decision and goal nodes
        decision, goal = self._extract_decision_goal(
            graph, decision_node_id, goal_node_id
        )

        if not decision or not goal:
            return self._create_no_nodes_result(decision, goal)

        # Compute cache key from graph topology
        cache_key = self._compute_topology_hash(graph, decision, goal)

        # Check cache
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            logger.info(
                "identifiability_cache_hit",
                extra={"decision": decision, "goal": goal},
            )
            return cached_result

        logger.info(
            "identifiability_analysis_started",
            extra={"decision": decision, "goal": goal},
        )

        # Convert to NetworkX graph for analysis
        nx_graph = self._convert_to_networkx(graph)

        # Check if there's a causal path
        if not nx.has_path(nx_graph, decision, goal):
            result = self._create_no_path_result(decision, goal, nx_graph)
            self._cache.put(cache_key, result)
            return result

        # Try Y₀ identification
        try:
            y0_graph = self._convert_to_y0(nx_graph)
            y0_result = identify_outcomes(
                graph=y0_graph,
                treatments={Variable(decision)},
                outcomes={Variable(goal)},
            )

            if y0_result:
                # Effect is identifiable
                result = self._create_identifiable_result(
                    decision, goal, nx_graph, y0_result
                )
            else:
                # Effect is not identifiable
                result = self._create_non_identifiable_result(
                    decision, goal, nx_graph
                )

        except Exception as e:
            logger.warning(f"y0_identification_failed: {e}")
            # Fall back to basic backdoor analysis
            result = self._fallback_analysis(decision, goal, nx_graph)

        # Cache result
        self._cache.put(cache_key, result)

        logger.info(
            "identifiability_analysis_completed",
            extra={
                "decision": decision,
                "goal": goal,
                "identifiable": result.identifiable,
                "method": result.method.value if result.method else None,
                "recommendation_status": result.recommendation_status.value,
            },
        )

        return result

    def analyze_from_dag(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        treatment: str,
        outcome: str,
    ) -> IdentifiabilityResult:
        """
        Analyze identifiability from simple DAG structure.

        Args:
            nodes: List of node names
            edges: List of (from, to) edge tuples
            treatment: Treatment/decision node
            outcome: Outcome/goal node

        Returns:
            IdentifiabilityResult with full analysis
        """
        # Compute cache key
        cache_key = self._compute_dag_topology_hash(nodes, edges, treatment, outcome)

        # Check cache
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            logger.info(
                "identifiability_dag_cache_hit",
                extra={"treatment": treatment, "outcome": outcome},
            )
            return cached_result

        logger.info(
            "identifiability_dag_analysis_started",
            extra={"treatment": treatment, "outcome": outcome},
        )

        # Build NetworkX graph
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)

        # Check if there's a causal path
        if not nx.has_path(nx_graph, treatment, outcome):
            result = self._create_no_path_result(treatment, outcome, nx_graph)
            self._cache.put(cache_key, result)
            return result

        # Try Y₀ identification
        try:
            y0_graph = self._convert_to_y0(nx_graph)
            y0_result = identify_outcomes(
                graph=y0_graph,
                treatments={Variable(treatment)},
                outcomes={Variable(outcome)},
            )

            if y0_result:
                result = self._create_identifiable_result(
                    treatment, outcome, nx_graph, y0_result
                )
            else:
                result = self._create_non_identifiable_result(
                    treatment, outcome, nx_graph
                )

        except Exception as e:
            logger.warning(f"y0_dag_identification_failed: {e}")
            result = self._fallback_analysis(treatment, outcome, nx_graph)

        # Cache result
        self._cache.put(cache_key, result)

        logger.info(
            "identifiability_dag_analysis_completed",
            extra={
                "treatment": treatment,
                "outcome": outcome,
                "identifiable": result.identifiable,
                "method": result.method.value if result.method else None,
            },
        )

        return result

    def _extract_decision_goal(
        self,
        graph: GraphV1,
        decision_override: Optional[str],
        goal_override: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract decision and goal nodes from graph."""
        decision = decision_override
        goal = goal_override

        # Find by node kind if not overridden
        for node in graph.nodes:
            if not decision and node.kind == NodeKind.DECISION:
                decision = node.id
            if not goal and node.kind == NodeKind.GOAL:
                goal = node.id

        return decision, goal

    def _compute_topology_hash(
        self, graph: GraphV1, decision: str, goal: str
    ) -> str:
        """Compute hash of graph topology for caching."""
        # Sort nodes and edges for deterministic hashing
        node_ids = sorted([n.id for n in graph.nodes])
        edges = sorted([(e.from_, e.to) for e in graph.edges])

        topology_str = f"{node_ids}|{edges}|{decision}|{goal}"
        return hashlib.sha256(topology_str.encode()).hexdigest()[:16]

    def _compute_dag_topology_hash(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        treatment: str,
        outcome: str,
    ) -> str:
        """Compute hash for simple DAG structure."""
        sorted_nodes = sorted(nodes)
        sorted_edges = sorted(edges)
        topology_str = f"{sorted_nodes}|{sorted_edges}|{treatment}|{outcome}"
        return hashlib.sha256(topology_str.encode()).hexdigest()[:16]

    def _convert_to_networkx(self, graph: GraphV1) -> nx.DiGraph:
        """Convert GraphV1 to NetworkX graph."""
        nx_graph = nx.DiGraph()
        for node in graph.nodes:
            nx_graph.add_node(node.id, kind=node.kind.value, label=node.label)
        for edge in graph.edges:
            nx_graph.add_edge(edge.from_, edge.to, weight=edge.weight)
        return nx_graph

    def _convert_to_y0(self, nx_graph: nx.DiGraph) -> NxMixedGraph:
        """Convert NetworkX graph to Y₀ format."""
        y0_graph = NxMixedGraph()
        for node in nx_graph.nodes():
            y0_graph.add_node(Variable(node))
        for from_node, to_node in nx_graph.edges():
            y0_graph.add_directed_edge(Variable(from_node), Variable(to_node))
        return y0_graph

    def _find_backdoor_paths(
        self, graph: nx.DiGraph, treatment: str, outcome: str
    ) -> List[List[str]]:
        """Find backdoor paths between treatment and outcome."""
        backdoor_paths = []
        parents = list(graph.predecessors(treatment))

        for parent in parents:
            if nx.has_path(graph, parent, outcome):
                paths = list(nx.all_simple_paths(graph, parent, outcome))
                for path in paths:
                    backdoor_paths.append([treatment] + path)

        return backdoor_paths

    def _find_adjustment_set(
        self, graph: nx.DiGraph, treatment: str, outcome: str
    ) -> Optional[List[str]]:
        """Find a valid adjustment set using backdoor criterion."""
        all_nodes = list(graph.nodes())
        potential_adjusters = [
            n for n in all_nodes if n not in [treatment, outcome]
        ]

        backdoor_paths = self._find_backdoor_paths(graph, treatment, outcome)

        if not backdoor_paths:
            return []  # No backdoor paths, no adjustment needed

        # Try individual nodes
        for node in potential_adjusters:
            if self._blocks_all_backdoors(graph, treatment, outcome, [node]):
                return [node]

        # Try pairs
        if len(potential_adjusters) >= 2:
            for i, n1 in enumerate(potential_adjusters):
                for n2 in potential_adjusters[i + 1 :]:
                    if self._blocks_all_backdoors(graph, treatment, outcome, [n1, n2]):
                        return [n1, n2]

        return None  # No valid adjustment set found

    def _blocks_all_backdoors(
        self,
        graph: nx.DiGraph,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
    ) -> bool:
        """Check if adjustment set blocks all backdoor paths."""
        parents = list(graph.predecessors(treatment))

        for parent in parents:
            temp_graph = graph.copy()
            temp_graph.remove_nodes_from(adjustment_set)

            if temp_graph.has_node(parent) and temp_graph.has_node(outcome):
                if nx.has_path(temp_graph, parent, outcome):
                    return False

        return True

    def _check_frontdoor(
        self, graph: nx.DiGraph, treatment: str, outcome: str
    ) -> Optional[List[str]]:
        """
        Check if frontdoor criterion is applicable.

        Frontdoor requires:
        1. Mediator M that intercepts all directed paths from X to Y
        2. No unblocked backdoor path from X to M
        3. All backdoor paths from M to Y are blocked by X
        """
        # Find mediators (nodes on directed paths from treatment to outcome)
        potential_mediators = []

        for path in nx.all_simple_paths(graph, treatment, outcome):
            for node in path[1:-1]:  # Exclude treatment and outcome
                if node not in potential_mediators:
                    potential_mediators.append(node)

        # Check each mediator for frontdoor criterion
        for mediator in potential_mediators:
            # Check: all paths from treatment to outcome go through mediator
            all_through_mediator = True
            for path in nx.all_simple_paths(graph, treatment, outcome):
                if mediator not in path:
                    all_through_mediator = False
                    break

            if not all_through_mediator:
                continue

            # Check: no backdoor from treatment to mediator
            backdoor_to_mediator = self._find_backdoor_paths(graph, treatment, mediator)
            if backdoor_to_mediator:
                continue

            # Check: X blocks backdoors from M to Y
            if self._blocks_all_backdoors(graph, mediator, outcome, [treatment]):
                return [mediator]

        return None

    def _generate_suggestions(
        self, graph: nx.DiGraph, treatment: str, outcome: str
    ) -> List[IdentifiabilitySuggestion]:
        """Generate suggestions for making effect identifiable."""
        suggestions = []

        # Find backdoor paths
        backdoor_paths = self._find_backdoor_paths(graph, treatment, outcome)

        if backdoor_paths:
            # Find nodes on backdoor paths that could be measured
            nodes_on_backdoors = set()
            for path in backdoor_paths:
                for node in path[1:-1]:  # Exclude treatment
                    nodes_on_backdoors.add(node)

            # Suggest measuring confounders
            for node in nodes_on_backdoors:
                suggestions.append(
                    IdentifiabilitySuggestion(
                        description=(
                            f"If '{node}' were observed as a confounder, "
                            f"conditioning on it might block the backdoor path"
                        ),
                        variable_to_add=f"{node}_observed",
                        priority="recommended",
                    )
                )

            # Suggest adding instrumental variable
            suggestions.append(
                IdentifiabilitySuggestion(
                    description=(
                        f"Adding an instrumental variable that affects '{treatment}' "
                        f"but not '{outcome}' directly could enable identification"
                    ),
                    variable_to_add=f"{treatment}_instrument",
                    edges_to_add=[(f"{treatment}_instrument", treatment)],
                    priority="recommended",
                )
            )

        # Suggest mediator if no direct path
        has_direct_edge = graph.has_edge(treatment, outcome)
        if not has_direct_edge:
            # Check if adding a mediator structure would help
            suggestions.append(
                IdentifiabilitySuggestion(
                    description=(
                        f"If the effect of '{treatment}' on '{outcome}' operates through "
                        f"a fully-observed mediator, frontdoor criterion may apply"
                    ),
                    variable_to_add=f"{treatment}_mechanism",
                    edges_to_add=[
                        (treatment, f"{treatment}_mechanism"),
                        (f"{treatment}_mechanism", outcome),
                    ],
                    priority="optional",
                )
            )

        # General suggestion
        suggestions.append(
            IdentifiabilitySuggestion(
                description=(
                    "Review the causal model structure to identify unmeasured "
                    "confounders that could be measured in practice"
                ),
                priority="critical",
            )
        )

        return suggestions

    def _create_identifiable_result(
        self,
        decision: str,
        goal: str,
        nx_graph: nx.DiGraph,
        y0_result: Any,
    ) -> IdentifiabilityResult:
        """Create result for identifiable effect."""
        # Determine method
        adjustment_set = self._find_adjustment_set(nx_graph, decision, goal)

        if adjustment_set is not None:
            method = IdentificationMethod.BACKDOOR
            if not adjustment_set:
                explanation = (
                    f"The effect of '{decision}' on '{goal}' is identifiable. "
                    f"No adjustment is needed (no confounding)."
                )
            else:
                adjustment_str = ", ".join(adjustment_set)
                explanation = (
                    f"The effect of '{decision}' on '{goal}' is identifiable "
                    f"using the backdoor criterion. Adjust for: {adjustment_str}."
                )
        else:
            # Check frontdoor
            frontdoor_set = self._check_frontdoor(nx_graph, decision, goal)
            if frontdoor_set:
                method = IdentificationMethod.FRONTDOOR
                mediator_str = ", ".join(frontdoor_set)
                explanation = (
                    f"The effect of '{decision}' on '{goal}' is identifiable "
                    f"using the frontdoor criterion through mediator(s): {mediator_str}."
                )
                adjustment_set = frontdoor_set
            else:
                method = IdentificationMethod.DO_CALCULUS
                explanation = (
                    f"The effect of '{decision}' on '{goal}' is identifiable "
                    f"via general do-calculus rules."
                )
                adjustment_set = []

        backdoor_paths = self._find_backdoor_paths(nx_graph, decision, goal)
        formatted_paths = (
            [" → ".join(path) for path in backdoor_paths] if backdoor_paths else None
        )

        return IdentifiabilityResult(
            effect=f"{decision} → {goal}",
            identifiable=True,
            method=method,
            adjustment_set=adjustment_set,
            confidence=ConfidenceLevel.HIGH,
            explanation=explanation,
            recommendation_status=RecommendationStatus.ACTIONABLE,
            recommendation_caveat=None,
            suggestions=None,
            backdoor_paths=formatted_paths,
        )

    def _create_non_identifiable_result(
        self, decision: str, goal: str, nx_graph: nx.DiGraph
    ) -> IdentifiabilityResult:
        """Create result for non-identifiable effect with hard rule enforcement."""
        backdoor_paths = self._find_backdoor_paths(nx_graph, decision, goal)
        formatted_paths = (
            [" → ".join(path) for path in backdoor_paths] if backdoor_paths else None
        )

        suggestions = self._generate_suggestions(nx_graph, decision, goal)

        explanation = (
            f"The effect of '{decision}' on '{goal}' is NOT identifiable from "
            f"the current model structure. There are confounding paths that cannot "
            f"be blocked with the available observed variables."
        )

        # HARD RULE: Non-identifiable effects get exploratory status
        caveat = (
            "Causal effect cannot be confirmed from current model structure. "
            "Recommendations should be treated as exploratory hypotheses, not "
            "actionable conclusions. Consider measuring additional variables or "
            "revising the causal model."
        )

        return IdentifiabilityResult(
            effect=f"{decision} → {goal}",
            identifiable=False,
            method=IdentificationMethod.NON_IDENTIFIABLE,
            adjustment_set=None,
            confidence=ConfidenceLevel.LOW,
            explanation=explanation,
            recommendation_status=RecommendationStatus.EXPLORATORY,
            recommendation_caveat=caveat,
            suggestions=suggestions,
            backdoor_paths=formatted_paths,
        )

    def _create_no_path_result(
        self, decision: str, goal: str, nx_graph: nx.DiGraph
    ) -> IdentifiabilityResult:
        """Create result when no causal path exists."""
        suggestions = [
            IdentifiabilitySuggestion(
                description=(
                    f"Add direct or mediated causal path from '{decision}' to '{goal}'"
                ),
                edges_to_add=[(decision, goal)],
                priority="critical",
            ),
            IdentifiabilitySuggestion(
                description=(
                    f"Verify that '{decision}' actually causally affects '{goal}' "
                    f"in the real-world domain"
                ),
                priority="critical",
            ),
        ]

        return IdentifiabilityResult(
            effect=f"{decision} → {goal}",
            identifiable=False,
            method=IdentificationMethod.NON_IDENTIFIABLE,
            adjustment_set=None,
            confidence=ConfidenceLevel.HIGH,
            explanation=(
                f"No causal path exists from '{decision}' to '{goal}'. "
                f"The decision cannot affect the goal in this model."
            ),
            recommendation_status=RecommendationStatus.EXPLORATORY,
            recommendation_caveat=(
                f"No causal connection exists between decision and goal. "
                f"Any recommendations would be meaningless."
            ),
            suggestions=suggestions,
            backdoor_paths=None,
        )

    def _create_no_nodes_result(
        self, decision: Optional[str], goal: Optional[str]
    ) -> IdentifiabilityResult:
        """Create result when required nodes are missing."""
        missing = []
        if not decision:
            missing.append("decision")
        if not goal:
            missing.append("goal")

        return IdentifiabilityResult(
            effect="? → ?",
            identifiable=False,
            method=None,
            adjustment_set=None,
            confidence=ConfidenceLevel.LOW,
            explanation=(
                f"Cannot analyze identifiability: missing {', '.join(missing)} node(s). "
                f"Graph must contain nodes with kind='decision' and kind='goal'."
            ),
            recommendation_status=RecommendationStatus.EXPLORATORY,
            recommendation_caveat=(
                f"Required node(s) missing: {', '.join(missing)}. "
                f"Cannot compute causal effects without decision and goal nodes."
            ),
            suggestions=[
                IdentifiabilitySuggestion(
                    description=f"Add a node with kind='{m}' to the graph",
                    priority="critical",
                )
                for m in missing
            ],
            backdoor_paths=None,
        )

    def _fallback_analysis(
        self, decision: str, goal: str, nx_graph: nx.DiGraph
    ) -> IdentifiabilityResult:
        """Fallback analysis when Y₀ fails."""
        # Try basic backdoor criterion
        adjustment_set = self._find_adjustment_set(nx_graph, decision, goal)

        if adjustment_set is not None:
            if not adjustment_set:
                return IdentifiabilityResult(
                    effect=f"{decision} → {goal}",
                    identifiable=True,
                    method=IdentificationMethod.BACKDOOR,
                    adjustment_set=[],
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=(
                        f"The effect of '{decision}' on '{goal}' appears identifiable "
                        f"(no confounding detected). Analysis used fallback method."
                    ),
                    recommendation_status=RecommendationStatus.ACTIONABLE,
                    recommendation_caveat=(
                        "Analysis used fallback method. Results should be verified."
                    ),
                    suggestions=None,
                    backdoor_paths=None,
                )
            else:
                adjustment_str = ", ".join(adjustment_set)
                return IdentifiabilityResult(
                    effect=f"{decision} → {goal}",
                    identifiable=True,
                    method=IdentificationMethod.BACKDOOR,
                    adjustment_set=adjustment_set,
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=(
                        f"The effect of '{decision}' on '{goal}' appears identifiable "
                        f"via backdoor adjustment for: {adjustment_str}. "
                        f"Analysis used fallback method."
                    ),
                    recommendation_status=RecommendationStatus.ACTIONABLE,
                    recommendation_caveat=(
                        "Analysis used fallback method. Results should be verified."
                    ),
                    suggestions=None,
                    backdoor_paths=self._find_backdoor_paths(nx_graph, decision, goal)
                    and [
                        " → ".join(p)
                        for p in self._find_backdoor_paths(nx_graph, decision, goal)
                    ],
                )
        else:
            # Cannot identify with backdoor, mark as non-identifiable
            return self._create_non_identifiable_result(decision, goal, nx_graph)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the identifiability cache."""
        self._cache.clear()
