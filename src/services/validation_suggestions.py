"""
Validation suggestion generation for non-identifiable DAGs.

Provides algorithmic, deterministic suggestions for making non-identifiable
causal models identifiable through:
- Adding measured confounders to block backdoor paths
- Adding mediators to enable front-door identification
- Reversing edge directions where theoretically plausible
- Suggesting conditional independence assumptions

All suggestions are purely algorithmic (no LLM calls) and deterministic.
"""

import logging
from typing import List, Optional, Set, Tuple

import networkx as nx

from src.models.responses import (
    ConditionalIndependence,
    SuggestionAction,
    ValidationSuggestion,
)
from src.utils.graph_parser import find_backdoor_paths, get_children, get_parents

logger = logging.getLogger(__name__)


def generate_validation_suggestions(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    all_nodes: List[str],
) -> List[ValidationSuggestion]:
    """
    Generate all applicable validation suggestions for a non-identifiable DAG.

    Args:
        graph: NetworkX directed graph
        treatment: Treatment variable
        outcome: Outcome variable
        all_nodes: All nodes in the graph

    Returns:
        List of validation suggestions, sorted by priority
    """
    suggestions = []

    # Generate backdoor-related suggestions
    suggestions.extend(suggest_backdoor_fixes(graph, treatment, outcome, all_nodes))

    # Generate mediator suggestions
    suggestions.extend(suggest_mediator_additions(graph, treatment, outcome))

    # Generate edge reversal suggestions
    suggestions.extend(suggest_edge_reversals(graph, treatment, outcome))

    # Sort by priority (critical > recommended > optional)
    priority_order = {"critical": 0, "recommended": 1, "optional": 2}
    suggestions.sort(key=lambda s: priority_order.get(s.priority, 3))

    return suggestions


def suggest_backdoor_fixes(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    all_nodes: List[str],
) -> List[ValidationSuggestion]:
    """
    Generate suggestions for blocking backdoor paths.

    Analyzes unblocked backdoor paths and suggests:
    1. Adding measured confounders at divergent nodes
    2. Making conditional independence assumptions

    Args:
        graph: NetworkX directed graph
        treatment: Treatment variable
        outcome: Outcome variable
        all_nodes: All nodes in the graph

    Returns:
        List of suggestions for blocking backdoor paths
    """
    suggestions = []

    # Safety check: ensure treatment and outcome are in graph
    if not graph.has_node(treatment) or not graph.has_node(outcome):
        return suggestions

    # Find all backdoor paths
    backdoor_paths = find_backdoor_paths(graph, treatment, outcome)

    if not backdoor_paths:
        return suggestions

    # Analyze each backdoor path
    confounders_found = set()

    for path in backdoor_paths:
        # Find potential confounders on this path
        # A confounder is a node that has paths to both treatment and outcome
        for node in path:
            if node not in [treatment, outcome]:
                # Check if this node is a common ancestor (confounder)
                if _is_confounder(graph, node, treatment, outcome):
                    confounders_found.add(node)

    # Generate suggestions for each confounder
    for confounder in confounders_found:
        # Check if confounder is already measured (in the graph)
        if confounder in all_nodes:
            # Already in graph - suggest including it in adjustment set
            path_str = " ← ".join(backdoor_paths[0])
            suggestions.append(
                ValidationSuggestion(
                    type="add_confounder",
                    description=f"Include {confounder} in adjustment set to control confounding",
                    technical_detail=(
                        f"Node {confounder} confounds {treatment} and {outcome}. "
                        f"Adjust for {confounder} to block backdoor path: {path_str}"
                    ),
                    priority="critical",
                    action=SuggestionAction(
                        add_edges=[[treatment, outcome]],  # Indicate adjustment needed
                    ),
                )
            )
        else:
            # Confounder not in graph - suggest adding it
            # Infer likely edges based on path structure
            suggested_edges = _infer_confounder_edges(
                graph, confounder, treatment, outcome, backdoor_paths
            )

            path_str = " ← ".join(backdoor_paths[0])
            suggestions.append(
                ValidationSuggestion(
                    type="add_confounder",
                    description=f"Measure {confounder} to control confounding",
                    technical_detail=(
                        f"Add {confounder} node to block backdoor path: {path_str}"
                    ),
                    priority="critical",
                    action=SuggestionAction(
                        add_node=confounder,
                        add_edges=suggested_edges,
                    ),
                )
            )

    # If confounders are complex, suggest conditional independence assumption
    if len(confounders_found) > 2:
        suggestions.append(
            ValidationSuggestion(
                type="add_conditional_independence",
                description=(
                    f"Assume {outcome} is independent of unmeasured confounders "
                    f"given measured variables"
                ),
                technical_detail=(
                    f"Make conditional independence assumption: "
                    f"{outcome} ⊥ Unmeasured | {{Measured Variables}}"
                ),
                priority="recommended",
                action=SuggestionAction(
                    assume_independence=ConditionalIndependence(
                        variable_a=outcome,
                        variable_b="UnmeasuredConfounders",
                        conditioning_set=list(all_nodes),
                    )
                ),
            )
        )

    return suggestions


def suggest_mediator_additions(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> List[ValidationSuggestion]:
    """
    Generate suggestions for adding mediators to enable front-door criterion.

    Analyzes direct treatment→outcome edges and suggests decomposition
    through mediating variables.

    Args:
        graph: NetworkX directed graph
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        List of mediator addition suggestions
    """
    suggestions = []

    # Safety check: ensure treatment and outcome are in graph
    if not graph.has_node(treatment) or not graph.has_node(outcome):
        return suggestions

    # Check if there's a direct edge from treatment to outcome
    if not graph.has_edge(treatment, outcome):
        return suggestions

    # Suggest adding a mediator to enable front-door identification
    # Generate a generic mediator name
    mediator_name = f"{treatment}Mechanism"

    suggestions.append(
        ValidationSuggestion(
            type="add_mediator",
            description=f"Add {mediator_name} to model the causal mechanism",
            technical_detail=(
                f"Replace direct edge {treatment}→{outcome} with "
                f"{treatment}→{mediator_name}→{outcome} to enable front-door criterion"
            ),
            priority="recommended",
            action=SuggestionAction(
                add_node=mediator_name,
                add_edges=[[treatment, mediator_name], [mediator_name, outcome]],
            ),
        )
    )

    # If there are multiple paths, suggest domain-specific mediators
    all_paths = list(nx.all_simple_paths(graph, treatment, outcome, cutoff=5))
    if len(all_paths) > 1:
        suggestions.append(
            ValidationSuggestion(
                type="add_mediator",
                description=(
                    f"Identify specific mediating variables between {treatment} and {outcome}"
                ),
                technical_detail=(
                    f"Multiple pathways exist. Explicitly model intermediate variables "
                    f"to capture the causal mechanism and potentially enable identification."
                ),
                priority="recommended",
                action=SuggestionAction(
                    add_node="<domain-specific-mediator>",
                    add_edges=None,
                ),
            )
        )

    return suggestions


def suggest_edge_reversals(
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
) -> List[ValidationSuggestion]:
    """
    Generate suggestions for reversing edge directions.

    Tests if reversing specific edges would make the effect identifiable.
    Only suggests reversals that are theoretically plausible (not involving
    treatment or outcome directly).

    Args:
        graph: NetworkX directed graph
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        List of edge reversal suggestions
    """
    suggestions = []

    # Safety check: ensure treatment and outcome are in graph
    if not graph.has_node(treatment) or not graph.has_node(outcome):
        return suggestions

    # Get all edges not directly involving treatment or outcome
    candidate_edges = [
        (u, v)
        for u, v in graph.edges()
        if u not in [treatment, outcome] and v not in [treatment, outcome]
    ]

    # Limit to prevent combinatorial explosion
    max_reversals_to_test = min(10, len(candidate_edges))

    for u, v in candidate_edges[:max_reversals_to_test]:
        # Create a copy with reversed edge
        test_graph = graph.copy()
        test_graph.remove_edge(u, v)
        test_graph.add_edge(v, u)

        # Check if this creates a cycle (invalid DAG)
        if not nx.is_directed_acyclic_graph(test_graph):
            continue

        # Check if reversal would help with identification
        # (simplified heuristic: does it reduce backdoor paths?)
        original_backdoor_count = len(find_backdoor_paths(graph, treatment, outcome))
        new_backdoor_count = len(find_backdoor_paths(test_graph, treatment, outcome))

        if new_backdoor_count < original_backdoor_count:
            suggestions.append(
                ValidationSuggestion(
                    type="reverse_edge",
                    description=f"Consider reversing edge {u}→{v} to {v}→{u}",
                    technical_detail=(
                        f"Current direction {u}→{v} contributes to confounding. "
                        f"Reversing to {v}→{u} reduces backdoor paths. "
                        f"Verify this direction is theoretically plausible in your domain."
                    ),
                    priority="optional",
                    action=SuggestionAction(
                        reverse_edge=(u, v),
                    ),
                )
            )

    return suggestions


# ============================================================================
# Helper Functions
# ============================================================================


def _is_confounder(
    graph: nx.DiGraph,
    node: str,
    treatment: str,
    outcome: str,
) -> bool:
    """
    Check if a node is a confounder (common cause of treatment and outcome).

    Args:
        graph: NetworkX directed graph
        node: Node to check
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        True if node has paths to both treatment and outcome
    """
    if node == treatment or node == outcome:
        return False

    has_path_to_treatment = nx.has_path(graph, node, treatment)
    has_path_to_outcome = nx.has_path(graph, node, outcome)

    return has_path_to_treatment and has_path_to_outcome


def _infer_confounder_edges(
    graph: nx.DiGraph,
    confounder: str,
    treatment: str,
    outcome: str,
    backdoor_paths: List[List[str]],
) -> List[Tuple[str, str]]:
    """
    Infer edges for a confounder node based on backdoor path structure.

    Args:
        graph: NetworkX directed graph
        confounder: Confounder node name
        treatment: Treatment variable
        outcome: Outcome variable
        backdoor_paths: List of backdoor paths

    Returns:
        List of (from, to) edge tuples
    """
    edges = []

    # Standard confounder pattern: confounder → treatment, confounder → outcome
    edges.append((confounder, treatment))
    edges.append((confounder, outcome))

    return edges


def _find_common_ancestor(
    graph: nx.DiGraph,
    path: List[str],
    treatment: str,
    outcome: str,
) -> Optional[str]:
    """
    Find the common ancestor (divergent node) in a backdoor path.

    Args:
        graph: NetworkX directed graph
        path: Backdoor path
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        Name of common ancestor node, or None if not found
    """
    # In a backdoor path, the common ancestor is the node where
    # the path diverges (has outgoing edges to both branches)
    for node in path:
        if node not in [treatment, outcome]:
            children = get_children(graph, node)
            # Check if this node has children in both directions
            if len(children) >= 2:
                return node

    return None
