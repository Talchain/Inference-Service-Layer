"""
Parameter Recommendation Service.

Provides weight and belief range recommendations based on causal graph topology.
"""

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.models.responses import ParameterRecommendation
from src.models.shared import GraphEdgeV1, GraphNodeV1, GraphV1
from src.services.cee_adapters import (
    calculate_node_centralities,
    find_critical_path_edges,
    graph_v1_to_networkx,
    identify_node_role,
    infer_outcome,
    infer_treatment,
)

logger = logging.getLogger(__name__)


def recommend_edge_weight(
    edge: GraphEdgeV1,
    is_critical: bool,
    from_centrality: float,
    to_centrality: float,
    graph: GraphV1,
) -> ParameterRecommendation:
    """
    Recommend weight range for an edge based on graph topology.

    Args:
        edge: Graph edge
        is_critical: Whether edge is on critical path
        from_centrality: Centrality of source node
        to_centrality: Centrality of target node
        graph: Full graph structure

    Returns:
        ParameterRecommendation with weight range
    """
    # Determine recommended range based on edge importance
    if is_critical:
        # Critical path edges need strong weights
        recommended_range = [1.2, 1.8]
        rationale = "Critical path edge connecting decision to outcome - requires strong causal influence"
        confidence = "high"
        importance = 0.9
    elif from_centrality > 0.6 or to_centrality > 0.6:
        # High centrality nodes need moderate-strong weights
        recommended_range = [0.8, 1.3]
        rationale = "High centrality in causal network - moderate to strong influence"
        confidence = "medium"
        importance = 0.7
    elif from_centrality > 0.3 or to_centrality > 0.3:
        # Medium centrality
        recommended_range = [0.5, 1.0]
        rationale = "Moderate centrality in causal network - balanced influence"
        confidence = "medium"
        importance = 0.5
    else:
        # Peripheral edges can be weaker
        recommended_range = [0.3, 0.8]
        rationale = "Supporting factor - moderate influence appropriate"
        confidence = "medium"
        importance = 0.4

    # Get node labels for better description
    from_node = next((n for n in graph.nodes if n.id == edge.from_), None)
    to_node = next((n for n in graph.nodes if n.id == edge.to), None)

    if from_node and to_node:
        rationale = f"{rationale} ({from_node.label} → {to_node.label})"

    return ParameterRecommendation(
        parameter=f"{edge.from_}_to_{edge.to}_weight",
        parameter_type="weight",
        current_value=edge.weight,
        recommended_range=recommended_range,
        recommended_typical=sum(recommended_range) / 2,
        rationale=rationale,
        importance=importance,
        confidence=confidence,
    )


def recommend_node_belief(
    node: GraphNodeV1,
    role: str,
    centrality: float,
    graph: GraphV1,
) -> ParameterRecommendation:
    """
    Recommend belief range for a node based on role and position.

    Args:
        node: Graph node
        role: Node role (treatment, outcome, mediator, confounder, other)
        centrality: Node centrality score
        graph: Full graph structure

    Returns:
        ParameterRecommendation with belief range
    """
    # Determine recommended range based on role
    if role in ["treatment", "outcome"]:
        # Core nodes need high certainty
        recommended_range = [0.75, 0.95]
        rationale = f"{role.capitalize()} node - high certainty recommended for robust analysis"
        confidence = "high"
        importance = 0.85
    elif node.kind.value == "risk":
        # Risk nodes are inherently uncertain
        recommended_range = [0.3, 0.6]
        rationale = "Risk factor - moderate uncertainty appropriate"
        confidence = "medium"
        importance = 0.6
    elif role == "mediator":
        # Mediators on causal pathway
        recommended_range = [0.65, 0.85]
        rationale = "Mediator on causal pathway - moderate-high certainty appropriate"
        confidence = "medium"
        importance = 0.7
    elif role == "confounder":
        # Confounders need careful estimation
        recommended_range = [0.7, 0.9]
        rationale = "Potential confounder - high certainty important for bias control"
        confidence = "high"
        importance = 0.75
    elif centrality > 0.5:
        # High centrality supporting nodes
        recommended_range = [0.6, 0.85]
        rationale = "High centrality supporting factor - moderate-high certainty"
        confidence = "medium"
        importance = 0.65
    else:
        # Peripheral supporting factors
        recommended_range = [0.5, 0.75]
        rationale = "Supporting factor - moderate certainty appropriate"
        confidence = "low"
        importance = 0.45

    rationale = f"{rationale} ({node.label})"

    return ParameterRecommendation(
        parameter=f"{node.id}_belief",
        parameter_type="belief",
        current_value=node.belief,
        recommended_range=recommended_range,
        recommended_typical=sum(recommended_range) / 2,
        rationale=rationale,
        importance=importance,
        confidence=confidence,
    )


def generate_parameter_recommendations(
    graph: GraphV1,
    current_parameters: Optional[Dict[str, float]] = None,
) -> Tuple[List[ParameterRecommendation], Dict]:
    """
    Generate parameter recommendations for all edges and nodes.

    Args:
        graph: GraphV1 structure
        current_parameters: Optional current parameter values

    Returns:
        Tuple of (recommendations list, graph_characteristics dict)
    """
    # Convert to NetworkX
    G = graph_v1_to_networkx(graph)
    treatment = infer_treatment(graph)
    outcome = infer_outcome(graph)

    # Analyze topology
    critical_edges = find_critical_path_edges(G, treatment, outcome)
    centralities = calculate_node_centralities(G)

    recommendations = []

    # Generate edge weight recommendations
    for edge in graph.edges:
        is_critical = (edge.from_, edge.to) in critical_edges
        from_centrality = centralities.get(edge.from_, 0.5)
        to_centrality = centralities.get(edge.to, 0.5)

        rec = recommend_edge_weight(
            edge, is_critical, from_centrality, to_centrality, graph
        )
        recommendations.append(rec)

    # Generate node belief recommendations
    for node in graph.nodes:
        # Only recommend beliefs for nodes that either have beliefs or are key nodes
        if node.belief is not None or node.kind.value in [
            "decision",
            "outcome",
            "goal",
            "risk",
        ]:
            role = identify_node_role(node.id, graph, treatment, outcome)
            centrality = centralities.get(node.id, 0.5)

            rec = recommend_node_belief(node, role, centrality, graph)
            recommendations.append(rec)

    # Sort by importance (descending)
    recommendations.sort(key=lambda x: x.importance, reverse=True)

    # Calculate graph characteristics
    max_path_length = 0
    try:
        if nx.has_path(G, treatment, outcome):
            paths = list(nx.all_simple_paths(G, treatment, outcome, cutoff=10))
            if paths:
                max_path_length = max(len(path) for path in paths)
    except (nx.NetworkXError, nx.NodeNotFound):
        pass

    graph_characteristics = {
        "num_critical_edges": len(critical_edges),
        "max_path_length": max_path_length,
        "avg_centrality": (
            sum(centralities.values()) / len(centralities) if centralities else 0
        ),
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "is_connected": nx.is_weakly_connected(G),
    }

    logger.info(
        "parameter_recommendations_generated",
        extra={
            "num_recommendations": len(recommendations),
            "num_critical_edges": len(critical_edges),
            "max_path_length": max_path_length,
        },
    )

    return recommendations, graph_characteristics
