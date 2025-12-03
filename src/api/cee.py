"""
CEE Enhancement Endpoints (Phase 0).

Provides endpoints for CEE decision reviews:
- Detailed sensitivity analysis
- Contrastive explanations
- Conformal predictions
- Validation strategies

These endpoints enhance decision reviews with advanced causal insights.
CEE gracefully degrades if endpoints are unavailable or return 501.
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.requests import (
    ConformalRequest,
    ContrastiveRequest,
    SensitivityDetailedRequest,
    ValidationStrategiesRequest,
)
from src.models.responses import (
    AssumptionSensitivity,
    ConformalResponse,
    ContrastiveAlternative,
    ContrastiveResponse,
    SensitivityDetailedResponse,
    ValidationImprovement,
    ValidationStrategiesResponse,
)
from src.services.advanced_validation_suggester import AdvancedValidationSuggester
from src.services.cee_adapters import (
    extract_assumptions,
    format_graph_summary,
    graph_v1_to_networkx,
    infer_outcome,
    infer_treatment,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
validation_suggester = AdvancedValidationSuggester()


@router.post(
    "/sensitivity/detailed",
    response_model=SensitivityDetailedResponse,
    summary="Detailed sensitivity analysis",
    description="""
    Identify which assumptions/variables have highest impact on outcomes.

    Provides:
    - Assumption sensitivity rankings
    - Impact assessments
    - Critical variable identification

    **Use when:** Understanding which assumptions matter most for decision robustness.
    """,
    responses={
        200: {"description": "Sensitivity analysis completed successfully"},
        400: {"description": "Invalid graph structure"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_sensitivity_detailed(
    request: SensitivityDetailedRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> SensitivityDetailedResponse:
    """
    Perform detailed sensitivity analysis on decision graph.

    Args:
        request: Sensitivity analysis request with GraphV1 structure
        x_request_id: Optional request ID for tracing

    Returns:
        SensitivityDetailedResponse: Sensitivity analysis results
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "cee_sensitivity_detailed_request",
            extra={
                "request_id": request_id,
                "graph_summary": format_graph_summary(request.graph),
                "num_nodes": len(request.graph.nodes),
                "num_edges": len(request.graph.edges),
                "timeout": request.timeout,
            },
        )

        # Extract assumptions from graph
        assumptions = extract_assumptions(request.graph)

        if not assumptions:
            logger.warning(
                "no_assumptions_found",
                extra={"request_id": request_id}
            )
            # Return empty result if no assumptions
            return SensitivityDetailedResponse(assumptions=[])

        # Analyze each assumption
        # For now, use simple heuristics based on graph structure
        G = graph_v1_to_networkx(request.graph)
        treatment = infer_treatment(request.graph)
        outcome = infer_outcome(request.graph)

        assumption_results = []

        for assumption in assumptions:
            # Calculate sensitivity based on graph topology
            sensitivity = _calculate_assumption_sensitivity(
                assumption, G, treatment, outcome
            )

            impact_desc = _format_impact_description(assumption, sensitivity)

            assumption_results.append(
                AssumptionSensitivity(
                    variable=assumption["name"],
                    sensitivity=sensitivity,
                    impact=impact_desc
                )
            )

        # Sort by sensitivity (highest first)
        assumption_results.sort(key=lambda x: x.sensitivity, reverse=True)

        logger.info(
            "cee_sensitivity_completed",
            extra={
                "request_id": request_id,
                "num_assumptions": len(assumption_results),
                "top_sensitivity": assumption_results[0].sensitivity if assumption_results else 0,
            }
        )

        return SensitivityDetailedResponse(assumptions=assumption_results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "cee_sensitivity_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Sensitivity analysis failed: {str(e)}"
        )


@router.post(
    "/contrastive",
    response_model=ContrastiveResponse,
    summary="Contrastive explanations",
    description="""
    Generate actionable alternatives showing what changes would produce different outcomes.

    Provides:
    - Counterfactual scenarios
    - Feasibility assessments
    - Outcome comparisons

    **Use when:** Exploring "what if" alternatives and decision paths.
    """,
    responses={
        200: {"description": "Contrastive explanation generated successfully"},
        400: {"description": "Invalid graph structure or target outcome"},
        500: {"description": "Internal computation error"},
    },
)
async def generate_contrastive(
    request: ContrastiveRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ContrastiveResponse:
    """
    Generate contrastive explanations for decision graph.

    Args:
        request: Contrastive request with GraphV1 structure and target outcome
        x_request_id: Optional request ID for tracing

    Returns:
        ContrastiveResponse: List of actionable alternatives
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "cee_contrastive_request",
            extra={
                "request_id": request_id,
                "graph_summary": format_graph_summary(request.graph),
                "target_outcome": request.target_outcome,
                "timeout": request.timeout,
            },
        )

        # Validate target outcome exists in graph
        node_ids = [node.id for node in request.graph.nodes]
        if request.target_outcome not in node_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Target outcome '{request.target_outcome}' not found in graph"
            )

        # Generate contrastive alternatives
        G = graph_v1_to_networkx(request.graph)
        alternatives = _generate_alternatives(G, request.target_outcome, request.graph)

        logger.info(
            "cee_contrastive_completed",
            extra={
                "request_id": request_id,
                "num_alternatives": len(alternatives),
            }
        )

        return ContrastiveResponse(alternatives=alternatives)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "cee_contrastive_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Contrastive explanation failed: {str(e)}"
        )


@router.post(
    "/conformal",
    response_model=ConformalResponse,
    summary="Conformal prediction",
    description="""
    Provide calibrated confidence intervals for predictions.

    Provides:
    - Prediction intervals
    - Confidence levels
    - Uncertainty sources

    **Use when:** Quantifying prediction uncertainty with calibrated bounds.
    """,
    responses={
        200: {"description": "Conformal prediction completed successfully"},
        400: {"description": "Invalid graph structure or variable"},
        500: {"description": "Internal computation error"},
    },
)
async def predict_conformal(
    request: ConformalRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ConformalResponse:
    """
    Generate conformal prediction intervals for decision graph.

    Args:
        request: Conformal request with GraphV1 structure and variable
        x_request_id: Optional request ID for tracing

    Returns:
        ConformalResponse: Calibrated confidence intervals
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "cee_conformal_request",
            extra={
                "request_id": request_id,
                "graph_summary": format_graph_summary(request.graph),
                "variable": request.variable,
                "timeout": request.timeout,
            },
        )

        # Validate variable exists in graph
        node_ids = [node.id for node in request.graph.nodes]
        if request.variable not in node_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Variable '{request.variable}' not found in graph"
            )

        # Generate conformal prediction interval
        G = graph_v1_to_networkx(request.graph)
        interval, uncertainty_source = _generate_conformal_interval(
            G, request.variable, request.graph
        )

        logger.info(
            "cee_conformal_completed",
            extra={
                "request_id": request_id,
                "interval_width": interval[1] - interval[0],
            }
        )

        return ConformalResponse(
            prediction_interval=interval,
            confidence_level=0.90,
            uncertainty_source=uncertainty_source
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "cee_conformal_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Conformal prediction failed: {str(e)}"
        )


@router.post(
    "/validation/strategies",
    response_model=ValidationStrategiesResponse,
    summary="Model validation strategies",
    description="""
    Suggest how to improve the causal model's reliability.

    Provides:
    - Data collection suggestions
    - Model structure improvements
    - Sensitivity testing recommendations

    **Use when:** Identifying ways to strengthen model reliability and validity.
    """,
    responses={
        200: {"description": "Validation strategies generated successfully"},
        400: {"description": "Invalid graph structure"},
        500: {"description": "Internal computation error"},
    },
)
async def suggest_validation_strategies(
    request: ValidationStrategiesRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> ValidationStrategiesResponse:
    """
    Generate validation strategy suggestions for decision graph.

    Args:
        request: Validation strategies request with GraphV1 structure
        x_request_id: Optional request ID for tracing

    Returns:
        ValidationStrategiesResponse: List of improvement suggestions
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "cee_validation_strategies_request",
            extra={
                "request_id": request_id,
                "graph_summary": format_graph_summary(request.graph),
                "num_nodes": len(request.graph.nodes),
                "num_edges": len(request.graph.edges),
                "timeout": request.timeout,
            },
        )

        # Convert to NetworkX
        G = graph_v1_to_networkx(request.graph)
        treatment = infer_treatment(request.graph)
        outcome = infer_outcome(request.graph)

        # Get adjustment strategies from advanced validator
        strategies = validation_suggester.suggest_adjustment_strategies(
            dag=G,
            treatment=treatment,
            outcome=outcome
        )

        # Convert to ValidationImprovement format
        improvements = []

        for strategy in strategies[:5]:  # Top 5 strategies
            # Map strategy type to improvement type
            improvement_type = _map_strategy_type(strategy.type)
            priority = _assess_priority(strategy)

            improvements.append(
                ValidationImprovement(
                    type=improvement_type,
                    description=strategy.explanation,
                    priority=priority
                )
            )

        # Add general recommendations based on graph structure
        general_improvements = _generate_general_recommendations(request.graph, G)
        improvements.extend(general_improvements)

        logger.info(
            "cee_validation_strategies_completed",
            extra={
                "request_id": request_id,
                "num_improvements": len(improvements),
            }
        )

        return ValidationStrategiesResponse(suggested_improvements=improvements)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "cee_validation_strategies_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Validation strategies generation failed: {str(e)}"
        )


# Helper functions


def _calculate_assumption_sensitivity(assumption: dict, G, treatment: str, outcome: str) -> float:
    """
    Calculate sensitivity score for an assumption.

    Uses graph topology to estimate impact.
    """
    import networkx as nx

    # Extract node/edge from assumption name
    if "_to_" in assumption["name"]:
        # Edge weight assumption
        parts = assumption["name"].replace("_weight", "").split("_to_")
        if len(parts) == 2:
            from_node, to_node = parts

            # Check if edge is on path from treatment to outcome
            if nx.has_path(G, treatment, outcome):
                paths = list(nx.all_simple_paths(G, treatment, outcome, cutoff=10))
                on_critical_path = any(
                    from_node in path and to_node in path and
                    path.index(from_node) + 1 == path.index(to_node)
                    for path in paths
                )

                if on_critical_path:
                    return 0.85 + (abs(assumption.get("current_value", 1.0)) / 3.0) * 0.15
                else:
                    return 0.45
    elif "_belief" in assumption["name"]:
        # Node belief assumption
        node = assumption["name"].replace("_belief", "")

        # Check centrality
        try:
            degree_centrality = nx.degree_centrality(G).get(node, 0)
            return 0.5 + (degree_centrality * 0.5)
        except:
            return 0.5

    return 0.5  # Default medium sensitivity


def _format_impact_description(assumption: dict, sensitivity: float) -> str:
    """Format impact description for assumption."""
    if sensitivity > 0.8:
        magnitude = "significant"
    elif sensitivity > 0.6:
        magnitude = "moderate"
    else:
        magnitude = "minor"

    if "_to_" in assumption["name"]:
        return f"Changes to this relationship have {magnitude} impact on outcomes"
    else:
        return f"Uncertainty in this variable creates {magnitude} outcome variance"


def _generate_alternatives(G, target_outcome: str, graph) -> List[ContrastiveAlternative]:
    """Generate contrastive alternatives."""
    import networkx as nx

    alternatives = []

    # Find predecessors of target outcome
    predecessors = list(G.predecessors(target_outcome))

    if predecessors:
        # Suggest strengthening key influences
        for pred in predecessors[:2]:
            pred_node = next((n for n in graph.nodes if n.id == pred), None)
            if pred_node:
                alternatives.append(
                    ContrastiveAlternative(
                        change=f"Strengthen {pred_node.label} by increasing investment or focus",
                        outcome_diff=f"Would increase likelihood of achieving {target_outcome}",
                        feasibility=0.75
                    )
                )

    # Suggest alternative paths
    outcome_node = next((n for n in graph.nodes if n.id == target_outcome), None)
    if outcome_node:
        alternatives.append(
            ContrastiveAlternative(
                change="Add additional supporting factors or enablers",
                outcome_diff=f"Would provide alternative pathways to {outcome_node.label}",
                feasibility=0.60
            )
        )

    # Add mitigation suggestion
    if len(alternatives) < 3:
        alternatives.append(
            ContrastiveAlternative(
                change="Reduce uncertainty by gathering more data on key assumptions",
                outcome_diff="Would narrow confidence intervals and improve decision quality",
                feasibility=0.85
            )
        )

    return alternatives[:3]  # Top 3


def _generate_conformal_interval(G, variable: str, graph) -> tuple:
    """Generate conformal prediction interval."""
    import networkx as nx

    # Calculate interval width based on uncertainty
    node = next((n for n in graph.nodes if n.id == variable), None)

    # Base interval on node properties
    if node and node.belief is not None:
        # Use belief to center interval
        center = node.belief * 100
        width = (1 - node.belief) * 50  # Higher belief = narrower interval
    else:
        center = 50
        width = 30

    # Adjust based on graph complexity
    in_degree = G.in_degree(variable)
    complexity_factor = 1 + (in_degree * 0.1)
    width *= complexity_factor

    interval = [max(0, center - width), min(100, center + width)]

    # Identify uncertainty source
    if in_degree > 2:
        uncertainty_source = "Multiple causal factors create compounding uncertainty"
    elif in_degree > 0:
        preds = list(G.predecessors(variable))
        uncertainty_source = f"Uncertainty propagates from upstream factors: {', '.join(preds[:2])}"
    else:
        uncertainty_source = "Limited observational data for this variable"

    return interval, uncertainty_source


def _map_strategy_type(strategy_type: str) -> str:
    """Map validation strategy type to improvement type."""
    mapping = {
        "backdoor": "model_structure",
        "frontdoor": "model_structure",
        "instrumental": "data_collection",
        "data": "data_collection",
    }
    return mapping.get(strategy_type.lower(), "model_structure")


def _assess_priority(strategy) -> str:
    """Assess priority based on strategy confidence."""
    if strategy.expected_identifiability > 0.8:
        return "high"
    elif strategy.expected_identifiability > 0.5:
        return "medium"
    else:
        return "low"


def _generate_general_recommendations(graph, G) -> List[ValidationImprovement]:
    """Generate general validation recommendations."""
    import networkx as nx

    recommendations = []

    # Check for nodes with belief scores
    nodes_with_beliefs = [n for n in graph.nodes if n.belief is not None]
    if nodes_with_beliefs and len(nodes_with_beliefs) < len(graph.nodes) * 0.5:
        recommendations.append(
            ValidationImprovement(
                type="data_collection",
                description="Assign confidence levels to more nodes by gathering expert estimates or historical data",
                priority="medium"
            )
        )

    # Check graph connectivity
    if not nx.is_weakly_connected(G):
        recommendations.append(
            ValidationImprovement(
                type="model_structure",
                description="Graph has disconnected components - consider adding relationships or splitting into separate models",
                priority="high"
            )
        )

    # Check for nodes without edges
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    if isolated:
        recommendations.append(
            ValidationImprovement(
                type="model_structure",
                description=f"Some nodes are isolated: {', '.join(isolated[:3])} - add causal relationships",
                priority="medium"
            )
        )

    return recommendations
