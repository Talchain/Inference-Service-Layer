"""
CEE Enhancement Endpoints (Phase 0).

Three of the four routes here are WITHDRAWN (ROADMAP 2.704) and answer a typed
501: ``/sensitivity/detailed``, ``/contrastive`` and ``/conformal``. They are
retained as instrumented refusals rather than deleted, so mounting this router
later still cannot serve a number. See ``src/api/withdrawn.py``.

``/validation/strategies`` is UNCHANGED and is NOT withdrawn. Note this
explicitly, because the 2026-08-07 triage recorded all four routes as
fabrication traps: re-derived at the bytes for ROADMAP 2.704, that blanket
verdict does not hold for this route. It delegates to a real
``AdvancedValidationSuggester`` over a real NetworkX DAG, passes the
suggester's own explanations through, and adds structural recommendations from
genuine graph checks (weak connectivity, isolated nodes, belief and weight
coverage). It emits only qualitative improvements — type, description,
priority — and publishes no numeric estimate at all, so there is nothing here
for a wiring job to serve as an invented number.

CEE gracefully degrades if endpoints are unavailable or return 501.
"""

import logging
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.withdrawn import (
    ASSERTED_COVERAGE_REASON,
    INVENTED_FEASIBILITY_REASON,
    STAMPED_CONSTANTS_REASON,
    refuse_withdrawn,
)
from src.models.isl_metadata import MetadataBuilder
from src.models.requests import ValidationStrategiesRequest
from src.models.responses import (
    ValidationImprovement,
    ValidationStrategiesResponse,
)
from src.services.advanced_validation_suggester import AdvancedValidationSuggester
from src.services.cee_adapters import (
    format_graph_summary,
    graph_v1_to_networkx,
    infer_outcome,
    infer_treatment,
)

router = APIRouter()
logger = logging.getLogger(__name__)

CEE_SENSITIVITY_DETAILED_ROUTE = "/api/v1/sensitivity/detailed"
CEE_CONTRASTIVE_ROUTE = "/api/v1/contrastive"
CEE_CONFORMAL_ROUTE = "/api/v1/conformal"

# Initialize services
validation_suggester = AdvancedValidationSuggester()


@router.post(
    "/sensitivity/detailed",
    summary="[WITHDRAWN] Detailed assumption sensitivity",
    description="""
    **WITHDRAWN (ROADMAP 2.704) — this route answers 501 and computes nothing.**

    It previously returned per-assumption "sensitivity" values with impact
    descriptions. Those values were hardcoded constants anchored on 0.85/0.45/
    0.5, selected by the assumption's position in the graph topology, and the
    response was stamped `algorithm="topology_sensitivity"` — a name that
    implied a computation that did not exist.
    """,
    responses={
        501: {"description": "Capability withdrawn — output was fabricated, not computed"},
    },
)
async def analyze_sensitivity_detailed(request: Request) -> JSONResponse:
    """Refuse: assumption sensitivity was a stamped constant.

    Takes the raw request rather than a validated model on purpose — no input
    is "valid" for a capability that does not exist.
    """
    return refuse_withdrawn(
        route=CEE_SENSITIVITY_DETAILED_ROUTE,
        reason=STAMPED_CONSTANTS_REASON,
        request=request,
    )


@router.post(
    "/contrastive",
    summary="[WITHDRAWN] Contrastive explanation",
    description="""
    **WITHDRAWN (ROADMAP 2.704) — this route answers 501 and computes nothing.**

    It previously returned "alternatives" with feasibility scores. The prose
    was templated ("Strengthen X by increasing investment or focus") and the
    feasibility numbers were invented constants (0.75 / 0.60 / 0.85).

    **Naming hazard:** this fake `/contrastive` sits beside the REAL
    `explain/contrastive`, whose minimal-intervention search delegates outcome
    simulation to the live counterfactual engine. Do not conflate them; if you
    are looking for contrastive explanation, that is the one.
    """,
    responses={
        501: {"description": "Capability withdrawn — output was fabricated, not computed"},
    },
)
async def generate_contrastive(request: Request) -> JSONResponse:
    """Refuse: alternatives were template prose with invented feasibility.

    Takes the raw request rather than a validated model on purpose — no input
    is "valid" for a capability that does not exist.
    """
    return refuse_withdrawn(
        route=CEE_CONTRASTIVE_ROUTE,
        reason=INVENTED_FEASIBILITY_REASON,
        request=request,
    )


@router.post(
    "/conformal",
    summary="[WITHDRAWN] Conformal prediction interval",
    description="""
    **WITHDRAWN (ROADMAP 2.704) — this route answers 501 and computes nothing.**

    It previously returned a `prediction_interval` with
    `confidence_level=0.90`, stamped `algorithm="conformal_prediction"`.
    No conformal prediction took place: the bounds were
    `belief*100 +/- (1-belief)*50`, widened by `1 + in_degree*0.1`, with a
    fallback of centre 50 / width 30 when the node carried no belief. There was
    no calibration set, no nonconformity score and no quantile — so the 0.90
    coverage was **asserted, not achieved**, which is the specific claim
    conformal prediction exists to make good on.
    """,
    responses={
        501: {"description": "Capability withdrawn — output was fabricated, not computed"},
    },
)
async def predict_conformal(request: Request) -> JSONResponse:
    """Refuse: the interval was arithmetic over belief and in-degree.

    Takes the raw request rather than a validated model on purpose — no input
    is "valid" for a capability that does not exist.
    """
    return refuse_withdrawn(
        route=CEE_CONFORMAL_ROUTE,
        reason=ASSERTED_COVERAGE_REASON,
        request=request,
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

    # Initialize metadata builder
    metadata_builder = MetadataBuilder(request_id)

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
            dag=G, treatment=treatment, outcome=outcome
        )

        # Convert to ValidationImprovement format
        improvements = []

        for strategy in strategies[:5]:  # Top 5 strategies
            # Map strategy type to improvement type
            improvement_type = _map_strategy_type(strategy.type)
            priority = _assess_priority(strategy)

            improvements.append(
                ValidationImprovement(
                    type=improvement_type, description=strategy.explanation, priority=priority
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
            },
        )

        response = ValidationStrategiesResponse(suggested_improvements=improvements)
        response.metadata = metadata_builder.build(algorithm="validation_strategies")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "cee_validation_strategies_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Validation strategies generation failed: {str(e)}"
        )


# Helper functions
#
# Only the helpers used by /validation/strategies survive. The fabricating
# helpers that served the three withdrawn routes are deleted outright:
# _calculate_assumption_sensitivity (0.85/0.45/0.5 constants),
# _format_impact_description, _generate_alternatives (template prose with
# invented feasibility 0.75/0.60/0.85) and _generate_conformal_interval
# (belief/in-degree arithmetic published as a conformal interval). Deleting
# them, rather than leaving them beside a refusing handler, is the point: a
# future wiring job cannot re-wire what is not there.


def _map_strategy_type(strategy_type: str) -> str:
    """Map validation strategy type to improvement type."""
    mapping = {
        "backdoor": "model_structure",
        "frontdoor": "model_structure",
        "instrumental": "data_collection",
        "data": "data_collection",
    }
    return mapping.get(strategy_type.lower(), "model_structure")


def _assess_priority(strategy: Any) -> str:
    """Assess priority based on strategy confidence."""
    if strategy.expected_identifiability > 0.8:
        return "high"
    elif strategy.expected_identifiability > 0.5:
        return "medium"
    else:
        return "low"


def _generate_general_recommendations(graph: Any, G: Any) -> List[ValidationImprovement]:
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
                priority="medium",
            )
        )

    # Check graph connectivity
    if not nx.is_weakly_connected(G):
        recommendations.append(
            ValidationImprovement(
                type="model_structure",
                description="Graph has disconnected components - consider adding relationships or splitting into separate models",
                priority="high",
            )
        )

    # Check for nodes without edges
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    if isolated:
        recommendations.append(
            ValidationImprovement(
                type="model_structure",
                description=f"Some nodes are isolated: {', '.join(isolated[:3])} - add causal relationships",
                priority="medium",
            )
        )

    # Check for missing weights
    edges_with_weights = [e for e in graph.edges if e.weight is not None]
    if len(edges_with_weights) < len(graph.edges) * 0.5:
        recommendations.append(
            ValidationImprovement(
                type="model_structure",
                description="Over 50% of edges lack explicit weights. Call /api/v1/causal/parameter-recommendations for suggested ranges based on causal topology.",
                priority="high",
            )
        )

    # Check for uniform weights (all edges have same value)
    weights = [e.weight for e in edges_with_weights]
    if weights and len(set(weights)) == 1 and len(weights) > 2:
        recommendations.append(
            ValidationImprovement(
                type="model_structure",
                description=f"All {len(weights)} edges have uniform weight ({weights[0]}). Call /api/v1/causal/parameter-recommendations to differentiate based on causal importance.",
                priority="medium",
            )
        )

    return recommendations
