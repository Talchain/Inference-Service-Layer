"""
Y₀ Identifiability Analysis API Endpoints.

Provides endpoints for computing causal effect identifiability
and enforcing the hard rule for non-identifiable effects.

Key features:
- Primary endpoint for identifiability analysis
- Alternative simple DAG format endpoint
- Integration with existing analysis endpoints
- Hard rule enforcement: non-identifiable → exploratory recommendations
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.models.isl_metadata import MetadataBuilder
from src.models.requests import IdentifiabilityFromDAGRequest, IdentifiabilityRequest
from src.models.responses import (
    ConcernSeverityEnum,
    ConcernTypeEnum,
    IdentifiabilityConcern as IdentifiabilityConcernResponse,
    IdentifiabilityInfo,
    IdentifiabilityResponse,
    IdentifiabilitySuggestionResponse,
    IdentificationMethodEnum,
    RecommendationStatusEnum,
)
from src.models.shared import ConfidenceLevel
from src.services.identifiability_analyzer import (
    ConcernSeverity,
    ConcernType,
    IdentifiabilityAnalyzer,
    IdentifiabilityConcern,
    IdentificationMethod,
    RecommendationStatus,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize service
identifiability_analyzer = IdentifiabilityAnalyzer()


def _convert_method(method: Optional[IdentificationMethod]) -> Optional[IdentificationMethodEnum]:
    """Convert service method enum to response enum."""
    if method is None:
        return None
    return IdentificationMethodEnum(method.value)


def _convert_status(status: RecommendationStatus) -> RecommendationStatusEnum:
    """Convert service status enum to response enum."""
    return RecommendationStatusEnum(status.value)


def _convert_confidence(confidence: ConfidenceLevel) -> ConfidenceLevel:
    """Pass through confidence level."""
    return confidence


def _convert_concern_type(concern_type: ConcernType) -> ConcernTypeEnum:
    """Convert service concern type to response enum."""
    return ConcernTypeEnum(concern_type.value)


def _convert_concern_severity(severity: ConcernSeverity) -> ConcernSeverityEnum:
    """Convert service concern severity to response enum."""
    return ConcernSeverityEnum(severity.value)


def _convert_concerns(
    concerns: Optional[list]
) -> Optional[list]:
    """Convert service concerns to response concerns."""
    if not concerns:
        return None
    return [
        IdentifiabilityConcernResponse(
            type=_convert_concern_type(c.type),
            severity=_convert_concern_severity(c.severity),
            description=c.description,
            affected_nodes=c.affected_nodes,
            affected_paths=c.affected_paths,
        )
        for c in concerns
    ]


@router.post(
    "/identifiability",
    response_model=IdentifiabilityResponse,
    summary="Analyze causal effect identifiability (Y₀)",
    description="""
    Determine whether a causal effect from decision to goal is identifiable
    from observational data given the graph structure.

    **Y₀ Algorithm:**
    Uses Pearl's complete identification algorithm to determine if P(Y | do(X))
    can be computed from observational data P(V) and the graph structure G.

    **Identification Methods:**
    - **Backdoor criterion**: Block all backdoor paths by conditioning on appropriate variables
    - **Frontdoor criterion**: Use mediating variables when backdoor is blocked
    - **Instrumental variables**: Use variables that affect X but not Y directly
    - **Do-calculus**: General graphical rules for effect identification

    **Hard Rule (Neil's Specification):**
    If the main decision→goal effect is non-identifiable:
    - `recommendation_status` is set to "exploratory" (not "actionable")
    - `recommendation_caveat` explains that conclusions should be treated as hypotheses
    - `suggestions` provide guidance on making the effect identifiable

    **Returns:**
    - Identifiability status and method
    - Adjustment set (variables to condition on)
    - Backdoor paths in the graph
    - Suggestions for non-identifiable effects
    - Hard rule enforcement via recommendation_status

    **Use when:**
    - Before presenting recommendations to users
    - When building causal decision models
    - When uncertain about effect identifiability
    """,
    responses={
        200: {"description": "Identifiability analysis completed successfully"},
        400: {"description": "Invalid input (e.g., missing decision or goal node)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_identifiability(
    request: IdentifiabilityRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> IdentifiabilityResponse:
    """
    Analyze causal effect identifiability using Y₀ algorithm.

    Args:
        request: Identifiability request with GraphV1 structure
        x_request_id: Optional request ID for tracing

    Returns:
        IdentifiabilityResponse with analysis results and hard rule status
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"
    metadata_builder = MetadataBuilder(request_id)

    try:
        logger.info(
            "identifiability_analysis_request",
            extra={
                "request_id": request_id,
                "num_nodes": len(request.graph.nodes),
                "num_edges": len(request.graph.edges),
                "decision_override": request.decision_node_id,
                "goal_override": request.goal_node_id,
            },
        )

        # Perform analysis
        result = identifiability_analyzer.analyze(
            graph=request.graph,
            decision_node_id=request.decision_node_id,
            goal_node_id=request.goal_node_id,
        )

        # Convert suggestions
        suggestions = None
        if result.suggestions:
            suggestions = [
                IdentifiabilitySuggestionResponse(
                    description=s.description,
                    variable_to_add=s.variable_to_add,
                    edges_to_add=s.edges_to_add,
                    priority=s.priority,
                )
                for s in result.suggestions
            ]

        # Convert concerns
        concerns = _convert_concerns(result.concerns)

        # Build response
        response = IdentifiabilityResponse(
            identifiability=IdentifiabilityInfo(
                effect=result.effect,
                identifiable=result.identifiable,
                method=_convert_method(result.method),
                adjustment_set=result.adjustment_set,
                confidence=result.confidence,
                explanation=result.explanation,
                concerns=concerns,
            ),
            recommendation_status=_convert_status(result.recommendation_status),
            recommendation_caveat=result.recommendation_caveat,
            suggestions=suggestions,
            backdoor_paths=result.backdoor_paths,
            concerns=concerns,
            scope="graph_structural",
        )

        # Add metadata
        response.metadata = metadata_builder.build(algorithm="y0_identification")

        logger.info(
            "identifiability_analysis_completed",
            extra={
                "request_id": request_id,
                "identifiable": result.identifiable,
                "method": result.method.value if result.method else None,
                "recommendation_status": result.recommendation_status.value,
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "identifiability_analysis_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform identifiability analysis. Check logs for details.",
        )


@router.post(
    "/identifiability/dag",
    response_model=IdentifiabilityResponse,
    summary="Analyze identifiability from simple DAG",
    description="""
    Alternative endpoint for identifiability analysis using simple DAG format.

    Same functionality as `/identifiability` but accepts a simpler input format
    with explicit node list, edge list, treatment, and outcome.

    **Use when:**
    - Working with simple causal graphs without typed nodes
    - Integrating with systems that don't use GraphV1 format
    """,
    responses={
        200: {"description": "Identifiability analysis completed successfully"},
        400: {"description": "Invalid input (e.g., treatment/outcome not in nodes)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_identifiability_from_dag(
    request: IdentifiabilityFromDAGRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> IdentifiabilityResponse:
    """
    Analyze causal effect identifiability from simple DAG format.

    Args:
        request: Identifiability request with simple DAG structure
        x_request_id: Optional request ID for tracing

    Returns:
        IdentifiabilityResponse with analysis results and hard rule status
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"
    metadata_builder = MetadataBuilder(request_id)

    try:
        logger.info(
            "identifiability_dag_analysis_request",
            extra={
                "request_id": request_id,
                "num_nodes": len(request.nodes),
                "num_edges": len(request.edges),
                "treatment": request.treatment,
                "outcome": request.outcome,
            },
        )

        # Perform analysis
        result = identifiability_analyzer.analyze_from_dag(
            nodes=request.nodes,
            edges=request.edges,
            treatment=request.treatment,
            outcome=request.outcome,
        )

        # Convert suggestions
        suggestions = None
        if result.suggestions:
            suggestions = [
                IdentifiabilitySuggestionResponse(
                    description=s.description,
                    variable_to_add=s.variable_to_add,
                    edges_to_add=s.edges_to_add,
                    priority=s.priority,
                )
                for s in result.suggestions
            ]

        # Convert concerns
        concerns = _convert_concerns(result.concerns)

        # Build response
        response = IdentifiabilityResponse(
            identifiability=IdentifiabilityInfo(
                effect=result.effect,
                identifiable=result.identifiable,
                method=_convert_method(result.method),
                adjustment_set=result.adjustment_set,
                confidence=result.confidence,
                explanation=result.explanation,
                concerns=concerns,
            ),
            recommendation_status=_convert_status(result.recommendation_status),
            recommendation_caveat=result.recommendation_caveat,
            suggestions=suggestions,
            backdoor_paths=result.backdoor_paths,
            concerns=concerns,
            scope="graph_structural",
        )

        # Add metadata
        response.metadata = metadata_builder.build(algorithm="y0_identification")

        logger.info(
            "identifiability_dag_analysis_completed",
            extra={
                "request_id": request_id,
                "identifiable": result.identifiable,
                "method": result.method.value if result.method else None,
                "recommendation_status": result.recommendation_status.value,
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "identifiability_dag_analysis_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to perform identifiability analysis. Check logs for details.",
        )
