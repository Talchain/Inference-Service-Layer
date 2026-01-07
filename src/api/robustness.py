"""
Robustness analysis API endpoint.

Provides FACET-based robustness verification for counterfactual recommendations.

Supports two schema versions:
- v1: Legacy single-uncertainty model (fixed edge weights)
- v2: Dual uncertainty model (edge existence + strength distribution)

Response versioning:
- response_version=1 (default): Original response format (backward compatible)
- response_version=2: Enhanced response with status fields, diagnostics, critiques
"""

import logging
import math
import uuid
from typing import Any, Dict, Optional, Union

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.constants import (
    DEFAULT_EXISTS_PROBABILITY_THRESHOLD,
    DEFAULT_RESPONSE_VERSION,
    DEFAULT_STRENGTH_THRESHOLD,
)
from src.models.metadata import create_response_metadata
from src.models.response_v2 import (
    DiagnosticsV2,
    FactorSensitivityV2,
    FragileEdgeV2,
    ISLResponseV2,
    ISLV2Error422,
    OptionResultV2,
    OutcomeDistributionV2,
    RobustnessResultV2,
)
from src.models.robustness import RobustnessRequest, RobustnessResponse
from src.models.robustness_v2 import (
    RobustnessRequestV2,
    RobustnessResponseV2,
    detect_schema_version,
)
from src.services.robustness_analyzer import RobustnessAnalyzer
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.business_metrics import track_robustness_analysis
from src.utils.response_builder import (
    ResponseBuilder,
    build_request_echo,
    determine_option_status,
    hash_node_id,
)
from src.utils.numerical_stability import validate_mc_samples
from src.utils.rng import compute_seed_from_graph
from src.utils.tracing import sanitize_request_id
from src.validation.degenerate_detector import detect_degenerate_outcomes
from src.validation.path_validator import PathValidationConfig
from src.validation.request_validator import RequestValidator

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
robustness_analyzer = RobustnessAnalyzer()
robustness_analyzer_v2 = RobustnessAnalyzerV2()


@router.post(
    "/analyze",
    response_model=RobustnessResponse,
    summary="Analyze robustness of counterfactual recommendation using FACET",
    description="""
    Verifies whether proposed intervention robustly achieves target outcomes
    by exploring intervention space and identifying robust regions.

    **FACET (Region-Based Robustness Analysis)**

    Instead of asking "What if price = £55?", FACET answers:
    - "Which price ranges reliably achieve £95k-£105k revenue?"
    - "How fragile is this recommendation to small changes?"
    - "Do I have flexibility in implementation?"

    **Use this when:**
    - Making high-stakes decisions
    - Need confidence in recommendations
    - Want to understand operating ranges
    - Assessing strategy fragility

    **Returns:**
    - Robust intervention regions (e.g., "price £52-£58 works")
    - Outcome guarantees (e.g., "revenue £95k-£105k")
    - Robustness score (0-1, higher = more robust)
    - Fragility warnings
    - Actionable recommendations

    **Performance:** 1-5 seconds depending on samples and dimensions

    **Example workflow:**
    1. Get counterfactual: "price=£55 → revenue=£100k"
    2. Check robustness: Is this recommendation stable?
    3. Receive: "Any price £52-£58 works" (robust) or "Only £54.8-£55.2 works" (fragile)
    4. Decide: Proceed with confidence or revise strategy
    """,
    responses={
        200: {"description": "Robustness analysis completed successfully"},
        400: {"description": "Invalid input (e.g., malformed model)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_robustness(
    request: RobustnessRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> RobustnessResponse:
    """
    Analyze robustness of counterfactual recommendation using FACET.

    Args:
        request: Robustness analysis request
        x_request_id: Optional request ID for tracing

    Returns:
        RobustnessResponse: Robustness analysis with regions and guarantees
    """
    # Generate request ID if not provided
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "robustness_analysis_request",
            extra={
                "request_id": request_id,
                "intervention_vars": len(request.intervention_proposal),
                "target_outcomes": len(request.target_outcome),
                "samples": request.min_samples,
                "perturbation_radius": request.perturbation_radius,
            },
        )

        # Perform FACET analysis
        analysis = robustness_analyzer.analyze_robustness(
            request=request,
            request_id=request_id,
        )

        # Track metrics
        track_robustness_analysis(
            status=analysis.status,
            robustness_score=analysis.robustness_score,
            is_fragile=analysis.is_fragile,
            regions_found=analysis.region_count,
        )

        # Create response with metadata
        response = RobustnessResponse(
            analysis=analysis,
            metadata=create_response_metadata(request_id),
        )

        logger.info(
            "robustness_analysis_completed",
            extra={
                "request_id": request_id,
                "status": analysis.status,
                "robustness_score": analysis.robustness_score,
                "regions_found": analysis.region_count,
                "is_fragile": analysis.is_fragile,
            },
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("robustness_analysis_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform robustness analysis: {str(e)}",
        )


@router.post(
    "/analyze/v2",
    response_model=None,  # Disable response model - we handle both v1 and v2 formats
    summary="Analyze robustness with dual uncertainty (v2.2 schema)",
    description="""
    Performs robustness analysis with dual uncertainty support:
    - **Structural uncertainty**: Probability that each edge exists (Bernoulli)
    - **Parametric uncertainty**: Distribution over effect magnitude (Normal)

    This enables answering:
    - "Is my decision robust to uncertainty about whether this relationship exists?"
    - "Is my decision robust to the effect being stronger/weaker than estimated?"

    **Input schema (v2.2):**
    - `graph`: Causal graph with nodes and edges
    - `edges`: Each has `exists_probability` and `strength: {mean, std}`
    - `options`: Decision alternatives to compare
    - `goal_node_id`: Target outcome to optimize

    **Output (response_version=1, default):**
    - Outcome distributions per option
    - Win probabilities
    - Sensitivity to edge existence AND magnitude
    - Overall robustness assessment

    **Output (response_version=2):**
    - All of the above, plus:
    - Explicit status fields (analysis_status, robustness_status, factor_sensitivity_status)
    - Structured critiques with severity levels
    - Optional diagnostics (when include_diagnostics=true)
    - Request echo for debugging

    **Performance:** 100-2000ms depending on n_samples (default 1000)
    """,
    responses={
        200: {
            "model": ISLResponseV2,
            "description": "Robustness analysis completed successfully",
        },
        400: {"description": "Invalid input schema"},
        422: {
            "model": ISLV2Error422,
            "description": "Validation error with structured critiques",
        },
        500: {"description": "Internal computation error"},
    },
)
async def analyze_robustness_v2(
    request: RobustnessRequestV2,
    response_version: int = Query(
        default=DEFAULT_RESPONSE_VERSION,
        ge=1,
        le=2,
        description="Response format version (1=legacy, 2=enhanced)",
    ),
    include_diagnostics: bool = Query(
        default=False,
        description="Include detailed diagnostics (V2 only)",
    ),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
    x_isl_response_version: Optional[int] = Header(
        default=None,
        alias="X-ISL-Response-Version",
        description="Alternative way to specify response version via header",
    ),
):
    """
    Analyze robustness with dual uncertainty (v2.2 schema).

    Args:
        request: V2 robustness analysis request with dual uncertainty edges
        response_version: Response format version (1=legacy, 2=enhanced)
        include_diagnostics: Whether to include detailed diagnostics (V2 only)
        x_request_id: Optional request ID for tracing
        x_isl_response_version: Alternative way to specify response version via header

    Returns:
        RobustnessResponseV2 (v1) or ISLResponseV2 (v2): Analysis results
    """
    # Determine response version (header takes precedence)
    version = x_isl_response_version or response_version

    # P2-ISL-2: Request ID handling with sanitization for security
    # Priority: request body > header > generated
    inbound_id = request.request_id or x_request_id
    if inbound_id:
        # Sanitize inbound ID (prevents log injection, header abuse)
        request_id, _ = sanitize_request_id(inbound_id)
    else:
        # Generate ISL-prefixed ID if none provided
        request_id = f"isl-{uuid.uuid4().hex[:12]}"

    # For V1 responses, use the legacy handler
    if version == 1:
        return await _analyze_robustness_v2_legacy(request, request_id)

    # V2 response format with validation and structured output
    return await _analyze_robustness_v2_enhanced(
        request, request_id, include_diagnostics
    )


async def _analyze_robustness_v2_legacy(
    request: RobustnessRequestV2,
    request_id: str,
) -> RobustnessResponseV2:
    """Legacy V1 response handler (backward compatible)."""
    try:
        # Enhanced logging for parameter uncertainty debugging
        param_uncertainties = request.parameter_uncertainties or []
        nodes_with_observed_state = sum(
            1 for n in request.graph.nodes
            if n.observed_state is not None and n.observed_state.value is not None
        )

        logger.info(
            "robustness_v2_analysis_request",
            extra={
                "request_id": request_id,
                "response_version": 1,
                "n_options": len(request.options),
                "n_edges": len(request.graph.edges),
                "n_samples": request.n_samples,
                "goal_node": request.goal_node_id,
                "n_parameter_uncertainties": len(param_uncertainties),
                "parameter_uncertainty_nodes": [u.node_id for u in param_uncertainties],
                "n_nodes_with_observed_state": nodes_with_observed_state,
                "analysis_types": request.analysis_types,
            },
        )

        # Perform v2 analysis
        response = robustness_analyzer_v2.analyze(request)

        # Track metrics
        track_robustness_analysis(
            status="robust" if response.robustness.is_robust else "fragile",
            robustness_score=response.recommendation_confidence,
            is_fragile=not response.robustness.is_robust,
            regions_found=0,
        )

        logger.info(
            "robustness_v2_analysis_completed",
            extra={
                "request_id": request_id,
                "response_version": 1,
                "recommended_option": response.recommended_option_id,
                "confidence": response.recommendation_confidence,
                "is_robust": response.robustness.is_robust,
                "execution_time_ms": response.metadata.execution_time_ms,
                "has_sensitivity": len(response.sensitivity) > 0,
                "n_sensitivity_results": len(response.sensitivity),
                "has_factor_sensitivity": len(response.factor_sensitivity) > 0,
                "n_factor_sensitivity_results": len(response.factor_sensitivity),
                "has_robustness": response.robustness is not None,
                "n_fragile_edges": len(response.robustness.fragile_edges),
            },
        )

        return response

    except ValidationError as e:
        logger.warning(
            "robustness_v2_validation_error",
            extra={"request_id": request_id, "errors": e.errors()},
        )
        raise HTTPException(status_code=422, detail=e.errors())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("robustness_v2_analysis_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform v2 robustness analysis: {str(e)}",
        )


async def _analyze_robustness_v2_enhanced(
    request: RobustnessRequestV2,
    request_id: str,
    include_diagnostics: bool,
) -> JSONResponse:
    """Enhanced V2 response handler with validation and structured output."""
    # Build request echo (no sensitive data)
    request_echo = build_request_echo(
        graph_node_count=len(request.graph.nodes),
        graph_edge_count=len(request.graph.edges),
        options_count=len(request.options),
        goal_node_id=request.goal_node_id,
        n_samples=request.n_samples,
        response_version=2,
        include_diagnostics=include_diagnostics,
    )

    # P2-ISL-1: Compute effective seed (single source of truth)
    # Must match analyzer's logic: request.seed or compute_seed_from_graph()
    effective_seed = (
        request.seed if request.seed is not None else compute_seed_from_graph(request.graph)
    )
    seed_str = str(effective_seed)
    builder = ResponseBuilder(
        request_id=request_id, request_echo=request_echo, seed_used=seed_str
    )

    try:
        # Convert request to dict for validation
        graph_dict = {
            "nodes": [n.model_dump() for n in request.graph.nodes],
            "edges": [e.model_dump(by_alias=True) for e in request.graph.edges],
        }
        options_dict = [o.model_dump() for o in request.options]

        # Validate request structure
        path_config = PathValidationConfig(
            exists_probability_threshold=DEFAULT_EXISTS_PROBABILITY_THRESHOLD,
            strength_threshold=DEFAULT_STRENGTH_THRESHOLD,
        )
        validator = RequestValidator(
            graph=graph_dict,
            options=options_dict,
            goal_node_id=request.goal_node_id,
            path_config=path_config,
        )
        validation = validator.validate()

        builder.add_critiques(validation.critiques)

        # Build diagnostics if requested
        if include_diagnostics:
            nodes_by_id = {n.id: n for n in request.graph.nodes}
            diagnostics = DiagnosticsV2(
                goal_node_id_hash=hash_node_id(request.goal_node_id),
                goal_node_found=request.goal_node_id in nodes_by_id,
                option_diagnostics=validation.option_diagnostics,
                n_samples_requested=request.n_samples,
                n_samples_completed=0,  # Updated after analysis
                identifiability_status="unknown",
                path_exists_probability_threshold=DEFAULT_EXISTS_PROBABILITY_THRESHOLD,
                path_strength_threshold=DEFAULT_STRENGTH_THRESHOLD,
            )
            builder.set_diagnostics(diagnostics)

        if validation.has_blockers:
            # Return 422 with unwrapped ISLV2Error422 (P2-ISL-3)
            logger.warning(
                "robustness_v2_validation_blocked",
                extra={
                    "request_id": request_id,
                    "blocker_count": sum(
                        1 for c in validation.critiques if c.severity == "blocker"
                    ),
                    "blocker_codes": [
                        c.code for c in validation.critiques if c.severity == "blocker"
                    ],
                },
            )
            error_response = builder.build_422_response()
            return JSONResponse(
                status_code=422,
                content=error_response.model_dump(),
                headers={
                    "X-Request-Id": request_id,
                    "X-Processing-Time-Ms": str(builder.get_processing_time_ms()),
                },
            )

        # Log request
        logger.info(
            "robustness_v2_analysis_request",
            extra={
                "request_id": request_id,
                "response_version": 2,
                "n_options": len(request.options),
                "n_edges": len(request.graph.edges),
                "n_samples": request.n_samples,
                "goal_node_hash": hash_node_id(request.goal_node_id),
                "include_diagnostics": include_diagnostics,
            },
        )

        # Run analysis
        v1_response = robustness_analyzer_v2.analyze(request)

        # Convert V1 response to V2 option results
        option_results = []
        for result in v1_response.results:
            dist = result.outcome_distribution

            # P2-ISL-5: Validate and clean MC samples with proper critiques
            n_total = request.n_samples
            if dist.samples is not None and len(dist.samples) > 0:
                # Use numerical stability utility for validation + critique emission
                samples_array = np.array(dist.samples)
                cleaned_samples, sample_critiques = validate_mc_samples(samples_array)

                # Add any numerical stability critiques (once per option set)
                for critique in sample_critiques:
                    # Add option context to critique
                    critique.affected_option_ids = [result.option_id]
                    builder.add_critique(critique)

                # Count valid samples from cleaned array
                n_valid = int(np.sum(np.isfinite(cleaned_samples)))
            else:
                # V1 analyzer doesn't track validity - assume all valid
                n_valid = n_total

            validity_ratio = n_valid / n_total if n_total > 0 else 0.0
            status = determine_option_status(n_valid, n_total)

            option_results.append(
                OptionResultV2(
                    id=result.option_id,
                    label=None,  # V1 doesn't include label in results
                    outcome=OutcomeDistributionV2(
                        mean=dist.mean,
                        std=dist.std,
                        p10=dist.ci_lower,  # Use CI as approximation
                        p50=dist.median,
                        p90=dist.ci_upper,
                        n_samples=n_total,
                        n_valid_samples=n_valid,
                        validity_ratio=validity_ratio,
                    ),
                    win_probability=result.win_probability,
                    probability_of_goal=result.probability_of_goal,
                    status=status,
                    status_reason=(
                        "Numerical issues in sampling"
                        if status != "computed"
                        else None
                    ),
                )
            )

        # Check for degenerate outcomes
        degen_critique = detect_degenerate_outcomes(option_results)
        if degen_critique:
            builder.add_critique(degen_critique)

        # Convert robustness result (include V1 fields for backward compatibility)
        robustness_result = None
        if v1_response.robustness:
            # Map is_robust to level
            if v1_response.robustness.is_robust:
                level = "high" if v1_response.robustness.confidence > 0.8 else "moderate"
            else:
                level = "low" if v1_response.robustness.confidence > 0.5 else "very_low"

            # Build enhanced fragile edges from fragile_edges_enhanced
            fragile_edges_v2 = None
            if v1_response.robustness.fragile_edges_enhanced:
                fragile_edges_v2 = [
                    FragileEdgeV2(
                        edge_id=fe["edge_id"],
                        from_id=fe["from_id"],
                        to_id=fe["to_id"],
                        alternative_winner_id=fe.get("alternative_winner_id"),
                        switch_probability=fe.get("switch_probability"),
                    )
                    for fe in v1_response.robustness.fragile_edges_enhanced
                ]

            robustness_result = RobustnessResultV2(
                # V2 fields
                level=level,
                confidence=v1_response.robustness.confidence,
                # V2 enhanced fragile edges
                fragile_edges=fragile_edges_v2,
                # V1 backward-compatibility fields
                is_robust=v1_response.robustness.is_robust,
                fragile_edges_v1=v1_response.robustness.fragile_edges,
                robust_edges=v1_response.robustness.robust_edges,
                recommendation_stability=v1_response.robustness.recommendation_stability,
            )

        # Convert factor sensitivity
        factor_sensitivity = None
        if v1_response.factor_sensitivity:
            factor_sensitivity = [
                FactorSensitivityV2(
                    node_id=fs.node_id,
                    label=fs.node_label,
                    sensitivity_score=fs.elasticity,
                    direction="positive" if fs.elasticity > 0 else "negative",
                    confidence=0.8,  # V1 doesn't provide per-factor confidence
                )
                for fs in v1_response.factor_sensitivity
            ]

        builder.set_results(
            options=option_results,
            robustness=robustness_result,
            factor_sensitivity=factor_sensitivity,
        )

        # Update diagnostics with sampling info
        if include_diagnostics and builder.diagnostics:
            builder.diagnostics.n_samples_completed = request.n_samples

        # Track metrics
        track_robustness_analysis(
            status="robust" if v1_response.robustness.is_robust else "fragile",
            robustness_score=v1_response.recommendation_confidence,
            is_fragile=not v1_response.robustness.is_robust,
            regions_found=0,
        )

        logger.info(
            "robustness_v2_analysis_completed",
            extra={
                "request_id": request_id,
                "response_version": 2,
                "recommended_option": v1_response.recommended_option_id,
                "confidence": v1_response.recommendation_confidence,
                "critique_count": len(builder.critiques),
                "analysis_status": "computed",
            },
        )

        # Return 200 with response (P2-ISL-2: Add tracing headers)
        response = builder.build()
        return JSONResponse(
            status_code=200,
            # Use by_alias for 'version' field, exclude_none for optional fields like probability_of_goal
            content=response.model_dump(by_alias=True, exclude_none=True),
            headers={
                "X-Request-Id": request_id,
                "X-Processing-Time-Ms": str(response.processing_time_ms),
            },
        )

    except Exception as e:
        logger.exception(f"Analysis failed for request {request_id}: {e}")
        response = builder.build_error_response(e)
        return JSONResponse(
            status_code=500,
            content=response.model_dump(by_alias=True),
            headers={
                "X-Request-Id": request_id,
                "X-Processing-Time-Ms": str(response.processing_time_ms),
            },
        )


@router.post(
    "/analyze/unified",
    summary="Unified robustness analysis (auto-detects schema version)",
    description="""
    Unified endpoint that accepts both v1 and v2.2 request schemas.

    Automatically detects schema version and routes to appropriate analyzer:
    - If request contains `graph` + `options`: Uses v2.2 analyzer
    - If request contains `causal_model`: Uses v1 analyzer

    **Backward compatible** - existing v1 clients continue to work.
    """,
    responses={
        200: {"description": "Analysis completed successfully"},
        400: {"description": "Unknown schema format"},
        422: {"description": "Validation error"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_robustness_unified(
    request: Dict[str, Any],
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> Union[RobustnessResponse, RobustnessResponseV2]:
    """
    Unified robustness analysis endpoint supporting both v1 and v2 schemas.

    Args:
        request: Raw request dict (v1 or v2 format)
        x_request_id: Optional request ID for tracing

    Returns:
        Response in corresponding schema version
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        # Log raw request keys for debugging schema issues
        logger.info(
            "robustness_unified_request_received",
            extra={
                "request_id": request_id,
                "top_level_keys": list(request.keys()),
                "has_graph": "graph" in request,
                "has_options": "options" in request,
                "has_parameter_uncertainties": "parameter_uncertainties" in request,
                "parameter_uncertainties_value": (
                    request.get("parameter_uncertainties")
                    if "parameter_uncertainties" in request
                    else "NOT_PROVIDED"
                ),
            },
        )

        # Detect schema version
        schema_version = detect_schema_version(request)

        if schema_version == "v2":
            # Validate and process v2 request
            validated_request = RobustnessRequestV2(**request)
            validated_request.request_id = (
                validated_request.request_id or request_id
            )
            return await analyze_robustness_v2(
                validated_request,
                x_request_id=request_id,
            )
        else:
            # Validate and process v1 request
            validated_request = RobustnessRequest(**request)
            return await analyze_robustness(
                validated_request,
                x_request_id=request_id,
            )

    except ValueError as e:
        # Schema detection failed
        logger.warning(
            "robustness_unified_schema_error",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        # Log detailed validation errors to help debug schema issues
        errors = e.errors()
        logger.warning(
            "robustness_unified_validation_error",
            extra={
                "request_id": request_id,
                "error_count": len(errors),
                "errors": errors,
                "has_parameter_uncertainties": "parameter_uncertainties" in request,
                "parameter_uncertainties_type": type(request.get("parameter_uncertainties")).__name__,
            },
        )
        raise HTTPException(status_code=422, detail=errors)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("unified_robustness_analysis_error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform robustness analysis: {str(e)}",
        )
