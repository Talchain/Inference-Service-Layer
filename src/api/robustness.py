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
import os
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
    BucketResultV2,
    ConditionalWinnerV2,
    ConstraintAnalysisV2,
    ConstraintResultV2,
    DiagnosticsV2,
    EdgeEValueV2,
    EdgeSensitivityV2,
    FactorSensitivityV2,
    FragileEdgeV2,
    ISLResponseV2,
    ISLV2Error422,
    OptionResultV2,
    OutcomeDistributionV2,
    PathContributionV2,
    PathDecompositionV2,
    RobustnessResultV2,
)
from src.models.robustness import RobustnessRequest, RobustnessResponse
from src.models.robustness_v2 import (
    RobustnessRequestV2,
    RobustnessResponseV2,
    detect_schema_version,
)
from src.services.robustness_analyzer import RobustnessAnalyzer
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2, compute_effective_seed
from src.config.stability_thresholds import (
    compute_factor_confidence,
    compute_graph_structural_confidence,
)
from src.utils.business_metrics import track_robustness_analysis
from src.utils.response_builder import (
    ResponseBuilder,
    build_request_echo,
    determine_option_status,
    hash_node_id,
)
from src.utils.numerical_stability import validate_mc_samples
from src.utils.tracing import sanitize_request_id
from src.validation.degenerate_detector import detect_degenerate_outcomes
from src.validation.path_validator import PathValidationConfig
from src.validation.request_validator import RequestValidator

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request complexity guard (DoS protection)
# ---------------------------------------------------------------------------
# Complexity score = n_samples × n_nodes × n_edges.
# This is a conservative heuristic using the three dominant cost factors.
# Parameter uncertainty count and robustness passes are second-order effects
# at PoC scale and are not included in the formula.
#
# Calibration:
#   Typical pilot graph (5 nodes, 8 edges, 1000 samples)   = 40K  — well within limit
#   Upper PoC bound   (12 nodes, 100 edges, 5000 samples)  = 6M   — within limit
#   Schema max        (50 nodes, 200 edges, 10000 samples)  = 100M — blocked
#
# Override via ISL_MAX_COMPUTE_COMPLEXITY env var (integer).
_DEFAULT_MAX_COMPLEXITY = 10_000_000


def _get_max_complexity() -> int:
    """Read complexity limit from env, falling back to default."""
    val = os.environ.get("ISL_MAX_COMPUTE_COMPLEXITY")
    if val is not None:
        try:
            return int(val)
        except ValueError:
            logger.warning(
                "ISL_MAX_COMPUTE_COMPLEXITY env var is not a valid integer (%s), using default %d",
                val,
                _DEFAULT_MAX_COMPLEXITY,
            )
    return _DEFAULT_MAX_COMPLEXITY


def compute_complexity_score(n_samples: int, n_nodes: int, n_edges: int) -> int:
    """Compute a heuristic complexity score for a robustness request."""
    return n_samples * n_nodes * n_edges


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
) -> Any:
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
    return await _analyze_robustness_v2_enhanced(request, request_id, include_diagnostics)


async def _analyze_robustness_v2_legacy(
    request: RobustnessRequestV2,
    request_id: str,
) -> RobustnessResponseV2:
    """Legacy V1 response handler (backward compatible)."""
    try:
        # Complexity guard — reject oversized requests early (DoS protection)
        n_nodes = len(request.graph.nodes)
        n_edges = len(request.graph.edges)
        complexity = compute_complexity_score(request.n_samples, n_nodes, n_edges)
        max_complexity = _get_max_complexity()
        logger.info(
            "robustness_v2_complexity",
            extra={
                "request_id": request_id,
                "complexity_score": complexity,
                "limit": max_complexity,
                "n_samples": request.n_samples,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
            },
        )
        if complexity > max_complexity:
            raise HTTPException(
                status_code=422,
                detail={
                    "detail": "Request complexity exceeds limit",
                    "complexity_score": complexity,
                    "limit": max_complexity,
                    "suggestion": (
                        f"Reduce n_samples (currently {request.n_samples}), "
                        f"node count (currently {n_nodes}), "
                        f"or edge count (currently {n_edges})"
                    ),
                },
            )

        # Enhanced logging for parameter uncertainty debugging
        param_uncertainties = request.parameter_uncertainties or []
        nodes_with_observed_state = sum(
            1
            for n in request.graph.nodes
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
    except ValueError as e:
        # Analyzer-level input rejections (e.g. GRAPH_CYCLE_DETECTED, goal
        # node filtered out) are client errors, not internal failures.
        # Fail closed with 422 so cyclic graphs never surface as 500s or,
        # worse, as plausible-looking results on this legacy path.
        logger.warning(
            "robustness_v2_invalid_input",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(status_code=422, detail=str(e)) from e
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

    # P2-ISL-1: Compute effective seed (single source of truth).
    # compute_effective_seed is the same derivation the analyzer uses: it
    # hashes the POST-FILTER inference graph, so the reported seed_used is
    # the seed the RNG streams actually consume even when organisational
    # nodes are filtered out before sampling.
    effective_seed, seed_source = compute_effective_seed(request)
    seed_str = str(effective_seed)
    builder = ResponseBuilder(
        request_id=request_id,
        request_echo=request_echo,
        seed_used=seed_str,
        seed_source=seed_source,
    )

    try:
        # Complexity guard — reject oversized requests early (DoS protection)
        n_nodes = len(request.graph.nodes)
        n_edges = len(request.graph.edges)
        complexity = compute_complexity_score(request.n_samples, n_nodes, n_edges)
        max_complexity = _get_max_complexity()
        logger.info(
            "robustness_v2_complexity",
            extra={
                "request_id": request_id,
                "complexity_score": complexity,
                "limit": max_complexity,
                "n_samples": request.n_samples,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
            },
        )
        if complexity > max_complexity:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Request complexity exceeds limit",
                    "complexity_score": complexity,
                    "limit": max_complexity,
                    "suggestion": (
                        f"Reduce n_samples (currently {request.n_samples}), "
                        f"node count (currently {n_nodes}), "
                        f"or edge count (currently {n_edges})"
                    ),
                },
                headers={"X-Request-Id": request_id},
            )

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

        # Reproducibility hardening: the envelope must report the seed the
        # analyzer actually used. Both derive from compute_effective_seed, so
        # this is normally a no-op — but syncing from the analysis result
        # guarantees seed_used can never drift from the RNG streams again.
        builder.seed_used = str(v1_response.metadata.seed_used)

        # Task 3: Build option label lookup for propagation to V2 response
        option_label_map = {opt.id: opt.label for opt in request.options}

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

            # Convert constraint_analysis if present
            constraint_analysis_v2 = None
            if result.constraint_analysis:
                ca = result.constraint_analysis
                constraint_analysis_v2 = ConstraintAnalysisV2(
                    constraints=[
                        ConstraintResultV2(
                            node_id=c.node_id,
                            operator=c.operator,
                            threshold=c.threshold,
                            label=c.label,
                            prob_satisfied=c.prob_satisfied,
                            failure_margin_median=c.failure_margin_median,
                            near_miss_fraction=c.near_miss_fraction,
                            binding=c.binding,
                        )
                        for c in ca.constraints
                    ],
                    joint_probability=ca.joint_probability,
                    conditional_probabilities=ca.conditional_probabilities,
                )

            # Task 2: Compute actual p10/p50/p90 from the *cleaned* samples
            # (same array used for n_valid_samples and critiques above) instead
            # of aliasing ci_lower (2.5th) → p10 and ci_upper (97.5th) → p90.
            # This fixes the percentile semantic drift where a 95% CI was
            # mislabelled as an 80% interval, and ensures percentiles reflect
            # the same sample set as validity metrics.
            #
            # CIL 0.2: return null when true percentiles unavailable, not
            #          mislabelled CI bounds. See code review C3.
            if dist.samples is not None and len(dist.samples) > 0:
                # Use cleaned_samples (from validate_mc_samples above) filtered
                # to finite values — identical population to n_valid_samples.
                finite_cleaned = cleaned_samples[np.isfinite(cleaned_samples)]
                if len(finite_cleaned) > 0:
                    # Task 4: single np.percentile call for efficiency
                    p10_val, p50_val, p90_val = (
                        float(v) for v in np.percentile(finite_cleaned, [10, 50, 90])
                    )
                    percentiles_source = "samples"
                else:
                    # All samples non-finite — cannot compute true percentiles
                    p10_val = None
                    p50_val = None
                    p90_val = None
                    percentiles_source = "unavailable"
            else:
                # No raw samples available — cannot compute true percentiles
                p10_val = None
                p50_val = None
                p90_val = None
                percentiles_source = "unavailable"

            option_results.append(
                OptionResultV2(
                    id=result.option_id,
                    # Task 3: Propagate option label from request to response
                    label=option_label_map.get(result.option_id),
                    outcome=OutcomeDistributionV2(
                        mean=dist.mean,
                        std=dist.std,
                        p10=p10_val,
                        p50=p50_val,
                        p90=p90_val,
                        n_samples=n_total,
                        n_valid_samples=n_valid,
                        validity_ratio=validity_ratio,
                        percentiles_source=percentiles_source,
                    ),
                    win_probability=result.win_probability,
                    probability_of_goal=result.probability_of_goal,
                    constraint_analysis=constraint_analysis_v2,
                    status=status,
                    status_reason=(
                        "Numerical issues in sampling" if status != "computed" else None
                    ),
                )
            )

        # Check for degenerate outcomes
        degen_critique = detect_degenerate_outcomes(option_results)
        if degen_critique:
            builder.add_critique(degen_critique)

        # Propagate analysis critiques from v1_response to enhanced response
        # (e.g., DEGENERATE_OPTION_ZERO_VARIANCE, HIGH_TIE_RATE)
        if v1_response.critiques:
            builder.add_critiques(v1_response.critiques)

        # Propagate inference warnings (e.g., STRENGTH_MEAN_CLAMPED, CONSTRAINT_NODE_DEFAULT_BASE)
        if v1_response.inference_warnings:
            builder.set_inference_warnings(v1_response.inference_warnings)

        # Convert robustness result (include V1 fields for backward compatibility)
        # Level mapping (implemented scheme, aligned with Decision Model Schema v2.6):
        #   is_robust (stability >= 0.7) AND confidence > 0.8 → "high"
        #   is_robust (stability >= 0.7) AND confidence <= 0.8 → "moderate"
        #   NOT robust AND confidence > 0.5 → "low"
        #   NOT robust AND confidence <= 0.5 → "very_low"
        robustness_result = None
        if v1_response.robustness:
            if v1_response.robustness.is_robust:
                level = "high" if v1_response.robustness.confidence > 0.8 else "moderate"
            else:
                level = "low" if v1_response.robustness.confidence > 0.5 else "very_low"

            # Convert FragileEdgeEnhanced to FragileEdgeV2 (same schema, different modules)
            fragile_edges_v2 = None
            if v1_response.robustness.fragile_edges_enhanced:
                fragile_edges_v2 = [
                    FragileEdgeV2(
                        edge_id=fe.edge_id,
                        from_id=fe.from_id,
                        to_id=fe.to_id,
                        alternative_winner_id=fe.alternative_winner_id,
                        switch_probability=fe.switch_probability,
                        marginal_switch_probability=fe.marginal_switch_probability,
                    )
                    for fe in v1_response.robustness.fragile_edges_enhanced
                ]

            # Convert E-values if present
            edge_e_values_v2 = None
            if v1_response.edge_e_values:
                edge_e_values_v2 = []
                for ev in v1_response.edge_e_values:
                    raw_e_val = ev.get("e_value", float("inf"))
                    unflippable = (
                        not isinstance(raw_e_val, (int, float))
                        or raw_e_val == float("inf")
                        or raw_e_val > 1e6
                    )
                    edge_e_values_v2.append(
                        EdgeEValueV2(
                            edge_id=ev["edge_id"],
                            from_id=ev["from_id"],
                            to_id=ev["to_id"],
                            e_value=None if unflippable else round(raw_e_val, 4),
                            is_unflippable=unflippable,
                            flip_direction=ev["flip_direction"],
                            current_mean=ev["current_mean"],
                            flip_mean=ev["flip_mean"],
                        )
                    )

            # T1-6: Edge-level sensitivity (additive wire completeness).
            # The analyzer always computed these forced-existence + magnitude
            # contrasts (v1_response.sensitivity) but the V2 envelope dropped
            # them, so consumers saw EDGE_SENSITIVITY_UNAVAILABLE_V2_WIRE.
            # Mirror them onto robustness.edge_sensitivity in V2 naming style.
            # sensitivity_score uses the same max-|elasticity| normalization as
            # factor_sensitivity; edge_id uses the same 'from->to' format as
            # fragile_edges and edge_e_values.
            edge_sensitivity_v2 = None
            if v1_response.sensitivity:
                max_abs_edge_elasticity = max(
                    (abs(s.elasticity) for s in v1_response.sensitivity), default=1.0
                )
                if max_abs_edge_elasticity < 1e-10:
                    max_abs_edge_elasticity = 1.0
                edge_sensitivity_v2 = [
                    EdgeSensitivityV2(
                        edge_id=f"{s.edge_from}->{s.edge_to}",
                        from_id=s.edge_from,
                        to_id=s.edge_to,
                        sensitivity_type=s.sensitivity_type,
                        sensitivity_score=min(1.0, abs(s.elasticity) / max_abs_edge_elasticity),
                        direction="positive" if s.elasticity > 0 else "negative",
                        elasticity=s.elasticity,
                        importance_rank=s.importance_rank,
                        interpretation=s.interpretation,
                    )
                    for s in v1_response.sensitivity
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
                # E-value analogue
                edge_e_values=edge_e_values_v2,
                # Edge-level sensitivity (T1-6 — additive)
                edge_sensitivity=edge_sensitivity_v2,
                # Trust penalty audit trail
                stability_penalty_factor=v1_response.robustness.stability_penalty_factor,
                defaulted_root_node_ids=v1_response.robustness.defaulted_root_node_ids,
            )

        # Convert factor sensitivity with normalized scores
        factor_sensitivity = None
        if v1_response.factor_sensitivity:
            # Compute max absolute elasticity for normalization
            max_abs_elasticity = max(
                (abs(fs.elasticity) for fs in v1_response.factor_sensitivity), default=1.0
            )
            # Avoid division by zero
            if max_abs_elasticity < 1e-10:
                max_abs_elasticity = 1.0

            n_factors = len(v1_response.factor_sensitivity)

            # Track S: per-factor provenance echo. Look up source nodes so we can
            # surface where each factor's value came from. Echo-only — not consumed
            # in inference.
            nodes_by_id = {n.id: n for n in request.graph.nodes}

            factor_sensitivity = []
            for fs in v1_response.factor_sensitivity:
                # Confidence: prefer bootstrap-derived, fall back to graph-structural
                bootstrap_confidence = compute_factor_confidence(
                    fs.attribution_stability,
                    fs.elasticity,
                    fs.elasticity_std,
                )
                if bootstrap_confidence is not None:
                    confidence_val = bootstrap_confidence
                    confidence_src = "bootstrap_sampling"
                else:
                    # Defensive fallback: in normal flow, bootstrap always runs
                    # when factor sensitivity exists, so this branch is unreachable.
                    # Kept as a safety net for future code paths that may bypass bootstrap.
                    confidence_val = compute_graph_structural_confidence(fs.influence_score)
                    confidence_src = "graph_structural"

                # Track S: echo where the factor's value came from. Pure passthrough
                # of request-supplied provenance; ISL does not consume it in inference.
                node = nodes_by_id.get(fs.node_id)
                obs = node.observed_state if node is not None else None
                value_source = obs.source if obs is not None else None
                value_extraction_type = obs.extractionType if obs is not None else None
                # Defaulted when no observed value was supplied (mean falls back to 0.0).
                # Same observed-value check as the ROOT_NODE_DEFAULT_VALUE warning logic.
                has_observed_value = obs is not None and obs.value is not None
                value_defaulted: Optional[bool] = (
                    True if (node is not None and not has_observed_value) else None
                )

                factor_sensitivity.append(
                    FactorSensitivityV2(
                        node_id=fs.node_id,
                        label=fs.node_label,
                        # Normalize to 0-1: relative to max sensitivity in this analysis
                        sensitivity_score=min(1.0, abs(fs.elasticity) / max_abs_elasticity),
                        # importance_score: 1.0 for rank 1, decreasing linearly
                        importance_score=(
                            1.0 - (fs.importance_rank - 1) / max(n_factors - 1, 1)
                            if n_factors > 1
                            else 1.0
                        ),
                        elasticity=fs.elasticity,  # Preserve raw elasticity
                        elasticity_display=fs.elasticity_display,  # Clamped for UI
                        direction="positive" if fs.elasticity > 0 else "negative",
                        confidence=confidence_val,
                        confidence_source=confidence_src,
                        importance_rank=fs.importance_rank,
                        observed_value=fs.observed_value,
                        interpretation=fs.interpretation,
                        # Debug fields (always included)
                        zero_reason=fs.zero_reason,
                        baseline_near_zero=fs.baseline_near_zero,
                        # Structural influence from graph path analysis
                        influence_score=fs.influence_score,
                        influence_rank=fs.influence_rank,
                        # Bootstrap uncertainty (3C)
                        elasticity_std=fs.elasticity_std,
                        attribution_stability=fs.attribution_stability,
                        rank_flip_rate=fs.rank_flip_rate,
                        stability_method=fs.stability_method,
                        # Provenance echo (Track S)
                        value_source=value_source,
                        value_extraction_type=value_extraction_type,
                        value_defaulted=value_defaulted,
                    )
                )

        builder.set_results(
            options=option_results,
            robustness=robustness_result,
            factor_sensitivity=factor_sensitivity,
            stability_thresholds=v1_response.stability_thresholds,  # 3C
            factor_evpi=v1_response.factor_evpi,
        )

        # B3: surface auto-noise disclosure on the V2 envelope so PLoT can read it
        # without consulting V1 _metadata. metadata is a required field on
        # RobustnessResponseV2, so this is always a concrete bool here; the
        # Optional on the V2 envelope covers error paths where build() runs
        # without metadata having been populated.
        builder.set_auto_noise_applied(v1_response.metadata.auto_noise_applied)

        # T1-6: Path decomposition passthrough (additive; request-gated by
        # include_path_decomposition so it only appears when asked for).
        # Previously computed by the analyzer on request but dropped from the
        # V2 envelope. Same schema, V2 module (PathDecompositionV2 mirror).
        if v1_response.path_decomposition is not None:
            pd = v1_response.path_decomposition
            builder.set_path_decomposition(
                PathDecompositionV2(
                    recommended_option_id=pd.recommended_option_id,
                    entry_nodes=pd.entry_nodes,
                    truncated=pd.truncated,
                    path_count=pd.path_count,
                    paths=[
                        PathContributionV2(
                            path=p.path,
                            path_effect=p.path_effect,
                            total_effect=p.total_effect,
                            signed_contribution=p.signed_contribution,
                            status=p.status,
                            mechanism=p.mechanism,
                        )
                        for p in pd.paths
                    ],
                )
            )

        # T1-5: Reference-option disclosure (additive). Edge sensitivity,
        # factor sensitivity, and the fragile-edge classification derived from
        # edge sensitivity are all computed against request.options[0] (see
        # RobustnessAnalyzerV2._compute_sensitivity /
        # _compute_factor_sensitivity: ref_option = request.options[0]).
        # Disclose it whenever any of those analyses produced results.
        if v1_response.sensitivity or v1_response.factor_sensitivity:
            builder.set_sensitivity_reference_option_id(request.options[0].id)

        # Convert conditional winners (V1 -> V2)
        if v1_response.conditional_winners is not None:
            conditional_winners_v2 = [
                ConditionalWinnerV2(
                    factor_id=cw.factor_id,
                    factor_label=cw.factor_label,
                    split_value=cw.split_value,
                    split_unit=cw.split_unit,
                    low_bucket=BucketResultV2(
                        n_samples=cw.low_bucket.n_samples,
                        winner_id=cw.low_bucket.winner_id,
                        winner_label=cw.low_bucket.winner_label,
                        winner_probability=cw.low_bucket.winner_probability,
                        runner_up_id=cw.low_bucket.runner_up_id,
                        runner_up_probability=cw.low_bucket.runner_up_probability,
                    ),
                    high_bucket=BucketResultV2(
                        n_samples=cw.high_bucket.n_samples,
                        winner_id=cw.high_bucket.winner_id,
                        winner_label=cw.high_bucket.winner_label,
                        winner_probability=cw.high_bucket.winner_probability,
                        runner_up_id=cw.high_bucket.runner_up_id,
                        runner_up_probability=cw.high_bucket.runner_up_probability,
                    ),
                    winner_flips=cw.winner_flips,
                )
                for cw in v1_response.conditional_winners
            ]
            builder.set_conditional_winners(conditional_winners_v2)

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
        # exclude_none=True: suppresses fields set to None (e.g. probability_of_goal when
        # goal_constraints are absent).  List fields with default_factory=list survive
        # exclude_none because they are [] not None:
        #   - inference_warnings: always a list ([] when empty), propagated from analyzer
        #   - critiques: always a list, never None
        # Sentinel fields (percentiles_source, robustness.confidence) must be set in the
        # response model before reaching here so they are never lost by exclude_none.
        response_dict = response.model_dump(by_alias=True, exclude_none=True)
        # Safety net: guarantee inference_warnings is always present even if model changes
        if "inference_warnings" not in response_dict:
            response_dict["inference_warnings"] = []
        return JSONResponse(
            status_code=200,
            content=response_dict,
            headers={
                "X-Request-Id": request_id,
                "X-Processing-Time-Ms": str(response.processing_time_ms),
            },
        )

    except Exception as e:
        logger.exception(f"Analysis failed for request {request_id}: {e}")
        response = builder.build_error_response(e)
        error_dict = response.model_dump(by_alias=True, exclude_none=True)
        # Safety net: guarantee inference_warnings is always present in error responses too
        if "inference_warnings" not in error_dict:
            error_dict["inference_warnings"] = []
        return JSONResponse(
            status_code=500,
            content=error_dict,
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
            validated_request.request_id = validated_request.request_id or request_id
            return await analyze_robustness_v2(  # type: ignore[no-any-return]
                validated_request,
                x_request_id=request_id,
            )
        else:
            # Validate and process v1 request
            validated_v1_request = RobustnessRequest(**request)
            return await analyze_robustness(
                validated_v1_request,
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
                "parameter_uncertainties_type": type(
                    request.get("parameter_uncertainties")
                ).__name__,
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
