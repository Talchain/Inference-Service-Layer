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

from typing import Any, Dict, NamedTuple, Optional, Union

import numpy as np

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.api.error_helpers import raise_invalid_input
from src.config.stability_thresholds import (
    compute_factor_confidence,
    compute_graph_structural_confidence,
    get_confidence_method_version,
)
from src.constants import (
    DEFAULT_EXISTS_PROBABILITY_THRESHOLD,
    DEFAULT_RESPONSE_VERSION,
    DEFAULT_STRENGTH_THRESHOLD,
)
from src.models.critique import LOW_EFFECTIVE_SAMPLES
from src.models.metadata import create_response_metadata
from src.models.response_v2 import (
    BucketResultV2,
    ConditionalWinnerV2,
    ConfidenceProvenance,
    ConstraintAnalysisV2,
    ConstraintResultV2,
    DiagnosticsV2,
    DownsideV2,
    EdgeEValueV2,
    EdgeSensitivityV2,
    FactorFlipValueV2,
    FactorSensitivityV2,
    FlipStabilityBandV2,
    FragileEdgeV2,
    ISLResponseV2,
    ISLV2Error422,
    OptionResultV2,
    OutcomeDistributionV2,
    PathContributionV2,
    PathDecompositionV2,
    RobustnessResultV2,
)
from src.models.responses import ErrorCode, ErrorResponse, RecoveryHints
from src.models.robustness import RobustnessRequest, RobustnessResponse
from src.models.robustness_v2 import (
    FactorSensitivityResult,
    RobustnessRequestV2,
    RobustnessResponseV2,
    detect_schema_version,
)
from src.services.analysis_pool import AnalysisDeadlineExceeded, run_offloaded
from src.services.compute_governor import RETRY_AFTER_SECONDS, ComputeGovernor, Overload
from src.services.robustness_analyzer import RobustnessAnalyzer
from src.services.robustness_analyzer_v2 import (
    COMPLEXITY_FORMULA_VERSION,
    RobustnessAnalyzerV2,
    WeightedCost,
    compute_effective_seed,
    compute_weighted_cost,
    get_max_cost_units,
)
from src.utils.business_metrics import track_robustness_analysis
from src.utils.downside import cvar_from_samples
from src.utils.numerical_stability import overflow_safe_mean_std, validate_mc_samples
from src.utils.response_builder import (
    ResponseBuilder,
    build_request_echo,
    determine_option_status,
    hash_node_id,
)
from src.utils.tracing import sanitize_request_id
from src.validation.degenerate_detector import detect_degenerate_outcomes
from src.validation.path_validator import PathValidationConfig
from src.validation.request_validator import RequestValidator

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request complexity guard (DoS protection) — Codex F8 weighted cost model
# ---------------------------------------------------------------------------
# The admission gate uses compute_weighted_cost(request) and get_max_cost_units()
# from src.services.robustness_analyzer_v2 (the single source of truth for the
# weights + ceiling, also advertised on /health). The pre-F8 scalar
# n_samples*n_nodes*n_edges heuristic (and its ISL_MAX_COMPUTE_COMPLEXITY env)
# is REPLACED: it omitted the option multiplier, used the wrong x-edges shape,
# and priced none of the optional phases. See the cost-model block in
# robustness_analyzer_v2.py for the formula and calibration provenance.


class _FactorConfidence(NamedTuple):
    """A factor's confidence figure resolved together with its source label and the
    S2 honest-disclosure marker. `provenance` is never None — see
    _resolve_factor_confidence for why the marker is unconditional."""

    value: float
    source: str
    provenance: ConfidenceProvenance


def _resolve_factor_confidence(
    fs: FactorSensitivityResult, bootstrap_method_version: str
) -> _FactorConfidence:
    """Resolve a factor's confidence together with its source and the S2 marker.

    Prefer the bootstrap-CV-blend confidence; fall back to graph-structural when
    bootstrap metrics are unavailable.

    S2 — honest disclosure marker: the marker rides EXACTLY alongside the
    confidence figure. On the wire (FactorSensitivityV2, exclude_none=True) it is
    present EXACTLY when `confidence` is present and OMITTED (never a JSON null)
    when confidence is absent; the F-3 model_validator on FactorSensitivityV2
    enforces that iff-invariant and stays as the fail-loud backstop. Both branches
    here ALWAYS produce a float (compute_graph_structural_confidence never returns
    None), so the marker is constructed UNCONDITIONALLY. The mapping is PROVISIONAL
    (Neil gate 1), so `calibrated` is always False until a validated calibration
    exists and `method_version` is bumped on any mapping change (guarded by
    tests/unit/test_confidence_provenance.py).
    """
    bootstrap_confidence = compute_factor_confidence(
        fs.attribution_stability,
        fs.elasticity,
        fs.elasticity_std,
    )
    if bootstrap_confidence is not None:
        return _FactorConfidence(
            bootstrap_confidence,
            "bootstrap_sampling",
            ConfidenceProvenance.bootstrap(bootstrap_method_version),
        )

    # Defensive fallback: in normal flow, bootstrap always runs when factor
    # sensitivity exists, so this branch is unreachable. Kept as a safety net for
    # future code paths that may bypass bootstrap.
    #
    # F-2: the fallback is a DIFFERENT method — stamp its own version, never the
    # bootstrap blend's, so the wire never names a method that did not produce the
    # number.
    return _FactorConfidence(
        compute_graph_structural_confidence(fs.influence_score),
        "graph_structural",
        ConfidenceProvenance.graph_structural(),
    )


def _admission_suggestion(request: RobustnessRequestV2, cost: WeightedCost) -> str:
    """Actionable reduction hint keyed on the dominant cost term."""
    dominant = cost.dominant_term
    n_uncertainties = len(request.parameter_uncertainties or [])
    if dominant == "evpi":
        return (
            f"EVPI dominates: reduce n_samples (currently {request.n_samples}), "
            f"options (currently {len(request.options)}), or parameter_uncertainties "
            f"(currently {n_uncertainties}); or set include_voi=false."
        )
    if dominant == "evppi_full":
        # v4 (OC-1): the term is O-flat (one shared multi-RHS SVD across options),
        # so reducing options would NOT reduce this term — don't suggest it.
        return (
            f"Per-factor EVPPI regression dominates: reduce n_samples (currently "
            f"{request.n_samples}) or parameter_uncertainties (currently "
            f"{n_uncertainties}); or set include_voi=false."
        )
    if dominant == "evpc":
        n_candidates = len(request.control_candidates or [])
        n_grid = sum(len(c.values) for c in (request.control_candidates or []))
        return (
            f"Value-of-control (EVPC) dominates: reduce n_samples (currently "
            f"{request.n_samples}), control_candidates (currently {n_candidates}), or the "
            f"number of candidate values ({n_grid} grid points total); or omit control_candidates."
        )
    if dominant == "sensitivity":
        return (
            f"Edge sensitivity dominates: reduce n_samples (currently {request.n_samples}) "
            "or remove 'sensitivity' from analysis_types."
        )
    if dominant in ("e_values", "bands"):
        return (
            f"E-value/stability-band analysis dominates: reduce edge count "
            f"(currently {len(request.graph.edges)}) or set include_e_values=false."
        )
    if dominant == "path_decomposition":
        return (
            f"Path decomposition dominates: reduce edge count "
            f"(currently {len(request.graph.edges)}) or set include_path_decomposition=false."
        )
    # base_mc (default)
    return (
        f"Base Monte-Carlo dominates: reduce n_samples (currently {request.n_samples}), "
        f"options (currently {len(request.options)}), or graph size "
        f"(currently {len(request.graph.nodes)} nodes, {len(request.graph.edges)} edges)."
    )


def _admission_error_body(request: RobustnessRequestV2, cost: WeightedCost, max_cost: int) -> dict:
    """Structured 422 body reporting the weighted cost + which term dominated."""
    return {
        "detail": "Request compute cost exceeds limit",
        "cost_units": cost.total,
        "limit": max_cost,
        "dominant_term": cost.dominant_term,
        "cost_breakdown": cost.terms,
        "complexity_formula_version": COMPLEXITY_FORMULA_VERSION,
        # Back-compat aliases for any consumer that read the pre-F8 keys.
        "complexity_score": cost.total,
        "suggestion": _admission_suggestion(request, cost),
    }


# ---------------------------------------------------------------------------
# Compute-governor overload + hard-deadline responses (Codex F15)
# ---------------------------------------------------------------------------
# The governor returns EARLY and TYPED on overload rather than leaving a caller
# hanging on a saturated pool: 429 when the caller's own concurrency is the cause,
# 503 when the service as a whole is busy. The hard deadline (worker terminated)
# returns 504. All three carry Retry-After and the canonical ErrorResponse body.


def _ensure_governor(app: Any) -> ComputeGovernor:
    """Return the app's compute governor, lazily creating a default if absent.

    Production installs the governor in the lifespan; a test using the
    ASGITransport client (no lifespan) has none. Lazily attaching a single default
    instance keeps the endpoint working (and bounded) rather than 500-ing on a
    missing attribute — part of the "analysis can never brick" guarantee.
    """
    gov = getattr(app.state, "governor", None)
    if gov is None:
        gov = ComputeGovernor()
        app.state.governor = gov
    return gov


def _overload_error_response(overload: Overload, request_id: str) -> ErrorResponse:
    """Typed body for a governor 429/503 (caller-concurrency / service-busy)."""
    if overload.status_code == 429:
        code = ErrorCode.RATE_LIMIT_EXCEEDED.value
        message = "Too many concurrent analyses from this caller. Retry shortly."
        hints = [
            "Reduce the number of simultaneous /analyze requests you issue",
            "Retry after the Retry-After interval",
        ]
    else:
        code = ErrorCode.SERVICE_UNAVAILABLE.value
        message = "Analysis service is at compute capacity. Retry shortly."
        hints = [
            "Retry after the Retry-After interval",
            "Consider smaller or fewer concurrent analyses",
        ]
    return ErrorResponse(
        code=code,
        message=message,
        reason=overload.reason,
        recovery=RecoveryHints(hints=hints, suggestion="Retry after Retry-After seconds"),
        retryable=True,
        source="isl",
        request_id=request_id,
    )


def _deadline_error_response(deadline: AnalysisDeadlineExceeded, request_id: str) -> ErrorResponse:
    """Typed 504 body for an analysis that blew the hard compute deadline."""
    return ErrorResponse(
        code=ErrorCode.TIMEOUT.value,
        message=(
            f"Analysis exceeded the {deadline.deadline_s:.0f}s hard compute deadline "
            "and was terminated."
        ),
        reason="analysis_hard_deadline_exceeded",
        recovery=RecoveryHints(
            hints=[
                "Reduce n_samples, options, or graph size",
                "Disable optional phases (include_voi / include_e_values / "
                "include_path_decomposition)",
            ],
            suggestion="Retry with a smaller analysis",
        ),
        retryable=True,
        source="isl",
        request_id=request_id,
    )


def _admission_cost_guard(
    request: RobustnessRequestV2, request_id: str
) -> tuple[WeightedCost, Optional[JSONResponse]]:
    """Compute the weighted admission cost, log it, and gate oversized requests.

    Shared by both v2 handlers (dedup + normalisation). Returns ``(cost, error)``
    where ``error`` is a 422 ``JSONResponse`` carrying the structured
    cost-breakdown body + ``X-Request-Id`` when the request exceeds the
    compute-cost ceiling, else ``None``. Both handlers now return the SAME
    normalised 422 — the legacy path previously ``raise``d ``HTTPException``,
    which the global handler rebuilt into the Olumi Error Schema (stringifying the
    structured body into ``message``) and dropped ``X-Request-Id``. Legacy is off
    the live V5 path, so this only tightens an already-rejected error shape.
    """
    n_nodes = len(request.graph.nodes)
    n_edges = len(request.graph.edges)
    cost = compute_weighted_cost(request)
    max_cost = get_max_cost_units()
    logger.info(
        "robustness_v2_complexity",
        extra={
            "request_id": request_id,
            "cost_units": cost.total,
            "limit": max_cost,
            "dominant_term": cost.dominant_term,
            "cost_breakdown": cost.terms,
            "n_samples": request.n_samples,
            "n_options": len(request.options),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        },
    )
    if cost.total > max_cost:
        return cost, JSONResponse(
            status_code=422,
            content=_admission_error_body(request, cost, max_cost),
            headers={"X-Request-Id": request_id},
        )
    return cost, None


async def _admit_and_run(
    app: Any,
    request: RobustnessRequestV2,
    request_id: str,
    api_key: Optional[str],
    cost: WeightedCost,
) -> tuple[Optional[RobustnessResponseV2], Optional[JSONResponse]]:
    """Admit through the compute governor and run the analysis offloaded.

    Shared by both v2 handlers. Returns ``(response, error)``: on success
    ``(response, None)``; on governor overload (429/503) or hard-deadline breach
    (504), ``(None, error)`` where ``error`` is the typed ``JSONResponse``
    carrying ``Retry-After`` + ``X-Request-Id``. The error is returned INLINE
    (not raised) so the canonical ErrorResponse body and the ``Retry-After``
    header survive — the global HTTPException handler rebuilds responses and
    drops custom headers. Exactly one of the two returned values is non-None.
    """
    governor = _ensure_governor(app)
    try:
        async with governor.admit(cost.total, api_key):
            response = await run_offloaded(app, request, request_id)
    except Overload as overload:
        body = _overload_error_response(overload, request_id)
        return None, JSONResponse(
            status_code=overload.status_code,
            content=body.model_dump(exclude_none=True),
            headers={
                "Retry-After": str(overload.retry_after),
                "X-Request-Id": request_id,
            },
        )
    except AnalysisDeadlineExceeded as deadline:
        body = _deadline_error_response(deadline, request_id)
        return None, JSONResponse(
            status_code=504,
            content=body.model_dump(exclude_none=True),
            headers={
                "Retry-After": str(RETRY_AFTER_SECONDS),
                "X-Request-Id": request_id,
            },
        )
    return response, None


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

        # Create the response, then attribute-assign the metadata.
        #
        # A3 Lane M (2026-07-23): RobustnessResponse.metadata is aliased `_metadata`.
        # Populating it as a constructor kwarg BY FIELD NAME (`metadata=...`) was
        # silently dropped by Pydantic v2 — the V1 wire served `_metadata: null`, so
        # isl_version / config_fingerprint / config_details never reached any V1
        # client — UNTIL C3 (2026-07-23) added `populate_by_name: True` to the model.
        # Attribute assignment sets the field by its Python name regardless of alias
        # and is retained here as the shared idiom of the counterfactual (causal.py)
        # and sequential (phase4.py) live routes. With populate_by_name now set, the
        # model ALSO survives the worker-offload round-trip (model_dump_json ->
        # model_validate_json) — see tests/unit/test_metadata_populate_by_name_invariant.py.
        #
        # config_details no longer advertises `monte_carlo_samples` (C2, F-3,
        # 2026-07-23). Robustness runs real Monte Carlo, but its budget is
        # `request.min_samples`-driven — NOT MAX_MONTE_CARLO_ITERATIONS. The V1 wire
        # served `monte_carlo_samples: 10000` beside its own `samples_tested`, a
        # self-contradiction, so the fabricated key was DELETED from
        # generate_config_details (no live path uses that cap as its operative budget).
        response = RobustnessResponse(analysis=analysis)
        response.metadata = create_response_metadata(request_id)

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
        429: {
            "model": ErrorResponse,
            "description": "Caller concurrency limit exceeded — retry after Retry-After",
        },
        503: {
            "model": ErrorResponse,
            "description": "Service at compute capacity — retry after Retry-After",
        },
        504: {
            "model": ErrorResponse,
            "description": "Analysis exceeded the hard compute deadline and was terminated",
        },
        500: {"description": "Internal computation error"},
    },
)
async def analyze_robustness_v2(
    request: RobustnessRequestV2,
    http_request: Request,
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
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
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
        return await _analyze_robustness_v2_legacy(request, request_id, http_request.app, x_api_key)

    # V2 response format with validation and structured output
    return await _analyze_robustness_v2_enhanced(
        request, request_id, include_diagnostics, http_request.app, x_api_key
    )


async def _analyze_robustness_v2_legacy(
    request: RobustnessRequestV2,
    request_id: str,
    app: Any,
    api_key: Optional[str],
) -> Union[RobustnessResponseV2, JSONResponse]:
    """Legacy V1 response handler (backward compatible)."""
    try:
        # Compute-admission guard (Codex F8 weighted cost) — reject oversized
        # requests early (DoS protection). Shared with the enhanced handler; the
        # 422 is now the normalised structured body + X-Request-Id (was a raised
        # HTTPException that the global handler reshaped and stripped the header).
        cost, cost_error = _admission_cost_guard(request, request_id)
        if cost_error is not None:
            return cost_error

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

        # Perform v2 analysis — offloaded to the process pool via the compute
        # governor (Codex F15). Overload (429/503) and the hard deadline (504) are
        # returned INLINE as typed JSONResponses so the canonical ErrorResponse
        # body AND Retry-After + X-Request-Id survive the global handler. Shared
        # with the enhanced handler.
        response, run_error = await _admit_and_run(app, request, request_id, api_key, cost)
        if run_error is not None:
            return run_error
        assert response is not None  # _admit_and_run returns exactly one non-None

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
        raise_invalid_input(logger, "robustness_v2_invalid_input", request_id, e)
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
    app: Any,
    api_key: Optional[str],
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
        # Compute-admission guard (Codex F8 weighted cost) — reject oversized
        # requests early (DoS protection). Shared with the legacy handler.
        cost, cost_error = _admission_cost_guard(request, request_id)
        if cost_error is not None:
            return cost_error

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

        # Run analysis — offloaded via the compute governor (Codex F15). Overload
        # (429/503) and the hard deadline (504) are returned INLINE as typed
        # JSONResponses so the outer `except Exception -> 500` never mis-maps them
        # and Retry-After + X-Request-Id survive. run_offloaded self-heals on
        # worker crash (in-process fallback). Shared with the legacy handler.
        v1_response, run_error = await _admit_and_run(app, request, request_id, api_key, cost)
        if run_error is not None:
            return run_error
        assert v1_response is not None  # _admit_and_run returns exactly one non-None

        # Reproducibility hardening: the envelope must report the seed the
        # analyzer actually used. Both derive from compute_effective_seed, so
        # this is normally a no-op — but syncing from the analysis result
        # guarantees seed_used can never drift from the RNG streams again.
        builder.seed_used = str(v1_response.metadata.seed_used)

        # Task 3: Build option label lookup for propagation to V2 response
        option_label_map = {opt.id: opt.label for opt in request.options}

        # B2 downside — JOINT expected regret is computed in the ANALYZER from the
        # PRE-noise CRN-aligned outcomes (the same population as win_probability)
        # and threaded here via OptionResult.pre_noise_expected_regret (a regular
        # serialized field, so it survives the offload worker->endpoint boundary).
        # It is deliberately NOT recomputed from outcome_distribution.samples,
        # which are POST `_apply_auto_scaled_noise` (independent per-option noise
        # that breaks CRN alignment and inflates regret ~80x for near-equivalent
        # options — see CODE-REVIEW-ISL F1). cvar_10 / p05 ARE marginal and stay on
        # the noised per-option samples, computed inside the loop below.

        # Convert V1 response to V2 option results
        option_results = []
        for result in v1_response.results:
            dist = result.outcome_distribution

            # P2-ISL-5: Validate and clean MC samples with proper critiques
            n_total = request.n_samples
            if dist.samples is not None and len(dist.samples) > 0:
                # Use numerical stability utility for validation + critique emission
                samples_array = np.array(dist.samples)
                _imputed, sample_critiques = validate_mc_samples(samples_array)

                # Add any numerical stability critiques (once per option set)
                for critique in sample_critiques:
                    # Add option context to critique
                    critique.affected_option_ids = [result.option_id]
                    builder.add_critique(critique)

                # 2.475 defect A: count valid samples from the RAW array, never
                # the imputed one. validate_mc_samples replaces every non-finite
                # sample with the median of the valid ones, so counting AFTER
                # replacement always yielded n_valid == n_total and the wire
                # fields n_valid_samples ("Samples without NaN/Inf") and
                # validity_ratio OVERSTATED validity — determine_option_status's
                # 'partial' branch was dead code at its only call site.
                finite_samples = samples_array[np.isfinite(samples_array)]
                n_valid = int(finite_samples.size)
            else:
                # V1 analyzer doesn't track validity - assume all valid
                finite_samples = None
                n_valid = n_total

            validity_ratio = n_valid / n_total if n_total > 0 else 0.0
            status = determine_option_status(n_valid, n_total)

            # 2.475: LOW_EFFECTIVE_SAMPLES — the user-facing disclosure for a
            # partially-valid option. Registered (with an S-bucket copy at CEE
            # and a deployed UI reader) but never emitted anywhere until now.
            # Success-path emission: the run still returns 200; the option is
            # honestly 'partial'. Threshold is the existing MIN_VALID_RATIO
            # (0.8) inside determine_option_status — no new number.
            if status == "partial":
                builder.add_critique(
                    LOW_EFFECTIVE_SAMPLES.build(
                        valid_count=n_valid,
                        total_count=n_total,
                        affected_option_ids=[result.option_id],
                        seed=v1_response.metadata.seed_used,
                    )
                )

            # Convert constraint_analysis if present
            constraint_analysis_v2 = None
            if result.constraint_analysis:
                ca = result.constraint_analysis
                constraint_analysis_v2 = ConstraintAnalysisV2(
                    constraints=[
                        ConstraintResultV2(
                            # Slice 6b: the internal->wire hop of the echo. This
                            # is the layer that actually reaches PLoT/CEE; the
                            # field must be threaded at every hop or it is lost
                            # here after surviving the analyzer.
                            constraint_id=c.constraint_id,
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

            # Task 2: Compute actual p10/p50/p90 from the *finite* samples
            # (same population n_valid_samples counts) instead of aliasing
            # ci_lower (2.5th) → p10 and ci_upper (97.5th) → p90.
            # This fixes the percentile semantic drift where a 95% CI was
            # mislabelled as an 80% interval, and ensures percentiles reflect
            # the same sample set as validity metrics. (2.475: the population
            # is now the finite subset of the RAW samples — before, it was the
            # median-IMPUTED array, which is identical whenever every sample is
            # finite, i.e. for every run that could serialize at all, but
            # double-weighted the median on the partial runs this fix unlocks.)
            #
            # CIL 0.2: return null when true percentiles unavailable, not
            #          mislabelled CI bounds. See code review C3.
            # B2 downside — cvar_10/p05 from the SAME finite (post-noise)
            # population as the percentiles; expected_regret is the PRE-noise
            # joint value threaded from the analyzer. Present exactly when
            # percentiles_source=="samples" AND the pre-noise regret is available.
            downside: Optional[DownsideV2] = None
            # 2.477(f): p10/p50/p90 can now be dropped to None when percentile
            # interpolation overflows, so they are Optional from the start
            # rather than narrowed to float by the first assignment.
            p10_val: Optional[float]
            p50_val: Optional[float]
            p90_val: Optional[float]
            if finite_samples is not None:
                finite_cleaned = finite_samples
                if len(finite_cleaned) > 0:
                    # Task 4: single np.percentile call for efficiency. B2 adds
                    # p05, extending the p10/p50/p90 family downward with the SAME
                    # percentile machinery/convention (no reimplementation).
                    p05_val, p10_val, p50_val, p90_val = (
                        float(v) for v in np.percentile(finite_cleaned, [5, 10, 50, 90])
                    )
                    percentiles_source = "samples"
                    # 2.477(f): np.percentile INTERPOLATES, and interpolation
                    # over a finite but extreme mixed-sign population overflows
                    # (executed: [-1.7e308, 1.7e308] -> [inf, inf, -inf, -inf]).
                    # p10/p50/p90 flowed from here into OutcomeDistributionV2
                    # UNGUARDED — only the downside path checked finiteness — so
                    # the render rejected the body and the whole run 500'd. The
                    # percentile family travels together (the model documents
                    # them as null together), so if any member cannot be
                    # computed honestly the family is 'unavailable' rather than
                    # part-null. Downside is gated on percentiles_source ==
                    # 'samples' by the model validator and is therefore omitted
                    # with them, which is correct: tail-risk numbers from a
                    # population whose own percentiles overflowed are not
                    # trustworthy. For every run that could serialize before,
                    # all four are finite and this branch is never taken.
                    if not all(math.isfinite(v) for v in (p05_val, p10_val, p50_val, p90_val)):
                        p10_val = None
                        p50_val = None
                        p90_val = None
                        percentiles_source = "unavailable"
                    # PRE-noise CRN-aligned joint regret (B2 CRN-fix F1), threaded
                    # from the analyzer. Present for every option with samples; if
                    # unexpectedly absent, omit downside rather than fabricate —
                    # absent != wrong (trap #13). cvar_10 (mean of the worst
                    # decile) and p05 stay on the finite (post-noise) population.
                    #
                    # 2.475: omit the WHOLE block when any component is
                    # non-finite. On a partial-validity run the joint regret is
                    # poisoned for EVERY option (max over options includes the
                    # inf samples), and cvar_10 can overflow on an extreme
                    # finite population — tail-risk numbers from a degraded
                    # population are not trustworthy, and a non-finite float
                    # is unserializable (the render rejects it). For every
                    # run that could serialize before this fix, all three are
                    # finite and the block is built exactly as before.
                    pre_noise_regret = result.pre_noise_expected_regret
                    if (
                        percentiles_source == "samples"
                        and pre_noise_regret is not None
                        and math.isfinite(pre_noise_regret)
                    ):
                        cvar_val = cvar_from_samples(finite_cleaned)
                        if math.isfinite(cvar_val) and math.isfinite(p05_val):
                            downside = DownsideV2(
                                cvar_10=cvar_val,
                                p05=p05_val,
                                expected_regret=pre_noise_regret,
                            )
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

            # 2.475: the analyzer's dist.mean/dist.std are np.mean/np.std over
            # the RAW samples array — one non-finite sample poisons both, and a
            # non-finite float in a required field makes the whole response
            # unserializable (the pre-fix behaviour: every partial-validity run
            # died as a 500 and no critique ever reached a user). Report the
            # finite sub-population's mean/std instead — the same population
            # n_valid_samples counts — via the overflow-safe helper (the finite
            # subset can sit near float64 max, where plain np.mean/np.std
            # overflow in the accumulator).
            #
            # ⚠ 2.477(f)+(g): 2.475 gated this on `status == "partial"`, and that
            # gate was TOO NARROW in both directions — measured, not reasoned:
            #  * (f) an option with 25 of 200 samples non-finite has validity
            #    0.875 >= MIN_VALID_RATIO (0.8), so its status is "computed" and
            #    it kept the POISONED dist.mean/dist.std verbatim. The entire
            #    validity band 0.8 <= ratio < 1.0 still 500'd. (The executed
            #    shape the adversarial review labelled a percentile overflow was
            #    in fact this: the field manifest showed outcome.mean = nan and
            #    outcome.std = nan, on a "computed" option.)
            #  * (g) samples that are ALL finite at ~1e299 overflow np.std's
            #    squared-deviation accumulator, so a fully-valid "computed"
            #    option shipped std = inf and 500'd.
            # The honest gate is FINITENESS, not status. `status == "partial"`
            # implies >=1 non-finite sample implies dist.mean is non-finite, so
            # the finiteness condition is a strict SUPERSET of the old one: every
            # 2.475 behaviour is preserved exactly, and the two holes close.
            #
            # NO-REGRESSION: a run that could serialize before had finite
            # dist.mean/dist.std by construction (the render rejects non-finite
            # floats), so it never enters this branch and its bytes are
            # unchanged.
            outcome_mean: Optional[float]
            outcome_std: Optional[float]
            if math.isfinite(dist.mean) and math.isfinite(dist.std):
                outcome_mean, outcome_std = dist.mean, dist.std
            elif finite_samples is not None and finite_samples.size > 0:
                outcome_mean, outcome_std = overflow_safe_mean_std(finite_samples)
            else:
                # 2.477(a): ZERO finite samples — there is no honest mean or std
                # to report, and inventing 0.0 would be a fabrication dressed as
                # a measurement. Omit both (exclude_none => ABSENT on the wire,
                # never a JSON null), exactly as p10/p50/p90 and downside are
                # already omitted for this option. The response now SHIPS as a
                # 200 carrying the option's MONTE_CARLO_FAILED critique, instead
                # of dying as a 500 that took every critique with it.
                outcome_mean, outcome_std = None, None

            option_results.append(
                OptionResultV2(
                    id=result.option_id,
                    # Task 3: Propagate option label from request to response
                    label=option_label_map.get(result.option_id),
                    outcome=OutcomeDistributionV2(
                        mean=outcome_mean,
                        std=outcome_std,
                        p10=p10_val,
                        p50=p50_val,
                        p90=p90_val,
                        n_samples=n_total,
                        n_valid_samples=n_valid,
                        validity_ratio=validity_ratio,
                        percentiles_source=percentiles_source,
                    ),
                    downside=downside,
                    win_probability=result.win_probability,
                    probability_of_goal=result.probability_of_goal,
                    constraint_analysis=constraint_analysis_v2,
                    status=status,
                    status_reason=(
                        "Numerical issues in sampling" if status != "computed" else None
                    ),
                )
            )

        # S1 (A3 VOI honesty, D-23.8): decision-level EVPI in OUTCOME UNITS.
        # decision_evpi = min_o expected_regret[o] = E[max_o U] − max_o E[U] on the
        # JOINT pre-noise CRN population (exact identity: min of E[max]−E[o] over o
        # is achieved at argmax_o E[o]). This is a one-line READ of the regret
        # already emitted per option in downside.expected_regret — zero new
        # sampling, no recompute. We take the min over the options that carry a
        # downside block (percentiles_source == 'samples').
        #
        # EXACTNESS / no-overstate — what ACTUALLY protects it (adversarial F-1):
        # min over the DOWNSIDE-bearing subset equals the true min_o regret only if
        # the argmax-mean (== argmin-regret) option is in that subset. It is NOT true
        # that "auto-noise preserves finiteness": _apply_auto_scaled_noise computes
        # outcome_std over an option's pre-noise samples, and a single non-finite
        # sample makes std nan/inf, so rng.normal(0, nan/inf) then DESTROYS a
        # partially-finite option's finiteness → downside omitted → excluded here →
        # the remaining min could OVERSTATE.
        #
        # ⚠ 2.477(e) — THE OLD ARGUMENT HERE IS WITHDRAWN, AND THIS IS THE FIX.
        # This comment used to say the overstatement "never reaches a 200 response"
        # because of an INCIDENTAL serializer guard: any non-finite sample also
        # poisoned outcome.mean, and the render rejects non-finite floats, so the
        # whole response 500'd before the number could ship — "so on every 200 the
        # population is all-finite and this min is the exact EVPI". 2.475 REMOVED
        # that guard (partial options now report an overflow-safe mean over their
        # finite sub-population, and the run returns 200), and 2.477(f)/(g) removed
        # what was left of it. A comment that reasons from a guard which no longer
        # exists is worse than no comment: the exactness claim became false the day
        # 2.475 merged, and nothing failed.
        #
        # So the exactness is now ENFORCED rather than argued. min over a PARTIAL
        # population can only be >= the true min_o (dropping candidates can only
        # raise a minimum), i.e. it OVERSTATES the value of perfect information —
        # a trust defect, and the wrong direction to be wrong in. decision_evpi is
        # therefore emitted only when the population is COMPLETE: every option that
        # HAS a usable sample population (percentiles_source == 'samples') carries
        # a downside. Otherwise it is honestly ABSENT — absent != wrong (trap #13),
        # and an absent number cannot mislead the way an inflated one can.
        #
        # Options with no samples at all are deliberately NOT counted as gaps: they
        # carry no honest regret, were never in the population, and their presence
        # must not suppress a perfectly good EVPI over the options that were
        # actually sampled. (expected_regret_per_option assigns such an option a
        # DEFENSIVE 0.0, which would be a fabricated minimum — it never reaches the
        # wire because the downside block requires >=1 finite sample.)
        #
        # For every run that could serialize before 2.475, every sampled option
        # carries a downside, so the population is complete and this is
        # byte-identical to the previous behaviour. The ISLResponseV2 validator
        # enforces the emission rule (including that absence is licensed only by a
        # wire-visible gap) and the value, both ways.
        decision_evpi_regrets = [
            o.downside.expected_regret for o in option_results if o.downside is not None
        ]
        decision_evpi_population_complete = all(
            o.downside is not None
            for o in option_results
            if o.outcome.percentiles_source == "samples"
        )
        builder.set_decision_evpi(
            min(decision_evpi_regrets)
            if decision_evpi_regrets and decision_evpi_population_complete
            else None
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
                    # Track S Phase 1: seed-sweep stability band — additive,
                    # default-on; absent only when the all-or-nothing band
                    # budget trips (exclude_none drops it from the wire).
                    stability_raw = ev.get("stability")
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
                            # ROADMAP 2.228-F3 — additive winner capture. .get()
                            # rather than [] so an analyzer result built before
                            # this field existed (or by a caller constructing the
                            # dict directly) maps to None instead of raising.
                            alternative_winner_id=ev.get("alternative_winner_id"),
                            baseline_winner_id=ev.get("baseline_winner_id"),
                            stability=(
                                FlipStabilityBandV2(**stability_raw) if stability_raw else None
                            ),
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

            # S2/F-1: bootstrap-blend method_version resolved ONCE per request. It
            # gains a "+env-override" suffix when the STABILITY_CV_* env levers have
            # moved the CV boundaries away from their defaults (a non-standard
            # method); env cannot change mid-request, so one resolution is exact.
            bootstrap_method_version = get_confidence_method_version()

            factor_sensitivity = []
            for fs in v1_response.factor_sensitivity:
                # Confidence figure + source + S2 disclosure marker, resolved
                # together (bootstrap-derived, else graph-structural fallback). The
                # marker is always populated here — both branches yield a float —
                # and the F-3 model_validator on FactorSensitivityV2 remains the
                # fail-loud backstop for the confidence<->marker iff-invariant.
                confidence = _resolve_factor_confidence(fs, bootstrap_method_version)

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
                        confidence=confidence.value,
                        confidence_source=confidence.source,
                        confidence_provenance=confidence.provenance,
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
            p_win_sensitivity=v1_response.p_win_sensitivity,  # S2 relabel (D-23.8)
            factor_evppi=v1_response.factor_evppi,  # S2 regression EVPPI (D-23.8)
            factor_evpc=v1_response.factor_evpc,  # S4 value-of-control (D-23.8)
        )

        # B3: surface auto-noise disclosure on the V2 envelope so PLoT can read it
        # without consulting V1 _metadata. metadata is a required field on
        # RobustnessResponseV2, so this is always a concrete bool here; the
        # Optional on the V2 envelope covers error paths where build() runs
        # without metadata having been populated.
        #
        # Arch step 1 (2026-07-26): the same call derives
        # `sample_population_provenance`, which says WHICH metrics each
        # population produced — the boolean alone cannot, and one envelope
        # carries both pre-noise CRN metrics and post-noise marginal ones. The
        # constraint-node mix already disclosed by CONSTRAINT_SAMPLES_UNNOISED is
        # threaded in so the two disclosures agree by construction.
        unnoised_constraint_nodes = [
            node_id
            for warning in (v1_response.inference_warnings or [])
            if warning.code == "CONSTRAINT_SAMPLES_UNNOISED"
            for node_id in (warning.detail or {}).get("node_ids", [])
        ]
        builder.set_auto_noise_applied(
            v1_response.metadata.auto_noise_applied, unnoised_constraint_nodes
        )

        # B3-S1: surface the correlated-factors disclosure (Gaussian copula method,
        # mandatory tail-independence caveat, any Higham PSD projection, suppressed-
        # attribution manifest). None when correlation is inactive → absent on wire.
        builder.set_correlation_model(v1_response.correlation_model)

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

        # ROADMAP 2.228-F3: per-root-factor flip thresholds (additive;
        # request-gated by include_factor_flips). The analyzer emits dict rows
        # like the other factor blocks; the WIRE model is typed, so a malformed
        # row fails loud here rather than reaching a consumer as an untyped bag.
        if v1_response.factor_flip_values is not None:
            builder.set_factor_flip_values(
                [FactorFlipValueV2(**row) for row in v1_response.factor_flip_values]
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
    http_request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
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
                http_request=http_request,
                x_request_id=request_id,
                x_api_key=x_api_key,
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
