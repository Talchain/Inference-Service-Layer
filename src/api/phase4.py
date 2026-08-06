"""
Phase 4: Sequential Decisions API.

Provides ONE endpoint: sequential decision analysis (backward induction), on
``sequential_router``, which main.py mounts as POST /api/v1/analysis/sequential.

RETIRED (ROADMAP 2.704): the dark ``router`` that carried
``/conditional-recommend``, ``/policy-tree`` and ``/stage-sensitivity`` is
DELETED, along with its handlers and their engine entry points. ROADMAP 2.363
ruled all three RETIRE and the verdicts were re-confirmed at this tip:

- ``/conditional-recommend``: the recommendation probability was a literal
  ``0.15  # Arbitrary``.
- ``/policy-tree``: the builder never recursed — children were flat edge dicts
  and ``_count_tree_nodes`` was self-commented "Simplified - doesn't recurse
  into children", so the "tree" was one level deep whatever the graph.
- ``/stage-sensitivity``: the flip threshold was hardcoded
  ``1.0 - variation_range`` and probability perturbation was un-renormalised.

These were deleted rather than refused because there is nothing to preserve a
route for: 2.363 already ruled them retired, and unlike the withdrawn routes in
``src/api/withdrawn.py`` they have no integrator worth telling apart from a
typo. The live sequential mount is deliberately untouched.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from src.api.error_helpers import raise_invalid_input
from src.models.metadata import create_response_metadata
from src.models.requests import SequentialAnalysisRequest
from src.models.responses import SequentialAnalysisResponse
from src.services.sequential_decision import SequentialDecisionEngine

# Selective mount (R-12): ONLY the sequential-analysis route is runtime-verified
# (A2 flip: honest engine + served-path value pins + D-12 422 mapping) and goes
# live. It keeps its own router name so main.py mounts exactly
# POST /api/v1/analysis/sequential. The sibling dark `router` that used to sit
# here was retired under 2.704 — do NOT reintroduce one to hang new routes on
# without its own runtime verification.
sequential_router = APIRouter()

logger = logging.getLogger(__name__)

# Initialize services
sequential_engine = SequentialDecisionEngine()

# F-4: the former module-level `_policy_cache` was an unbounded, never-evicted,
# write-only dict — populated on every live sequential request (key derived from a
# client-controllable X-Request-Id), with zero readers (the dark policy-tree route
# recomputed rather than reading it). With the mount that was a monotonic memory
# leak. It is removed, and with 2.704 its only would-be reader is gone too.


@sequential_router.post(
    "/sequential",
    response_model=SequentialAnalysisResponse,
    summary="Analyze sequential decision problem",
    description="""
    Solve multi-stage decision problems using backward induction.

    Many strategic decisions are sequential: "If we launch, then we'll need to
    decide on pricing. If pricing works, we'll decide on expansion."
    Optimal strategy requires reasoning backward from final outcomes.

    **Algorithm:**
    1. Start from final stage (terminal payoffs)
    2. For each state at stage T, compute optimal action and value
    3. Move backward, computing optimal action given immediate payoff + continuation value
    4. Repeat until stage 0

    **One decision per stage:** each stage may declare at most one decision node.
    The engine builds exactly one decision rule per stage and the response has no
    field in which a dropped decision could be disclosed, so more than one is
    rejected (422, `MULTI_DECISION_STAGE_UNSUPPORTED`) rather than truncated.

    **Not returned:** `value_of_flexibility` and `sensitivity_to_timing` are
    omitted (arch step 1, 2026-07-26) — the committed leg averaged a chance
    node's branches while the flexible leg probability-weighted them, so the
    difference measured an estimator gap, not the value of waiting. See
    `SequentialAnalysisResponse`.

    **Use when:** Planning multi-stage strategies with uncertainty resolution.
    """,
    responses={
        200: {"description": "Sequential analysis completed successfully"},
        400: {"description": "Invalid input (e.g., invalid graph structure)"},
        500: {"description": "Internal computation error"},
    },
)
async def analyze_sequential_decision(
    request: SequentialAnalysisRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
) -> SequentialAnalysisResponse:
    """
    Analyze a sequential decision problem.

    Args:
        request: Sequential analysis request with graph and stages

    Returns:
        SequentialAnalysisResponse with optimal policy and stage analyses
    """
    request_id = x_request_id or f"req_{uuid.uuid4().hex[:12]}"

    try:
        logger.info(
            "sequential_analysis_request",
            extra={
                "request_id": request_id,
                "num_nodes": len(request.graph.nodes),
                "num_edges": len(request.graph.edges),
                "num_stages": len(request.stages),
                "discount_factor": request.discount_factor,
                "risk_tolerance": request.risk_tolerance,
            },
        )

        result = sequential_engine.analyze(request)

        # F-4: no _policy_cache write — the former write-only cache was an unbounded
        # leak with no live reader (see the module-level note). policy-tree recomputes.

        logger.info(
            "sequential_analysis_completed",
            extra={
                "request_id": request_id,
                "expected_value": result.optimal_policy.expected_total_value,
                "num_policy_stages": len(result.optimal_policy.stages),
            },
        )

        # Add metadata. config_details no longer advertises `monte_carlo_samples`
        # (C2, F-3, 2026-07-23): the key was DELETED from generate_config_details as a
        # fabricated disclosure (no live path uses MAX_MONTE_CARLO_ITERATIONS as its
        # budget). The sequential-decision engine is deterministic (expectimax over the
        # decision tree — it draws no Monte Carlo samples) regardless.
        result.metadata = create_response_metadata(request_id)

        return result

    except HTTPException:
        raise
    except ValueError as e:
        # D-12: the engine fails loud (ValueError) on client-input defects the
        # request model cannot catch — dangling edges (to_node not validated
        # against nodes), cycles (backward induction requires a DAG), or an
        # unsupported node type. These are client errors, not internal failures.
        # Fail closed with 422 (matching the robustness v2 handler) so a
        # dangling-edge-behind-p=0 graph surfaces as a clean validation error,
        # never a 500 or a plausible-looking 200.
        raise_invalid_input(logger, "sequential_analysis_invalid_input", request_id, e)
    except Exception as e:
        logger.error(
            "sequential_analysis_error",
            extra={"request_id": request_id, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze sequential decision. Check logs for details.",
        )
