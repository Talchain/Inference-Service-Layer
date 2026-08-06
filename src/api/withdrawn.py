"""
Instrumented typed refusal for ISL routes that cannot compute what they claim.

A route reaches this helper when its published numbers were shown NOT to be
derived from the request — they were constants, random draws, keyword matches,
or arithmetic over a fabricated input — while being presented to the caller as
a computed analysis, often behind a model card naming a real method.

Every route below was confirmed at the handler bytes by the science-expansion
triage of 2026-08-07 (ROADMAP 2.691) and re-verified at this tip (ROADMAP
2.704). The reason constants in this module ARE those findings: each one names
the specific fabrication, so an integrator reading a 501 body learns what was
wrong rather than merely that something was.

WHY REFUSE RATHER THAN DELETE
-----------------------------
These routers are all currently DARK (their ``include_router`` lines in
``src/api/main.py`` are commented out), so nothing can reach them today. That
is exactly why quarantining is cheap now and why it must happen now: the whole
hazard is a FUTURE wiring job. Someone reading ``team_aligner.align()`` or a
"Bayesian Teaching (Zhu et al.)" docstring, in good faith, mounts the router
and ships invented numbers under a real citation.

Deleting outright would answer 404 — indistinguishable from a typo, and it
destroys the record of what the capability was and why it went away. A typed
501 at the handler says: this existed, it was withdrawn, here is the specific
reason, and retrying will not help. Crucially, the refusal is IN the handler,
so it survives the router being mounted: mounting these routers later still
cannot serve a number.

(Routes whose fabrication was judged not worth preserving even as a refusal —
``deliberation``, ``decision_robustness``, and the phase4 dark trio — were
DELETED under the same ruling rather than refused. The disposition split is
recorded in the ROADMAP 2.704 lane report.)

WHY 501
-------
The server understood the request and is declining to implement it.
``retryable=False`` — an identical retry produces an identical refusal.

WHY A JSONResponse RATHER THAN ``raise HTTPException``
------------------------------------------------------
``main.py``'s HTTPException handler maps unrecognised status codes onto
``ErrorCode.VALIDATION_ERROR`` with ``reason="http_error"`` — it has no 501
branch, so raising would silently FLATTEN the typed code this module exists to
publish. Returning the response directly means the wire shape cannot be
rewritten by a handler that does not know about withdrawal. The body is still
ISL's canonical ``ErrorResponse`` (Olumi Error Schema v1.0), so nothing
downstream needs a special parser.

A KNOWN AND DELIBERATE LIMIT
----------------------------
FastAPI validates the declared request body BEFORE the handler runs, so a
malformed body to a withdrawn route yields 422, not this 501. PLoT's
equivalent refuses before validation. The difference is immaterial to the
guarantee that matters here — neither path serves a number — and keeping the
typed request model preserves the OpenAPI record of what the route accepted.
Pinned by ``test_withdrawn_routes.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from src.models.responses import ErrorCode, ErrorResponse, RecoveryHints
from src.utils.business_metrics import capability_withdrawn_refusals_total

logger = logging.getLogger(__name__)


# --- Reason constants: one per confirmed fabrication class -------------------
# These are the triage findings, not paraphrases of them. Each cites the code
# that was read. Kept as constants so a test can assert the exact wire string
# and so two routes sharing a fabrication cannot drift into two descriptions.

STAMPED_CONSTANTS_REASON = (
    "route published hardcoded constants as computed analysis: assumption "
    "'sensitivity' was anchored on literals 0.85/0.45/0.5 derived from graph "
    "topology position and stamped algorithm='topology_sensitivity'; see "
    "science-expansion triage 2026-08-07"
)

ASSERTED_COVERAGE_REASON = (
    "route published an interval as a CONFORMAL prediction without performing "
    "conformal prediction: the bounds were belief*100 +/- (1-belief)*50 scaled "
    "by node in-degree, with confidence_level=0.90 hardcoded and stamped "
    "algorithm='conformal_prediction'; there was no calibration set, no "
    "nonconformity score and no quantile, so the stated coverage was asserted "
    "rather than achieved; see science-expansion triage 2026-08-07"
)

INVENTED_FEASIBILITY_REASON = (
    "route published template prose with invented feasibility constants "
    "(0.75/0.60/0.85) as a contrastive analysis; note the real minimal-"
    "intervention implementation is explain/contrastive, which this route's "
    "name collides with; see science-expansion triage 2026-08-07"
)

RANDOM_OUTCOMES_REASON = (
    "route published np.random.uniform() draws as teaching-example scenario "
    "outcomes behind an 'Optimal Bayesian Teaching (Zhu et al.)' model card; "
    "the teaching-value score is a heuristic over those random numbers; see "
    "science-expansion triage 2026-08-07"
)

KEYWORD_MATCH_REASON = (
    "route published keyword substring matching as team-alignment analysis: "
    "satisfaction score is the fraction of priority keywords appearing in "
    "option attribute text x100 (default 50.0), and trade-offs are hardcoded "
    "prose; see science-expansion triage 2026-08-07"
)

FABRICATED_QUERY_REASON = (
    "route fed a fabricated empty query to a real Bayesian updater: the "
    "elicitation query is a placeholder with NO scenarios, and the likelihood "
    "is computed FROM those scenarios, so the posterior cannot reflect what "
    "was asked; the storage seam was never built; see science-expansion "
    "triage 2026-08-07"
)

DEFAULT_OUTCOME_REASON = (
    "route published a hardcoded 50000.0 default outcome whenever equation "
    "string-parsing failed, and reported analysis FAILURE as maximal "
    "robustness (robustness_score=1.0, 'Analysis failed - assuming robust'); "
    "see science-expansion triage 2026-08-07"
)

INDEX_DIRECTION_REASON = (
    "route's default discovery engine assigned causal edge DIRECTION by "
    "variable index rather than by any causal criterion, so the arrow "
    "reflects column order in the input; see science-expansion triage "
    "2026-08-07"
)


def refuse_withdrawn(
    route: str,
    reason: str,
    request: Optional[Request] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """
    Refuse a withdrawn capability, and record that someone asked.

    Instrumentation is a reason the handler still exists rather than being
    deleted, so it happens first and unconditionally; a metrics failure must
    never convert a clean refusal into a 500.

    Args:
        route: Stable route identifier, e.g. "/api/v1/team/align". Used as the
            metric label and the log field, so it must not embed request data.
        reason: One of the module-level ``*_REASON`` constants.
        request: Optional FastAPI request, used only for correlation headers.
        request_id: Explicit correlation id; wins over the request headers.

    Returns:
        JSONResponse: 501 carrying ISL's canonical ErrorResponse body.
    """
    try:
        capability_withdrawn_refusals_total.labels(route=route).inc()
    except Exception:  # pragma: no cover - instrumentation must never break the refusal
        logger.debug("withdrawn_refusal_metric_failed", exc_info=True)

    resolved_request_id = request_id
    if resolved_request_id is None and request is not None:
        resolved_request_id = request.headers.get("X-Request-Id") or request.headers.get(
            "X-Trace-Id"
        )

    logger.warning(
        "capability_withdrawn",
        extra={
            "evt": "capability_withdrawn",
            "route": route,
            "code": ErrorCode.CAPABILITY_WITHDRAWN.value,
            "reason": reason,
            "request_id": resolved_request_id,
        },
    )

    # `ErrorResponse.request_id` is a non-optional str with a default_factory,
    # so an explicit None fails validation — omit the key instead and let the
    # model mint a correlation id.
    optional_fields = (
        {"request_id": resolved_request_id} if resolved_request_id is not None else {}
    )

    error_response = ErrorResponse(
        code=ErrorCode.CAPABILITY_WITHDRAWN.value,
        message=f"{route} has been withdrawn: {reason}",
        reason="capability_withdrawn",
        recovery=RecoveryHints(
            hints=[
                "This capability was withdrawn because its output was fabricated, "
                "not computed. Retrying will not help.",
                "Do not re-mount this route expecting numbers; the underlying "
                "computation was never implemented.",
            ],
            suggestion=(
                "Use the live analysis surface (/api/v1/robustness/*) or raise a "
                "ROADMAP row to build this capability honestly."
            ),
        ),
        retryable=False,
        source="isl",
        **optional_fields,
    )

    return JSONResponse(
        status_code=501,
        content=error_response.model_dump(exclude_none=True),
    )
