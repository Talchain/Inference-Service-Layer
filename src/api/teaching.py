"""
Bayesian teaching endpoint — WITHDRAWN (ROADMAP 2.704).

The route is retained as an instrumented typed 501 rather than deleted, so
that mounting this router later still cannot serve a number, and so that any
caller that appears is recorded. See ``src/api/withdrawn.py`` for the full
rationale.

What was wrong (confirmed at the handler bytes, science-expansion triage
2026-08-07, re-verified at this tip): ``BayesianTeacher`` generated its
teaching examples from templates whose scenario outcomes were
``np.random.uniform(0, 100)`` — random numbers — and then scored each example's
"teaching value" as ``0.4*novelty + 0.4*clarity + 0.2*relevance`` over those
random outcomes. All of it was presented behind a docstring citing "Optimal
Bayesian Teaching (Zhu et al.)".

This route is a particular hazard for the coaching roadmap: it is exactly where
a lane looking for dormant decision-science teaching code would land, and its
"concepts" (confounding, trade-offs) read as a ready-made bias-education leg.
It must never be presented as one.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.withdrawn import RANDOM_OUTCOMES_REASON, refuse_withdrawn

router = APIRouter()
logger = logging.getLogger(__name__)

TEACHING_TEACH_ROUTE = "/api/v1/teaching/teach"


@router.post(
    "/teach",
    summary="[WITHDRAWN] Generate teaching examples",
    description="""
    **WITHDRAWN (ROADMAP 2.704) — this route answers 501 and computes nothing.**

    It previously returned "pedagogically optimized" teaching examples with
    scenario outcomes and teaching-value scores. The scenario outcomes were
    `np.random.uniform(0, 100)` draws and the scores were a heuristic over
    those random numbers, published under a "Bayesian Teaching (Zhu et al.)"
    model card.

    Do not treat this as a dormant coaching or bias-education capability.
    Building one honestly is a new piece of work — raise a ROADMAP row.
    """,
    responses={
        501: {"description": "Capability withdrawn — output was fabricated, not computed"},
    },
)
async def generate_teaching_examples(request: Request) -> JSONResponse:
    """Refuse: teaching examples were random draws, not computed pedagogy.

    Takes the raw request rather than a validated model on purpose — no input
    is "valid" for a capability that does not exist, and a 422 would imply a
    well-formed body would have been answered.
    """
    return refuse_withdrawn(
        route=TEACHING_TEACH_ROUTE,
        reason=RANDOM_OUTCOMES_REASON,
        request=request,
    )
