"""
Team alignment endpoint — WITHDRAWN (ROADMAP 2.704).

The route is retained as an instrumented typed 501 rather than deleted, so
that mounting this router later still cannot serve a number, and so that any
caller that appears is recorded. See ``src/api/withdrawn.py`` for the full
rationale.

What was wrong (confirmed at the handler bytes, science-expansion triage
2026-08-07, re-verified at this tip): ``TeamAligner._calculate_satisfaction``
scored an option against a stakeholder's priorities by checking whether the
priority keyword appeared as a SUBSTRING of the option's attribute text,
multiplied the hit fraction by 100, and defaulted to 50.0; ``_identify_tradeoff``
returned hardcoded prose ("Some priorities" / "Alternative benefits") with a
special case keyed on a literal "speed" attribute. The result was presented as
team-alignment analysis with satisfaction scores and conflict resolutions.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.withdrawn import KEYWORD_MATCH_REASON, refuse_withdrawn

router = APIRouter()
logger = logging.getLogger(__name__)

TEAM_ALIGN_ROUTE = "/api/v1/team/align"


@router.post(
    "/align",
    summary="[WITHDRAWN] Find team alignment",
    description="""
    **WITHDRAWN (ROADMAP 2.704) — this route answers 501 and computes nothing.**

    It previously reported stakeholder "satisfaction scores" and trade-offs.
    Those scores were keyword substring matches against option attribute text
    (x100, default 50.0) and the trade-off text was hardcoded. No alignment
    analysis was performed.

    Rebuilding this capability honestly requires a real preference model over
    stakeholders and options — raise a ROADMAP row rather than re-mounting.
    """,
    responses={
        501: {"description": "Capability withdrawn — output was fabricated, not computed"},
    },
)
async def align_team(request: Request) -> JSONResponse:
    """Refuse: team alignment was never computed.

    Takes the raw request rather than a validated model on purpose — no input
    is "valid" for a capability that does not exist, and a 422 would imply a
    well-formed body would have been answered.
    """
    return refuse_withdrawn(
        route=TEAM_ALIGN_ROUTE,
        reason=KEYWORD_MATCH_REASON,
        request=request,
    )
