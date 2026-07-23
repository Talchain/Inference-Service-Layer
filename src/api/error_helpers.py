"""Shared error-mapping helpers for API route handlers.

Small, deliberately un-magical helpers that dedup identical error-handling
*tails* across route handlers WITHOUT hiding the routing decision. The
per-route ``except`` blocks (and their load-bearing proof comments about which
client inputs are being mapped) stay at each call site — this module only
carries the byte-identical body those blocks share.
"""

import logging
from typing import NoReturn

from fastapi import HTTPException


def raise_invalid_input(
    logger: logging.Logger, event: str, request_id: str, exc: ValueError
) -> NoReturn:
    """Log a client-input ``ValueError`` at WARNING and re-raise it as a 422.

    Shared body for the per-route D-12 ``ValueError`` -> 422 mappings
    (sequential analysis, counterfactual, robustness v2). Each call site keeps
    its own ``except ValueError`` block and proof comment documenting *which*
    engine-level input defects are client errors; this only replaces the
    identical warning-log + 422-raise tail those blocks duplicated.

    ``logger`` is passed in so the emitted record keeps the caller's module
    logger name (``src.api.phase4`` / ``causal`` / ``robustness``) — the log
    output is byte-identical to the inlined form.

    Deliberately NOT a global ``add_exception_handler``: route-local visibility
    of which inputs map to 422 is intentional (adjudicated).
    """
    logger.warning(event, extra={"request_id": request_id, "error": str(exc)})
    raise HTTPException(status_code=422, detail=str(exc)) from exc
