"""Shared error-mapping helpers for API route handlers.

Small, deliberately un-magical helpers that dedup identical error-handling
*tails* across route handlers WITHOUT hiding the routing decision. The
per-route ``except`` blocks (and their load-bearing proof comments about which
client inputs are being mapped) stay at each call site — this module only
carries the byte-identical body those blocks share.
"""

import logging
from typing import Any, Dict, NoReturn

from fastapi import HTTPException


def raise_invalid_input(
    logger: logging.Logger, event: str, request_id: str, exc: ValueError
) -> NoReturn:
    """Log a client-input ``ValueError`` at WARNING (the SINGLE owner record) and
    re-raise it as a 422.

    Shared body for the per-route D-12 ``ValueError`` -> 422 mappings
    (sequential analysis, counterfactual, robustness v2). Each call site keeps
    its own ``except ValueError`` block and proof comment documenting *which*
    engine-level input defects are client errors; this only replaces the
    identical warning-log + 422-raise tail those blocks duplicated.

    ``logger`` is passed in so the emitted record keeps the caller's module
    logger name (``src.api.phase4`` / ``causal`` / ``robustness``) — the log
    output is byte-identical to the inlined form.

    F6 / D-23.15 — SINGLE log owner + redaction. This is the one layer with a
    ``request_id``, so the engine layers propagate client-input defects WITHOUT
    re-logging and this is the sole record for a client-input 422. When the
    exception is a redaction-aware typed error (it exposes ``safe_log_extra()``,
    e.g. ``CounterfactualClientInputError``), its REDACTED structured fields
    (``code`` / ``variable_hash`` / ``equation_hash`` / ``category`` — never raw
    client text or identifiers) are merged into the record, and the LOGGED
    ``error`` field is the stable ``code`` — NOT ``str(exc)``: the public
    message deliberately names the client's variable (their own data, useful on
    the 422 wire), but that same name must not ride into the log via the
    message (Codex re-confirm F6: the raw sentinel appeared in the owner record
    exactly this way). The client-facing 422 ``detail`` keeps the full public
    message — wire behaviour unchanged. For a plain ``ValueError`` the
    behaviour is byte-identical to before (``extra={request_id, error:
    str(exc)}``, ``detail=str(exc)``), so the sequential/robustness callers
    (whose ValueErrors carry no client model text — swept, D-23.18) and every
    already-safe counterfactual 422 are unchanged.

    Deliberately NOT a global ``add_exception_handler``: route-local visibility
    of which inputs map to 422 is intentional (adjudicated).
    """
    safe_log_extra = getattr(exc, "safe_log_extra", None)
    if callable(safe_log_extra):
        # Redaction-aware typed error: the log carries the stable code + the
        # safe structured digests, never the message (which names the variable).
        extra: Dict[str, Any] = {
            "request_id": request_id,
            "error": getattr(exc, "code", type(exc).__name__),
        }
        extra.update(safe_log_extra())
    else:
        extra = {"request_id": request_id, "error": str(exc)}
    logger.warning(event, extra=extra)
    raise HTTPException(status_code=422, detail=str(exc)) from exc
