"""A3 Codex-fix-D / F6 (D-23.15): a client STRUCTURAL EQUATION is private.

A structural equation can encode a client's proprietary model structure, factor
labels, and constants. On a malformed / evaluation-failing equation the engine
previously wrote the RAW equation text, in clear, to THREE records:

  1. ``_evaluate_equation`` engine WARNING  (``Syntax error in equation '<eq>'``)
  2. ``analyze()`` engine WARNING           (``counterfactual_analysis_failed`` -> str(e))
  3. route ``raise_invalid_input`` WARNING  (``counterfactual_invalid_input`` -> str(exc))

...and echoed it back to the client as the 422 ``message``.

The fix (D-23.15) raises a typed ``CounterfactualClientInputError`` carrying a
SAFE public message (variable id + category, never the equation), a stable
``code``, the offending ``variable`` and a short ``equation_hash`` for
correlation; the ROUTE is the single log owner (only layer with request_id);
the engine layers propagate without re-logging.

Every absence assertion below is a trap-#13 positive control: it first proves
the capture harness CAN SEE the sentinel (emitting it through the same loggers),
so the "sentinel is absent" assertion is non-vacuous.
"""

import hashlib
import logging
import traceback

import pytest

# The string sentinel appears ONLY inside the client equation, so its presence in
# any log record or client message is unambiguous leakage of client model text.
EQUATION_SENTINEL = "SYNTHETIC_CLIENT_MODEL_SENTINEL"
MALFORMED_EQUATION = f"{EQUATION_SENTINEL} +* X"  # `+*` -> ast.parse SyntaxError
EQUATION_HASH = hashlib.sha256(MALFORMED_EQUATION.encode("utf-8")).hexdigest()[:12]

# Loggers on the three former leak sites.
_LEAK_LOGGERS = ("src.services.counterfactual_engine", "src.api.causal")


def _rendered(rec: logging.LogRecord) -> str:
    """Everything an operator could ever read off one record: the interpolated
    message, every structured/extra attribute (``repr(__dict__)``), and the
    FULLY FORMATTED record including any exc_info traceback (a SyntaxError's
    formatted traceback prints its ``.text`` — the source line — so this catches
    the equation leaking via a chained/exc_info path too)."""
    parts = [str(rec.msg), rec.getMessage(), repr(rec.__dict__)]
    try:
        parts.append(logging.Formatter().format(rec))
    except Exception:  # pragma: no cover - formatting must never mask the scan
        pass
    return "\n".join(parts)


def _records_leaking(records, needle):
    return [r for r in records if needle in _rendered(r)]


def _malformed_request():
    return {
        "model": {
            "variables": ["X", "Y"],
            "equations": {"Y": MALFORMED_EQUATION},
            "distributions": {},
        },
        "intervention": {"X": 1.0},
        "outcome": "Y",
    }


class TestF6EquationTextPrivacy:
    """F6 / D-23.15 — the raw equation must not reach ANY log or the client wire."""

    @pytest.mark.asyncio
    async def test_equation_text_absent_from_all_logs_and_client_message(self, cf_client, caplog):
        """RED at HEAD: the sentinel equation appears in 3 log records AND the
        client 422 message. GREEN: 0 log records carry it, and the client sees a
        422 naming the VARIABLE + category, never the equation."""
        caplog.set_level(logging.DEBUG)

        # --- positive control FIRST: the harness is NOT blind to the sentinel ---
        for name in _LEAK_LOGGERS:
            logging.getLogger(name).warning(
                "a3_f6_probe_control", extra={"error": f"leaked: {MALFORMED_EQUATION}"}
            )
        probe = _records_leaking(caplog.records, EQUATION_SENTINEL)
        assert probe, (
            "positive control failed: harness cannot SEE the sentinel even when "
            "deliberately logged — an absence assertion here would be vacuous"
        )
        caplog.clear()

        # --- production request with the malformed client equation ---
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=_malformed_request())

        # client wire: a clean 422 that NAMES the variable + category, no equation
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "Y" in body["message"], "422 must still name the offending variable"
        assert "equation" in body["message"].lower()
        assert "syntax" in body["message"].lower()
        assert (
            EQUATION_SENTINEL not in body["message"]
        ), "raw client equation echoed to the client 422 message"

        # logs: the sentinel appears in NO record (the core F6 assertion)
        leaks = _records_leaking(caplog.records, EQUATION_SENTINEL)
        assert not leaks, f"raw client equation leaked to {len(leaks)} log record(s): " + "; ".join(
            f"{r.name}:{r.levelname}:{r.msg!r}" for r in leaks
        )

        # the single OWNER record (route) carries enough to correlate/debug —
        # reads REAL structured data, so the absence assertions above are not vacuous
        owner = [r for r in caplog.records if r.msg == "counterfactual_invalid_input"]
        assert owner, "expected the route's counterfactual_invalid_input owner log"
        rec = owner[0]
        assert getattr(rec, "code", None) == "cf_equation_syntax_invalid"
        assert getattr(rec, "variable", None) == "Y"
        assert getattr(rec, "equation_hash", None) == EQUATION_HASH
        assert getattr(rec, "category", None) == "equation_syntax_invalid"
        assert rec.levelname == "WARNING", "a client input defect must not page as ERROR"

    @pytest.mark.asyncio
    async def test_no_duplicate_owner_records(self, cf_client, caplog):
        """One owner record only: the engine layers propagate without re-logging
        (kills the 3x duplication). Exactly one WARNING carries the redacted
        equation fields."""
        caplog.set_level(logging.DEBUG)
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=_malformed_request())
        assert resp.status_code == 422, resp.text
        with_fields = [
            r for r in caplog.records if getattr(r, "equation_hash", None) == EQUATION_HASH
        ]
        assert len(with_fields) == 1, (
            "expected exactly ONE owner record with the equation fields, got "
            f"{[(r.name, r.msg) for r in with_fields]}"
        )
        # and the range-added engine duplicate is gone
        eng_dupes = [
            r
            for r in caplog.records
            if r.name == "src.services.counterfactual_engine"
            and r.msg == "counterfactual_analysis_failed"
        ]
        assert not eng_dupes, "engine still re-logs counterfactual_analysis_failed (duplicate)"

    def test_eval_failure_branch_also_redacted(self, caplog):
        """The second branch (parses, fails to evaluate) is redacted too: an
        undefined-variable equation carrying the sentinel raises the typed error
        with a safe message and no engine log of the equation."""
        from src.models.requests import CounterfactualRequest, StructuralModel
        from src.services.counterfactual_engine import (
            CounterfactualClientInputError,
            CounterfactualEngine,
        )

        engine = CounterfactualEngine()
        eq = f"{EQUATION_SENTINEL}_UNDEFINED + X"  # parses, but the name is undefined
        request = CounterfactualRequest(
            model=StructuralModel(variables=["X", "Y"], equations={"Y": eq}, distributions={}),
            intervention={"X": 1.0},
            outcome="Y",
            context=None,
        )
        caplog.set_level(logging.DEBUG, logger="src.services.counterfactual_engine")
        with pytest.raises(CounterfactualClientInputError) as ei:
            engine.analyze(request)
        exc = ei.value
        assert EQUATION_SENTINEL not in str(exc)
        assert "Y" in str(exc)
        assert exc.variable == "Y"
        assert exc.code == "cf_equation_eval_failed"
        # engine emitted no record carrying the equation text
        assert not _records_leaking(caplog.records, EQUATION_SENTINEL)
        # Chain-suppression pin (adversarial M3, 2026-07-24): `from None` is
        # load-bearing — the suppressed underlying exception here is
        # ValueError("Unknown variable: <sentinel>_UNDEF"), so if a refactor drops
        # `from None`, the rendered exc_info chain re-leaks the equation fragment.
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True
        rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        assert (
            EQUATION_SENTINEL not in rendered
        ), "underlying exception (equation fragment) leaked into the rendered chain"

    def test_syntax_branch_chain_suppressed(self):
        """Chain-suppression pin for the SYNTAX branch (adversarial M3,
        2026-07-24): a SyntaxError's `.text` IS the raw equation line, and
        `traceback.format_exception` prints it — so `from None` is the mechanism
        that keeps the equation out of any exc_info rendering. Pin it: dropping
        `from None` must turn this RED."""
        import numpy as np

        from src.services.counterfactual_engine import (
            CounterfactualClientInputError,
            CounterfactualEngine,
        )

        engine = CounterfactualEngine()
        with pytest.raises(CounterfactualClientInputError) as ei:
            engine._evaluate_equation(MALFORMED_EQUATION, {"X": np.array([1.0])}, var_name="Y")
        exc = ei.value
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True
        rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        assert (
            EQUATION_SENTINEL not in rendered
        ), "SyntaxError.text (the raw equation) leaked into the rendered chain"

    def test_typed_error_is_valueerror_subclass(self):
        """The typed error must subclass ValueError so the existing route/engine
        ``except ValueError`` mappings (D-12(cf) -> 422) still catch it and the
        client wire status is unchanged."""
        from src.services.counterfactual_engine import CounterfactualClientInputError

        assert issubclass(CounterfactualClientInputError, ValueError)

    def test_genuine_internal_fault_logs_error_trace_no_client_equation(self, caplog):
        """A genuine INTERNAL fault still logs at ERROR with a traceback (enough
        to debug) and carries NO client equation text."""
        from src.models.requests import CounterfactualRequest, StructuralModel
        from src.services.counterfactual_engine import CounterfactualEngine

        engine = CounterfactualEngine()
        secret_eq = "2 * SecretModelStructureVar"

        def _boom(*a, **k):
            raise RuntimeError("synthetic internal fault")

        engine._compute_prediction = _boom
        request = CounterfactualRequest(
            model=StructuralModel(
                variables=["SecretModelStructureVar", "Y"],
                equations={"Y": secret_eq},
                distributions={},
            ),
            intervention={"SecretModelStructureVar": 5.0},
            outcome="Y",
            context=None,
        )
        caplog.set_level(logging.DEBUG, logger="src.services.counterfactual_engine")
        with pytest.raises(RuntimeError):
            engine.analyze(request)
        errs = [
            r
            for r in caplog.records
            if r.msg == "counterfactual_analysis_failed" and r.levelname == "ERROR"
        ]
        assert errs, "genuine internal fault must still log ERROR"
        assert any(r.exc_info is not None for r in errs), "ERROR must carry a traceback"
        assert not _records_leaking(
            caplog.records, secret_eq
        ), "internal-fault ERROR log leaked client equation text"
