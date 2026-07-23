"""A3 counterfactual-honesty remediation (Codex deep-review, 2026-07-22).

RED-first tests for the counterfactual engine/route hardening:

* F2  — do(X) + observe(X) is contradictory: an intervention key that also
        appears in `context` is rejected (was: context silently overwrote the
        intervention, so a declared do(X=10) was computed at X=context, e.g. 6).
* F3d — an empty intervention on the interventional /counterfactual route is
        rejected (a counterfactual with no do() is not a counterfactual).
* F11 — raw intervention VALUES must never reach the structured logs; they are
        redacted to sorted key names + a count (the R-004 correlation key stays
        the canonical request hash the engine already computes).

Each F11 test is a trap-#13 positive control: it first proves the capture
harness CAN SEE the sentinel value (emitting it through the same logger), then
asserts the production log does NOT carry it.
"""

import logging

import pytest

from src.models.requests import ConformalCounterfactualRequest

# `cf_client` (LIVE counterfactual + conformal routes with the production
# exception handlers), the F11 `redaction_sentinel`, and the
# `assert_harness_can_see_value` positive-control helper are shared fixtures in
# tests/integration/conftest.py (C4 dedup).


# ===========================================================================
# F2 — do(X) and observe(X) is contradictory
# ===========================================================================
class TestF2InterventionContextConflict:
    @pytest.mark.asyncio
    async def test_do_and_observe_same_variable_returns_422(self, cf_client):
        """RED at HEAD: intervention {X:10} + context {X:3} on Y=2X returns 200
        with point_estimate 6 (=2*3) — the context write silently overwrote the
        declared do(X=10). do(X) and observe(X) are contradictory; reject -> 422.
        """
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "2 * X"},
                "distributions": {},
            },
            "intervention": {"X": 10.0},
            "context": {"X": 3.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "X" in body["message"]

    @pytest.mark.asyncio
    async def test_disjoint_intervention_and_context_still_200(self, cf_client):
        """Positive control: an intervention and context on DIFFERENT variables
        is legitimate (do(X), observe(Z)) and must still 200 — the guard rejects
        only the do/observe COLLISION, not any use of context."""
        request = {
            "model": {
                "variables": ["X", "Z", "Y"],
                "equations": {"Y": "X + Z"},
                "distributions": {},
            },
            "intervention": {"X": 5.0},
            "context": {"Z": 3.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prediction"]["point_estimate"] == pytest.approx(8.0, abs=0.5)


# ===========================================================================
# F3d — an empty intervention is not a counterfactual
# ===========================================================================
class TestF3dEmptyIntervention:
    @pytest.mark.asyncio
    async def test_empty_intervention_returns_422(self, cf_client):
        """RED at HEAD: an empty intervention {} on the interventional
        /counterfactual route is accepted (200). A counterfactual with no do()
        operator is not a counterfactual; reject -> 422 (D-12(cf))."""
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "2 * X"},
                "distributions": {"X": {"type": "normal", "parameters": {"mean": 5, "std": 1}}},
            },
            "intervention": {},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "intervention" in body["message"].lower()


# ===========================================================================
# F11 — redact raw intervention VALUES from the structured logs (3 sites)
# ===========================================================================
# The `redaction_sentinel` value and the `assert_harness_can_see_value` positive
# control live in tests/integration/conftest.py (shared with the intervention-key
# validation suite). Each F11 test is a trap-#13 positive control: it first proves
# the capture harness CAN SEE the sentinel, then asserts the production log omits it.


class TestF11EngineLogRedaction:
    """Engine site (counterfactual_engine.py :98, `counterfactual_analysis_started`)."""

    @pytest.mark.asyncio
    async def test_engine_start_log_redacts_intervention_values(
        self, cf_client, caplog, redaction_sentinel, assert_harness_can_see_value
    ):
        from src.models.requests import CounterfactualRequest, StructuralModel
        from src.services.counterfactual_engine import CounterfactualEngine

        engine = CounterfactualEngine()
        request = CounterfactualRequest(
            model=StructuralModel(
                variables=["SecretPrice", "Y"],
                equations={"Y": "2 * SecretPrice"},
                distributions={},
            ),
            intervention={"SecretPrice": redaction_sentinel},
            outcome="Y",
            context=None,
        )
        with caplog.at_level(logging.INFO, logger="src.services.counterfactual_engine"):
            engine.analyze(request)
            # positive control FIRST — the harness must be able to see the value
            assert_harness_can_see_value(caplog.records, "src.services.counterfactual_engine")

        recs = [r for r in caplog.records if r.msg == "counterfactual_analysis_started"]
        assert recs, "expected the counterfactual_analysis_started log"
        rec = recs[0]
        # redacted structured fields present (reads REAL data, not vacuous)
        assert getattr(rec, "intervention_keys", None) == ["SecretPrice"]
        assert getattr(rec, "intervention_count", None) == 1
        # RED at HEAD: the raw dict field is gone and the value appears nowhere
        assert not hasattr(rec, "intervention"), "raw intervention dict still logged"
        assert str(redaction_sentinel) not in repr(rec.__dict__)
        assert str(int(redaction_sentinel)) not in repr(rec.__dict__)


class TestF11RouteLogRedaction:
    """Route site (causal.py :265, `counterfactual_request`)."""

    @pytest.mark.asyncio
    async def test_counterfactual_request_log_redacts_intervention_values(
        self, cf_client, caplog, redaction_sentinel, assert_harness_can_see_value
    ):
        request = {
            "model": {
                "variables": ["SecretPrice", "Y"],
                "equations": {"Y": "2 * SecretPrice"},
                "distributions": {},
            },
            "intervention": {"SecretPrice": redaction_sentinel},
            "outcome": "Y",
        }
        with caplog.at_level(logging.INFO, logger="src.api.causal"):
            resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
            assert resp.status_code == 200, resp.text
            assert_harness_can_see_value(caplog.records, "src.api.causal")

        recs = [r for r in caplog.records if r.msg == "counterfactual_request"]
        assert recs, "expected the counterfactual_request log"
        rec = recs[0]
        assert getattr(rec, "intervention_keys", None) == ["SecretPrice"]
        assert getattr(rec, "intervention_count", None) == 1
        assert not hasattr(rec, "intervention"), "raw intervention dict still logged"
        assert str(redaction_sentinel) not in repr(rec.__dict__)
        assert str(int(redaction_sentinel)) not in repr(rec.__dict__)


class TestF11ConformalLogRedaction:
    """Conformal site (causal.py :472, `conformal_counterfactual_invalid_intervention`).

    The reachable invalid payload here is the empty intervention (the type
    Dict[str, float] rejects null values before the route runs), so no numeric
    value flows through this site — the redaction is consistency hardening: the
    raw dict is replaced by sorted key names + count, matching the other two
    sites. This test locks the redacted SHAPE (RED at HEAD: raw `intervention`
    field present, no `intervention_keys`).
    """

    @pytest.mark.asyncio
    async def test_conformal_invalid_intervention_log_is_redacted(self, caplog):
        from src.api.causal import conformal_counterfactual_prediction

        req = ConformalCounterfactualRequest(
            model={"variables": ["X", "Y"], "equations": {"Y": "2 * X"}, "distributions": {}},
            intervention={},
        )
        with caplog.at_level(logging.WARNING, logger="src.api.causal"):
            resp = await conformal_counterfactual_prediction(req, x_request_id="t-f11")
        assert resp.status_code == 400
        recs = [
            r for r in caplog.records if r.msg == "conformal_counterfactual_invalid_intervention"
        ]
        assert recs, "expected the conformal invalid-intervention warning"
        rec = recs[0]
        assert getattr(rec, "intervention_keys", None) == []
        assert getattr(rec, "intervention_count", None) == 0
        assert not hasattr(rec, "intervention"), "raw intervention dict still logged"


# ===========================================================================
# F5 — an invalid client equation is a client error, not a server incident
# ===========================================================================
class TestF5EquationErrorSeverity:
    """C6 (F-5, 2026-07-23): `_evaluate_equation` logged client equation
    syntax/eval errors at ERROR, so every invalid-equation 422 paged as a server
    incident. Demote to WARNING (the #95 convention). The genuine internal-fault
    ERROR + traceback path (analyze()'s except Exception) stays intact."""

    def test_invalid_client_equation_logs_warning_not_error(self, caplog):
        """RED at HEAD: `_evaluate_equation`'s syntax-error log is ERROR-level.
        `2 */ X` passes the equation charset but fails ast.parse -> ValueError ->
        422; its engine log must be WARNING, not ERROR."""
        from src.models.requests import CounterfactualRequest, StructuralModel
        from src.services.counterfactual_engine import CounterfactualEngine

        engine = CounterfactualEngine()
        request = CounterfactualRequest(
            model=StructuralModel(
                variables=["X", "Y"], equations={"Y": "2 */ X"}, distributions={}
            ),
            intervention={"X": 5.0},
            outcome="Y",
            context=None,
        )
        with caplog.at_level(logging.DEBUG, logger="src.services.counterfactual_engine"):
            with pytest.raises(ValueError):
                engine.analyze(request)

        eqn_recs = [r for r in caplog.records if "equation" in str(r.msg).lower()]
        assert eqn_recs, "expected an equation-error log from _evaluate_equation"
        assert all(r.levelname != "ERROR" for r in eqn_recs), (
            "invalid client equation still logged at ERROR (should be WARNING)"
        )
        assert any(r.levelname == "WARNING" for r in eqn_recs)

    def test_genuine_internal_fault_still_logs_error_with_traceback(self, caplog):
        """Positive control: the ERROR + traceback path for a REAL server fault
        (analyze()'s except Exception) is intact — not collateral of the demotion."""
        from src.models.requests import CounterfactualRequest, StructuralModel
        from src.services.counterfactual_engine import CounterfactualEngine

        engine = CounterfactualEngine()

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic internal fault")

        engine._compute_prediction = _boom  # a genuine (non-ValueError) server fault
        request = CounterfactualRequest(
            model=StructuralModel(
                variables=["X", "Y"],
                equations={"Y": "2 * X"},
                distributions={"X": {"type": "normal", "parameters": {"mean": 5, "std": 1}}},
            ),
            intervention={"X": 5.0},
            outcome="Y",
            context=None,
        )
        with caplog.at_level(logging.DEBUG, logger="src.services.counterfactual_engine"):
            with pytest.raises(RuntimeError):
                engine.analyze(request)

        errs = [
            r
            for r in caplog.records
            if r.msg == "counterfactual_analysis_failed" and r.levelname == "ERROR"
        ]
        assert errs, "genuine internal fault must still log ERROR"
        assert any(r.exc_info is not None for r in errs), "ERROR must carry a traceback (exc_info)"
