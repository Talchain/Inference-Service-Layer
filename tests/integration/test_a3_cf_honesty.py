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
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.models.requests import ConformalCounterfactualRequest


# ---------------------------------------------------------------------------
# Local app: the LIVE counterfactual route + the conformal route, both with the
# production exception handlers, so the ValueError->422 mapping (D-12(cf)) and
# the conformal 400 envelope are exercised at the router level.
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def cf_client():
    from fastapi import FastAPI, HTTPException

    from src.api import main as isl_main
    from src.api.causal import counterfactual_router, router as causal_full_router

    test_app = FastAPI()
    test_app.include_router(counterfactual_router, prefix="/api/v1/causal")
    # conformal lives on the full causal router (dark in prod); mount it here so
    # the F11 site-3 redaction is exercised through the real handler.
    test_app.include_router(causal_full_router, prefix="/api/v1/causal")
    test_app.add_exception_handler(HTTPException, isl_main.http_exception_handler)
    test_app.add_exception_handler(Exception, isl_main.global_exception_handler)

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        yield ac


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
_SENTINEL = 987654.321  # a value that cannot arise from a hash/seed/percentile


def _assert_harness_can_see_value(caplog_records_before, logger_name):
    """Trap #13 positive control: prove the capture harness is NOT blind to the
    sentinel by emitting it through the same logger and confirming it is seen."""
    log = logging.getLogger(logger_name)
    log.info("a3_f11_probe_control", extra={"intervention": {"Probe": _SENTINEL}})
    seen = [r for r in caplog_records_before if getattr(r, "msg", None) == "a3_f11_probe_control"]
    assert seen, "positive control failed: caplog did not capture the probe log at all"
    assert any(str(_SENTINEL) in repr(r.__dict__) for r in seen), (
        "positive control failed: the harness cannot SEE the raw value even when "
        "it is deliberately logged — an absence assertion here would be vacuous"
    )


class TestF11EngineLogRedaction:
    """Engine site (counterfactual_engine.py :98, `counterfactual_analysis_started`)."""

    @pytest.mark.asyncio
    async def test_engine_start_log_redacts_intervention_values(self, cf_client, caplog):
        from src.models.requests import CounterfactualRequest, StructuralModel
        from src.services.counterfactual_engine import CounterfactualEngine

        engine = CounterfactualEngine()
        request = CounterfactualRequest(
            model=StructuralModel(
                variables=["SecretPrice", "Y"],
                equations={"Y": "2 * SecretPrice"},
                distributions={},
            ),
            intervention={"SecretPrice": _SENTINEL},
            outcome="Y",
            context=None,
        )
        with caplog.at_level(logging.INFO, logger="src.services.counterfactual_engine"):
            engine.analyze(request)
            # positive control FIRST — the harness must be able to see the value
            _assert_harness_can_see_value(caplog.records, "src.services.counterfactual_engine")

        recs = [r for r in caplog.records if r.msg == "counterfactual_analysis_started"]
        assert recs, "expected the counterfactual_analysis_started log"
        rec = recs[0]
        # redacted structured fields present (reads REAL data, not vacuous)
        assert getattr(rec, "intervention_keys", None) == ["SecretPrice"]
        assert getattr(rec, "intervention_count", None) == 1
        # RED at HEAD: the raw dict field is gone and the value appears nowhere
        assert not hasattr(rec, "intervention"), "raw intervention dict still logged"
        assert str(_SENTINEL) not in repr(rec.__dict__)
        assert "987654" not in repr(rec.__dict__)


class TestF11RouteLogRedaction:
    """Route site (causal.py :265, `counterfactual_request`)."""

    @pytest.mark.asyncio
    async def test_counterfactual_request_log_redacts_intervention_values(self, cf_client, caplog):
        request = {
            "model": {
                "variables": ["SecretPrice", "Y"],
                "equations": {"Y": "2 * SecretPrice"},
                "distributions": {},
            },
            "intervention": {"SecretPrice": _SENTINEL},
            "outcome": "Y",
        }
        with caplog.at_level(logging.INFO, logger="src.api.causal"):
            resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
            assert resp.status_code == 200, resp.text
            _assert_harness_can_see_value(caplog.records, "src.api.causal")

        recs = [r for r in caplog.records if r.msg == "counterfactual_request"]
        assert recs, "expected the counterfactual_request log"
        rec = recs[0]
        assert getattr(rec, "intervention_keys", None) == ["SecretPrice"]
        assert getattr(rec, "intervention_count", None) == 1
        assert not hasattr(rec, "intervention"), "raw intervention dict still logged"
        assert str(_SENTINEL) not in repr(rec.__dict__)
        assert "987654" not in repr(rec.__dict__)


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
