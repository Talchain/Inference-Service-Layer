"""Graceful 429 + Paul-ruled lenient limit value pins (ISL lane 2, 2026-07-17).

RED-first evidence for the 429 fix: at base 85566321 the blocked branch of
RateLimitMiddleware.dispatch crashed with
``SecurityAuditLogger.log_rate_limit_exceeded() got an unexpected keyword
argument 'request_id'`` — rate-limited callers received a bare 500 instead of
the structured 429 (2026-07-17 budget-review surprise #2; guarantee-theatre
class: the graceful-degradation machinery itself was broken). The crash site
carried a ``# type: ignore[call-arg]`` that suppressed the exact mypy error
which would have caught it. TestGraceful429 was written and run RED against
the unfixed middleware before the fix landed, and asserts BOTH halves of the
restored machinery: the well-formed 429 response AND the audit log event.

TestPaulRuledLenientValuePins pins the four lenient defaults raised by Paul's
2026-07-17 lenient-latency ruling so a silent revert turns RED.
"""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.middleware.rate_limiting as rate_limiting_module
from src.middleware.rate_limiting import (
    RateLimiter,
    RateLimitMiddleware,
    RedisRateLimiter,
)

TINY_LIMIT = 2


@pytest.fixture()
def tiny_limit_client(monkeypatch):
    """TestClient for an app running the REAL RateLimitMiddleware with a
    2-req/min in-memory limiter injected in place of the module singleton.

    raise_server_exceptions=False so an unhandled middleware exception is
    served as the bare 500 the live caller would see (the base-tip defect)
    rather than re-raised — the RED failure then reads 500 != 429, exactly
    mirroring the observed production behaviour.
    """
    fallback = RateLimiter(requests_per_minute=TINY_LIMIT)
    limiter = RedisRateLimiter(redis_client=None, requests_per_minute=TINY_LIMIT, fallback=fallback)
    monkeypatch.setattr(rate_limiting_module, "_rate_limiter", limiter)

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    @app.get("/probe")
    def probe() -> dict:
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


class TestGraceful429:
    """The rate limiter's degradation path must degrade, not crash."""

    def test_rate_limited_request_gets_structured_429(self, tiny_limit_client, caplog):
        with caplog.at_level(logging.WARNING, logger="security.audit"):
            for _ in range(TINY_LIMIT):
                assert tiny_limit_client.get("/probe").status_code == 200
            resp = tiny_limit_client.get("/probe", headers={"X-Request-Id": "req-lane2-429"})

        # Half 1 — graceful, well-formed 429 (Olumi Error Schema v1.0),
        # not the bare 500 the TypeError produced at base.
        assert resp.status_code == 429
        body = resp.json()
        assert body["code"] == "ISL_RATE_LIMIT_EXCEEDED"
        assert body["reason"] == "too_many_requests"
        assert body["retryable"] is True
        assert body["source"] == "isl"
        assert body["request_id"] == "req-lane2-429"
        assert body["recovery"]["hints"]
        assert int(resp.headers["Retry-After"]) > 0

        # Half 2 — the alarm fires: the security-audit event the crash was
        # silencing, correlated by the caller's request id.
        events = [r for r in caplog.records if r.getMessage() == "rate_limit_exceeded"]
        assert len(events) == 1
        assert events[0].request_id == "req-lane2-429"
        assert events[0].identifier.startswith("ip:")
        assert isinstance(events[0].limit, int) and events[0].limit > 0

    def test_requests_within_limit_pass_through_untouched(self, tiny_limit_client):
        """Positive control: below the limit the middleware is transparent."""
        resp = tiny_limit_client.get("/probe")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert "X-RateLimit-Limit" in resp.headers


class TestPaulRuledLenientValuePins:
    """Paul-ruled lenient defaults 2026-07-17 — a silent revert turns these RED.

    Each pin asserts the CODE default (no Render env override exists on
    isl-staging for any of these — code governs the deployed value).
    """

    def test_rate_limit_default_pinned_at_1000(self):
        from src.config import Settings

        # Pin the field default itself so an ambient env var cannot mask a revert.
        assert Settings.model_fields["RATE_LIMIT_REQUESTS_PER_MINUTE"].default == 1000

    def test_compute_admission_ceiling_pinned(self):
        # Codex F8 (2026-07-18) REPLACED the scalar complexity ceiling
        # (_DEFAULT_MAX_COMPLEXITY = 30M, in scalar n_samples*n_nodes*n_edges
        # units) with the weighted cost model's DEFAULT_MAX_COST_UNITS (in cost
        # units — a DIFFERENT scale). PROVISIONAL, Paul-DIRECTED 24M (the more-
        # lenient choice); staging recalibration still owed. Canonical pin +
        # positive controls live in test_admission_calibration.py; this asserts the
        # same constant so a silent revert of the lenient ceiling turns RED here too.
        from src.services.robustness_analyzer_v2 import DEFAULT_MAX_COST_UNITS

        assert DEFAULT_MAX_COST_UNITS == 24_000_000

    def test_evpi_sample_cap_pinned_at_2000(self):
        from src.services.robustness_analyzer_v2 import EVPI_SAMPLE_CAP

        assert EVPI_SAMPLE_CAP == 2000

    def test_e_value_budget_pinned_at_8000(self):
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        assert RobustnessAnalyzerV2.E_VALUE_BUDGET_MS == 8000
