"""A3 counterfactual cache-determinism remediation (HUNT-VALIDATION F-4, 2026-07-23).

Two PRE-EXISTING live-route defects on POST /api/v1/causal/counterfactual:

* Defect 1 — SAME request bytes returned 200 / 422 / 500 depending on PROCESS
  HISTORY. The topo-sort cache key is `json.dumps(equations, sort_keys=True)`
  (content-order-insensitive) but Kahn's order depended on dict INSERTION order
  whenever two variables were simultaneously ready (a tie). Two requests with
  identical equation CONTENT but different key order collided on one cache key
  while computing different evaluation orders, so whichever arrived first decided
  every later permutation's outcome. FIX: a variable-name tie-break makes the
  order a pure function of content, so it agrees with the cache key.

* Defect 2 — a CONSTANT structural equation (`{"Y": "5"}`) returned 500. A 0-d
  array's `.tolist()` scalar hit `len()` in adaptive Monte Carlo -> TypeError.
  A constant IS a valid structural equation; support it (point_estimate = the
  constant, CI collapsed).

The M1/M2 shapes are the exact HUNT-VALIDATION F-4 repro (`{"log": "5", ...}`
uses a math-function-named variable so the log->Y dependency edge is stripped,
creating the false tie). `do(D)` is a disconnected no-op that only satisfies the
route's non-empty-intervention guard; it does not feed the equations under test.

RED at HEAD (executed, engine level): M1 cold=500 / M1 after-M2=422 ;
M2 cold=422 / M2 after-M1=500 (the demonstrated matrix). Constant `{"Y":"5"}`=500.
"""

import json

import pytest

from src.api.causal import counterfactual_engine

_HDR = {"X-Request-Id": "req_fixed_for_determinism_pin"}
_URL = "/api/v1/causal/counterfactual"

# Identical CONTENT, opposite key insertion orders (the F-4 collision pair).
_M1 = {
    "model": {"variables": ["log", "Y", "D"], "equations": {"log": "5", "Y": "log + 1"}, "distributions": {}},
    "intervention": {"D": 1}, "outcome": "Y",
}
_M2 = {
    "model": {"variables": ["log", "Y", "D"], "equations": {"Y": "log + 1", "log": "5"}, "distributions": {}},
    "intervention": {"D": 1}, "outcome": "Y",
}


def _clear_cache():
    """Reset the module-singleton engine's topo cache to a COLD state."""
    counterfactual_engine._topo_sort_cache.clear()


class TestCacheDeterminismMatrix:
    """Defect 1: identical request bytes -> identical status+body regardless of
    which insertion order populated the topo cache first. Reverting the
    variable-name tie-break re-introduces the 500-vs-422 process-history flip and
    makes these RED."""

    @pytest.mark.asyncio
    async def test_M1_cold_equals_M1_after_M2(self, cf_client):
        _clear_cache()
        cold = await cf_client.post(_URL, json=_M1, headers=_HDR)
        _clear_cache()
        await cf_client.post(_URL, json=_M2, headers=_HDR)  # poison cache with Y-first order
        warm = await cf_client.post(_URL, json=_M1, headers=_HDR)
        assert cold.status_code == warm.status_code, (cold.status_code, warm.status_code)
        assert cold.json() == warm.json()

    @pytest.mark.asyncio
    async def test_M2_cold_equals_M2_after_M1(self, cf_client):
        _clear_cache()
        cold = await cf_client.post(_URL, json=_M2, headers=_HDR)
        _clear_cache()
        await cf_client.post(_URL, json=_M1, headers=_HDR)  # poison cache with log-first order
        warm = await cf_client.post(_URL, json=_M2, headers=_HDR)
        assert cold.status_code == warm.status_code, (cold.status_code, warm.status_code)
        assert cold.json() == warm.json()

    @pytest.mark.asyncio
    async def test_same_content_different_insertion_order_identical(self, cf_client):
        """M1 and M2 are the SAME equation set in different key order; both must
        produce byte-identical status+body (the canonical outcome)."""
        _clear_cache()
        r1 = await cf_client.post(_URL, json=_M1, headers=_HDR)
        _clear_cache()
        r2 = await cf_client.post(_URL, json=_M2, headers=_HDR)
        assert r1.status_code == r2.status_code == 422, (r1.status_code, r2.status_code)
        assert r1.json() == r2.json()
        # The canonical name-sorted order evaluates Y before `log` (which the engine
        # treats as a reserved math-function name), so `log` is unresolved -> a
        # DETERMINISTIC 422 (no longer 500-or-422 by history). retryable is honest.
        body = r1.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert body["retryable"] is False
        assert "log" in body["message"]
