"""A3 counterfactual constant-equation support (HUNT-VALIDATION F-4, 2026-07-23).

Defect 2 (PRE-EXISTING, live route POST /api/v1/causal/counterfactual): a CONSTANT
structural equation (`{"Y": "5"}`) returned HTTP 500. `_evaluate_equation("5")`
yields a 0-d numpy array; `_run_adaptive_monte_carlo` calls `.tolist()` on it,
getting a Python scalar, and the subsequent `len()` raises TypeError -> a
mislabeled 500 on legal client input.

A constant IS a valid structural equation: `Y = 5` means Y takes the value 5 in
every Monte Carlo sample. FIX: broadcast a 0-d equation result to the sample
dimension. Outcome: point_estimate = the constant, CI collapsed to [const, const].

RED at HEAD (executed, engine level): `{"Y": "5"}` -> 500
`TypeError: object of type 'int' has no len()`.
"""

import pytest

from src.api.causal import counterfactual_engine

_HDR = {"X-Request-Id": "req_fixed_for_constant_pin"}
_URL = "/api/v1/causal/counterfactual"


class TestConstantEquationRoute:
    @pytest.mark.asyncio
    async def test_constant_outcome_equation_returns_200_constant(self, cf_client):
        """RED at HEAD: 500. Y=5 is constant -> 200, point_estimate 5.0, CI [5, 5].
        `do(D)` is a disconnected no-op only to satisfy the non-empty-intervention
        route guard."""
        request = {
            "model": {"variables": ["Y", "D"], "equations": {"Y": "5"}, "distributions": {}},
            "intervention": {"D": 1}, "outcome": "Y",
        }
        counterfactual_engine._topo_sort_cache.clear()
        resp = await cf_client.post(_URL, json=request, headers=_HDR)
        assert resp.status_code == 200, resp.text
        pred = resp.json()["prediction"]
        assert pred["point_estimate"] == pytest.approx(5.0, abs=1e-9)
        assert pred["confidence_interval"]["lower"] == pytest.approx(5.0, abs=1e-9)
        assert pred["confidence_interval"]["upper"] == pytest.approx(5.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_constant_chain_to_outcome_returns_200(self, cf_client):
        """A chain of CONSTANT equations reaching the outcome keeps it 0-d and was
        500: A=3, B=4, Y=A+B -> Y=7. (A constant feeding an outcome that also has a
        sampled/intervened operand already broadcast to 1-d and never crashed.)"""
        request = {
            "model": {"variables": ["A", "B", "Y", "D"],
                      "equations": {"A": "3", "B": "4", "Y": "A + B"}, "distributions": {}},
            "intervention": {"D": 1}, "outcome": "Y",
        }
        counterfactual_engine._topo_sort_cache.clear()
        resp = await cf_client.post(_URL, json=request, headers=_HDR)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prediction"]["point_estimate"] == pytest.approx(7.0, abs=1e-9)
