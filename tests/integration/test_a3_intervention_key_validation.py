"""A3 honesty-residuals (2026-07-23): reject-not-fabricate for unknown
intervention/context keys on the LIVE /api/v1/causal/counterfactual route.

The F3d residual: an intervention (or context) key naming a variable that is
ABSENT from the structural model silently no-ops. `_run_fixed_monte_carlo`
writes `samples[key] = np.full(...)`, but nothing ever reads it, so the model
returns the OBSERVATIONAL BASELINE with HTTP 200 — a plausible answer to a
question that was never evaluated (a client typo like {"Q": 5} on a model of
X,Z,Y). The `context` channel has the identical hole (same unread np.full write).

Fix: validate intervention AND context keys against the model's known-variable
set (variables | equations | distributions) in the ONE validation home that
already carries the PR#93 do/observe collision guard
(`CounterfactualEngine._require_resolvable_outcome`). Unknown keys -> ValueError
-> route D-12(cf) -> 422 naming the unknown key(s) and the known-set SIZE. The
message names KEYS + a size only; it never echoes request VALUES (they are the
client's private scenario inputs; the message is also written to the
`counterfactual_invalid_input` warning log).

RED-first: `test_unknown_intervention_key_returns_422` and
`test_unknown_context_key_returns_422` FAIL at HEAD (route returns 200 with the
observational baseline) and pass after the fix (422).
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


# A distinctive value that cannot arise from a hash/seed/percentile, used to
# prove the 422 message never leaks a request VALUE (keys are fine, values are
# not — the message is also logged).
_SENTINEL_VALUE = 424242.4242


@pytest_asyncio.fixture
async def cf_client():
    """LIVE counterfactual route mounted with the production exception handlers,
    so the ValueError->422 mapping (D-12(cf)) is exercised at the router level."""
    from fastapi import FastAPI, HTTPException

    from src.api import main as isl_main
    from src.api.causal import counterfactual_router

    test_app = FastAPI()
    test_app.include_router(counterfactual_router, prefix="/api/v1/causal")
    test_app.add_exception_handler(HTTPException, isl_main.http_exception_handler)
    test_app.add_exception_handler(Exception, isl_main.global_exception_handler)

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        yield ac


# Model of X, Z, Y with Y = 2*X + Z. X and Z are exogenous (distributions), so
# the outcome Y is computable WITHOUT any intervention — this is what lets an
# unknown intervention key no-op to the observational baseline at HEAD.
def _model_xzy_with_distributions() -> dict:
    return {
        "variables": ["X", "Z", "Y"],
        "equations": {"Y": "2 * X + Z"},
        "distributions": {
            "X": {"type": "normal", "parameters": {"mean": 10.0, "std": 1.0}},
            "Z": {"type": "normal", "parameters": {"mean": 3.0, "std": 1.0}},
        },
    }


class TestUnknownInterventionKey:
    @pytest.mark.asyncio
    async def test_unknown_intervention_key_returns_422(self, cf_client):
        """RED at HEAD: intervention {"Q": 5} on a model of X,Z,Y is accepted
        (200) and returns the observational baseline — Q is written to `samples`
        but never read. Q is absent from the model; reject -> 422."""
        request = {
            "model": _model_xzy_with_distributions(),
            "intervention": {"Q": 5.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "Q" in body["message"]

    @pytest.mark.asyncio
    async def test_unknown_context_key_returns_422(self, cf_client):
        """RED at HEAD: the context channel has the identical hole — context
        {"Q": 5} names an absent variable, is written to `samples` and never
        read, so a legitimate do(X) request 200s while silently ignoring the
        typo'd observation. Reject -> 422 naming Q."""
        request = {
            "model": _model_xzy_with_distributions(),
            "intervention": {"X": 10.0},
            "context": {"Q": 5.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "Q" in body["message"]

    @pytest.mark.asyncio
    async def test_message_names_unknown_keys_and_set_size_not_values(self, cf_client):
        """The 422 message must NAME the unknown key(s) and the known-variable
        set SIZE, and must NOT echo the request VALUE (redaction: the message is
        also written to the counterfactual_invalid_input warning log)."""
        request = {
            "model": _model_xzy_with_distributions(),
            "intervention": {"Qtypo": _SENTINEL_VALUE},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        message = resp.json()["message"]
        # names the unknown key
        assert "Qtypo" in message
        # names the known-variable set SIZE (X, Z, Y -> 3)
        assert "3" in message
        # NEVER leaks the raw value
        assert str(_SENTINEL_VALUE) not in message
        assert "424242" not in message


class TestKnownKeyPositiveControls:
    @pytest.mark.asyncio
    async def test_disjoint_do_observe_deterministic_23(self, cf_client):
        """Positive control (a): a valid disjoint do(X=10) + observe(Z=3) on the
        fully-deterministic model Y=2*X+Z (no distributions) still 200s with the
        exact point_estimate 23.0."""
        request = {
            "model": {
                "variables": ["X", "Z", "Y"],
                "equations": {"Y": "2 * X + Z"},
                "distributions": {},
            },
            "intervention": {"X": 10.0},
            "context": {"Z": 3.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prediction"]["point_estimate"] == pytest.approx(23.0)

    @pytest.mark.asyncio
    async def test_intervene_on_all_model_variables_still_200(self, cf_client):
        """Positive control (b): intervening on EVERY model variable (including
        the equation-defined outcome) is legitimate do() and still 200s."""
        request = {
            "model": {
                "variables": ["X", "Z", "Y"],
                "equations": {"Y": "2 * X + Z"},
                "distributions": {},
            },
            "intervention": {"X": 1.0, "Z": 2.0, "Y": 3.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prediction"]["point_estimate"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_context_on_exogenous_distribution_variable_still_200(self, cf_client):
        """Positive control: a context key naming an EXOGENOUS variable that is
        declared only in `distributions` (not in `variables`) is known and must
        still 200 — the known-set includes distributions, mirroring the schema
        example's context {"baseline_brand": ...}."""
        request = {
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "baseline + 500 * Price"},
                "distributions": {
                    "baseline": {"type": "normal", "parameters": {"mean": 100.0, "std": 1.0}}
                },
            },
            "intervention": {"Price": 15.0},
            "context": {"baseline": 120.0},
            "outcome": "Revenue",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prediction"]["point_estimate"] == pytest.approx(120.0 + 500 * 15.0)
