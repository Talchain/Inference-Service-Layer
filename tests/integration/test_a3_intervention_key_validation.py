"""A3 honesty-residuals (2026-07-23): reject-not-fabricate for malformed
intervention/context INPUTS on the LIVE /api/v1/causal/counterfactual route.

Two residual classes, both closed in the single validation home
(`CounterfactualEngine._validate_counterfactual_inputs`, which already carries the
PR#93 do/observe collision guard):

1. The F3d residual (unknown KEY): an intervention (or context) key naming a
   variable ABSENT from the structural model silently no-ops.
   `_run_fixed_monte_carlo` writes `samples[key] = np.full(...)`, but nothing
   ever reads it, so the model returns the OBSERVATIONAL BASELINE with HTTP 200 —
   a plausible answer to a question that was never evaluated (a client typo like
   {"Q": 5} on a model of X,Z,Y). The `context` channel has the identical hole.
   Fix: validate keys against the model's known-variable set (variables |
   equations | distributions) -> unknown keys ValueError -> route D-12(cf) -> 422
   naming the unknown key(s) and the known-set SIZE.

2. The C1 residual (non-finite VALUE, F-1): a NON-FINITE intervention/context
   value (NaN or +/-inf) on a declared-but-DISCONNECTED variable (one no equation
   reads) passes the unknown-key AND finite-outcome guards, leaves the outcome
   finite, is echoed into `scenario.intervention`/`scenario.context`, and crashes
   Starlette's `allow_nan=False` JSON render OUTSIDE the route try -> an unhandled
   500 (ISL_COMPUTATION_ERROR, retryable:true — which is false: it fails forever).
   Fix: sweep intervention/context VALUES for non-finiteness -> ValueError -> a
   clean 422 naming the offending KEY(s).

Both messages name KEYS (+ a size for class 1) only; they NEVER echo request
VALUES (client-private scenario inputs; the message is also written to the
`counterfactual_invalid_input` warning log).

RED-first: `test_unknown_*_key_returns_422` FAIL at HEAD with 200 (observational
baseline); the `TestNonFiniteInterventionValues` tests FAIL at HEAD with 500
(serialization crash, retryable:true). All pass after the fix (422).
"""

import pytest

# The `cf_client` fixture and the F11 `redaction_sentinel` value (a distinctive
# value that cannot arise from a hash/seed/percentile, used to prove the 422
# message never leaks a request VALUE) are shared fixtures in
# tests/integration/conftest.py (C4 dedup).


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
    async def test_message_names_unknown_keys_and_set_size_not_values(
        self, cf_client, redaction_sentinel
    ):
        """The 422 message must NAME the unknown key(s) and the known-variable
        set SIZE, and must NOT echo the request VALUE (redaction: the message is
        also written to the counterfactual_invalid_input warning log)."""
        request = {
            "model": _model_xzy_with_distributions(),
            "intervention": {"Qtypo": redaction_sentinel},
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
        assert str(redaction_sentinel) not in message
        assert str(int(redaction_sentinel)) not in message


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


# D is declared in `variables` but referenced by NO equation (Y depends only on
# X); a non-finite value on D therefore leaves the outcome finite and slips past
# both the unknown-key guard (D is "known") and the finite-outcome guard (Y is
# finite) — the exact class the C1 non-finite sweep closes.
def _disconnected_d_model() -> dict:
    return {"variables": ["X", "Y", "D"], "equations": {"Y": "2 * X"}, "distributions": {}}


class TestNonFiniteInterventionValues:
    """C1 (F-1, 2026-07-23): non-finite intervention/context VALUE -> 500 today,
    422 after the fix.

    RED at HEAD: each request below returns 500 (ISL_COMPUTATION_ERROR,
    retryable:true — Starlette allow_nan=False crash OUTSIDE the route try) before
    the fix; a clean 422 (validation_failed, retryable:false) after.
    """

    @pytest.mark.asyncio
    async def test_nonfinite_nan_intervention_returns_422(self, cf_client):
        """do(D=NaN) on the disconnected D -> 500 at HEAD, 422 after."""
        request = {
            "model": _disconnected_d_model(),
            "intervention": {"D": float("nan")},
            "context": {"X": 5.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        # The HEAD 500 falsely advertised retryable:true (it fails forever). The
        # 422 correction must NOT claim retryable.
        assert body["retryable"] is False
        assert "D" in body["message"]
        assert "non-finite" in body["message"]

    @pytest.mark.asyncio
    async def test_nonfinite_nan_context_returns_422(self, cf_client):
        """The context channel has the identical hole: observe(D=NaN) alongside a
        valid do(X=5) -> 500 at HEAD, 422 after, naming D."""
        request = {
            "model": _disconnected_d_model(),
            "intervention": {"X": 5.0},
            "context": {"D": float("nan")},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["reason"] == "validation_failed"
        assert body["retryable"] is False
        assert "D" in body["message"]

    @pytest.mark.asyncio
    async def test_nonfinite_positive_infinity_returns_422(self, cf_client):
        request = {
            "model": _disconnected_d_model(),
            "intervention": {"D": float("inf")},
            "context": {"X": 5.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        assert resp.json()["retryable"] is False

    @pytest.mark.asyncio
    async def test_nonfinite_negative_infinity_returns_422(self, cf_client):
        request = {
            "model": _disconnected_d_model(),
            "intervention": {"D": float("-inf")},
            "context": {"X": 5.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        assert resp.json()["retryable"] is False

    @pytest.mark.asyncio
    async def test_nonfinite_on_connected_variable_returns_422_naming_key(self, cf_client):
        """do(X=NaN) where X FEEDS the outcome: the non-finite sweep now fires
        first (before the finite-outcome guard), so the 422 names the offending
        KEY 'X' rather than blaming 'the structural model'."""
        request = {
            "model": _disconnected_d_model(),
            "intervention": {"X": float("nan")},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["retryable"] is False
        assert "X" in body["message"]
        assert "non-finite" in body["message"]

    @pytest.mark.asyncio
    async def test_nonfinite_message_names_key_not_covalues(self, cf_client, redaction_sentinel):
        """Redaction (F11): only the offending KEY is named. A finite sentinel
        VALUE co-submitted on another known key must NOT appear in the message
        (proves the guard names keys, not the intervention dict)."""
        model = {
            "variables": ["X", "Y", "D", "W"],
            "equations": {"Y": "2 * X"},
            "distributions": {},
        }
        request = {
            "model": model,
            "intervention": {"D": float("nan"), "W": redaction_sentinel},
            "context": {"X": 5.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 422, resp.text
        message = resp.json()["message"]
        assert "D" in message
        assert "non-finite" in message
        # The finite co-value on W is never echoed.
        assert str(redaction_sentinel) not in message
        assert str(int(redaction_sentinel)) not in message

    @pytest.mark.asyncio
    async def test_finite_disconnected_value_still_200(self, cf_client):
        """Positive control: a finite (even large) value on the disconnected D is
        a legitimate no-op probe and still 200s with the exact baseline estimate
        (Y = 2 * X = 10.0 at X=5)."""
        request = {
            "model": _disconnected_d_model(),
            "intervention": {"D": 500000.0},
            "context": {"X": 5.0},
            "outcome": "Y",
        }
        resp = await cf_client.post("/api/v1/causal/counterfactual", json=request)
        assert resp.status_code == 200, resp.text
        assert resp.json()["prediction"]["point_estimate"] == pytest.approx(10.0)
