"""
Integration tests for causal inference endpoints.

NOTE: Tests converted to async to avoid Starlette TestClient async middleware bug.
Uses httpx.AsyncClient with pytest-asyncio.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_causal_validation_identifiable(client, sample_dag):
    """Test causal validation with identifiable case."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={"dag": sample_dag, "treatment": "X", "outcome": "Y"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ["identifiable", "uncertain", "cannot_identify"]
    assert "explanation" in data
    assert "summary" in data["explanation"]
    assert "reasoning" in data["explanation"]
    assert "assumptions" in data["explanation"]


@pytest.mark.asyncio
async def test_causal_validation_pricing_scenario(client, pricing_dag):
    """Test causal validation with realistic pricing scenario."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={"dag": pricing_dag, "treatment": "Price", "outcome": "Revenue"},
    )

    assert response.status_code == 200
    data = response.json()

    # Price-Revenue should be identifiable
    assert data["status"] == "identifiable"
    assert data["adjustment_sets"] is not None
    assert len(data["adjustment_sets"]) > 0


@pytest.mark.asyncio
async def test_causal_validation_invalid_dag(client):
    """Test causal validation with invalid DAG (Pydantic validation error)."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={
            "dag": {"nodes": [], "edges": []},  # Empty DAG
            "treatment": "X",
            "outcome": "Y",
        },
    )

    # FastAPI/Pydantic returns 422 for validation errors
    assert response.status_code == 422
    data = response.json()
    # ISL uses custom error response format
    assert "code" in data
    assert data["code"] == "ISL_VALIDATION_ERROR"
    assert "message" in data


@pytest.mark.asyncio
async def test_causal_validation_missing_node(client, sample_dag):
    """Test causal validation with missing node."""
    response = await client.post(
        "/api/v1/causal/validate",
        json={"dag": sample_dag, "treatment": "NotExist", "outcome": "Y"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_counterfactual_basic(client, sample_structural_model):
    """Test basic counterfactual analysis."""
    response = await client.post(
        "/api/v1/causal/counterfactual",
        json={
            "model": sample_structural_model,
            "intervention": {"X": 5},
            "outcome": "Y",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "point_estimate" in data["prediction"]
    assert "confidence_interval" in data["prediction"]
    assert "uncertainty" in data
    assert "robustness" in data
    assert "explanation" in data


@pytest.mark.asyncio
async def test_counterfactual_deterministic(client, sample_structural_model):
    """Test that counterfactual analysis is deterministic."""
    request_data = {
        "model": sample_structural_model,
        "intervention": {"X": 5},
        "outcome": "Y",
    }

    # Make two identical requests
    response1 = await client.post("/api/v1/causal/counterfactual", json=request_data)
    response2 = await client.post("/api/v1/causal/counterfactual", json=request_data)

    assert response1.status_code == 200
    assert response2.status_code == 200

    data1 = response1.json()
    data2 = response2.json()

    # Results should be identical
    assert data1["prediction"]["point_estimate"] == data2["prediction"]["point_estimate"]
    assert (
        data1["prediction"]["confidence_interval"]["lower"]
        == data2["prediction"]["confidence_interval"]["lower"]
    )
    assert (
        data1["prediction"]["confidence_interval"]["upper"]
        == data2["prediction"]["confidence_interval"]["upper"]
    )


# ---------------------------------------------------------------------------
# D-12(cf): counterfactual-engine input-rejection ValueErrors -> 422, not 500.
# ---------------------------------------------------------------------------
#
# The counterfactual engine fails loud (ValueError) on client-input defects the
# request model cannot catch: a structural equation referencing an undefined
# variable, a malformed equation, or circular equation dependencies. The router
# previously mapped any Exception -> 500 (group-a-verify cf_bad.resp.json). D-12
# maps these client-input rejections to the repo's 422 validation-error envelope,
# matching the sequential (phase4) D-12 mapping and the robustness v2 handler
# (`except ValueError: HTTPException(422, str(e))`).
#
# This fixture mounts the counterfactual router with the PRODUCTION exception handlers
# on a local app so the ValueError->422 mapping is exercised at the router level
# regardless of whether the global app has mounted the route yet (C1 precedes the
# C3 selective mount). It asserts the real Olumi ErrorResponse envelope.


@pytest_asyncio.fixture
async def counterfactual_error_client():
    """Local app: counterfactual route + production HTTPException/Exception handlers."""
    from fastapi import FastAPI, HTTPException

    from src.api import main as isl_main
    from src.api.causal import counterfactual_router

    test_app = FastAPI()
    test_app.include_router(counterfactual_router, prefix="/api/v1/causal")
    test_app.add_exception_handler(HTTPException, isl_main.http_exception_handler)
    test_app.add_exception_handler(Exception, isl_main.global_exception_handler)

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        yield ac


class TestCounterfactualEngineErrorMapping:
    """D-12(cf): engine ValueError (bad equation / cycle) -> 422, not 500."""

    @pytest.mark.asyncio
    async def test_malformed_equation_undefined_var_returns_422_envelope(
        self, counterfactual_error_client
    ):
        """RED at HEAD: undefined variable in a structural equation -> 500.

        This is the group-a-verify cf_bad.json case (`"Revenue": "500 * NopeVar"`,
        NopeVar undefined). AST evaluation fails loud (ValueError). Pydantic's
        equation validator only blocks unsafe characters, not undefined-variable
        references, so the request reaches the engine. That is a client-input
        rejection -> 422 (Olumi envelope), never a 500.
        """
        request = {
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "500 * NopeVar"},
                "distributions": {},
            },
            "intervention": {"Price": 15},
            "outcome": "Revenue",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert body["code"] == "ISL_VALIDATION_ERROR"
        # message carries the engine's diagnostic naming the offending equation
        assert "NopeVar" in body["message"]

    @pytest.mark.asyncio
    async def test_circular_equation_dependencies_returns_422_envelope(
        self, counterfactual_error_client
    ):
        """RED at HEAD: circular equation dependencies -> 500.

        `A = B + 1`, `B = A + 1` is a dependency cycle; topological ordering of the
        structural equations fails loud. -> 422, not 500.
        """
        request = {
            "model": {
                "variables": ["A", "B"],
                "equations": {"A": "B + 1", "B": "A + 1"},
                "distributions": {},
            },
            "intervention": {},
            "outcome": "A",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "circular" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_malformed_equation_syntax_returns_422_envelope(
        self, counterfactual_error_client
    ):
        """RED at HEAD: a syntactically malformed equation -> 500.

        `"500 * "` is not a parseable expression; AST parsing raises SyntaxError
        which the engine converts to a fail-loud ValueError. -> 422, not 500.
        """
        request = {
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "500 * "},
                "distributions": {},
            },
            "intervention": {"Price": 15},
            "outcome": "Revenue",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "equation" in body["message"].lower()


# ---------------------------------------------------------------------------
# Input hardening (A3, 2026-07-22): three client-reachable input-defect classes
# that fail-closed as 500 on the LIVE mount are hardened to a clean 422 through
# the same D-12(cf) `except ValueError -> 422` locus. All three are fail-closed
# today (no wrong value) but the endpoint is live and a consumer will wire in;
# this is the counterfactual analogue of the phase4 sequential F-2/F-3 pre-mount
# hardening. Documented in a3-flip/ADVERSARIAL.md findings 1 (KeyError classes)
# and 2 (non-finite outcomes). Reuses the counterfactual_error_client fixture
# (route + production handlers) so the mapping is exercised at the router level.


class TestCounterfactualUndefinedOutcome:
    """Shape 1: `outcome` names a variable the model can never value -> 422, not 500.

    The MC sampler populates `samples` from exactly four sources (exogenous
    distributions, intervention, context, structural equations). An outcome absent
    from all four is never sampled, so `samples[outcome]` (and, earlier,
    `_run_adaptive_monte_carlo`'s `batch_samples[request.outcome]`) raises KeyError
    -> a mislabeled 500. It is a client-input defect (a typo'd / dangling outcome
    name from the upstream graph-builder), so it must be a clean 422 naming the
    unresolved variable.
    """

    @pytest.mark.asyncio
    async def test_undefined_outcome_returns_422_envelope(self, counterfactual_error_client):
        """RED at HEAD: outcome 'Ghost' absent from eq/dist/intervention/context -> 500
        (KeyError 'Ghost' @ counterfactual_engine.py:252). -> 422 after the guard.
        """
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "10 + 2 * X"},
                "distributions": {},
            },
            "intervention": {"X": 5},
            "outcome": "Ghost",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert body["code"] == "ISL_VALIDATION_ERROR"
        # message names the unresolved outcome variable
        assert "Ghost" in body["message"]

    @pytest.mark.asyncio
    async def test_outcome_resolvable_via_distribution_only_still_200(
        self, counterfactual_error_client
    ):
        """Positive control: an outcome that is a pure exogenous distribution
        variable (no structural equation) is legitimately resolvable and MUST still
        200 — proving the guard's resolvable set includes distributions, not only
        equations (i.e. it does not over-reject).
        """
        request = {
            "model": {
                "variables": ["X"],
                "equations": {},
                "distributions": {"X": {"type": "normal", "parameters": {"mean": 10, "std": 1}}},
            },
            "intervention": {},
            "outcome": "X",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 200
        body = response.json()
        assert "point_estimate" in body["prediction"]


class TestCounterfactualMissingDistributionParameter:
    """Shape 2: an exogenous distribution spec omitting a required parameter -> 422,
    not 500.

    `_sample_distribution` dereferences the client-supplied distribution parameters
    positionally (`params["std"]`, `params["max"]`, ...). A missing key raises
    KeyError -> a mislabeled 500. This is a client-input defect. The fix re-raises
    it as ValueError (route D-12(cf) -> 422) naming the variable and the exact
    missing parameter, DERIVED from the KeyError (not a hand-maintained
    required-params mirror), so the two dist types below prove it is not hardcoded
    to one parameter name.
    """

    @pytest.mark.asyncio
    async def test_normal_missing_std_returns_422_envelope(self, counterfactual_error_client):
        """RED at HEAD: a `normal` distribution without `std` -> 500 (KeyError 'std'
        @ counterfactual_engine.py:340). -> 422 after the guard.
        """
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "10 + 2 * X"},
                "distributions": {"X": {"type": "normal", "parameters": {"mean": 5}}},
            },
            "intervention": {},
            "outcome": "Y",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert body["code"] == "ISL_VALIDATION_ERROR"
        # message names the offending variable AND the exact missing parameter
        assert "X" in body["message"]
        assert "std" in body["message"]

    @pytest.mark.asyncio
    async def test_uniform_missing_max_returns_422_envelope(self, counterfactual_error_client):
        """RED at HEAD: a `uniform` distribution without `max` -> 500 (KeyError
        'max'). Proves the missing-parameter name is DERIVED from the real
        dereference, not hardcoded to 'std'.
        """
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "10 + 2 * X"},
                "distributions": {"X": {"type": "uniform", "parameters": {"min": 0}}},
            },
            "intervention": {},
            "outcome": "Y",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "X" in body["message"]
        assert "max" in body["message"]


class TestCounterfactualNonFiniteOutcome:
    """Shape 3: a structural model that computes a non-finite outcome (NaN/+/-inf)
    on the sampled inputs -> 422, not a serialization 500.

    log() of a non-positive number, division by zero, or overflow in a structural
    equation yields NaN/inf values. The engine returns them successfully; the
    non-finite point estimate / interval then serialize-fails in Starlette's
    JSONResponse (allow_nan=False) -> an unhandled 500 at RESPONSE RENDERING,
    OUTSIDE the route's try (adversarial finding 2; via the local error fixture the
    same defect surfaces as a raised `ValueError: Out of range float values are not
    JSON compliant`). The guard rejects (does NOT clamp — a finite substitute would
    be a fabricated value) with a clean 422 BEFORE the response is built. The
    positive controls prove the guard does not false-positive on legitimate finite
    outcomes, including a legitimately-zero and a large-but-finite value.
    """

    @pytest.mark.asyncio
    async def test_log_of_negative_nan_returns_422_envelope(self, counterfactual_error_client):
        """RED at HEAD: Y=log(X), do(X=-5) -> all-NaN outcome -> 500 at serialization
        (non-finite point_estimate; JSONResponse allow_nan=False). -> 422 after.
        """
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "log(X)"},
                "distributions": {},
            },
            "intervention": {"X": -5},
            "outcome": "Y",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert body["code"] == "ISL_VALIDATION_ERROR"
        assert "non-finite" in body["message"].lower()
        assert "Y" in body["message"]

    @pytest.mark.asyncio
    async def test_division_by_zero_inf_returns_422_envelope(self, counterfactual_error_client):
        """RED at HEAD: Y=1/X, do(X=0) -> +inf outcome -> 500 at serialization.
        -> 422 after. A distinct non-finite class (inf, not NaN).
        """
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "1 / X"},
                "distributions": {},
            },
            "intervention": {"X": 0},
            "outcome": "Y",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "non-finite" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_legitimate_zero_outcome_still_200(self, counterfactual_error_client):
        """Positive control: Y=X-5, do(X=5) -> outcome exactly 0.0 (finite). MUST
        still 200 — proves the non-finite guard does not reject a legitimately-zero
        outcome.
        """
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "X - 5"},
                "distributions": {},
            },
            "intervention": {"X": 5},
            "outcome": "Y",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"]["point_estimate"] == 0.0

    @pytest.mark.asyncio
    async def test_large_but_finite_outcome_still_200(self, counterfactual_error_client):
        """Positive control: Y=X*1e9, do(X=1e9) -> 1e18 (large but finite). MUST
        still 200 — proves the guard keys on finiteness, not magnitude.
        """
        request = {
            "model": {
                "variables": ["X", "Y"],
                "equations": {"Y": "X * 1000000000"},
                "distributions": {},
            },
            "intervention": {"X": 1000000000},
            "outcome": "Y",
        }
        response = await counterfactual_error_client.post(
            "/api/v1/causal/counterfactual", json=request
        )
        assert response.status_code == 200
        body = response.json()
        assert response.json()["prediction"]["point_estimate"] == 1e18
