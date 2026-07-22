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
# This fixture mounts the causal router with the PRODUCTION exception handlers on
# a local app so the ValueError->422 mapping is exercised at the router level
# regardless of whether the global app has mounted the route yet (C1 precedes the
# C3 selective mount). It asserts the real Olumi ErrorResponse envelope.


@pytest_asyncio.fixture
async def counterfactual_error_client():
    """Local app: counterfactual route + production HTTPException/Exception handlers."""
    from fastapi import FastAPI, HTTPException

    from src.api import main as isl_main
    from src.api.causal import router as causal_router

    test_app = FastAPI()
    test_app.include_router(causal_router, prefix="/api/v1/causal")
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
