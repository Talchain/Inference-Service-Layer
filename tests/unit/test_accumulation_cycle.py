"""Accumulation cycles must use the structural blocked response at the HTTP boundary."""

from fastapi.testclient import TestClient

from src.api.main import app
from tests.unit.test_accumulation_identity import CARRIER, GOAL, STOCK, accumulation_request


def test_cyclic_accumulation_returns_structural_blocked_422():
    request = accumulation_request()
    carrier = next(node for node in request["graph"]["nodes"] if node["id"] == CARRIER)
    carrier["nonlinear_identity"]["factor_ids"][0] = GOAL
    stock_edge = next(
        edge for edge in request["graph"]["edges"]
        if edge["from"] == STOCK and edge["to"] == CARRIER
    )
    stock_edge["from"] = GOAL

    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2",
        json=request,
        headers={"X-ISL-Response-Version": "2"},
    )

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["analysis_status"] == "blocked"
    assert any(
        critique["code"] == "GRAPH_CYCLE_DETECTED" and critique["severity"] == "blocker"
        for critique in body["critiques"]
    )
