"""Negative accumulation counts are refused at the request or draw boundary."""

import pytest

from fastapi.testclient import TestClient

from src.api.main import app
from tests.unit.test_accumulation_identity import (
    CARRIER,
    GOAL,
    INFLOW,
    RATE,
    STOCK,
    accumulation_request,
    assert_finite_floats,
    stepped,
)


def post(request):
    return TestClient(app).post(
        "/api/v1/robustness/analyze/v2",
        json=request,
        headers={"X-ISL-Response-Version": "2"},
    )


@pytest.mark.parametrize(
    "node_id,label,figure,level_field",
    [
        (STOCK, "Pro subscribers today", -5.0, "baseline"),
        (INFLOW, "New subscribers per month", -3.0, "value"),
    ],
    ids=["stock-minus-5", "inflow-minus-3"],
)
def test_negative_today_returns_worded_422(node_id, label, figure, level_field):
    request = accumulation_request()
    node = next(node for node in request["graph"]["nodes"] if node["id"] == node_id)
    node["label"] = label
    observed = node["observed_state"]
    if level_field == "value":
        observed.pop("baseline")
    observed[level_field] = figure / node["execution_frame"]["frame"]

    response = post(request)

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["analysis_status"] == "blocked"
    assert any(
        critique["code"] == "VALIDATION_ERROR"
        and critique["severity"] == "blocker"
        and label in critique["message"]
        and f"is {figure:g};" in critique["message"]
        and "can't be negative" in critique["message"]
        and "month-12 figure can't be computed" in critique["message"]
        for critique in body["critiques"]
    ), body


def test_zero_stock_and_inflow_evaluate():
    request = accumulation_request()
    for node in request["graph"]["nodes"]:
        if node["id"] in (STOCK, INFLOW):
            node["observed_state"].update(value=0.0, baseline=0.0)
        if node["id"] == GOAL:
            # No contradictory nonzero TODAY claim on this zero-stock product.
            node.pop("observed_state")

    response = post(request)

    assert response.status_code == 200, response.text
    body = response.json()
    for option in body["options"]:
        assert option["outcome"]["n_valid_samples"] == request["n_samples"]
        assert option["outcome"]["mean"] == 0.0
    assert_finite_floats(body)


@pytest.mark.parametrize(
    "node_id,figure", [(STOCK, -5.0), (INFLOW, -3.0)], ids=["stock", "inflow"]
)
def test_negative_option_withholds_the_whole_option(node_id, figure):
    request = accumulation_request()
    node = next(node for node in request["graph"]["nodes"] if node["id"] == node_id)
    request["options"].append(
        {
            "id": "invalid",
            "label": "Negative count",
            "interventions": {node_id: figure / node["execution_frame"]["frame"]},
        }
    )

    response = post(request)

    assert response.status_code == 200, response.text
    body = response.json()
    options = {option["id"]: option for option in body["options"]}
    invalid = options["invalid"]
    assert invalid["status"] == "failed"
    assert invalid["outcome"]["n_valid_samples"] == 0
    for field in ("mean", "std", "p10", "p50", "p90"):
        assert invalid["outcome"].get(field) is None
    for field in ("win_probability", "probability_of_goal", "downside"):
        assert invalid.get(field) is None
    horizon_stock = stepped(250.0, 0.03, 20.0, 12)
    for option_id, price in (("keep", 49.0), ("raise", 59.0)):
        assert options[option_id]["outcome"]["n_valid_samples"] == request["n_samples"]
        assert options[option_id]["outcome"]["mean"] == pytest.approx(
            price * horizon_stock / 100000.0, rel=1e-12
        )
    assert_finite_floats(body)


@pytest.mark.parametrize("operation", ["product", "sum"])
def test_other_identity_negative_operand_still_evaluates(operation):
    request = accumulation_request()
    nodes = {node["id"]: node for node in request["graph"]["nodes"]}
    nodes[STOCK]["observed_state"].update(value=-0.005, baseline=-0.005)
    carrier = nodes[CARRIER]
    carrier["nonlinear_identity"] = {
        "operation": operation,
        "factor_ids": [STOCK, INFLOW],
        "stated_in_brief": False,
    }
    carrier.pop("observed_state")
    request["graph"]["edges"] = [
        edge for edge in request["graph"]["edges"]
        if not (edge["from"] == RATE and edge["to"] == CARRIER)
    ]
    nodes[GOAL].pop("nonlinear_identity")
    request["goal_node_id"] = CARRIER
    request["options"] = [
        {"id": "keep", "label": "Keep inflow", "interventions": {INFLOW: 0.2}},
        {"id": "raise", "label": "Raise inflow", "interventions": {INFLOW: 0.3}},
    ]

    response = post(request)

    assert response.status_code == 200, response.text
    body = response.json()
    identity = next(row for row in body["identity_evaluations"] if row["node_id"] == CARRIER)
    assert identity["operation"] == operation and identity["evaluated"] is True
    for option in body["options"]:
        assert option["outcome"]["n_valid_samples"] == request["n_samples"]
        inflow = 20.0 if option["id"] == "keep" else 30.0
        expected = -5.0 * inflow if operation == "product" else -5.0 + inflow
        assert option["outcome"]["mean"] == pytest.approx(expected / 1000.0)
    assert_finite_floats(body)
