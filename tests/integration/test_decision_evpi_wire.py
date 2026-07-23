"""S1 (A3 VOI honesty, D-23.8) — decision_evpi end-to-end on the robustness v2 wire.

Asserts the top-level ``decision_evpi`` reaches the V2 HTTP envelope, equals the
MINIMUM per-option ``downside.expected_regret`` (= E[max]−max E in outcome units),
is seed-deterministic, and is additive-only (existing fields unchanged).

Seed-42 pins are hand-derived from the regret values already pinned in
test_downside_wire.py (low=0.04396019818411503, high=0.05405519879844565;
near-equivalent low=0.0010990049546028757, high=0.0013513799699611433). Since
decision_evpi = min_o expected_regret[o], the wire value MUST be the 'low' regret.
"""

import pytest

from fastapi.testclient import TestClient

from src.api.main import app

V2_URL = "/api/v1/robustness/analyze/v2"


@pytest.fixture
def client():
    return TestClient(app)


def base_request(**overrides):
    """3-node chain (identical to test_downside_wire.base_request) — two options
    with clearly different means so the decision regret is non-degenerate."""
    request = {
        "graph": {
            "nodes": [
                {"id": "price", "kind": "factor", "label": "Price",
                 "observed_state": {"value": 0.5}},
                {"id": "demand", "kind": "chance", "label": "Demand"},
                {"id": "revenue", "kind": "outcome", "label": "Revenue"},
            ],
            "edges": [
                {"from": "price", "to": "demand", "exists_probability": 0.9,
                 "strength": {"mean": -0.6, "std": 0.1}},
                {"from": "demand", "to": "revenue", "exists_probability": 0.95,
                 "strength": {"mean": 0.8, "std": 0.1}},
                {"from": "price", "to": "revenue", "exists_probability": 0.8,
                 "strength": {"mean": 0.5, "std": 0.15}},
            ],
        },
        "options": [
            {"id": "low", "label": "Low price", "interventions": {"price": 0.3}},
            {"id": "high", "label": "High price", "interventions": {"price": 0.7}},
        ],
        "goal_node_id": "revenue",
        "n_samples": 400,
        "seed": 42,
    }
    request.update(overrides)
    return request


def near_equivalent_request(**overrides):
    request = base_request()
    request["options"] = [
        {"id": "low", "label": "Price 0.50", "interventions": {"price": 0.50}},
        {"id": "high", "label": "Price 0.51", "interventions": {"price": 0.51}},
    ]
    request.update(overrides)
    return request


def post_v2(client, request):
    resp = client.post(f"{V2_URL}?response_version=2", json=request)
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestDecisionEvpiOnWire:
    def test_decision_evpi_present_and_scalar(self, client):
        body = post_v2(client, base_request())
        assert "decision_evpi" in body, "decision_evpi must be present on the wire"
        assert isinstance(body["decision_evpi"], (int, float))
        assert body["decision_evpi"] >= 0.0

    def test_decision_evpi_equals_min_expected_regret_on_wire(self, client):
        """decision_evpi == min_o downside.expected_regret (the exact identity, read
        off the SAME wire response — non-vacuous: uses the emitted per-option regret)."""
        body = post_v2(client, base_request())
        regrets = [
            o["downside"]["expected_regret"]
            for o in body["options"]
            if o.get("downside") is not None
        ]
        assert regrets, "expected downside.expected_regret on the wire"
        assert body["decision_evpi"] == pytest.approx(min(regrets), rel=1e-12)

    def test_decision_evpi_seed42_value_pin(self, client):
        """Wire-exact pin (hand-derived: min(low=0.04396019818411503,
        high=0.05405519879844565) = the 'low' regret). A min->max mutation would
        emit 0.05405519879844565 and RED this pin; a wrong-field mutation moves it."""
        body = post_v2(client, base_request())
        assert body["decision_evpi"] == pytest.approx(0.04396019818411503, rel=1e-12)

    def test_decision_evpi_positive_control_not_the_max(self, client):
        """Positive control (trap #13): the emitted value is MATERIALLY different
        from the max regret (0.0540552) — the pin can SEE a min->max mutation, it is
        not vacuously green."""
        body = post_v2(client, base_request())
        assert abs(body["decision_evpi"] - 0.05405519879844565) > 1e-3

    def test_decision_evpi_near_equivalent_pin(self, client):
        """Near-equivalent options -> tiny decision EVPI (~0.0011): min(low, high)."""
        body = post_v2(client, near_equivalent_request())
        assert body["decision_evpi"] == pytest.approx(0.0010990049546028757, rel=1e-12)
        assert body["decision_evpi"] < 0.01

    def test_decision_evpi_seed_deterministic(self, client):
        b1 = post_v2(client, base_request(seed=7))
        b2 = post_v2(client, base_request(seed=7))
        assert b1["decision_evpi"] == b2["decision_evpi"]

    def test_decision_evpi_additive_downside_and_outcome_unchanged(self, client):
        """decision_evpi must not perturb the existing downside/outcome values
        (additive-only guarantee): the per-option regrets still match their
        test_downside_wire pins."""
        body = post_v2(client, base_request())
        by_id = {o["id"]: o for o in body["options"]}
        assert by_id["low"]["downside"]["expected_regret"] == pytest.approx(
            0.04396019818411503, rel=1e-12
        )
        assert by_id["high"]["downside"]["expected_regret"] == pytest.approx(
            0.05405519879844565, rel=1e-12
        )
