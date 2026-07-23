"""S4 (A3 value-of-control, D-23.8) — factor_evpc end-to-end on the robustness v2 wire.

Asserts the per-lever ``factor_evpc`` block reaches the V2 HTTP envelope with the
grid-do() fields, is request-driven (present iff control_candidates supplied),
ABSENT (key omitted, never null) when the field is not supplied, additive-only
(every other wire field byte-identical with vs without the field), and that the
request-validation 422s (unknown factor / non-finite / caps) reach the client with
factor-named messages.
"""

import pytest

from fastapi.testclient import TestClient

from src.api.main import app

V2_URL = "/api/v1/robustness/analyze/v2"


@pytest.fixture
def client():
    return TestClient(app)


def base_request(**overrides):
    """3-node chain; two options pin price ∈ {0.3, 0.7}. 'demand' is a controllable
    intermediate the user could pull."""
    request = {
        "request_id": "evpc-wire",
        "graph": {
            "nodes": [
                {
                    "id": "price",
                    "kind": "factor",
                    "label": "Price",
                    "observed_state": {"value": 0.5},
                },
                {"id": "demand", "kind": "chance", "label": "Demand"},
                {"id": "revenue", "kind": "outcome", "label": "Revenue"},
            ],
            "edges": [
                {
                    "from": "price",
                    "to": "demand",
                    "exists_probability": 0.9,
                    "strength": {"mean": -0.6, "std": 0.1},
                },
                {
                    "from": "demand",
                    "to": "revenue",
                    "exists_probability": 0.95,
                    "strength": {"mean": 0.8, "std": 0.1},
                },
                {
                    "from": "price",
                    "to": "revenue",
                    "exists_probability": 0.8,
                    "strength": {"mean": 0.5, "std": 0.15},
                },
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


def with_candidates(**overrides):
    return base_request(
        control_candidates=[{"factor_id": "demand", "values": [0.2, 0.5, 0.8]}], **overrides
    )


def post_v2(client, request, expect=200):
    resp = client.post(f"{V2_URL}?response_version=2", json=request)
    assert resp.status_code == expect, resp.text
    return resp.json()


class TestFactorEvpcOnWire:
    def test_factor_evpc_present_with_expected_fields(self, client):
        body = post_v2(client, with_candidates())
        assert "factor_evpc" in body, "factor_evpc must be present on the wire"
        assert isinstance(body["factor_evpc"], list) and body["factor_evpc"]
        entry = body["factor_evpc"][0]
        for key in (
            "factor_id",
            "evpc",
            "evpc_raw",
            "best_candidate_value",
            "baseline_max_expected_utility",
            "best_do_expected_utility",
            "units",
            "method",
            "n_samples",
            "n_candidate_values",
            "clamped_low",
            "correlation_active",
        ):
            assert key in entry, f"factor_evpc entry missing {key}"
        assert entry["units"] == "outcome"
        assert entry["method"] == "grid_do_v1"
        assert entry["evpc"] >= 0.0
        # clamp identity holds on the wire (the value-integrity validator guarantees it).
        assert entry["evpc"] == pytest.approx(max(0.0, entry["evpc_raw"]), abs=1e-9)
        # best_candidate_value is one of the supplied grid values.
        assert entry["best_candidate_value"] in (0.2, 0.5, 0.8)

    def test_factor_evpc_absent_when_not_requested(self, client):
        """No control_candidates ⇒ the key is OMITTED entirely (exclude_none), never
        a JSON null."""
        body = post_v2(client, base_request())
        assert "factor_evpc" not in body

    def test_factor_evpc_is_additive_only_on_wire(self, client):
        """Every wire field except factor_evpc is byte-identical with vs without
        control_candidates. Wall-clock / timestamp fields (timestamp,
        processing_time_ms, _metadata.execution_time_ms) are stripped — they vary
        run-to-run and are not part of the analysis payload."""
        with_body = post_v2(client, with_candidates())
        without_body = post_v2(client, base_request())
        with_body.pop("factor_evpc", None)
        for b in (with_body, without_body):
            b.pop("timestamp", None)
            b.pop("processing_time_ms", None)
            b.get("_metadata", {}).pop("execution_time_ms", None)
        assert with_body == without_body

    def test_seed_deterministic_on_wire(self, client):
        a = post_v2(client, with_candidates())["factor_evpc"]
        b = post_v2(client, with_candidates())["factor_evpc"]
        assert a == b


class TestFactorEvpcWire422s:
    def test_unknown_factor_422(self, client):
        req = base_request(control_candidates=[{"factor_id": "ghost", "values": [0.1]}])
        body = post_v2(client, req, expect=422)
        assert "ghost" in str(body)

    def test_goal_node_422(self, client):
        req = base_request(control_candidates=[{"factor_id": "revenue", "values": [0.1]}])
        body = post_v2(client, req, expect=422)
        assert "goal node" in str(body)

    def test_non_finite_value_422(self, client):
        # JSON has no NaN/Inf literal; send the string form FastAPI/pydantic rejects,
        # OR use a value that fails the finite check. Use 1e400 → inf on parse.
        req = base_request(control_candidates=[{"factor_id": "demand", "values": [0.1, 1e400]}])
        body = post_v2(client, req, expect=422)
        assert "finite" in str(body).lower()

    def test_too_many_values_422(self, client):
        req = base_request(
            control_candidates=[
                {"factor_id": "demand", "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
            ]
        )
        post_v2(client, req, expect=422)

    def test_too_many_candidates_422(self, client):
        req = base_request(
            control_candidates=[{"factor_id": f"n{i}", "values": [0.1]} for i in range(6)]
        )
        post_v2(client, req, expect=422)
