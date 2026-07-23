"""A3 honesty-residuals (2026-07-23): per-endpoint config_details honesty.

`generate_config_details()` unconditionally emitted
`monte_carlo_samples: settings.MAX_MONTE_CARLO_ITERATIONS` (10000) into every
response's `_metadata.config_details`, via `create_response_metadata`. That is
honest for the robustness endpoint (real Monte Carlo) but was emitted verbatim
on the sequential engine — which draws no samples at all — advertising a
Monte-Carlo sample budget for a deterministic computation.

Fix: `create_response_metadata` / `generate_config_details` take an explicit
`sampling: bool`. Each ENDPOINT declares its own nature (derive-don't-mirror: no
route list in the helper). `monte_carlo_samples` is emitted ONLY where
`sampling=True`. On the deterministic routes the key is ABSENT (absent-not-null).

Hard constraint: the robustness `_metadata` wire is BYTE-UNCHANGED — with
`sampling=True` the config_details dict keeps its exact keys AND order, so the
serialized bytes are identical.

RED-first (route level):
* counterfactual and sequential config_details carry monte_carlo_samples at HEAD
  -> the "absent" assertions FAIL at HEAD, pass after the fix.

Positive control (trap #13 — the test can SEE the key when present): asserted at
the HELPER level (`generate_config_details(sampling=True)` and the default both
carry monte_carlo_samples) AND, since A3 Lane M (2026-07-23), at the WIRE level on
the robustness V1 /analyze route.

A3 Lane M (2026-07-23): the pre-existing alias-drop bug is now FIXED.
RobustnessResponse.metadata is aliased `_metadata` with no populate_by_name, so the
V1 /analyze route's `RobustnessResponse(metadata=create_response_metadata(...))`
(passed by FIELD NAME) was silently dropped by Pydantic v2 and `_metadata` served
None on the wire — isl_version / config_fingerprint / config_details never reached
any V1 client. The route now attribute-assigns the metadata after construction
(matching the counterfactual/sequential live routes' working pattern), so the wire
carries a POPULATED `_metadata`. `test_robustness_wire_metadata_is_populated`
asserts that (and doubles as the wire-level positive control).
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.models.metadata import create_response_metadata, generate_config_details


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _config_details(response_json: dict) -> dict:
    return response_json["_metadata"]["config_details"]


# ---------------------------------------------------------------------------
# Valid payloads for the three LIVE routes.
# ---------------------------------------------------------------------------
def _robustness_payload() -> dict:
    return {
        "causal_model": {
            "nodes": ["price", "demand", "revenue"],
            "edges": [["price", "demand"], ["demand", "revenue"]],
        },
        "intervention_proposal": {"price": 55.0},
        "target_outcome": {"revenue": (95000.0, 105000.0)},
        "perturbation_radius": 0.1,
        "min_samples": 50,
        "confidence_level": 0.95,
    }


def _counterfactual_payload() -> dict:
    return {
        "model": {
            "variables": ["X", "Y"],
            "equations": {"Y": "2 * X"},
            "distributions": {"X": {"type": "normal", "parameters": {"mean": 5.0, "std": 1.0}}},
        },
        "intervention": {"X": 10.0},
        "outcome": "Y",
    }


def _sequential_payload() -> dict:
    return {
        "graph": {
            "nodes": [
                {"id": "invest", "type": "decision", "label": "Invest"},
                {"id": "market", "type": "chance", "label": "Market"},
                {"id": "success", "type": "terminal", "label": "Success", "payoff": 100000},
                {"id": "failure", "type": "terminal", "label": "Failure", "payoff": -20000},
                {"id": "no_invest", "type": "terminal", "label": "No Investment", "payoff": 0},
            ],
            "edges": [
                {"from": "invest", "to": "market", "action": "invest", "immediate_payoff": -10000},
                {"from": "invest", "to": "no_invest", "action": "wait"},
                {"from": "market", "to": "success", "outcome": "favorable", "probability": 0.6},
                {"from": "market", "to": "failure", "outcome": "unfavorable", "probability": 0.4},
            ],
            "stage_assignments": {
                "invest": 0,
                "market": 1,
                "success": 2,
                "failure": 2,
                "no_invest": 1,
            },
        },
        "stages": [
            {"stage_index": 0, "stage_label": "Investment", "decision_nodes": ["invest"]},
            {
                "stage_index": 1,
                "stage_label": "Market",
                "decision_nodes": [],
                "resolution_nodes": ["market"],
            },
            {"stage_index": 2, "stage_label": "Terminal", "decision_nodes": []},
        ],
        "discount_factor": 0.95,
        "risk_tolerance": "neutral",
    }


# ===========================================================================
# Wire-level: monte_carlo_samples present ONLY where sampling happens
# ===========================================================================
class TestConfigDetailsPerRouteHonesty:
    @pytest.mark.asyncio
    async def test_robustness_wire_metadata_is_populated(self, client):
        """A3 Lane M (2026-07-23): the robustness V1 /analyze wire emits a
        POPULATED `_metadata`.

        Previously the route constructed
        `RobustnessResponse(metadata=create_response_metadata(...))` by FIELD NAME
        against the `_metadata` alias with no populate_by_name, so Pydantic v2
        silently dropped it and the wire served `_metadata: null` —
        isl_version / config_fingerprint / config_details never reached any V1
        client. The route now attribute-assigns the metadata after construction
        (the counterfactual/sequential live routes' working pattern).

        Doubles as the WIRE-level positive control (trap #13): the suite can SEE a
        populated `_metadata` on a live route."""
        resp = await client.post("/api/v1/robustness/analyze", json=_robustness_payload())
        assert resp.status_code == 200, resp.text
        meta = resp.json()["_metadata"]
        assert meta is not None, "V1 _metadata dropped — alias-drop regression"
        assert isinstance(meta["isl_version"], str) and meta["isl_version"]
        assert isinstance(meta["config_fingerprint"], str) and meta["config_fingerprint"]
        assert meta["request_id"]
        cd = meta["config_details"]
        # config_details no longer carries monte_carlo_samples (C2, F-3): robustness
        # draws real Monte Carlo but its budget is request.min_samples-driven, NOT
        # MAX_MONTE_CARLO_ITERATIONS, so the fixed 10000 was a fabricated disclosure.
        assert "monte_carlo_samples" not in cd
        # the four determinism-relevant transparency keys remain
        assert list(cd.keys()) == [
            "confidence_level",
            "response_timeout",
            "redis_enabled",
            "deterministic_mode",
        ]

    @pytest.mark.asyncio
    async def test_counterfactual_omits_monte_carlo_samples(self, client):
        """RED at HEAD: the counterfactual config_details carries
        monte_carlo_samples: 10000. The route declares sampling=False, so the key
        is ABSENT (absent-not-null)."""
        resp = await client.post("/api/v1/causal/counterfactual", json=_counterfactual_payload())
        assert resp.status_code == 200, resp.text
        cd = _config_details(resp.json())
        assert "monte_carlo_samples" not in cd
        # the other transparency keys remain
        assert "confidence_level" in cd
        assert "deterministic_mode" in cd

    @pytest.mark.asyncio
    async def test_sequential_omits_monte_carlo_samples(self, client):
        """RED at HEAD: the sequential config_details carries
        monte_carlo_samples: 10000 despite the sequential engine drawing NO
        samples. The route declares sampling=False, so the key is ABSENT."""
        resp = await client.post("/api/v1/analysis/sequential", json=_sequential_payload())
        assert resp.status_code == 200, resp.text
        cd = _config_details(resp.json())
        assert "monte_carlo_samples" not in cd
        assert "confidence_level" in cd
        assert "deterministic_mode" in cd


# ===========================================================================
# Helper-level: config_details emits exactly four keys, no monte_carlo_samples
# ===========================================================================
class TestGenerateConfigDetails:
    def test_config_details_is_exactly_four_keys_in_order(self):
        """config_details emits EXACTLY the four determinism-relevant keys, in
        order — monte_carlo_samples is gone (C2, F-3). Presence positive control
        AND order/absence pin: re-adding the key makes this RED (C2 mutation)."""
        d = generate_config_details()
        assert list(d.keys()) == [
            "confidence_level",
            "response_timeout",
            "redis_enabled",
            "deterministic_mode",
        ]
        assert "monte_carlo_samples" not in d

    def test_create_response_metadata_config_details_omits_mc(self):
        """create_response_metadata's config_details carries the same four keys and
        never monte_carlo_samples (no route re-introduces the fabricated cap)."""
        cd = create_response_metadata("req_a").config_details
        assert "monte_carlo_samples" not in cd
        assert list(cd.keys()) == [
            "confidence_level",
            "response_timeout",
            "redis_enabled",
            "deterministic_mode",
        ]
