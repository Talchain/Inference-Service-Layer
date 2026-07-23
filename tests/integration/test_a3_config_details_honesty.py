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
carry monte_carlo_samples). NOTE: a WIRE-level "present on robustness" control is
impossible — RobustnessResponse.metadata is aliased `_metadata` with no
populate_by_name, so the V1 /analyze route's `metadata=create_response_metadata(...)`
(passed by field name) is silently dropped and `_metadata` is None on the wire
(pre-existing alias-drop bug, out of scope). `test_robustness_wire_metadata_is_none`
locks that byte-unchanged invariant so this change never accidentally starts
populating it.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.config import get_settings
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
    async def test_robustness_wire_metadata_is_none(self, client):
        """Byte-unchanged invariant: the robustness V1 /analyze wire carries
        `_metadata: None` (pre-existing alias-drop — the route constructs
        RobustnessResponse(metadata=...) by field name against the `_metadata`
        alias with no populate_by_name, so it is silently dropped). This change
        must NOT alter that; if the alias bug is later fixed, this tripwire flips
        and the robustness golden must be re-reviewed."""
        resp = await client.post("/api/v1/robustness/analyze", json=_robustness_payload())
        assert resp.status_code == 200, resp.text
        assert resp.json()["_metadata"] is None

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
# Helper-level: the sampling flag + robustness byte-order invariance
# ===========================================================================
class TestGenerateConfigDetailsSamplingFlag:
    def test_sampling_true_has_key_in_original_order(self):
        """sampling=True (the robustness/default path) is byte-order-invariant:
        monte_carlo_samples is present, first, and the full key order is
        unchanged -> the robustness _metadata JSON is byte-identical."""
        d = generate_config_details(sampling=True)
        assert list(d.keys()) == [
            "monte_carlo_samples",
            "confidence_level",
            "response_timeout",
            "redis_enabled",
            "deterministic_mode",
        ]
        assert d["monte_carlo_samples"] == get_settings().MAX_MONTE_CARLO_ITERATIONS

    def test_sampling_false_omits_key_only(self):
        """sampling=False drops monte_carlo_samples and NOTHING else; the
        remaining keys keep their relative order."""
        d = generate_config_details(sampling=False)
        assert "monte_carlo_samples" not in d
        assert list(d.keys()) == [
            "confidence_level",
            "response_timeout",
            "redis_enabled",
            "deterministic_mode",
        ]

    def test_create_response_metadata_threads_sampling_flag(self):
        """create_response_metadata forwards sampling to config_details."""
        with_mc = create_response_metadata("req_a", sampling=True).config_details
        without_mc = create_response_metadata("req_b", sampling=False).config_details
        assert "monte_carlo_samples" in with_mc
        assert "monte_carlo_samples" not in without_mc

    def test_default_preserves_key_for_untouched_callers(self):
        """The default is sampling=True so untouched/dark callers keep their
        current wire (monte_carlo_samples present)."""
        assert "monte_carlo_samples" in generate_config_details()
        assert "monte_carlo_samples" in create_response_metadata("req_c").config_details
