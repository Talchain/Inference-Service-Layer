"""A3 sequential-decision honesty remediation (Codex deep-review, 2026-07-22).

* F4a — the unread `monte_carlo_samples` request param is REMOVED (it was
        accepted + bounds-validated + documented but had ZERO readers; mcs in
        {100, 1k, 10k} produced identical output).
* F4b — the fabricated policy `value_distribution` (normal params with
        std = 0.2*|root|) is OMITTED (a made-up std reported as if measured; the
        engine draws no noise).
* F4c — the misnamed `information_value` (actually sqrt(Σ variance) = a
        payoff-unit outcome-dispersion magnitude, NOT value-of-information) is
        relabeled `resolved_uncertainty`, dropping the decision-difference claim.
* F4d — an ALL-ZERO-MASS chance node (every branch probability 0) is REJECTED
        -> 422 instead of being silently valued 0 (which collapsed the root to a
        wrong value at HTTP 200). A non-zero but non-unit sum is still
        renormalised — that is deliberate and load-bearing for the internal
        stage-sensitivity perturbation; see the engine comment.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.models.requests import SequentialAnalysisRequest


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _base_seq_request():
    """invest -> market(chance) -> {success, failure}; market probs specified."""
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
                "invest": 0, "market": 1, "success": 2, "failure": 2, "no_invest": 1,
            },
        },
        "stages": [
            {"stage_index": 0, "stage_label": "Investment", "decision_nodes": ["invest"]},
            {"stage_index": 1, "stage_label": "Market", "decision_nodes": [], "resolution_nodes": ["market"]},
            {"stage_index": 2, "stage_label": "Terminal", "decision_nodes": []},
        ],
        "discount_factor": 0.95,
        "risk_tolerance": "neutral",
    }


# ===========================================================================
# F4d — chance-node branch probabilities must form a valid distribution
# ===========================================================================
class TestF4dChanceProbabilities:
    @pytest.mark.asyncio
    async def test_all_zero_mass_chance_node_returns_422(self, client):
        """RED at HEAD: a chance node whose branches ALL have probability 0.0 is
        accepted, valued 0 (no mass, no normalisation), and the root collapses to
        a wrong value with HTTP 200. Reject the zero-mass distribution -> 422."""
        req = _base_seq_request()
        for e in req["graph"]["edges"]:
            if e["from"] == "market":
                e["probability"] = 0.0
        resp = await client.post("/api/v1/analysis/sequential", json=req)
        assert resp.status_code == 422, resp.text
        assert "market" in resp.text and "probabil" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_valid_distribution_still_200(self, client):
        """Positive control: a well-formed sum-to-1 distribution (0.6/0.4) is
        accepted (200) — the guard rejects only invalid distributions."""
        resp = await client.post("/api/v1/analysis/sequential", json=_base_seq_request())
        assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_omitted_probabilities_equal_split_still_200(self, client):
        """Positive control: OMITTED probabilities default to an equal split
        (sum-to-1) and must still 200 — the guard must not reject the equal-split
        default path."""
        req = _base_seq_request()
        for e in req["graph"]["edges"]:
            if e["from"] == "market":
                e.pop("probability", None)
        resp = await client.post("/api/v1/analysis/sequential", json=req)
        assert resp.status_code == 200, resp.text


# ===========================================================================
# F4a — remove the unread monte_carlo_samples param
# ===========================================================================
class TestF4aMonteCarloSamplesRemoved:
    def test_field_removed_from_request_model(self):
        """RED at HEAD: `monte_carlo_samples` is a field of the request model
        (accepted + bounds-validated + documented) despite having zero readers."""
        assert "monte_carlo_samples" not in SequentialAnalysisRequest.model_fields


# ===========================================================================
# F4b — omit the fabricated policy value_distribution
# ===========================================================================
class TestF4bValueDistributionOmitted:
    @pytest.mark.asyncio
    async def test_policy_has_no_value_distribution(self, client):
        """RED at HEAD: optimal_policy carries `value_distribution` = normal params
        with a fabricated std (0.2*|root|). Omitted — a made-up std must not be
        reported as if measured."""
        resp = await client.post("/api/v1/analysis/sequential", json=_base_seq_request())
        assert resp.status_code == 200, resp.text
        policy = resp.json()["optimal_policy"]
        assert "expected_total_value" in policy
        assert "value_distribution" not in policy


# ===========================================================================
# F4c — relabel the misnamed VOI
# ===========================================================================
class TestF4cInformationValueRelabelled:
    @pytest.mark.asyncio
    async def test_stage_analysis_uses_honest_uncertainty_name(self, client):
        """RED at HEAD: stage_analyses carry `information_value`, which is actually
        sqrt(Σ variance) — a payoff-unit dispersion magnitude, not value-of-
        information. It is relabeled `resolved_uncertainty`; the misnamed field is
        gone."""
        resp = await client.post("/api/v1/analysis/sequential", json=_base_seq_request())
        assert resp.status_code == 200, resp.text
        for sa in resp.json()["stage_analyses"]:
            assert "information_value" not in sa
            assert "resolved_uncertainty" in sa
            assert sa["resolved_uncertainty"] >= 0.0
