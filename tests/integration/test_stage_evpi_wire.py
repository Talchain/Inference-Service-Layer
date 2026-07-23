"""S3 (A3 VOI honesty, D-23.8) — stage_evpi on the sequential wire; the removed
`optimal_waiting_value` heuristic must be gone.

End-to-end through /api/v1/analysis/sequential (the one selectively-mounted phase4
route). Reuses the A3 risk_neutral honesty fixture; pins the stage-0 stage_evpi to
the hand-derived 11220 and asserts optimal_waiting_value is absent everywhere.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app

URL = "/api/v1/analysis/sequential"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _base_seq_request():
    """invest -> market(chance) -> {success, failure}; the F4c/A3 fixture."""
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
            "stage_assignments": {"invest": 0, "market": 1, "success": 2, "failure": 2, "no_invest": 1},
        },
        "stages": [
            {"stage_index": 0, "stage_label": "Investment", "decision_nodes": ["invest"]},
            {"stage_index": 1, "stage_label": "Market", "decision_nodes": [], "resolution_nodes": ["market"]},
            {"stage_index": 2, "stage_label": "Terminal", "decision_nodes": []},
        ],
        "discount_factor": 0.95,
        "risk_tolerance": "neutral",
    }


class TestStageEvpiWire:
    @pytest.mark.asyncio
    async def test_stage_evpi_present_and_pinned_on_wire(self, client):
        resp = await client.post(URL, json=_base_seq_request())
        assert resp.status_code == 200, resp.text
        stages = {sa["stage_index"]: sa for sa in resp.json()["stage_analyses"]}
        assert "stage_evpi" in stages[0]
        assert stages[0]["stage_evpi"] == pytest.approx(11220.0, rel=1e-12)

    @pytest.mark.asyncio
    async def test_optimal_waiting_value_gone_from_wire(self, client):
        """The removed sqrt-heuristic key must not appear on ANY stage."""
        resp = await client.post(URL, json=_base_seq_request())
        assert resp.status_code == 200, resp.text
        for sa in resp.json()["stage_analyses"]:
            assert "optimal_waiting_value" not in sa

    @pytest.mark.asyncio
    async def test_stage_evpi_null_for_decisionless_stage(self, client):
        """A stage with no decision node reports stage_evpi = null (NOT 0), with
        stage_evpi_status='no_decision_node' disclosing the cause. This endpoint does
        not omit None fields — the status distinguishes 'no decision to inform' from a
        genuine 0 and from a size-skip (F-3)."""
        resp = await client.post(URL, json=_base_seq_request())
        stages = {sa["stage_index"]: sa for sa in resp.json()["stage_analyses"]}
        assert stages[1]["stage_evpi"] is None
        assert stages[1]["stage_evpi_status"] == "no_decision_node"
        assert stages[2]["stage_evpi"] is None
        # the computed stage carries a real value and no status
        assert stages[0]["stage_evpi"] == pytest.approx(11220.0, rel=1e-12)
        assert stages[0]["stage_evpi_status"] is None

    @pytest.mark.asyncio
    async def test_stage_evpi_over_cap_returns_200_fast_with_status(self, client):
        """F-3: a LEGAL request that would enumerate 2^K joint cells returns 200
        FAST (not a hang / not a 422) with stage_evpi=null + status. K=30 => 2^30
        cells would take ~hours WITHOUT the guard; with it, milliseconds."""
        import time as _t

        # decision D with 30 actions, each -> its own binary chance node.
        nodes = [{"id": "D", "type": "decision", "label": "D"},
                 {"id": "hi", "type": "terminal", "label": "hi", "payoff": 200},
                 {"id": "lo", "type": "terminal", "label": "lo", "payoff": -50}]
        edges = []
        sa = {"D": 0, "hi": 2, "lo": 2}
        for i in range(30):
            nodes.append({"id": f"c{i}", "type": "chance", "label": f"c{i}"})
            sa[f"c{i}"] = 1
            edges.append({"from": "D", "to": f"c{i}", "action": f"a{i}"})
            edges.append({"from": f"c{i}", "to": "hi", "outcome": "up", "probability": 0.5})
            edges.append({"from": f"c{i}", "to": "lo", "outcome": "down", "probability": 0.5})
        req = {
            "graph": {"nodes": nodes, "edges": edges, "stage_assignments": sa},
            "stages": [
                {"stage_index": 0, "stage_label": "s0", "decision_nodes": ["D"]},
                {"stage_index": 1, "stage_label": "s1", "decision_nodes": []},
                {"stage_index": 2, "stage_label": "s2", "decision_nodes": []},
            ],
            "discount_factor": 0.95,
            "risk_tolerance": "neutral",
        }
        t0 = _t.monotonic()
        resp = await client.post(URL, json=req)
        elapsed = _t.monotonic() - t0
        assert resp.status_code == 200, resp.text  # honest skip, NOT a 422
        sd = {sa_["stage_index"]: sa_ for sa_ in resp.json()["stage_analyses"]}
        assert sd[0]["stage_evpi"] is None
        assert sd[0]["stage_evpi_status"] == "skipped_joint_space_too_large"
        assert elapsed < 5.0, f"stage_evpi enumeration was not bounded: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_resolved_uncertainty_and_flexibility_untouched(self, client):
        """S3 leaves the honest neighbours in place: resolved_uncertainty (honestly
        labelled) and value_of_flexibility (a real decision-difference) still ride."""
        resp = await client.post(URL, json=_base_seq_request())
        body = resp.json()
        assert "value_of_flexibility" in body
        for sa in body["stage_analyses"]:
            assert "resolved_uncertainty" in sa and sa["resolved_uncertainty"] >= 0.0
