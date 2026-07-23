"""Unique / capped parameter_uncertainties validation (Codex F8.3).

Positive control PC-1 (memory trap #13): the pre-F8 behaviour — proven on base
in acceptance-evidence/a3-verify-2026-07-16/codex-f8-isl-BUILD.md — was that 3
identical `demand` parameter_uncertainties produced 3 BYTE-IDENTICAL factor_evpi
rows (a full EVPI Monte-Carlo pass each, at zero admitted cost). This suite
asserts the flipped behaviour: a typed 422 at parse time, plus a defensive dedup
in the analyzer so a direct internal caller cannot re-trigger the multiplier.
"""
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

ENDPOINT = "/api/v1/robustness/analyze/v2"


def _base_body(n_factor_nodes: int) -> dict:
    """A small valid graph with n_factor_nodes observed factor nodes + a goal."""
    nodes = [
        {
            "id": f"f{i}",
            "kind": "factor",
            "label": f"F{i}",
            "observed_state": {"value": 50.0, "std": 5.0},
        }
        for i in range(n_factor_nodes)
    ]
    nodes.append({"id": "goal", "kind": "outcome", "label": "Goal"})
    edges = [
        {
            "from": f"f{i}",
            "to": "goal",
            "exists_probability": 0.9,
            "strength": {"mean": 0.3, "std": 0.1},
        }
        for i in range(n_factor_nodes)
    ]
    return {
        "graph": {"nodes": nodes, "edges": edges},
        "options": [
            {"id": "low", "label": "Low", "interventions": {"f0": 0.3}},
            {"id": "high", "label": "High", "interventions": {"f0": 0.7}},
        ],
        "goal_node_id": "goal",
        "n_samples": 1000,
        "seed": 42,
        "include_voi": True,
    }


# ---------------------------------------------------------------------------
# PC-1 — duplicate node_ids rejected at parse time
# ---------------------------------------------------------------------------
class TestDuplicateRejected:
    def test_three_identical_uncertainties_raise_validation_error(self):
        body = _base_body(2)
        body["parameter_uncertainties"] = [
            {"node_id": "f0", "distribution": "normal", "std": 5.0} for _ in range(3)
        ]
        with pytest.raises(ValidationError) as exc:
            RobustnessRequestV2(**body)
        msgs = " ".join(d["msg"] for d in exc.value.errors())
        assert "Duplicate parameter_uncertainties node_ids" in msgs
        assert "f0" in msgs

    def test_duplicate_endpoint_returns_422(self):
        client = TestClient(app)
        body = _base_body(2)
        body["parameter_uncertainties"] = [
            {"node_id": "f0", "distribution": "normal", "std": 5.0} for _ in range(3)
        ]
        resp = client.post(ENDPOINT, json=body, headers={"X-ISL-Response-Version": "2"})
        assert resp.status_code == 422
        assert "Duplicate parameter_uncertainties" in json.dumps(resp.json())

    def test_distinct_node_ids_accepted(self):
        """Positive control: distinct node_ids are NOT rejected."""
        body = _base_body(3)
        body["parameter_uncertainties"] = [
            {"node_id": f"f{i}", "distribution": "normal", "std": 5.0} for i in range(3)
        ]
        req = RobustnessRequestV2(**body)  # must not raise
        assert len(req.parameter_uncertainties) == 3


# ---------------------------------------------------------------------------
# Boundary — max_length cap
# ---------------------------------------------------------------------------
class TestLengthCap:
    @staticmethod
    def _chain_body(n_nodes: int) -> dict:
        """A valid n_nodes-node chain graph (last node is the goal/outcome).

        n_nodes is capped at the 50-node graph limit, so it can host up to 50
        distinct parameter_uncertainties referencing n0..n{n-1}.
        """
        nodes = [
            {
                "id": f"n{i}",
                "kind": "factor",
                "label": f"N{i}",
                "observed_state": {"value": 50.0, "std": 5.0},
            }
            for i in range(n_nodes)
        ]
        nodes[-1]["kind"] = "outcome"
        edges = [
            {
                "from": f"n{i}",
                "to": f"n{i + 1}",
                "exists_probability": 0.9,
                "strength": {"mean": 0.3, "std": 0.1},
            }
            for i in range(n_nodes - 1)
        ]
        return {
            "graph": {"nodes": nodes, "edges": edges},
            "options": [{"id": "o0", "label": "O0", "interventions": {"n0": 0.5}}],
            "goal_node_id": f"n{n_nodes - 1}",
            "n_samples": 500,
            "seed": 1,
            "include_voi": True,
        }

    def test_exactly_50_unique_admits(self):
        body = self._chain_body(50)  # graph at the 50-node cap
        body["parameter_uncertainties"] = [
            {"node_id": f"n{i}", "distribution": "normal", "std": 5.0} for i in range(50)
        ]
        req = RobustnessRequestV2(**body)  # exactly at the field cap → allowed
        assert len(req.parameter_uncertainties) == 50

    def test_51_uncertainties_rejected(self):
        # Only 50 distinct nodes can exist (node cap), so a 51st entry must repeat
        # one; the field max_length=50 rejects the 51-element list before the
        # uniqueness model-validator even runs. Both are correct rejections.
        body = self._chain_body(50)
        pu = [{"node_id": f"n{i}", "distribution": "normal", "std": 5.0} for i in range(50)]
        pu.append({"node_id": "n0", "distribution": "normal", "std": 5.0})  # 51st
        body["parameter_uncertainties"] = pu
        with pytest.raises(ValidationError) as exc:
            RobustnessRequestV2(**body)
        errs = exc.value.errors()
        assert any(
            "at most 50" in d["msg"] or "Duplicate" in d["msg"] or "too_long" in d["type"]
            for d in errs
        )


# ---------------------------------------------------------------------------
# Defensive dedup in _compute_evpi (defence-in-depth)
# ---------------------------------------------------------------------------
class TestEvpiDefensiveDedup:
    def test_internal_caller_with_duplicates_gets_one_row(self):
        """A direct internal caller bypasses model validation; the analyzer must
        still dedup so a repeated node_id cannot re-run the EVPI multiplier."""
        body = _base_body(1)
        body["parameter_uncertainties"] = [{"node_id": "f0", "distribution": "normal", "std": 5.0}]
        req = RobustnessRequestV2(**body)
        # Simulate a bypassed caller: inject duplicates AFTER validation.
        dup = req.parameter_uncertainties[0]
        req.parameter_uncertainties = [dup, dup, dup]
        resp = RobustnessAnalyzerV2().analyze(req)
        rows = resp.p_win_sensitivity or []
        # Deduped to the single unique node_id — NOT 3 rows.
        assert len(rows) == 1
        assert rows[0]["factor_id"] == "f0"
