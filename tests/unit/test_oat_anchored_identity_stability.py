"""No stability claim from a blind probe (AIQ #72 5882847470).

An evaluated identity with a STATED level is anchored to each draw's status quo:
``T = o + k * (term - term_sq) + ...``. The one-at-a-time probe perturbs a factor under the reference
option, which moves the draw and its status quo together, so through that node it reads only the reference
option's own change. On Paul's served wire the reference option keeps price at £49 and MRR is the goal, so
``T = o`` whatever any factor does: every row read ``attribution_stability: negligible``, a statement about
the probe, not the factor. The stability read off that probe is withheld with a reason; every other field
keeps its shape and value.
"""

from __future__ import annotations

import copy

from typing import Any, Dict

import pytest

import src.services.robustness_analyzer_v2 as rav2

from src.services.robustness_analyzer_v2 import anchored_blind_factor_ids
from tests.unit.test_identity_zero_operand_unstated import GATED, zero_price
from tests.unit.test_r3_identity_evaluation import MRR, graph_of, v2_body, wire

STABILITY = ("attribution_stability", "elasticity_std", "rank_flip_rate", "stability_method")
CONFIDENCE = ("confidence", "confidence_source", "confidence_provenance")


def rows(body: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {r["node_id"]: r for r in body.get("factor_sensitivity") or []}


def cohort(d: Dict[str, Any]) -> set:
    return {u["node_id"] for u in d.get("parameter_uncertainties") or []}


@pytest.fixture(scope="module")
def anchored() -> Dict[str, Any]:
    return v2_body({**wire(), "n_samples": 400})


@pytest.fixture(scope="module")
def unwithheld() -> Dict[str, Any]:
    """The same request with the withhold switched off: the mutant, and the reference for "no other change"."""
    mp = pytest.MonkeyPatch()
    mp.setattr(rav2, "anchored_blind_factor_ids", lambda *a, **k: {})
    try:
        return v2_body({**wire(), "n_samples": 400})
    finally:
        mp.undo()


class TestTheProbeIsBlindOnPaulsWire:
    def test_precondition_the_probe_read_every_factor_as_negligible(self, unwithheld):
        # Without the withhold, the blind probe's claim is on every row: what this change removes.
        assert rows(unwithheld), "no factor_sensitivity rows"
        assert {r.get("attribution_stability") for r in rows(unwithheld).values()} == {"negligible"}

    def test_no_row_carries_a_stability_claim(self, anchored):
        assert set(rows(anchored)) == cohort(wire())  # no row is dropped
        for node_id, r in rows(anchored).items():
            for field in STABILITY:
                assert r.get(field) is None, (node_id, field)

    def test_every_other_field_is_unchanged(self, anchored, unwithheld):
        a, b = rows(anchored), rows(unwithheld)
        assert set(a) == set(b)
        for node_id in a:
            strip = lambda r: {
                k: v for k, v in r.items() if k not in STABILITY + CONFIDENCE
            }  # noqa: E731
            assert strip(a[node_id]) == strip(b[node_id]), node_id

    def test_the_confidence_is_the_disclosed_structural_method_not_the_probe(
        self, anchored, unwithheld
    ):
        """ISL's own ``confidence`` reads the bootstrap first and, with no stability, its graph-structural method,
        stamped as such (``confidence_source``/``method_version``): never a figure from the blind probe.
        """
        for node_id, r in rows(anchored).items():
            assert r.get("confidence_source") == "graph_structural", node_id
            assert (r.get("confidence_provenance") or {}).get(
                "method_version"
            ) == "graph-structural-v1", node_id
        assert {r.get("confidence_source") for r in rows(unwithheld).values()} == {
            "bootstrap_sampling"
        }

    def test_the_withhold_says_why(self, anchored):
        (critique,) = [c for c in anchored["critiques"] if c["code"] == "FACTOR_STABILITY_ANCHORED"]
        assert sorted(critique.get("affected_node_ids") or []) == sorted(cohort(wire()))
        assert "anchored to its stated level" in critique["message"] and MRR in critique["message"]


class TestNotAnchoredNoChange:
    def test_an_evaluated_identity_with_no_stated_level_is_not_anchored(self):
        """T = term + A + L per draw (no status quo in it): the probe sees the factors. #214's gated rows are
        omitted; every other row keeps its stability."""
        body = v2_body({**zero_price(stated=False), "n_samples": 400})
        assert not [c for c in body["critiques"] if c["code"] == "FACTOR_STABILITY_ANCHORED"]
        kept = rows(body)
        assert kept and not GATED & set(kept)
        for node_id, r in kept.items():
            assert r.get("stability_method") is not None, node_id

    def test_a_linear_graph_is_not_anchored(self):
        body = v2_body({**wire(identity=None), "n_samples": 400})
        assert not [c for c in body["critiques"] if c["code"] == "FACTOR_STABILITY_ANCHORED"]
        assert all(r.get("stability_method") is not None for r in rows(body).values())


class TestOnlyAFactorSeenSolelyThroughTheAnchorIsBlind:
    """The goal sits downstream of the anchored MRR: a factor with its own path to the goal is seen."""

    def graph(self):
        d = copy.deepcopy(wire())
        nodes = {n["id"]: n for n in d["graph"]["nodes"]}
        nodes[MRR]["kind"] = "factor"
        d["graph"]["nodes"] += [
            {"id": "profit", "kind": "goal", "label": "Profit"},
            {"id": "costs", "kind": "factor", "label": "Costs", "observed_state": {"value": 0.3}},
        ]
        d["graph"]["edges"] += [
            {"from": MRR, "to": "profit", "strength": {"mean": 1.0, "std": 0.01}},
            {"from": "costs", "to": "profit", "strength": {"mean": -0.5, "std": 0.05}},
            {"from": "other_mrr_growth", "to": "profit", "strength": {"mean": 0.2, "std": 0.05}},
        ]
        d["goal_node_id"] = "profit"
        return graph_of(d)

    def test_the_bypass_is_seen_and_the_rest_is_blind(self):
        factors = ["pro_plan_price", "pro_paying_subscribers", "other_mrr_growth", "costs", MRR]
        blind = anchored_blind_factor_ids(self.graph(), factors, "profit")
        assert set(blind) == {"pro_plan_price", "pro_paying_subscribers", MRR}
        assert all(anchors == [MRR] for anchors in blind.values())
