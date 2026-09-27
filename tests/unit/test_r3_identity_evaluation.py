"""R3 slice 1 — a declared accounting identity is EVALUATED, in user units (AIQ rows R3-1, R3-8).

Paul's pricing brief: MRR = price x paying subscribers (+ other MRR growth). The linear model
reads it as guessed slopes; the identity reads it exactly. Rulings: AIQ #70 5859633012 (CEE
declares, ISL never infers), 5860087988 (tau = 5%, withhold codes, ratio form, R3-1 split).

The graph is the UNEDITED served wire (PLoT a6da42b capture of Paul's a295e4a1) plus the
declaration CEE mints for it and the execution frames PLoT resolves (runtime metadata):
    mrr                   cap  125,000   (0.6   = £75,000)
    pro_plan_price        cap      200   (0.245 = £49)
    pro_paying_subscribers pair  10,000  (0.15  = 1,500)
    other_mrr_growth      pair  50,000   (0.02  = £1,000)
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import GraphV2, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import SCMEvaluatorV2, resolve_identity_plans

SERVED_WIRE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "anchored_delta"
    / "paul_a295e4a1_served_wire_plot_a6da42b.json"
)

MRR, PRICE, SUBS, OTHER = "mrr", "pro_plan_price", "pro_paying_subscribers", "other_mrr_growth"
FRAMES = {
    MRR: {"frame": 125_000.0, "carrier": "cap"},
    PRICE: {"frame": 200.0, "carrier": "cap"},
    SUBS: {"frame": 10_000.0, "carrier": "pair"},
    OTHER: {"frame": 50_000.0, "carrier": "pair"},
}
PRODUCT = {"operation": "product", "factor_ids": [PRICE, SUBS], "stated_in_brief": True}
WITH_ADDEND = {**PRODUCT, "addends": [OTHER]}

# R3-1 (AIQ 5860087988): £59 at unchanged subscribers, other MRR growth declared as an addend.
# The product term is anchored at o - A = £75,000 - £1,000 and scaled by 59/49.
R3_1_GBP = 74_000.0 * (59.0 / 49.0 - 1.0)  # 15,102.0408...


def wire(
    identity: Optional[Dict[str, Any]] = WITH_ADDEND, frames: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    d = json.loads(SERVED_WIRE.read_text())
    nodes = {n["id"]: n for n in d["graph"]["nodes"]}
    if identity is not None:
        nodes[MRR]["nonlinear_identity"] = copy.deepcopy(identity)
    for node_id, frame in (FRAMES if frames is None else frames).items():
        nodes[node_id]["execution_frame"] = dict(frame)
    return d


def graph_of(d: Dict[str, Any]) -> GraphV2:
    return RobustnessRequestV2.model_validate(d).graph


def means(d: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    return {(e["from"], e["to"]): e["strength"]["mean"] for e in d["graph"]["edges"]}


def roots_at_today(d: Dict[str, Any]) -> Dict[str, float]:
    """Every root at its level today: the deterministic query R3-1 is (no sampling at all)."""
    targets = {e["to"] for e in d["graph"]["edges"]}
    return {
        n["id"]: n["observed_state"]["value"]
        for n in d["graph"]["nodes"]
        if n["id"] not in targets and (n.get("observed_state") or {}).get("value") is not None
    }


def query_gbp(d: Dict[str, Any], interventions: Dict[str, float]) -> float:
    """MRR effect of ``interventions`` against the status quo, in £, edges at their means."""
    evaluator = SCMEvaluatorV2(graph_of(d))
    edges, factors = means(d), roots_at_today(d)
    option = evaluator.evaluate(edges, interventions, MRR, factor_values=factors)
    today = evaluator.evaluate(edges, {}, MRR, factor_values=factors)
    return (option - today) * FRAMES[MRR]["frame"]


# Subscribers "unchanged": set to their level today, which B1a-5 reads as no change, and pinned so
# price's two paths into subscribers are cut (R3-1-path: that is a QUERY, not what the graph does).
AT_59_SUBS_HELD = {PRICE: 0.295, SUBS: 0.15}


class TestR3_1TheProductAtUnchangedSubscribers:
    def test_r3_1_is_plus_15_102_04(self):
        effect = query_gbp(wire(), AT_59_SUBS_HELD)
        assert abs(effect - R3_1_GBP) <= 1e-6
        assert round(effect, 2) == 15_102.04

    def test_the_status_quo_sits_at_the_stated_75k(self):
        d = wire()
        evaluator = SCMEvaluatorV2(graph_of(d))
        today = evaluator.evaluate(means(d), {}, MRR, factor_values=roots_at_today(d))
        assert today * FRAMES[MRR]["frame"] == pytest.approx(75_000.0, abs=1e-6)

    def test_control_the_linear_model_gives_the_slope_answer(self):
        """No declaration: the same query is the guessed-slope figure, not +£15,102.04."""
        effect = query_gbp(wire(identity=None), AT_59_SUBS_HELD)
        assert abs(effect - R3_1_GBP) > 1_000.0

    def test_without_the_addend_the_whole_75k_scales(self):
        """Undeclared, other MRR growth stays a sampled belief edge (AIQ item 5): the product term is
        anchored at o - L_sq, the edge's own contribution, measured rather than asserted."""
        d = wire(identity=PRODUCT)
        belief = 0.4 * 0.02 * FRAMES[MRR]["frame"]  # other_mrr_growth -> mrr at its mean, £
        expected = (75_000.0 - belief) * (59.0 / 49.0 - 1.0)
        assert abs(query_gbp(d, AT_59_SUBS_HELD) - expected) <= 1e-6

    def test_r3_8_mutant_operands_left_normalised_misses(self, monkeypatch):
        """R3-8: every frame read as 1 (f on normalised operands) -> the reconciliation compares
        0.245 x 0.15 + 0.02 with 0.6 and the identity is withheld; R3-1 is not reached."""
        d = wire(frames={k: {**v, "frame": 1.0} for k, v in FRAMES.items()})
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        assert plan.withheld_reason == rav2.IDENTITY_INCONSISTENT
        assert abs(query_gbp(d, AT_59_SUBS_HELD) - R3_1_GBP) > 1_000.0


class TestReconciliation:
    """AIQ 5860087988 items 1-2: tau = 5%, ONE constant; above it the identity is WITHHELD."""

    def test_pauls_case_is_0_67_percent_with_the_addend(self):
        (plan,) = resolve_identity_plans(graph_of(wire())).values()
        assert plan.evaluated
        assert plan.reconstructed == pytest.approx(74_500.0)
        assert plan.stated == pytest.approx(75_000.0)
        assert plan.mismatch_share == pytest.approx(500.0 / 75_000.0)

    def test_pauls_case_is_2_0_percent_without_it(self):
        (plan,) = resolve_identity_plans(graph_of(wire(identity=PRODUCT))).values()
        assert plan.evaluated
        assert plan.mismatch_share == pytest.approx(1_500.0 / 75_000.0)

    def test_a_real_contradiction_is_withheld_not_approximated(self):
        """1,000 subscribers against £75,000: 49 x 1,000 + 1,000 = £50,000, a third out."""
        d = wire()
        (subs,) = [n for n in d["graph"]["nodes"] if n["id"] == SUBS]
        subs["observed_state"].update(value=0.1, raw_value=1_000)
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        assert plan.withheld_reason == rav2.IDENTITY_INCONSISTENT
        assert plan.reconstructed == pytest.approx(50_000.0)

    def test_tau_is_one_constant_at_five_percent(self):
        assert rav2.IDENTITY_RECONCILIATION_TOLERANCE == 0.05


class TestWithheldNeverApproximated:
    def _plan(self, d: Dict[str, Any]):
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        return plan

    def test_an_operand_with_no_frame(self):
        frames = {k: v for k, v in FRAMES.items() if k != SUBS}
        assert self._plan(wire(frames=frames)).withheld_reason == rav2.IDENTITY_FRAME_MISSING

    def test_the_target_with_no_frame(self):
        frames = {k: v for k, v in FRAMES.items() if k != MRR}
        assert self._plan(wire(frames=frames)).withheld_reason == rav2.IDENTITY_FRAME_MISSING

    def test_an_operand_with_no_level(self):
        d = wire()
        (subs,) = [n for n in d["graph"]["nodes"] if n["id"] == SUBS]
        subs["observed_state"] = None
        assert self._plan(d).withheld_reason == rav2.IDENTITY_OPERAND_MISSING

    def test_a_zero_operand(self):
        d = wire()
        (price,) = [n for n in d["graph"]["nodes"] if n["id"] == PRICE]
        price["observed_state"]["value"] = 0.0
        assert self._plan(d).withheld_reason == rav2.IDENTITY_ZERO_LEVEL

    def test_a_withheld_identity_leaves_the_node_linear(self):
        """The evaluator does not evaluate it; the analysis withholds what depends on it."""
        frames = {k: v for k, v in FRAMES.items() if k != SUBS}
        effect = query_gbp(wire(frames=frames), AT_59_SUBS_HELD)
        assert effect == pytest.approx(query_gbp(wire(identity=None), AT_59_SUBS_HELD), abs=1e-9)


class TestNoStatedTarget:
    """AIQ 5860087988 item 4: with no level for the target, it is evaluated from its inputs."""

    def test_the_status_quo_is_the_inputs_and_the_effect_is_unscaled(self):
        d = wire()
        (mrr,) = [n for n in d["graph"]["nodes"] if n["id"] == MRR]
        mrr["observed_state"] = None  # no level today; its frame still rides execution_frame
        d["goal_threshold"] = None
        d["goal_threshold_frame"] = None
        d["goal_constraints"] = []
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        assert plan.evaluated and plan.target_level is None
        # 49 x 1,500 + 1,000 = £74,500 today; £59 adds £10 x 1,500, unscaled.
        assert query_gbp(d, AT_59_SUBS_HELD) == pytest.approx(15_000.0, abs=1e-6)


class TestTheDeclarationIsStrict:
    """PLoT refuses an unknown key or operation (rung a); ISL's reader is strict for the same reason."""

    @pytest.mark.parametrize(
        "identity",
        [
            {**PRODUCT, "operation": "ratio"},
            {**PRODUCT, "weights": [1, 2]},
            {**PRODUCT, "factor_ids": [PRICE, PRICE]},
            {**PRODUCT, "addends": [PRICE]},
            {**PRODUCT, "factor_ids": [PRICE, "not_a_node"]},
            {**PRODUCT, "factor_ids": [PRICE, "monthly_churn"]},  # a node, but not mrr's parent
        ],
    )
    def test_a_malformed_declaration_is_refused(self, identity):
        with pytest.raises(Exception):
            graph_of(wire(identity=identity))

    def test_a_graph_with_no_declaration_has_no_plan(self):
        assert resolve_identity_plans(graph_of(wire(identity=None))) == {}


# ---------------------------------------------------------------------------------------------------------
# Withheld on the decision path -> the analysis is withheld (blocked 422), never approximated
# ---------------------------------------------------------------------------------------------------------


def inconsistent_wire() -> Dict[str, Any]:
    d = wire()
    (subs,) = [n for n in d["graph"]["nodes"] if n["id"] == SUBS]
    subs["observed_state"].update(value=0.1, raw_value=1_000)
    return d


class TestAWithheldIdentityWithholdsTheAnalysis:
    def test_the_route_returns_the_blocked_422_naming_both_figures(self):
        from fastapi.testclient import TestClient

        from src.api.main import app

        response = TestClient(app).post(
            "/api/v1/robustness/analyze/v2",
            json=inconsistent_wire(),
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422, response.text
        body = response.json()
        assert body["analysis_status"] == "blocked"
        (critique,) = [c for c in body["critiques"] if c["code"] == "IDENTITY_NOT_EVALUATED"]
        assert critique["severity"] == "blocker"
        assert "identity_inconsistent" in critique["message"]
        assert "50,000.00" in critique["message"] and "75,000.00" in critique["message"]
        assert critique["affected_node_ids"] == [MRR, PRICE, SUBS, OTHER]
        assert "results" not in body and "options" not in body  # no number leaks

    def test_the_analyzer_refuses_every_other_caller(self):
        request = RobustnessRequestV2.model_validate(inconsistent_wire())
        with pytest.raises(rav2.IdentityNotEvaluatedError):
            rav2.RobustnessAnalyzerV2().analyze(request)

    def test_an_evaluated_identity_is_not_blocked(self):
        assert rav2.identity_blocking_critiques(RobustnessRequestV2.model_validate(wire())) == []

    def test_a_withheld_identity_off_every_decision_path_does_not_block(self):
        """Declared on a node that reaches neither the goal nor a limit: nothing depends on it."""
        d = wire(identity=None)
        d["graph"]["nodes"].append(
            {
                "id": "side_total",
                "kind": "factor",
                "label": "Side total",
                "nonlinear_identity": {
                    "operation": "sum",
                    "factor_ids": [OTHER],
                    "stated_in_brief": False,
                },
            }
        )
        d["graph"]["edges"].append(
            {"from": OTHER, "to": "side_total", "strength": {"mean": 1.0, "std": 0.01}}
        )
        request = RobustnessRequestV2.model_validate(d)
        (plan,) = resolve_identity_plans(request.graph).values()
        assert plan.withheld_reason == rav2.IDENTITY_FRAME_MISSING
        assert rav2.identity_blocking_critiques(request) == []


# ---------------------------------------------------------------------------------------------------------
# The Monte Carlo run with the identity evaluated (R3-1-MC: MEASURED, reported with its SE)
# ---------------------------------------------------------------------------------------------------------


class TestTheMonteCarloRun:
    @pytest.fixture(scope="class")
    def response(self):
        d = wire()
        d["options"] = d["options"] + [{"id": "hold", "label": "Status quo", "interventions": {}}]
        return rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))

    def _samples(self, response, option_id):
        (result,) = [r for r in response.results if r.option_id == option_id]
        import numpy as np

        return np.array(result.outcome_distribution.samples)

    def test_the_status_quo_is_the_stated_75k_on_every_draw(self, response):
        import numpy as np

        hold = self._samples(response, "hold") * FRAMES[MRR]["frame"]
        assert np.allclose(hold, 75_000.0, atol=1e-6)

    def test_keep_current_is_the_status_quo_on_every_draw(self, response):
        import numpy as np

        keep = self._samples(response, "keep_current_49_price")
        assert np.array_equal(keep, self._samples(response, "hold"))

    def test_59_is_above_keep_current_through_the_identity(self, response):
        """R3-1-MC is MEASURED: price also moves subscribers through two paths, so it is not the
        held-subscribers +£15,102.04 (R3-1-path). Only its sign and scale are asserted here."""
        diff = (
            self._samples(response, "increase_price_to_59")
            - self._samples(response, "keep_current_49_price")
        ) * FRAMES[MRR]["frame"]
        assert 5_000.0 < float(diff.mean()) < 20_000.0
