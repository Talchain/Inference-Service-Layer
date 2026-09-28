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
        anchored at o - L, the edge's EXPECTED contribution (AIQ #72 5866317942: one central k)."""
        d = wire(identity=PRODUCT)
        # other_mrr_growth -> mrr at its effective strength, mean 0.4 x exists_probability 0.8, £800
        belief = 0.8 * 0.4 * 0.02 * FRAMES[MRR]["frame"]
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


# ---------------------------------------------------------------------------------------------------------
# R3-5: the consumer matrix — every output that MOVES under an evaluated identity is named; an unnamed
# move is RED. (Measured on the served wire, with and without CEE's declaration.)
# ---------------------------------------------------------------------------------------------------------

# RECOMPUTED on the identity: the option comparison and everything read from it.
RECOMPUTED = {
    "results",  # outcome samples, P(goal), win share per option
    "recommendation_confidence",
    "sensitivity",  # edge elasticities of the win share (all 0.0 here: £59 wins every draw)
    "factor_sensitivity",
    "robustness",  # fragile edges (none here: no draw flips the winner)
    "edge_e_values",
    "factor_flip_values",
    "factor_evppi",  # R3-6, MEASURED: 0.0, below_resolution (see below)
    "critiques",  # analysis critiques are recomputed on the new samples
    "identity_evaluations",  # the new disclosure itself (absent when nothing is declared)
    "structural_influence",  # R3-5: every factor node's influence (absent without an evaluated identity)
    "metadata",  # execution_time_ms differs on ANY two runs; edge_existence_rates move through the
    # tie-break coupling (£59 no longer ties, so the edge stream is consumed differently)
}


class TestR3_5TheConsumerMatrix:
    @pytest.fixture(scope="class")
    def pair(self):
        analyzer = rav2.RobustnessAnalyzerV2()
        on = analyzer.analyze(RobustnessRequestV2.model_validate(wire(identity=PRODUCT)))
        off = analyzer.analyze(RobustnessRequestV2.model_validate(wire(identity=None)))
        return on.model_dump(), off.model_dump()

    def test_every_move_is_named(self, pair):
        on, off = pair
        moved = {
            k
            for k in on
            if json.dumps(on[k], sort_keys=True, default=str)
            != json.dumps(off[k], sort_keys=True, default=str)
        }
        assert moved == RECOMPUTED

    def test_r3_6_evppi_of_subscribers_is_measured(self, pair):
        """R3-6: MEASURED, never predicted. With the identity £59 leads on every draw, so no single
        figure can change which option leads: EVPPI(subscribers) is 0 and below resolution."""
        on, _ = pair
        (row,) = [r for r in on["factor_evppi"] if r["factor_id"] == SUBS]
        assert row["evppi"] == 0.0
        assert row["status"] == "below_resolution"


# ---------------------------------------------------------------------------------------------------------
# The disclosure: declared is not evaluated (R3-4). Only evaluated=true licenses a numerical claim.
# ---------------------------------------------------------------------------------------------------------


def v2_body(d: Dict[str, Any]) -> Dict[str, Any]:
    from fastapi.testclient import TestClient

    from src.api.main import app

    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2", json=d, headers={"X-ISL-Response-Version": "2"}
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestTheDisclosureOnTheV2Wire:
    def test_an_evaluated_identity_says_so_with_its_reconciliation(self):
        (entry,) = v2_body(wire())["identity_evaluations"]
        assert entry["node_id"] == MRR and entry["operation"] == "product"
        assert entry["factor_ids"] == [PRICE, SUBS] and entry["addends"] == [OTHER]
        assert entry["evaluated"] is True and entry.get("withheld_reason") is None
        assert entry["level_source"] == "stated_level"
        assert entry["stated_in_brief"] is True
        rec = entry["reconciliation"]
        assert rec["reconstructed"] == pytest.approx(74_500.0)
        assert rec["stated"] == pytest.approx(75_000.0)
        assert rec["mismatch_share"] == pytest.approx(500.0 / 75_000.0)

    def test_no_declaration_no_key(self):
        """A graph that declares nothing is unchanged on the wire."""
        assert v2_body(wire(identity=None, frames={})).get("identity_evaluations") is None

    def test_a_declared_but_unused_identity_says_it_was_not_evaluated(self):
        d = wire(identity=None, frames={})
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
        (entry,) = v2_body(d)["identity_evaluations"]
        assert entry["evaluated"] is False
        assert entry["withheld_reason"] == "identity_frame_missing"
        assert entry.get("level_source") is None


# ---------------------------------------------------------------------------------------------------------
# SUM (R3-2 shape, synthetic until CEE mints `sum` on 0e19bb82 — MG rung c): a spend tally is exactly
# the sum of its parts, in user units. The expected tally: features £20,000 · advertising £20,000 ·
# carry-on £0.
# ---------------------------------------------------------------------------------------------------------


def tally_request(tally_level: Optional[float]) -> Dict[str, Any]:
    spend = {"frame": 100_000.0, "carrier": "cap"}
    tally_state = (
        None
        if tally_level is None
        else {"value": tally_level, "cap": 100_000, "source": "brief_extraction"}
    )
    return {
        "request_id": "r3-2-sum",
        "graph": {
            "nodes": [
                {
                    "id": "features_spend",
                    "kind": "factor",
                    "label": "Features spend",
                    "observed_state": {"value": 0.0, "cap": 100_000, "source": "brief_extraction"},
                    "execution_frame": spend,
                },
                {
                    "id": "advertising_spend",
                    "kind": "factor",
                    "label": "Advertising spend",
                    "observed_state": {"value": 0.0, "cap": 100_000, "source": "brief_extraction"},
                    "execution_frame": spend,
                },
                {
                    "id": "total_spend",
                    "kind": "outcome",
                    "label": "Six-month spend",
                    "observed_state": tally_state,
                    "execution_frame": spend,
                    "nonlinear_identity": {
                        "operation": "sum",
                        "factor_ids": ["features_spend", "advertising_spend"],
                        "stated_in_brief": True,
                    },
                },
            ],
            "edges": [
                # Belief strengths a 0.5 slope would use (R3-2c's mutant); the identity ignores them.
                {
                    "from": "features_spend",
                    "to": "total_spend",
                    "strength": {"mean": 0.5, "std": 0.1},
                },
                {
                    "from": "advertising_spend",
                    "to": "total_spend",
                    "strength": {"mean": 0.5, "std": 0.1},
                },
            ],
        },
        "options": [
            {"id": "features", "label": "Build features", "interventions": {"features_spend": 0.2}},
            {
                "id": "advertising",
                "label": "Advertise",
                "interventions": {"advertising_spend": 0.2},
            },
            {"id": "carry_on", "label": "Carry on", "interventions": {}},
        ],
        "goal_node_id": "total_spend",
        "n_samples": 200,
        "seed": 7,
    }


def tally_effects(d: Dict[str, Any]) -> Dict[str, float]:
    evaluator = SCMEvaluatorV2(graph_of(d))
    edges = means(d)
    factors = {"features_spend": 0.0, "advertising_spend": 0.0}
    today = evaluator.evaluate(edges, {}, "total_spend", factor_values=factors)
    return {
        o["id"]: (
            evaluator.evaluate(edges, o["interventions"], "total_spend", factor_values=factors)
            - today
        )
        * 100_000.0
        for o in d["options"]
    }


class TestSum:
    def test_a_tally_with_no_stated_level_is_the_sum_of_its_parts(self):
        effects = tally_effects(tally_request(None))
        assert effects == pytest.approx(
            {"features": 20_000.0, "advertising": 20_000.0, "carry_on": 0.0}
        )

    def test_control_the_linear_model_reads_the_half_slope(self):
        """R3-2c's mutant shape: the same graph undeclared gives the 0.5-slope £10,000, not £20,000."""
        d = tally_request(None)
        d["graph"]["nodes"][2].pop("nonlinear_identity")
        assert tally_effects(d)["features"] == pytest.approx(10_000.0)

    def test_r3_2_a_stated_zero_tally_evaluates(self):
        """AIQ ISL #187 5860770241 (1): a sum has no ratio, so a stated £0 is an ordinary level; both
        sides 0 reconcile exactly. R3-2: features £20,000 · advertising £20,000 · carry-on £0. The
        product's relative check applied to a sum (mutant) withholds it -> RED."""
        d = tally_request(0.0)
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        assert plan.evaluated and plan.mismatch_share == 0.0
        assert tally_effects(d) == pytest.approx(
            {"features": 20_000.0, "advertising": 20_000.0, "carry_on": 0.0}
        )

    def test_a_stated_zero_tally_whose_parts_are_not_zero_is_withheld(self):
        """o = £0 against parts summing to £5,000: the scaled absolute check fails (share 1.0)."""
        d = tally_request(0.0)
        d["graph"]["nodes"][0]["observed_state"]["value"] = 0.05  # £5,000 of features spend today
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        assert plan.withheld_reason == rav2.IDENTITY_INCONSISTENT
        assert plan.mismatch_share == pytest.approx(1.0)

    def test_a_product_keeps_the_zero_level_withhold(self):
        d = wire()
        (mrr,) = [n for n in d["graph"]["nodes"] if n["id"] == MRR]
        mrr["observed_state"].update(value=0.0, baseline=0.0, raw_value=0)
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        assert plan.withheld_reason == rav2.IDENTITY_ZERO_LEVEL

    def test_a_stated_nonzero_tally_moves_by_the_unscaled_sum(self):
        d = tally_request(0.1)  # £10,000 stated
        for node in d["graph"]["nodes"][:2]:
            node["observed_state"]["value"] = 0.05  # £5,000 each: reconciles exactly
        evaluator = SCMEvaluatorV2(graph_of(d))
        factors = {"features_spend": 0.05, "advertising_spend": 0.05}
        today = evaluator.evaluate(means(d), {}, "total_spend", factor_values=factors)
        more = evaluator.evaluate(
            means(d), {"features_spend": 0.25}, "total_spend", factor_values=factors
        )
        assert today * 100_000.0 == pytest.approx(10_000.0)
        assert (more - today) * 100_000.0 == pytest.approx(20_000.0)


# ---------------------------------------------------------------------------------------------------------
# R3-9 (AIQ ISL #187 5860770241 (3), a SERVING condition): an evaluated identity's operand edges are
# definitions, so no edge-level output lists them.
# ---------------------------------------------------------------------------------------------------------

DEFINITIONAL = {(PRICE, MRR), (SUBS, MRR)}


def edge_pairs_listed(v1: Any, v2: Dict[str, Any]) -> set:
    listed = {(row.edge_from, row.edge_to) for row in v1.sensitivity}
    listed |= {(row["from_id"], row["to_id"]) for row in (v1.edge_e_values or [])}
    for edge_id in [*v1.robustness.fragile_edges, *v1.robustness.robust_edges]:
        source, target = edge_id.split("->")
        listed.add((source, target))
    robustness = v2.get("robustness") or {}
    for key in ("edge_sensitivity", "edge_e_values", "fragile_edges", "robust_edges"):
        for row in robustness.get(key) or []:
            if isinstance(row, str):
                source, target = row.split("->")
                listed.add((source, target))
            else:
                listed.add(
                    (
                        row.get("from_id") or row.get("edge_from") or row.get("from"),
                        row.get("to_id") or row.get("edge_to") or row.get("to"),
                    )
                )
    return listed


class TestR3_9DefinitionalEdgesAreNotBeliefs:
    def _pair(self, identity):
        d = wire(identity=identity)
        v1 = rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))
        return v1, v2_body(d)

    def test_no_edge_level_output_lists_an_operand_edge(self):
        v1, v2 = self._pair(PRODUCT)
        assert not (edge_pairs_listed(v1, v2) & DEFINITIONAL)

    def test_control_without_the_identity_they_are_listed(self):
        """Non-vacuity: the same outputs DO list them when the link is a belief."""
        v1, v2 = self._pair(None)
        assert edge_pairs_listed(v1, v2) >= DEFINITIONAL

    def test_every_other_edge_is_still_listed(self):
        v1, v2 = self._pair(PRODUCT)
        listed = edge_pairs_listed(v1, v2)
        assert (OTHER, MRR) in listed  # undeclared addend: still a sampled belief edge
        assert (PRICE, "monthly_new_pro_subscribers") in listed


# ---------------------------------------------------------------------------------------------------------
# DL CHANGES_REQUIRED on ISL #187 @45a36d60 (BLOCKING): flip thresholds read a FRESH status quo.
# ``_flip_mean_under_background.winner_at`` bisects one edge of a SHARED background dict IN PLACE and calls
# ``evaluate`` on it. A status-quo cache keyed on the dict OBJECT returned the reading of a different edge
# strength on every step after the first, so any option with a today's-level (``unchanged``) or
# ``set_levels`` setting moved its threshold with no identity declared (measured at 45a36d60 on this wire:
# pro_plan_price -> monthly_new_pro_subscribers, background 1, -0.3491... against base 0.3750...).
# ---------------------------------------------------------------------------------------------------------

# Base d1cef9a (staging), measured on the no-identity served wire below: every (edge, background) whose
# background admits a flip. Every other pair is None (no flip). Backgrounds: 0 = the expected-value
# background (mean x exists_probability), 1-3 = ``_sample_flip_backgrounds(request, seed, 3, ...)``.
BASE_D1CEF9A_FLIP_MEANS = {
    (PRICE, MRR, 0): 0.007482051849365234,
    (PRICE, MRR, 1): 0.006546497344970703,
    (PRICE, MRR, 2): 0.02470541000366211,
    (PRICE, MRR, 3): 0.002655506134033203,
    (PRICE, "monthly_new_pro_subscribers", 1): 0.37500038146972653,
    ("monthly_churn", SUBS, 1): -0.6806744575500487,
    ("monthly_new_pro_subscribers", SUBS, 1): 0.07612380981445313,
    ("monthly_new_pro_subscribers", SUBS, 2): 0.6398595809936525,
    (SUBS, MRR, 1): -2.3841857911061326e-07,
}


def _no_cache_status_quo(self, edge_strengths, base_values, factor_values):
    return self._propagate(edge_strengths, {}, base_values, factor_values, noise=False)


def _object_keyed_status_quo(self, edge_strengths, base_values, factor_values):
    """45a36d60's cache, reinstated as an in-test CONTROL: keyed on the objects passed."""
    cached = getattr(self, "_object_cache", None)
    if (
        cached is not None
        and cached[0] is edge_strengths
        and cached[1] is base_values
        and cached[2] is factor_values
    ):
        return cached[3]
    status_quo = self._propagate(edge_strengths, {}, base_values, factor_values, noise=False)
    self._object_cache = (edge_strengths, base_values, factor_values, status_quo)
    return status_quo


def flip_means(d: Dict[str, Any]) -> Dict[Tuple[str, str, int], Optional[float]]:
    """``_flip_mean_under_background`` for every edge on four backgrounds, as the sweep calls it."""
    request = RobustnessRequestV2.model_validate(d)
    analyzer = rav2.RobustnessAnalyzerV2()
    evaluator = SCMEvaluatorV2(request.graph)  # no epsilon: the post-MC structural state
    expected = {
        (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
    }
    backgrounds = [expected] + analyzer._sample_flip_backgrounds(
        request, request.seed, 3, "flip_stability"
    )
    return {
        (edge.from_, edge.to, i): analyzer._flip_mean_under_background(
            request, evaluator, edge, background
        )
        for edge in request.graph.edges
        for i, background in enumerate(backgrounds)
    }


class TestFlipThresholdsReadAFreshStatusQuo:
    def test_the_wire_exercises_both_framings(self):
        """Non-vacuity: the reference option holds price at TODAY's level (B1a-5 ``unchanged``) and
        two set a non-root level (``set_levels``), whose status-quo sample moves with the edges."""
        d = wire(identity=None)
        evaluator = SCMEvaluatorV2(graph_of(d))
        (keep,) = [o for o in d["options"] if o["id"] == "keep_current_49_price"]
        assert rav2.is_todays_level(keep["interventions"][PRICE], evaluator._todays_levels[PRICE])
        set_level_nodes = {
            node_id
            for o in d["options"]
            for node_id in o["interventions"]
            if node_id in evaluator._status_quo_levels
        }
        assert set_level_nodes == {"monthly_new_pro_subscribers", "monthly_churn"}

    def test_no_identity_thresholds_are_base_d1cef9a(self, monkeypatch, former_truncated_edge_sampler):
        """(a) The no-identity path is base: identical to d1cef9a's measured thresholds, and to a run in
        which no status quo is ever cached. d1cef9a sampled strengths under the former +/-1 bound, so
        this row does too (AIQ #72 5868664986)."""
        d = wire(identity=None)
        served = flip_means(d)
        expected = {key: BASE_D1CEF9A_FLIP_MEANS.get(key) for key in served}
        assert len(served) == 40 and sum(v is not None for v in expected.values()) == 9
        assert served == expected
        monkeypatch.setattr(SCMEvaluatorV2, "_status_quo", _no_cache_status_quo)
        assert flip_means(d) == served

    def test_evaluated_identity_thresholds_equal_a_no_cache_computation(self, monkeypatch):
        """(b) With the identity EVALUATED, the cached anchor gives exactly the thresholds a run that
        recomputes the status quo on every call gives."""
        d = wire()
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        assert plan.evaluated
        served = flip_means(d)
        monkeypatch.setattr(SCMEvaluatorV2, "_status_quo", _no_cache_status_quo)
        assert flip_means(d) == served

    def test_control_an_object_keyed_cache_moves_them(self, monkeypatch):
        """Discriminating control for (b): 45a36d60's object-keyed cache on the same graph gives
        DIFFERENT thresholds, so (b) can see the defect."""
        d = wire()
        monkeypatch.setattr(SCMEvaluatorV2, "_status_quo", _no_cache_status_quo)
        fresh = flip_means(d)
        monkeypatch.setattr(SCMEvaluatorV2, "_status_quo", _object_keyed_status_quo)
        assert flip_means(d) != fresh


# ---------------------------------------------------------------------------------------------------------
# The typed ``identity`` on the IDENTITY_NOT_EVALUATED critique (R&C 5860893532; DL ISL #187). PLoT carries
# this shape and CEE states it as a typed ask, so nobody parses the message.
# ---------------------------------------------------------------------------------------------------------


def blocked_422(d: Dict[str, Any]) -> Dict[str, Any]:
    from fastapi.testclient import TestClient

    from src.api.main import app

    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2", json=d, headers={"X-ISL-Response-Version": "2"}
    )
    assert response.status_code == 422, response.text
    return response.json()


def identity_critique(body: Dict[str, Any]) -> Dict[str, Any]:
    (critique,) = [c for c in body["critiques"] if c["code"] == "IDENTITY_NOT_EVALUATED"]
    return critique


class TestTheTypedIdentityOnTheCritique:
    def test_identity_inconsistent_carries_both_figures(self):
        """49 x 1,000 + 1,000 = £50,000 against a stated £75,000: a third apart."""
        identity = identity_critique(blocked_422(inconsistent_wire()))["identity"]
        assert set(identity) == {
            "node_id",
            "operation",
            "participants",
            "withheld_reason",
            "reconstructed",
            "stated",
            "mismatch_share",
        }
        assert identity["node_id"] == MRR
        assert identity["operation"] == "product"
        assert identity["participants"] == [PRICE, SUBS, OTHER]
        assert identity["withheld_reason"] == "identity_inconsistent"
        assert identity["reconstructed"] == pytest.approx(50_000.0)
        assert identity["stated"] == pytest.approx(75_000.0)
        assert identity["mismatch_share"] == pytest.approx(25_000.0 / 75_000.0)

    @pytest.mark.parametrize(
        "reason, edit",
        [
            ("identity_operand_missing", lambda subs, price: subs.update(observed_state=None)),
            ("identity_zero_level", lambda subs, price: price["observed_state"].update(value=0.0)),
        ],
    )
    def test_a_reason_with_no_reconciliation_carries_no_figures(self, reason, edit):
        d = wire()
        nodes = {n["id"]: n for n in d["graph"]["nodes"]}
        edit(nodes[SUBS], nodes[PRICE])
        identity = identity_critique(blocked_422(d))["identity"]
        assert identity["withheld_reason"] == reason
        assert identity["node_id"] == MRR and identity["participants"] == [PRICE, SUBS, OTHER]
        assert identity["reconstructed"] is None
        assert identity["stated"] is None
        assert identity["mismatch_share"] is None

    def test_every_other_critique_has_no_identity_key(self):
        """On the 422 (dumped WITHOUT exclude_none) beside another blocker, and on a computed 200."""
        d = inconsistent_wire()
        d["options"] = d["options"] + [
            {"id": "keep_copy", "label": "Keep (copy)", "interventions": {PRICE: 0.245}}
        ]
        body = blocked_422(d)
        others = [c for c in body["critiques"] if c["code"] != "IDENTITY_NOT_EVALUATED"]
        assert "IDENTICAL_OPTIONS" in {c["code"] for c in others}
        assert all("identity" not in c for c in others)
        assert "identity" in identity_critique(body)
        computed = v2_body(wire())["critiques"]
        assert computed and all("identity" not in c for c in computed)

    def test_a_non_finite_figure_is_none_never_nan(self):
        from src.models.response_v2 import CritiqueIdentityV2

        identity = CritiqueIdentityV2(
            node_id=MRR,
            operation="product",
            participants=[PRICE, SUBS],
            withheld_reason="identity_inconsistent",
            reconstructed=float("inf"),
            stated=75_000.0,
            mismatch_share=float("nan"),
        )
        assert identity.reconstructed is None and identity.mismatch_share is None
        assert identity.stated == 75_000.0
        json.dumps(identity.model_dump(), allow_nan=False)  # strict JSON: raises on NaN/Infinity

    def test_a_numpy_float32_figure_is_screened_too(self):
        """MG ISL #187 5861838085 (5): np.float32 is not a Python float, so a float32 NaN passed the
        isinstance check and reached the model as NaN."""
        import numpy as np

        from src.models.response_v2 import CritiqueIdentityV2

        identity = CritiqueIdentityV2(
            node_id=MRR,
            operation="product",
            participants=[PRICE, SUBS],
            withheld_reason="identity_inconsistent",
            reconstructed=np.float32("nan"),
            stated=np.float32(1.5),
            mismatch_share=np.float64("inf"),
        )
        assert identity.reconstructed is None
        assert identity.stated == 1.5 and isinstance(identity.stated, float)
        assert identity.mismatch_share is None
        json.dumps(identity.model_dump(), allow_nan=False)

    def test_the_message_names_an_addend_as_an_addend(self):
        """MG ISL #187 5861838085 (6): MRR = price x subscribers + other MRR growth. The message listed
        the addend inside the product; the typed ``participants`` (CEE reads it) is unchanged."""
        critique = identity_critique(blocked_422(inconsistent_wire()))
        assert critique["message"].startswith(
            f"{MRR} is declared as the product of {PRICE}, {SUBS} plus {OTHER} but cannot be "
            "computed exactly (identity_inconsistent"
        )
        assert f"{SUBS}, {OTHER}" not in critique["message"]
        assert critique["identity"]["participants"] == [PRICE, SUBS, OTHER]

    def test_without_addends_the_message_is_unchanged(self):
        d = wire(identity=PRODUCT)
        (subs,) = [n for n in d["graph"]["nodes"] if n["id"] == SUBS]
        subs["observed_state"].update(value=0.1, raw_value=1_000)
        critique = identity_critique(blocked_422(d))
        assert critique["message"] == (
            f"{MRR} is declared as the product of {PRICE}, {SUBS} but cannot be computed exactly "
            "(identity_inconsistent: its parts give 49,000.00 where the stated level is 75,000.00, "
            "34.7% apart); the analysis is withheld rather than approximated"
        )
        assert critique["identity"]["participants"] == [PRICE, SUBS]


# ---------------------------------------------------------------------------------------------------------
# DL ISL #187 (not blocking): a NaN draw is DROPPED by the aggregator: the wire's mean is the finite draws'
# mean, n_valid_samples counts only them, and no option wins that draw. It is never averaged in. (A zero
# status-quo product term no longer produces one: AIQ #72 5866317942, below. A non-finite factor draw does.)
# ---------------------------------------------------------------------------------------------------------


class TestANaNIdentityDrawIsDroppedNotAveraged:
    N_SAMPLES, EVERY = 200, 10

    def _price_every_tenth_draw(self, monkeypatch, value: float):
        """Today's price reads ``value`` on every tenth factor draw (the same draw for every option)."""
        original = rav2.FactorSampler.sample_factor_values
        calls = {"n": 0}

        def sample(sampler):
            values = original(sampler)
            calls["n"] += 1
            if calls["n"] % self.EVERY == 0:
                values[PRICE] = value
            return values

        monkeypatch.setattr(rav2.FactorSampler, "sample_factor_values", sample)
        return calls

    def test_a_non_finite_draw_is_dropped_not_averaged(self, monkeypatch):
        import numpy as np

        d = wire()
        d["n_samples"] = self.N_SAMPLES
        calls = self._price_every_tenth_draw(monkeypatch, float("nan"))
        v1 = rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))
        calls["n"] = 0
        v2 = v2_body(d)

        n_nan = self.N_SAMPLES // self.EVERY
        finite_means = {}
        nan_masks = []
        for result in v1.results:
            samples = np.array(result.outcome_distribution.samples)
            nan_masks.append(~np.isfinite(samples))
            finite_means[result.option_id] = float(np.mean(samples[np.isfinite(samples)]))
        assert all(int(mask.sum()) == n_nan for mask in nan_masks)
        assert all(np.array_equal(mask, nan_masks[0]) for mask in nan_masks)  # whole draws

        for option in v2["options"]:
            outcome = option["outcome"]
            assert outcome["n_samples"] == self.N_SAMPLES
            assert outcome["n_valid_samples"] == self.N_SAMPLES - n_nan
            assert math.isfinite(outcome["mean"])
            assert outcome["mean"] == pytest.approx(finite_means[option["id"]], rel=1e-12)
        # A NaN draw is won by no option: the shares sum to the informative fraction.
        assert sum(o["win_probability"] for o in v2["options"]) == pytest.approx(
            (self.N_SAMPLES - n_nan) / self.N_SAMPLES
        )

    def test_a_zero_status_quo_term_draw_is_finite(self, monkeypatch):
        """AIQ #72 5866317942: no draw divides by its own status-quo term, so a draw on which today's
        price reads 0 is an ordinary (finite) draw, not an uninformative one. It was NaN under the
        ratio form."""
        import numpy as np

        d = wire()
        d["n_samples"] = self.N_SAMPLES
        self._price_every_tenth_draw(monkeypatch, 0.0)
        v1 = rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))
        for result in v1.results:
            assert np.isfinite(np.array(result.outcome_distribution.samples)).all()

    # MG ISL #187 5861838085 (4): the legacy V1 body carries those NaN draws raw (outcome_distribution
    # mean/std/median/CI and samples); starlette renders with allow_nan=False, so the run was a 500.

    @staticmethod
    def _v1(d: Dict[str, Any]) -> Any:
        from fastapi.testclient import TestClient

        from src.api.main import app

        return TestClient(app, raise_server_exceptions=False).post(
            "/api/v1/robustness/analyze/v2", json=d, headers={"X-ISL-Response-Version": "1"}
        )

    def test_the_v1_body_renders_a_nan_draw_as_null_not_a_500(self, monkeypatch):
        def refuse(token: str) -> Any:
            raise ValueError(f"not strict JSON: {token}")

        d = wire()
        d["n_samples"] = self.N_SAMPLES
        self._price_every_tenth_draw(monkeypatch, float("nan"))
        response = self._v1(d)
        assert response.status_code == 200, response.text[:300]
        body = json.loads(response.text, parse_constant=refuse)

        n_nan = self.N_SAMPLES // self.EVERY
        for result in body["results"]:
            samples = result["outcome_distribution"]["samples"]
            assert sum(s is None for s in samples) == n_nan
            assert all(math.isfinite(s) for s in samples if s is not None)
            assert result["outcome_distribution"]["mean"] is None  # NaN in, null out

    def test_without_nan_draws_the_v1_body_is_byte_identical(self, monkeypatch):
        """The same request with no NaN draw renders exactly what FastAPI rendered from the returned
        model before the fix (response_model=None: jsonable_encoder, then JSONResponse)."""
        from fastapi.encoders import jsonable_encoder
        from fastapi.responses import JSONResponse

        import src.api.robustness as route

        returned: Dict[str, Any] = {}
        run_offloaded = route.run_offloaded

        async def capture(*args: Any, **kwargs: Any) -> Any:
            returned["response"] = await run_offloaded(*args, **kwargs)
            return returned["response"]

        monkeypatch.setattr(route, "run_offloaded", capture)
        d = wire()
        d["n_samples"] = self.N_SAMPLES
        response = self._v1(d)
        assert response.status_code == 200, response.text[:300]
        assert response.content == JSONResponse(content=jsonable_encoder(returned["response"])).body


# ---------------------------------------------------------------------------------------------------------
# AIQ #72 5866317942 (MG ISL #187 5861838085 follow-up 2): the product is scaled by ONE central constant
# k = (o - A - L) / term, never by a sampled status-quo draw's own term. The ratio form divided by that
# draw's term; a draw near 0 contradicts the stated o and the division amplified it (MG: std 11.3 at price
# std 0.2; measured here at base 14f1a3a, 10,000 draws: std 230.9, mean -2.44).
# ---------------------------------------------------------------------------------------------------------


def with_price_std(d: Dict[str, Any], std: float) -> Dict[str, Any]:
    (price,) = [u for u in d["parameter_uncertainties"] if u["node_id"] == PRICE]
    price["std"] = std
    return d


def option_samples(d: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    response = rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))
    return {r.option_id: np.array(r.outcome_distribution.samples) for r in response.results}


class TestTheCentralScale:
    SERVED_59_GBP, SERVED_59_SE = 14_381.96, 13.31  # base 14f1a3a, served wire (R3-1-MC)

    @pytest.fixture(scope="class")
    def served(self):
        return option_samples(wire(identity=PRODUCT))

    @pytest.fixture(scope="class")
    def wide(self):
        return option_samples(with_price_std(wire(identity=PRODUCT), 0.2))

    def test_k_is_the_reconciliation_factor(self):
        """With the addend declared, k = (75,000 - 1,000) / 73,500 (Paul's figures), one number."""
        evaluator = SCMEvaluatorV2(graph_of(wire()))
        assert evaluator._identity_scales == {MRR: pytest.approx(74_000.0 / 73_500.0, rel=1e-12)}

    def test_a_sum_and_an_ungraphed_identity_have_no_scale(self):
        assert SCMEvaluatorV2(graph_of(tally_request(0.0)))._identity_scales == {}
        assert SCMEvaluatorV2(graph_of(wire(identity=None)))._identity_scales == {}

    def test_a_wide_price_keeps_the_tails_finite_and_the_mean(self, served, wide):
        """MG's table, at price std 0.2 (£40 on £49): the £59 option's spread is the difference form's
        (subscribers x the price spread, about 0.48 of the frame), NOT 11.3; its mean does not move.
        Mutant: the sampled denominator -> std 230.9 -> RED."""
        import numpy as np

        at_59 = wide["increase_price_to_59"]
        assert np.isfinite(at_59).all()
        difference_form = (74_000.0 / 73_500.0) * 1_500.0 * (0.2 * 200.0) / FRAMES[MRR]["frame"]
        assert 0.5 * difference_form < float(at_59.std()) < 1.5 * difference_form
        assert float(at_59.mean()) == pytest.approx(float(served["increase_price_to_59"].mean()), abs=0.005)

    def test_a_sampled_non_root_parent_is_read_at_its_centre_in_k(self):
        """DL ISL #193 CHANGES_REQUIRED: churn is non-root, uncertain and not an operand. With a
        churn -> MRR edge, k was computed with churn at base 0 while every draw reads it at its centre
        (3%): the £59 effect fell to +£14,389.35 (16 SE). Base 14f1a3a gives +£14,673.50 (SE £17.31).
        Mutant: k computed with no factor centres -> RED."""
        d = wire(identity=PRODUCT)
        d["graph"]["edges"].append(
            {
                "from": "monthly_churn",
                "to": MRR,
                "strength": {"mean": -0.5, "std": 0.1},
                "exists_probability": 0.8,
            }
        )
        samples = option_samples(d)
        diff = (samples["increase_price_to_59"] - samples["keep_current_49_price"]) * FRAMES[MRR]["frame"]
        assert abs(float(diff.mean()) - 14_673.50) < 3 * 17.31

    def test_k_reads_the_sampler_centres(self):
        request = RobustnessRequestV2.model_validate(wire(identity=PRODUCT))
        centres = rav2.factor_centres(request)
        assert centres[SUBS] == 0.15 and centres[PRICE] == 0.245
        assert "monthly_churn" in centres

    def test_the_served_59_effect_stays_within_3_se(self, served):
        """Paul's served wire (price std 1e-4): the £59 effect is +£14,381.96 at base; one central k keeps
        it within 3 SE. Mutant: k with each belief edge at its bare mean (not x exists_probability) moves
        it by £50, 3.8 SE -> RED."""
        diff = (served["increase_price_to_59"] - served["keep_current_49_price"]) * FRAMES[MRR]["frame"]
        assert abs(float(diff.mean()) - self.SERVED_59_GBP) < 3 * self.SERVED_59_SE


# ---------------------------------------------------------------------------------------------------------
# R3-5 RED 1 (AIQ #72 5867263914; rows 5866603688): structural influence walks an evaluated identity's
# operand/addend edges at the identity's own partial derivative at the centre, never at the guessed slope
# the evaluator ignores. Measured on the served wire at base: the scores were IDENTICAL with and without the
# identity (price 1.000 · other growth 0.794 · subscribers 0.298).
# ---------------------------------------------------------------------------------------------------------


def influence(d: Dict[str, Any]) -> Dict[str, float]:
    request = RobustnessRequestV2.model_validate(d)
    factors = [u.node_id for u in request.parameter_uncertainties or []]
    scores, truncated = rav2.RobustnessAnalyzerV2()._compute_structural_influence(
        request.graph, factors, request.goal_node_id, factor_centres=rav2.factor_centres(request)
    )
    assert truncated == []
    return scores


def central_slope(d: Dict[str, Any], operand: str, h: float = 1e-4) -> float:
    """d(MRR)/d(operand) through the evaluator itself, at today's levels, in normalised frames: the
    operand set h either side of today (price held at today, so its path into subscribers is cut)."""
    evaluator = SCMEvaluatorV2(graph_of(d))
    edges, factors = means(d), roots_at_today(d)
    today = {PRICE: 0.245, SUBS: 0.15, OTHER: 0.02}
    held = {PRICE: today[PRICE], SUBS: today[SUBS]}

    def at(value: float) -> float:
        return evaluator.evaluate(edges, {**held, operand: value}, MRR, factor_values=factors)

    return (at(today[operand] + h) - at(today[operand] - h)) / (2 * h)


class TestInfluenceReadsTheIdentity:
    def test_subscribers_rank_above_price_on_the_served_wire(self):
        """The product's partials at today's levels: subscribers k x £49 x 10,000/125,000 = 3.96 against
        price k x 1,500 x 200/125,000 = 2.42 (plus price's path through new subscribers)."""
        scores = influence(wire(identity=PRODUCT))
        assert max(scores, key=scores.get) == SUBS
        assert scores[SUBS] == 1.0
        assert scores[PRICE] == pytest.approx(0.638, abs=5e-4)
        assert scores[OTHER] == pytest.approx(0.081, abs=5e-4)  # still a belief edge: 0.4 x 0.8 / 3.96

    def test_control_no_declaration_walks_the_slopes_exactly_as_at_base(self):
        """Base 14f1a3a, the same wire with no declaration: price 1.000 · other 0.794 · subscribers 0.298."""
        assert rav2.identity_partials(graph_of(wire(identity=None))) == {}
        scores = influence(wire(identity=None))
        assert scores[PRICE] == 1.0
        assert scores[OTHER] == pytest.approx(0.794, abs=5e-4)
        assert scores[SUBS] == pytest.approx(0.298, abs=5e-4)

    @pytest.mark.parametrize("identity", [PRODUCT, WITH_ADDEND], ids=["no_addend", "addend"])
    @pytest.mark.parametrize("operand", [PRICE, SUBS])
    def test_each_partial_is_the_evaluators_own_slope(self, identity, operand):
        d = wire(identity=identity)
        partial = rav2.identity_partials(graph_of(d))[(operand, MRR)]
        assert partial == pytest.approx(central_slope(d, operand), rel=1e-6)

    def test_a_declared_addend_moves_one_for_one_in_user_units(self):
        partials = rav2.identity_partials(graph_of(wire(identity=WITH_ADDEND)))
        assert partials[(OTHER, MRR)] == pytest.approx(50_000.0 / 125_000.0, rel=1e-12)
        k = 74_000.0 / 73_500.0
        assert partials[(PRICE, MRR)] == pytest.approx(k * 1_500.0 * 200.0 / 125_000.0, rel=1e-12)
        assert partials[(SUBS, MRR)] == pytest.approx(k * 49.0 * 10_000.0 / 125_000.0, rel=1e-12)

    def test_a_sum_moves_one_for_one(self):
        partials = rav2.identity_partials(graph_of(tally_request(0.0)))
        assert partials == {("features_spend", "total_spend"): 1.0, ("advertising_spend", "total_spend"): 1.0}

    def test_a_withheld_identity_keeps_its_belief_edges(self):
        frames = {k: v for k, v in FRAMES.items() if k != PRICE}  # identity_frame_missing
        assert rav2.identity_partials(graph_of(wire(frames=frames))) == {}


# ---------------------------------------------------------------------------------------------------------
# R3-9 (engine; AIQ #72 5866734772, DL order 5867659507 item 3): an evaluated identity's DEFINITIONAL edges
# draw nothing from any RNG. They were still drawn (and ignored), so their existence/strength parameters
# shifted the draws of every edge after them: measured at base, perturbing price -> MRR and subscribers -> MRR
# moved results, edge_e_values, factor_evppi, factor_flip_values and edge_existence_rates.
# ---------------------------------------------------------------------------------------------------------


DEFINITIONS = ((PRICE, MRR), (SUBS, MRR))


def with_definitions_perturbed(d: Dict[str, Any]) -> Dict[str, Any]:
    d = copy.deepcopy(d)
    for edge in d["graph"]["edges"]:
        if (edge["from"], edge["to"]) in DEFINITIONS:
            edge["exists_probability"] = 0.3
            edge["strength"] = {"mean": 0.9, "std": 0.05}
    return d


def dumped(d: Dict[str, Any]) -> Dict[str, Any]:
    d = copy.deepcopy(d)
    d["n_samples"] = 500
    out = rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d)).model_dump()
    out["metadata"].pop("execution_time_ms")  # differs on ANY two runs
    return json.loads(json.dumps(out, sort_keys=True, default=str))


def fifty_nine_first(d: Dict[str, Any]) -> Dict[str, Any]:
    """Edge sensitivity reads the FIRST option. Served, that is keep-current, whose MRR is the stated
    £75,000 on every draw, so no edge's sensitivity can move; £59 first makes them readable."""
    d = copy.deepcopy(d)
    d["options"].sort(key=lambda option: option["id"] != "increase_price_to_59")
    return d


def definitions_of(d: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    """``definitional_strengths`` as every sampler calls it: at the request's own sampler centres."""
    request = RobustnessRequestV2.model_validate(d)
    return rav2.definitional_strengths(request.graph, rav2.factor_centres(request))


class TestDefinitionsDrawNothing:
    @pytest.mark.parametrize("order", [lambda d: d, fifty_nine_first], ids=["served", "59_first"])
    def test_a_definitions_ignored_parameters_move_nothing(self, order):
        d = order(wire(identity=PRODUCT))
        assert dumped(with_definitions_perturbed(d)) == dumped(d)

    def test_the_59_first_variant_reads_edge_sensitivity(self):
        """Non-vacuity for the row above: with £59 first, belief edges carry non-zero sensitivity."""
        rows = dumped(fifty_nine_first(wire(identity=PRODUCT)))["sensitivity"]
        assert any(abs(row["elasticity"]) > 0 for row in rows)

    def test_control_without_the_identity_they_are_beliefs_and_move_the_results(self):
        d = wire(identity=None)
        assert dumped(with_definitions_perturbed(d))["results"] != dumped(d)["results"]

    def test_definitions_consume_no_draws(self):
        """Belief edges draw exactly what they would if the definitional edges were not in the graph."""
        graph = graph_of(wire(identity=PRODUCT))
        fixed = definitions_of(wire(identity=PRODUCT))
        assert set(fixed) == set(DEFINITIONS)
        with_defs = rav2.DualUncertaintySampler(graph.edges, rav2.SeededRNG(7), fixed)
        without = rav2.DualUncertaintySampler(
            [e for e in graph.edges if (e.from_, e.to) not in fixed], rav2.SeededRNG(7)
        )
        for _ in range(50):
            drawn = with_defs.sample_edge_configuration()
            assert {k: v for k, v in drawn.items() if k not in fixed} == without.sample_edge_configuration()
            assert {k: drawn[k] for k in fixed} == fixed

    def test_a_definition_sits_at_its_central_strength_and_lists_no_existence_rate(self):
        graph = graph_of(wire(identity=PRODUCT))
        fixed = definitions_of(wire(identity=PRODUCT))
        assert fixed == {
            (PRICE, MRR): pytest.approx(0.5 * 0.8),
            (SUBS, MRR): pytest.approx(0.15 * 0.8),
        }
        sampler = rav2.DualUncertaintySampler(graph.edges, rav2.SeededRNG(7), fixed)
        sampler.sample_edge_configuration()
        rates = sampler.get_existence_rates()
        assert f"{PRICE}->{MRR}" not in rates and f"{SUBS}->{MRR}" not in rates
        assert f"{OTHER}->{MRR}" in rates  # an undeclared addend is still a belief

    def test_no_evaluated_identity_no_fixed_edges(self):
        assert definitions_of(wire(identity=None)) == {}
        frames = {k: v for k, v in FRAMES.items() if k != PRICE}  # identity_frame_missing: withheld
        assert definitions_of(wire(frames=frames)) == {}


# ---------------------------------------------------------------------------------------------------------
# AIQ #72 5868227452 (DL ISL #193 non-blocking note): the ONE central scale k must lie in [0.5, 2], else the
# product is withheld as ``identity_scale_out_of_range``. A belief parent of MRR carrying L at the centre gives
# k = (o - A - L) / term = (74,000 - L) / 73,500 with the addend declared; the reconciliation (which ignores L)
# still passes, so only this check can catch it.
# ---------------------------------------------------------------------------------------------------------


def with_belief_parent(k: float) -> Dict[str, Any]:
    """The addend wire plus a root belief parent ``promo`` (level 1.0, certain edge) whose central
    contribution L = 125,000 x mean puts the scale at ``k``."""
    d = wire(identity=WITH_ADDEND)
    d["graph"]["nodes"].append(
        {
            "id": "promo",
            "kind": "factor",
            "label": "Promotion",
            "observed_state": {"value": 1.0, "source": "brief_extraction"},
        }
    )
    mean = (74_000.0 - k * 73_500.0) / FRAMES[MRR]["frame"]
    d["graph"]["edges"].append(
        {"from": "promo", "to": MRR, "strength": {"mean": mean, "std": 0.01}, "exists_probability": 1.0}
    )
    return d


class TestTheScaleRange:
    def test_the_served_scale_is_evaluated(self):
        (plan,) = resolve_identity_plans(graph_of(wire())).values()
        assert plan.evaluated and plan.scale == pytest.approx(74_000.0 / 73_500.0, rel=1e-12)

    @pytest.mark.parametrize("k", [0.49, 2.01, 0.0, -0.1], ids=["0.49", "2.01", "L=o-A", "L>o-A"])
    def test_a_scale_outside_half_to_two_is_withheld(self, k):
        (plan,) = resolve_identity_plans(graph_of(with_belief_parent(k))).values()
        assert plan.withheld_reason == rav2.IDENTITY_SCALE_OUT_OF_RANGE
        assert plan.scale == pytest.approx(k, abs=1e-9)
        assert plan.mismatch_share == pytest.approx(500.0 / 75_000.0)  # it DID reconcile (0.67%)

    @pytest.mark.parametrize("k", [0.51, 1.99])
    def test_a_scale_inside_is_evaluated(self, k):
        (plan,) = resolve_identity_plans(graph_of(with_belief_parent(k))).values()
        assert plan.evaluated and plan.scale == pytest.approx(k, abs=1e-9)

    def test_on_the_decision_path_it_is_the_blocked_422_naming_the_scale(self):
        from fastapi.testclient import TestClient

        from src.api.main import app

        response = TestClient(app).post(
            "/api/v1/robustness/analyze/v2",
            json=with_belief_parent(2.01),
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422, response.text[:300]
        (critique,) = [
            c for c in response.json()["critiques"] if c["code"] == "IDENTITY_NOT_EVALUATED"
        ]
        assert critique["identity"]["withheld_reason"] == "identity_scale_out_of_range"
        assert "2.010, outside [0.5, 2]" in critique["message"]

    def test_the_disclosure_and_the_filter_read_the_same_decision(self):
        graph = graph_of(with_belief_parent(0.49))
        (entry,) = rav2.identity_evaluations(graph)
        assert not entry.evaluated and entry.withheld_reason == "identity_scale_out_of_range"
        assert rav2.definitional_edges(graph) == set()  # withheld: its edges stay beliefs

    @pytest.mark.parametrize(
        "k, churn_mean, side",
        [(0.52, 1.0, "below"), (1.97, -1.0, "above")],
        ids=["centre_k_below_0.5", "centre_k_above_2"],
    )
    def test_the_samplers_hold_nothing_the_evaluator_withheld(self, k, churn_mean, side):
        """R3-9 x the scale guard: ``definitional_strengths`` decides "evaluated" at the SAME sampler
        centres as the evaluator. churn (non-root, sampled, not a participant) carries L only at its centre
        (3%), so k crosses the band edge between base 0 and the centre: 0.519 -> 0.468, or 1.971 -> 2.022.
        The evaluator withholds the identity, so its edges are beliefs and must be drawn. Mutant: the
        samplers' fixed set read without the centres holds all three edges -> RED."""
        d = with_belief_parent(k)
        d["graph"]["edges"].append(
            {
                "from": "monthly_churn",
                "to": MRR,
                "strength": {"mean": churn_mean, "std": 0.1},
                "exists_probability": 1.0,
            }
        )
        request = RobustnessRequestV2.model_validate(d)
        centres = rav2.factor_centres(request)
        at_centre = resolve_identity_plans(request.graph, centres)[MRR]
        at_base = resolve_identity_plans(request.graph)[MRR]
        # Non-vacuity: the seam is live on this graph (the two readings disagree).
        assert at_centre.withheld_reason == rav2.IDENTITY_SCALE_OUT_OF_RANGE and at_base.evaluated
        assert (at_centre.scale < 0.5) if side == "below" else (at_centre.scale > 2.0)
        assert definitions_of(d) == {}


# ---------------------------------------------------------------------------------------------------------
# R3-9 x the scale guard, pinned at the SOURCE (Paul, #72 14:4xZ): "evaluated" now depends on k, and k on the
# sampler centres, so a reader that omits them decides a different identity set from the evaluator's. That
# seam bit three times (DL #193 CHANGES_REQUIRED, the bare-mean k, #197 x #199). Every call in src/ to a plan
# reader or the evaluator must pass the centres, and every sampler must pass the fixed set.
# ---------------------------------------------------------------------------------------------------------

import ast  # noqa: E402

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
CENTRE_READERS = {
    "resolve_identity_plans",
    "definitional_edges",
    "definitional_strengths",
    "identity_evaluations",
}


def _callee(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def unwired_calls(source: str, where: str) -> Tuple[Dict[str, int], list]:
    """(calls seen per callee, the calls that drop the centres or the fixed set)."""
    seen: Dict[str, int] = {}
    bad = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        keywords = {kw.arg for kw in node.keywords}
        if name in CENTRE_READERS:
            ok = len(node.args) >= 2 or "factor_centres" in keywords
        elif name == "SCMEvaluatorV2":
            ok = len(node.args) >= 3 or "factor_centres" in keywords
        elif name == "DualUncertaintySampler":
            ok = len(node.args) >= 3 or "fixed" in keywords
        else:
            continue
        seen[name] = seen.get(name, 0) + 1
        if not ok:
            bad.append(f"{where}:{node.lineno} {name}")
    return seen, bad


class TestTheCentresAreWiredAtEveryReader:
    def test_every_reader_passes_the_centres_and_every_sampler_the_fixed_set(self):
        seen: Dict[str, int] = {}
        bad: list = []
        for path in sorted(SRC_ROOT.rglob("*.py")):
            counts, missing = unwired_calls(path.read_text(), str(path.relative_to(SRC_ROOT)))
            for name, n in counts.items():
                seen[name] = seen.get(name, 0) + n
            bad.extend(missing)
        # Non-vacuity: the scan sees the call sites it guards (5 samplers, the evaluator, the readers).
        assert seen.get("DualUncertaintySampler", 0) >= 5, seen
        assert seen.get("SCMEvaluatorV2", 0) >= 4, seen
        assert seen.get("definitional_strengths", 0) >= 5, seen
        assert all(seen.get(name, 0) >= 1 for name in CENTRE_READERS), seen
        assert bad == [], bad

    @pytest.mark.parametrize(
        "mutant",
        [
            "plans = resolve_identity_plans(request.graph)",
            "fixed = definitional_edges(graph)",
            "evaluator = SCMEvaluatorV2(request.graph, epsilon_rng=rng)",
            "sampler = DualUncertaintySampler(request.graph.edges, rng)",
        ],
    )
    def test_the_scan_catches_a_dropped_argument(self, mutant):
        """Mutant rows: each shape the seam took, with the centres or the fixed set dropped -> caught."""
        _, bad = unwired_calls(mutant, "mutant")
        assert len(bad) == 1, bad
