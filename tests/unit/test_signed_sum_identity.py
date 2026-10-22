"""An exact profit definition remains revenue MINUS costs at every numeric reader."""
import copy

import pytest
from pydantic import ValidationError

from src.models.robustness_v2 import NonlinearIdentityV2, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    RobustnessAnalyzerV2, SCMEvaluatorV2, identity_partials, resolve_identity_plans,
)
from tests.unit.test_r3_identity_evaluation import v2_body


def profit_request(*, stated=True):
    total = {
        "id": "profit", "label": "Profit", "kind": "outcome",
        "execution_frame": {"frame": 100, "carrier": "cap"},
        "nonlinear_identity": {"operation": "sum", "factor_ids": ["revenue", "costs"],
                               "subtract": ["costs"], "stated_in_brief": True},
    }
    if stated:
        total["observed_state"] = {"value": .3, "baseline": .3, "raw_value": 30, "source": "brief_extraction"}
    return {
        "graph": {
            "nodes": [
                {"id": "revenue", "label": "Revenue", "kind": "factor",
                 "observed_state": {"value": .5, "raw_value": 100, "source": "brief_extraction"},
                 "execution_frame": {"frame": 200, "carrier": "cap"}},
                {"id": "costs", "label": "Costs", "kind": "factor",
                 "observed_state": {"value": .35, "raw_value": 70, "source": "brief_extraction"},
                 "execution_frame": {"frame": 200, "carrier": "cap"}}, total,
            ],
            "edges": [{"from": p, "to": "profit", "strength": {"mean": 1, "std": .1},
                       "exists_probability": 1} for p in ["revenue", "costs"]],
        },
        "options": [
            {"id": "grow", "label": "Revenue to 120", "interventions": {"revenue": .6}},
            {"id": "spend", "label": "Costs to 80", "interventions": {"costs": .4}},
        ],
        "goal_node_id": "profit", "goal_threshold": .4, "goal_threshold_frame": "level",
        "n_samples": 100, "seed": 4242,
    }


@pytest.mark.parametrize("stated", [True, False])
def test_profit_goal_chance_uses_subtraction_with_and_without_stated_current(stated):
    d = profit_request(stated=stated)
    before = copy.deepcopy(d)
    req = RobustnessRequestV2.model_validate(d)
    response = RobustnessAnalyzerV2().analyze(req)
    assert {r.option_id: r.probability_of_goal for r in response.results} == {"grow": 1, "spend": 0}
    plan = resolve_identity_plans(req.graph)["profit"]
    assert plan.evaluated and plan.subtract == ("costs",)
    assert d == before
    body = v2_body(d)
    assert body["identity_evaluations"][0]["subtract"] == ["costs"]


def test_partials_and_reconciliation_agree_on_the_sign_in_natural_units():
    req = RobustnessRequestV2.model_validate(profit_request())
    partials = identity_partials(req.graph)
    assert partials[("revenue", "profit")] == 2
    assert partials[("costs", "profit")] == -2
    plan = resolve_identity_plans(req.graph)["profit"]
    assert plan.reconstructed == pytest.approx(30)
    assert plan.mismatch_share == pytest.approx(0)


def test_an_actual_loss_is_negative_not_absolute_or_added():
    d = profit_request(stated=False)
    req = RobustnessRequestV2.model_validate(d)
    value = SCMEvaluatorV2(req.graph).evaluate(
        {("revenue", "profit"): 1, ("costs", "profit"): 1}, {"revenue": .1}, "profit")
    assert value * 100 == pytest.approx(-50)


@pytest.mark.parametrize("patch", [
    {"subtract": ["missing"]}, {"subtract": ["costs", "costs"]}, {"subtract": []},
    {"subtract": None}, {"operation": "product"},
])
def test_invalid_or_non_sum_subtraction_is_rejected_not_dropped(patch):
    carrier = profit_request()["graph"]["nodes"][-1]["nonlinear_identity"]
    with pytest.raises(ValidationError):
        NonlinearIdentityV2.model_validate({**carrier, **patch})


def test_legacy_sum_without_subtraction_keeps_its_wire_and_answer():
    d = profit_request(stated=False)
    del d["graph"]["nodes"][-1]["nonlinear_identity"]["subtract"]
    req = RobustnessRequestV2.model_validate(d)
    assert "subtract" not in req.graph.nodes[-1].nonlinear_identity.model_dump()
    value = SCMEvaluatorV2(req.graph).evaluate({("revenue", "profit"): 1, ("costs", "profit"): 1}, {}, "profit")
    assert value * 100 == pytest.approx(170)


def bakery_request(*, revenue=2850, costs=1500, profit=1350):
    d = profit_request()
    for node, raw, frame in zip(d["graph"]["nodes"], [revenue, costs, profit], [20000, 20000, 3000]):
        node["execution_frame"]["frame"] = frame
        node["observed_state"] = {"value": raw / frame, "raw_value": raw, "source": "brief_extraction"}
        if node["id"] == "profit":
            node["observed_state"]["baseline"] = raw / frame
    return d


@pytest.mark.parametrize("revenue,costs,stated", [(2850, 1500, 1567), (10000, 9500, -400)])
def test_reconciliation_rejects_net_contradiction_even_when_gross_gap_is_under_five_percent(revenue, costs, stated):
    from src.services.robustness_analyzer_v2 import identity_blocking_critiques

    d = bakery_request(revenue=revenue, costs=costs, profit=stated)
    req = RobustnessRequestV2.model_validate(d)
    plan = resolve_identity_plans(req.graph)["profit"]
    assert not plan.evaluated and plan.withheld_reason == "identity_inconsistent"
    assert plan.reconstructed == pytest.approx(revenue - costs)
    assert plan.mismatch_share == pytest.approx(abs(stated - (revenue - costs)) / max(abs(stated), abs(revenue - costs)))
    critique = identity_blocking_critiques(req)[0].model_dump()
    assert "revenue minus costs" in critique["message"]
    assert critique["identity"]["subtract"] == ["costs"]
    assert critique["identity"]["participants"] == ["revenue", "costs"]


def test_signed_reconciliation_keeps_exact_break_even():
    req = RobustnessRequestV2.model_validate(bakery_request(revenue=100, costs=100, profit=0))
    plan = resolve_identity_plans(req.graph)["profit"]
    assert plan.evaluated and plan.mismatch_share == 0


@pytest.mark.parametrize("threshold", [0, -100])
def test_positive_profit_today_does_not_floor_a_later_loss_into_goal_achievement(threshold):
    d = bakery_request()
    d["options"] = [
        {"id": "spend", "label": "Costs to 3000", "interventions": {"costs": 3000 / 20000}},
        {"id": "spend_more", "label": "Costs to 3100", "interventions": {"costs": 3100 / 20000}},
    ]
    d["goal_threshold"] = threshold / 3000
    response = RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))
    assert {r.option_id: r.probability_of_goal for r in response.results} == {"spend": 0, "spend_more": 0}


@pytest.mark.parametrize("cost_level", [.35, .375])
def test_stated_profit_is_reproduced_at_status_quo_even_on_a_changed_cost_draw(cost_level):
    """The reference term must subtract costs too; a plus-only term_sq shifts today's profit."""
    req = RobustnessRequestV2.model_validate(profit_request())
    evaluator = SCMEvaluatorV2(req.graph)
    value = evaluator.evaluate(
        {("revenue", "profit"): 1, ("costs", "profit"): 1}, {}, "profit",
        factor_values={"revenue": .5, "costs": cost_level},
    )
    assert value * 100 == pytest.approx(30)


def test_request_schema_matches_runtime_non_nullable_subtract_contract():
    prop = NonlinearIdentityV2.model_json_schema()["properties"]["subtract"]
    assert prop["type"] == "array" and prop["minItems"] == 1
    assert "anyOf" not in prop
