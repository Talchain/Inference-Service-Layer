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
        total["observed_state"] = {"value": .3, "raw_value": 30, "source": "brief_extraction"}
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
