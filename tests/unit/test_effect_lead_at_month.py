"""SPIKE S8: an option's effect that starts after a lead, scored at month M.

Hiring shape: "+10% productivity at month 3; a hire is fully in place L months from now".
Step assumption (AIE 6179731987, T7): no effect before L, the full step from L on. At
month M a step is in force iff its from_month <= M; a step not yet in force leaves the
node as it is without the option (the status quo), on the SAME draw.

Hand calculation (independent of the engine), from the recovery numerical check's
witness: S ~ Normal(0.5, 0.1); f is 0.5 today; setting f = x adds S * (x - 0.5) to
the goal. Relative-change target: a raw rise of 10 on a raw baseline of 70 over a
raw span of 100, so the normalised threshold T = 0.1.
  full step (x = 1.0):  mean 0.25,  sd 0.05  -> P = Phi((0.25 - 0.1)/0.05)   = Phi(3) = 0.99865
  half step (x = 0.75): mean 0.125, sd 0.025 -> P = Phi((0.125 - 0.1)/0.025) = Phi(1) = 0.84134
  not yet in force:     the paired change is exactly 0 < 0.1               -> P = 0 exactly
For contrast, the window-average k surrogate at H = 3, L = 2 would scale the effect by
1/3 (k - 1 = (r - 1) * (H - L) / H), giving Phi((0.0833 - 0.1)/0.0167) = Phi(-1) = 0.159:
the wrong quantity for a LEVEL read at month 3.
"""

import math

import pytest
from pydantic import ValidationError

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from tests.unit.test_recovery_numerical_check import TOL, linear_request


def phi(z):
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0


def hiring_request(at_month, steps_by_option, *, drop_steps_for=None):
    """The wire body CEE/PLoT would send: options with effect_steps, request at_month."""
    base = linear_request("change_rel")
    base.goal_threshold = 10.0 / 70.0  # a raw rise of 10 on the raw baseline 70
    body = base.model_dump(by_alias=True, exclude_none=True)
    body["options"] = [{"id": "hold", "label": "Status quo", "interventions": {}}]
    for option_id, steps in steps_by_option.items():
        option = {"id": option_id, "label": option_id, "interventions": {"f": 1.0}}
        if option_id != drop_steps_for:
            option["effect_steps"] = {
                "f": [{"from_month": m, "value": v} for m, v in steps]
            }
        body["options"].append(option)
    body["at_month"] = at_month
    return RobustnessRequestV2.model_validate(body)


HIRING = {
    "lead_in_time": [(2.0, 1.0)],  # start 1 + settling-in 1: fully in place at month 2
    "lead_too_late": [(4.0, 1.0)],  # fully in place at month 4, after the goal's month 3
    "two_hires": [(2.0, 0.75), (4.0, 1.0)],  # hire 1 in force at 2, hire 2 at 4 (S2 = S1 + delay)
}


def by_id(response):
    return {result.option_id: result for result in response.results}


def test_at_month_three_scores_only_the_steps_in_force():
    response = RobustnessAnalyzerV2().analyze(hiring_request(3.0, HIRING))
    results = by_id(response)
    assert results["lead_in_time"].probability_of_goal == pytest.approx(phi(3.0), abs=TOL)
    assert results["two_hires"].probability_of_goal == pytest.approx(phi(1.0), abs=TOL)
    # Not yet in force at month 3: exactly the status quo, never the full effect.
    assert results["lead_too_late"].probability_of_goal == 0.0
    assert results["hold"].probability_of_goal == 0.0
    echo = response.at_month_scoring
    assert echo is not None and echo.at_month == 3.0
    assert echo.step_shape == "no_effect_before_from_month_full_step_after"
    in_force = {o.option_id: (o.in_force, o.not_yet_in_force) for o in echo.options}
    assert in_force["lead_in_time"] == ({"f": 1.0}, [])
    assert in_force["two_hires"] == ({"f": 0.75}, ["f"])
    assert in_force["lead_too_late"] == ({}, ["f"])


def test_at_month_four_every_step_is_in_force():
    results = by_id(RobustnessAnalyzerV2().analyze(hiring_request(4.0, HIRING)))
    for option_id in ("lead_in_time", "lead_too_late", "two_hires"):
        assert results[option_id].probability_of_goal == pytest.approx(phi(3.0), abs=TOL)


def test_a_missing_lead_is_refused_never_read_as_zero():
    with pytest.raises(ValidationError, match="EFFECT_LEAD_MISSING"):
        hiring_request(3.0, HIRING, drop_steps_for="two_hires")


def test_last_step_must_be_the_options_full_setting():
    with pytest.raises(ValidationError, match="EFFECT_STEPS_FULL_SETTING"):
        hiring_request(3.0, {"bad": [(2.0, 0.75)]})



@pytest.mark.asyncio
async def test_wire_v2_envelope_carries_the_at_month_echo(monkeypatch):
    """PLoT reads /analyze/v2?response_version=2; the echo must reach that envelope.

    The route's own validator blocks an EMPTY or IDENTICAL option on the RAW request, so the
    wire body uses distinct full settings and a status quo that holds f at today's 0.5.
    """
    from httpx import ASGITransport, AsyncClient

    from src.api.main import app

    request = hiring_request(3.0, {"lead_in_time": [(2.0, 1.0)]})
    body = request.model_dump(mode="json", by_alias=True, exclude_none=True)
    body["options"][0] = {"id": "hold", "label": "Status quo", "interventions": {"f": 0.5},
                          "effect_steps": {"f": [{"from_month": 0.0, "value": 0.5}]}}
    body["options"].append({"id": "lead_too_late", "label": "late", "interventions": {"f": 0.9},
                            "effect_steps": {"f": [{"from_month": 4.0, "value": 0.9}]}})
    monkeypatch.setattr(app.state, "analysis_pool", None, raising=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/robustness/analyze/v2?response_version=2", json=body)
    assert response.status_code == 200, response.text
    wire = response.json()
    echo = wire.get("at_month_scoring")
    assert echo is not None, "V2 envelope lost the at-month echo"
    assert echo["at_month"] == 3.0
    assert {o["option_id"]: o["in_force"] for o in echo["options"]} == {
        "hold": {"f": 0.5}, "lead_in_time": {"f": 1.0}, "lead_too_late": {}}
