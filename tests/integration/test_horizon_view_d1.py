"""
Horizon view (SCIENCE/DSK EXPERIMENT, #85 5947449217) through ISL's real route on D1.

FIXTURE PROVENANCE (``tests/fixtures/horizon/d1-isl-request.json``): R3's served D1 capture (CEE
``rc-served-signal-cases.json`` case ``A-Q-D1-BUILD``) posted to PLoT ``/v2/run`` at PLoT staging ``4526e432``,
with the ISL client mocked to record the outgoing ``/api/v1/robustness/analyze/v2`` body. Options as CEE submits
them: the Olumi-proposed excluded option left out, and the baseline held at its factors' observed values
(``analysable-option-gate.ts``). Seed 42, 10,000 draws.

The rows run with the optional post-Monte-Carlo phases off (e-values, VOI, factor flips): the view is read from the
draws, which are computed before those phases, and the full-request view was checked equal by the experiment
harness. The ONSETS below are illustrative experiment inputs (T4 §3: the AI module ships after 6 weeks; the bug
fix ramps over 2 weeks). Nothing in D1's brief states them.
"""

import copy
import json
import pathlib

import pytest

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "horizon" / "d1-isl-request.json"
ROUTE = "/api/v1/robustness/analyze/v2"
AI, FIX, KEEP = "ai_reporting_module_sprint", "integration_bug_fix_sprint", "continue_current_plan"
T4_ONSETS = [
    {"option_id": AI, "onset_weeks": 6, "ramp_weeks": 0},
    {"option_id": FIX, "onset_weeks": 1, "ramp_weeks": 2},
]


def d1(horizon=None):
    q = json.loads(FIXTURE.read_text())
    q.update({"include_e_values": False, "include_voi": False, "include_factor_flips": False, "analysis_types": ["comparison"]})
    if horizon is not None:
        q["horizon"] = horizon
    return q


def without_clock(body):
    b = copy.deepcopy(body)
    b.get("_metadata", {}).pop("execution_time_ms", None)
    return b


async def post(client, q):
    r = await client.post(ROUTE, json=q)
    assert r.status_code == 200, r.text[:500]
    return r.json()


async def test_h2_absent_horizon_adds_no_key(client):
    body = await post(client, d1())
    assert "horizon_view" not in body


async def test_h1_no_onsets_every_week_equals_todays_win_probability_and_nothing_else_moves(client):
    plain = await post(client, d1())
    asked = await post(client, d1({"horizon_weeks": 13, "evaluate": "cumulative", "onsets": []}))
    win_p = {o["option_id"]: o["win_probability"] for o in plain["results"]}
    view = asked.pop("horizon_view")
    assert [c["p_best"] for c in view["checkpoints"]] == [win_p] * 13
    assert without_clock(asked) == without_clock(plain)


async def test_h3_d1_bug_fix_leads_until_the_ai_module_ships_then_the_ai_module_leads(client):
    body = await post(client, d1({"horizon_weeks": 13, "evaluate": "cumulative", "onsets": T4_ONSETS}))
    v = body["horizon_view"]
    assert v["flips"] == [
        {"week": 2, "from_option_id": None, "to_option_id": FIX},
        {"week": 7, "from_option_id": FIX, "to_option_id": AI},
    ]
    assert v["leader_at_horizon"] == AI
    wk6, wk13 = v["checkpoints"][5]["p_best"], v["checkpoints"][12]["p_best"]
    assert round(wk6[FIX], 3) == 0.589
    assert {k: round(p, 3) for k, p in wk13.items()} == {AI: 0.569, FIX: 0.324, KEEP: 0.107}


async def test_h4_same_request_same_view(client):
    q = d1({"horizon_weeks": 13, "evaluate": "at", "onsets": T4_ONSETS})
    assert (await post(client, q))["horizon_view"] == (await post(client, q))["horizon_view"]


async def test_unknown_onset_option_is_a_client_error(client):
    r = await client.post(ROUTE, json=d1({"horizon_weeks": 4, "onsets": [{"option_id": "nope", "onset_weeks": 1}]}))
    assert r.status_code == 422
    assert "HORIZON_ONSET_UNKNOWN_OPTION" in r.text
