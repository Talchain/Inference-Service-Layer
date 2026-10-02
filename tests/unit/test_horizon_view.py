"""
Horizon view (SCIENCE/DSK EXPERIMENT, #85 5947449217): the pure closed-form model in ``src/services/horizon_view.py``.

Every winner below is decided by the analyser's REAL canonical owner (``RobustnessAnalyzerV2._winners_for_draw``),
so a row cannot pass on a local ``max`` that agrees with the owner only under maximise.
"""

import math

import pytest

from src.services.horizon_view import compute_horizon_view, fraction_for, realised_fraction
from src.services.robustness_analyzer_v2 import ObjectivePlan, RobustnessAnalyzerV2


def owner(sense: str):
    plan = ObjectivePlan(sense=sense, attested=True)
    return lambda finite, sq: RobustnessAnalyzerV2._winners_for_draw(finite, plan, None)


# Three draws, three options. Under maximise A wins draws 0 and 1 and B wins draw 2; under minimise C wins all.
OUTCOMES = {"A": [5.0, 4.0, 1.0], "B": [3.0, 2.0, 6.0], "C": [0.5, 0.5, 0.5]}
SQ = [1.0, 1.0, 1.0]


def test_realised_fraction_step_ramp_and_immediate():
    assert [realised_fraction(u, 0, 0) for u in (1, 2)] == [1.0, 1.0]  # today's assumption: in force from week 1
    assert [realised_fraction(u, 6, 0) for u in (6, 7)] == [0.0, 1.0]  # ships after week 6
    assert [realised_fraction(u, 1, 2) for u in (1, 2, 3, 4)] == [0.0, 0.5, 1.0, 1.0]  # ramps over weeks 2-3
    assert fraction_for(4, 1, 2, "cumulative") == pytest.approx((0 + 0.5 + 1 + 1) / 4)


def test_identity_no_onsets_reproduces_the_owner_win_share_exactly():
    view = compute_horizon_view(["A", "B", "C"], OUTCOMES, SQ, {}, 3, "cumulative", 3, owner("maximise"))
    for c in view["checkpoints"]:
        assert c["p_best"] == {"A": 2 / 3, "B": 1 / 3, "C": 0.0}
    assert view["flips"] == [] and view["leader_at_horizon"] == "A"


def test_every_draw_goes_through_the_owner_minimise_ranks_the_horizon_too():
    view = compute_horizon_view(["A", "B", "C"], OUTCOMES, SQ, {}, 2, "at", 3, owner("minimise"))
    assert view["checkpoints"][0]["p_best"] == {"A": 0.0, "B": 0.0, "C": 1.0}
    assert view["leader_at_horizon"] == "C"


def test_full_effect_is_the_outcome_verbatim_even_without_a_status_quo():
    nan_sq = [math.nan, math.nan, math.nan]
    view = compute_horizon_view(["A", "B", "C"], OUTCOMES, nan_sq, {}, 1, "at", 3, owner("maximise"))
    assert view["checkpoints"][0]["p_best"] == {"A": 2 / 3, "B": 1 / 3, "C": 0.0}


def test_before_onset_an_option_is_the_status_quo_and_ties_split():
    # A acts from week 3; B never acts within 2 weeks; C in force. Week 1: A = B = SQ (1.0) > C (0.5) → A and B tie.
    onsets = {"A": (2.0, 0.0), "B": (5.0, 0.0)}
    view = compute_horizon_view(["A", "B", "C"], OUTCOMES, SQ, onsets, 3, "at", 3, owner("maximise"))
    wk1, wk3 = view["checkpoints"][0], view["checkpoints"][2]
    assert wk1["p_best"] == {"A": 0.5, "B": 0.5, "C": 0.0} and wk1["leader_option_id"] is None
    # A in force (5, 4, 1) vs B still at SQ (1): A wins draws 0 and 1, and ties B at draw 2.
    assert wk3["p_best"] == {"A": 2.5 / 3, "B": 0.5 / 3, "C": 0.0}
    assert view["flips"] == [{"week": 3, "from_option_id": None, "to_option_id": "A"}]


def test_uninformative_draws_credit_nobody():
    out = {"A": [math.nan, 2.0], "B": [math.nan, 1.0]}
    view = compute_horizon_view(["A", "B"], out, [0.0, 0.0], {}, 1, "at", 2, owner("maximise"))
    assert view["checkpoints"][0]["p_best"] == {"A": 0.5, "B": 0.0}


def test_unknown_onset_option_is_refused():
    with pytest.raises(ValueError, match="HORIZON_ONSET_UNKNOWN_OPTION"):
        compute_horizon_view(["A"], {"A": [1.0]}, [0.0], {"Z": (1.0, 0.0)}, 1, "at", 1, owner("maximise"))
