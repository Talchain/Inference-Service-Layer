"""
Decision-level flip threshold (SCIENCE ROBUSTNESS step 1, EXPERIMENT): the pure grid + bisection search in
``src/services/decision_flip.py``. ``leader_at`` stands in for "the analyser's recommendation at link value x".
"""

import pytest

from src.services.decision_flip import decision_flip_threshold


def step_at(t, below="B", above="A"):
    return lambda x: above if x >= t else below


def test_finds_the_threshold_within_tol_and_names_the_new_leader():
    res = decision_flip_threshold(step_at(0.0625), current=0.25, bound=0.0, tol=0.001)
    assert res["exists"] is True and res["leader"] == "A" and res["to_option_id"] == "B"
    lo, hi = res["bracket"]
    assert lo < 0.0625 <= hi and hi - lo <= 0.001
    assert abs(res["threshold"] - 0.0625) <= 0.001


def test_no_change_before_the_bound_is_an_honest_none():
    res = decision_flip_threshold(lambda x: "A", current=0.25, bound=0.0)
    assert res == {"exists": False, "leader": "A", "threshold": None, "bracket": None, "to_option_id": None,
                   "evaluations": 9}


def test_the_nearest_change_wins_when_the_curve_changes_twice():
    # A on [0.2, 0.25], B on [0.05, 0.2), A again below 0.05: a bisection between current and bound alone would see
    # A at both ends and report "no change"; the grid finds the nearest change at 0.2.
    leader = lambda x: "A" if x >= 0.2 or x < 0.05 else "B"
    res = decision_flip_threshold(leader, current=0.25, bound=0.0, tol=0.001)
    assert res["exists"] is True and res["to_option_id"] == "B"
    assert abs(res["threshold"] - 0.2) <= 0.001


def test_a_change_exactly_at_the_bound_is_found():
    res = decision_flip_threshold(lambda x: "A" if x > 0 else "B", current=0.25, bound=0.0, tol=0.001)
    assert res["exists"] is True and res["bracket"][0] == 0.0 and res["bracket"][1] <= 0.001


def test_stronger_direction_searches_upwards():
    res = decision_flip_threshold(lambda x: "B" if x > 0.7 else "A", current=0.25, bound=1.0, tol=0.001)
    assert res["exists"] is True and abs(res["threshold"] - 0.7) <= 0.001


@pytest.mark.parametrize("kw", [{"grid": 0}, {"tol": 0}])
def test_bad_parameters_are_refused(kw):
    with pytest.raises(ValueError, match="DECISION_FLIP_BAD_PARAMS"):
        decision_flip_threshold(lambda x: "A", 0.25, 0.0, **kw)
