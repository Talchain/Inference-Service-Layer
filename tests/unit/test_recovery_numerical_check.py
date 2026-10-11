"""Batch 0: closed-form checks of attainment, ranking, frames, and units.

The only random input is S ~ Normal(0.5, 0.1). The factor is currently
x0 = 0.5; pushing it to x = 1 gives the paired effect S * (x - x0).
Thus P(effect >= T) = Phi((mu * (x - x0) - T) / (sigma * abs(x - x0))).
All expectations below come from this formula, independently of the engine.
"""

import math

import pytest

from src.models.robustness_v2 import (
    InterventionOption,
    ObservedState,
    RawRange,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from tests.unit.test_goal_node_propagation import _edge
from tests.unit.test_goal_threshold_frame import build_request


# Absolute Monte Carlo tolerance, about four standard errors at 10,000 draws.
TOL = 0.02
N_SAMPLES = 10_000
SEED = 42


def phi(z):
    """Standard Normal CDF, with no engine-derived expected values."""
    return (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0


def linear_request(frame, *, factor_scale=1.0, raw_unit_scale=1.0, sigma=0.1):
    """Adapt the existing witness builder; keep the real analyser call path."""
    # Goal raw range [20, 120], baseline 70 -> normalised baseline 0.5.
    # Re-express raw targets and range together when changing goal display units.
    raw_min, raw_max = 20.0 * raw_unit_scale, 120.0 * raw_unit_scale
    span = raw_max - raw_min
    raw_baseline = 70.0 * raw_unit_scale
    thresholds = {
        "level": (100.0 * raw_unit_scale - raw_min) / span,
        "change_abs": 20.0 * raw_unit_scale / span,
        "change_rel": 25.0 * raw_unit_scale / raw_baseline,
    }
    request = build_request(
        goal_threshold=thresholds[frame],
        goal_threshold_frame=frame,
        baseline=0.5,
        goal_observed_value=0.5,
        goal_has_observed_state=frame != "change_abs",
        factor_now=0.5 * factor_scale,
        factor_pu=False,
    )
    request.graph.edges = [_edge("f", "g", 0.5 / factor_scale, sigma / factor_scale)]
    goal = next(node for node in request.graph.nodes if node.id == "g")
    goal.quantity_frame = "change" if frame == "change_abs" else "level"
    goal.raw_range = RawRange(min=raw_min, max=raw_max)
    if frame != "change_abs":
        goal.observed_state = ObservedState(
            value=0.5, baseline=0.5, source="user_confirmed"
        )
    request.options = [
        InterventionOption(id="hold", label="Status quo", interventions={}),
        InterventionOption(
            id="push", label="Push", interventions={"f": 1.0 * factor_scale}
        ),
    ]
    request.n_samples = N_SAMPLES
    request.seed = SEED
    request.goal_direction = "maximise"
    request.analysis_types = ["comparison"]
    return request


def test_level_attainment_is_not_win_share():
    """Catch a win share shown as a goal chance, or a level anchored at zero."""
    response = RobustnessAnalyzerV2().analyze(linear_request("level"))
    results = {result.option_id: result for result in response.results}
    # Goal level = 0.5 + 0.5*S; target = (100-20)/100 = 0.8.
    # P(goal) = Phi((0.5*0.5 - (0.8-0.5))/(0.1*0.5))
    #         = Phi(-1) = 0.1586552539.
    expected_goal = phi((0.5 * 0.5 - 0.3) / (0.1 * 0.5))
    # Push beats hold iff S*1 > S*0.5, i.e. S>0:
    # P(win) = Phi(0.5/0.1) = Phi(5) = 0.9999997133.
    expected_win = phi(0.5 / 0.1)
    assert results["push"].probability_of_goal == pytest.approx(expected_goal, abs=TOL)
    assert results["push"].win_probability == pytest.approx(expected_win, abs=TOL)
    # Doing nothing leaves the goal at 0.5, below 0.8 on every draw.
    assert results["hold"].probability_of_goal == 0.0
    assert results["hold"].win_probability == pytest.approx(1.0 - expected_win, abs=TOL)


def test_absolute_change_uses_paired_status_quo_without_baseline():
    """Catch absolute propagated scores mistaken for paired change, or a required baseline."""
    request = linear_request("change_abs")
    goal = next(node for node in request.graph.nodes if node.id == "g")
    assert goal.quantity_frame == "change"
    assert goal.observed_state is None
    response = RobustnessAnalyzerV2().analyze(request)
    # T=20/100=0.2, effect=S*(1-0.5):
    # P(goal) = Phi((0.5*0.5-0.2)/(0.1*0.5)) = Phi(1) = 0.8413447461.
    expected = phi((0.5 * 0.5 - 0.2) / (0.1 * 0.5))
    results = {result.option_id: result for result in response.results}
    assert results["push"].probability_of_goal == pytest.approx(expected, abs=TOL)
    # Paired change for hold is exactly zero, despite a nonzero propagated score.
    assert results["hold"].probability_of_goal == 0.0


def test_relative_change_uses_user_baseline_and_nonzero_raw_origin():
    """Catch relative change computed against zero or against the normalised baseline."""
    response = RobustnessAnalyzerV2().analyze(linear_request("change_rel"))
    # B_raw=0.5*(120-20)+20=70; requested relative change r=25/70=5/14.
    # Required normalised effect T=r*B_raw/(120-20)=0.25.
    # P(goal) = Phi((0.5*0.5-0.25)/(0.1*0.5)) = Phi(0) = 0.5.
    expected = phi((0.5 * 0.5 - (5.0 / 14.0) * 70.0 / 100.0) / (0.1 * 0.5))
    results = {result.option_id: result for result in response.results}
    assert results["push"].probability_of_goal == pytest.approx(expected, abs=TOL)
    assert results["hold"].probability_of_goal == 0.0


@pytest.mark.parametrize("threshold, expected", [(0.4, 1.0), (0.6, 0.0)])
def test_near_deterministic_status_quo_has_exact_attainment(threshold, expected):
    """Catch spurious progress or noise in the paired status quo at a level target."""
    # 0.0011 is the smallest convenient valid sigma (the schema requires >0.001).
    request = linear_request("level", sigma=0.0011)
    request.goal_threshold = threshold
    response = RobustnessAnalyzerV2().analyze(request)
    hold = next(result for result in response.results if result.option_id == "hold")
    # Hold's level = B + S*(x0-x0) = 0.5 EXACTLY, even for unbounded Normal S.
    # Hence P(level>=0.4)=1 and P(level>=0.6)=0, not merely rounded Normal tails.
    assert hold.probability_of_goal == expected


@pytest.mark.parametrize(
    "frame, z", [("level", -1.0), ("change_abs", 1.0), ("change_rel", 0.0)]
)
def test_units_preserve_goal_probability_and_win_shares(frame, z):
    """Catch unit-dependent attainment/ranking or failure to rescale strength uncertainty."""
    # x'=k*x and S'=S/k preserve S'*(x'-x0')=S*(x-x0), including its variance.
    # Factor-only changes leave a goal-unit threshold unchanged. Also re-express
    # goal raw units by k: [20,120]->[200,1200], B_raw 70->700, level target
    # 100->1000, absolute target 20->200, relative target 25/70->250/700.
    # Normalised wire thresholds therefore remain 0.8, 0.2, and 5/14.
    original = RobustnessAnalyzerV2().analyze(linear_request(frame))
    scaled = RobustnessAnalyzerV2().analyze(
        linear_request(frame, factor_scale=10.0, raw_unit_scale=10.0)
    )
    before = {result.option_id: result for result in original.results}
    after = {result.option_id: result for result in scaled.results}
    # Same hand-derived probabilities as above: Phi(-1), Phi(1), Phi(0).
    # Push wins iff S>0: Phi(5); hold wins iff S<0: 1-Phi(5).
    for option_id, expected_goal, expected_win in (
        ("push", phi(z), phi(5.0)),
        ("hold", 0.0, 1.0 - phi(5.0)),
    ):
        for result in (before[option_id], after[option_id]):
            assert result.probability_of_goal == pytest.approx(expected_goal, abs=TOL)
            assert result.win_probability == pytest.approx(expected_win, abs=TOL)
        assert after[option_id].probability_of_goal == pytest.approx(
            before[option_id].probability_of_goal, abs=TOL
        )
        assert after[option_id].win_probability == pytest.approx(
            before[option_id].win_probability, abs=TOL
        )
