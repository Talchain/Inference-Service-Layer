"""Science 0d (#87 6027634829, ruling 1): an UNSIZED link still yields a driver row.

CEE's goal-chance RANGE reads one ISL driver row for the link nobody sized. Those links reach ISL with Olumi's
placeholder spread (CEE ``sizeLink`` placeholder: std = |mean| / 2; the projected default: mean 0.5, std 0.125) and
Olumi's existence prior (0.8). This pins that ISL ranks such a link as a ``link_strength`` (and ``link_existence``) driver
with two real group chances, so the range has a producer and is never manufactured downstream.
"""

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


def _payload(strength_mean, strength_std, exists=0.8, a_std=0.02, threshold=0.75):
    nodes = [
        {"id": "a", "kind": "factor", "label": "A", "observed_state": {"value": 0.5}},
        {"id": "lever", "kind": "factor", "label": "Lever", "observed_state": {"value": 0.0}},
        {"id": "goal", "kind": "outcome", "label": "Goal", "observed_state": {"value": 0.5, "baseline": 0.5}},
    ]
    edges = [
        {"from": "a", "to": "goal", "exists_probability": 1.0, "strength": {"mean": 1.0, "std": 0.01}},
        # The unsized link: Olumi's placeholder spread and existence prior.
        {"from": "lever", "to": "goal", "exists_probability": exists,
         "strength": {"mean": strength_mean, "std": strength_std}},
    ]
    return {
        "graph": {"nodes": nodes, "edges": edges},
        "options": [
            {"id": "act", "label": "Act", "interventions": {"lever": 1.0}},
            {"id": "hold", "label": "Hold", "interventions": {"lever": 0.0}},
        ],
        "goal_node_id": "goal",
        "parameter_uncertainties": [{"node_id": "a", "distribution": "normal", "std": a_std}],
        "n_samples": 2000,
        "seed": 7,
        "analysis_types": ["comparison"],
        # Level frame on the goal: baseline 0.5 + (option - status quo), so "act" meets it iff the link adds >= threshold - 0.5.
        "goal_threshold": threshold,
        "goal_threshold_frame": "delta",
    }


def _act(**kwargs):
    response = RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**_payload(**kwargs)))
    return {r.option_id: r for r in response.results}["act"]


def _row(option, kind):
    rows = [r for r in option.probability_of_goal_drivers.drivers if r.quantity_id == "lever->goal" and r.kind == kind]
    assert len(rows) == 1, f"expected one {kind} row for lever->goal, got {len(rows)}"
    return rows[0]


def test_placeholder_link_is_a_resolved_strength_driver_with_two_group_chances():
    act = _act(strength_mean=0.3, strength_std=0.15, threshold=0.75)  # CEE placeholder: std = |mean| / 2
    assert act.probability_of_goal is not None and 0.0 < act.probability_of_goal < 1.0
    row = _row(act, "link_strength")
    assert row.status == "resolved"
    assert row.n_low >= 30 and row.n_high >= 30
    assert 0.0 <= row.p_goal_if_low < row.p_goal_if_high <= 1.0  # a positive link: the weak third falls
    existence = _row(act, "link_existence")
    assert existence.n_absent > 0 and existence.n_present > 0


def test_projected_default_link_is_also_a_strength_driver():
    act = _act(strength_mean=0.5, strength_std=0.125, exists=0.8, threshold=0.95)  # STRENGTH_DEFAULT_SIGNATURE
    row = _row(act, "link_strength")
    assert row.n_low >= 30 and row.n_high >= 30
    assert row.p_goal_if_low <= row.p_goal_if_high


def test_contrast_a_near_fixed_link_gives_no_resolved_strength_row():
    # ISL refuses std <= 0.001; a near-fixed link (the existing suite's held std) cannot separate its thirds, so no
    # resolved strength row exists for it and CEE's range has nothing to read: it is never manufactured.
    act = _act(strength_mean=0.3, strength_std=0.0011, exists=1.0, threshold=0.75)
    rows = [r for r in act.probability_of_goal_drivers.drivers if r.quantity_id == "lever->goal" and r.kind == "link_strength"]
    assert all(r.status != "resolved" for r in rows)


# Science ruling 8 (Paul's science-validation-harness rule): a KNOWN ANSWER, derived analytically, never from ISL itself.
# In the delta frame "act" meets the goal iff the unsized link adds at least d (the other path is common to both options;
# its node is held near-fixed so the analytic answer is exact). With w ~ N(0.5, 0.1) and d at w's 1/6 quantile:
# P(goal) = 5/6; the weakest third of w (below its 1/3 quantile) meets it in (1/3 - 1/6) / (1/3) = 1/2 of runs; the
# strongest third always does. With existence 0.8 the absent runs never meet it, the present runs meet it at 5/6, and
# P(goal) = 0.8 * 5/6 = 2/3. Averaged over 8 seeds (n = 2000 each): tolerances are ~3 standard errors of the MEAN.
# Measured 7 Oct at 9ddca912: P 0.8361, weakest third 0.5079, strongest 1.0 (single seed 7: 0.819 / 0.4565 — why 8).
from statistics import NormalDist, mean

from src.models.robustness_v2 import RobustnessRequestV2 as _Req

_MU, _SD = 0.5, 0.1
_D = NormalDist(_MU, _SD).inv_cdf(1 / 6)
_SEEDS = range(1, 9)


def _act_seeded(seed, exists):
    payload = _payload(strength_mean=_MU, strength_std=_SD, exists=exists, a_std=0.0011, threshold=0.5 + _D)
    payload["seed"] = seed
    return {r.option_id: r for r in RobustnessAnalyzerV2().analyze(_Req(**payload)).results}["act"]


def test_known_answer_strength_row_matches_the_analytic_tercile_conditional_chance():
    acts = [_act_seeded(seed, exists=1.0) for seed in _SEEDS]
    rows = [_row(act, "link_strength") for act in acts]
    assert all(row.status == "resolved" for row in rows)
    assert abs(mean(act.probability_of_goal for act in acts) - 5 / 6) <= 0.01
    assert abs(mean(row.p_goal_if_low for row in rows) - 0.5) <= 0.021
    assert min(row.p_goal_if_high for row in rows) >= 0.99


def test_known_answer_existence_row_matches_the_analytic_present_and_absent_chances():
    acts = [_act_seeded(seed, exists=0.8) for seed in _SEEDS]
    rows = [_row(act, "link_existence") for act in acts]
    assert abs(mean(act.probability_of_goal for act in acts) - 2 / 3) <= 0.013
    assert max(row.p_goal_if_absent for row in rows) <= 0.01
    assert abs(mean(row.p_goal_if_present for row in rows) - 5 / 6) <= 0.012
