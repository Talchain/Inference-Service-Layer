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
