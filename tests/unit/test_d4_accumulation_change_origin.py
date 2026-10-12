"""D4 (i') (DL 12 Oct; AIE #94 6133159652): an accumulation-dependent LEVEL target stated as a change from today.

"Reduce it by at least 15% relative to today's 8 weeks": the change origin is the target's stated TODAY level, a
positively held ``observed_state.baseline``, read through the existing known-base/level resolver. The cutoff is
8 x (1 - 0.15) = 6.8 weeks, compared with the goal evaluated at the resolved horizon. There, carrying on is
8 + 2/3 x 3 = 10 weeks: the forecast, never the reference and never "no change". Any other origin stays refused
(``accumulation_change_frame_unsupported``): a carrier's own observed field, a CHANGE node, or a value-only target
(its paired change would read the status quo AT THE HORIZON). The month count is the diagnostic 3-month design
(D4's own deadline is the end of the next calendar quarter); 0 provider calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

import src.services.robustness_analyzer_v2 as rav2

from src.models.robustness_v2 import RobustnessRequestV2
from tests.unit.test_r3_identity_evaluation import v2_body


STOCK, RATE, INFLOW, CARRIER, GOAL, ACT = (
    "cycle_today",
    "cycle_at_month_3_net_zero_rate",
    "net_change_per_month",
    "cycle_at_month_3",
    "cycle",
    "automation",
)
REASON = "accumulation_change_frame_unsupported"


def d4(
    *,
    frame: str = "change_rel",
    threshold: float = -0.15,
    unit_scale: float = 1.0,
    cap: float = 20.0,
    goal_observed: str = "baseline",
) -> Dict[str, Any]:
    """Today 8 weeks, a net 2/3 week a month for 3 months, an option that takes 4 weeks off at the horizon.

    ``unit_scale`` re-expresses every quantity in another native unit (7: days). ``cap`` re-frames every
    quantity (the option's normalised effect scales with it, so the raw world is the same).
    """
    s = unit_scale
    frames = {
        STOCK: cap * s,
        RATE: 100.0,
        INFLOW: 5.0 * s,
        CARRIER: cap * s,
        GOAL: cap * s,
        ACT: 1.0,
    }
    levels = {STOCK: 8.0 * s, RATE: 0.0, INFLOW: 2.0 / 3.0 * s, GOAL: 8.0 * s, ACT: 0.0}
    nodes: List[Dict[str, Any]] = []
    for node_id, node_frame in frames.items():
        node: Dict[str, Any] = {
            "id": node_id,
            "kind": "goal" if node_id == GOAL else ("outcome" if node_id == CARRIER else "factor"),
            "label": node_id,
            "execution_frame": {"frame": node_frame, "carrier": "cap"},
        }
        if node_id in levels and not (node_id == GOAL and goal_observed == "none"):
            observed = {"value": levels[node_id] / node_frame, "source": "brief_extraction"}
            if node_id != GOAL or goal_observed == "baseline":
                observed["baseline"] = levels[node_id] / node_frame
            node["observed_state"] = observed
        nodes.append(node)
    by_id = {n["id"]: n for n in nodes}
    by_id[CARRIER]["nonlinear_identity"] = {
        "operation": "accumulation",
        "factor_ids": [STOCK, RATE, INFLOW],
        "horizon_months": 3,
        "rate_scale": 0.01,
        "stated_in_brief": True,
    }
    by_id[GOAL]["nonlinear_identity"] = {
        "operation": "sum",
        "factor_ids": [CARRIER],
        "stated_in_brief": True,
    }
    by_id[GOAL]["raw_range"] = {"min": 0.0, "max": cap * s}
    effect = -4.0 / cap  # -4 weeks (spread 0.2 weeks), in the goal's normalised frame
    return {
        "request_id": "d4-change-origin",
        "graph": {
            "nodes": nodes,
            "edges": [
                {
                    "from": a,
                    "to": b,
                    "exists_probability": 1.0,
                    "strength": {"mean": m, "std": 0.2 / cap},
                }
                for a, b, m in [
                    (STOCK, CARRIER, 0.5),
                    (RATE, CARRIER, 0.5),
                    (INFLOW, CARRIER, 0.5),
                    (CARRIER, GOAL, 0.5),
                    (ACT, GOAL, effect),
                ]
            ],
        },
        "options": [
            {"id": "carry_on", "label": "Carry on", "interventions": {ACT: 0.0}},
            {"id": "automate", "label": "Automate", "interventions": {ACT: 1.0}},
        ],
        "goal_node_id": GOAL,
        "goal_threshold": threshold,
        "goal_threshold_frame": frame,
        "goal_direction": "minimise",
        "n_samples": 200,
        "seed": 7,
        "include_e_values": False,
        "include_voi": False,
        "include_factor_flips": False,
        "include_path_decomposition": False,
        "analysis_types": ["comparison"],
    }


def plan_of(d: Dict[str, Any]):
    return rav2.RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(
        RobustnessRequestV2.model_validate(d)
    )


def options(d: Dict[str, Any]) -> List[Tuple[str, float, Any]]:
    return [
        (o["id"], o["outcome"]["mean"], o.get("probability_of_goal")) for o in v2_body(d)["options"]
    ]


def refusals(d: Dict[str, Any]) -> List[str]:
    return [
        w["detail"].get("reason")
        for w in v2_body(d)["inference_warnings"]
        if w["detail"].get("reason") == REASON
    ]


def test_minus_15_percent_of_todays_8_is_6_8_against_10_at_the_horizon():
    plan, _ = plan_of(d4())
    assert plan is not None
    # Read through the known-base/level resolver: origin = today's stated 8 (0.4 x 20), cutoff 6.8 (0.34 x 20).
    assert (plan.stated_origin, plan.level_threshold, plan.frame_verdict) == (
        pytest.approx(0.4),
        pytest.approx(0.34),
        "scored",
    )
    (carry_id, carry_mean, carry_p), (auto_id, auto_mean, auto_p) = options(d4())
    assert (carry_id, auto_id) == ("carry_on", "automate")
    # Carrying on is the forecast at the horizon, 8 + 2/3 x 3 = 10 weeks: above 6.8 on every draw.
    assert carry_mean * 20.0 == pytest.approx(10.0, abs=1e-9) and carry_p == 0.0
    # The option takes 4 weeks off: about 6 weeks, at or below 6.8 on every draw.
    assert auto_mean * 20.0 == pytest.approx(6.0, abs=0.05) and auto_p == 1.0
    assert refusals(d4()) == []
    # The goal's sum is a horizon identity: never reconciled against today's 8 (no tolerance involved).
    goal_plan = rav2.resolve_identity_plans(RobustnessRequestV2.model_validate(d4()).graph)[GOAL]
    assert (goal_plan.withheld_reason, goal_plan.reconstructed, goal_plan.target_level) == (
        None,
        None,
        None,
    )


def test_the_same_cutoff_as_its_level_and_change_abs_twins_exactly():
    rel = options(d4())
    assert options(d4(frame="level", threshold=6.8 / 20.0)) == rel
    assert options(d4(frame="change_abs", threshold=-1.2 / 20.0)) == rel
    assert plan_of(d4(frame="change_abs", threshold=-1.2 / 20.0))[
        0
    ].level_threshold == pytest.approx(0.34)


@pytest.mark.parametrize("unit_scale, cap", [(7.0, 20.0), (1.0, 40.0)])
def test_native_unit_and_reframed_equivalents(unit_scale: float, cap: float):
    """8 weeks = 56 days (every frame x 7); and the same weeks on a cap of 40: the same cutoff in raw units."""
    d = d4(unit_scale=unit_scale, cap=cap)
    plan, _ = plan_of(d)
    frame = cap * unit_scale
    assert plan.level_threshold * frame == pytest.approx(6.8 * unit_scale)
    assert plan.stated_origin * frame == pytest.approx(8.0 * unit_scale)
    assert [(i, p) for i, _, p in options(d)] == [("carry_on", 0.0), ("automate", 1.0)]
    assert [m * frame for _, m, _ in options(d)] == pytest.approx(
        [10.0 * unit_scale, 6.0 * unit_scale], abs=0.05 * unit_scale
    )


@pytest.mark.parametrize("goal_observed", ["value_only", "none"])
def test_no_held_baseline_is_still_refused(goal_observed: str):
    """A value with no baseline would read the paired change (the status quo AT THE HORIZON); none at all has no origin."""
    d = d4(goal_observed=goal_observed)
    plan, warning = plan_of(d)
    assert plan is None and warning.detail["reason"] == REASON
    assert [p for _, _, p in options(d)] == [None, None]


@pytest.mark.parametrize("frame", ["change_abs", "change_rel"])
def test_a_carriers_own_observed_level_is_never_the_origin(frame: str):
    d = d4()
    carrier = next(n for n in d["graph"]["nodes"] if n["id"] == CARRIER)
    carrier["observed_state"] = {"value": 0.4, "baseline": 0.4, "source": "brief_extraction"}
    carrier["raw_range"] = {"min": 0.0, "max": 20.0}
    d["goal_constraints"] = [
        {
            "constraint_id": "carrier-change",
            "node_id": CARRIER,
            "operator": "<=",
            "value": -0.15,
            "value_frame": frame,
        }
    ]
    body = v2_body(d)
    rows = [
        w
        for w in body["inference_warnings"]
        if w["detail"].get("constraint_id") == "carrier-change"
    ]
    assert [(w["code"], w["detail"]["reason"]) for w in rows] == [
        ("CONSTRAINT_NOT_CONVERTIBLE", REASON)
    ]


def test_a_change_node_downstream_of_the_carrier_is_still_refused():
    d = d4(frame="change_abs", threshold=-1.2 / 20.0)
    next(n for n in d["graph"]["nodes"] if n["id"] == GOAL)["quantity_frame"] = "change"
    plan, warning = plan_of(d)
    assert plan is None and warning.detail["reason"] == REASON
