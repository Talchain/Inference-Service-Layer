"""R1 S2 (DL #72 5871412823; meaning AIQ 5871459631 / 5872801411; wire 5872798858): ONE typed
target contract, ISL half.

A target is now stated in one of four frames (``level`` · legacy ``delta`` · ``change_abs`` ·
``change_rel``) on a node that measures a ``level`` or a ``change`` (``NodeV2.quantity_frame``).
Before R1, every producer stamped ``level`` and a change from today had no honest spelling: Paul's
"maintain code quality" (a change node, target Δ >= 0) was never scored, and "+10% productivity"
had to be pre-multiplied into a level on a base nobody had checked.

THE RULES (AIQ Q-A / Q-B, adversarially checked on ISL e24c88c):
- ``change_abs`` c on a LEVEL node with a known base b goes through THE SAME level resolver at
  b + c, so P equals the level twin's P by construction (clip, pinned exact compare and the
  epsilon refusal included).
- With no base, it is the paired change Δ = option − status quo, compared with c.
- A pinned option compares (x − today) with c exactly; a pinned root uses its central value as
  today. An unpinned root has change 0 exactly (P = 1[0 op c]).
- A CHANGE node compares its own paired change: 0 today by definition, so carrying on meets
  "maintain" by definition. ``change_rel`` on a change node is refused by name
  (CHANGE_OF_A_CHANGE).
- ``change_rel`` r is a change of r · b_raw, with b_raw = b_n·(max − min) + min from the node's
  ``raw_range``; no base → GOAL_BASE_MISSING, b_raw = 0 or no raw_range → refused by name.

THE WITNESS GRAPH, every expected number derived by hand::

    f  root driver, observed 0.5, uniform(0, 1)
    g  goal, non-root, g = 0.5 f
    c  target, non-root, c = 0.5 f  (observed value 0.7, baseline 0.7 unless a row removes it)
    option PUSH = do(f := 1)  ->  Δc = 0.5 (1 − f) ∈ [0, 0.5]
    option CUT  = do(f := 0)  ->  Δc = −0.5 f     ∈ [−0.5, 0]

    P(Δc >= 0.1 | PUSH) = P(f <= 0.8) = 0.8

n = 10,000, so one MC standard error on p ~ 0.8 is ~0.004 and TOL = 0.02 is ~5 SE. Every
constraint is located by its constraint_id, never by a value predicate.
"""

from typing import Dict, List, Optional

import pytest
from pydantic import ValidationError

from src.models.robustness_v2 import (
    EdgeV2,
    GoalConstraint,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
    RawRange,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import (
    BASELINE_OWNER_BY_OBSERVED_SOURCE,
    RobustnessAnalyzerV2,
    baseline_owner,
)

N_SAMPLES = 10_000
SEED = 42
TOL = 0.02
CID = "cid-r1"
PUSH = {"f": 1.0}
CUT = {"f": 0.0}


def edge(src: str, dst: str, mean: float) -> EdgeV2:
    return EdgeV2(
        **{"from": src, "to": dst},
        exists_probability=1.0,
        strength=StrengthDistribution(mean=mean, std=0.0011),
    )


def build(
    *,
    value: float = 0.1,
    frame: Optional[str] = "change_abs",
    operator: str = ">=",
    options: Optional[Dict[str, Dict[str, float]]] = None,
    target_observed: Optional[ObservedState] = ObservedState(value=0.7, baseline=0.7),
    target_quantity: Optional[str] = None,
    target_raw_range: Optional[RawRange] = None,
    target_strength: float = 0.5,
    target_is_root: bool = False,
    goal_threshold: Optional[float] = None,
    goal_threshold_frame: Optional[str] = None,
    goal_observed: Optional[ObservedState] = None,
    constraint_on_target: bool = True,
) -> RobustnessRequestV2:
    nodes = [
        NodeV2(id="f", kind="factor", label="Driver", observed_state=ObservedState(value=0.5)),
        NodeV2(id="g", kind="outcome", label="Goal", observed_state=goal_observed),
        NodeV2(
            id="c",
            kind="outcome",
            label="Target",
            observed_state=target_observed,
            quantity_frame=target_quantity,
            raw_range=target_raw_range,
        ),
    ]
    edges = [edge("f", "g", 0.5)]
    uncertainties = [
        ParameterUncertainty(node_id="f", distribution="uniform", range_min=0.0, range_max=1.0)
    ]
    if target_is_root:
        uncertainties.append(
            ParameterUncertainty(node_id="c", distribution="uniform", range_min=0.0, range_max=1.0)
        )
    else:
        edges.append(edge("f", "c", target_strength))
    constraints: List[GoalConstraint] = []
    if constraint_on_target:
        kwargs = {
            "constraint_id": CID,
            "node_id": "c",
            "operator": operator,
            "value": value,
            "label": "Target",
        }
        if frame is not None:
            kwargs["value_frame"] = frame
        constraints.append(GoalConstraint(**kwargs))
    return RobustnessRequestV2(
        request_id="r1-s2",
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(id=oid, label=oid, interventions=iv)
            for oid, iv in (options or {"push": PUSH}).items()
        ],
        goal_node_id="g",
        n_samples=N_SAMPLES,
        seed=SEED,
        goal_threshold=goal_threshold,
        goal_threshold_frame=goal_threshold_frame,
        goal_constraints=constraints or None,
        parameter_uncertainties=uncertainties,
    )


def run(request: RobustnessRequestV2):
    return RobustnessAnalyzerV2().analyze(request)


def result_of(response, option_id: str):
    matches = [r for r in response.results if r.option_id == option_id]
    assert len(matches) == 1, [r.option_id for r in response.results]
    return matches[0]


def p_satisfied(response, option_id: str, constraint_id: str = CID) -> float:
    analysis = result_of(response, option_id).constraint_analysis
    assert analysis is not None, [w.code for w in response.inference_warnings]
    rows = [c for c in analysis.constraints if c.constraint_id == constraint_id]
    assert len(rows) == 1, [c.constraint_id for c in analysis.constraints]
    return rows[0].prob_satisfied


def constraint_row(response, option_id: str, constraint_id: str = CID):
    analysis = result_of(response, option_id).constraint_analysis
    assert analysis is not None, [w.code for w in response.inference_warnings]
    (row,) = [c for c in analysis.constraints if c.constraint_id == constraint_id]
    return row


def refusal(response, code: str):
    return [w for w in response.inference_warnings if w.code == code]


def no_row(response, option_id: str, constraint_id: str = CID) -> bool:
    analysis = result_of(response, option_id).constraint_analysis
    return analysis is None or not any(c.constraint_id == constraint_id for c in analysis.constraints)


# ---------------------------------------------------------------------------------------------------------
# The contract parses: four target frames, the node quantity, the raw range, the base owner.
# ---------------------------------------------------------------------------------------------------------


class TestTheContractParses:
    @pytest.mark.parametrize("frame", ["level", "delta", "change_abs", "change_rel"])
    def test_every_frame_is_accepted_on_a_limit_and_on_the_goal(self, frame):
        request = build(frame=frame, goal_threshold=0.1, goal_threshold_frame=frame)
        assert request.goal_constraints[0].value_frame == frame
        assert request.goal_threshold_frame == frame

    def test_an_unknown_frame_is_still_rejected(self):
        with pytest.raises(ValidationError):
            build(frame="percent")

    def test_quantity_frame_absent_means_level(self):
        request = build()
        (target,) = [n for n in request.graph.nodes if n.id == "c"]
        assert target.quantity_frame is None

    def test_quantity_frame_takes_only_level_or_change(self):
        with pytest.raises(ValidationError):
            NodeV2(id="x", kind="outcome", label="x", quantity_frame="rate")

    def test_a_raw_range_must_be_finite_and_increasing(self):
        assert RawRange(min=20.0, max=120.0).max == 120.0
        with pytest.raises(ValidationError):
            RawRange(min=5.0, max=5.0)
        with pytest.raises(ValidationError):
            RawRange(min=0.0, max=float("inf"))

    def test_no_wire_carries_an_owner_field(self):
        # decision (b): the owner is read from observed_state.source; a stray field is ignored, never read.
        assert "baseline_owner" not in ObservedState.model_fields
        stray = ObservedState(value=0.5, baseline=0.5, baseline_owner="user")
        assert not hasattr(stray, "baseline_owner")

    # The 0.61.0 CHANGELOG's classes, row for row. Anything else is UNKNOWN (None), the fail-safe side.
    @pytest.mark.parametrize(
        "source, owner",
        [
            ("brief_extraction", "user"),
            ("explicit", "user"),
            ("user_override", "user"),
            ("user_confirmed", "user"),
            ("user", "user"),
            ("user_edited", "user"),
            ("user_calibration", "user"),
            ("panel_elicited", "user"),
            ("cee_inference", "olumi"),
            ("inferred", "olumi"),
            ("cee_repair", "olumi"),
            ("user_assumption", None),
            ("user_stated", None),
            ("computed", None),
            (None, None),
        ],
    )
    def test_the_owner_is_the_documented_class_of_the_source(self, source, owner):
        assert baseline_owner(ObservedState(value=0.5, baseline=0.5, source=source)) == owner

    def test_the_owner_map_is_exactly_the_documented_classes(self):
        assert dict(BASELINE_OWNER_BY_OBSERVED_SOURCE) == {
            **dict.fromkeys(
                (
                    "brief_extraction",
                    "explicit",
                    "user_override",
                    "user_confirmed",
                    "user",
                    "user_edited",
                    "user_calibration",
                    "panel_elicited",
                ),
                "user",
            ),
            **dict.fromkeys(("cee_inference", "inferred", "cee_repair"), "olumi"),
        }
        assert baseline_owner(None) is None


# ---------------------------------------------------------------------------------------------------------
# Q-A: change_abs on a LEVEL node.
# ---------------------------------------------------------------------------------------------------------


class TestChangeAbsOnALevelNode:
    def test_with_a_base_it_equals_its_level_twin_exactly(self):
        """Row (a): level (b + c) ≡ change_abs c, per option, because both run the SAME resolver.
        Mutant: a separate paired-Δ arithmetic for the base-known case would still agree here, so the
        byte-equality is paired with the pinned and clip rows below."""
        options = {"push": PUSH, "keep": {}}
        change = run(build(value=0.1, frame="change_abs", options=options))
        level = run(build(value=0.8, frame="level", options=options))
        for option_id in options:
            assert p_satisfied(change, option_id) == p_satisfied(level, option_id)
        assert p_satisfied(change, "push") == pytest.approx(0.8, abs=TOL)
        assert p_satisfied(change, "keep") == 0.0

    def test_with_no_base_it_is_the_paired_change(self):
        """No baseline: the level frame refuses (missing_target_baseline), change_abs does not need
        one. Δ = 0.5 (1 − f) regardless of the base, so P = 0.8."""
        observed = ObservedState(value=0.7)
        level = run(build(value=0.8, frame="level", target_observed=observed))
        assert no_row(level, "push")
        change = run(build(value=0.1, frame="change_abs", target_observed=observed))
        assert p_satisfied(change, "push") == pytest.approx(0.8, abs=TOL)

    def test_with_no_observed_state_at_all_it_is_still_the_paired_change(self):
        change = run(build(value=0.1, frame="change_abs", target_observed=None))
        assert p_satisfied(change, "push") == pytest.approx(0.8, abs=TOL)

    def test_a_pinned_option_is_compared_at_its_own_change_exactly(self):
        """do(c := 0.9) on a node whose today is 0.7: change 0.2, exactly, on every draw.
        Mutant (compare x itself with c): 0.9 >= 0.25 → 1.0 on the second row → RED."""
        options = {"set": {"c": 0.9}}
        met = run(build(value=0.15, frame="change_abs", options=options))
        assert p_satisfied(met, "set") == 1.0
        missed = run(build(value=0.25, frame="change_abs", options=options))
        assert p_satisfied(missed, "set") == 0.0

    def test_a_pinned_option_with_no_today_is_refused_by_name(self):
        options = {"set": {"c": 0.9}}
        response = run(build(value=0.15, frame="change_abs", options=options, target_observed=None))
        assert no_row(response, "set")
        (warning,) = refusal(response, "GOAL_BASE_MISSING")
        assert warning.detail["constraint_id"] == CID

    def test_an_unpinned_root_has_no_change_at_all(self):
        """A root no option touches: change 0 on every draw, so P = 1[0 >= c], exactly. Its LEVEL
        twin reads the root's sampled level instead (uniform(0, 1) >= 0.4 → 0.6): a separate row."""
        at_zero = run(build(value=0.0, frame="change_abs", target_is_root=True))
        assert p_satisfied(at_zero, "push") == 1.0
        above = run(build(value=0.05, frame="change_abs", target_is_root=True))
        assert p_satisfied(above, "push") == 0.0
        level = run(build(value=0.4, frame="level", target_is_root=True))
        assert p_satisfied(level, "push") == pytest.approx(0.6, abs=TOL)

    def test_a_pinned_root_is_compared_against_its_central_value(self):
        """do(c := 0.9) on a root whose central value is 0.5 (uniform(0, 1)): change 0.4 exactly.
        Mutant (paired against the status-quo DRAW): P(0.9 − U >= 0.35) = 0.55 → RED."""
        options = {"set": {"c": 0.9}}
        met = run(build(value=0.35, frame="change_abs", options=options, target_is_root=True))
        assert p_satisfied(met, "set") == 1.0
        missed = run(build(value=0.45, frame="change_abs", options=options, target_is_root=True))
        assert p_satisfied(missed, "set") == 0.0


# ---------------------------------------------------------------------------------------------------------
# Q-B: a CHANGE node compares its own paired change; change_rel on it is refused by name.
# ---------------------------------------------------------------------------------------------------------


class TestAChangeNode:
    def test_carrying_on_meets_maintain_by_definition(self):
        """"Code quality change" with a NEGATIVE driver: raw samples are −0.5 f < 0 on almost every
        draw, so the raw-samples mutant scores "maintain (>= 0)" at ~0 for carrying on. The paired
        change is 0 today by definition → exactly 1."""
        options = {"keep": {}, "cut": CUT}
        response = run(
            build(value=0.0, frame="change_abs", options=options, target_quantity="change", target_strength=-0.5)
        )
        assert p_satisfied(response, "keep") == 1.0
        # CUT: Δ = −0.5 (0 − f) = +0.5 f >= 0 on every draw.
        assert p_satisfied(response, "cut") == 1.0

    def test_a_level_frame_on_a_change_node_reads_the_same_paired_change(self):
        options = {"keep": {}, "push": PUSH}
        change = run(build(value=0.1, frame="change_abs", options=options, target_quantity="change"))
        level = run(build(value=0.1, frame="level", options=options, target_quantity="change"))
        for option_id in options:
            assert p_satisfied(level, option_id) == p_satisfied(change, option_id)
        assert p_satisfied(change, "push") == pytest.approx(0.8, abs=TOL)

    def test_a_pinned_change_node_is_compared_at_the_change_it_sets(self):
        options = {"set": {"c": 0.2}}
        response = run(build(value=0.15, frame="change_abs", options=options, target_quantity="change"))
        assert p_satisfied(response, "set") == 1.0

    def test_change_rel_on_a_change_node_is_refused_by_name(self):
        response = run(
            build(
                value=0.1,
                frame="change_rel",
                target_quantity="change",
                target_raw_range=RawRange(min=0.0, max=100.0),
            )
        )
        assert no_row(response, "push")
        (warning,) = refusal(response, "CHANGE_OF_A_CHANGE")
        assert warning.detail["constraint_id"] == CID


# ---------------------------------------------------------------------------------------------------------
# change_rel: a change of r · b_raw, on the node's own raw range.
# ---------------------------------------------------------------------------------------------------------


class TestChangeRel:
    def test_a_relative_change_on_a_zero_origin_range(self):
        """raw_range [0, 100], b_n 0.7 → b_raw 70; r = 1/7 → c_n = 0.1 → P = 0.8."""
        response = run(build(value=1 / 7, frame="change_rel", target_raw_range=RawRange(min=0.0, max=100.0)))
        assert p_satisfied(response, "push") == pytest.approx(0.8, abs=TOL)

    def test_a_relative_change_on_an_offset_range_reads_the_raw_base(self):
        """raw_range [20, 120], b_n 0.7 → b_raw 90; r = 1/9 → c_n = 0.1 → P = 0.8.
        Mutant b_n·(1 + r) (c_n = 0.0778): P(f <= 0.844) = 0.844 → RED at TOL 0.02."""
        response = run(build(value=1 / 9, frame="change_rel", target_raw_range=RawRange(min=20.0, max=120.0)))
        assert p_satisfied(response, "push") == pytest.approx(0.8, abs=TOL)

    def test_the_sign_survives_both_ways(self):
        """AIQ row 2: "cut by at least 20%" = change_rel <= −0.20; "by at most 20%" = >= −0.20.
        CUT: Δ = −0.5 f; b_raw 70 on [0, 100] → c_n = −0.14.
        P(Δ <= −0.14) = P(f >= 0.28) = 0.72; P(Δ >= −0.14) = 0.28. Mutant (sign dropped, c_n +0.14):
        P(Δ <= 0.14) = 1 → RED."""
        raw = RawRange(min=0.0, max=100.0)
        at_least = run(build(value=-0.2, frame="change_rel", operator="<=", options={"cut": CUT}, target_raw_range=raw))
        at_most = run(build(value=-0.2, frame="change_rel", operator=">=", options={"cut": CUT}, target_raw_range=raw))
        assert p_satisfied(at_least, "cut") == pytest.approx(0.72, abs=TOL)
        assert p_satisfied(at_most, "cut") == pytest.approx(0.28, abs=TOL)

    def test_no_base_is_refused_by_name(self):
        response = run(
            build(value=0.1, frame="change_rel", target_observed=None, target_raw_range=RawRange(min=0.0, max=100.0))
        )
        assert no_row(response, "push")
        (warning,) = refusal(response, "GOAL_BASE_MISSING")
        assert warning.detail["constraint_id"] == CID

    def test_a_zero_raw_base_is_refused_by_name(self):
        """"20% of nothing": b_n 0.2 on [−25, 100] → b_raw = 0.2·125 − 25 = 0."""
        response = run(
            build(
                value=0.2,
                frame="change_rel",
                target_observed=ObservedState(value=0.2, baseline=0.2),
                target_raw_range=RawRange(min=-25.0, max=100.0),
            )
        )
        assert no_row(response, "push")
        (warning,) = refusal(response, "CONSTRAINT_NOT_CONVERTIBLE")
        assert warning.detail["reason"] == "change_rel_base_zero"

    def test_no_raw_range_is_refused_by_name_never_guessed(self):
        response = run(build(value=1 / 7, frame="change_rel"))
        assert no_row(response, "push")
        (warning,) = refusal(response, "CONSTRAINT_NOT_CONVERTIBLE")
        assert warning.detail["reason"] == "change_rel_raw_range_missing"


# ---------------------------------------------------------------------------------------------------------
# Whose base (AIQ 5871459631, wire 5872798858): only change_rel reads the base's OWNER.
# ---------------------------------------------------------------------------------------------------------


class TestFrameVerdict:
    RAW = RawRange(min=0.0, max=100.0)

    @pytest.mark.parametrize(
        "source, verdict",
        [
            ("brief_extraction", "scored"),
            ("user_confirmed", "scored"),
            ("cee_inference", "estimate_only"),
            ("user_assumption", "estimate_only"),
            (None, "estimate_only"),
        ],
    )
    def test_change_rel_is_scored_only_on_the_users_base(self, source, verdict):
        observed = ObservedState(value=0.7, baseline=0.7, source=source)
        response = run(build(value=1 / 7, frame="change_rel", target_observed=observed, target_raw_range=self.RAW))
        assert constraint_row(response, "push").frame_verdict == verdict

    def test_a_stray_owner_field_never_scores_olumis_base(self):
        observed = ObservedState(value=0.7, baseline=0.7, source="cee_inference", baseline_owner="user")
        response = run(build(value=1 / 7, frame="change_rel", target_observed=observed, target_raw_range=self.RAW))
        assert constraint_row(response, "push").frame_verdict == "estimate_only"

    @pytest.mark.parametrize("frame, value", [("level", 0.8), ("change_abs", 0.1), ("delta", 0.4)])
    def test_every_other_frame_is_scored_whoever_owns_the_base(self, frame, value):
        observed = ObservedState(value=0.7, baseline=0.7, source="cee_inference")
        response = run(build(value=value, frame=frame, target_observed=observed))
        assert constraint_row(response, "push").frame_verdict == "scored"


# ---------------------------------------------------------------------------------------------------------
# Channel A (the goal) runs the same rules.
# ---------------------------------------------------------------------------------------------------------


class TestTheGoalChannel:
    def test_a_change_abs_goal_equals_its_level_twin(self):
        observed = ObservedState(value=0.7, baseline=0.7)
        change = run(
            build(goal_threshold=0.1, goal_threshold_frame="change_abs", goal_observed=observed, constraint_on_target=False)
        )
        level = run(
            build(goal_threshold=0.8, goal_threshold_frame="level", goal_observed=observed, constraint_on_target=False)
        )
        assert result_of(change, "push").probability_of_goal == result_of(level, "push").probability_of_goal
        assert result_of(change, "push").probability_of_goal == pytest.approx(0.8, abs=TOL)

    def test_a_change_abs_goal_with_no_base_is_the_paired_change(self):
        response = run(
            build(goal_threshold=0.1, goal_threshold_frame="change_abs", goal_observed=None, constraint_on_target=False)
        )
        assert result_of(response, "push").probability_of_goal == pytest.approx(0.8, abs=TOL)

    def test_a_change_rel_goal_with_no_base_is_refused_by_name(self):
        response = run(
            build(goal_threshold=0.1, goal_threshold_frame="change_rel", goal_observed=None, constraint_on_target=False)
        )
        assert result_of(response, "push").probability_of_goal is None
        (warning,) = refusal(response, "GOAL_BASE_MISSING")
        assert warning.detail["goal_node_id"] == "g"


# ---------------------------------------------------------------------------------------------------------
# Admission prices the status-quo phase for every frame that can need it.
# ---------------------------------------------------------------------------------------------------------


class TestAdmissionPricing:
    @pytest.mark.parametrize("frame", ["change_abs", "change_rel"])
    def test_a_change_goal_is_priced_like_a_level_goal(self, frame):
        from src.services.robustness_analyzer_v2 import compute_weighted_cost

        level = compute_weighted_cost(build(goal_threshold=0.8, goal_threshold_frame="level", constraint_on_target=False))
        change = compute_weighted_cost(build(goal_threshold=0.1, goal_threshold_frame=frame, constraint_on_target=False))
        assert "status_quo" in level.terms
        assert change.terms.get("status_quo") == level.terms["status_quo"]


# ---------------------------------------------------------------------------------------------------------
# The route, bound by identity: a base-known change_abs IS the level plan (AIQ Q-A "no new paired-delta
# arithmetic"), so the clip and every level refusal apply to it unchanged. Without a binding clip the
# paired-Δ arithmetic gives the same P, so the P rows above cannot tell the two routes apart; this can.
# ---------------------------------------------------------------------------------------------------------


class TestTheRoute:
    def plan_for(self, request):
        plans, warnings = RobustnessAnalyzerV2._resolve_scored_constraint_plans(request)
        assert warnings == [], [w.detail for w in warnings]
        return plans[0]

    def test_a_base_known_change_abs_is_the_level_plan_at_b_plus_c(self):
        plan = self.plan_for(build(value=0.1, frame="change_abs"))
        assert plan.change_frame is False
        assert plan.goal_baseline == 0.7
        assert plan.level_threshold == pytest.approx(0.8)
        assert (plan.stated_origin, plan.stated_scale) == (0.7, 1.0)

    def test_no_base_is_the_paired_change(self):
        plan = self.plan_for(build(value=0.1, frame="change_abs", target_observed=None))
        assert plan.change_frame is True
        assert plan.goal_baseline == 0.0
        assert plan.level_threshold == 0.1

    def test_change_rel_carries_its_raw_scale(self):
        plan = self.plan_for(build(value=1 / 9, frame="change_rel", target_raw_range=RawRange(min=20.0, max=120.0)))
        # b_raw = 0.7 · 100 + 20 = 90 → scale = span / b_raw = 100 / 90.
        assert plan.stated_scale == pytest.approx(100 / 90)
        assert plan.level_threshold == pytest.approx(0.7 + 0.1)


class TestAChangeIsNeverClippedAsALevel:
    def test_the_goals_level_domain_never_clips_a_paired_change(self):
        """The goal is a percent node (cap 100) and a level limit on it states its domain [0, 1], so
        the goal's LEVEL is clipped there. A paired change is not a level: CUT gives Δg = −0.5 f, and
        "cut it by at least 0.1" (minimise, change_abs −0.1) is met when f >= 0.2 → 0.8.
        Mutant (the change clipped to [0, 1]): Δ becomes 0 → never <= −0.1 → 0.0 → RED."""
        from src.models.robustness_v2 import LevelDomain

        base = build(options={"cut": CUT}, constraint_on_target=False)
        goal = NodeV2(
            id="g",
            kind="outcome",
            label="Goal",
            observed_state=ObservedState(value=0.2, cap=100.0, source="brief_extraction"),
        )
        nodes = [goal if n.id == "g" else n for n in base.graph.nodes]
        limit = GoalConstraint(
            constraint_id="cid-goal-level",
            node_id="g",
            operator="<=",
            value=0.5,
            value_frame="level",
            level_domain=LevelDomain(min=0.0, max=1.0),
            label="Goal level",
        )
        request = base.model_copy(
            update={
                "graph": GraphV2(nodes=nodes, edges=base.graph.edges),
                "goal_threshold": -0.1,
                "goal_threshold_frame": "change_abs",
                "goal_direction": "minimise",
                "goal_constraints": [limit],
            }
        )
        analyzer = RobustnessAnalyzerV2()
        parents: Dict[str, List[str]] = {}
        for e in request.graph.edges:
            parents.setdefault(e.to, []).append(e.from_)
        (frame,) = [
            f
            for f in analyzer._node_level_frames(request, parents, {"f"})
            if f.node_id == "g"
        ]
        # Fixture control: the goal IS anchored with a domain, so its level is clipped and the
        # mutant below has something to act on.
        assert frame.frame == "anchored_level" and frame.level_domain_min == 0.0, frame
        response = run(request)
        assert result_of(response, "cut").probability_of_goal == pytest.approx(0.8, abs=TOL)
