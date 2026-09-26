"""An option that SETS a limit's target is compared at the level it sets.

THE DEFECT (olumi-programme-docs #70 5844032568, WIRE on PLoT b09c0f2 + ISL 2795a8c).
``_resolve_threshold_in_sample_frame`` refused a ``level`` threshold outright when
ANY option intervened on its target ("pinned samples are not change-from-origin").
``_resolve_constraint_plans`` then omitted the whole ``constraint_analysis`` block
(all-or-nothing, which is right for ``joint_probability``). So one option that
states its own level for the limited quantity — a win-back offer that "cuts churn
to 3%" beside a "churn under 10%" limit, or every option on a limit about a lever
("keep the price under £60") — erased every limit check on the run:
  * L1: churn level limit, no option sets churn   -> computed, 0.842 / 0.842
  * L4: the same, one option also sets churn 5%   -> unavailable, CONSTRAINT_NOT_CONVERTIBLE

THE ARITHMETIC, from the evaluator (``SCMEvaluatorV2.evaluate``): an intervened node
takes ``node_values[T] = x`` on every draw, before any propagation. So under THAT
option the samples are exactly the level it sets — already in the threshold's frame,
the identity a root's samples take — and they are compared untouched
(``GoalThresholdPlan.identity_option_ids``). Every OTHER option keeps the plan the
existing rules give it (root identity, or ``baseline + (option - status quo)``), and
every existing refusal still refuses the whole threshold.

THE WITNESS GRAPH, chosen so every expected probability is computable by hand::

    f  root driver, observed 0.2               f -> c (0.5), f -> g (0.5)
    c  the limit's target, NON-root, observed baseline 0.6
    g  the goal

    options: "winback" sets c := 0.3 · "push" sets f := 1.0 · "hold" sets nothing
    limit:   c <= 0.5, level frame

    winback  identity 0.3                            -> 0.3 <= 0.5 on every draw: 1.0
    push     0.6 + (0.5*1.0 - 0.5*0.2) = 1.0         -> never:                  0.0
    hold     0.6 + (0.1 - 0.1)         = 0.6         -> never:                  0.0

A converted winback (0.6 + (0.3 - 0.1) = 0.8) would score 0.0 — the discriminating row.
"""

from typing import Dict, List, Optional

import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GoalConstraint,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

N_SAMPLES = 2_000
SEED = 42
EXACT = 1e-12
LIMIT = "cid-churn-limit"
LEVER_LIMIT = "cid-lever-limit"


def edge(src: str, dst: str) -> EdgeV2:
    return EdgeV2(
        **{"from": src, "to": dst},
        exists_probability=1.0,
        strength=StrengthDistribution(mean=0.5, std=0.0011),
    )


def build_request(
    *,
    options: Optional[Dict[str, Dict[str, float]]] = None,
    threshold: float = 0.5,
    target_baseline: Optional[float] = 0.6,
    lever_limit: bool = False,
    with_limits: bool = True,
) -> RobustnessRequestV2:
    options = (
        options
        if options is not None
        else {"winback": {"c": 0.3}, "push": {"f": 1.0}, "hold": {}}
    )
    nodes = [
        NodeV2(id="f", kind="factor", label="Driver", observed_state=ObservedState(value=0.2)),
        NodeV2(
            id="c",
            kind="factor",
            label="Churn",
            observed_state=(
                ObservedState(value=target_baseline, baseline=target_baseline, unit="norm")
                if target_baseline is not None
                else None
            ),
        ),
        NodeV2(id="g", kind="outcome", label="Goal"),
    ]
    constraints: List[GoalConstraint] = [
        GoalConstraint(
            constraint_id=LIMIT, node_id="c", operator="<=", value=threshold, value_frame="level"
        )
    ]
    if lever_limit:
        # A limit ON the lever an option sets: f is a root, and "push" sets it.
        constraints.append(
            GoalConstraint(
                constraint_id=LEVER_LIMIT, node_id="f", operator="<=", value=0.9, value_frame="level"
            )
        )
    return RobustnessRequestV2(
        request_id="rm-pinned-option-levels",
        graph=GraphV2(nodes=nodes, edges=[edge("f", "c"), edge("f", "g")]),
        options=[
            InterventionOption(id=oid, label=oid.title(), interventions=iv)
            for oid, iv in options.items()
        ],
        goal_node_id="g",
        n_samples=N_SAMPLES,
        seed=SEED,
        goal_constraints=constraints if with_limits else None,
    )


def analyse(**kwargs):
    return RobustnessAnalyzerV2().analyze(build_request(**kwargs))


def result_for(response, option_id: str):
    matches = [r for r in response.results if r.option_id == option_id]
    assert len(matches) == 1, f"expected exactly one result for '{option_id}'"
    return matches[0]


def row(response, option_id: str, constraint_id: str = LIMIT):
    """Locate the constraint by its ECHOED IDENTITY, never by a value predicate."""
    analysis = result_for(response, option_id).constraint_analysis
    assert analysis is not None, f"{option_id}: constraint_analysis omitted"
    rows = [c for c in analysis.constraints if c.constraint_id == constraint_id]
    assert len(rows) == 1, f"expected one row for {constraint_id}, got {len(rows)}"
    return rows[0]


def refusals(response):
    return [
        w
        for w in (response.inference_warnings or [])
        if w.code in {"CONSTRAINT_NOT_CONVERTIBLE", "CONSTRAINT_FRAME_UNSPECIFIED"}
    ]


class TestSomeOptionsSetTheTarget:
    """The L4 shape on a NON-root target: the served churn case."""

    def test_the_option_that_sets_the_target_is_compared_at_the_level_it_sets(self):
        response = analyse()

        assert refusals(response) == [], "no option's comparison may be refused"
        assert row(response, "winback").prob_satisfied == pytest.approx(1.0, abs=EXACT)

    def test_every_other_option_keeps_the_status_quo_anchored_level(self):
        """push: 0.6 + (0.5 - 0.1) = 1.0; hold: 0.6. Neither meets <= 0.5 — the
        same answer the unpinned rules give them with no pinned sibling."""
        response = analyse()

        assert row(response, "push").prob_satisfied == pytest.approx(0.0, abs=EXACT)
        assert row(response, "hold").prob_satisfied == pytest.approx(0.0, abs=EXACT)

    def test_the_others_answer_is_unchanged_by_a_pinned_sibling(self):
        """CONTRAST: remove the win-back and the others' rows must not move."""
        with_pin = analyse()
        without = analyse(options={"push": {"f": 1.0}, "hold": {}})

        for oid in ("push", "hold"):
            assert row(with_pin, oid).prob_satisfied == row(without, oid).prob_satisfied

    def test_a_threshold_below_the_set_level_is_missed_on_every_draw(self):
        """The mirror row: 0.3 <= 0.25 never holds. A pass that always said 1.0
        for a pinned option would fail here."""
        response = analyse(threshold=0.25)

        assert row(response, "winback").prob_satisfied == pytest.approx(0.0, abs=EXACT)


class TestEveryOptionSetsTheTarget:
    """The L3 shape: no option needs a conversion, so no baseline is needed."""

    def test_scored_without_a_baseline(self):
        response = analyse(
            options={"winback": {"c": 0.3}, "hike": {"c": 0.7}}, target_baseline=None
        )

        assert refusals(response) == []
        assert row(response, "winback").prob_satisfied == pytest.approx(1.0, abs=EXACT)
        assert row(response, "hike").prob_satisfied == pytest.approx(0.0, abs=EXACT)

    def test_contrast_the_same_graph_unpinned_still_refuses_for_want_of_a_baseline(self):
        """CONTRAST CONTROL: the baseline refusal is untouched for options that
        DO need the conversion — the all-pinned exit is what scored the row above."""
        response = analyse(options={"push": {"f": 1.0}, "hold": {}}, target_baseline=None)

        assert [w.detail["reason"] for w in refusals(response)] == ["missing_target_baseline"]
        assert result_for(response, "push").constraint_analysis is None


class TestTheOtherLimitsSurvive:
    def test_a_second_limit_on_a_lever_is_scored_beside_it(self):
        """f <= 0.9: push sets f := 1.0 (identity, misses); the others leave the
        root at 0.2 (meets). Before, EITHER pin omitted BOTH limits."""
        response = analyse(lever_limit=True)

        assert refusals(response) == []
        assert row(response, "push", LEVER_LIMIT).prob_satisfied == pytest.approx(0.0, abs=EXACT)
        assert row(response, "winback", LEVER_LIMIT).prob_satisfied == pytest.approx(1.0, abs=EXACT)
        assert row(response, "hold", LEVER_LIMIT).prob_satisfied == pytest.approx(1.0, abs=EXACT)
        joint = {
            oid: result_for(response, oid).constraint_analysis.joint_probability
            for oid in ("winback", "push", "hold")
        }
        assert joint == pytest.approx({"winback": 1.0, "push": 0.0, "hold": 0.0}, abs=EXACT)


class TestWhatStillRefuses:
    def test_a_set_level_in_raw_units_is_refused_by_the_domain_guard(self):
        response = analyse(options={"winback": {"c": 3.0}, "hold": {}})

        found = refusals(response)
        assert [w.detail["reason"] for w in found] == ["constraint_values_outside_normalised_domain"]
        assert "pinned_level[winback]" in found[0].detail["out_of_domain"]
        assert result_for(response, "winback").constraint_analysis is None


class TestWinProbabilityIsUntouched:
    def test_win_probability_is_identical_with_and_without_the_limits(self):
        """The plan only resolves constraint series; the ranking never reads it."""
        with_limits = analyse(lever_limit=True)
        without = analyse(with_limits=False)

        for oid in ("winback", "push", "hold"):
            assert (
                result_for(with_limits, oid).win_probability
                == result_for(without, oid).win_probability
            )



class TestChannelAKeepsItsRefusal:
    """CONTRAST: the goal channel is NOT changed. Its plan also decides what "wins"
    means for a target objective, so a goal an option sets stays refused there and
    the target ranking stays withheld — win_probability cannot move through here."""

    @staticmethod
    def goal_request(goal_direction=None) -> RobustnessRequestV2:
        return RobustnessRequestV2(
            request_id="rm-pinned-goal-mixed",
            graph=GraphV2(
                nodes=[
                    NodeV2(id="f", kind="factor", label="Driver", observed_state=ObservedState(value=0.2)),
                    NodeV2(
                        id="g",
                        kind="outcome",
                        label="Goal",
                        observed_state=ObservedState(value=0.6, baseline=0.6, unit="norm"),
                    ),
                ],
                edges=[edge("f", "g")],
            ),
            options=[
                InterventionOption(id="set", label="Set", interventions={"g": 0.5}),
                InterventionOption(id="push", label="Push", interventions={"f": 1.0}),
            ],
            goal_node_id="g",
            n_samples=N_SAMPLES,
            seed=SEED,
            goal_threshold=0.75,
            goal_threshold_frame="level",
            **({"goal_direction": goal_direction} if goal_direction else {}),
        )

    def test_a_goal_an_option_sets_is_still_refused_in_the_goal_channel(self):
        response = RobustnessAnalyzerV2().analyze(self.goal_request())

        assert [r.probability_of_goal for r in response.results] == [None, None]
        assert [
            w.detail["reason"]
            for w in response.inference_warnings
            if w.code == "GOAL_THRESHOLD_NOT_CONVERTIBLE"
        ] == ["goal_pinned_by_intervention"]

    def test_the_target_ranking_stays_withheld(self):
        response = RobustnessAnalyzerV2().analyze(self.goal_request("target"))

        assert response.objective_ranking.status == "withheld"
        assert response.objective_ranking.withheld_reason == "target_not_resolvable_in_sample_frame"
