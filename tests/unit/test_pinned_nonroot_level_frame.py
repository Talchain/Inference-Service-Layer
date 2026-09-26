"""An option that SETS a non-root quantity is read in the model's frame, not as a raw number (N6).

THE DEFECT (olumi-programme-docs #70 5844447523, SERVED on PLoT b09c0f2 + ISL 2795a8c).
``SCMEvaluatorV2`` writes ``node_values[T] = x`` for an intervened node. On a ROOT that is right: a root's
samples ARE levels (base = its observed or sampled value). On a NON-root it is not: the node's samples are
its parents' propagated composition plus its own base, so they are not the quantity's level. Writing the
level ``x`` there compares it against siblings measured in that other frame. On Paul's served churn model:
  * "carry on, but churn SET to 4%" (today's level: no change) scored ABOVE carrying on;
  * settings up to about 8% (a RISE) scored above carrying on too.

THE RULE (the frame the level plan already uses, ``_resolve_constraint_series``)::

    level  = status_quo_level + (sample - status_quo_sample)        # samples -> level
    sample = status_quo_sample + (level - status_quo_level)          # level -> samples (this change)

``status_quo_sample`` is the node under NO interventions on the SAME draw (edges and factor draws shared:
common random numbers). So "set T to today's level" IS the status quo on every draw. The effect of setting
T to ``x`` is exactly ``(x - today) x`` T's effect on the goal. And the level plan maps the pinned
option's samples back to exactly ``x``.

THE WITNESS GRAPH, every value by hand::

    f  root driver, observed 0.2         f -> c (0.5),  c -> g (0.5)
    c  NON-root quantity, today's level (observed_state.baseline) 0.6
    g  the goal

    hold  {}          c = 0.5*0.2 = 0.1                      g = 0.05
    keep  {c: 0.6}    c = 0.1 + (0.6 - 0.6) = 0.1            g = 0.05   (raw pin: c = 0.6, g = 0.30)
    cut   {c: 0.3}    c = 0.1 + (0.3 - 0.6) = -0.2           g = -0.10  (raw pin: g = 0.15)
    push  {f: 1.0}    ROOT pin, unchanged: c = 0.5           g = 0.25
"""

from typing import Dict, Optional

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
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2, SCMEvaluatorV2
from src.utils.rng import SeededRNG

N_SAMPLES = 2_000
SEED = 42
EXACT = 1e-12
STRENGTHS = {("f", "c"): 0.5, ("c", "g"): 0.5}


def edge(src: str, dst: str) -> EdgeV2:
    return EdgeV2(
        **{"from": src, "to": dst},
        exists_probability=1.0,
        strength=StrengthDistribution(mean=0.5, std=0.0011),
    )


def graph(c_state: Optional[ObservedState] = ObservedState(value=0.6, baseline=0.6)) -> GraphV2:
    return GraphV2(
        nodes=[
            NodeV2(id="f", kind="factor", label="Driver", observed_state=ObservedState(value=0.2)),
            NodeV2(id="c", kind="factor", label="Churn", observed_state=c_state),
            NodeV2(id="g", kind="outcome", label="Goal"),
        ],
        edges=[edge("f", "c"), edge("c", "g")],
    )


def goal_at(interventions: Dict[str, float], g: Optional[GraphV2] = None) -> float:
    return SCMEvaluatorV2(g or graph()).evaluate(
        edge_strengths=STRENGTHS, interventions=interventions, goal_node="g"
    )


def analyse(options: Dict[str, Dict[str, float]], limit: Optional[float] = None):
    return RobustnessAnalyzerV2().analyze(
        RobustnessRequestV2(
            request_id="n6-pinned-nonroot-level",
            graph=graph(),
            options=[
                InterventionOption(id=oid, label=oid.title(), interventions=iv)
                for oid, iv in options.items()
            ],
            goal_node_id="g",
            n_samples=N_SAMPLES,
            seed=SEED,
            goal_constraints=(
                None
                if limit is None
                else [
                    GoalConstraint(
                        constraint_id="lim-c", node_id="c", operator="<=", value=limit, value_frame="level"
                    )
                ]
            ),
        )
    )


def result_for(response, option_id: str):
    matches = [r for r in response.results if r.option_id == option_id]
    assert len(matches) == 1, f"expected exactly one result for '{option_id}'"
    return matches[0]


class TestTheEvaluatorReadsANonRootLevelInTheModelsFrame:
    """Per draw, exact: the evaluator is the one place every analysis reads an option through."""

    def test_setting_the_quantity_at_todays_level_is_no_change(self):
        assert goal_at({"c": 0.6}) == pytest.approx(goal_at({}), abs=EXACT)
        assert goal_at({}) == pytest.approx(0.05, abs=EXACT)

    def test_setting_it_below_today_moves_the_goal_by_the_change_times_its_effect(self):
        assert goal_at({"c": 0.3}) - goal_at({}) == pytest.approx((0.3 - 0.6) * 0.5, abs=EXACT)

    def test_two_options_that_both_set_it_keep_their_difference(self):
        """CONTROL (true before and after): only the anchor moves, never the gap between two set levels."""
        assert goal_at({"c": 0.3}) - goal_at({"c": 0.45}) == pytest.approx((0.3 - 0.45) * 0.5, abs=EXACT)

    def test_a_root_pin_is_still_its_level(self):
        """CONTROL: a root's samples ARE levels, so a root pin is untouched."""
        assert goal_at({"f": 1.0}) == pytest.approx(0.25, abs=EXACT)

    def test_a_root_pin_ignores_the_roots_own_draw(self):
        """A root with a sampled base (0.35 on this draw, observed 0.2) is still set to exactly 1.0.
        Framing a root like a non-root would give 0.35 + (1.0 - 0.2) = 1.15, and g = 0.2875."""
        pinned = SCMEvaluatorV2(graph()).evaluate(
            edge_strengths=STRENGTHS, interventions={"f": 1.0}, goal_node="g", factor_values={"f": 0.35}
        )
        assert pinned == pytest.approx(0.25, abs=EXACT)

    def test_todays_reading_draws_no_epsilon(self):
        """The status-quo reading is noise-free, like the reference ``_run_monte_carlo`` differences
        against (its evaluator has no epsilon stream). With epsilon on the driver, keeping c at today's
        level still reads today's structural value on this draw: 0.5 * 0.2 = 0.1."""
        noisy = GraphV2(
            nodes=[
                NodeV2(id="f", kind="factor", label="Driver", observed_state=ObservedState(value=0.2), epsilon_std=0.2),
                NodeV2(id="c", kind="factor", label="Churn", observed_state=ObservedState(value=0.6, baseline=0.6)),
                NodeV2(id="g", kind="outcome", label="Goal"),
            ],
            edges=[edge("f", "c"), edge("c", "g")],
        )
        evaluator = SCMEvaluatorV2(noisy, epsilon_rng=SeededRNG(7))
        assert evaluator._in_model_frame(STRENGTHS, {"c": 0.6}, None, None) == {"c": pytest.approx(0.1, abs=EXACT)}

    def test_the_setting_cuts_the_quantity_off_from_what_the_option_does_upstream(self):
        """do(): an option that ALSO moves the driver does not move a quantity it sets. The level is today's
        world plus the stated change, whatever else the option does."""
        assert goal_at({"f": 1.0, "c": 0.3}) == pytest.approx(goal_at({"c": 0.3}), abs=EXACT)

    def test_todays_level_is_the_baseline_when_both_are_stated(self):
        """observed_state.baseline is the level plan's anchor, so it is the evaluator's too."""
        g = graph(ObservedState(value=0.7, baseline=0.6))
        assert goal_at({"c": 0.6}, g) == pytest.approx(goal_at({}, g), abs=EXACT)

    def test_todays_level_falls_back_to_the_observed_value(self):
        """The served churn node carries value only (PLoT sends no baseline for a factor)."""
        g = graph(ObservedState(value=0.6))
        assert goal_at({"c": 0.6}, g) == pytest.approx(goal_at({}, g), abs=EXACT)

    def test_with_no_level_for_today_the_setting_is_written_as_given(self):
        """STATED: nothing to anchor the level to, so the old behaviour stands (and the level plan refuses)."""
        g = graph(None)
        assert goal_at({"c": 0.6}, g) == pytest.approx(0.3, abs=EXACT)


class TestWhatTheUserSees:
    def test_holding_the_quantity_at_todays_level_ties_with_carrying_on(self):
        """The served symptom: "SET to 4%" (no change) beat carrying on. It must TIE (fair split)."""
        response = analyse({"hold": {}, "keep": {"c": 0.6}})

        assert result_for(response, "keep").win_probability == pytest.approx(0.5, abs=EXACT)
        assert result_for(response, "hold").win_probability == pytest.approx(0.5, abs=EXACT)
        assert result_for(response, "keep").outcome_distribution.mean == pytest.approx(
            result_for(response, "hold").outcome_distribution.mean, abs=EXACT
        )

    def test_a_setting_scores_by_its_change_from_today(self):
        response = analyse({"hold": {}, "cut": {"c": 0.3}})

        gap = result_for(response, "cut").outcome_distribution.mean - result_for(response, "hold").outcome_distribution.mean
        assert gap == pytest.approx((0.3 - 0.6) * 0.5, abs=5e-3)
        assert result_for(response, "hold").win_probability == pytest.approx(1.0, abs=EXACT)

    def test_a_limit_on_the_set_quantity_still_compares_the_level_the_option_sets(self):
        """The level plan maps the pinned option's samples back to EXACTLY the level it sets:
        0.6 + ((0.1 + (0.3 - 0.6)) - 0.1) = 0.3. So 0.3 <= 0.5 holds, and 0.3 <= 0.25 never does."""
        passes = analyse({"hold": {}, "cut": {"c": 0.3}}, limit=0.5)
        misses = analyse({"hold": {}, "cut": {"c": 0.3}}, limit=0.25)

        def prob(response, oid):
            rows = [
                c for c in result_for(response, oid).constraint_analysis.constraints if c.constraint_id == "lim-c"
            ]
            assert len(rows) == 1
            return rows[0].prob_satisfied

        assert prob(passes, "cut") == pytest.approx(1.0, abs=EXACT)
        assert prob(misses, "cut") == pytest.approx(0.0, abs=EXACT)
        assert prob(passes, "hold") == pytest.approx(0.0, abs=EXACT)

    def test_a_level_set_exactly_at_the_limit_meets_it_on_every_draw(self):
        """"Cut churn to 0.05" against "churn at most 0.05". Mapped back from the model's frame, the level
        carries float rounding: 0.6 + ((r + (0.05 - 0.6)) - r) lands just above 0.05 on every draw here
        (r ~ 0.1), and the limit would read as missed. The option is compared at the level it states."""
        response = analyse({"hold": {}, "cut": {"c": 0.05}}, limit=0.05)
        rows = [c for c in result_for(response, "cut").constraint_analysis.constraints if c.constraint_id == "lim-c"]
        assert len(rows) == 1
        assert rows[0].prob_satisfied == 1.0
