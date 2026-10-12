"""Batch C: exact enumeration and closed-form witnesses for joint attainment.

Normal expectations use Phi directly; correlated zero-threshold normals obey
P(X>0,Y>0)=1/4+asin(rho)/(2*pi). No marginal multiplication in the engine.
"""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from fastapi.testclient import TestClient

from src.models.response_v2 import ConstraintAnalysisV2
from src.models.robustness_v2 import (
    ConstraintAnalysis,
    GoalConstraint,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
)
from src.services.robustness_analyzer_v2 import (
    CONSTRAINT_OPERATOR_SENSE,
    GoalThresholdPlan,
    RobustnessAnalyzerV2,
    _clamped_limit_probability,
    _lognormal_limit_probability,
    meets_threshold,
)
from tests.unit.test_goal_node_propagation import _edge
from tests.unit.test_recovery_numerical_check import TOL, linear_request, phi


def constraint(node, operator=">=", value=0.0, frame="delta"):
    return GoalConstraint(
        constraint_id=f"{node}:{operator}:{frame}",
        node_id=node,
        operator=operator,
        value=value,
        value_frame=frame,
    )


def analyze_draws(values, constraints, *, plans=None, reference=None, **kwargs):
    if plans is None:
        plans = {i: GoalThresholdPlan(delta_threshold=c.value) for i, c in enumerate(constraints)}
    return RobustnessAnalyzerV2()._compute_constraint_analysis(
        {"push": values}, constraints, "push", plans, reference, **kwargs
    )


def assert_precision(analysis, n):
    pairs = [
        (row["prob_satisfied"], row["prob_satisfied_precision"]) for row in analysis["constraints"]
    ]
    if analysis["joint_probability"] is not None:
        pairs.append((analysis["joint_probability"], analysis["joint_probability_precision"]))
    for p, precision in pairs:
        assert precision["n_informative"] == n
        assert precision["standard_error"] == pytest.approx(math.sqrt(p * (1 - p) / n), abs=1e-14)


@pytest.mark.parametrize("op,expected", [(">", 0.25), (">=", 0.75), ("<", 0.25), ("<=", 0.75)])
def test_discrete_ties_follow_the_operator(op, expected):
    # Exactly four equiprobable states: -1, 0, 0, 1. Second condition always holds.
    result = analyze_draws(
        {"x": [-1, 0, 0, 1], "y": [1] * 4}, [constraint("x", op), constraint("y")]
    )
    assert result["joint_probability"] == expected
    assert result["constraints"][0]["prob_satisfied"] == expected
    assert result["constraints"][0]["failure_margin_median"] >= 0
    assert_precision(result, 4)


# AIE 6115686431 (C numerical decision): ONE helper decides every single-goal, marginal and joint
# draw. A draw within eight binary64 epsilons of its threshold, RELATIVE to the compared operands,
# 8 * eps * max(|value|, |threshold|), is a numerical tie: met by >= and <=, not by > and <. Beyond
# that budget the literal sign decides. The budget is an operational allowance for the
# paired/conversion arithmetic, not a measurement resolution; no semantic tolerance or unit floor
# applies, so at zero only zero is a tie (independent delta check r2: a spacing count, including
# the subnormal steps it admitted at zero, changed verdicts under an exact change of units).
OPERATORS = (">", ">=", "<", "<=")
STRICT = (">", "<")
TIE_CASES = ("tie", "step_up", "step_down")
SIGNED_THRESHOLDS = [0.0, 0.3, -0.3, 2.5e6, -2.5e6]


def up(x):
    return float(np.nextafter(x, math.inf))


def down(x):
    return float(np.nextafter(x, -math.inf))


def held_verdict(op, case, compared_at_zero=False):
    """The chance for a draw that sits on the tie, against a target placed by `case`.

    'above'/'below' place the target a genuine offset above/below the draw, not a rounding error.
    `compared_at_zero`: the draw and target are COMPARED at zero (a change node: the paired change
    0 against the stated change). There the one-step targets are subnormal and outside the
    relative budget, so they take their literal side: step_up is above the draw, step_down below
    it. A change stated on a LEVEL node (change_rel here) is compared as the level b + change
    (ISL re-scope I-2), where a subnormal change rounds into b: an exact tie again.
    """
    if compared_at_zero and case in ("step_up", "step_down"):
        case = "above" if case == "step_up" else "below"
    if case in TIE_CASES:
        return 0.0 if op in STRICT else 1.0
    draw_is_past = (case == "below") == (op in (">", ">="))
    return 1.0 if draw_is_past else 0.0


# Contrast 1: an exact tie, all four operators, at zero, negative and large thresholds.
@pytest.mark.parametrize("threshold", SIGNED_THRESHOLDS)
@pytest.mark.parametrize("op", OPERATORS)
def test_an_exact_tie_meets_only_the_non_strict_operators(op, threshold):
    minimise, strict = CONSTRAINT_OPERATOR_SENSE[op]
    met = meets_threshold(np.array([threshold]), threshold, minimise=minimise, strict=strict)
    assert met.tolist() == [op not in STRICT]


# Contrast 2: one representable (nextafter) step either side of a nonzero threshold is a numerical
# tie. At zero that step is the smallest subnormal, outside the relative budget: see Contrast 2b.
@pytest.mark.parametrize("threshold", [t for t in SIGNED_THRESHOLDS if t != 0.0])
@pytest.mark.parametrize("op,expected", [(">", 0.2), (">=", 0.8), ("<", 0.2), ("<=", 0.8)])
def test_one_representable_step_either_side_is_a_numerical_tie(op, expected, threshold):
    # Five equiprobable draws: one clearly past each side, three numerical ties.
    draws = [threshold - 1, down(threshold), threshold, up(threshold), threshold + 1]
    result = analyze_draws({"x": draws}, [constraint("x", op, threshold)])
    assert result["constraints"][0]["prob_satisfied"] == pytest.approx(expected, abs=1e-15)
    assert result["joint_probability"] == pytest.approx(expected, abs=1e-15)


# Contrast 2b: at a threshold of exactly zero only zero is a tie; one subnormal step takes its
# literal side (a fixed subnormal allowance cannot survive an exact change of units).
@pytest.mark.parametrize("op,expected", [(">", 0.4), (">=", 0.6), ("<", 0.4), ("<=", 0.6)])
def test_at_zero_only_zero_is_a_numerical_tie(op, expected):
    draws = [-1.0, down(0.0), 0.0, up(0.0), 1.0]
    result = analyze_draws({"x": draws}, [constraint("x", op, 0.0)])
    assert result["constraints"][0]["prob_satisfied"] == pytest.approx(expected, abs=1e-15)
    assert result["joint_probability"] == pytest.approx(expected, abs=1e-15)


# Contrast 3: genuine offsets lie outside the budget, so the literal sign decides. 1e-12 at 0.3
# and 2.5e-6 at 2,500,000 (both inside the superseded 1e-9 * max(1, |t|) band), mirrored below
# zero, and small ordinary values at zero: none of them is a tie.
@pytest.mark.parametrize(
    "threshold,offset",
    [(0.3, 1e-12), (-0.3, 1e-12), (2.5e6, 2.5e-6), (-2.5e6, 2.5e-6), (0.0, 1e-12), (0.0, 1e-300)],
)
@pytest.mark.parametrize("op,expected", [(">", 0.4), (">=", 0.6), ("<", 0.4), ("<=", 0.6)])
def test_genuine_offsets_near_the_threshold_take_their_literal_side(op, expected, threshold, offset):
    # Five equiprobable draws: one tie, two genuinely on each side.
    draws = [threshold - 1, threshold - offset, threshold, threshold + offset, threshold + 1]
    result = analyze_draws({"x": draws}, [constraint("x", op, threshold)])
    assert result["constraints"][0]["prob_satisfied"] == pytest.approx(expected, abs=1e-15)
    assert result["joint_probability"] == pytest.approx(expected, abs=1e-15)


# Contrast 5 (control): a non-finite draw is never a numerical tie, and the informative
# finite population (the precision denominator) is unchanged by the tie budget.
@pytest.mark.parametrize("op,expected", [(">", 0.25), (">=", 0.75), ("<", 0.25), ("<=", 0.75)])
def test_non_finite_draws_stay_outside_the_population_beside_numerical_ties(op, expected):
    t = 0.3
    draws = [t - 1, down(t), up(t), t + 1, float("nan"), float("inf"), float("-inf")]
    result = analyze_draws({"x": draws}, [constraint("x", op, t)])
    assert result["constraints"][0]["prob_satisfied"] == pytest.approx(expected, abs=1e-15)
    assert result["joint_probability"] == pytest.approx(expected, abs=1e-15)
    assert_precision(result, 4)



# Independent delta check (12 Oct) BLOCKING: np.spacing(float64 max) is inf, so the tie budget became
# unbounded and the largest finite draws read as ties — ">=" met at -max, ">" missed at +max, with a
# negative failure margin. The budget stays finite at every finite magnitude; these take their literal side.
BIG = float(np.finfo(float).max)


@pytest.mark.parametrize(
    "op,draw,expected",
    [
        (">=", -BIG, 0.0), (">", -BIG, 0.0), (">", BIG, 1.0), (">=", BIG, 1.0),
        ("<=", BIG, 0.0), ("<", BIG, 0.0), ("<", -BIG, 1.0), ("<=", -BIG, 1.0),
    ],
)
def test_the_largest_finite_draws_take_their_literal_side(op, draw, expected):
    t = 0.3
    minimise, strict = CONSTRAINT_OPERATOR_SENSE[op]
    assert meets_threshold(np.array([draw]), t, minimise=minimise, strict=strict).tolist() == [expected == 1.0]
    result = analyze_draws({"x": [draw]}, [constraint("x", op, t)])
    row = result["constraints"][0]
    assert row["prob_satisfied"] == expected
    assert result["joint_probability"] == expected
    # A failing draw's margin is its distance on the failing side: never negative.
    assert row["failure_margin_median"] is None or row["failure_margin_median"] >= 0


# The same budget at a threshold of float64 max: an exact tie and one step below are ties (met only by
# the non-strict operator); a far draw is literal.
@pytest.mark.parametrize(
    "op,draw,expected",
    [(">=", BIG, True), (">", BIG, False), (">=", None, True), (">", None, False), (">=", 0.3, False), (">", 0.3, False)],
)
def test_a_threshold_at_the_largest_finite_value_keeps_a_finite_tie_budget(op, draw, expected):
    value = down(BIG) if draw is None else draw
    minimise, strict = CONSTRAINT_OPERATOR_SENSE[op]
    assert meets_threshold(np.array([value]), BIG, minimise=minimise, strict=strict).tolist() == [expected]



# Independent delta check r2 (12 Oct) BLOCKING: counting spacings is unit-dependent — ulp(10t)/ulp(t) is not
# 10 — so draw 1.0000000000000018 against 1 with ">" gave 0 while the same pair times ten (both products exact)
# gave 1. The budget is eight binary64 epsilons RELATIVE to the compared operands, so an exact rescaling of
# both sides can never move a draw across it.
def exact_scale(x, c):
    """x * c when that product is exactly representable, else None."""
    from fractions import Fraction
    y = x * c
    return y if math.isfinite(y) and Fraction(y) == Fraction(x) * Fraction(c) else None


def test_the_checkers_witness_keeps_its_verdict_when_units_scale_by_ten():
    v, t = 1.0000000000000018, 1.0  # v = 1 + 8 eps: |v - t| = 8 eps <= 8 eps * max(|v|, |t|), a tie
    assert exact_scale(v, 10) is not None and exact_scale(t, 10) is not None
    for op in OPERATORS:
        minimise, strict = CONSTRAINT_OPERATOR_SENSE[op]
        before = meets_threshold(np.array([v]), t, minimise=minimise, strict=strict).tolist()
        after = meets_threshold(np.array([v * 10]), t * 10, minimise=minimise, strict=strict).tolist()
        assert before == after == [op not in STRICT], op
        # The checker's production witness: the joint probability, through the constraint analysis.
        joints = [
            analyze_draws({"x": [draw]}, [constraint("x", op, limit)])["joint_probability"]
            for draw, limit in ((v, t), (v * 10, t * 10))
        ]
        assert joints == [0.0 if op in STRICT else 1.0] * 2, op


@pytest.mark.parametrize("factor", [10.0, 100.0, 1000.0, 0.5, 2.0, 1024.0, 3.0, 1e6])
def test_every_exact_rescaling_keeps_every_verdict(factor):
    checked = 0
    for t in [1.0, 0.3, -0.3, 0.8, 2.5e6, -2.5e6, 7.0, 1e-200, 1e200]:
        candidates = [t]
        for direction in (math.inf, -math.inf):
            v = t
            for _ in range(24):
                v = float(np.nextafter(v, direction))
                candidates.append(v)
        st = exact_scale(t, factor)
        if st is None:
            continue
        for v in candidates:
            sv = exact_scale(v, factor)
            if sv is None:
                continue
            for op in OPERATORS:
                minimise, strict = CONSTRAINT_OPERATOR_SENSE[op]
                before = meets_threshold(np.array([v]), t, minimise=minimise, strict=strict).tolist()
                after = meets_threshold(np.array([sv]), st, minimise=minimise, strict=strict).tolist()
                assert before == after, (t, v, factor, op)
                checked += 1
    assert checked > 0  # an empty sweep proves nothing


# ISL re-scope I-1: the budget is computed EXACTLY. The tie test is the real-number inequality
# |v − t| <= 8 · eps · max(|v|, |t|), checked against exact rationals at every binade from the smallest
# subnormal to float64 max, both signs, k = 0..40 representable steps either side, and far draws. That
# exactness is what makes every exact rescaling verdict-preserving; a spacing count (dbddcd5e) fails it.
EIGHT_EPS = 8 * Fraction(float(np.finfo(float).eps))
SUBNORMAL = 5e-324
NORMAL_MIN = float(np.finfo(float).tiny)


def exact_tie(v, t):
    V, T = Fraction(v), Fraction(t)
    return abs(V - T) <= EIGHT_EPS * max(abs(V), abs(T))


def oracle_thresholds():
    fixed = [0.0, -0.0, SUBNORMAL, -SUBNORMAL, NORMAL_MIN, -NORMAL_MIN, 1e-310, 0.3, -0.3, 0.8, 1.0,
             1.0000000000000018, 7.0, 10.0, 2.5e6, -2.5e6, 1e-200, 1e200, BIG, -BIG, down(BIG)]
    grid = [math.ldexp(m, e) for e in range(-1074, 1024, 61) for m in (0.5, 0.71, -0.999)]
    return fixed + [t for t in grid if t != 0.0 and math.isfinite(t)]


def test_the_tie_budget_is_the_exact_real_number_inequality_at_every_magnitude():
    checked = 0
    for t in oracle_thresholds():
        near = [t]
        for direction in (math.inf, -math.inf):
            v = t
            for _ in range(40):
                v = float(np.nextafter(v, direction))
                near.append(v)
        far = [0.0, SUBNORMAL, -SUBNORMAL, BIG, -BIG, -t, t / 2, t * 2]
        draws = np.array([x for x in near + far if math.isfinite(x)])
        for minimise in (False, True):
            inclusive = meets_threshold(draws, t, minimise=minimise, strict=False).tolist()
            exclusive = meets_threshold(draws, t, minimise=minimise, strict=True).tolist()
            for x, met_inclusive, met_strict in zip(draws.tolist(), inclusive, exclusive, strict=True):
                tie = exact_tie(x, t)
                literal = x < t if minimise else x > t
                assert (met_inclusive, met_strict) == (literal or tie, literal and not tie), (x, t, minimise)
                checked += 1
    assert checked > 10_000  # an empty sweep proves nothing


# r2 NON-BLOCKING / ISL re-scope I-3: a draw that is a numerical tie and fails a strict operator sits AT
# the threshold, on whichever side of it the draw lies: its failure margin is exactly zero (one step on
# the failing side is not a one-ulp shortfall), and no margin is ever negative.
@pytest.mark.parametrize(
    "op,draw", [(">", "up"), (">", "tie"), (">", "down"), ("<", "down"), ("<", "tie"), ("<", "up")]
)
def test_a_tie_that_fails_a_strict_operator_has_a_zero_margin(op, draw):
    t = 0.3
    value = {"up": up(t), "down": down(t), "tie": t}[draw]
    row = analyze_draws({"x": [value]}, [constraint("x", op, t)])["constraints"][0]
    assert row["prob_satisfied"] == 0.0
    assert row["failure_margin_median"] == 0.0


# I-3 through a change frame: a change_rel limit on a LEVEL node is compared as the level (I-2), so
# "hold", at today's level b, ties a target one representable step from b. Its distance in the
# STATED frame (±r) is not a shortfall: the margin is zero on both sides.
@pytest.mark.parametrize("op", STRICT)
@pytest.mark.parametrize("side", [math.inf, -math.inf])
def test_a_change_stated_tie_that_fails_a_strict_operator_has_a_zero_margin(op, side):
    b, span, base_raw = 0.5, 100.0, 70.0  # linear_request("change_rel"): raw range [20, 120], today 70
    r = (float(np.nextafter(b, side)) - b) * span / base_raw
    request = goal_and_constraint_request("change_rel", op, r)
    sent = request.goal_constraints[0].constraint_id
    (hold,) = [option for option in RobustnessAnalyzerV2().analyze(request).results if option.option_id == "hold"]
    (row,) = [c for c in hold.constraint_analysis.constraints if c.constraint_id == sent]
    assert hold.probability_of_goal == row.prob_satisfied == 0.0
    assert row.failure_margin_median == 0.0


def goal_and_constraint_request(frame, op, threshold, **kwargs):
    """One request asking the same target both ways: as the single goal and as a one-limit joint."""
    request = linear_request(frame, **kwargs)
    request.n_samples = 200
    request.goal_threshold = threshold
    request.goal_direction = "minimise" if op in ("<", "<=") else "maximise"
    request.goal_threshold_strict = op in STRICT
    request.goal_constraints = [constraint("g", op, threshold, frame)]
    return request


def chances_asked_both_ways(request):
    chances = {}
    for option in RobustnessAnalyzerV2().analyze(request).results:
        ca = option.constraint_analysis
        assert option.probability_of_goal is not None
        assert ca.constraints[0].prob_satisfied == option.probability_of_goal
        assert ca.joint_probability == option.probability_of_goal
        chances[option.option_id] = option.probability_of_goal
    return chances


# Contrasts 1-3 through both routes. "hold" sits exactly on `tie` in every draw (its paired
# effect is 0). The target is that tie, one nextafter step from it, or a genuine 1e-12 above
# or below it; single goal, marginal and joint give the same verdict.
@pytest.mark.parametrize("frame,tie", [("level", 0.5), ("change_abs", 0.0), ("change_rel", 0.0)])
@pytest.mark.parametrize("case", [*TIE_CASES, "above", "below"])
@pytest.mark.parametrize("op", OPERATORS)
def test_single_goal_and_one_constraint_joint_agree_at_and_near_a_tie(op, case, frame, tie):
    threshold = {
        "tie": tie,
        "step_up": up(tie),
        "step_down": down(tie),
        "above": tie + 1e-12,
        "below": tie - 1e-12,
    }[case]
    chances = chances_asked_both_ways(goal_and_constraint_request(frame, op, threshold))
    assert chances["hold"] == held_verdict(op, case, compared_at_zero=frame == "change_abs")


# Contrast 4: one physical target stated in fraction (raw units x0.01), percent (x1) and scaled
# raw units (x25,000: today's level 1,750,000) reaches the engine through the existing frame
# conversion with the same verdicts and chances. "hold" stays at today's level; the target is
# today (the caller's conversion lands on it exactly at all three scales), one nextafter step
# from it (round-off), or a genuine 1e-9 of today's raw level away.
RAW_SCALES = (0.01, 1.0, 25_000.0)


def stated_target(frame, k, relative_offset):
    """The wire value a caller sends for today's raw level moved by `relative_offset` of it."""
    raw_min, raw_max = 20.0 * k, 120.0 * k
    span = raw_max - raw_min
    raw_today = 70.0 * k
    raw_change = relative_offset * raw_today
    if frame == "level":
        return (raw_today + raw_change - raw_min) / span
    if frame == "change_abs":
        return raw_change / span
    return raw_change / raw_today


@pytest.mark.parametrize("frame", ["level", "change_abs", "change_rel"])
@pytest.mark.parametrize("case", [*TIE_CASES, "above", "below"])
@pytest.mark.parametrize("op", OPERATORS)
def test_fraction_percent_and_scaled_raw_units_give_the_same_verdicts(op, case, frame):
    seen = []
    for k in RAW_SCALES:
        today = stated_target(frame, k, 0.0)
        threshold = {
            "tie": today,
            "step_up": up(today),
            "step_down": down(today),
            "above": stated_target(frame, k, 1e-9),
            "below": stated_target(frame, k, -1e-9),
        }[case]
        request = goal_and_constraint_request(frame, op, threshold, raw_unit_scale=k)
        chances = chances_asked_both_ways(request)
        assert chances["hold"] == held_verdict(op, case, compared_at_zero=frame == "change_abs"), k
        seen.append(chances)
    assert seen[1] == seen[0] and seen[2] == seen[0]


# ISL re-scope I-2: a goal LEVEL that is non-finite on a draw stays non-finite through the domain clip
# (GoalThresholdPlan.compared), so the draw leaves the population on BOTH routes. Before, the goal route's
# np.clip turned +inf into the domain's ceiling and -inf into its floor and COUNTED them, while the
# constraint route excluded them: goal 5/8 or 3/8 beside a marginal and joint of 3/6. "push" is level
# 0.5 + effect on eight draws: three at 0.9, three at 0.1, two at +inf or -inf; domain [0, 1]; target 0.5.
@pytest.mark.parametrize("op", [">=", "<="])
@pytest.mark.parametrize("non_finite", [math.inf, -math.inf])
def test_a_non_finite_goal_level_leaves_the_population_on_both_routes(non_finite, op):
    request = linear_request("level")
    request.goal_threshold = 0.5
    request.goal_direction = "minimise" if op == "<=" else "maximise"
    request.goal_constraints = [constraint("g", op, 0.5, "level")]
    analyser = RobustnessAnalyzerV2()
    goal_plan, warning = analyser._resolve_goal_threshold_in_sample_frame(request)
    assert warning is None and goal_plan is not None and goal_plan.level_threshold == 0.5
    constraint_plans, warnings = analyser._resolve_constraint_plans(request)
    assert not warnings and constraint_plans == {0: goal_plan}  # one target, one plan
    status_quo = [0.0] * 8
    push = [0.4, 0.4, -0.4, -0.4, 0.4, -0.4, non_finite, non_finite]
    results = analyser._compute_option_results(
        outcomes={"hold": status_quo, "push": push},
        wins={"hold": 0, "push": 0},
        request=request,
        goal_threshold_plan=goal_plan,
        constraint_node_values={"hold": {"g": status_quo}, "push": {"g": push}},
        status_quo_outcomes=status_quo,
        constraint_plans=constraint_plans,
        status_quo_node_values={"g": status_quo},
        level_domains={"g": (0.0, 1.0)},
    )
    (option,) = [result for result in results if result.option_id == "push"]
    ca = option.constraint_analysis
    sent = request.goal_constraints[0].constraint_id
    (row,) = [c for c in ca.constraints if c.constraint_id == sent]
    assert option.probability_of_goal == row.prob_satisfied == ca.joint_probability == 3 / 6
    assert option.probability_of_goal_precision.n_informative == 6
    assert row.prob_satisfied_precision.n_informative == 6
    assert ca.joint_probability_precision.n_informative == 6


@pytest.mark.parametrize("rho", [0.0, 0.8, -0.8])
def test_two_normals_closed_form_joint(rho):
    # Independent Z,W; X=Z, Y=rho*Z+sqrt(1-rho^2)*W.
    z, w = np.random.default_rng(42).normal(size=(2, 100_000))
    y = rho * z + math.sqrt(1 - rho * rho) * w
    result = analyze_draws(
        {"x": z.tolist(), "y": y.tolist()}, [constraint("x", ">"), constraint("y", ">")]
    )
    p = [c["prob_satisfied"] for c in result["constraints"]]
    expected = 0.25 + math.asin(rho) / (2 * math.pi)
    assert p == pytest.approx([0.5, 0.5], abs=0.006)
    assert result["joint_probability"] == pytest.approx(expected, abs=0.006)
    if rho == 0:
        assert result["joint_probability"] == pytest.approx(p[0] * p[1], abs=0.006)
    else:
        assert abs(result["joint_probability"] - p[0] * p[1]) > 0.1
    assert result["joint_probability"] <= min(p)
    assert_precision(result, 100_000)


def test_frechet_bounds_for_every_four_draw_boolean_pair():
    # Enumerate every pair of four-draw events, including impossible/certain events.
    for x, y in itertools.product(itertools.product([0, 1], repeat=4), repeat=2):
        result = analyze_draws(
            {"x": x, "y": y}, [constraint("x", ">", 0), constraint("y", ">=", 1)]
        )
        p, q = [c["prob_satisfied"] for c in result["constraints"]]
        assert max(0, p + q - 1) <= result["joint_probability"] <= min(p, q)


def test_precision_uses_shared_finite_population_and_withholds_incomplete_joint():
    constraints = [constraint("x"), constraint("y")]
    result = analyze_draws({"x": [-1, 1, 1, float("nan")], "y": [1, -1, 1, 1]}, constraints)
    assert result["joint_probability"] == pytest.approx(1 / 3)
    assert [c["prob_satisfied"] for c in result["constraints"]] == pytest.approx([2 / 3, 2 / 3])
    assert_precision(result, 3)
    partial = analyze_draws(
        {"x": [-1, 1]}, constraints, plans={0: GoalThresholdPlan(delta_threshold=0)}
    )
    assert partial["joint_probability"] is None
    assert partial["joint_probability_precision"] is None
    assert_precision(partial, 2)
    assert analyze_draws({"x": [float("nan")], "y": [1]}, constraints) is None


@pytest.mark.parametrize("frame,z", [("level", -1), ("change_abs", 1), ("change_rel", 0)])
def test_joint_frames_use_paired_status_quo_and_match_single_goal(frame, z):
    request = linear_request(frame)
    request.goal_constraints = [
        constraint("g", ">=", request.goal_threshold, frame),
        constraint("g", "<=", 1.0, frame),
    ]
    response = RobustnessAnalyzerV2().analyze(request)
    for option in response.results:
        ca = option.constraint_analysis
        expected = phi(z) if option.option_id == "push" else 0.0
        assert ca.joint_probability == pytest.approx(expected, abs=TOL)
        assert ca.constraints[0].prob_satisfied == option.probability_of_goal
        assert ca.joint_probability == ca.constraints[0].prob_satisfied
        assert ca.joint_probability_precision.n_informative == request.n_samples
        # Process-pool JSON roundtrip cannot silently drop the new precision fields.
        assert ConstraintAnalysis.model_validate_json(ca.model_dump_json()) == ca


@pytest.mark.parametrize("frame", ["level", "change_abs", "change_rel"])
def test_joint_unit_invariance(frame):
    results = []
    for scale in [1.0, 10.0]:
        request = linear_request(frame, factor_scale=scale, raw_unit_scale=scale)
        request.goal_constraints = [
            constraint("g", ">", request.goal_threshold, frame),
            constraint("g", "<", 1.0, frame),
        ]
        results.append(RobustnessAnalyzerV2().analyze(request))
    for before, after in zip(results[0].results, results[1].results, strict=True):
        old = before.constraint_analysis.model_dump()
        new = after.constraint_analysis.model_dump()
        for old_row, new_row in zip(old["constraints"], new["constraints"], strict=True):
            old_margin = old_row.pop("failure_margin_median")
            new_margin = new_row.pop("failure_margin_median")
            if old_margin is None:
                assert new_margin is None
            else:
                assert new_margin == pytest.approx(old_margin, abs=1e-14)
        # Probabilities, counts, precision and all other diagnostics stay exact.
        assert new == old


# ISL re-scope I-2: the EVPI joint (p_win_sensitivity's current_metric, P(joint goal) for the
# recommended option) reads the SAME comparison as the wire block. Before, it compared a change_rel
# limit's LEVELS (0.5 + effect) against its STATED relative change (25/70), true on every draw: 1.0
# beside a wire joint of ~0.48. Its own draws (EVPI seeds, at most 2,000) differ from the wire
# block's 10,000, so the two agree within Monte Carlo error (about four standard errors), not exactly.
@pytest.mark.parametrize("frame", ["level", "change_abs", "change_rel"])
def test_the_evpi_joint_reads_the_wire_blocks_comparison(frame):
    request = linear_request(frame)
    request.include_voi = True
    request.parameter_uncertainties = [
        ParameterUncertainty(node_id="f", distribution="uniform", range_min=0.0, range_max=1.0)
    ]
    request.goal_constraints = [constraint("g", ">=", request.goal_threshold, frame)]
    response = RobustnessAnalyzerV2().analyze(request)
    assert response.recommended_option_id == "push"
    (push,) = [option for option in response.results if option.option_id == "push"]
    wire_joint = push.constraint_analysis.joint_probability
    assert 0.2 < wire_joint < 0.8  # a genuinely uncertain limit, not a structural 0 or 1
    (sensitivity,) = [entry for entry in response.p_win_sensitivity if entry["factor_id"] == "f"]
    assert sensitivity["metric_type"] == "p_joint_goal"
    assert sensitivity["current_metric"] == pytest.approx(wire_joint, abs=0.05)


def test_two_independent_normal_targets_through_real_sampler():
    request = linear_request("level")
    request.graph.nodes.append(
        NodeV2(
            id="c",
            kind="outcome",
            label="Customers",
            observed_state=ObservedState(value=0.5, baseline=0.5, source="user_confirmed"),
        )
    )
    request.graph.edges.append(_edge("f", "c", 0.5, 0.1))
    request.goal_threshold = 0.75
    # Independent S_g,S_c ~ N(.5,.1); each paired change is .5*S.
    # Level_g > .75 AND change_c >= .25 each have probability .5; joint=.25.
    request.goal_constraints = [
        constraint("g", ">", 0.75, "level"),
        constraint("c", ">=", 0.25, "change_abs"),
    ]
    response = RobustnessAnalyzerV2().analyze(request)
    for option in response.results:
        ca = option.constraint_analysis
        expected = 0.25 if option.option_id == "push" else 0.0
        assert ca.joint_probability == pytest.approx(expected, abs=TOL)
        if option.option_id == "push":
            assert [c.prob_satisfied for c in ca.constraints] == pytest.approx([0.5, 0.5], abs=TOL)
        assert ca.joint_probability <= min(c.prob_satisfied for c in ca.constraints)


def test_range_joint_withheld_also_omits_precision():
    constraints = [constraint("x", ">", 0.5, "level"), constraint("y")]
    plans = {
        0: GoalThresholdPlan(level_threshold=0.5, goal_baseline=0, pinned_levels=(("push", 0.5),)),
        1: GoalThresholdPlan(delta_threshold=0),
    }
    result = analyze_draws(
        {"x": [0.5] * 4, "y": [-1, 1, -1, 1]},
        constraints,
        plans=plans,
        range_series={"x": [0.25, 0.5, 0.75, 1]},
    )
    assert result["joint_probability"] is None
    assert result["joint_probability_precision"] is None
    assert result["constraints"][0]["prob_satisfied"] == 0.5
    assert_precision(result, 4)


def test_pairing_cancels_draw_specific_background_in_every_constraint():
    # Four equiprobable worlds. Different backgrounds for X and Y cancel per draw.
    # Effects are X=[0,.25,.5,.75], Y=[.75,.5,.25,0]. Both >.2 in two worlds.
    constraints = [constraint("x", ">", 0.7, "level"), constraint("y", ">", 0.7, "level")]
    plans = {i: GoalThresholdPlan(level_threshold=0.7, goal_baseline=0.5) for i in range(2)}
    result = analyze_draws(
        {"x": [4, 8.25, 16.5, 32.75], "y": [32.75, 16.5, 8.25, 4]},
        constraints,
        plans=plans,
        reference={"x": [4, 8, 16, 32], "y": [32, 16, 8, 4]},
    )
    assert result["joint_probability"] == 0.5
    assert [c["prob_satisfied"] for c in result["constraints"]] == [0.75, 0.75]
    assert_precision(result, 4)


@pytest.mark.parametrize(
    "op,threshold,expected", [(">", 1, 0), (">=", 1, 0.5), ("<", 0, 0), ("<=", 0, 0.25)]
)
def test_clipped_draws_preserve_strict_ties(op, threshold, expected):
    c = constraint("x", op, threshold, "level")
    plan = GoalThresholdPlan(level_threshold=threshold, goal_baseline=0)
    result = analyze_draws(
        {"x": [-1, 0.5, 1, 2]},
        [c],
        plans={0: plan},
        reference={"x": [0] * 4},
        level_domains={"x": (0, 1)},
    )
    assert result["joint_probability"] == expected


@pytest.mark.parametrize(
    "op,threshold,expected",
    [
        ("<", 1, 0),
        ("<=", 1, 0.25),
        (">", 1, 0.75),
        (">=", 1, 1),
        ("<", 2, 0.25),
        ("<=", 2, 1),
        (">", 2, 0),
        (">=", 2, 0.75),
        ("<", 3, 1),
        (">", 0, 1),
    ],
)
def test_clamped_analytic_diagnostic_respects_boundary_mass(op, threshold, expected):
    # Raw continuous CDF at either bound is supplied as 1/4 in this exact case.
    raw_probability = 0.25 if op in ("<", "<=") else 0.75
    assert _clamped_limit_probability(raw_probability, op, threshold, 1, 2) == expected


@pytest.mark.parametrize("op,expected", [("<", 0.5), ("<=", 0.5), (">", 0.5), (">=", 0.5)])
def test_continuous_lognormal_diagnostic(op, expected):
    assert _lognormal_limit_probability(2, 0.5, op, 2) == expected
    assert _lognormal_limit_probability(2, 0.5, op, 0) == (0 if op in ("<", "<=") else 1)


def test_wire_contract_keeps_precision_through_response_builder():
    from src.api.main import app

    request = linear_request("level")
    request.n_samples = 200
    request.options[0].interventions = {"f": 0.5}
    request.goal_constraints = [
        constraint("g", ">", 0.8, "level"),
        constraint("g", "<", 0.95, "level"),
    ]
    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2",
        json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
        headers={"X-ISL-Response-Version": "2"},
    )
    assert response.status_code == 200, response.text
    for option in response.json()["options"]:
        ca = option["constraint_analysis"]
        assert [c["operator"] for c in ca["constraints"]] == [">", "<"]
        assert_precision(ca, 200)
        assert ConstraintAnalysisV2.model_validate(ca).joint_probability_precision is not None
    # New readers still accept an old producer's payload with no precision blocks.
    legacy = {
        "constraints": [
            {"node_id": "g", "operator": ">=", "threshold": 0.5, "prob_satisfied": 0.5}
        ],
        "joint_probability": 0.5,
    }
    assert ConstraintAnalysisV2.model_validate(legacy).joint_probability_precision is None
