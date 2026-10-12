"""Batch C ISL re-scope I-2 / I-4(c): ONE comparison per target.

A target asked both as the single goal and as a one-limit joint (same node, same frame, same
threshold) is decided by ``meets_threshold`` on the SAME compared draws and threshold, built from
the one plan. So for every option, and in particular for "hold", which sits exactly on the tie on
every draw (its paired effect is 0):

    probability_of_goal == prob_satisfied == joint_probability

swept over the level, change_abs and change_rel frames, ties at several magnitudes, targets k
representable steps either side of the tie (k = 0..32) and genuine offsets, all four operators.
The bug this pins: a change-stated limit on a LEVEL node was compared in its stated frame,
``(level − today) × scale`` against the stated value, while the goal compared the level against
the converted level — two arithmetic routes that agree in exact arithmetic, not near a tie.

Where the compared threshold is known without the engine's conversion arithmetic, the held
option's verdict is also checked against the tie rule's exact rational statement (AIE
6115686431): a tie iff |x − T| <= 8 · eps · max(|x|, |T|); met by >=/<=, not by >/<. And (I-3)
no failure margin is ever negative, and a held tie that fails a strict operator has margin 0.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from src.models.robustness_v2 import ObservedState
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from tests.unit.test_recovery_joint_goal import OPERATORS, STRICT, constraint
from tests.unit.test_recovery_numerical_check import linear_request

EIGHT_EPS = 8 * Fraction(float(np.finfo(float).eps))
K_STEPS = range(33)
GENUINE = (1e-12, -1e-12, 1e-9, -1e-9)


def steps(x, k, direction):
    for _ in range(k):
        x = float(np.nextafter(x, direction))
    return x


def k_steps_either_side(x):
    """x itself, then k = 1..32 representable steps above and below it."""
    out = [x]
    for k in K_STEPS[1:]:
        out += [steps(x, k, math.inf), steps(x, k, -math.inf)]
    return out


def exact_verdict(op, x, target):
    """The tie rule's exact statement: the held option's chance (0.0/1.0) at compared value x."""
    X, T = Fraction(x), Fraction(target)
    if abs(X - T) <= EIGHT_EPS * max(abs(X), abs(T)):
        return 0.0 if op in STRICT else 1.0
    met = X > T if op in (">", ">=") else X < T
    return 1.0 if met else 0.0


# (id, frame, node measures, today's normalised level b, raw unit scale). "change node" is a
# quantity_frame 'change' node (today = 0, no base); "level node" carries the user's base b.
CASES = [
    ("level-b0.5", "level", "level", 0.5, 1.0),
    ("level-b-0.3", "level", "level", -0.3, 1.0),
    ("level-b1.25", "level", "level", 1.25, 1.0),
    ("level-b1e-6", "level", "level", 1e-6, 1.0),
    ("change_abs-change-node", "change_abs", "change", None, 1.0),
    ("change_abs-level-b0.5", "change_abs", "level", 0.5, 1.0),
    ("change_abs-level-b-0.3", "change_abs", "level", -0.3, 1.0),
    ("change_abs-level-b1.25", "change_abs", "level", 1.25, 1.0),
    ("change_rel-b0.5-x1", "change_rel", "level", 0.5, 1.0),
    ("change_rel-b0.3-x25000", "change_rel", "level", 0.3, 25_000.0),
    ("change_rel-b1.25-x0.01", "change_rel", "level", 1.25, 0.01),
]


def request_for(frame, measures, b, raw_scale, op, threshold):
    request = linear_request("change_abs" if measures == "change" else "level", raw_unit_scale=raw_scale)
    request.goal_threshold_frame = frame
    if measures == "level":
        goal = next(node for node in request.graph.nodes if node.id == "g")
        goal.observed_state = ObservedState(value=b, baseline=b, source="user_confirmed")
    request.n_samples = 100
    request.goal_threshold = threshold
    request.goal_direction = "minimise" if op in ("<", "<=") else "maximise"
    request.goal_threshold_strict = op in STRICT
    request.goal_constraints = [constraint("g", op, threshold, frame)]
    return request


def targets_for(frame, measures, b, raw_scale):
    """(stated target, the compared threshold when it is known without the engine's conversion)."""
    if frame == "level":
        return [(t, t) for t in k_steps_either_side(b)] + [(b + g, b + g) for g in GENUINE]
    if measures == "change":
        # Compared as the paired change against c itself (today = 0).
        return [(c, c) for c in k_steps_either_side(0.0)] + [(g, g) for g in GENUINE]
    near_level = k_steps_either_side(b)
    if frame == "change_abs":
        # c = (level k steps from b) − b, exact (Sterbenz), so the level b + c is that step exactly;
        # plus the subnormal steps of c at zero, whose level rounds to b + c in binary64.
        return (
            [(level - b, level) for level in near_level]
            + [(c, b + c) for c in k_steps_either_side(0.0)[1:]]
            + [(g, b + g) for g in GENUINE]
        )
    # change_rel: r of today's RAW level. The converted level has two roundings of the engine's
    # own conversion, so only agreement is asserted for these (no exact compared threshold).
    span = 100.0 * raw_scale
    base_raw = b * span + 20.0 * raw_scale
    return (
        [((level - b) * span / base_raw, None) for level in near_level]
        + [(r, None) for r in k_steps_either_side(0.0)[1:]]
        + [(g, None) for g in GENUINE]
    )


@pytest.mark.parametrize("op", OPERATORS)
@pytest.mark.parametrize(
    "frame,measures,b,raw_scale", [case[1:] for case in CASES], ids=[case[0] for case in CASES]
)
def test_the_goal_and_its_constraint_twin_agree_on_every_draw(frame, measures, b, raw_scale, op):
    disagreements, wrong, margins = [], [], []
    checked = 0
    for stated, compared_threshold in targets_for(frame, measures, b, raw_scale):
        request = request_for(frame, measures, b, raw_scale, op, stated)
        sent_id = request.goal_constraints[0].constraint_id
        for option in RobustnessAnalyzerV2().analyze(request).results:
            ca = option.constraint_analysis
            assert option.probability_of_goal is not None and ca is not None, (option.option_id, stated)
            (row,) = [c for c in ca.constraints if c.constraint_id == sent_id]
            triple = (option.probability_of_goal, row.prob_satisfied, ca.joint_probability)
            if len(set(triple)) != 1:
                disagreements.append((option.option_id, stated, triple))
            margin = row.failure_margin_median
            if margin is not None and not margin >= 0.0:
                margins.append((option.option_id, stated, margin))
            if option.option_id == "hold":
                checked += 1
                held = 0.0 if measures == "change" else b
                if compared_threshold is not None:
                    expected = exact_verdict(op, held, compared_threshold)
                    if option.probability_of_goal != expected:
                        wrong.append((stated, compared_threshold, option.probability_of_goal, expected))
                    tie = exact_verdict(">=", held, compared_threshold) == exact_verdict("<=", held, compared_threshold) == 1.0
                    if tie and op in STRICT and margin != 0.0:
                        margins.append(("hold", stated, margin))
    assert checked >= 65  # an empty sweep proves nothing
    assert not disagreements, f"{len(disagreements)} goal/marginal/joint disagreements: {disagreements[:8]}"
    assert not wrong, f"{len(wrong)} held verdicts off the exact tie rule: {wrong[:8]}"
    assert not margins, f"{len(margins)} negative margins, or nonzero margins on a strict tie: {margins[:8]}"
