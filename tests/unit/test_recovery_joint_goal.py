"""Batch C: exact enumeration and closed-form witnesses for joint attainment.

Normal expectations use Phi directly; correlated zero-threshold normals obey
P(X>0,Y>0)=1/4+asin(rho)/(2*pi). No marginal multiplication in the engine.
"""

import itertools
import math

import numpy as np
import pytest

from fastapi.testclient import TestClient

from src.models.response_v2 import ConstraintAnalysisV2
from src.models.robustness_v2 import ConstraintAnalysis, GoalConstraint, NodeV2, ObservedState
from src.services.robustness_analyzer_v2 import (
    GoalThresholdPlan,
    RobustnessAnalyzerV2,
    _clamped_limit_probability,
    _lognormal_limit_probability,
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
def test_discrete_ties_are_compared_exactly(op, expected):
    # Exactly four equiprobable states: -1, 0, 0, 1. Second condition always holds.
    result = analyze_draws(
        {"x": [-1, 0, 0, 1], "y": [1] * 4}, [constraint("x", op), constraint("y")]
    )
    assert result["joint_probability"] == expected
    assert result["constraints"][0]["prob_satisfied"] == expected
    assert result["constraints"][0]["failure_margin_median"] >= 0
    assert_precision(result, 4)


# Follow-up 1 (independent check C): a draw ON the threshold is decided by the single-goal
# tie band (|v - t| <= 1e-9 * max(1, |t|)), so one rounding step cannot flip a constraint's
# chance between 0 and 1. On the band: met by >= and <=, not met by > and <.
@pytest.mark.parametrize("threshold", [0.3, 2.5e6])
@pytest.mark.parametrize("op,expected", [(">", 0.2), (">=", 0.8), ("<", 0.2), ("<=", 0.8)])
def test_constraint_draws_within_rounding_of_the_threshold_are_on_it(op, expected, threshold):
    eps = 1e-12 * max(1.0, threshold)
    # Five equiprobable draws: one clearly past each side, three on the threshold.
    draws = [threshold - 1, threshold - eps, threshold, threshold + eps, threshold + 1]
    result = analyze_draws({"x": draws}, [constraint("x", op, threshold)])
    assert result["constraints"][0]["prob_satisfied"] == pytest.approx(expected, abs=1e-15)
    assert result["joint_probability"] == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("frame,tie", [("level", 0.5), ("change_abs", 0.0), ("change_rel", 0.0)])
@pytest.mark.parametrize("offset", [0.0, 1e-12, -1e-12])
@pytest.mark.parametrize("op", [">", ">=", "<", "<="])
def test_single_goal_and_one_constraint_joint_agree_at_and_near_a_tie(op, offset, frame, tie):
    # Same request, same draws, same target and operator, asked both ways. "hold" sits
    # exactly on `tie` in every draw (its paired effect is 0), so the target is a tie
    # (offset 0) or within 1e-12 of one.
    request = linear_request(frame)
    request.n_samples = 200
    threshold = tie + offset
    request.goal_threshold = threshold
    request.goal_direction = "minimise" if op in ("<", "<=") else "maximise"
    request.goal_threshold_strict = op in (">", "<")
    request.goal_constraints = [constraint("g", op, threshold, frame)]
    response = RobustnessAnalyzerV2().analyze(request)
    on_the_threshold = 0.0 if op in (">", "<") else 1.0
    for option in response.results:
        ca = option.constraint_analysis
        assert option.probability_of_goal is not None
        if option.option_id == "hold":
            assert option.probability_of_goal == on_the_threshold
        assert ca.constraints[0].prob_satisfied == option.probability_of_goal
        assert ca.joint_probability == option.probability_of_goal


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
