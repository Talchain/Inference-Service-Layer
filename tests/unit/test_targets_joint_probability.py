"""Batch I: the TARGET-ONLY joint, a separately named figure beside joint_probability.

AIE #94 6130197656 item 1 / DL #94 6132232654. Rows marked role='goal_target' are the goal's
own targets; rows without a role are limits. targets_joint_probability is P(every target row
holds), counted on the SAME informative mask and satisfaction matrix as joint_probability, with
the same sqrt(p(1-p)/n_informative) precision. Every expected value below is hand-calculated
from the listed draws; none is read back from the engine.
"""

import math
import sys
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from src.models.response_v2 import ConstraintAnalysisV2
from src.models.robustness_v2 import ConstraintAnalysis, GoalConstraint, NodeV2, ObservedState
from src.services.robustness_analyzer_v2 import GoalThresholdPlan, RobustnessAnalyzerV2
from tests.unit.test_goal_node_propagation import _edge
from tests.unit.test_recovery_numerical_check import TOL, linear_request, phi

# Isolation sentinel: the module under test is THIS tree's, not another checkout's.
assert (
    Path(sys.modules[RobustnessAnalyzerV2.__module__].__file__)
    .resolve()
    .is_relative_to(Path(__file__).resolve().parents[2])
)


def row(node, operator, value, *, target=False, frame="delta"):
    return GoalConstraint(
        constraint_id=f"{node}:{operator}",
        node_id=node,
        operator=operator,
        value=value,
        value_frame=frame,
        **({"role": "goal_target"} if target else {}),
    )


def analyze(values, constraints, *, plans=None, range_series=None, option_id="push"):
    if plans is None:
        plans = {i: GoalThresholdPlan(delta_threshold=c.value) for i, c in enumerate(constraints)}
    return RobustnessAnalyzerV2()._compute_constraint_analysis(
        {option_id: values},
        constraints,
        option_id,
        plans,
        None,
        level_domains=None,
        range_series=range_series,
    )


# ---------------------------------------------------------------------------------------------
# THE NUMERICAL ROW. Ten equiprobable draws; two targets and one limit.
#   T1 conversion  x >  0.5   (strict: draw 2 sits EXACTLY on 0.5 and must NOT count)
#   T2 customers   y >= 20    (non-strict: draw 3 sits exactly on 20 and DOES count)
#   L  budget      z <= 100   (a limit, no role: draw 9 sits exactly on 100 and counts)
#
#   draw   x     y    z    T1 T2 L
#    0    0.60  25    50    1  1  1
#    1    0.70  30   150    1  1  0   <- limit fails, targets hold
#    2    0.50  22    50    0  1  1   <- strict tie: T1 fails
#    3    0.80  20   120    1  1  0   <- non-strict tie: T2 holds; limit fails
#    4    0.40  10    50    0  0  1
#    5    0.30  15    80    0  0  1
#    6    0.90  40    90    1  1  1
#    7    0.55  19    60    1  0  1
#    8    0.20   5   200    0  0  0
#    9    0.65  21   100    1  1  1   <- limit tie: holds
#
#   P(T1) = 6/10 = 0.6    P(T2) = 6/10 = 0.6    P(L) = 7/10 = 0.7
#   targets joint  P(T1 and T2)        = draws {0,1,3,6,9}  = 5/10 = 0.5
#   all-condition  P(T1 and T2 and L)  = draws {0,6,9}      = 3/10 = 0.3   (< 0.5: the limit fails on 1, 3)
#   product of the target marginals    = 0.6 * 0.6          = 0.36         (!= 0.5: the targets are correlated)
#   precision  sqrt(0.5 * 0.5 / 10) = sqrt(0.025)
# ---------------------------------------------------------------------------------------------
X = [0.60, 0.70, 0.50, 0.80, 0.40, 0.30, 0.90, 0.55, 0.20, 0.65]
Y = [25, 30, 22, 20, 10, 15, 40, 19, 5, 21]
Z = [50, 150, 50, 120, 50, 80, 90, 60, 200, 100]
EXPECTED = {
    "targets_joint_probability": 0.5,
    "joint_probability": 0.3,
    "marginals": [0.6, 0.6, 0.7],
}


def numerical_row(scale_x=1.0, scale_y=1.0, scale_z=1.0):
    constraints = [
        row("conversion", ">", 0.5 * scale_x, target=True),
        row("customers", ">=", 20 * scale_y, target=True),
        row("budget", "<=", 100 * scale_z),
    ]
    values = {
        "conversion": [v * scale_x for v in X],
        "customers": [v * scale_y for v in Y],
        "budget": [v * scale_z for v in Z],
    }
    return analyze(values, constraints)


def figures(result):
    return {
        "targets_joint_probability": result["targets_joint_probability"],
        "joint_probability": result["joint_probability"],
        "marginals": [c["prob_satisfied"] for c in result["constraints"]],
    }


def test_numerical_row_hand_calculated():
    result = numerical_row()
    assert figures(result) == EXPECTED
    # Separate from the all-condition joint, and not the product of the marginals.
    assert result["targets_joint_probability"] > result["joint_probability"]
    assert result["targets_joint_probability"] != 0.6 * 0.6
    assert result["targets_joint_probability_precision"] == {
        "n_informative": 10,
        "standard_error": math.sqrt(0.025),
    }
    # The all-condition joint and its precision are exactly what they were.
    assert result["joint_probability_precision"] == {
        "n_informative": 10,
        "standard_error": math.sqrt(0.3 * 0.7 / 10),
    }


@pytest.mark.parametrize(
    "scale_x,scale_y,scale_z",
    [
        (100.0, 1.0, 1.0),  # conversion as a percent
        (1.0, 0.001, 1.0),  # customers in thousands
        (1.0, 1.0, 1000.0),  # budget in units rather than thousands
        (100.0, 0.001, 1000.0),
        (1024.0, 0.5, 0.25),
    ],
)
def test_numerical_row_keeps_its_answer_when_units_rescale(scale_x, scale_y, scale_z):
    assert figures(numerical_row(scale_x, scale_y, scale_z)) == EXPECTED


def test_targets_are_selected_by_role_not_by_position():
    # Same draws, rows reordered so the limit sits BETWEEN the targets.
    constraints = [
        row("customers", ">=", 20, target=True),
        row("budget", "<=", 100),
        row("conversion", ">", 0.5, target=True),
    ]
    result = analyze({"conversion": X, "customers": Y, "budget": Z}, constraints)
    assert result["targets_joint_probability"] == 0.5
    assert result["joint_probability"] == 0.3


# ---------------------------------------------------------------------------------------------
# GATES. An unscored LIMIT does not block the target-only joint; an unscored TARGET does.
# ---------------------------------------------------------------------------------------------
def test_an_unscored_limit_does_not_block_it_and_positions_follow_the_scored_rows():
    # The limit is FIRST and refused (no plan), so the targets sit at scored positions 0 and 1.
    constraints = [
        row("budget", "<=", 100),
        row("conversion", ">", 0.5, target=True),
        row("customers", ">=", 20, target=True),
    ]
    plans = {1: GoalThresholdPlan(delta_threshold=0.5), 2: GoalThresholdPlan(delta_threshold=20)}
    result = analyze({"conversion": X, "customers": Y, "budget": Z}, constraints, plans=plans)
    assert result["joint_probability"] is None  # B5, unchanged
    assert result["targets_joint_probability"] == 0.5
    assert result["targets_joint_probability_precision"]["n_informative"] == 10


def test_an_unscored_target_withholds_it():
    constraints = [
        row("conversion", ">", 0.5, target=True),
        row("customers", ">=", 20, target=True),
        row("budget", "<=", 100),
    ]
    plans = {0: GoalThresholdPlan(delta_threshold=0.5), 2: GoalThresholdPlan(delta_threshold=100)}
    result = analyze({"conversion": X, "customers": Y, "budget": Z}, constraints, plans=plans)
    assert result["targets_joint_probability"] is None
    assert result["targets_joint_probability_precision"] is None
    # The scored rows keep their own figures.
    assert [c["prob_satisfied"] for c in result["constraints"]] == [0.6, 0.7]


def test_one_target_is_not_a_conjunction():
    constraints = [row("conversion", ">", 0.5, target=True), row("budget", "<=", 100)]
    result = analyze({"conversion": X, "budget": Z}, constraints)
    assert result["targets_joint_probability"] is None
    assert result["targets_joint_probability_precision"] is None


def test_a_range_scored_target_withholds_it_and_a_range_scored_limit_does_not():
    # A row is scored from a stated range when its node has a range series and the option is
    # pinned in its plan (TEMPORAL step 1). Four draws, all targets hold on draws 1 and 3.
    values = {"a": [0.5] * 4, "b": [-1, 1, -1, 1], "c": [-1, 1, -1, 1]}
    range_plan = GoalThresholdPlan(
        level_threshold=0.5, goal_baseline=0, pinned_levels=(("push", 0.5),)
    )
    delta = GoalThresholdPlan(delta_threshold=0)
    series = {"a": [0.25, 0.5, 0.75, 1.0]}

    target_ranged = analyze(
        values,
        [
            row("a", ">", 0.5, target=True, frame="level"),
            row("b", ">", 0, target=True),
            row("c", ">", 0),
        ],
        plans={0: range_plan, 1: delta, 2: delta},
        range_series=series,
    )
    assert target_ranged["targets_joint_probability"] is None

    limit_ranged = analyze(
        values,
        [
            row("a", ">", 0.5, frame="level"),
            row("b", ">", 0, target=True),
            row("c", ">", 0, target=True),
        ],
        plans={0: range_plan, 1: delta, 2: delta},
        range_series=series,
    )
    assert limit_ranged["joint_probability"] is None  # TEMPORAL step 1, unchanged
    assert limit_ranged["targets_joint_probability"] == 0.5


def test_partial_coverage_extremum_withholds_only_this_figure():
    # Draw 3 is non-finite on the scored limit, so 3 of 4 draws are informative. On those three
    # the targets never hold together (0/3, a partial-population extremum). A second limit is
    # unscored (no plan), so joint_probability is withheld (B5) and every EMITTED existing
    # figure is mid-range: the block stands, and only the new figure is withheld. (Folding it
    # into the block refusal would have removed the limit checks too.)
    values = {"t1": [1, -1, 1, 1], "t2": [-1, 1, -1, 1], "lim": [1, 1, -1, float("nan")]}
    constraints = [
        row("t1", ">", 0, target=True),
        row("t2", ">", 0, target=True),
        row("lim", ">", 0),
        row("unscored_limit", ">", 0),
    ]
    plans = {i: GoalThresholdPlan(delta_threshold=0) for i in range(3)}
    result = analyze(values, constraints, plans=plans)
    assert result is not None
    assert result["joint_probability"] is None
    assert result["targets_joint_probability"] is None
    assert result["targets_joint_probability_precision"] is None
    assert [c["prob_satisfied"] for c in result["constraints"]] == [2 / 3, 1 / 3, 2 / 3]
    assert [c["prob_satisfied_precision"]["n_informative"] for c in result["constraints"]] == [
        3
    ] * 3


def test_partial_coverage_mid_range_is_emitted_on_the_shared_population():
    # Draw 4 is non-finite on the limit: n_informative = 4 of 5. Targets both hold on draws 0, 3.
    values = {
        "t1": [1, 1, -1, 1, 1],
        "t2": [1, -1, 1, 1, 1],
        "lim": [1, -1, 1, -1, float("inf")],
    }
    constraints = [
        row("t1", ">", 0, target=True),
        row("t2", ">", 0, target=True),
        row("lim", ">", 0),
    ]
    result = analyze(values, constraints)
    assert result["targets_joint_probability"] == 0.5
    assert result["targets_joint_probability_precision"] == {
        "n_informative": 4,
        "standard_error": math.sqrt(0.25 / 4),
    }
    assert result["joint_probability"] == 0.25  # draw 0 only


# ---------------------------------------------------------------------------------------------
# INERTNESS. No role on any row: the new keys are None in the analysis and ABSENT on the wire.
# (The byte-for-byte comparison against the base commit is a separate run, recorded in the
# handback.)
# ---------------------------------------------------------------------------------------------
def test_no_role_leaves_the_block_unchanged():
    constraints = [
        row("conversion", ">", 0.5),
        row("customers", ">=", 20),
        row("budget", "<=", 100),
    ]
    result = analyze({"conversion": X, "customers": Y, "budget": Z}, constraints)
    assert result["targets_joint_probability"] is None
    assert result["targets_joint_probability_precision"] is None
    assert result["joint_probability"] == 0.3


# ---------------------------------------------------------------------------------------------
# THROUGH THE REAL SAMPLER AND THE WIRE. Independent S_g, S_c ~ N(.5, .1); push's paired change
# on each is .5 * S, so level_g ~ N(.75, .05) and change_c ~ N(.25, .05).
#   targets: level_g > .75 (P .5), change_c >= .25 (P .5), independent -> targets joint .25
#   limit:   level_g < .80 (P Phi(1))
#   all-condition joint = (Phi(1) - .5) * .5
# ---------------------------------------------------------------------------------------------
def two_target_request():
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
    request.goal_constraints = [
        GoalConstraint(
            constraint_id="t_g",
            node_id="g",
            operator=">",
            value=0.75,
            value_frame="level",
            role="goal_target",
        ),
        GoalConstraint(
            constraint_id="t_c",
            node_id="c",
            operator=">=",
            value=0.25,
            value_frame="change_abs",
            role="goal_target",
        ),
        GoalConstraint(
            constraint_id="l_g", node_id="g", operator="<", value=0.8, value_frame="level"
        ),
    ]
    return request


def test_real_sampler_closed_form():
    response = RobustnessAnalyzerV2().analyze(two_target_request())
    for option in response.results:
        ca = option.constraint_analysis
        if option.option_id == "push":
            assert ca.targets_joint_probability == pytest.approx(0.25, abs=TOL)
            assert ca.joint_probability == pytest.approx((phi(1) - 0.5) * 0.5, abs=TOL)
        else:
            assert ca.targets_joint_probability == 0.0
        assert ca.targets_joint_probability >= ca.joint_probability
        assert ca.targets_joint_probability_precision.n_informative == (
            ca.joint_probability_precision.n_informative
        )
        # The process-pool JSON round trip keeps the new fields.
        assert ConstraintAnalysis.model_validate_json(ca.model_dump_json()) == ca


def test_wire_carries_the_figure_through_the_api_hop_and_omits_it_without_roles():
    from src.api.main import app

    client = TestClient(app)

    def post(request):
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200, response.text
        return {o["id"]: o["constraint_analysis"] for o in response.json()["options"]}

    request = two_target_request()
    request.n_samples = 2000
    request.options[0].interventions = {
        "f": 0.5
    }  # the API refuses an empty option; 0.5 is today's f
    internal = {
        o.option_id: o.constraint_analysis for o in RobustnessAnalyzerV2().analyze(request).results
    }
    wire = post(request)
    assert set(wire) == {"hold", "push"}
    for option_id, ca in wire.items():
        assert ca["targets_joint_probability"] == internal[option_id].targets_joint_probability
        p, n = (
            ca["targets_joint_probability"],
            ca["targets_joint_probability_precision"]["n_informative"],
        )
        assert n == 2000
        assert ca["targets_joint_probability_precision"]["standard_error"] == pytest.approx(
            math.sqrt(p * (1 - p) / n), abs=1e-14
        )
        assert ConstraintAnalysisV2.model_validate(ca).targets_joint_probability == p

    for constraint in request.goal_constraints:
        constraint.role = None
    for ca in post(request).values():
        assert "targets_joint_probability" not in ca
        assert "targets_joint_probability_precision" not in ca
