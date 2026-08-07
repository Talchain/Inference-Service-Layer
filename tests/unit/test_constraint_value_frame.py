"""ROADMAP 2.798 — Channel B (`goal_constraints`) gets Channel A's fail-closed contract.

THE DEFECT. ISL emits two semantically different goal quantities:

    probability_of_goal        <- goal_threshold      (Channel A)
    constraint_analysis
      .joint_probability       <- goal_constraints    (Channel B)
                                  (PLoT renames this `probability_of_joint_goal`)

Channel A was hardened by ROADMAP 2.258 / 2.286: a threshold must be ATTESTED
(``goal_threshold_frame``), a ``level`` threshold is converted per draw against a
status-quo reference, and anything unconvertible is REFUSED — the field is
omitted and a typed warning names what was missing.

Channel B never received that machinery. ``_check_constraint_satisfied`` did a
bare ``value >= constraint.threshold``, comparing a level (or a count) verbatim
against change-from-origin Monte Carlo samples. No frame, no unit, no refusal,
no warning. ``GoalConstraint`` had no frame member at all and
``model_config = {"extra": "ignore"}``, so a frame sent by a producer died
silently at parse.

WHAT THAT COSTS, ON THE WITNESS GRAPH BELOW. The user asks "will `c` reach 0.8?"
The honest answer under the pushed option is **0.8** — an 80% chance. Channel B
compared 0.8 against samples whose whole support is ~0.5, and answered **0.0**.
Not a rounding error and not a finding: arithmetically forced, threshold-agnostic
and option-agnostic. Two unrelated real decisions returned near-zero for every
joint goal this way (see
PHASE0-EVIDENCE-2026-07-28/goal-fit-near-zero-attribution-2026-08-08.md).

THE WITNESS GRAPH used throughout this module, chosen so every expected
probability is computable BY HAND rather than read back off the code::

    f  root factor, observed_state{value: 0.5}, ParameterUncertainty uniform(0, 1)
    g  goal,   NON-ROOT outcome, parent f, strength s = 0.5
    c  target, NON-ROOT outcome, parent f, strength s = 0.5,
       observed_state{value: 0.7, baseline: B = 0.7}

    sample(c)   = 0 (non-root base) + intercept 0 + f * s   =  0.5 * f
    status_quo  = same draw, no interventions                =  0.5 * f
    option PUSH = do(f := 1.0)                               =  0.5

    level_i     = B + (option_i - status_quo_i) = 0.7 + 0.5 * (1 - f)

    P(level >= 0.8) = P(0.5 * (1 - f) >= 0.1) = P(f <= 0.8) = 0.8

The pre-fix comparison instead asks ``P(0.5 >= 0.8)``, which is 0 for every draw.
So the two answers on this graph are **0.8 (true) vs 0.0 (emitted)** — the widest
possible separation, which is what makes the pins below discriminating rather
than decorative.

Auto-scaled noise is default-OFF (``ENABLE_AUTO_SCALED_NOISE``), so samples here
are purely model-driven and the hand arithmetic is exact up to MC error. With
n_samples = 10000 one standard error on a p ~ 0.8 estimate is ~0.004, so the
0.02 tolerance below is ~5 s.e. — wide enough never to flake, far tighter than
the 0.8-wide gap any frame collision opens.

ASSERTION DISCIPLINE. Every constraint is located by its ``constraint_id`` (an
opaque identity echoed back by ISL), never by a value predicate — a sibling
constraint with the same threshold must not be able to satisfy an assertion
written about this one.
"""

from typing import Optional

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
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

N_SAMPLES = 10_000
SEED = 42
TOL = 0.02

# The driver at its ceiling. A level-framed probability for a do-nothing option is
# degenerate by construction (the effect is identically zero), so every
# level-frame fixture analyses an option that actually does something.
PUSH_DRIVER = {"f": 1.0}

# P(level >= 0.8) = P(f <= 0.8) on the witness graph. Derived in the module
# docstring, not read back off the implementation.
TRUE_LEVEL_PROBABILITY = 0.8

# What the frame-blind comparison emits instead: the option's samples sit at 0.5
# for every draw, so `0.5 >= 0.8` is false 10000 times out of 10000.
FORCED_ZERO = 0.0

CID = "cid-target-c"


def build_request(
    *,
    threshold: float = 0.8,
    value_frame: Optional[str] = None,
    operator: str = ">=",
    target_baseline: Optional[float] = 0.7,
    target_has_observed_state: bool = True,
    target_is_root: bool = False,
    target_intercept: float = 0.0,
    pu_on_target: bool = False,
    intervene_on_target: bool = False,
    parent_epsilon_std: float = 0.0,
    target_epsilon_std: float = 0.0,
    option_interventions=None,
    extra_constraints=None,
    goal_threshold: Optional[float] = None,
    goal_threshold_frame: Optional[str] = None,
    constrain_the_goal_node: bool = False,
    include_voi: bool = False,
):
    """The witness graph. Every knob exists to drive one adversarial fixture."""
    target_id = "g" if constrain_the_goal_node else "c"

    target_observed = (
        ObservedState(value=0.7, baseline=target_baseline, unit="norm")
        if target_has_observed_state
        else None
    )

    nodes = [
        NodeV2(id="f", kind="factor", label="Driver", observed_state=ObservedState(value=0.5)),
        NodeV2(id="g", kind="outcome", label="Goal"),
    ]
    edges = [
        EdgeV2(
            **{"from": "f", "to": "g"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=0.5, std=0.0011),
        )
    ]
    uncertainties = [
        ParameterUncertainty(node_id="f", distribution="uniform", range_min=0.0, range_max=1.0)
    ]

    if constrain_the_goal_node:
        # Parity fixture: one node is both the goal and the constraint target, so
        # Channel A and Channel B are answering the SAME question about the SAME
        # samples and any disagreement is a genuine second dialect.
        nodes[1] = NodeV2(
            id="g",
            kind="outcome",
            label="Goal",
            intercept=target_intercept,
            epsilon_std=target_epsilon_std,
            observed_state=target_observed,
        )
        if target_is_root:
            edges = []
            uncertainties.append(
                ParameterUncertainty(
                    node_id="g", distribution="uniform", range_min=0.0, range_max=1.0
                )
            )
    else:
        nodes.append(
            NodeV2(
                id="c",
                kind="outcome",
                label="Target",
                intercept=target_intercept,
                epsilon_std=target_epsilon_std,
                observed_state=target_observed,
            )
        )
        if not target_is_root:
            edges.append(
                EdgeV2(
                    **{"from": "f", "to": "c"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.0011),
                )
            )
        else:
            # A root target needs SOME variation or it is a degenerate constant.
            uncertainties.append(
                ParameterUncertainty(
                    node_id="c", distribution="uniform", range_min=0.0, range_max=1.0
                )
            )

    if parent_epsilon_std:
        nodes[0] = NodeV2(
            id="f",
            kind="factor",
            label="Driver",
            epsilon_std=parent_epsilon_std,
            observed_state=ObservedState(value=0.5),
        )
    if pu_on_target:
        uncertainties.append(
            ParameterUncertainty(node_id=target_id, distribution="normal", std=0.1)
        )

    interventions = dict(option_interventions or {})
    if intervene_on_target:
        interventions[target_id] = 0.5

    constraint_kwargs = {
        "constraint_id": CID,
        "node_id": target_id,
        "operator": operator,
        "value": threshold,
        "label": "Target level",
    }
    if value_frame is not None:
        constraint_kwargs["value_frame"] = value_frame

    constraints = [GoalConstraint(**constraint_kwargs)]
    constraints.extend(extra_constraints or [])

    return RobustnessRequestV2(
        request_id="rm-2798",
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[InterventionOption(id="push", label="Push", interventions=interventions)],
        goal_node_id="g",
        n_samples=N_SAMPLES,
        seed=SEED,
        goal_threshold=goal_threshold,
        goal_threshold_frame=goal_threshold_frame,
        goal_constraints=constraints,
        parameter_uncertainties=uncertainties or None,
        include_voi=include_voi,
    )


def analyse(**kwargs):
    kwargs.setdefault("option_interventions", PUSH_DRIVER)
    return RobustnessAnalyzerV2().analyze(build_request(**kwargs))


def only_result(response):
    assert len(response.results) == 1, "witness graph declares exactly one option"
    return response.results[0]


def constraint_by_id(analysis, constraint_id):
    """Locate a constraint result by IDENTITY, never by a value predicate.

    A sibling constraint carrying the same threshold, operator or probability must
    not be able to satisfy an assertion written about this one (CLAUDE.md trap 19).
    """
    matches = [c for c in analysis.constraints if c.constraint_id == constraint_id]
    assert len(matches) == 1, (
        f"expected exactly one constraint with constraint_id={constraint_id!r}, "
        f"got {[c.constraint_id for c in analysis.constraints]}"
    )
    return matches[0]


def refusals(response):
    return [w for w in response.inference_warnings if w.code == "CONSTRAINT_NOT_CONVERTIBLE"]


def unstamped_warnings(response):
    return [w for w in response.inference_warnings if w.code == "CONSTRAINT_FRAME_UNSPECIFIED"]


def sole_refusal_reason(response):
    """The single refusal reason, asserting there is exactly one refusal."""
    all_refusals = refusals(response) + unstamped_warnings(response)
    assert len(all_refusals) == 1, (
        "expected exactly one constraint refusal, got "
        f"{[(w.code, w.detail.get('reason')) for w in all_refusals]}"
    )
    return all_refusals[0].detail["reason"]


# =============================================================================
# 0. THE FABRICATION. RED-first on the forced zero.
# =============================================================================


class TestTheForcedZero:
    """A frame-blind comparison must not be able to emit a probability at all."""

    def test_an_unattested_constraint_is_refused_not_scored_zero(self):
        """The defect, reproduced and closed.

        Pre-fix this returns joint_probability == 0.0 while the honest answer is
        0.8. Post-fix there is no number: the block is omitted because ISL cannot
        know which frame an unstamped threshold is in, and a guessed frame is a
        manufactured attestation.
        """
        response = analyse(value_frame=None)
        result = only_result(response)

        assert result.constraint_analysis is None, (
            "an unattested constraint threshold cannot be compared against the "
            "target's samples, so the whole constraint block must be omitted — "
            "got "
            + repr(
                None
                if result.constraint_analysis is None
                else result.constraint_analysis.joint_probability
            )
        )

    def test_the_refusal_names_the_missing_frame_and_the_constraint(self):
        response = analyse(value_frame=None)

        warnings = unstamped_warnings(response)
        assert len(warnings) == 1, [w.code for w in response.inference_warnings]
        warning = warnings[0]

        assert (
            warning.severity == "warning"
        ), "a degradation disclosure must not ride as 'info' — PLoT hides info"
        assert warning.detail["constraint_id"] == CID
        assert warning.detail["node_id"] == "c"
        assert warning.detail["reason"] == "frame_not_stamped"

    def test_the_honest_answer_on_this_graph_is_not_zero(self):
        """Guards the fixture itself: the forced zero must not be the true answer.

        Without this the suite could 'prove' a refusal on a graph where 0.0 was
        simply correct, and every pin above would be vacuous.
        """
        response = analyse(value_frame="level")
        analysis = only_result(response).constraint_analysis

        assert analysis is not None
        assert analysis.joint_probability == pytest.approx(TRUE_LEVEL_PROBABILITY, abs=TOL)
        assert abs(analysis.joint_probability - FORCED_ZERO) > 0.5


# =============================================================================
# 1. THE READER. ISL is the blocking layer for value_frame (schemas 0.38.0).
# =============================================================================


class TestValueFrameIsDeclared:
    """`extra: "ignore"` means an undeclared field dies silently at parse.

    Until ISL declares `value_frame`, a CEE stamp and a PLoT forward are
    guaranteed no-ops. Reader before producer.
    """

    def test_goal_constraint_declares_value_frame(self):
        constraint = GoalConstraint(
            **{"constraint_id": CID, "node_id": "c", "operator": ">=", "value": 0.8},
            value_frame="level",
        )
        assert constraint.value_frame == "level"

    def test_value_frame_survives_a_dict_payload_from_the_wire(self):
        """The producer sends JSON, not kwargs — prove the field is not dropped."""
        constraint = GoalConstraint.model_validate(
            {
                "constraint_id": CID,
                "node_id": "c",
                "operator": "<=",
                "value": 0.2,
                "value_frame": "delta",
            }
        )
        assert constraint.value_frame == "delta"

    def test_value_frame_absent_means_unstamped_not_defaulted(self):
        constraint = GoalConstraint(
            **{"constraint_id": CID, "node_id": "c", "operator": ">=", "value": 0.8}
        )
        assert constraint.value_frame is None, (
            "a defaulted frame is a manufactured attestation — the exact "
            "fabrication class 2.258/2.286 exist to kill"
        )

    def test_an_unknown_dialect_is_rejected_rather_than_ignored(self):
        with pytest.raises(ValidationError):
            GoalConstraint.model_validate(
                {
                    "constraint_id": CID,
                    "node_id": "c",
                    "operator": ">=",
                    "value": 0.8,
                    "value_frame": "normalised",
                }
            )


# =============================================================================
# 2. THE LEVEL CONVERSION — the same arithmetic Channel A already uses.
# =============================================================================


class TestLevelFrameIsConvertedAgainstTheStatusQuo:
    def test_joint_probability_matches_the_hand_computed_level_probability(self):
        response = analyse(value_frame="level")
        analysis = only_result(response).constraint_analysis

        assert analysis is not None
        assert analysis.joint_probability == pytest.approx(TRUE_LEVEL_PROBABILITY, abs=TOL)

    def test_per_constraint_probability_is_the_same_number_for_a_lone_constraint(self):
        response = analyse(value_frame="level")
        analysis = only_result(response).constraint_analysis

        assert analysis is not None
        assert constraint_by_id(analysis, CID).prob_satisfied == pytest.approx(
            analysis.joint_probability, abs=1e-12
        )

    def test_a_less_than_constraint_converts_in_the_same_frame(self):
        """P(level <= 0.8) = 1 - P(level >= 0.8) = 0.2 on this graph."""
        response = analyse(value_frame="level", operator="<=")
        analysis = only_result(response).constraint_analysis

        assert analysis is not None
        assert analysis.joint_probability == pytest.approx(1.0 - TRUE_LEVEL_PROBABILITY, abs=TOL)

    def test_failure_margins_are_reported_in_level_space(self):
        """The margin the user is shown must be a shortfall in their own quantity.

        Pre-fix the margin was ``threshold - raw_sample`` = 0.8 - 0.5 = 0.3 — a
        distance between two quantities that are not the same kind of thing, which
        downstream renders as "you are X short". In level space the failing draws
        are those with f > 0.8, whose levels run from 0.7 to 0.8, so the median
        shortfall is ~0.05 (f ~ U(0.8, 1] => level ~ U(0.7, 0.8), median 0.75).
        """
        response = analyse(value_frame="level")
        analysis = only_result(response).constraint_analysis

        assert analysis is not None
        margin = constraint_by_id(analysis, CID).failure_margin_median
        assert margin is not None
        assert margin == pytest.approx(0.05, abs=0.01), (
            "failure_margin_median must be computed on the resolved LEVEL series, "
            "not on the raw change-frame samples"
        )


class TestDeltaFrameIsCallerAttested:
    """`delta` says the number is already in the samples' own frame. Unchanged."""

    def test_delta_framed_constraint_compares_raw_samples(self):
        # Samples sit at 0.5; a delta threshold of 0.4 is met by every draw.
        response = analyse(value_frame="delta", threshold=0.4)
        analysis = only_result(response).constraint_analysis

        assert analysis is not None
        assert analysis.joint_probability == pytest.approx(1.0, abs=TOL)

    def test_delta_framed_constraint_needs_no_baseline(self):
        response = analyse(value_frame="delta", threshold=0.4, target_has_observed_state=False)
        analysis = only_result(response).constraint_analysis

        assert analysis is not None
        assert analysis.joint_probability == pytest.approx(1.0, abs=TOL)


# =============================================================================
# 3. THE REFUSALS — Channel A's convertibility preconditions, applied to the
#    CONSTRAINT's target node.
# =============================================================================


class TestConvertibilityRefusals:
    """Every path that cannot be PROVED convertible omits the block and says why."""

    @pytest.mark.parametrize(
        "kwargs,expected_reason",
        [
            pytest.param(
                {"target_is_root": True},
                "root_target",
                id="root_target_takes_its_base_from_observed_value",
            ),
            pytest.param(
                {"target_baseline": None},
                "missing_target_baseline",
                id="level_frame_needs_a_baseline_to_convert_against",
            ),
            pytest.param(
                {"target_has_observed_state": False},
                "missing_target_baseline",
                id="no_observed_state_at_all",
            ),
            pytest.param(
                {"intervene_on_target": True},
                "target_pinned_by_intervention",
                id="pinned_samples_are_not_change_from_origin",
            ),
            pytest.param(
                {"threshold": 250000.0},
                "constraint_values_outside_normalised_domain",
                id="raw_user_units_where_normalised_values_were_expected",
            ),
            pytest.param(
                {"parent_epsilon_std": 0.05},
                "epsilon_breaks_status_quo_reference",
                id="epsilon_on_an_ancestor_is_not_crn_matchable",
            ),
            pytest.param(
                {"target_epsilon_std": 0.05},
                "epsilon_breaks_status_quo_reference",
                id="epsilon_on_the_target_itself",
            ),
            pytest.param(
                {"pu_on_target": True},
                "target_parameter_uncertainty_shifts_base",
                id="a_per_draw_base_has_no_single_static_conversion",
            ),
        ],
    )
    def test_unconvertible_level_constraint_omits_the_block_with_a_named_reason(
        self, kwargs, expected_reason
    ):
        response = analyse(value_frame="level", **kwargs)
        result = only_result(response)

        assert (
            result.constraint_analysis is None
        ), f"{expected_reason}: the block must be omitted, not scored"
        assert sole_refusal_reason(response) == expected_reason

    def test_the_domain_guard_refusal_reports_the_offending_operands(self):
        """A refusal that does not say WHICH operand was out of domain is unusable."""
        response = analyse(value_frame="level", threshold=250000.0)
        warning = refusals(response)[0]

        assert warning.detail["constraint_id"] == CID
        assert "constraint_threshold" in warning.detail["out_of_domain"]
        assert warning.detail["out_of_domain"]["constraint_threshold"] == 250000.0

    def test_a_refused_constraint_never_reaches_the_p_win_sensitivity_metric(self):
        """The SECOND producer of joint satisfaction must ride the same plans.

        `_compute_evpi_metric` folds `_compute_constraint_probabilities` into the
        per-factor sensitivity metric ("P(joint_goal) when goal_constraints
        exist"). A fabricated joint probability there is the same untruth wearing
        a different field name, and fixing only the visible channel would leave a
        twin behind.
        """
        response = analyse(value_frame=None, include_voi=True)

        assert only_result(response).constraint_analysis is None
        assert response.p_win_sensitivity is None, (
            "a sensitivity metric built on an unresolvable joint probability is "
            "fabricated, not degraded — the phase is all-or-nothing"
        )
        disclosures = [
            w
            for w in response.inference_warnings
            if w.code == "EVPI_UNAVAILABLE"
            and w.detail.get("reason") == "constraints_not_convertible"
        ]
        assert len(disclosures) == 1, (
            "the omission must be disclosed with its real reason, not silently "
            "attributed to a time budget: "
            f"{[(w.code, w.detail.get('reason')) for w in response.inference_warnings]}"
        )


# =============================================================================
# 4. WHOLE-BLOCK REFUSAL — a joint probability is a conjunction.
# =============================================================================


class TestPartialAttestationSuppressesTheJoint:
    """P(ALL satisfied) is unresolvable if ANY conjunct is unresolvable.

    `ConstraintAnalysisV2.joint_probability` is a REQUIRED wire field with a live
    consumer in another repo, so a partially-populated block cannot be emitted
    honestly. `constraint_analysis` is already Optional and already absent on the
    no-constraints path, so absence is the shape every consumer already handles.
    """

    def test_one_unattested_constraint_suppresses_the_whole_block(self):
        unattested = GoalConstraint(
            **{
                "constraint_id": "cid-second",
                "node_id": "c",
                "operator": "<=",
                "value": 0.95,
            }
        )
        response = analyse(value_frame="level", extra_constraints=[unattested])

        assert only_result(response).constraint_analysis is None, (
            "a joint probability computed over a resolved conjunct AND an "
            "unresolved one is not a probability of anything"
        )

    def test_the_refusal_names_the_unattested_constraint_not_the_attested_one(self):
        unattested = GoalConstraint(
            **{
                "constraint_id": "cid-second",
                "node_id": "c",
                "operator": "<=",
                "value": 0.95,
            }
        )
        response = analyse(value_frame="level", extra_constraints=[unattested])

        warnings = unstamped_warnings(response)
        assert [w.detail["constraint_id"] for w in warnings] == [
            "cid-second"
        ], "the attested constraint is not the problem and must not be blamed"

    def test_two_attested_constraints_still_produce_a_joint(self):
        """The suppression must be caused by the REFUSAL, not by having two rows."""
        second = GoalConstraint(
            **{
                "constraint_id": "cid-second",
                "node_id": "c",
                "operator": "<=",
                "value": 0.95,
            },
            value_frame="level",
        )
        response = analyse(value_frame="level", extra_constraints=[second])
        analysis = only_result(response).constraint_analysis

        assert analysis is not None
        # level = 0.7 + 0.5*(1-f) in [0.7, 1.2]; <= 0.95 iff f >= 0.5. Joint with
        # (f <= 0.8) is f in [0.5, 0.8] => 0.3.
        assert analysis.joint_probability == pytest.approx(0.3, abs=TOL)
        assert constraint_by_id(analysis, "cid-second").prob_satisfied == pytest.approx(
            0.5, abs=TOL
        )


# =============================================================================
# 5. ONE DIALECT — the anti-twin binding.
# =============================================================================


class TestChannelAAndChannelBAreOneImplementation:
    """Two same-named-but-different code paths have burned this estate before.

    These bind the two channels BEHAVIOURALLY: on one graph where the goal node is
    also the constraint target, both channels answer the same question about the
    same samples. A divergence is a second dialect, whatever the code looks like.
    """

    def test_the_two_channels_agree_exactly_on_the_same_threshold(self):
        response = analyse(
            constrain_the_goal_node=True,
            value_frame="level",
            goal_threshold=0.8,
            goal_threshold_frame="level",
        )
        result = only_result(response)

        assert result.probability_of_goal is not None
        assert result.constraint_analysis is not None
        assert result.constraint_analysis.joint_probability == pytest.approx(
            result.probability_of_goal, abs=1e-12
        ), "one quantity computed by two channels must be one number"

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"target_is_root": True}, id="root"),
            pytest.param({"target_baseline": None}, id="missing_baseline"),
            pytest.param({"intervene_on_target": True}, id="pinned_by_intervention"),
            pytest.param({"threshold": 250000.0}, id="out_of_domain"),
            pytest.param({"parent_epsilon_std": 0.05}, id="epsilon"),
            pytest.param({"pu_on_target": True}, id="parameter_uncertainty"),
        ],
    )
    def test_both_channels_refuse_on_exactly_the_same_preconditions(self, kwargs):
        threshold = kwargs.get("threshold", 0.8)
        response = analyse(
            constrain_the_goal_node=True,
            value_frame="level",
            goal_threshold=threshold,
            goal_threshold_frame="level",
            **kwargs,
        )
        result = only_result(response)

        assert result.probability_of_goal is None, "Channel A must refuse here"
        assert result.constraint_analysis is None, (
            "Channel B must refuse on the SAME precondition — a channel that "
            "accepts what its sibling refuses is a second dialect"
        )
