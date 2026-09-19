"""A ROOT target's samples ARE levels, so a `level` threshold needs no conversion.

THE DEFECT. ``_resolve_threshold_in_sample_frame`` refused EVERY ``level``-framed
threshold whose target node has no parents, with reason ``root_target`` /
``root_goal``::

    "<noun> '<id>' has no parents. A root node takes its base from
     observed_state.value, so its samples are not in the non-root
     change-from-origin frame this conversion is derived for."

Both halves of that sentence are true and the conclusion does not follow. The
conversion this limb guards is not needed for a root — the identity is.

THE ARITHMETIC, derived from the evaluator rather than assumed
(``SCMEvaluatorV2.evaluate``, robustness_analyzer_v2.py:1448-1470)::

    node_values[T] = base + intercept + SUM(parent_value * strength)

For a ROOT the sum is empty, so ``sample = base + intercept``, and ``base`` is,
in the evaluator's own priority order:

  * ``factor_values[T]``     — FactorSampler's draw for this node. Every family
    draws an ABSOLUTE value: ``uniform`` over ``[range_min, range_max]``,
    ``normal`` around the central value, ``point_mass`` exactly it
    (``_sample_from_distribution``, :1272-1289). Not a change from anything.
  * ``observed_state.value`` — the node's absolute current level (:1460-1466).
  * ``0.0``                  — nothing measured about this node at all.

In the first two cases the sample IS the node's level on that draw. A level
threshold is therefore already in the samples' own frame, and the correct plan is
the identity — byte-identical to what the caller-attested ``delta`` branch
returns. In the third it is a fabricated zero, and refusing is the only honest
answer.

WHAT THE REFUSAL COST, end to end. ``_resolve_constraint_plans`` refuses the
BLOCK when any single constraint is unresolvable (:4703-4716, and that
all-or-nothing rule is right — ``joint_probability`` is a conjunction). So one
root-targeted constraint erased every constraint score on the run. CEE then sees
an empty evaluated set, takes the ``unevaluated`` verdict, and
``MAY_NAME_LEADING_OPTION[unevaluated] = false`` withholds the leading option. A
user who wrote "keep churn under 4%" against a root factor got no constraint
check AND no recommendation — and the sentence offered them ("tell me the limit
in your own words and run the analysis again") could not change either outcome.

PLoT had already derived this ruling from ISL's own evaluator and shipped it:
``constraint-reliability.ts`` ``resolveConstraintSampleFrameAnchor`` returns
``'root_observed_level'`` for a root carrying a finite ``observed_state.value``.
The two services held opposite rulings on one node class; this file settles ISL's
half in PLoT's favour, because PLoT's reading of the evaluator is the correct one.

WHAT STILL REFUSES, and why each is a different question:
  * no measured base (no observed value AND no distribution to draw from) — the
    samples are a fabricated 0.0, so there is no level to compare;
  * a non-zero ``intercept`` — the modelled level is then ``value + intercept``
    and disagrees with the observed level it was stamped against;
  * an option intervening on the target — pinned samples, unchanged limb;
  * operands outside the normalised domain — unchanged guard, now applied to the
    root anchor instead of a baseline it does not have;
  * every NON-root target — the change-from-origin conversion is untouched.

THE WITNESS GRAPH, chosen so every expected probability is computable by hand::

    d  root factor, observed_state{value: 0.2}          (the driver an option pushes)
    t  the constraint target (root by default)
    g  outcome, parent d, strength 0.5                  (the goal)

    options: "hold" (nothing) and "push" (do(d := 1.0))

``t`` is never intervened on, so its samples are its own base on every draw and
both options report the same probability for it — which is the honest answer when
no option touches the quantity the user constrained.
"""

from typing import List, Optional

import pytest

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

CID = "cid-root-target"


def build_request(
    *,
    threshold: float = 0.3,
    operator: str = "<=",
    value_frame: Optional[str] = "level",
    target_is_root: bool = True,
    target_observed_value: Optional[float] = 0.6,
    target_baseline: Optional[float] = 0.6,
    target_has_observed_state: bool = True,
    target_intercept: float = 0.0,
    target_pu: Optional[ParameterUncertainty] = None,
    intervene_on_target: bool = False,
    n_samples: int = N_SAMPLES,
):
    """The witness graph. Every knob exists to drive one adversarial fixture."""
    target_observed = (
        ObservedState(value=target_observed_value, baseline=target_baseline, unit="norm")
        if target_has_observed_state
        else None
    )

    nodes = [
        NodeV2(id="d", kind="factor", label="Driver", observed_state=ObservedState(value=0.2)),
        NodeV2(id="g", kind="outcome", label="Goal"),
        NodeV2(
            id="t",
            kind="factor",
            label="Target",
            intercept=target_intercept,
            observed_state=target_observed,
        ),
    ]
    edges = [
        EdgeV2(
            **{"from": "d", "to": "g"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=0.5, std=0.0011),
        )
    ]
    if not target_is_root:
        edges.append(
            EdgeV2(
                **{"from": "d", "to": "t"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.0011),
            )
        )

    uncertainties: List[ParameterUncertainty] = []
    if target_pu is not None:
        uncertainties.append(target_pu)

    return RobustnessRequestV2(
        request_id="rm-root-level-identity",
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(
                id="hold",
                label="Hold",
                interventions={"t": 0.5} if intervene_on_target else {},
            ),
            InterventionOption(id="push", label="Push", interventions={"d": 1.0}),
        ],
        goal_node_id="g",
        n_samples=n_samples,
        seed=SEED,
        goal_constraints=[
            GoalConstraint(
                constraint_id=CID,
                node_id="t",
                operator=operator,
                value=threshold,
                value_frame=value_frame,
            )
        ],
        parameter_uncertainties=uncertainties or None,
    )


def analyse(**kwargs):
    return RobustnessAnalyzerV2().analyze(build_request(**kwargs))


def result_for(response, option_id: str):
    matches = [r for r in response.results if r.option_id == option_id]
    assert len(matches) == 1, f"expected exactly one result for '{option_id}'"
    return matches[0]


def constraint_row(analysis):
    """Locate the constraint by its ECHOED IDENTITY, never by a value predicate."""
    rows = [c for c in analysis.constraints if c.constraint_id == CID]
    assert len(rows) == 1, f"expected exactly one row for {CID}, got {len(rows)}"
    return rows[0]


def refusals(response):
    return [
        w
        for w in (response.inference_warnings or [])
        if w.code in {"CONSTRAINT_NOT_CONVERTIBLE", "CONSTRAINT_FRAME_UNSPECIFIED"}
    ]


def sole_refusal_reason(response) -> str:
    found = refusals(response)
    assert len(found) == 1, f"expected exactly one refusal, got {[w.detail for w in found]}"
    return found[0].detail["reason"]


# =============================================================================
# PRECONDITION PINS — a fixture that does not reach the branch under test
# measures a different arm and every assertion below it is decoration.
# =============================================================================


class TestFixtureReachesTheBranchUnderTest:
    def test_the_target_is_a_root_with_a_finite_observed_value_and_zero_intercept(self):
        request = build_request()
        target = next(n for n in request.graph.nodes if n.id == "t")

        assert [e for e in request.graph.edges if e.to == "t"] == [], (
            "the target must have NO parents, or this fixture exercises the "
            "non-root conversion path instead of the root identity path"
        )
        assert target.observed_state is not None
        assert target.observed_state.value == 0.6
        assert target.intercept == 0.0
        assert request.goal_constraints[0].value_frame == "level", (
            "a 'delta' frame returns before the root limb is ever reached"
        )
        assert not any("t" in o.interventions for o in request.options), (
            "an intervention on the target refuses one limb EARLIER, so this "
            "fixture would prove nothing about the root limb"
        )

    def test_the_non_root_variant_really_gives_the_target_a_parent(self):
        """CONTRAST CONTROL for the pin above: the knob must actually move."""
        request = build_request(target_is_root=False)
        assert [e.from_ for e in request.graph.edges if e.to == "t"] == ["d"]


# =============================================================================
# THE IDENTITY PATH — the evaluable cases now evaluate.
# =============================================================================


class TestRootTargetLevelIsEvaluated:
    def test_a_sampled_root_target_reports_its_own_level_probability(self):
        """t ~ U(0, 1) on every draw, so P(t <= 0.3) = 0.3.

        Three outcomes are distinguishable here, which is what makes the pin
        discriminating rather than decorative:
          0.3  — the identity, the truth;
          0.0  — what the change-from-origin conversion would give
                 (t is never intervened on, so option == status quo every draw
                 and the recovered "level" would be the constant baseline 0.6,
                 which fails `<= 0.3` on all 10000 draws);
          absent — the pre-fix refusal.
        """
        response = analyse(
            target_pu=ParameterUncertainty(
                node_id="t", distribution="uniform", range_min=0.0, range_max=1.0
            )
        )
        analysis = result_for(response, "hold").constraint_analysis

        assert analysis is not None, "the block must be scored, not refused"
        assert refusals(response) == []
        assert constraint_row(analysis).prob_satisfied == pytest.approx(0.3, abs=TOL)
        assert analysis.joint_probability == pytest.approx(0.3, abs=TOL)

    def test_a_prior_only_root_target_is_evaluated_from_the_distribution_it_draws_from(self):
        """t ~ U(0.4, 0.8) with NO observed value: P(t <= 0.6) = 0.5.

        `resolve_factor_central_value` reports PRIOR_MIDPOINT here — the sampler
        ignores observed_state for the uniform family, so the anchor must too.
        """
        response = analyse(
            threshold=0.6,
            target_has_observed_state=False,
            target_pu=ParameterUncertainty(
                node_id="t", distribution="uniform", range_min=0.4, range_max=0.8
            ),
        )
        analysis = result_for(response, "hold").constraint_analysis

        assert analysis is not None
        assert constraint_row(analysis).prob_satisfied == pytest.approx(0.5, abs=TOL)

    def test_an_unsampled_root_target_is_its_observed_level_on_every_draw(self):
        """No PU: t sits at its observed 0.6 in every draw, so P(t >= 0.5) = 1.0.

        1.0 is the discriminating direction. The structural-zero defect this
        channel was hardened against produces 0.0, and the pre-fix refusal
        produces no block at all — neither can be mistaken for this.
        """
        response = analyse(operator=">=", threshold=0.5, n_samples=2_000)
        analysis = result_for(response, "hold").constraint_analysis

        assert analysis is not None
        assert constraint_row(analysis).prob_satisfied == pytest.approx(1.0, abs=1e-12)

    def test_an_unmet_limit_on_a_root_target_is_reported_as_unmet_not_as_absent(self):
        """The same fixture, the other way round: P(t <= 0.3) = 0.0 with t == 0.6.

        A scored 0.0 and an omitted block read identically in a summary and mean
        opposite things to a user: "this limit is broken" versus "this limit was
        never checked".
        """
        response = analyse(operator="<=", threshold=0.3, n_samples=2_000)
        analysis = result_for(response, "hold").constraint_analysis

        assert analysis is not None
        assert constraint_row(analysis).prob_satisfied == pytest.approx(0.0, abs=1e-12)

    def test_every_option_gets_the_block_not_just_the_first(self):
        response = analyse(operator=">=", threshold=0.5, n_samples=2_000)

        for option_id in ("hold", "push"):
            analysis = result_for(response, option_id).constraint_analysis
            assert analysis is not None, f"{option_id}: block missing"
            assert constraint_row(analysis).prob_satisfied == pytest.approx(1.0, abs=1e-12)


# =============================================================================
# WHAT STILL REFUSES — each limb is a different question, and none of them is
# "does this node have parents".
# =============================================================================


class TestRootTargetRefusalsThatSurvive:
    def test_a_root_with_no_measured_base_still_refuses(self):
        """No observed value and no distribution: base falls through to 0.0.

        That zero is fabricated — it is the absence of data wearing a number's
        clothes — so there is no level here to compare a threshold against.
        """
        response = analyse(target_has_observed_state=False, n_samples=2_000)

        assert result_for(response, "hold").constraint_analysis is None
        assert sole_refusal_reason(response) == "root_target"
        assert "no measured base" in refusals(response)[0].detail["message"]

    def test_a_root_whose_only_distribution_centres_on_a_fabricated_zero_still_refuses(self):
        """normal(std=0.1) with no observed value draws around 0.0, not around a
        measured level. The uniform family is the opposite case (its bounds ARE
        the data), which is why the two are classified by the ONE central-value
        resolver rather than by the presence of a PU."""
        response = analyse(
            target_has_observed_state=False,
            target_pu=ParameterUncertainty(node_id="t", distribution="normal", std=0.1),
            n_samples=2_000,
        )

        assert result_for(response, "hold").constraint_analysis is None
        assert sole_refusal_reason(response) == "root_target"

    def test_a_non_zero_intercept_still_refuses(self):
        """sample = value + intercept, so the modelled level is 0.85 while the
        observed level stamped on the node is 0.6. Comparing the user's
        threshold against either one is a guess about which they meant."""
        response = analyse(target_intercept=0.25, n_samples=2_000)

        assert result_for(response, "hold").constraint_analysis is None
        assert sole_refusal_reason(response) == "root_target"
        assert refusals(response)[0].detail["root_intercept"] == 0.25

    def test_an_intervened_root_target_still_refuses_on_the_earlier_limb(self):
        """CONTRAST CONTROL on the ordering: the intervention limb precedes the
        root limb and must keep doing so — pinned samples are not this node's
        own level."""
        response = analyse(intervene_on_target=True, n_samples=2_000)

        assert result_for(response, "hold").constraint_analysis is None
        assert sole_refusal_reason(response) == "target_pinned_by_intervention"

    def test_the_domain_guard_still_fires_on_the_root_anchor(self):
        """Raw user units where normalised values were expected. The guard is the
        same guard; only the operand it anchors on changes."""
        response = analyse(threshold=250_000.0, n_samples=2_000)
        warning = refusals(response)[0]

        assert result_for(response, "hold").constraint_analysis is None
        assert warning.detail["reason"] == "constraint_values_outside_normalised_domain"
        assert "constraint_threshold" in warning.detail["out_of_domain"]

    def test_the_non_root_conversion_path_is_untouched(self):
        """CONTRAST CONTROL for the whole change: a non-root target with no
        baseline must still refuse for want of a baseline. If this ever reported
        the identity, the change would have replaced the conversion rather than
        added a sibling to it."""
        response = analyse(target_is_root=False, target_baseline=None, n_samples=2_000)

        assert result_for(response, "hold").constraint_analysis is None
        assert sole_refusal_reason(response) == "missing_target_baseline"
