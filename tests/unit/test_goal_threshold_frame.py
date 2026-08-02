"""ROADMAP 2.258 / 2.286 — resolving goal_threshold against the right reference.

THE ORIGINAL DEFECT (2.258). CEE mints ``goal_threshold`` as a normalised LEVEL
(0.8 == a GBP 6.0m target against a GBP 7.5m cap). ISL compared it directly
against the goal's samples, which are not levels, and reported a STRUCTURAL zero
rendered to users as "< 1% chance of hitting your goal".

⚠ THE FIX WAS WRONG TOO (2.286), and wrong in the more dangerous direction.
2.258 converted with ``delta_threshold = T - B + intercept``, derived from::

    sample      = intercept + S            (S = parents' propagated contribution)
    real_level  = baseline  + S            <-- FALSE

The second line assumes S is a CHANGE, i.e. that S == 0 when nothing is done.
It is not. ``SCMEvaluatorV2.evaluate`` seeds ``observed_state.value`` as the base
of ROOT nodes ONLY, and ``FactorSampler`` centres factor draws on that same
observed value, so parents carry ABSOLUTE current values and a non-root goal —
whose own base is 0.0 — receives their absolute propagated sum. Under the status
quo S is emphatically non-zero, so anchoring at zero shifted every level
comparison by exactly S_sq. Measured on staging tip 71a962e8 (f=0.5, s=0.5,
B=0.7, T=0.9, do-nothing option): status quo scores 0.25 against a converted
threshold of 0.20, so ISL reported **probability_of_goal = 1.0** for a goal the
status quo does not reach. A confident INVERSION, not a missing number.

THE ARITHMETIC NOW. Levels are recovered per draw against a status-quo
REFERENCE, under common random numbers::

    level_i = B + (option_sample_i - status_quo_sample_i)

S_sq cancels because it appears in both terms — and so does the intercept, which
is why the ``+ intercept`` term is gone rather than re-derived.

THE WITNESS GRAPH used throughout this module, chosen so every expected
probability is computable BY HAND rather than read back off the code::

    f  root factor, observed_state{value: 0.5}, ParameterUncertainty uniform(0, 1)
    g  goal, NON-ROOT outcome, observed_state{baseline: B}, intercept I
    f -> g  strength mean s=0.5, std 0.0011 (near-deterministic), exists_prob 1.0

    sample(g)   = 0 (non-root base) + I + f * s     ~  Uniform(I, I + 0.5)
    level(g)    = B + s * (1 - f)                   under `analyse_level`, which
                                                    intervenes do(f := 1.0)

    P(level >= T) = P(1 - f >= (T - B)/s) = 1 - 2 * (T - B)   for 0 <= 2(T-B) <= 1

Note this is the SAME closed form the 2.258 fixture produced, because ``1 - f``
is also U(0, 1). Every expected probability in this module is therefore
numerically UNCHANGED by the 2.286 fix — the old numbers were arithmetically
right about the wrong quantity. What proves the fix is not these pins but
``TestStatusQuoIsNotProgress``, where the two quantities disagree completely.

The intercept I CANCELS, now structurally rather than by a remembered term.

Auto-scaled noise is default-OFF (``ENABLE_AUTO_SCALED_NOISE``), so the samples
here are purely model-driven and the hand arithmetic is exact up to MC error.
With n_samples=10000 one standard error on a p~0.5 estimate is ~0.005, so every
assertion below uses a 0.02 tolerance (~4 s.e.).
"""

from typing import get_args

import pytest

from src.models.robustness_v2 import (
    EdgeV2,
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
# ~4 standard errors at n=10k. Wide enough never to flake, tight enough that a
# wrong conversion (which moves the answer by whole tenths) cannot hide inside it.
TOL = 0.02


def build_request(
    *,
    goal_threshold=None,
    goal_threshold_frame=None,
    baseline=0.7,
    intercept=0.0,
    goal_observed_value=0.7,
    goal_has_observed_state=True,
    goal_is_root=False,
    pu_on_goal=False,
    intervene_on_goal=False,
    goal_epsilon_std=0.0,
    parent_epsilon_std=0.0,
    strength_mean=0.5,
    option_interventions=None,
    factor_now=0.5,
    factor_pu=True,
):
    """The witness graph. Every knob exists to drive one adversarial fixture."""
    goal_observed = (
        ObservedState(value=goal_observed_value, baseline=baseline, unit="norm")
        if goal_has_observed_state
        else None
    )
    nodes = [
        NodeV2(
            id="g",
            kind="outcome",
            label="Goal",
            intercept=intercept,
            epsilon_std=goal_epsilon_std,
            observed_state=goal_observed,
        )
    ]
    edges = []
    uncertainties = []
    if not goal_is_root:
        nodes.insert(
            0,
            NodeV2(
                id="f",
                kind="factor",
                label="Driver",
                epsilon_std=parent_epsilon_std,
                observed_state=ObservedState(value=factor_now),
            ),
        )
        edges.append(
            EdgeV2(
                **{"from": "f", "to": "g"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=strength_mean, std=0.0011),
            )
        )
        if factor_pu:
            uncertainties.append(
                ParameterUncertainty(
                    node_id="f", distribution="uniform", range_min=0.0, range_max=1.0
                )
            )
    if pu_on_goal:
        uncertainties.append(ParameterUncertainty(node_id="g", distribution="normal", std=0.1))
    if goal_is_root:
        # A root goal needs SOME variation or it is a degenerate constant.
        uncertainties.append(
            ParameterUncertainty(node_id="g", distribution="uniform", range_min=0.0, range_max=1.0)
        )

    return RobustnessRequestV2(
        request_id="rm-2258",
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(
                id="hold",
                label="Hold",
                interventions=(
                    {"g": 0.5} if intervene_on_goal else (option_interventions or {})
                ),
            )
        ],
        goal_node_id="g",
        n_samples=N_SAMPLES,
        seed=SEED,
        goal_threshold=goal_threshold,
        goal_threshold_frame=goal_threshold_frame,
        parameter_uncertainties=uncertainties or None,
    )


def analyse(**kwargs):
    return RobustnessAnalyzerV2().analyze(build_request(**kwargs))


# Push the driver to its ceiling. Level-frame requests are analysed with an option
# that actually DOES something, because a reference-anchored probability for a
# do-nothing option is degenerate by construction — that degeneracy is itself the
# subject of TestStatusQuoIsNotProgress below, and is where the 2.286 defect lived.
PUSH_DRIVER = {"f": 1.0}


def analyse_level(**kwargs):
    """Analyse with the driver pushed to 1.0 — the level-frame workhorse.

    P(goal reaches T) = P(B + s*(1 - f) >= T) = P(f <= 1 - (T-B)/s)
                      = 1 - 2*(T - B)                        for s = 0.5, f ~ U(0,1)

    Note this is the SAME closed form the pre-2.286 fixture produced, because
    ``1 - f`` is also U(0, 1). Every expected probability in this module is
    therefore numerically unchanged by the fix — what changed is that the number
    now answers the user's question ("will my goal reach T if I do this?")
    instead of "is the model's absolute propagated sum above T - B?".
    """
    return analyse(option_interventions=PUSH_DRIVER, **kwargs)


def warnings_by_code(response, code):
    return [w for w in response.inference_warnings if w.code == code]


# =============================================================================
# 0. ROADMAP 2.286 — the INVERSION. RED-first on the zero-anchor defect.
# =============================================================================


class TestStatusQuoIsNotProgress:
    """A do-nothing option cannot reach a goal it does not already reach.

    THE DEFECT 2.286 REPAIRS. The 2.258 conversion anchored a level threshold at
    ZERO: ``delta_threshold = T - B + intercept``, on the premise that the goal's
    samples are a CHANGE from its current level. They are not. The evaluator
    seeds ``observed_state.value`` as the base of ROOT nodes only, so factors
    carry their ABSOLUTE current values and propagate ``parent_value * strength``
    into a non-root goal whose own base is 0.0.

    So under the status quo the goal still scores ``S_sq = 0.5 * 0.5 = 0.25``,
    the converted threshold was ``0.9 - 0.7 + 0 = 0.20``, and ``0.25 >= 0.20``
    held for every single draw. ISL reported **100%** confidence in a goal the
    status quo does not reach at all. The truth is 0%: doing nothing leaves the
    goal at 0.7, and 0.7 < 0.9.

    This is the failure mode that matters most — not a missing number, and not a
    pessimistic one, but a CONFIDENTLY INVERTED one, which a UI renders as
    certainty and a user acts on.
    """

    def test_status_quo_does_not_reach_an_unreached_goal(self):
        """RED-FIRST PIN, and the exact scenario measured on staging tip 71a962e8.

        f pinned at its observed 0.5 (no parameter uncertainty), so every draw is
        the same story and the arithmetic is a single line rather than a
        distribution: status-quo sample 0.25, old converted threshold 0.20,
        0.25 >= 0.20 on all 10,000 draws -> the old code returns exactly 1.0.
        """
        response = analyse(
            goal_threshold=0.9,
            goal_threshold_frame="level",
            baseline=0.7,
            option_interventions={},  # explicit: change nothing
            factor_pu=False,  # f stays at its observed 0.5
        )
        prob = response.results[0].probability_of_goal

        assert prob is not None, "a convertible level threshold must produce a probability"
        assert prob == 0.0, (
            f"expected 0.0 — the status quo leaves the goal at its baseline 0.7, "
            f"which does not reach 0.9 — but got {prob}. 1.0 is the 2.286 "
            f"inversion: the threshold was anchored at zero instead of at the "
            f"status quo's own score of 0.25."
        )

    def test_status_quo_does_not_reach_it_under_parameter_uncertainty_either(self):
        """The same claim with f ~ U(0, 1) restored.

        Uncertainty about where the driver currently SITS is not an opportunity
        to reach the goal by doing nothing: under common random numbers it
        cancels between the option and the reference, so the answer stays a hard
        0.0. The old code returned ~0.60 here — a different wrong number from the
        one above, from the same zero anchor.
        """
        response = analyse(
            goal_threshold=0.9,
            goal_threshold_frame="level",
            baseline=0.7,
            option_interventions={},
        )
        assert response.results[0].probability_of_goal == 0.0

    def test_status_quo_reaches_a_goal_it_already_meets(self):
        """The mirror image, so the pin cannot be satisfied by always returning 0.

        Without this, `probability_of_goal = 0.0` would pass the tests above while
        being just as wrong in the other direction.
        """
        response = analyse(
            goal_threshold=0.6,
            goal_threshold_frame="level",
            baseline=0.7,
            option_interventions={},
        )
        assert response.results[0].probability_of_goal == 1.0, (
            "the goal is already at 0.7 and the target is 0.6, so doing nothing "
            "meets it with certainty"
        )

    def test_the_effect_is_measured_against_the_status_quo_not_against_zero(self):
        """The mechanism itself, isolated from any particular probability.

        Two runs identical except for the driver's CURRENT value. The option
        pushes f to 1.0 either way, so a model anchored at zero sees the same
        option score and must return the same probability. A correctly anchored
        one does not: there is less room to gain from 0.8 than from 0.2.
        """
        # No parameter uncertainty on f: a PU draw REPLACES the observed value in
        # both the option and the reference, so factor_now would be unread and
        # this test would compare two identical runs and pass vacuously.
        from_low = analyse(
            goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7,
            option_interventions=PUSH_DRIVER, factor_now=0.2, factor_pu=False,
        )
        from_high = analyse(
            goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7,
            option_interventions=PUSH_DRIVER, factor_now=0.8, factor_pu=False,
        )
        assert from_low.results[0].probability_of_goal is not None
        assert from_low.results[0].probability_of_goal > from_high.results[0].probability_of_goal, (
            "the same intervention must be worth MORE when the driver starts "
            "lower; if these are equal the status-quo reference is not being read"
        )


# =============================================================================
# 1. The witness — hand-computed, and the RED-first pin on the defect
# =============================================================================


class TestFrameConversionWitness:
    """The conversion produces the BY-HAND probability, not the structural zero."""

    def test_level_threshold_yields_hand_computed_probability(self):
        """B=0.7, T=0.9, s=0.5 -> P = 1 - 2*(0.9-0.7) = 0.60.

        The option pushes f to 1.0, so the goal gains s*(1 - f) over its baseline
        and reaches 0.9 exactly when f <= 0.6.
        """
        response = analyse_level(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        prob = response.results[0].probability_of_goal

        assert prob is not None, "a convertible level threshold must produce a probability"
        assert prob == pytest.approx(0.60, abs=TOL), (
            f"expected the hand-computed 0.60, got {prob}."
        )

    def test_samples_are_neither_levels_nor_changes_from_the_goals_own_level(self):
        """The premise of the row, asserted rather than assumed — and CORRECTED.

        ⚠ 2.286. The predecessor of this test asserted the samples were
        "change-from-origin", which licensed the zero anchor. They are not. A
        non-root goal's sample is the propagated sum of its parents' ABSOLUTE
        current values — so under the status quo it is NOT zero, which is exactly
        why anchoring a level threshold at zero inverted answers.

        Both halves are pinned here, because the samples being neither thing is
        the whole reason a status-quo REFERENCE is required.
        """
        response = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        samples = response.results[0].outcome_distribution.samples

        assert min(samples) == pytest.approx(0.0, abs=0.01)
        assert max(samples) == pytest.approx(0.5, abs=0.01)
        assert not (min(samples) <= 0.7 <= max(samples)), (
            "the goal's baseline level sits inside the sample range — the samples "
            "would then be levels and this converter needs re-deriving"
        )

        # ... and NOT changes from the goal's current level either: with f at its
        # observed 0.5 and nothing intervening, a change-from-origin sample would
        # be 0.0. It is 0.25.
        status_quo = analyse(
            goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7,
            option_interventions={}, factor_pu=False,
        )
        sq_samples = status_quo.results[0].outcome_distribution.samples
        assert sum(sq_samples) / len(sq_samples) == pytest.approx(0.25, abs=0.01), (
            "the status quo must score 0.5 * 0.5 = 0.25, NOT 0.0 — if this ever "
            "reads 0.0 the samples really are change-from-origin and the status-quo "
            "reference could be dropped"
        )

    def test_conversion_changes_only_the_comparison_not_the_samples(self):
        """Control: resolving the frame must not perturb sampling.

        Same seed, one request stamped 'level' and one 'delta' — byte-identical
        sample series, different probabilities.

        2.286 gives this control a SECOND job, and it is the sharper one. The
        level path now runs an EXTRA evaluation per draw (the status-quo
        reference). If that reference were drawn from the shared evaluator it
        would consume the epsilon RNG stream and shift every subsequent sample,
        silently changing results across the repo. It is given its own
        epsilon-free evaluator precisely so this equality still holds — so this
        assertion is what pins that decision.
        """
        level = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        delta = analyse(goal_threshold=0.2, goal_threshold_frame="delta", baseline=0.7)

        assert (
            level.results[0].outcome_distribution.samples
            == delta.results[0].outcome_distribution.samples
        ), "the frame field must not touch sampling"
        # The probabilities now legitimately DIFFER: 'delta' asks whether the
        # model's propagated sum clears 0.2 (0.60 of the time), 'level' asks
        # whether this option gets the goal to 0.9 (never — it does nothing).
        assert delta.results[0].probability_of_goal == pytest.approx(0.60, abs=TOL)
        assert level.results[0].probability_of_goal == 0.0

    def test_intercept_cancels_out(self):
        """A non-zero goal intercept must NOT move the answer.

        Under 2.258 this held by arithmetic: the '+ intercept' term in
        `T - B + I` cancelled the shift in the samples. Under 2.286 it holds
        STRUCTURALLY — the intercept is present in both the option sample and the
        status-quo sample, so it cancels in the difference and no term has to be
        remembered. That is why the '+ intercept' term is gone rather than
        preserved, and this test is the guard on the change.
        """
        without = analyse_level(
            goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7, intercept=0.0
        )
        with_intercept = analyse_level(
            goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7, intercept=0.1
        )

        assert with_intercept.results[0].probability_of_goal == pytest.approx(0.60, abs=TOL)
        assert with_intercept.results[0].probability_of_goal == pytest.approx(
            without.results[0].probability_of_goal, abs=TOL
        )


# =============================================================================
# 2. Fail closed and LOUD — the control that hunts for the absence
# =============================================================================


class TestFailClosedWarnings:
    """No frame, or no baseline => NO probability, and a loud reason."""

    def test_frame_absent_omits_probability_and_warns(self):
        """THE CONTROL. A fixture built to demonstrate the conversion cannot hunt
        for its absence, so this asserts the no-frame path emits NO probability
        AND DOES emit the warning — both halves, or the absence claim is vacuous.
        """
        response = analyse(goal_threshold=0.9, goal_threshold_frame=None, baseline=0.7)

        assert (
            response.results[0].probability_of_goal is None
        ), "an unattested threshold must never produce a probability"
        found = warnings_by_code(response, "GOAL_THRESHOLD_FRAME_UNSPECIFIED")
        assert len(found) == 1, (
            f"expected exactly one GOAL_THRESHOLD_FRAME_UNSPECIFIED, got "
            f"{[w.code for w in response.inference_warnings]}"
        )
        assert found[0].detail["reason"] == "frame_not_stamped"
        assert found[0].detail["goal_threshold"] == 0.9

    def test_fail_closed_warning_severity_is_warning_not_info(self):
        """PLoT HIDES severity=='info'. A reason nobody sees is not a disclosure.

        This is the enforcing test for the hand-maintained severity comment in
        response_v2.py (trap 12).
        """
        response = analyse(goal_threshold=0.9, goal_threshold_frame=None, baseline=0.7)
        assert warnings_by_code(response, "GOAL_THRESHOLD_FRAME_UNSPECIFIED")[0].severity == (
            "warning"
        )

        no_baseline = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=None)
        assert warnings_by_code(no_baseline, "GOAL_THRESHOLD_NOT_CONVERTIBLE")[0].severity == (
            "warning"
        )

    def test_probability_absent_from_the_wire_not_null(self):
        """Omitted means OMITTED — exclude_none must drop the key entirely."""
        response = analyse(goal_threshold=0.9, goal_threshold_frame=None, baseline=0.7)
        dumped = response.results[0].model_dump(exclude_none=True)
        assert "probability_of_goal" not in dumped

    def test_level_without_baseline_omits_and_names_the_missing_field(self):
        response = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=None)

        assert response.results[0].probability_of_goal is None
        found = warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert len(found) == 1
        assert found[0].detail["reason"] == "missing_goal_baseline"
        assert found[0].field == "nodes[g].observed_state.baseline"
        assert found[0].detail["observed_state_present"] is True

    def test_level_without_any_observed_state_omits_and_warns(self):
        response = analyse(
            goal_threshold=0.9, goal_threshold_frame="level", goal_has_observed_state=False
        )

        assert response.results[0].probability_of_goal is None
        found = warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert found[0].detail["reason"] == "missing_goal_baseline"
        assert found[0].detail["observed_state_present"] is False

    @pytest.mark.parametrize(
        "kwargs,reason",
        [
            ({"goal_is_root": True}, "root_goal"),
            ({"pu_on_goal": True}, "goal_parameter_uncertainty_shifts_base"),
            ({"intervene_on_goal": True}, "goal_pinned_by_intervention"),
        ],
    )
    def test_unprovable_frames_refuse_with_their_own_reason(self, kwargs, reason):
        """Each of these breaks `sample = intercept + S`, so the conversion is
        not valid and the only honest answer is to refuse — naming which one."""
        response = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7, **kwargs)

        assert response.results[0].probability_of_goal is None
        found = warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert len(found) == 1
        assert found[0].detail["reason"] == reason

    def test_no_threshold_requested_is_silent(self):
        """Control on the control: no threshold => no probability AND no warning.

        Without this, a warning that fired unconditionally would still pass every
        assertion above.
        """
        response = analyse(goal_threshold=None, goal_threshold_frame=None)

        assert response.results[0].probability_of_goal is None
        assert warnings_by_code(response, "GOAL_THRESHOLD_FRAME_UNSPECIFIED") == []
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []


# =============================================================================
# 3. 'delta' passthrough — the pre-2.258 behaviour, now attested
# =============================================================================


class TestDeltaPassthrough:
    def test_delta_is_used_unconverted(self):
        """Samples ~U(0,0.5); a delta threshold of 0.2 gives 1 - 0.2/0.5 = 0.60.

        The baseline is present and non-zero, so if 'delta' were wrongly
        converted the answer would be P(sample >= 0.2-0.7) = 1.0, not 0.60.
        """
        response = analyse(goal_threshold=0.2, goal_threshold_frame="delta", baseline=0.7)
        assert response.results[0].probability_of_goal == pytest.approx(0.60, abs=TOL)

    def test_delta_needs_no_baseline_at_all(self):
        response = analyse(goal_threshold=0.2, goal_threshold_frame="delta", baseline=None)
        assert response.results[0].probability_of_goal == pytest.approx(0.60, abs=TOL)
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []


# =============================================================================
# 4. Adversarial fixtures — each states its expected outcome
# =============================================================================


class TestAdversarialBaselines:
    def test_zero_baseline_is_a_value_not_a_missing_field(self):
        """B=0.0 is FALSY. A `if not baseline` check would wrongly refuse here.

        B=0.0, T=0.25 -> delta 0.25 -> P = 1 - 2*0.25 = 0.50.
        """
        response = analyse_level(goal_threshold=0.25, goal_threshold_frame="level", baseline=0.0)

        assert (
            response.results[0].probability_of_goal is not None
        ), "baseline=0.0 must be treated as a supplied value, not as absent"
        assert response.results[0].probability_of_goal == pytest.approx(0.50, abs=TOL)
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []

    def test_negative_baseline_converts(self):
        """B=-0.3, T=0.1 -> delta 0.4 -> P = 1 - 2*0.4 = 0.20."""
        response = analyse_level(goal_threshold=0.1, goal_threshold_frame="level", baseline=-0.3)
        assert response.results[0].probability_of_goal == pytest.approx(0.20, abs=TOL)

    def test_converted_threshold_may_be_negative_and_is_not_clamped(self):
        """B=0.7, T=0.6 -> delta = -0.1: the goal is ALREADY met at baseline.

        The goal starts at 0.7 and the option can only raise it, so every draw
        clears 0.6 => P = 1.0 exactly. A target already met must not be refused
        or clamped away.
        """
        response = analyse_level(goal_threshold=0.6, goal_threshold_frame="level", baseline=0.7)
        assert response.results[0].probability_of_goal == 1.0

    def test_honest_zero_is_still_reported(self):
        """B=0.7, T=1.5, best reachable level 0.7 + 0.5 = 1.2 => P = 0.0.

        This zero is TRUE — the goal is genuinely unreachable in this model — and
        must still be emitted. Fail-closed withholds UNPROVABLE numbers, never
        merely unwelcome ones.
        """
        response = analyse_level(goal_threshold=1.5, goal_threshold_frame="level", baseline=0.7)
        assert response.results[0].probability_of_goal == 0.0
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []


# =============================================================================
# 5. The resolver in isolation
# =============================================================================


class TestResolverUnit:
    """Direct tests of the arithmetic, independent of Monte Carlo."""

    @staticmethod
    def resolve(**kwargs):
        return RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(build_request(**kwargs))

    @pytest.mark.parametrize(
        "threshold,baseline,intercept",
        [
            (0.9, 0.7, 0.0),  # the witness
            (0.9, 0.7, 0.1),  # non-zero intercept
            (0.25, 0.0, 0.0),  # zero baseline
            (0.1, -0.3, 0.0),  # negative baseline
            (0.6, 0.7, 0.0),  # target already met at baseline
        ],
    )
    def test_level_plan_carries_the_threshold_and_baseline_verbatim(
        self, threshold, baseline, intercept
    ):
        """A level plan does NO arithmetic — that is the point of 2.286.

        Its predecessor asserted ``value == T - B + I``. That single number was
        the defect: collapsing the comparison to a constant known before the
        Monte Carlo runs is only possible if the samples' origin is known in
        advance, and it is not — it depends on where the factors currently sit.
        The plan therefore carries T and B forward untouched, and the anchor is
        applied per draw against the status-quo reference.
        """
        plan, warning = self.resolve(
            goal_threshold=threshold,
            goal_threshold_frame="level",
            baseline=baseline,
            intercept=intercept,
        )
        assert warning is None
        assert plan.level_threshold == pytest.approx(threshold, abs=1e-12)
        assert plan.goal_baseline == pytest.approx(baseline, abs=1e-12)
        assert plan.delta_threshold is None
        assert plan.needs_status_quo_reference is True

    def test_the_intercept_is_no_longer_an_operand(self):
        """Two requests differing only in intercept must yield the SAME plan.

        Under 2.258 the intercept entered the converted number. Under 2.286 it
        cancels in the per-draw difference, so it must not appear in the plan at
        all — if it reappears, it is being double-counted.
        """
        flat, _ = self.resolve(goal_threshold=0.9, goal_threshold_frame="level", intercept=0.0)
        raised, _ = self.resolve(goal_threshold=0.9, goal_threshold_frame="level", intercept=0.9)
        assert flat == raised

    def test_delta_returns_the_input_unchanged(self):
        plan, warning = self.resolve(goal_threshold=0.2, goal_threshold_frame="delta")
        assert plan.delta_threshold == 0.2
        assert plan.level_threshold is None
        assert plan.needs_status_quo_reference is False, (
            "a delta plan compares raw samples, so it must not pay for the "
            "status-quo reference"
        )
        assert warning is None

    def test_absent_frame_returns_none_and_a_warning(self):
        value, warning = self.resolve(goal_threshold=0.9, goal_threshold_frame=None)
        assert value is None
        assert warning is not None
        assert warning.code == "GOAL_THRESHOLD_FRAME_UNSPECIFIED"

    def test_no_threshold_returns_none_and_no_warning(self):
        assert self.resolve(goal_threshold=None) == (None, None)

    def test_result_is_always_finite_when_returned(self):
        value, warning = self.resolve(
            goal_threshold=1e308, goal_threshold_frame="level", baseline=-1e308
        )
        # Operands that would overflow are refused. Since the domain guard bounds
        # every operand by 1.5, they are caught THERE — |converted| <= 4.5 makes a
        # non-finite result unreachable, which is why no post-conversion finiteness
        # branch exists to be tested.
        assert value is None
        assert warning.detail["reason"] == "goal_values_outside_normalised_domain"

    def test_nan_operand_is_refused_despite_passing_a_magnitude_test(self):
        """`abs(nan) > 1.5` is False, so a NaN would sail through the domain guard.

        Pins that the finiteness check runs FIRST. NaN reaches the resolver only
        via a directly-constructed model (the field validator rejects a NaN
        goal_threshold at parse time), so it is built by hand here.
        """
        request = build_request(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        goal = next(n for n in request.graph.nodes if n.id == "g")
        object.__setattr__(goal.observed_state, "baseline", float("nan"))

        value, warning = RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(request)
        assert value is None
        assert warning.detail["reason"] == "non_finite_conversion_input"


# =============================================================================
# 6. Epsilon breaks the status-quo reference — widened by 2.286
# =============================================================================


class TestEpsilonBreaksTheStatusQuoReference:
    """Any epsilon that can REACH the goal now refuses, and the widening is forced.

    2.258 refused only when the GOAL carried epsilon AND the converted threshold
    escaped (0, 1], because the only hazard then was the evaluator's [0, 1] clamp
    falsifying `sample = intercept + S`.

    2.286 resolves a level threshold by DIFFERENCING each option's sample against
    a status-quo sample from the same draw, and that adds a second, independent
    hazard: `SCMEvaluatorV2.evaluate` draws epsilon inside each call, so the two
    evaluations get two independent noise vectors. Their difference would carry
    ~2x epsilon variance that no option caused — manufacturing UNCERTAINTY, which
    is the same class of untruth as manufacturing confidence. The clamp hazard is
    still present too, and is not additive.

    Neither hazard is fixable without pre-drawing epsilon per node per sample,
    which would change RNG consumption for every existing caller. So the honest
    move is the one this seam already makes everywhere else: refuse, and name it.

    Witness graph for this class (strength_mean=1.0, intercept=1.0, B=0.5,
    T=0.7), with do(f := 1.0)::

        level(g) = 0.5 + 1.0 * (1 - f)
        P(level >= 0.7) = P(1 - f >= 0.2) = 0.80        by hand
    """

    @staticmethod
    def over_unit(**kwargs):
        return analyse_level(
            goal_threshold=0.7,
            goal_threshold_frame="level",
            baseline=0.5,
            intercept=1.0,
            strength_mean=1.0,
            **kwargs,
        )

    def test_goal_epsilon_refuses(self):
        """RED-FIRST on the widening. 2.258 accepted a goal epsilon whenever the
        converted threshold happened to land inside (0, 1]; the reference cannot
        be CRN-matched through per-call noise regardless of where it lands."""
        noised = self.over_unit(goal_epsilon_std=0.001)

        assert noised.results[0].probability_of_goal is None
        refusals = warnings_by_code(noised, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert len(refusals) == 1
        assert refusals[0].detail["reason"] == "epsilon_breaks_status_quo_reference"
        assert refusals[0].detail["noisy_node_ids"] == ["g"]

    def test_parent_epsilon_also_refuses(self):
        """⚠ REVERSED BY 2.286, deliberately.

        Its predecessor was a CONTROL asserting a parent's epsilon was harmless,
        on the reasoning that it "perturbs S, which the identity already
        accommodates". Under a per-draw reference that reasoning no longer holds:
        the parent's noise is drawn twice — once for the option evaluation, once
        for the reference — so it does NOT cancel and lands in the effect
        estimate as fabricated spread.
        """
        noised = self.over_unit(parent_epsilon_std=0.001)

        assert noised.results[0].probability_of_goal is None
        refusals = warnings_by_code(noised, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert refusals[0].detail["reason"] == "epsilon_breaks_status_quo_reference"
        assert refusals[0].detail["noisy_node_ids"] == ["f"]

    def test_no_epsilon_anywhere_converts(self):
        """POSITIVE CONTROL on the fixture. Without it every assertion above
        would pass on a fixture that refuses for some unrelated reason."""
        clean = self.over_unit(goal_epsilon_std=0.0)
        assert clean.results[0].probability_of_goal == pytest.approx(0.80, abs=TOL)
        assert warnings_by_code(clean, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []

    def test_epsilon_on_a_node_that_cannot_reach_the_goal_does_not_refuse(self):
        """The guard is scoped to the goal's ANCESTORS, not to the whole graph.

        Over-refusal has its own cost — a user sees "not available" for an answer
        ISL could have given honestly — so a noisy node in a disconnected branch
        must not veto the conversion. Written with an explicit graph because the
        shared fixture has no unrelated node to make noisy.
        """
        request = build_request(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        request.graph.nodes.append(
            NodeV2(id="unrelated", kind="factor", label="Elsewhere", epsilon_std=0.5)
        )
        plan, warning = RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(request)

        assert warning is None, (
            f"a noisy node with no path to the goal must not refuse, got "
            f"{warning.detail['reason'] if warning else None}"
        )
        assert plan is not None


# =============================================================================
# 7. A4 Tier-2 — magnitude guard on the normalised domain
# =============================================================================


class TestNormalisedDomainGuard:
    """A 'level' threshold and the baseline must share a domain, and ISL cannot
    verify the producer's attestation. It CAN reject operands that are obviously
    raw user units — the failure mode fail-closed does not otherwise cover,
    because it yields a WRONG NUMBER rather than no number.

    The 1.5 bound derives from the evaluator's own [0, 1] node-value clamp
    (:1189 "keep normalised node values in valid range") plus slack.
    """

    def test_raw_user_units_are_refused(self):
        """RED-FIRST. ObservedState's own normative example is RAW: value 59.0 /
        baseline 49.0 for £59k/£49k. Paired with a normalised threshold of 0.8
        that silently produced a converted threshold of 0.8-49.0 = -48.2, i.e.
        P = 1.0 — a confident, wrong 100%."""
        response = analyse(
            goal_threshold=0.8,
            goal_threshold_frame="level",
            baseline=49.0,
            goal_observed_value=59.0,
        )
        assert response.results[0].probability_of_goal is None
        found = warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert len(found) == 1
        assert found[0].detail["reason"] == "goal_values_outside_normalised_domain"
        assert found[0].detail["out_of_domain"] == {"goal_baseline": 49.0}

    def test_legitimate_negative_baseline_still_converts(self):
        """CONTROL on the guard: it must REFUSE, never clamp, and it must not
        catch legitimate in-domain negatives. B=-0.3, T=0.1 -> P = 0.20."""
        response = analyse_level(goal_threshold=0.1, goal_threshold_frame="level", baseline=-0.3)
        assert response.results[0].probability_of_goal == pytest.approx(0.20, abs=TOL)
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []

    def test_guard_is_symmetric_in_sign(self):
        """A one-sided bound would let a large NEGATIVE raw value through."""
        value, warning = RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(
            build_request(goal_threshold=0.1, goal_threshold_frame="level", baseline=-3.0)
        )
        assert value is None
        assert warning.detail["reason"] == "goal_values_outside_normalised_domain"
        assert warning.detail["out_of_domain"] == {"goal_baseline": -3.0}

    @pytest.mark.parametrize(
        "baseline,accepted", [(1.5, True), (1.51, False), (-1.5, True), (-1.51, False)]
    )
    def test_bound_is_inclusive_at_1_5(self, baseline, accepted):
        value, _ = RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(
            build_request(goal_threshold=0.1, goal_threshold_frame="level", baseline=baseline)
        )
        assert (value is not None) is accepted

    @pytest.mark.parametrize("field", ["goal_threshold", "goal_intercept"])
    def test_every_operand_is_guarded_not_just_the_baseline(self, field):
        kwargs = {"goal_threshold": 0.1, "goal_threshold_frame": "level", "baseline": 0.5}
        kwargs["goal_threshold" if field == "goal_threshold" else "intercept"] = 90.0
        value, warning = RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(
            build_request(**kwargs)
        )
        assert value is None
        assert warning.detail["out_of_domain"] == {field: 90.0}

    def test_delta_frame_is_not_domain_guarded(self):
        """CONTROL. 'delta' is in the SAMPLES' frame, which carries no domain
        promise — ISL must not impose the normalised bound on it. Samples here are
        ~U(0, 0.5), so a delta threshold of 59.0 is legitimately P = 0.0.
        """
        response = analyse(goal_threshold=59.0, goal_threshold_frame="delta", baseline=49.0)
        assert response.results[0].probability_of_goal == 0.0
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []


# =============================================================================
# 8. ROADMAP 2.279 — GOAL_OBSERVED_VALUE_UNUSED fires only when ACTIONABLE
# =============================================================================


def _declared_frames():
    """The frame Literal, read from the model rather than hand-listed here."""
    frames: set = set()
    for arg in get_args(RobustnessRequestV2.model_fields["goal_threshold_frame"].annotation):
        frames.update(get_args(arg))
    return frames


class TestObservedValueUnusedIsActionable:
    """A warning that fires on every SUCCESS is a broken alarm.

    ``ObservedState.value`` is a REQUIRED field, so once CEE populates a goal's
    observed_state (CEE #787) every successfully-converted 'level' analysis
    carries one — and GOAL_OBSERVED_VALUE_UNUSED, which fires for any non-root
    goal whose observed_state.value is present, would fire on all of them. The
    UI renders it on five surfaces, so that is visible noise on every healthy
    goal analysis.

    The suppression is keyed to CONSUMPTION, not to "a threshold was asked
    for": when this run's conversion actually read the goal's
    observed_state.baseline, the observed_state did its job and the warning has
    nothing to say. On every run where it did NOT — no threshold, 'delta'
    frame, unstamped frame, or any convertibility refusal — the observed_state
    genuinely IS unused, the warning is actionable, and it must keep firing.
    That firing case is also the diagnostic surface for #786/#787, so blanket
    suppression would blind it.
    """

    CODE = "GOAL_OBSERVED_VALUE_UNUSED"

    def test_converted_run_drops_the_warning(self):
        """RED-FIRST (a). Before the fix this run carried the warning.

        The probability assertion is a positive control ON THE FIXTURE: it
        proves the conversion actually SUCCEEDED, so the absence below is
        suppression-on-consumption and not an analysis that quietly failed.
        """
        response = analyse_level(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)

        assert response.results[0].probability_of_goal == pytest.approx(0.60, abs=TOL), (
            "fixture control: this run must CONVERT, otherwise the assertion "
            "below would pass for the wrong reason"
        )
        assert warnings_by_code(response, self.CODE) == [], (
            "the conversion consumed observed_state.baseline, so the "
            "observed_state was used — this warning fires on every successful "
            "goal analysis and is a broken alarm"
        )

    def test_no_threshold_keeps_the_warning(self):
        """POSITIVE CONTROL (b). No threshold -> no conversion -> the
        observed_state genuinely went unused. Green BEFORE and AFTER the fix;
        this is what proves the fix is not a blanket suppression."""
        response = analyse(goal_threshold=None, goal_threshold_frame=None)

        found = warnings_by_code(response, self.CODE)
        assert len(found) == 1
        assert found[0].field == "nodes[g].observed_state.value"
        assert found[0].detail["observed_value"] == pytest.approx(0.7)
        assert found[0].detail["reason"] == "non_root_goal_forward_propagation"
        # Name and severity are deliberately untouched by 2.279.
        assert found[0].code == "GOAL_OBSERVED_VALUE_UNUSED"
        assert found[0].severity == "info"

    def test_delta_frame_keeps_the_warning(self):
        """POSITIVE CONTROL. 'delta' resolves to a non-None threshold WITHOUT
        reading the baseline, so the observed_state is still unused. Pins that
        the predicate keys on consumption, not on "a threshold resolved"."""
        response = analyse(goal_threshold=0.2, goal_threshold_frame="delta", baseline=0.7)

        assert response.results[0].probability_of_goal == pytest.approx(0.60, abs=TOL)
        assert len(warnings_by_code(response, self.CODE)) == 1

    @pytest.mark.parametrize(
        "kwargs,reason",
        [
            ({"goal_threshold_frame": None}, "frame_not_stamped"),
            ({"goal_threshold_frame": "level", "baseline": None}, "missing_goal_baseline"),
            (
                {"goal_threshold_frame": "level", "baseline": 59.0},
                "goal_values_outside_normalised_domain",
            ),
        ],
    )
    def test_every_refusal_keeps_the_warning(self, kwargs, reason):
        """POSITIVE CONTROLS. A conversion that REFUSED consumed nothing, so the
        observed_state is genuinely unused and the warning is actionable.

        These are also #786/#787's diagnostic surface — suppressing here would
        blind the exact runs an operator needs to see.
        """
        response = analyse(goal_threshold=0.9, **kwargs)

        refusals = [
            w
            for w in response.inference_warnings
            if w.code in ("GOAL_THRESHOLD_NOT_CONVERTIBLE", "GOAL_THRESHOLD_FRAME_UNSPECIFIED")
        ]
        assert [w.detail["reason"] for w in refusals] == [reason], "fixture control"
        assert len(warnings_by_code(response, self.CODE)) == 1

    def test_epsilon_refusal_keeps_the_warning(self):
        """POSITIVE CONTROL. The epsilon refusal reads the baseline on its way to
        the guard, then refuses — nothing was CONSUMED because no plan was
        returned, so the warning stays."""
        response = analyse(
            goal_threshold=0.6,
            goal_threshold_frame="level",
            baseline=0.7,
            goal_epsilon_std=0.001,
        )

        refusals = warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert [w.detail["reason"] for w in refusals] == [
            "epsilon_breaks_status_quo_reference"
        ], "fixture control"
        assert len(warnings_by_code(response, self.CODE)) == 1

    def test_only_this_warning_differs_between_level_and_delta(self):
        """CONTROL ON THE BLAST RADIUS. A 'level' run and a 'delta' run must
        differ by EXACTLY this one code — nothing else suppressed, added or
        reordered.

        2.286 note: the two no longer produce the same NUMBER, and must not be
        asserted to. 'delta' asks whether the model's propagated sum clears 0.2;
        'level' asks whether this option gets the goal to 0.9. Under 2.258 those
        two questions were conflated, which is precisely the defect. Both are
        asserted to be present, so this stays a control on warning codes rather
        than passing because one of them silently declined to answer.
        """
        level = analyse_level(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        delta = analyse(goal_threshold=0.2, goal_threshold_frame="delta", baseline=0.7)

        assert level.results[0].probability_of_goal is not None, "fixture control"
        assert delta.results[0].probability_of_goal is not None, "fixture control"
        level_codes = [w.code for w in level.inference_warnings]
        delta_codes = [w.code for w in delta.inference_warnings]
        assert level_codes == [c for c in delta_codes if c != "GOAL_OBSERVED_VALUE_UNUSED"]

    def test_root_goal_is_untouched(self):
        """CONTROL. A root goal never emitted this warning (only non-root goals
        do) and still does not — 2.279 changes no other limb."""
        response = analyse(
            goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7, goal_is_root=True
        )
        assert warnings_by_code(response, self.CODE) == []


class TestConsumptionPredicate:
    """The predicate in isolation, and a guard against it going stale."""

    def test_every_declared_frame_is_deliberately_classified(self):
        """DRIFT GUARD (trap 12). The predicate treats 'level' as the only
        baseline-consuming frame. If a third frame is ever added to the model,
        this REDs instead of silently landing on the wrong side of it."""
        assert _declared_frames() == {"level", "delta"}, (
            "goal_threshold_frame gained or lost a value — re-derive whether it "
            "reads observed_state.baseline and update "
            "RobustnessAnalyzerV2._goal_baseline_was_consumed"
        )

    @pytest.mark.parametrize("frame,consumed", [("level", True), ("delta", False)])
    def test_predicate_classifies_each_declared_frame(self, frame, consumed):
        request = build_request(goal_threshold=0.2, goal_threshold_frame=frame, baseline=0.7)
        value, warning = RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(request)

        assert value is not None and warning is None, "fixture control: both frames resolve"
        assert RobustnessAnalyzerV2._goal_baseline_was_consumed(request, value) is consumed

    def test_unresolved_threshold_is_never_consumed(self):
        """A refusal returns None; None can never count as consumption."""
        request = build_request(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        assert RobustnessAnalyzerV2._goal_baseline_was_consumed(request, None) is False
