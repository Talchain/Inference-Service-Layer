"""ROADMAP 2.258 — goal_threshold FRAME conversion.

THE DEFECT. CEE mints ``goal_threshold`` as a normalised LEVEL (0.8 == a
GBP 6.0m target against a GBP 7.5m cap). ISL's goal samples, for a NON-ROOT
goal, are the forward-propagated composition of the goal's parents measured from
an origin of ``intercept`` (0.0 by default) — a CHANGE, not a level. Nobody
converted, so ``probability_of_goal`` computed "P(change >= level)": a
STRUCTURAL zero, rendered to users as "< 1% chance of hitting your goal".

THE ARITHMETIC, derived from ``SCMEvaluatorV2.evaluate`` and pinned
by the witness below::

    sample      = intercept + S            (S = parents' propagated contribution)
    real_level  = baseline  + S
    => sample >= level_threshold - baseline + intercept

    delta_threshold = level_threshold - goal_baseline + goal_intercept

THE WITNESS GRAPH used throughout this module, chosen so every expected
probability is computable BY HAND rather than read back off the code::

    f  root factor, ParameterUncertainty uniform(0, 1)
    g  goal, NON-ROOT outcome, observed_state{baseline: B}, intercept I
    f -> g  strength mean 0.5, std 0.0011 (near-deterministic), exists_prob 1.0

    sample(g) = 0 (non-root base) + I + f * 0.5    ~  Uniform(I, I + 0.5)

    P(sample >= T - B + I)
        = P(I + 0.5f >= T - B + I)
        = P(f >= 2 * (T - B))
        = 1 - 2 * (T - B)                for 0 <= 2 * (T - B) <= 1

Note the intercept I CANCELS. That is not a coincidence — the conversion adds
back exactly what the samples were shifted by — and it is the property that
makes ``test_intercept_cancels_out`` bite if the ``+ intercept`` term is dropped.

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
                observed_state=ObservedState(value=0.5),
            ),
        )
        edges.append(
            EdgeV2(
                **{"from": "f", "to": "g"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=strength_mean, std=0.0011),
            )
        )
        uncertainties.append(
            ParameterUncertainty(node_id="f", distribution="uniform", range_min=0.0, range_max=1.0)
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
                interventions={"g": 0.5} if intervene_on_goal else {},
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


def warnings_by_code(response, code):
    return [w for w in response.inference_warnings if w.code == code]


# =============================================================================
# 1. The witness — hand-computed, and the RED-first pin on the defect
# =============================================================================


class TestFrameConversionWitness:
    """The conversion produces the BY-HAND probability, not the structural zero."""

    def test_level_threshold_yields_hand_computed_probability(self):
        """B=0.7, T=0.9, I=0 -> delta=0.2 -> P = 1 - 2*(0.9-0.7) = 0.60.

        RED-FIRST PIN. Before the converter, this same request compared samples
        (~U(0, 0.5)) against the raw LEVEL 0.9 and returned exactly 0.0 — the
        structural zero of the defect. Reverting the conversion turns this
        assertion RED at 0.0 vs 0.60.
        """
        response = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        prob = response.results[0].probability_of_goal

        assert prob is not None, "a convertible level threshold must produce a probability"
        assert prob == pytest.approx(0.60, abs=TOL), (
            f"expected the hand-computed 0.60, got {prob}. A value of 0.0 means the "
            f"raw LEVEL was compared against change-from-origin samples (the 2.258 defect)."
        )

    def test_samples_are_change_from_origin_not_levels(self):
        """The premise of the whole row, asserted rather than assumed.

        The goal's own level (baseline 0.7) lies OUTSIDE the sample range, so the
        samples cannot be levels of the goal quantity.
        """
        response = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        samples = response.results[0].outcome_distribution.samples

        assert min(samples) == pytest.approx(0.0, abs=0.01)
        assert max(samples) == pytest.approx(0.5, abs=0.01)
        assert not (min(samples) <= 0.7 <= max(samples)), (
            "the goal's baseline level sits inside the sample range — the frame "
            "premise of ROADMAP 2.258 no longer holds and this converter needs re-deriving"
        )

    def test_conversion_changes_only_the_comparison_not_the_samples(self):
        """Control: the converter must not perturb sampling.

        Same seed, one request stamped 'level' and one 'delta' — identical
        sample series, different probabilities.
        """
        level = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        delta = analyse(goal_threshold=0.2, goal_threshold_frame="delta", baseline=0.7)

        assert (
            level.results[0].outcome_distribution.samples
            == delta.results[0].outcome_distribution.samples
        ), "the frame field must not touch sampling"
        # 0.9 - 0.7 == 0.2, so the two must agree exactly on the probability too.
        assert level.results[0].probability_of_goal == delta.results[0].probability_of_goal

    def test_intercept_cancels_out(self):
        """delta = T - B + I, so a non-zero goal intercept must NOT move the answer.

        Samples shift to U(I, I+0.5) and the threshold shifts by the same I.
        Drop the '+ intercept' term and this reds: P becomes 1-2*(T-B-I) = 0.80.
        """
        without = analyse(
            goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7, intercept=0.0
        )
        with_intercept = analyse(
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
        response = analyse(goal_threshold=0.25, goal_threshold_frame="level", baseline=0.0)

        assert (
            response.results[0].probability_of_goal is not None
        ), "baseline=0.0 must be treated as a supplied value, not as absent"
        assert response.results[0].probability_of_goal == pytest.approx(0.50, abs=TOL)
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []

    def test_negative_baseline_converts(self):
        """B=-0.3, T=0.1 -> delta 0.4 -> P = 1 - 2*0.4 = 0.20."""
        response = analyse(goal_threshold=0.1, goal_threshold_frame="level", baseline=-0.3)
        assert response.results[0].probability_of_goal == pytest.approx(0.20, abs=TOL)

    def test_converted_threshold_may_be_negative_and_is_not_clamped(self):
        """B=0.7, T=0.6 -> delta = -0.1: the goal is ALREADY met at baseline.

        A negative converted threshold is legitimate, so it must not be clamped
        to 0 or refused. Every sample (>= 0) clears -0.1 => P = 1.0 exactly.
        """
        response = analyse(goal_threshold=0.6, goal_threshold_frame="level", baseline=0.7)
        assert response.results[0].probability_of_goal == 1.0

    def test_honest_zero_is_still_reported(self):
        """B=0.7, T=1.5 -> delta 0.8, above every sample (max 0.5) => P = 0.0.

        This zero is TRUE — the goal is genuinely unreachable in this model — and
        must still be emitted. Fail-closed withholds UNPROVABLE numbers, never
        merely unwelcome ones.
        """
        response = analyse(goal_threshold=1.5, goal_threshold_frame="level", baseline=0.7)
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
        "threshold,baseline,intercept,expected",
        [
            (0.9, 0.7, 0.0, 0.2),  # the witness
            (0.9, 0.7, 0.1, 0.3),  # + intercept
            (0.25, 0.0, 0.0, 0.25),  # zero baseline
            (0.1, -0.3, 0.0, 0.4),  # negative baseline
            (0.6, 0.7, 0.0, -0.1),  # negative result, unclamped
        ],
    )
    def test_arithmetic(self, threshold, baseline, intercept, expected):
        value, warning = self.resolve(
            goal_threshold=threshold,
            goal_threshold_frame="level",
            baseline=baseline,
            intercept=intercept,
        )
        assert warning is None
        assert value == pytest.approx(expected, abs=1e-12)

    def test_delta_returns_the_input_unchanged(self):
        value, warning = self.resolve(goal_threshold=0.2, goal_threshold_frame="delta")
        assert value == 0.2
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
# 6. A1 — goal-node epsilon_std clamps samples to [0, 1]
# =============================================================================


class TestGoalEpsilonClamp:
    """SCMEvaluatorV2 clamps an epsilon-noised node to [0, 1] (:1187-1189).

    That clamp FALSIFIES `sample = intercept + S`, so it breaks the conversion
    identity. Witness graph for this class (strength_mean=1.0, intercept=1.0)::

        sample(g) = 0 + 1.0 + f * 1.0  ~  Uniform(1, 2)     when eps == 0
        B = 0.5, T = 0.7  ->  converted = 0.7 - 0.5 + 1.0 = 1.20
        P(sample >= 1.20), sample ~ U(1, 2)  =  0.80        by hand

    With eps > 0 on the goal every sample is clamped to <= 1.0, so P collapses to
    a silent 0.0 — the 2.258 untruth, re-manufactured by the converter itself.
    """

    @staticmethod
    def over_unit(**kwargs):
        """Converted threshold 1.20, i.e. OUTSIDE (0, 1]."""
        return analyse(
            goal_threshold=0.7,
            goal_threshold_frame="level",
            baseline=0.5,
            intercept=1.0,
            strength_mean=1.0,
            **kwargs,
        )

    def test_goal_epsilon_over_unit_interval_refuses_instead_of_faking_zero(self):
        """RED-FIRST. Before this guard the same request returned 0.0 with ZERO
        warnings: measured 0.7967 at eps=0.0 and 0.0 at eps=0.001."""
        clean = self.over_unit(goal_epsilon_std=0.0)
        assert clean.results[0].probability_of_goal == pytest.approx(0.80, abs=TOL)

        noised = self.over_unit(goal_epsilon_std=0.001)
        assert noised.results[0].probability_of_goal is None, (
            "a goal epsilon clamps samples to [0,1]; a converted threshold of 1.20 "
            "can then only ever return a manufactured 0.0"
        )
        found = warnings_by_code(noised, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert len(found) == 1
        assert found[0].detail["reason"] == "goal_epsilon_noise_clamps_samples"
        assert found[0].field == "nodes[g].epsilon_std"

    def test_goal_epsilon_inside_unit_interval_still_converts(self):
        """Inside (0, 1] the clamp provably cannot change P, so refusing would be
        over-broad. B=0.5, T=0.9, I=0, strength 1.0 -> samples ~U(0,1),
        converted 0.4 -> P = 1 - 0.4 = 0.60."""
        response = analyse(
            goal_threshold=0.9,
            goal_threshold_frame="level",
            baseline=0.5,
            strength_mean=1.0,
            goal_epsilon_std=0.001,
        )
        assert response.results[0].probability_of_goal == pytest.approx(0.60, abs=TOL)
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []

    def test_parent_epsilon_does_not_trigger_the_guard(self):
        """CONTROL. Only the GOAL's epsilon clamps the goal's samples; a parent's
        epsilon perturbs S, which the identity already accommodates."""
        response = self.over_unit(parent_epsilon_std=0.001)
        assert response.results[0].probability_of_goal == pytest.approx(0.80, abs=TOL)
        assert warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE") == []

    def test_no_goal_epsilon_means_no_guard_at_all(self):
        """CONTROL. Without a goal epsilon there is no clamp, so a converted
        threshold outside (0, 1] is perfectly legitimate and must convert."""
        response = self.over_unit(goal_epsilon_std=0.0)
        assert response.results[0].probability_of_goal == pytest.approx(0.80, abs=TOL)

    @pytest.mark.parametrize(
        "threshold,baseline,converted,accepted",
        [
            (1.0, 0.0, 1.0, True),  # converted == 1 -> ACCEPTED (pins `<= 1`)
            (0.9, 0.9, 0.0, False),  # converted == 0 -> REFUSED  (pins `0 <`)
        ],
    )
    def test_unit_interval_boundaries(self, threshold, baseline, converted, accepted):
        value, warning = RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(
            build_request(
                goal_threshold=threshold,
                goal_threshold_frame="level",
                baseline=baseline,
                goal_epsilon_std=0.001,
            )
        )
        assert value == pytest.approx(converted, abs=1e-12) if accepted else value is None
        if not accepted:
            assert warning.detail["reason"] == "goal_epsilon_noise_clamps_samples"


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
        response = analyse(goal_threshold=0.1, goal_threshold_frame="level", baseline=-0.3)
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
        response = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)

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

    def test_epsilon_clamp_refusal_keeps_the_warning(self):
        """POSITIVE CONTROL. The clamp refusal reads the baseline to compute the
        candidate, then refuses — nothing was CONSUMED because no threshold was
        returned, so the warning stays."""
        response = analyse(
            goal_threshold=0.6,
            goal_threshold_frame="level",
            baseline=0.7,
            goal_epsilon_std=0.001,
        )

        refusals = warnings_by_code(response, "GOAL_THRESHOLD_NOT_CONVERTIBLE")
        assert [w.detail["reason"] for w in refusals] == [
            "goal_epsilon_noise_clamps_samples"
        ], "fixture control"
        assert len(warnings_by_code(response, self.CODE)) == 1

    def test_only_this_warning_differs_between_level_and_delta(self):
        """CONTROL ON THE BLAST RADIUS. The same comparison expressed as a
        converted 'level' and as an equivalent 'delta' must differ by EXACTLY
        this one code — nothing else was suppressed, added or reordered."""
        level = analyse(goal_threshold=0.9, goal_threshold_frame="level", baseline=0.7)
        delta = analyse(goal_threshold=0.2, goal_threshold_frame="delta", baseline=0.7)

        assert level.results[0].probability_of_goal == pytest.approx(
            delta.results[0].probability_of_goal, abs=1e-12
        )
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
