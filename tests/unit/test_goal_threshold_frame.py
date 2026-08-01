"""ROADMAP 2.258 — goal_threshold FRAME conversion.

THE DEFECT. CEE mints ``goal_threshold`` as a normalised LEVEL (0.8 == a
GBP 6.0m target against a GBP 7.5m cap). ISL's goal samples, for a NON-ROOT
goal, are the forward-propagated composition of the goal's parents measured from
an origin of ``intercept`` (0.0 by default) — a CHANGE, not a level. Nobody
converted, so ``probability_of_goal`` computed "P(change >= level)": a
STRUCTURAL zero, rendered to users as "< 1% chance of hitting your goal".

THE ARITHMETIC, derived from ``StructuralEquationEvaluator.evaluate`` and pinned
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

import math

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
            observed_state=goal_observed,
        )
    ]
    edges = []
    uncertainties = []
    if not goal_is_root:
        nodes.insert(
            0,
            NodeV2(id="f", kind="factor", label="Driver", observed_state=ObservedState(value=0.5)),
        )
        edges.append(
            EdgeV2(
                **{"from": "f", "to": "g"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.5, std=0.0011),
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
        # Overflow to inf must be refused, never returned as a comparison operand.
        assert value is None or math.isfinite(value)
        if value is None:
            assert warning.detail["reason"] == "non_finite_converted_threshold"
