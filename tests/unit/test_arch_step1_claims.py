"""Architecture step 1 — "stop the false claims".

Normative tests for the five adjudicated science-claim defects. Every assertion
here states the CORRECTED truth, not the behaviour at
``3aea011c88ef25461c7ab99bba8b24964c14e943``. Written RED-first: each one failed
against that commit.

Evidence for each defect: ``CODEX-SCIENCE-CLAIMS-VERIFY-2026-07-26.md``
(adjudicated read-only lane, file:line verified at the bytes).

The five items:

1. ``value_of_flexibility`` — OMITTED. ``_calculate_average_continuation``
   averaged chance branches with ``np.mean`` while the flexible leg
   probability-weighted them, so the field reported the gap between two
   different estimators of the SAME quantity as economic value.
2. ``RobustnessResult.confidence`` — an uncalibrated heuristic
   ``min(.99, stability*(1-1/sqrt(n)))`` presented as a confidence level.
3. Counterfactual "95% confidence interval" — a 2.5/97.5 percentile band over
   Monte-Carlo draws, i.e. a prediction interval.
4. Auto-scaled noise (~sqrt(2) spread inflation) — was unconditional for every
   request that reached it, with a single boolean as its whole disclosure.
5. Multi-decision-per-stage — the request permitted 20 decision nodes per stage
   and the engine used the first only, silently.
"""

import math

import pytest

from pydantic import ValidationError

from src.models.requests import (
    DecisionStage,
    SequentialAnalysisRequest,
    SequentialGraph,
    SequentialGraphEdge,
    SequentialGraphNode,
)
from src.models.responses import SequentialAnalysisResponse
from src.services.sequential_decision import SequentialDecisionEngine

# ---------------------------------------------------------------------------
# Item 1 — value_of_flexibility
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return SequentialDecisionEngine()


@pytest.fixture
def no_future_choice_graph():
    """The adjudication's reproduction graph.

    ``D0`` is the ONLY decision node in the whole graph. Nothing is ever chosen
    after the chance node resolves, so information arriving later cannot change
    any action, so the value of flexibility is definitionally 0::

        D0 (stage 0, decision)
          --"invest"--> C (stage 1, chance) --p=0.9--> T_hi  payoff 100
                                            --p=0.1--> T_lo  payoff   0
          --"abort" --> T_abort payoff 10

    At ``3aea011`` the engine reported 40.0 here: ``mean(100, 0) = 50`` for the
    committed leg against ``0.9*100 + 0.1*0 = 90`` for the flexible one.
    """
    nodes = [
        SequentialGraphNode(id="D0", type="decision", label="Invest decision"),
        SequentialGraphNode(id="C", type="chance", label="Market"),
        SequentialGraphNode(id="T_hi", type="terminal", label="Boom", payoff=100),
        SequentialGraphNode(id="T_lo", type="terminal", label="Bust", payoff=0),
        SequentialGraphNode(id="T_abort", type="terminal", label="Abort", payoff=10),
    ]
    edges = [
        SequentialGraphEdge(from_node="D0", to_node="C", action="invest"),
        SequentialGraphEdge(from_node="D0", to_node="T_abort", action="abort"),
        SequentialGraphEdge(from_node="C", to_node="T_hi", outcome="boom", probability=0.9),
        SequentialGraphEdge(from_node="C", to_node="T_lo", outcome="bust", probability=0.1),
    ]
    graph = SequentialGraph(
        nodes=nodes,
        edges=edges,
        stage_assignments={"D0": 0, "C": 1, "T_hi": 1, "T_lo": 1, "T_abort": 1},
    )
    stages = [
        DecisionStage(stage_index=0, stage_label="Invest decision", decision_nodes=["D0"]),
        DecisionStage(stage_index=1, stage_label="Resolution", decision_nodes=[]),
    ]
    return graph, stages


class TestValueOfFlexibilityOmitted:
    """The field is gone from the served contract until a correct estimator ships."""

    def test_field_absent_from_response_model(self):
        """``value_of_flexibility`` must not be a field of the response model.

        Presence, not value: this is what pins the omission. A future re-add
        must come WITH the invariant test below un-xfailed.
        """
        assert "value_of_flexibility" not in SequentialAnalysisResponse.model_fields, (
            "value_of_flexibility is served again. It may only return with an "
            "estimator that satisfies test_no_future_choice_implies_zero_flexibility "
            "(remove that test's xfail marker at the same time)."
        )

    def test_sensitivity_to_timing_absent_from_response_model(self):
        """``sensitivity_to_timing`` was a pure function of the same broken number.

        ``_assess_timing_sensitivity`` bucketed ``value_of_flexibility /
        |best_stage_0_value|`` into high/medium/low. Re-deriving the dependent
        claim: if the input is unsupported, so is the label built from it.
        """
        assert "sensitivity_to_timing" not in SequentialAnalysisResponse.model_fields, (
            "sensitivity_to_timing is served again, but it is a relabelling of "
            "value_of_flexibility — it cannot return before that field does."
        )

    def test_omission_does_not_break_the_rest_of_the_analysis(self, engine, no_future_choice_graph):
        """Positive control: the exact analysis is untouched by the omission.

        Backward induction still returns the correct policy value
        (0.9*100 + 0.1*0 = 90 beats abort's 10), so this test proves the
        two absence assertions above are about the omitted fields specifically
        and not about a response that failed to build.
        """
        graph, stages = no_future_choice_graph
        request = SequentialAnalysisRequest(
            graph=graph, stages=stages, discount_factor=1.0, risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        assert result.optimal_policy.expected_total_value == pytest.approx(90.0)
        assert result.optimal_policy.stages[0].decision_rule.default_action == "invest"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "value_of_flexibility is OMITTED (arch step 1). At 3aea011 this graph "
            "returned 40.0 where the true value is 0. Note also that under a "
            "correct estimator on THIS request schema the field is identically 0 "
            "for every input — see the PR body. When a meaningful estimator ships "
            "(it needs information sets the schema does not currently express), "
            "remove this marker."
        ),
    )
    def test_no_future_choice_implies_zero_flexibility(self, engine, no_future_choice_graph):
        """INVARIANT: a graph with no choice after stage 0 has zero flexibility value.

        This replaces ``assert result.value_of_flexibility >= 0``, which was
        unfalsifiable by construction — ``_compute_value_of_flexibility`` clamped
        with ``max(0, ...)`` before returning, so ``>= 0`` could not fail.

        Unlike that assertion, this one has a hand-derivable answer and would
        have caught the defect.
        """
        graph, stages = no_future_choice_graph
        request = SequentialAnalysisRequest(
            graph=graph, stages=stages, discount_factor=1.0, risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        assert result.value_of_flexibility == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Item 5 — one decision node per stage, enforced at the request boundary
# ---------------------------------------------------------------------------


class TestOneDecisionNodePerStage:
    """The engine solves one decision rule per stage; the request must say so."""

    def test_two_decision_nodes_in_one_stage_is_rejected(self):
        """>1 decision node per stage is a typed validation error, not silent truncation.

        At 3aea011 ``decision_nodes`` carried ``max_length=20`` while
        ``_build_policy`` ``break``-ed after the first node it resolved. Nodes
        2..20 were dropped with no warning, no InferenceWarning and no status
        field — and ``StagePolicy`` has no shape in which the truncation could
        even be disclosed.
        """
        with pytest.raises(ValidationError) as exc_info:
            DecisionStage(
                stage_index=0,
                stage_label="Two decisions at once",
                decision_nodes=["decide_a", "decide_b"],
            )

        message = str(exc_info.value)
        assert "decision_nodes" in message
        assert "MULTI_DECISION_STAGE_UNSUPPORTED" in message, (
            "The rejection must carry a stable machine-readable code so a client "
            "can distinguish it from a generic length error."
        )

    def test_one_decision_node_is_accepted(self):
        """Positive control for the validator: the supported shape still passes."""
        stage = DecisionStage(
            stage_index=0, stage_label="Single decision", decision_nodes=["decide_a"]
        )
        assert stage.decision_nodes == ["decide_a"]

    def test_zero_decision_nodes_is_accepted(self):
        """Positive control: a terminal stage declares no decision node."""
        stage = DecisionStage(stage_index=2, stage_label="Terminal", decision_nodes=[])
        assert stage.decision_nodes == []

    def test_declared_max_length_matches_what_the_engine_solves(self):
        """The published contract must not advertise capacity the engine lacks.

        ``max_length`` on the field is what a consumer reads out of
        ``openapi.json`` (it surfaces as ``maxItems``). It said 20.
        """
        field = DecisionStage.model_fields["decision_nodes"]
        max_lengths = [meta.max_length for meta in field.metadata if hasattr(meta, "max_length")]
        assert max_lengths == [1], (
            f"decision_nodes advertises max_length={max_lengths}; the engine builds "
            "exactly one decision rule per stage, so the contract must say 1."
        )


# ---------------------------------------------------------------------------
# Item 3 — counterfactual interval is a prediction interval
# ---------------------------------------------------------------------------


class TestCounterfactualIntervalLabelling:
    """A 2.5/97.5 percentile band over MC draws is not a confidence interval."""

    def _explanation(self):
        from src.services.explanation_generator import ExplanationGenerator

        return ExplanationGenerator().generate_counterfactual_explanation(
            outcome="revenue",
            intervention={"spend": 100.0},
            point_estimate=50.0,
            ci_lower=40.0,
            ci_upper=60.0,
            uncertainty_level="low",
        )

    def test_reasoning_says_prediction_interval(self):
        """The prose names the object it actually computed."""
        reasoning = self._explanation().reasoning
        assert "95% prediction interval" in reasoning, reasoning

    def test_reasoning_does_not_say_confidence_interval(self):
        """The mislabel is gone.

        The bounds carry no frequentist coverage guarantee over the estimand —
        they describe the spread of the model's own simulated outcomes.
        """
        reasoning = self._explanation().reasoning
        assert "confidence interval" not in reasoning.lower(), reasoning

    def test_technical_basis_does_not_say_ci(self):
        technical = self._explanation().technical_basis
        assert "CI:" not in technical, technical
        assert "PI:" in technical, technical

    def test_low_uncertainty_prose_does_not_claim_high_confidence(self):
        """The uncertainty level is a bare coefficient-of-variation bucket.

        ``counterfactual_engine._analyze_uncertainty`` thresholds ``std/|mean|``
        at 0.1 / 0.3. That is a dispersion measure of the simulated outcome; it
        is not evidence about whether the prediction is right, so it cannot
        license the phrase "high confidence".
        """
        reasoning = self._explanation().reasoning
        assert "high confidence" not in reasoning.lower(), reasoning
        assert "low simulated dispersion" in reasoning.lower(), reasoning

    def test_high_uncertainty_still_warns(self):
        """Positive control: the caution branch survives the rewording."""
        from src.services.explanation_generator import ExplanationGenerator

        explanation = ExplanationGenerator().generate_counterfactual_explanation(
            outcome="revenue",
            intervention={"spend": 100.0},
            point_estimate=50.0,
            ci_lower=0.0,
            ci_upper=100.0,
            uncertainty_level="high",
        )
        assert "caution" in explanation.reasoning.lower()

    def test_discarded_sample_defect_is_documented_in_code(self):
        """The adaptive-sampling defect is NOT fixed here — but it is recorded.

        ``_run_adaptive_monte_carlo`` throws away every accumulated batch and
        re-draws from an advanced RNG state, so the ``cv < 0.1`` convergence
        test was evaluated on a population that no longer exists and the
        returned samples were never tested for convergence. Fixing that changes
        numbers and needs its own careful change; leaving it undocumented in the
        code does not.
        """
        import inspect

        from src.services.counterfactual_engine import CounterfactualEngine

        source = inspect.getsource(CounterfactualEngine._run_adaptive_monte_carlo)
        assert (
            "KNOWN DEFECT" in source
        ), "The discarded-batch defect must stay flagged at the site until it is fixed."
        assert (
            "CODEX-SCIENCE-CLAIMS-VERIFY-2026-07-26" in source
        ), "The in-code note must cite the adjudication that established the defect."


# ---------------------------------------------------------------------------
# Item 4 — auto-scaled noise is default-off
# ---------------------------------------------------------------------------


class TestAutoScaledNoiseDefaultOff:
    """The ~sqrt(2) spread inflation is a PoC heuristic pending calibration.

    Its own docstring says so ("Status: PoC heuristic ... Pending formal review
    and calibration against pilot outcome data"). An uncalibrated widening of
    every served interval must be opt-in, not the default a client cannot decline.
    """

    def test_settings_default_is_off(self):
        from src.config import Settings

        assert (
            Settings().ENABLE_AUTO_SCALED_NOISE is False
        ), "Auto-scaled noise must default to OFF until it is calibrated."

    def test_analyzer_does_not_noise_by_default(self):
        """With default settings the served samples are the model's own."""
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2, SeededRNG

        analyzer = RobustnessAnalyzerV2()
        samples = {"a": [1.0, 2.0, 3.0, 4.0]}
        node = _outcome_node("goal")

        out, applied = analyzer._apply_auto_scaled_noise(
            {k: list(v) for k, v in samples.items()}, "goal", [node], SeededRNG(7)
        )

        assert applied is False
        assert out["a"] == samples["a"]

    def test_explicit_opt_in_still_applies_noise(self):
        """Positive control: the capability is disabled, not deleted."""
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2, SeededRNG

        analyzer = RobustnessAnalyzerV2()
        samples = {"a": [1.0, 2.0, 3.0, 4.0]}
        node = _outcome_node("goal")

        out, applied = analyzer._apply_auto_scaled_noise(
            {k: list(v) for k, v in samples.items()},
            "goal",
            [node],
            SeededRNG(7),
            noise_multiplier=1.0,
            enabled=True,
        )

        assert applied is True
        assert out["a"] != samples["a"]
        assert all(math.isfinite(v) for v in out["a"])


class TestNoiseProvenanceDisclosure:
    """One boolean cannot say WHICH of the metrics in front of you were noised."""

    def test_provenance_field_exists_on_the_envelope(self):
        from src.models.response_v2 import ISLResponseV2

        assert "sample_population_provenance" in ISLResponseV2.model_fields

    def test_provenance_names_the_pre_noise_metrics(self):
        """expected_regret / win_probability / factor_evppi / factor_evpc are
        computed from the PRE-noise CRN-aligned population (see the B2 CRN-fix
        comment in ``analyze``); p10/p50/p90/mean/cvar_10/p05 come from the
        POST-noise one. Both ride in one envelope."""
        from src.utils.response_builder import POST_NOISE_METRICS, PRE_NOISE_METRICS

        assert "expected_regret" in PRE_NOISE_METRICS
        assert "win_probability" in PRE_NOISE_METRICS
        assert "p10" in POST_NOISE_METRICS
        assert "cvar_10" in POST_NOISE_METRICS
        assert not (
            set(PRE_NOISE_METRICS) & set(POST_NOISE_METRICS)
        ), "A metric cannot come from both populations."


# ---------------------------------------------------------------------------
# Item 2 — the heuristic "confidence"
# ---------------------------------------------------------------------------


class TestHeuristicConfidenceDeLabelled:
    """``min(.99, stability*(1-1/sqrt(n)))`` is not a confidence level.

    Nothing calibrates it. The ``(1 - 1/sqrt(n))`` term is a monotone function
    of sample COUNT alone — at n=1000 it is a fixed 0.968 factor no matter what
    the estimator's actual sampling error is — so the published number moved
    with how long the simulation ran, not with how likely the recommendation was
    to be right.

    Positive control that this codebase does document calibration when it has
    it: ``FactorSensitivityV2.confidence`` ships a mandatory
    ``ConfidenceProvenance`` marker carrying ``method_version`` and a
    ``calibrated`` boolean, enforced by an iff-invariant
    (``_confidence_provenance_iff_confidence``). The robustness ``confidence``
    had no such marker and no calibration behind it.
    """

    def test_sample_count_no_longer_moves_the_number(self):
        """The served figure is the stability fraction, not a sample-count function.

        RED at 3aea011: 0.9 * (1 - 1/sqrt(100)) = 0.81 vs
        0.9 * (1 - 1/sqrt(10000)) = 0.891 — the same recommendation, the same
        stability, two different "confidences", because only n changed.
        """
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        low_n = RobustnessAnalyzerV2._stability_confidence_figure(0.9, 100)
        high_n = RobustnessAnalyzerV2._stability_confidence_figure(0.9, 10000)

        assert low_n == high_n == pytest.approx(0.9)

    def test_figure_equals_recommendation_stability(self):
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        for stability in (0.0, 0.42, 0.7, 1.0):
            assert RobustnessAnalyzerV2._stability_confidence_figure(
                stability, 1000
            ) == pytest.approx(stability)

    def test_interpretation_does_not_claim_a_confidence_level(self):
        """The user-facing string must not say "ROBUST with X% confidence"."""
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        interpretation = RobustnessAnalyzerV2._build_robustness_interpretation(
            is_robust=True,
            recommendation_stability=0.92,
            most_frequent_winner="option_a",
            fragile_edges=[],
        )

        assert "confidence" not in interpretation.lower(), interpretation
        assert "92%" in interpretation, interpretation
        assert "scenario" in interpretation.lower(), interpretation

    def test_interpretation_still_distinguishes_the_three_verdicts(self):
        """Positive control: the de-labelling did not flatten the verdicts."""
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        robust = RobustnessAnalyzerV2._build_robustness_interpretation(
            is_robust=True,
            recommendation_stability=0.92,
            most_frequent_winner="a",
            fragile_edges=["e1"],
        )
        moderate = RobustnessAnalyzerV2._build_robustness_interpretation(
            is_robust=False,
            recommendation_stability=0.6,
            most_frequent_winner="a",
            fragile_edges=["e1"],
        )
        fragile = RobustnessAnalyzerV2._build_robustness_interpretation(
            is_robust=False,
            recommendation_stability=0.3,
            most_frequent_winner="a",
            fragile_edges=["e1"],
        )

        assert "ROBUST" in robust and "MODERATELY" not in robust
        assert "MODERATELY ROBUST" in moderate
        assert "FRAGILE" in fragile
        assert "e1" in robust  # the fragile-edge list still rides

    @pytest.mark.parametrize(
        "model_path,model_name",
        [
            ("src.models.robustness_v2", "RobustnessResult"),
            ("src.models.response_v2", "RobustnessResultV2"),
        ],
    )
    def test_field_description_denies_the_confidence_reading(self, model_path, model_name):
        """The published contract must not sell a stability fraction as confidence.

        On the live wire the V2 description had degraded all the way to
        "Confidence [0, 1]" — a consumer reading the spec had no signal that
        this was an uncalibrated heuristic.
        """
        import importlib

        model = getattr(importlib.import_module(model_path), model_name)
        description = (model.model_fields["confidence"].description or "").lower()

        assert "not a confidence" in description, description
        assert "uncalibrated" in description, description
        assert "recommendation_stability" in description, description

    @pytest.mark.parametrize(
        "model_path,model_name",
        [
            ("src.models.robustness_v2", "RobustnessResult"),
            ("src.models.response_v2", "RobustnessResultV2"),
        ],
    )
    def test_confidence_basis_marker_is_machine_readable(self, model_path, model_name):
        """A prose caveat is not enough — a consumer needs a field it can branch on.

        Mirrors the ``ConfidenceProvenance`` marker that already rides beside
        ``FactorSensitivityV2.confidence``.
        """
        import importlib

        model = getattr(importlib.import_module(model_path), model_name)
        assert "confidence_basis" in model.model_fields


def _outcome_node(node_id: str):
    """Minimal stand-in for a graph node with ``kind='outcome'``."""

    class _Node:
        def __init__(self, nid: str) -> None:
            self.id = nid
            self.kind = "outcome"

    return _Node(node_id)
