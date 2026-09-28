"""factor_evppi ``resolved`` must mean "learning this factor can change the DECISION".

The permutation-null floor shuffles theta, so it tests theta-outcome ASSOCIATION. A factor
that moves only an option that never wins is associated with the outcomes but cannot change
the choice: its true EVPPI is exactly 0. Its fitted curves can still cross the leader's
through fit noise, while the shuffled fits stay flat and apart (floor ~0), so it escaped as
``resolved`` in 28/200 seeds at n=2000 and 52/200 at n=500 (ISL 14f1a3a; study in
``experiments/r3b_sim/evppi_gate/``). The independence-null calibration
(``test_evppi_estimator.TestBelowResolutionFloorCalibration``) never exercised this shape.

The gate: a 2-fold cross-fitted test that the decision rule LEARNED from theta beats the
best fixed option on held-out draws. Under "theta cannot change the decision" the expected
held-out gain of any learned rule is <= 0, whatever the association.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.services.robustness_analyzer_v2 as analyzer_mod
from pydantic import ValidationError
from src.models.robustness_v2 import ParameterUncertainty
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.evppi import FactorEvppiEstimate, factor_evppi_estimate

from tests.unit.test_factor_evppi_emission import _mediator_request


def _two_option(mean_b, n: int, ds: int):
    """Option A = 10 + noise; option B = mean_b(theta) + noise; theta ~ N(0, 1). Deterministic."""
    rng = np.random.default_rng(ds)
    theta = rng.normal(0.0, 1.0, n)
    return theta, {
        "a": 10.0 + rng.normal(0.0, 1.0, n),
        "b": mean_b(theta) + rng.normal(0.0, 1.0, n),
    }


def _resolved(est: FactorEvppiEstimate) -> bool:
    """The analyzer's wire rule (``robustness_analyzer_v2`` status line). ``getattr`` so the
    same row measures the pre-gate estimator too (RED at base on the real escape rate)."""
    evppi = round(max(0.0, est.evppi_raw), 6)
    return evppi > round(est.noise_floor, 6) and getattr(est, "decision_gain_passes", True)


class TestDominatedDependentFactorIsNotResolved:
    """theta moves B only, and B never beats A: sup E[B | theta] = 9.8 < 10, true EVPPI = 0."""

    N = 40
    TARGET = 0.05

    @staticmethod
    def _dominated(theta):
        return 9.0 + 0.8 * np.tanh(theta)

    @pytest.mark.parametrize("n", [500, 2000])
    def test_escape_rate_below_target(self, n):
        escapes = sum(
            _resolved(factor_evppi_estimate(*_two_option(self._dominated, n, ds), seed=1000 + ds))
            for ds in range(self.N)
        )
        assert (
            escapes / self.N <= self.TARGET
        ), f"{escapes}/{self.N} decision-irrelevant factors shipped as resolved at n={n}"


class TestDecisionRelevantFactorStaysResolved:
    """B beats A for theta > 1: true EVPPI = E[(theta - 1)+] = 0.0833."""

    N = 40

    def test_true_positive_resolved_at_the_served_sample_size(self):
        kept = sum(
            _resolved(
                factor_evppi_estimate(
                    *_two_option(lambda th: 9.0 + 1.0 * th, 2000, ds), seed=1000 + ds
                )
            )
            for ds in range(self.N)
        )
        assert kept == self.N

    def test_gate_is_deterministic_given_seed(self):
        theta, oo = _two_option(lambda th: 9.0 + 1.0 * th, 2000, 3)
        assert factor_evppi_estimate(theta, oo, seed=5) == factor_evppi_estimate(theta, oo, seed=5)

    def test_degenerate_factor_never_passes(self):
        est = factor_evppi_estimate(
            np.full(500, 0.3), {"a": np.arange(500.0), "b": -np.arange(500.0)}, seed=1
        )
        assert est.degenerate and not est.decision_gain_passes



class TestGateFoldsAreSeeded:
    """The fold split is part of the verdict, so it must come from the per-factor seed, on a
    stream separate from the floor's (``default_rng((seed, 1))``). DL verdict on ISL #198
    (5869397004): the old determinism row did not discriminate, because an always-passing case
    stays green under an unseeded fold. These rows use a BORDERLINE case (true EVPPI 0.0042, n=2000,
    data seed 8) whose verdict does depend on the fold seed."""

    SEEDS = range(16)
    PINNED = "0011101111100110"  # decision_gain_passes for fold seeds 0..15 (numpy PCG64)

    @staticmethod
    def _verdicts():
        theta, oo = _two_option(lambda th: 9.0 + 0.5 * th, 2000, 8)
        return "".join(
            "1" if factor_evppi_estimate(theta, oo, seed=s).decision_gain_passes else "0"
            for s in TestGateFoldsAreSeeded.SEEDS
        )

    def test_the_fold_seed_matters_on_the_borderline_case(self):
        """Non-vacuity: some fold seeds pass and some fail, so determinism is not automatic."""
        assert set(self._verdicts()) == {"0", "1"}

    def test_same_seed_same_verdict_and_pinned(self):
        first = self._verdicts()
        assert self._verdicts() == first
        assert first == self.PINNED

    def test_folds_come_from_their_own_seeded_stream(self, monkeypatch):
        calls = []
        real = np.random.default_rng

        def spy(*args, **kwargs):
            calls.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(np.random, "default_rng", spy)
        theta, oo = _two_option(lambda th: 9.0 + 1.0 * th, 500, 3)
        calls.clear()
        factor_evppi_estimate(theta, oo, seed=5)
        assert ((5, 1),) in calls, calls
        assert (5,) in calls, calls  # the floor's stream, distinct from the folds'
        assert () not in calls, "an unseeded generator makes the verdict irreproducible"

class TestAnalyzerStatusUsesTheGate:
    """The wire status reads the gate: a row above its floor whose learned rule does not
    beat the best fixed option on held-out draws is below_resolution."""

    def _patched(self, monkeypatch, *, raw, floor, gate):
        def fake(theta, option_outcomes, *, seed, **kw):
            return FactorEvppiEstimate(
                evppi_raw=raw,
                conditional_max_expected_utility=0.0,
                baseline_max_expected_utility=0.0,
                noise_floor=floor,
                degree_used=4,
                n_samples=len(list(theta)),
                degenerate=False,
                decision_gain_passes=gate,
            )

        monkeypatch.setattr(analyzer_mod, "factor_evppi_estimate", fake)

    def _theta_row(self):
        r = RobustnessAnalyzerV2().analyze(_mediator_request(n_samples=800))
        return {e["factor_id"]: e for e in r.factor_evppi}["theta"]

    def test_above_floor_but_gate_fails_is_below_resolution(self, monkeypatch):
        self._patched(monkeypatch, raw=0.5, floor=0.0, gate=False)
        row = self._theta_row()
        assert row["evppi"] > row["noise_floor"]
        assert (row["status"], row["status_reason"]) == (
            "below_resolution",
            "decision_gain_not_significant",
        )

    def test_control_above_floor_and_gate_passes_is_resolved(self, monkeypatch):
        self._patched(monkeypatch, raw=0.5, floor=0.01, gate=True)
        row = self._theta_row()
        assert (row["status"], row["status_reason"]) == ("resolved", None)

    def test_gate_never_rescues_a_row_at_or_below_its_floor(self, monkeypatch):
        self._patched(monkeypatch, raw=0.01, floor=0.02, gate=True)
        row = self._theta_row()
        assert (row["status"], row["status_reason"]) == (
            "below_resolution",
            "at_or_below_noise_floor",
        )

    def test_floor_reason_takes_precedence_when_both_fail(self, monkeypatch):
        self._patched(monkeypatch, raw=0.01, floor=0.02, gate=False)
        assert self._theta_row()["status_reason"] == "at_or_below_noise_floor"


class TestAiqRulingRows:
    """AIQ #72 5867782904, rows as ruled (200 seeds each): dominated <= 1/200, independent
    nulls <= 1/200, true EVPPI 0.083 resolved in >= 180/200 at n=500."""

    SEEDS = 200

    def _count(self, mean_b, n):
        return sum(
            _resolved(factor_evppi_estimate(*_two_option(mean_b, n, ds), seed=1000 + ds))
            for ds in range(self.SEEDS)
        )

    @pytest.mark.parametrize("n", [500, 2000])
    def test_dominated_at_most_one_in_200(self, n):
        assert self._count(lambda th: 9.0 + 0.8 * np.tanh(th), n) <= 1

    @pytest.mark.parametrize("n", [500, 2000])
    def test_independent_null_at_most_one_in_200(self, n):
        assert self._count(lambda th: 9.0 + 0.0 * th, n) <= 1

    @pytest.mark.parametrize("n", [500, 2000])
    def test_near_tie_null_at_most_two_in_200(self, n):
        """theta irrelevant, options 0.02 apart: a comparator chosen on the training fold let
        the learned rule 'beat' a mis-chosen option (5/200 here); the full-sample comparator
        (EVPPI's own baseline) does not."""
        assert self._count(lambda th: 9.98 + 0.0 * th, n) <= 2

    def test_true_positive_at_least_180_in_200_at_n500(self):
        assert self._count(lambda th: 9.0 + 1.0 * th, 500) >= 180


class TestSpreadSourceEcho:
    """AIQ #72 5867782904: each factor_evppi row says whose spread it is, echoed from the
    request's ParameterUncertainty.spread_source; absent means the caller did not say (None)."""

    def _row(self, spread_source):
        req = _mediator_request(n_samples=800)
        pu = req.parameter_uncertainties[0]
        fields = pu.model_dump(exclude_none=True)
        if spread_source is not None:
            fields["spread_source"] = spread_source
        req = req.model_copy(update={"parameter_uncertainties": [ParameterUncertainty(**fields)]})
        r = RobustnessAnalyzerV2().analyze(req)
        return {e["factor_id"]: e for e in r.factor_evppi}["theta"]

    @pytest.mark.parametrize("source", ["user", "template"])
    def test_stated_source_is_echoed(self, source):
        assert self._row(source)["spread_source"] == source

    def test_absent_source_is_none_never_inferred(self):
        assert self._row(None)["spread_source"] is None

    def test_unknown_source_is_refused(self):
        with pytest.raises(ValidationError):
            ParameterUncertainty(node_id="theta", std=1.0, spread_source="olumi")


class TestMediatorAnalyticCaseStaysResolvedEndToEnd:
    """The analytic mediator (EVPPI = 0.5 sigma sqrt(2/pi)) is decision-relevant by
    construction: the gate must not withhold it on the real analyzer path."""

    def test_mediator_resolved(self):
        r = RobustnessAnalyzerV2().analyze(_mediator_request(n_samples=2000))
        theta = {e["factor_id"]: e for e in r.factor_evppi}["theta"]
        assert theta["status"] == "resolved"
