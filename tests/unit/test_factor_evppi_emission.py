"""S2 (D-23.8) analyzer-level tests for factor_evppi emission.

Covers the end-to-end path through RobustnessAnalyzerV2.analyze():
* a constructed mediator graph with an analytically-known positive EVPPI,
* the per-factor <= whole-decision EVPI bound,
* D-U lever omission (option-controlled factors are ABSENT, not zero),
* Howard non-negativity + per-factor<=total clamps with disclosure (monkeypatched
  so the clamp mechanism is tested deterministically),
* emission under active correlation (with p_win_sensitivity suppressed),
* method/units tags, gating, and determinism.
"""

from __future__ import annotations

import math

import pytest

import src.services.robustness_analyzer_v2 as analyzer_mod

from src.models.robustness_v2 import (
    EdgeV2,
    FactorCorrelation,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.evppi import REGRESSION_EVPPI_METHOD, FactorEvppiEstimate

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _mediator_request(n_samples=6000, sigma=2.0, seed=777):
    """theta -> m (+1); m -> outcome (+1); theta -> outcome (-0.5 direct).

    opt_pin intervenes m=0  => outcome = -0.5*theta
    opt_free leaves m free  => outcome = +1*theta - 0.5*theta = +0.5*theta
    The winner flips at theta=0, so with perfect info you pick 0.5*|theta|:
        EVPPI_theta = 0.5 * E|theta| = 0.5 * sigma * sqrt(2/pi).
    theta is a NON-lever factor (options intervene on m, never on theta); m is the
    lever. sigma large so theta swings sign.
    """
    nodes = [
        NodeV2(id="theta", kind="factor", label="Theta", observed_state=ObservedState(value=0.0)),
        NodeV2(id="m", kind="factor", label="M", observed_state=ObservedState(value=0.0)),
        NodeV2(id="outcome", kind="outcome", label="Out", observed_state=ObservedState(value=0.0)),
    ]
    edges = [
        EdgeV2(**{"from": "theta"}, to="m", strength=StrengthDistribution(mean=1.0, std=0.002), exists_probability=1.0),
        EdgeV2(**{"from": "m"}, to="outcome", strength=StrengthDistribution(mean=1.0, std=0.002), exists_probability=1.0),
        EdgeV2(**{"from": "theta"}, to="outcome", strength=StrengthDistribution(mean=-0.5, std=0.002), exists_probability=1.0),
    ]
    return RobustnessRequestV2(
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(id="opt_pin", label="Pin m", interventions={"m": 0.0}),
            InterventionOption(id="opt_free", label="Free", interventions={}),
        ],
        goal_node_id="outcome",
        seed=seed,
        n_samples=n_samples,
        parameter_uncertainties=[ParameterUncertainty(node_id="theta", distribution="normal", std=sigma)],
        include_voi=True,
    )


def _lever_request(n_samples=2000, correlated=False):
    """factor_a is intervened by BOTH options -> a D-U lever (must be OMITTED from
    factor_evppi). factor_b is a free uncertain factor (gets an entry)."""
    nodes = [
        NodeV2(id="factor_a", kind="factor", label="A", observed_state=ObservedState(value=0.5)),
        NodeV2(id="factor_b", kind="factor", label="B", observed_state=ObservedState(value=0.3)),
        NodeV2(id="outcome", kind="outcome", label="Rev", observed_state=ObservedState(value=0.0)),
    ]
    edges = [
        EdgeV2(**{"from": "factor_a"}, to="outcome", strength=StrengthDistribution(mean=0.8, std=0.1), exists_probability=0.95),
        EdgeV2(**{"from": "factor_b"}, to="outcome", strength=StrengthDistribution(mean=0.3, std=0.2), exists_probability=0.9),
    ]
    kw = {}
    if correlated:
        kw["factor_correlations"] = [FactorCorrelation(factor_a="factor_a", factor_b="factor_b", rho=0.5)]
    return RobustnessRequestV2(
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(id="opt_high", label="High", interventions={"factor_a": 0.9}),
            InterventionOption(id="opt_low", label="Low", interventions={"factor_a": 0.1}),
        ],
        goal_node_id="outcome",
        seed=12345,
        n_samples=n_samples,
        parameter_uncertainties=[
            ParameterUncertainty(node_id="factor_a", distribution="normal", std=0.1),
            ParameterUncertainty(node_id="factor_b", distribution="normal", std=0.15),
        ],
        include_voi=True,
        **kw,
    )


def _decision_evpi_bound(response):
    regrets = [
        o.pre_noise_expected_regret
        for o in response.results
        if o.pre_noise_expected_regret is not None and math.isfinite(o.pre_noise_expected_regret)
    ]
    return min(regrets) if regrets else None


class TestPositiveEvppiAnalytic:
    def test_mediator_evppi_matches_half_sigma_sqrt_2_over_pi(self):
        r = RobustnessAnalyzerV2().analyze(_mediator_request(sigma=2.0))
        entries = {e["factor_id"]: e for e in (r.factor_evppi or [])}
        assert "theta" in entries
        theta = entries["theta"]
        analytic = 0.5 * 2.0 * SQRT_2_OVER_PI  # ~0.7979
        assert theta["evppi"] == pytest.approx(analytic, abs=0.04)
        assert theta["units"] == "outcome"
        assert theta["method"] == "regression_evppi_v1"
        assert theta["status"] == "resolved"
        assert theta["clamped_low"] is False

    def test_evppi_raw_respects_decision_evpi_bound(self):
        """The RAW estimate (pre-clamp) already respects EVPPI_i <= decision_evpi
        on this shape — proving the estimator, not just the clamp."""
        r = RobustnessAnalyzerV2().analyze(_mediator_request(sigma=2.0))
        bound = _decision_evpi_bound(r)
        assert bound is not None
        for e in r.factor_evppi:
            assert e["evppi_raw"] <= bound + 1e-3
            assert e["evppi"] <= bound + 1e-9  # emitted value is clamped to the bound


class TestLeverOmission:
    def test_lever_factor_absent_free_factor_present(self):
        r = RobustnessAnalyzerV2().analyze(_lever_request())
        fids = [e["factor_id"] for e in (r.factor_evppi or [])]
        assert "factor_a" not in fids, "D-U lever leaked into factor_evppi"
        assert "factor_b" in fids
        # The relabelled p_win_sensitivity keeps BOTH factors (byte-identical
        # legacy behaviour); only the new EVPPI applies lever suppression.
        assert {e["factor_id"] for e in r.p_win_sensitivity} == {"factor_a", "factor_b"}

    def test_all_factors_lever_yields_no_block(self):
        """If every uncertain factor is a lever, factor_evppi is None (not [])."""
        nodes = [
            NodeV2(id="a", kind="factor", label="A", observed_state=ObservedState(value=0.5)),
            NodeV2(id="outcome", kind="outcome", label="O", observed_state=ObservedState(value=0.0)),
        ]
        edges = [EdgeV2(**{"from": "a"}, to="outcome", strength=StrengthDistribution(mean=0.8, std=0.1), exists_probability=0.95)]
        req = RobustnessRequestV2(
            graph=GraphV2(nodes=nodes, edges=edges),
            options=[
                InterventionOption(id="hi", label="hi", interventions={"a": 0.9}),
                InterventionOption(id="lo", label="lo", interventions={"a": 0.1}),
            ],
            goal_node_id="outcome",
            seed=1,
            n_samples=500,
            parameter_uncertainties=[ParameterUncertainty(node_id="a", distribution="normal", std=0.1)],
            include_voi=True,
        )
        r = RobustnessAnalyzerV2().analyze(req)
        assert r.factor_evppi is None


class TestInterventionFactorUnion:
    """C1: the D-U lever set is a single shared helper consulted by BOTH
    _compute_factor_sensitivity and _compute_factor_evppi (derive-don't-mirror)."""

    def test_union_across_all_options_not_just_first(self):
        req = _lever_request()  # opt_high & opt_low each intervene on factor_a
        # Add a factor intervened ONLY by the SECOND option -> still a lever (union).
        req.options[1].interventions["factor_b"] = 0.2
        union = RobustnessAnalyzerV2._intervention_factor_union(req)
        assert union == {"factor_a", "factor_b"}

    def test_empty_when_no_interventions(self):
        req = _mediator_request(n_samples=200)
        req.options = [
            InterventionOption(id="a", label="A", interventions={}),
            InterventionOption(id="b", label="B", interventions={}),
        ]
        assert RobustnessAnalyzerV2._intervention_factor_union(req) == set()

    def test_factor_evppi_omits_union_lever_via_shared_helper(self):
        # factor_a is a lever (both options intervene); factor_b is free. The
        # shared helper drives factor_evppi's omission -> factor_a absent, factor_b
        # present. (If a site stopped using the helper, this would break.)
        r = RobustnessAnalyzerV2().analyze(_lever_request())
        ids = {e["factor_id"] for e in (r.factor_evppi or [])}
        assert "factor_a" not in ids
        assert "factor_b" in ids


class TestClampDisclosure:
    """The Howard (>=0) and per-factor<=total clamps are tested deterministically
    by monkeypatching the estimator to return out-of-range raw values."""

    def _patched(self, monkeypatch, raw):
        def fake(theta, option_outcomes, *, seed, **kw):
            return FactorEvppiEstimate(
                evppi_raw=raw,
                conditional_max_expected_utility=0.0,
                baseline_max_expected_utility=0.0,
                noise_floor=0.0,
                degree_used=4,
                n_samples=len(list(theta)),
                degenerate=False,
            )

        monkeypatch.setattr(analyzer_mod, "factor_evppi_estimate", fake)

    def test_negative_raw_clamped_to_zero_and_disclosed(self, monkeypatch):
        self._patched(monkeypatch, raw=-0.5)
        r = RobustnessAnalyzerV2().analyze(_mediator_request(n_samples=800))
        theta = {e["factor_id"]: e for e in r.factor_evppi}["theta"]
        assert theta["evppi"] == 0.0
        assert theta["clamped_low"] is True
        assert theta["clamped_high"] is False

    def test_raw_above_total_evpi_capped_and_disclosed(self, monkeypatch):
        self._patched(monkeypatch, raw=1e6)  # absurdly large -> must cap at bound
        r = RobustnessAnalyzerV2().analyze(_mediator_request(n_samples=800))
        bound = _decision_evpi_bound(r)
        theta = {e["factor_id"]: e for e in r.factor_evppi}["theta"]
        assert theta["clamped_high"] is True
        # F-2 (B4): emitted value respects the RAW bound AFTER rounding
        # (round-then-clamp). This fixture's bound sits in a round-UP window
        # (round(bound,6) > bound), so the emitted value is the raw bound, NOT
        # round(bound,6) — which pre-fix exceeded decision_evpi by ~5e-7.
        assert theta["evppi"] == min(round(bound, 6), bound)
        assert theta["evppi"] <= bound


def _two_free_factor_request(n_samples=2000, seed=999):
    """m is the lever (both options intervene); theta1 and theta2 are two FREE
    uncertain factors, each with signal to the outcome. Both get factor_evppi rows.
    Used to prove per-factor degradation: if one factor's estimator raises, the
    OTHER factor's computable row must survive (hunter F-4)."""
    nodes = [
        NodeV2(id="theta1", kind="factor", label="T1", observed_state=ObservedState(value=0.0)),
        NodeV2(id="theta2", kind="factor", label="T2", observed_state=ObservedState(value=0.0)),
        NodeV2(id="m", kind="factor", label="M", observed_state=ObservedState(value=0.0)),
        NodeV2(id="outcome", kind="outcome", label="Out", observed_state=ObservedState(value=0.0)),
    ]
    edges = [
        EdgeV2(**{"from": "theta1"}, to="m", strength=StrengthDistribution(mean=1.0, std=0.002), exists_probability=1.0),
        EdgeV2(**{"from": "m"}, to="outcome", strength=StrengthDistribution(mean=1.0, std=0.002), exists_probability=1.0),
        EdgeV2(**{"from": "theta1"}, to="outcome", strength=StrengthDistribution(mean=-0.5, std=0.002), exists_probability=1.0),
        EdgeV2(**{"from": "theta2"}, to="outcome", strength=StrengthDistribution(mean=0.7, std=0.002), exists_probability=1.0),
    ]
    return RobustnessRequestV2(
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(id="opt_pin", label="Pin m", interventions={"m": 0.0}),
            InterventionOption(id="opt_free", label="Free", interventions={}),
        ],
        goal_node_id="outcome",
        seed=seed,
        n_samples=n_samples,
        parameter_uncertainties=[
            ParameterUncertainty(node_id="theta1", distribution="normal", std=2.0),
            ParameterUncertainty(node_id="theta2", distribution="normal", std=2.0),
        ],
        include_voi=True,
    )


class TestPerFactorEstimatorDegrade:
    """Hunter F-4: one factor whose estimator raises (e.g. poisoned ±inf theta ->
    Polynomial.fit LinAlgError) must NOT drop the ENTIRE factor_evppi block — the
    other factors' computable rows are kept, the failing factor is omitted."""

    def test_one_factor_estimator_error_keeps_other_rows(self, monkeypatch):
        real = analyzer_mod.factor_evppi_estimate
        state = {"n": 0}

        def flaky(theta, option_outcomes, *, seed, **kw):
            # First non-lever factor processed raises like Polynomial.fit on inf
            # theta; the rest compute normally via the real estimator.
            state["n"] += 1
            if state["n"] == 1:
                raise __import__("numpy").linalg.LinAlgError(
                    "SVD did not converge in Linear Least Squares"
                )
            return real(theta, option_outcomes, seed=seed, **kw)

        monkeypatch.setattr(analyzer_mod, "factor_evppi_estimate", flaky)
        r = RobustnessAnalyzerV2().analyze(_two_free_factor_request())

        # Pre-fix: the raise propagates out of _compute_factor_evppi, the outer
        # except drops the WHOLE block -> factor_evppi is None (RED here).
        # Post-fix: the failing factor is omitted, the computable factor's row is kept.
        assert r.factor_evppi is not None, "whole factor_evppi block was dropped"
        ids = {e["factor_id"] for e in r.factor_evppi}
        # Exactly one of the two free factors failed; the other survives.
        assert len(r.factor_evppi) == 1, f"expected 1 surviving row, got {ids}"
        # The surviving row is a real, honest computation (has a method + status).
        surviving = r.factor_evppi[0]
        assert surviving["method"] == REGRESSION_EVPPI_METHOD
        assert surviving["status"] in ("resolved", "below_resolution")


class TestFactorEvppiEntryTyped:
    """C3 (altitude Q1): factor_evppi is a typed FactorEvppiEntryV2 model, not a bare
    Dict[str, Any] — a producer typo / dropped status / wrong-typed audit field now
    fails loud in Pydantic instead of serialising clean. Wire byte-identity to the
    prior dict payload was verified pre/post on 4 shapes incl. correlation-active."""

    def _valid(self, **over):
        d = dict(
            factor_id="fa",
            evppi=0.1,
            evppi_raw=0.11,
            baseline_max_expected_utility=1.0,
            conditional_max_expected_utility=1.1,
            units="outcome",
            method=REGRESSION_EVPPI_METHOD,
            regression_degree=4,
            n_samples=2000,
            clamped_low=False,
            clamped_high=False,
            noise_floor=0.01,
            status="resolved",
            correlation_active=False,
        )
        d.update(over)
        return d

    def test_wire_model_types_factor_evppi(self):
        # The V2 WIRE model (ISLResponseV2) types factor_evppi as the model, so a
        # bad entry fails loud at the consumer boundary + openapi documents the shape.
        # (The analyzer/V1 model keeps Dict[str, Any] by design — see C3 scope note.)
        from typing import get_args

        from src.models.response_v2 import FactorEvppiEntryV2, ISLResponseV2

        ann = ISLResponseV2.model_fields["factor_evppi"].annotation
        # Optional[List[FactorEvppiEntryV2]] -> FactorEvppiEntryV2 appears in the args.
        flat = str(ann)
        assert "FactorEvppiEntryV2" in flat, flat

    def test_valid_dict_coerces(self):
        from src.models.response_v2 import FactorEvppiEntryV2

        FactorEvppiEntryV2(**self._valid())

    def test_missing_field_rejected(self):
        from src.models.response_v2 import FactorEvppiEntryV2

        bad = self._valid()
        del bad["status"]
        with pytest.raises(Exception):
            FactorEvppiEntryV2(**bad)

    def test_wrong_typed_field_rejected(self):
        from src.models.response_v2 import FactorEvppiEntryV2

        with pytest.raises(Exception):
            FactorEvppiEntryV2(**self._valid(clamped_low="nope"))


class TestClampRoundOrdering:
    """Hunter F-2: clamp-then-round(.,6) can ship evppi > decision_evpi by <=5e-7
    when clamped_high binds and the bound sits in a round-UP window. The emitted
    value must respect the (raw) bound AFTER rounding."""

    def test_emitted_evppi_never_exceeds_bound_in_rounding_window(self, monkeypatch):
        # Force clamped_high: a raw far above the bound -> evppi clamped to the bound.
        def fake(theta, option_outcomes, *, seed, **kw):
            return FactorEvppiEstimate(
                evppi_raw=0.99,
                conditional_max_expected_utility=1.0,
                baseline_max_expected_utility=0.01,
                noise_floor=0.0,
                degree_used=4,
                n_samples=len(list(theta)),
                degenerate=False,
            )

        monkeypatch.setattr(analyzer_mod, "factor_evppi_estimate", fake)

        # A bound whose round(.,6) rounds UP past itself: round(0.12345675, 6)=0.123457.
        bound = 0.12345675
        assert round(bound, 6) > bound  # the hunter's rounding-window precondition

        n = 8
        req = RobustnessRequestV2(
            graph=GraphV2(
                nodes=[
                    NodeV2(id="f", kind="factor", label="F", observed_state=ObservedState(value=0.0)),
                    NodeV2(id="m", kind="factor", label="M", observed_state=ObservedState(value=0.0)),
                    NodeV2(id="out", kind="outcome", label="Out", observed_state=ObservedState(value=0.0)),
                ],
                edges=[
                    EdgeV2(**{"from": "f"}, to="out", strength=StrengthDistribution(mean=1.0, std=0.01), exists_probability=1.0),
                    EdgeV2(**{"from": "m"}, to="out", strength=StrengthDistribution(mean=1.0, std=0.01), exists_probability=1.0),
                ],
            ),
            options=[
                InterventionOption(id="o1", label="A", interventions={"m": 0.0}),
                InterventionOption(id="o2", label="B", interventions={"m": 1.0}),
            ],
            goal_node_id="out",
            seed=1,
            n_samples=100,  # request min; the direct call below uses the array length
            parameter_uncertainties=[ParameterUncertainty(node_id="f", distribution="normal", std=1.0)],
            include_voi=True,
        )
        pre_noise = {"o1": [float(i) for i in range(n)], "o2": [float(i) + 0.5 for i in range(n)]}
        factor_values = [{"f": float(i) * 0.1} for i in range(n)]

        rows = RobustnessAnalyzerV2()._compute_factor_evppi(
            req, pre_noise, factor_values, seed=1, decision_evpi_bound=bound, correlation_active=False
        )
        row = {r["factor_id"]: r for r in rows}["f"]
        assert row["clamped_high"] is True
        # RED pre-fix: emitted 0.123457 > bound 0.12345675 by 2.5e-7.
        assert row["evppi"] <= bound, f"emitted evppi {row['evppi']} exceeds decision_evpi bound {bound}"


class TestCorrelationEmission:
    def test_factor_evppi_emitted_under_correlation(self):
        r = RobustnessAnalyzerV2().analyze(_lever_request(correlated=True))
        assert r.factor_evppi is not None
        for e in r.factor_evppi:
            assert e["correlation_active"] is True
        # p_win_sensitivity IS suppressed under correlation; factor_evppi is NOT.
        assert r.p_win_sensitivity is None
        assert "p_win_sensitivity" in r.correlation_model.suppressed_attributions
        assert "factor_evppi" not in r.correlation_model.suppressed_attributions


class TestGatingAndDeterminism:
    def test_no_block_without_include_voi(self):
        req = _mediator_request(n_samples=500)
        req.include_voi = False
        assert RobustnessAnalyzerV2().analyze(req).factor_evppi is None

    def test_no_block_without_uncertainties(self):
        nodes = [
            NodeV2(id="a", kind="factor", label="A", observed_state=ObservedState(value=0.5)),
            NodeV2(id="outcome", kind="outcome", label="O", observed_state=ObservedState(value=0.0)),
        ]
        edges = [EdgeV2(**{"from": "a"}, to="outcome", strength=StrengthDistribution(mean=0.8, std=0.1), exists_probability=0.95)]
        req = RobustnessRequestV2(
            graph=GraphV2(nodes=nodes, edges=edges),
            options=[InterventionOption(id="x", label="x", interventions={})],
            goal_node_id="outcome",
            seed=1,
            n_samples=500,
            include_voi=True,
        )
        assert RobustnessAnalyzerV2().analyze(req).factor_evppi is None

    def test_deterministic_same_seed(self):
        r1 = RobustnessAnalyzerV2().analyze(_mediator_request(n_samples=3000))
        r2 = RobustnessAnalyzerV2().analyze(_mediator_request(n_samples=3000))
        assert r1.factor_evppi == r2.factor_evppi
