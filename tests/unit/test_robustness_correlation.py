"""B3-S1 — correlated-factors (Gaussian copula) tests.

Doctrine (D-23.4, research-sharpened): independence is the default, correlation
is inert-when-absent (the no-correlation path is byte-identical to base), the
copula activates only when supplied, non-PSD inputs are Higham-projected with
disclosure, tail-independence is disclosed, and independence-assuming per-factor
attributions are suppressed with a marker.

Each mechanism has a mutation check documented in NOTES.md: reverting the
mechanism turns the paired test(s) RED (proof the test discriminates).
"""

import numpy as np
import pytest
from pydantic import ValidationError

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
from src.utils.correlation import (
    HIGHAM_METHOD,
    assemble_correlation_matrix,
    build_correlation_plan,
    is_positive_semidefinite,
    nearest_correlation_higham,
)


# =============================================================================
# Fixtures — a 2-factor same-direction chain (fa, fb both push revenue up)
# =============================================================================


def _two_factor_graph():
    return GraphV2(
        nodes=[
            NodeV2(id="fa", kind="factor", label="Fa", observed_state=ObservedState(value=1.0)),
            NodeV2(id="fb", kind="factor", label="Fb", observed_state=ObservedState(value=1.0)),
            NodeV2(id="rev", kind="outcome", label="Rev"),
        ],
        edges=[
            EdgeV2(
                **{"from": "fa", "to": "rev"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=1.0, std=0.05),
            ),
            EdgeV2(
                **{"from": "fb", "to": "rev"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=1.0, std=0.05),
            ),
        ],
    )


_OPTS = [
    InterventionOption(id="o1", label="O1", interventions={}),
    InterventionOption(id="o2", label="O2", interventions={"fa": 1.0}),
]
_UNC_NORMAL = [
    ParameterUncertainty(node_id="fa", distribution="normal", std=0.5),
    ParameterUncertainty(node_id="fb", distribution="normal", std=0.5),
]


def _request(correlations, *, unc=None, graph=None, options=None, n=2000, seed=42, voi=True):
    return RobustnessRequestV2(
        request_id="corr-test",
        graph=graph or _two_factor_graph(),
        options=options or _OPTS,
        goal_node_id="rev",
        n_samples=n,
        seed=seed,
        analysis_types=["comparison", "sensitivity", "robustness"],
        parameter_uncertainties=unc or _UNC_NORMAL,
        include_voi=voi,
        factor_correlations=correlations,
    )


def _outcome(resp, option_id="o1"):
    return next(r for r in resp.results if r.option_id == option_id).outcome_distribution


@pytest.fixture(scope="module")
def analyzer():
    return RobustnessAnalyzerV2()


# =============================================================================
# Request-channel validation → 422 (pydantic ValidationError)
# =============================================================================


class TestFactorCorrelationValidation:
    def test_absent_is_valid(self):
        req = _request(None)
        assert req.factor_correlations is None

    def test_valid_pair_accepted(self):
        req = _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.6)])
        assert len(req.factor_correlations) == 1

    def test_self_pair_rho_one_rejected(self):
        # B3 P3-2: a self-pair {fa, fa, rho} is NOT a harmless no-op. Even at
        # rho=1.0 it ACTIVATES the correlation regime (correlation_model disclosure
        # + attribution suppression) while declaring no dependence between distinct
        # factors — attributions withheld for nothing. Rejected outright, any rho.
        with pytest.raises(ValidationError, match="self-correlation"):
            _request([FactorCorrelation(factor_a="fa", factor_b="fa", rho=1.0)])

    def test_rho_above_one_rejected(self):
        with pytest.raises(ValidationError):
            FactorCorrelation(factor_a="fa", factor_b="fb", rho=1.5)

    def test_rho_below_minus_one_rejected(self):
        with pytest.raises(ValidationError):
            FactorCorrelation(factor_a="fa", factor_b="fb", rho=-1.5)

    def test_unknown_factor_rejected(self):
        with pytest.raises(ValidationError, match="non-existent factor"):
            _request([FactorCorrelation(factor_a="fa", factor_b="ghost", rho=0.5)])

    def test_factor_without_uncertainty_rejected(self):
        # 'rev' exists but has no parameter_uncertainty.
        with pytest.raises(ValidationError, match="no parameter_uncertainty"):
            _request([FactorCorrelation(factor_a="fa", factor_b="rev", rho=0.5)])

    def test_unsupported_distribution_family_rejected_allowlist(self):
        # B3 INFO-2: the correlation marginal check ALLOWLISTS supported families
        # {normal, uniform}. A future distribution family that slips past
        # ParameterUncertainty's own family validator must still be rejected by the
        # correlation validator — not silently accepted (which would reach
        # _copula_transform's fail-closed branch and, on the enhanced handler, a
        # 500). A point_mass-only blocklist would let a new family through.
        #
        # Inject the fake family by mutating .distribution AFTER construction:
        # ParameterUncertainty has no validate_assignment, so this bypasses its own
        # family validator and simulates a family the request layer would accept.
        unc = [
            ParameterUncertainty(node_id="fa", distribution="normal", std=0.5),
            ParameterUncertainty(node_id="fb", distribution="normal", std=0.5),
        ]
        req = _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.6)], unc=unc)
        req.parameter_uncertainties[0].distribution = "lognormal"  # fresh obj, no shared state
        with pytest.raises(ValueError) as exc:
            req.validate_factor_correlations()
        msg = str(exc.value)
        assert "'fa'" in msg  # names the offending factor
        assert "lognormal" in msg  # names the unsupported family, no other request value

    def test_point_mass_factor_rejected(self):
        unc = _UNC_NORMAL + [ParameterUncertainty(node_id="rev", distribution="point_mass")]
        # 'rev' is the goal/outcome; give it a point_mass uncertainty and correlate.
        graph = _two_factor_graph()
        with pytest.raises(ValidationError, match="point_mass"):
            RobustnessRequestV2(
                graph=graph,
                options=_OPTS,
                goal_node_id="rev",
                n_samples=200,
                seed=1,
                parameter_uncertainties=unc,
                factor_correlations=[FactorCorrelation(factor_a="fa", factor_b="rev", rho=0.5)],
            )

    def test_self_pair_rho_not_one_rejected(self):
        with pytest.raises(ValidationError, match="self-correlation"):
            _request([FactorCorrelation(factor_a="fa", factor_b="fa", rho=0.5)])

    def test_duplicate_pair_rejected(self):
        # Same unordered pair {fa, fb} supplied twice (order-swapped) → rejected.
        with pytest.raises(ValidationError, match="duplicate pair"):
            _request(
                [
                    FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.6),
                    FactorCorrelation(factor_a="fb", factor_b="fa", rho=0.3),
                ]
            )

    def test_error_message_names_factor_not_other_values(self):
        # The 422 message names the factor id but must not echo unrelated request
        # values (e.g. observed_state values, edge strengths).
        try:
            _request([FactorCorrelation(factor_a="fa", factor_b="ghost", rho=0.5)])
            assert False, "should have raised"
        except ValidationError as exc:
            msg = str(exc)
            assert "ghost" in msg
            assert "observed_state" not in msg
            assert "strength" not in msg


# =============================================================================
# Inert-when-absent (byte-identity guard at the unit level)
# =============================================================================


class TestInertWhenAbsent:
    def test_no_correlation_model_when_absent(self, analyzer):
        resp = analyzer.analyze(_request(None))
        assert resp.correlation_model is None

    def test_absent_path_emits_attributions_positive_control(self, analyzer):
        # Positive control: absent-correlation path DOES compute the attributions
        # (so their suppression under correlation is a real change, not vacuous).
        resp = analyzer.analyze(_request(None))
        assert len(resp.factor_sensitivity) > 0
        assert resp.p_win_sensitivity is not None

    def test_factor_sampler_plan_none_is_deterministic_independent(self):
        # A plan=None FactorSampler must draw exactly the independent stream.
        from src.services.robustness_analyzer_v2 import FactorSampler
        from src.utils.rng import SeededRNG

        nodes = _two_factor_graph().nodes
        fs = FactorSampler(nodes, _UNC_NORMAL, SeededRNG(43), correlation_plan=None)
        draws = [fs.sample_factor_values() for _ in range(3)]
        # Reproduce with a fresh identical sampler → identical stream.
        fs2 = FactorSampler(nodes, _UNC_NORMAL, SeededRNG(43), correlation_plan=None)
        draws2 = [fs2.sample_factor_values() for _ in range(3)]
        assert draws == draws2
        # And the values are the raw marginal draws (fa,fb present each iter).
        assert set(draws[0].keys()) == {"fa", "fb"}


# =============================================================================
# Activation — direction + exact seeded values (the HARD numerical gate)
# =============================================================================


class TestCorrelationActivation:
    def test_positive_correlation_widens_spread(self, analyzer):
        # rho=0.9 on a same-direction chain widens the outcome spread.
        independent = _outcome(analyzer.analyze(_request(None)))
        correlated = _outcome(
            analyzer.analyze(_request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.9)]))
        )
        assert correlated.std > independent.std
        assert correlated.std / independent.std > 1.3  # direction pinned

    def test_positive_correlation_exact_seeded_value(self, analyzer):
        # Exact seeded pin (numpy 1.26.x / PCG64, seed=42, n=2000, this fixture).
        # Stable across 3 runs; reverting the copula draw moves this value (mutation
        # check M2 in NOTES.md).
        correlated = _outcome(
            analyzer.analyze(_request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.9)]))
        )
        assert correlated.std == pytest.approx(1.3234972880067288, abs=1e-9)

    def test_rho_zero_all_normal_is_bit_identical_to_absent(self, analyzer):
        # rho=0 over an all-normal correlated set spanning the full uncertainty
        # list reproduces the independent draws BIT-for-BIT (identity Cholesky +
        # numpy's mean+std*z == normal(mean,std)).
        absent = _outcome(analyzer.analyze(_request(None)))
        rho0 = _outcome(
            analyzer.analyze(_request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.0)]))
        )
        assert rho0.samples == absent.samples  # exact equality
        assert rho0.std == absent.std
        assert rho0.mean == absent.mean

    def test_rho_zero_with_uniform_factor_differs_from_absent(self, analyzer):
        # DISCLOSED divergence: a uniform factor in the correlated set is drawn in
        # normal space + Phi-mapped, so rho=0 is NOT bit-identical to omitting.
        graph = _two_factor_graph()
        unc = [
            ParameterUncertainty(node_id="fa", distribution="normal", std=0.5),
            ParameterUncertainty(
                node_id="fb", distribution="uniform", range_min=0.0, range_max=2.0
            ),
        ]
        absent = _outcome(analyzer.analyze(_request(None, unc=unc, graph=graph)))
        rho0 = _outcome(
            analyzer.analyze(
                _request(
                    [FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.0)],
                    unc=unc,
                    graph=graph,
                )
            )
        )
        assert rho0.samples != absent.samples

    def test_determinism_same_request_same_result(self, analyzer):
        corr = [FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.7)]
        a = _outcome(analyzer.analyze(_request(corr)))
        b = _outcome(analyzer.analyze(_request(corr)))
        assert a.samples == b.samples

    def test_uniform_copula_draws_stay_in_range(self, analyzer):
        # The Gaussian-copula uniform coupling must keep draws in [lo, hi].
        graph = _two_factor_graph()
        unc = [
            ParameterUncertainty(node_id="fa", distribution="normal", std=0.5),
            ParameterUncertainty(
                node_id="fb", distribution="uniform", range_min=0.2, range_max=0.9
            ),
        ]
        from src.services.robustness_analyzer_v2 import FactorSampler
        from src.utils.rng import SeededRNG

        plan = build_correlation_plan(["fa", "fb"], [("fa", "fb", 0.8)])
        fs = FactorSampler(graph.nodes, unc, SeededRNG(5), correlation_plan=plan)
        for _ in range(500):
            vals = fs.sample_factor_values()
            assert 0.2 <= vals["fb"] <= 0.9


# =============================================================================
# Attribution suppression + disclosure
# =============================================================================


class TestCorrelationSuppression:
    def test_attributions_suppressed_under_correlation(self, analyzer):
        resp = analyzer.analyze(
            _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.8)])
        )
        assert resp.factor_sensitivity == []
        assert resp.p_win_sensitivity is None
        assert resp.conditional_winners is None

    def test_correlation_model_disclosure_present(self, analyzer):
        resp = analyzer.analyze(
            _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.8)])
        )
        cm = resp.correlation_model
        assert cm is not None
        assert cm.method == "gaussian_copula_v1"
        assert cm.active is True
        assert set(cm.correlated_factors) == {"fa", "fb"}
        assert cm.n_pairs == 1

    def test_mandatory_tail_dependence_disclosure(self, analyzer):
        resp = analyzer.analyze(
            _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.8)])
        )
        cm = resp.correlation_model
        assert cm.tail_dependence == "none"
        assert "tail" in cm.tail_dependence_note.lower()
        assert cm.tail_dependence_note  # non-empty — the load-bearing caveat

    def test_suppressed_attribution_manifest(self, analyzer):
        resp = analyzer.analyze(
            _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.8)])
        )
        cm = resp.correlation_model
        assert "factor_sensitivity" in cm.suppressed_attributions
        assert "p_win_sensitivity" in cm.suppressed_attributions
        assert "conditional_winners" in cm.suppressed_attributions
        assert cm.suppression_reason == "not_separable_under_correlation"

    def test_suppression_manifest_excludes_unenabled_attributions(self, analyzer):
        # factor_evpi is only listed as suppressed when include_voi would have
        # emitted it — a field never going to be produced is not claimed.
        resp = analyzer.analyze(
            _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.8)], voi=False)
        )
        cm = resp.correlation_model
        assert "p_win_sensitivity" not in cm.suppressed_attributions
        assert "factor_sensitivity" in cm.suppressed_attributions

    def test_stability_thresholds_named_in_manifest_and_actually_vanishes(self, analyzer):
        # Hunter F-1: stability_thresholds rides on factor_sensitivity's bootstrap,
        # so under active correlation it silently vanishes. RECORD-not-PREDICT now
        # names it in the manifest at the factor_sensitivity skip site.
        corr = [FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.8)]

        # Positive control: WITHOUT correlation, stability_thresholds IS emitted
        # (otherwise "it was suppressed" would be a vacuous claim).
        baseline = analyzer.analyze(_request([]))
        assert baseline.correlation_model is None
        assert baseline.stability_thresholds is not None, (
            "positive control: stability_thresholds must emit when correlation is off"
        )

        # Under active correlation it vanishes from the response...
        resp = analyzer.analyze(_request(corr))
        assert resp.stability_thresholds is None
        # ...and is NAMED as suppressed (RED pre-fix: absent from the manifest).
        assert "stability_thresholds" in resp.correlation_model.suppressed_attributions
        # co-recorded with its parent factor_sensitivity.
        assert "factor_sensitivity" in resp.correlation_model.suppressed_attributions

    def test_manifest_is_recorded_not_predicted_exact_set(self, analyzer):
        # The full F-1 shape (sensitivity + 2 options + voi + factors, correlation
        # active) records exactly the four blocks it actually skipped, in gate order.
        resp = analyzer.analyze(
            _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.8)])
        )
        assert resp.correlation_model.suppressed_attributions == [
            "factor_sensitivity",
            "stability_thresholds",
            "conditional_winners",
            "p_win_sensitivity",
        ]

    def test_joint_quantities_preserved(self, analyzer):
        # win_probability + downside stay present under correlation (joint-valid).
        resp = analyzer.analyze(
            _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.8)])
        )
        for r in resp.results:
            assert r.win_probability is not None


# =============================================================================
# PSD projection (Higham) disclosure
# =============================================================================


def _three_factor_graph():
    nodes = [
        NodeV2(id=f, kind="factor", label=f, observed_state=ObservedState(value=1.0))
        for f in ("fa", "fb", "fc")
    ]
    nodes.append(NodeV2(id="rev", kind="outcome", label="Rev"))
    edges = [
        EdgeV2(
            **{"from": f, "to": "rev"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=1.0, std=0.05),
        )
        for f in ("fa", "fb", "fc")
    ]
    return GraphV2(nodes=nodes, edges=edges)


_UNC3 = [
    ParameterUncertainty(node_id="fa", distribution="normal", std=0.5),
    ParameterUncertainty(node_id="fb", distribution="normal", std=0.5),
    ParameterUncertainty(node_id="fc", distribution="normal", std=0.5),
]


class TestPSDProjection:
    def test_psd_input_no_projection(self, analyzer):
        resp = analyzer.analyze(
            _request([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.9)])
        )
        assert resp.correlation_model.psd_projection is None

    def test_non_psd_triggers_higham_with_disclosure(self, analyzer):
        # 0.9 / 0.9 / -0.9 over three factors is indefinite → Higham projection.
        non_psd = [
            FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.9),
            FactorCorrelation(factor_a="fb", factor_b="fc", rho=0.9),
            FactorCorrelation(factor_a="fa", factor_b="fc", rho=-0.9),
        ]
        resp = analyzer.analyze(
            _request(
                non_psd,
                unc=_UNC3,
                graph=_three_factor_graph(),
                options=[
                    InterventionOption(id="o1", label="O1", interventions={}),
                    InterventionOption(id="o2", label="O2", interventions={"fa": 1.0}),
                ],
            )
        )
        proj = resp.correlation_model.psd_projection
        assert proj is not None
        assert proj.applied is True
        assert proj.method == HIGHAM_METHOD
        assert proj.frobenius_distance > 0.0
        assert proj.iterations > 0


# =============================================================================
# Correlation utility (pure math)
# =============================================================================


class TestCorrelationUtility:
    def test_assemble_matrix_symmetric_unit_diagonal(self):
        m = assemble_correlation_matrix(["a", "b"], [("a", "b", 0.6)])
        assert m[0, 1] == 0.6
        assert m[1, 0] == 0.6
        assert m[0, 0] == 1.0 and m[1, 1] == 1.0

    def test_rho_zero_gives_identity_cholesky(self):
        plan = build_correlation_plan(["a", "b"], [("a", "b", 0.0)])
        assert np.array_equal(plan.cholesky, np.eye(2))
        assert plan.projection is None

    def test_psd_detection(self):
        assert is_positive_semidefinite(np.array([[1.0, 0.9], [0.9, 1.0]]))
        indef = assemble_correlation_matrix(
            ["a", "b", "c"], [("a", "b", 0.9), ("b", "c", 0.9), ("a", "c", -0.9)]
        )
        assert not is_positive_semidefinite(indef)

    def test_higham_returns_psd_unit_diagonal(self):
        indef = assemble_correlation_matrix(
            ["a", "b", "c"], [("a", "b", 0.9), ("b", "c", 0.9), ("a", "c", -0.9)]
        )
        projected, iters = nearest_correlation_higham(indef)
        assert iters > 0
        assert np.allclose(np.diag(projected), 1.0)
        assert np.linalg.eigvalsh(projected)[0] >= -1e-8

    def test_perfect_correlation_singular_cholesky_handled(self):
        plan = build_correlation_plan(["a", "b"], [("a", "b", 1.0)])
        llt = plan.cholesky @ plan.cholesky.T
        assert np.allclose(llt, [[1.0, 1.0], [1.0, 1.0]], atol=1e-6)

    def test_build_plan_cholesky_reconstructs_matrix(self):
        plan = build_correlation_plan(["a", "b"], [("a", "b", 0.7)])
        llt = plan.cholesky @ plan.cholesky.T
        assert np.allclose(llt, [[1.0, 0.7], [0.7, 1.0]])
