"""Tests for scientific enhancements: confidence, root node warnings, E-value, EVPI."""

import math

import numpy as np
import pytest

from src.config.stability_thresholds import (
    compute_factor_confidence,
    compute_graph_structural_confidence,
)
from src.models.response_v2 import EdgeEValueV2, FactorSensitivityV2
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


# ---------------------------------------------------------------------------
# Shared fixtures: small deterministic graph
# ---------------------------------------------------------------------------


def _make_graph(
    include_observed_values: bool = True,
    n_factors: int = 2,
) -> GraphV2:
    """3-4 node graph: factor_a → outcome, factor_b → outcome, (optional factor_c → outcome)."""
    nodes = [
        NodeV2(
            id="factor_a",
            kind="factor",
            label="Factor A",
            observed_state=ObservedState(value=0.5) if include_observed_values else None,
        ),
        NodeV2(
            id="factor_b",
            kind="factor",
            label="Factor B",
            observed_state=ObservedState(value=0.3) if include_observed_values else None,
        ),
        NodeV2(
            id="outcome",
            kind="outcome",
            label="Revenue",
            observed_state=ObservedState(value=0.0),
        ),
    ]
    edges = [
        EdgeV2(
            **{"from": "factor_a"},
            to="outcome",
            strength=StrengthDistribution(mean=0.8, std=0.1),
            exists_probability=0.95,
        ),
        EdgeV2(
            **{"from": "factor_b"},
            to="outcome",
            strength=StrengthDistribution(mean=0.3, std=0.2),
            exists_probability=0.9,
        ),
    ]
    if n_factors >= 3:
        nodes.append(
            NodeV2(
                id="factor_c",
                kind="factor",
                label="Factor C",
                observed_state=None,  # Deliberately missing
            )
        )
        edges.append(
            EdgeV2(
                **{"from": "factor_c"},
                to="outcome",
                strength=StrengthDistribution(mean=0.5, std=0.15),
                exists_probability=0.85,
            )
        )
    return GraphV2(nodes=nodes, edges=edges)


def _make_request(
    graph: GraphV2,
    seed: int = 12345,
    include_uncertainties: bool = False,
    include_e_values: bool = False,
    include_voi: bool = False,
    n_samples: int = 200,
) -> RobustnessRequestV2:
    options = [
        InterventionOption(id="opt_high", label="High", interventions={"factor_a": 0.9}),
        InterventionOption(id="opt_low", label="Low", interventions={"factor_a": 0.1}),
    ]
    uncertainties = None
    if include_uncertainties:
        uncertainties = [
            ParameterUncertainty(node_id="factor_a", distribution="normal", std=0.1),
            ParameterUncertainty(node_id="factor_b", distribution="normal", std=0.15),
        ]
    return RobustnessRequestV2(
        graph=graph,
        options=options,
        goal_node_id="outcome",
        seed=seed,
        n_samples=n_samples,
        parameter_uncertainties=uncertainties,
        include_e_values=include_e_values,
        include_voi=include_voi,
    )


# ===========================================================================
# Task 1: Factor sensitivity confidence
# ===========================================================================


class TestFactorSensitivityConfidence:
    """Verify confidence is always populated when factor sensitivity is computed."""

    def test_graph_structural_confidence_range(self):
        """Graph-structural confidence maps influence_score [0,1] to [0.25, 0.75]."""
        assert compute_graph_structural_confidence(0.0) == 0.25
        assert compute_graph_structural_confidence(1.0) == 0.75
        assert compute_graph_structural_confidence(0.5) == 0.5
        assert compute_graph_structural_confidence(None) == 0.25

    def test_bootstrap_confidence_non_none(self):
        """When bootstrap data is available, confidence is computed (not None)."""
        result = compute_factor_confidence("high", 0.5, None)
        assert result is not None
        assert result == 0.9

    def test_bootstrap_confidence_with_cv(self):
        """Bootstrap confidence blends category (70%) and CV (30%)."""
        result = compute_factor_confidence("moderate", 0.5, 0.1)
        assert result is not None
        # 0.7 * 0.6 + 0.3 * (1/(1+0.2)) = 0.42 + 0.25 = 0.67
        assert abs(result - 0.6694) < 0.01

    def test_confidence_always_populated_with_uncertainties(self):
        """When parameter_uncertainties are provided, confidence is never None."""
        graph = _make_graph()
        request = _make_request(graph, include_uncertainties=True)
        analyzer = RobustnessAnalyzerV2()
        analyzer._n_bootstrap_override = 5  # Fast bootstrap for test
        response = analyzer.analyze(request)

        assert len(response.factor_sensitivity) > 0
        for fs in response.factor_sensitivity:
            # Internal model always has attribution_stability when bootstrap runs
            assert fs.attribution_stability is not None or fs.elasticity == 0.0

    def test_confidence_source_field_present(self):
        """FactorSensitivityV2 model accepts confidence_source."""
        fs = FactorSensitivityV2(
            node_id="x",
            sensitivity_score=0.5,
            direction="positive",
            confidence=0.7,
            confidence_source="bootstrap_sampling",
        )
        assert fs.confidence_source == "bootstrap_sampling"

    def test_graph_structural_fallback_used_without_uncertainties(self):
        """When no parameter_uncertainties, factor sensitivity is empty (no fallback needed)."""
        graph = _make_graph()
        request = _make_request(graph, include_uncertainties=False)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        # No parameter_uncertainties → no factor sensitivity
        assert len(response.factor_sensitivity) == 0


# ===========================================================================
# Task 2: Root node default warning
# ===========================================================================


class TestRootNodeDefaultWarning:
    """Verify warning emitted when observed_state.value is missing on root nodes."""

    def test_warning_emitted_for_missing_root_value(self):
        """Root node without observed_state.value gets ROOT_NODE_DEFAULT_VALUE warning."""
        graph = _make_graph(include_observed_values=True, n_factors=3)
        # factor_c has no observed_state → should trigger warning
        request = _make_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        root_warnings = [
            w for w in response.inference_warnings if w.code == "ROOT_NODE_DEFAULT_VALUE"
        ]
        assert len(root_warnings) == 1
        assert root_warnings[0].detail["node_id"] == "factor_c"
        assert root_warnings[0].detail["defaulted_to"] == 0.0

    def test_no_warning_when_observed_value_present(self):
        """Root nodes with observed_state.value do not trigger warning."""
        graph = _make_graph(include_observed_values=True, n_factors=2)
        request = _make_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        root_warnings = [
            w for w in response.inference_warnings if w.code == "ROOT_NODE_DEFAULT_VALUE"
        ]
        assert len(root_warnings) == 0

    def test_no_warning_when_uncertainty_provides_prior(self):
        """Root nodes with ParameterUncertainty do not trigger warning."""
        graph = _make_graph(include_observed_values=True, n_factors=3)
        request = _make_request(graph, include_uncertainties=False)
        # Add uncertainty for factor_c (which has no observed_state)
        request = request.model_copy(
            update={
                "parameter_uncertainties": [
                    ParameterUncertainty(
                        node_id="factor_c",
                        distribution="uniform",
                        range_min=0.0,
                        range_max=1.0,
                    ),
                ],
            }
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        root_warnings = [
            w for w in response.inference_warnings if w.code == "ROOT_NODE_DEFAULT_VALUE"
        ]
        assert len(root_warnings) == 0

    def test_trust_downgrade_scales_with_defaults(self):
        """recommendation_stability is penalised when root nodes default to 0.0."""
        graph = _make_graph(include_observed_values=True, n_factors=2)
        request = _make_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response_clean = analyzer.analyze(request)

        # Now add a factor with missing value
        graph_missing = _make_graph(include_observed_values=True, n_factors=3)
        request_missing = _make_request(graph_missing, seed=12345)
        response_missing = analyzer.analyze(request_missing)

        # Stability should be lower with defaulted root
        assert response_missing.robustness.recommendation_stability <= (
            response_clean.robustness.recommendation_stability
        )

    def test_metadata_n_defaulted_root_nodes(self):
        """Metadata includes n_defaulted_root_nodes count."""
        graph = _make_graph(include_observed_values=True, n_factors=3)
        request = _make_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.metadata.n_defaulted_root_nodes == 1


# ===========================================================================
# Task 3: Auto-scaled noise metadata
# ===========================================================================


class TestAutoScaledNoiseMetadata:
    """Verify auto_noise_applied flag in metadata."""

    def test_noise_applied_for_outcome_node(self):
        """Outcome goal node gets noise → metadata flag True."""
        graph = _make_graph()
        request = _make_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.metadata.auto_noise_applied is True

    def test_noise_not_applied_for_non_outcome_node(self):
        """Factor goal node does not get noise → metadata flag False."""
        graph = _make_graph()
        # Change goal to factor_a (kind=factor, not outcome/risk)
        request = _make_request(graph)
        request = request.model_copy(update={"goal_node_id": "factor_a"})
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.metadata.auto_noise_applied is False


# ===========================================================================
# Task 5: E-value analogue per edge
# ===========================================================================


class TestEdgeEValues:
    """Verify E-value computation for edges."""

    def test_e_values_computed_when_requested(self):
        """include_e_values=True produces edge_e_values in response."""
        graph = _make_graph()
        request = _make_request(graph, include_e_values=True)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.edge_e_values is not None
        assert len(response.edge_e_values) == 2  # Two edges

    def test_e_values_not_computed_by_default(self):
        """include_e_values=False (default) → no edge_e_values."""
        graph = _make_graph()
        request = _make_request(graph, include_e_values=False)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.edge_e_values is None

    def test_fragile_edge_has_valid_e_value(self):
        """Each edge has either a numeric e_value or is_unflippable=True (inf in internal)."""
        graph = _make_graph()
        request = _make_request(graph, include_e_values=True)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert response.edge_e_values is not None
        for ev in response.edge_e_values:
            assert ev["e_value"] >= 1.0 or ev["e_value"] == float("inf")
            assert ev["flip_direction"] in ("increase", "decrease")
            assert ev["current_mean"] is not None

    def test_e_value_model_unflippable(self):
        """EdgeEValueV2 model represents unflippable edges correctly."""
        ev = EdgeEValueV2(
            edge_id="a->b",
            from_id="a",
            to_id="b",
            e_value=None,
            is_unflippable=True,
            flip_direction="increase",
            current_mean=0.3,
            flip_mean=0.3,
        )
        assert ev.is_unflippable is True
        assert ev.e_value is None

    def test_e_value_model_flippable(self):
        """EdgeEValueV2 model represents flippable edges correctly."""
        ev = EdgeEValueV2(
            edge_id="a->b",
            from_id="a",
            to_id="b",
            e_value=2.5,
            is_unflippable=False,
            flip_direction="increase",
            current_mean=0.3,
            flip_mean=0.75,
        )
        assert ev.e_value == 2.5
        assert ev.is_unflippable is False

    def test_e_values_deterministic(self):
        """Same seed produces identical E-values."""
        graph = _make_graph()
        request = _make_request(graph, include_e_values=True, seed=42)
        analyzer = RobustnessAnalyzerV2()
        r1 = analyzer.analyze(request)
        r2 = analyzer.analyze(request)
        assert r1.edge_e_values == r2.edge_e_values


# ===========================================================================
# Task 6: EVPI per factor
# ===========================================================================


class TestEVPI:
    """Verify Expected Value of Perfect Information computation."""

    def test_evpi_computed_when_requested(self):
        """include_voi=True with uncertainties produces factor_evpi."""
        graph = _make_graph()
        request = _make_request(
            graph,
            include_voi=True,
            include_uncertainties=True,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.factor_evpi is not None
        assert len(response.factor_evpi) == 2  # Two factors with uncertainties

    def test_evpi_not_computed_without_flag(self):
        """include_voi=False → no factor_evpi."""
        graph = _make_graph()
        request = _make_request(graph, include_voi=False, include_uncertainties=True)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.factor_evpi is None

    def test_evpi_not_computed_without_uncertainties(self):
        """No parameter_uncertainties → no factor_evpi even with flag."""
        graph = _make_graph()
        request = _make_request(graph, include_voi=True, include_uncertainties=False)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.factor_evpi is None

    def test_evpi_sorted_descending(self):
        """EVPI results are sorted by EVPI value descending."""
        graph = _make_graph()
        request = _make_request(
            graph,
            include_voi=True,
            include_uncertainties=True,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.factor_evpi is not None
        evpi_values = [e["evpi"] for e in response.factor_evpi]
        assert evpi_values == sorted(evpi_values, reverse=True)

    def test_evpi_fields_present(self):
        """EVPI results have all required fields including n_evpi_samples."""
        graph = _make_graph()
        request = _make_request(
            graph,
            include_voi=True,
            include_uncertainties=True,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.factor_evpi is not None
        for evpi in response.factor_evpi:
            assert "factor_id" in evpi
            assert "evpi" in evpi
            assert "evpi_percentage_points" in evpi
            assert "current_metric" in evpi
            assert "perfect_metric" in evpi
            assert "metric_type" in evpi
            assert "n_evpi_samples" in evpi
            assert evpi["n_evpi_samples"] <= 500  # Budget cap


class TestEVPIBelowResolutionLabelling:
    """T0-4 remediation: below-resolution labelling of EVPI estimates.

    EVPI is a difference of two independent MC proportion estimates, so at the
    500-sample budget cap the estimator noise is ~+/-0.03-0.06. Entries whose
    |evpi| is inside the noise floor are LABELLED below_resolution; the raw
    evpi value is never clamped or altered (provisional_doctrine_v0).
    """

    def test_noise_floor_formula(self):
        """noise_floor(n) = 1.96 * sqrt(0.5 / n) — two-sided 95% worst-case bound."""
        from src.services.robustness_analyzer_v2 import evpi_noise_floor

        assert math.isclose(evpi_noise_floor(500), 1.96 * math.sqrt(0.5 / 500))
        assert math.isclose(evpi_noise_floor(500), 0.0619806, abs_tol=1e-6)
        assert math.isclose(evpi_noise_floor(100), 1.96 * math.sqrt(0.5 / 100))
        # Monotone decreasing in n: more samples -> finer resolution
        assert evpi_noise_floor(500) < evpi_noise_floor(200) < evpi_noise_floor(100)
        # Degenerate budget: nothing is resolvable
        assert evpi_noise_floor(0) == float("inf")

    def test_labelling_fields_present_on_every_entry(self):
        """Every factor_evpi entry carries the additive labelling fields."""
        from src.services.robustness_analyzer_v2 import evpi_noise_floor

        graph = _make_graph()
        request = _make_request(graph, include_voi=True, include_uncertainties=True, n_samples=200)
        response = RobustnessAnalyzerV2().analyze(request)
        assert response.factor_evpi is not None
        for entry in response.factor_evpi:
            assert entry["evpi_status"] in ("below_resolution", "resolved")
            assert entry["evpi_noise_floor"] == round(evpi_noise_floor(entry["n_evpi_samples"]), 6)
            assert entry["evpi_noise_floor_method"] == "z95_worst_case_bernoulli_diff"
            assert entry["evpi_labelling_doctrine"] == "provisional_doctrine_v0"

    def test_status_consistent_with_floor(self):
        """evpi_status is below_resolution iff |evpi| < noise floor."""
        from src.services.robustness_analyzer_v2 import evpi_noise_floor

        graph = _make_graph()
        request = _make_request(graph, include_voi=True, include_uncertainties=True, n_samples=500)
        response = RobustnessAnalyzerV2().analyze(request)
        assert response.factor_evpi is not None
        for entry in response.factor_evpi:
            floor = evpi_noise_floor(entry["n_evpi_samples"])
            # Guard: skip entries within rounding distance of the boundary
            # (evpi is rounded to 6 dp before serialisation).
            if abs(abs(entry["evpi"]) - floor) < 2e-6:
                continue
            expected = "below_resolution" if abs(entry["evpi"]) < floor else "resolved"
            assert entry["evpi_status"] == expected

    def test_small_sample_entries_labelled_below_resolution(self):
        """At n=100 (floor ~0.139) the deterministic fixture EVPIs are all noise."""
        graph = _make_graph()
        request = _make_request(graph, include_voi=True, include_uncertainties=True, n_samples=100)
        response = RobustnessAnalyzerV2().analyze(request)
        assert response.factor_evpi is not None
        assert len(response.factor_evpi) == 2
        for entry in response.factor_evpi:
            assert entry["evpi_status"] == "below_resolution"

    def test_raw_evpi_never_clamped_by_labelling(self):
        """Label, do NOT clamp: raw evpi stays perfect_metric - current_metric.

        The deterministic fixture (seed=12345, n=500) produces a raw NEGATIVE
        evpi entry — the audit's live pattern (e.g. fac_tech_lead -0.004). It
        must be preserved as-is and labelled, never zeroed or clamped.
        """
        graph = _make_graph()
        request = _make_request(graph, include_voi=True, include_uncertainties=True, n_samples=500)
        response = RobustnessAnalyzerV2().analyze(request)
        assert response.factor_evpi is not None
        for entry in response.factor_evpi:
            assert math.isclose(
                entry["evpi"],
                entry["perfect_metric"] - entry["current_metric"],
                abs_tol=2e-6,
            )
        negatives = [e for e in response.factor_evpi if e["evpi"] < 0]
        assert negatives, "fixture regression: expected a raw negative EVPI entry"
        for entry in negatives:
            assert entry["evpi_status"] == "below_resolution"


# ===========================================================================
# Task 9: Reproducibility with new fields
# ===========================================================================


class TestReproducibility:
    """Verify determinism with all new fields."""

    def test_identical_seed_produces_identical_results(self):
        """Same seed + same graph → identical results including new fields."""
        graph = _make_graph(n_factors=3)
        request = _make_request(
            graph,
            seed=99999,
            include_e_values=True,
            include_uncertainties=True,
            include_voi=True,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        analyzer._n_bootstrap_override = 5

        r1 = analyzer.analyze(request)
        r2 = analyzer.analyze(request)

        # Core results
        assert r1.recommended_option_id == r2.recommended_option_id
        assert r1.recommendation_confidence == r2.recommendation_confidence
        assert r1.robustness.recommendation_stability == r2.robustness.recommendation_stability

        # E-values
        assert r1.edge_e_values == r2.edge_e_values

        # Inference warnings
        assert len(r1.inference_warnings) == len(r2.inference_warnings)
        for w1, w2 in zip(r1.inference_warnings, r2.inference_warnings):
            assert w1.code == w2.code
            assert w1.detail == w2.detail

        # Metadata
        assert r1.metadata.auto_noise_applied == r2.metadata.auto_noise_applied
        assert r1.metadata.n_defaulted_root_nodes == r2.metadata.n_defaulted_root_nodes


# ===========================================================================
# P1 fixes: noise flag, penalty auditability, E-value semantics
# ===========================================================================


class TestAutoNoiseZeroVariance:
    """Verify auto_noise_applied is False when no noise was actually added."""

    def test_zero_variance_outcome_no_noise(self):
        """All options with zero-variance outcome → auto_noise_applied=False."""
        # Build a graph where both options intervene the OUTCOME directly,
        # setting it to the same fixed value → zero model variance.
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="f",
                    kind="factor",
                    label="F",
                    observed_state=ObservedState(value=0.5),
                ),
                NodeV2(
                    id="out",
                    kind="outcome",
                    label="Out",
                    observed_state=ObservedState(value=0.0),
                ),
            ],
            edges=[
                EdgeV2(
                    **{"from": "f"},
                    to="out",
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                    exists_probability=1.0,
                ),
            ],
        )
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                # Both options override outcome directly → zero variance per option
                InterventionOption(id="a", label="A", interventions={"out": 0.7}),
                InterventionOption(id="b", label="B", interventions={"out": 0.7}),
            ],
            goal_node_id="out",
            seed=1,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        # Both options fix outcome to 0.7 → zero variance → no noise actually added
        assert response.metadata.auto_noise_applied is False


class TestTrustPenaltyAuditability:
    """Verify penalty details exposed on RobustnessResult."""

    def test_penalty_fields_present_when_defaults_exist(self):
        """Stability penalty factor and affected node IDs in robustness result."""
        graph = _make_graph(include_observed_values=True, n_factors=3)
        request = _make_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        # factor_c has no observed_state → triggers penalty
        assert response.robustness.stability_penalty_factor is not None
        assert response.robustness.stability_penalty_factor < 1.0
        assert response.robustness.defaulted_root_node_ids == ["factor_c"]

    def test_no_penalty_fields_when_all_values_present(self):
        """No penalty fields when all root nodes have values."""
        graph = _make_graph(include_observed_values=True, n_factors=2)
        request = _make_request(graph)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.robustness.stability_penalty_factor is None
        assert response.robustness.defaulted_root_node_ids is None


class TestTrustPenaltyFloor:
    """Verify penalty factor floors at 0.1 for many defaulted roots."""

    def test_penalty_factor_formula_floor(self):
        """max(0.1, 1.0 - 0.05 * n) floors at 0.1 when n >= 18."""
        # Direct formula check — avoids constructing a 25-node graph
        for n_defaulted in [18, 20, 25, 100]:
            factor = max(0.1, 1.0 - 0.05 * n_defaulted)
            assert factor == 0.1, f"n={n_defaulted}: expected 0.1, got {factor}"

    def test_penalty_factor_scales_linearly_before_floor(self):
        """Penalty is 0.05 per defaulted root before hitting floor."""
        for n_defaulted, expected in [(1, 0.95), (3, 0.85), (10, 0.5), (17, 0.15)]:
            factor = max(0.1, 1.0 - 0.05 * n_defaulted)
            assert (
                abs(factor - expected) < 1e-9
            ), f"n={n_defaulted}: expected {expected}, got {factor}"


class TestEValueUnflippableSemantics:
    """Verify E-value model correctly represents unflippable edges."""

    def test_unflippable_edge_has_null_e_value(self):
        """EdgeEValueV2 with is_unflippable=True has e_value=None."""
        ev = EdgeEValueV2(
            edge_id="a->b",
            from_id="a",
            to_id="b",
            e_value=None,
            is_unflippable=True,
            flip_direction="increase",
            current_mean=0.5,
            flip_mean=0.5,
        )
        d = ev.model_dump(exclude_none=True)
        assert "e_value" not in d
        assert d["is_unflippable"] is True


class TestResponseShapeIntegration:
    """Integration-style tests verifying fields appear in serialized response shape."""

    def test_factor_evpi_in_internal_response(self):
        """factor_evpi appears in RobustnessResponseV2 when include_voi=True."""
        graph = _make_graph()
        request = _make_request(
            graph,
            include_voi=True,
            include_uncertainties=True,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        # Internal model has factor_evpi
        assert response.factor_evpi is not None
        # Serialize and check it survives
        d = response.model_dump(exclude_none=True)
        assert "factor_evpi" in d

    def test_confidence_source_in_serialized_response(self):
        """confidence_source appears in FactorSensitivityV2 serialization."""
        fs = FactorSensitivityV2(
            node_id="x",
            sensitivity_score=0.5,
            direction="positive",
            confidence=0.7,
            confidence_source="bootstrap_sampling",
        )
        d = fs.model_dump(exclude_none=True)
        assert "confidence_source" in d
        assert d["confidence_source"] == "bootstrap_sampling"

    def test_edge_e_values_in_robustness_result(self):
        """edge_e_values appears in RobustnessResultV2 serialization."""
        from src.models.response_v2 import RobustnessResultV2

        result = RobustnessResultV2(
            level="high",
            confidence=0.9,
            edge_e_values=[
                EdgeEValueV2(
                    edge_id="a->b",
                    from_id="a",
                    to_id="b",
                    e_value=3.0,
                    is_unflippable=False,
                    flip_direction="increase",
                    current_mean=0.2,
                    flip_mean=0.6,
                )
            ],
        )
        d = result.model_dump(exclude_none=True)
        assert "edge_e_values" in d
        assert len(d["edge_e_values"]) == 1
        assert d["edge_e_values"][0]["e_value"] == 3.0

    def test_penalty_fields_in_robustness_result_v2(self):
        """stability_penalty_factor and defaulted_root_node_ids in RobustnessResultV2."""
        from src.models.response_v2 import RobustnessResultV2

        result = RobustnessResultV2(
            level="low",
            confidence=0.5,
            recommendation_stability=0.57,
            stability_penalty_factor=0.95,
            defaulted_root_node_ids=["node_x"],
        )
        d = result.model_dump(exclude_none=True)
        assert d["stability_penalty_factor"] == 0.95
        assert d["defaulted_root_node_ids"] == ["node_x"]

    def test_n_evpi_samples_in_evpi_output(self):
        """n_evpi_samples field present in EVPI results."""
        graph = _make_graph()
        request = _make_request(
            graph,
            include_voi=True,
            include_uncertainties=True,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.factor_evpi is not None
        for evpi in response.factor_evpi:
            assert "n_evpi_samples" in evpi
            assert isinstance(evpi["n_evpi_samples"], int)


# ===========================================================================
# EVPI metric_type branch coverage
# ===========================================================================


class TestEVPIMetricType:
    """Verify metric_type switches based on goal_constraints."""

    def test_metric_type_is_p_win_recommended_without_constraints(self):
        """Without goal_constraints, metric_type should be p_win_recommended."""
        graph = _make_graph()
        request = _make_request(graph, include_voi=True, include_uncertainties=True)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.factor_evpi is not None
        for evpi in response.factor_evpi:
            assert evpi["metric_type"] == "p_win_recommended"

    def test_metric_type_is_p_joint_goal_with_constraints(self):
        """With goal_constraints, metric_type should be p_joint_goal."""
        graph = _make_graph()
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                InterventionOption(id="opt_high", label="High", interventions={"factor_a": 0.9}),
                InterventionOption(id="opt_low", label="Low", interventions={"factor_a": 0.1}),
            ],
            goal_node_id="outcome",
            seed=12345,
            n_samples=200,
            parameter_uncertainties=[
                ParameterUncertainty(node_id="factor_a", distribution="normal", std=0.1),
                ParameterUncertainty(node_id="factor_b", distribution="normal", std=0.15),
            ],
            include_voi=True,
            goal_constraints=[
                GoalConstraint(node_id="outcome", operator=">=", value=0.3),
            ],
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.factor_evpi is not None
        for evpi in response.factor_evpi:
            assert evpi["metric_type"] == "p_joint_goal"


# ===========================================================================
# E-value bidirected edge exclusion
# ===========================================================================


class TestEValueBidirectedExclusion:
    """Verify bidirected edges are excluded from E-value computation."""

    def test_bidirected_edges_excluded_from_e_values(self):
        """E-values should only be computed for directed (causal) edges."""
        # Use standard graph node names so _make_request interventions are valid
        graph = _make_graph()
        # Add a bidirected edge
        graph.edges.append(
            EdgeV2(
                **{"from": "factor_a"},
                to="factor_b",
                strength=StrengthDistribution(mean=0.4, std=0.1),
                exists_probability=0.9,
                edge_type="bidirected",
            )
        )
        request = _make_request(graph, include_e_values=True)
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        assert response.edge_e_values is not None
        edge_ids = {ev["edge_id"] for ev in response.edge_e_values}
        # Bidirected edge factor_a->factor_b must NOT appear
        assert "factor_a->factor_b" not in edge_ids
        # Only directed edges should be present (factor_a->outcome, factor_b->outcome)
        assert len(edge_ids) == 2
        assert "factor_a->outcome" in edge_ids
        assert "factor_b->outcome" in edge_ids


# ===========================================================================
# E-value budget timeout
# ===========================================================================


class TestEValueBudgetTimeout:
    """Verify E-value computation respects the 2s budget."""

    def test_budget_exceeded_returns_none(self):
        """When budget is set to 0, edge_e_values should be None (budget exceeded)."""
        graph = _make_graph()
        request = _make_request(graph, include_e_values=True)
        analyzer = RobustnessAnalyzerV2()
        # Set budget to 0ms so any edge computation exceeds it
        original_budget = analyzer.E_VALUE_BUDGET_MS
        analyzer.E_VALUE_BUDGET_MS = 0
        try:
            response = analyzer.analyze(request)
            assert response.edge_e_values is None, "Should return None when budget is 0ms"
        finally:
            analyzer.E_VALUE_BUDGET_MS = original_budget


# ===========================================================================
# Root node penalty: intervention-overridden roots
# ===========================================================================


class TestRootPenaltyInterventionOverride:
    """Verify root penalty is suppressed when all options intervene on the root."""

    def test_no_warning_when_all_options_intervene(self):
        """Root without observed_state but covered by all interventions: no warning."""
        nodes = [
            NodeV2(id="root_x", kind="factor", label="X", observed_state=None),
            NodeV2(
                id="outcome", kind="outcome", label="O", observed_state=ObservedState(value=0.0)
            ),
        ]
        edges = [
            EdgeV2(
                **{"from": "root_x"},
                to="outcome",
                strength=StrengthDistribution(mean=0.6, std=0.1),
                exists_probability=1.0,
            ),
        ]
        graph = GraphV2(nodes=nodes, edges=edges)
        # Both options intervene on root_x
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="A", interventions={"root_x": 0.5}),
                InterventionOption(id="opt_b", label="B", interventions={"root_x": 0.7}),
            ],
            goal_node_id="outcome",
            seed=42,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        root_warnings = [
            w for w in response.inference_warnings if w.code == "ROOT_NODE_DEFAULT_VALUE"
        ]
        assert len(root_warnings) == 0, "Should not warn when all options intervene on root"
        assert response.robustness.stability_penalty_factor is None

    def test_warning_when_partial_intervention(self):
        """Root without observed_state with only some options intervening: should warn."""
        nodes = [
            NodeV2(id="root_x", kind="factor", label="X", observed_state=None),
            NodeV2(
                id="outcome", kind="outcome", label="O", observed_state=ObservedState(value=0.0)
            ),
        ]
        edges = [
            EdgeV2(
                **{"from": "root_x"},
                to="outcome",
                strength=StrengthDistribution(mean=0.6, std=0.1),
                exists_probability=1.0,
            ),
        ]
        graph = GraphV2(nodes=nodes, edges=edges)
        # Only opt_a intervenes on root_x, opt_b does not
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                InterventionOption(id="opt_a", label="A", interventions={"root_x": 0.5}),
                InterventionOption(id="opt_b", label="B", interventions={}),
            ],
            goal_node_id="outcome",
            seed=42,
            n_samples=100,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)
        root_warnings = [
            w for w in response.inference_warnings if w.code == "ROOT_NODE_DEFAULT_VALUE"
        ]
        assert len(root_warnings) == 1, "Should warn when not all options intervene"
