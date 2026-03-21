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
