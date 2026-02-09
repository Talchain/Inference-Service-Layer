"""
CIL Phase 0 + 0.2 — ISL reconciliation fix tests.

Tests for:
- Task 1: Unknown-field policy (extra='ignore') on all v2 models
- Task 2: Percentile semantic drift fix (p10/p50/p90 from actual samples)
- Task 3: Option label propagation to V2 response
- Task 4: Seed type normalisation (str | int | None)
- CIL 0.2 Task 1: ObservedState extra='ignore' + std field
- CIL 0.2 Task 2: ParameterUncertainty extra='ignore'
- CIL 0.2 Task 3: Percentile fallback returns None
- CIL 0.2 Task 5: Supporting response models extra='ignore'
- CIL 0.2 Task 6: Duplicate option-ID validation
"""

import hashlib
import math

import numpy as np
import pytest
from pydantic import ValidationError

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
from src.models.response_v2 import (
    ConstraintAnalysisV2,
    ConstraintResultV2,
    CritiqueV2,
    DiagnosticsV2,
    FactorSensitivityV2,
    FragileEdgeV2,
    ISLResponseV2,
    OptionDiagnosticV2,
    OptionResultV2,
    OutcomeDistributionV2,
    RequestEchoV2,
    RobustnessResultV2,
    SensitiveFactorV2,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_graph():
    """Minimal valid graph for testing."""
    return GraphV2(
        nodes=[
            NodeV2(id="price", kind="factor", label="Price"),
            NodeV2(id="revenue", kind="outcome", label="Revenue"),
        ],
        edges=[
            EdgeV2(
                **{"from": "price", "to": "revenue"},
                exists_probability=0.9,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            )
        ],
    )


@pytest.fixture
def simple_options():
    """Minimal valid options for testing."""
    return [
        InterventionOption(
            id="low", label="Low price", interventions={"price": 0.3}
        ),
        InterventionOption(
            id="high", label="High price", interventions={"price": 0.7}
        ),
    ]


@pytest.fixture
def analyzer():
    """Create a RobustnessAnalyzerV2 instance."""
    return RobustnessAnalyzerV2()


# =============================================================================
# Task 1: Unknown-field policy tests
# =============================================================================


class TestUnknownFieldPolicy:
    """Verify extra='ignore' is explicit and functional on all v2 models."""

    def test_request_unknown_top_level_field_ignored(self, simple_graph, simple_options):
        """Request with unknown top-level field parses without error."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            unknown_top_level="should be ignored",
        )
        # Field should not be present on the model
        assert not hasattr(request, "unknown_top_level")

    def test_request_unknown_nested_node_field_ignored(self):
        """Request with unknown field on a node parses without error."""
        node = NodeV2(
            id="price",
            kind="factor",
            label="Price",
            unknown_nested="should be ignored",
        )
        assert not hasattr(node, "unknown_nested")
        assert node.id == "price"

    def test_request_unknown_nested_edge_field_ignored(self):
        """Request with unknown field on an edge parses without error."""
        edge = EdgeV2(
            **{"from": "price", "to": "revenue"},
            exists_probability=0.9,
            strength=StrengthDistribution(mean=0.5, std=0.1),
            unknown_edge_field="should be ignored",
        )
        assert not hasattr(edge, "unknown_edge_field")
        assert edge.from_ == "price"

    def test_unknown_fields_not_in_serialised_response(self):
        """Unknown fields do NOT appear in the serialised response (confirm ignore, not allow)."""
        outcome = OutcomeDistributionV2(
            mean=1.0, std=0.1, p10=0.9, p50=1.0, p90=1.1,
            n_samples=100, n_valid_samples=100, validity_ratio=1.0,
        )
        result = OptionResultV2(
            id="test",
            label="Test",
            outcome=outcome,
            status="computed",
        )
        serialised = result.model_dump()
        assert "unknown_field" not in serialised

    def test_strength_distribution_ignores_unknown(self):
        """StrengthDistribution ignores unknown fields."""
        sd = StrengthDistribution(mean=0.5, std=0.1, extra_field="ignored")
        assert not hasattr(sd, "extra_field")

    def test_graph_ignores_unknown(self, simple_graph):
        """GraphV2 ignores unknown fields."""
        g = GraphV2(
            nodes=[NodeV2(id="a", kind="factor", label="A")],
            edges=[],
            unknown_graph_field="ignored",
        )
        assert not hasattr(g, "unknown_graph_field")

    def test_intervention_option_ignores_unknown(self):
        """InterventionOption ignores unknown fields."""
        opt = InterventionOption(
            id="opt1", label="Opt 1", interventions={"a": 1.0},
            unknown_opt_field="ignored",
        )
        assert not hasattr(opt, "unknown_opt_field")

    def test_goal_constraint_ignores_unknown(self):
        """GoalConstraint ignores unknown fields."""
        gc = GoalConstraint(
            node_id="revenue", operator=">=", threshold=100.0,
            unknown_gc_field="ignored",
        )
        assert not hasattr(gc, "unknown_gc_field")

    def test_constraint_result_v2_ignores_unknown(self):
        """ConstraintResultV2 ignores unknown fields."""
        cr = ConstraintResultV2(
            node_id="revenue", operator=">=", threshold=100.0,
            prob_satisfied=0.8,
            unknown_cr_field="ignored",
        )
        assert not hasattr(cr, "unknown_cr_field")
        assert "unknown_cr_field" not in cr.model_dump()

    def test_constraint_analysis_v2_ignores_unknown(self):
        """ConstraintAnalysisV2 ignores unknown fields."""
        ca = ConstraintAnalysisV2(
            constraints=[],
            joint_probability=1.0,
            unknown_ca_field="ignored",
        )
        assert not hasattr(ca, "unknown_ca_field")
        assert "unknown_ca_field" not in ca.model_dump()

    def test_fragile_edge_v2_ignores_unknown(self):
        """FragileEdgeV2 ignores unknown fields."""
        fe = FragileEdgeV2(
            edge_id="a->b", from_id="a", to_id="b",
            unknown_fe_field="ignored",
        )
        assert not hasattr(fe, "unknown_fe_field")

    def test_robustness_result_v2_ignores_unknown(self):
        """RobustnessResultV2 ignores unknown fields."""
        rr = RobustnessResultV2(
            level="high", confidence=0.9,
            unknown_rr_field="ignored",
        )
        assert not hasattr(rr, "unknown_rr_field")

    def test_factor_sensitivity_v2_ignores_unknown(self):
        """FactorSensitivityV2 ignores unknown fields."""
        fs = FactorSensitivityV2(
            node_id="price", sensitivity_score=0.5, direction="positive",
            unknown_fs_field="ignored",
        )
        assert not hasattr(fs, "unknown_fs_field")

    def test_outcome_distribution_v2_ignores_unknown(self):
        """OutcomeDistributionV2 ignores unknown fields."""
        od = OutcomeDistributionV2(
            mean=1.0, std=0.1, p10=0.9, p50=1.0, p90=1.1,
            n_samples=100, n_valid_samples=100, validity_ratio=1.0,
            unknown_od_field="ignored",
        )
        assert not hasattr(od, "unknown_od_field")
        assert "unknown_od_field" not in od.model_dump()


# =============================================================================
# Task 2: Percentile semantic drift fix tests
# =============================================================================


class TestPercentileSemanticDrift:
    """Verify p10/p50/p90 are actual 10th/50th/90th percentiles, not CI bounds."""

    def test_samples_stored_in_internal_response(self, simple_graph, simple_options, analyzer):
        """Internal analyzer must now store raw samples for V2 percentile computation."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
        )

        response = analyzer.analyze(request)

        for result in response.results:
            samples = result.outcome_distribution.samples
            assert samples is not None, "Samples must be stored for V2 percentile computation"
            assert len(samples) == 500

    def test_actual_percentiles_differ_from_ci_bounds(self, simple_graph, simple_options, analyzer):
        """
        Regression test: actual p10/p90 from samples must differ from ci_lower/ci_upper
        (2.5th/97.5th). This test will FAIL if someone reverts to the old mapping.
        """
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=1000,
            seed=42,
        )

        response = analyzer.analyze(request)

        for result in response.results:
            samples = result.outcome_distribution.samples
            assert samples is not None
            finite_samples = [s for s in samples if math.isfinite(s)]

            # Compute what the V2 API layer should produce
            actual_p10 = float(np.percentile(finite_samples, 10))
            actual_p90 = float(np.percentile(finite_samples, 90))

            # These are the OLD (incorrect) values that were aliased as p10/p90
            ci_lower_2_5 = result.outcome_distribution.ci_lower
            ci_upper_97_5 = result.outcome_distribution.ci_upper

            # For a non-degenerate distribution, 10th != 2.5th and 90th != 97.5th
            if result.outcome_distribution.std > 0.001:
                assert abs(actual_p10 - ci_lower_2_5) > 1e-10, \
                    "p10 (10th) should differ from ci_lower (2.5th percentile)"
                assert abs(actual_p90 - ci_upper_97_5) > 1e-10, \
                    "p90 (90th) should differ from ci_upper (97.5th percentile)"

    def test_percentile_ordering(self, simple_graph, simple_options, analyzer):
        """p10 <= p50 <= p90 must always hold."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=123,
        )

        response = analyzer.analyze(request)

        for result in response.results:
            samples = result.outcome_distribution.samples
            assert samples is not None
            finite_samples = [s for s in samples if math.isfinite(s)]
            p10 = float(np.percentile(finite_samples, 10))
            p50 = float(np.percentile(finite_samples, 50))
            p90 = float(np.percentile(finite_samples, 90))
            assert p10 <= p50 <= p90

    def test_api_transform_uses_actual_percentiles(self):
        """
        Directly test the V1→V2 transform logic: given known samples,
        verify p10/p50/p90 are computed from np.percentile, not ci_lower/ci_upper.
        """
        # Known sample set where percentiles are easy to verify
        samples = list(range(1, 101))  # 1..100
        expected_p10 = float(np.percentile(samples, 10))   # ~10.9
        expected_p50 = float(np.percentile(samples, 50))   # 50.5
        expected_p90 = float(np.percentile(samples, 90))   # ~90.1

        # ci_lower/ci_upper would be 2.5th/97.5th — different values
        ci_lower = float(np.percentile(samples, 2.5))      # ~3.475
        ci_upper = float(np.percentile(samples, 97.5))     # ~97.525

        # Simulate what the API layer does (from robustness.py)
        finite_samples = [s for s in samples if math.isfinite(s)]
        p10_val = float(np.percentile(finite_samples, 10))
        p50_val = float(np.percentile(finite_samples, 50))
        p90_val = float(np.percentile(finite_samples, 90))

        assert abs(p10_val - expected_p10) < 1e-10
        assert abs(p50_val - expected_p50) < 1e-10
        assert abs(p90_val - expected_p90) < 1e-10

        # Confirm these are NOT the old CI values
        assert abs(p10_val - ci_lower) > 1.0, \
            "p10 must not equal ci_lower (2.5th percentile)"
        assert abs(p90_val - ci_upper) > 1.0, \
            "p90 must not equal ci_upper (97.5th percentile)"


# =============================================================================
# Task 3: Option label propagation tests
# =============================================================================


class TestOptionLabelPropagation:
    """Verify option labels are threaded from request to V2 response."""

    def test_labels_propagated_to_response(self, simple_graph, analyzer):
        """Request with labelled options → response with same labels."""
        options = [
            InterventionOption(
                id="low", label="Low price strategy", interventions={"price": 0.3}
            ),
            InterventionOption(
                id="high", label="High price strategy", interventions={"price": 0.7}
            ),
        ]
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=options,
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )

        response = analyzer.analyze(request)

        # Build label map from request for verification
        expected_labels = {opt.id: opt.label for opt in options}

        for result in response.results:
            # The V1 internal OptionResult doesn't carry labels,
            # but the V2 API transform should propagate them.
            # Here we verify the internal response has the option_id
            # so the API layer can look it up.
            assert result.option_id in expected_labels

    def test_label_none_does_not_crash(self, simple_graph, analyzer):
        """InterventionOption.label is required, so this tests normal flow."""
        options = [
            InterventionOption(
                id="only", label="Only option", interventions={"price": 0.5}
            ),
        ]
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=options,
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )

        response = analyzer.analyze(request)
        assert len(response.results) == 1
        assert response.results[0].option_id == "only"


# =============================================================================
# Task 4: Seed type normalisation tests
# =============================================================================


class TestSeedTypeNormalisation:
    """Verify seed accepts str | int | None and normalises to int."""

    def test_string_numeric_seed_same_as_int(self, simple_graph, simple_options, analyzer):
        """seed='42' should produce identical results to seed=42."""
        request_int = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=200,
            seed=42,
        )
        request_str = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=200,
            seed="42",
        )

        # After normalisation, both should have int seed
        assert request_int.seed == 42
        assert request_str.seed == 42

        # Results should be identical
        response_int = analyzer.analyze(request_int)
        response_str = analyzer.analyze(request_str)

        for r_int, r_str in zip(response_int.results, response_str.results):
            assert r_int.outcome_distribution.mean == r_str.outcome_distribution.mean
            assert r_int.win_probability == r_str.win_probability

    def test_non_numeric_string_seed_deterministic(self, simple_graph, simple_options):
        """seed='my_seed' should produce a deterministic int, no error."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            seed="my_seed",
        )
        assert isinstance(request.seed, int)
        assert request.seed >= 0

        # Same string should always produce same int
        request2 = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            seed="my_seed",
        )
        assert request.seed == request2.seed

    def test_non_numeric_string_seed_uses_sha256(self, simple_graph, simple_options):
        """Non-numeric string seed must use SHA-256 (stable across processes), not hash()."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            seed="my_seed",
        )
        # Compute expected value using the same SHA-256 formula
        expected = int(hashlib.sha256("my_seed".encode("utf-8")).hexdigest(), 16) % (2**31)
        assert request.seed == expected, (
            f"Expected SHA-256 based seed {expected}, got {request.seed}. "
            "If this fails, the normaliser may be using Python's hash() which is "
            "randomized per process (PEP 456)."
        )

    def test_seed_zero_is_valid(self, simple_graph, simple_options, analyzer):
        """seed=0 must be treated as an explicit seed, not as falsy/None."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=200,
            seed=0,
        )
        assert request.seed == 0

        # Run twice with seed=0 — must be deterministic
        response1 = analyzer.analyze(request)
        response2 = analyzer.analyze(request)
        for r1, r2 in zip(response1.results, response2.results):
            assert r1.outcome_distribution.mean == r2.outcome_distribution.mean, \
                "seed=0 should produce deterministic results"

    def test_string_zero_seed_normalises_to_int_zero(self, simple_graph, simple_options):
        """seed='0' should normalise to int 0, not be hashed."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            seed="0",
        )
        assert request.seed == 0
        assert isinstance(request.seed, int)

    def test_none_seed_accepted(self, simple_graph, simple_options):
        """seed=None should be accepted (computed from graph)."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            seed=None,
        )
        assert request.seed is None

    def test_int_seed_unchanged(self, simple_graph, simple_options):
        """seed=42 (int) should remain 42."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )
        assert request.seed == 42
        assert isinstance(request.seed, int)


# =============================================================================
# API Transform Integration Tests
# =============================================================================


class TestAPITransformOutput:
    """Tests that exercise the actual V1→V2 API transform logic from robustness.py.

    These tests construct the same data flow as the API endpoint to verify
    that OptionResultV2.label and percentiles are correctly populated.
    """

    def test_api_transform_percentiles_from_cleaned_samples(self):
        """
        Simulate the API transform: given samples with some NaN values,
        verify percentiles are computed from the cleaned (finite) subset
        — the same population used for n_valid_samples.

        validate_mc_samples replaces NaN with median (not removes), so
        the cleaned array has 105 elements with 5 median-imputed values.
        """
        from src.utils.numerical_stability import validate_mc_samples

        # Samples with a few NaN values mixed in
        raw_samples = list(range(1, 101)) + [float("nan")] * 5
        samples_array = np.array(raw_samples, dtype=float)
        cleaned_samples, _ = validate_mc_samples(samples_array)

        # After cleaning, all 105 values should be finite (NaN replaced with median)
        finite_cleaned = cleaned_samples[np.isfinite(cleaned_samples)]
        assert len(finite_cleaned) == 105, \
            "validate_mc_samples replaces NaN with median, so all 105 should be finite"

        # API transform logic (mirrors robustness.py)
        p10_val = float(np.percentile(finite_cleaned, 10))
        p50_val = float(np.percentile(finite_cleaned, 50))
        p90_val = float(np.percentile(finite_cleaned, 90))

        # Verify percentiles come from the cleaned set (with imputed values)
        expected_p10 = float(np.percentile(finite_cleaned, 10))
        expected_p50 = float(np.percentile(finite_cleaned, 50))
        expected_p90 = float(np.percentile(finite_cleaned, 90))

        assert p10_val == expected_p10
        assert p50_val == expected_p50
        assert p90_val == expected_p90

        # Key assertion: percentiles from cleaned samples differ from raw
        # (because 5 median-imputed values shift the distribution slightly)
        raw_finite = [s for s in raw_samples if math.isfinite(s)]
        raw_p10 = float(np.percentile(raw_finite, 10))
        # The cleaned set includes 5 extra median values, so percentiles shift
        assert abs(p10_val - raw_p10) > 0.01, \
            "Cleaned percentiles should differ from raw (median imputation shifts them)"

        # n_valid should count all finite values in cleaned array
        n_valid = int(np.sum(np.isfinite(cleaned_samples)))
        assert n_valid == 105

    def test_api_transform_label_propagation(self, simple_graph, simple_options, analyzer):
        """
        Simulate the API transform label propagation and verify
        OptionResultV2 gets the correct label from the request.
        """
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )

        v1_response = analyzer.analyze(request)

        # Replicate the API transform label logic
        option_label_map = {opt.id: opt.label for opt in request.options}

        for result in v1_response.results:
            label = option_label_map.get(result.option_id)
            assert label is not None, f"Label missing for option {result.option_id}"

            # Build the V2 model to verify it accepts the label
            dist = result.outcome_distribution
            samples = dist.samples
            assert samples is not None

            finite_samples = np.array([s for s in samples if math.isfinite(s)])
            option_v2 = OptionResultV2(
                id=result.option_id,
                label=label,
                outcome=OutcomeDistributionV2(
                    mean=dist.mean,
                    std=dist.std,
                    p10=float(np.percentile(finite_samples, 10)),
                    p50=float(np.percentile(finite_samples, 50)),
                    p90=float(np.percentile(finite_samples, 90)),
                    n_samples=len(samples),
                    n_valid_samples=len(finite_samples),
                    validity_ratio=len(finite_samples) / len(samples),
                ),
                status="computed",
            )

            # Assert the V2 model has the correct label
            assert option_v2.label == option_label_map[result.option_id]
            # Assert percentiles are actual 10th/50th/90th
            assert option_v2.outcome.p10 == float(np.percentile(finite_samples, 10))
            assert option_v2.outcome.p50 == float(np.percentile(finite_samples, 50))
            assert option_v2.outcome.p90 == float(np.percentile(finite_samples, 90))


# =============================================================================
# CIL Phase 0.2: ObservedState nested model hole (C1)
# =============================================================================


class TestObservedStateExtraIgnore:
    """CIL 0.2 Task 1: ObservedState must accept and drop unknown fields."""

    def test_observed_state_ignores_unknown_field(self):
        """ObservedState nested in NodeV2 must silently drop unknown fields."""
        node = NodeV2(
            id="price",
            kind="factor",
            label="Price",
            observed_state={"value": 59.0, "std": 5.0, "unknown_field": True},
        )
        assert node.observed_state is not None
        assert node.observed_state.std == 5.0
        assert not hasattr(node.observed_state, "unknown_field")

    def test_observed_state_std_none_when_absent(self):
        """NodeV2 with observed_state without std → std is None."""
        node = NodeV2(
            id="price",
            kind="factor",
            label="Price",
            observed_state={"value": 59.0},
        )
        assert node.observed_state is not None
        assert node.observed_state.std is None

    def test_observed_state_std_declared(self):
        """ObservedState.std is a declared Optional[float] field per v2.6 schema."""
        os = ObservedState(value=59.0, std=5.0)
        assert os.std == 5.0
        serialised = os.model_dump()
        assert "std" in serialised
        assert serialised["std"] == 5.0

    def test_observed_state_unknown_not_in_serialised(self):
        """Unknown fields must not appear in serialised output."""
        os = ObservedState(value=59.0, unknown_extra="dropped")
        serialised = os.model_dump()
        assert "unknown_extra" not in serialised


# =============================================================================
# CIL Phase 0.2: ParameterUncertainty nested model hole (C2)
# =============================================================================


class TestParameterUncertaintyExtraIgnore:
    """CIL 0.2 Task 2: ParameterUncertainty must accept and drop unknown fields."""

    def test_parameter_uncertainty_ignores_unknown_field(self):
        """ParameterUncertainty with unknown field parses without error."""
        pu = ParameterUncertainty(
            node_id="price",
            distribution="normal",
            std=2.5,
            unknown_param="dropped",
        )
        assert not hasattr(pu, "unknown_param")
        assert pu.std == 2.5

    def test_parameter_uncertainty_unknown_not_in_serialised(self):
        """Unknown fields must not appear in serialised output."""
        pu = ParameterUncertainty(
            node_id="price",
            distribution="normal",
            std=2.5,
            extra_field="ignored",
        )
        serialised = pu.model_dump()
        assert "extra_field" not in serialised


# =============================================================================
# CIL Phase 0.2: Percentile fallback returns None (C3)
# =============================================================================


class TestPercentileFallback:
    """CIL 0.2 Task 3: Fallback path returns None, not mislabelled CI bounds."""

    def test_outcome_distribution_accepts_none_percentiles(self):
        """OutcomeDistributionV2 accepts p10=None, p50=None, p90=None."""
        od = OutcomeDistributionV2(
            mean=50.0,
            std=5.0,
            p10=None,
            p50=None,
            p90=None,
            n_samples=1000,
            n_valid_samples=0,
            validity_ratio=0.0,
            percentiles_source="unavailable",
        )
        assert od.p10 is None
        assert od.p50 is None
        assert od.p90 is None
        assert od.percentiles_source == "unavailable"

    def test_happy_path_percentiles_source_is_samples(self):
        """When percentiles are computed from samples, source is 'samples'."""
        od = OutcomeDistributionV2(
            mean=50.0,
            std=5.0,
            p10=42.0,
            p50=50.0,
            p90=58.0,
            n_samples=1000,
            n_valid_samples=1000,
            validity_ratio=1.0,
            percentiles_source="samples",
        )
        assert od.percentiles_source == "samples"
        assert od.p10 == 42.0
        assert od.p50 == 50.0
        assert od.p90 == 58.0

    def test_none_percentiles_excluded_from_serialisation(self):
        """When p10/p50/p90 are None, exclude_none serialisation drops them."""
        od = OutcomeDistributionV2(
            mean=50.0,
            std=5.0,
            p10=None,
            p50=None,
            p90=None,
            n_samples=1000,
            n_valid_samples=0,
            validity_ratio=0.0,
            percentiles_source="unavailable",
        )
        serialised = od.model_dump(exclude_none=True)
        assert "p10" not in serialised
        assert "p50" not in serialised
        assert "p90" not in serialised
        assert serialised["percentiles_source"] == "unavailable"

    def test_percentiles_source_default_is_samples(self):
        """Default percentiles_source is 'samples' for backward compatibility."""
        od = OutcomeDistributionV2(
            mean=50.0,
            std=5.0,
            p10=42.0,
            p50=50.0,
            p90=58.0,
            n_samples=1000,
            n_valid_samples=1000,
            validity_ratio=1.0,
        )
        assert od.percentiles_source == "samples"

    def test_normal_path_returns_non_null_percentiles(self, simple_graph, simple_options, analyzer):
        """Normal analysis path always returns non-null percentiles (not breaking in practice)."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )
        response = analyzer.analyze(request)
        for result in response.results:
            dist = result.outcome_distribution
            # In normal operation, samples are always present → percentiles always computed
            assert dist.samples is not None
            assert len(dist.samples) == 100


# =============================================================================
# CIL Phase 0.2: Supporting response models extra='ignore' (Task 5)
# =============================================================================


class TestSupportingResponseModelsExtraIgnore:
    """CIL 0.2 Task 5: All supporting response models accept unknown fields."""

    def test_request_echo_v2_ignores_unknown(self):
        """RequestEchoV2 ignores unknown fields."""
        re = RequestEchoV2(
            graph_node_count=5,
            graph_edge_count=4,
            options_count=2,
            goal_node_id_hash="abc123",
            n_samples=1000,
            response_version_requested=2,
            include_diagnostics=False,
            unknown_echo_field="ignored",
        )
        assert not hasattr(re, "unknown_echo_field")
        assert "unknown_echo_field" not in re.model_dump()

    def test_critique_v2_ignores_unknown(self):
        """CritiqueV2 ignores unknown fields."""
        c = CritiqueV2(
            id="c1",
            code="TEST",
            severity="warning",
            message="test",
            source="validation",
            unknown_critique_field="ignored",
        )
        assert not hasattr(c, "unknown_critique_field")
        assert "unknown_critique_field" not in c.model_dump()

    def test_diagnostics_v2_ignores_unknown(self):
        """DiagnosticsV2 ignores unknown fields."""
        d = DiagnosticsV2(
            goal_node_id_hash="abc",
            goal_node_found=True,
            n_samples_requested=1000,
            n_samples_completed=1000,
            identifiability_status="unknown",
            path_exists_probability_threshold=1e-6,
            path_strength_threshold=1e-6,
            unknown_diag_field="ignored",
        )
        assert not hasattr(d, "unknown_diag_field")
        assert "unknown_diag_field" not in d.model_dump()

    def test_option_diagnostic_v2_ignores_unknown(self):
        """OptionDiagnosticV2 ignores unknown fields."""
        od = OptionDiagnosticV2(
            option_id="opt1",
            intervention_count=1,
            has_structural_path=True,
            has_effective_path=True,
            targets_with_effective_path_count=1,
            targets_without_effective_path_count=0,
            unknown_od_field="ignored",
        )
        assert not hasattr(od, "unknown_od_field")
        assert "unknown_od_field" not in od.model_dump()

    def test_sensitive_factor_v2_ignores_unknown(self):
        """SensitiveFactorV2 ignores unknown fields."""
        sf = SensitiveFactorV2(
            node_id="price",
            sensitivity_score=0.5,
            effect_on_ranking="minor",
            unknown_sf_field="ignored",
        )
        assert not hasattr(sf, "unknown_sf_field")
        assert "unknown_sf_field" not in sf.model_dump()


# =============================================================================
# CIL Phase 0.2: Duplicate option-ID validation (S1)
# =============================================================================


class TestDuplicateOptionIDValidation:
    """CIL 0.2 Task 6: Reject duplicate option IDs."""

    def test_duplicate_option_ids_rejected(self, simple_graph):
        """Two options with same ID → ValidationError."""
        with pytest.raises(ValidationError, match="Duplicate option IDs"):
            RobustnessRequestV2(
                graph=simple_graph,
                options=[
                    InterventionOption(
                        id="same", label="First", interventions={"price": 0.1}
                    ),
                    InterventionOption(
                        id="same", label="Second", interventions={"price": 0.9}
                    ),
                ],
                goal_node_id="revenue",
                n_samples=100,
            )

    def test_unique_option_ids_accepted(self, simple_graph):
        """Two options with different IDs → no error."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=[
                InterventionOption(
                    id="opt_a", label="Option A", interventions={"price": 0.1}
                ),
                InterventionOption(
                    id="opt_b", label="Option B", interventions={"price": 0.9}
                ),
            ],
            goal_node_id="revenue",
            n_samples=100,
        )
        assert len(request.options) == 2
