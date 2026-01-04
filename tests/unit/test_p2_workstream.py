"""
Unit tests for P2 Workstream implementation.

Tests cover:
- P2-ISL-1: V2 Response Format
- P2-ISL-2: X-Request-Id Tracing
- P2-ISL-3: 422 Error Schema
- P2-ISL-4: Critique Codes
- P2-ISL-5: Baseline Near-Zero Protection
"""

import math

import numpy as np
import pytest

from src.constants import BASELINE_EPSILON
from src.models.critique import (
    BASELINE_NEAR_ZERO,
    CRITIQUES,
    EDGE_STD_INVALID,
    GRAPH_EMPTY,
    INSUFFICIENT_OPTIONS,
    INVALID_NODE_ID,
    MONTE_CARLO_FAILED,
)
from src.models.response_v2 import ISLV2Error422, ISLResponseV2
from src.utils.numerical_stability import (
    check_baseline_near_zero,
    compute_analysis_status_with_numerical_checks,
    safe_percent_change,
    safe_sensitivity,
    validate_mc_samples,
)
from src.utils.response_builder import ResponseBuilder, build_request_echo


class TestP2ISL1ResponseFormat:
    """P2-ISL-1: V2 Response Format tests."""

    def test_response_has_version_alias(self):
        """Response can use 'version' alias for response_schema_version."""
        response = ISLResponseV2(
            response_schema_version="2.0",
            endpoint_version="analyze/v2",
            engine_version="1.0.0",
            analysis_status="computed",
            robustness_status="computed",
            factor_sensitivity_status="computed",
            critiques=[],
            request_echo=build_request_echo(
                graph_node_count=3,
                graph_edge_count=2,
                options_count=2,
                goal_node_id="goal",
                n_samples=1000,
                response_version=2,
                include_diagnostics=False,
            ),
            request_id="test-123",
            processing_time_ms=100,
        )
        # Check that the field is properly serialized with alias
        data = response.model_dump(by_alias=True)
        assert "version" in data
        assert data["version"] == "2.0"

    def test_response_has_timestamp(self):
        """Response includes timestamp in ISO 8601 format."""
        response = ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version="1.0.0",
            analysis_status="computed",
            robustness_status="computed",
            factor_sensitivity_status="computed",
            critiques=[],
            request_echo=build_request_echo(
                graph_node_count=3,
                graph_edge_count=2,
                options_count=2,
                goal_node_id="goal",
                n_samples=1000,
                response_version=2,
                include_diagnostics=False,
            ),
            request_id="test-123",
            processing_time_ms=100,
        )
        assert hasattr(response, "timestamp")
        assert response.timestamp.endswith("Z")

    def test_response_has_seed_used(self):
        """Response can include seed_used field."""
        response = ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version="1.0.0",
            analysis_status="computed",
            robustness_status="computed",
            factor_sensitivity_status="computed",
            critiques=[],
            request_echo=build_request_echo(
                graph_node_count=3,
                graph_edge_count=2,
                options_count=2,
                goal_node_id="goal",
                n_samples=1000,
                response_version=2,
                include_diagnostics=False,
            ),
            request_id="test-123",
            processing_time_ms=100,
            seed_used="42",
        )
        assert response.seed_used == "42"

    def test_response_no_response_hash(self):
        """Response should NOT include response_hash (PLoT owns it)."""
        response = ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version="1.0.0",
            analysis_status="computed",
            robustness_status="computed",
            factor_sensitivity_status="computed",
            critiques=[],
            request_echo=build_request_echo(
                graph_node_count=3,
                graph_edge_count=2,
                options_count=2,
                goal_node_id="goal",
                n_samples=1000,
                response_version=2,
                include_diagnostics=False,
            ),
            request_id="test-123",
            processing_time_ms=100,
        )
        data = response.model_dump()
        assert "response_hash" not in data


class TestP2ISL2TracingHeader:
    """P2-ISL-2: X-Request-Id Tracing tests."""

    def test_request_id_included_in_response(self):
        """Request ID is included in response body."""
        response = ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version="1.0.0",
            analysis_status="computed",
            robustness_status="computed",
            factor_sensitivity_status="computed",
            critiques=[],
            request_echo=build_request_echo(
                graph_node_count=3,
                graph_edge_count=2,
                options_count=2,
                goal_node_id="goal",
                n_samples=1000,
                response_version=2,
                include_diagnostics=False,
            ),
            request_id="isl-a1b2c3d4e5f6",
            processing_time_ms=100,
        )
        assert response.request_id == "isl-a1b2c3d4e5f6"

    def test_processing_time_in_response(self):
        """Processing time is included in response."""
        response = ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version="1.0.0",
            analysis_status="computed",
            robustness_status="computed",
            factor_sensitivity_status="computed",
            critiques=[],
            request_echo=build_request_echo(
                graph_node_count=3,
                graph_edge_count=2,
                options_count=2,
                goal_node_id="goal",
                n_samples=1000,
                response_version=2,
                include_diagnostics=False,
            ),
            request_id="test-123",
            processing_time_ms=150,
        )
        assert response.processing_time_ms == 150


class TestP2ISL3Error422Schema:
    """P2-ISL-3: 422 Error Schema tests."""

    def test_422_error_has_blocked_status(self):
        """422 error response has analysis_status='blocked'."""
        error = ISLV2Error422(
            status_reason="Validation failed",
            critiques=[],
            request_id="test-123",
        )
        assert error.analysis_status == "blocked"

    def test_422_error_is_unwrapped(self):
        """422 error response should not be wrapped in envelope."""
        error = ISLV2Error422(
            status_reason="Invalid node ID",
            critiques=[INVALID_NODE_ID.build(id="Bad Node!")],
            request_id="test-123",
        )
        data = error.model_dump()
        # Should NOT have envelope fields
        assert "error" not in data
        assert "success" not in data
        # Should have direct fields
        assert "analysis_status" in data
        assert "status_reason" in data
        assert "critiques" in data

    def test_422_error_includes_critiques(self):
        """422 error includes structured critiques."""
        error = ISLV2Error422(
            status_reason="Multiple validation errors",
            critiques=[
                INVALID_NODE_ID.build(id="bad_node!"),
                GRAPH_EMPTY.build(),
            ],
            request_id="test-123",
        )
        assert len(error.critiques) == 2
        assert error.critiques[0].code == "INVALID_NODE_ID"
        assert error.critiques[1].code == "GRAPH_EMPTY"

    def test_response_builder_creates_422(self):
        """ResponseBuilder.build_422_response creates correct format."""
        builder = ResponseBuilder(
            request_id="test-123",
            request_echo=build_request_echo(
                graph_node_count=0,
                graph_edge_count=0,
                options_count=0,
                goal_node_id="goal",
                n_samples=1000,
                response_version=2,
                include_diagnostics=False,
            ),
        )
        builder.add_critique(GRAPH_EMPTY.build())

        error = builder.build_422_response()
        assert error.analysis_status == "blocked"
        assert len(error.critiques) == 1
        assert error.critiques[0].code == "GRAPH_EMPTY"


class TestP2ISL4CritiqueCodes:
    """P2-ISL-4: Critique Codes tests."""

    def test_all_critique_codes_registered(self):
        """All P2 critique codes are in the registry."""
        required_codes = [
            # Graph structure
            "GRAPH_EMPTY",
            "GRAPH_DISCONNECTED",
            "GRAPH_CYCLE_DETECTED",
            # Nodes
            "INVALID_NODE_ID",
            "DUPLICATE_NODE_ID",
            # Edges
            "EDGE_STRENGTH_OUT_OF_RANGE",
            "EDGE_STD_INVALID",
            "EDGE_ENDPOINT_MISSING",
            "NEGLIGIBLE_EDGE_STRENGTH",
            # Options
            "INSUFFICIENT_OPTIONS",
            "OPTION_NO_INTERVENTIONS",
            "DUPLICATE_OPTION_ID",
            "INTERVENTION_VALUE_INVALID",
            # Inference
            "MONTE_CARLO_FAILED",
            "BASELINE_NEAR_ZERO",
            "INFERENCE_TIMEOUT",
            "SEED_INVALID",
        ]
        for code in required_codes:
            assert code in CRITIQUES, f"Missing critique code: {code}"

    def test_critique_has_suggestion(self):
        """Critiques include actionable suggestions."""
        critique = EDGE_STD_INVALID.build(
            from_node="a", to_node="b", value=-0.1
        )
        assert critique.suggestion is not None
        assert len(critique.suggestion) > 0

    def test_critique_severity_levels(self):
        """Critiques have correct severity levels."""
        assert GRAPH_EMPTY.severity == "blocker"
        assert EDGE_STD_INVALID.severity == "blocker"
        assert BASELINE_NEAR_ZERO.severity == "warning"
        assert INSUFFICIENT_OPTIONS.severity == "blocker"

    def test_critique_sources(self):
        """Critiques have correct sources."""
        assert GRAPH_EMPTY.source == "validation"
        assert MONTE_CARLO_FAILED.source == "analysis"
        assert BASELINE_NEAR_ZERO.source == "analysis"


class TestP2ISL5NumericalStability:
    """P2-ISL-5: Baseline Near-Zero Protection tests."""

    def test_safe_sensitivity_normal_baseline(self):
        """Normal baseline returns unguarded sensitivity."""
        sens, guarded = safe_sensitivity(10.0, 100.0)
        assert sens == pytest.approx(0.1, rel=1e-6)
        assert guarded is False

    def test_safe_sensitivity_near_zero_baseline(self):
        """Near-zero baseline is epsilon-guarded."""
        sens, guarded = safe_sensitivity(10.0, 1e-10)
        assert guarded is True
        assert math.isfinite(sens)

    def test_safe_sensitivity_exact_zero_baseline(self):
        """Exact zero baseline is epsilon-guarded."""
        sens, guarded = safe_sensitivity(10.0, 0.0)
        assert guarded is True
        assert math.isfinite(sens)

    def test_safe_percent_change_normal(self):
        """Normal percent change calculation."""
        pct, guarded = safe_percent_change(110.0, 100.0)
        assert pct == pytest.approx(10.0, rel=1e-6)
        assert guarded is False

    def test_safe_percent_change_zero_baseline(self):
        """Zero baseline is epsilon-guarded."""
        pct, guarded = safe_percent_change(10.0, 0.0)
        assert guarded is True
        assert math.isfinite(pct)

    def test_check_baseline_near_zero_emits_warning(self):
        """Near-zero baseline emits warning critique."""
        critiques = []
        result = check_baseline_near_zero(1e-10, critiques)
        assert result is True
        assert len(critiques) == 1
        assert critiques[0].code == "BASELINE_NEAR_ZERO"

    def test_check_baseline_normal_no_warning(self):
        """Normal baseline does not emit warning."""
        critiques = []
        result = check_baseline_near_zero(100.0, critiques)
        assert result is False
        assert len(critiques) == 0

    def test_validate_mc_samples_clean(self):
        """Clean samples pass through unchanged."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cleaned, critiques = validate_mc_samples(samples)
        np.testing.assert_array_equal(cleaned, samples)
        assert len(critiques) == 0

    def test_validate_mc_samples_with_nan(self):
        """NaN samples are replaced and warning emitted."""
        samples = np.array([1.0, np.nan, 3.0])
        cleaned, critiques = validate_mc_samples(samples)
        assert np.all(np.isfinite(cleaned))
        assert len(critiques) == 1
        assert critiques[0].code == "NUMERICAL_INSTABILITY"

    def test_validate_mc_samples_with_inf(self):
        """Inf samples are replaced and warning emitted."""
        samples = np.array([1.0, np.inf, 3.0])
        cleaned, critiques = validate_mc_samples(samples)
        assert np.all(np.isfinite(cleaned))
        assert len(critiques) == 1
        assert critiques[0].code == "NUMERICAL_INSTABILITY"

    def test_validate_mc_samples_all_invalid(self):
        """All invalid samples emit MONTE_CARLO_FAILED."""
        samples = np.array([np.nan, np.inf, -np.inf])
        _, critiques = validate_mc_samples(samples)
        assert len(critiques) == 1
        assert critiques[0].code == "MONTE_CARLO_FAILED"

    def test_compute_status_with_numerical_warnings(self):
        """Numerical warnings downgrade computed to partial."""
        critiques = [BASELINE_NEAR_ZERO.build(value="1e-10")]
        status = compute_analysis_status_with_numerical_checks("computed", critiques)
        assert status == "partial"

    def test_compute_status_without_numerical_warnings(self):
        """No numerical warnings keeps computed status."""
        critiques = []
        status = compute_analysis_status_with_numerical_checks("computed", critiques)
        assert status == "computed"

    def test_baseline_epsilon_value(self):
        """BASELINE_EPSILON is set correctly."""
        assert BASELINE_EPSILON == 1e-8
