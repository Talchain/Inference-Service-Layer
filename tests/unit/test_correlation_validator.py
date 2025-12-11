"""
Unit tests for CorrelationValidator service.

Tests cover:
- Correlation group validation
- Matrix positive semi-definite checking
- Implied matrix construction
- Factor reference validation
- High correlation warnings
- Edge cases and error conditions
"""

import pytest
import numpy as np

from src.models.requests import (
    CorrelationGroup,
    CorrelationMatrix,
    CorrelationValidationRequest,
)
from src.models.shared import GraphNodeV1, GraphV1, NodeKind
from src.services.correlation_validator import CorrelationValidator


@pytest.fixture
def validator():
    """Create a CorrelationValidator instance."""
    return CorrelationValidator()


@pytest.fixture
def simple_groups():
    """Create simple correlation groups."""
    return [
        CorrelationGroup(
            group_id="market_conditions",
            factors=["demand", "competition"],
            correlation=0.7,
            label="Market factors tend to move together",
        )
    ]


@pytest.fixture
def multiple_groups():
    """Create multiple correlation groups."""
    return [
        CorrelationGroup(
            group_id="market_conditions",
            factors=["demand", "competition"],
            correlation=0.7,
        ),
        CorrelationGroup(
            group_id="cost_factors",
            factors=["labor_cost", "material_cost"],
            correlation=0.6,
        ),
    ]


@pytest.fixture
def sample_graph():
    """Create a sample graph with factor nodes."""
    return GraphV1(
        nodes=[
            GraphNodeV1(id="demand", label="Demand", kind=NodeKind.FACTOR),
            GraphNodeV1(id="competition", label="Competition", kind=NodeKind.FACTOR),
            GraphNodeV1(id="labor_cost", label="Labor Cost", kind=NodeKind.FACTOR),
            GraphNodeV1(id="material_cost", label="Material Cost", kind=NodeKind.FACTOR),
            GraphNodeV1(id="revenue", label="Revenue", kind=NodeKind.GOAL),
        ],
        edges=[],
    )


class TestCorrelationGroupValidation:
    """Tests for individual correlation group validation."""

    def test_valid_correlation_group(self, validator, simple_groups):
        """Test validation of valid correlation group."""
        request = CorrelationValidationRequest(correlation_groups=simple_groups)

        result = validator.validate(request)

        assert result.valid is True
        assert len(result.validated_groups) == 1
        assert result.validated_groups[0].is_valid is True
        assert result.validated_groups[0].group_id == "market_conditions"

    def test_perfect_positive_correlation_warning(self, validator):
        """Test warning for perfect positive correlation."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b"],
                correlation=1.0,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        # Still valid but with issues flagged
        assert result.validated_groups[0].group_id == "test"
        assert len(result.validated_groups[0].issues) > 0
        assert "1.0" in result.validated_groups[0].issues[0]

    def test_perfect_negative_correlation_error(self, validator):
        """Test error for perfect negative correlation with >2 factors."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b", "c"],
                correlation=-1.0,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        assert result.validated_groups[0].is_valid is False
        assert any("cannot" in issue for issue in result.validated_groups[0].issues)

    def test_high_correlation_many_factors_warning(self, validator):
        """Test warning for high correlation among many factors."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b", "c", "d", "e", "f"],
                correlation=0.85,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        assert len(result.validated_groups[0].issues) > 0
        assert "numerical" in result.validated_groups[0].issues[0].lower()


class TestMatrixPositiveSemiDefinite:
    """Tests for positive semi-definite checking."""

    def test_valid_psd_matrix(self, validator):
        """Test validation of valid PSD matrix."""
        # Identity matrix is always PSD
        matrix = CorrelationMatrix(
            factors=["a", "b", "c"],
            matrix=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, check_positive_definite=True
        )

        result = validator.validate(request)

        assert result.valid is True
        assert result.matrix_analysis is not None
        assert result.matrix_analysis.is_positive_semi_definite is True

    def test_valid_correlation_matrix_psd(self, validator):
        """Test validation of valid correlation matrix."""
        # A valid correlation matrix with moderate correlations
        matrix = CorrelationMatrix(
            factors=["a", "b", "c"],
            matrix=[
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, check_positive_definite=True
        )

        result = validator.validate(request)

        assert result.valid is True
        assert result.matrix_analysis.is_positive_semi_definite is True

    def test_non_psd_matrix_invalid(self, validator):
        """Test that non-PSD matrix is flagged as invalid."""
        # This matrix is NOT positive semi-definite
        # (eigenvalues will include negative values)
        matrix = CorrelationMatrix(
            factors=["a", "b", "c"],
            matrix=[
                [1.0, 0.9, 0.9],
                [0.9, 1.0, -0.9],
                [0.9, -0.9, 1.0],
            ],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, check_positive_definite=True
        )

        result = validator.validate(request)

        assert result.valid is False
        assert result.matrix_analysis is not None
        assert result.matrix_analysis.is_positive_semi_definite is False
        assert result.matrix_analysis.min_eigenvalue is not None
        assert result.matrix_analysis.min_eigenvalue < 0
        assert result.matrix_analysis.suggested_regularization is not None

    def test_skip_psd_check(self, validator):
        """Test skipping PSD check when requested."""
        matrix = CorrelationMatrix(
            factors=["a", "b"],
            matrix=[
                [1.0, 0.5],
                [0.5, 1.0],
            ],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, check_positive_definite=False
        )

        result = validator.validate(request)

        assert result.valid is True
        # Matrix analysis should still exist but eigenvalues not computed
        assert result.matrix_analysis.min_eigenvalue is None

    def test_condition_number_computed(self, validator):
        """Test condition number is computed for PSD matrices."""
        matrix = CorrelationMatrix(
            factors=["a", "b"],
            matrix=[
                [1.0, 0.5],
                [0.5, 1.0],
            ],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, check_positive_definite=True
        )

        result = validator.validate(request)

        assert result.matrix_analysis.condition_number is not None


class TestImpliedMatrixConstruction:
    """Tests for implied matrix construction from groups."""

    def test_implied_matrix_from_single_group(self, validator, simple_groups):
        """Test implied matrix construction from single group."""
        request = CorrelationValidationRequest(correlation_groups=simple_groups)

        result = validator.validate(request)

        assert result.implied_matrix is not None
        assert len(result.implied_matrix.factors) == 2
        assert "demand" in result.implied_matrix.factors
        assert "competition" in result.implied_matrix.factors
        # Matrix should be 2x2 with 0.7 off-diagonal
        matrix = result.implied_matrix.matrix
        assert matrix[0][0] == 1.0  # Diagonal
        assert matrix[1][1] == 1.0  # Diagonal
        # Off-diagonal should be 0.7
        off_diag = matrix[0][1] if matrix[0][1] != 1.0 else matrix[1][0]
        assert off_diag == 0.7

    def test_implied_matrix_from_multiple_groups(self, validator, multiple_groups):
        """Test implied matrix construction from multiple groups."""
        request = CorrelationValidationRequest(correlation_groups=multiple_groups)

        result = validator.validate(request)

        assert result.implied_matrix is not None
        # Should have 4 unique factors
        assert len(result.implied_matrix.factors) == 4
        # Matrix should be 4x4
        assert len(result.implied_matrix.matrix) == 4
        assert all(len(row) == 4 for row in result.implied_matrix.matrix)

    def test_conflicting_correlations_warning(self, validator):
        """Test warning when same pair has different correlations."""
        groups = [
            CorrelationGroup(
                group_id="group1",
                factors=["a", "b"],
                correlation=0.7,
            ),
            CorrelationGroup(
                group_id="group2",
                factors=["a", "b", "c"],
                correlation=0.5,
            ),
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        conflict_warnings = [
            w for w in result.warnings if w.code == "CONFLICTING_CORRELATION"
        ]
        assert len(conflict_warnings) == 1


class TestFactorReferenceValidation:
    """Tests for factor reference validation against graph."""

    def test_valid_factor_references(self, validator, simple_groups, sample_graph):
        """Test validation with valid factor references."""
        request = CorrelationValidationRequest(
            correlation_groups=simple_groups, graph=sample_graph
        )

        result = validator.validate(request)

        missing_warnings = [
            w for w in result.warnings if w.code == "MISSING_FACTOR_NODES"
        ]
        assert len(missing_warnings) == 0

    def test_missing_factor_node_warning(self, validator, sample_graph):
        """Test warning when factor not found in graph."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["demand", "missing_factor"],
                correlation=0.5,
            )
        ]
        request = CorrelationValidationRequest(
            correlation_groups=groups, graph=sample_graph
        )

        result = validator.validate(request)

        assert result.valid is True  # Still valid, just warning
        missing_warnings = [
            w for w in result.warnings if w.code == "MISSING_FACTOR_NODES"
        ]
        assert len(missing_warnings) == 1
        assert "missing_factor" in str(missing_warnings[0].affected_factors)

    def test_non_factor_node_warning(self, validator, sample_graph):
        """Test warning when referencing non-FACTOR node type."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["demand", "revenue"],  # revenue is a GOAL node
                correlation=0.5,
            )
        ]
        request = CorrelationValidationRequest(
            correlation_groups=groups, graph=sample_graph
        )

        result = validator.validate(request)

        non_factor_warnings = [
            w for w in result.warnings if w.code == "NON_FACTOR_NODES"
        ]
        assert len(non_factor_warnings) == 1
        assert "revenue" in str(non_factor_warnings[0].affected_factors)


class TestHighCorrelationWarnings:
    """Tests for high correlation detection."""

    def test_high_correlation_warning(self, validator):
        """Test warning for high correlation (>=0.9)."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b"],
                correlation=0.95,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        high_warnings = [w for w in result.warnings if w.code == "HIGH_CORRELATION"]
        assert len(high_warnings) == 1
        assert "redundancy" in high_warnings[0].message.lower()

    def test_high_negative_correlation_warning(self, validator):
        """Test warning for high negative correlation."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b"],
                correlation=-0.95,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        high_warnings = [w for w in result.warnings if w.code == "HIGH_CORRELATION"]
        assert len(high_warnings) == 1

    def test_no_warning_for_moderate_correlation(self, validator):
        """Test no warning for moderate correlation."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b"],
                correlation=0.7,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        high_warnings = [w for w in result.warnings if w.code == "HIGH_CORRELATION"]
        assert len(high_warnings) == 0


class TestDirectMatrixInput:
    """Tests for direct matrix input mode."""

    def test_direct_matrix_validation(self, validator):
        """Test validation with direct matrix input."""
        matrix = CorrelationMatrix(
            factors=["a", "b", "c"],
            matrix=[
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ],
        )
        request = CorrelationValidationRequest(correlation_matrix=matrix)

        result = validator.validate(request)

        assert result.valid is True
        assert result.implied_matrix is not None
        assert result.implied_matrix.factors == ["a", "b", "c"]

    def test_direct_matrix_with_graph_validation(self, validator, sample_graph):
        """Test direct matrix with graph validation."""
        matrix = CorrelationMatrix(
            factors=["demand", "competition"],
            matrix=[
                [1.0, 0.7],
                [0.7, 1.0],
            ],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, graph=sample_graph
        )

        result = validator.validate(request)

        assert result.valid is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_pair_group(self, validator):
        """Test validation with minimum factors (2)."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b"],
                correlation=0.5,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        assert result.valid is True
        assert len(result.implied_matrix.factors) == 2

    def test_zero_correlation(self, validator):
        """Test validation with zero correlation."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b"],
                correlation=0.0,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        assert result.valid is True
        # Zero correlation means independent - should be valid

    def test_empty_groups_list(self, validator):
        """Test validation with empty groups list."""
        request = CorrelationValidationRequest(correlation_groups=[])

        result = validator.validate(request)

        # Empty groups means no matrix to analyze
        assert result.implied_matrix is None
        assert result.matrix_analysis is None

    def test_large_matrix_psd_check(self, validator):
        """Test PSD check on larger matrix."""
        # Create 5x5 identity matrix (always PSD)
        n = 5
        matrix = CorrelationMatrix(
            factors=[f"f{i}" for i in range(n)],
            matrix=[[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, check_positive_definite=True
        )

        result = validator.validate(request)

        assert result.valid is True
        assert result.matrix_analysis.is_positive_semi_definite is True

    def test_response_schema_version(self, validator, simple_groups):
        """Test response includes schema version."""
        request = CorrelationValidationRequest(correlation_groups=simple_groups)

        result = validator.validate(request)

        assert result.schema_version == "correlation.v1"

    def test_correlation_type_preserved(self, validator):
        """Test that correlation type is preserved in validated groups."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b"],
                correlation=0.5,
                correlation_type="spearman",
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        # The validated groups should preserve the input
        assert result.validated_groups[0].correlation == 0.5


class TestMatrixAnalysisDetails:
    """Tests for detailed matrix analysis."""

    def test_regularization_suggestion(self, validator):
        """Test regularization suggestion for non-PSD matrix."""
        # Create a matrix that's definitely not PSD
        matrix = CorrelationMatrix(
            factors=["a", "b", "c"],
            matrix=[
                [1.0, 0.99, 0.99],
                [0.99, 1.0, -0.99],
                [0.99, -0.99, 1.0],
            ],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, check_positive_definite=True
        )

        result = validator.validate(request)

        assert result.valid is False
        assert result.matrix_analysis.suggested_regularization is not None
        # Regularization should be positive
        assert result.matrix_analysis.suggested_regularization > 0

    def test_eigenvalue_precision(self, validator):
        """Test eigenvalue is rounded appropriately."""
        matrix = CorrelationMatrix(
            factors=["a", "b"],
            matrix=[
                [1.0, 0.5],
                [0.5, 1.0],
            ],
        )
        request = CorrelationValidationRequest(
            correlation_matrix=matrix, check_positive_definite=True
        )

        result = validator.validate(request)

        # min_eigenvalue should be a reasonable precision
        assert result.matrix_analysis.min_eigenvalue is not None
        # Should be 0.5 for this matrix
        assert abs(result.matrix_analysis.min_eigenvalue - 0.5) < 0.001


class TestGroupOverlap:
    """Tests for overlapping correlation groups."""

    def test_overlapping_groups_same_correlation(self, validator):
        """Test overlapping groups with same correlation value."""
        groups = [
            CorrelationGroup(
                group_id="group1",
                factors=["a", "b"],
                correlation=0.5,
            ),
            CorrelationGroup(
                group_id="group2",
                factors=["b", "c"],
                correlation=0.5,
            ),
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        assert result.valid is True
        # Should have 3 unique factors
        assert len(result.implied_matrix.factors) == 3

    def test_three_factor_group(self, validator):
        """Test group with three factors (pairwise correlations)."""
        groups = [
            CorrelationGroup(
                group_id="test",
                factors=["a", "b", "c"],
                correlation=0.5,
            )
        ]
        request = CorrelationValidationRequest(correlation_groups=groups)

        result = validator.validate(request)

        assert result.valid is True
        # All three pairs should have correlation 0.5
        matrix = result.implied_matrix.matrix
        factors = result.implied_matrix.factors
        idx = {f: i for i, f in enumerate(factors)}
        assert matrix[idx["a"]][idx["b"]] == 0.5
        assert matrix[idx["b"]][idx["c"]] == 0.5
        assert matrix[idx["a"]][idx["c"]] == 0.5
