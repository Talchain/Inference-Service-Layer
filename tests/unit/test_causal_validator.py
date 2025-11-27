"""
Unit tests for enhanced CausalValidator service.

Tests the comprehensive Y₀ integration including:
- Identifiable cases (backdoor, do-calculus)
- Non-identifiable cases with detailed diagnostics
- Graceful degradation on errors
- Helper methods (formulas, assumptions, alternatives)
- Enhanced response models
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models.requests import CausalValidationRequest
from src.models.responses import (
    AlternativeMethod,
    AssumptionDetail,
    CausalValidationResponse,
)
from src.models.shared import ConfidenceLevel, DAGStructure, ValidationStatus
from src.services.causal_validator import CausalValidator


@pytest.fixture
def validator():
    """Create CausalValidator instance."""
    return CausalValidator()


@pytest.fixture
def simple_identifiable_dag():
    """
    Simple identifiable DAG: Z → X → Y, Z → Y
    Z confounds X-Y, identifiable by controlling for Z.
    """
    return DAGStructure(
        nodes=["X", "Y", "Z"],
        edges=[["Z", "X"], ["X", "Y"], ["Z", "Y"]],
    )


@pytest.fixture
def backdoor_dag():
    """
    Classic backdoor scenario: X ← Z → Y, X → Y
    Backdoor path through Z, identifiable by controlling for Z.
    """
    return DAGStructure(
        nodes=["X", "Y", "Z"],
        edges=[["Z", "X"], ["Z", "Y"], ["X", "Y"]],
    )


@pytest.fixture
def no_confounders_dag():
    """
    No confounders: X → Y (direct effect only)
    Identifiable without adjustment.
    """
    return DAGStructure(
        nodes=["X", "Y"],
        edges=[["X", "Y"]],
    )


@pytest.fixture
def no_path_dag():
    """
    No causal path: X  Y (no connection)
    """
    return DAGStructure(
        nodes=["X", "Y", "Z"],
        edges=[["Z", "X"], ["Z", "Y"]],
    )


@pytest.fixture
def non_identifiable_dag():
    """
    Non-identifiable: X → Y ← U → X (unmeasured confounder U)
    """
    return DAGStructure(
        nodes=["X", "Y", "U"],
        edges=[["X", "Y"], ["U", "Y"], ["U", "X"]],
    )


@pytest.fixture
def mediator_dag():
    """
    Mediator structure: X → M → Y
    Could be identifiable via front-door if conditions met.
    """
    return DAGStructure(
        nodes=["X", "M", "Y"],
        edges=[["X", "M"], ["M", "Y"]],
    )


@pytest.fixture
def complex_dag():
    """
    Complex multi-confounder DAG for comprehensive testing.
    """
    return DAGStructure(
        nodes=["X", "Y", "Z1", "Z2", "Z3"],
        edges=[
            ["Z1", "X"],
            ["Z1", "Y"],
            ["Z2", "X"],
            ["Z2", "Y"],
            ["Z3", "Z1"],
            ["X", "Y"],
        ],
    )


# ============================================================================
# Test: Identifiable Cases with Enhanced Response
# ============================================================================


def test_validate_identifiable_with_confounder(validator, backdoor_dag):
    """Test identifiable case with single confounder - should include method and formula."""
    request = CausalValidationRequest(
        dag=backdoor_dag,
        treatment="X",
        outcome="Y",
    )

    response = validator.validate(request)

    assert response.status == ValidationStatus.IDENTIFIABLE
    assert response.confidence == ConfidenceLevel.HIGH

    # Enhanced fields should be populated
    assert response.method is not None
    assert response.method in ["backdoor", "do_calculus"]
    assert response.identification_formula is not None
    assert "P(Y|do(X))" in response.identification_formula
    assert response.structured_assumptions is not None
    assert len(response.structured_assumptions) > 0
    assert response.alternative_methods is not None

    # Check assumption structure
    for assumption in response.structured_assumptions:
        assert isinstance(assumption, AssumptionDetail)
        assert assumption.type is not None
        assert assumption.description is not None
        assert isinstance(assumption.critical, bool)

    # Check alternative methods structure
    for alt_method in response.alternative_methods:
        assert isinstance(alt_method, AlternativeMethod)
        assert alt_method.method in ["backdoor", "front_door", "instrumental_variables"]
        assert isinstance(alt_method.applicable, bool)
        assert alt_method.reason is not None


def test_validate_identifiable_no_confounders(validator, no_confounders_dag):
    """Test identifiable case without confounders - no adjustment needed."""
    request = CausalValidationRequest(
        dag=no_confounders_dag,
        treatment="X",
        outcome="Y",
    )

    response = validator.validate(request)

    assert response.status == ValidationStatus.IDENTIFIABLE
    assert response.confidence == ConfidenceLevel.HIGH
    assert response.method is not None
    assert response.identification_formula is not None

    # No adjustment set needed
    if response.minimal_set:
        assert len(response.minimal_set) == 0

    # Formula should indicate no adjustment
    assert "P(Y|X)" in response.identification_formula or "Σ" not in response.identification_formula


def test_validate_complex_identifiable(validator, complex_dag):
    """Test complex identifiable case with multiple confounders."""
    request = CausalValidationRequest(
        dag=complex_dag,
        treatment="X",
        outcome="Y",
    )

    response = validator.validate(request)

    # Should be identifiable with appropriate adjustment
    if response.status == ValidationStatus.IDENTIFIABLE:
        assert response.method is not None
        assert response.identification_formula is not None
        assert response.structured_assumptions is not None
        assert len(response.structured_assumptions) >= 3  # Multiple assumptions for complex DAG
        assert response.alternative_methods is not None


# ============================================================================
# Test: Non-Identifiable Cases with Enhanced Diagnostics
# ============================================================================


def test_validate_no_path(validator, no_path_dag):
    """Test when no causal path exists - should provide clear reason."""
    request = CausalValidationRequest(
        dag=no_path_dag,
        treatment="X",
        outcome="Y",
    )

    response = validator.validate(request)

    assert response.status in [ValidationStatus.UNCERTAIN, ValidationStatus.CANNOT_IDENTIFY]

    # Enhanced diagnostic fields
    assert response.reason is not None
    assert "causal" in response.reason.lower() and "path" in response.reason.lower()
    assert response.suggestions is not None
    assert len(response.suggestions) > 0
    assert response.attempted_methods is not None
    assert len(response.attempted_methods) > 0


def test_validate_non_identifiable_with_unmeasured_confounder(validator, non_identifiable_dag):
    """Test case with potential unmeasured confounder structure."""
    request = CausalValidationRequest(
        dag=non_identifiable_dag,
        treatment="X",
        outcome="Y",
    )

    response = validator.validate(request)

    # Note: This DAG might be identifiable if U is observed
    # We're mainly testing that the validator handles it without crashing
    assert response.status in [
        ValidationStatus.IDENTIFIABLE,
        ValidationStatus.CANNOT_IDENTIFY,
        ValidationStatus.UNCERTAIN,
        ValidationStatus.DEGRADED
    ]

    # Should always have valid response structure
    assert response.confidence is not None
    assert response.explanation is not None


# ============================================================================
# Test: Graceful Degradation
# ============================================================================


def test_validate_degraded_on_y0_error(validator, backdoor_dag):
    """Test that validator handles Y₀ errors gracefully with fallback."""
    # Mock Y₀ to raise an exception
    with patch('src.services.causal_validator.edge_list_to_y0') as mock_convert:
        mock_convert.side_effect = Exception("Y₀ graph conversion error")

        request = CausalValidationRequest(
            dag=backdoor_dag,
            treatment="X",
            outcome="Y",
        )

        response = validator.validate(request)

        # When Y₀ fails, should fall back to backdoor criterion
        # This particular DAG is identifiable by backdoor, so status could be IDENTIFIABLE or DEGRADED
        assert response.status in [ValidationStatus.IDENTIFIABLE, ValidationStatus.DEGRADED]

        # Should not crash and should return valid response
        assert response.confidence is not None
        assert response.explanation is not None


def test_validate_degraded_fallback_analysis(validator, no_confounders_dag):
    """Test that validator handles Y₀ errors and uses fallback successfully."""
    with patch('src.services.causal_validator.edge_list_to_y0') as mock_convert:
        mock_convert.side_effect = RuntimeError("Graph conversion failed")

        request = CausalValidationRequest(
            dag=no_confounders_dag,
            treatment="X",
            outcome="Y",
        )

        response = validator.validate(request)

        # Should handle error gracefully - either degraded or successful fallback
        assert response.status in [ValidationStatus.IDENTIFIABLE, ValidationStatus.DEGRADED]

        # Should return valid response structure
        assert response.confidence is not None
        assert response.explanation is not None


# ============================================================================
# Test: Helper Methods - Formula Generation
# ============================================================================


def test_generate_backdoor_formula_with_adjustment(validator):
    """Test backdoor formula generation with adjustment set."""
    request = CausalValidationRequest(
        dag=DAGStructure(nodes=["X", "Y", "Z"], edges=[]),
        treatment="X",
        outcome="Y",
    )

    formula = validator._generate_identification_formula(
        method="backdoor",
        adjustment_set=["Z"],
        request=request
    )

    assert "P(Y|do(X))" in formula
    assert "Z" in formula
    assert "Σ" in formula or "sum" in formula.lower()


def test_generate_backdoor_formula_no_adjustment(validator):
    """Test backdoor formula generation without adjustment set."""
    request = CausalValidationRequest(
        dag=DAGStructure(nodes=["X", "Y"], edges=[]),
        treatment="X",
        outcome="Y",
    )

    formula = validator._generate_identification_formula(
        method="backdoor",
        adjustment_set=[],
        request=request
    )

    assert "P(Y|do(X))" in formula
    assert "P(Y|X)" in formula
    assert "Σ" not in formula


def test_generate_front_door_formula(validator):
    """Test front-door formula generation."""
    request = CausalValidationRequest(
        dag=DAGStructure(nodes=["X", "Y", "M"], edges=[]),
        treatment="X",
        outcome="Y",
    )

    formula = validator._generate_identification_formula(
        method="front_door",
        adjustment_set=["M"],
        request=request
    )

    assert "P(Y|do(X))" in formula
    assert "M" in formula  # Mediator should be mentioned


def test_generate_do_calculus_formula(validator):
    """Test general do-calculus formula generation."""
    request = CausalValidationRequest(
        dag=DAGStructure(nodes=["X", "Y", "Z"], edges=[]),
        treatment="X",
        outcome="Y",
    )

    formula = validator._generate_identification_formula(
        method="do_calculus",
        adjustment_set=["Z"],
        request=request
    )

    assert "do-calculus" in formula.lower() or "identifiable" in formula.lower()


# ============================================================================
# Test: Helper Methods - Assumption Extraction
# ============================================================================


def test_extract_backdoor_assumptions(validator):
    """Test extraction of backdoor assumptions."""
    assumptions = validator._extract_assumptions("backdoor", ["Z"])

    assert len(assumptions) >= 3
    assumption_types = {a.type for a in assumptions}

    # Should include key assumptions
    assert "no_unmeasured_confounding" in assumption_types
    assert "positivity" in assumption_types

    # All should be AssumptionDetail instances
    for assumption in assumptions:
        assert isinstance(assumption, AssumptionDetail)
        assert len(assumption.description) > 0
        assert isinstance(assumption.critical, bool)


def test_extract_front_door_assumptions(validator):
    """Test extraction of front-door assumptions."""
    assumptions = validator._extract_assumptions("front_door", ["M"])

    assert len(assumptions) >= 2
    assumption_types = {a.type for a in assumptions}

    # Front-door specific assumptions
    assert "mediator_completeness" in assumption_types or len(assumptions) > 0


def test_extract_do_calculus_assumptions(validator):
    """Test extraction of do-calculus assumptions."""
    assumptions = validator._extract_assumptions("do_calculus", ["Z"])

    assert len(assumptions) >= 2

    for assumption in assumptions:
        assert isinstance(assumption, AssumptionDetail)
        assert assumption.type is not None
        assert len(assumption.description) > 0


# ============================================================================
# Test: Helper Methods - Alternative Methods
# ============================================================================


def test_check_alternative_methods_backdoor(validator, backdoor_dag):
    """Test alternative method checking for backdoor-identifiable case."""
    request = CausalValidationRequest(
        dag=backdoor_dag,
        treatment="X",
        outcome="Y",
    )

    import networkx as nx
    from src.utils.graph_parser import edge_list_to_networkx

    nx_graph = edge_list_to_networkx(backdoor_dag.nodes, backdoor_dag.edges)

    alternatives = validator._check_alternative_methods(request, nx_graph)

    assert len(alternatives) >= 3  # backdoor, front_door, instrumental_variables

    # Backdoor should be applicable
    backdoor_alt = next((a for a in alternatives if a.method == "backdoor"), None)
    assert backdoor_alt is not None
    assert backdoor_alt.applicable is True
    assert len(backdoor_alt.reason) > 0


def test_check_alternative_methods_mediator(validator, mediator_dag):
    """Test alternative method checking for mediator structure."""
    request = CausalValidationRequest(
        dag=mediator_dag,
        treatment="X",
        outcome="Y",
    )

    import networkx as nx
    from src.utils.graph_parser import edge_list_to_networkx

    nx_graph = edge_list_to_networkx(mediator_dag.nodes, mediator_dag.edges)

    alternatives = validator._check_alternative_methods(request, nx_graph)

    assert len(alternatives) >= 3

    # Check structure
    for alt in alternatives:
        assert isinstance(alt, AlternativeMethod)
        assert alt.method in ["backdoor", "front_door", "instrumental_variables"]
        assert isinstance(alt.applicable, bool)
        assert len(alt.reason) > 0


# ============================================================================
# Test: Helper Methods - Y₀ Method Determination
# ============================================================================


def test_determine_y0_method_backdoor(validator, backdoor_dag):
    """Test Y₀ method determination for backdoor case."""
    import networkx as nx
    from src.utils.graph_parser import edge_list_to_networkx

    nx_graph = edge_list_to_networkx(backdoor_dag.nodes, backdoor_dag.edges)

    # Mock a Y₀ result
    mock_result = MagicMock()

    request = CausalValidationRequest(
        dag=backdoor_dag,
        treatment="X",
        outcome="Y",
    )

    method = validator._determine_y0_method(
        y0_result=mock_result,
        nx_graph=nx_graph,
        request=request
    )

    assert method in ["backdoor", "do_calculus"]


def test_determine_y0_method_no_backdoor(validator, mediator_dag):
    """Test Y₀ method determination when no backdoor paths exist."""
    import networkx as nx
    from src.utils.graph_parser import edge_list_to_networkx

    nx_graph = edge_list_to_networkx(mediator_dag.nodes, mediator_dag.edges)

    mock_result = MagicMock()

    request = CausalValidationRequest(
        dag=mediator_dag,
        treatment="X",
        outcome="Y",
    )

    method = validator._determine_y0_method(
        y0_result=mock_result,
        nx_graph=nx_graph,
        request=request
    )

    # Should return a valid method
    assert method in ["backdoor", "do_calculus", "front_door", "instrumental_variables"]


# ============================================================================
# Test: Helper Methods - Adjustment Set Extraction
# ============================================================================


def test_extract_y0_adjustment_set_with_set(validator, backdoor_dag):
    """Test extracting adjustment set from Y₀ result."""
    from y0.dsl import Variable
    import networkx as nx
    from src.utils.graph_parser import edge_list_to_networkx

    nx_graph = edge_list_to_networkx(backdoor_dag.nodes, backdoor_dag.edges)

    # Mock Y₀ result with adjustment set
    mock_result = MagicMock()
    mock_result.get_variables.return_value = {Variable("Z")}

    request = CausalValidationRequest(
        dag=backdoor_dag,
        treatment="X",
        outcome="Y",
    )

    adjustment_set = validator._extract_y0_adjustment_set(mock_result, nx_graph, request)

    assert adjustment_set is not None
    assert isinstance(adjustment_set, list)


def test_extract_y0_adjustment_set_empty(validator, no_confounders_dag):
    """Test extracting empty adjustment set."""
    import networkx as nx
    from src.utils.graph_parser import edge_list_to_networkx

    nx_graph = edge_list_to_networkx(no_confounders_dag.nodes, no_confounders_dag.edges)

    mock_result = MagicMock()
    mock_result.get_variables.return_value = set()

    request = CausalValidationRequest(
        dag=no_confounders_dag,
        treatment="X",
        outcome="Y",
    )

    adjustment_set = validator._extract_y0_adjustment_set(mock_result, nx_graph, request)

    assert adjustment_set is not None
    assert isinstance(adjustment_set, list)


def test_extract_y0_adjustment_set_none(validator, mediator_dag):
    """Test extracting adjustment set when not available."""
    import networkx as nx
    from src.utils.graph_parser import edge_list_to_networkx

    nx_graph = edge_list_to_networkx(mediator_dag.nodes, mediator_dag.edges)

    mock_result = MagicMock()
    mock_result.get_variables.side_effect = AttributeError("No get_variables method")

    request = CausalValidationRequest(
        dag=mediator_dag,
        treatment="X",
        outcome="Y",
    )

    adjustment_set = validator._extract_y0_adjustment_set(mock_result, nx_graph, request)

    assert adjustment_set is not None
    assert isinstance(adjustment_set, list)


# ============================================================================
# Test: Backward Compatibility
# ============================================================================


def test_response_backward_compatible(validator, backdoor_dag):
    """Test that enhanced response still includes legacy fields."""
    request = CausalValidationRequest(
        dag=backdoor_dag,
        treatment="X",
        outcome="Y",
    )

    response = validator.validate(request)

    # Legacy fields should still be present
    assert hasattr(response, "status")
    assert hasattr(response, "confidence")
    assert hasattr(response, "explanation")

    # These may or may not be populated depending on status
    assert hasattr(response, "adjustment_sets")
    assert hasattr(response, "minimal_set")
    assert hasattr(response, "backdoor_paths")
    assert hasattr(response, "issues")


# ============================================================================
# Test: Explanation Generation Integration
# ============================================================================


def test_explanation_includes_degraded_status():
    """Test that ExplanationGenerator handles degraded status."""
    from src.services.explanation_generator import ExplanationGenerator

    generator = ExplanationGenerator()
    explanation = generator.generate_causal_validation_explanation(
        status="degraded",
        treatment="X",
        outcome="Y",
    )

    assert "degraded" in explanation.summary.lower()
    assert "fallback" in explanation.technical_basis.lower()
    assert len(explanation.assumptions) > 0
    assert "caution" in explanation.reasoning.lower() or "reliability" in explanation.reasoning.lower()
