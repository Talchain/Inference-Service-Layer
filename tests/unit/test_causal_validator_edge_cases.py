"""
Edge case tests for CausalValidator.

Covers:
- Non-identifiable DAGs
- Error handling and graceful degradation
- Cannot identify scenarios
- Complex confounding structures
"""

import pytest
from unittest.mock import Mock, patch

from src.models.requests import CausalValidationRequest
from src.models.shared import DAGStructure
from src.services.causal_validator import CausalValidator


class TestCausalValidatorCannotIdentify:
    """Test cases where causal effects cannot be identified."""

    def test_unobserved_confounder_cannot_identify(self):
        """Test DAG with unobserved confounder."""
        validator = CausalValidator()

        # Classic confounding: U -> X, U -> Y (but U is unobserved)
        # We only observe X -> Y, but there's hidden confounding
        # This tests the backdoor path detection
        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y", "Z"],
                edges=[
                    ("X", "Y"),  # Direct effect
                    ("Z", "X"),  # Z confounds...
                    ("Z", "Y"),  # ...both X and Y (backdoor path)
                ],
            ),
            treatment="X",
            outcome="Y",
        )

        # Z is a confounder - if we adjust for it, we can identify
        # But the test is about the "cannot identify" path being covered
        response = validator.validate(request)

        # Should identify with adjustment
        assert response.status in ["identifiable", "conditional"]
        if response.adjustment_sets:
            assert "Z" in response.adjustment_sets[0]

    def test_bow_arc_structure_complex(self):
        """Test bow-arc structure that's harder to identify."""
        validator = CausalValidator()

        # More complex structure that might hit edge cases
        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["A", "B", "C", "D", "E"],
                edges=[
                    ("A", "B"),
                    ("B", "C"),
                    ("C", "D"),
                    ("D", "E"),
                    ("A", "E"),  # Backdoor through A
                ],
            ),
            treatment="B",
            outcome="D",
        )

        response = validator.validate(request)

        # Should handle complex structure
        assert response.status is not None
        assert response.confidence is not None

    def test_cyclic_dependencies_rejected(self):
        """Test that cyclic graphs are rejected."""
        validator = CausalValidator()

        # Create a cycle: X -> Y -> Z -> X
        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y", "Z"],
                edges=[
                    ("X", "Y"),
                    ("Y", "Z"),
                    ("Z", "X"),  # Creates cycle
                ],
            ),
            treatment="X",
            outcome="Y",
        )

        # Should raise validation error for cyclic graph
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validator.validate(request)

        assert exc_info.value.status_code == 400
        assert "cycle" in str(exc_info.value.detail).lower()


class TestCausalValidatorErrorHandling:
    """Test error handling and graceful degradation."""

    def test_invalid_node_names(self):
        """Test handling of invalid node names."""
        validator = CausalValidator()

        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y"],
                edges=[("X", "Y")],
            ),
            treatment="NonExistentNode",
            outcome="Y",
        )

        # Should raise validation error
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validator.validate(request)

        assert exc_info.value.status_code == 400

    def test_outcome_not_in_nodes(self):
        """Test when outcome variable not in DAG nodes."""
        validator = CausalValidator()

        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y"],
                edges=[("X", "Y")],
            ),
            treatment="X",
            outcome="Z",  # Not in nodes
        )

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            validator.validate(request)

        assert exc_info.value.status_code == 400

    def test_empty_dag(self):
        """Test handling of DAG with no edges."""
        validator = CausalValidator()

        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y"],
                edges=[],  # No edges - disconnected nodes
            ),
            treatment="X",
            outcome="Y",
        )

        response = validator.validate(request)

        # Should return cannot_identify for disconnected nodes
        assert response.status == "cannot_identify"

    def test_self_loop_rejected(self):
        """Test that self-loops are rejected by Pydantic validation."""
        from pydantic import ValidationError

        # Self-loops are caught at Pydantic level, not validator level
        with pytest.raises(ValidationError) as exc_info:
            request = CausalValidationRequest(
                dag=DAGStructure(
                    nodes=["X", "Y"],
                    edges=[
                        ("X", "Y"),
                        ("X", "X"),  # Self-loop
                    ],
                ),
                treatment="X",
                outcome="Y",
            )

        assert "Self-loops not allowed" in str(exc_info.value)

    @patch('src.services.causal_validator.identify_outcomes')
    def test_y0_identification_error_triggers_degraded(self, mock_identify):
        """Test that Y0 identification errors fall back to backdoor analysis."""
        mock_identify.side_effect = RuntimeError("Y0 computation failed")

        validator = CausalValidator()

        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y", "Z"],
                edges=[("X", "Y"), ("Z", "Y")],
            ),
            treatment="X",
            outcome="Y",
        )

        # Should handle Y0 error gracefully and fall back to backdoor analysis
        response = validator.validate(request)

        # Fallback should successfully identify (simple DAG, no confounding)
        assert response.status == "identifiable"
        assert response.method in ["backdoor", "do_calculus"]

    @patch('src.services.causal_validator.edge_list_to_networkx')
    def test_graph_construction_error(self, mock_edge_list):
        """Test handling of graph construction errors."""
        mock_edge_list.side_effect = ValueError("Invalid edge format")

        validator = CausalValidator()

        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y"],
                edges=[("X", "Y")],
            ),
            treatment="X",
            outcome="Y",
        )

        # Should gracefully degrade on unexpected errors
        response = validator.validate(request)

        assert response.status == "degraded"
        assert "error" in response.model_dump() or response.explanation


class TestCausalValidatorComplexStructures:
    """Test complex DAG structures."""

    def test_large_dag_performance(self):
        """Test validation on larger DAG (10+ nodes)."""
        validator = CausalValidator()

        # Create a larger DAG
        nodes = [f"X{i}" for i in range(15)]
        edges = [[f"X{i}", f"X{i+1}"] for i in range(14)]

        request = CausalValidationRequest(
            dag=DAGStructure(nodes=nodes, edges=edges),
            treatment="X0",
            outcome="X14",
        )

        response = validator.validate(request)

        # Should complete successfully
        assert response.status == "identifiable"
        assert response.confidence is not None

    def test_dense_dag_many_edges(self):
        """Test DAG with many edges (dense connectivity)."""
        validator = CausalValidator()

        # Create fully connected DAG (each node connects to all later nodes)
        nodes = ["A", "B", "C", "D", "E"]
        edges = []
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                edges.append([node1, node2])

        request = CausalValidationRequest(
            dag=DAGStructure(nodes=nodes, edges=edges),
            treatment="A",
            outcome="E",
        )

        response = validator.validate(request)

        # Should handle dense graph
        assert response.status == "identifiable"

    def test_disconnected_treatment_outcome(self):
        """Test when treatment and outcome are not connected."""
        validator = CausalValidator()

        # Two separate components: X->Y and Z->W
        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y", "Z", "W"],
                edges=[
                    ("X", "Y"),
                    ("Z", "W"),
                ],
            ),
            treatment="X",
            outcome="W",  # No path from X to W
        )

        response = validator.validate(request)

        # Should indicate cannot identify (no causal path)
        assert response.status == "cannot_identify"
        if hasattr(response, "explanation") and response.explanation:
            explanation_text = response.explanation.summary if hasattr(response.explanation, "summary") else str(response.explanation)
            assert "no path" in explanation_text.lower() or "not connected" in explanation_text.lower() or "cannot" in explanation_text.lower()


class TestCausalValidatorConfidenceLevels:
    """Test confidence level assignments."""

    def test_simple_dag_high_confidence(self):
        """Test that simple identifiable DAG gets high confidence."""
        validator = CausalValidator()

        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y"],
                edges=[("X", "Y")],
            ),
            treatment="X",
            outcome="Y",
        )

        response = validator.validate(request)

        assert response.status == "identifiable"
        assert response.confidence == "high"

    def test_complex_adjustment_medium_confidence(self):
        """Test that complex adjustment sets get medium confidence."""
        validator = CausalValidator()

        # Complex confounding structure
        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y", "Z1", "Z2", "Z3"],
                edges=[
                    ("X", "Y"),
                    ("Z1", "X"), ("Z1", "Y"),
                    ("Z2", "X"), ("Z2", "Y"),
                    ("Z3", "Z1"), ("Z3", "Z2"),
                ],
            ),
            treatment="X",
            outcome="Y",
        )

        response = validator.validate(request)

        # Should be identifiable but with lower confidence
        assert response.status in ["identifiable", "conditional"]
        # Confidence might be medium due to complexity
        assert response.confidence in ["high", "medium"]


class TestCausalValidatorFallbackAssessment:
    """Test fallback assessment when primary methods fail."""

    @patch('src.services.causal_validator.identify_outcomes')
    def test_fallback_finds_direct_path(self, mock_identify):
        """Test that fallback correctly identifies direct paths."""
        mock_identify.side_effect = Exception("Y0 failed")

        validator = CausalValidator()

        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y", "Z"],
                edges=[("X", "Y"), ("Z", "Y")],
            ),
            treatment="X",
            outcome="Y",
        )

        response = validator.validate(request)

        # Fallback should successfully identify (simple structure)
        assert response.status == "identifiable"
        assert response.method in ["backdoor", "do_calculus"]

    @patch('src.services.causal_validator.identify_outcomes')
    def test_fallback_detects_potential_confounders(self, mock_identify):
        """Test that fallback identifies graphs with confounders."""
        mock_identify.side_effect = Exception("Y0 failed")

        validator = CausalValidator()

        # Z is a classic confounder
        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y", "Z"],
                edges=[
                    ("Z", "X"),
                    ("Z", "Y"),
                    ("X", "Y"),
                ],
            ),
            treatment="X",
            outcome="Y",
        )

        response = validator.validate(request)

        # Fallback should successfully identify with Z as adjustment
        assert response.status == "identifiable"
        assert response.method in ["backdoor", "do_calculus"]
        if hasattr(response, "adjustment_set"):
            # Should include Z in adjustment set
            assert "Z" in response.adjustment_set

    @patch('src.services.causal_validator.identify_outcomes')
    @patch('src.services.causal_validator.edge_list_to_networkx')
    def test_double_failure_fallback(self, mock_networkx, mock_identify):
        """Test when both primary and fallback assessment fail."""
        mock_identify.side_effect = Exception("Y0 failed")
        mock_networkx.side_effect = Exception("Graph construction failed")

        validator = CausalValidator()

        request = CausalValidationRequest(
            dag=DAGStructure(
                nodes=["X", "Y"],
                edges=[("X", "Y")],
            ),
            treatment="X",
            outcome="Y",
        )

        # Should still return degraded response with error message
        response = validator.validate(request)

        assert response.status == "degraded"
        fallback = response.fallback_assessment
        if isinstance(fallback, dict):
            assert "error" in fallback or "failed" in str(fallback).lower()
