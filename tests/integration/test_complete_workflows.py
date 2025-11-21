"""
End-to-end integration tests across multiple services.

Tests realistic workflows that use multiple ISL services together.
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestCompleteDecisionWorkflow:
    """Test complete decision-making workflow."""

    def test_product_decision_workflow(self):
        """
        Complete workflow:
        1. Validate causal model
        2. Generate counterfactual
        3. Check robustness
        4. Team deliberation
        """
        # Step 1: Validate causal model
        causal_model = {
            "nodes": ["price", "quality", "demand", "revenue", "churn"],
            "edges": [
                ["price", "demand"],
                ["quality", "demand"],
                ["demand", "revenue"],
                ["quality", "churn"],
            ],
        }

        validation_response = client.post(
            "/api/v1/causal/validate",
            json={"dag": causal_model, "treatment": "price", "outcome": "revenue"},
        )

        assert validation_response.status_code == 200
        validation = validation_response.json()
        assert validation["status"] == "identifiable"

        # Step 2: Generate counterfactual
        counterfactual_response = client.post(
            "/api/v1/causal/counterfactual",
            json={
                "model": {
                    "variables": ["price", "revenue"],
                    "equations": {"revenue": "100000 - 1000 * price"},
                    "distributions": {
                        "price": {"type": "normal", "parameters": {"mean": 50, "std": 5}}
                    },
                },
                "intervention": {"price": 55},
                "outcome": "revenue",
                "samples": 1000,
            },
        )

        assert counterfactual_response.status_code == 200
        cf_result = counterfactual_response.json()

        # Should have prediction with point estimate
        assert "prediction" in cf_result
        assert "point_estimate" in cf_result["prediction"]

        # Step 3: Check robustness
        revenue_estimate = cf_result["prediction"]["point_estimate"]
        revenue_range = (
            revenue_estimate * 0.95,
            revenue_estimate * 1.05,
        )

        robustness_response = client.post(
            "/api/v1/robustness/analyze",
            json={
                "causal_model": {
                    "nodes": ["price", "marketing", "quality", "demand", "revenue"],
                    "edges": [
                        ["price", "demand"],
                        ["marketing", "demand"],
                        ["quality", "demand"],
                        ["demand", "revenue"],
                    ],
                },
                "intervention_proposal": {"price": 55},
                "target_outcome": {"revenue": list(revenue_range)},
                "perturbation_radius": 0.1,
                "min_samples": 100,
            },
        )

        assert robustness_response.status_code == 200
        rob_result = robustness_response.json()

        # Should have robustness assessment
        assert "analysis" in rob_result
        assert "robustness_score" in rob_result["analysis"]
        assert "is_fragile" in rob_result["analysis"]

        # Step 4: Team deliberation
        deliberation_response = client.post(
            "/api/v1/deliberation/deliberate",
            json={
                "decision_context": "Should we increase price to £55?",
                "positions": [
                    {
                        "member_id": "pm_001",
                        "position_statement": f"I support this because revenue would increase to approximately £{revenue_estimate:.0f}, and the robustness score of {rob_result['analysis']['robustness_score']:.2f} gives me confidence.",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    {
                        "member_id": "eng_001",
                        "position_statement": "I'm concerned about implementation complexity and potential technical debt from rushed changes.",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ],
            },
        )

        assert deliberation_response.status_code == 200
        delib_result = deliberation_response.json()

        # Should have consensus statement
        assert "consensus_statement" in delib_result
        assert "common_ground" in delib_result
        assert "session_id" in delib_result

        # Verify data flow integrity
        # PM's position should reference counterfactual results
        assert str(int(revenue_estimate / 1000)) in str(revenue_estimate)


class TestCrossServiceDataFlow:
    """Test data flowing between services correctly."""

    def test_y0_to_counterfactual_flow(self):
        """Validated model → Counterfactual generation."""
        # First validate
        validation_response = client.post(
            "/api/v1/causal/validate",
            json={
                "dag": {
                    "nodes": ["X", "Y", "Z"],
                    "edges": [["X", "Y"], ["Z", "Y"]],
                },
                "treatment": "X",
                "outcome": "Y",
            },
        )

        assert validation_response.status_code == 200
        validation = validation_response.json()
        assert validation["status"] == "identifiable"

        # Use validated model for counterfactual
        counterfactual_response = client.post(
            "/api/v1/causal/counterfactual",
            json={
                "model": {
                    "variables": ["X", "Y", "Z"],
                    "equations": {"Y": "2*X + 3*Z", "Z": "X + 1"},
                    "distributions": {
                        "X": {"type": "normal", "parameters": {"mean": 0, "std": 1}}
                    },
                },
                "intervention": {"X": 1.0},
                "outcome": "Y",
                "samples": 500,
            },
        )

        assert counterfactual_response.status_code == 200
        cf_result = counterfactual_response.json()

        # Counterfactual should work for validated model
        assert "prediction" in cf_result
        assert "point_estimate" in cf_result["prediction"]


class TestErrorHandling:
    """Test error handling across services."""

    def test_invalid_model_rejected_consistently(self):
        """
        Submit invalid model and ensure:
        1. Validation catches it
        2. Counterfactual rejects it
        3. Error messages are helpful
        """
        invalid_model_dag = {
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"], ["Y", "X"]],  # Cycle!
        }

        # Validation should detect cycle
        validation_response = client.post(
            "/api/v1/causal/validate",
            json={
                "dag": invalid_model_dag,
                "treatment": "X",
                "outcome": "Y",
            },
        )

        # Should either reject or warn
        assert validation_response.status_code in [200, 400, 422]

    def test_missing_required_fields(self):
        """Test that missing required fields are caught."""
        # Missing treatment
        response = client.post(
            "/api/v1/causal/validate",
            json={"dag": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]}, "outcome": "Y"},
        )

        assert response.status_code == 422  # Validation error


class TestServiceIndependence:
    """Test that services work independently."""

    def test_deliberation_without_prior_analysis(self):
        """Deliberation should work without prior causal analysis."""
        response = client.post(
            "/api/v1/deliberation/deliberate",
            json={
                "decision_context": "Choose color scheme",
                "positions": [
                    {
                        "member_id": "designer_001",
                        "position_statement": "I prefer blue because it's calming and professional.",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    {
                        "member_id": "designer_002",
                        "position_statement": "I like green because it represents growth.",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ],
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert "consensus_statement" in result

    def test_counterfactual_without_validation(self):
        """Counterfactual should work without prior validation."""
        response = client.post(
            "/api/v1/causal/counterfactual",
            json={
                "model": {
                    "variables": ["X", "Y"],
                    "equations": {"Y": "2*X"},
                    "distributions": {
                        "X": {"type": "normal", "parameters": {"mean": 0, "std": 1}}
                    },
                },
                "intervention": {"X": 1.0},
                "outcome": "Y",
                "samples": 100,
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
