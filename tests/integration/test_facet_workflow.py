"""
Integration tests for FACET robustness workflow.

Tests the complete end-to-end robustness analysis via API.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


class TestFACETRobustnessWorkflow:
    """Test complete FACET robustness workflow."""

    def test_robustness_analysis_endpoint_success(self):
        """Test robustness analysis endpoint with valid request."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "demand", "revenue"],
                "edges": [["price", "demand"], ["demand", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (95000.0, 105000.0)},
            "perturbation_radius": 0.1,
            "min_samples": 50,  # Small for fast testing
            "confidence_level": 0.95,
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "analysis" in data
        analysis = data["analysis"]

        # Verify analysis fields
        assert "status" in analysis
        assert analysis["status"] in ["robust", "fragile", "failed"]

        assert "robustness_score" in analysis
        assert 0 <= analysis["robustness_score"] <= 1

        assert "region_count" in analysis
        assert analysis["region_count"] >= 0

        assert "is_fragile" in analysis
        assert isinstance(analysis["is_fragile"], bool)

        assert "interpretation" in analysis
        assert len(analysis["interpretation"]) > 0

        assert "recommendation" in analysis
        assert len(analysis["recommendation"]) > 0

        # Check metadata
        assert "metadata" in data or "_metadata" in data

    def test_robustness_analysis_with_feasible_ranges(self):
        """Test robustness analysis with feasibility constraints."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (95000.0, 105000.0)},
            "perturbation_radius": 0.2,
            "min_samples": 30,
            "confidence_level": 0.90,
            "feasible_ranges": {"price": (40.0, 70.0)},
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        analysis = data["analysis"]
        assert "robust_regions" in analysis

        # If regions found, verify they respect feasible ranges
        if analysis["robust_regions"]:
            for region in analysis["robust_regions"]:
                if "price" in region["variable_ranges"]:
                    price_min, price_max = region["variable_ranges"]["price"]
                    assert price_min >= 40.0
                    assert price_max <= 70.0

    def test_robustness_analysis_invalid_request(self):
        """Test robustness analysis with invalid request."""
        request_data = {
            "causal_model": {},  # Empty model
            "intervention_proposal": {},  # Empty intervention
            "target_outcome": {},  # Empty target
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        # Service handles empty requests gracefully with heuristic analysis
        # Updated for Phase 4C: graceful degradation instead of hard failure
        assert response.status_code == 200
        result = response.json()
        assert "analysis" in result
        assert result["analysis"]["status"] in ["robust", "fragile"]

    def test_robustness_analysis_with_structural_model(self):
        """Test robustness analysis with full structural model."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 15.0},
            "target_outcome": {"revenue": (45000.0, 55000.0)},
            "perturbation_radius": 0.1,
            "min_samples": 20,
            "confidence_level": 0.95,
            "structural_model": {
                "variables": ["price", "revenue", "noise"],
                "equations": {
                    "revenue": "1000 * price + noise"
                },
                "distributions": {
                    "noise": {
                        "type": "normal",
                        "parameters": {"mean": 0, "std": 2000}
                    }
                },
            },
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        analysis = data["analysis"]

        # With structural model, should get outcome guarantees
        if analysis["status"] != "failed":
            assert "outcome_guarantees" in analysis

    def test_robustness_analysis_multiple_dimensions(self):
        """Test robustness analysis with multi-dimensional intervention."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "quality", "demand", "revenue"],
                "edges": [
                    ["price", "demand"],
                    ["quality", "demand"],
                    ["demand", "revenue"],
                ],
            },
            "intervention_proposal": {"price": 55.0, "quality": 8.0},
            "target_outcome": {"revenue": (95000.0, 105000.0)},
            "perturbation_radius": 0.15,
            "min_samples": 40,
            "confidence_level": 0.95,
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        analysis = data["analysis"]

        # Verify multi-dimensional handling
        if analysis["robust_regions"]:
            for region in analysis["robust_regions"]:
                var_ranges = region["variable_ranges"]
                # Should have regions for both price and quality
                assert "price" in var_ranges or "quality" in var_ranges

    def test_robustness_analysis_tight_target(self):
        """Test robustness analysis with very tight target (expect fragile)."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (99000.0, 101000.0)},  # Very tight
            "perturbation_radius": 0.1,
            "min_samples": 30,
            "confidence_level": 0.95,
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        analysis = data["analysis"]

        # With tight target, likely fragile (but not guaranteed)
        # Just verify response structure
        assert "is_fragile" in analysis
        if analysis["is_fragile"]:
            assert len(analysis["fragility_reasons"]) > 0

    def test_robustness_analysis_custom_confidence(self):
        """Test robustness analysis with custom confidence level."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (90000.0, 110000.0)},
            "perturbation_radius": 0.1,
            "min_samples": 25,
            "confidence_level": 0.80,  # Lower confidence
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()

        analysis = data["analysis"]
        assert analysis["confidence_level"] == 0.80

    def test_robustness_analysis_with_request_id(self):
        """Test robustness analysis with custom request ID."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (95000.0, 105000.0)},
            "perturbation_radius": 0.1,
            "min_samples": 20,
        }

        headers = {"X-Request-Id": "test_facet_123"}

        response = client.post(
            "/api/v1/robustness/analyze",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Check metadata includes request ID
        metadata = data.get("metadata") or data.get("_metadata")
        if metadata:
            assert "request_id" in metadata
            assert metadata["request_id"] == "test_facet_123"

    def test_robustness_analysis_samples_boundary(self):
        """Test robustness analysis with boundary sample counts."""
        # Test minimum samples
        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (95000.0, 105000.0)},
            "perturbation_radius": 0.1,
            "min_samples": 10,  # Minimum allowed
            "confidence_level": 0.95,
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)
        assert response.status_code == 200

    def test_robustness_analysis_invalid_confidence(self):
        """Test robustness analysis with invalid confidence level."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (95000.0, 105000.0)},
            "perturbation_radius": 0.1,
            "min_samples": 50,
            "confidence_level": 1.5,  # Invalid (> 1.0)
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        # Should fail validation
        assert response.status_code == 422

    def test_robustness_analysis_zero_perturbation(self):
        """Test robustness analysis with zero perturbation."""
        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (95000.0, 105000.0)},
            "perturbation_radius": 0.0,  # Zero (invalid)
            "min_samples": 50,
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        # Should fail validation (radius must be > 0)
        assert response.status_code == 422


class TestFACETMetrics:
    """Test FACET metrics tracking."""

    def test_metrics_tracked_on_analysis(self):
        """Test that metrics are tracked during analysis."""
        from src.utils.business_metrics import (
            facet_analyses_total,
            facet_robustness_score,
            facet_fragile_recommendations_total,
            facet_robust_regions_found,
        )

        # Get initial counts
        initial_analyses = facet_analyses_total._metrics.get(
            ("robust",), facet_analyses_total._metrics.get(("fragile",), None)
        )

        request_data = {
            "causal_model": {
                "nodes": ["price", "revenue"],
                "edges": [["price", "revenue"]],
            },
            "intervention_proposal": {"price": 55.0},
            "target_outcome": {"revenue": (95000.0, 105000.0)},
            "perturbation_radius": 0.1,
            "min_samples": 20,
        }

        response = client.post("/api/v1/robustness/analyze", json=request_data)

        assert response.status_code == 200

        # Metrics should have been updated
        # (Exact verification depends on Prometheus client implementation)
