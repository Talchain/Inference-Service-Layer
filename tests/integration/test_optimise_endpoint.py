"""
Integration tests for POST /api/v1/analysis/optimise endpoint.

Tests the continuous optimization endpoint including:
- Grid search optimization
- Constraint handling (feasibility filtering)
- Confidence intervals
- Sensitivity analysis around optimum
- Edge cases and warnings
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestOptimiseEndpointBasic:
    """Basic integration tests for the optimise endpoint."""

    def test_optimize_single_variable_maximize(self, client):
        """Test maximizing single variable."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 2.0},
                "constant": 0.0,
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["schema_version"] == "optimise.v1"
        assert data["optimal_point"] is not None
        assert data["optimal_point"]["objective_value"] == pytest.approx(20.0, rel=1e-4)
        assert data["optimal_point"]["variable_values"]["x"] == pytest.approx(10.0, rel=1e-4)
        assert data["optimal_point"]["is_boundary"] is True

    def test_optimize_single_variable_minimize(self, client):
        """Test minimizing single variable."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "minimize",
                "coefficients": {"x": 2.0},
                "constant": 0.0,
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["optimal_point"]["objective_value"] == pytest.approx(0.0, rel=1e-4)
        assert data["optimal_point"]["variable_values"]["x"] == pytest.approx(0.0, rel=1e-4)

    def test_optimize_two_variables(self, client):
        """Test optimizing two variables."""
        request_data = {
            "objective": {
                "variable_id": "profit",
                "direction": "maximize",
                "coefficients": {"x": 3.0, "y": 2.0},
                "constant": 0.0,
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 5},
                {"variable_id": "y", "lower_bound": 0, "upper_bound": 10},
            ],
            "grid_points": 6,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Maximum at x=5, y=10 → f = 15 + 20 = 35
        assert data["optimal_point"]["objective_value"] == pytest.approx(35.0, rel=1e-4)
        assert data["optimal_point"]["variable_values"]["x"] == pytest.approx(5.0, rel=1e-4)
        assert data["optimal_point"]["variable_values"]["y"] == pytest.approx(10.0, rel=1e-4)


class TestOptimiseEndpointConstraints:
    """Test constraint handling."""

    def test_less_than_constraint(self, client):
        """Test less-than-or-equal constraint."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 2.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "constraints": [
                {
                    "constraint_id": "limit",
                    "coefficients": {"x": 1.0},
                    "relation": "le",
                    "rhs": 5.0,
                }
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should stop at x=5 due to constraint
        assert data["optimal_point"]["variable_values"]["x"] == pytest.approx(5.0, rel=1e-4)
        assert data["optimal_point"]["objective_value"] == pytest.approx(10.0, rel=1e-4)

    def test_greater_than_constraint(self, client):
        """Test greater-than-or-equal constraint."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "minimize",
                "coefficients": {"x": 2.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "constraints": [
                {
                    "constraint_id": "minimum",
                    "coefficients": {"x": 1.0},
                    "relation": "ge",
                    "rhs": 3.0,
                }
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should stop at x=3 due to constraint
        assert data["optimal_point"]["variable_values"]["x"] == pytest.approx(3.0, rel=1e-4)
        assert data["optimal_point"]["objective_value"] == pytest.approx(6.0, rel=1e-4)

    def test_multiple_constraints(self, client):
        """Test multiple constraints (budget allocation problem)."""
        request_data = {
            "objective": {
                "variable_id": "profit",
                "direction": "maximize",
                "coefficients": {"price": 3.0, "quantity": 2.0},
            },
            "decision_variables": [
                {"variable_id": "price", "lower_bound": 0, "upper_bound": 100},
                {"variable_id": "quantity", "lower_bound": 0, "upper_bound": 100},
            ],
            "constraints": [
                {
                    "constraint_id": "budget",
                    "coefficients": {"price": 1.0, "quantity": 1.0},
                    "relation": "le",
                    "rhs": 100.0,
                },
                {
                    "constraint_id": "min_price",
                    "coefficients": {"price": 1.0},
                    "relation": "ge",
                    "rhs": 10.0,
                },
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["optimal_point"] is not None
        # Budget constraint should be satisfied
        total = (
            data["optimal_point"]["variable_values"]["price"]
            + data["optimal_point"]["variable_values"]["quantity"]
        )
        assert total <= 100.0 + 0.01  # Allow small tolerance

    def test_no_feasible_solution(self, client):
        """Test conflicting constraints return no solution."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 1.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 100}
            ],
            "constraints": [
                {
                    "constraint_id": "c1",
                    "coefficients": {"x": 1.0},
                    "relation": "ge",
                    "rhs": 60.0,
                },
                {
                    "constraint_id": "c2",
                    "coefficients": {"x": 1.0},
                    "relation": "le",
                    "rhs": 40.0,
                },
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["optimal_point"] is None
        assert data["grid_metrics"]["feasible_points"] == 0
        assert len(data["warnings"]) > 0
        assert any(w["code"] == "NO_FEASIBLE_SOLUTION" for w in data["warnings"])


class TestOptimiseEndpointConfidenceIntervals:
    """Test confidence interval computation."""

    def test_confidence_interval_included(self, client):
        """Test confidence interval is computed."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 10.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
            "confidence_level": 0.95,
            "noise_std": 5.0,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["optimal_point"] is not None
        ci = data["optimal_point"]["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert ci["confidence_level"] == 0.95
        # CI should contain the optimal value
        assert ci["lower"] < data["optimal_point"]["objective_value"] < ci["upper"]

    def test_different_confidence_levels(self, client):
        """Test different confidence levels give different interval widths."""
        base_request = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 10.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
            "noise_std": 5.0,
        }

        # 90% CI
        request_90 = {**base_request, "confidence_level": 0.90}
        response_90 = client.post("/api/v1/analysis/optimise", json=request_90)

        # 99% CI
        request_99 = {**base_request, "confidence_level": 0.99}
        response_99 = client.post("/api/v1/analysis/optimise", json=request_99)

        assert response_90.status_code == 200
        assert response_99.status_code == 200

        ci_90 = response_90.json()["optimal_point"]["confidence_interval"]
        ci_99 = response_99.json()["optimal_point"]["confidence_interval"]

        width_90 = ci_90["upper"] - ci_90["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]

        # 99% CI should be wider than 90% CI
        assert width_99 > width_90


class TestOptimiseEndpointSensitivityAnalysis:
    """Test sensitivity analysis at optimum."""

    def test_sensitivity_analysis_included(self, client):
        """Test sensitivity analysis is computed."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 5.0, "y": 3.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10},
                {"variable_id": "y", "lower_bound": 0, "upper_bound": 10},
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["sensitivity"] is not None
        sensitivity = data["sensitivity"]

        # Check gradient
        assert "gradient_at_optimum" in sensitivity
        assert sensitivity["gradient_at_optimum"]["x"] == pytest.approx(5.0, rel=1e-4)
        assert sensitivity["gradient_at_optimum"]["y"] == pytest.approx(3.0, rel=1e-4)

        # Check 5% tolerance range
        assert "range_within_5pct" in sensitivity
        assert "x" in sensitivity["range_within_5pct"]
        assert "y" in sensitivity["range_within_5pct"]

        # Check robustness
        assert "robustness" in sensitivity
        assert sensitivity["robustness"] in ["robust", "moderate", "fragile"]
        assert 0 <= sensitivity["robustness_score"] <= 1

    def test_critical_variables_identified(self, client):
        """Test critical variables are identified."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 10.0, "y": 1.0},  # x much more important
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10},
                {"variable_id": "y", "lower_bound": 0, "upper_bound": 10},
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # x should be critical (higher coefficient)
        assert "x" in data["sensitivity"]["critical_variables"]


class TestOptimiseEndpointWarnings:
    """Test warning generation."""

    def test_boundary_optimum_warning(self, client):
        """Test boundary optimum generates warning."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 1.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["optimal_point"]["is_boundary"] is True
        assert any(w["code"] == "BOUNDARY_OPTIMUM" for w in data["warnings"])

    def test_flat_objective_warning(self, client):
        """Test flat objective generates warning."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"y": 1.0},  # y not in decision variables
                "constant": 10.0,
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert any(w["code"] == "FLAT_OBJECTIVE" for w in data["warnings"])


class TestOptimiseEndpointGridMetrics:
    """Test grid metrics reporting."""

    def test_grid_metrics_reported(self, client):
        """Test grid metrics are accurately reported."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 1.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        metrics = data["grid_metrics"]
        assert metrics["grid_points_evaluated"] == 11
        assert metrics["feasible_points"] == 11
        assert metrics["computation_time_ms"] > 0
        assert metrics["convergence_achieved"] is True

    def test_two_variable_grid_size(self, client):
        """Test two variable grid has correct size."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 1.0, "y": 1.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10},
                {"variable_id": "y", "lower_bound": 0, "upper_bound": 10},
            ],
            "grid_points": 5,
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # 5 points per variable = 5 * 5 = 25 total
        assert data["grid_metrics"]["grid_points_evaluated"] == 25


class TestOptimiseEndpointMetadata:
    """Test metadata handling."""

    def test_request_id_in_metadata(self, client):
        """Test request ID is included in response metadata."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 1.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 11,
        }

        response = client.post(
            "/api/v1/analysis/optimise",
            json=request_data,
            headers={"X-Request-Id": "test-opt-123"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "metadata" in data
        assert data["metadata"]["request_id"] == "test-opt-123"


class TestOptimiseEndpointValidation:
    """Test request validation."""

    def test_invalid_direction_rejected(self, client):
        """Test invalid direction is rejected."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "optimal",  # Invalid
                "coefficients": {"x": 1.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)
        assert response.status_code == 422

    def test_upper_bound_less_than_lower_rejected(self, client):
        """Test upper_bound < lower_bound is rejected."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 1.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 100, "upper_bound": 10}  # Invalid
            ],
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)
        assert response.status_code == 422

    def test_empty_coefficients_rejected(self, client):
        """Test empty coefficients rejected."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {},  # Empty
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)
        assert response.status_code == 422

    def test_too_few_grid_points_rejected(self, client):
        """Test too few grid points rejected."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 1.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10}
            ],
            "grid_points": 2,  # Too few (< 5)
        }

        response = client.post("/api/v1/analysis/optimise", json=request_data)
        assert response.status_code == 422


class TestOptimiseEndpointReproducibility:
    """Test reproducibility with seed."""

    def test_same_seed_same_result(self, client):
        """Test same seed produces same result."""
        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 3.0, "y": 2.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 10},
                {"variable_id": "y", "lower_bound": 0, "upper_bound": 10},
            ],
            "grid_points": 11,
            "seed": 42,
        }

        response1 = client.post("/api/v1/analysis/optimise", json=request_data)
        response2 = client.post("/api/v1/analysis/optimise", json=request_data)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        assert data1["optimal_point"]["objective_value"] == data2["optimal_point"]["objective_value"]
        assert data1["optimal_point"]["variable_values"] == data2["optimal_point"]["variable_values"]


class TestOptimiseEndpointPerformance:
    """Test performance requirements."""

    def test_performance_under_2_seconds(self, client):
        """Test optimization completes in under 2 seconds."""
        import time

        request_data = {
            "objective": {
                "variable_id": "output",
                "direction": "maximize",
                "coefficients": {"x": 3.0, "y": 2.0},
            },
            "decision_variables": [
                {"variable_id": "x", "lower_bound": 0, "upper_bound": 100},
                {"variable_id": "y", "lower_bound": 0, "upper_bound": 100},
            ],
            "grid_points": 20,
        }

        start = time.time()
        response = client.post("/api/v1/analysis/optimise", json=request_data)
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 2.0, f"Optimization took {elapsed}s, expected <2s"
