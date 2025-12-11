"""
Integration tests for Phase 1 critical endpoints.

Tests:
- POST /api/v1/validation/feasibility
- POST /api/v1/validation/coherence
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestFeasibilityEndpoint:
    """Integration tests for /api/v1/validation/feasibility endpoint."""

    def test_feasibility_endpoint_success(self, client):
        """Test successful feasibility check."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Maximize Profit"},
                    {"id": "option_a", "kind": "option", "label": "Option A"},
                    {"id": "option_b", "kind": "option", "label": "Option B"},
                ],
                "edges": [],
            },
            "constraints": [
                {
                    "id": "budget_limit",
                    "constraint_type": "budget",
                    "target_variable": "total_cost",
                    "relation": "le",
                    "threshold": 100000.0,
                    "label": "Budget constraint",
                    "priority": "hard",
                }
            ],
            "options": [
                {
                    "option_id": "option_a",
                    "name": "Option A",
                    "variable_values": {"total_cost": 85000},
                    "expected_value": 50000,
                },
                {
                    "option_id": "option_b",
                    "name": "Option B",
                    "variable_values": {"total_cost": 120000},
                    "expected_value": 80000,
                },
            ],
        }

        response = client.post("/api/v1/validation/feasibility", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["schema_version"] == "feasibility.v1"
        assert "feasibility" in data
        assert "option_a" in data["feasibility"]["feasible_options"]
        assert len(data["feasibility"]["infeasible_options"]) == 1
        assert data["feasibility"]["infeasible_options"][0]["option_id"] == "option_b"

    def test_feasibility_endpoint_all_feasible(self, client):
        """Test when all options are feasible."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "constraints": [
                {
                    "id": "budget",
                    "constraint_type": "budget",
                    "target_variable": "cost",
                    "relation": "le",
                    "threshold": 100000.0,
                    "priority": "hard",
                }
            ],
            "options": [
                {"option_id": "opt1", "name": "Opt1", "variable_values": {"cost": 50000}},
                {"option_id": "opt2", "name": "Opt2", "variable_values": {"cost": 60000}},
            ],
        }

        response = client.post("/api/v1/validation/feasibility", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert len(data["feasibility"]["feasible_options"]) == 2
        assert len(data["feasibility"]["infeasible_options"]) == 0

    def test_feasibility_endpoint_all_infeasible(self, client):
        """Test when all options are infeasible."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "constraints": [
                {
                    "id": "budget",
                    "constraint_type": "budget",
                    "target_variable": "cost",
                    "relation": "le",
                    "threshold": 50000.0,
                    "priority": "hard",
                }
            ],
            "options": [
                {"option_id": "opt1", "name": "Opt1", "variable_values": {"cost": 100000}},
                {"option_id": "opt2", "name": "Opt2", "variable_values": {"cost": 150000}},
            ],
        }

        response = client.post("/api/v1/validation/feasibility", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert len(data["feasibility"]["feasible_options"]) == 0
        assert len(data["feasibility"]["infeasible_options"]) == 2
        assert any("No feasible options" in w for w in data["warnings"])

    def test_feasibility_endpoint_constraint_validation(self, client):
        """Test constraint validation in response."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "constraints": [
                {
                    "id": "valid_constraint",
                    "constraint_type": "threshold",
                    "target_variable": "value",
                    "relation": "ge",
                    "threshold": 100.0,
                    "priority": "hard",
                },
                {
                    "id": "invalid_constraint",
                    "constraint_type": "threshold",
                    "target_variable": "nonexistent_var",
                    "relation": "le",
                    "threshold": 50.0,
                    "priority": "hard",
                },
            ],
            "options": [
                {"option_id": "opt1", "name": "Opt1", "variable_values": {"value": 150}},
            ],
        }

        response = client.post("/api/v1/validation/feasibility", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check constraint validation results
        assert len(data["constraint_validation"]) == 2

        valid_result = next(
            cv for cv in data["constraint_validation"]
            if cv["constraint_id"] == "valid_constraint"
        )
        assert valid_result["is_valid"]

        invalid_result = next(
            cv for cv in data["constraint_validation"]
            if cv["constraint_id"] == "invalid_constraint"
        )
        assert not invalid_result["is_valid"]
        assert any("not found" in issue for issue in invalid_result["issues"])

    def test_feasibility_endpoint_multiple_constraints(self, client):
        """Test with multiple constraints."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "constraints": [
                {
                    "id": "budget",
                    "constraint_type": "budget",
                    "target_variable": "cost",
                    "relation": "le",
                    "threshold": 100000.0,
                    "priority": "hard",
                },
                {
                    "id": "quality",
                    "constraint_type": "threshold",
                    "target_variable": "quality",
                    "relation": "ge",
                    "threshold": 80.0,
                    "priority": "hard",
                },
            ],
            "options": [
                {
                    "option_id": "good",
                    "name": "Good",
                    "variable_values": {"cost": 90000, "quality": 85},
                },
                {
                    "option_id": "expensive",
                    "name": "Expensive",
                    "variable_values": {"cost": 120000, "quality": 95},
                },
                {
                    "option_id": "low_quality",
                    "name": "Low Quality",
                    "variable_values": {"cost": 50000, "quality": 60},
                },
            ],
        }

        response = client.post("/api/v1/validation/feasibility", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Only "good" should be feasible
        assert data["feasibility"]["feasible_options"] == ["good"]
        assert len(data["feasibility"]["infeasible_options"]) == 2

    def test_feasibility_endpoint_violation_details(self, client):
        """Test violation details are properly returned."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "constraints": [
                {
                    "id": "budget",
                    "constraint_type": "budget",
                    "target_variable": "cost",
                    "relation": "le",
                    "threshold": 100000.0,
                    "label": "Budget limit",
                    "priority": "hard",
                }
            ],
            "options": [
                {
                    "option_id": "over_budget",
                    "name": "Over Budget",
                    "variable_values": {"cost": 130000},
                },
            ],
        }

        response = client.post("/api/v1/validation/feasibility", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert len(data["feasibility"]["infeasible_options"]) == 1
        infeasible = data["feasibility"]["infeasible_options"][0]

        assert infeasible["option_id"] == "over_budget"
        assert "budget" in infeasible["violated_constraints"]
        assert len(infeasible["violation_details"]) == 1

        violation = infeasible["violation_details"][0]
        assert violation["constraint_id"] == "budget"
        assert violation["actual_value"] == 130000
        assert violation["threshold"] == 100000
        assert violation["violation_magnitude"] == 30000
        assert violation["is_hard_violation"]

    def test_feasibility_endpoint_with_request_id(self, client):
        """Test that X-Request-Id header is respected."""
        request_data = {
            "graph": {
                "nodes": [{"id": "goal", "kind": "goal", "label": "Goal"}],
                "edges": [],
            },
            "constraints": [
                {
                    "id": "c1",
                    "constraint_type": "threshold",
                    "target_variable": "v",
                    "relation": "le",
                    "threshold": 100.0,
                    "priority": "hard",
                }
            ],
            "options": [
                {"option_id": "opt1", "name": "Opt1", "variable_values": {"v": 50}},
            ],
        }

        response = client.post(
            "/api/v1/validation/feasibility",
            json=request_data,
            headers={"X-Request-Id": "test-request-123"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["request_id"] == "test-request-123"


class TestCoherenceEndpoint:
    """Integration tests for /api/v1/validation/coherence endpoint."""

    def test_coherence_endpoint_success(self, client):
        """Test successful coherence analysis."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Maximize Revenue"},
                    {"id": "option_a", "kind": "option", "label": "Option A"},
                    {"id": "option_b", "kind": "option", "label": "Option B"},
                ],
                "edges": [],
            },
            "options": [
                {"option_id": "option_a", "name": "Option A", "expected_value": 50000, "rank": 1},
                {"option_id": "option_b", "name": "Option B", "expected_value": 40000, "rank": 2},
            ],
            "perturbation_magnitude": 0.1,
            "close_race_threshold": 0.05,
            "num_perturbations": 50,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["schema_version"] == "coherence.v1"
        assert "coherence_analysis" in data
        assert "stability_analysis" in data
        assert "recommendations" in data

        # Verify coherence analysis fields
        coherence = data["coherence_analysis"]
        assert "top_option_positive" in coherence
        assert "margin_to_second" in coherence
        assert "margin_to_second_pct" in coherence
        assert "ranking_stability" in coherence
        assert "stability_score" in coherence
        assert "warnings" in coherence

    def test_coherence_endpoint_negative_top_value(self, client):
        """Test detection of negative expected value at top."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "Option A", "expected_value": -10000, "rank": 1},
                {"option_id": "opt_b", "name": "Option B", "expected_value": -20000, "rank": 2},
            ],
            "num_perturbations": 20,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["coherence_analysis"]["top_option_positive"] is False
        assert any("negative expected value" in w.lower()
                   for w in data["coherence_analysis"]["warnings"])

    def test_coherence_endpoint_close_race(self, client):
        """Test detection of close race."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "Option A", "expected_value": 50000, "rank": 1},
                {"option_id": "opt_b", "name": "Option B", "expected_value": 49000, "rank": 2},  # 2% diff
            ],
            "close_race_threshold": 0.05,  # 5% threshold
            "num_perturbations": 20,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should detect close race (2% < 5%)
        assert any("close race" in w.lower() for w in data["coherence_analysis"]["warnings"])

    def test_coherence_endpoint_stability_analysis(self, client):
        """Test stability analysis in response."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "Option A", "expected_value": 100000, "rank": 1},
                {"option_id": "opt_b", "name": "Option B", "expected_value": 10000, "rank": 2},
            ],
            "perturbation_magnitude": 0.1,
            "num_perturbations": 100,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200
        data = response.json()

        stability = data["stability_analysis"]
        assert stability["num_perturbations"] == 100
        assert 0 <= stability["ranking_change_rate"] <= 1
        assert isinstance(stability["ranking_changes"], int)

        # With 10x margin, should be stable
        assert data["coherence_analysis"]["ranking_stability"] == "stable"

    def test_coherence_endpoint_sample_perturbations(self, client):
        """Test sample perturbations are included."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "Option A", "expected_value": 50000, "rank": 1},
                {"option_id": "opt_b", "name": "Option B", "expected_value": 48000, "rank": 2},
            ],
            "num_perturbations": 100,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200
        data = response.json()

        perturbations = data["stability_analysis"]["sample_perturbations"]
        assert len(perturbations) > 0
        assert len(perturbations) <= 10

        for perturb in perturbations:
            assert "perturbation_id" in perturb
            assert "top_option_id" in perturb
            assert "ranking_changed" in perturb
            assert "value_change_pct" in perturb

    def test_coherence_endpoint_recommendations(self, client):
        """Test that recommendations are generated."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "Option A", "expected_value": 100000, "rank": 1},
                {"option_id": "opt_b", "name": "Option B", "expected_value": 10000, "rank": 2},
            ],
            "num_perturbations": 50,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert len(data["recommendations"]) > 0

    def test_coherence_endpoint_with_confidence_intervals(self, client):
        """Test with confidence intervals."""
        request_data = {
            "graph": {
                "nodes": [
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
            "options": [
                {
                    "option_id": "opt_a",
                    "name": "Option A",
                    "expected_value": 50000,
                    "confidence_interval": [40000, 60000],
                    "rank": 1,
                },
                {
                    "option_id": "opt_b",
                    "name": "Option B",
                    "expected_value": 45000,
                    "confidence_interval": [35000, 55000],  # Overlaps with opt_a
                    "rank": 2,
                },
            ],
            "num_perturbations": 20,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should warn about overlapping intervals
        assert any("overlap" in r.lower() or "significantly different" in r.lower()
                   for r in data["recommendations"])

    def test_coherence_endpoint_with_request_id(self, client):
        """Test that X-Request-Id header is respected."""
        request_data = {
            "graph": {
                "nodes": [{"id": "goal", "kind": "goal", "label": "Goal"}],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "A", "expected_value": 50000, "rank": 1},
                {"option_id": "opt_b", "name": "B", "expected_value": 40000, "rank": 2},
            ],
            "num_perturbations": 10,
        }

        response = client.post(
            "/api/v1/validation/coherence",
            json=request_data,
            headers={"X-Request-Id": "coherence-test-456"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["request_id"] == "coherence-test-456"

    def test_coherence_endpoint_minimum_options(self, client):
        """Test with minimum required options (2)."""
        request_data = {
            "graph": {
                "nodes": [{"id": "goal", "kind": "goal", "label": "Goal"}],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "A", "expected_value": 50000},
                {"option_id": "opt_b", "name": "B", "expected_value": 40000},
            ],
            "num_perturbations": 10,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200

    def test_coherence_endpoint_multiple_options(self, client):
        """Test with multiple options."""
        request_data = {
            "graph": {
                "nodes": [{"id": "goal", "kind": "goal", "label": "Goal"}],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "A", "expected_value": 100000, "rank": 1},
                {"option_id": "opt_b", "name": "B", "expected_value": 80000, "rank": 2},
                {"option_id": "opt_c", "name": "C", "expected_value": 60000, "rank": 3},
                {"option_id": "opt_d", "name": "D", "expected_value": 40000, "rank": 4},
            ],
            "num_perturbations": 50,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["coherence_analysis"]["top_option_positive"]
        assert data["coherence_analysis"]["margin_to_second"] == 20000


class TestEndpointValidation:
    """Tests for request validation."""

    def test_feasibility_missing_constraints(self, client):
        """Test error when constraints are missing."""
        request_data = {
            "graph": {
                "nodes": [{"id": "goal", "kind": "goal", "label": "Goal"}],
                "edges": [],
            },
            "constraints": [],  # Empty constraints
            "options": [
                {"option_id": "opt1", "name": "Opt1", "variable_values": {"cost": 50000}},
            ],
        }

        response = client.post("/api/v1/validation/feasibility", json=request_data)

        # Should fail validation (min_length=1 for constraints)
        assert response.status_code == 422

    def test_coherence_insufficient_options(self, client):
        """Test error when fewer than 2 options provided."""
        request_data = {
            "graph": {
                "nodes": [{"id": "goal", "kind": "goal", "label": "Goal"}],
                "edges": [],
            },
            "options": [
                {"option_id": "opt_a", "name": "A", "expected_value": 50000},
            ],
            "num_perturbations": 10,
        }

        response = client.post("/api/v1/validation/coherence", json=request_data)

        # Should fail validation (min_length=2 for options)
        assert response.status_code == 422
