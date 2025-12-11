"""
Integration tests for Phase 2 Schema & Contracts endpoints.

Tests:
- POST /api/v1/utility/validate
- POST /api/v1/validation/correlations
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestUtilityValidateEndpoint:
    """Integration tests for /api/v1/utility/validate endpoint."""

    def test_utility_validate_equal_weights(self, client):
        """Test utility validation with no weights (equal weighting default)."""
        request_data = {
            "utility_spec": {
                "goals": [
                    {"goal_id": "profit", "label": "Maximize Profit", "direction": "maximize"},
                    {"goal_id": "cost", "label": "Minimize Cost", "direction": "minimize"},
                ],
                "aggregation_method": "weighted_sum",
                "risk_tolerance": "risk_neutral",
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["schema_version"] == "utility.v1"
        assert data["valid"] is True
        assert data["aggregation_method"] == "weighted_sum"
        assert data["risk_tolerance"] == "risk_neutral"

        # Check equal weights applied
        assert data["normalised_weights"]["profit"] == pytest.approx(0.5, rel=1e-4)
        assert data["normalised_weights"]["cost"] == pytest.approx(0.5, rel=1e-4)

        # Should have default weights warning
        assert any("DEFAULT_WEIGHTS" in w.get("code", "") for w in data["warnings"])

    def test_utility_validate_explicit_weights(self, client):
        """Test utility validation with explicit weights."""
        request_data = {
            "utility_spec": {
                "goals": [
                    {
                        "goal_id": "profit",
                        "label": "Maximize Profit",
                        "direction": "maximize",
                        "weight": 0.7,
                    },
                    {
                        "goal_id": "cost",
                        "label": "Minimize Cost",
                        "direction": "minimize",
                        "weight": 0.3,
                    },
                ],
                "aggregation_method": "weighted_sum",
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert data["normalised_weights"]["profit"] == pytest.approx(0.7, rel=1e-4)
        assert data["normalised_weights"]["cost"] == pytest.approx(0.3, rel=1e-4)

    def test_utility_validate_weight_normalization(self, client):
        """Test utility validation normalizes weights not summing to 1."""
        # Weights are valid (<=1) but don't sum to 1
        request_data = {
            "utility_spec": {
                "goals": [
                    {"goal_id": "g1", "label": "Goal 1", "direction": "maximize", "weight": 0.3},
                    {"goal_id": "g2", "label": "Goal 2", "direction": "maximize", "weight": 0.3},
                ],
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        # 0.3 / 0.6 = 0.5
        assert data["normalised_weights"]["g1"] == pytest.approx(0.5, rel=1e-4)
        assert data["normalised_weights"]["g2"] == pytest.approx(0.5, rel=1e-4)

        # Should have normalization warning
        assert any("WEIGHTS_NORMALIZED" in w.get("code", "") for w in data["warnings"])

    def test_utility_validate_lexicographic_method(self, client):
        """Test utility validation with lexicographic method."""
        request_data = {
            "utility_spec": {
                "goals": [
                    {"goal_id": "g1", "label": "Primary Goal", "priority": 1},
                    {"goal_id": "g2", "label": "Secondary Goal", "priority": 2},
                ],
                "aggregation_method": "lexicographic",
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert data["aggregation_method"] == "lexicographic"

    def test_utility_validate_lexicographic_no_priority_warning(self, client):
        """Test lexicographic method without priorities warns."""
        request_data = {
            "utility_spec": {
                "goals": [
                    {"goal_id": "g1", "label": "Goal 1"},
                    {"goal_id": "g2", "label": "Goal 2"},
                ],
                "aggregation_method": "lexicographic",
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        # Should warn about missing priorities
        assert any("AGGREGATION_METHOD" in w.get("code", "") for w in data["warnings"])

    def test_utility_validate_risk_averse(self, client):
        """Test utility validation with risk averse tolerance."""
        request_data = {
            "utility_spec": {
                "goals": [{"goal_id": "g1", "label": "Goal"}],
                "risk_tolerance": "risk_averse",
                "risk_coefficient": 2.0,
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert data["risk_tolerance"] == "risk_averse"

    def test_utility_validate_risk_averse_no_coefficient_warning(self, client):
        """Test risk averse without coefficient warns."""
        request_data = {
            "utility_spec": {
                "goals": [{"goal_id": "g1", "label": "Goal"}],
                "risk_tolerance": "risk_averse",
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert any("RISK_TOLERANCE" in w.get("code", "") for w in data["warnings"])

    def test_utility_validate_with_graph_reference(self, client):
        """Test utility validation with graph reference."""
        request_data = {
            "utility_spec": {
                "goals": [
                    {"goal_id": "profit", "label": "Profit", "weight": 0.6},
                    {"goal_id": "cost", "label": "Cost", "weight": 0.4},
                ],
            },
            "graph": {
                "nodes": [
                    {"id": "profit", "kind": "goal", "label": "Profit Goal"},
                    {"id": "cost", "kind": "goal", "label": "Cost Goal"},
                    {"id": "factor1", "kind": "factor", "label": "Factor 1"},
                ],
                "edges": [],
            },
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True

    def test_utility_validate_graph_missing_reference_warning(self, client):
        """Test warning when goal not in graph."""
        request_data = {
            "utility_spec": {
                "goals": [
                    {"goal_id": "profit", "label": "Profit", "weight": 0.5},
                    {"goal_id": "missing", "label": "Missing", "weight": 0.5},
                ],
            },
            "graph": {
                "nodes": [
                    {"id": "profit", "kind": "goal", "label": "Profit Goal"},
                ],
                "edges": [],
            },
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert any("GRAPH_REFERENCE" in w.get("code", "") for w in data["warnings"])

    def test_utility_validate_normalised_goals_output(self, client):
        """Test normalised_goals output structure."""
        request_data = {
            "utility_spec": {
                "goals": [
                    {"goal_id": "g1", "label": "Goal 1", "direction": "maximize", "weight": 0.6, "priority": 1},
                    {"goal_id": "g2", "label": "Goal 2", "direction": "minimize", "weight": 0.4, "priority": 2},
                ],
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert len(data["normalised_goals"]) == 2
        g1 = next(g for g in data["normalised_goals"] if g["goal_id"] == "g1")
        assert g1["label"] == "Goal 1"
        assert g1["direction"] == "maximize"
        assert g1["normalised_weight"] == pytest.approx(0.6, rel=1e-4)
        assert g1["original_weight"] == 0.6
        assert g1["priority"] == 1

    def test_utility_validate_all_aggregation_methods(self, client):
        """Test all aggregation methods are accepted."""
        methods = ["weighted_sum", "weighted_product", "lexicographic", "min_max"]

        for method in methods:
            request_data = {
                "utility_spec": {
                    "goals": [
                        {"goal_id": "g1", "label": "Goal 1", "weight": 0.5, "priority": 1},
                        {"goal_id": "g2", "label": "Goal 2", "weight": 0.5, "priority": 2},
                    ],
                    "aggregation_method": method,
                }
            }

            response = client.post("/api/v1/utility/validate", json=request_data)
            assert response.status_code == 200
            assert response.json()["aggregation_method"] == method

    def test_utility_validate_request_id_header(self, client):
        """Test request ID header is included in response metadata."""
        request_data = {
            "utility_spec": {
                "goals": [{"goal_id": "g1", "label": "Goal"}],
            }
        }

        response = client.post(
            "/api/v1/utility/validate",
            json=request_data,
            headers={"X-Request-Id": "test-req-123"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "metadata" in data
        assert data["metadata"]["request_id"] == "test-req-123"

    def test_utility_validate_invalid_request(self, client):
        """Test validation error for invalid request."""
        # Missing required fields
        request_data = {
            "utility_spec": {
                "aggregation_method": "weighted_sum",
                # Missing goals
            }
        }

        response = client.post("/api/v1/utility/validate", json=request_data)

        assert response.status_code == 422


class TestCorrelationValidateEndpoint:
    """Integration tests for /api/v1/validation/correlations endpoint."""

    def test_correlations_validate_single_group(self, client):
        """Test correlation validation with single group."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "market_factors",
                    "factors": ["demand", "competition"],
                    "correlation": 0.7,
                    "label": "Market factors tend to move together",
                }
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["schema_version"] == "correlation.v1"
        assert data["valid"] is True
        assert len(data["validated_groups"]) == 1
        assert data["validated_groups"][0]["is_valid"] is True

        # Check implied matrix
        assert data["implied_matrix"] is not None
        assert len(data["implied_matrix"]["factors"]) == 2

    def test_correlations_validate_multiple_groups(self, client):
        """Test correlation validation with multiple groups."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "market",
                    "factors": ["demand", "price"],
                    "correlation": 0.7,
                },
                {
                    "group_id": "costs",
                    "factors": ["labor", "materials"],
                    "correlation": 0.5,
                },
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert len(data["validated_groups"]) == 2
        assert len(data["implied_matrix"]["factors"]) == 4

    def test_correlations_validate_matrix_psd_check(self, client):
        """Test PSD check on implied matrix."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["a", "b", "c"],
                    "correlation": 0.5,
                }
            ],
            "check_positive_definite": True,
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert data["matrix_analysis"] is not None
        assert data["matrix_analysis"]["is_positive_semi_definite"] is True

    def test_correlations_validate_non_psd_matrix(self, client):
        """Test detection of non-PSD matrix."""
        # Direct matrix input that's not PSD
        request_data = {
            "correlation_matrix": {
                "factors": ["a", "b", "c"],
                "matrix": [
                    [1.0, 0.9, 0.9],
                    [0.9, 1.0, -0.9],
                    [0.9, -0.9, 1.0],
                ],
            },
            "check_positive_definite": True,
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is False
        assert data["matrix_analysis"]["is_positive_semi_definite"] is False
        assert data["matrix_analysis"]["suggested_regularization"] is not None
        assert any("positive semi-definite" in err for err in data["errors"])

    def test_correlations_validate_direct_matrix(self, client):
        """Test validation with direct matrix input."""
        request_data = {
            "correlation_matrix": {
                "factors": ["a", "b", "c"],
                "matrix": [
                    [1.0, 0.5, 0.3],
                    [0.5, 1.0, 0.4],
                    [0.3, 0.4, 1.0],
                ],
            }
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert data["implied_matrix"]["factors"] == ["a", "b", "c"]

    def test_correlations_validate_high_correlation_warning(self, client):
        """Test warning for high correlations."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["a", "b"],
                    "correlation": 0.95,
                }
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert any(w["code"] == "HIGH_CORRELATION" for w in data["warnings"])

    def test_correlations_validate_with_graph(self, client):
        """Test correlation validation with graph reference."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["demand", "competition"],
                    "correlation": 0.7,
                }
            ],
            "graph": {
                "nodes": [
                    {"id": "demand", "kind": "factor", "label": "Demand"},
                    {"id": "competition", "kind": "factor", "label": "Competition"},
                ],
                "edges": [],
            },
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        # No missing factor warnings
        missing_warnings = [w for w in data["warnings"] if w["code"] == "MISSING_FACTOR_NODES"]
        assert len(missing_warnings) == 0

    def test_correlations_validate_missing_factor_warning(self, client):
        """Test warning when factor not in graph."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["demand", "missing"],
                    "correlation": 0.5,
                }
            ],
            "graph": {
                "nodes": [
                    {"id": "demand", "kind": "factor", "label": "Demand"},
                ],
                "edges": [],
            },
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert any(w["code"] == "MISSING_FACTOR_NODES" for w in data["warnings"])

    def test_correlations_validate_non_factor_node_warning(self, client):
        """Test warning when correlating non-FACTOR nodes."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["demand", "goal"],
                    "correlation": 0.5,
                }
            ],
            "graph": {
                "nodes": [
                    {"id": "demand", "kind": "factor", "label": "Demand"},
                    {"id": "goal", "kind": "goal", "label": "Goal"},
                ],
                "edges": [],
            },
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        assert any(w["code"] == "NON_FACTOR_NODES" for w in data["warnings"])

    def test_correlations_validate_conflicting_correlation_warning(self, client):
        """Test warning for conflicting correlations."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "group1",
                    "factors": ["a", "b"],
                    "correlation": 0.7,
                },
                {
                    "group_id": "group2",
                    "factors": ["a", "b", "c"],
                    "correlation": 0.5,  # Conflicts with group1 for a-b pair
                },
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert any(w["code"] == "CONFLICTING_CORRELATION" for w in data["warnings"])

    def test_correlations_validate_perfect_correlation_issue(self, client):
        """Test handling of perfect correlation (1.0)."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["a", "b"],
                    "correlation": 1.0,
                }
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should have an issue about perfect correlation
        assert len(data["validated_groups"][0]["issues"]) > 0

    def test_correlations_validate_skip_psd_check(self, client):
        """Test skipping PSD check."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["a", "b"],
                    "correlation": 0.5,
                }
            ],
            "check_positive_definite": False,
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True
        # Eigenvalue not computed
        assert data["matrix_analysis"]["min_eigenvalue"] is None

    def test_correlations_validate_request_id_header(self, client):
        """Test request ID header in response metadata."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["a", "b"],
                    "correlation": 0.5,
                }
            ]
        }

        response = client.post(
            "/api/v1/validation/correlations",
            json=request_data,
            headers={"X-Request-Id": "corr-test-456"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metadata"]["request_id"] == "corr-test-456"

    def test_correlations_validate_eigenvalue_analysis(self, client):
        """Test eigenvalue analysis in response."""
        request_data = {
            "correlation_matrix": {
                "factors": ["a", "b"],
                "matrix": [
                    [1.0, 0.5],
                    [0.5, 1.0],
                ],
            },
            "check_positive_definite": True,
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["matrix_analysis"]["is_positive_semi_definite"] is True
        assert data["matrix_analysis"]["min_eigenvalue"] == pytest.approx(0.5, rel=0.01)
        assert data["matrix_analysis"]["condition_number"] is not None

    def test_correlations_validate_zero_correlation(self, client):
        """Test validation with zero correlation (independence)."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "independent",
                    "factors": ["a", "b"],
                    "correlation": 0.0,
                }
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True

    def test_correlations_validate_negative_correlation(self, client):
        """Test validation with negative correlation."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "inverse",
                    "factors": ["supply", "price"],
                    "correlation": -0.6,
                }
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["valid"] is True

    def test_correlations_validate_invalid_request_no_input(self, client):
        """Test error when no groups or matrix provided."""
        request_data = {}

        response = client.post("/api/v1/validation/correlations", json=request_data)

        # Should be validation error (422)
        assert response.status_code == 422

    def test_correlations_validate_matrix_symmetry(self, client):
        """Test that implied matrix is symmetric."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["a", "b", "c"],
                    "correlation": 0.5,
                }
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        matrix = data["implied_matrix"]["matrix"]
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                assert matrix[i][j] == matrix[j][i]

    def test_correlations_validate_diagonal_ones(self, client):
        """Test that diagonal is all 1.0."""
        request_data = {
            "correlation_groups": [
                {
                    "group_id": "test",
                    "factors": ["a", "b", "c"],
                    "correlation": 0.5,
                }
            ]
        }

        response = client.post("/api/v1/validation/correlations", json=request_data)

        assert response.status_code == 200
        data = response.json()

        matrix = data["implied_matrix"]["matrix"]
        n = len(matrix)
        for i in range(n):
            assert matrix[i][i] == 1.0
