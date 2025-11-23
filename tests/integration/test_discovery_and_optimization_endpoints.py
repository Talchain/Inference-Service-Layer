"""
Integration tests for causal discovery and sequential optimization endpoints (Features 3 & 4).

Tests complete end-to-end flow through API.
"""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI


@pytest.mark.asyncio
class TestDiscoveryFromDataEndpoint:
    """Test /api/v1/causal/discover/from-data endpoint (Feature 3)."""

    async def test_simple_data_discovery(self, client: AsyncClient):
        """Test discovery from simple dataset."""
        request = {
            "data": [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [3.0, 6.0, 9.0],
                [4.0, 8.0, 12.0],
                [5.0, 10.0, 15.0],
                [6.0, 12.0, 18.0],
                [7.0, 14.0, 21.0],
                [8.0, 16.0, 24.0],
                [9.0, 18.0, 27.0],
                [10.0, 20.0, 30.0],
            ],
            "variable_names": ["X", "Y", "Z"],
            "threshold": 0.5,
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/discover/from-data", json=request)

        assert response.status_code == 200
        data = response.json()

        assert "discovered_dags" in data
        assert "explanation" in data
        assert len(data["discovered_dags"]) > 0

        # Check DAG structure
        dag = data["discovered_dags"][0]
        assert "nodes" in dag
        assert "edges" in dag
        assert "confidence" in dag
        assert "method" in dag
        assert dag["method"] == "correlation"

    async def test_prior_knowledge_enforced(self, client: AsyncClient):
        """Test that prior knowledge is enforced."""
        request = {
            "data": [[1.0, 2.0] for _ in range(20)],
            "variable_names": ["A", "B"],
            "prior_knowledge": {
                "required_edges": [["A", "B"]]
            },
            "threshold": 0.3,
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/discover/from-data", json=request)

        assert response.status_code == 200
        data = response.json()

        dag = data["discovered_dags"][0]
        # Required edge should be present
        assert ["A", "B"] in dag["edges"]

    async def test_confidence_scoring(self, client: AsyncClient):
        """Test confidence score is provided."""
        request = {
            "data": [[float(i), float(i*2)] for i in range(50)],
            "variable_names": ["X", "Y"],
            "threshold": 0.3,
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/discover/from-data", json=request)

        assert response.status_code == 200
        data = response.json()

        dag = data["discovered_dags"][0]
        assert 0 <= dag["confidence"] <= 1

    async def test_determinism(self, client: AsyncClient):
        """Test deterministic results with same seed."""
        request = {
            "data": [[float(i), float(i+1)] for i in range(30)],
            "variable_names": ["X", "Y"],
            "threshold": 0.3,
            "seed": 42,
        }

        response1 = await client.post("/api/v1/causal/discover/from-data", json=request)
        response2 = await client.post("/api/v1/causal/discover/from-data", json=request)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Same seed should produce same results
        assert data1["discovered_dags"][0]["edges"] == data2["discovered_dags"][0]["edges"]

    async def test_insufficient_data(self, client: AsyncClient):
        """Test error handling with insufficient data."""
        request = {
            "data": [[1.0, 2.0]],  # Only 1 sample (need min 10)
            "variable_names": ["X", "Y"],
            "threshold": 0.3,
        }

        response = await client.post("/api/v1/causal/discover/from-data", json=request)

        # Should return 400 (validation error)
        assert response.status_code == 400

    async def test_metadata_included(self, client: AsyncClient):
        """Test metadata in response."""
        request = {
            "data": [[float(i), float(i)] for i in range(20)],
            "variable_names": ["A", "B"],
            "threshold": 0.3,
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/discover/from-data", json=request)

        assert response.status_code == 200
        data = response.json()

        assert "_metadata" in data
        assert "request_id" in data["_metadata"]

    async def test_explanation_structure(self, client: AsyncClient):
        """Test explanation structure."""
        request = {
            "data": [[float(i), float(i*2)] for i in range(25)],
            "variable_names": ["X", "Y"],
            "threshold": 0.3,
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/discover/from-data", json=request)

        assert response.status_code == 200
        data = response.json()

        explanation = data["explanation"]
        assert "summary" in explanation
        assert "reasoning" in explanation
        assert "technical_basis" in explanation
        assert "assumptions" in explanation


@pytest.mark.asyncio
class TestDiscoveryFromKnowledgeEndpoint:
    """Test /api/v1/causal/discover/from-knowledge endpoint (Feature 3)."""

    async def test_simple_knowledge_discovery(self, client: AsyncClient):
        """Test discovery from domain knowledge."""
        request = {
            "domain_description": "Price affects revenue, quality affects both price and revenue",
            "variable_names": ["Price", "Quality", "Revenue"],
            "top_k": 3,
        }

        response = await client.post("/api/v1/causal/discover/from-knowledge", json=request)

        assert response.status_code == 200
        data = response.json()

        assert "discovered_dags" in data
        assert len(data["discovered_dags"]) > 0
        assert len(data["discovered_dags"]) <= 3

    async def test_top_k_parameter(self, client: AsyncClient):
        """Test top_k limits results."""
        request = {
            "domain_description": "Variables interact in complex ways",
            "variable_names": ["A", "B", "C"],
            "top_k": 1,
        }

        response = await client.post("/api/v1/causal/discover/from-knowledge", json=request)

        assert response.status_code == 200
        data = response.json()

        assert len(data["discovered_dags"]) <= 1

    async def test_prior_knowledge_applied(self, client: AsyncClient):
        """Test prior knowledge constraints."""
        request = {
            "domain_description": "Simple model",
            "variable_names": ["X", "Y", "Z"],
            "prior_knowledge": {
                "required_edges": [["X", "Y"]]
            },
            "top_k": 3,
        }

        response = await client.post("/api/v1/causal/discover/from-knowledge", json=request)

        assert response.status_code == 200
        data = response.json()

        # At least one DAG should have required edge
        has_required = any(
            ["X", "Y"] in dag["edges"]
            for dag in data["discovered_dags"]
        )
        assert has_required

    async def test_dag_structure(self, client: AsyncClient):
        """Test DAG structure in response."""
        request = {
            "domain_description": "Marketing affects sales",
            "variable_names": ["Marketing", "Sales"],
            "top_k": 2,
        }

        response = await client.post("/api/v1/causal/discover/from-knowledge", json=request)

        assert response.status_code == 200
        data = response.json()

        for dag in data["discovered_dags"]:
            assert "nodes" in dag
            assert "edges" in dag
            assert "confidence" in dag
            assert "method" in dag
            assert dag["method"] == "knowledge"
            assert 0 <= dag["confidence"] <= 1

    async def test_minimal_description(self, client: AsyncClient):
        """Test with minimal description."""
        request = {
            "domain_description": "variables",
            "variable_names": ["A", "B"],
        }

        response = await client.post("/api/v1/causal/discover/from-knowledge", json=request)

        assert response.status_code == 200
        data = response.json()

        assert len(data["discovered_dags"]) > 0


@pytest.mark.asyncio
class TestExperimentRecommendationEndpoint:
    """Test /api/v1/causal/experiment/recommend endpoint (Feature 4)."""

    async def test_simple_recommendation(self, client: AsyncClient):
        """Test basic experiment recommendation."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "effect_price",
                    "distribution_type": "normal",
                    "parameters": {"mean": 500, "std": 50},
                }
            ],
            "objective": {
                "target_variable": "Revenue",
                "goal": "maximize",
            },
            "constraints": {
                "budget": 100000,
                "time_horizon": 10,
                "feasible_interventions": {
                    "Price": [30, 100],
                },
            },
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response.status_code == 200
        data = response.json()

        assert "recommendation" in data
        assert "explanation" in data

        rec = data["recommendation"]
        assert "intervention" in rec
        assert "expected_outcome" in rec
        assert "expected_information_gain" in rec
        assert "cost_estimate" in rec
        assert "rationale" in rec
        assert "exploration_vs_exploitation" in rec

    async def test_feasibility_constraints(self, client: AsyncClient):
        """Test that recommendations respect feasibility constraints."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "effect_x",
                    "distribution_type": "normal",
                    "parameters": {"mean": 1, "std": 0.1},
                }
            ],
            "objective": {
                "target_variable": "Y",
                "goal": "maximize",
            },
            "constraints": {
                "budget": 50000,
                "time_horizon": 5,
                "feasible_interventions": {
                    "X": [0, 10],
                },
            },
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response.status_code == 200
        data = response.json()

        intervention = data["recommendation"]["intervention"]
        assert "X" in intervention
        assert 0 <= intervention["X"] <= 10

    async def test_with_history(self, client: AsyncClient):
        """Test recommendation with experimental history."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "effect_price",
                    "distribution_type": "normal",
                    "parameters": {"mean": 500, "std": 50},
                }
            ],
            "objective": {
                "target_variable": "Revenue",
                "goal": "maximize",
            },
            "constraints": {
                "budget": 100000,
                "time_horizon": 5,
                "feasible_interventions": {
                    "Price": [30, 100],
                },
            },
            "history": [
                {
                    "intervention": {"Price": 45},
                    "outcome": {"Revenue": 32000},
                    "cost": 5000,
                },
                {
                    "intervention": {"Price": 50},
                    "outcome": {"Revenue": 35000},
                    "cost": 5000,
                },
            ],
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response.status_code == 200
        data = response.json()

        # Should still produce valid recommendation
        assert "recommendation" in data

    async def test_determinism(self, client: AsyncClient):
        """Test deterministic recommendations with same seed."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "effect_x",
                    "distribution_type": "uniform",
                    "parameters": {"low": 0, "high": 5},
                }
            ],
            "objective": {
                "target_variable": "Y",
                "goal": "maximize",
            },
            "constraints": {
                "budget": 10000,
                "time_horizon": 5,
                "feasible_interventions": {
                    "X": [0, 10],
                },
            },
            "seed": 42,
        }

        response1 = await client.post("/api/v1/causal/experiment/recommend", json=request)
        response2 = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Same seed should produce same recommendation
        assert data1["recommendation"]["intervention"] == data2["recommendation"]["intervention"]

    async def test_information_gain_range(self, client: AsyncClient):
        """Test information gain is in valid range."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "param1",
                    "distribution_type": "normal",
                    "parameters": {"mean": 1, "std": 0.5},
                }
            ],
            "objective": {
                "target_variable": "Y",
                "goal": "maximize",
            },
            "constraints": {
                "budget": 10000,
                "time_horizon": 10,
                "feasible_interventions": {
                    "X": [0, 10],
                },
            },
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response.status_code == 200
        data = response.json()

        info_gain = data["recommendation"]["expected_information_gain"]
        assert 0 <= info_gain <= 1

    async def test_exploration_exploitation_range(self, client: AsyncClient):
        """Test exploration/exploitation score is in valid range."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "param1",
                    "distribution_type": "normal",
                    "parameters": {"mean": 2, "std": 0.2},
                }
            ],
            "objective": {
                "target_variable": "Y",
                "goal": "minimize",
            },
            "constraints": {
                "budget": 10000,
                "time_horizon": 10,
                "feasible_interventions": {
                    "X": [0, 10],
                },
            },
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response.status_code == 200
        data = response.json()

        exploration_score = data["recommendation"]["exploration_vs_exploitation"]
        assert 0 <= exploration_score <= 1

    async def test_multiple_beliefs(self, client: AsyncClient):
        """Test with multiple parameter beliefs."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "effect_price",
                    "distribution_type": "normal",
                    "parameters": {"mean": 500, "std": 50},
                },
                {
                    "parameter_name": "effect_quality",
                    "distribution_type": "uniform",
                    "parameters": {"low": 100, "high": 300},
                },
            ],
            "objective": {
                "target_variable": "Revenue",
                "goal": "maximize",
            },
            "constraints": {
                "budget": 100000,
                "time_horizon": 10,
                "feasible_interventions": {
                    "Price": [30, 100],
                    "Quality": [5, 10],
                },
            },
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response.status_code == 200
        data = response.json()

        # Should handle multiple parameters
        assert "recommendation" in data

    async def test_metadata_included(self, client: AsyncClient):
        """Test metadata in response."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "effect_x",
                    "distribution_type": "normal",
                    "parameters": {"mean": 1, "std": 0.1},
                }
            ],
            "objective": {
                "target_variable": "Y",
                "goal": "maximize",
            },
            "constraints": {
                "budget": 10000,
                "time_horizon": 5,
                "feasible_interventions": {
                    "X": [0, 10],
                },
            },
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response.status_code == 200
        data = response.json()

        assert "_metadata" in data
        assert "request_id" in data["_metadata"]

    async def test_explanation_structure(self, client: AsyncClient):
        """Test explanation structure."""
        request = {
            "beliefs": [
                {
                    "parameter_name": "effect_x",
                    "distribution_type": "normal",
                    "parameters": {"mean": 1, "std": 0.1},
                }
            ],
            "objective": {
                "target_variable": "Y",
                "goal": "maximize",
            },
            "constraints": {
                "budget": 10000,
                "time_horizon": 5,
                "feasible_interventions": {
                    "X": [0, 10],
                },
            },
            "seed": 42,
        }

        response = await client.post("/api/v1/causal/experiment/recommend", json=request)

        assert response.status_code == 200
        data = response.json()

        explanation = data["explanation"]
        assert "summary" in explanation
        assert "reasoning" in explanation
        assert "technical_basis" in explanation
        assert "assumptions" in explanation
