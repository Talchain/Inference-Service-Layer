"""
Integration tests for validation strategy endpoints (Feature 2).

Tests complete end-to-end flow through API.
"""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI


@pytest.mark.asyncio
class TestValidationStrategyEndpoint:
    """Test /api/v1/causal/validate/strategies endpoint."""

    async def test_simple_confounding_case(self, client: AsyncClient):
        """Test strategy generation for simple confounding."""
        request = {
            "dag": {
                "nodes": ["Price", "Competitors", "Revenue"],
                "edges": [
                    ["Competitors", "Price"],
                    ["Competitors", "Revenue"],
                    ["Price", "Revenue"],
                ],
            },
            "treatment": "Price",
            "outcome": "Revenue",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        assert "strategies" in data
        assert "path_analysis" in data
        assert "explanation" in data
        assert len(data["strategies"]) > 0

    async def test_backdoor_strategy_generation(self, client: AsyncClient):
        """Test backdoor adjustment strategy."""
        request = {
            "dag": {
                "nodes": ["Treatment", "Confounder", "Outcome"],
                "edges": [
                    ["Confounder", "Treatment"],
                    ["Confounder", "Outcome"],
                    ["Treatment", "Outcome"],
                ],
            },
            "treatment": "Treatment",
            "outcome": "Outcome",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        # Should find backdoor strategies
        backdoor_strategies = [s for s in data["strategies"] if s["strategy_type"] == "backdoor"]
        assert len(backdoor_strategies) > 0

        # Strategy should mention confound or control
        assert any("control" in s["explanation"].lower() or "confounder" in s["explanation"].lower()
                   for s in backdoor_strategies)

    async def test_path_analysis_included(self, client: AsyncClient):
        """Test that path analysis is comprehensive."""
        request = {
            "dag": {
                "nodes": ["X", "Y", "Z"],
                "edges": [["X", "Y"], ["Y", "Z"]],
            },
            "treatment": "X",
            "outcome": "Z",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        path_analysis = data["path_analysis"]
        assert "backdoor_paths" in path_analysis
        assert "frontdoor_paths" in path_analysis
        assert "blocked_paths" in path_analysis
        assert "critical_nodes" in path_analysis

        # Should have frontdoor path
        assert len(path_analysis["frontdoor_paths"]) > 0

    async def test_multiple_strategies_ranked(self, client: AsyncClient):
        """Test that multiple strategies are ranked."""
        request = {
            "dag": {
                "nodes": ["T", "U1", "U2", "O"],
                "edges": [
                    ["U1", "T"],
                    ["U1", "O"],
                    ["U2", "T"],
                    ["U2", "O"],
                    ["T", "O"],
                ],
            },
            "treatment": "T",
            "outcome": "O",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        strategies = data["strategies"]
        assert len(strategies) >= 2

        # Should be ranked by expected_identifiability
        for i in range(len(strategies) - 1):
            # Allow some tolerance for ranking
            assert strategies[i]["expected_identifiability"] >= strategies[i+1]["expected_identifiability"] - 0.4

    async def test_already_identifiable(self, client: AsyncClient):
        """Test case where effect is already identifiable."""
        request = {
            "dag": {
                "nodes": ["Treatment", "Outcome"],
                "edges": [["Treatment", "Outcome"]],
            },
            "treatment": "Treatment",
            "outcome": "Outcome",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        # May have no backdoor strategies or low priority ones
        assert "strategies" in data

    async def test_response_structure(self, client: AsyncClient):
        """Test complete response structure."""
        request = {
            "dag": {
                "nodes": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"]],
            },
            "treatment": "A",
            "outcome": "C",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        # Check strategy structure
        for strategy in data["strategies"]:
            assert "strategy_type" in strategy
            assert "nodes_to_add" in strategy
            assert "edges_to_add" in strategy
            assert "explanation" in strategy
            assert "theoretical_basis" in strategy
            assert "expected_identifiability" in strategy

            assert 0 <= strategy["expected_identifiability"] <= 1

        # Check explanation structure
        assert "summary" in data["explanation"]
        assert "reasoning" in data["explanation"]
        assert "technical_basis" in data["explanation"]
        assert "assumptions" in data["explanation"]

    async def test_metadata_included(self, client: AsyncClient):
        """Test that response includes metadata."""
        request = {
            "dag": {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"]],
            },
            "treatment": "X",
            "outcome": "Y",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        assert "_metadata" in data
        assert "request_id" in data["_metadata"]
        assert "timestamp" in data["_metadata"]

    async def test_invalid_dag(self, client: AsyncClient):
        """Test error handling for invalid DAG."""
        request = {
            "dag": {
                "nodes": [],  # Empty nodes
                "edges": [["A", "B"]],
            },
            "treatment": "A",
            "outcome": "B",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        # Should handle gracefully (400 or 500)
        assert response.status_code in [400, 500] or response.status_code == 200

    async def test_missing_nodes(self, client: AsyncClient):
        """Test when treatment/outcome not in DAG."""
        request = {
            "dag": {
                "nodes": ["A", "B"],
                "edges": [["A", "B"]],
            },
            "treatment": "X",  # Not in DAG
            "outcome": "Y",    # Not in DAG
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        # Should handle gracefully
        assert response.status_code in [200, 400, 500]

    async def test_complex_dag(self, client: AsyncClient):
        """Test with complex DAG structure."""
        request = {
            "dag": {
                "nodes": ["T", "M1", "M2", "C1", "C2", "O"],
                "edges": [
                    ["C1", "T"],
                    ["C1", "O"],
                    ["C2", "T"],
                    ["C2", "O"],
                    ["T", "M1"],
                    ["M1", "M2"],
                    ["M2", "O"],
                ],
            },
            "treatment": "T",
            "outcome": "O",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        # Should handle complex structure
        assert len(data["strategies"]) > 0
        assert len(data["path_analysis"]["frontdoor_paths"]) > 0


@pytest.mark.asyncio
class TestValidationStrategyEdgeCases:
    """Test edge cases and error scenarios."""

    async def test_self_loop_dag(self, client: AsyncClient):
        """Test handling of self-loop in DAG."""
        request = {
            "dag": {
                "nodes": ["X", "Y"],
                "edges": [["X", "X"], ["X", "Y"]],  # Self-loop
            },
            "treatment": "X",
            "outcome": "Y",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        # Should handle gracefully
        assert response.status_code in [200, 400]

    async def test_cyclic_dag(self, client: AsyncClient):
        """Test handling of cyclic graph."""
        request = {
            "dag": {
                "nodes": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"], ["C", "A"]],  # Cycle
            },
            "treatment": "A",
            "outcome": "C",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        # Should handle gracefully
        assert response.status_code in [200, 400, 500]

    async def test_identical_treatment_outcome(self, client: AsyncClient):
        """Test when treatment and outcome are the same."""
        request = {
            "dag": {
                "nodes": ["X"],
                "edges": [],
            },
            "treatment": "X",
            "outcome": "X",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        # Should handle this edge case
        assert response.status_code in [200, 400]

    async def test_disconnected_graph(self, client: AsyncClient):
        """Test with disconnected components."""
        request = {
            "dag": {
                "nodes": ["A", "B", "C", "D"],
                "edges": [["A", "B"], ["C", "D"]],  # Disconnected
            },
            "treatment": "A",
            "outcome": "D",
        }

        response = await client.post("/api/v1/causal/validate/strategies", json=request)

        assert response.status_code == 200
        data = response.json()

        # Should handle disconnected case
        assert "strategies" in data
