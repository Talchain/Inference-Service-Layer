"""
Tests for ISL discovery API (causal discovery).
"""

import pytest
from unittest.mock import patch
import httpx

from isl_client import ISLClient


@pytest.mark.asyncio
async def test_discover_from_data():
    """Test causal discovery from observational data."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "dag": {
            "nodes": ["X", "Y", "Z"],
            "edges": [
                {"from": "X", "to": "Y"},
                {"from": "Y", "to": "Z"}
            ]
        },
        "confidence": 0.85,
        "algorithm": "PC",
        "metadata": {}
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.discovery.from_data(
            data=[
                {"X": 1.0, "Y": 2.0, "Z": 3.0},
                {"X": 1.5, "Y": 2.5, "Z": 3.5},
            ],
            algorithm="PC"
        )

        assert len(result.dag["nodes"]) == 3
        assert len(result.dag["edges"]) == 2
        assert result.confidence > 0.0

    await client.close()


@pytest.mark.asyncio
async def test_discover_with_notears():
    """Test NOTEARS algorithm for discovery."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "dag": {
            "nodes": ["A", "B", "C"],
            "edges": [{"from": "A", "to": "C"}]
        },
        "confidence": 0.90,
        "algorithm": "NOTEARS",
        "metadata": {}
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.discovery.from_data(
            data=[
                {"A": 1, "B": 2, "C": 3},
                {"A": 2, "B": 3, "C": 4},
            ],
            algorithm="NOTEARS"
        )

        assert result.algorithm == "NOTEARS"

    await client.close()


@pytest.mark.asyncio
async def test_extract_factors():
    """Test factor extraction from unstructured text."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "factors": [
            {
                "name": "Price Sensitivity",
                "strength": 0.8,
                "representative_texts": ["price increased", "pricing strategy"],
                "keywords": ["price", "cost", "pricing"],
                "prevalence": 0.6
            },
            {
                "name": "Quality Perception",
                "strength": 0.7,
                "representative_texts": ["product quality", "high quality"],
                "keywords": ["quality", "excellent", "premium"],
                "prevalence": 0.5
            }
        ],
        "suggested_dag": {
            "nodes": ["Price Sensitivity", "Quality Perception", "Revenue"],
            "edges": []
        },
        "confidence": 0.75,
        "method": "sentence-transformers",
        "summary": "Extracted 2 causal factors"
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.discovery.extract_factors(
            texts=[
                "The price increase led to lower sales",
                "Better quality improved customer satisfaction",
            ],
            n_factors=2,
            outcome_variable="Revenue"
        )

        assert len(result.factors) == 2
        assert result.confidence > 0.0
        assert "Price" in result.factors[0].name or "Quality" in result.factors[1].name

    await client.close()


@pytest.mark.asyncio
async def test_discovery_with_constraints():
    """Test discovery with structural constraints."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "dag": {
            "nodes": ["Treatment", "Outcome"],
            "edges": [{"from": "Treatment", "to": "Outcome"}]
        },
        "confidence": 0.95,
        "algorithm": "PC",
        "metadata": {}
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.discovery.from_data(
            data=[
                {"Treatment": 0, "Outcome": 10},
                {"Treatment": 1, "Outcome": 20},
            ],
            algorithm="PC",
            forbidden_edges=[("Outcome", "Treatment")]  # No reverse causation
        )

        # Verify no forbidden edges in result
        for edge in result.dag["edges"]:
            assert not (edge["from"] == "Outcome" and edge["to"] == "Treatment")

    await client.close()


@pytest.mark.asyncio
async def test_discovery_confidence_threshold():
    """Test that discovery returns confidence scores."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "dag": {
            "nodes": ["X", "Y"],
            "edges": [{"from": "X", "to": "Y", "confidence": 0.92}]
        },
        "confidence": 0.88,
        "algorithm": "PC",
        "metadata": {}
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.discovery.from_data(
            data=[{"X": 1, "Y": 2}],
            algorithm="PC",
            min_confidence=0.8
        )

        assert result.confidence >= 0.8

    await client.close()
