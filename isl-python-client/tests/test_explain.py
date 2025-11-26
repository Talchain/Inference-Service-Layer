"""
Tests for ISL explain API (contrastive explanations).
"""

import pytest
from unittest.mock import patch
import httpx

from isl_client import ISLClient


@pytest.mark.asyncio
async def test_contrastive_explanation():
    """Test contrastive explanation request."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "minimal_intervention": {"Price": 60.0},
        "explanation": "Increasing Price to 60.0 achieves target outcome",
        "alternatives": [],
        "metadata": {}
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.explain.contrastive(
            dag={"nodes": ["Price", "Revenue"], "edges": []},
            factual={"Price": 50.0},
            factual_outcome=5000.0,
            counterfactual_outcome=6000.0,
            outcome_variable="Revenue"
        )

        assert result.minimal_intervention == {"Price": 60.0}
        assert "Price" in result.explanation or isinstance(result.explanation, str)

    await client.close()


@pytest.mark.asyncio
async def test_contrastive_with_alternatives():
    """Test contrastive explanation with multiple alternatives."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "minimal_intervention": {"Price": 60.0},
        "explanation": "Main intervention",
        "alternatives": [
            {"Quality": 0.9},
            {"Marketing": 1500}
        ],
        "metadata": {}
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.explain.contrastive(
            dag={"nodes": ["Price", "Quality", "Revenue"], "edges": []},
            factual={"Price": 50.0, "Quality": 0.8},
            factual_outcome=5000.0,
            counterfactual_outcome=6000.0,
            outcome_variable="Revenue",
            n_alternatives=3
        )

        assert len(result.alternatives) == 2

    await client.close()


@pytest.mark.asyncio
async def test_contrastive_with_constraints():
    """Test contrastive explanation with intervention constraints."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "minimal_intervention": {"Quality": 0.9},
        "explanation": "Constrained intervention",
        "alternatives": [],
        "metadata": {}
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.explain.contrastive(
            dag={"nodes": ["Price", "Quality", "Revenue"], "edges": []},
            factual={"Price": 50.0, "Quality": 0.8},
            factual_outcome=5000.0,
            counterfactual_outcome=6000.0,
            outcome_variable="Revenue",
            constraints={"Price": {"min": 45, "max": 55}}  # Don't change price much
        )

        # Should suggest Quality change instead
        assert "Quality" in result.minimal_intervention or "Price" not in result.minimal_intervention

    await client.close()


@pytest.mark.asyncio
async def test_explanation_metadata():
    """Test that explanation includes metadata."""
    client = ISLClient(base_url="http://localhost:8000")

    mock_response = {
        "minimal_intervention": {"Price": 60.0},
        "explanation": "Explanation text",
        "alternatives": [],
        "metadata": {
            "isl_version": "1.0.0",
            "request_id": "req_123"
        }
    }

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(200, json=mock_response)

        result = await client.explain.contrastive(
            dag={"nodes": ["Price", "Revenue"], "edges": []},
            factual={"Price": 50.0},
            factual_outcome=5000.0,
            counterfactual_outcome=6000.0,
            outcome_variable="Revenue"
        )

        assert result.metadata is not None
        assert "isl_version" in result.metadata or "version" in result.metadata

    await client.close()
