"""Tests for synchronous wrapper."""

import pytest
from unittest.mock import MagicMock, patch

from isl_client import ISLClientSync
from isl_client.models import ValidationResponse


def test_sync_client_context_manager():
    """Test sync client works as context manager."""
    with ISLClientSync("http://localhost:8000") as client:
        assert client._async_client is not None
        assert client.causal is not None


def test_sync_validate():
    """Test synchronous validate method."""
    with ISLClientSync("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "identifiable",
            "method": "backdoor",
            "adjustment_sets": [["Z"]],
            "suggestions": None,
            "explanation": {
                "summary": "Effect is identifiable",
                "assumptions": [],
                "caveats": [],
                "confidence": 0.95,
            },
        }

        with patch.object(client._async_client, "post", return_value=mock_response):
            result = client.causal.validate(
                dag={"nodes": ["X", "Y", "Z"], "edges": [["X", "Y"]]},
                treatment="X",
                outcome="Y",
            )

            assert isinstance(result, ValidationResponse)
            assert result.status == "identifiable"


def test_sync_counterfactual():
    """Test synchronous counterfactual method."""
    with ISLClientSync("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "intervention": {"Price": 45.0},
            "prediction": {
                "prediction": {"Revenue": 1250.0},
                "uncertainty": None,
                "explanation": "Prediction complete",
            },
            "model_assumptions": [],
            "explanation": {
                "summary": "Counterfactual computed",
                "assumptions": [],
                "caveats": [],
                "confidence": 0.85,
            },
        }

        with patch.object(client._async_client, "post", return_value=mock_response):
            result = client.causal.counterfactual(
                model={"equations": {}},
                intervention={"Price": 45.0},
            )

            assert result.intervention["Price"] == 45.0
            assert result.prediction.prediction["Revenue"] == 1250.0


def test_sync_health():
    """Test synchronous health check."""
    with ISLClientSync("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2025-11-23T10:00:00Z",
        }

        with patch.object(client._async_client, "get", return_value=mock_response):
            result = client.health()

            assert result["status"] == "healthy"


def test_sync_client_initialization():
    """Test sync client initializes correctly."""
    client = ISLClientSync(
        base_url="http://localhost:8000",
        api_key="test_key",
        timeout=30.0,
        max_retries=3,
    )

    assert client._async_client.base_url == "http://localhost:8000"
    assert client._async_client.api_key == "test_key"
    assert client._async_client.max_retries == 3

    client.close()
