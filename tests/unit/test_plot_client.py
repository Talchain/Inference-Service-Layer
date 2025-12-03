"""
Unit tests for PLoT Engine client.

Tests client behavior, parsing, error handling, and logging.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.clients.plot_engine_client import PLoTEngineClient
from src.models.plot_engine import (
    ChangeAttribution,
    ChangeDriver,
    CompareOption,
    CompareRequest,
    CompareResponse,
    EvidenceFreshness,
    IdempotencyMismatchError,
    ModelCard,
    RunRequest,
    RunResponse,
)


@pytest.fixture
def plot_client():
    """Create PLoT client for testing."""
    return PLoTEngineClient(
        base_url="https://plot.test.olumi.ai",
        api_key="test_key_123",
        timeout=30,
        max_retries=3
    )


def test_parses_evidence_freshness():
    """Test that EvidenceFreshness is correctly parsed."""
    data = {
        "total": 100,
        "with_timestamp": 85,
        "oldest_days": 365,
        "newest_days": 7,
        "buckets": {
            "FRESH": 40,
            "AGING": 30,
            "STALE": 15,
            "UNKNOWN": 15
        }
    }

    evidence = EvidenceFreshness(**data)

    assert evidence.total == 100
    assert evidence.with_timestamp == 85
    assert evidence.oldest_days == 365
    assert evidence.newest_days == 7
    assert evidence.buckets["FRESH"] == 40
    assert evidence.buckets["STALE"] == 15


def test_evidence_freshness_validates_data():
    """Test that EvidenceFreshness validates input data."""
    # Negative total should fail
    with pytest.raises(ValueError, match="total must be non-negative"):
        EvidenceFreshness(total=-1, with_timestamp=0)

    # with_timestamp > total should fail
    with pytest.raises(ValueError, match="with_timestamp must be between 0 and total"):
        EvidenceFreshness(total=50, with_timestamp=60)


def test_parses_change_attribution():
    """Test that ChangeAttribution is correctly parsed."""
    data = {
        "outcome_delta": 25.5,
        "primary_drivers": [
            {
                "change_type": "structural",
                "description": "Increased marketing investment",
                "contribution_to_delta": 15.0,
                "contribution_pct": 60.0,
                "affected_nodes": [{"id": "n_marketing", "label": "Marketing"}]
            },
            {
                "change_type": "parametric",
                "description": "Market conditions improved",
                "contribution_to_delta": 10.5,
                "contribution_pct": 40.0,
                "affected_nodes": [{"id": "n_market", "label": "Market"}]
            }
        ],
        "summary": "Outcome increased by 25.5 due to marketing and market conditions"
    }

    attribution = ChangeAttribution(
        outcome_delta=data["outcome_delta"],
        primary_drivers=[
            ChangeDriver(**driver) for driver in data["primary_drivers"]
        ],
        summary=data["summary"]
    )

    assert attribution.outcome_delta == 25.5
    assert len(attribution.primary_drivers) == 2
    assert attribution.primary_drivers[0].contribution_pct == 60.0
    assert attribution.primary_drivers[1].change_type == "parametric"
    assert attribution.summary == "Outcome increased by 25.5 due to marketing and market conditions"


def test_change_driver_validates_percentage():
    """Test that ChangeDriver validates contribution percentage."""
    # Valid percentage
    driver = ChangeDriver(
        change_type="structural",
        description="Test",
        contribution_to_delta=10.0,
        contribution_pct=50.0
    )
    assert driver.contribution_pct == 50.0

    # Invalid percentage > 100
    with pytest.raises(ValueError, match="contribution_pct must be between 0 and 100"):
        ChangeDriver(
            change_type="structural",
            description="Test",
            contribution_to_delta=10.0,
            contribution_pct=150.0
        )

    # Invalid negative percentage
    with pytest.raises(ValueError, match="contribution_pct must be between 0 and 100"):
        ChangeDriver(
            change_type="structural",
            description="Test",
            contribution_to_delta=10.0,
            contribution_pct=-10.0
        )


@pytest.mark.asyncio
async def test_handles_409_idempotency_mismatch(plot_client):
    """Test that 409 idempotency mismatch is handled correctly and not retried."""
    # Mock HTTP client to return 409
    mock_response = MagicMock()
    mock_response.status_code = 409
    mock_response.json.return_value = {
        "code": "IDEMPOTENCY_MISMATCH",
        "message": "Request with different body for same idempotency key"
    }

    plot_client.client.post = AsyncMock(return_value=mock_response)

    # Should raise IdempotencyMismatchError
    request = RunRequest(
        graph={"nodes": [], "edges": []},
        idempotency_key="test_key_123"
    )

    with pytest.raises(IdempotencyMismatchError) as exc_info:
        await plot_client.run(request)

    assert "idempotency key" in str(exc_info.value).lower()
    assert exc_info.value.idempotency_key == "test_key_123"

    # Verify only called once (no retries)
    assert plot_client.client.post.call_count == 1


@pytest.mark.asyncio
async def test_retries_network_errors_with_backoff(plot_client):
    """Test that network errors are retried with exponential backoff."""
    import httpx

    # Mock to fail first 2 times, then succeed
    mock_responses = [
        AsyncMock(side_effect=httpx.NetworkError("Connection failed")),
        AsyncMock(side_effect=httpx.NetworkError("Connection failed")),
        AsyncMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {
                "run_id": "test_run_123",
                "status": "completed",
                "model_card": {}
            }
        ))
    ]

    plot_client.client.post = AsyncMock(side_effect=mock_responses)

    request = RunRequest(graph={"nodes": [], "edges": []})

    # Should succeed after retries
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        response = await plot_client.run(request)

    assert response.run_id == "test_run_123"
    assert plot_client.client.post.call_count == 3

    # Verify exponential backoff
    assert mock_sleep.call_count == 2
    # First retry: 2^1 = 2 seconds
    assert mock_sleep.call_args_list[0][0][0] == 2
    # Second retry: 2^2 = 4 seconds
    assert mock_sleep.call_args_list[1][0][0] == 4


@pytest.mark.asyncio
async def test_logs_evidence_quality(plot_client):
    """Test that evidence quality metrics are logged."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "run_id": "test_run_123",
        "status": "completed",
        "model_card": {
            "evidence_freshness": {
                "total": 100,
                "with_timestamp": 80,
                "oldest_days": 180,
                "newest_days": 5,
                "buckets": {"FRESH": 50, "AGING": 30, "STALE": 20}
            }
        }
    }

    plot_client.client.post = AsyncMock(return_value=mock_response)

    request = RunRequest(graph={"nodes": [], "edges": []})

    with patch("src.clients.plot_engine_client.logger") as mock_logger:
        await plot_client.run(request)

        # Verify evidence quality was logged
        log_calls = [call for call in mock_logger.info.call_args_list
                     if "plot_evidence_quality" in str(call)]
        assert len(log_calls) > 0


@pytest.mark.asyncio
async def test_logs_change_attribution(plot_client):
    """Test that change attribution is logged for compare requests."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "compare_id": "test_compare_123",
        "status": "completed",
        "options": [
            {
                "option_id": "opt1",
                "label": "Option 1",
                "outcome_value": 100.0,
                "change_attribution": {
                    "outcome_delta": 25.0,
                    "primary_drivers": [
                        {
                            "change_type": "structural",
                            "description": "Increased budget",
                            "contribution_to_delta": 25.0,
                            "contribution_pct": 100.0
                        }
                    ],
                    "summary": "Outcome increased by 25 due to budget increase"
                }
            }
        ],
        "model_card": {}
    }

    plot_client.client.post = AsyncMock(return_value=mock_response)

    request = CompareRequest(
        graph={"nodes": [], "edges": []},
        scenarios=[{"name": "base"}, {"name": "alt"}]
    )

    with patch("src.clients.plot_engine_client.logger") as mock_logger:
        response = await plot_client.compare(request)

        # Verify attribution was logged
        log_calls = [call for call in mock_logger.info.call_args_list
                     if "plot_change_attribution" in str(call)]
        assert len(log_calls) > 0

        # Verify response contains attribution
        assert len(response.options) == 1
        assert response.options[0].change_attribution is not None
        assert response.options[0].change_attribution.outcome_delta == 25.0


@pytest.mark.asyncio
async def test_handles_optional_fields_safely(plot_client):
    """Test that optional fields are handled safely when missing."""
    # Response with minimal data (no optional fields)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "run_id": "test_run_123",
        "status": "completed",
        "model_card": {}  # No evidence_freshness
    }

    plot_client.client.post = AsyncMock(return_value=mock_response)

    request = RunRequest(graph={"nodes": [], "edges": []})
    response = await plot_client.run(request)

    # Should not crash, evidence_freshness should be None
    assert response.run_id == "test_run_123"
    assert response.model_card.evidence_freshness is None


@pytest.mark.asyncio
async def test_compare_handles_optional_attribution(plot_client):
    """Test that compare handles options without change attribution."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "compare_id": "test_compare_123",
        "status": "completed",
        "options": [
            {
                "option_id": "opt1",
                "label": "Option 1",
                "outcome_value": 100.0
                # No change_attribution
            }
        ],
        "model_card": {}
    }

    plot_client.client.post = AsyncMock(return_value=mock_response)

    request = CompareRequest(
        graph={"nodes": [], "edges": []},
        scenarios=[{"name": "base"}]
    )

    response = await plot_client.compare(request)

    # Should not crash, attribution should be None
    assert len(response.options) == 1
    assert response.options[0].change_attribution is None


@pytest.mark.asyncio
async def test_client_context_manager(plot_client):
    """Test that client works as async context manager."""
    async with PLoTEngineClient("https://test.olumi.ai") as client:
        assert client.client is not None

    # Client should be closed after context
    assert client.client.is_closed
