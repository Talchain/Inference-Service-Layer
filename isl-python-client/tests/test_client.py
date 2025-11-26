"""Tests for core ISL client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from isl_client import ISLClient
from isl_client.exceptions import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailable,
    TimeoutError,
    ValidationError,
)


@pytest.mark.asyncio
async def test_client_initialization():
    """Test client initializes correctly."""
    client = ISLClient(
        base_url="http://localhost:8000",
        api_key="test_key",
        timeout=30.0,
        max_retries=3,
    )

    assert client.base_url == "http://localhost:8000"
    assert client.api_key == "test_key"
    assert client.max_retries == 3

    await client.close()


@pytest.mark.asyncio
async def test_client_context_manager():
    """Test client works as async context manager."""
    async with ISLClient("http://localhost:8000") as client:
        assert client._client is not None

    # Client should be closed after context exit
    # (we can't directly test this without implementation details)


@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    async with ISLClient("http://localhost:8000") as client:
        # Mock the response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2025-11-23T10:00:00Z",
        }

        with patch.object(client, "get", return_value=mock_response):
            result = await client.health()

            assert result["status"] == "healthy"
            assert result["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_authentication_error():
    """Test 401 raises AuthenticationError."""
    async with ISLClient("http://localhost:8000") as client:
        # Mock 401 response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401

        with patch.object(client._client, "request", return_value=mock_response):
            with pytest.raises(AuthenticationError, match="Authentication failed"):
                await client.get("/test")


@pytest.mark.asyncio
async def test_not_found_error():
    """Test 404 raises NotFoundError."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404

        with patch.object(client._client, "request", return_value=mock_response):
            with pytest.raises(NotFoundError, match="Endpoint not found"):
                await client.get("/nonexistent")


@pytest.mark.asyncio
async def test_rate_limit_error():
    """Test 429 raises RateLimitError."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}

        with patch.object(client._client, "request", return_value=mock_response):
            with pytest.raises(RateLimitError) as exc_info:
                await client.get("/test")

            assert exc_info.value.retry_after == 60


@pytest.mark.asyncio
async def test_validation_error():
    """Test 400 raises ValidationError."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "message": "Invalid parameters",
            "details": {"field": "treatment", "error": "required"},
        }

        with patch.object(client._client, "request", return_value=mock_response):
            with pytest.raises(ValidationError) as exc_info:
                await client.post("/test", json={})

            assert "Invalid parameters" in str(exc_info.value)
            assert exc_info.value.details["field"] == "treatment"


@pytest.mark.asyncio
async def test_retry_on_500():
    """Test retry logic on 500 errors."""
    async with ISLClient("http://localhost:8000", max_retries=3) as client:
        # Mock responses: 2 failures, then success
        responses = [
            MagicMock(spec=httpx.Response, status_code=500),
            MagicMock(spec=httpx.Response, status_code=500),
            MagicMock(spec=httpx.Response, status_code=200, json=lambda: {"result": "ok"}),
        ]

        call_count = [0]

        async def mock_request(*args, **kwargs):
            result = responses[call_count[0]]
            call_count[0] += 1
            return result

        with patch.object(client._client, "request", side_effect=mock_request):
            with patch("asyncio.sleep", new_callable=AsyncMock):  # Don't actually sleep
                response = await client.get("/test")

                # Should have retried and succeeded
                assert call_count[0] == 3
                assert response.status_code == 200


@pytest.mark.asyncio
async def test_retry_exhaustion():
    """Test ServiceUnavailable after retries exhausted."""
    async with ISLClient("http://localhost:8000", max_retries=2) as client:
        # Always return 500
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500

        with patch.object(client._client, "request", return_value=mock_response):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ServiceUnavailable, match="after 2 retries"):
                    await client.get("/test")


@pytest.mark.asyncio
async def test_timeout_error():
    """Test timeout raises TimeoutError."""
    async with ISLClient("http://localhost:8000", max_retries=2) as client:
        async def mock_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Request timed out")

        with patch.object(client._client, "request", side_effect=mock_timeout):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(TimeoutError, match="timed out after 2 attempts"):
                    await client.get("/test")


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test exponential backoff delays."""
    async with ISLClient("http://localhost:8000", max_retries=3, retry_backoff_factor=2.0) as client:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500

        sleep_delays = []

        async def mock_sleep(delay):
            sleep_delays.append(delay)

        with patch.object(client._client, "request", return_value=mock_response):
            with patch("asyncio.sleep", side_effect=mock_sleep):
                with pytest.raises(ServiceUnavailable):
                    await client.get("/test")

                # Should have exponential delays: 2^0=1, 2^1=2
                assert len(sleep_delays) == 2
                assert sleep_delays[0] == 1.0  # 2.0^0
                assert sleep_delays[1] == 2.0  # 2.0^1


@pytest.mark.asyncio
async def test_successful_request():
    """Test successful request flow."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}

        with patch.object(client._client, "request", return_value=mock_response):
            response = await client.get("/test")

            assert response.status_code == 200
            assert response.json() == {"data": "test"}


@pytest.mark.asyncio
async def test_api_key_header():
    """Test API key is included in headers."""
    client = ISLClient("http://localhost:8000", api_key="secret_key")

    assert "Authorization" in client._client.headers
    assert client._client.headers["Authorization"] == "Bearer secret_key"

    await client.close()


@pytest.mark.asyncio
async def test_no_api_key():
    """Test client works without API key."""
    client = ISLClient("http://localhost:8000")

    assert "Authorization" not in client._client.headers

    await client.close()
