"""
Tests for ISL client error handling and retry logic.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from isl_client import ISLClient
from isl_client.exceptions import (
    ISLException,
    ValidationError,
    ServiceUnavailable,
    RateLimitError,
    AuthenticationError,
    NotFoundError,
    TimeoutError,
)


@pytest.mark.asyncio
async def test_retry_on_server_error():
    """Test that client retries on 500 errors."""
    client = ISLClient(base_url="http://localhost:8000", max_retries=3)

    # Mock responses: fail twice, then succeed
    responses = [
        httpx.Response(500, text="Server Error"),
        httpx.Response(500, text="Server Error"),
        httpx.Response(200, json={"status": "identifiable"}),
    ]

    with patch.object(client._client, 'post') as mock_post:
        mock_post.side_effect = responses

        result = await client.causal.validate(
            dag={"nodes": ["A", "B"], "edges": []},
            treatment="A",
            outcome="B"
        )

        # Should succeed after retries
        assert result.status == "identifiable"
        assert mock_post.call_count == 3

    await client.close()


@pytest.mark.asyncio
async def test_exhausted_retries_raises():
    """Test that client raises after exhausting retries."""
    client = ISLClient(base_url="http://localhost:8000", max_retries=2)

    with patch.object(client._client, 'post') as mock_post:
        mock_post.side_effect = [
            httpx.Response(500, text="Server Error"),
            httpx.Response(500, text="Server Error"),
            httpx.Response(500, text="Server Error"),
        ]

        with pytest.raises(ServiceUnavailable):
            await client.causal.validate(
                dag={"nodes": ["A"], "edges": []},
                treatment="A",
                outcome="A"
            )

    await client.close()


@pytest.mark.asyncio
async def test_validation_error_no_retry():
    """Test that validation errors (422) don't trigger retries."""
    client = ISLClient(base_url="http://localhost:8000", max_retries=3)

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(
            422,
            json={"detail": [{"msg": "field required", "type": "value_error"}]}
        )

        with pytest.raises(ValidationError):
            await client.causal.validate(
                dag={},  # Invalid DAG
                treatment="A",
                outcome="B"
            )

        # Should NOT retry on validation error
        assert mock_post.call_count == 1

    await client.close()


@pytest.mark.asyncio
async def test_authentication_error():
    """Test authentication error handling."""
    client = ISLClient(
        base_url="http://localhost:8000",
        api_key="invalid_key"
    )

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(
            401,
            json={"detail": "Invalid authentication credentials"}
        )

        with pytest.raises(AuthenticationError):
            await client.causal.validate(
                dag={"nodes": ["A"], "edges": []},
                treatment="A",
                outcome="A"
            )

    await client.close()


@pytest.mark.asyncio
async def test_rate_limit_error():
    """Test rate limit error handling."""
    client = ISLClient(base_url="http://localhost:8000")

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(
            429,
            json={"detail": "Rate limit exceeded"}
        )

        with pytest.raises(RateLimitError):
            await client.causal.validate(
                dag={"nodes": ["A"], "edges": []},
                treatment="A",
                outcome="A"
            )

    await client.close()


@pytest.mark.asyncio
async def test_not_found_error():
    """Test 404 not found error."""
    client = ISLClient(base_url="http://localhost:8000")

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(
            404,
            json={"detail": "Not found"}
        )

        with pytest.raises(NotFoundError):
            await client.causal.validate(
                dag={"nodes": ["A"], "edges": []},
                treatment="A",
                outcome="A"
            )

    await client.close()


@pytest.mark.asyncio
async def test_timeout_error():
    """Test timeout handling."""
    client = ISLClient(base_url="http://localhost:8000", timeout=1.0)

    with patch.object(client._client, 'post') as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        with pytest.raises(TimeoutError):
            await client.causal.validate(
                dag={"nodes": ["A"], "edges": []},
                treatment="A",
                outcome="A"
            )

    await client.close()


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test exponential backoff between retries."""
    import time

    client = ISLClient(
        base_url="http://localhost:8000",
        max_retries=3,
        retry_backoff_factor=0.1  # Small factor for fast test
    )

    with patch.object(client._client, 'post') as mock_post:
        with patch('asyncio.sleep') as mock_sleep:
            mock_post.side_effect = [
                httpx.Response(500, text="Error"),
                httpx.Response(500, text="Error"),
                httpx.Response(200, json={"status": "identifiable"}),
            ]

            await client.causal.validate(
                dag={"nodes": ["A"], "edges": []},
                treatment="A",
                outcome="A"
            )

            # Should sleep with exponential backoff
            # First retry: 0.1s, Second retry: 0.2s
            assert mock_sleep.call_count >= 2

    await client.close()


@pytest.mark.asyncio
async def test_error_message_extraction():
    """Test error message extraction from various formats."""
    client = ISLClient(base_url="http://localhost:8000")

    # String detail
    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(
            400,
            json={"detail": "Simple error message"}
        )

        with pytest.raises(ISLException) as exc_info:
            await client.causal.validate(
                dag={"nodes": [], "edges": []},
                treatment="A",
                outcome="B"
            )

        assert "Simple error message" in str(exc_info.value)

    # List detail (Pydantic format)
    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(
            422,
            json={
                "detail": [
                    {"msg": "field required", "loc": ["dag", "nodes"]}
                ]
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            await client.causal.validate(
                dag={},
                treatment="A",
                outcome="B"
            )

        error_msg = str(exc_info.value)
        assert "field required" in error_msg

    await client.close()


@pytest.mark.asyncio
async def test_context_manager_closes_client():
    """Test that context manager properly closes client."""
    closed = False

    async def mock_aclose():
        nonlocal closed
        closed = True

    with patch('httpx.AsyncClient.aclose', new=mock_aclose):
        async with ISLClient(base_url="http://localhost:8000") as client:
            assert client is not None

        assert closed


@pytest.mark.asyncio
async def test_network_error_handling():
    """Test handling of network errors."""
    client = ISLClient(base_url="http://localhost:8000", max_retries=2)

    with patch.object(client._client, 'post') as mock_post:
        mock_post.side_effect = httpx.NetworkError("Connection failed")

        with pytest.raises(ServiceUnavailable):
            await client.causal.validate(
                dag={"nodes": ["A"], "edges": []},
                treatment="A",
                outcome="A"
            )

    await client.close()


@pytest.mark.asyncio
async def test_malformed_response_handling():
    """Test handling of malformed JSON responses."""
    client = ISLClient(base_url="http://localhost:8000")

    with patch.object(client._client, 'post') as mock_post:
        mock_post.return_value = httpx.Response(
            200,
            text="Not JSON"
        )

        with pytest.raises(ISLException):
            await client.causal.validate(
                dag={"nodes": ["A"], "edges": []},
                treatment="A",
                outcome="A"
            )

    await client.close()


def test_invalid_base_url():
    """Test that invalid base URL is handled."""
    # Should not raise during initialization
    client = ISLClient(base_url="not-a-url")
    assert client.base_url == "not-a-url"


@pytest.mark.asyncio
async def test_custom_timeout():
    """Test custom timeout configuration."""
    client = ISLClient(base_url="http://localhost:8000", timeout=5.0)

    assert client._client.timeout.read == 5.0

    await client.close()


@pytest.mark.asyncio
async def test_headers_include_api_key():
    """Test that API key is included in headers."""
    client = ISLClient(
        base_url="http://localhost:8000",
        api_key="test_key_123"
    )

    assert "Authorization" in client._client.headers
    assert client._client.headers["Authorization"] == "Bearer test_key_123"

    await client.close()


@pytest.mark.asyncio
async def test_headers_without_api_key():
    """Test headers when no API key provided."""
    client = ISLClient(base_url="http://localhost:8000")

    assert "Authorization" not in client._client.headers

    await client.close()
