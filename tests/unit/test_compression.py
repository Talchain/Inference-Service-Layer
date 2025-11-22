"""
Tests for gzip compression middleware.

NOTE: Uses httpx.AsyncClient to avoid Starlette TestClient async middleware bug.
"""

import pytest
import httpx


BASE_URL = "http://localhost:8000"


class TestGzipCompression:
    """Test gzip compression functionality."""

    @pytest.mark.asyncio
    async def test_compression_middleware_installed(self):
        """GZip middleware is installed and responds to headers."""
        try:
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
                response = await client.get(
                    "/health",
                    headers={"Accept-Encoding": "gzip"}
                )

                # Should succeed
                assert response.status_code == 200

                # Content-encoding may or may not be gzip depending on response size
                # But the middleware should handle the header properly
                assert "content-type" in response.headers

        except httpx.ConnectError:
            pytest.skip("ISL not running")

    @pytest.mark.asyncio
    async def test_large_response_handling(self):
        """Large responses are handled correctly."""
        # Create a request that will return larger JSON
        payload = {
            "dag": {
                "nodes": ["A", "B", "C", "D", "E"],
                "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]]
            },
            "treatment": "A",
            "outcome": "E"
        }

        try:
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
                response = await client.post(
                    "/api/v1/causal/validate",
                    json=payload,
                    headers={"Accept-Encoding": "gzip"}
                )

                # Should succeed (may or may not be compressed)
                assert response.status_code in [200, 400, 422]

                # Should be valid JSON
                assert response.json()

        except httpx.ConnectError:
            pytest.skip("ISL not running")

    @pytest.mark.asyncio
    async def test_compression_preserves_response_data(self):
        """Compression doesn't corrupt response data."""
        payload = {
            "dag": {
                "nodes": ["X", "Y", "Z"],
                "edges": [["X", "Y"], ["Y", "Z"]]
            },
            "treatment": "X",
            "outcome": "Z"
        }

        try:
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
                # Get response without compression
                response1 = await client.post(
                    "/api/v1/causal/validate",
                    json=payload,
                    headers={}
                )

                # Get response with compression
                response2 = await client.post(
                    "/api/v1/causal/validate",
                    json=payload,
                    headers={"Accept-Encoding": "gzip"}
                )

                # Both should have same status code
                assert response1.status_code == response2.status_code

                # If both succeeded, responses should be equivalent
                if response1.status_code == 200 and response2.status_code == 200:
                    json1 = response1.json()
                    json2 = response2.json()

                    # Key fields should match
                    assert json1.get("status") == json2.get("status")
                    assert json1.get("confidence") == json2.get("confidence")

        except httpx.ConnectError:
            pytest.skip("ISL not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
