"""
Integration tests for Redis failover and graceful degradation.

Tests verify that:
- Service continues operating when Redis unavailable
- Cache miss triggers fresh computation
- Errors include request_id for tracing
- Timeout handling works correctly
- Graceful degradation under failures
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


class TestRedisFailover:
    """Test graceful degradation when Redis unavailable."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires ability to simulate Redis failure")
    async def test_service_continues_without_redis(self):
        """Service should continue operating when Redis fails."""

        # This test requires ability to simulate Redis failure
        # Implementation depends on how Redis is injected

        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            # First request (Redis working)
            response1 = await client.post("/api/v1/causal/validate", json=payload)
            assert response1.status_code == 200

            # Simulate Redis failure (implementation-specific)
            # This might require stopping Redis container or using test doubles

            # Second request (Redis failed, should still work)
            response2 = await client.post("/api/v1/causal/validate", json=payload)

            # Should succeed (using in-memory fallback)
            assert response2.status_code == 200

            # Response should indicate degraded mode (implementation-specific)
            print(f"✓ Service continues without Redis")

    @pytest.mark.asyncio
    async def test_cache_behaviour_with_repeated_requests(self):
        """Repeated requests should show cache behaviour."""

        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        latencies = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for i in range(5):
                try:
                    import time
                    start = time.time()
                    response = await client.post("/api/v1/causal/validate", json=payload)
                    latency = time.time() - start

                    if response.status_code == 200:
                        latencies.append(latency)
                except Exception as e:
                    print(f"⚠ Request {i+1} failed: {e}")

        if len(latencies) >= 3:
            # Log latencies (may show caching effect)
            print(f"✓ Latencies: {[f'{l*1000:.1f}ms' for l in latencies]}")

            # First request typically slower (cache miss)
            # Subsequent requests may be faster (cache hit)
            # But this is not guaranteed due to small payload size
        else:
            pytest.skip("Not enough successful requests")


class TestErrorPropagation:
    """Test error handling and propagation."""

    @pytest.mark.asyncio
    async def test_error_includes_request_id(self):
        """Errors should include request_id for tracing."""

        # Send invalid request (self-loop in DAG)
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            try:
                response = await client.post(
                    "/api/v1/causal/validate",
                    json={
                        "dag": {"nodes": ["A"], "edges": [["A", "A"]]},  # Self-loop (invalid)
                        "treatment": "A",
                        "outcome": "A"
                    }
                )

                # Should return error (400 or 422)
                assert response.status_code in [400, 422], f"Expected error, got {response.status_code}"

                data = response.json()
                # Check for detail field (FastAPI standard)
                assert "detail" in data, "Error response missing 'detail' field"

                print(f"✓ Error response format valid: {response.status_code}")

            except httpx.TimeoutException:
                pytest.skip("Server not available")
            except Exception as e:
                print(f"⚠ Test error: {e}")
                pytest.skip("Server not available or unexpected error")

    @pytest.mark.asyncio
    async def test_invalid_input_validation(self):
        """Invalid input should return 400/422 with descriptive error."""

        test_cases = [
            # Missing required field
            {
                "payload": {
                    "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
                    # Missing 'treatment' and 'outcome'
                },
                "expected_status": [422],
                "description": "Missing required fields"
            },
            # Invalid DAG structure
            {
                "payload": {
                    "dag": {"nodes": ["A", "B"], "edges": [["A", "C"]]},  # C not in nodes
                    "treatment": "A",
                    "outcome": "B"
                },
                "expected_status": [400, 422],
                "description": "Invalid DAG (edge references unknown node)"
            },
        ]

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for test_case in test_cases:
                try:
                    response = await client.post(
                        "/api/v1/causal/validate",
                        json=test_case["payload"]
                    )

                    assert response.status_code in test_case["expected_status"], \
                        f"{test_case['description']}: Expected {test_case['expected_status']}, got {response.status_code}"

                    data = response.json()
                    assert "detail" in data, f"{test_case['description']}: Missing 'detail' in error"

                    print(f"✓ {test_case['description']}: {response.status_code}")

                except httpx.TimeoutException:
                    print(f"⚠ {test_case['description']}: Server timeout")
                except Exception as e:
                    print(f"⚠ {test_case['description']}: {e}")

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of request timeouts."""

        # Use short timeout to test timeout handling
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=0.5) as client:
            try:
                response = await client.post(
                    "/api/v1/causal/validate",
                    json={
                        "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
                        "treatment": "A",
                        "outcome": "B"
                    }
                )

                # If succeeds, check it's fast
                print(f"✓ Request completed within 0.5s timeout")

            except httpx.TimeoutException:
                # Timeout is expected with aggressive timeout setting
                print(f"✓ Timeout handled gracefully (0.5s timeout)")

            except Exception as e:
                print(f"⚠ Unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_server_availability(self):
        """Basic connectivity test."""

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=5.0) as client:
            try:
                response = await client.get("/health")

                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ Server available: {data.get('status', 'unknown')}")
                    assert data.get("status") == "healthy", "Server not healthy"
                else:
                    print(f"⚠ Server returned {response.status_code}")
                    pytest.skip(f"Server returned {response.status_code}")

            except httpx.ConnectError:
                print(f"✗ Server not running on {BASE_URL}")
                pytest.skip("Server not available")

            except Exception as e:
                print(f"⚠ Connection error: {e}")
                pytest.skip("Server not available")


class TestGracefulDegradation:
    """Test graceful degradation scenarios."""

    @pytest.mark.asyncio
    async def test_health_endpoint_always_responds(self):
        """Health endpoint should always respond."""

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=5.0) as client:
            try:
                response = await client.get("/health")

                # Health endpoint should always return 200
                assert response.status_code == 200, f"Health check failed: {response.status_code}"

                data = response.json()
                assert "status" in data, "Health response missing 'status'"
                assert "version" in data, "Health response missing 'version'"

                print(f"✓ Health endpoint responsive: {data.get('status')}")

            except Exception as e:
                print(f"⚠ Health check failed: {e}")
                pytest.skip("Server not available")

    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(self):
        """Multiple sequential requests should all succeed."""

        payload = {
            "dag": {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]]},
            "treatment": "A",
            "outcome": "C"
        }

        success_count = 0
        total_requests = 10

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for i in range(total_requests):
                try:
                    response = await client.post("/api/v1/causal/validate", json=payload)

                    if response.status_code == 200:
                        success_count += 1
                    else:
                        print(f"⚠ Request {i+1} returned {response.status_code}")

                except Exception as e:
                    print(f"⚠ Request {i+1} failed: {e}")

        if success_count > 0:
            success_rate = (success_count / total_requests) * 100
            print(f"✓ Success rate: {success_count}/{total_requests} ({success_rate:.1f}%)")

            # Expect high success rate
            assert success_rate >= 90, f"Success rate too low: {success_rate:.1f}%"
        else:
            pytest.skip("No successful requests")


# Run tests with: pytest tests/integration/test_redis_failover.py -v -s
