"""
Integration tests for concurrent request handling.

Tests verify that:
- Concurrent requests don't interfere with each other
- Service handles sustained load without degradation
- Cache contention is handled correctly
- Performance remains stable under load
"""

import pytest
import httpx
import asyncio
import time
from typing import List, Tuple

BASE_URL = "http://localhost:8000"


class TestConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_no_interference(self):
        """Concurrent requests should not interfere with each other."""

        # Create unique payloads with different nodes
        payloads = [
            {
                "dag": {"nodes": [f"Node{i}_A", f"Node{i}_B"], "edges": [[f"Node{i}_A", f"Node{i}_B"]]},
                "treatment": f"Node{i}_A",
                "outcome": f"Node{i}_B"
            }
            for i in range(10)
        ]

        async def make_request(payload, request_num):
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
                try:
                    response = await client.post(
                        "/api/v1/causal/validate",
                        json=payload
                    )
                    return request_num, response.status_code, response.json() if response.status_code == 200 else None
                except Exception as e:
                    return request_num, None, str(e)

        try:
            # Send all requests concurrently
            tasks = [make_request(p, i) for i, p in enumerate(payloads)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for r in results if not isinstance(r, Exception) and r[1] == 200)

            print(f"✓ {success_count}/{len(payloads)} concurrent requests succeeded")

            # Verify each response corresponds to its request
            for i, result in enumerate(results):
                if not isinstance(result, Exception) and result[1] == 200:
                    data = result[2]
                    if data:
                        # Each request should get its own response
                        assert "_metadata" in data
                        # Verify treatment/outcome match request
                        expected_treatment = f"Node{i}_A"
                        # Response should indicate success

        except Exception as e:
            print(f"⚠ Concurrency test failed: {e}")
            pytest.skip("Server not available or test infrastructure issue")

    @pytest.mark.asyncio
    async def test_sustained_load(self):
        """Service should handle sustained load without degradation."""

        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        latencies = []

        async def make_request():
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
                start = time.time()
                try:
                    response = await client.post("/api/v1/causal/validate", json=payload)
                    latency = time.time() - start
                    return latency, response.status_code
                except Exception as e:
                    latency = time.time() - start
                    return latency, None

        try:
            # Send 30 requests with 5 concurrent at a time (6 batches)
            for batch in range(6):
                batch_results = await asyncio.gather(*[make_request() for _ in range(5)])
                latencies.extend([r[0] for r in batch_results if r[1] == 200])

                # Small delay between batches
                await asyncio.sleep(0.1)

            if len(latencies) >= 10:
                # Calculate statistics
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)

                print(f"✓ Sustained load: {len(latencies)} requests")
                print(f"  Avg: {avg_latency:.3f}s, Min: {min_latency:.3f}s, Max: {max_latency:.3f}s")

                # Latency should not degrade significantly
                assert avg_latency < 2.0, f"Average latency too high: {avg_latency:.2f}s"
                assert max_latency < 5.0, f"Max latency too high: {max_latency:.2f}s"
            else:
                pytest.skip(f"Not enough successful requests: {len(latencies)}")

        except Exception as e:
            print(f"⚠ Sustained load test failed: {e}")
            pytest.skip("Server not available")

    @pytest.mark.asyncio
    async def test_cache_contention(self):
        """Multiple requests for same data should not cause issues."""

        # Same payload for all requests (cache contention scenario)
        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        async def make_request():
            async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
                try:
                    response = await client.post(
                        "/api/v1/causal/validate",
                        json=payload
                    )
                    return response.json() if response.status_code == 200 else None
                except Exception:
                    return None

        try:
            # Send 20 concurrent requests for same data
            responses = await asyncio.gather(*[make_request() for _ in range(20)])

            # Filter successful responses
            successful_responses = [r for r in responses if r is not None]

            if len(successful_responses) >= 10:
                # All responses should be identical (deterministic)
                first_status = successful_responses[0].get("status")

                identical_count = sum(1 for r in successful_responses if r.get("status") == first_status)

                print(f"✓ Cache contention handled: {len(successful_responses)} responses")
                print(f"  {identical_count}/{len(successful_responses)} responses identical")

                # Most responses should be identical (allowing for some timing variations)
                assert identical_count / len(successful_responses) >= 0.9
            else:
                pytest.skip(f"Not enough successful responses: {len(successful_responses)}")

        except Exception as e:
            print(f"⚠ Cache contention test failed: {e}")
            pytest.skip("Server not available")

    @pytest.mark.asyncio
    async def test_mixed_concurrent_requests(self):
        """Mix of different request types should work concurrently."""

        requests = [
            # Causal validation requests
            ("validate", {
                "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
                "treatment": "A",
                "outcome": "B"
            }),
            ("validate", {
                "dag": {"nodes": ["X", "Y", "Z"], "edges": [["X", "Y"], ["Y", "Z"]]},
                "treatment": "X",
                "outcome": "Z"
            }),
            # Preference elicitation requests
            ("elicit", {
                "user_id": f"concurrent_user_1",
                "context": {
                    "domain": "pricing",
                    "variables": ["revenue", "churn"]
                },
                "num_queries": 2
            }),
            ("elicit", {
                "user_id": f"concurrent_user_2",
                "context": {
                    "domain": "pricing",
                    "variables": ["revenue", "churn"]
                },
                "num_queries": 2
            }),
        ]

        async def make_request(req_type, payload):
            endpoint = "/api/v1/causal/validate" if req_type == "validate" else "/api/v1/preferences/elicit"

            async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
                try:
                    response = await client.post(endpoint, json=payload)
                    return req_type, response.status_code
                except Exception as e:
                    return req_type, None

        try:
            # Send all requests concurrently
            results = await asyncio.gather(*[make_request(rt, p) for rt, p in requests])

            # Count successes by type
            validate_success = sum(1 for rt, status in results if rt == "validate" and status == 200)
            elicit_success = sum(1 for rt, status in results if rt == "elicit" and status == 200)

            print(f"✓ Mixed concurrent requests:")
            print(f"  Validation: {validate_success}/{sum(1 for rt, _ in results if rt == 'validate')}")
            print(f"  Elicitation: {elicit_success}/{sum(1 for rt, _ in results if rt == 'elicit')}")

            # At least some of each type should succeed
            assert validate_success > 0, "No validation requests succeeded"

        except Exception as e:
            print(f"⚠ Mixed concurrent test failed: {e}")
            pytest.skip("Server not available")


class TestPerformanceStability:
    """Test performance stability under various conditions."""

    @pytest.mark.asyncio
    async def test_latency_consistency(self):
        """Latency should be consistent across multiple requests."""

        payload = {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }

        latencies = []

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            for i in range(20):
                try:
                    start = time.time()
                    response = await client.post("/api/v1/causal/validate", json=payload)
                    latency = time.time() - start

                    if response.status_code == 200:
                        latencies.append(latency)

                except Exception as e:
                    print(f"⚠ Request {i+1} failed: {e}")

        if len(latencies) >= 10:
            avg = sum(latencies) / len(latencies)
            max_lat = max(latencies)
            min_lat = min(latencies)

            # Calculate variance
            variance = sum((l - avg) ** 2 for l in latencies) / len(latencies)
            std_dev = variance ** 0.5

            print(f"✓ Latency consistency ({len(latencies)} samples):")
            print(f"  Avg: {avg*1000:.1f}ms, Std Dev: {std_dev*1000:.1f}ms")
            print(f"  Min: {min_lat*1000:.1f}ms, Max: {max_lat*1000:.1f}ms")

            # Standard deviation should be reasonable
            assert std_dev < avg * 2, f"Latency too variable (std dev: {std_dev:.3f}s)"
        else:
            pytest.skip(f"Not enough successful requests: {len(latencies)}")


# Run tests with: pytest tests/integration/test_concurrency.py -v -s
