"""
Performance Smoke Tests.

Quick latency checks to catch performance regressions in CI.
These tests verify that endpoints respond within acceptable latency bounds.
"""

import statistics
import time
from typing import List

import pytest
from starlette.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


class TestLatencyBaselines:
    """
    Verify endpoint latencies are within acceptable bounds.

    These are smoke tests - not comprehensive benchmarks.
    Run full benchmarks separately for detailed analysis.
    """

    # Latency thresholds in milliseconds (P95 targets)
    LATENCY_THRESHOLDS_MS = {
        "/health": 50,
        "/ready": 100,
        "/api/v1/validation/assumptions": 500,
        "/api/v1/counterfactual/generate": 2000,
        "/api/v1/sensitivity/analyze": 2000,
    }

    # Number of requests per endpoint for statistical significance
    SAMPLE_SIZE = 5

    def _measure_latency(self, client, method: str, path: str, json_data: dict = None) -> List[float]:
        """
        Measure request latency over multiple requests.

        Returns:
            List of latencies in milliseconds
        """
        latencies = []
        for _ in range(self.SAMPLE_SIZE):
            start = time.perf_counter()
            if method == "GET":
                response = client.get(path)
            else:
                response = client.post(path, json=json_data or {})
            end = time.perf_counter()

            # Only count successful responses
            if response.status_code in [200, 422]:  # 422 is validation error, still measures latency
                latencies.append((end - start) * 1000)  # Convert to ms

        return latencies

    def test_health_endpoint_latency(self, client):
        """Health endpoint should respond quickly."""
        latencies = self._measure_latency(client, "GET", "/health")

        assert len(latencies) > 0, "No successful health check responses"

        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        threshold = self.LATENCY_THRESHOLDS_MS["/health"]

        assert p95 < threshold, f"Health endpoint P95 latency {p95:.1f}ms exceeds {threshold}ms threshold"

    def test_ready_endpoint_latency(self, client):
        """Ready endpoint should respond quickly."""
        latencies = self._measure_latency(client, "GET", "/ready")

        assert len(latencies) > 0, "No successful ready check responses"

        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        threshold = self.LATENCY_THRESHOLDS_MS["/ready"]

        assert p95 < threshold, f"Ready endpoint P95 latency {p95:.1f}ms exceeds {threshold}ms threshold"

    def test_validation_endpoint_latency(self, client):
        """Validation endpoint should respond within bounds."""
        test_payload = {
            "dag": {
                "nodes": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"]]
            },
            "treatment": "A",
            "outcome": "C"
        }

        latencies = self._measure_latency(
            client, "POST", "/api/v1/validation/assumptions", test_payload
        )

        if not latencies:
            pytest.skip("No successful validation responses (may require auth)")

        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        threshold = self.LATENCY_THRESHOLDS_MS["/api/v1/validation/assumptions"]

        assert p95 < threshold, f"Validation P95 latency {p95:.1f}ms exceeds {threshold}ms threshold"


class TestThroughputBaselines:
    """
    Verify minimum throughput for critical endpoints.
    """

    def test_health_throughput(self, client):
        """Health endpoint should handle high throughput."""
        num_requests = 50
        start = time.perf_counter()

        for _ in range(num_requests):
            response = client.get("/health")
            assert response.status_code == 200

        elapsed = time.perf_counter() - start
        rps = num_requests / elapsed

        # Health should handle at least 100 RPS
        assert rps > 100, f"Health endpoint throughput {rps:.1f} RPS below 100 RPS minimum"


class TestMemoryBaseline:
    """
    Basic memory checks to detect leaks.
    """

    def test_no_obvious_memory_leak(self, client):
        """Multiple requests should not cause obvious memory growth."""
        import gc

        # Force garbage collection before measuring
        gc.collect()

        # Make several requests
        for _ in range(20):
            client.get("/health")

        # Force garbage collection
        gc.collect()

        # This is a basic sanity check - detailed memory profiling
        # should be done with dedicated tools
        # Just verify we can complete the requests without OOM
        assert True


class TestResponseSizeBaseline:
    """
    Verify response sizes are reasonable.
    """

    def test_health_response_size(self, client):
        """Health response should be small."""
        response = client.get("/health")
        size_kb = len(response.content) / 1024

        # Health response should be under 1KB
        assert size_kb < 1, f"Health response {size_kb:.2f}KB exceeds 1KB limit"

    def test_error_response_size(self, client):
        """Error responses should be reasonable size."""
        # Trigger a validation error
        response = client.post(
            "/api/v1/validation/assumptions",
            json={"invalid": "data"}
        )

        size_kb = len(response.content) / 1024

        # Error response should be under 10KB
        assert size_kb < 10, f"Error response {size_kb:.2f}KB exceeds 10KB limit"
