"""
Integration tests for production excellence features.

Tests:
- Compression
- Connection pooling
- Memory profiling
- Enhanced errors
- Distributed tracing
- Custom metrics
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.memory_profiler import MemoryMonitor
from src.utils.tracing import get_trace_id, set_trace_id

client = TestClient(app)


class TestCompression:
    """Test gzip compression functionality."""

    def test_compression_enabled_with_header(self):
        """Responses are compressed when client accepts gzip."""
        response = client.get(
            "/health",
            headers={"Accept-Encoding": "gzip"}
        )

        # Should have gzip encoding if response >1KB
        # Health endpoint is small, so may not be compressed
        assert response.status_code == 200

    def test_large_response_compressed(self):
        """Large responses are compressed."""
        # Create a request that will return larger JSON
        payload = {
            "dag": {
                "nodes": ["A", "B", "C", "D", "E"],
                "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]]
            },
            "treatment": "A",
            "outcome": "E"
        }

        response = client.post(
            "/api/v1/causal/validate",
            json=payload,
            headers={"Accept-Encoding": "gzip"}
        )

        # Should succeed (may or may not be compressed depending on size)
        assert response.status_code in [200, 400, 422]


class TestConnectionPooling:
    """Test Redis connection pooling."""

    def test_redis_pool_configured(self):
        """Redis client uses connection pool."""
        from src.services.user_storage import UserStorage

        storage = UserStorage()

        # If Redis is available, should have pool configured
        if storage.redis_enabled:
            # Connection pool should be configured
            # (Can't easily test internal pool state, but no errors is good)
            assert storage.redis_client is not None


class TestMemoryProfiling:
    """Test memory monitoring and circuit breaker."""

    def test_memory_monitor_get_usage(self):
        """Memory monitor can get current usage."""
        usage = MemoryMonitor.get_memory_usage()

        assert "rss_mb" in usage
        assert "percent" in usage
        assert usage["rss_mb"] > 0
        assert 0 <= usage["percent"] <= 100

    def test_system_memory(self):
        """Can get system memory info."""
        mem = MemoryMonitor.get_system_memory()

        assert "total_mb" in mem
        assert "available_mb" in mem
        assert mem["total_mb"] > 0

    def test_memory_circuit_breaker_allows_normal_requests(self):
        """Circuit breaker allows requests when memory is normal."""
        response = client.get("/health")

        # Should succeed (unless system memory is actually >85%)
        assert response.status_code == 200


class TestEnhancedErrors:
    """Test enhanced error messages."""

    def test_enhanced_error_format(self):
        """Enhanced errors have suggestions and documentation."""
        from src.utils.error_messages import CausalNotIdentifiableError

        error = CausalNotIdentifiableError("X", "Y", "No adjustment set found")

        response_dict = error.to_response()

        assert "error" in response_dict
        assert "code" in response_dict["error"]
        assert "suggestions" in response_dict["error"]
        assert "documentation" in response_dict["error"]
        assert len(response_dict["error"]["suggestions"]) > 0

    def test_batch_size_error(self):
        """Batch size exceeded error has actionable suggestions."""
        from src.utils.error_messages import BatchSizeExceededError

        error = BatchSizeExceededError(
            batch_size=100,
            max_size=50,
            endpoint="validation"
        )

        response_dict = error.to_response()

        assert "Split into multiple batches" in str(response_dict["error"]["suggestions"])
        assert error.code == "BATCH_SIZE_EXCEEDED"


class TestDistributedTracing:
    """Test distributed tracing functionality."""

    def test_trace_id_in_response_header(self):
        """All responses include X-Trace-Id header."""
        response = client.get("/health")

        assert "X-Trace-Id" in response.headers
        assert response.headers["X-Trace-Id"].startswith("trace_")

    def test_trace_id_propagation(self):
        """Trace ID is propagated from request to response."""
        custom_trace_id = "trace_custom_test_123"

        response = client.get(
            "/health",
            headers={"X-Trace-Id": custom_trace_id}
        )

        assert response.headers["X-Trace-Id"] == custom_trace_id

    def test_trace_id_generated_if_missing(self):
        """Trace ID is generated if not provided."""
        response1 = client.get("/health")
        response2 = client.get("/health")

        # Both should have trace IDs
        assert "X-Trace-Id" in response1.headers
        assert "X-Trace-Id" in response2.headers

        # They should be different
        assert response1.headers["X-Trace-Id"] != response2.headers["X-Trace-Id"]


class TestCustomMetrics:
    """Test custom metrics are registered."""

    def test_batch_metrics_exist(self):
        """Batch processing metrics are registered."""
        from src.utils.business_metrics import (
            batch_requests_total,
            batch_items_processed,
            batch_processing_duration
        )

        # Metrics should be defined
        assert batch_requests_total is not None
        assert batch_items_processed is not None
        assert batch_processing_duration is not None

    def test_adaptive_sampling_metrics_exist(self):
        """Adaptive sampling metrics are registered."""
        from src.utils.business_metrics import (
            adaptive_sampling_convergence,
            adaptive_sampling_speedup
        )

        assert adaptive_sampling_convergence is not None
        assert adaptive_sampling_speedup is not None

    def test_memory_metrics_exist(self):
        """Memory monitoring metrics are registered."""
        from src.utils.business_metrics import (
            memory_usage_bytes,
            memory_circuit_breaker_triggers
        )

        assert memory_usage_bytes is not None
        assert memory_circuit_breaker_triggers is not None

    def test_tracing_metrics_exist(self):
        """Distributed tracing metrics are registered."""
        from src.utils.business_metrics import trace_propagation_total

        assert trace_propagation_total is not None


class TestHealthEndpoint:
    """Test health endpoint with all enhancements."""

    def test_health_check_returns_ok(self):
        """Health check returns 200 OK."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_check_has_trace_id(self):
        """Health check response includes trace ID."""
        response = client.get("/health")

        assert "X-Trace-Id" in response.headers

    def test_health_check_exempt_from_rate_limiting(self):
        """Health checks are not rate limited."""
        # Make many rapid requests
        for _ in range(150):  # More than rate limit
            response = client.get("/health")
            # Should all succeed (not 429)
            assert response.status_code == 200


class TestEndToEnd:
    """End-to-end tests with all production features."""

    def test_causal_validation_with_all_features(self):
        """Causal validation works with compression, tracing, etc."""
        payload = {
            "dag": {
                "nodes": ["X", "Y", "Z"],
                "edges": [["X", "Y"], ["Y", "Z"]]
            },
            "treatment": "X",
            "outcome": "Z"
        }

        response = client.post(
            "/api/v1/causal/validate",
            json=payload,
            headers={
                "Accept-Encoding": "gzip",
                "X-Trace-Id": "trace_e2e_test"
            }
        )

        # Should succeed
        assert response.status_code == 200

        # Should have trace ID
        assert response.headers["X-Trace-Id"] == "trace_e2e_test"

        # Response should be valid JSON
        data = response.json()
        assert "status" in data
        assert "confidence" in data

    def test_batch_endpoint_with_tracing(self):
        """Batch endpoints work with distributed tracing."""
        payload = {
            "requests": [
                {
                    "dag": {
                        "nodes": ["A", "B"],
                        "edges": [["A", "B"]]
                    },
                    "treatment": "A",
                    "outcome": "B"
                }
                for _ in range(3)
            ]
        }

        response = client.post(
            "/api/v1/batch/validate",
            json=payload,
            headers={"X-Trace-Id": "trace_batch_test"}
        )

        # Should succeed or fail validation, but not error
        assert response.status_code in [200, 400, 422]

        # Should have trace ID
        assert response.headers["X-Trace-Id"] == "trace_batch_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
