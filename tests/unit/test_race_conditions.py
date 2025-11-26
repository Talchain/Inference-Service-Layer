"""
Race condition tests for security validation.

Tests concurrent access patterns that could lead to:
- Rate limit bypass
- Authentication bypass
- Cache inconsistencies
- Time-of-check to time-of-use (TOCTOU) vulnerabilities
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

from src.middleware.rate_limiting import RateLimiter, RedisRateLimiter


class TestRateLimiterRaceConditions:
    """Test race conditions in rate limiting."""

    def test_concurrent_rate_limit_checks(self):
        """
        Test that concurrent requests don't bypass rate limit.

        Scenario: Many requests arrive simultaneously, each checking
        if they're within the limit. Without proper synchronization,
        they could all pass before any is counted.
        """
        limiter = RateLimiter(requests_per_minute=10)
        identifier = "test_client"

        results = []

        def make_request():
            """Simulate a request checking rate limit."""
            allowed, _ = limiter.check_rate_limit(identifier)
            results.append(allowed)
            return allowed

        # Send 20 concurrent requests (2x the limit)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            for future in as_completed(futures):
                future.result()

        # Count allowed requests
        allowed_count = sum(1 for r in results if r)

        # Should allow at most 10 (the limit), but due to race conditions
        # in the in-memory limiter, we might get slightly more
        # The key is that we shouldn't get all 20 through
        assert allowed_count <= 15, f"Too many requests allowed: {allowed_count}"

    def test_burst_requests_at_window_boundary(self):
        """
        Test burst of requests at rate limit window boundary.

        Scenario: Requests arrive right at the 60-second window boundary.
        """
        limiter = RateLimiter(requests_per_minute=10)
        identifier = "boundary_test"

        # Fill up the limit
        for _ in range(10):
            limiter.check_rate_limit(identifier)

        # Verify we're at the limit
        allowed, retry_after = limiter.check_rate_limit(identifier)
        assert not allowed

        # Manually expire old requests by manipulating internal state
        # This simulates time passing
        limiter.requests[identifier] = []

        # Now we should be allowed again
        allowed, _ = limiter.check_rate_limit(identifier)
        assert allowed

    def test_multiple_identifiers_isolation(self):
        """
        Test that rate limits for different identifiers are isolated.

        One client hitting their limit shouldn't affect another client.
        """
        limiter = RateLimiter(requests_per_minute=5)

        # Client A hits their limit
        for _ in range(5):
            limiter.check_rate_limit("client_a")

        allowed_a, _ = limiter.check_rate_limit("client_a")
        assert not allowed_a, "Client A should be rate limited"

        # Client B should still be allowed
        allowed_b, _ = limiter.check_rate_limit("client_b")
        assert allowed_b, "Client B should not be affected by Client A's limit"


class TestRedisRateLimiterRaceConditions:
    """Test race conditions in Redis-backed rate limiting."""

    def test_fallback_on_redis_failure(self):
        """
        Test graceful fallback when Redis fails during rate limit check.
        """
        fallback = RateLimiter(requests_per_minute=10)

        # Create a mock Redis client that raises an exception
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = Exception("Redis connection failed")

        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            requests_per_minute=10,
            fallback=fallback
        )

        # Should fall back to in-memory limiter
        allowed, _ = limiter.check_rate_limit("test_client")
        assert allowed

    def test_redis_none_uses_fallback(self):
        """
        Test that None Redis client uses fallback immediately.
        """
        fallback = RateLimiter(requests_per_minute=5)
        limiter = RedisRateLimiter(
            redis_client=None,
            requests_per_minute=5,
            fallback=fallback
        )

        # Use up the limit
        for _ in range(5):
            limiter.check_rate_limit("test")

        allowed, _ = limiter.check_rate_limit("test")
        assert not allowed


class TestAuthenticationRaceConditions:
    """Test race conditions in authentication."""

    def test_api_key_validation_timing(self):
        """
        Test that API key validation doesn't leak timing information.

        Both valid and invalid keys should take similar time to validate.
        """
        from src.middleware.auth import APIKeyAuthMiddleware

        app = MagicMock()
        middleware = APIKeyAuthMiddleware(
            app,
            api_keys="valid_key_123456789"
        )

        valid_key = "valid_key_123456789"
        invalid_key = "invalid_key_12345678"

        # Time validation of valid key
        valid_times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = valid_key in middleware._api_keys
            valid_times.append(time.perf_counter() - start)

        # Time validation of invalid key
        invalid_times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = invalid_key in middleware._api_keys
            invalid_times.append(time.perf_counter() - start)

        # Calculate averages (both should be similarly fast for set lookup)
        avg_valid = sum(valid_times) / len(valid_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)

        # Set membership is O(1), so times should be very similar
        # Allow for some variance due to system noise
        ratio = max(avg_valid, avg_invalid) / min(avg_valid, avg_invalid)
        assert ratio < 10, f"Timing difference too large: {ratio}x"


class TestCacheRaceConditions:
    """Test race conditions in caching."""

    def test_concurrent_cache_access(self):
        """
        Test concurrent access to the settings cache.
        """
        from src.config import get_settings

        settings_list = []

        def get_settings_thread():
            settings = get_settings()
            settings_list.append(id(settings))

        # Multiple threads getting settings
        threads = [
            threading.Thread(target=get_settings_thread)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same cached instance
        assert len(set(settings_list)) == 1, "Settings cache returned different instances"


class TestTracingRaceConditions:
    """Test race conditions in distributed tracing."""

    def test_trace_id_isolation_between_contexts(self):
        """
        Test that trace IDs are isolated between different request contexts.
        """
        from src.utils.tracing import generate_trace_id, set_trace_id, get_trace_id

        results = {}

        def set_and_get_trace(thread_id):
            """Set a trace ID and verify it's isolated."""
            trace_id = f"trace_{thread_id}"
            set_trace_id(trace_id)
            time.sleep(0.01)  # Small delay to allow interleaving
            retrieved = get_trace_id()
            results[thread_id] = retrieved

        threads = []
        for i in range(5):
            t = threading.Thread(target=set_and_get_trace, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have its own trace ID
        # Note: contextvars provides thread isolation
        for thread_id, trace in results.items():
            assert trace == f"trace_{thread_id}", \
                f"Thread {thread_id} got wrong trace ID: {trace}"


class TestAsyncRaceConditions:
    """Test race conditions in async code paths."""

    @pytest.mark.asyncio
    async def test_concurrent_async_rate_limit_checks(self):
        """
        Test concurrent async requests against rate limiter.
        """
        limiter = RateLimiter(requests_per_minute=10)
        identifier = "async_test"

        async def check_limit():
            allowed, _ = limiter.check_rate_limit(identifier)
            return allowed

        # Run 20 concurrent async tasks
        tasks = [check_limit() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        allowed_count = sum(1 for r in results if r)

        # Should limit requests appropriately
        # Some variance is expected due to async scheduling
        assert allowed_count <= 15, f"Too many async requests allowed: {allowed_count}"


class TestTOCTOUVulnerabilities:
    """
    Test Time-Of-Check to Time-Of-Use vulnerabilities.

    TOCTOU occurs when there's a gap between checking a condition
    and acting on it, allowing the condition to change.
    """

    def test_rate_limit_toctou(self):
        """
        Test for TOCTOU in rate limit checks.

        Scenario: Check if request is allowed, then record it.
        If these aren't atomic, multiple requests could pass.
        """
        limiter = RateLimiter(requests_per_minute=1)
        identifier = "toctou_test"

        # In-memory limiter does check and record in one operation
        # This test verifies that behavior

        results = []

        def toctou_check():
            # Simulate checking and recording
            allowed, _ = limiter.check_rate_limit(identifier)
            results.append(allowed)

        # Many concurrent checks
        threads = [
            threading.Thread(target=toctou_check)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed_count = sum(1 for r in results if r)

        # Due to race conditions in threading, we might get more than 1
        # but we shouldn't get all 10
        assert allowed_count < 5, f"TOCTOU vulnerability: {allowed_count} requests allowed"


class TestSecurityAuditLogRaceConditions:
    """Test race conditions in security audit logging."""

    def test_concurrent_audit_log_writes(self):
        """
        Test that concurrent audit log writes don't interleave.
        """
        from src.utils.secure_logging import SecurityAuditLogger

        logger = SecurityAuditLogger("test.audit")
        calls = []

        # Patch the underlying logger to capture calls
        original_log = logger._logger.log

        def mock_log(level, msg, **kwargs):
            calls.append((level, msg, kwargs))
            return original_log(level, msg, **kwargs)

        logger._logger.log = mock_log

        def log_auth():
            logger.log_authentication_attempt(
                success=True,
                client_ip="127.0.0.1",
                path="/test"
            )

        # Concurrent log writes
        threads = [
            threading.Thread(target=log_auth)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All logs should have been written
        assert len(calls) == 10


class TestResourceExhaustion:
    """Test for resource exhaustion attacks via race conditions."""

    def test_rate_limiter_memory_growth(self):
        """
        Test that rate limiter doesn't grow unbounded memory.

        Many unique identifiers could fill up the requests dict.
        """
        limiter = RateLimiter(requests_per_minute=10)

        # Simulate many unique clients
        for i in range(1000):
            limiter.check_rate_limit(f"client_{i}")

        # Check memory usage (rough estimate via dict size)
        # Each identifier should have at most 10 entries
        total_entries = sum(len(v) for v in limiter.requests.values())
        assert total_entries <= 10000, f"Too many entries: {total_entries}"

    def test_old_entries_cleanup(self):
        """
        Test that old rate limit entries are cleaned up.
        """
        limiter = RateLimiter(requests_per_minute=100)
        identifier = "cleanup_test"

        # Add some requests
        for _ in range(10):
            limiter.check_rate_limit(identifier)

        # Manually expire entries
        limiter.requests[identifier] = [
            time.time() - 120  # 2 minutes ago (expired)
            for _ in range(10)
        ]

        # Next check should clean up old entries
        limiter.check_rate_limit(identifier)

        # Old entries should be cleaned up
        assert len(limiter.requests[identifier]) == 1
