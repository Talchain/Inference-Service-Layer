"""
Unit tests for rate limiting middleware.

Tests in-memory and Redis-backed rate limiters.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from src.middleware.rate_limiting import (
    RateLimiter,
    RedisRateLimiter,
    RateLimitMiddleware,
)
from src.utils.ip_extraction import get_client_ip, _is_trusted_proxy


class TestRateLimiter:
    """Test cases for in-memory RateLimiter."""

    def test_initial_request_allowed(self):
        """Test that first request is always allowed."""
        limiter = RateLimiter(requests_per_minute=100)
        allowed, retry_after = limiter.check_rate_limit("test_client")

        assert allowed is True
        assert retry_after == 0

    def test_within_limit_allowed(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(requests_per_minute=10)

        for i in range(10):
            allowed, _ = limiter.check_rate_limit("test_client")
            assert allowed is True

    def test_over_limit_blocked(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(requests_per_minute=5)

        # Make 5 requests (allowed)
        for _ in range(5):
            limiter.check_rate_limit("test_client")

        # 6th request should be blocked
        allowed, retry_after = limiter.check_rate_limit("test_client")

        assert allowed is False
        assert retry_after > 0

    def test_different_clients_independent(self):
        """Test that different clients have independent limits."""
        limiter = RateLimiter(requests_per_minute=2)

        # Client A makes 2 requests
        limiter.check_rate_limit("client_a")
        limiter.check_rate_limit("client_a")

        # Client A is now blocked
        allowed_a, _ = limiter.check_rate_limit("client_a")
        assert allowed_a is False

        # Client B should still be allowed
        allowed_b, _ = limiter.check_rate_limit("client_b")
        assert allowed_b is True

    def test_get_stats(self):
        """Test getting rate limit statistics."""
        limiter = RateLimiter(requests_per_minute=100)

        # Make 5 requests
        for _ in range(5):
            limiter.check_rate_limit("test_client")

        stats = limiter.get_stats("test_client")

        assert stats["current_requests"] == 5
        assert stats["limit"] == 100
        assert stats["remaining"] == 95

    def test_old_requests_cleaned_up(self):
        """Test that old requests are removed from tracking."""
        limiter = RateLimiter(requests_per_minute=2)

        # Make 2 requests at current time
        limiter.check_rate_limit("test_client")
        limiter.check_rate_limit("test_client")

        # Simulate time passing by manipulating the internal state
        old_time = time.time() - 120  # 2 minutes ago
        limiter.requests["test_client"] = [old_time, old_time]

        # Next request should be allowed (old ones cleaned up)
        allowed, _ = limiter.check_rate_limit("test_client")
        assert allowed is True


class TestRedisRateLimiter:
    """Test cases for Redis-backed RateLimiter."""

    def test_fallback_when_redis_none(self):
        """Test fallback to in-memory when Redis is None."""
        fallback = RateLimiter(requests_per_minute=10)
        limiter = RedisRateLimiter(
            redis_client=None,
            requests_per_minute=10,
            fallback=fallback
        )

        allowed, _ = limiter.check_rate_limit("test_client")
        assert allowed is True

    def test_fallback_on_redis_error(self):
        """Test fallback when Redis raises an error."""
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = Exception("Redis error")

        fallback = RateLimiter(requests_per_minute=10)
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            requests_per_minute=10,
            fallback=fallback
        )

        allowed, _ = limiter.check_rate_limit("test_client")
        assert allowed is True  # Fallback should work

    def test_redis_pipeline_called(self):
        """Test that Redis pipeline is used for atomic operations."""
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_pipe.execute.return_value = [None, None, 1, None]  # count=1

        fallback = RateLimiter(requests_per_minute=10)
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            requests_per_minute=10,
            fallback=fallback
        )

        allowed, _ = limiter.check_rate_limit("test_client")

        assert allowed is True
        mock_redis.pipeline.assert_called_once()
        mock_pipe.zremrangebyscore.assert_called_once()
        mock_pipe.zadd.assert_called_once()
        mock_pipe.zcard.assert_called_once()
        mock_pipe.expire.assert_called_once()

    def test_rate_limit_exceeded_in_redis(self):
        """Test rate limit exceeded detection in Redis."""
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_pipe.execute.return_value = [None, None, 101, None]  # count > limit

        mock_redis.zrange.return_value = [("timestamp", time.time() - 30)]

        fallback = RateLimiter(requests_per_minute=100)
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            requests_per_minute=100,
            fallback=fallback
        )

        allowed, retry_after = limiter.check_rate_limit("test_client")

        assert allowed is False
        assert retry_after > 0


class TestRateLimitMiddleware:
    """Test cases for RateLimitMiddleware."""

    def test_exempt_paths(self):
        """Test that health/metrics endpoints are exempt."""
        assert "/health" in RateLimitMiddleware.EXEMPT_PATHS
        assert "/ready" in RateLimitMiddleware.EXEMPT_PATHS
        assert "/metrics" in RateLimitMiddleware.EXEMPT_PATHS

    def test_get_client_ip_from_direct_connection(self):
        """Test extracting IP from direct connection."""
        request = MagicMock()
        request.headers.get.return_value = None
        request.client.host = "192.168.1.100"

        with patch("src.utils.ip_extraction.get_settings") as mock_settings:
            mock_settings.return_value.get_trusted_proxies_list.return_value = []
            ip = get_client_ip(request)

        assert ip == "192.168.1.100"

    def test_get_client_ip_from_x_forwarded_for(self):
        """Test extracting IP from X-Forwarded-For header."""
        request = MagicMock()
        request.headers.get.side_effect = lambda h: {
            "X-Forwarded-For": "203.0.113.1, 10.0.0.1",
            "X-Real-IP": None,
        }.get(h)

        with patch("src.utils.ip_extraction.get_settings") as mock_settings:
            mock_settings.return_value.get_trusted_proxies_list.return_value = []
            ip = get_client_ip(request)

        assert ip == "203.0.113.1"

    def test_trusted_proxy_handling(self):
        """Test that trusted proxies are skipped in X-Forwarded-For chain."""
        request = MagicMock()
        # Client -> Trusted Proxy 1 -> Trusted Proxy 2 -> Server
        request.headers.get.side_effect = lambda h: {
            "X-Forwarded-For": "203.0.113.1, 10.0.0.1, 10.0.0.2",
            "X-Real-IP": None,
        }.get(h)

        with patch("src.utils.ip_extraction.get_settings") as mock_settings:
            mock_settings.return_value.get_trusted_proxies_list.return_value = [
                "10.0.0.0/8"  # Trust all 10.x.x.x addresses
            ]
            ip = get_client_ip(request)

        # Should return the first non-trusted IP (working backwards)
        assert ip == "203.0.113.1"

    def test_is_trusted_proxy_single_ip(self):
        """Test checking if single IP is trusted."""
        assert _is_trusted_proxy("10.0.0.1", ["10.0.0.1"])
        assert not _is_trusted_proxy("10.0.0.2", ["10.0.0.1"])

    def test_is_trusted_proxy_cidr(self):
        """Test checking if IP is in trusted CIDR range."""
        # 10.0.0.0/8 should match any 10.x.x.x
        assert _is_trusted_proxy("10.0.0.1", ["10.0.0.0/8"])
        assert _is_trusted_proxy("10.255.255.255", ["10.0.0.0/8"])
        assert not _is_trusted_proxy("192.168.1.1", ["10.0.0.0/8"])

    def test_is_trusted_proxy_invalid_ip(self):
        """Test handling of invalid IP address."""
        # Invalid IP should return False
        assert not _is_trusted_proxy("invalid", ["10.0.0.0/8"])
        assert not _is_trusted_proxy("", ["10.0.0.0/8"])
