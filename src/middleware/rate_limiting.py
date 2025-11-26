"""
Rate limiting middleware for DoS protection.

Implements distributed rate limiting using Redis with in-memory fallback.
Supports proxy-aware IP detection and per-API-key rate limiting.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings
from src.utils.secure_logging import get_security_audit_logger

logger = logging.getLogger(__name__)
security_audit = get_security_audit_logger()

# Prometheus metrics for rate limiting
rate_limit_hits = Counter(
    "isl_rate_limit_hits_total",
    "Total number of rate limit violations",
    ["identifier_type"],  # "ip" or "api_key"
)
rate_limit_checks = Counter(
    "isl_rate_limit_checks_total",
    "Total number of rate limit checks",
    ["result"],  # "allowed" or "blocked"
)


class RateLimiter:
    """
    In-memory rate limiter using sliding window.

    For production with multiple replicas, use RedisRateLimiter.
    """

    def __init__(self, requests_per_minute: int = 100):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per identifier
        """
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def check_rate_limit(self, identifier: str) -> Tuple[bool, int]:
        """
        Check if request is within rate limit.

        Args:
            identifier: Client identifier (IP address or API key)

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int)
        """
        now = time.time()
        minute_ago = now - 60

        # Remove old requests (older than 1 minute)
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > minute_ago
        ]

        # Check if over limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            # Calculate retry_after based on oldest request
            oldest = self.requests[identifier][0]
            retry_after = int(60 - (now - oldest)) + 1  # Add 1 for safety
            return False, retry_after

        # Allow request
        self.requests[identifier].append(now)
        return True, 0

    def get_stats(self, identifier: str) -> Dict[str, int]:
        """
        Get rate limiting stats for identifier.

        Args:
            identifier: Client identifier

        Returns:
            Dict with current_requests, limit, and remaining
        """
        now = time.time()
        minute_ago = now - 60

        # Count requests in last minute
        current_requests = sum(
            1 for req_time in self.requests[identifier]
            if req_time > minute_ago
        )

        return {
            "current_requests": current_requests,
            "limit": self.requests_per_minute,
            "remaining": max(0, self.requests_per_minute - current_requests)
        }


class RedisRateLimiter:
    """
    Redis-backed rate limiter using sliding window algorithm.

    Provides distributed rate limiting that works across multiple replicas.
    Falls back to in-memory rate limiting if Redis is unavailable.
    """

    def __init__(
        self,
        redis_client,
        requests_per_minute: int = 100,
        fallback: Optional[RateLimiter] = None
    ):
        """
        Initialize Redis rate limiter.

        Args:
            redis_client: Redis client instance (can be None)
            requests_per_minute: Maximum requests per minute per identifier
            fallback: Fallback in-memory rate limiter
        """
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.fallback = fallback or RateLimiter(requests_per_minute)
        self.key_prefix = "isl:ratelimit:"
        self.window_seconds = 60

    def check_rate_limit(self, identifier: str) -> Tuple[bool, int]:
        """
        Check if request is within rate limit using Redis sorted sets.

        Uses sliding window algorithm with Redis ZADD/ZREMRANGEBYSCORE.

        Args:
            identifier: Client identifier (IP or API key)

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int)
        """
        if not self.redis:
            return self.fallback.check_rate_limit(identifier)

        try:
            key = f"{self.key_prefix}{identifier}"
            now = time.time()
            window_start = now - self.window_seconds

            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)

            # Add current request
            pipe.zadd(key, {str(now): now})

            # Count requests in window
            pipe.zcard(key)

            # Set TTL to auto-expire
            pipe.expire(key, self.window_seconds + 1)

            results = pipe.execute()
            count = results[2]

            if count > self.requests_per_minute:
                # Over limit - calculate retry_after
                # Get oldest entry to determine when window clears
                oldest = self.redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(self.window_seconds - (now - oldest[0][1])) + 1
                else:
                    retry_after = self.window_seconds
                return False, max(1, retry_after)

            return True, 0

        except Exception as e:
            logger.warning(
                f"Redis rate limit check failed, using fallback: {e}",
                extra={"identifier": identifier[:16] + "..." if len(identifier) > 16 else identifier}
            )
            return self.fallback.check_rate_limit(identifier)

    def get_stats(self, identifier: str) -> Dict[str, int]:
        """
        Get rate limiting stats for identifier.

        Args:
            identifier: Client identifier

        Returns:
            Dict with current_requests, limit, and remaining
        """
        if not self.redis:
            return self.fallback.get_stats(identifier)

        try:
            key = f"{self.key_prefix}{identifier}"
            now = time.time()
            window_start = now - self.window_seconds

            # Clean and count in pipeline
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = pipe.execute()

            current_requests = results[1]

            return {
                "current_requests": current_requests,
                "limit": self.requests_per_minute,
                "remaining": max(0, self.requests_per_minute - current_requests)
            }
        except Exception:
            return self.fallback.get_stats(identifier)


# Global rate limiter instance
# Will be initialized with Redis client if available
_rate_limiter: Optional[RedisRateLimiter] = None
_in_memory_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RedisRateLimiter:
    """
    Get the global rate limiter instance.

    Initializes with Redis if available, otherwise uses in-memory fallback.

    Returns:
        RedisRateLimiter instance
    """
    global _rate_limiter, _in_memory_limiter

    if _rate_limiter is None:
        settings = get_settings()
        _in_memory_limiter = RateLimiter(settings.RATE_LIMIT_REQUESTS_PER_MINUTE)

        # Try to get Redis client
        try:
            from src.infrastructure.redis_client import get_redis_client
            redis_client = get_redis_client()
            _rate_limiter = RedisRateLimiter(
                redis_client=redis_client,
                requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                fallback=_in_memory_limiter
            )
            if redis_client:
                logger.info("Rate limiter using Redis backend")
            else:
                logger.info("Rate limiter using in-memory backend (Redis unavailable)")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis rate limiter: {e}")
            _rate_limiter = RedisRateLimiter(
                redis_client=None,
                requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                fallback=_in_memory_limiter
            )

    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Applies rate limiting to all requests using either IP address or API key.
    """

    # Endpoints exempt from rate limiting
    EXEMPT_PATHS: Set[str] = {"/health", "/ready", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response (200 if allowed, 429 if rate limited)
        """
        # Skip rate limiting for exempt endpoints
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Get rate limiter
        limiter = get_rate_limiter()

        # Determine identifier (prefer API key over IP)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            identifier = f"key:{api_key[:16]}"  # Truncate for safety
            identifier_type = "api_key"
        else:
            identifier = f"ip:{self._get_client_ip(request)}"
            identifier_type = "ip"

        # Check rate limit
        allowed, retry_after = limiter.check_rate_limit(identifier)

        if not allowed:
            # Record metrics
            rate_limit_hits.labels(identifier_type=identifier_type).inc()
            rate_limit_checks.labels(result="blocked").inc()

            # Get client IP for audit logging
            client_ip = self._get_client_ip(request)

            # Log rate limit violation via security audit logger
            settings = get_settings()
            security_audit.log_rate_limit_exceeded(
                client_ip=client_ip,
                identifier=identifier,
                limit=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                window_seconds=60,
                path=request.url.path,
            )

            # Return 429 Too Many Requests
            return JSONResponse(
                status_code=429,
                content={
                    "schema": "error.v1",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please wait before trying again.",
                    "retry_after": retry_after,
                    "suggested_action": "retry_later"
                },
                headers={"Retry-After": str(retry_after)}
            )

        # Record allowed request
        rate_limit_checks.labels(result="allowed").inc()

        # Allow request
        response = await call_next(request)

        # Add rate limit headers to response
        stats = limiter.get_stats(identifier)
        response.headers["X-RateLimit-Limit"] = str(stats["limit"])
        response.headers["X-RateLimit-Remaining"] = str(stats["remaining"])

        return response

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address, respecting proxy headers.

        Supports X-Forwarded-For and X-Real-IP headers from trusted proxies.

        Args:
            request: The incoming request

        Returns:
            Client IP address
        """
        settings = get_settings()
        trusted_proxies = settings.get_trusted_proxies_list()

        # Check X-Forwarded-For header (set by proxies/load balancers)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2
            # Parse the chain and find the first non-trusted IP
            ips = [ip.strip() for ip in forwarded_for.split(",")]

            # If we have trusted proxies configured, walk back through the chain
            if trusted_proxies:
                for ip in reversed(ips):
                    if not self._is_trusted_proxy(ip, trusted_proxies):
                        return ip
                # All IPs are trusted, return the first one (original client)
                return ips[0]
            else:
                # No trusted proxies configured, return the first IP
                return ips[0]

        # Check X-Real-IP header (set by nginx)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _is_trusted_proxy(self, ip: str, trusted_proxies: List[str]) -> bool:
        """
        Check if an IP is in the trusted proxy list.

        Args:
            ip: IP address to check
            trusted_proxies: List of trusted proxy IPs or CIDRs

        Returns:
            True if IP is trusted, False otherwise
        """
        import ipaddress

        try:
            check_ip = ipaddress.ip_address(ip)

            for proxy in trusted_proxies:
                try:
                    # Check if it's a CIDR range
                    if "/" in proxy:
                        network = ipaddress.ip_network(proxy, strict=False)
                        if check_ip in network:
                            return True
                    else:
                        # Single IP
                        if check_ip == ipaddress.ip_address(proxy):
                            return True
                except ValueError:
                    continue

            return False
        except ValueError:
            return False
