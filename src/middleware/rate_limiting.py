"""
Rate limiting middleware for DoS protection.

Implements simple in-memory rate limiting (100 req/min per IP).
For production with multiple replicas, use Redis-backed rate limiting.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter.

    For production with multiple replicas, consider:
    - slowapi (Redis-backed)
    - fastapi-limiter (Redis-backed)
    - Cloud-native rate limiting (API Gateway)
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
            identifier: Client identifier (IP address)

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int)

        Example:
            >>> limiter = RateLimiter(requests_per_minute=100)
            >>> allowed, retry_after = limiter.check_rate_limit("192.168.1.1")
            >>> if not allowed:
            ...     print(f"Rate limited. Retry after {retry_after}s")
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
            Dict with current_requests and limit

        Example:
            >>> limiter.get_stats("192.168.1.1")
            {'current_requests': 15, 'limit': 100, 'remaining': 85}
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


# Global rate limiter instance
# For production: Replace with Redis-backed implementation
rate_limiter = RateLimiter(requests_per_minute=100)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Applies rate limiting to all requests.
    """

    def _get_client_identifier(self, request: Request) -> str:
        """
        Extract client IP address, handling proxy headers.

        Checks X-Forwarded-For and X-Real-IP headers for requests
        behind load balancers or proxies (AWS ALB, Kubernetes ingress, etc.)

        Args:
            request: FastAPI Request object

        Returns:
            Client IP address as string
        """
        # Check X-Forwarded-For first (from proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take first IP (client IP, rest are proxies)
            # Format: "client_ip, proxy1_ip, proxy2_ip"
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header (alternative proxy header)
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection IP
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response (200 if allowed, 429 if rate limited)
        """
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Identify by IP address (handles proxy headers)
        identifier = self._get_client_identifier(request)

        # Check rate limit
        allowed, retry_after = rate_limiter.check_rate_limit(identifier)

        if not allowed:
            # Log rate limit violation
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": identifier,
                    "path": request.url.path,
                    "retry_after": retry_after
                }
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

        # Allow request
        response = await call_next(request)

        # Add rate limit headers to response
        stats = rate_limiter.get_stats(identifier)
        response.headers["X-RateLimit-Limit"] = str(stats["limit"])
        response.headers["X-RateLimit-Remaining"] = str(stats["remaining"])

        return response


def get_rate_limiter() -> RateLimiter:
    """
    Get global rate limiter instance.

    Returns:
        RateLimiter instance

    Example:
        >>> limiter = get_rate_limiter()
        >>> limiter.get_stats("192.168.1.1")
    """
    return rate_limiter
