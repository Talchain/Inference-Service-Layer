"""
Circuit breaker middleware for memory and health protection.

Stops accepting requests when system resources are constrained.
"""

import logging
from typing import Callable

import psutil
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class MemoryCircuitBreaker(BaseHTTPMiddleware):
    """
    Circuit breaker that rejects requests when memory usage is high.

    This prevents cascading failures due to memory exhaustion.
    """

    def __init__(self, app, threshold_percent: float = 85.0):
        """
        Initialize circuit breaker.

        Args:
            app: FastAPI app
            threshold_percent: Memory usage threshold (0-100)
        """
        super().__init__(app)
        self.threshold = threshold_percent
        self.last_log_time = 0

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Check memory before processing request.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response (503 if circuit open, normal response otherwise)
        """
        # Skip circuit breaker for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Check system memory
        mem = psutil.virtual_memory()

        if mem.percent > self.threshold:
            # Extract request ID for correlation
            from src.utils.tracing import get_trace_id
            request_id = request.headers.get("X-Request-Id") or request.headers.get("X-Trace-Id") or get_trace_id()

            # Log periodically (not on every request to avoid log spam)
            import time
            now = time.time()
            if now - self.last_log_time > 10:  # Log every 10 seconds
                logger.warning(
                    "circuit_breaker_open",
                    extra={
                        "memory_percent": mem.percent,
                        "threshold": self.threshold,
                        "available_mb": mem.available / 1024 / 1024,
                        "request_id": request_id,
                    }
                )
                self.last_log_time = now

            # Return 503 Service Unavailable with Olumi Error Schema v1.0
            from src.models.responses import ErrorCode, ErrorResponse, RecoveryHints

            error_response = ErrorResponse(
                code=ErrorCode.SERVICE_UNAVAILABLE.value,
                message=f"Service temporarily unavailable due to high memory usage ({round(mem.percent, 1)}%)",
                reason="memory_circuit_breaker_open",
                recovery=RecoveryHints(
                    hints=[
                        "Wait 30 seconds before retrying",
                        "Simplify your request to reduce memory usage",
                        "Consider reducing batch sizes or complexity"
                    ],
                    suggestion="Retry after 30 seconds when memory usage decreases",
                ),
                retryable=True,
                source="isl",
                request_id=request_id,
            )

            return JSONResponse(
                status_code=503,
                content=error_response.model_dump(exclude_none=True),
                headers={"Retry-After": "30"}
            )

        # Memory OK, process request normally
        return await call_next(request)


class HealthCircuitBreaker(BaseHTTPMiddleware):
    """
    Circuit breaker that checks overall system health.

    Rejects requests when dependencies are unhealthy.
    """

    def __init__(self, app, redis_client=None):
        """
        Initialize health circuit breaker.

        Args:
            app: FastAPI app
            redis_client: Optional Redis client to check connectivity
        """
        super().__init__(app)
        self.redis_client = redis_client
        self.last_health_check = 0
        self.is_healthy = True

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Check health before processing request.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response (503 if unhealthy, normal response otherwise)
        """
        # Skip circuit breaker for health endpoints
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Periodic health check (every 30 seconds)
        import time
        now = time.time()
        if now - self.last_health_check > 30:
            self.is_healthy = self._check_health()
            self.last_health_check = now

        if not self.is_healthy:
            # Extract request ID for correlation
            from src.utils.tracing import get_trace_id
            request_id = request.headers.get("X-Request-Id") or request.headers.get("X-Trace-Id") or get_trace_id()

            logger.warning(
                "circuit_breaker_health_check_failed",
                extra={"request_id": request_id}
            )

            # Return 503 with Olumi Error Schema v1.0
            from src.models.responses import ErrorCode, ErrorResponse, RecoveryHints

            error_response = ErrorResponse(
                code=ErrorCode.SERVICE_UNAVAILABLE.value,
                message="Service temporarily unavailable due to health check failure",
                reason="health_circuit_breaker_open",
                recovery=RecoveryHints(
                    hints=[
                        "Wait 30 seconds before retrying",
                        "Check service status page",
                        "Contact support if issue persists"
                    ],
                    suggestion="Retry after 30 seconds",
                ),
                retryable=True,
                source="isl",
                request_id=request_id,
            )

            return JSONResponse(
                status_code=503,
                content=error_response.model_dump(exclude_none=True),
                headers={"Retry-After": "30"}
            )

        # Health OK, process request
        return await call_next(request)

    def _check_health(self) -> bool:
        """
        Check system health.

        Returns:
            True if healthy, False otherwise
        """
        # Check Redis connectivity if client available
        if self.redis_client:
            try:
                self.redis_client.ping()
            except Exception as e:
                logger.error(
                    "redis_health_check_failed",
                    extra={"error": str(e)}
                )
                # Don't fail the circuit breaker for Redis (graceful degradation)
                # But log the issue
                pass

        # Add other health checks here (database, external APIs, etc.)

        return True  # System is healthy
