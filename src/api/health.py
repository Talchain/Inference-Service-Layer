"""
Health check endpoint.

Provides a simple health check for monitoring and load balancers.
"""

from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter

from src.__version__ import __version__
from src.config import get_settings
from src.infrastructure.memory_cache import get_memory_cache
from src.models.metadata import generate_config_fingerprint
from src.models.responses import HealthResponse
from src.utils.error_recovery import health_monitor

router = APIRouter()
settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status for monitoring and load balancers.",
    responses={
        200: {"description": "Service is healthy"},
    },
)
async def health_check() -> HealthResponse:
    """
    Check service health.

    Returns:
        HealthResponse: Health status with version and timestamp
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow().isoformat() + "Z",
        config_fingerprint=generate_config_fingerprint(),
    )


@router.get(
    "/cache/stats",
    response_model=Dict[str, Any],
    summary="Cache statistics",
    description="Returns in-memory cache performance statistics for monitoring.",
    responses={
        200: {"description": "Cache statistics"},
    },
)
async def cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns cache performance metrics including:
    - Hit/miss counts and rates
    - Current cache size
    - Eviction and expiration counts

    Returns:
        Dict with cache statistics
    """
    cache = get_memory_cache()
    stats = cache.get_stats()

    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "hit_rate_percent": round(stats.hit_rate, 2),
        "total_requests": stats.total_requests,
        "evictions": stats.evictions,
        "expirations": stats.expirations,
        "current_size": stats.current_size,
        "max_size": stats.max_size,
        "utilization_percent": round(
            (stats.current_size / stats.max_size * 100) if stats.max_size > 0 else 0, 2
        ),
    }


@router.get(
    "/health/services",
    response_model=Dict[str, Any],
    summary="Service health status",
    description="Returns health status of all monitored services for error recovery monitoring.",
    responses={
        200: {"description": "Service health information"},
    },
)
async def service_health() -> Dict[str, Any]:
    """
    Get health status of all monitored services.

    Returns health metrics for each service including:
    - Overall health status (HEALTHY, DEGRADED, FAILING)
    - Success rate percentage
    - Total requests, successes, failures, fallbacks
    - Uptime and last check timestamp

    Returns:
        Dict with service health information
    """
    services = [
        "conformal_prediction",
        "advanced_discovery",
        "validation_suggester",
        "path_analysis",
    ]

    service_status = {}
    overall_healthy = True

    for service_name in services:
        health = health_monitor.get_health(service_name)

        service_status[service_name] = {
            "status": health.status.value,
            "success_rate_percent": round(health.success_rate * 100, 2),
            "total_requests": health.total_requests,
            "successes": health.successes,
            "failures": health.failures,
            "fallbacks": health.fallbacks,
            "uptime_seconds": round(health.uptime, 2),
            "last_check": health.last_check.isoformat() + "Z" if health.last_check else None,
        }

        # Mark overall as unhealthy if any service is failing
        if health.status.value == "FAILING":
            overall_healthy = False

    return {
        "overall_status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": service_status,
    }


@router.get(
    "/health/circuit-breakers",
    response_model=Dict[str, Any],
    summary="Circuit breaker status",
    description="Returns status of all circuit breakers for expensive operations monitoring.",
    responses={
        200: {"description": "Circuit breaker information"},
    },
)
async def circuit_breaker_status() -> Dict[str, Any]:
    """
    Get status of all circuit breakers.

    Returns circuit breaker state for each protected operation including:
    - State (CLOSED, OPEN, HALF_OPEN)
    - Failure count
    - Last failure time
    - Timeout configuration

    This helps operators understand which operations are experiencing
    repeated failures and may need intervention.

    Returns:
        Dict with circuit breaker status
    """
    # Import circuit breakers
    from src.services.causal_discovery_engine import _notears_breaker, _pc_breaker
    from src.services.advanced_validation_suggester import (
        _path_analysis_breaker,
        _strategy_generation_breaker,
    )

    circuit_breakers = {
        "notears_discovery": _notears_breaker,
        "pc_discovery": _pc_breaker,
        "path_analysis": _path_analysis_breaker,
        "strategy_generation": _strategy_generation_breaker,
    }

    breaker_status = {}
    any_open = False

    for name, breaker in circuit_breakers.items():
        breaker_status[name] = {
            "state": breaker.state.value,
            "failure_count": breaker.failure_count,
            "success_count": breaker.success_count,
            "failure_threshold": breaker.failure_threshold,
            "success_threshold": breaker.success_threshold,
            "timeout_seconds": breaker.timeout,
            "last_failure_time": (
                breaker.last_failure_time.isoformat() + "Z"
                if breaker.last_failure_time
                else None
            ),
        }

        if breaker.state.value == "OPEN":
            any_open = True

    return {
        "overall_status": "operational" if not any_open else "some_circuits_open",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "circuit_breakers": breaker_status,
        "explanation": (
            "CLOSED = Normal operation, OPEN = Repeated failures (circuit tripped), "
            "HALF_OPEN = Testing recovery"
        ),
    }
