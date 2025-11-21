"""
Health check endpoint.

Provides a simple health check for monitoring and load balancers.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter

from src.__version__ import __version__
from src.config import get_settings
from src.infrastructure.memory_cache import get_memory_cache
from src.models.metadata import generate_config_fingerprint
from src.models.responses import HealthResponse

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
