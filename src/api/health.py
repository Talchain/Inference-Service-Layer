"""
Health check endpoint.

Provides a simple health check for monitoring and load balancers.
"""

from datetime import datetime

from fastapi import APIRouter

from src.config import get_settings
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
        version=settings.VERSION,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
