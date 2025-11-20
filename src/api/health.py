"""
Health check endpoint.

Provides a simple health check for monitoring and load balancers.
"""

from datetime import datetime

from fastapi import APIRouter

from src.__version__ import __version__
from src.config import get_settings
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
