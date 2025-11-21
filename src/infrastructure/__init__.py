"""Infrastructure utilities for ISL."""

from .redis_client import get_redis_client

__all__ = ["get_redis_client"]
