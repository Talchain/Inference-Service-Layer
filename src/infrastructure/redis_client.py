"""
Redis client for ISL infrastructure.

Provides centralized Redis connection management.
"""

import logging
import os
from functools import lru_cache
from typing import Optional

import redis

logger = logging.getLogger(__name__)


@lru_cache()
def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client instance (cached).

    Returns None if Redis is not available.
    """
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
        )

        # Test connection
        client.ping()

        logger.info("Redis connection established")
        return client

    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(f"Redis not available: {e}")
        return None

    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        return None
