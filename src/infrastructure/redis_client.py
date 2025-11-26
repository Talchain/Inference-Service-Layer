"""
Redis client for ISL infrastructure.

Provides centralized Redis connection management with:
- TLS/SSL support for secure connections
- Password authentication
- Connection pooling with configurable limits
- Automatic retry with exponential backoff
- Health monitoring and metrics
"""

import logging
import os
import ssl
from functools import lru_cache
from typing import Optional

import redis
from prometheus_client import Counter, Gauge
from redis.backoff import ExponentialBackoff
from redis.retry import Retry

from src.config import get_settings

logger = logging.getLogger(__name__)

# Prometheus metrics for Redis client
# Note: Using unique names to avoid collision with metrics.py
redis_client_ops = Counter(
    "isl_redis_client_ops_total",
    "Total Redis client operations",
    ["operation", "status"],  # status: "success" or "error"
)
redis_client_status = Gauge(
    "isl_redis_client_connected",
    "Redis client connection status (1=connected, 0=disconnected)",
)


class RedisHealthCheck:
    """Health check wrapper for Redis client."""

    def __init__(self, client: Optional[redis.Redis]):
        self.client = client
        self._last_check_success = False

    def is_healthy(self) -> bool:
        """
        Check if Redis is healthy and responding.

        Returns:
            True if Redis is responding to PING, False otherwise
        """
        if not self.client:
            self._last_check_success = False
            redis_client_status.set(0)
            return False

        try:
            self.client.ping()
            self._last_check_success = True
            redis_client_status.set(1)
            redis_client_ops.labels(operation="ping", status="success").inc()
            return True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis health check failed: {e}")
            self._last_check_success = False
            redis_client_status.set(0)
            redis_client_ops.labels(operation="ping", status="error").inc()
            return False
        except Exception as e:
            logger.error(f"Redis health check error: {e}")
            self._last_check_success = False
            redis_client_status.set(0)
            redis_client_ops.labels(operation="ping", status="error").inc()
            return False

    def get_info(self) -> dict:
        """
        Get Redis server info for diagnostics.

        Returns:
            Dict with Redis info or error details
        """
        if not self.client:
            return {"status": "disconnected", "error": "No client configured"}

        try:
            info = self.client.info("server")
            return {
                "status": "connected",
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global Redis client and health check
_redis_client: Optional[redis.Redis] = None
_redis_health: Optional[RedisHealthCheck] = None


def _create_ssl_context() -> Optional[ssl.SSLContext]:
    """
    Create SSL context for Redis TLS connections.

    Returns:
        SSL context if TLS is enabled, None otherwise
    """
    settings = get_settings()

    if not settings.REDIS_TLS_ENABLED:
        return None

    ssl_context = ssl.create_default_context()

    # For production: verify certificates
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    # Optional: Load custom CA certificate
    ca_cert_path = os.getenv("REDIS_CA_CERT_PATH")
    if ca_cert_path:
        ssl_context.load_verify_locations(ca_cert_path)

    # Optional: Load client certificate for mTLS
    client_cert_path = os.getenv("REDIS_CLIENT_CERT_PATH")
    client_key_path = os.getenv("REDIS_CLIENT_KEY_PATH")
    if client_cert_path and client_key_path:
        ssl_context.load_cert_chain(client_cert_path, client_key_path)

    return ssl_context


def _create_retry_policy() -> Retry:
    """
    Create retry policy with exponential backoff.

    Returns:
        Retry instance configured for resilient connections
    """
    return Retry(
        backoff=ExponentialBackoff(cap=10, base=0.1),  # Max 10s between retries
        retries=3,  # Retry up to 3 times
    )


@lru_cache()
def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client instance (cached).

    Creates a Redis client with:
    - TLS/SSL if REDIS_TLS_ENABLED=true
    - Password authentication if REDIS_PASSWORD is set
    - Connection pooling with configurable limits
    - Automatic retry with exponential backoff
    - Health check interval

    Returns:
        Redis client instance, or None if connection fails
    """
    global _redis_client

    settings = get_settings()

    try:
        # Create SSL context if TLS is enabled
        ssl_context = _create_ssl_context()

        # Create retry policy
        retry = _create_retry_policy()

        # Build connection parameters
        connection_kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "db": settings.REDIS_DB,
            "decode_responses": True,
            "socket_connect_timeout": 5,
            "socket_timeout": 10,
            "retry": retry,
            "retry_on_timeout": True,
            "health_check_interval": 30,
            "max_connections": settings.REDIS_MAX_CONNECTIONS,
        }

        # Add password if configured
        if settings.REDIS_PASSWORD:
            connection_kwargs["password"] = settings.REDIS_PASSWORD

        # Add SSL if configured
        if ssl_context:
            connection_kwargs["ssl"] = True
            connection_kwargs["ssl_context"] = ssl_context
            logger.info("Redis TLS enabled")

        # Create client
        _redis_client = redis.Redis(**connection_kwargs)

        # Test connection
        _redis_client.ping()

        logger.info(
            "Redis connection established",
            extra={
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "tls_enabled": ssl_context is not None,
            }
        )

        redis_client_status.set(1)
        redis_client_ops.labels(operation="connect", status="success").inc()

        return _redis_client

    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(
            f"Redis not available (connection): {e}",
            extra={
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
            }
        )
        redis_client_status.set(0)
        redis_client_ops.labels(operation="connect", status="error").inc()
        return None

    except redis.AuthenticationError as e:
        logger.error(
            f"Redis authentication failed: {e}",
            extra={
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
            }
        )
        redis_client_status.set(0)
        redis_client_ops.labels(operation="connect", status="error").inc()
        return None

    except Exception as e:
        logger.error(
            f"Redis connection error: {e}",
            extra={
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
            }
        )
        redis_client_status.set(0)
        redis_client_ops.labels(operation="connect", status="error").inc()
        return None


def get_redis_health() -> RedisHealthCheck:
    """
    Get Redis health check instance.

    Returns:
        RedisHealthCheck instance for monitoring Redis status
    """
    global _redis_health

    if _redis_health is None:
        client = get_redis_client()
        _redis_health = RedisHealthCheck(client)

    return _redis_health


def clear_redis_cache() -> None:
    """
    Clear the cached Redis client.

    Useful for testing or when reconnection is needed.
    """
    global _redis_client, _redis_health
    get_redis_client.cache_clear()
    _redis_client = None
    _redis_health = None
