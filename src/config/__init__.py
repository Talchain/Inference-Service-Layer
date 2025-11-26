"""
Configuration management for Inference Service Layer.

Handles all environment variables and application settings.
"""

import logging
import sys
from functools import lru_cache
from typing import List, Optional

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings
from pythonjsonlogger import jsonlogger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment Configuration
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment: development, staging, or production"
    )

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Olumi Inference Service Layer"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Deterministic scientific computation core for decision enhancement"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 1

    # Logging
    LOG_LEVEL: str = "INFO"

    # Authentication
    # Supports both ISL_API_KEYS (preferred) and ISL_API_KEY (legacy) for backward compatibility
    ISL_API_KEYS: Optional[str] = Field(
        default=None,
        description="Comma-separated list of valid API keys for authentication"
    )
    ISL_API_KEY: Optional[str] = Field(
        default=None,
        description="Legacy: Single API key for authentication (use ISL_API_KEYS instead)"
    )

    # CORS Configuration
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Comma-separated list of allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=False,
        description="Allow credentials in CORS requests"
    )

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=100,
        description="Maximum requests per minute per client"
    )
    TRUSTED_PROXIES: str = Field(
        default="",
        description="Comma-separated list of trusted proxy IPs/CIDRs"
    )

    # Redis Configuration
    REDIS_HOST: str = Field(default="localhost", description="Redis server hostname")
    REDIS_PORT: int = Field(default=6379, description="Redis server port")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_TLS_ENABLED: bool = Field(default=False, description="Enable TLS for Redis")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, description="Max Redis connections")

    # Request Limits
    MAX_REQUEST_SIZE_MB: int = Field(
        default=10,
        description="Maximum request body size in megabytes"
    )
    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=60,
        description="Maximum time for request processing"
    )

    # Computation Settings
    DEFAULT_CONFIDENCE_LEVEL: float = 0.95
    MAX_MONTE_CARLO_ITERATIONS: int = 10000
    RESPONSE_TIMEOUT_SECONDS: int = 30

    # FACET Configuration
    FACET_ENABLED: bool = True
    FACET_ROBUSTNESS_CHECKS: int = 100

    # Feature Flags
    TEAM_ALIGNMENT_ENABLED: bool = True
    SENSITIVITY_ANALYSIS_ENABLED: bool = True

    # Determinism
    ENABLE_DETERMINISTIC_MODE: bool = True

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True
    )

    @field_validator("ISL_API_KEYS")
    @classmethod
    def validate_api_keys_in_production(cls, v, info):
        """Require API keys in production environment."""
        # Note: info.data may not have ENVIRONMENT yet during validation
        # This is a soft validation; hard validation happens in startup check
        return v

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v, info):
        """Validate CORS origins don't contain wildcards in production."""
        if "*" in v:
            # Get environment from info.data if available
            env = info.data.get("ENVIRONMENT", "development") if info.data else "development"
            if env == "production":
                raise ValueError("Wildcard CORS origins not allowed in production")
        return v

    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        if not self.CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def get_trusted_proxies_list(self) -> List[str]:
        """Get trusted proxies as a list."""
        if not self.TRUSTED_PROXIES:
            return []
        return [proxy.strip() for proxy in self.TRUSTED_PROXIES.split(",") if proxy.strip()]

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    def validate_production_config(self) -> List[str]:
        """
        Validate configuration for production environment.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.is_production():
            return errors

        # Required settings in production
        if not self.ISL_API_KEYS and not self.ISL_API_KEY:
            errors.append("ISL_API_KEYS (or ISL_API_KEY) is required in production")

        if "*" in self.CORS_ORIGINS:
            errors.append("Wildcard CORS origins not allowed in production")

        if not self.CORS_ORIGINS or self.CORS_ORIGINS == "http://localhost:3000,http://localhost:8080":
            errors.append("CORS_ORIGINS should be configured for production domains")

        return errors


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application configuration
    """
    return Settings()


def setup_logging() -> logging.Logger:
    """
    Configure structured JSON logging.

    Returns:
        logging.Logger: Configured root logger
    """
    settings = get_settings()

    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL)

    # Remove existing handlers
    logger.handlers = []

    # Create stdout handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
