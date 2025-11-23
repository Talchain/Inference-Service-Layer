"""
Configuration management for Inference Service Layer.

Handles all environment variables and application settings.
"""

import logging
import sys
from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from pythonjsonlogger import jsonlogger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

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
