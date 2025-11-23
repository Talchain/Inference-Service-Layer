"""
Response metadata for determinism and reproducibility.
"""

import hashlib
import json
import os
from typing import Any, Dict

from pydantic import BaseModel, Field

from src.__version__ import __version__
from src.config import get_settings


class ResponseMetadata(BaseModel):
    """
    Metadata attached to all ISL responses for reproducibility.

    Enables PLoT to:
    - Verify same ISL version across reruns
    - Detect config changes that affect determinism
    - Audit what configuration produced results
    """

    isl_version: str = Field(
        description="ISL service version (semver)",
        json_schema_extra={"example": "1.0.0"}
    )

    config_fingerprint: str = Field(
        description="Hash of determinism-critical config",
        json_schema_extra={"example": "a3f8c9d2e1b4"}
    )

    config_details: Dict[str, Any] = Field(
        description="Key configuration values",
        json_schema_extra={
            "example": {
                "monte_carlo_samples": 10000,
                "confidence_level": 0.95,
                "learning_rate": 0.1,
            }
        }
    )

    request_id: str = Field(
        description="Request ID for tracing",
        json_schema_extra={"example": "req_abc123"}
    )


def generate_config_fingerprint() -> str:
    """
    Generate deterministic hash of configuration.

    Only includes settings that affect computational results.
    Changes to logging, monitoring, etc. don't change fingerprint.

    Returns:
        12-character hex hash of config
    """
    settings = get_settings()

    # Only include determinism-critical settings
    config_snapshot = {
        # Monte Carlo settings
        "max_monte_carlo_iterations": settings.MAX_MONTE_CARLO_ITERATIONS,
        "confidence_level": settings.DEFAULT_CONFIDENCE_LEVEL,
        # Learning parameters (hardcoded in BeliefUpdater)
        "learning_rate": 0.1,
        "uncertainty_reduction": 0.9,
        # Validation thresholds (hardcoded in validators)
        "complexity_min_nodes": 5,
        "complexity_max_nodes": 20,
        # Teaching parameters (hardcoded in BayesianTeacher)
        "teaching_kl_threshold": 0.1,
        # Deterministic mode
        "deterministic_mode": settings.ENABLE_DETERMINISTIC_MODE,
        # Random seed (if set globally)
        "global_seed": os.getenv("RANDOM_SEED"),
    }

    # Sort keys for consistent hashing
    config_json = json.dumps(config_snapshot, sort_keys=True)
    hash_obj = hashlib.sha256(config_json.encode())

    return hash_obj.hexdigest()[:12]


def generate_config_details() -> Dict[str, Any]:
    """
    Extract key configuration values for transparency.

    Returns detailed config without sensitive info.
    """
    settings = get_settings()

    # Check if Redis is available
    redis_enabled = False
    try:
        import redis

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        r = redis.Redis(host=redis_host, port=redis_port, socket_connect_timeout=1)
        r.ping()
        redis_enabled = True
    except Exception:
        pass

    return {
        "monte_carlo_samples": settings.MAX_MONTE_CARLO_ITERATIONS,
        "confidence_level": settings.DEFAULT_CONFIDENCE_LEVEL,
        "response_timeout": settings.RESPONSE_TIMEOUT_SECONDS,
        "redis_enabled": redis_enabled,
        "deterministic_mode": settings.ENABLE_DETERMINISTIC_MODE,
    }


def create_response_metadata(request_id: str) -> ResponseMetadata:
    """
    Create metadata for response.

    Args:
        request_id: Request ID from X-Request-Id header

    Returns:
        ResponseMetadata with version, fingerprint, config
    """
    return ResponseMetadata(
        isl_version=__version__,
        config_fingerprint=generate_config_fingerprint(),
        config_details=generate_config_details(),
        request_id=request_id,
    )
