"""
ISL response metadata models.

Provides standard metadata for all ISL responses to enable
tracing, debugging, and performance monitoring.
"""

from typing import Optional
from pydantic import BaseModel, Field
import time


class ISLResponseMetadata(BaseModel):
    """
    Standard metadata included in all ISL responses.

    Attributes:
        request_id: Unique identifier for request tracing
        computation_time_ms: Time taken for computation in milliseconds
        isl_version: ISL service version
        algorithm: Algorithm/method used for computation
        cache_hit: Whether result was served from cache
    """

    request_id: str = Field(
        ...,
        description="Unique request identifier for tracing",
        min_length=1,
        max_length=100
    )

    computation_time_ms: float = Field(
        ...,
        description="Computation time in milliseconds",
        ge=0.0
    )

    isl_version: str = Field(
        default="2.0",
        description="ISL service version"
    )

    algorithm: Optional[str] = Field(
        None,
        description="Algorithm/method used (e.g., 'weighted_sum', 'skyline_pareto')",
        max_length=100
    )

    cache_hit: bool = Field(
        default=False,
        description="Whether result was served from cache"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_a1b2c3d4e5f6",
                "computation_time_ms": 123.45,
                "isl_version": "2.0",
                "algorithm": "weighted_sum",
                "cache_hit": False
            }
        }


class MetadataBuilder:
    """Helper class for building metadata objects."""

    def __init__(self, request_id: str):
        """
        Initialize metadata builder.

        Args:
            request_id: Request identifier
        """
        self.request_id = request_id
        self.start_time = time.perf_counter()

    def build(
        self,
        algorithm: Optional[str] = None,
        cache_hit: bool = False
    ) -> ISLResponseMetadata:
        """
        Build metadata object with computed timing.

        Args:
            algorithm: Algorithm used for computation
            cache_hit: Whether result was cached

        Returns:
            ISLResponseMetadata object
        """
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        return ISLResponseMetadata(
            request_id=self.request_id,
            computation_time_ms=elapsed_ms,
            algorithm=algorithm,
            cache_hit=cache_hit
        )


def create_isl_metadata(
    request_id: str,
    computation_time_ms: float,
    algorithm: Optional[str] = None,
    cache_hit: bool = False
) -> ISLResponseMetadata:
    """
    Create an ISL response metadata object.

    Args:
        request_id: Request identifier for tracing
        computation_time_ms: Computation time in milliseconds
        algorithm: Algorithm used for computation
        cache_hit: Whether result was cached

    Returns:
        ISLResponseMetadata object
    """
    return ISLResponseMetadata(
        request_id=request_id,
        computation_time_ms=computation_time_ms,
        algorithm=algorithm,
        cache_hit=cache_hit
    )
