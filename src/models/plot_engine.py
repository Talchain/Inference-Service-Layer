"""
PLoT Engine request/response models.

Models for interfacing with the PLoT (Platform for Learning over Time) engine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceFreshness:
    """
    Evidence freshness metrics for model card.

    Tracks how recent the evidence supporting the model is.
    """

    total: int
    with_timestamp: int
    oldest_days: Optional[int] = None
    newest_days: Optional[int] = None
    buckets: Dict[str, int] = field(default_factory=dict)  # FRESH, AGING, STALE, UNKNOWN

    def __post_init__(self):
        """Validate evidence freshness data."""
        if self.total < 0:
            raise ValueError("total must be non-negative")
        if self.with_timestamp < 0 or self.with_timestamp > self.total:
            raise ValueError("with_timestamp must be between 0 and total")


@dataclass
class ChangeDriver:
    """
    Individual driver of change attribution.

    Explains what contributed to an outcome delta.
    """

    change_type: str
    description: str
    contribution_to_delta: float
    contribution_pct: float
    affected_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate change driver data."""
        if not 0 <= self.contribution_pct <= 100:
            raise ValueError("contribution_pct must be between 0 and 100")


@dataclass
class ChangeAttribution:
    """
    Attribution analysis for outcome changes.

    Explains why outcomes changed between scenarios.
    """

    outcome_delta: float
    primary_drivers: List[ChangeDriver] = field(default_factory=list)
    summary: str = ""

    def __post_init__(self):
        """Validate change attribution data."""
        if self.primary_drivers:
            total_pct = sum(driver.contribution_pct for driver in self.primary_drivers)
            # Allow some tolerance for floating point errors
            if abs(total_pct - 100.0) > 0.1:
                # Log warning but don't fail - PLoT might not always sum to 100%
                pass


@dataclass
class ModelCard:
    """
    Model metadata and quality metrics.

    Provides transparency about model construction and evidence quality.
    """

    model_id: Optional[str] = None
    version: Optional[str] = None
    created_at: Optional[str] = None
    evidence_freshness: Optional[EvidenceFreshness] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareOption:
    """
    Comparison option with change attribution.

    Represents one scenario in a comparison analysis.
    """

    option_id: str
    label: str
    outcome_value: float
    change_attribution: Optional[ChangeAttribution] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRequest:
    """Request for PLoT /v1/run endpoint."""

    graph: Dict[str, Any]
    idempotency_key: Optional[str] = None
    timeout_ms: int = 30000
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResponse:
    """Response from PLoT /v1/run endpoint."""

    run_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    model_card: ModelCard = field(default_factory=ModelCard)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareRequest:
    """Request for PLoT /v1/compare endpoint."""

    graph: Dict[str, Any]
    scenarios: List[Dict[str, Any]]
    idempotency_key: Optional[str] = None
    timeout_ms: int = 30000
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareResponse:
    """Response from PLoT /v1/compare endpoint."""

    compare_id: str
    status: str
    options: List[CompareOption] = field(default_factory=list)
    model_card: ModelCard = field(default_factory=ModelCard)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IdempotencyMismatchError(Exception):
    """
    Exception raised when idempotency key mismatch occurs (409).

    This error should not be retried - indicates conflicting request with same key.
    """

    def __init__(self, message: str, idempotency_key: Optional[str] = None):
        """
        Initialize idempotency mismatch error.

        Args:
            message: Error message from PLoT
            idempotency_key: The conflicting idempotency key
        """
        super().__init__(message)
        self.idempotency_key = idempotency_key
