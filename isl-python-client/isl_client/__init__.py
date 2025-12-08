"""ISL Python Client - Type-safe client for Olumi Inference Service Layer."""

__version__ = "0.1.0"

from .client import ISLClient
from .exceptions import (
    AuthenticationError,
    CircuitBreakerError,
    ISLException,
    NotFoundError,
    RateLimitError,
    ServiceUnavailable,
    TimeoutError,
    ValidationError,
)
from .models import (
    AdjustmentStrategy,
    BatchCounterfactualResponse,
    CircuitBreakerStatus,
    CircuitBreakersResponse,
    ConformalInterval,
    ConformalResponse,
    ContrastiveExplanation,
    ContrastiveResponse,
    CounterfactualPrediction,
    CounterfactualResponse,
    CoverageGuarantee,
    DAGStructure,
    DiscoveryResponse,
    ErrorResponse,
    ExplanationMetadata,
    HealthResponse,
    InteractionAnalysis,
    IntervalComparison,
    OptimizationResponse,
    OptimizationStep,
    ScenarioResult,
    ServiceHealth,
    ServicesHealthResponse,
    StrategiesResponse,
    SuggestionAction,
    TransportabilityResponse,
    ValidationResponse,
    ValidationSuggestion,
)
from .sync import ISLClientSync

__all__ = [
    # Client
    "ISLClient",
    "ISLClientSync",
    # Exceptions
    "ISLException",
    "ValidationError",
    "ServiceUnavailable",
    "RateLimitError",
    "AuthenticationError",
    "NotFoundError",
    "TimeoutError",
    "CircuitBreakerError",
    # Models - Causal
    "ValidationResponse",
    "ValidationSuggestion",
    "SuggestionAction",
    "StrategiesResponse",
    "AdjustmentStrategy",
    "CounterfactualResponse",
    "CounterfactualPrediction",
    "ConformalResponse",
    "ConformalInterval",
    "CoverageGuarantee",
    "IntervalComparison",
    "BatchCounterfactualResponse",
    "ScenarioResult",
    "InteractionAnalysis",
    "TransportabilityResponse",
    # Models - Discovery
    "DiscoveryResponse",
    "DAGStructure",
    # Models - Explanation
    "ContrastiveResponse",
    "ContrastiveExplanation",
    # Models - Optimization
    "OptimizationResponse",
    "OptimizationStep",
    # Models - Health
    "HealthResponse",
    "ServicesHealthResponse",
    "ServiceHealth",
    "CircuitBreakersResponse",
    "CircuitBreakerStatus",
    # Models - Shared
    "ExplanationMetadata",
    "ErrorResponse",
]
