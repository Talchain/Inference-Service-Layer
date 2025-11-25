"""Pydantic models for ISL API responses."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ===== Explanation & Metadata =====


class ExplanationMetadata(BaseModel):
    """Metadata for explaining results to users."""

    summary: str = Field(description="Plain English summary")
    assumptions: list[str] = Field(description="Key assumptions made")
    caveats: list[str] = Field(default_factory=list, description="Important caveats")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in result (0-1)")


# ===== Causal Validation =====


class SuggestionAction(BaseModel):
    """Action to take for a validation suggestion."""

    action_type: Literal["collect_data", "add_edges", "control_for", "manual_review"]
    details: dict[str, Any]


class ValidationSuggestion(BaseModel):
    """Suggestion for making DAG identifiable."""

    type: str = Field(description="Suggestion type (backdoor, frontdoor, instrumental)")
    description: str = Field(description="Plain English description")
    technical_detail: str = Field(description="Technical explanation")
    priority: Literal["critical", "recommended", "optional"]
    action: SuggestionAction


class AdjustmentStrategy(BaseModel):
    """Complete adjustment strategy for identifiability."""

    strategy_type: str = Field(description="backdoor, frontdoor, instrumental, manual")
    nodes_to_add: list[str] = Field(description="Variables to measure/add")
    edges_to_add: list[tuple[str, str]] = Field(description="Edges to add to DAG")
    explanation: str = Field(description="Plain English explanation")
    theoretical_basis: str = Field(description="Theoretical justification")
    expected_identifiability: float = Field(ge=0.0, le=1.0, description="Confidence (0-1)")


class ValidationResponse(BaseModel):
    """Response from causal validation endpoint."""

    status: Literal["identifiable", "uncertain", "cannot_identify"]
    method: str | None = Field(None, description="Identification method used")
    adjustment_sets: list[list[str]] | None = Field(
        None, description="Valid adjustment sets"
    )
    suggestions: list[ValidationSuggestion] | None = Field(
        None, description="Suggestions if not identifiable"
    )
    explanation: ExplanationMetadata


class StrategiesResponse(BaseModel):
    """Response with complete adjustment strategies."""

    strategies: list[AdjustmentStrategy] = Field(description="Ranked adjustment strategies")
    path_analysis: dict[str, Any] | None = Field(None, description="Detailed path analysis")
    explanation: ExplanationMetadata


# ===== Conformal Prediction =====


class ConformalInterval(BaseModel):
    """Conformal prediction interval."""

    lower: float = Field(description="Lower bound")
    upper: float = Field(description="Upper bound")
    width: float = Field(description="Interval width")
    point_estimate: float = Field(description="Point prediction")


class CoverageGuarantee(BaseModel):
    """Coverage guarantee for conformal intervals."""

    guaranteed: bool = Field(description="Whether coverage is guaranteed")
    theoretical_coverage: float = Field(ge=0.0, le=1.0, description="Theoretical coverage")
    finite_sample_valid: bool = Field(description="Valid for finite samples")
    assumptions: list[str] = Field(description="Required assumptions")


class IntervalComparison(BaseModel):
    """Comparison between conformal and Monte Carlo intervals."""

    monte_carlo_width: float
    conformal_width: float
    width_ratio: float = Field(description="conformal_width / monte_carlo_width")
    relative_efficiency: float = Field(description="Efficiency gain/loss")


class ConformalResponse(BaseModel):
    """Response from conformal prediction endpoint."""

    conformal_interval: ConformalInterval
    coverage_guarantee: CoverageGuarantee
    comparison_to_monte_carlo: IntervalComparison
    explanation: ExplanationMetadata


# ===== Counterfactual =====


class CounterfactualPrediction(BaseModel):
    """Single counterfactual prediction."""

    prediction: dict[str, float] = Field(description="Predicted outcomes")
    uncertainty: dict[str, float] | None = Field(None, description="Uncertainty estimates")
    explanation: str = Field(description="How intervention affects outcome")


class CounterfactualResponse(BaseModel):
    """Response from counterfactual prediction endpoint."""

    intervention: dict[str, float] = Field(description="Applied intervention")
    prediction: CounterfactualPrediction
    model_assumptions: list[str] = Field(description="SCM assumptions")
    explanation: ExplanationMetadata


# ===== Batch Scenarios =====


class ScenarioResult(BaseModel):
    """Result for a single scenario."""

    scenario_id: str
    intervention: dict[str, float]
    prediction: dict[str, float]
    uncertainty: dict[str, float] | None = None
    explanation: str


class InteractionAnalysis(BaseModel):
    """Analysis of interactions between scenarios."""

    has_synergy: bool = Field(description="Whether synergistic effects detected")
    synergy_score: float | None = Field(None, description="Strength of synergy")
    summary: str = Field(description="Plain English summary")
    details: dict[str, Any] = Field(default_factory=dict)


class BatchCounterfactualResponse(BaseModel):
    """Response from batch counterfactual analysis."""

    scenarios: list[ScenarioResult]
    interactions: InteractionAnalysis | None = None
    optimal_scenario: str | None = Field(None, description="Best scenario ID")
    explanation: ExplanationMetadata


# ===== Transportability =====


class TransportabilityResponse(BaseModel):
    """Response from transportability analysis."""

    transportable: bool = Field(description="Whether effect transports")
    validity_conditions: list[str] = Field(description="Conditions for transport")
    adaptation_required: bool = Field(description="Whether adaptation needed")
    suggestions: list[str] = Field(default_factory=list, description="How to adapt")
    explanation: ExplanationMetadata


# ===== Causal Discovery =====


class DAGStructure(BaseModel):
    """DAG structure representation."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoveryResponse(BaseModel):
    """Response from causal discovery endpoint."""

    dag: DAGStructure = Field(description="Discovered DAG structure")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in structure")
    algorithm_used: str = Field(description="Discovery algorithm")
    alternative_dags: list[DAGStructure] = Field(
        default_factory=list, description="Alternative structures"
    )
    explanation: ExplanationMetadata


# ===== Contrastive Explanation =====


class ContrastiveExplanation(BaseModel):
    """Contrastive explanation comparing scenarios."""

    scenario_a_id: str
    scenario_b_id: str
    key_differences: list[str] = Field(description="Main differences causing outcome")
    counterfactual_path: str = Field(description="How to change A to get B")
    importance_scores: dict[str, float] = Field(description="Variable importance")
    explanation: str = Field(description="Plain English explanation")


class ContrastiveResponse(BaseModel):
    """Response from contrastive explanation endpoint."""

    explanations: list[ContrastiveExplanation]
    summary: str = Field(description="Overall summary")
    explanation: ExplanationMetadata


# ===== Sequential Optimization =====


class OptimizationStep(BaseModel):
    """Single step in sequential optimization."""

    step: int
    intervention: dict[str, float]
    predicted_outcome: float
    utility: float = Field(description="Expected utility")
    explanation: str


class OptimizationResponse(BaseModel):
    """Response from sequential optimization endpoint."""

    optimal_sequence: list[OptimizationStep]
    total_utility: float = Field(description="Total expected utility")
    convergence_achieved: bool = Field(description="Whether optimization converged")
    alternative_sequences: list[list[OptimizationStep]] = Field(
        default_factory=list, description="Alternative sequences"
    )
    explanation: ExplanationMetadata


# ===== Health & Status =====


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "failing"]
    version: str
    timestamp: str
    config_fingerprint: str | None = None


class ServiceHealth(BaseModel):
    """Health status for a specific service."""

    status: Literal["HEALTHY", "DEGRADED", "FAILING"]
    success_rate_percent: float
    total_requests: int
    successes: int
    failures: int
    fallbacks: int
    uptime_seconds: float
    last_check: str | None


class ServicesHealthResponse(BaseModel):
    """Response from /health/services endpoint."""

    overall_status: Literal["healthy", "degraded"]
    timestamp: str
    services: dict[str, ServiceHealth]


class CircuitBreakerStatus(BaseModel):
    """Status of a circuit breaker."""

    state: Literal["CLOSED", "OPEN", "HALF_OPEN"]
    failure_count: int
    success_count: int
    failure_threshold: int
    success_threshold: int
    timeout_seconds: int
    last_failure_time: str | None


class CircuitBreakersResponse(BaseModel):
    """Response from /health/circuit-breakers endpoint."""

    overall_status: Literal["operational", "some_circuits_open"]
    timestamp: str
    circuit_breakers: dict[str, CircuitBreakerStatus]
    explanation: str


# ===== Error Response =====


class ErrorResponse(BaseModel):
    """Standard error response from ISL."""

    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool
    suggested_action: str
