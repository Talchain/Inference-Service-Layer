"""
V2 Response Schemas for ISL Enhanced Response Format.

Provides explicit status fields, structured critiques, and diagnostics
for improved integration with PLoT and UI components.

P2 Brief Alignment:
- Adds `version` as alias for `response_schema_version`
- Adds `timestamp` in ISO 8601 format
- Adds `seed_used` for determinism (PLoT owns response_hash)
- 422 responses use unwrapped ISLV2Error422 format
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.constants import RESPONSE_SCHEMA_VERSION_V2


# =============================================================================
# Request Echo (for debugging, no sensitive data)
# =============================================================================


class RequestEchoV2(BaseModel):
    """Echo of request parameters for debugging (no sensitive data)."""

    graph_node_count: int = Field(..., description="Number of nodes in graph")
    graph_edge_count: int = Field(..., description="Number of edges in graph")
    options_count: int = Field(..., description="Number of options provided")
    goal_node_id_hash: str = Field(
        ..., description="SHA-256 hash of goal node ID (truncated)"
    )
    n_samples: int = Field(..., description="Number of samples requested")
    response_version_requested: int = Field(
        ..., description="Response version requested"
    )
    include_diagnostics: bool = Field(
        ..., description="Whether diagnostics were requested"
    )


# =============================================================================
# Critique (structured error/warning information)
# =============================================================================


class CritiqueV2(BaseModel):
    """Structured critique for UI display."""

    id: str = Field(..., description="Unique identifier for this critique")
    code: str = Field(
        ..., description="Machine-readable code, e.g., 'NO_PATH_TO_GOAL'"
    )
    severity: Literal["info", "warning", "error", "blocker"] = Field(
        ..., description="Severity level"
    )
    message: str = Field(..., description="Human-readable message (sanitised)")
    source: Literal["validation", "analysis", "engine"] = Field(
        ..., description="Source of the critique (explicit, not derived)"
    )
    affected_option_ids: Optional[List[str]] = Field(
        None, description="Option IDs affected by this critique"
    )
    affected_node_ids: Optional[List[str]] = Field(
        None, description="Node IDs affected by this critique"
    )
    suggestion: Optional[str] = Field(
        None, description="Actionable suggestion to resolve the issue"
    )


# =============================================================================
# Diagnostics (optional detailed information)
# =============================================================================


class OptionDiagnosticV2(BaseModel):
    """Diagnostic information for a single option."""

    option_id: str = Field(..., description="Option identifier")
    intervention_count: int = Field(..., description="Number of interventions")
    has_structural_path: bool = Field(
        ...,
        description="Path exists with exists_probability >= threshold",
    )
    has_effective_path: bool = Field(
        ...,
        description="Structural path AND abs(strength.mean) >= threshold",
    )
    targets_with_effective_path_count: int = Field(
        ..., description="Number of intervention targets with effective path"
    )
    targets_without_effective_path_count: int = Field(
        ..., description="Number of intervention targets without effective path"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Option-specific warnings"
    )


class DiagnosticsV2(BaseModel):
    """Diagnostic information (only included when requested)."""

    goal_node_id_hash: str = Field(..., description="Hashed goal node ID")
    goal_node_found: bool = Field(
        ..., description="Whether goal node exists in graph"
    )
    option_diagnostics: List[OptionDiagnosticV2] = Field(
        default_factory=list, description="Per-option diagnostics"
    )
    n_samples_requested: int = Field(..., description="Samples requested")
    n_samples_completed: int = Field(..., description="Samples completed")
    identifiability_status: Literal["identifiable", "not_identifiable", "unknown"] = (
        Field(..., description="Causal identifiability status")
    )
    identifiability_reason: Optional[str] = Field(
        None, description="Reason for identifiability status"
    )
    path_exists_probability_threshold: float = Field(
        ..., description="Threshold used for exists_probability"
    )
    path_strength_threshold: float = Field(
        ..., description="Threshold used for strength.mean"
    )


# =============================================================================
# Outcome Distribution
# =============================================================================


class OutcomeDistributionV2(BaseModel):
    """Outcome distribution with core percentiles."""

    mean: float = Field(..., description="Mean outcome value")
    std: float = Field(..., description="Standard deviation")
    p10: float = Field(..., description="10th percentile")
    p50: float = Field(..., description="50th percentile (median)")
    p90: float = Field(..., description="90th percentile")
    n_samples: int = Field(..., description="Total samples")
    n_valid_samples: int = Field(
        ..., description="Samples without NaN/Inf"
    )
    validity_ratio: float = Field(
        ..., description="n_valid_samples / n_samples"
    )


# =============================================================================
# Option Result
# =============================================================================


class OptionResultV2(BaseModel):
    """Analysis result for a single option."""

    id: str = Field(..., description="Option identifier")
    label: Optional[str] = Field(None, description="Human-readable label")
    outcome: OutcomeDistributionV2 = Field(..., description="Outcome distribution")
    win_probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="P(this option is best) - fraction of samples where this option had highest outcome"
    )
    probability_of_goal: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="P(outcome >= goal_threshold). Only present when goal_threshold is provided in request."
    )
    status: Literal["computed", "partial", "failed"] = Field(
        ..., description="Option-specific status"
    )
    status_reason: Optional[str] = Field(
        None, description="Reason for non-computed status"
    )


# =============================================================================
# Sensitive Factor
# =============================================================================


class SensitiveFactorV2(BaseModel):
    """Factor sensitivity information."""

    node_id: str = Field(..., description="Factor node ID")
    sensitivity_score: float = Field(..., description="Sensitivity score")
    effect_on_ranking: Literal["none", "minor", "moderate", "major"] = Field(
        ..., description="Effect on option ranking"
    )


# =============================================================================
# Fragile Edge (V2 enhanced format)
# =============================================================================


class FragileEdgeV2(BaseModel):
    """Fragile edge with alternative winner analysis.

    Identifies edges where the recommendation is sensitive to assumption changes
    and what option would win if the edge is weaker than modelled.
    """

    edge_id: str = Field(..., description="Edge identifier in 'from->to' format")
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Target node ID")
    alternative_winner_id: Optional[str] = Field(
        None,
        description="Option that wins most often when this edge is weak (bottom quartile). "
        "Null if same option wins regardless of edge strength.",
    )
    switch_probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Proportion of MC samples where alternative wins when edge is weak. "
        "0.0 if same option wins (stable), null only if no data available.",
    )


# =============================================================================
# Robustness Result
# =============================================================================


class RobustnessResultV2(BaseModel):
    """Robustness analysis result."""

    # V2 fields
    level: Literal["high", "moderate", "low", "very_low"] = Field(
        ..., description="Robustness level"
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence [0, 1]")
    sensitive_factors: Optional[List[SensitiveFactorV2]] = Field(
        None, description="Factor sensitivity breakdown"
    )

    # V2 enhanced fragile edges with alternative winner analysis
    fragile_edges: Optional[List[FragileEdgeV2]] = Field(
        None,
        description="Edges that could flip the decision, with alternative winner analysis",
    )

    # V1 backward-compatibility fields (for PLoT integration)
    is_robust: Optional[bool] = Field(
        None, description="Whether recommendation is robust (V1 compat)"
    )
    fragile_edges_v1: Optional[List[str]] = Field(
        None,
        description="Edges that could flip the decision (V1 compat, string format)",
    )
    robust_edges: Optional[List[str]] = Field(
        None, description="Edges that don't significantly affect decision (V1 compat)"
    )
    recommendation_stability: Optional[float] = Field(
        None, ge=0, le=1, description="P(same recommendation across samples) (V1 compat)"
    )


# =============================================================================
# Factor Sensitivity
# =============================================================================


class FactorSensitivityV2(BaseModel):
    """Factor sensitivity for drivers analysis."""

    node_id: str = Field(..., description="Factor node ID")
    label: Optional[str] = Field(None, description="Human-readable node label")
    sensitivity_score: float = Field(..., description="Sensitivity score")
    direction: Literal["positive", "negative"] = Field(
        ..., description="Direction of effect"
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")


# =============================================================================
# Main V2 Response
# =============================================================================


class ISLResponseV2(BaseModel):
    """Enhanced response with explicit status fields and diagnostics."""

    # Version information (P2-ISL-1: added `version` alias for PLoT compatibility)
    response_schema_version: str = Field(
        default=RESPONSE_SCHEMA_VERSION_V2,
        description="Response schema version",
        alias="version",
    )
    endpoint_version: str = Field(..., description="Endpoint version, e.g., 'analyze/v2'")
    engine_version: str = Field(..., description="ISL engine version")

    # P2-ISL-1: Timestamp in ISO 8601 format
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        description="Response timestamp in ISO 8601 format",
    )

    # Explicit status fields (CRITICAL for UI)
    analysis_status: Literal["computed", "partial", "failed"] = Field(
        ..., description="Overall analysis status"
    )
    robustness_status: Literal["computed", "skipped", "unavailable", "error"] = Field(
        ..., description="Robustness analysis status"
    )
    factor_sensitivity_status: Literal["computed", "skipped", "unavailable", "error"] = (
        Field(..., description="Factor sensitivity status")
    )

    # Reason for non-computed status (sanitised, no internal details)
    status_reason: Optional[str] = Field(
        None, description="Reason for non-computed status"
    )

    # Structured critiques (for UI display)
    critiques: List[CritiqueV2] = Field(
        default_factory=list, description="Structured critiques"
    )

    # Request echo (for debugging integration issues)
    request_echo: RequestEchoV2 = Field(..., description="Echo of request parameters")

    # Diagnostics (OPTIONAL - only when requested)
    diagnostics: Optional[DiagnosticsV2] = Field(
        None, description="Detailed diagnostics (when requested)"
    )

    # Analysis results (only if analysis_status in ["computed", "partial"])
    options: Optional[List[OptionResultV2]] = Field(
        None, description="Option results"
    )
    robustness: Optional[RobustnessResultV2] = Field(
        None, description="Robustness assessment"
    )
    factor_sensitivity: Optional[List[FactorSensitivityV2]] = Field(
        None, description="Factor sensitivity results"
    )

    # Correlation
    request_id: str = Field(..., description="Request ID for correlation")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    # P2-ISL-1: Determinism (ISL only returns seed_used; response_hash is PLoT-owned)
    seed_used: Optional[str] = Field(
        None, description="RNG seed used for deterministic reproduction"
    )

    model_config = {
        "populate_by_name": True,  # Allow both 'version' and 'response_schema_version'
        "json_schema_extra": {
            "example": {
                "version": "2.0",
                "endpoint_version": "analyze/v2",
                "engine_version": "1.0.0",
                "timestamp": "2025-01-15T10:30:00Z",
                "analysis_status": "computed",
                "robustness_status": "computed",
                "factor_sensitivity_status": "computed",
                "status_reason": None,
                "critiques": [],
                "request_echo": {
                    "graph_node_count": 5,
                    "graph_edge_count": 4,
                    "options_count": 2,
                    "goal_node_id_hash": "abc123def456",
                    "n_samples": 1000,
                    "response_version_requested": 2,
                    "include_diagnostics": False,
                },
                "options": [
                    {
                        "id": "option_a",
                        "label": "Option A",
                        "outcome": {
                            "mean": 50000.0,
                            "std": 5000.0,
                            "p10": 42000.0,
                            "p50": 50000.0,
                            "p90": 58000.0,
                            "n_samples": 1000,
                            "n_valid_samples": 1000,
                            "validity_ratio": 1.0,
                        },
                        "status": "computed",
                    }
                ],
                "robustness": {
                    "level": "high",
                    "confidence": 0.92,
                },
                "request_id": "req_abc123",
                "processing_time_ms": 150,
                "seed_used": "42",
            }
        }
    }


# =============================================================================
# 422 Error Response (P2-ISL-3)
# =============================================================================


class ISLV2Error422(BaseModel):
    """
    422 error response — MUST be returned unwrapped (no envelope).

    Per P2 brief: This is the exact shape PLoT/UI expect for validation failures.
    DO NOT wrap in {"error": {...}} or add success: false.
    """

    analysis_status: Literal["blocked"] = Field(
        default="blocked",
        description="Always 'blocked' for 422 responses",
    )
    status_reason: str = Field(
        ..., description="Human-readable reason for blocking"
    )
    critiques: List[CritiqueV2] = Field(
        ..., description="Structured critiques explaining the validation failure"
    )
    request_id: Optional[str] = Field(
        None, description="Request ID for correlation (echoed if available)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "analysis_status": "blocked",
                "status_reason": "Intervention targets non-existent node",
                "critiques": [
                    {
                        "id": "critique_a1b2c3d4",
                        "code": "INVALID_INTERVENTION_TARGET",
                        "severity": "blocker",
                        "source": "validation",
                        "message": "Intervention targets non-existent node: 'nonexistent_node'",
                        "suggestion": "Interventions must target nodes that exist in the graph",
                        "affected_node_ids": ["nonexistent_node"],
                    }
                ],
                "request_id": "isl-a1b2c3d4e5f6",
            }
        }
    }
