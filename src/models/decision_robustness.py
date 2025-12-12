"""
Decision Robustness Suite models.

Response and request schemas for the unified robustness analysis endpoint.
Implements Brief 7: ISL — Decision Robustness Suite.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .isl_metadata import ISLResponseMetadata
from .shared import GraphV1


# ============================================================================
# Enums
# ============================================================================


class ConfidenceLevelEnum(str, Enum):
    """Confidence level for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationStatusEnum(str, Enum):
    """Recommendation status based on identifiability and robustness."""
    ACTIONABLE = "actionable"
    EXPLORATORY = "exploratory"


class RobustnessLabelEnum(str, Enum):
    """Overall robustness classification."""
    ROBUST = "robust"
    MODERATE = "moderate"
    FRAGILE = "fragile"


class ImpactDirectionEnum(str, Enum):
    """Direction of parameter impact on outcome."""
    POSITIVE = "positive"
    NEGATIVE = "negative"


# ============================================================================
# Nested Response Models
# ============================================================================


class UtilityDistribution(BaseModel):
    """Distribution of utility values at percentiles."""

    p5: float = Field(..., description="5th percentile")
    p25: float = Field(..., description="25th percentile")
    p50: float = Field(..., description="50th percentile (median)")
    p75: float = Field(..., description="75th percentile")
    p95: float = Field(..., description="95th percentile")

    model_config = {
        "json_schema_extra": {
            "example": {
                "p5": 85000.0,
                "p25": 92000.0,
                "p50": 100000.0,
                "p75": 108000.0,
                "p95": 115000.0
            }
        }
    }


class RankedOption(BaseModel):
    """A decision option with its ranking and utility metrics."""

    option_id: str = Field(..., description="Unique option identifier")
    option_label: str = Field(..., description="Human-readable option name")
    expected_utility: float = Field(..., description="Expected utility value")
    utility_distribution: UtilityDistribution = Field(
        ...,
        description="Distribution of utility values"
    )
    rank: int = Field(..., description="Rank (1 = best)", ge=1)
    vs_baseline: Optional[float] = Field(
        default=None,
        description="Absolute difference vs baseline option"
    )
    vs_baseline_pct: Optional[float] = Field(
        default=None,
        description="Percentage difference vs baseline option"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "option_a",
                "option_label": "Aggressive Marketing",
                "expected_utility": 150000.0,
                "utility_distribution": {
                    "p5": 120000.0,
                    "p25": 135000.0,
                    "p50": 150000.0,
                    "p75": 165000.0,
                    "p95": 180000.0
                },
                "rank": 1,
                "vs_baseline": 25000.0,
                "vs_baseline_pct": 20.0
            }
        }
    }


class Recommendation(BaseModel):
    """Top recommendation from the analysis."""

    option_id: str = Field(..., description="Recommended option ID")
    option_label: str = Field(..., description="Recommended option label")
    confidence: ConfidenceLevelEnum = Field(
        ...,
        description="Confidence in recommendation"
    )
    recommendation_status: RecommendationStatusEnum = Field(
        ...,
        description="Whether recommendation is actionable or exploratory"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "option_a",
                "option_label": "Aggressive Marketing",
                "confidence": "high",
                "recommendation_status": "actionable"
            }
        }
    }


class SensitiveParameter(BaseModel):
    """A parameter identified as sensitive to the decision outcome."""

    parameter_id: str = Field(..., description="Parameter identifier")
    parameter_label: str = Field(..., description="Human-readable parameter name")
    sensitivity_score: float = Field(
        ...,
        description="Normalized sensitivity score (0-1)",
        ge=0,
        le=1
    )
    current_value: float = Field(..., description="Current parameter value")
    impact_direction: ImpactDirectionEnum = Field(
        ...,
        description="Whether increasing parameter improves or hurts outcome"
    )
    description: str = Field(
        ...,
        description="Human-readable description of impact"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter_id": "price_elasticity",
                "parameter_label": "Price Elasticity",
                "sensitivity_score": 0.85,
                "current_value": -1.2,
                "impact_direction": "negative",
                "description": "Increasing Price Elasticity decreases expected revenue"
            }
        }
    }


class RobustnessBound(BaseModel):
    """Threshold at which a parameter change would flip the recommendation."""

    parameter_id: str = Field(..., description="Parameter identifier")
    parameter_label: str = Field(..., description="Human-readable parameter name")
    flip_threshold: float = Field(
        ...,
        description="Absolute change required to flip recommendation"
    )
    flip_threshold_pct: float = Field(
        ...,
        description="Percentage change required to flip recommendation"
    )
    flip_to_option: str = Field(
        ...,
        description="Which option would become recommended after flip"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter_id": "marketing_roi",
                "parameter_label": "Marketing ROI",
                "flip_threshold": 0.15,
                "flip_threshold_pct": 25.0,
                "flip_to_option": "option_b"
            }
        }
    }


class ValueOfInformation(BaseModel):
    """Value of information analysis for a parameter."""

    parameter_id: str = Field(..., description="Parameter identifier")
    parameter_label: str = Field(..., description="Human-readable parameter name")
    evpi: float = Field(
        ...,
        description="Expected Value of Perfect Information"
    )
    evsi: float = Field(
        ...,
        description="Expected Value of Sample Information"
    )
    current_uncertainty: float = Field(
        ...,
        description="Current variance/entropy of parameter"
    )
    recommendation: str = Field(
        ...,
        description="Recommendation based on VoI analysis"
    )
    data_collection_suggestion: str = Field(
        ...,
        description="Actionable suggestion for gathering data"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter_id": "churn_rate",
                "parameter_label": "Customer Churn Rate",
                "evpi": 15000.0,
                "evsi": 8500.0,
                "current_uncertainty": 0.25,
                "recommendation": "High value - consider gathering data",
                "data_collection_suggestion": "Survey 50 customers about retention factors"
            }
        }
    }


class ParetoPoint(BaseModel):
    """A point on the Pareto frontier for multi-goal optimization."""

    option_id: str = Field(..., description="Option identifier")
    option_label: str = Field(..., description="Human-readable option name")
    goal_values: Dict[str, float] = Field(
        ...,
        description="Value achieved for each goal"
    )
    is_dominated: bool = Field(
        ...,
        description="Whether this option is dominated by another"
    )
    trade_off_description: str = Field(
        ...,
        description="Description of trade-offs"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "option_a",
                "option_label": "Aggressive Marketing",
                "goal_values": {"revenue": 150000.0, "risk": 0.3},
                "is_dominated": False,
                "trade_off_description": "Sacrifices 10% risk reduction for 25% higher revenue"
            }
        }
    }


class ParetoResult(BaseModel):
    """Pareto frontier analysis for multi-goal decisions."""

    goals: List[str] = Field(..., description="Goals analyzed")
    frontier_options: List[ParetoPoint] = Field(
        ...,
        description="Options on the Pareto frontier"
    )
    current_selection_pareto_efficient: bool = Field(
        ...,
        description="Whether current recommendation is Pareto efficient"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "goals": ["revenue", "risk"],
                "frontier_options": [
                    {
                        "option_id": "option_a",
                        "option_label": "Aggressive Marketing",
                        "goal_values": {"revenue": 150000.0, "risk": 0.3},
                        "is_dominated": False,
                        "trade_off_description": "Highest revenue, moderate risk"
                    }
                ],
                "current_selection_pareto_efficient": True
            }
        }
    }


# ============================================================================
# Main Response Schema
# ============================================================================


class RobustnessResult(BaseModel):
    """
    Unified response schema for Decision Robustness Suite.

    Contains all robustness metrics in a single payload:
    - Option rankings with utility distributions
    - Top recommendation with confidence
    - Sensitivity analysis (top sensitive parameters)
    - Robustness bounds (flip thresholds)
    - Value of information (EVPI/EVSI)
    - Pareto frontier (for multi-goal)
    - Narrative summary
    """

    # Rankings (ISL's core responsibility)
    option_rankings: List[RankedOption] = Field(
        ...,
        description="All options ranked by expected utility"
    )
    recommendation: Recommendation = Field(
        ...,
        description="Top recommendation with confidence"
    )

    # Sensitivity Analysis
    sensitivity: List[SensitiveParameter] = Field(
        ...,
        description="Most sensitive parameters (ranked)"
    )

    # Robustness Assessment
    robustness_label: RobustnessLabelEnum = Field(
        ...,
        description="Overall robustness classification"
    )
    robustness_summary: str = Field(
        ...,
        description="Plain language robustness summary"
    )
    robustness_bounds: List[RobustnessBound] = Field(
        ...,
        description="Parameter thresholds that would flip recommendation"
    )

    # Value of Information
    value_of_information: List[ValueOfInformation] = Field(
        ...,
        description="VoI analysis for uncertain parameters"
    )

    # Pareto Frontier (optional, only for multi-goal)
    pareto: Optional[ParetoResult] = Field(
        default=None,
        description="Pareto frontier analysis (multi-goal only)"
    )

    # Unified Narrative
    narrative: str = Field(
        ...,
        description="Plain language summary combining all analyses"
    )

    # Metadata
    metadata: Optional[ISLResponseMetadata] = Field(
        default=None,
        description="Response metadata",
        alias="_metadata"
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "option_rankings": [
                    {
                        "option_id": "option_a",
                        "option_label": "Aggressive Marketing",
                        "expected_utility": 150000.0,
                        "utility_distribution": {
                            "p5": 120000.0, "p25": 135000.0, "p50": 150000.0,
                            "p75": 165000.0, "p95": 180000.0
                        },
                        "rank": 1,
                        "vs_baseline": 25000.0,
                        "vs_baseline_pct": 20.0
                    }
                ],
                "recommendation": {
                    "option_id": "option_a",
                    "option_label": "Aggressive Marketing",
                    "confidence": "high",
                    "recommendation_status": "actionable"
                },
                "sensitivity": [
                    {
                        "parameter_id": "price_elasticity",
                        "parameter_label": "Price Elasticity",
                        "sensitivity_score": 0.85,
                        "current_value": -1.2,
                        "impact_direction": "negative",
                        "description": "Increasing Price Elasticity decreases expected revenue"
                    }
                ],
                "robustness_label": "robust",
                "robustness_summary": "Your decision is robust to typical parameter uncertainty",
                "robustness_bounds": [
                    {
                        "parameter_id": "marketing_roi",
                        "parameter_label": "Marketing ROI",
                        "flip_threshold": 0.15,
                        "flip_threshold_pct": 25.0,
                        "flip_to_option": "option_b"
                    }
                ],
                "value_of_information": [
                    {
                        "parameter_id": "churn_rate",
                        "parameter_label": "Customer Churn Rate",
                        "evpi": 15000.0,
                        "evsi": 8500.0,
                        "current_uncertainty": 0.25,
                        "recommendation": "High value - consider gathering data",
                        "data_collection_suggestion": "Survey 50 customers"
                    }
                ],
                "narrative": "Your decision is robust. Price changes up to 30% wouldn't change the recommendation."
            }
        }
    }


class RobustnessResponse(BaseModel):
    """
    API response wrapper for robustness analysis.
    """

    result: RobustnessResult = Field(
        ...,
        description="Robustness analysis result"
    )
    metadata: Optional[ISLResponseMetadata] = Field(
        default=None,
        description="Response metadata",
        alias="_metadata"
    )

    model_config = {
        "populate_by_name": True
    }


# ============================================================================
# Request Schema
# ============================================================================


class UtilitySpecification(BaseModel):
    """Specification of utility function for ranking options."""

    goal_node_id: str = Field(
        ...,
        description="Node ID representing the primary goal/outcome"
    )
    additional_goals: Optional[List[str]] = Field(
        default=None,
        description="Additional goal node IDs for multi-goal analysis"
    )
    maximize: bool = Field(
        default=True,
        description="Whether to maximize (True) or minimize (False) the goal"
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for multiple goals (must sum to 1.0)"
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v):
        """Ensure weights sum to approximately 1.0."""
        if v is not None:
            total = sum(v.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "goal_node_id": "revenue",
                "additional_goals": ["customer_satisfaction"],
                "maximize": True,
                "weights": {"revenue": 0.7, "customer_satisfaction": 0.3}
            }
        }
    }


class AnalysisOptions(BaseModel):
    """Configuration options for the robustness analysis."""

    sensitivity_top_n: int = Field(
        default=5,
        description="Number of top sensitive parameters to return",
        ge=1,
        le=20
    )
    perturbation_range: float = Field(
        default=0.5,
        description="Maximum perturbation for robustness bounds (0.5 = ±50%)",
        gt=0,
        le=1.0
    )
    monte_carlo_samples: int = Field(
        default=1000,
        description="Number of Monte Carlo samples for uncertainty",
        ge=100,
        le=10000
    )
    include_pareto: bool = Field(
        default=True,
        description="Include Pareto frontier analysis for multi-goal"
    )
    include_voi: bool = Field(
        default=True,
        description="Include Value of Information analysis"
    )
    sample_sizes_for_evsi: List[int] = Field(
        default=[10, 50, 100],
        description="Sample sizes to consider for EVSI calculation"
    )
    timeout_ms: Optional[int] = Field(
        default=5000,
        description="Maximum computation time in milliseconds",
        ge=1000,
        le=60000
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "sensitivity_top_n": 5,
                "perturbation_range": 0.5,
                "monte_carlo_samples": 1000,
                "include_pareto": True,
                "include_voi": True,
                "sample_sizes_for_evsi": [10, 50, 100],
                "timeout_ms": 5000
            }
        }
    }


class DecisionOption(BaseModel):
    """A decision option to evaluate."""

    id: str = Field(
        ...,
        description="Unique option identifier",
        min_length=1,
        max_length=100
    )
    label: str = Field(
        ...,
        description="Human-readable option name",
        min_length=1,
        max_length=200
    )
    interventions: Dict[str, float] = Field(
        ...,
        description="Intervention values for this option (node_id → value)"
    )
    is_baseline: bool = Field(
        default=False,
        description="Whether this is the baseline/status quo option"
    )

    @field_validator("interventions")
    @classmethod
    def validate_interventions_size(cls, v):
        """Validate interventions dict size."""
        from src.utils.security_validators import validate_dict_size
        validate_dict_size(v, "interventions")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "option_a",
                "label": "Aggressive Marketing",
                "interventions": {"marketing_spend": 100000, "price": 49.99},
                "is_baseline": False
            }
        }
    }


class RobustnessRequest(BaseModel):
    """
    Request for unified robustness analysis.

    Accepts a graph, decision options, utility specification, and analysis options.
    Returns a comprehensive robustness analysis in a single response.
    """

    graph: GraphV1 = Field(
        ...,
        description="Decision graph with nodes and edges"
    )
    options: List[DecisionOption] = Field(
        ...,
        description="Decision options to evaluate",
        min_length=2,
        max_length=20
    )
    utility: UtilitySpecification = Field(
        ...,
        description="How to compute utility/ranking"
    )
    analysis_options: Optional[AnalysisOptions] = Field(
        default=None,
        description="Configuration for the analysis"
    )

    # Optional: parameter uncertainties for VoI
    parameter_uncertainties: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Uncertainty specification for parameters: {param_id: {mean: x, std: y}}"
    )

    @field_validator("options")
    @classmethod
    def validate_unique_option_ids(cls, v):
        """Ensure all option IDs are unique."""
        ids = [opt.id for opt in v]
        if len(ids) != len(set(ids)):
            raise ValueError("All option IDs must be unique")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "graph": {
                    "nodes": [
                        {"id": "marketing_spend", "kind": "decision", "label": "Marketing Spend"},
                        {"id": "price", "kind": "decision", "label": "Product Price"},
                        {"id": "demand", "kind": "factor", "label": "Customer Demand"},
                        {"id": "revenue", "kind": "goal", "label": "Revenue"}
                    ],
                    "edges": [
                        {"from": "marketing_spend", "to": "demand", "weight": 2.0},
                        {"from": "price", "to": "demand", "weight": -1.5},
                        {"from": "demand", "to": "revenue", "weight": 2.5}
                    ]
                },
                "options": [
                    {
                        "id": "option_a",
                        "label": "Aggressive Marketing",
                        "interventions": {"marketing_spend": 100000, "price": 49.99},
                        "is_baseline": False
                    },
                    {
                        "id": "option_b",
                        "label": "Premium Pricing",
                        "interventions": {"marketing_spend": 50000, "price": 79.99},
                        "is_baseline": True
                    }
                ],
                "utility": {
                    "goal_node_id": "revenue",
                    "maximize": True
                }
            }
        }
    }


# ============================================================================
# Outcome Logging Models (Task 8)
# ============================================================================


class OutcomeLog(BaseModel):
    """
    Outcome logging for future calibration.

    Records decisions and their outcomes for analyzing recommendation accuracy.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Auto-generated UUID"
    )
    decision_id: str = Field(
        ...,
        description="Unique identifier for this decision"
    )
    graph_hash: str = Field(
        ...,
        description="Hash of the graph structure"
    )
    response_hash: str = Field(
        ...,
        description="Hash of the analysis response"
    )
    chosen_option: str = Field(
        ...,
        description="Option ID the user chose"
    )
    recommendation_option: str = Field(
        ...,
        description="Option ID ISL recommended"
    )
    recommendation_followed: bool = Field(
        ...,
        description="Whether user followed the recommendation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When decision was logged"
    )
    outcome_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Actual outcome values (filled in later)"
    )
    outcome_timestamp: Optional[datetime] = Field(
        default=None,
        description="When outcome was recorded"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant/organization identifier"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "log_abc123",
                "decision_id": "decision_456",
                "graph_hash": "sha256_abc...",
                "response_hash": "sha256_def...",
                "chosen_option": "option_a",
                "recommendation_option": "option_a",
                "recommendation_followed": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "outcome_values": {"revenue": 155000.0},
                "user_id": "user_789",
                "tenant_id": "tenant_xyz"
            }
        }
    }


class OutcomeLogRequest(BaseModel):
    """Request to log a decision outcome."""

    decision_id: str = Field(
        ...,
        description="Unique identifier for this decision"
    )
    graph_hash: str = Field(
        ...,
        description="Hash of the graph structure"
    )
    response_hash: str = Field(
        ...,
        description="Hash of the analysis response"
    )
    chosen_option: str = Field(
        ...,
        description="Option ID the user chose"
    )
    recommendation_option: str = Field(
        ...,
        description="Option ID ISL recommended"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant/organization identifier"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes"
    )


class OutcomeUpdateRequest(BaseModel):
    """Request to update an outcome log with actual values."""

    outcome_values: Dict[str, float] = Field(
        ...,
        description="Actual outcome values"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes about the outcome"
    )


class OutcomeSummary(BaseModel):
    """Summary statistics for outcome logging."""

    total_logged: int = Field(..., description="Total decisions logged")
    with_outcomes: int = Field(..., description="Decisions with recorded outcomes")
    recommendations_followed: int = Field(
        ...,
        description="Number where recommendation was followed"
    )
    recommendations_followed_pct: float = Field(
        ...,
        description="Percentage of recommendations followed"
    )
    avg_outcome_when_followed: Optional[float] = Field(
        default=None,
        description="Average outcome when recommendation followed"
    )
    avg_outcome_when_not_followed: Optional[float] = Field(
        default=None,
        description="Average outcome when recommendation not followed"
    )
    metadata: Optional[ISLResponseMetadata] = Field(
        default=None,
        description="Response metadata",
        alias="_metadata"
    )

    model_config = {
        "populate_by_name": True
    }
