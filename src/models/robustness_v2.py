"""
Robustness analysis models for v2.2 dual uncertainty schema.

Supports both structural uncertainty (edge existence) and parametric
uncertainty (effect magnitude) for proper robustness analysis.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
import re

from src.models.response_v2 import CritiqueV2


# =============================================================================
# Enums
# =============================================================================

class NodeKindV2(str, Enum):
    """Node types in v2 causal graphs."""
    FACTOR = "factor"
    DECISION = "decision"
    CHANCE = "chance"
    OPTION = "option"
    OUTCOME = "outcome"
    GOAL = "goal"
    RISK = "risk"
    ACTION = "action"


class SensitivityType(str, Enum):
    """Types of sensitivity analysis."""
    EXISTENCE = "existence"
    MAGNITUDE = "magnitude"


# =============================================================================
# Core V2 Schema Components
# =============================================================================

class StrengthDistribution(BaseModel):
    """
    Parametric uncertainty over edge effect magnitude.

    Represents a Normal distribution over the causal effect strength.
    Positive mean = positive causal effect (increase in cause -> increase in effect)
    Negative mean = negative causal effect (increase in cause -> decrease in effect)
    """

    mean: float = Field(
        ...,
        description="Expected effect size (SIGNED: negative = negative effect)"
    )
    std: float = Field(
        ...,
        gt=0,
        description="Standard deviation of effect size"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "mean": 0.5,
                "std": 0.1
            }
        }
    }


class ObservedState(BaseModel):
    """
    Observed state for quantitative factor nodes.

    Captures the current observed value of a factor, along with optional
    baseline for comparison, display unit, and data provenance.

    This supports the v2.2 schema where factor nodes can carry actual
    observed values from CEE extraction or user input.
    """

    value: float = Field(
        ...,
        description="Current observed value in user units (e.g., 59 for £59k revenue)"
    )
    baseline: Optional[float] = Field(
        None,
        description="Reference/baseline value for comparison (e.g., 49 for £49k baseline)"
    )
    unit: Optional[str] = Field(
        None,
        max_length=50,
        description="Display unit (e.g., '£', '%', 'users', 'k')"
    )
    source: Optional[str] = Field(
        None,
        max_length=100,
        description="Data provenance (e.g., 'brief_extraction', 'user_input', 'computed')"
    )

    @field_validator("value")
    @classmethod
    def value_must_be_finite(cls, v: float) -> float:
        """Validate that value is a finite number (not NaN or infinity)."""
        if not math.isfinite(v):
            raise ValueError("value must be finite (not NaN or infinity)")
        return v

    @field_validator("baseline")
    @classmethod
    def baseline_must_be_finite(cls, v: Optional[float]) -> Optional[float]:
        """Validate that baseline, if provided, is finite."""
        if v is not None and not math.isfinite(v):
            raise ValueError("baseline must be finite (not NaN or infinity)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "value": 59.0,
                "baseline": 49.0,
                "unit": "£k",
                "source": "brief_extraction"
            }
        }
    }


class ParameterUncertainty(BaseModel):
    """
    Uncertainty specification for a factor node's value.

    Defines how to sample a factor's value during Monte Carlo analysis.
    The mean is typically taken from the node's `observed_state.value`.

    Supported distributions:
    - "normal": Sample from Normal(observed_value, std)
    - "uniform": Sample uniformly from [range_min, range_max]
    - "point_mass": Use observed_value exactly (no sampling)
    """

    node_id: str = Field(
        ...,
        pattern=r"^[a-z0-9_:-]+$",
        description="ID of the factor node this uncertainty applies to"
    )
    distribution: str = Field(
        default="normal",
        description="Distribution family: 'normal', 'uniform', 'point_mass'"
    )
    std: Optional[float] = Field(
        None,
        ge=0,
        description="Standard deviation for Normal sampling around observed_state.value"
    )
    # For uniform distribution
    range_min: Optional[float] = Field(
        None,
        description="Minimum value for uniform distribution"
    )
    range_max: Optional[float] = Field(
        None,
        description="Maximum value for uniform distribution"
    )

    @model_validator(mode="after")
    def validate_distribution_params(self) -> "ParameterUncertainty":
        """Validate distribution-specific parameters."""
        if self.distribution == "normal":
            if self.std is None or self.std <= 0:
                raise ValueError(
                    f"For normal distribution, 'std' must be provided and > 0 "
                    f"(got std={self.std})"
                )
        elif self.distribution == "uniform":
            if self.range_min is None or self.range_max is None:
                raise ValueError(
                    "For uniform distribution, both 'range_min' and 'range_max' must be provided"
                )
            if self.range_min >= self.range_max:
                raise ValueError(
                    f"For uniform distribution, range_min ({self.range_min}) "
                    f"must be less than range_max ({self.range_max})"
                )
        elif self.distribution == "point_mass":
            pass  # No additional params needed
        else:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. "
                f"Supported: 'normal', 'uniform', 'point_mass'"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "node_id": "marketing_spend",
                "distribution": "normal",
                "std": 2.5
            }
        }
    }


class EdgeV2(BaseModel):
    """
    Edge with dual uncertainty.

    Combines structural uncertainty (does the edge exist?) with
    parametric uncertainty (how strong is the effect?).
    """

    from_: str = Field(
        ...,
        alias="from",
        pattern=r"^[a-z0-9_:-]+$",
        description="Source node ID"
    )
    to: str = Field(
        ...,
        pattern=r"^[a-z0-9_:-]+$",
        description="Target node ID"
    )
    exists_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="P(edge exists) - structural uncertainty"
    )
    strength: StrengthDistribution = Field(
        ...,
        description="Effect magnitude distribution - parametric uncertainty"
    )
    label: Optional[str] = Field(
        None,
        description="Human-readable edge description",
        max_length=500
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "from": "marketing",
                "to": "demand",
                "exists_probability": 0.9,
                "strength": {"mean": 0.6, "std": 0.15},
                "label": "Marketing increases demand"
            }
        },
        "populate_by_name": True
    }


class NodeV2(BaseModel):
    """Node in the v2 causal graph."""

    id: str = Field(
        ...,
        pattern=r"^[a-z0-9_:-]+$",
        description="Unique node identifier"
    )
    kind: str = Field(
        ...,
        description="Node type (factor, decision, chance, outcome, etc.)"
    )
    label: str = Field(
        ...,
        description="Human-readable node name",
        max_length=500
    )
    body: Optional[str] = Field(
        None,
        description="Detailed description",
        max_length=5000
    )
    observed_state: Optional[ObservedState] = Field(
        None,
        description="Observed state for quantitative factor nodes (value, baseline, unit, source)"
    )
    intercept: float = Field(
        default=0.0,
        description="Node intercept term (constant added to structural equation). "
                    "Represents the baseline value when all parent contributions are zero."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "revenue",
                "kind": "outcome",
                "label": "Total Revenue",
                "intercept": 0.0,
                "observed_state": {
                    "value": 59.0,
                    "baseline": 49.0,
                    "unit": "£k"
                }
            }
        }
    }


class GraphV2(BaseModel):
    """
    Causal graph with dual uncertainty edges.

    Represents a directed acyclic graph where each edge has both
    structural uncertainty (exists_probability) and parametric
    uncertainty (strength distribution).
    """

    nodes: List[NodeV2] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of graph nodes"
    )
    edges: List[EdgeV2] = Field(
        ...,
        max_length=300,
        description="List of directed edges with dual uncertainty"
    )

    @field_validator("nodes")
    @classmethod
    def validate_unique_node_ids(cls, v: List[NodeV2]) -> List[NodeV2]:
        """Validate node IDs are unique."""
        node_ids = [node.id for node in v]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
            raise ValueError(f"Duplicate node IDs found: {list(set(duplicates))}")
        return v

    @field_validator("edges")
    @classmethod
    def validate_edges_reference_nodes(
        cls, v: List[EdgeV2], info
    ) -> List[EdgeV2]:
        """Validate edges reference existing nodes."""
        if "nodes" in info.data:
            node_ids = {node.id for node in info.data["nodes"]}
            for edge in v:
                if edge.from_ not in node_ids:
                    raise ValueError(
                        f"Edge references non-existent source node: {edge.from_}"
                    )
                if edge.to not in node_ids:
                    raise ValueError(
                        f"Edge references non-existent target node: {edge.to}"
                    )
        return v

    @field_validator("edges")
    @classmethod
    def validate_no_self_loops(cls, v: List[EdgeV2]) -> List[EdgeV2]:
        """Validate no self-loops exist."""
        for edge in v:
            if edge.from_ == edge.to:
                raise ValueError(f"Self-loop detected on node: {edge.from_}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "nodes": [
                    {"id": "price", "kind": "decision", "label": "Price"},
                    {"id": "revenue", "kind": "outcome", "label": "Revenue"}
                ],
                "edges": [
                    {
                        "from": "price",
                        "to": "revenue",
                        "exists_probability": 0.95,
                        "strength": {"mean": 0.5, "std": 0.1}
                    }
                ]
            }
        }
    }


class InterventionOption(BaseModel):
    """A decision option with its interventions."""

    id: str = Field(
        ...,
        description="Unique option identifier"
    )
    label: str = Field(
        ...,
        description="Human-readable option name",
        max_length=500
    )
    interventions: Dict[str, float] = Field(
        ...,
        description="node_id -> intervention value mapping"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "low_price",
                "label": "Keep price at $49",
                "interventions": {"price": 0.49}
            }
        }
    }


# =============================================================================
# Request Schema
# =============================================================================

class RobustnessRequestV2(BaseModel):
    """
    V2.2 robustness analysis request.

    Accepts a causal graph with dual uncertainty edges, a set of decision
    options, and configuration for Monte Carlo sampling and analysis.
    """

    request_id: Optional[str] = Field(
        None,
        description="Optional request ID for tracing. Generated if not provided."
    )
    graph: GraphV2 = Field(
        ...,
        description="Causal graph with dual uncertainty edges"
    )
    options: List[InterventionOption] = Field(
        ...,
        min_length=1,
        description="Decision options to compare"
    )
    goal_node_id: str = Field(
        ...,
        description="Target outcome node to optimize"
    )

    # Sampling configuration
    n_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of Monte Carlo samples"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility; if None, computed from graph"
    )

    # Analysis configuration
    analysis_types: List[str] = Field(
        default=["comparison", "sensitivity", "robustness"],
        description="Types of analysis to perform"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals"
    )

    # Factor uncertainty configuration (Phase 2A Part 2)
    parameter_uncertainties: Optional[List[ParameterUncertainty]] = Field(
        None,
        description="Uncertainty specifications for factor node values. "
        "If not provided, factor nodes use observed_state.value as fixed values."
    )

    # Goal threshold configuration
    goal_threshold: Optional[float] = Field(
        None,
        description="Success threshold for goal outcome. When provided, "
        "computes probability_of_goal (fraction of samples meeting/exceeding threshold)."
    )

    @field_validator("goal_threshold")
    @classmethod
    def validate_goal_threshold_finite(cls, v: Optional[float]) -> Optional[float]:
        """Reject NaN and infinite values for goal_threshold."""
        import math
        if v is not None and (math.isnan(v) or math.isinf(v)):
            raise ValueError("goal_threshold must be a finite number, not NaN or infinite")
        return v

    @field_validator("goal_node_id")
    @classmethod
    def validate_goal_node_exists(cls, v: str, info) -> str:
        """Validate goal node exists in graph."""
        if "graph" in info.data:
            node_ids = {node.id for node in info.data["graph"].nodes}
            if v not in node_ids:
                raise ValueError(f"Goal node '{v}' not found in graph")
        return v

    @model_validator(mode="after")
    def validate_interventions_reference_nodes(self) -> "RobustnessRequestV2":
        """Validate all intervention nodes exist in graph."""
        node_ids = {node.id for node in self.graph.nodes}
        for option in self.options:
            for node_id in option.interventions.keys():
                if node_id not in node_ids:
                    raise ValueError(
                        f"Option '{option.id}' references non-existent node: {node_id}"
                    )
        return self

    @model_validator(mode="after")
    def validate_parameter_uncertainties_reference_nodes(self) -> "RobustnessRequestV2":
        """Validate all parameter uncertainty node_ids exist in graph."""
        if self.parameter_uncertainties:
            node_ids = {node.id for node in self.graph.nodes}
            for uncertainty in self.parameter_uncertainties:
                if uncertainty.node_id not in node_ids:
                    raise ValueError(
                        f"ParameterUncertainty references non-existent node: {uncertainty.node_id}"
                    )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req-001",
                "graph": {
                    "nodes": [
                        {"id": "price", "kind": "decision", "label": "Price"},
                        {"id": "revenue", "kind": "outcome", "label": "Revenue"}
                    ],
                    "edges": [
                        {
                            "from": "price",
                            "to": "revenue",
                            "exists_probability": 0.9,
                            "strength": {"mean": -0.5, "std": 0.15}
                        }
                    ]
                },
                "options": [
                    {"id": "low", "label": "Low price", "interventions": {"price": 0.3}},
                    {"id": "high", "label": "High price", "interventions": {"price": 0.7}}
                ],
                "goal_node_id": "revenue",
                "n_samples": 1000
            }
        }
    }


# =============================================================================
# Response Schema
# =============================================================================

class OutcomeDistribution(BaseModel):
    """Distribution of outcomes from Monte Carlo sampling."""

    mean: float = Field(..., description="Mean outcome value")
    std: float = Field(..., description="Standard deviation")
    median: float = Field(..., description="Median outcome value")
    ci_lower: float = Field(..., description="Lower bound of confidence interval")
    ci_upper: float = Field(..., description="Upper bound of confidence interval")
    samples: Optional[List[float]] = Field(
        None,
        description="Raw samples if requested"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "mean": 50000.0,
                "std": 5000.0,
                "median": 49500.0,
                "ci_lower": 40000.0,
                "ci_upper": 60000.0
            }
        }
    }


class OptionResult(BaseModel):
    """Results for a single decision option."""

    option_id: str = Field(..., description="Option identifier")
    outcome_distribution: OutcomeDistribution = Field(
        ...,
        description="Distribution of outcomes"
    )
    win_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="P(this option is best)"
    )
    probability_of_goal: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="P(outcome >= goal_threshold). Only present when goal_threshold is provided in request."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "opt1",
                "outcome_distribution": {
                    "mean": 50000.0,
                    "std": 5000.0,
                    "median": 49500.0,
                    "ci_lower": 40000.0,
                    "ci_upper": 60000.0
                },
                "win_probability": 0.65,
                "probability_of_goal": 0.72
            }
        }
    }


class SensitivityResult(BaseModel):
    """Sensitivity to a single edge."""

    edge_from: str = Field(..., description="Source node of edge")
    edge_to: str = Field(..., description="Target node of edge")
    sensitivity_type: str = Field(
        ...,
        description="Type: 'existence' or 'magnitude'"
    )
    elasticity: float = Field(
        ...,
        description="% change in outcome per % change in parameter"
    )
    importance_rank: int = Field(
        ...,
        ge=1,
        description="Rank by importance (1 = most important)"
    )
    interpretation: str = Field(
        ...,
        description="Human-readable explanation"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "edge_from": "marketing",
                "edge_to": "demand",
                "sensitivity_type": "existence",
                "elasticity": 0.45,
                "importance_rank": 1,
                "interpretation": "Decision is moderately sensitive to marketing->demand existence"
            }
        }
    }


class FactorSensitivityResult(BaseModel):
    """Sensitivity to a factor node's value (Phase 2A Part 2)."""

    node_id: str = Field(..., description="Factor node ID")
    node_label: Optional[str] = Field(None, description="Human-readable node label")
    elasticity: float = Field(
        ...,
        description="% change in outcome per % change in factor value"
    )
    importance_rank: int = Field(
        ...,
        ge=1,
        description="Rank by importance (1 = most important)"
    )
    observed_value: Optional[float] = Field(
        None,
        description="Observed value from node's observed_state"
    )
    interpretation: str = Field(
        ...,
        description="Human-readable explanation"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "node_id": "marketing_spend",
                "node_label": "Marketing Spend",
                "elasticity": 0.32,
                "importance_rank": 2,
                "observed_value": 50000.0,
                "interpretation": "Decision is moderately sensitive to marketing_spend value"
            }
        }
    }


class FragileEdgeEnhanced(BaseModel):
    """Enhanced fragile edge data with alternative winner analysis.

    Used internally by the analyzer. Maps 1:1 with FragileEdgeV2 in response_v2.py
    for API responses.
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
        description="Probability of alternative winner in weak-edge scenarios. "
        "0.0 if same option wins (stable), null only if no data available.",
    )


class RobustnessResult(BaseModel):
    """Overall robustness assessment."""

    is_robust: bool = Field(
        ...,
        description="Whether recommendation is robust"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in robustness assessment"
    )
    fragile_edges: List[str] = Field(
        default_factory=list,
        description="Edges that could flip the decision (format: 'from->to')"
    )
    fragile_edges_enhanced: Optional[List[FragileEdgeEnhanced]] = Field(
        default=None,
        description="Enhanced fragile edge data with alternative winner analysis"
    )
    robust_edges: List[str] = Field(
        default_factory=list,
        description="Edges that don't significantly affect decision"
    )
    recommendation_stability: float = Field(
        ...,
        ge=0,
        le=1,
        description="P(same recommendation across samples)"
    )
    interpretation: str = Field(
        ...,
        description="Human-readable robustness summary"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_robust": True,
                "confidence": 0.92,
                "fragile_edges": ["marketing->demand"],
                "robust_edges": ["price->revenue"],
                "recommendation_stability": 0.88,
                "interpretation": "Recommendation is robust with 92% confidence"
            }
        }
    }


class ClampMetrics(BaseModel):
    """Tracks out-of-bounds sampling for diagnostics."""

    total_node_samples: int = Field(
        ...,
        description="Total node value samples"
    )
    clamped_samples: int = Field(
        ...,
        description="Samples that were clamped to bounds"
    )
    clamp_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of samples clamped"
    )
    nodes_with_high_clamp_rate: List[str] = Field(
        default_factory=list,
        description="Nodes with >10% clamp rate"
    )


class ResponseMetadataV2(BaseModel):
    """Execution metadata for v2 responses."""

    schema_version: str = Field(
        default="2.2",
        description="Schema version"
    )
    isl_version: str = Field(
        ...,
        description="ISL service version"
    )
    n_samples_used: int = Field(
        ...,
        description="Actual samples used"
    )
    seed_used: int = Field(
        ...,
        description="Random seed used"
    )
    execution_time_ms: int = Field(
        ...,
        description="Execution time in milliseconds"
    )
    edge_existence_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Actual sampling rates per edge (format: 'from->to': rate)"
    )
    clamp_metrics: Optional[ClampMetrics] = Field(
        None,
        description="Out-of-bounds sampling metrics"
    )
    config_fingerprint: str = Field(
        ...,
        description="Hash of determinism-critical config"
    )
    tie_count: Optional[int] = Field(
        None,
        description="Number of Monte Carlo samples with tied outcomes"
    )
    tie_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Fraction of samples with tied outcomes (tie_count / n_samples)"
    )


class RobustnessResponseV2(BaseModel):
    """V2.2 robustness analysis response."""

    request_id: str = Field(
        ...,
        description="Request identifier for tracing"
    )

    # Core results
    results: List[OptionResult] = Field(
        ...,
        description="Results for each decision option"
    )
    recommended_option_id: str = Field(
        ...,
        description="ID of recommended option"
    )
    recommendation_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in recommendation"
    )

    # Sensitivity analysis
    sensitivity: List[SensitivityResult] = Field(
        default_factory=list,
        description="Sensitivity results per edge"
    )

    # Factor sensitivity analysis (Phase 2A Part 2)
    factor_sensitivity: List[FactorSensitivityResult] = Field(
        default_factory=list,
        description="Sensitivity results per factor node value"
    )

    # Robustness analysis
    robustness: RobustnessResult = Field(
        ...,
        description="Overall robustness assessment"
    )

    # Metadata
    metadata: ResponseMetadataV2 = Field(
        ...,
        description="Execution metadata",
        alias="_metadata"
    )

    # Analysis critiques (warnings about degenerate options, high tie rates, etc.)
    critiques: List[CritiqueV2] = Field(
        default_factory=list,
        description="Analysis critiques and warnings"
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "request_id": "req-001",
                "results": [
                    {
                        "option_id": "opt1",
                        "outcome_distribution": {
                            "mean": 50000.0,
                            "std": 5000.0,
                            "median": 49500.0,
                            "ci_lower": 40000.0,
                            "ci_upper": 60000.0
                        },
                        "win_probability": 0.65
                    }
                ],
                "recommended_option_id": "opt1",
                "recommendation_confidence": 0.65,
                "sensitivity": [],
                "robustness": {
                    "is_robust": True,
                    "confidence": 0.92,
                    "fragile_edges": [],
                    "robust_edges": [],
                    "recommendation_stability": 0.88,
                    "interpretation": "Robust recommendation"
                },
                "_metadata": {
                    "schema_version": "2.2",
                    "isl_version": "1.0.0",
                    "n_samples_used": 1000,
                    "seed_used": 12345,
                    "execution_time_ms": 150,
                    "edge_existence_rates": {},
                    "config_fingerprint": "abc123",
                    "tie_count": 0,
                    "tie_rate": 0.0
                },
                "critiques": []
            }
        }
    }


# =============================================================================
# Schema Detection
# =============================================================================

def detect_schema_version(request: Dict[str, Any]) -> str:
    """
    Detect request schema version from request structure.

    Args:
        request: Raw request dictionary

    Returns:
        "v2" for v2.2 schema, "v1" for legacy schema

    Raises:
        ValueError: If schema cannot be determined
    """
    if "graph" in request and "options" in request:
        return "v2"
    elif "causal_model" in request:
        return "v1"
    else:
        raise ValueError(
            "Unknown request schema - must contain 'graph'+'options' (v2) "
            "or 'causal_model' (v1)"
        )
