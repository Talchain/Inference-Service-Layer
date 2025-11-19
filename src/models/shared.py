"""
Shared Pydantic models used across requests and responses.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    """Confidence level for analysis results."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RobustnessLevel(str, Enum):
    """Robustness level for analysis results."""

    ROBUST = "robust"
    MODERATE = "moderate"
    FRAGILE = "fragile"


class ImportanceLevel(str, Enum):
    """Importance level for assumptions."""

    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"


class ValidationStatus(str, Enum):
    """Status of causal validation."""

    IDENTIFIABLE = "identifiable"
    UNCERTAIN = "uncertain"
    CANNOT_IDENTIFY = "cannot_identify"


class ValidationIssueType(str, Enum):
    """Type of validation issue."""

    MISSING_CONNECTION = "missing_connection"
    CONFOUNDING = "confounding"
    AMBIGUOUS_RELATIONSHIP = "ambiguous_relationship"


class ConflictSeverity(str, Enum):
    """Severity of team alignment conflict."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class UncertaintyLevel(str, Enum):
    """Overall uncertainty level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DistributionType(str, Enum):
    """Types of probability distributions."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    BETA = "beta"
    EXPONENTIAL = "exponential"


class DAGStructure(BaseModel):
    """Directed Acyclic Graph structure."""

    nodes: List[str] = Field(..., description="List of node names in the graph")
    edges: List[Tuple[str, str]] = Field(
        ...,
        description="List of directed edges as (from, to) tuples",
    )

    @field_validator("nodes")
    @classmethod
    def validate_nodes_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure at least one node exists."""
        if not v:
            raise ValueError("DAG must contain at least one node")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "nodes": ["Price", "Brand", "Revenue"],
                "edges": [["Price", "Revenue"], ["Brand", "Price"], ["Brand", "Revenue"]],
            }
        }
    }


class Distribution(BaseModel):
    """Probability distribution specification."""

    type: DistributionType = Field(..., description="Type of distribution")
    parameters: Dict[str, float] = Field(..., description="Distribution parameters")

    model_config = {
        "json_schema_extra": {
            "example": {"type": "normal", "parameters": {"mean": 0, "std": 1}}
        }
    }


class StructuralModel(BaseModel):
    """Structural causal model specification."""

    variables: List[str] = Field(..., description="List of variable names")
    equations: Dict[str, str] = Field(
        ...,
        description="Structural equations mapping variable to expression",
    )
    distributions: Dict[str, Distribution] = Field(
        ...,
        description="Prior distributions for exogenous variables",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "variables": ["X", "Y", "Z"],
                "equations": {"Y": "10 + 2*X + 3*Z", "Z": "5 + 0.5*X"},
                "distributions": {
                    "X": {"type": "normal", "parameters": {"mean": 0, "std": 1}}
                },
            }
        }
    }


class ExplanationMetadata(BaseModel):
    """Explanation metadata included in all responses."""

    summary: str = Field(..., description="One-line summary of the result")
    reasoning: str = Field(..., description="Plain English explanation of why")
    technical_basis: str = Field(..., description="Mathematical/technical justification")
    assumptions: List[str] = Field(..., description="Key assumptions made in analysis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "summary": "Effect is identifiable by controlling for Brand",
                "reasoning": "Brand influences both Price and Revenue, creating confounding. "
                "Controlling for Brand blocks the backdoor path.",
                "technical_basis": "Backdoor criterion satisfied with adjustment set {Brand}",
                "assumptions": [
                    "No unmeasured confounding",
                    "Correct causal structure specified",
                ],
            }
        }
    }


class ConfidenceInterval(BaseModel):
    """Confidence interval for numerical estimates."""

    lower: float = Field(..., description="Lower bound of confidence interval")
    upper: float = Field(..., description="Upper bound of confidence interval")
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level (default 95%)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"lower": 45000, "upper": 55000, "confidence_level": 0.95}
        }
    }


class SensitivityRange(BaseModel):
    """Sensitivity range showing best/worst plausible cases."""

    optimistic: float = Field(..., description="Optimistic scenario value")
    pessimistic: float = Field(..., description="Pessimistic scenario value")
    explanation: str = Field(..., description="What creates this range")

    model_config = {
        "json_schema_extra": {
            "example": {
                "optimistic": 62000,
                "pessimistic": 38000,
                "explanation": "Range accounts for uncertainty in competitive response",
            }
        }
    }
