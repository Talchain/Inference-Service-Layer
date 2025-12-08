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
    DEGRADED = "degraded"  # Y₀ analysis failed, using fallback


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

    nodes: List[str] = Field(
        ...,
        description="List of node names in the graph",
        min_length=1,
        max_length=50
    )
    edges: List[Tuple[str, str]] = Field(
        ...,
        description="List of directed edges as (from, to) tuples",
        max_length=200
    )

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, v: List[str]) -> List[str]:
        """Validate nodes: not empty, no duplicates, valid identifiers."""
        from src.utils.security_validators import (
            validate_no_duplicate_nodes,
            validate_node_names
        )

        if not v:
            raise ValueError("DAG must contain at least one node")

        validate_no_duplicate_nodes(v)
        validate_node_names(v)

        return v

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, v: List[Tuple[str, str]], info) -> List[Tuple[str, str]]:
        """Validate edges: no self-loops, reference existing nodes."""
        from src.utils.security_validators import (
            validate_no_self_loops,
            validate_edges_reference_nodes
        )

        validate_no_self_loops(v)

        # Validate edges reference nodes (if nodes already validated)
        if 'nodes' in info.data:
            validate_edges_reference_nodes(v, info.data['nodes'])

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

    variables: List[str] = Field(
        ...,
        description="List of variable names",
        max_length=50
    )
    equations: Dict[str, str] = Field(
        ...,
        description="Structural equations mapping variable to expression",
    )
    distributions: Dict[str, Distribution] = Field(
        ...,
        description="Prior distributions for exogenous variables",
    )

    @field_validator("variables")
    @classmethod
    def validate_variable_names(cls, v: List[str]) -> List[str]:
        """Validate variables are valid identifiers."""
        from src.utils.security_validators import validate_node_names
        validate_node_names(v)
        return v

    @field_validator("equations")
    @classmethod
    def validate_equations(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate equations contain only safe characters."""
        from src.utils.security_validators import (
            validate_equations_safe,
            validate_dict_size
        )

        validate_dict_size(v, "equations")
        validate_equations_safe(v)

        return v

    @field_validator("distributions")
    @classmethod
    def validate_distributions_size(cls, v: Dict[str, Distribution]) -> Dict[str, Distribution]:
        """Validate distributions dict size."""
        from src.utils.security_validators import validate_dict_size
        validate_dict_size(v, "distributions")
        return v

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

    # Enhanced explanation fields (Feature 6)
    simple_explanation: Optional[str] = Field(
        None,
        description="Non-technical explanation for general audience",
        max_length=200
    )
    learn_more_url: Optional[str] = Field(
        None,
        description="Link to documentation for this concept"
    )
    visual_type: Optional[str] = Field(
        None,
        description="Suggested visualization type (e.g., 'path_diagram', 'interval_plot')"
    )

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
                "simple_explanation": "We can estimate this effect by controlling for Brand.",
                "learn_more_url": "https://docs.inference-service-layer.com/docs/methods/backdoor-adjustment",
                "visual_type": "dag_with_adjustment"
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


class NodeKind(str, Enum):
    """Node kind for GraphV1."""

    GOAL = "goal"
    DECISION = "decision"
    OPTION = "option"
    OUTCOME = "outcome"
    RISK = "risk"
    ACTION = "action"


class GraphNodeV1(BaseModel):
    """Node in GraphV1 structure."""

    id: str = Field(..., description="Unique node identifier", max_length=100)
    kind: NodeKind = Field(..., description="Type of node")
    label: str = Field(..., description="Human-readable label", max_length=500)
    body: Optional[str] = Field(None, description="Detailed description", max_length=5000)
    belief: Optional[float] = Field(
        None,
        description="Belief probability (0-1) for probabilistic nodes",
        ge=0.0,
        le=1.0
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate node ID is a safe identifier."""
        from src.utils.security_validators import validate_node_names
        validate_node_names([v])
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "n_market_size",
                "kind": "outcome",
                "label": "Market Size Estimate",
                "body": "Total addressable market for product launch",
                "belief": 0.85,
                "metadata": {"unit": "millions", "currency": "USD"}
            }
        }
    }


class GraphEdgeV1(BaseModel):
    """Edge in GraphV1 structure."""

    from_: str = Field(..., alias="from", description="Source node ID", max_length=100)
    to: str = Field(..., description="Target node ID", max_length=100)
    weight: Optional[float] = Field(
        None,
        description="Edge weight (-3 to +3)",
        ge=-3.0,
        le=3.0
    )
    label: Optional[str] = Field(None, description="Edge label", max_length=500)

    model_config = {
        "json_schema_extra": {
            "example": {
                "from": "n_marketing_budget",
                "to": "n_market_penetration",
                "weight": 2.5,
                "label": "Strong positive influence"
            }
        }
    }


class GraphV1(BaseModel):
    """
    GraphV1 structure for CEE decision reviews.

    Represents causal decision graphs with typed nodes and weighted edges.
    """

    nodes: List[GraphNodeV1] = Field(
        ...,
        description="List of graph nodes",
        min_length=1,
        max_length=100
    )
    edges: List[GraphEdgeV1] = Field(
        ...,
        description="List of directed edges",
        max_length=300
    )

    @field_validator("nodes")
    @classmethod
    def validate_unique_node_ids(cls, v: List[GraphNodeV1]) -> List[GraphNodeV1]:
        """Validate node IDs are unique."""
        node_ids = [node.id for node in v]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [id for id in node_ids if node_ids.count(id) > 1]
            raise ValueError(f"Duplicate node IDs found: {duplicates}")
        return v

    @field_validator("edges")
    @classmethod
    def validate_edges_reference_nodes(cls, v: List[GraphEdgeV1], info) -> List[GraphEdgeV1]:
        """Validate edges reference existing nodes."""
        if 'nodes' in info.data:
            node_ids = {node.id for node in info.data['nodes']}
            for edge in v:
                if edge.from_ not in node_ids:
                    raise ValueError(f"Edge references non-existent node: {edge.from_}")
                if edge.to not in node_ids:
                    raise ValueError(f"Edge references non-existent node: {edge.to}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "nodes": [
                    {
                        "id": "n_launch_product",
                        "kind": "decision",
                        "label": "Launch New Product",
                        "body": "Decision to launch product in Q1"
                    },
                    {
                        "id": "n_market_success",
                        "kind": "outcome",
                        "label": "Market Success",
                        "belief": 0.75
                    }
                ],
                "edges": [
                    {
                        "from": "n_launch_product",
                        "to": "n_market_success",
                        "weight": 2.0,
                        "label": "Expected positive impact"
                    }
                ]
            }
        }
    }
