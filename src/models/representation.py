"""
Causal representation learning models for extracting factors from unstructured data.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .metadata import ResponseMetadata
from .shared import DAGStructure


class CausalFactor(BaseModel):
    """A causal factor extracted from unstructured data."""

    name: str = Field(
        ...,
        description="Human-readable name for this factor",
        min_length=1,
        max_length=100
    )

    strength: float = Field(
        ...,
        description="Factor coherence/strength (0-1)",
        ge=0.0,
        le=1.0
    )

    representative_texts: List[str] = Field(
        ...,
        description="Example texts that represent this factor",
        min_length=0,
        max_length=10
    )

    keywords: List[str] = Field(
        default_factory=list,
        description="Key terms associated with this factor",
        max_length=20
    )

    prevalence: float = Field(
        default=0.0,
        description="Proportion of data exhibiting this factor (0-1)",
        ge=0.0,
        le=1.0
    )


class ExtractedFactors(BaseModel):
    """Complete result of factor extraction."""

    factors: List[CausalFactor] = Field(
        ...,
        description="Extracted causal factors",
        min_length=0,
        max_length=20
    )

    suggested_dag: Optional[DAGStructure] = Field(
        None,
        description="Suggested DAG structure based on factors"
    )

    confidence: float = Field(
        ...,
        description="Overall confidence in extraction (0-1)",
        ge=0.0,
        le=1.0
    )

    method: str = Field(
        ...,
        description="Method used for extraction",
        max_length=100
    )

    summary: str = Field(
        ...,
        description="Plain English summary of findings",
        max_length=500
    )

    metadata: Optional[ResponseMetadata] = Field(
        None,
        description="Response metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "factors": [
                    {
                        "name": "usability_issues",
                        "strength": 0.82,
                        "representative_texts": ["Can't find settings", "Navigation confusing"],
                        "keywords": ["confusing", "hard", "find", "navigate"],
                        "prevalence": 0.35
                    },
                    {
                        "name": "performance_problems",
                        "strength": 0.76,
                        "representative_texts": ["Slow load times", "App freezes"],
                        "keywords": ["slow", "freeze", "lag", "performance"],
                        "prevalence": 0.28
                    }
                ],
                "suggested_dag": {
                    "nodes": ["usability_issues", "performance_problems", "churn"],
                    "edges": [["usability_issues", "churn"], ["performance_problems", "churn"]]
                },
                "confidence": 0.75,
                "method": "Sentence embedding + clustering",
                "summary": "Identified 2 primary factors driving churn: usability issues (35%) and performance problems (28%)"
            }
        }
    }


class FactorExtractionRequest(BaseModel):
    """Request for factor extraction from unstructured data."""

    data_type: str = Field(
        ...,
        description="Type of data (e.g., 'support_tickets', 'reviews', 'feedback')",
        max_length=50
    )

    texts: List[str] = Field(
        ...,
        description="Raw text data to analyze",
        min_length=10,
        max_length=1000
    )

    domain: Optional[str] = Field(
        None,
        description="Domain context (e.g., 'SaaS product', 'e-commerce')",
        max_length=100
    )

    n_factors: int = Field(
        default=5,
        description="Number of factors to extract",
        ge=2,
        le=20
    )

    outcome_variable: Optional[str] = Field(
        None,
        description="Known outcome variable (e.g., 'churn', 'revenue')",
        max_length=50
    )

    min_cluster_size: int = Field(
        default=3,
        description="Minimum cluster size for factor extraction",
        ge=2,
        le=100
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "data_type": "support_tickets",
                "texts": ["User can't find settings...", "Slow load times...", "..."],
                "domain": "SaaS product",
                "n_factors": 5,
                "outcome_variable": "churn",
                "min_cluster_size": 3
            }
        }
    }
