"""
Request Pydantic models for API endpoints.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .shared import DAGStructure, Distribution, DistributionType, StructuralModel


class CausalValidationRequest(BaseModel):
    """Request model for causal validation endpoint."""

    dag: DAGStructure = Field(..., description="Directed acyclic graph structure")
    treatment: str = Field(
        ...,
        description="Treatment variable name",
        min_length=1,
        max_length=100
    )
    outcome: str = Field(
        ...,
        description="Outcome variable name",
        min_length=1,
        max_length=100
    )
    observed: Optional[List[str]] = Field(
        default=None,
        description="Optional list of observed variables",
        max_length=50
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "dag": {
                    "nodes": ["Price", "Brand", "Revenue", "CustomerAcquisition"],
                    "edges": [
                        ["Price", "Revenue"],
                        ["Brand", "Price"],
                        ["Brand", "Revenue"],
                        ["CustomerAcquisition", "Revenue"],
                    ],
                },
                "treatment": "Price",
                "outcome": "Revenue",
            }
        }
    }


class CounterfactualRequest(BaseModel):
    """Request model for counterfactual analysis endpoint."""

    model: StructuralModel = Field(..., description="Structural causal model")
    intervention: Dict[str, float] = Field(
        ...,
        description="Intervention values mapping variable names to values",
    )
    outcome: str = Field(
        ...,
        description="Outcome variable to predict",
        min_length=1,
        max_length=100
    )
    context: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional observed context values",
    )
    # Forward compatibility fields for Phase 1+
    preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Reserved for ActiVA preferences (Phase 1)",
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Reserved for Bayesian Teaching context (Phase 1)",
    )

    @field_validator("intervention", "context")
    @classmethod
    def validate_dict_size(cls, v, info):
        """Validate dictionary sizes."""
        if v is not None:
            from src.utils.security_validators import validate_dict_size
            field_name = info.field_name
            validate_dict_size(v, field_name)
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": {
                    "variables": ["Price", "Brand", "Revenue"],
                    "equations": {
                        "Brand": "baseline_brand + 0.3 * Price",
                        "Revenue": "10000 + 500 * Price - 200 * Brand",
                    },
                    "distributions": {
                        "baseline_brand": {
                            "type": "normal",
                            "parameters": {"mean": 50, "std": 5},
                        }
                    },
                },
                "intervention": {"Price": 15},
                "outcome": "Revenue",
                "context": {"baseline_brand": 52},
            }
        }
    }


class TeamPerspective(BaseModel):
    """Individual team member's perspective."""

    role: str = Field(
        ...,
        description="Role name (e.g., PM, Designer, Engineer)",
        min_length=1,
        max_length=100
    )
    priorities: List[str] = Field(
        ...,
        description="What this role cares about",
        max_length=20
    )
    constraints: List[str] = Field(
        ...,
        description="What limits this role",
        max_length=20
    )
    preferred_options: Optional[List[str]] = Field(
        default=None,
        description="Preferred option IDs",
        max_length=50
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "role": "Product Manager",
                "priorities": ["User acquisition", "Revenue growth", "Fast time-to-market"],
                "constraints": ["Limited budget", "Q4 deadline"],
                "preferred_options": ["option_a", "option_b"],
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
    name: str = Field(
        ...,
        description="Human-readable option name",
        min_length=1,
        max_length=200
    )
    attributes: Dict[str, Any] = Field(
        ...,
        description="Characteristics of this option",
    )

    @field_validator("attributes")
    @classmethod
    def validate_attributes_size(cls, v):
        """Validate attributes dict size."""
        from src.utils.security_validators import validate_dict_size
        validate_dict_size(v, "attributes")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "option_a",
                "name": "Quick MVP launch",
                "attributes": {
                    "speed": "fast",
                    "quality": "medium",
                    "acquisition_potential": "high",
                },
            }
        }
    }


class TeamAlignmentRequest(BaseModel):
    """Request model for team alignment endpoint."""

    perspectives: List[TeamPerspective] = Field(
        ...,
        description="Team member perspectives",
        min_length=2,
        max_length=20
    )
    options: List[DecisionOption] = Field(
        ...,
        description="Decision options to evaluate",
        min_length=2,
        max_length=50
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "perspectives": [
                    {
                        "role": "Product Manager",
                        "priorities": ["User acquisition", "Revenue growth"],
                        "constraints": ["Limited budget"],
                        "preferred_options": ["option_a"],
                    },
                    {
                        "role": "Designer",
                        "priorities": ["User experience", "Brand consistency"],
                        "constraints": ["Design system limitations"],
                        "preferred_options": ["option_b"],
                    },
                ],
                "options": [
                    {
                        "id": "option_a",
                        "name": "Quick MVP",
                        "attributes": {"speed": "fast", "quality": "medium"},
                    },
                    {
                        "id": "option_b",
                        "name": "Polished feature",
                        "attributes": {"speed": "medium", "quality": "high"},
                    },
                ],
            }
        }
    }


class Assumption(BaseModel):
    """An assumption to test in sensitivity analysis."""

    name: str = Field(
        ...,
        description="Assumption name",
        min_length=1,
        max_length=200
    )
    current_value: Union[str, float, Dict[str, Any]] = Field(
        ...,
        description="Current assumed value",
    )
    type: str = Field(
        ...,
        description="Assumption type: parametric, structural, or distributional",
        min_length=1,
        max_length=50
    )
    variation_range: Optional[Dict[str, float]] = Field(
        default=None,
        description="Range to test (min/max values)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Customer price sensitivity",
                "current_value": 0.5,
                "type": "parametric",
                "variation_range": {"min": 0.3, "max": 0.8},
            }
        }
    }


class SensitivityAnalysisRequest(BaseModel):
    """Request model for sensitivity analysis endpoint."""

    model: StructuralModel = Field(..., description="Structural causal model")
    baseline_result: float = Field(..., description="Baseline analysis result")
    assumptions: List[Assumption] = Field(
        ...,
        description="Assumptions to test",
        min_length=1,
        max_length=30
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": {
                    "variables": ["Price", "Revenue"],
                    "equations": {"Revenue": "10000 + sensitivity * Price"},
                    "distributions": {
                        "sensitivity": {"type": "normal", "parameters": {"mean": 500, "std": 50}}
                    },
                },
                "baseline_result": 51000,
                "assumptions": [
                    {
                        "name": "Price sensitivity",
                        "current_value": 500,
                        "type": "parametric",
                        "variation_range": {"min": 300, "max": 700},
                    }
                ],
            }
        }
    }
