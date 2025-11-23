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


class InterventionConstraints(BaseModel):
    """Constraints for finding minimal interventions."""

    feasible: List[str] = Field(
        ...,
        description="Variables that can be changed",
        min_length=1,
        max_length=20
    )
    fixed: Optional[List[str]] = Field(
        default=None,
        description="Variables that cannot be changed",
        max_length=20
    )
    max_changes: int = Field(
        default=1,
        description="Maximum number of variables to change simultaneously",
        ge=1,
        le=5
    )
    minimize: str = Field(
        default="change_magnitude",
        description="Optimization criterion: change_magnitude, cost, or feasibility",
    )
    variable_bounds: Optional[Dict[str, tuple[float, float]]] = Field(
        default=None,
        description="Optional bounds for each variable (min, max)",
    )

    @field_validator("variable_bounds")
    @classmethod
    def validate_bounds_size(cls, v):
        """Validate variable bounds dict size."""
        if v is not None:
            from src.utils.security_validators import validate_dict_size
            validate_dict_size(v, "variable_bounds")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "feasible": ["Price", "Marketing"],
                "fixed": ["Quality"],
                "max_changes": 2,
                "minimize": "cost",
                "variable_bounds": {
                    "Price": (30, 100),
                    "Marketing": (10000, 100000),
                },
            }
        }
    }


class ContrastiveExplanationRequest(BaseModel):
    """Request model for contrastive explanation endpoint."""

    model: StructuralModel = Field(..., description="Structural causal model")
    current_state: Dict[str, float] = Field(
        ...,
        description="Current values for all variables",
    )
    observed_outcome: Dict[str, float] = Field(
        ...,
        description="Current observed outcome values",
    )
    target_outcome: Dict[str, tuple[float, float]] = Field(
        ...,
        description="Target outcome ranges (variable → (min, max))",
    )
    constraints: InterventionConstraints = Field(
        ...,
        description="Constraints on which variables can change and how",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic results",
    )

    @field_validator("current_state", "observed_outcome", "target_outcome")
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
                    "variables": ["Price", "Quality", "Marketing", "Brand", "Revenue"],
                    "equations": {
                        "Brand": "50 + 0.3 * Quality - 0.1 * Price",
                        "Revenue": "10000 + 800 * Price + 200 * Quality + 0.5 * Marketing + 300 * Brand",
                    },
                    "distributions": {
                        "baseline_noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
                    },
                },
                "current_state": {
                    "Price": 40,
                    "Quality": 7.5,
                    "Marketing": 30000,
                },
                "observed_outcome": {"Revenue": 40000},
                "target_outcome": {"Revenue": (50000, 55000)},
                "constraints": {
                    "feasible": ["Price", "Marketing"],
                    "fixed": ["Quality"],
                    "max_changes": 2,
                    "minimize": "cost",
                },
                "seed": 42,
            }
        }
    }


class ScenarioSpec(BaseModel):
    """Specification for a single counterfactual scenario."""

    id: str = Field(
        ...,
        description="User-defined scenario identifier",
        min_length=1,
        max_length=100
    )
    intervention: Dict[str, float] = Field(
        ...,
        description="Intervention values for this scenario",
    )
    label: Optional[str] = Field(
        default=None,
        description="Optional human-readable label",
        max_length=200
    )

    @field_validator("intervention")
    @classmethod
    def validate_intervention_size(cls, v):
        """Validate intervention dict size."""
        from src.utils.security_validators import validate_dict_size
        validate_dict_size(v, "intervention")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "aggressive_pricing",
                "intervention": {"Price": 50, "Marketing": 50000},
                "label": "Aggressive market expansion strategy",
            }
        }
    }


class BatchCounterfactualRequest(BaseModel):
    """Request model for batch counterfactual analysis."""

    model: StructuralModel = Field(..., description="Structural causal model")
    scenarios: List[ScenarioSpec] = Field(
        ...,
        description="List of scenarios to evaluate",
        min_length=2,
        max_length=20
    )
    outcome: str = Field(
        ...,
        description="Outcome variable to predict",
        min_length=1,
        max_length=100
    )
    analyze_interactions: bool = Field(
        default=True,
        description="Whether to detect variable interactions",
    )
    robustness_radius: float = Field(
        default=0.1,
        description="Perturbation radius for robustness analysis",
        gt=0,
        le=0.5,
    )
    samples: int = Field(
        default=1000,
        description="Monte Carlo samples per scenario",
        ge=100,
        le=10000,
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic results",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": {
                    "variables": ["Price", "Quality", "Revenue"],
                    "equations": {"Revenue": "10000 + 500*Price + 200*Quality"},
                    "distributions": {
                        "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
                    },
                },
                "scenarios": [
                    {"id": "baseline", "intervention": {"Price": 40}, "label": "Current pricing"},
                    {"id": "increase", "intervention": {"Price": 50}, "label": "10% increase"},
                    {"id": "combined", "intervention": {"Price": 50, "Quality": 8.5}, "label": "Price + Quality"},
                ],
                "outcome": "Revenue",
                "analyze_interactions": True,
                "samples": 1000,
                "seed": 42,
            }
        }
    }
