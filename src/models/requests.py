"""
Request Pydantic models for API endpoints.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .shared import DAGStructure, Distribution, DistributionType, GraphV1, StructuralModel


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


class DataSummary(BaseModel):
    """Summary of available data in a domain."""

    n_samples: int = Field(..., description="Number of samples available", ge=0)
    available_variables: List[str] = Field(
        ...,
        description="Variables measured in this domain",
        max_length=50
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Additional notes about data availability",
        max_length=10
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "n_samples": 1000,
                "available_variables": ["Price", "Revenue", "CustomerAge"],
                "notes": ["Covariates only", "No outcome data"],
            }
        }
    }


class DomainSpec(BaseModel):
    """Specification for a single domain (e.g., market, region)."""

    name: str = Field(
        ...,
        description="Domain name",
        min_length=1,
        max_length=100
    )
    dag: DAGStructure = Field(..., description="Causal graph structure for this domain")
    data_summary: Optional[DataSummary] = Field(
        default=None,
        description="Summary of available data (optional)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "UK Market",
                "dag": {
                    "nodes": ["Price", "Brand", "Revenue"],
                    "edges": [["Price", "Revenue"], ["Brand", "Price"], ["Brand", "Revenue"]],
                },
                "data_summary": {
                    "n_samples": 5000,
                    "available_variables": ["Price", "Brand", "Revenue"],
                    "notes": ["Complete data available"],
                },
            }
        }
    }


class TransportabilityRequest(BaseModel):
    """Request model for transportability analysis."""

    source_domain: DomainSpec = Field(..., description="Source domain (where effect is identified)")
    target_domain: DomainSpec = Field(..., description="Target domain (where effect is transported)")
    treatment: str = Field(
        ...,
        description="Treatment variable",
        min_length=1,
        max_length=100
    )
    outcome: str = Field(
        ...,
        description="Outcome variable",
        min_length=1,
        max_length=100
    )
    selection_variables: Optional[List[str]] = Field(
        default=None,
        description="Variables affected by domain selection (if known)",
        max_length=20
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_domain": {
                    "name": "UK Market",
                    "dag": {
                        "nodes": ["Price", "Brand", "Revenue"],
                        "edges": [["Price", "Revenue"], ["Brand", "Price"], ["Brand", "Revenue"]],
                    },
                },
                "target_domain": {
                    "name": "EU Market",
                    "dag": {
                        "nodes": ["Price", "Brand", "Revenue"],
                        "edges": [["Price", "Revenue"], ["Brand", "Price"], ["Brand", "Revenue"]],
                    },
                },
                "treatment": "Price",
                "outcome": "Revenue",
                "selection_variables": ["Brand"],
            }
        }
    }


class ObservationPoint(BaseModel):
    """
    Single observation for calibration in conformal prediction.

    Contains both inputs (intervention/context) and observed outcomes
    used to calibrate prediction intervals.
    """

    inputs: Dict[str, float] = Field(
        ...,
        description="Input values (interventions or context variables)",
    )
    outcome: Dict[str, float] = Field(
        ...,
        description="Observed outcome values for this observation",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "inputs": {"Price": 45, "Quality": 8},
                "outcome": {"Revenue": 52000},
            }
        }
    }


class ConformalCounterfactualRequest(BaseModel):
    """
    Request model for conformal prediction endpoint.

    Provides finite-sample valid prediction intervals with
    guaranteed coverage using conformal prediction methods.
    """

    model: StructuralModel = Field(
        ...,
        description="Structural causal model",
    )
    intervention: Dict[str, float] = Field(
        ...,
        description="Intervention to apply",
    )
    calibration_data: Optional[List[ObservationPoint]] = Field(
        default=None,
        description="Historical observations for calibration (optional if model has built-in calibration)",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Target coverage level (1-alpha), e.g., 0.95 for 95% coverage",
        ge=0.5,
        le=0.999,
    )
    method: str = Field(
        default="split",
        description="Conformal method: 'split', 'cv+', or 'jackknife+'",
    )
    samples: int = Field(
        default=1000,
        description="Number of samples for Monte Carlo comparison",
        ge=100,
        le=10000,
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate conformal method."""
        allowed = ["split", "cv+", "jackknife+"]
        if v not in allowed:
            raise ValueError(f"Method must be one of {allowed}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": {
                    "variables": ["Price", "Revenue"],
                    "equations": {"Revenue": "10000 + 500*Price"},
                    "distributions": {
                        "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
                    },
                },
                "intervention": {"Price": 50},
                "calibration_data": [
                    {"inputs": {"Price": 40}, "outcome": {"Revenue": 30000}},
                    {"inputs": {"Price": 45}, "outcome": {"Revenue": 32500}},
                    {"inputs": {"Price": 50}, "outcome": {"Revenue": 35000}},
                ],
                "confidence_level": 0.95,
                "method": "split",
                "samples": 1000,
                "seed": 42,
            }
        }
    }


class ValidationStrategyRequest(BaseModel):
    """
    Request model for enhanced Y₀ validation strategies.

    Provides complete adjustment strategies for non-identifiable DAGs.
    """

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

    model_config = {
        "json_schema_extra": {
            "example": {
                "dag": {
                    "nodes": ["Price", "Competitors", "Revenue"],
                    "edges": [
                        ["Price", "Revenue"],
                        ["Competitors", "Revenue"],
                    ],
                },
                "treatment": "Price",
                "outcome": "Revenue",
            }
        }
    }


class DiscoveryFromDataRequest(BaseModel):
    """
    Request model for causal discovery from observational data.

    Automatically suggests DAG structures from data.
    """

    data: List[List[float]] = Field(
        ...,
        description="Data matrix (rows = samples, columns = variables)",
        min_length=10,
        max_length=10000
    )
    variable_names: List[str] = Field(
        ...,
        description="Names of variables (must match data columns)",
        min_length=2,
        max_length=50
    )
    prior_knowledge: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Prior knowledge about structure (required_edges, forbidden_edges)",
    )
    threshold: float = Field(
        default=0.3,
        description="Correlation threshold for edge detection",
        ge=0.0,
        le=1.0,
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [
                    [40, 7.5, 30000],
                    [45, 8.0, 32500],
                    [50, 8.5, 35000],
                ],
                "variable_names": ["Price", "Quality", "Revenue"],
                "prior_knowledge": {
                    "required_edges": [["Price", "Revenue"]],
                    "forbidden_edges": [["Revenue", "Price"]],
                },
                "threshold": 0.3,
                "seed": 42,
            }
        }
    }


class DiscoveryFromKnowledgeRequest(BaseModel):
    """
    Request model for knowledge-guided causal discovery.

    Uses domain knowledge to suggest DAG structures.
    """

    domain_description: str = Field(
        ...,
        description="Natural language description of the domain",
        min_length=10,
        max_length=2000
    )
    variable_names: List[str] = Field(
        ...,
        description="Names of variables in the domain",
        min_length=2,
        max_length=50
    )
    prior_knowledge: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Prior knowledge about structure",
    )
    top_k: int = Field(
        default=3,
        description="Number of candidate DAGs to return",
        ge=1,
        le=10,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "domain_description": "An e-commerce pricing model where price affects revenue, quality affects both price and revenue",
                "variable_names": ["Price", "Quality", "Revenue"],
                "prior_knowledge": {
                    "required_edges": [["Price", "Revenue"]],
                },
                "top_k": 3,
            }
        }
    }


class BeliefDistribution(BaseModel):
    """Parameter belief distribution for sequential optimization."""

    parameter_name: str = Field(..., description="Parameter name")
    distribution_type: str = Field(
        ...,
        description="Distribution type: normal, uniform, beta"
    )
    parameters: Dict[str, float] = Field(
        ...,
        description="Distribution parameters (mean/std for normal, low/high for uniform, etc.)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter_name": "effect_price",
                "distribution_type": "normal",
                "parameters": {"mean": 500, "std": 50},
            }
        }
    }


class OptimizationObjectiveSpec(BaseModel):
    """Optimization objective specification."""

    target_variable: str = Field(
        ...,
        description="Variable to optimize",
        min_length=1,
        max_length=100
    )
    goal: str = Field(
        ...,
        description="Optimization goal: maximize, minimize, or target"
    )
    target_value: Optional[float] = Field(
        default=None,
        description="Target value (required if goal=target)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "target_variable": "Revenue",
                "goal": "maximize",
            }
        }
    }


class ExperimentConstraintsSpec(BaseModel):
    """Constraints on experiments."""

    budget: float = Field(..., description="Total remaining budget", gt=0)
    time_horizon: int = Field(
        ...,
        description="Number of experiments remaining",
        ge=1,
        le=100
    )
    feasible_interventions: Dict[str, tuple[float, float]] = Field(
        ...,
        description="Feasible ranges for each intervention variable (min, max)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "budget": 100000,
                "time_horizon": 5,
                "feasible_interventions": {
                    "Price": (30, 100),
                    "Marketing": (10000, 100000),
                },
            }
        }
    }


class ExperimentHistoryPoint(BaseModel):
    """A single historical experiment for sequential optimization."""

    intervention: Dict[str, float] = Field(
        ...,
        description="Intervention values",
    )
    outcome: Dict[str, float] = Field(
        ...,
        description="Observed outcome values",
    )
    cost: Optional[float] = Field(
        default=None,
        description="Actual cost of the experiment",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "intervention": {"Price": 45},
                "outcome": {"Revenue": 32500},
                "cost": 5000,
            }
        }
    }


class ExperimentRecommendationRequest(BaseModel):
    """
    Request model for sequential experiment recommendation.

    Uses Thompson sampling to recommend next experiment.
    """

    beliefs: List[BeliefDistribution] = Field(
        ...,
        description="Current beliefs about model parameters",
        min_length=1,
        max_length=30
    )
    objective: OptimizationObjectiveSpec = Field(
        ...,
        description="Optimization objective",
    )
    constraints: ExperimentConstraintsSpec = Field(
        ...,
        description="Experiment constraints",
    )
    history: Optional[List[ExperimentHistoryPoint]] = Field(
        default=None,
        description="Previous experiment results",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "beliefs": [
                    {
                        "parameter_name": "effect_price",
                        "distribution_type": "normal",
                        "parameters": {"mean": 500, "std": 50},
                    }
                ],
                "objective": {
                    "target_variable": "Revenue",
                    "goal": "maximize",
                },
                "constraints": {
                    "budget": 100000,
                    "time_horizon": 5,
                    "feasible_interventions": {
                        "Price": (30, 100),
                    },
                },
                "history": [
                    {
                        "intervention": {"Price": 45},
                        "outcome": {"Revenue": 32500},
                        "cost": 5000,
                    }
                ],
                "seed": 42,
            }
        }
    }


# ============================================================================
# CEE Enhancement Endpoints (Phase 0)
# ============================================================================


class SensitivityDetailedRequest(BaseModel):
    """Request model for detailed sensitivity analysis endpoint."""

    graph: GraphV1 = Field(..., description="Decision graph structure")
    timeout: Optional[int] = Field(
        default=12000,
        description="Request timeout in milliseconds",
        ge=1000,
        le=30000
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "graph": {
                    "nodes": [
                        {
                            "id": "n_market_size",
                            "kind": "outcome",
                            "label": "Market Size",
                            "belief": 0.75
                        },
                        {
                            "id": "n_roi",
                            "kind": "outcome",
                            "label": "ROI"
                        }
                    ],
                    "edges": [
                        {
                            "from": "n_market_size",
                            "to": "n_roi",
                            "weight": 2.5
                        }
                    ]
                },
                "timeout": 12000
            }
        }
    }


class ContrastiveRequest(BaseModel):
    """Request model for contrastive explanation endpoint."""

    graph: GraphV1 = Field(..., description="Decision graph structure")
    target_outcome: str = Field(
        ...,
        description="Desired outcome or node ID to analyze",
        max_length=100
    )
    timeout: Optional[int] = Field(
        default=12000,
        description="Request timeout in milliseconds",
        ge=1000,
        le=30000
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "graph": {
                    "nodes": [
                        {
                            "id": "n_launch_product",
                            "kind": "decision",
                            "label": "Launch Product"
                        },
                        {
                            "id": "n_market_success",
                            "kind": "outcome",
                            "label": "Market Success"
                        }
                    ],
                    "edges": [
                        {
                            "from": "n_launch_product",
                            "to": "n_market_success",
                            "weight": 2.0
                        }
                    ]
                },
                "target_outcome": "n_market_success",
                "timeout": 12000
            }
        }
    }


class ConformalRequest(BaseModel):
    """Request model for conformal prediction endpoint."""

    graph: GraphV1 = Field(..., description="Decision graph structure")
    variable: str = Field(
        ...,
        description="Variable or outcome to predict",
        max_length=100
    )
    timeout: Optional[int] = Field(
        default=12000,
        description="Request timeout in milliseconds",
        ge=1000,
        le=30000
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "graph": {
                    "nodes": [
                        {
                            "id": "n_revenue",
                            "kind": "outcome",
                            "label": "Revenue"
                        }
                    ],
                    "edges": []
                },
                "variable": "n_revenue",
                "timeout": 12000
            }
        }
    }


class ValidationStrategiesRequest(BaseModel):
    """Request model for validation strategies endpoint."""

    graph: GraphV1 = Field(..., description="Decision graph structure")
    timeout: Optional[int] = Field(
        default=12000,
        description="Request timeout in milliseconds",
        ge=1000,
        le=30000
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "graph": {
                    "nodes": [
                        {
                            "id": "n_decision",
                            "kind": "decision",
                            "label": "Strategic Decision"
                        },
                        {
                            "id": "n_outcome",
                            "kind": "outcome",
                            "label": "Business Outcome"
                        }
                    ],
                    "edges": [
                        {
                            "from": "n_decision",
                            "to": "n_outcome",
                            "weight": 1.5
                        }
                    ]
                },
                "timeout": 12000
            }
        }
    }


class ParameterRecommendationRequest(BaseModel):
    """Request model for parameter recommendation endpoint."""

    graph: GraphV1 = Field(..., description="Decision graph structure")
    timeout: Optional[int] = Field(
        default=12000,
        description="Request timeout in milliseconds",
        ge=1000,
        le=30000
    )
    current_parameters: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional current parameter values to compare against"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "graph": {
                    "nodes": [
                        {
                            "id": "n_decision",
                            "kind": "decision",
                            "label": "Launch Product",
                            "belief": 0.8
                        },
                        {
                            "id": "n_outcome",
                            "kind": "outcome",
                            "label": "Market Success",
                            "belief": 0.6
                        }
                    ],
                    "edges": [
                        {
                            "from": "n_decision",
                            "to": "n_outcome",
                            "weight": 2.0
                        }
                    ]
                },
                "timeout": 12000,
                "current_parameters": {
                    "n_decision_to_n_outcome_weight": 2.0,
                    "n_decision_belief": 0.8
                }
            }
        }
    }


# ============================================================================
# Dominance Detection Endpoint - Request Models
# ============================================================================


class DominanceOption(BaseModel):
    """Single option with scores across multiple criteria."""

    option_id: str = Field(
        ...,
        description="Unique option identifier",
        min_length=1,
        max_length=200
    )
    option_label: str = Field(
        ...,
        description="Human-readable option label",
        min_length=1,
        max_length=500
    )
    scores: Dict[str, float] = Field(
        ...,
        description="Normalized scores (0-1) by criterion_id"
    )

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v):
        """Validate scores are normalized 0-1 and finite."""
        if not v:
            raise ValueError("scores cannot be empty")

        for criterion_id, score in v.items():
            if not isinstance(score, (int, float)):
                raise ValueError(f"Score for {criterion_id} must be numeric")
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Score for {criterion_id} must be in range [0, 1], got {score}")
            if not float('-inf') < score < float('inf'):
                raise ValueError(f"Score for {criterion_id} must be finite, got {score}")

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "opt_launch_q1",
                "option_label": "Launch in Q1 2025",
                "scores": {
                    "revenue": 0.85,
                    "risk": 0.60,
                    "timeline": 0.90
                }
            }
        }
    }


class DominanceRequest(BaseModel):
    """Request model for dominance detection endpoint."""

    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for tracing (auto-generated if not provided)"
    )
    options: List[DominanceOption] = Field(
        ...,
        description="Options to analyze for dominance relationships",
        min_length=2,
        max_length=100
    )
    criteria: List[str] = Field(
        ...,
        description="Criterion IDs to consider (must match keys in option scores)",
        min_length=1,
        max_length=10
    )

    @field_validator("options")
    @classmethod
    def validate_option_consistency(cls, v, info):
        """Validate all options have scores for all criteria."""
        if len(v) < 2:
            raise ValueError("At least 2 options required for dominance analysis")

        # Get criteria from context if available
        criteria = info.data.get("criteria", [])
        if criteria:
            for option in v:
                missing = set(criteria) - set(option.scores.keys())
                if missing:
                    raise ValueError(
                        f"Option {option.option_id} missing scores for criteria: {missing}"
                    )
                extra = set(option.scores.keys()) - set(criteria)
                if extra:
                    raise ValueError(
                        f"Option {option.option_id} has scores for unexpected criteria: {extra}"
                    )

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_abc123",
                "options": [
                    {
                        "option_id": "opt_a",
                        "option_label": "Option A: Aggressive Growth",
                        "scores": {"revenue": 0.90, "risk": 0.40, "timeline": 0.70}
                    },
                    {
                        "option_id": "opt_b",
                        "option_label": "Option B: Conservative Growth",
                        "scores": {"revenue": 0.60, "risk": 0.80, "timeline": 0.90}
                    },
                    {
                        "option_id": "opt_c",
                        "option_label": "Option C: Balanced Approach",
                        "scores": {"revenue": 0.75, "risk": 0.75, "timeline": 0.80}
                    }
                ],
                "criteria": ["revenue", "risk", "timeline"]
            }
        }
    }


# ============================================================================
# Pareto Frontier Endpoint - Request Models
# ============================================================================


class ParetoRequest(BaseModel):
    """Request model for Pareto frontier endpoint."""

    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for tracing (auto-generated if not provided)"
    )
    options: List[DominanceOption] = Field(
        ...,
        description="Options to analyze for Pareto frontier",
        min_length=2,
        max_length=100
    )
    criteria: List[str] = Field(
        ...,
        description="Criterion IDs to consider (must match keys in option scores)",
        min_length=1,
        max_length=10
    )
    max_frontier_size: Optional[int] = Field(
        default=20,
        description="Maximum number of frontier options to return (for large frontiers)",
        ge=1,
        le=100
    )

    @field_validator("options")
    @classmethod
    def validate_option_consistency(cls, v, info):
        """Validate all options have scores for all criteria."""
        if len(v) < 2:
            raise ValueError("At least 2 options required for Pareto analysis")

        # Get criteria from context if available
        criteria = info.data.get("criteria", [])
        if criteria:
            for option in v:
                missing = set(criteria) - set(option.scores.keys())
                if missing:
                    raise ValueError(
                        f"Option {option.option_id} missing scores for criteria: {missing}"
                    )
                extra = set(option.scores.keys()) - set(criteria)
                if extra:
                    raise ValueError(
                        f"Option {option.option_id} has scores for unexpected criteria: {extra}"
                    )

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_abc123",
                "options": [
                    {
                        "option_id": "opt_a",
                        "option_label": "Option A: Aggressive Growth",
                        "scores": {"revenue": 0.90, "risk": 0.40, "timeline": 0.70}
                    },
                    {
                        "option_id": "opt_b",
                        "option_label": "Option B: Conservative Growth",
                        "scores": {"revenue": 0.60, "risk": 0.80, "timeline": 0.90}
                    },
                    {
                        "option_id": "opt_c",
                        "option_label": "Option C: Balanced Approach",
                        "scores": {"revenue": 0.75, "risk": 0.75, "timeline": 0.80}
                    }
                ],
                "criteria": ["revenue", "risk", "timeline"],
                "max_frontier_size": 20
            }
        }
    }


# ============================================================================
# Multi-Criteria Aggregation Endpoint - Request Models
# ============================================================================


class OptionScore(BaseModel):
    """Scores for a single option across percentiles."""

    option_id: str = Field(..., description="Option identifier")
    option_label: str = Field(..., description="Human-readable option label")
    p10: float = Field(..., description="Pessimistic (10th percentile) score")
    p50: float = Field(..., description="Expected (50th percentile) score")
    p90: float = Field(..., description="Optimistic (90th percentile) score")

    @field_validator("p10", "p50", "p90")
    @classmethod
    def validate_scores(cls, v):
        """Validate scores are 0-1 and finite."""
        if not isinstance(v, (int, float)):
            raise ValueError("Score must be numeric")
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Score must be in range [0, 1], got {v}")
        if not float('-inf') < v < float('inf'):
            raise ValueError(f"Score must be finite, got {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "opt_launch_q1",
                "option_label": "Launch in Q1 2025",
                "p10": 0.65,
                "p50": 0.85,
                "p90": 0.95
            }
        }
    }


class CriterionResult(BaseModel):
    """Results for a single criterion with scores for all options."""

    criterion_id: str = Field(
        ...,
        description="Criterion identifier",
        min_length=1,
        max_length=100
    )
    criterion_name: str = Field(
        ...,
        description="Human-readable criterion name",
        min_length=1,
        max_length=200
    )
    options: List[OptionScore] = Field(
        ...,
        description="Scores for each option on this criterion",
        min_length=2,
        max_length=100
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "criterion_id": "revenue",
                "criterion_name": "Expected Revenue",
                "options": [
                    {
                        "option_id": "opt_a",
                        "option_label": "Option A",
                        "p10": 0.65,
                        "p50": 0.85,
                        "p90": 0.95
                    },
                    {
                        "option_id": "opt_b",
                        "option_label": "Option B",
                        "p10": 0.45,
                        "p50": 0.60,
                        "p90": 0.75
                    }
                ]
            }
        }
    }


class MultiCriteriaRequest(BaseModel):
    """Request model for multi-criteria aggregation endpoint."""

    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for tracing"
    )
    criteria: List[CriterionResult] = Field(
        ...,
        description="Results for each criterion with option scores",
        min_length=1,
        max_length=10
    )
    aggregation_method: str = Field(
        ...,
        description="Aggregation method to use",
        pattern="^(weighted_sum|weighted_product|lexicographic)$"
    )
    weights: Dict[str, float] = Field(
        ...,
        description="Weights by criterion_id (should sum to 1.0, will auto-normalize if not)"
    )
    percentile: Optional[str] = Field(
        default="p50",
        description="Which percentile to use for aggregation",
        pattern="^(p10|p50|p90)$"
    )
    trade_off_threshold: Optional[float] = Field(
        default=0.05,
        description="Minimum score difference to report as trade-off",
        ge=0.0,
        le=1.0
    )
    timeout_ms: Optional[int] = Field(
        default=5000,
        description="Request timeout in milliseconds",
        ge=100,
        le=30000
    )

    @field_validator("weights")
    @classmethod
    def validate_weights_positive(cls, v):
        """Validate that weights are positive."""
        if not v:
            raise ValueError("weights cannot be empty")

        for criterion_id, weight in v.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for {criterion_id} must be numeric")
            if weight < 0:
                raise ValueError(f"Weight for {criterion_id} must be non-negative, got {weight}")
            if not float('-inf') < weight < float('inf'):
                raise ValueError(f"Weight for {criterion_id} must be finite, got {weight}")

        total = sum(v.values())
        if total == 0:
            raise ValueError("Cannot normalize: all weights are zero")

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_abc123",
                "criteria": [
                    {
                        "criterion_id": "revenue",
                        "criterion_name": "Expected Revenue",
                        "options": [
                            {"option_id": "opt_a", "option_label": "Option A", "p10": 0.65, "p50": 0.85, "p90": 0.95},
                            {"option_id": "opt_b", "option_label": "Option B", "p10": 0.45, "p50": 0.60, "p90": 0.75}
                        ]
                    },
                    {
                        "criterion_id": "risk",
                        "criterion_name": "Risk Level",
                        "options": [
                            {"option_id": "opt_a", "option_label": "Option A", "p10": 0.30, "p50": 0.40, "p90": 0.55},
                            {"option_id": "opt_b", "option_label": "Option B", "p10": 0.70, "p50": 0.80, "p90": 0.90}
                        ]
                    }
                ],
                "aggregation_method": "weighted_sum",
                "weights": {"revenue": 0.6, "risk": 0.4},
                "percentile": "p50",
                "trade_off_threshold": 0.05
            }
        }
    }


# ============================================================================
# Risk Adjustment Endpoint - Request Models
# ============================================================================


class RiskOption(BaseModel):
    """Option with uncertainty for risk adjustment."""

    option_id: str = Field(
        ...,
        description="Option identifier",
        min_length=1,
        max_length=200
    )
    option_label: str = Field(
        ...,
        description="Human-readable option label",
        min_length=1,
        max_length=500
    )
    # Support both mean/std_dev and percentile representations
    mean: Optional[float] = Field(
        default=None,
        description="Mean score (for mean-variance representation)"
    )
    std_dev: Optional[float] = Field(
        default=None,
        description="Standard deviation (for mean-variance representation)",
        ge=0.0
    )
    p10: Optional[float] = Field(
        default=None,
        description="10th percentile score (for percentile representation)"
    )
    p50: Optional[float] = Field(
        default=None,
        description="50th percentile score (for percentile representation)"
    )
    p90: Optional[float] = Field(
        default=None,
        description="90th percentile score (for percentile representation)"
    )

    @field_validator("mean", "std_dev", "p10", "p50", "p90")
    @classmethod
    def validate_scores_finite(cls, v):
        """Validate scores are finite if provided."""
        if v is not None:
            if not isinstance(v, (int, float)):
                raise ValueError("Score must be numeric")
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Score must be in range [0, 1], got {v}")
            if not float('-inf') < v < float('inf'):
                raise ValueError(f"Score must be finite, got {v}")
        return v

    def model_post_init(self, __context):
        """Validate that either mean/std_dev or percentiles are provided."""
        has_mean_std = self.mean is not None and self.std_dev is not None
        has_percentiles = (
            self.p10 is not None and self.p50 is not None and self.p90 is not None
        )

        if not has_mean_std and not has_percentiles:
            raise ValueError(
                "Must provide either (mean, std_dev) or (p10, p50, p90)"
            )

        if has_mean_std and has_percentiles:
            raise ValueError(
                "Cannot provide both (mean, std_dev) and (p10, p50, p90) - choose one representation"
            )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "option_id": "opt_a",
                    "option_label": "Option A: High Risk",
                    "mean": 0.75,
                    "std_dev": 0.15
                },
                {
                    "option_id": "opt_b",
                    "option_label": "Option B: Low Risk",
                    "p10": 0.50,
                    "p50": 0.55,
                    "p90": 0.60
                }
            ]
        }
    }


class RiskAdjustmentRequest(BaseModel):
    """Request model for risk adjustment endpoint."""

    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for tracing"
    )
    options: List[RiskOption] = Field(
        ...,
        description="Options with uncertainty to adjust for risk",
        min_length=2,
        max_length=100
    )
    risk_coefficient: float = Field(
        ...,
        description="Risk coefficient from CEE risk profile (>0 for risk aversion)",
        ge=0.0,
        le=10.0
    )
    risk_type: str = Field(
        ...,
        description="Risk attitude type",
        pattern="^(risk_averse|risk_neutral|risk_seeking)$"
    )

    @field_validator("risk_type")
    @classmethod
    def validate_risk_type_coefficient(cls, v, info):
        """Validate risk_type matches coefficient."""
        risk_coefficient = info.data.get("risk_coefficient")
        if risk_coefficient is not None:
            if v == "risk_neutral" and risk_coefficient != 0.0:
                raise ValueError(
                    "risk_neutral requires risk_coefficient=0.0"
                )
            if v in ["risk_averse", "risk_seeking"] and risk_coefficient == 0.0:
                raise ValueError(
                    f"{v} requires risk_coefficient > 0.0"
                )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_risk_abc123",
                "options": [
                    {
                        "option_id": "opt_aggressive",
                        "option_label": "Aggressive Growth Strategy",
                        "mean": 0.80,
                        "std_dev": 0.20
                    },
                    {
                        "option_id": "opt_conservative",
                        "option_label": "Conservative Growth Strategy",
                        "mean": 0.60,
                        "std_dev": 0.05
                    },
                    {
                        "option_id": "opt_balanced",
                        "option_label": "Balanced Approach",
                        "p10": 0.55,
                        "p50": 0.70,
                        "p90": 0.85
                    }
                ],
                "risk_coefficient": 2.0,
                "risk_type": "risk_averse"
            }
        }
    }


# ============================================================================
# Threshold Identification Endpoint - Request Models
# ============================================================================


class ParameterSweep(BaseModel):
    """Parameter sweep with scores at different parameter values."""

    parameter_id: str = Field(
        ...,
        description="Parameter identifier",
        min_length=1,
        max_length=200
    )
    parameter_label: str = Field(
        ...,
        description="Human-readable parameter label",
        min_length=1,
        max_length=500
    )
    values: List[float] = Field(
        ...,
        description="Parameter values tested (in sweep order)",
        min_length=2,
        max_length=1000
    )
    scores_by_value: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Scores for each option at each parameter value. "
                    "Format: {parameter_value: {option_id: score}}"
    )

    @field_validator("values")
    @classmethod
    def validate_values_unique(cls, v):
        """Validate that parameter values are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Parameter values must be unique")
        return v

    @field_validator("scores_by_value")
    @classmethod
    def validate_scores_by_value(cls, v, info):
        """Validate that scores_by_value has entries for all values."""
        values = info.data.get("values", [])
        if values:
            # Check that all values have score entries
            values_str = [str(val) for val in values]
            for val_str in values_str:
                if val_str not in v:
                    raise ValueError(
                        f"scores_by_value missing entry for parameter value {val_str}"
                    )

            # Check that all score dictionaries have same option_ids
            if v:
                first_options = set(next(iter(v.values())).keys())
                for val_str, scores in v.items():
                    if set(scores.keys()) != first_options:
                        raise ValueError(
                            f"All parameter values must have scores for same options. "
                            f"Value {val_str} has different options."
                        )

                # Validate score ranges
                for val_str, scores in v.items():
                    for option_id, score in scores.items():
                        if not isinstance(score, (int, float)):
                            raise ValueError(
                                f"Score for {option_id} at {val_str} must be numeric"
                            )
                        if not (0.0 <= score <= 1.0):
                            raise ValueError(
                                f"Score for {option_id} at {val_str} must be in [0, 1], got {score}"
                            )

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter_id": "price",
                "parameter_label": "Product Price",
                "values": [30.0, 40.0, 50.0, 60.0, 70.0],
                "scores_by_value": {
                    "30.0": {"opt_a": 0.90, "opt_b": 0.60, "opt_c": 0.75},
                    "40.0": {"opt_a": 0.85, "opt_b": 0.70, "opt_c": 0.75},
                    "50.0": {"opt_a": 0.75, "opt_b": 0.80, "opt_c": 0.70},
                    "60.0": {"opt_a": 0.65, "opt_b": 0.85, "opt_c": 0.65},
                    "70.0": {"opt_a": 0.50, "opt_b": 0.90, "opt_c": 0.60}
                }
            }
        }
    }


class ThresholdIdentificationRequest(BaseModel):
    """Request model for threshold identification endpoint."""

    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for tracing"
    )
    parameter_sweeps: List[ParameterSweep] = Field(
        ...,
        description="Parameter sweeps with scores at different values",
        min_length=1,
        max_length=20
    )
    baseline_ranking: Optional[List[str]] = Field(
        default=None,
        description="Optional baseline ranking (option_ids in rank order). "
                    "If not provided, uses ranking at first parameter value."
    )
    confidence_threshold: Optional[float] = Field(
        default=0.1,
        description="Minimum score difference to consider ranking change meaningful",
        ge=0.0,
        le=1.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_threshold_abc123",
                "parameter_sweeps": [
                    {
                        "parameter_id": "price",
                        "parameter_label": "Product Price",
                        "values": [30.0, 40.0, 50.0, 60.0, 70.0],
                        "scores_by_value": {
                            "30.0": {"opt_a": 0.90, "opt_b": 0.60, "opt_c": 0.75},
                            "40.0": {"opt_a": 0.85, "opt_b": 0.70, "opt_c": 0.75},
                            "50.0": {"opt_a": 0.75, "opt_b": 0.80, "opt_c": 0.70},
                            "60.0": {"opt_a": 0.65, "opt_b": 0.85, "opt_c": 0.65},
                            "70.0": {"opt_a": 0.50, "opt_b": 0.90, "opt_c": 0.60}
                        }
                    }
                ],
                "baseline_ranking": ["opt_a", "opt_c", "opt_b"],
                "confidence_threshold": 0.1
            }
        }
    }
