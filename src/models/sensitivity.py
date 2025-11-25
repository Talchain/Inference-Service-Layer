"""
Sensitivity analysis models for quantifying assumption robustness.

Provides continuous sensitivity metrics instead of discrete scores.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .metadata import ResponseMetadata
from .shared import ConfidenceLevel


class AssumptionType(str, Enum):
    """Types of causal assumptions."""

    NO_UNOBSERVED_CONFOUNDING = "no_unobserved_confounding"
    LINEAR_EFFECTS = "linear_effects"
    NO_SELECTION_BIAS = "no_selection_bias"
    CAUSAL_SUFFICIENCY = "causal_sufficiency"
    POSITIVITY = "positivity"
    CONSISTENCY = "consistency"
    MARKOV_PROPERTY = "markov_property"


class ViolationType(str, Enum):
    """Types of assumption violations."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class CausalAssumption(BaseModel):
    """
    A causal assumption that can be tested for sensitivity.

    Examples:
    - No unobserved confounding between treatment and outcome
    - Linear relationship between variables
    - Positivity (all subgroups have both treatment values)
    """

    name: str = Field(
        ...,
        description="Human-readable name of assumption",
        min_length=1,
        max_length=200
    )

    type: AssumptionType = Field(
        ...,
        description="Type of assumption"
    )

    description: str = Field(
        ...,
        description="Detailed explanation of what this assumption means",
        max_length=1000
    )

    violated_by: Optional[str] = Field(
        None,
        description="What would violate this assumption",
        max_length=500
    )

    testable: bool = Field(
        default=False,
        description="Whether this assumption can be empirically tested"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "No Unobserved Confounding",
                "type": "no_unobserved_confounding",
                "description": "There are no unmeasured variables that affect both treatment and outcome",
                "violated_by": "Hidden factors like socioeconomic status affecting both",
                "testable": False
            }
        }
    }


class ViolationScenario(BaseModel):
    """
    A specific violation of an assumption.

    Used to test how sensitive results are to assumption violations.
    """

    assumption_name: str = Field(
        ...,
        description="Name of assumption being violated"
    )

    severity: ViolationType = Field(
        ...,
        description="How severe is this violation"
    )

    magnitude: float = Field(
        ...,
        description="Quantitative measure of violation strength (0-1)",
        ge=0.0,
        le=1.0
    )

    description: str = Field(
        ...,
        description="What this violation represents",
        max_length=500
    )

    parameters: Dict[str, float] = Field(
        default_factory=dict,
        description="Violation-specific parameters"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "assumption_name": "No Unobserved Confounding",
                "severity": "moderate",
                "magnitude": 0.3,
                "description": "Unmeasured confounder with correlation 0.3",
                "parameters": {"confounder_correlation": 0.3}
            }
        }
    }


class SensitivityMetric(BaseModel):
    """
    Quantitative sensitivity metric for one assumption.

    Measures how much outcomes change when assumption is violated.
    """

    assumption: str = Field(
        ...,
        description="Name of assumption tested"
    )

    baseline_outcome: float = Field(
        ...,
        description="Predicted outcome under assumption"
    )

    outcome_range: Tuple[float, float] = Field(
        ...,
        description="Range of outcomes across all violations (min, max)"
    )

    elasticity: float = Field(
        ...,
        description="Percent change in outcome per percent change in assumption violation",
        ge=0.0
    )

    critical: bool = Field(
        ...,
        description="Whether this assumption is critical (high sensitivity)"
    )

    max_deviation_percent: float = Field(
        ...,
        description="Maximum percent deviation from baseline",
        ge=0.0
    )

    robustness_score: float = Field(
        ...,
        description="How robust results are to this assumption (0=fragile, 1=robust)",
        ge=0.0,
        le=1.0
    )

    interpretation: str = Field(
        ...,
        description="Plain English explanation of sensitivity",
        max_length=500
    )

    violation_details: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Detailed results for each violation tested"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "assumption": "no_unobserved_confounding",
                "baseline_outcome": 50000.0,
                "outcome_range": (42000.0, 58000.0),
                "elasticity": 2.3,
                "critical": True,
                "max_deviation_percent": 16.0,
                "robustness_score": 0.34,
                "interpretation": "10% confounder bias → 23% outcome change. CRITICAL assumption.",
                "violation_details": []
            }
        }
    }


class SensitivityReport(BaseModel):
    """
    Complete sensitivity analysis report.

    Shows how sensitive results are to each assumption.
    """

    sensitivities: Dict[str, SensitivityMetric] = Field(
        ...,
        description="Sensitivity metric for each assumption tested"
    )

    most_critical: List[str] = Field(
        ...,
        description="Names of critical assumptions (ordered by sensitivity)"
    )

    least_critical: List[str] = Field(
        default_factory=list,
        description="Names of least critical assumptions"
    )

    overall_robustness_score: float = Field(
        ...,
        description="Aggregate robustness across all assumptions (0=fragile, 1=robust)",
        ge=0.0,
        le=1.0
    )

    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Confidence in sensitivity analysis results"
    )

    summary: str = Field(
        ...,
        description="Plain English summary of sensitivity findings",
        max_length=1000
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for strengthening assumptions",
        max_length=10
    )

    metadata: ResponseMetadata = Field(
        ...,
        description="Response metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "sensitivities": {
                    "no_unobserved_confounding": {
                        "assumption": "no_unobserved_confounding",
                        "baseline_outcome": 50000.0,
                        "outcome_range": (42000.0, 58000.0),
                        "elasticity": 2.3,
                        "critical": True,
                        "max_deviation_percent": 16.0,
                        "robustness_score": 0.34,
                        "interpretation": "10% bias → 23% change",
                        "violation_details": []
                    }
                },
                "most_critical": ["no_unobserved_confounding"],
                "least_critical": ["linear_effects"],
                "overall_robustness_score": 0.67,
                "confidence_level": "high",
                "summary": "Results are moderately robust. Most critical: no unobserved confounding.",
                "recommendations": [
                    "Measure additional potential confounders",
                    "Use instrumental variables if available"
                ],
                "metadata": {
                    "timestamp": "2025-11-23T12:00:00Z",
                    "version": "1.0.0"
                }
            }
        }
    }


class SensitivityRequest(BaseModel):
    """Request for detailed sensitivity analysis."""

    model: Dict = Field(
        ...,
        description="Structural causal model specification"
    )

    intervention: Dict[str, float] = Field(
        ...,
        description="Intervention values to analyze"
    )

    assumptions: List[str] = Field(
        ...,
        description="List of assumption types to test",
        min_length=1,
        max_length=10
    )

    violation_levels: List[float] = Field(
        default=[0.1, 0.2, 0.3, 0.5],
        description="Violation magnitudes to test (0-1)",
        min_length=1,
        max_length=20
    )

    n_samples: int = Field(
        default=100,
        description="Number of samples for stochastic violations",
        ge=10,
        le=1000
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": {
                    "type": "linear",
                    "equations": {
                        "Revenue": "100 * Price + 5000 * Quality - 200"
                    }
                },
                "intervention": {"Price": 45.0},
                "assumptions": ["no_unobserved_confounding", "linear_effects"],
                "violation_levels": [0.1, 0.2, 0.3],
                "n_samples": 100
            }
        }
    }
