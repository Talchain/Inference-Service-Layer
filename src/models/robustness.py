"""
Robustness analysis models for FACET.

Represents intervention regions, outcome guarantees, and robustness metrics
for region-based counterfactual verification.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .metadata import ResponseMetadata
from .shared import ConfidenceLevel


class InterventionRegion(BaseModel):
    """
    Multi-dimensional region of intervention values.

    Represents set of interventions that achieve target outcome.
    Each region is defined as a hyper-rectangle in intervention space.
    """

    variable_ranges: Dict[str, Tuple[float, float]] = Field(
        ...,
        description="Variable name → (min, max) range",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "variable_ranges": {
                    "price": (52.0, 58.0),
                    "quality_score": (7.5, 8.5),
                }
            }
        }
    }

    def contains(self, intervention: Dict[str, float]) -> bool:
        """
        Check if intervention point is within region.

        Args:
            intervention: Dict mapping variable names to values

        Returns:
            True if intervention is within region bounds
        """
        for var, (min_val, max_val) in self.variable_ranges.items():
            value = intervention.get(var)
            if value is None or not (min_val <= value <= max_val):
                return False
        return True

    def volume(self) -> float:
        """
        Compute region volume (normalized).

        Returns:
            Product of range sizes normalized to [0,1] scale
        """
        vol = 1.0
        for var, (min_val, max_val) in self.variable_ranges.items():
            # Normalize by assuming variable range is [0, 100]
            # (Simplified - production would use actual bounds from context)
            range_size = (max_val - min_val) / 100.0
            vol *= range_size
        return vol

    def sample_random(self, n: int = 100, seed: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Sample random interventions from region.

        Args:
            n: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            List of intervention dicts sampled from region
        """
        if seed is not None:
            np.random.seed(seed)

        samples = []
        for _ in range(n):
            sample = {}
            for var, (min_val, max_val) in self.variable_ranges.items():
                sample[var] = float(np.random.uniform(min_val, max_val))
            samples.append(sample)
        return samples

    def center_point(self) -> Dict[str, float]:
        """
        Get center of region.

        Returns:
            Dict mapping each variable to its midpoint value
        """
        return {
            var: (min_val + max_val) / 2
            for var, (min_val, max_val) in self.variable_ranges.items()
        }


class OutcomeGuarantee(BaseModel):
    """
    Guaranteed outcome range across intervention region.

    With high confidence, all interventions in region produce
    outcomes within this range.
    """

    outcome_variable: str = Field(..., description="Outcome variable name")
    minimum: float = Field(..., description="Guaranteed minimum outcome")
    maximum: float = Field(..., description="Guaranteed maximum outcome")
    confidence: float = Field(..., description="Confidence level (0-1)", ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "outcome_variable": "revenue",
                "minimum": 95000.0,
                "maximum": 105000.0,
                "confidence": 0.95,
            }
        }
    }

    def satisfies_target(
        self,
        target_min: Optional[float] = None,
        target_max: Optional[float] = None,
    ) -> bool:
        """
        Check if guarantee satisfies target requirements.

        Args:
            target_min: Minimum required outcome (if any)
            target_max: Maximum allowed outcome (if any)

        Returns:
            True if guarantee satisfies all target constraints
        """
        if target_min is not None and self.minimum < target_min:
            return False
        if target_max is not None and self.maximum > target_max:
            return False
        return True


class FACETRobustnessAnalysis(BaseModel):
    """
    Complete FACET robustness analysis result.

    Contains robust intervention region(s), outcome guarantees,
    robustness metrics, and fragility warnings.
    """

    status: str = Field(
        ...,
        description="Analysis status: robust, fragile, or failed",
    )

    # Robust regions found
    robust_regions: List[InterventionRegion] = Field(
        default_factory=list,
        description="Regions achieving target outcome",
    )

    # Outcome guarantees per region
    outcome_guarantees: Dict[str, OutcomeGuarantee] = Field(
        default_factory=dict,
        description="Guaranteed outcomes by outcome variable",
    )

    # Robustness metrics
    robustness_score: float = Field(
        ...,
        description="Overall robustness (0-1, higher = more robust)",
        ge=0,
        le=1,
    )
    region_count: int = Field(..., description="Number of robust regions found", ge=0)
    total_volume: float = Field(
        ...,
        description="Total volume of robust regions",
        ge=0,
    )

    # Fragility indicators
    is_fragile: bool = Field(..., description="True if recommendation is fragile")
    fragility_reasons: List[str] = Field(
        default_factory=list,
        description="Why recommendation might be fragile",
    )

    # Verification details
    samples_tested: int = Field(..., description="Samples used in verification", ge=0)
    samples_successful: int = Field(..., description="Samples achieving target", ge=0)
    confidence_level: float = Field(
        0.95,
        description="Statistical confidence",
        ge=0,
        le=1,
    )

    # User-facing interpretation
    interpretation: str = Field(..., description="Plain English summary")
    recommendation: str = Field(..., description="Actionable guidance")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "robust",
                "robust_regions": [
                    {"variable_ranges": {"price": (52.0, 58.0)}}
                ],
                "outcome_guarantees": {
                    "revenue": {
                        "outcome_variable": "revenue",
                        "minimum": 95000.0,
                        "maximum": 105000.0,
                        "confidence": 0.95,
                    }
                },
                "robustness_score": 0.75,
                "region_count": 1,
                "total_volume": 0.06,
                "is_fragile": False,
                "fragility_reasons": [],
                "samples_tested": 500,
                "samples_successful": 500,
                "confidence_level": 0.95,
                "interpretation": "ROBUST RECOMMENDATION (robustness: 0.75). Multiple intervention strategies achieve the target outcome.",
                "recommendation": "Recommendation: Proceed with confidence. Operating ranges: price: 52.0-58.0. Strategy is robust to reasonable variations.",
            }
        }
    }


class RobustnessRequest(BaseModel):
    """Request for FACET robustness analysis."""

    # Causal model structure
    causal_model: Dict = Field(..., description="Causal DAG structure (nodes and edges)")

    # Target intervention and outcome
    intervention_proposal: Dict[str, float] = Field(
        ...,
        description="Proposed intervention (e.g., {'price': 55})",
    )
    target_outcome: Dict[str, Tuple[float, float]] = Field(
        ...,
        description="Target outcome ranges (e.g., {'revenue': (95000, 105000)})",
    )

    # Search parameters
    perturbation_radius: float = Field(
        0.1,
        description="How far to search around proposal (0.1 = ±10%)",
        gt=0,
        le=1,
    )
    min_samples: int = Field(
        100,
        description="Minimum samples for verification per region",
        ge=10,
        le=1000,
    )
    confidence_level: float = Field(
        0.95,
        description="Required confidence level",
        ge=0.5,
        le=0.99,
    )

    # Optional constraints
    feasible_ranges: Optional[Dict[str, Tuple[float, float]]] = Field(
        None,
        description="Feasible ranges for intervention variables",
    )

    # Structural model for counterfactual simulation
    structural_model: Optional[Dict] = Field(
        None,
        description="Structural equations and distributions for simulation",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "causal_model": {
                    "nodes": ["price", "demand", "revenue"],
                    "edges": [["price", "demand"], ["demand", "revenue"]],
                },
                "intervention_proposal": {"price": 55.0},
                "target_outcome": {"revenue": (95000.0, 105000.0)},
                "perturbation_radius": 0.1,
                "min_samples": 100,
                "confidence_level": 0.95,
            }
        }
    }


class RobustnessResponse(BaseModel):
    """Response with FACET robustness analysis."""

    analysis: FACETRobustnessAnalysis = Field(
        ...,
        description="Robustness analysis result",
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Response metadata",
        alias="_metadata",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "analysis": {
                    "status": "robust",
                    "robustness_score": 0.75,
                    "region_count": 2,
                    "is_fragile": False,
                    "interpretation": "ROBUST RECOMMENDATION",
                    "recommendation": "Proceed with confidence",
                }
            }
        }
    }
