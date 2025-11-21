"""
Response Pydantic models for API endpoints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .metadata import ResponseMetadata
from .shared import (
    ConfidenceInterval,
    ConfidenceLevel,
    ConflictSeverity,
    ExplanationMetadata,
    ImportanceLevel,
    RobustnessLevel,
    SensitivityRange,
    UncertaintyLevel,
    ValidationIssueType,
    ValidationStatus,
)


class ErrorCode(str, Enum):
    """Error codes for structured error responses."""

    INVALID_DAG = "invalid_dag_structure"
    INVALID_MODEL = "invalid_structural_model"
    COMPUTATION_ERROR = "computation_error"
    Y0_ERROR = "y0_library_error"
    FACET_ERROR = "facet_computation_error"
    VALIDATION_ERROR = "validation_error"


class ErrorResponse(BaseModel):
    """Structured error response."""

    error_code: ErrorCode = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details",
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique trace ID for debugging",
    )
    retryable: bool = Field(..., description="Whether the request can be retried")
    suggested_action: str = Field(..., description="Suggested action to resolve error")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error_code": "invalid_dag_structure",
                "message": "DAG contains cycles",
                "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                "retryable": False,
                "suggested_action": "fix_input",
            }
        }
    }


class ValidationIssue(BaseModel):
    """Issue identified during causal validation."""

    type: ValidationIssueType = Field(..., description="Type of validation issue")
    description: str = Field(..., description="Description of the issue")
    affected_nodes: List[str] = Field(..., description="Nodes affected by this issue")
    suggested_action: str = Field(..., description="How to resolve the issue")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "missing_connection",
                "description": "CustomerAcquisition may confound the Price-Revenue relationship",
                "affected_nodes": ["CustomerAcquisition", "Price", "Revenue"],
                "suggested_action": "Clarify: Does CustomerAcquisition affect Price?",
            }
        }
    }


class AssumptionDetail(BaseModel):
    """
    Individual assumption required for causal identification.

    Provides structured information about each assumption, including
    its type, description, and criticality.
    """

    type: str = Field(..., description="Assumption type (e.g., 'no_unmeasured_confounding', 'positivity')")
    description: str = Field(..., description="Human-readable explanation of the assumption")
    critical: bool = Field(..., description="Whether this assumption is critical for identification")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "no_unmeasured_confounding",
                "description": "No unmeasured confounders after adjusting for adjustment set",
                "critical": True
            }
        }
    }


class AlternativeMethod(BaseModel):
    """
    Alternative identification method assessment.

    Documents whether alternative methods (backdoor, front-door, IV, do-calculus)
    are applicable and why.
    """

    method: str = Field(..., description="Method name (e.g., 'backdoor', 'front_door', 'instrumental_variables')")
    applicable: bool = Field(..., description="Whether this method is applicable to the problem")
    reason: str = Field(..., description="Why the method is or is not applicable")

    model_config = {
        "json_schema_extra": {
            "example": {
                "method": "front_door",
                "applicable": False,
                "reason": "No mediator set completely captures causal pathway"
            }
        }
    }


class CausalValidationResponse(BaseModel):
    """
    Enhanced response model for causal validation endpoint.

    Provides comprehensive Y₀-powered identifiability analysis including:
    - Identification method used (backdoor, front-door, IV, do-calculus)
    - Adjustment sets and identification formula
    - Structured assumptions with criticality
    - Alternative methods considered
    - Structured failure diagnosis when non-identifiable
    - Graceful degradation with fallback assessment
    """

    status: ValidationStatus = Field(..., description="Validation status")

    # Enhanced identifiable case fields
    method: Optional[str] = Field(
        default=None,
        description="Identification method used: backdoor, front_door, instrumental_variables, do_calculus"
    )
    adjustment_sets: Optional[List[List[str]]] = Field(
        default=None,
        description="Valid adjustment sets for causal identification",
    )
    minimal_set: Optional[List[str]] = Field(
        default=None,
        description="Minimal sufficient adjustment set",
    )
    identification_formula: Optional[str] = Field(
        default=None,
        description="Human-readable identification formula"
    )
    structured_assumptions: Optional[List[AssumptionDetail]] = Field(
        default=None,
        description="Required assumptions for valid identification (structured)"
    )
    alternative_methods: Optional[List[AlternativeMethod]] = Field(
        default=None,
        description="Alternative identification methods considered"
    )
    backdoor_paths: Optional[List[str]] = Field(
        default=None,
        description="Backdoor paths that create confounding",
    )

    # Enhanced non-identifiable case fields
    reason: Optional[str] = Field(
        default=None,
        description="Why identification failed (e.g., 'no_causal_path', 'unmeasured_confounding', 'selection_bias')"
    )
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggested remedies to achieve identifiability"
    )
    attempted_methods: Optional[List[str]] = Field(
        default=None,
        description="Methods attempted during identification"
    )
    issues: Optional[List[ValidationIssue]] = Field(
        default=None,
        description="Issues preventing identification (legacy compatibility)",
    )

    # Degraded mode fields
    fallback_assessment: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Best-effort assessment when Y₀ fails"
    )

    # Always present
    confidence: ConfidenceLevel = Field(..., description="Confidence in the analysis")
    explanation: ExplanationMetadata = Field(..., description="Explanation metadata")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Identifiable (Backdoor)",
                    "value": {
                        "status": "identifiable",
                        "method": "backdoor",
                        "adjustment_sets": [["Brand"], ["Brand", "CustomerAcquisition"]],
                        "minimal_set": ["Brand"],
                        "identification_formula": "P(revenue|do(price)) = Σ_Brand P(revenue|price, Brand) P(Brand)",
                        "structured_assumptions": [
                            {
                                "type": "no_unmeasured_confounding",
                                "description": "No unmeasured confounders after adjusting for Brand",
                                "critical": True
                            },
                            {
                                "type": "positivity",
                                "description": "All price values possible at all Brand levels",
                                "critical": True
                            }
                        ],
                        "alternative_methods": [
                            {
                                "method": "backdoor",
                                "applicable": True,
                                "reason": "Valid adjustment set exists"
                            },
                            {
                                "method": "front_door",
                                "applicable": False,
                                "reason": "No mediator set completely captures pathway"
                            }
                        ],
                        "backdoor_paths": ["Price ← Brand → Revenue"],
                        "confidence": "high",
                        "explanation": {
                            "summary": "Effect is identifiable by controlling for Brand",
                            "reasoning": "Brand influences both Price and Revenue, creating confounding. Controlling for Brand blocks the backdoor path.",
                            "technical_basis": "Backdoor criterion satisfied with adjustment set {Brand}",
                            "assumptions": ["No unmeasured confounding", "Correct causal structure"]
                        }
                    }
                },
                {
                    "title": "Non-Identifiable",
                    "value": {
                        "status": "cannot_identify",
                        "reason": "unmeasured_confounding",
                        "suggestions": [
                            "Add measured confounders to the model",
                            "Consider using instrumental variables if available",
                            "Explore front-door criterion with full mediation"
                        ],
                        "attempted_methods": ["backdoor", "front_door", "do_calculus"],
                        "confidence": "high",
                        "explanation": {
                            "summary": "Effect cannot be identified due to unmeasured confounding",
                            "reasoning": "Backdoor paths exist but no valid adjustment set found with measured variables.",
                            "technical_basis": "No identification formula available",
                            "assumptions": ["DAG structure correct"]
                        }
                    }
                }
            ]
        }
    }


class ScenarioDescription(BaseModel):
    """Description of the counterfactual scenario."""

    intervention: Dict[str, float] = Field(..., description="Intervention values")
    outcome: str = Field(..., description="Outcome variable")
    context: Optional[Dict[str, float]] = Field(
        default=None,
        description="Observed context",
    )


class PredictionResults(BaseModel):
    """Prediction results for counterfactual analysis."""

    point_estimate: float = Field(..., description="Most likely outcome value")
    confidence_interval: ConfidenceInterval = Field(
        ...,
        description="Confidence interval for prediction",
    )
    sensitivity_range: SensitivityRange = Field(
        ...,
        description="Range of plausible outcomes",
    )


class UncertaintySource(BaseModel):
    """Source of uncertainty in the analysis."""

    factor: str = Field(..., description="Name of uncertainty factor")
    impact: float = Field(..., description="Contribution to variance")
    confidence: ConfidenceLevel = Field(..., description="Confidence in this estimate")
    explanation: str = Field(..., description="Plain English explanation")
    basis: str = Field(..., description="Evidence source for this factor")

    model_config = {
        "json_schema_extra": {
            "example": {
                "factor": "Brand perception lag",
                "impact": 3000,
                "confidence": "medium",
                "explanation": "Brand changes take 2-4 weeks to affect revenue",
                "basis": "Historical data from 3 previous price changes",
            }
        }
    }


class UncertaintyBreakdown(BaseModel):
    """Breakdown of uncertainty sources."""

    overall: UncertaintyLevel = Field(..., description="Overall uncertainty level")
    sources: List[UncertaintySource] = Field(..., description="Individual uncertainty sources")


class CriticalAssumption(BaseModel):
    """A critical assumption affecting robustness."""

    assumption: str = Field(..., description="The assumption being tested")
    impact: float = Field(..., description="How much result changes if assumption is wrong")
    confidence: ConfidenceLevel = Field(..., description="Confidence in this assumption")
    recommendation: str = Field(..., description="Recommended action")

    model_config = {
        "json_schema_extra": {
            "example": {
                "assumption": "Customer price sensitivity remains constant",
                "impact": 15000,
                "confidence": "medium",
                "recommendation": "Consider A/B testing before full rollout",
            }
        }
    }


class RobustnessAnalysis(BaseModel):
    """Robustness analysis results."""

    score: RobustnessLevel = Field(..., description="Overall robustness score")
    critical_assumptions: List[CriticalAssumption] = Field(
        ...,
        description="Critical assumptions",
    )


class CounterfactualResponse(BaseModel):
    """Response model for counterfactual analysis endpoint."""

    scenario: ScenarioDescription = Field(..., description="Scenario description")
    prediction: PredictionResults = Field(..., description="Prediction results")
    uncertainty: UncertaintyBreakdown = Field(..., description="Uncertainty breakdown")
    robustness: RobustnessAnalysis = Field(..., description="Robustness analysis")
    explanation: ExplanationMetadata = Field(..., description="Explanation metadata")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "scenario": {"intervention": {"Price": 15}, "outcome": "Revenue"},
                "prediction": {
                    "point_estimate": 51000,
                    "confidence_interval": {"lower": 45000, "upper": 55000},
                    "sensitivity_range": {
                        "optimistic": 62000,
                        "pessimistic": 38000,
                        "explanation": "Range accounts for competitive effects",
                    },
                },
                "uncertainty": {
                    "overall": "medium",
                    "sources": [
                        {
                            "factor": "Brand perception lag",
                            "impact": 3000,
                            "confidence": "medium",
                            "explanation": "Brand changes take time",
                            "basis": "Historical data",
                        }
                    ],
                },
                "robustness": {
                    "score": "moderate",
                    "critical_assumptions": [
                        {
                            "assumption": "Price sensitivity constant",
                            "impact": 15000,
                            "confidence": "medium",
                            "recommendation": "A/B test before rollout",
                        }
                    ],
                },
                "explanation": {
                    "summary": "Revenue likely increases by £45k-£55k",
                    "reasoning": "Price increase drives growth...",
                    "technical_basis": "FACET region-based analysis",
                    "assumptions": ["Structural equations correct"],
                },
            }
        }
    }


class CommonGround(BaseModel):
    """Common ground among team perspectives."""

    shared_goals: List[str] = Field(..., description="Goals shared by all roles")
    shared_constraints: List[str] = Field(..., description="Constraints shared by all roles")
    agreement_level: float = Field(..., description="Agreement percentage (0-100)")


class Tradeoff(BaseModel):
    """A tradeoff for a specific role."""

    role: str = Field(..., description="Role making the tradeoff")
    gives: str = Field(..., description="What this role compromises on")
    gets: str = Field(..., description="What this role gains")


class AlignedOption(BaseModel):
    """An option with alignment analysis."""

    option: str = Field(..., description="Option ID")
    satisfies_roles: List[str] = Field(..., description="Roles satisfied by this option")
    satisfaction_score: float = Field(..., description="Overall satisfaction (0-100)")
    tradeoffs: List[Tradeoff] = Field(..., description="Tradeoffs for each role")


class Conflict(BaseModel):
    """A conflict between team perspectives."""

    between: List[str] = Field(..., description="Roles in conflict")
    about: str = Field(..., description="What the conflict is about")
    severity: ConflictSeverity = Field(..., description="Conflict severity")
    suggestion: str = Field(..., description="Potential resolution")


class Recommendation(BaseModel):
    """Recommendation for team alignment."""

    top_option: str = Field(..., description="Recommended option ID")
    rationale: str = Field(..., description="Why this option is recommended")
    confidence: ConfidenceLevel = Field(..., description="Confidence in recommendation")
    next_steps: List[str] = Field(..., description="Suggested next steps")


class TeamAlignmentResponse(BaseModel):
    """Response model for team alignment endpoint."""

    common_ground: CommonGround = Field(..., description="Shared goals and constraints")
    aligned_options: List[AlignedOption] = Field(..., description="Options with alignment scores")
    conflicts: List[Conflict] = Field(..., description="Identified conflicts")
    recommendation: Recommendation = Field(..., description="Top recommendation")
    explanation: ExplanationMetadata = Field(..., description="Explanation metadata")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "common_ground": {
                    "shared_goals": ["Deliver value to users", "Meet Q4 deadline"],
                    "shared_constraints": ["Limited budget"],
                    "agreement_level": 72,
                },
                "aligned_options": [
                    {
                        "option": "option_b",
                        "satisfies_roles": ["PM", "Designer", "Engineer"],
                        "satisfaction_score": 85,
                        "tradeoffs": [],
                    }
                ],
                "conflicts": [],
                "recommendation": {
                    "top_option": "option_b",
                    "rationale": "Balances all priorities",
                    "confidence": "high",
                    "next_steps": ["Define MVP scope"],
                },
                "explanation": {
                    "summary": "Team can align on polished feature set",
                    "reasoning": "All roles value quality...",
                    "technical_basis": "Overlap analysis with priority weighting",
                    "assumptions": ["Priorities accurate"],
                },
            }
        }
    }


class ImpactAssessment(BaseModel):
    """Impact of an assumption being wrong."""

    if_wrong: float = Field(..., description="Absolute change in result if assumption wrong")
    percentage: float = Field(..., description="Percentage of total variance explained")


class AssumptionAnalysis(BaseModel):
    """Analysis of a single assumption."""

    name: str = Field(..., description="Assumption name")
    current_value: Any = Field(..., description="Current assumed value")
    importance: ImportanceLevel = Field(..., description="Importance level")
    impact: ImpactAssessment = Field(..., description="Impact assessment")
    confidence: ConfidenceLevel = Field(..., description="Confidence in assumption")
    evidence: str = Field(..., description="Evidence for this assumption")
    recommendation: str = Field(..., description="Recommended action")


class Breakpoint(BaseModel):
    """A breakpoint where conclusion changes."""

    assumption: str = Field(..., description="Assumption name")
    threshold: str = Field(..., description="Threshold where conclusion flips")


class RobustnessScore(BaseModel):
    """Overall robustness score for sensitivity analysis."""

    overall: RobustnessLevel = Field(..., description="Overall robustness level")
    summary: str = Field(..., description="Summary of robustness")
    breakpoints: List[Breakpoint] = Field(..., description="Critical breakpoints")


class ConclusionStatement(BaseModel):
    """Conclusion statement for sensitivity analysis."""

    statement: str = Field(..., description="The conclusion being tested")
    base_case: float = Field(..., description="Base case result value")


class SensitivityAnalysisResponse(BaseModel):
    """Response model for sensitivity analysis endpoint."""

    conclusion: ConclusionStatement = Field(..., description="Conclusion being tested")
    assumptions: List[AssumptionAnalysis] = Field(..., description="Analyzed assumptions")
    robustness: RobustnessScore = Field(..., description="Overall robustness")
    explanation: ExplanationMetadata = Field(..., description="Explanation metadata")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "conclusion": {
                    "statement": "Revenue increases by £45k-£55k per month",
                    "base_case": 51000,
                },
                "assumptions": [
                    {
                        "name": "Price sensitivity",
                        "current_value": 0.5,
                        "importance": "critical",
                        "impact": {"if_wrong": 15000, "percentage": 58},
                        "confidence": "medium",
                        "evidence": "Historical data",
                        "recommendation": "Validate with A/B test",
                    }
                ],
                "robustness": {
                    "overall": "moderate",
                    "summary": "Holds unless price sensitivity substantially different",
                    "breakpoints": [
                        {
                            "assumption": "Price sensitivity",
                            "threshold": "If sensitivity > 0.75, impact becomes negative",
                        }
                    ],
                },
                "explanation": {
                    "summary": "Moderately robust with one critical assumption",
                    "reasoning": "Depends on price sensitivity...",
                    "technical_basis": "One-at-a-time sensitivity analysis",
                    "assumptions": ["Assumptions independent"],
                },
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(default="healthy", description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Current timestamp")
    config_fingerprint: Optional[str] = Field(
        default=None,
        description="Config fingerprint for determinism verification"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }
    }
