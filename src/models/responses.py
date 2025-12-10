"""
Response Pydantic models for API endpoints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
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
    """
    ISL-specific error codes following Olumi platform standard.

    All codes use ISL_ prefix for clarity in multi-service debugging.
    """

    # DAG Structure Errors
    INVALID_DAG = "ISL_INVALID_DAG"
    DAG_CYCLIC = "ISL_DAG_CYCLIC"
    DAG_EMPTY = "ISL_DAG_EMPTY"
    DAG_DISCONNECTED = "ISL_DAG_DISCONNECTED"
    NODE_NOT_FOUND = "ISL_NODE_NOT_FOUND"

    # Model Errors
    INVALID_MODEL = "ISL_INVALID_MODEL"
    INVALID_EQUATION = "ISL_INVALID_EQUATION"
    INVALID_DISTRIBUTION = "ISL_INVALID_DISTRIBUTION"
    MISSING_VARIABLE = "ISL_MISSING_VARIABLE"

    # Validation Errors
    VALIDATION_ERROR = "ISL_VALIDATION_ERROR"
    CAUSAL_NOT_IDENTIFIABLE = "ISL_CAUSAL_NOT_IDENTIFIABLE"
    NO_ADJUSTMENT_SET = "ISL_NO_ADJUSTMENT_SET"
    UNMEASURED_CONFOUNDING = "ISL_UNMEASURED_CONFOUNDING"

    # Computation Errors
    COMPUTATION_ERROR = "ISL_COMPUTATION_ERROR"
    Y0_ERROR = "ISL_Y0_ERROR"
    FACET_ERROR = "ISL_FACET_ERROR"
    MONTE_CARLO_ERROR = "ISL_MONTE_CARLO_ERROR"
    CONVERGENCE_ERROR = "ISL_CONVERGENCE_ERROR"

    # Input Errors
    INVALID_INPUT = "ISL_INVALID_INPUT"
    INVALID_PROBABILITY = "ISL_INVALID_PROBABILITY"
    INVALID_CONFIDENCE_LEVEL = "ISL_INVALID_CONFIDENCE_LEVEL"
    BATCH_SIZE_EXCEEDED = "ISL_BATCH_SIZE_EXCEEDED"

    # Resource Errors
    TIMEOUT = "ISL_TIMEOUT"
    MEMORY_LIMIT = "ISL_MEMORY_LIMIT"
    RATE_LIMIT_EXCEEDED = "ISL_RATE_LIMIT_EXCEEDED"
    SERVICE_UNAVAILABLE = "ISL_SERVICE_UNAVAILABLE"

    # Cache Errors
    CACHE_ERROR = "ISL_CACHE_ERROR"
    REDIS_ERROR = "ISL_REDIS_ERROR"


class RecoveryHints(BaseModel):
    """Recovery hints for error resolution."""

    hints: List[str] = Field(..., description="List of actionable hints")
    suggestion: str = Field(..., description="Primary suggestion")
    example: Optional[str] = Field(default=None, description="Example fix")


class ErrorResponse(BaseModel):
    """
    Olumi Error Response (v1.0) - ISL Implementation.

    Follows the canonical error schema for all Olumi services.
    See platform error schema standard for full specification.
    """

    # Domain-specific fields (ISL defines these)
    code: str = Field(..., description="Error code (e.g., 'ISL_INVALID_DAG')")
    message: str = Field(..., description="Human-readable error message")
    reason: Optional[str] = Field(
        default=None,
        description="Fine-grained reason code for the error"
    )

    # Service-specific details (optional)
    recovery: Optional[RecoveryHints] = Field(
        default=None,
        description="Recovery suggestions and hints"
    )

    # ISL domain-specific fields
    validation_failures: Optional[List[str]] = Field(
        default=None,
        description="List of validation failures (ISL-specific)"
    )
    node_count: Optional[int] = Field(
        default=None,
        description="Number of nodes in DAG (for structure errors)"
    )
    edge_count: Optional[int] = Field(
        default=None,
        description="Number of edges in DAG (for structure errors)"
    )
    missing_nodes: Optional[List[str]] = Field(
        default=None,
        description="List of missing nodes referenced in equations"
    )
    attempted_methods: Optional[List[str]] = Field(
        default=None,
        description="Identification methods attempted (for validation errors)"
    )

    # Platform-required fields (all services MUST include)
    retryable: bool = Field(..., description="Can client retry this request?")
    source: str = Field(default="isl", description="Service that generated error")
    request_id: str = Field(
        default_factory=lambda: f"req_{uuid4().hex[:16]}",
        description="Request ID for correlation (from X-Request-Id header)"
    )
    degraded: Optional[bool] = Field(
        default=False,
        description="Result is partial/incomplete"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "DAG Structure Error",
                    "value": {
                        "code": "ISL_DAG_CYCLIC",
                        "reason": "cycle_detected",
                        "message": "DAG contains cycles and cannot be processed",
                        "recovery": {
                            "hints": [
                                "Check for circular dependencies in your model",
                                "Use a DAG visualization tool to identify cycles"
                            ],
                            "suggestion": "Remove edges that create cycles",
                            "example": "If A→B→C→A exists, remove one edge"
                        },
                        "validation_failures": ["Cycle: Price → Revenue → CustomerSatisfaction → Price"],
                        "node_count": 5,
                        "edge_count": 6,
                        "retryable": False,
                        "source": "isl",
                        "request_id": "req_abc123def456",
                        "degraded": False
                    }
                },
                {
                    "title": "Validation Error",
                    "value": {
                        "code": "ISL_CAUSAL_NOT_IDENTIFIABLE",
                        "reason": "unmeasured_confounding",
                        "message": "Causal effect cannot be identified",
                        "recovery": {
                            "hints": [
                                "Add measured confounders to the model",
                                "Consider using instrumental variables",
                                "Explore front-door criterion"
                            ],
                            "suggestion": "Measure and include confounding variables"
                        },
                        "attempted_methods": ["backdoor", "front_door", "do_calculus"],
                        "retryable": False,
                        "source": "isl",
                        "request_id": "req_xyz789",
                        "degraded": False
                    }
                },
                {
                    "title": "Timeout Error",
                    "value": {
                        "code": "ISL_TIMEOUT",
                        "message": "Computation exceeded time budget",
                        "recovery": {
                            "hints": [
                                "Simplify your causal model",
                                "Reduce Monte Carlo iterations"
                            ],
                            "suggestion": "Retry with a simpler model"
                        },
                        "retryable": True,
                        "source": "isl",
                        "request_id": "req_timeout123",
                        "degraded": False
                    }
                }
            ]
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


class ConditionalIndependence(BaseModel):
    """
    Conditional independence assumption.

    Represents an assumption that X ⊥ Y | Z (X is independent of Y given Z).
    """

    variable_a: str = Field(..., description="First variable")
    variable_b: str = Field(..., description="Second variable")
    conditioning_set: List[str] = Field(default_factory=list, description="Variables to condition on")

    model_config = {
        "json_schema_extra": {
            "example": {
                "variable_a": "Revenue",
                "variable_b": "Competitors",
                "conditioning_set": ["Brand"]
            }
        }
    }


class SuggestionAction(BaseModel):
    """
    Specific action to implement a suggestion.

    Contains the concrete graph modifications or assumptions needed.
    """

    # For add_mediator/add_confounder
    add_node: Optional[str] = Field(default=None, description="Node to add to the graph")
    add_edges: Optional[List[Tuple[str, str]]] = Field(
        default=None,
        description="Edges to add (list of [from, to] pairs)"
    )

    # For reverse_edge
    reverse_edge: Optional[Tuple[str, str]] = Field(
        default=None,
        description="Edge to reverse ([from, to] pair)"
    )

    # For remove_edge
    remove_edge: Optional[Tuple[str, str]] = Field(
        default=None,
        description="Edge to remove ([from, to] pair)"
    )

    # For conditional independence
    assume_independence: Optional[ConditionalIndependence] = Field(
        default=None,
        description="Conditional independence assumption to make"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Add Confounder",
                    "value": {
                        "add_node": "Competitors",
                        "add_edges": [["Competitors", "Price"], ["Competitors", "Revenue"]]
                    }
                },
                {
                    "title": "Reverse Edge",
                    "value": {
                        "reverse_edge": ["A", "B"]
                    }
                }
            ]
        }
    }


class ValidationSuggestion(BaseModel):
    """
    Actionable suggestion for making a non-identifiable DAG identifiable.

    Provides specific, algorithmic guidance on how to modify the causal graph
    or make additional assumptions to achieve identification.
    """

    type: str = Field(
        ...,
        description="Suggestion type: add_mediator, add_confounder, reverse_edge, add_conditional_independence, remove_edge"
    )
    description: str = Field(..., description="Plain English description of the suggestion")
    technical_detail: str = Field(..., description="Precise causal graph operation or assumption")
    priority: str = Field(
        ...,
        description="Priority level: critical, recommended, optional"
    )
    action: SuggestionAction = Field(..., description="Specific action to take")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Add Confounder",
                    "value": {
                        "type": "add_confounder",
                        "description": "Measure competitor activity to control confounding",
                        "technical_detail": "Add Competitors node to block backdoor path: Price ← Competitors → Revenue",
                        "priority": "critical",
                        "action": {
                            "add_node": "Competitors",
                            "add_edges": [["Competitors", "Price"], ["Competitors", "Revenue"]]
                        }
                    }
                },
                {
                    "title": "Add Mediator",
                    "value": {
                        "type": "add_mediator",
                        "description": "Add CustomerSentiment to model the causal mechanism",
                        "technical_detail": "Replace Price→Revenue with Price→CustomerSentiment→Revenue",
                        "priority": "recommended",
                        "action": {
                            "add_node": "CustomerSentiment",
                            "add_edges": [["Price", "CustomerSentiment"], ["CustomerSentiment", "Revenue"]]
                        }
                    }
                }
            ]
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
    suggestions: Optional[List[ValidationSuggestion]] = Field(
        default=None,
        description="Structured, actionable suggestions to achieve identifiability"
    )
    legacy_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Legacy string suggestions (deprecated, use 'suggestions' instead)"
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


class InterventionChange(BaseModel):
    """A single variable change in an intervention."""

    variable: str = Field(..., description="Variable name")
    from_value: float = Field(..., description="Current value")
    to_value: float = Field(..., description="Proposed new value")
    delta: float = Field(..., description="Absolute change (to_value - from_value)")
    relative_change: float = Field(..., description="Relative change as percentage")
    unit: Optional[str] = Field(default=None, description="Optional unit of measurement")

    model_config = {
        "json_schema_extra": {
            "example": {
                "variable": "Price",
                "from_value": 40,
                "to_value": 45,
                "delta": 5,
                "relative_change": 12.5,
                "unit": "£",
            }
        }
    }


class MinimalIntervention(BaseModel):
    """A minimal intervention achieving target outcome."""

    rank: int = Field(..., description="Rank by optimization criterion (1 = best)", ge=1)
    changes: Dict[str, InterventionChange] = Field(
        ...,
        description="Variables to change and their values",
    )
    expected_outcome: Dict[str, float] = Field(
        ...,
        description="Expected outcome values after intervention",
    )
    confidence_interval: Dict[str, ConfidenceInterval] = Field(
        ...,
        description="Confidence intervals for each outcome",
    )
    feasibility: float = Field(
        ...,
        description="Feasibility score (0-1, higher = more feasible)",
        ge=0,
        le=1,
    )
    cost_estimate: str = Field(
        ...,
        description="Cost estimate: low, medium, high",
    )
    robustness: RobustnessLevel = Field(
        ...,
        description="Robustness level from FACET analysis",
    )
    robustness_score: float = Field(
        ...,
        description="Numerical robustness score (0-1)",
        ge=0,
        le=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "rank": 1,
                "changes": {
                    "Price": {
                        "variable": "Price",
                        "from_value": 40,
                        "to_value": 45,
                        "delta": 5,
                        "relative_change": 12.5,
                        "unit": "£",
                    }
                },
                "expected_outcome": {"Revenue": 51000},
                "confidence_interval": {
                    "Revenue": {"lower": 48000, "upper": 54000, "confidence_level": 0.95}
                },
                "feasibility": 0.95,
                "cost_estimate": "low",
                "robustness": "robust",
                "robustness_score": 0.85,
            }
        }
    }


class InterventionComparison(BaseModel):
    """Comparison of multiple interventions."""

    best_by_cost: int = Field(..., description="Rank of best intervention by cost")
    best_by_robustness: int = Field(..., description="Rank of best intervention by robustness")
    best_by_feasibility: int = Field(..., description="Rank of best intervention by feasibility")
    synergies: str = Field(..., description="Analysis of combining multiple interventions")
    tradeoffs: str = Field(..., description="Key tradeoffs between interventions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "best_by_cost": 1,
                "best_by_robustness": 1,
                "best_by_feasibility": 2,
                "synergies": "Combining Price and Marketing changes yields diminishing returns (expected gain: £52k vs £51k individually)",
                "tradeoffs": "Price increase is cheaper but less robust; Marketing spend is more robust but costlier",
            }
        }
    }


class ContrastiveExplanationResponse(BaseModel):
    """Response model for contrastive explanation endpoint."""

    minimal_interventions: List[MinimalIntervention] = Field(
        ...,
        description="Minimal interventions ranked by optimization criterion",
    )
    comparison: InterventionComparison = Field(
        ...,
        description="Comparison of interventions",
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Plain English explanation of recommendations",
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "minimal_interventions": [
                    {
                        "rank": 1,
                        "changes": {
                            "Price": {
                                "variable": "Price",
                                "from_value": 40,
                                "to_value": 45,
                                "delta": 5,
                                "relative_change": 12.5,
                                "unit": "£",
                            }
                        },
                        "expected_outcome": {"Revenue": 51000},
                        "confidence_interval": {
                            "Revenue": {"lower": 48000, "upper": 54000, "confidence_level": 0.95}
                        },
                        "feasibility": 0.95,
                        "cost_estimate": "low",
                        "robustness": "robust",
                        "robustness_score": 0.85,
                    }
                ],
                "comparison": {
                    "best_by_cost": 1,
                    "best_by_robustness": 1,
                    "best_by_feasibility": 1,
                    "synergies": "Single intervention sufficient for target",
                    "tradeoffs": "No significant tradeoffs",
                },
                "explanation": {
                    "summary": "Increase Price from £40 to £45 to achieve target revenue",
                    "reasoning": "Price has strong causal effect on Revenue. A 12.5% increase yields expected £11k revenue gain.",
                    "technical_basis": "Binary search with FACET robustness verification",
                    "assumptions": ["Price elasticity remains constant", "No competitive response"],
                },
            }
        }
    }


class ScenarioResult(BaseModel):
    """Result for a single counterfactual scenario."""

    scenario_id: str = Field(..., description="Scenario identifier")
    intervention: Dict[str, float] = Field(..., description="Intervention values")
    label: Optional[str] = Field(default=None, description="Human-readable label")
    prediction: PredictionResults = Field(..., description="Prediction results")
    uncertainty: UncertaintyBreakdown = Field(..., description="Uncertainty breakdown")
    robustness: RobustnessAnalysis = Field(..., description="Robustness analysis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "scenario_id": "aggressive_pricing",
                "intervention": {"Price": 50},
                "label": "10% price increase",
                "prediction": {
                    "point_estimate": 35000,
                    "confidence_interval": {"lower": 33000, "upper": 37000},
                    "sensitivity_range": {"optimistic": 38000, "pessimistic": 32000, "explanation": "Range accounts for uncertainty"},
                },
                "uncertainty": {
                    "overall": "medium",
                    "sources": [],
                },
                "robustness": {
                    "score": "robust",
                    "critical_assumptions": [],
                },
            }
        }
    }


class PairwiseInteraction(BaseModel):
    """Pairwise interaction between two variables."""

    variables: Tuple[str, str] = Field(..., description="Pair of variables")
    type: str = Field(
        ...,
        description="Interaction type: synergistic, antagonistic, or additive"
    )
    effect_size: float = Field(..., description="Interaction effect size in outcome units")
    significance: float = Field(..., description="Significance score (0-1)", ge=0, le=1)
    explanation: str = Field(..., description="Plain English explanation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "variables": ("Price", "Quality"),
                "type": "synergistic",
                "effect_size": 5000,
                "significance": 0.85,
                "explanation": "Price and Quality interact synergistically: combined effect (£55k) exceeds sum of individual effects (£48k)",
            }
        }
    }


class InteractionAnalysis(BaseModel):
    """Analysis of variable interactions."""

    pairwise: List[PairwiseInteraction] = Field(
        default_factory=list,
        description="Pairwise interactions detected",
    )
    summary: str = Field(..., description="Plain English summary of interactions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pairwise": [
                    {
                        "variables": ("Price", "Quality"),
                        "type": "synergistic",
                        "effect_size": 5000,
                        "significance": 0.85,
                        "explanation": "Price and Quality interact synergistically",
                    }
                ],
                "summary": "Strong synergistic interaction between Price and Quality (£5k additional gain)",
            }
        }
    }


class ScenarioComparison(BaseModel):
    """Comparison of multiple scenarios."""

    best_outcome: str = Field(..., description="Scenario ID with best outcome")
    most_robust: str = Field(..., description="Scenario ID with highest robustness")
    marginal_gains: Dict[str, float] = Field(
        ...,
        description="Marginal uplift of each scenario vs baseline",
    )
    ranking: List[str] = Field(..., description="Scenarios ranked by outcome")

    model_config = {
        "json_schema_extra": {
            "example": {
                "best_outcome": "combined",
                "most_robust": "baseline",
                "marginal_gains": {
                    "increase": 5000,
                    "combined": 12000,
                },
                "ranking": ["combined", "increase", "baseline"],
            }
        }
    }


class BatchCounterfactualResponse(BaseModel):
    """Response model for batch counterfactual analysis."""

    scenarios: List[ScenarioResult] = Field(
        ...,
        description="Results for each scenario",
    )
    interactions: Optional[InteractionAnalysis] = Field(
        default=None,
        description="Interaction analysis (if enabled)",
    )
    comparison: ScenarioComparison = Field(
        ...,
        description="Comparison across scenarios",
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Overall explanation",
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "scenarios": [
                    {
                        "scenario_id": "baseline",
                        "intervention": {"Price": 40},
                        "label": "Current pricing",
                        "prediction": {"point_estimate": 30000, "confidence_interval": {"lower": 28000, "upper": 32000}},
                    },
                    {
                        "scenario_id": "increase",
                        "intervention": {"Price": 50},
                        "label": "10% increase",
                        "prediction": {"point_estimate": 35000, "confidence_interval": {"lower": 33000, "upper": 37000}},
                    },
                ],
                "interactions": {
                    "pairwise": [],
                    "summary": "No significant interactions detected",
                },
                "comparison": {
                    "best_outcome": "increase",
                    "most_robust": "baseline",
                    "marginal_gains": {"increase": 5000},
                    "ranking": ["increase", "baseline"],
                },
                "explanation": {
                    "summary": "Price increase yields £5k marginal gain with moderate robustness",
                    "reasoning": "10% price increase produces consistent revenue uplift across scenarios",
                    "technical_basis": "Batch counterfactual analysis with interaction detection",
                    "assumptions": ["Structural equations correct", "No external shocks"],
                },
            }
        }
    }


class TransportAssumption(BaseModel):
    """
    Individual assumption required for transportability.

    Documents what must hold for causal effects to transport
    from source domain to target domain.
    """

    type: str = Field(
        ...,
        description="Assumption type (e.g., 'same_mechanism', 'no_selection_bias', 'common_support')"
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the assumption"
    )
    critical: bool = Field(
        ...,
        description="Whether this assumption is critical for transportability"
    )
    testable: bool = Field(
        ...,
        description="Whether this assumption can be empirically tested"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "same_mechanism",
                "description": "The causal mechanism Price→Revenue is the same in both UK and Germany",
                "critical": True,
                "testable": False
            }
        }
    }


class TransportabilityResponse(BaseModel):
    """
    Response model for transportability analysis endpoint.

    Indicates whether causal effects identified in source domain
    can be transported to target domain, along with required
    assumptions and robustness assessment.
    """

    transportable: bool = Field(
        ...,
        description="Whether the causal effect can be transported"
    )
    method: Optional[str] = Field(
        default=None,
        description="Transportability method used (e.g., 'selection_diagram', 'weighting', 'direct')"
    )
    formula: Optional[str] = Field(
        default=None,
        description="Transport formula if transportable"
    )
    required_assumptions: List[TransportAssumption] = Field(
        default_factory=list,
        description="Assumptions required for valid transport"
    )
    robustness: str = Field(
        ...,
        description="Robustness assessment: robust, moderate, fragile"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason if not transportable"
    )
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggestions if not transportable"
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence in transportability assessment"
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Plain English explanation"
    )

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
                    "title": "Transportable Effect",
                    "value": {
                        "transportable": True,
                        "method": "selection_diagram",
                        "formula": "P_target(revenue|do(price)) = Σ_S P_source(revenue|price,S) P_target(S)",
                        "required_assumptions": [
                            {
                                "type": "same_mechanism",
                                "description": "Price→Revenue mechanism identical in UK and Germany",
                                "critical": True,
                                "testable": False
                            },
                            {
                                "type": "no_selection_bias",
                                "description": "Selection into domains doesn't affect mechanism",
                                "critical": True,
                                "testable": True
                            }
                        ],
                        "robustness": "moderate",
                        "confidence": "medium",
                        "explanation": {
                            "summary": "Effect can be transported by adjusting for domain selection",
                            "reasoning": "Selection diagram shows effect is transportable via re-weighting by domain-specific covariates",
                            "technical_basis": "Y₀ transportability algorithm with selection nodes",
                            "assumptions": ["Same causal mechanism", "Measured selection variables"]
                        }
                    }
                },
                {
                    "title": "Non-Transportable Effect",
                    "value": {
                        "transportable": False,
                        "reason": "different_mechanisms",
                        "suggestions": [
                            "Collect data on market structure differences",
                            "Test if price elasticity differs between domains",
                            "Consider stratified analysis by market segment"
                        ],
                        "robustness": "fragile",
                        "confidence": "high",
                        "explanation": {
                            "summary": "Effect cannot be transported due to mechanism differences",
                            "reasoning": "Regulatory differences between UK and Germany alter Price→Revenue mechanism",
                            "technical_basis": "Y₀ transportability algorithm - no valid transport formula found",
                            "assumptions": ["DAG structure correct", "Selection variables complete"]
                        }
                    }
                }
            ]
        }
    }


class ConformalInterval(BaseModel):
    """
    Conformal prediction interval with guaranteed coverage.

    Provides lower and upper bounds with finite-sample validity.
    """

    lower_bound: Dict[str, float] = Field(
        ...,
        description="Lower bound of prediction interval for each outcome variable",
    )
    upper_bound: Dict[str, float] = Field(
        ...,
        description="Upper bound of prediction interval for each outcome variable",
    )
    point_estimate: Dict[str, float] = Field(
        ...,
        description="Point estimate for each outcome variable",
    )
    interval_width: Dict[str, float] = Field(
        ...,
        description="Width of prediction interval (upper - lower)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "lower_bound": {"Revenue": 48000},
                "upper_bound": {"Revenue": 56000},
                "point_estimate": {"Revenue": 52000},
                "interval_width": {"Revenue": 8000},
            }
        }
    }


class CoverageGuarantee(BaseModel):
    """
    Coverage guarantee for conformal prediction.

    Documents the theoretical guarantee that the true value
    will fall within the interval with specified probability.
    """

    nominal_coverage: float = Field(
        ...,
        description="Requested coverage level (e.g., 0.95)",
    )
    guaranteed_coverage: float = Field(
        ...,
        description="Provable coverage level accounting for finite samples",
    )
    finite_sample_valid: bool = Field(
        ...,
        description="Whether guarantee holds for finite samples (always True for conformal)",
    )
    assumptions: List[str] = Field(
        ...,
        description="Assumptions required for coverage guarantee",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "nominal_coverage": 0.95,
                "guaranteed_coverage": 0.9474,
                "finite_sample_valid": True,
                "assumptions": [
                    "Exchangeability of calibration and test points"
                ],
            }
        }
    }


class CalibrationMetrics(BaseModel):
    """
    Metrics assessing calibration quality.

    Provides statistics about the calibration set and
    how well it covers the prediction space.
    """

    calibration_size: int = Field(
        ...,
        description="Number of calibration samples",
    )
    residual_statistics: Dict[str, float] = Field(
        ...,
        description="Statistics of calibration residuals (mean, std, median, iqr)",
    )
    interval_adaptivity: float = Field(
        ...,
        description="How much intervals adapt to local uncertainty (higher = more adaptive)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "calibration_size": 50,
                "residual_statistics": {
                    "mean": 2500,
                    "std": 1200,
                    "median": 2300,
                    "iqr": 1500,
                },
                "interval_adaptivity": 0.48,
            }
        }
    }


class ComparisonMetrics(BaseModel):
    """
    Comparison of conformal vs standard Monte Carlo intervals.

    Shows how conformal intervals differ from traditional
    uncertainty quantification approaches.
    """

    monte_carlo_interval: Dict[str, ConfidenceInterval] = Field(
        ...,
        description="Standard Monte Carlo confidence intervals",
    )
    conformal_interval: Dict[str, tuple[float, float]] = Field(
        ...,
        description="Conformal prediction intervals",
    )
    width_ratio: Dict[str, float] = Field(
        ...,
        description="Ratio of conformal width to Monte Carlo width",
    )
    interpretation: str = Field(
        ...,
        description="Plain English interpretation of the comparison",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "monte_carlo_interval": {
                    "Revenue": {"lower": 49000, "upper": 55000, "confidence_level": 0.95}
                },
                "conformal_interval": {
                    "Revenue": (48000, 56000)
                },
                "width_ratio": {
                    "Revenue": 1.33
                },
                "interpretation": "Conformal interval is 33% wider, providing more honest uncertainty quantification with finite-sample guarantees",
            }
        }
    }


class ConformalCounterfactualResponse(BaseModel):
    """
    Response model for conformal counterfactual prediction.

    Provides prediction intervals with finite-sample valid
    coverage guarantees, calibration quality metrics, and
    comparison to standard methods.
    """

    prediction_interval: ConformalInterval = Field(
        ...,
        description="Conformal prediction interval",
    )
    coverage_guarantee: CoverageGuarantee = Field(
        ...,
        description="Coverage guarantee details",
    )
    calibration_quality: CalibrationMetrics = Field(
        ...,
        description="Quality metrics for calibration set",
    )
    comparison_to_standard: ComparisonMetrics = Field(
        ...,
        description="Comparison to standard Monte Carlo intervals",
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Plain English explanation",
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction_interval": {
                    "lower_bound": {"Revenue": 48000},
                    "upper_bound": {"Revenue": 56000},
                    "point_estimate": {"Revenue": 52000},
                    "interval_width": {"Revenue": 8000},
                },
                "coverage_guarantee": {
                    "nominal_coverage": 0.95,
                    "guaranteed_coverage": 0.9474,
                    "finite_sample_valid": True,
                    "assumptions": ["Exchangeability of calibration and test points"],
                },
                "calibration_quality": {
                    "calibration_size": 50,
                    "residual_statistics": {
                        "mean": 2500,
                        "std": 1200,
                        "median": 2300,
                        "iqr": 1500,
                    },
                    "interval_adaptivity": 0.48,
                },
                "comparison_to_standard": {
                    "monte_carlo_interval": {
                        "Revenue": {"lower": 49000, "upper": 55000, "confidence_level": 0.95}
                    },
                    "conformal_interval": {"Revenue": (48000, 56000)},
                    "width_ratio": {"Revenue": 1.33},
                    "interpretation": "Conformal interval is 33% wider",
                },
                "explanation": {
                    "summary": "Conformal prediction provides 94.7% guaranteed coverage",
                    "reasoning": "Using split conformal with 50 calibration points",
                    "technical_basis": "Finite-sample conformal prediction (Vovk et al. 2005)",
                    "assumptions": ["Exchangeability between calibration and test data"],
                },
            }
        }
    }


class AdjustmentStrategyDetail(BaseModel):
    """
    Complete adjustment strategy for identifiability.

    Specifies exactly what nodes/edges to add to make a non-identifiable
    DAG identifiable.
    """

    strategy_type: str = Field(
        ...,
        description="Strategy type: backdoor, frontdoor, or instrumental"
    )
    nodes_to_add: List[str] = Field(
        ...,
        description="Nodes that need to be added/measured"
    )
    edges_to_add: List[Tuple[str, str]] = Field(
        ...,
        description="Edges to add to the DAG"
    )
    explanation: str = Field(
        ...,
        description="Plain English explanation of the strategy"
    )
    theoretical_basis: str = Field(
        ...,
        description="Theoretical justification (e.g., Pearl's backdoor criterion)"
    )
    expected_identifiability: float = Field(
        ...,
        description="Confidence that this strategy achieves identifiability (0-1)",
        ge=0,
        le=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "strategy_type": "backdoor",
                "nodes_to_add": ["Competitors"],
                "edges_to_add": [
                    ("Competitors", "Price"),
                    ("Competitors", "Revenue"),
                ],
                "explanation": "Add and measure Competitors variable, then control for it to block backdoor paths",
                "theoretical_basis": "Pearl's backdoor criterion",
                "expected_identifiability": 0.9,
            }
        }
    }


class PathAnalysisDetail(BaseModel):
    """
    Analysis of causal paths in a DAG.

    Identifies backdoor paths, frontdoor paths, and critical nodes.
    """

    backdoor_paths: List[List[str]] = Field(
        ...,
        description="Backdoor paths from treatment to outcome"
    )
    frontdoor_paths: List[List[str]] = Field(
        ...,
        description="Directed paths from treatment to outcome"
    )
    blocked_paths: List[List[str]] = Field(
        ...,
        description="Paths that are already blocked by colliders"
    )
    critical_nodes: List[str] = Field(
        ...,
        description="Nodes that block multiple paths if controlled"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "backdoor_paths": [
                    ["Price", "Competitors", "Revenue"]
                ],
                "frontdoor_paths": [
                    ["Price", "Revenue"]
                ],
                "blocked_paths": [],
                "critical_nodes": ["Competitors"],
            }
        }
    }


class ValidationStrategyResponse(BaseModel):
    """
    Response model for enhanced validation strategies.

    Provides complete adjustment strategies and path analysis.
    """

    strategies: List[AdjustmentStrategyDetail] = Field(
        ...,
        description="Adjustment strategies ranked by expected success"
    )
    path_analysis: PathAnalysisDetail = Field(
        ...,
        description="Comprehensive path analysis"
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Plain English explanation"
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "strategies": [
                    {
                        "strategy_type": "backdoor",
                        "nodes_to_add": [],
                        "edges_to_add": [],
                        "explanation": "Control for existing variable Competitors to block backdoor paths",
                        "theoretical_basis": "Pearl's backdoor criterion",
                        "expected_identifiability": 0.9,
                    }
                ],
                "path_analysis": {
                    "backdoor_paths": [["Price", "Competitors", "Revenue"]],
                    "frontdoor_paths": [["Price", "Revenue"]],
                    "blocked_paths": [],
                    "critical_nodes": ["Competitors"],
                },
                "explanation": {
                    "summary": "Effect is identifiable by controlling for Competitors",
                    "reasoning": "Competitors creates a backdoor path between Price and Revenue",
                    "technical_basis": "Backdoor path analysis and adjustment set identification",
                    "assumptions": ["DAG structure is correct"],
                },
            }
        }
    }


class DiscoveredDAG(BaseModel):
    """A discovered DAG structure with confidence score."""

    nodes: List[str] = Field(..., description="Variable names")
    edges: List[Tuple[str, str]] = Field(..., description="Directed edges")
    confidence: float = Field(
        ...,
        description="Confidence in this structure (0-1)",
        ge=0,
        le=1,
    )
    method: str = Field(
        ...,
        description="Discovery method used (correlation, knowledge, hybrid)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "nodes": ["Price", "Quality", "Revenue"],
                "edges": [
                    ("Price", "Revenue"),
                    ("Quality", "Revenue"),
                    ("Quality", "Price"),
                ],
                "confidence": 0.75,
                "method": "correlation",
            }
        }
    }


class DiscoveryResponse(BaseModel):
    """
    Response model for causal discovery.

    Returns discovered DAG structures with confidence scores.
    """

    discovered_dags: List[DiscoveredDAG] = Field(
        ...,
        description="Discovered DAG structures ranked by confidence"
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Plain English explanation"
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "discovered_dags": [
                    {
                        "nodes": ["Price", "Quality", "Revenue"],
                        "edges": [
                            ("Price", "Revenue"),
                            ("Quality", "Revenue"),
                        ],
                        "confidence": 0.85,
                        "method": "correlation",
                    }
                ],
                "explanation": {
                    "summary": "Discovered DAG structure from data with 85% confidence",
                    "reasoning": "Strong correlations between Price→Revenue (0.82) and Quality→Revenue (0.75)",
                    "technical_basis": "Correlation-based structure learning with threshold 0.3",
                    "assumptions": ["Linear relationships", "No hidden confounders"],
                },
            }
        }
    }


class RecommendedExperimentDetail(BaseModel):
    """
    Recommended experiment specification.

    Specifies intervention values, expected outcomes, and information gain.
    """

    intervention: Dict[str, float] = Field(
        ...,
        description="Recommended intervention values"
    )
    expected_outcome: Dict[str, float] = Field(
        ...,
        description="Expected outcome values"
    )
    expected_information_gain: float = Field(
        ...,
        description="Expected information gain from this experiment (0-1)",
        ge=0,
        le=1,
    )
    cost_estimate: float = Field(
        ...,
        description="Estimated cost of the experiment"
    )
    rationale: str = Field(
        ...,
        description="Plain English rationale for this recommendation"
    )
    exploration_vs_exploitation: float = Field(
        ...,
        description="Exploration vs exploitation score (0=exploit, 1=explore)",
        ge=0,
        le=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "intervention": {"Price": 55},
                "expected_outcome": {"Revenue": 38000},
                "expected_information_gain": 0.75,
                "cost_estimate": 10000,
                "rationale": "Explore: Test Price=55 to learn more (high information gain: 0.75)",
                "exploration_vs_exploitation": 0.8,
            }
        }
    }


class ExperimentRecommendationResponse(BaseModel):
    """
    Response model for experiment recommendation.

    Provides recommended next experiment using Thompson sampling.
    """

    recommendation: RecommendedExperimentDetail = Field(
        ...,
        description="Recommended next experiment"
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Plain English explanation"
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "recommendation": {
                    "intervention": {"Price": 55},
                    "expected_outcome": {"Revenue": 38000},
                    "expected_information_gain": 0.75,
                    "cost_estimate": 10000,
                    "rationale": "Explore: Test Price=55 to learn more (high information gain: 0.75)",
                    "exploration_vs_exploitation": 0.8,
                },
                "explanation": {
                    "summary": "Recommend testing Price=55 to maximize learning",
                    "reasoning": "This intervention has high information gain (0.75) and explores undersampled region",
                    "technical_basis": "Thompson sampling with 100 posterior samples",
                    "assumptions": ["Parameter beliefs accurate", "Cost-benefit tradeoff acceptable"],
                },
            }
        }
    }


# ============================================================================
# CEE Enhancement Endpoints (Phase 0) - Response Models
# ============================================================================


class AssumptionSensitivity(BaseModel):
    """Sensitivity information for a single assumption/variable."""

    variable: str = Field(..., description="Node ID or variable name", max_length=100)
    sensitivity: float = Field(
        ...,
        description="Sensitivity score (0-1), higher = more sensitive",
        ge=0.0,
        le=1.0
    )
    impact: str = Field(..., description="Description of impact", max_length=500)

    model_config = {
        "json_schema_extra": {
            "example": {
                "variable": "n_market_size",
                "sensitivity": 0.92,
                "impact": "10% change in market size estimate leads to 25% swing in projected ROI"
            }
        }
    }


class SensitivityDetailedResponse(BaseModel):
    """Response model for detailed sensitivity analysis."""

    assumptions: List[AssumptionSensitivity] = Field(
        ...,
        description="List of assumptions with sensitivity analysis"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "assumptions": [
                    {
                        "variable": "n_market_size",
                        "sensitivity": 0.92,
                        "impact": "10% change in market size estimate leads to 25% swing in projected ROI"
                    },
                    {
                        "variable": "n_adoption_rate",
                        "sensitivity": 0.78,
                        "impact": "Adoption rate uncertainty accounts for most outcome variance"
                    }
                ]
            }
        }
    }


class ContrastiveAlternative(BaseModel):
    """Single contrastive alternative suggestion."""

    change: str = Field(..., description="What to change", max_length=1000)
    outcome_diff: str = Field(
        ...,
        description="How outcome would differ",
        max_length=1000
    )
    feasibility: float = Field(
        ...,
        description="Implementation feasibility (0-1)",
        ge=0.0,
        le=1.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "change": "Increase initial marketing budget by 40%",
                "outcome_diff": "Would increase market penetration from 12% to 18% in Q1",
                "feasibility": 0.85
            }
        }
    }


class ContrastiveResponse(BaseModel):
    """Response model for contrastive explanation."""

    alternatives: List[ContrastiveAlternative] = Field(
        ...,
        description="List of actionable alternatives"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "alternatives": [
                    {
                        "change": "Increase initial marketing budget by 40%",
                        "outcome_diff": "Would increase market penetration from 12% to 18% in Q1",
                        "feasibility": 0.85
                    },
                    {
                        "change": "Partner with established distributor",
                        "outcome_diff": "Would reduce time-to-market by 3 months but increase costs 15%",
                        "feasibility": 0.60
                    }
                ]
            }
        }
    }


class ConformalResponse(BaseModel):
    """Response model for conformal prediction."""

    prediction_interval: List[float] = Field(
        ...,
        description="Numeric bounds [lower, upper]",
        min_length=2,
        max_length=2
    )
    confidence_level: float = Field(
        ...,
        description="Confidence level (0-1)",
        ge=0.0,
        le=1.0
    )
    uncertainty_source: str = Field(
        ...,
        description="Main source of uncertainty",
        max_length=500
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction_interval": [45000, 78000],
                "confidence_level": 0.90,
                "uncertainty_source": "Market demand volatility and competitor response timing"
            }
        }
    }


class ValidationImprovement(BaseModel):
    """Single model improvement suggestion."""

    type: str = Field(
        ...,
        description="Category (e.g., data_collection, model_structure, sensitivity_test)",
        max_length=100
    )
    description: str = Field(
        ...,
        description="What to do",
        max_length=1000
    )
    priority: str = Field(
        ...,
        description="Implementation priority",
        pattern="^(high|medium|low)$"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "data_collection",
                "description": "Gather historical adoption data from 3-5 comparable product launches to calibrate adoption rate assumptions",
                "priority": "high"
            }
        }
    }


class ValidationStrategiesResponse(BaseModel):
    """Response model for validation strategies."""

    suggested_improvements: List[ValidationImprovement] = Field(
        ...,
        description="List of suggested model improvements"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "suggested_improvements": [
                    {
                        "type": "data_collection",
                        "description": "Gather historical adoption data from 3-5 comparable product launches to calibrate adoption rate assumptions",
                        "priority": "high"
                    },
                    {
                        "type": "model_structure",
                        "description": "Add explicit node for competitor response to capture strategic interaction effects",
                        "priority": "medium"
                    },
                    {
                        "type": "sensitivity_test",
                        "description": "Test model robustness by varying market size estimate ±20%",
                        "priority": "low"
                    }
                ]
            }
        }
    }


# ============================================================================
# Parameter Recommendation Endpoint - Response Models
# ============================================================================


class ParameterRecommendation(BaseModel):
    """Single parameter recommendation with range and rationale."""

    parameter: str = Field(
        ...,
        description="Parameter name (e.g., 'n_decision_to_n_outcome_weight')",
        max_length=200
    )
    parameter_type: str = Field(
        ...,
        description="Type of parameter",
        pattern="^(weight|belief)$"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current parameter value (if provided in request)"
    )
    recommended_range: List[float] = Field(
        ...,
        description="Recommended range [min, max]",
        min_length=2,
        max_length=2
    )
    recommended_typical: float = Field(
        ...,
        description="Typical/center value of recommended range"
    )
    rationale: str = Field(
        ...,
        description="Explanation of why this range is recommended",
        max_length=500
    )
    importance: float = Field(
        ...,
        description="Parameter importance (0-1, from sensitivity analysis)",
        ge=0.0,
        le=1.0
    )
    confidence: str = Field(
        ...,
        description="Confidence in recommendation",
        pattern="^(high|medium|low)$"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter": "n_decision_to_n_outcome_weight",
                "parameter_type": "weight",
                "current_value": 2.0,
                "recommended_range": [1.2, 1.8],
                "recommended_typical": 1.5,
                "rationale": "Critical path edge connecting decision to outcome - requires strong causal influence",
                "importance": 0.92,
                "confidence": "high"
            }
        }
    }


class ParameterRecommendationResponse(BaseModel):
    """Response model for parameter recommendation endpoint."""

    recommendations: List[ParameterRecommendation] = Field(
        ...,
        description="List of parameter recommendations sorted by importance"
    )
    graph_characteristics: Dict[str, Any] = Field(
        ...,
        description="Metadata about graph structure"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "recommendations": [
                    {
                        "parameter": "n_decision_to_n_outcome_weight",
                        "parameter_type": "weight",
                        "current_value": 2.0,
                        "recommended_range": [1.2, 1.8],
                        "recommended_typical": 1.5,
                        "rationale": "Critical path edge connecting decision to outcome - requires strong causal influence",
                        "importance": 0.92,
                        "confidence": "high"
                    },
                    {
                        "parameter": "n_decision_belief",
                        "parameter_type": "belief",
                        "current_value": 0.8,
                        "recommended_range": [0.75, 0.95],
                        "recommended_typical": 0.85,
                        "rationale": "Treatment node - high certainty recommended for robust analysis",
                        "importance": 0.78,
                        "confidence": "high"
                    }
                ],
                "graph_characteristics": {
                    "num_critical_edges": 3,
                    "max_path_length": 4,
                    "avg_centrality": 0.42
                }
            }
        }
    }


# ============================================================================
# Phase 4: Sequential Decisions & Conditional Recommendations - Response Models
# ============================================================================


class ConditionExpression(BaseModel):
    """Machine-readable condition expression."""

    parameter: str = Field(
        ...,
        description="Parameter name being conditioned on"
    )
    operator: str = Field(
        ...,
        description="Comparison operator",
        pattern="^(<|<=|>|>=|==|in_range)$"
    )
    value: Any = Field(
        ...,
        description="Threshold value or range tuple for 'in_range'"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "parameter": "marketing_roi",
                "operator": "<",
                "value": 0.3
            }
        }
    }


class PrimaryRecommendation(BaseModel):
    """Primary recommendation for an option."""

    option_id: str = Field(..., description="Recommended option ID")
    label: str = Field(..., description="Human-readable label")
    confidence: str = Field(
        ...,
        description="Confidence level",
        pattern="^(high|medium|low)$"
    )
    expected_value: float = Field(..., description="Expected value of this option")

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "option_a",
                "label": "Aggressive Expansion",
                "confidence": "high",
                "expected_value": 75000.0
            }
        }
    }


class ConditionalRecommendation(BaseModel):
    """A conditional recommendation that depends on parameter values."""

    condition_id: str = Field(..., description="Unique condition identifier")
    condition_type: str = Field(
        ...,
        description="Type of condition",
        pattern="^(threshold|dominance|risk_profile|scenario)$"
    )
    condition_description: str = Field(
        ...,
        description="Human-readable description of the condition"
    )
    condition_expression: ConditionExpression = Field(
        ...,
        description="Machine-readable condition expression"
    )
    triggered_recommendation: PrimaryRecommendation = Field(
        ...,
        description="Recommendation when condition is true"
    )
    probability_of_condition: Optional[float] = Field(
        default=None,
        description="Estimated probability that condition is true",
        ge=0.0,
        le=1.0
    )
    impact_magnitude: str = Field(
        ...,
        description="Impact magnitude if condition is true",
        pattern="^(high|medium|low)$"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "condition_id": "cond_001",
                "condition_type": "threshold",
                "condition_description": "If marketing ROI drops below 0.3",
                "condition_expression": {
                    "parameter": "marketing_roi",
                    "operator": "<",
                    "value": 0.3
                },
                "triggered_recommendation": {
                    "option_id": "option_b",
                    "label": "Conservative Growth",
                    "confidence": "medium",
                    "expected_value": 50000.0
                },
                "probability_of_condition": 0.25,
                "impact_magnitude": "high"
            }
        }
    }


class RobustnessSummary(BaseModel):
    """Summary of recommendation robustness."""

    recommendation_stability: str = Field(
        ...,
        description="Overall stability classification",
        pattern="^(robust|moderate|fragile)$"
    )
    conditions_count: int = Field(
        ...,
        description="Number of conditions that could flip the recommendation",
        ge=0
    )
    closest_flip_point: Optional[ConditionExpression] = Field(
        default=None,
        description="Nearest condition that would flip the recommendation"
    )
    safety_margin: Optional[float] = Field(
        default=None,
        description="Distance to nearest flip point (in parameter units)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "recommendation_stability": "moderate",
                "conditions_count": 3,
                "closest_flip_point": {
                    "parameter": "market_size",
                    "operator": "<",
                    "value": 0.75
                },
                "safety_margin": 0.15
            }
        }
    }


class ConditionalRecommendResponse(BaseModel):
    """Response model for conditional recommendation endpoint."""

    schema_version: str = Field(
        default="conditional_recommend.v1",
        description="Schema version for this response"
    )
    primary_recommendation: PrimaryRecommendation = Field(
        ...,
        description="Primary (unconditional) recommendation"
    )
    conditional_recommendations: List[ConditionalRecommendation] = Field(
        ...,
        description="List of conditional recommendations"
    )
    robustness_summary: RobustnessSummary = Field(
        ...,
        description="Summary of recommendation robustness"
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "schema_version": "conditional_recommend.v1",
                "primary_recommendation": {
                    "option_id": "option_a",
                    "label": "Aggressive Expansion",
                    "confidence": "high",
                    "expected_value": 75000.0
                },
                "conditional_recommendations": [
                    {
                        "condition_id": "cond_001",
                        "condition_type": "threshold",
                        "condition_description": "If marketing ROI drops below 30%",
                        "condition_expression": {
                            "parameter": "marketing_roi",
                            "operator": "<",
                            "value": 0.3
                        },
                        "triggered_recommendation": {
                            "option_id": "option_b",
                            "label": "Conservative Growth",
                            "confidence": "medium",
                            "expected_value": 50000.0
                        },
                        "probability_of_condition": 0.25,
                        "impact_magnitude": "high"
                    }
                ],
                "robustness_summary": {
                    "recommendation_stability": "moderate",
                    "conditions_count": 3,
                    "closest_flip_point": {
                        "parameter": "market_size",
                        "operator": "<",
                        "value": 0.75
                    },
                    "safety_margin": 0.15
                }
            }
        }
    }


# Sequential Decision Support Response Models


class ConditionalAction(BaseModel):
    """A conditional action in a decision rule."""

    condition: str = Field(
        ...,
        description="Human-readable condition string"
    )
    action: str = Field(
        ...,
        description="Action to take if condition is met"
    )
    expected_value_if_taken: float = Field(
        ...,
        description="Expected value if this action is taken"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "condition": "If demand_signal > 0.7",
                "action": "expand",
                "expected_value_if_taken": 85000.0
            }
        }
    }


class DecisionRule(BaseModel):
    """Conditional decision rule based on observed state."""

    default_action: str = Field(
        ...,
        description="Default action if no conditions are met"
    )
    conditional_actions: List[ConditionalAction] = Field(
        default_factory=list,
        description="List of conditional actions"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "default_action": "wait",
                "conditional_actions": [
                    {
                        "condition": "If market_favorable",
                        "action": "expand",
                        "expected_value_if_taken": 85000.0
                    },
                    {
                        "condition": "If market_unfavorable",
                        "action": "exit",
                        "expected_value_if_taken": 10000.0
                    }
                ]
            }
        }
    }


class StagePolicy(BaseModel):
    """Optimal decision policy for a single stage."""

    stage_index: int = Field(..., description="Stage index", ge=0)
    stage_label: str = Field(..., description="Human-readable stage label")
    decision_rule: DecisionRule = Field(..., description="Decision rule for this stage")
    contingent_on: List[str] = Field(
        default_factory=list,
        description="What must be observed before this stage"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "stage_index": 1,
                "stage_label": "Pricing Decision",
                "decision_rule": {
                    "default_action": "standard_pricing",
                    "conditional_actions": [
                        {
                            "condition": "If market_favorable",
                            "action": "premium_pricing",
                            "expected_value_if_taken": 100000.0
                        }
                    ]
                },
                "contingent_on": ["market_outcome"]
            }
        }
    }


class PolicyDistribution(BaseModel):
    """Distribution of policy values."""

    type: str = Field(..., description="Distribution type")
    parameters: Dict[str, float] = Field(..., description="Distribution parameters")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "normal",
                "parameters": {"mean": 45000, "std": 15000}
            }
        }
    }


class Policy(BaseModel):
    """Optimal policy across all stages."""

    stages: List[StagePolicy] = Field(
        ...,
        description="Policy for each stage"
    )
    expected_total_value: float = Field(
        ...,
        description="Expected total value following this policy"
    )
    value_distribution: PolicyDistribution = Field(
        ...,
        description="Distribution of total value"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "stages": [
                    {
                        "stage_index": 0,
                        "stage_label": "Investment",
                        "decision_rule": {
                            "default_action": "invest",
                            "conditional_actions": []
                        },
                        "contingent_on": []
                    }
                ],
                "expected_total_value": 45000.0,
                "value_distribution": {
                    "type": "normal",
                    "parameters": {"mean": 45000, "std": 15000}
                }
            }
        }
    }


class StageOption(BaseModel):
    """Analysis of a single option at a stage."""

    option_id: str = Field(..., description="Option identifier")
    label: str = Field(..., description="Human-readable label")
    immediate_value: float = Field(
        ...,
        description="Immediate payoff of choosing this option"
    )
    continuation_value: float = Field(
        ...,
        description="Expected value of future stages"
    )
    total_value: float = Field(
        ...,
        description="Total value (immediate + discounted continuation)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "invest",
                "label": "Make Investment",
                "immediate_value": -50000,
                "continuation_value": 95000,
                "total_value": 45000
            }
        }
    }


class StageAnalysis(BaseModel):
    """Analysis for a single stage."""

    stage_index: int = Field(..., description="Stage index", ge=0)
    stage_label: str = Field(..., description="Human-readable stage label")
    options_at_stage: List[StageOption] = Field(
        ...,
        description="Options available at this stage with analysis"
    )
    information_value: float = Field(
        ...,
        description="Value of information revealed at this stage"
    )
    optimal_waiting_value: Optional[float] = Field(
        default=None,
        description="Value of delaying decision (if applicable)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "stage_index": 0,
                "stage_label": "Investment Decision",
                "options_at_stage": [
                    {
                        "option_id": "invest",
                        "label": "Make Investment",
                        "immediate_value": -50000,
                        "continuation_value": 95000,
                        "total_value": 45000
                    },
                    {
                        "option_id": "wait",
                        "label": "Wait and See",
                        "immediate_value": 0,
                        "continuation_value": 20000,
                        "total_value": 20000
                    }
                ],
                "information_value": 15000.0,
                "optimal_waiting_value": 20000.0
            }
        }
    }


class SequentialAnalysisResponse(BaseModel):
    """Response model for sequential decision analysis."""

    schema_version: str = Field(
        default="sequential.v1",
        description="Schema version for this response"
    )
    optimal_policy: Policy = Field(
        ...,
        description="Optimal policy across all stages"
    )
    stage_analyses: List[StageAnalysis] = Field(
        ...,
        description="Detailed analysis for each stage"
    )
    value_of_flexibility: float = Field(
        ...,
        description="Value of waiting vs committing now"
    )
    sensitivity_to_timing: str = Field(
        ...,
        description="How sensitive results are to timing",
        pattern="^(high|medium|low)$"
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "schema_version": "sequential.v1",
                "optimal_policy": {
                    "stages": [
                        {
                            "stage_index": 0,
                            "stage_label": "Investment",
                            "decision_rule": {
                                "default_action": "invest",
                                "conditional_actions": []
                            },
                            "contingent_on": []
                        },
                        {
                            "stage_index": 1,
                            "stage_label": "Pricing",
                            "decision_rule": {
                                "default_action": "standard",
                                "conditional_actions": [
                                    {
                                        "condition": "If market_favorable",
                                        "action": "premium",
                                        "expected_value_if_taken": 100000
                                    }
                                ]
                            },
                            "contingent_on": ["market_outcome"]
                        }
                    ],
                    "expected_total_value": 45000.0,
                    "value_distribution": {
                        "type": "normal",
                        "parameters": {"mean": 45000, "std": 15000}
                    }
                },
                "stage_analyses": [
                    {
                        "stage_index": 0,
                        "stage_label": "Investment Decision",
                        "options_at_stage": [
                            {
                                "option_id": "invest",
                                "label": "Make Investment",
                                "immediate_value": -50000,
                                "continuation_value": 95000,
                                "total_value": 45000
                            }
                        ],
                        "information_value": 15000.0,
                        "optimal_waiting_value": 20000.0
                    }
                ],
                "value_of_flexibility": 25000.0,
                "sensitivity_to_timing": "medium"
            }
        }
    }


class PolicyTreeNode(BaseModel):
    """Node in a policy tree representation."""

    node_id: str = Field(..., description="Node identifier")
    stage: int = Field(..., description="Stage this node belongs to")
    node_type: str = Field(
        ...,
        description="Type of node",
        pattern="^(decision|chance|terminal)$"
    )
    label: str = Field(..., description="Human-readable label")
    optimal_action: Optional[str] = Field(
        default=None,
        description="Optimal action at this node (for decision nodes)"
    )
    expected_value: float = Field(..., description="Expected value at this node")
    children: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Child nodes with transition info"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "node_id": "root",
                "stage": 0,
                "node_type": "decision",
                "label": "Investment Decision",
                "optimal_action": "invest",
                "expected_value": 45000,
                "children": [
                    {
                        "action": "invest",
                        "child_id": "market_node",
                        "probability": 1.0
                    }
                ]
            }
        }
    }


class PolicyTreeResponse(BaseModel):
    """Response model for policy tree endpoint."""

    schema_version: str = Field(
        default="policy_tree.v1",
        description="Schema version"
    )
    root: PolicyTreeNode = Field(
        ...,
        description="Root node of the policy tree"
    )
    total_stages: int = Field(..., description="Total number of stages")
    total_nodes: int = Field(..., description="Total number of nodes in tree")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "schema_version": "policy_tree.v1",
                "root": {
                    "node_id": "root",
                    "stage": 0,
                    "node_type": "decision",
                    "label": "Investment Decision",
                    "optimal_action": "invest",
                    "expected_value": 45000,
                    "children": []
                },
                "total_stages": 3,
                "total_nodes": 7
            }
        }
    }


class StageSensitivityResult(BaseModel):
    """Sensitivity analysis result for a single stage."""

    stage_index: int = Field(..., description="Stage index")
    stage_label: str = Field(..., description="Stage label")
    parameter_sensitivities: Dict[str, float] = Field(
        ...,
        description="Mapping of parameter names to sensitivity scores"
    )
    policy_changes_at: Optional[Dict[str, float]] = Field(
        default=None,
        description="Parameter values where optimal policy changes"
    )
    robustness_score: float = Field(
        ...,
        description="Stage robustness score (0-1)",
        ge=0.0,
        le=1.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "stage_index": 0,
                "stage_label": "Investment Decision",
                "parameter_sensitivities": {
                    "market_probability": 0.85,
                    "investment_cost": 0.45
                },
                "policy_changes_at": {
                    "market_probability": 0.35
                },
                "robustness_score": 0.72
            }
        }
    }


class StageSensitivityResponse(BaseModel):
    """Response model for stage sensitivity analysis."""

    schema_version: str = Field(
        default="stage_sensitivity.v1",
        description="Schema version"
    )
    stage_results: List[StageSensitivityResult] = Field(
        ...,
        description="Sensitivity results for each stage"
    )
    overall_robustness: float = Field(
        ...,
        description="Overall policy robustness score",
        ge=0.0,
        le=1.0
    )
    most_sensitive_parameters: List[str] = Field(
        ...,
        description="Parameters with highest sensitivity"
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Plain English explanation"
    )

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "schema_version": "stage_sensitivity.v1",
                "stage_results": [
                    {
                        "stage_index": 0,
                        "stage_label": "Investment Decision",
                        "parameter_sensitivities": {
                            "market_probability": 0.85,
                            "investment_cost": 0.45
                        },
                        "policy_changes_at": {
                            "market_probability": 0.35
                        },
                        "robustness_score": 0.72
                    }
                ],
                "overall_robustness": 0.68,
                "most_sensitive_parameters": ["market_probability", "payoff_high"],
                "explanation": {
                    "summary": "Policy is moderately robust to parameter changes",
                    "reasoning": "Market probability has high sensitivity - small changes could flip optimal decision",
                    "technical_basis": "One-at-a-time sensitivity analysis with policy tracking",
                    "assumptions": ["Parameters vary independently"]
                }
            }
        }
    }
