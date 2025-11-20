"""
Phase 1 Pydantic models for advanced features.

Includes models for:
- Preference elicitation (ActiVA)
- Bayesian teaching
- Advanced model validation
- User belief representation
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .metadata import ResponseMetadata
from .shared import ConfidenceLevel, Distribution, ExplanationMetadata


# ============================================================================
# Preference Elicitation Models (ActiVA)
# ============================================================================


class DecisionContext(BaseModel):
    """Context for a decision domain."""

    domain: str = Field(
        ...,
        description="Decision domain (e.g., 'pricing', 'feature_prioritization')",
        min_length=1,
        max_length=100
    )
    variables: List[str] = Field(
        ...,
        description="Relevant decision variables",
        max_length=50
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Domain-specific constraints",
    )

    @field_validator("constraints")
    @classmethod
    def validate_constraints_size(cls, v):
        """Validate constraints dict size."""
        if v is not None:
            from src.utils.security_validators import validate_dict_size
            validate_dict_size(v, "constraints")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "domain": "pricing",
                "variables": ["price", "revenue", "churn", "brand_perception"],
                "constraints": {"industry": "SaaS", "current_price": 49},
            }
        }
    }


class UserBeliefModel(BaseModel):
    """Bayesian model of user preferences and beliefs."""

    value_weights: Dict[str, Distribution] = Field(
        ...,
        description="Probability distributions over value weights",
    )
    risk_tolerance: Distribution = Field(..., description="Risk aversion parameter distribution")
    time_horizon: Distribution = Field(
        ...,
        description="Short-term vs long-term focus distribution",
    )
    uncertainty_estimates: Dict[str, float] = Field(
        ...,
        description="Confidence in each belief (0-1)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "value_weights": {
                    "revenue": {"type": "normal", "parameters": {"mean": 0.6, "std": 0.2}},
                    "churn": {"type": "normal", "parameters": {"mean": 0.4, "std": 0.2}},
                },
                "risk_tolerance": {"type": "beta", "parameters": {"alpha": 2, "beta": 2}},
                "time_horizon": {"type": "normal", "parameters": {"mean": 12, "std": 3}},
                "uncertainty_estimates": {"revenue_weight": 0.3, "churn_weight": 0.4},
            }
        }
    }


class Scenario(BaseModel):
    """A decision scenario for comparison."""

    description: str = Field(
        ...,
        description="Natural language description",
        max_length=10000
    )
    outcomes: Dict[str, float] = Field(..., description="Predicted outcomes")
    trade_offs: List[str] = Field(
        ...,
        description="What's gained vs lost",
        max_length=20
    )

    @field_validator("outcomes")
    @classmethod
    def validate_outcomes_size(cls, v):
        """Validate outcomes dict size."""
        from src.utils.security_validators import validate_dict_size
        validate_dict_size(v, "outcomes")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "description": "Conservative growth, preserve relationships",
                "outcomes": {"revenue": 45000, "churn": 0.015, "brand": 0},
                "trade_offs": ["Lower revenue", "Better retention"],
            }
        }
    }


class QueryStrategy(str, Enum):
    """Strategy for query selection."""

    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    EXPECTED_IMPROVEMENT = "expected_improvement"
    EXPLORATION = "exploration"


class QueryStrategyInfo(BaseModel):
    """Information about the query selection strategy."""

    type: QueryStrategy = Field(..., description="Strategy type")
    rationale: str = Field(..., description="Why this strategy was chosen")
    focus_areas: List[str] = Field(..., description="Which preferences being targeted")


class CounterfactualQuery(BaseModel):
    """A counterfactual query for preference elicitation."""

    id: str = Field(
        ...,
        description="Unique query identifier",
        min_length=1,
        max_length=100
    )
    question: str = Field(
        ...,
        description="Natural language question",
        max_length=5000
    )
    scenario_a: Scenario = Field(..., description="First scenario option")
    scenario_b: Scenario = Field(..., description="Second scenario option")
    information_gain: float = Field(
        ...,
        description="Expected reduction in uncertainty",
        ge=0.0,
        le=1.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "query_001",
                "question": "Which outcome would you prefer?\n\nOption A: ...",
                "scenario_a": {
                    "description": "Conservative approach",
                    "outcomes": {"revenue": 45000},
                    "trade_offs": ["Lower risk"],
                },
                "scenario_b": {
                    "description": "Aggressive approach",
                    "outcomes": {"revenue": 55000},
                    "trade_offs": ["Higher risk"],
                },
                "information_gain": 0.34,
            }
        }
    }


class PreferenceElicitationRequest(BaseModel):
    """Request for preference elicitation."""

    user_id: str = Field(
        ...,
        description="User identifier",
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_\-]+$'
    )
    context: DecisionContext = Field(..., description="Decision context")
    current_beliefs: Optional[UserBeliefModel] = Field(
        default=None,
        description="Current user beliefs (None for first elicitation)",
    )
    num_queries: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of queries to generate",
    )


class PreferenceElicitationResponse(BaseModel):
    """Response from preference elicitation."""

    queries: List[CounterfactualQuery] = Field(..., description="Generated queries")
    strategy: QueryStrategyInfo = Field(..., description="Query selection strategy")
    expected_information_gain: float = Field(
        ...,
        description="Expected total information gain",
    )
    estimated_queries_remaining: int = Field(
        ...,
        description="Estimated queries needed after these",
    )
    explanation: ExplanationMetadata = Field(..., description="Explanation metadata")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )


class PreferenceChoice(str, Enum):
    """User's preference choice."""

    A = "A"
    B = "B"
    INDIFFERENT = "indifferent"


class PreferenceUpdateRequest(BaseModel):
    """Request to update user beliefs based on response."""

    user_id: str = Field(
        ...,
        description="User identifier",
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_\-]+$'
    )
    query_id: str = Field(
        ...,
        description="Query being responded to",
        min_length=1,
        max_length=100
    )
    response: PreferenceChoice = Field(..., description="User's choice")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="User's confidence in choice (0-1)",
    )


class LearningSummary(BaseModel):
    """Summary of what has been learned about user."""

    top_priorities: List[str] = Field(..., description="Inferred top priorities")
    confidence: float = Field(..., description="Overall confidence in beliefs (0-1)")
    insights: List[str] = Field(..., description="Key insights about user preferences")
    ready_for_recommendations: bool = Field(
        ...,
        description="Whether enough has been learned",
    )


class PreferenceUpdateResponse(BaseModel):
    """Response from belief update."""

    updated_beliefs: UserBeliefModel = Field(..., description="Updated belief model")
    queries_completed: int = Field(..., description="Total queries completed")
    estimated_queries_remaining: int = Field(..., description="Estimated queries still needed")
    next_queries: List[CounterfactualQuery] = Field(
        ...,
        description="Next batch of queries",
    )
    learning_summary: LearningSummary = Field(..., description="Learning progress summary")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )


# ============================================================================
# Bayesian Teaching Models
# ============================================================================


class TeachingExample(BaseModel):
    """A pedagogically valuable example."""

    scenario: Scenario = Field(..., description="Example scenario")
    key_insight: str = Field(..., description="What this teaches")
    why_this_example: str = Field(..., description="Why this is pedagogically valuable")
    information_value: float = Field(..., description="Teaching value score")


class BayesianTeachingRequest(BaseModel):
    """Request for teaching examples."""

    user_id: str = Field(
        ...,
        description="User identifier",
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_\-]+$'
    )
    current_beliefs: UserBeliefModel = Field(..., description="Current user beliefs")
    target_concept: str = Field(
        ...,
        description="Concept to teach (e.g., 'confounding', 'trade_offs')",
        min_length=1,
        max_length=100
    )
    context: DecisionContext = Field(..., description="Decision context")
    max_examples: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum examples to return",
    )


class BayesianTeachingResponse(BaseModel):
    """Response with teaching examples."""

    examples: List[TeachingExample] = Field(..., description="Selected teaching examples")
    explanation: str = Field(..., description="Overall teaching strategy explanation")
    learning_objectives: List[str] = Field(..., description="What user will learn")
    expected_learning_time: str = Field(..., description="Estimated time to learn concept")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )


# ============================================================================
# Advanced Model Validation Models
# ============================================================================


class ValidationLevel(str, Enum):
    """Level of validation to perform."""

    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class QualityLevel(str, Enum):
    """Overall model quality."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class ValidationCheck(BaseModel):
    """A single validation check result."""

    name: str = Field(..., description="Check name")
    status: ValidationStatus = Field(..., description="Check status")
    description: str = Field(..., description="Check result description")
    recommendation: Optional[str] = Field(
        default=None,
        description="Recommendation if check failed",
    )


class StructuralValidation(BaseModel):
    """Structural validation results."""

    checks: List[ValidationCheck] = Field(..., description="Structural checks performed")


class StatisticalValidation(BaseModel):
    """Statistical validation results."""

    checks: List[ValidationCheck] = Field(..., description="Statistical checks performed")


class DomainValidation(BaseModel):
    """Domain-specific validation results."""

    checks: List[ValidationCheck] = Field(..., description="Domain checks performed")


class ValidationResults(BaseModel):
    """All validation results."""

    structural: StructuralValidation = Field(..., description="Structural validation")
    statistical: StatisticalValidation = Field(..., description="Statistical validation")
    domain: DomainValidation = Field(..., description="Domain validation")


class SuggestionType(str, Enum):
    """Type of model suggestion."""

    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    ADD_NODE = "add_node"
    RESTRUCTURE = "restructure"


class ImpactLevel(str, Enum):
    """Impact level of suggestion."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ModelSuggestion(BaseModel):
    """Suggestion for model improvement."""

    type: SuggestionType = Field(..., description="Type of suggestion")
    description: str = Field(..., description="What to change")
    rationale: str = Field(..., description="Why this change would help")
    confidence: float = Field(..., description="Confidence in suggestion (0-1)")
    impact: ImpactLevel = Field(..., description="Expected impact")


class BestPracticeStatus(str, Enum):
    """Best practice adherence status."""

    FOLLOWED = "followed"
    PARTIAL = "partial"
    NOT_FOLLOWED = "not_followed"


class BestPractice(BaseModel):
    """Best practice check."""

    practice: str = Field(..., description="Best practice name")
    status: BestPracticeStatus = Field(..., description="Adherence status")
    description: str = Field(..., description="Status description")


class AdvancedValidationRequest(BaseModel):
    """Request for advanced model validation."""

    dag: Any = Field(..., description="DAG structure")  # Using Any to avoid circular import
    structural_model: Optional[Any] = Field(
        default=None,
        description="Optional structural model",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context (e.g., domain)",
    )
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.STANDARD,
        description="Level of validation to perform",
    )


class AdvancedValidationResponse(BaseModel):
    """Response from advanced validation."""

    overall_quality: QualityLevel = Field(..., description="Overall quality assessment")
    quality_score: float = Field(..., description="Numerical quality score (0-100)")
    validation_results: ValidationResults = Field(..., description="Detailed validation results")
    suggestions: List[ModelSuggestion] = Field(..., description="Improvement suggestions")
    best_practices: List[BestPractice] = Field(..., description="Best practice adherence")
    explanation: ExplanationMetadata = Field(..., description="Explanation metadata")

    # Metadata for determinism and reproducibility
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Metadata for determinism verification",
        alias="_metadata"
    )
