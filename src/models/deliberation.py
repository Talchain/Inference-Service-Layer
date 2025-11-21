"""
Team deliberation models for Habermas Machine.

Represents individual values, common ground, consensus statements,
and deliberation state for democratic team decision-making.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .metadata import ResponseMetadata


class ValueStatement(BaseModel):
    """
    Individual team member's value statement.

    Captures what matters to them and why.
    """

    value_name: str = Field(..., description="Value identifier (e.g., 'user_satisfaction')")
    weight: float = Field(..., ge=0.0, le=1.0, description="Priority weight (0-1)")
    rationale: str = Field(..., description="Why this value matters to this person")
    examples: List[str] = Field(
        default_factory=list,
        description="Concrete examples of this value",
    )


class ConcernStatement(BaseModel):
    """
    Individual concern about a decision/option.

    Captures risks, worries, or objections.
    """

    concern_name: str = Field(..., description="Concern identifier")
    severity: float = Field(..., ge=0.0, le=1.0, description="How serious (0-1)")
    explanation: str = Field(..., description="What's the concern and why")
    conditions: Optional[str] = Field(
        None,
        description="Under what conditions would this be addressed?",
    )


class MemberPosition(BaseModel):
    """
    Complete position of one team member.

    Includes values, concerns, and reasoning.
    """

    member_id: str = Field(..., description="Team member identifier")
    member_name: Optional[str] = Field(None, description="Display name")
    role: Optional[str] = Field(None, description="Role (PM, Engineer, Designer, etc.)")

    # What they value
    values: List[ValueStatement] = Field(
        default_factory=list,
        description="What matters to this member",
    )

    # What they're concerned about
    concerns: List[ConcernStatement] = Field(
        default_factory=list,
        description="Risks or objections",
    )

    # Preferred option (if choosing between options)
    preferred_option: Optional[str] = Field(
        None,
        description="Which option they prefer (if applicable)",
    )

    # Open-ended position statement
    position_statement: Optional[str] = Field(
        None,
        description="Free-form explanation of their position",
    )

    timestamp: str = Field(..., description="When position was stated")


class SharedValue(BaseModel):
    """
    Value that multiple team members share.

    Part of common ground.
    """

    value_name: str = Field(..., description="Shared value identifier")
    agreement_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Degree of agreement (0-1)",
    )
    supporting_members: List[str] = Field(
        ...,
        description="Member IDs who share this value",
    )
    average_weight: float = Field(..., description="Average priority weight among supporters")
    synthesized_rationale: str = Field(
        ...,
        description="Combined explanation of why this matters",
    )


class HabermasCommonGround(BaseModel):
    """
    Shared values and concerns across team.

    Foundation for consensus building.
    """

    shared_values: List[SharedValue] = Field(
        default_factory=list,
        description="Values team agrees on",
    )

    shared_concerns: List[SharedValue] = Field(
        default_factory=list,
        description="Concerns team shares",
    )

    agreement_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall agreement score",
    )

    participants: List[str] = Field(..., description="Member IDs included in common ground")


class ConsensusStatement(BaseModel):
    """
    AI-generated consensus statement.

    Reflects common ground and can be iteratively refined.
    """

    statement_id: str = Field(..., description="Unique statement identifier")
    version: int = Field(1, description="Version number (increments with edits)")

    # The statement
    text: str = Field(..., description="The consensus statement itself")

    # What it captures
    incorporated_values: List[str] = Field(
        default_factory=list,
        description="Which values are reflected",
    )
    incorporated_concerns: List[str] = Field(
        default_factory=list,
        description="Which concerns are addressed",
    )

    # Support levels
    supporting_members: List[str] = Field(
        default_factory=list,
        description="Members who support this statement",
    )
    support_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of team supporting (0-1)",
    )

    # Remaining issues
    unresolved_disagreements: List[str] = Field(
        default_factory=list,
        description="Issues not yet resolved",
    )

    # Metadata
    generated_at: str = Field(..., description="When statement was generated")
    generated_by: str = Field("habermas_machine", description="Generation method")


class EditSuggestion(BaseModel):
    """
    Team member's suggested edit to consensus statement.

    Part of iterative refinement.
    """

    member_id: str = Field(..., description="Who suggested this edit")
    edit_type: str = Field(
        ...,
        description="Type: 'addition', 'deletion', 'clarification', 'reframing'",
    )
    suggestion: str = Field(..., description="The specific edit suggestion")
    rationale: str = Field(..., description="Why this edit improves the statement")
    priority: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How important is this edit (0-1)",
    )


class DeliberationRound(BaseModel):
    """
    Single round of deliberation.

    Contains positions, common ground found, consensus generated.
    """

    round_number: int = Field(..., description="Deliberation round (1, 2, 3, ...)")

    # Inputs
    positions: List[MemberPosition] = Field(
        default_factory=list,
        description="Member positions at start of round",
    )

    # Analysis
    common_ground: Optional[HabermasCommonGround] = Field(
        None,
        description="Common ground identified",
    )

    # Output
    consensus_statement: Optional[ConsensusStatement] = Field(
        None,
        description="Generated consensus statement",
    )

    # Feedback
    edit_suggestions: List[EditSuggestion] = Field(
        default_factory=list,
        description="Edits suggested by team",
    )

    # Metrics
    agreement_level: float = Field(..., description="Overall agreement level (0-1)")
    participation_rate: float = Field(
        ...,
        description="Fraction of team participating (0-1)",
    )

    timestamp: str = Field(..., description="When round occurred")


class DeliberationSession(BaseModel):
    """
    Complete deliberation session.

    Tracks entire journey from initial positions to final consensus.
    """

    session_id: str = Field(..., description="Unique session identifier")
    decision_context: str = Field(..., description="What decision is being made")

    # Participants
    team_members: List[Dict[str, str]] = Field(
        ...,
        description="Team member metadata (id, name, role)",
    )

    # Deliberation history
    rounds: List[DeliberationRound] = Field(
        default_factory=list,
        description="All deliberation rounds",
    )

    # Current state
    current_consensus: Optional[ConsensusStatement] = Field(
        None,
        description="Latest consensus statement",
    )

    # Status
    status: str = Field("active", description="active, converged, diverged, abandoned")
    convergence_criteria: Dict[str, float] = Field(
        default_factory=dict,
        description="Thresholds for convergence (support_threshold, etc.)",
    )

    # Metrics
    total_rounds: int = Field(0, description="Number of rounds so far")
    final_agreement_level: Optional[float] = Field(
        None,
        description="Final agreement level if converged",
    )

    # Timestamps
    started_at: str = Field(..., description="Session start time")
    updated_at: str = Field(..., description="Last update time")
    converged_at: Optional[str] = Field(None, description="Convergence time")


class DeliberationRequest(BaseModel):
    """Request to initiate or continue deliberation."""

    session_id: Optional[str] = Field(
        None,
        description="Existing session ID (None for new session)",
    )
    decision_context: str = Field(..., description="What's being decided")

    # Team members and their positions
    positions: List[MemberPosition] = Field(
        ...,
        description="Current positions from team members",
    )

    # Previous consensus (if continuing)
    previous_consensus: Optional[ConsensusStatement] = Field(
        None,
        description="Previous consensus being refined",
    )

    # Edit suggestions (if continuing)
    edit_suggestions: Optional[List[EditSuggestion]] = Field(
        None,
        description="Edits suggested by team",
    )

    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Deliberation configuration (thresholds, etc.)",
    )


class DeliberationResponse(BaseModel):
    """Response from deliberation round."""

    session_id: str = Field(..., description="Session identifier")
    round_number: int = Field(..., description="Current round number")

    # Analysis results
    common_ground: HabermasCommonGround = Field(..., description="Identified common ground")

    # Generated consensus
    consensus_statement: ConsensusStatement = Field(
        ...,
        description="New or revised consensus statement",
    )

    # Status
    status: str = Field(..., description="active, converged, needs_more_input")
    convergence_assessment: Dict[str, Any] = Field(
        ...,
        description="Progress toward convergence",
    )

    # Next steps
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next actions",
    )

    # Metadata
    metadata: Optional[ResponseMetadata] = Field(
        default=None,
        description="Response metadata",
        alias="_metadata",
    )

    model_config = {
        "populate_by_name": True,  # Allow using both 'metadata' and '_metadata'
        "json_schema_extra": {
            "example": {
                "session_id": "delib_abc123",
                "round_number": 1,
                "common_ground": {
                    "shared_values": [
                        {
                            "value_name": "user_satisfaction",
                            "agreement_score": 0.9,
                            "supporting_members": ["alice", "bob", "charlie"],
                            "average_weight": 0.8,
                            "synthesized_rationale": "We all prioritize user satisfaction",
                        }
                    ],
                    "agreement_level": 0.75,
                },
                "consensus_statement": {
                    "statement_id": "consensus_001",
                    "version": 1,
                    "text": "We agree that user satisfaction is our top priority...",
                    "support_score": 0.9,
                },
                "status": "active",
            }
        }
    }
