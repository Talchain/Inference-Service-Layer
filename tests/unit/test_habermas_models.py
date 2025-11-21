"""
Unit tests for Habermas Machine deliberation models.

Tests all Pydantic models and validation logic for team deliberation.
"""

import pytest
from datetime import datetime

from src.models.deliberation import (
    ValueStatement,
    ConcernStatement,
    MemberPosition,
    SharedValue,
    HabermasCommonGround,
    ConsensusStatement,
    EditSuggestion,
    DeliberationRound,
    DeliberationSession,
    DeliberationRequest,
    DeliberationResponse,
)
from src.models.metadata import ResponseMetadata


class TestValueStatement:
    """Test ValueStatement model."""

    def test_valid_value_statement(self):
        """Test creating valid value statement."""
        value = ValueStatement(
            value_name="user_satisfaction",
            weight=0.8,
            rationale="Users should be happy",
            examples=["Fast response times", "Intuitive UI"],
        )

        assert value.value_name == "user_satisfaction"
        assert value.weight == 0.8
        assert value.rationale == "Users should be happy"
        assert len(value.examples) == 2

    def test_weight_validation_in_range(self):
        """Test weight must be between 0 and 1."""
        value = ValueStatement(
            value_name="test",
            weight=0.5,
            rationale="test",
        )
        assert 0.0 <= value.weight <= 1.0

    def test_weight_validation_too_high(self):
        """Test weight cannot exceed 1.0."""
        with pytest.raises(ValueError):
            ValueStatement(
                value_name="test",
                weight=1.5,
                rationale="test",
            )

    def test_weight_validation_negative(self):
        """Test weight cannot be negative."""
        with pytest.raises(ValueError):
            ValueStatement(
                value_name="test",
                weight=-0.1,
                rationale="test",
            )

    def test_empty_examples_default(self):
        """Test examples defaults to empty list."""
        value = ValueStatement(
            value_name="test",
            weight=0.5,
            rationale="test",
        )
        assert value.examples == []


class TestConcernStatement:
    """Test ConcernStatement model."""

    def test_valid_concern_statement(self):
        """Test creating valid concern statement."""
        concern = ConcernStatement(
            concern_name="technical_risk",
            severity=0.7,
            explanation="System might not scale",
            conditions="Add load testing",
        )

        assert concern.concern_name == "technical_risk"
        assert concern.severity == 0.7
        assert concern.explanation == "System might not scale"
        assert concern.conditions == "Add load testing"

    def test_severity_validation(self):
        """Test severity must be between 0 and 1."""
        # Valid
        concern = ConcernStatement(
            concern_name="test",
            severity=0.5,
            explanation="test",
        )
        assert 0.0 <= concern.severity <= 1.0

    def test_optional_conditions(self):
        """Test conditions is optional."""
        concern = ConcernStatement(
            concern_name="test",
            severity=0.5,
            explanation="test",
        )
        assert concern.conditions is None


class TestMemberPosition:
    """Test MemberPosition model."""

    def test_complete_member_position(self):
        """Test creating complete member position."""
        position = MemberPosition(
            member_id="alice",
            member_name="Alice Smith",
            role="PM",
            values=[
                ValueStatement(
                    value_name="user_satisfaction",
                    weight=0.9,
                    rationale="Users first",
                )
            ],
            concerns=[
                ConcernStatement(
                    concern_name="schedule_risk",
                    severity=0.6,
                    explanation="Tight deadline",
                )
            ],
            preferred_option="Option A",
            position_statement="I prefer Option A because...",
            timestamp="2025-01-01T00:00:00",
        )

        assert position.member_id == "alice"
        assert position.member_name == "Alice Smith"
        assert position.role == "PM"
        assert len(position.values) == 1
        assert len(position.concerns) == 1
        assert position.preferred_option == "Option A"

    def test_minimal_member_position(self):
        """Test member position with only required fields."""
        position = MemberPosition(
            member_id="bob",
            timestamp="2025-01-01T00:00:00",
        )

        assert position.member_id == "bob"
        assert position.values == []
        assert position.concerns == []
        assert position.member_name is None
        assert position.role is None


class TestSharedValue:
    """Test SharedValue model."""

    def test_valid_shared_value(self):
        """Test creating valid shared value."""
        shared = SharedValue(
            value_name="quality",
            agreement_score=0.85,
            supporting_members=["alice", "bob", "charlie"],
            average_weight=0.75,
            synthesized_rationale="We all value quality",
        )

        assert shared.value_name == "quality"
        assert shared.agreement_score == 0.85
        assert len(shared.supporting_members) == 3
        assert shared.average_weight == 0.75

    def test_agreement_score_validation(self):
        """Test agreement score must be between 0 and 1."""
        shared = SharedValue(
            value_name="test",
            agreement_score=0.5,
            supporting_members=["alice"],
            average_weight=0.5,
            synthesized_rationale="test",
        )
        assert 0.0 <= shared.agreement_score <= 1.0


class TestHabermasCommonGround:
    """Test HabermasCommonGround model."""

    def test_valid_common_ground(self):
        """Test creating valid common ground."""
        common_ground = HabermasCommonGround(
            shared_values=[
                SharedValue(
                    value_name="quality",
                    agreement_score=0.9,
                    supporting_members=["alice", "bob"],
                    average_weight=0.8,
                    synthesized_rationale="Quality matters",
                )
            ],
            shared_concerns=[],
            agreement_level=0.75,
            participants=["alice", "bob", "charlie"],
        )

        assert len(common_ground.shared_values) == 1
        assert len(common_ground.shared_concerns) == 0
        assert common_ground.agreement_level == 0.75
        assert len(common_ground.participants) == 3

    def test_empty_common_ground(self):
        """Test common ground with no shared items."""
        common_ground = HabermasCommonGround(
            shared_values=[],
            shared_concerns=[],
            agreement_level=0.0,
            participants=[],
        )

        assert common_ground.agreement_level == 0.0


class TestConsensusStatement:
    """Test ConsensusStatement model."""

    def test_valid_consensus_statement(self):
        """Test creating valid consensus statement."""
        consensus = ConsensusStatement(
            statement_id="consensus_001",
            version=1,
            text="We agree to prioritize user satisfaction",
            incorporated_values=["user_satisfaction", "quality"],
            incorporated_concerns=["schedule_risk"],
            supporting_members=["alice", "bob"],
            support_score=0.9,
            unresolved_disagreements=[],
            generated_at="2025-01-01T00:00:00",
        )

        assert consensus.statement_id == "consensus_001"
        assert consensus.version == 1
        assert len(consensus.incorporated_values) == 2
        assert consensus.support_score == 0.9

    def test_default_version(self):
        """Test version defaults to 1."""
        consensus = ConsensusStatement(
            statement_id="test",
            text="test",
            support_score=0.5,
            generated_at="2025-01-01T00:00:00",
        )
        assert consensus.version == 1

    def test_support_score_validation(self):
        """Test support score must be between 0 and 1."""
        consensus = ConsensusStatement(
            statement_id="test",
            text="test",
            support_score=0.5,
            generated_at="2025-01-01T00:00:00",
        )
        assert 0.0 <= consensus.support_score <= 1.0


class TestEditSuggestion:
    """Test EditSuggestion model."""

    def test_valid_edit_suggestion(self):
        """Test creating valid edit suggestion."""
        edit = EditSuggestion(
            member_id="alice",
            edit_type="addition",
            suggestion="Add focus on performance",
            rationale="Performance is critical",
            priority=0.8,
        )

        assert edit.member_id == "alice"
        assert edit.edit_type == "addition"
        assert edit.priority == 0.8

    def test_priority_validation(self):
        """Test priority must be between 0 and 1."""
        edit = EditSuggestion(
            member_id="test",
            edit_type="clarification",
            suggestion="test",
            rationale="test",
            priority=0.5,
        )
        assert 0.0 <= edit.priority <= 1.0


class TestDeliberationRound:
    """Test DeliberationRound model."""

    def test_valid_deliberation_round(self):
        """Test creating valid deliberation round."""
        round_obj = DeliberationRound(
            round_number=1,
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
            common_ground=HabermasCommonGround(
                shared_values=[],
                shared_concerns=[],
                agreement_level=0.5,
                participants=["alice"],
            ),
            consensus_statement=ConsensusStatement(
                statement_id="test",
                text="test",
                support_score=0.5,
                generated_at="2025-01-01T00:00:00",
            ),
            edit_suggestions=[],
            agreement_level=0.5,
            participation_rate=1.0,
            timestamp="2025-01-01T00:00:00",
        )

        assert round_obj.round_number == 1
        assert len(round_obj.positions) == 1
        assert round_obj.participation_rate == 1.0


class TestDeliberationSession:
    """Test DeliberationSession model."""

    def test_valid_session(self):
        """Test creating valid deliberation session."""
        session = DeliberationSession(
            session_id="delib_abc123",
            decision_context="Choose new feature approach",
            team_members=[
                {"id": "alice", "name": "Alice", "role": "PM"},
                {"id": "bob", "name": "Bob", "role": "Engineer"},
            ],
            rounds=[],
            status="active",
            convergence_criteria={
                "support_threshold": 0.8,
                "agreement_threshold": 0.7,
            },
            total_rounds=0,
            started_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )

        assert session.session_id == "delib_abc123"
        assert len(session.team_members) == 2
        assert session.status == "active"
        assert session.total_rounds == 0

    def test_status_default(self):
        """Test status defaults to 'active'."""
        session = DeliberationSession(
            session_id="test",
            decision_context="test",
            team_members=[],
            started_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )
        assert session.status == "active"


class TestDeliberationRequest:
    """Test DeliberationRequest model."""

    def test_valid_request(self):
        """Test creating valid deliberation request."""
        request = DeliberationRequest(
            session_id="delib_123",
            decision_context="Choose feature X or Y",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        assert request.session_id == "delib_123"
        assert request.decision_context == "Choose feature X or Y"
        assert len(request.positions) == 1

    def test_new_session_request(self):
        """Test request for new session (no session_id)."""
        request = DeliberationRequest(
            decision_context="New decision",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        assert request.session_id is None


class TestDeliberationResponse:
    """Test DeliberationResponse model."""

    def test_valid_response(self):
        """Test creating valid deliberation response."""
        response = DeliberationResponse(
            session_id="delib_123",
            round_number=1,
            common_ground=HabermasCommonGround(
                shared_values=[],
                shared_concerns=[],
                agreement_level=0.5,
                participants=["alice"],
            ),
            consensus_statement=ConsensusStatement(
                statement_id="test",
                text="test",
                support_score=0.5,
                generated_at="2025-01-01T00:00:00",
            ),
            status="active",
            convergence_assessment={
                "support_met": False,
                "agreement_met": False,
            },
            next_steps=["Review consensus", "Submit feedback"],
        )

        assert response.session_id == "delib_123"
        assert response.round_number == 1
        assert response.status == "active"
        assert len(response.next_steps) == 2
