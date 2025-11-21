"""
Unit tests for ConsensusGenerator service.

Tests generation and refinement of consensus statements.
"""

import pytest

from src.models.deliberation import (
    MemberPosition,
    ValueStatement,
    ConcernStatement,
    HabermasCommonGround,
    SharedValue,
    ConsensusStatement,
    EditSuggestion,
)
from src.services.consensus_generator import ConsensusGenerator


class TestConsensusGenerator:
    """Test consensus statement generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ConsensusGenerator()

    def test_generate_new_statement_basic(self):
        """Test generating new consensus statement."""
        common_ground = HabermasCommonGround(
            shared_values=[
                SharedValue(
                    value_name="quality",
                    agreement_score=0.9,
                    supporting_members=["alice", "bob"],
                    average_weight=0.8,
                    synthesized_rationale="We value quality",
                )
            ],
            shared_concerns=[],
            agreement_level=0.75,
            participants=["alice", "bob"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=None,
            edit_suggestions=None,
            decision_context="Feature decision",
            request_id="test_001",
        )

        assert consensus.version == 1
        assert len(consensus.text) > 0
        assert consensus.support_score == 0.75  # Should match agreement level
        assert "quality" in consensus.incorporated_values

    def test_statement_includes_context(self):
        """Test that generated statement includes decision context."""
        common_ground = HabermasCommonGround(
            shared_values=[
                SharedValue(
                    value_name="speed",
                    agreement_score=0.8,
                    supporting_members=["alice"],
                    average_weight=0.7,
                    synthesized_rationale="Speed matters",
                )
            ],
            shared_concerns=[],
            agreement_level=0.7,
            participants=["alice"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00")
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=None,
            edit_suggestions=None,
            decision_context="API Architecture",
            request_id="test_002",
        )

        # Statement should mention the context
        assert "API Architecture" in consensus.text

    def test_statement_includes_shared_values(self):
        """Test that statement incorporates shared values."""
        common_ground = HabermasCommonGround(
            shared_values=[
                SharedValue(
                    value_name="quality",
                    agreement_score=0.9,
                    supporting_members=["alice", "bob"],
                    average_weight=0.8,
                    synthesized_rationale="Quality is key",
                ),
                SharedValue(
                    value_name="speed",
                    agreement_score=0.8,
                    supporting_members=["alice", "bob"],
                    average_weight=0.7,
                    synthesized_rationale="Speed matters",
                ),
            ],
            shared_concerns=[],
            agreement_level=0.8,
            participants=["alice", "bob"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=None,
            edit_suggestions=None,
            decision_context="Decision",
            request_id="test_003",
        )

        # Should incorporate both values
        assert "quality" in consensus.incorporated_values
        assert "speed" in consensus.incorporated_values

    def test_statement_includes_shared_concerns(self):
        """Test that statement addresses shared concerns."""
        common_ground = HabermasCommonGround(
            shared_values=[],
            shared_concerns=[
                SharedValue(  # Reusing SharedValue for concerns
                    value_name="technical_risk",
                    agreement_score=0.8,
                    supporting_members=["alice", "bob"],
                    average_weight=0.7,
                    synthesized_rationale="Worried about scalability",
                )
            ],
            agreement_level=0.7,
            participants=["alice", "bob"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=None,
            edit_suggestions=None,
            decision_context="Decision",
            request_id="test_004",
        )

        # Should incorporate concerns
        assert "technical_risk" in consensus.incorporated_concerns

    def test_refine_existing_statement(self):
        """Test refining existing consensus with edits."""
        previous = ConsensusStatement(
            statement_id="consensus_001",
            version=1,
            text="We agree quality is important.",
            support_score=0.7,
            generated_at="2025-01-01T00:00:00",
        )

        edits = [
            EditSuggestion(
                member_id="alice",
                edit_type="addition",
                suggestion="Also prioritize speed",
                rationale="Speed is critical too",
                priority=0.8,
            )
        ]

        common_ground = HabermasCommonGround(
            shared_values=[
                SharedValue(
                    value_name="quality",
                    agreement_score=0.8,
                    supporting_members=["alice", "bob"],
                    average_weight=0.7,
                    synthesized_rationale="Quality matters",
                )
            ],
            shared_concerns=[],
            agreement_level=0.8,
            participants=["alice", "bob"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=previous,
            edit_suggestions=edits,
            decision_context="Decision",
            request_id="test_005",
        )

        # Version should increment
        assert consensus.version == 2

        # Statement ID should be preserved
        assert consensus.statement_id == "consensus_001"

        # Should incorporate edit suggestion
        assert "speed" in consensus.text.lower()

    def test_refinement_adds_version_note(self):
        """Test that refined statements include version note."""
        previous = ConsensusStatement(
            statement_id="consensus_001",
            version=1,
            text="Original statement",
            support_score=0.7,
            generated_at="2025-01-01T00:00:00",
        )

        edits = [
            EditSuggestion(
                member_id="alice",
                edit_type="clarification",
                suggestion="Clarify the scope",
                rationale="Needs clarity",
                priority=0.7,
            )
        ]

        common_ground = HabermasCommonGround(
            shared_values=[],
            shared_concerns=[],
            agreement_level=0.7,
            participants=["alice"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00")
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=previous,
            edit_suggestions=edits,
            decision_context="Decision",
            request_id="test_006",
        )

        # Should include version note
        assert "v2" in consensus.text

    def test_identify_unresolved_disagreements(self):
        """Test identifying unresolved disagreements."""
        common_ground = HabermasCommonGround(
            shared_values=[
                SharedValue(
                    value_name="quality",
                    agreement_score=0.5,  # Only 50% agree
                    supporting_members=["alice"],
                    average_weight=0.8,
                    synthesized_rationale="Quality matters",
                )
            ],
            shared_concerns=[],
            agreement_level=0.5,
            participants=["alice", "bob"],
        )

        positions = [
            MemberPosition(
                member_id="alice",
                values=[
                    ValueStatement(value_name="quality", weight=0.8, rationale="test"),
                    ValueStatement(value_name="speed", weight=0.9, rationale="test"),
                ],
                timestamp="2025-01-01T00:00:00",
            ),
            MemberPosition(
                member_id="bob",
                values=[
                    ValueStatement(value_name="cost", weight=0.9, rationale="test"),
                    ValueStatement(value_name="speed", weight=0.8, rationale="test"),
                ],
                timestamp="2025-01-01T00:00:00",
            ),
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=None,
            edit_suggestions=None,
            decision_context="Decision",
            request_id="test_007",
        )

        # Should identify unresolved disagreements
        # (values held by 2+ people but not shared)
        # Speed should be unresolved (both care but not in shared_values)
        assert len(consensus.unresolved_disagreements) > 0

    def test_supporting_members_list(self):
        """Test that supporting members list is correct."""
        common_ground = HabermasCommonGround(
            shared_values=[
                SharedValue(
                    value_name="quality",
                    agreement_score=0.9,
                    supporting_members=["alice", "bob", "charlie"],
                    average_weight=0.8,
                    synthesized_rationale="Quality",
                )
            ],
            shared_concerns=[],
            agreement_level=0.9,
            participants=["alice", "bob", "charlie"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="charlie", timestamp="2025-01-01T00:00:00"),
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=None,
            edit_suggestions=None,
            decision_context="Decision",
            request_id="test_008",
        )

        # All participants should be supporting
        assert len(consensus.supporting_members) == 3

    def test_statement_id_generated_if_new(self):
        """Test that statement ID is generated for new statements."""
        common_ground = HabermasCommonGround(
            shared_values=[],
            shared_concerns=[],
            agreement_level=0.5,
            participants=["alice"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00")
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=None,
            edit_suggestions=None,
            decision_context="Decision",
            request_id="test_009",
        )

        # Should generate statement ID
        assert consensus.statement_id is not None
        assert len(consensus.statement_id) > 0
        assert "consensus_" in consensus.statement_id

    def test_generated_timestamp_present(self):
        """Test that generated_at timestamp is present."""
        common_ground = HabermasCommonGround(
            shared_values=[],
            shared_concerns=[],
            agreement_level=0.5,
            participants=["alice"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00")
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=None,
            edit_suggestions=None,
            decision_context="Decision",
            request_id="test_010",
        )

        assert consensus.generated_at is not None
        assert len(consensus.generated_at) > 0

    def test_multiple_edits_incorporated(self):
        """Test that multiple edit suggestions are incorporated."""
        previous = ConsensusStatement(
            statement_id="consensus_001",
            version=1,
            text="Original statement",
            support_score=0.7,
            generated_at="2025-01-01T00:00:00",
        )

        edits = [
            EditSuggestion(
                member_id="alice",
                edit_type="addition",
                suggestion="Add point A",
                rationale="Important",
                priority=0.9,
            ),
            EditSuggestion(
                member_id="bob",
                edit_type="addition",
                suggestion="Add point B",
                rationale="Also important",
                priority=0.8,
            ),
            EditSuggestion(
                member_id="charlie",
                edit_type="clarification",
                suggestion="Clarify point C",
                rationale="Needs clarity",
                priority=0.7,
            ),
        ]

        common_ground = HabermasCommonGround(
            shared_values=[],
            shared_concerns=[],
            agreement_level=0.7,
            participants=["alice", "bob", "charlie"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="charlie", timestamp="2025-01-01T00:00:00"),
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=previous,
            edit_suggestions=edits,
            decision_context="Decision",
            request_id="test_011",
        )

        # Should incorporate high-priority edits
        text_lower = consensus.text.lower()
        assert "point a" in text_lower or "additional points" in text_lower

    def test_edits_sorted_by_priority(self):
        """Test that edits are processed by priority."""
        previous = ConsensusStatement(
            statement_id="consensus_001",
            version=1,
            text="Original",
            support_score=0.7,
            generated_at="2025-01-01T00:00:00",
        )

        # Create edits with different priorities
        edits = [
            EditSuggestion(
                member_id="alice",
                edit_type="addition",
                suggestion="Low priority edit",
                rationale="test",
                priority=0.3,
            ),
            EditSuggestion(
                member_id="bob",
                edit_type="addition",
                suggestion="High priority edit",
                rationale="test",
                priority=0.9,
            ),
        ]

        common_ground = HabermasCommonGround(
            shared_values=[],
            shared_concerns=[],
            agreement_level=0.7,
            participants=["alice", "bob"],
        )

        positions = [
            MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
            MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
        ]

        consensus = self.generator.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=previous,
            edit_suggestions=edits,
            decision_context="Decision",
            request_id="test_012",
        )

        # High priority edit should be included
        assert "high priority" in consensus.text.lower()
