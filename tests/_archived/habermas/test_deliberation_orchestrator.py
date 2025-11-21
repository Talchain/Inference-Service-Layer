"""
Unit tests for DeliberationOrchestrator service.

Tests orchestration of multi-round deliberation sessions.
"""

import pytest

from src.models.deliberation import (
    DeliberationRequest,
    MemberPosition,
    ValueStatement,
    ConsensusStatement,
    EditSuggestion,
)
from src.services.deliberation_orchestrator import DeliberationOrchestrator


class TestDeliberationOrchestrator:
    """Test deliberation orchestration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = DeliberationOrchestrator()

    def test_conduct_first_round(self):
        """Test conducting first round of deliberation."""
        request = DeliberationRequest(
            decision_context="Choose API design",
            positions=[
                MemberPosition(
                    member_id="alice",
                    values=[
                        ValueStatement(
                            value_name="simplicity",
                            weight=0.8,
                            rationale="Simple is better",
                        )
                    ],
                    timestamp="2025-01-01T00:00:00",
                ),
                MemberPosition(
                    member_id="bob",
                    values=[
                        ValueStatement(
                            value_name="simplicity",
                            weight=0.7,
                            rationale="Keep it simple",
                        )
                    ],
                    timestamp="2025-01-01T00:00:00",
                ),
            ],
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_001",
        )

        # Check response structure
        assert response.round_number == 1
        assert response.session_id is not None
        assert response.common_ground is not None
        assert response.consensus_statement is not None
        assert response.status in ["active", "converged"]

    def test_session_creation(self):
        """Test that new session is created for first round."""
        request = DeliberationRequest(
            decision_context="Test decision",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_002",
        )

        # Session should be created and stored
        session = self.orchestrator.get_session(response.session_id)
        assert session is not None
        assert session.session_id == response.session_id
        assert session.decision_context == "Test decision"

    def test_session_continuation(self):
        """Test continuing existing session."""
        # First round
        request1 = DeliberationRequest(
            decision_context="Test decision",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response1 = self.orchestrator.conduct_deliberation_round(
            request=request1,
            request_id="test_003a",
        )

        session_id = response1.session_id

        # Second round with same session
        request2 = DeliberationRequest(
            session_id=session_id,
            decision_context="Test decision",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response2 = self.orchestrator.conduct_deliberation_round(
            request=request2,
            request_id="test_003b",
        )

        # Should continue same session
        assert response2.session_id == session_id
        assert response2.round_number == 2

    def test_convergence_detection_high_support(self):
        """Test convergence when support is high."""
        # Create positions with high alignment
        request = DeliberationRequest(
            decision_context="Test decision",
            positions=[
                MemberPosition(
                    member_id="alice",
                    values=[
                        ValueStatement(
                            value_name="quality",
                            weight=0.9,
                            rationale="Quality first",
                        )
                    ],
                    timestamp="2025-01-01T00:00:00",
                ),
                MemberPosition(
                    member_id="bob",
                    values=[
                        ValueStatement(
                            value_name="quality",
                            weight=0.9,
                            rationale="Quality is key",
                        )
                    ],
                    timestamp="2025-01-01T00:00:00",
                ),
            ],
            config={
                "convergence_criteria": {
                    "support_threshold": 0.7,  # Lower threshold for testing
                    "agreement_threshold": 0.6,
                    "max_rounds": 10,
                }
            },
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_004",
        )

        # Check convergence assessment
        assessment = response.convergence_assessment
        assert "support_score" in assessment
        assert "agreement_level" in assessment

    def test_max_rounds_convergence(self):
        """Test convergence when max rounds reached."""
        # Create request with max_rounds=1
        request = DeliberationRequest(
            decision_context="Test decision",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
            config={
                "convergence_criteria": {
                    "support_threshold": 0.99,  # Impossible to reach
                    "agreement_threshold": 0.99,
                    "max_rounds": 1,
                }
            },
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_005",
        )

        # Should converge due to max rounds
        assert response.convergence_assessment["max_rounds_reached"] is True

    def test_participation_rate_calculation(self):
        """Test participation rate calculation."""
        # First round with 2 members
        request1 = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00"),
                MemberPosition(member_id="bob", timestamp="2025-01-01T00:00:00"),
            ],
        )

        response1 = self.orchestrator.conduct_deliberation_round(
            request=request1,
            request_id="test_006a",
        )

        session_id = response1.session_id

        # Second round with only 1 member (50% participation)
        request2 = DeliberationRequest(
            session_id=session_id,
            decision_context="Test",
            positions=[
                MemberPosition(member_id="alice", timestamp="2025-01-01T00:00:00")
            ],
        )

        response2 = self.orchestrator.conduct_deliberation_round(
            request=request2,
            request_id="test_006b",
        )

        # Get session to check rounds
        session = self.orchestrator.get_session(session_id)
        assert session.rounds[1].participation_rate == 0.5  # 1/2 = 50%

    def test_next_steps_generated(self):
        """Test that next steps are generated."""
        request = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_007",
        )

        # Should have next steps
        assert len(response.next_steps) > 0

    def test_next_steps_converged(self):
        """Test next steps when converged."""
        request = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    values=[
                        ValueStatement(
                            value_name="quality",
                            weight=0.9,
                            rationale="test",
                        )
                    ],
                    timestamp="2025-01-01T00:00:00",
                )
            ],
            config={
                "convergence_criteria": {
                    "support_threshold": 0.5,
                    "agreement_threshold": 0.5,
                    "max_rounds": 1,
                }
            },
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_008",
        )

        # If converged, next steps should indicate completion
        if response.status == "converged":
            next_steps_text = " ".join(response.next_steps).lower()
            assert "complete" in next_steps_text or "ready" in next_steps_text

    def test_edit_suggestions_processed(self):
        """Test that edit suggestions are processed in refinement."""
        # First round
        request1 = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response1 = self.orchestrator.conduct_deliberation_round(
            request=request1,
            request_id="test_009a",
        )

        # Second round with edits
        request2 = DeliberationRequest(
            session_id=response1.session_id,
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
            previous_consensus=response1.consensus_statement,
            edit_suggestions=[
                EditSuggestion(
                    member_id="alice",
                    edit_type="addition",
                    suggestion="Add more detail",
                    rationale="Needs clarity",
                    priority=0.8,
                )
            ],
        )

        response2 = self.orchestrator.conduct_deliberation_round(
            request=request2,
            request_id="test_009b",
        )

        # Version should increment
        assert response2.consensus_statement.version == 2

    def test_session_status_updates(self):
        """Test that session status is updated correctly."""
        request = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
            config={
                "convergence_criteria": {
                    "support_threshold": 0.5,
                    "agreement_threshold": 0.5,
                    "max_rounds": 1,
                }
            },
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_010",
        )

        session = self.orchestrator.get_session(response.session_id)

        # Status should be updated based on convergence
        assert session.status in ["active", "converged"]

    def test_session_timestamps_updated(self):
        """Test that session timestamps are updated."""
        request = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_011",
        )

        session = self.orchestrator.get_session(response.session_id)

        # Timestamps should be present
        assert session.started_at is not None
        assert session.updated_at is not None

    def test_convergence_criteria_defaults(self):
        """Test that default convergence criteria are applied."""
        request = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
            # No config provided
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_012",
        )

        session = self.orchestrator.get_session(response.session_id)

        # Should have default criteria
        assert "support_threshold" in session.convergence_criteria
        assert "agreement_threshold" in session.convergence_criteria
        assert "max_rounds" in session.convergence_criteria

    def test_multiple_rounds_tracked(self):
        """Test that multiple rounds are tracked in session."""
        request1 = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response1 = self.orchestrator.conduct_deliberation_round(
            request=request1,
            request_id="test_013a",
        )

        request2 = DeliberationRequest(
            session_id=response1.session_id,
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response2 = self.orchestrator.conduct_deliberation_round(
            request=request2,
            request_id="test_013b",
        )

        session = self.orchestrator.get_session(response1.session_id)

        # Should have 2 rounds
        assert len(session.rounds) == 2
        assert session.total_rounds == 2

    def test_metadata_included_in_response(self):
        """Test that metadata is included in response."""
        request = DeliberationRequest(
            decision_context="Test",
            positions=[
                MemberPosition(
                    member_id="alice",
                    timestamp="2025-01-01T00:00:00",
                )
            ],
        )

        response = self.orchestrator.conduct_deliberation_round(
            request=request,
            request_id="test_014",
        )

        # Metadata should be present
        assert response.metadata is not None
        assert response.metadata.request_id == "test_014"

    def test_get_nonexistent_session(self):
        """Test getting session that doesn't exist."""
        session = self.orchestrator.get_session("nonexistent_id")
        assert session is None
