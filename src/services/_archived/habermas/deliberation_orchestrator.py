"""
Orchestrate multi-round deliberation sessions.

Manages deliberation state, convergence detection, and
iteration through refinement rounds using Habermas Machine principles.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.models.deliberation import (
    ConsensusStatement,
    DeliberationRequest,
    DeliberationResponse,
    DeliberationRound,
    DeliberationSession,
    MemberPosition,
)
from src.models.metadata import create_response_metadata
from src.services.common_ground_finder import CommonGroundFinder
from src.services.consensus_generator import ConsensusGenerator
from src.services.value_extractor import ValueExtractor

logger = logging.getLogger(__name__)


class DeliberationOrchestrator:
    """
    Orchestrate Habermas Machine deliberation.

    Coordinates:
    1. Value extraction from positions
    2. Common ground identification
    3. Consensus generation
    4. Iterative refinement
    5. Convergence detection
    """

    def __init__(
        self,
        value_extractor: Optional[ValueExtractor] = None,
        common_ground_finder: Optional[CommonGroundFinder] = None,
        consensus_generator: Optional[ConsensusGenerator] = None,
    ):
        """Initialize deliberation orchestrator."""
        self.value_extractor = value_extractor or ValueExtractor()
        self.common_ground_finder = common_ground_finder or CommonGroundFinder()
        self.consensus_generator = consensus_generator or ConsensusGenerator()

        # Session storage (in production, would use database)
        self.sessions: Dict[str, DeliberationSession] = {}

    def conduct_deliberation_round(
        self,
        request: DeliberationRequest,
        request_id: str,
    ) -> DeliberationResponse:
        """
        Conduct one round of deliberation.

        Args:
            request: Deliberation request
            request_id: Request ID

        Returns:
            DeliberationResponse with results
        """
        # Get or create session
        session = self._get_or_create_session(request, request_id)

        round_number = session.total_rounds + 1

        logger.info(
            "Starting deliberation round",
            extra={
                "request_id": request_id,
                "session_id": session.session_id,
                "round_number": round_number,
                "positions": len(request.positions),
            },
        )

        # Step 1: Find common ground
        common_ground = self.common_ground_finder.find_common_ground(
            positions=request.positions,
            request_id=request_id,
        )

        # Step 2: Generate consensus statement
        consensus_statement = self.consensus_generator.generate_consensus(
            common_ground=common_ground,
            positions=request.positions,
            previous_consensus=request.previous_consensus,
            edit_suggestions=request.edit_suggestions,
            decision_context=request.decision_context,
            request_id=request_id,
        )

        # Step 3: Create deliberation round
        round_obj = DeliberationRound(
            round_number=round_number,
            positions=request.positions,
            common_ground=common_ground,
            consensus_statement=consensus_statement,
            edit_suggestions=request.edit_suggestions or [],
            agreement_level=common_ground.agreement_level,
            participation_rate=self._compute_participation_rate(request.positions, session),
            timestamp=datetime.utcnow().isoformat(),
        )

        # Step 4: Update session
        session.rounds.append(round_obj)
        session.current_consensus = consensus_statement
        session.total_rounds = round_number
        session.updated_at = datetime.utcnow().isoformat()

        # Step 5: Check convergence
        has_converged, convergence_assessment = self._check_convergence(
            session=session,
            common_ground=common_ground,
            consensus=consensus_statement,
        )

        if has_converged:
            session.status = "converged"
            session.final_agreement_level = common_ground.agreement_level
            session.converged_at = datetime.utcnow().isoformat()

        # Step 6: Save session
        self.sessions[session.session_id] = session

        # Step 7: Generate next steps
        next_steps = self._generate_next_steps(
            session=session,
            consensus=consensus_statement,
            convergence_assessment=convergence_assessment,
        )

        # Step 8: Build response
        response = DeliberationResponse(
            session_id=session.session_id,
            round_number=round_number,
            common_ground=common_ground,
            consensus_statement=consensus_statement,
            status="converged" if has_converged else "active",
            convergence_assessment=convergence_assessment,
            next_steps=next_steps,
            metadata=create_response_metadata(request_id),
        )

        logger.info(
            "Deliberation round complete",
            extra={
                "request_id": request_id,
                "session_id": session.session_id,
                "round_number": round_number,
                "agreement_level": common_ground.agreement_level,
                "converged": has_converged,
            },
        )

        return response

    def get_session(self, session_id: str) -> Optional[DeliberationSession]:
        """Get existing session by ID."""
        return self.sessions.get(session_id)

    def _get_or_create_session(
        self,
        request: DeliberationRequest,
        request_id: str,
    ) -> DeliberationSession:
        """Get existing session or create new one."""
        if request.session_id and request.session_id in self.sessions:
            return self.sessions[request.session_id]

        # Create new session
        team_members = [
            {
                "id": p.member_id,
                "name": p.member_name or p.member_id,
                "role": p.role or "Team Member",
            }
            for p in request.positions
        ]

        # Default convergence criteria
        default_criteria = {
            "support_threshold": 0.8,  # 80% support required
            "agreement_threshold": 0.7,  # 70% agreement required
            "max_rounds": 10,
        }
        criteria = request.config.get("convergence_criteria", default_criteria)

        session = DeliberationSession(
            session_id=request.session_id or f"delib_{uuid.uuid4().hex[:12]}",
            decision_context=request.decision_context,
            team_members=team_members,
            rounds=[],
            current_consensus=None,
            status="active",
            convergence_criteria=criteria,
            total_rounds=0,
            final_agreement_level=None,
            started_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )

        logger.info(
            "Created new deliberation session",
            extra={
                "request_id": request_id,
                "session_id": session.session_id,
                "team_size": len(team_members),
            },
        )

        return session

    def _check_convergence(
        self,
        session: DeliberationSession,
        common_ground: Any,
        consensus: ConsensusStatement,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if deliberation has converged.

        Convergence when:
        - High support for consensus (>threshold)
        - High agreement level (>threshold)
        - No significant unresolved disagreements
        - OR max rounds reached
        """
        criteria = session.convergence_criteria

        # Extract thresholds
        support_threshold = criteria.get("support_threshold", 0.8)
        agreement_threshold = criteria.get("agreement_threshold", 0.7)
        max_rounds = criteria.get("max_rounds", 10)

        # Check conditions
        support_met = consensus.support_score >= support_threshold
        agreement_met = common_ground.agreement_level >= agreement_threshold
        no_major_disagreements = len(consensus.unresolved_disagreements) == 0
        max_rounds_reached = session.total_rounds >= max_rounds

        # Converged if all conditions met OR max rounds
        has_converged = (
            support_met and agreement_met and no_major_disagreements
        ) or max_rounds_reached

        # Assessment details
        assessment = {
            "support_score": consensus.support_score,
            "support_threshold": support_threshold,
            "support_met": support_met,
            "agreement_level": common_ground.agreement_level,
            "agreement_threshold": agreement_threshold,
            "agreement_met": agreement_met,
            "unresolved_count": len(consensus.unresolved_disagreements),
            "unresolved_list": consensus.unresolved_disagreements,
            "rounds_used": session.total_rounds,
            "max_rounds": max_rounds,
            "max_rounds_reached": max_rounds_reached,
            "convergence_type": "consensus" if not max_rounds_reached else "rounds_limit",
        }

        return has_converged, assessment

    def _compute_participation_rate(
        self,
        positions: List[MemberPosition],
        session: DeliberationSession,
    ) -> float:
        """Compute fraction of team participating in this round."""
        participating_ids = {p.member_id for p in positions}
        total_team_size = len(session.team_members)

        if total_team_size == 0:
            return 1.0

        return len(participating_ids) / total_team_size

    def _generate_next_steps(
        self,
        session: DeliberationSession,
        consensus: ConsensusStatement,
        convergence_assessment: Dict[str, Any],
    ) -> List[str]:
        """Generate recommended next steps."""
        next_steps = []

        if session.status == "converged":
            next_steps.append("✓ Deliberation complete - consensus reached")
            next_steps.append("Final consensus ready for implementation")
            return next_steps

        # Ongoing deliberation
        if not convergence_assessment["support_met"]:
            next_steps.append(
                f"Build support: Currently {convergence_assessment['support_score']:.0%}, "
                f"need {convergence_assessment['support_threshold']:.0%}"
            )

        if not convergence_assessment["agreement_met"]:
            next_steps.append(
                f"Strengthen common ground: Currently {convergence_assessment['agreement_level']:.0%}, "
                f"need {convergence_assessment['agreement_threshold']:.0%}"
            )

        if convergence_assessment["unresolved_count"] > 0:
            next_steps.append(
                f"Address {convergence_assessment['unresolved_count']} unresolved issue(s):"
            )
            for disagreement in convergence_assessment["unresolved_list"][:2]:
                next_steps.append(f"  - {disagreement}")

        # Suggest actions
        next_steps.append("\nRecommended actions:")
        next_steps.append("1. Team members review consensus statement")
        next_steps.append("2. Submit edit suggestions to clarify/improve")
        next_steps.append("3. Conduct next deliberation round with feedback")

        return next_steps
