"""
Generate consensus statements from common ground.

Creates statements that reflect shared values while respecting all voices.
This implementation uses template-based generation that can be upgraded to LLM.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from src.models.deliberation import (
    ConsensusStatement,
    EditSuggestion,
    HabermasCommonGround,
    MemberPosition,
)

logger = logging.getLogger(__name__)


class ConsensusGenerator:
    """
    Generate and refine consensus statements.

    Creates statements that reflect common ground and can be
    iteratively refined through team feedback.

    Current implementation: Template-based (can be upgraded to LLM)
    """

    def __init__(self):
        """Initialize consensus generator."""
        pass

    def generate_consensus(
        self,
        common_ground: HabermasCommonGround,
        positions: List[MemberPosition],
        previous_consensus: Optional[ConsensusStatement],
        edit_suggestions: Optional[List[EditSuggestion]],
        decision_context: str,
        request_id: str,
    ) -> ConsensusStatement:
        """
        Generate consensus statement from common ground.

        Args:
            common_ground: Identified shared values/concerns
            positions: All member positions (for context)
            previous_consensus: Previous statement being refined (if any)
            edit_suggestions: Suggested edits from team (if any)
            decision_context: What's being decided
            request_id: Request ID

        Returns:
            Generated consensus statement
        """
        logger.info(
            "Generating consensus statement",
            extra={
                "request_id": request_id,
                "shared_values": len(common_ground.shared_values),
                "shared_concerns": len(common_ground.shared_concerns),
                "refining_previous": previous_consensus is not None,
                "edit_suggestions": len(edit_suggestions) if edit_suggestions else 0,
            },
        )

        # Determine version
        version = (previous_consensus.version + 1) if previous_consensus else 1

        # Generate statement text
        if previous_consensus and edit_suggestions:
            # Refine existing statement
            statement_text = self._refine_statement(
                previous_consensus, edit_suggestions, common_ground
            )
        else:
            # Generate new statement
            statement_text = self._generate_new_statement(common_ground, decision_context)

        # Identify incorporated elements
        incorporated_values = [sv.value_name for sv in common_ground.shared_values]
        incorporated_concerns = [sc.value_name for sc in common_ground.shared_concerns]

        # Identify unresolved disagreements
        unresolved = self._identify_unresolved_disagreements(common_ground, positions)

        # Determine support
        supporting_members = [p.member_id for p in positions]
        support_score = common_ground.agreement_level

        # Create statement
        consensus = ConsensusStatement(
            statement_id=(
                previous_consensus.statement_id
                if previous_consensus
                else f"consensus_{uuid.uuid4().hex[:8]}"
            ),
            version=version,
            text=statement_text,
            incorporated_values=incorporated_values,
            incorporated_concerns=incorporated_concerns,
            supporting_members=supporting_members,
            support_score=support_score,
            unresolved_disagreements=unresolved,
            generated_at=datetime.utcnow().isoformat(),
        )

        logger.info(
            "Consensus statement generated",
            extra={
                "request_id": request_id,
                "statement_id": consensus.statement_id,
                "version": consensus.version,
                "text_length": len(consensus.text),
                "support_score": consensus.support_score,
            },
        )

        return consensus

    def _generate_new_statement(
        self,
        common_ground: HabermasCommonGround,
        decision_context: str,
    ) -> str:
        """
        Generate new consensus statement from common ground.

        Uses template-based generation with common ground elements.
        """
        parts = []

        # Opening
        parts.append(f"Regarding {decision_context}, we've identified the following alignment:")

        # Shared values
        if common_ground.shared_values:
            parts.append("\nShared Values:")
            for sv in common_ground.shared_values[:5]:  # Top 5
                support_pct = int(sv.agreement_score * 100)
                parts.append(
                    f"- {sv.value_name.replace('_', ' ').title()} "
                    f"({support_pct}% agreement): {sv.synthesized_rationale[:100]}"
                )

        # Shared concerns
        if common_ground.shared_concerns:
            parts.append("\nShared Concerns:")
            for sc in common_ground.shared_concerns[:3]:  # Top 3
                support_pct = int(sc.agreement_score * 100)
                parts.append(
                    f"- {sc.value_name.replace('_', ' ').title()} "
                    f"({support_pct}% agreement): {sc.synthesized_rationale[:100]}"
                )

        # Commitment statement
        agreement_pct = int(common_ground.agreement_level * 100)
        parts.append(f"\nWith {agreement_pct}% alignment, we commit to:")

        # Generate commitments from top shared values
        for sv in common_ground.shared_values[:3]:
            value_name = sv.value_name.replace("_", " ")
            parts.append(f"• Prioritizing {value_name} in our decision-making")

        # Add concern acknowledgment if any
        if common_ground.shared_concerns:
            parts.append("\nWe acknowledge these concerns and will address them:")
            for sc in common_ground.shared_concerns[:2]:
                concern_name = sc.value_name.replace("_", " ")
                parts.append(f"• Mitigating {concern_name}")

        statement = "\n".join(parts)

        return statement

    def _refine_statement(
        self,
        previous: ConsensusStatement,
        edits: List[EditSuggestion],
        common_ground: HabermasCommonGround,
    ) -> str:
        """
        Refine existing statement based on edit suggestions.

        Incorporates edits while maintaining common ground.
        """
        text = previous.text

        # Sort edits by priority
        sorted_edits = sorted(edits, key=lambda e: e.priority, reverse=True)

        # Apply top edits
        additions = []
        clarifications = []

        for edit in sorted_edits[:5]:  # Top 5 edits
            if edit.edit_type == "addition":
                additions.append(f"• {edit.suggestion}")
            elif edit.edit_type == "clarification":
                clarifications.append(f"Specifically: {edit.suggestion}")
            elif edit.edit_type == "deletion":
                # Simple deletion by keyword removal
                # In production, would use more sophisticated text manipulation
                pass

        # Incorporate additions
        if additions:
            text += "\n\nAdditional Points (from team feedback):\n"
            text += "\n".join(additions)

        # Incorporate clarifications
        if clarifications:
            text += "\n\nClarifications:\n"
            text += "\n".join(clarifications)

        # Add version note
        text += f"\n\n(Consensus Statement v{previous.version + 1}, refined based on team feedback)"

        return text

    def _identify_unresolved_disagreements(
        self,
        common_ground: HabermasCommonGround,
        positions: List[MemberPosition],
    ) -> List[str]:
        """
        Identify disagreements not yet resolved in common ground.

        Looks for values/concerns held by minority that aren't shared.
        """
        unresolved = []

        # Collect all values mentioned
        all_value_names = set()
        for pos in positions:
            for value in pos.values:
                all_value_names.add(value.value_name)

        # Find values not in shared values
        shared_value_names = {sv.value_name for sv in common_ground.shared_values}
        unshared_values = all_value_names - shared_value_names

        # Check if significant (held by >1 person)
        for value_name in unshared_values:
            count = sum(
                1
                for pos in positions
                if any(v.value_name == value_name for v in pos.values)
            )

            if count >= 2:  # At least 2 people care
                pct = int((count / len(positions)) * 100)
                unresolved.append(
                    f"{value_name.replace('_', ' ').title()} "
                    f"(valued by {pct}% but not yet shared consensus)"
                )

        return unresolved[:3]  # Top 3 unresolved issues
