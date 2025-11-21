"""
LLM-powered consensus statement generation.

Generates natural, inclusive consensus statements that reflect
common ground while respecting all voices.
"""

import hashlib
import json
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
from src.services.consensus_generator import ConsensusGenerator as TemplateBasedGenerator
from src.services.llm_client import LLMClient
from src.utils.business_metrics import track_llm_fallback, track_llm_request

logger = logging.getLogger(__name__)


class ConsensusGeneratorLLM:
    """
    Generate consensus statements using LLM.

    Creates statements that:
    - Reflect common ground
    - Use inclusive language
    - Acknowledge remaining disagreements
    - Are specific and actionable
    """

    def __init__(
        self,
        llm_client: LLMClient,
        fallback_generator: Optional[TemplateBasedGenerator] = None,
    ):
        """Initialize with LLM client."""
        self.llm = llm_client
        self.fallback = fallback_generator or TemplateBasedGenerator()

    def generate_consensus(
        self,
        common_ground: HabermasCommonGround,
        positions: List[MemberPosition],
        previous_consensus: Optional[ConsensusStatement],
        edit_suggestions: Optional[List[EditSuggestion]],
        decision_context: str,
        session_id: Optional[str] = None,
        request_id: str = "unknown",
    ) -> ConsensusStatement:
        """
        Generate consensus statement using LLM.

        Uses better model (GPT-4) for higher quality consensus.
        """
        logger.info(
            "Generating consensus with LLM",
            extra={
                "request_id": request_id,
                "shared_values": len(common_ground.shared_values),
                "refining": previous_consensus is not None,
                "edits": len(edit_suggestions) if edit_suggestions else 0,
            },
        )

        # Build messages
        messages = self._build_consensus_messages(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=previous_consensus,
            edit_suggestions=edit_suggestions,
            decision_context=decision_context,
        )

        # Generate cache key
        cache_key = hashlib.md5(
            json.dumps(
                {
                    "context": decision_context,
                    "values": [v.value_name for v in common_ground.shared_values],
                    "concerns": [c.value_name for c in common_ground.shared_concerns],
                    "previous": previous_consensus.statement_id if previous_consensus else None,
                    "edits": len(edit_suggestions) if edit_suggestions else 0,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        # Call LLM (use better model for consensus)
        try:
            response = self.llm.complete(
                messages=messages,
                model=self.llm.config.consensus_model,  # GPT-4 for quality
                temperature=0.7,  # Some creativity
                session_id=session_id,
                cache_key=cache_key,
                request_id=request_id,
            )

            # Track metrics
            track_llm_request(
                model=self.llm.config.consensus_model,
                endpoint="consensus_generation",
                cost=response["cost"],
                input_tokens=response["usage"]["input_tokens"],
                output_tokens=response["usage"]["output_tokens"],
                cached=response["cached"],
            )

            # Parse response
            result = json.loads(response["content"])

            # Determine version
            version = (previous_consensus.version + 1) if previous_consensus else 1

            # Build consensus
            consensus = ConsensusStatement(
                statement_id=result.get(
                    "statement_id",
                    previous_consensus.statement_id if previous_consensus else f"consensus_{uuid.uuid4().hex[:8]}",
                ),
                version=version,
                text=result["statement_text"],
                incorporated_values=result.get("incorporated_values", []),
                incorporated_concerns=result.get("incorporated_concerns", []),
                supporting_members=[p.member_id for p in positions],
                support_score=common_ground.agreement_level,
                unresolved_disagreements=result.get("unresolved_disagreements", []),
                generated_at=datetime.utcnow().isoformat(),
            )

            logger.info(
                "Consensus generated",
                extra={
                    "request_id": request_id,
                    "statement_id": consensus.statement_id,
                    "version": version,
                    "cost": response["cost"],
                    "cached": response["cached"],
                },
            )

            return consensus

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM consensus response: {e}")
            track_llm_fallback(reason="json_parse_error")
            return self._fallback_consensus(
                common_ground, positions, previous_consensus, edit_suggestions, decision_context, request_id
            )

        except ValueError as e:
            if "budget" in str(e).lower():
                logger.warning(f"Budget exceeded, falling back to template: {e}")
                track_llm_fallback(reason="budget_exceeded")
            else:
                logger.error(f"LLM consensus error: {e}")
                track_llm_fallback(reason="value_error")
            return self._fallback_consensus(
                common_ground, positions, previous_consensus, edit_suggestions, decision_context, request_id
            )

        except Exception as e:
            logger.error(f"LLM consensus generation failed: {e}")
            track_llm_fallback(reason="api_error")
            return self._fallback_consensus(
                common_ground, positions, previous_consensus, edit_suggestions, decision_context, request_id
            )

    def _build_consensus_messages(
        self,
        common_ground: HabermasCommonGround,
        positions: List[MemberPosition],
        previous_consensus: Optional[ConsensusStatement],
        edit_suggestions: Optional[List[EditSuggestion]],
        decision_context: str,
    ) -> List[dict]:
        """Build messages for consensus generation."""
        system_prompt = """You are facilitating democratic deliberation.

**Your task:** Draft a consensus statement that:
1. Reflects the identified shared values and concerns
2. Respects ALL voices - no steamrolling minorities
3. Is clear, specific, and actionable
4. Uses inclusive language ("we agree that...")
5. Acknowledges remaining disagreements honestly

**Principles:**
- Build on common ground, don't force agreement
- Be concrete, not vague platitudes
- Facilitate understanding, not just vote counting
- Protect minority voices while building consensus

**Output valid JSON only:**
{
  "statement_id": "consensus_xyz",
  "statement_text": "We agree that [specific consensus]...",
  "incorporated_values": ["value1", "value2"],
  "incorporated_concerns": ["concern1"],
  "unresolved_disagreements": ["issue X remains under discussion"]
}"""

        # Build user prompt
        user_parts = []
        user_parts.append(f"**Decision Context:**\n{decision_context}\n")

        user_parts.append(f"**Common Ground:**")
        user_parts.append(f"Agreement Level: {common_ground.agreement_level:.0%}\n")

        if common_ground.shared_values:
            user_parts.append("Shared Values:")
            for sv in common_ground.shared_values:
                user_parts.append(
                    f"- {sv.value_name} (supported by {len(sv.supporting_members)}/{len(positions)}, "
                    f"weight: {sv.average_weight:.2f})"
                )
                user_parts.append(f"  Rationale: {sv.synthesized_rationale}")

        if common_ground.shared_concerns:
            user_parts.append("\nShared Concerns:")
            for sc in common_ground.shared_concerns:
                user_parts.append(
                    f"- {sc.value_name} (shared by {len(sc.supporting_members)}/{len(positions)}, "
                    f"severity: {sc.average_weight:.2f})"
                )
                user_parts.append(f"  Explanation: {sc.synthesized_rationale}")

        # Previous consensus and edits
        if previous_consensus and edit_suggestions:
            user_parts.append(
                f"\n**Previous Consensus (v{previous_consensus.version}):**"
            )
            user_parts.append(previous_consensus.text)

            user_parts.append("\n**Edit Suggestions:**")
            for edit in edit_suggestions:
                user_parts.append(
                    f"- {edit.edit_type.upper()} (priority {edit.priority:.1f}): "
                    f"{edit.suggestion}"
                )
                user_parts.append(f"  Reason: {edit.rationale}")

            user_parts.append(
                "\nGenerate an improved consensus incorporating these edits."
            )
        else:
            user_parts.append(
                "\nGenerate a consensus statement reflecting this common ground."
            )

        user_parts.append("\nOutput JSON only.")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    def _fallback_consensus(
        self,
        common_ground: HabermasCommonGround,
        positions: List[MemberPosition],
        previous_consensus: Optional[ConsensusStatement],
        edit_suggestions: Optional[List[EditSuggestion]],
        decision_context: str,
        request_id: str,
    ) -> ConsensusStatement:
        """Fallback to template-based consensus."""
        logger.warning(
            "Using template-based fallback for consensus generation",
            extra={"request_id": request_id},
        )

        return self.fallback.generate_consensus(
            common_ground=common_ground,
            positions=positions,
            previous_consensus=previous_consensus,
            edit_suggestions=edit_suggestions,
            decision_context=decision_context,
            request_id=request_id,
        )
