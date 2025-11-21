"""
LLM-powered value extraction from free-form positions.

Replaces rule-based keyword matching with semantic understanding.
Falls back to rule-based extraction on errors or budget limits.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import List, Optional

from src.models.deliberation import ConcernStatement, MemberPosition, ValueStatement
from src.services.llm_client import LLMClient
from src.services.value_extractor import ValueExtractor as RuleBasedExtractor
from src.utils.business_metrics import track_llm_fallback, track_llm_request

logger = logging.getLogger(__name__)


class ValueExtractorLLM:
    """
    Extract structured values using LLM semantic understanding.

    Transforms free-form position statements into structured
    values, concerns, and rationales.

    Falls back to rule-based extraction on:
    - LLM API errors
    - Budget exceeded
    - Invalid JSON responses
    """

    def __init__(self, llm_client: LLMClient, fallback_extractor: Optional[RuleBasedExtractor] = None):
        """
        Initialize with LLM client.

        Args:
            llm_client: LLM client for API calls
            fallback_extractor: Rule-based extractor for fallback
        """
        self.llm = llm_client
        self.fallback = fallback_extractor or RuleBasedExtractor()

    def extract_values_and_concerns(
        self,
        position_text: str,
        member_id: str,
        context: str,
        session_id: Optional[str] = None,
        request_id: str = "unknown",
    ) -> MemberPosition:
        """
        Extract structured position using LLM.

        Args:
            position_text: Team member's position statement
            member_id: Member identifier
            context: Decision context
            session_id: Session ID for cost tracking
            request_id: Request ID for tracing

        Returns:
            Structured member position with values and concerns
        """
        logger.info(
            "Extracting values with LLM",
            extra={
                "request_id": request_id,
                "member_id": member_id,
                "text_length": len(position_text),
            },
        )

        # Build prompt
        messages = self._build_extraction_messages(
            position_text=position_text,
            context=context,
        )

        # Generate cache key (deterministic based on input)
        cache_key = hashlib.md5(f"{context}:{position_text}".encode()).hexdigest()

        # Call LLM
        try:
            response = self.llm.complete(
                messages=messages,
                model=self.llm.config.extraction_model,  # Cheaper model
                temperature=0.3,  # Lower for consistency
                session_id=session_id,
                cache_key=cache_key,
                request_id=request_id,
            )

            # Track metrics
            track_llm_request(
                model=self.llm.config.extraction_model,
                endpoint="value_extraction",
                cost=response["cost"],
                input_tokens=response["usage"]["input_tokens"],
                output_tokens=response["usage"]["output_tokens"],
                cached=response["cached"],
            )

            # Parse JSON response
            result = json.loads(response["content"])

            # Build structured position
            values = [
                ValueStatement(
                    value_name=v["name"],
                    weight=v["weight"],
                    rationale=v["rationale"],
                    examples=v.get("examples", []),
                )
                for v in result.get("values", [])
            ]

            concerns = [
                ConcernStatement(
                    concern_name=c["name"],
                    severity=c["severity"],
                    explanation=c["explanation"],
                    conditions=c.get("conditions"),
                )
                for c in result.get("concerns", [])
            ]

            position = MemberPosition(
                member_id=member_id,
                values=values,
                concerns=concerns,
                position_statement=position_text,
                timestamp=datetime.utcnow().isoformat(),
            )

            logger.info(
                "Value extraction complete",
                extra={
                    "request_id": request_id,
                    "member_id": member_id,
                    "values_found": len(values),
                    "concerns_found": len(concerns),
                    "cost": response["cost"],
                    "cached": response["cached"],
                },
            )

            return position

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM response: {e}",
                extra={"request_id": request_id, "response": response.get("content", "")[:200]},
            )
            # Fallback to rule-based
            track_llm_fallback(reason="json_parse_error")
            return self._fallback_extraction(position_text, member_id, context, request_id)

        except ValueError as e:
            # Budget exceeded or similar
            if "budget" in str(e).lower():
                logger.warning(f"Budget exceeded, falling back to rules: {e}")
                track_llm_fallback(reason="budget_exceeded")
            else:
                logger.error(f"LLM value error: {e}")
                track_llm_fallback(reason="value_error")
            return self._fallback_extraction(position_text, member_id, context, request_id)

        except Exception as e:
            logger.error(
                f"LLM extraction failed: {e}", extra={"request_id": request_id}
            )
            track_llm_fallback(reason="api_error")
            return self._fallback_extraction(position_text, member_id, context, request_id)

    def _build_extraction_messages(
        self,
        position_text: str,
        context: str,
    ) -> List[dict]:
        """Build messages for LLM extraction."""
        system_prompt = """You are an expert at understanding what people value in decisions.

Your task: Extract structured information from a team member's position statement.

1. VALUES: What matters to them and why
   - Assign priority weights (0.0-1.0, where 1.0 = most important)
   - Provide rationale in their own words
   - Include concrete examples if mentioned

2. CONCERNS: What they're worried about
   - Assign severity (0.0-1.0, where 1.0 = critical concern)
   - Explain the concern clearly
   - Note conditions that would address it

**Be precise and faithful to their actual words. Don't invent values they didn't express.**

Output valid JSON only:
{
  "values": [
    {
      "name": "descriptive_identifier",
      "weight": 0.8,
      "rationale": "Why this matters to them",
      "examples": ["concrete example 1"]
    }
  ],
  "concerns": [
    {
      "name": "descriptive_identifier",
      "severity": 0.7,
      "explanation": "What's the concern and why",
      "conditions": "What would address this?"
    }
  ]
}"""

        user_prompt = f"""Decision Context:
{context}

Team Member's Position:
{position_text}

Extract values and concerns. Output JSON only."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _fallback_extraction(
        self,
        position_text: str,
        member_id: str,
        context: str,
        request_id: str,
    ) -> MemberPosition:
        """Fallback to rule-based extraction."""
        logger.warning(
            "Using rule-based fallback for value extraction",
            extra={"request_id": request_id, "member_id": member_id},
        )

        return self.fallback.extract_values_and_concerns(
            position_text=position_text,
            member_id=member_id,
            context=context,
            request_id=request_id,
        )
