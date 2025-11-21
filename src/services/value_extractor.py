"""
Extract and model team member values from free-form input.

Uses intelligent parsing to understand what matters to team members and why.
This implementation uses rule-based extraction that can be upgraded to LLM later.
"""

import logging
import re
from datetime import datetime
from typing import List

from src.models.deliberation import ConcernStatement, MemberPosition, ValueStatement

logger = logging.getLogger(__name__)


class ValueExtractor:
    """
    Extract structured values from team member input.

    Transforms free-form position statements into structured
    values, concerns, and rationales.

    Current implementation: Rule-based (can be upgraded to LLM)
    """

    # Common value keywords
    VALUE_KEYWORDS = {
        "user": ["user", "customer", "client", "end-user", "satisfaction"],
        "quality": ["quality", "excellence", "robust", "reliable", "stable"],
        "speed": ["speed", "fast", "quick", "rapid", "time-to-market", "velocity"],
        "innovation": ["innovation", "novel", "creative", "cutting-edge", "new"],
        "cost": ["cost", "budget", "affordable", "economical", "cheap"],
        "technical": ["technical", "engineering", "architecture", "scalable"],
        "business": ["business", "revenue", "profit", "growth", "market"],
        "team": ["team", "collaboration", "communication", "morale"],
    }

    # Concern indicators
    CONCERN_INDICATORS = [
        "worried",
        "concerned",
        "risk",
        "problem",
        "issue",
        "challenge",
        "fear",
        "doubt",
    ]

    def __init__(self):
        """Initialize value extractor."""
        pass

    def extract_values_and_concerns(
        self,
        position_text: str,
        member_id: str,
        context: str,
        request_id: str,
    ) -> MemberPosition:
        """
        Extract structured position from free-form text.

        Args:
            position_text: Team member's position statement
            member_id: Member identifier
            context: Decision context
            request_id: Request ID for tracing

        Returns:
            Structured member position with values and concerns
        """
        logger.info(
            "Extracting values from position",
            extra={
                "request_id": request_id,
                "member_id": member_id,
                "text_length": len(position_text),
            },
        )

        # Extract values
        values = self._extract_values(position_text)

        # Extract concerns
        concerns = self._extract_concerns(position_text)

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
            },
        )

        return position

    def _extract_values(self, text: str) -> List[ValueStatement]:
        """
        Extract values from text.

        Uses keyword matching and sentence analysis.
        """
        values = []
        text_lower = text.lower()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        for value_category, keywords in self.VALUE_KEYWORDS.items():
            # Check if any keyword is present
            matches = [kw for kw in keywords if kw in text_lower]

            if matches:
                # Find relevant sentences
                relevant_sentences = [
                    s
                    for s in sentences
                    if any(kw in s.lower() for kw in matches)
                ]

                if relevant_sentences:
                    # Determine weight based on emphasis
                    weight = self._calculate_weight(text_lower, matches)

                    # Build rationale from relevant sentences
                    rationale = " ".join(relevant_sentences[:2])  # Top 2 sentences

                    # Extract examples
                    examples = self._extract_examples(relevant_sentences)

                    value = ValueStatement(
                        value_name=value_category,
                        weight=weight,
                        rationale=rationale if rationale else f"Values {value_category}",
                        examples=examples,
                    )

                    values.append(value)

        # If no values found, create default
        if not values:
            values.append(
                ValueStatement(
                    value_name="general_quality",
                    weight=0.5,
                    rationale="General preference for quality outcomes",
                    examples=[],
                )
            )

        return values

    def _extract_concerns(self, text: str) -> List[ConcernStatement]:
        """
        Extract concerns from text.

        Looks for worry/risk indicators.
        """
        concerns = []
        text_lower = text.lower()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Find sentences with concern indicators
        concern_sentences = [
            s
            for s in sentences
            if any(indicator in s.lower() for indicator in self.CONCERN_INDICATORS)
        ]

        if concern_sentences:
            # Group by topic
            for sentence in concern_sentences[:3]:  # Max 3 concerns
                # Determine concern type
                concern_type = self._identify_concern_type(sentence)

                # Calculate severity
                severity = self._calculate_severity(sentence)

                concern = ConcernStatement(
                    concern_name=concern_type,
                    severity=severity,
                    explanation=sentence,
                    conditions=None,  # Could extract conditions with more sophisticated parsing
                )

                concerns.append(concern)

        return concerns

    def _calculate_weight(self, text: str, keywords: List[str]) -> float:
        """
        Calculate weight/priority based on emphasis.

        Looks for emphasis indicators like "most important", "critical", etc.
        """
        emphasis_indicators = {
            "critical": 0.95,
            "essential": 0.9,
            "crucial": 0.9,
            "most important": 0.9,
            "key": 0.8,
            "important": 0.75,
            "priority": 0.75,
            "prefer": 0.6,
            "like": 0.5,
        }

        weight = 0.5  # Default

        for indicator, indicator_weight in emphasis_indicators.items():
            if indicator in text:
                # Check if it's near our keywords
                for keyword in keywords:
                    keyword_pos = text.find(keyword)
                    indicator_pos = text.find(indicator)

                    if keyword_pos != -1 and indicator_pos != -1:
                        # Within reasonable distance
                        if abs(keyword_pos - indicator_pos) < 100:
                            weight = max(weight, indicator_weight)

        return weight

    def _calculate_severity(self, sentence: str) -> float:
        """Calculate concern severity."""
        severity_indicators = {
            "critical": 0.95,
            "serious": 0.85,
            "major": 0.8,
            "significant": 0.7,
            "moderate": 0.5,
            "minor": 0.3,
        }

        sentence_lower = sentence.lower()

        for indicator, severity_value in severity_indicators.items():
            if indicator in sentence_lower:
                return severity_value

        # Default based on concern words
        if "very" in sentence_lower or "really" in sentence_lower:
            return 0.7
        else:
            return 0.5

    def _identify_concern_type(self, sentence: str) -> str:
        """Identify what the concern is about."""
        sentence_lower = sentence.lower()

        concern_types = {
            "technical_risk": ["technical", "bug", "scale", "performance", "architecture"],
            "schedule_risk": ["time", "deadline", "schedule", "delay", "late"],
            "quality_risk": ["quality", "defect", "error", "stability"],
            "resource_risk": ["resource", "budget", "cost", "capacity"],
            "user_risk": ["user", "customer", "adoption", "satisfaction"],
        }

        for concern_type, keywords in concern_types.items():
            if any(kw in sentence_lower for kw in keywords):
                return concern_type

        return "general_risk"

    def _extract_examples(self, sentences: List[str]) -> List[str]:
        """
        Extract concrete examples from sentences.

        Looks for example indicators.
        """
        examples = []
        example_indicators = ["for example", "such as", "like", "e.g.", "including"]

        for sentence in sentences:
            sentence_lower = sentence.lower()

            for indicator in example_indicators:
                if indicator in sentence_lower:
                    # Extract the part after indicator
                    parts = sentence_lower.split(indicator)
                    if len(parts) > 1:
                        example = parts[1].strip()
                        examples.append(example[:100])  # Truncate long examples

        return examples[:3]  # Max 3 examples
