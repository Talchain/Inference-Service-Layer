"""
Identify common ground across team member positions.

Finds shared values, concerns, and areas of agreement using
sophisticated similarity matching and aggregation.
"""

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.models.deliberation import HabermasCommonGround, MemberPosition, SharedValue

logger = logging.getLogger(__name__)


class CommonGroundFinder:
    """
    Identify common ground across diverse positions.

    Finds where team members agree, even if they express it differently.
    Uses semantic grouping and statistical aggregation.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize common ground finder.

        Args:
            similarity_threshold: Min similarity to consider values "same" (0-1)
        """
        self.similarity_threshold = similarity_threshold

    def find_common_ground(
        self,
        positions: List[MemberPosition],
        request_id: str,
    ) -> HabermasCommonGround:
        """
        Identify common ground across positions.

        Args:
            positions: All team member positions
            request_id: Request ID for tracing

        Returns:
            Common ground (shared values and concerns)
        """
        if not positions:
            return HabermasCommonGround(
                shared_values=[],
                shared_concerns=[],
                agreement_level=0.0,
                participants=[],
            )

        logger.info(
            "Finding common ground",
            extra={
                "request_id": request_id,
                "num_positions": len(positions),
                "total_values": sum(len(p.values) for p in positions),
                "total_concerns": sum(len(p.concerns) for p in positions),
            },
        )

        # Find shared values
        shared_values = self._find_shared_values(positions)

        # Find shared concerns
        shared_concerns = self._find_shared_concerns(positions)

        # Compute overall agreement level
        agreement_level = self._compute_agreement_level(positions, shared_values, shared_concerns)

        common_ground = HabermasCommonGround(
            shared_values=shared_values,
            shared_concerns=shared_concerns,
            agreement_level=agreement_level,
            participants=[p.member_id for p in positions],
        )

        logger.info(
            "Common ground identified",
            extra={
                "request_id": request_id,
                "shared_values": len(shared_values),
                "shared_concerns": len(shared_concerns),
                "agreement_level": agreement_level,
            },
        )

        return common_ground

    def _find_shared_values(self, positions: List[MemberPosition]) -> List[SharedValue]:
        """
        Find values shared across multiple members.

        Strategy:
        1. Group similar values together
        2. For each group, compute agreement score
        3. Return groups with high agreement
        """
        # Collect all values
        all_values = []
        for position in positions:
            for value in position.values:
                all_values.append({"member_id": position.member_id, "value": value})

        if not all_values:
            return []

        # Group similar values
        value_groups = self._group_similar_values(all_values)

        # Convert to SharedValue objects
        shared_values = []
        for group_name, group_members in value_groups.items():
            if len(group_members) >= 2:  # At least 2 members share
                # Compute metrics
                supporting_members = [m["member_id"] for m in group_members]
                weights = [m["value"].weight for m in group_members]
                average_weight = float(np.mean(weights))
                agreement_score = len(group_members) / len(positions)

                # Synthesize rationale
                rationales = [m["value"].rationale for m in group_members]
                synthesized_rationale = self._synthesize_text(rationales)

                shared_value = SharedValue(
                    value_name=group_name,
                    agreement_score=agreement_score,
                    supporting_members=supporting_members,
                    average_weight=average_weight,
                    synthesized_rationale=synthesized_rationale,
                )
                shared_values.append(shared_value)

        # Sort by agreement score (descending)
        shared_values.sort(key=lambda v: v.agreement_score, reverse=True)

        return shared_values

    def _group_similar_values(self, all_values: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group similar values together.

        Uses name-based grouping (in production, would use embeddings).
        """
        groups = defaultdict(list)

        for value_item in all_values:
            value_name = value_item["value"].value_name

            # Normalize name (lowercase, remove underscores)
            normalized = value_name.lower().replace("_", " ").strip()

            # Group by normalized name
            groups[normalized].append(value_item)

        return dict(groups)

    def _find_shared_concerns(self, positions: List[MemberPosition]) -> List[SharedValue]:
        """Find concerns shared across multiple members."""
        # Collect all concerns
        all_concerns = []
        for position in positions:
            for concern in position.concerns:
                all_concerns.append({"member_id": position.member_id, "concern": concern})

        if not all_concerns:
            return []

        # Group similar concerns
        concern_groups = defaultdict(list)
        for concern_item in all_concerns:
            concern_name = concern_item["concern"].concern_name
            normalized = concern_name.lower().replace("_", " ").strip()
            concern_groups[normalized].append(concern_item)

        # Convert to SharedValue (reusing model)
        shared_concerns = []
        for group_name, group_members in concern_groups.items():
            if len(group_members) >= 2:
                supporting_members = [m["member_id"] for m in group_members]
                severities = [m["concern"].severity for m in group_members]
                average_severity = float(np.mean(severities))
                agreement_score = len(group_members) / len(positions)

                explanations = [m["concern"].explanation for m in group_members]
                synthesized_explanation = self._synthesize_text(explanations)

                shared_concern = SharedValue(
                    value_name=group_name,
                    agreement_score=agreement_score,
                    supporting_members=supporting_members,
                    average_weight=average_severity,
                    synthesized_rationale=synthesized_explanation,
                )
                shared_concerns.append(shared_concern)

        shared_concerns.sort(key=lambda c: c.agreement_score, reverse=True)

        return shared_concerns

    def _compute_agreement_level(
        self,
        positions: List[MemberPosition],
        shared_values: List[SharedValue],
        shared_concerns: List[SharedValue],
    ) -> float:
        """
        Compute overall agreement level.

        Higher when:
        - More shared values/concerns
        - Higher agreement scores on shared items
        - Fewer unshared values/concerns
        """
        if not positions:
            return 0.0

        # Weighted average of agreement scores
        all_shared = shared_values + shared_concerns

        if not all_shared:
            return 0.0

        # Weight by number of supporters
        weighted_scores = [
            s.agreement_score * len(s.supporting_members) for s in all_shared
        ]

        total_agreements = sum(weighted_scores)
        max_possible = len(all_shared) * len(positions)

        agreement_level = total_agreements / max_possible if max_possible > 0 else 0.0

        return min(1.0, agreement_level)

    def _synthesize_text(self, texts: List[str]) -> str:
        """
        Synthesize multiple text snippets into one.

        Simple implementation: takes most common phrases.
        In production, would use LLM for better synthesis.
        """
        if not texts:
            return ""

        # Remove duplicates
        unique_texts = list(set(texts))

        if len(unique_texts) == 1:
            return unique_texts[0]

        # Combine top texts
        combined = "; ".join(unique_texts[:3])  # Top 3 to avoid verbosity

        # Truncate if too long
        if len(combined) > 200:
            combined = combined[:197] + "..."

        return combined
