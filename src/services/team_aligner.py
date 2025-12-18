"""
Team Aligner service for finding common ground across perspectives.

Identifies shared goals, analyzes option satisfaction, and provides
recommendations for team alignment.
"""

import logging
from typing import Dict, List, Set

from src.models.requests import DecisionOption, TeamAlignmentRequest, TeamPerspective
from src.models.responses import (
    AlignedOption,
    CommonGround,
    Conflict,
    Recommendation,
    TeamAlignmentResponse,
    Tradeoff,
)
from src.models.shared import ConfidenceLevel, ConflictSeverity
from src.services.explanation_generator import ExplanationGenerator
from src.utils.determinism import canonical_hash, make_deterministic

logger = logging.getLogger(__name__)


class TeamAligner:
    """
    Finds alignment across team perspectives.

    Analyzes multiple stakeholder viewpoints to identify common ground,
    rank options, and recommend decisions that satisfy all parties.
    """

    def __init__(self) -> None:
        """Initialize the aligner."""
        self.explanation_generator = ExplanationGenerator()

    def align(self, request: TeamAlignmentRequest) -> TeamAlignmentResponse:
        """
        Find team alignment across perspectives.

        Args:
            request: Team alignment request

        Returns:
            TeamAlignmentResponse: Alignment analysis
        """
        # Make computation deterministic
        rng = make_deterministic(request.model_dump())

        logger.info(
            "team_alignment_started",
            extra={
                "request_hash": canonical_hash(request.model_dump()),
                "num_perspectives": len(request.perspectives),
                "num_options": len(request.options),
                "seed": rng.seed,
            },
        )

        try:
            # Find common ground
            common_ground = self._find_common_ground(request.perspectives)

            # Analyze each option
            aligned_options = self._analyze_options(
                request.perspectives, request.options
            )

            # Identify conflicts
            conflicts = self._identify_conflicts(request.perspectives)

            # Generate recommendation
            recommendation = self._generate_recommendation(
                aligned_options, common_ground, conflicts
            )

            # Generate explanation
            explanation = self.explanation_generator.generate_team_alignment_explanation(
                num_perspectives=len(request.perspectives),
                agreement_level=common_ground.agreement_level,
                top_option=recommendation.top_option,
                satisfaction_score=aligned_options[0].satisfaction_score
                if aligned_options
                else 0,
                num_conflicts=len(conflicts),
            )

            return TeamAlignmentResponse(
                common_ground=common_ground,
                aligned_options=aligned_options,
                conflicts=conflicts,
                recommendation=recommendation,
                explanation=explanation,
            )

        except Exception as e:
            logger.error("team_alignment_failed", exc_info=True)
            raise

    def _find_common_ground(
        self, perspectives: List[TeamPerspective]
    ) -> CommonGround:
        """
        Find shared goals and constraints across perspectives.

        Args:
            perspectives: List of team perspectives

        Returns:
            CommonGround with shared elements
        """
        if not perspectives:
            return CommonGround(
                shared_goals=[],
                shared_constraints=[],
                agreement_level=0.0,
            )

        # Extract all priorities and constraints
        all_priorities = [set(p.priorities) for p in perspectives]
        all_constraints = [set(p.constraints) for p in perspectives]

        # Find intersection (shared by all)
        shared_goals = list(set.intersection(*all_priorities)) if all_priorities else []
        shared_constraints = (
            list(set.intersection(*all_constraints)) if all_constraints else []
        )

        # Calculate agreement level
        # (ratio of shared items to total unique items)
        all_unique_priorities = set.union(*all_priorities) if all_priorities else set()
        all_unique_constraints = set.union(*all_constraints) if all_constraints else set()

        total_unique = len(all_unique_priorities) + len(all_unique_constraints)
        total_shared = len(shared_goals) + len(shared_constraints)

        agreement_level = (
            (total_shared / total_unique * 100) if total_unique > 0 else 0.0
        )

        return CommonGround(
            shared_goals=shared_goals,
            shared_constraints=shared_constraints,
            agreement_level=round(agreement_level, 1),
        )

    def _analyze_options(
        self,
        perspectives: List[TeamPerspective],
        options: List[DecisionOption],
    ) -> List[AlignedOption]:
        """
        Analyze how well each option satisfies all perspectives.

        Args:
            perspectives: List of team perspectives
            options: List of decision options

        Returns:
            List of AlignedOption, sorted by satisfaction score
        """
        aligned_options = []

        for option in options:
            # Count how many perspectives this option satisfies
            satisfies_roles: List[str] = []
            tradeoffs: List[Tradeoff] = []
            total_satisfaction = 0.0

            for perspective in perspectives:
                # Check if this option is in preferred options
                if (
                    perspective.preferred_options
                    and option.id in perspective.preferred_options
                ):
                    satisfies_roles.append(perspective.role)
                    total_satisfaction += 100.0
                else:
                    # Calculate partial satisfaction based on attribute matching
                    satisfaction = self._calculate_satisfaction(
                        perspective, option
                    )
                    total_satisfaction += satisfaction

                    if satisfaction >= 60:  # Threshold for "satisfies"
                        satisfies_roles.append(perspective.role)
                    else:
                        # This is a tradeoff - role doesn't fully get what they want
                        tradeoff = self._identify_tradeoff(perspective, option)
                        if tradeoff:
                            tradeoffs.append(tradeoff)

            # Average satisfaction across all perspectives
            avg_satisfaction = (
                total_satisfaction / len(perspectives) if perspectives else 0.0
            )

            aligned_options.append(
                AlignedOption(
                    option=option.id,
                    satisfies_roles=satisfies_roles,
                    satisfaction_score=round(avg_satisfaction, 1),
                    tradeoffs=tradeoffs,
                )
            )

        # Sort by satisfaction score (descending)
        aligned_options.sort(key=lambda x: x.satisfaction_score, reverse=True)

        return aligned_options

    def _calculate_satisfaction(
        self, perspective: TeamPerspective, option: DecisionOption
    ) -> float:
        """
        Calculate how well an option satisfies a perspective.

        Args:
            perspective: Team perspective
            option: Decision option

        Returns:
            Satisfaction score (0-100)
        """
        # Simple heuristic: check if option attributes match priorities
        matches = 0
        total_priorities = len(perspective.priorities)

        for priority in perspective.priorities:
            # Check if any attribute name or value contains the priority keyword
            priority_lower = priority.lower()
            for attr_name, attr_value in option.attributes.items():
                if priority_lower in attr_name.lower() or priority_lower in str(
                    attr_value
                ).lower():
                    matches += 1
                    break

        return (matches / total_priorities * 100) if total_priorities > 0 else 50.0

    def _identify_tradeoff(
        self, perspective: TeamPerspective, option: DecisionOption
    ) -> Tradeoff:
        """
        Identify what a perspective gives up and gets with this option.

        Args:
            perspective: Team perspective
            option: Decision option

        Returns:
            Tradeoff or None
        """
        # Simplified tradeoff identification
        # In a real implementation, this would be more sophisticated

        gives = "Some priorities"
        gets = "Alternative benefits"

        # Check attributes to infer tradeoffs
        if "speed" in option.attributes:
            if option.attributes["speed"] == "fast":
                gives = "Maximum quality"
                gets = "Faster time-to-market"
            elif option.attributes["speed"] == "slow":
                gives = "Speed"
                gets = "Higher quality"

        return Tradeoff(
            role=perspective.role,
            gives=gives,
            gets=gets,
        )

    def _identify_conflicts(
        self, perspectives: List[TeamPerspective]
    ) -> List[Conflict]:
        """
        Identify conflicts between perspectives.

        Args:
            perspectives: List of team perspectives

        Returns:
            List of conflicts
        """
        conflicts = []

        # Check for mutually exclusive priorities
        # For each pair of perspectives
        for i, p1 in enumerate(perspectives):
            for p2 in perspectives[i + 1 :]:
                # Check if they have conflicting preferred options
                if p1.preferred_options and p2.preferred_options:
                    p1_only = set(p1.preferred_options) - set(p2.preferred_options)
                    p2_only = set(p2.preferred_options) - set(p1.preferred_options)

                    if p1_only and p2_only:
                        # They prefer different options
                        conflict = Conflict(
                            between=[p1.role, p2.role],
                            about="Preferred options",
                            severity=ConflictSeverity.MODERATE,
                            suggestion=f"Find compromise option or sequence: {p1.role}'s priority first, then {p2.role}'s",
                        )
                        conflicts.append(conflict)

                # Check for conflicting priorities (heuristic)
                conflicting_pairs = [
                    ("speed", "quality"),
                    ("cost", "features"),
                    ("innovation", "stability"),
                ]

                for term1, term2 in conflicting_pairs:
                    p1_has_term1 = any(term1 in p.lower() for p in p1.priorities)
                    p2_has_term2 = any(term2 in p.lower() for p in p2.priorities)

                    if p1_has_term1 and p2_has_term2:
                        conflict = Conflict(
                            between=[p1.role, p2.role],
                            about=f"{term1.title()} vs {term2.title()} tradeoff",
                            severity=ConflictSeverity.MINOR,
                            suggestion=f"Negotiate acceptable balance between {term1} and {term2}",
                        )
                        conflicts.append(conflict)

        return conflicts

    def _generate_recommendation(
        self,
        aligned_options: List[AlignedOption],
        common_ground: CommonGround,
        conflicts: List[Conflict],
    ) -> Recommendation:
        """
        Generate top recommendation.

        Args:
            aligned_options: Analyzed options
            common_ground: Common ground
            conflicts: Identified conflicts

        Returns:
            Recommendation
        """
        if not aligned_options:
            return Recommendation(
                top_option="none",
                rationale="No options provided",
                confidence=ConfidenceLevel.LOW,
                next_steps=["Define decision options to evaluate"],
            )

        top_option = aligned_options[0]

        # Determine confidence
        if top_option.satisfaction_score >= 80 and len(conflicts) == 0:
            confidence = ConfidenceLevel.HIGH
        elif top_option.satisfaction_score >= 60:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        # Generate rationale
        rationale = (
            f"This option achieves {top_option.satisfaction_score:.0f}% overall satisfaction, "
            f"satisfying {len(top_option.satisfies_roles)} of the team roles. "
        )

        if common_ground.agreement_level >= 70:
            rationale += "There is strong alignment on shared goals, making implementation smoother."
        elif conflicts:
            rationale += f"However, {len(conflicts)} conflict(s) should be addressed."

        # Generate next steps
        next_steps = []
        if conflicts:
            next_steps.append("Resolve identified conflicts through team discussion")
        if top_option.tradeoffs:
            next_steps.append("Review and approve necessary tradeoffs")
        next_steps.append("Define implementation plan with clear milestones")
        if common_ground.shared_goals:
            next_steps.append(f"Align implementation with shared goals: {', '.join(common_ground.shared_goals[:2])}")

        return Recommendation(
            top_option=top_option.option,
            rationale=rationale,
            confidence=confidence,
            next_steps=next_steps[:4],  # Max 4 next steps
        )
