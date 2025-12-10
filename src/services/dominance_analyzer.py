"""
Dominance Detection Service.

Analyzes options to identify dominance relationships and Pareto frontier.
An option B dominates option A if:
- B is better or equal to A on all criteria AND
- B is strictly better than A on at least one criterion
"""

import logging
from typing import Dict, List, Set, Tuple

from src.models.requests import DominanceOption
from src.models.responses import DominanceRelation

logger = logging.getLogger(__name__)


class DominanceAnalyzer:
    """Service for detecting dominance relationships between options."""

    def analyze(
        self,
        options: List[DominanceOption],
        criteria: List[str]
    ) -> Tuple[List[DominanceRelation], List[str]]:
        """
        Analyze options for dominance relationships.

        Args:
            options: List of options with scores on multiple criteria
            criteria: List of criterion IDs to consider

        Returns:
            Tuple of (dominated_relations, non_dominated_ids)
            - dominated_relations: List of DominanceRelation objects
            - non_dominated_ids: List of option IDs on Pareto frontier

        Algorithm:
            O(n²) pairwise comparison where n = number of options
            For each pair (A, B), check if B dominates A
        """
        logger.info(
            "dominance_analysis_start",
            extra={
                "num_options": len(options),
                "num_criteria": len(criteria)
            }
        )

        # Build dominance relationships
        dominance_map: Dict[str, Set[str]] = {}  # dominated_id -> set of dominating_ids

        for i, option_a in enumerate(options):
            for option_b in options[i+1:]:
                # Check if B dominates A
                if self._dominates(option_b, option_a, criteria):
                    if option_a.option_id not in dominance_map:
                        dominance_map[option_a.option_id] = set()
                    dominance_map[option_a.option_id].add(option_b.option_id)

                # Check if A dominates B
                if self._dominates(option_a, option_b, criteria):
                    if option_b.option_id not in dominance_map:
                        dominance_map[option_b.option_id] = set()
                    dominance_map[option_b.option_id].add(option_a.option_id)

        # Build dominated relations
        dominated_relations = []
        option_lookup = {opt.option_id: opt for opt in options}

        for dominated_id, dominating_ids in dominance_map.items():
            dominated_option = option_lookup[dominated_id]
            dominated_relations.append(
                DominanceRelation(
                    dominated_option_id=dominated_id,
                    dominated_option_label=dominated_option.option_label,
                    dominated_by=sorted(list(dominating_ids)),  # Sorted for determinism
                    degree=len(dominating_ids)
                )
            )

        # Sort by degree (most dominated first)
        dominated_relations.sort(key=lambda x: x.degree, reverse=True)

        # Identify non-dominated options (Pareto frontier)
        all_option_ids = {opt.option_id for opt in options}
        dominated_ids = set(dominance_map.keys())
        non_dominated_ids = sorted(list(all_option_ids - dominated_ids))  # Sorted for determinism

        logger.info(
            "dominance_analysis_complete",
            extra={
                "num_dominated": len(dominated_relations),
                "frontier_size": len(non_dominated_ids)
            }
        )

        return dominated_relations, non_dominated_ids

    def _dominates(
        self,
        option_a: DominanceOption,
        option_b: DominanceOption,
        criteria: List[str]
    ) -> bool:
        """
        Check if option_a dominates option_b.

        Dominance definition:
        - A dominates B if A >= B on all criteria AND A > B on at least one

        Args:
            option_a: First option (potential dominator)
            option_b: Second option (potentially dominated)
            criteria: List of criteria to compare

        Returns:
            True if option_a dominates option_b, False otherwise
        """
        better_or_equal_on_all = True
        strictly_better_on_one = False

        for criterion in criteria:
            score_a = option_a.scores[criterion]
            score_b = option_b.scores[criterion]

            if score_a < score_b:
                # A is worse than B on this criterion -> cannot dominate
                better_or_equal_on_all = False
                break

            if score_a > score_b:
                # A is strictly better than B on this criterion
                strictly_better_on_one = True

        return better_or_equal_on_all and strictly_better_on_one

    def get_frontier_size(self, options: List[DominanceOption], criteria: List[str]) -> int:
        """
        Get the size of the Pareto frontier without computing full dominance.

        Useful for quick checks or validation.

        Args:
            options: List of options
            criteria: List of criteria

        Returns:
            Number of non-dominated options
        """
        _, non_dominated_ids = self.analyze(options, criteria)
        return len(non_dominated_ids)

    def is_dominated(
        self,
        target_option: DominanceOption,
        other_options: List[DominanceOption],
        criteria: List[str]
    ) -> bool:
        """
        Check if a specific option is dominated by any other option.

        Args:
            target_option: Option to check
            other_options: Other options to compare against
            criteria: List of criteria

        Returns:
            True if target_option is dominated by at least one other option
        """
        for other in other_options:
            if other.option_id == target_option.option_id:
                continue

            if self._dominates(other, target_option, criteria):
                return True

        return False
