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
        Analyze options for dominance relationships using optimized Skyline algorithm.

        Args:
            options: List of options with scores on multiple criteria
            criteria: List of criterion IDs to consider

        Returns:
            Tuple of (dominated_relations, non_dominated_ids)
            - dominated_relations: List of DominanceRelation objects
            - non_dominated_ids: List of option IDs on Pareto frontier

        Algorithm:
            Optimized Sort-Filter-Skyline approach:
            1. Sort by sum of scores (O(n log n))
            2. Incrementally build frontier (O(n·m) where m = frontier size << n)
            3. Compare dominated points only against frontier (O((n-m)·m))

            Complexity: O(n log n + n·m) where m is typically much smaller than n
            Best case: O(n log n) when m is constant
            Worst case: O(n²) when all points are on frontier (rare in practice)
        """
        logger.info(
            "dominance_analysis_start",
            extra={
                "num_options": len(options),
                "num_criteria": len(criteria)
            }
        )

        if not options:
            return [], []

        # Step 1: Sort by sum of scores (O(n log n))
        # This heuristic places likely frontier candidates first
        sorted_options = sorted(
            options,
            key=lambda opt: sum(opt.scores.get(c, 0) for c in criteria),
            reverse=True
        )

        # Step 2: Incrementally build Pareto frontier
        frontier: List[DominanceOption] = []
        dominated_by_frontier: Dict[str, Set[str]] = {}  # dominated_id -> frontier_ids

        for option in sorted_options:
            # Check if this option is dominated by any frontier member
            dominated_by = []
            for frontier_option in frontier:
                if self._dominates(frontier_option, option, criteria):
                    dominated_by.append(frontier_option.option_id)

            if dominated_by:
                # Option is dominated - record who dominates it
                dominated_by_frontier[option.option_id] = set(dominated_by)
            else:
                # Option is not dominated by frontier - add to frontier
                # But first, check if it dominates any existing frontier members
                # (This handles cases where sorting heuristic isn't perfect)
                new_frontier = []
                for frontier_option in frontier:
                    if not self._dominates(option, frontier_option, criteria):
                        # Keep frontier option (not dominated by new option)
                        new_frontier.append(frontier_option)
                    else:
                        # New option dominates this frontier option
                        if frontier_option.option_id not in dominated_by_frontier:
                            dominated_by_frontier[frontier_option.option_id] = set()
                        dominated_by_frontier[frontier_option.option_id].add(option.option_id)

                new_frontier.append(option)
                frontier = new_frontier

        # Step 3: Build full dominance map by checking dominated options against each other
        # This is necessary to capture all dominance relationships, not just frontier-based ones
        dominance_map: Dict[str, Set[str]] = dominated_by_frontier.copy()

        dominated_options = [opt for opt in sorted_options if opt.option_id in dominated_by_frontier]

        # For dominated options, also check pairwise dominance among themselves
        # This is still faster than full O(n²) because we skip frontier comparisons
        for i, option_a in enumerate(dominated_options):
            for option_b in dominated_options[i+1:]:
                if self._dominates(option_b, option_a, criteria):
                    dominance_map[option_a.option_id].add(option_b.option_id)
                elif self._dominates(option_a, option_b, criteria):
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

        # Extract non-dominated IDs from frontier
        non_dominated_ids = sorted([opt.option_id for opt in frontier])

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
