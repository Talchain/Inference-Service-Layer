"""
Threshold identification service for parameter sensitivity analysis.

Detects parameter values where option rankings change, enabling
users to understand critical decision points and parameter sensitivity.
"""

from typing import Dict, List, Optional, Set, Tuple

from src.models.requests import ParameterSweep
from src.models.responses import ParameterSensitivity, RankingThreshold


class ThresholdIdentifier:
    """
    Identifies parameter thresholds where option rankings change.

    Analyzes parameter sweeps to find critical values where the
    relative ranking of options shifts, indicating decision sensitivity.
    """

    def identify(
        self,
        parameter_sweeps: List[ParameterSweep],
        baseline_ranking: Optional[List[str]],
        confidence_threshold: float
    ) -> Tuple[List[RankingThreshold], List[ParameterSensitivity], int, List[str]]:
        """
        Identify thresholds across all parameter sweeps.

        Compares rankings at each parameter value against a baseline to detect
        where option rankings change. The baseline is either provided explicitly
        or derived from the first parameter value.

        Args:
            parameter_sweeps: List of parameter sweeps to analyze
            baseline_ranking: Optional baseline ranking to compare against.
                If provided, first threshold is where ranking diverges from baseline.
                If None, uses ranking at first parameter value as baseline.
            confidence_threshold: Minimum score difference to report change

        Returns:
            Tuple of:
            - List of RankingThreshold objects
            - List of ParameterSensitivity objects (sorted by sensitivity)
            - Total threshold count
            - List of monotonic parameter IDs
        """
        all_thresholds = []
        sensitivity_data = []
        monotonic_params = []

        for sweep in parameter_sweeps:
            # Identify thresholds for this parameter
            thresholds = self._identify_parameter_thresholds(
                sweep=sweep,
                baseline_ranking=baseline_ranking,
                confidence_threshold=confidence_threshold
            )

            all_thresholds.extend(thresholds)

            # Compute sensitivity metrics
            if len(thresholds) == 0:
                monotonic_params.append(sweep.parameter_id)
                sensitivity_score = 0.0
                sensitive_range = None
            else:
                # Sensitivity score based on number of changes relative to sweep length
                max_possible_changes = len(sweep.values) - 1
                sensitivity_score = min(1.0, len(thresholds) / max_possible_changes)

                # Most sensitive range = min to max threshold values
                threshold_values = [t.threshold_value for t in thresholds]
                sensitive_range = [min(threshold_values), max(threshold_values)]

            sensitivity_data.append(ParameterSensitivity(
                parameter_id=sweep.parameter_id,
                parameter_label=sweep.parameter_label,
                changes_count=len(thresholds),
                most_sensitive_range=sensitive_range,
                sensitivity_score=sensitivity_score
            ))

        # Sort sensitivity ranking by changes_count (descending)
        sensitivity_data.sort(key=lambda x: x.changes_count, reverse=True)

        return (
            all_thresholds,
            sensitivity_data,
            len(all_thresholds),
            monotonic_params
        )

    def _identify_parameter_thresholds(
        self,
        sweep: ParameterSweep,
        baseline_ranking: Optional[List[str]],
        confidence_threshold: float
    ) -> List[RankingThreshold]:
        """
        Identify thresholds for a single parameter sweep.

        Processes parameter values in the order provided by the caller
        (ascending, descending, or custom order). Caller is responsible
        for ordering values appropriately for their use case.

        Args:
            sweep: Parameter sweep to analyze
            baseline_ranking: Optional baseline ranking to compare against.
                If provided, used as initial comparison point.
                If None, ranking at first value is used as baseline.
            confidence_threshold: Minimum score difference to report

        Returns:
            List of RankingThreshold objects for this parameter
        """
        thresholds = []

        # Process values in caller-provided order (no sorting)
        values = sweep.values

        # Initialize comparison ranking
        if baseline_ranking is not None:
            # Use provided baseline as initial comparison point
            prev_ranking = baseline_ranking
            start_index = 0
        else:
            # Use ranking at first value as baseline
            first_value_str = str(values[0])
            first_scores = sweep.scores_by_value[first_value_str]
            prev_ranking = self._rank_options(first_scores, confidence_threshold)
            start_index = 1

        # Compare each subsequent value against previous ranking
        for i in range(start_index, len(values)):
            value = values[i]

            # Get scores at this parameter value
            value_str = str(value)
            scores = sweep.scores_by_value[value_str]

            # Rank options by score (descending)
            current_ranking = self._rank_options(scores, confidence_threshold)

            # Compare to previous ranking
            if current_ranking != prev_ranking:
                # Ranking changed - this is a threshold
                affected_options = self._find_affected_options(
                    prev_ranking, current_ranking
                )

                # Compute score gap between most affected options
                score_gap = self._compute_score_gap(
                    scores, affected_options
                )

                thresholds.append(RankingThreshold(
                    parameter_id=sweep.parameter_id,
                    parameter_label=sweep.parameter_label,
                    threshold_value=value,
                    ranking_before=prev_ranking,
                    ranking_after=current_ranking,
                    options_affected=affected_options,
                    score_gap=score_gap
                ))

            prev_ranking = current_ranking

        return thresholds

    def _rank_options(
        self,
        scores: Dict[str, float],
        confidence_threshold: float
    ) -> List[str]:
        """
        Rank options by score, handling ties with confidence threshold.

        Options within confidence_threshold of each other are considered tied
        and maintain stable ordering (alphabetical by option_id).

        Args:
            scores: Dictionary of option_id -> score
            confidence_threshold: Threshold for considering scores equal

        Returns:
            List of option_ids in rank order (best to worst)
        """
        # Sort by score (descending), then by option_id (for stable tie-breaking)
        sorted_items = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0])  # Negative score for descending
        )

        # Group options that are within confidence_threshold
        # This creates stable rankings even with small score differences
        ranked_options = []
        i = 0

        while i < len(sorted_items):
            option_id, score = sorted_items[i]
            group = [option_id]

            # Check if next options are within confidence threshold
            j = i + 1
            while j < len(sorted_items):
                next_option_id, next_score = sorted_items[j]
                if abs(score - next_score) <= confidence_threshold:
                    group.append(next_option_id)
                    j += 1
                else:
                    break

            # Sort group alphabetically for stable ordering
            group.sort()
            ranked_options.extend(group)
            i = j

        return ranked_options

    def _find_affected_options(
        self,
        ranking_before: List[str],
        ranking_after: List[str]
    ) -> List[str]:
        """
        Find which options changed position between rankings.

        Args:
            ranking_before: Previous ranking
            ranking_after: Current ranking

        Returns:
            List of option_ids that changed position
        """
        affected = []

        # Pre-build position lookup dictionaries for O(1) access
        # This optimization reduces complexity from O(n²) to O(n)
        position_before = {option_id: idx for idx, option_id in enumerate(ranking_before)}
        position_after = {option_id: idx for idx, option_id in enumerate(ranking_after)}

        for option_id in ranking_before:
            idx_before = position_before[option_id]
            idx_after = position_after.get(option_id)

            # Check if option exists in both rankings and changed position
            if idx_after is not None and idx_before != idx_after:
                affected.append(option_id)

        return affected

    def _compute_score_gap(
        self,
        scores: Dict[str, float],
        affected_options: List[str]
    ) -> Optional[float]:
        """
        Compute score gap between affected options.

        Returns the maximum pairwise score difference among affected options.

        Args:
            scores: Current scores
            affected_options: Options that changed rank

        Returns:
            Maximum score difference, or None if < 2 affected options
        """
        if len(affected_options) < 2:
            return None

        # Get scores for affected options
        affected_scores = [scores[opt_id] for opt_id in affected_options]

        # Compute max pairwise difference
        max_gap = 0.0
        for i in range(len(affected_scores)):
            for j in range(i + 1, len(affected_scores)):
                gap = abs(affected_scores[i] - affected_scores[j])
                max_gap = max(max_gap, gap)

        return max_gap


# Global instance
threshold_identifier = ThresholdIdentifier()
