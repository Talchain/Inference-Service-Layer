"""
Multi-Criteria Aggregation Service.

Implements three aggregation methods for combining scores across multiple criteria:
- weighted_sum: Compensatory, allows trade-offs
- weighted_product: Balanced, penalizes low scores
- lexicographic: Hierarchical, no trade-offs

Includes weight normalization and trade-off detection.
"""

import logging
import math
from typing import Dict, List, Tuple

from src.models.requests import MultiCriteriaRequest, OptionScore
from src.models.responses import AggregatedRanking, TradeOff, ValidationWarning
from src.utils.request_validators import normalize_weights, validate_weights_sum

logger = logging.getLogger(__name__)


class MultiCriteriaAggregator:
    """Service for aggregating multi-criteria option scores."""

    def aggregate(
        self,
        request: MultiCriteriaRequest
    ) -> Tuple[List[AggregatedRanking], List[TradeOff], List[ValidationWarning]]:
        """
        Aggregate option scores across multiple criteria.

        Args:
            request: Multi-criteria aggregation request

        Returns:
            Tuple of (aggregated_rankings, trade_offs, warnings)
        """
        logger.info(
            "aggregation_start",
            extra={
                "num_criteria": len(request.criteria),
                "method": request.aggregation_method,
                "percentile": request.percentile
            }
        )

        warnings = []

        # Normalize weights if needed
        weights = request.weights
        is_valid, weight_sum = validate_weights_sum(weights)
        if not is_valid:
            normalized_weights = normalize_weights(weights)
            warnings.append(
                ValidationWarning(
                    code="WEIGHTS_NORMALIZED",
                    message=f"Weights normalized from sum={weight_sum:.4f} to sum=1.0",
                    affected_items=list(weights.keys())
                )
            )
            weights = normalized_weights

        # Extract scores for selected percentile
        scores_by_option = self._extract_scores(request.criteria, request.percentile)

        # Apply aggregation method
        if request.aggregation_method == "weighted_sum":
            rankings = self._weighted_sum(scores_by_option, weights)
        elif request.aggregation_method == "weighted_product":
            rankings = self._weighted_product(scores_by_option, weights)
        elif request.aggregation_method == "lexicographic":
            rankings = self._lexicographic(scores_by_option, weights, list(weights.keys()))
        else:
            raise ValueError(f"Unknown aggregation method: {request.aggregation_method}")

        # Detect trade-offs
        trade_offs = self._detect_trade_offs(
            rankings,
            scores_by_option,
            request.trade_off_threshold
        )

        logger.info(
            "aggregation_complete",
            extra={
                "num_options": len(rankings),
                "num_trade_offs": len(trade_offs),
                "num_warnings": len(warnings)
            }
        )

        return rankings, trade_offs, warnings

    def _extract_scores(
        self,
        criteria: List,
        percentile: str
    ) -> Dict[str, Dict[str, Tuple[str, float]]]:
        """
        Extract scores for the selected percentile.

        Args:
            criteria: List of CriterionResult objects
            percentile: Which percentile to use (p10, p50, p90)

        Returns:
            Dict mapping option_id -> criterion_id -> (label, score)
        """
        scores_by_option: Dict[str, Dict[str, Tuple[str, float]]] = {}

        for criterion in criteria:
            for option in criterion.options:
                if option.option_id not in scores_by_option:
                    scores_by_option[option.option_id] = {}

                # Extract score for selected percentile
                if percentile == "p10":
                    score = option.p10
                elif percentile == "p50":
                    score = option.p50
                elif percentile == "p90":
                    score = option.p90
                else:
                    score = option.p50  # Default

                scores_by_option[option.option_id][criterion.criterion_id] = (
                    option.option_label,
                    score
                )

        return scores_by_option

    def _weighted_sum(
        self,
        scores_by_option: Dict[str, Dict[str, Tuple[str, float]]],
        weights: Dict[str, float]
    ) -> List[AggregatedRanking]:
        """
        Weighted sum aggregation: Score = Σ(weight_i × score_i) × 100

        Compensatory method: high scores can compensate for low scores.
        """
        rankings = []

        for option_id, criterion_scores in scores_by_option.items():
            option_label = next(iter(criterion_scores.values()))[0]

            # Compute weighted sum
            aggregated_score = sum(
                weights.get(crit_id, 0) * score
                for crit_id, (_, score) in criterion_scores.items()
            )

            # Scale to 0-100
            aggregated_score *= 100

            scores_dict = {
                crit_id: score
                for crit_id, (_, score) in criterion_scores.items()
            }

            rankings.append(
                AggregatedRanking(
                    option_id=option_id,
                    option_label=option_label,
                    rank=0,  # Will be set after sorting
                    aggregated_score=aggregated_score,
                    scores_by_criterion=scores_dict
                )
            )

        # Sort by score (descending) and assign ranks
        rankings.sort(key=lambda x: x.aggregated_score, reverse=True)
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings

    def _weighted_product(
        self,
        scores_by_option: Dict[str, Dict[str, Tuple[str, float]]],
        weights: Dict[str, float]
    ) -> List[AggregatedRanking]:
        """
        Weighted product aggregation: Score = ∏(score_i ^ weight_i) × 100

        Balanced method: all criteria must be decent, penalizes low scores.
        """
        rankings = []

        for option_id, criterion_scores in scores_by_option.items():
            option_label = next(iter(criterion_scores.values()))[0]

            # Compute weighted product
            aggregated_score = 1.0
            for crit_id, (_, score) in criterion_scores.items():
                weight = weights.get(crit_id, 0)
                # Handle zero scores (use small epsilon to avoid math errors)
                safe_score = max(score, 0.001)
                aggregated_score *= math.pow(safe_score, weight)

            # Scale to 0-100
            aggregated_score *= 100

            scores_dict = {
                crit_id: score
                for crit_id, (_, score) in criterion_scores.items()
            }

            rankings.append(
                AggregatedRanking(
                    option_id=option_id,
                    option_label=option_label,
                    rank=0,
                    aggregated_score=aggregated_score,
                    scores_by_criterion=scores_dict
                )
            )

        # Sort by score (descending) and assign ranks
        rankings.sort(key=lambda x: x.aggregated_score, reverse=True)
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings

    def _lexicographic(
        self,
        scores_by_option: Dict[str, Dict[str, Tuple[str, float]]],
        weights: Dict[str, float],
        criterion_order: List[str]
    ) -> List[AggregatedRanking]:
        """
        Lexicographic aggregation: Sort by criterion_1, then criterion_2, etc.

        Non-compensatory method: criteria are in strict hierarchy based on weights.
        Higher weight = higher priority.
        """
        # Sort criteria by weight (descending) to determine priority order
        sorted_criteria = sorted(
            criterion_order,
            key=lambda c: weights.get(c, 0),
            reverse=True
        )

        # Build list of options with scores
        option_list = []
        for option_id, criterion_scores in scores_by_option.items():
            option_label = next(iter(criterion_scores.values()))[0]
            scores_dict = {
                crit_id: score
                for crit_id, (_, score) in criterion_scores.items()
            }
            option_list.append((option_id, option_label, scores_dict))

        # Sort lexicographically
        def lex_key(item) -> tuple:
            _, _, scores = item
            return tuple(scores.get(crit, 0) for crit in sorted_criteria)

        option_list.sort(key=lex_key, reverse=True)

        # Build rankings (score is based on top criterion for interpretability)
        rankings = []
        top_criterion = sorted_criteria[0]

        for i, (option_id, option_label, scores_dict) in enumerate(option_list):
            # Use top criterion score as the "aggregated" score for interpretability
            aggregated_score = scores_dict.get(top_criterion, 0) * 100

            rankings.append(
                AggregatedRanking(
                    option_id=option_id,
                    option_label=option_label,
                    rank=i + 1,
                    aggregated_score=aggregated_score,
                    scores_by_criterion=scores_dict
                )
            )

        return rankings

    def _detect_trade_offs(
        self,
        rankings: List[AggregatedRanking],
        scores_by_option: Dict[str, Dict[str, Tuple[str, float]]],
        threshold: float
    ) -> List[TradeOff]:
        """
        Detect significant trade-offs between top options.

        A trade-off exists when options differ significantly on criteria.
        """
        trade_offs = []

        # Compare top 5 options pairwise
        top_options = rankings[:5]

        for i, opt_a in enumerate(top_options):
            for opt_b in top_options[i+1:]:
                a_better = []
                b_better = []
                max_diff = 0.0

                # Compare on each criterion
                for crit_id in opt_a.scores_by_criterion.keys():
                    score_a = opt_a.scores_by_criterion.get(crit_id, 0)
                    score_b = opt_b.scores_by_criterion.get(crit_id, 0)

                    diff = abs(score_a - score_b)
                    max_diff = max(max_diff, diff)

                    if score_a > score_b + threshold:
                        a_better.append(crit_id)
                    elif score_b > score_a + threshold:
                        b_better.append(crit_id)

                # Only report if there's a significant trade-off
                if a_better and b_better and max_diff >= threshold:
                    trade_offs.append(
                        TradeOff(
                            option_a_id=opt_a.option_id,
                            option_a_label=opt_a.option_label,
                            option_b_id=opt_b.option_id,
                            option_b_label=opt_b.option_label,
                            a_better_on=sorted(a_better),
                            b_better_on=sorted(b_better),
                            max_difference=max_diff
                        )
                    )

        return trade_offs
