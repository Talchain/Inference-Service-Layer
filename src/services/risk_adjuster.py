"""
Risk adjustment service for options under uncertainty.

Implements mean-variance approach to compute certainty equivalents
based on user's risk profile (averse, neutral, or seeking).
"""

import math
from typing import List, Tuple

from src.models.requests import RiskOption
from src.models.responses import AdjustedScore, RankingChange


class RiskAdjuster:
    """
    Risk adjustment using mean-variance approach.

    Computes certainty equivalents based on risk attitude:
    - Risk averse: CE = mean - (coefficient/2) × variance
    - Risk neutral: CE = mean
    - Risk seeking: CE = mean + (coefficient/2) × variance
    """

    def adjust(
        self,
        options: List[RiskOption],
        risk_coefficient: float,
        risk_type: str
    ) -> Tuple[List[AdjustedScore], bool, List[RankingChange], str]:
        """
        Apply risk adjustment to options.

        Args:
            options: Options with uncertainty (mean/std_dev or percentiles)
            risk_coefficient: Risk coefficient (>0 for risk aversion)
            risk_type: 'risk_averse', 'risk_neutral', or 'risk_seeking'

        Returns:
            Tuple of:
            - List of AdjustedScore (sorted by CE, descending)
            - Boolean indicating if rankings changed
            - List of RankingChange details (if rankings changed)
            - Plain English interpretation string
        """
        # Extract mean and variance for each option
        scores_data = []
        for option in options:
            mean, variance = self._extract_mean_variance(option)
            scores_data.append({
                'option_id': option.option_id,
                'option_label': option.option_label,
                'mean': mean,
                'variance': variance
            })

        # Compute certainty equivalents
        adjusted_scores = []
        for data in scores_data:
            ce = self._compute_certainty_equivalent(
                mean=data['mean'],
                variance=data['variance'],
                coefficient=risk_coefficient,
                risk_type=risk_type
            )

            # Clamp CE to [0, 1] range
            ce = max(0.0, min(1.0, ce))

            adjusted_scores.append(AdjustedScore(
                option_id=data['option_id'],
                option_label=data['option_label'],
                original_score=data['mean'],
                certainty_equivalent=ce,
                adjustment=ce - data['mean'],
                variance=data['variance'] if data['variance'] > 0 else None
            ))

        # Sort by original score (descending) to determine original ranks
        original_ranking = sorted(
            adjusted_scores,
            key=lambda x: x.original_score,
            reverse=True
        )
        original_ranks = {
            score.option_id: rank + 1
            for rank, score in enumerate(original_ranking)
        }

        # Sort by certainty equivalent (descending) to determine adjusted ranks
        adjusted_scores.sort(key=lambda x: x.certainty_equivalent, reverse=True)
        adjusted_ranks = {
            score.option_id: rank + 1
            for rank, score in enumerate(adjusted_scores)
        }

        # Check if rankings changed
        rankings_changed = original_ranks != adjusted_ranks

        # Compute ranking changes if changed
        ranking_changes = []
        if rankings_changed:
            for option_id in original_ranks:
                orig_rank = original_ranks[option_id]
                adj_rank = adjusted_ranks[option_id]
                if orig_rank != adj_rank:
                    # Find option label
                    option_label = next(
                        s.option_label for s in adjusted_scores
                        if s.option_id == option_id
                    )
                    ranking_changes.append(RankingChange(
                        option_id=option_id,
                        option_label=option_label,
                        original_rank=orig_rank,
                        adjusted_rank=adj_rank,
                        rank_change=orig_rank - adj_rank  # Positive = improved
                    ))

            # Sort by absolute rank change (largest changes first)
            ranking_changes.sort(key=lambda x: abs(x.rank_change), reverse=True)

        # Generate interpretation
        interpretation = self._generate_interpretation(
            adjusted_scores=adjusted_scores,
            risk_coefficient=risk_coefficient,
            risk_type=risk_type,
            rankings_changed=rankings_changed,
            ranking_changes=ranking_changes
        )

        return adjusted_scores, rankings_changed, ranking_changes, interpretation

    def _extract_mean_variance(self, option: RiskOption) -> Tuple[float, float]:
        """
        Extract mean and variance from option.

        Supports both mean/std_dev and percentile representations.
        For percentiles, approximates mean and variance using:
        - mean ≈ p50
        - variance ≈ ((p90 - p10) / 2.56)²  (assuming normal distribution)

        Args:
            option: Risk option with either (mean, std_dev) or (p10, p50, p90)

        Returns:
            Tuple of (mean, variance)
        """
        if option.mean is not None and option.std_dev is not None:
            # Direct mean-variance representation
            return option.mean, option.std_dev ** 2

        elif option.p10 is not None and option.p50 is not None and option.p90 is not None:
            # Percentile representation - approximate mean and variance
            # For normal distribution:
            # - p50 is the median (≈ mean for symmetric distributions)
            # - (p90 - p10) ≈ 2.56 standard deviations (90% - 10% = 80% interval)
            mean = option.p50
            std_dev_approx = (option.p90 - option.p10) / 2.56
            variance = std_dev_approx ** 2
            return mean, variance

        else:
            # Should not reach here due to validation
            raise ValueError(
                f"Option {option.option_id} must have either (mean, std_dev) "
                "or (p10, p50, p90)"
            )

    def _compute_certainty_equivalent(
        self,
        mean: float,
        variance: float,
        coefficient: float,
        risk_type: str
    ) -> float:
        """
        Compute certainty equivalent using mean-variance approach.

        Args:
            mean: Expected value
            variance: Variance of distribution
            coefficient: Risk coefficient
            risk_type: 'risk_averse', 'risk_neutral', or 'risk_seeking'

        Returns:
            Certainty equivalent score
        """
        if risk_type == "risk_neutral":
            # Risk neutral: CE = mean (no adjustment)
            return mean

        elif risk_type == "risk_averse":
            # Risk averse: CE = mean - (coefficient/2) × variance
            # Penalizes variance (uncertainty)
            adjustment = (coefficient / 2.0) * variance
            return mean - adjustment

        elif risk_type == "risk_seeking":
            # Risk seeking: CE = mean + (coefficient/2) × variance
            # Rewards variance (uncertainty)
            adjustment = (coefficient / 2.0) * variance
            return mean + adjustment

        else:
            raise ValueError(f"Unknown risk_type: {risk_type}")

    def _generate_interpretation(
        self,
        adjusted_scores: List[AdjustedScore],
        risk_coefficient: float,
        risk_type: str,
        rankings_changed: bool,
        ranking_changes: List[RankingChange]
    ) -> str:
        """
        Generate plain English interpretation of risk adjustment.

        Args:
            adjusted_scores: Adjusted scores sorted by CE
            risk_coefficient: Risk coefficient used
            risk_type: Risk type used
            rankings_changed: Whether rankings changed
            ranking_changes: Ranking change details

        Returns:
            Interpretation string
        """
        # Build interpretation parts
        parts = []

        # Risk attitude description
        if risk_type == "risk_averse":
            parts.append(
                f"Risk aversion (coefficient={risk_coefficient:.1f}) penalizes "
                "high-variance options."
            )
        elif risk_type == "risk_seeking":
            parts.append(
                f"Risk seeking (coefficient={risk_coefficient:.1f}) rewards "
                "high-variance options."
            )
        else:
            parts.append(
                "Risk neutral attitude: rankings based solely on expected values "
                "(no adjustment for variance)."
            )

        # Ranking changes
        if rankings_changed and ranking_changes:
            # Describe top ranking changes
            top_changes = ranking_changes[:2]  # Top 2 changes

            for change in top_changes:
                direction = "improves" if change.rank_change > 0 else "drops"
                parts.append(
                    f"{change.option_label} {direction} from rank "
                    f"{change.original_rank} to rank {change.adjusted_rank}"
                )

            # Add variance context for the most significant change
            if top_changes:
                top_change = top_changes[0]
                top_option = next(
                    s for s in adjusted_scores
                    if s.option_id == top_change.option_id
                )
                if top_option.variance is not None and top_option.variance > 0:
                    parts[-1] += f" due to variance={top_option.variance:.4f}"

        else:
            parts.append("Rankings unchanged after risk adjustment.")

        return " ".join(parts)


# Global instance
risk_adjuster = RiskAdjuster()
