"""
Coherence Analyzer Service.

Provides coherence checks for inference results including:
- Detection of negative expected value for top option
- Close race detection (options within threshold)
- Ranking instability under perturbations
"""

import logging
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from src.models.requests import CoherenceAnalysisRequest, RankedOption
from src.models.responses import (
    CoherenceAnalysis,
    CoherenceAnalysisResponse,
    PerturbationResult,
    RankingStability,
    StabilityAnalysis,
)

logger = logging.getLogger(__name__)


class CoherenceAnalyzer:
    """
    Service for analyzing coherence of inference results.

    Performs checks for:
    - Negative expected value at top
    - Close races between options
    - Ranking stability under perturbations
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the coherence analyzer.

        Args:
            seed: Optional random seed for reproducibility
        """
        self._rng = np.random.default_rng(seed)

    def analyze(self, request: CoherenceAnalysisRequest) -> CoherenceAnalysisResponse:
        """
        Analyze coherence of ranked options.

        Args:
            request: Coherence analysis request

        Returns:
            CoherenceAnalysisResponse with analysis results
        """
        # Sort options by expected value (descending)
        sorted_options = sorted(
            request.options, key=lambda x: x.expected_value, reverse=True
        )

        # Get top option and second option
        top_option = sorted_options[0]
        second_option = sorted_options[1] if len(sorted_options) > 1 else None

        # Core coherence analysis
        coherence_analysis = self._analyze_coherence(
            top_option, second_option, request.close_race_threshold
        )

        # Stability analysis through perturbations
        stability_analysis = self._analyze_stability(
            sorted_options,
            request.perturbation_magnitude,
            request.num_perturbations,
        )

        # Update coherence with stability info
        coherence_analysis = self._update_coherence_with_stability(
            coherence_analysis, stability_analysis
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            coherence_analysis, stability_analysis, sorted_options
        )

        return CoherenceAnalysisResponse(
            coherence_analysis=coherence_analysis,
            stability_analysis=stability_analysis,
            recommendations=recommendations,
        )

    def _analyze_coherence(
        self,
        top_option: RankedOption,
        second_option: Optional[RankedOption],
        close_race_threshold: float,
    ) -> CoherenceAnalysis:
        """
        Analyze basic coherence metrics.

        Args:
            top_option: The top-ranked option
            second_option: The second-ranked option (if exists)
            close_race_threshold: Threshold for detecting close races

        Returns:
            CoherenceAnalysis with basic metrics
        """
        warnings = []

        # Check 1: Top option has negative expected value
        top_positive = top_option.expected_value > 0
        if not top_positive:
            warnings.append(
                f"Top option '{top_option.name}' has negative expected value "
                f"({top_option.expected_value:.2f})"
            )

        # Check 2: Margin to second option
        if second_option:
            margin = top_option.expected_value - second_option.expected_value

            # Calculate relative margin (percentage)
            if top_option.expected_value != 0:
                margin_pct = abs(margin / top_option.expected_value) * 100
            else:
                margin_pct = 0.0 if margin == 0 else 100.0

            # Check for close race
            if abs(top_option.expected_value) > 0:
                relative_margin = abs(margin) / abs(top_option.expected_value)
            else:
                relative_margin = 0.0 if margin == 0 else float('inf')

            if relative_margin < close_race_threshold:
                warnings.append(
                    f"Close race detected: '{top_option.name}' leads '{second_option.name}' "
                    f"by only {margin_pct:.1f}%"
                )
        else:
            margin = 0.0
            margin_pct = 0.0

        # Initial stability (will be updated after perturbation analysis)
        return CoherenceAnalysis(
            top_option_positive=top_positive,
            margin_to_second=margin,
            margin_to_second_pct=margin_pct,
            ranking_stability=RankingStability.STABLE,  # Placeholder
            stability_score=1.0,  # Placeholder
            warnings=warnings,
        )

    def _analyze_stability(
        self,
        options: List[RankedOption],
        perturbation_magnitude: float,
        num_perturbations: int,
    ) -> StabilityAnalysis:
        """
        Analyze ranking stability under perturbations.

        Applies random perturbations to expected values and checks
        how often rankings change.

        Args:
            options: Sorted list of options
            perturbation_magnitude: Magnitude of perturbations (relative)
            num_perturbations: Number of perturbation scenarios

        Returns:
            StabilityAnalysis with perturbation results
        """
        if len(options) < 2:
            return StabilityAnalysis(
                num_perturbations=0,
                ranking_changes=0,
                ranking_change_rate=0.0,
                most_frequent_alternative=None,
                sample_perturbations=[],
            )

        baseline_top = options[0].option_id
        ranking_changes = 0
        alternative_tops: List[str] = []
        sample_perturbations: List[PerturbationResult] = []

        for i in range(num_perturbations):
            # Apply perturbations to all options
            perturbed_values = []
            for opt in options:
                # Perturbation proportional to value magnitude
                if opt.expected_value != 0:
                    std = abs(opt.expected_value) * perturbation_magnitude
                else:
                    std = perturbation_magnitude

                perturbed_value = opt.expected_value + self._rng.normal(0, std)
                perturbed_values.append((opt.option_id, perturbed_value))

            # Find new top option
            perturbed_values.sort(key=lambda x: x[1], reverse=True)
            new_top_id = perturbed_values[0][0]
            new_top_value = perturbed_values[0][1]

            changed = new_top_id != baseline_top
            if changed:
                ranking_changes += 1
                alternative_tops.append(new_top_id)

            # Calculate value change for top option
            original_top_value = options[0].expected_value
            if original_top_value != 0:
                value_change_pct = ((new_top_value - original_top_value) / abs(original_top_value)) * 100
            else:
                value_change_pct = 0.0

            # Store sample perturbations (first 10)
            if i < 10:
                sample_perturbations.append(
                    PerturbationResult(
                        perturbation_id=i + 1,
                        top_option_id=new_top_id,
                        ranking_changed=changed,
                        value_change_pct=round(value_change_pct, 2),
                    )
                )

        # Find most frequent alternative
        most_frequent = None
        if alternative_tops:
            counter = Counter(alternative_tops)
            most_frequent = counter.most_common(1)[0][0]

        ranking_change_rate = ranking_changes / num_perturbations if num_perturbations > 0 else 0.0

        return StabilityAnalysis(
            num_perturbations=num_perturbations,
            ranking_changes=ranking_changes,
            ranking_change_rate=round(ranking_change_rate, 4),
            most_frequent_alternative=most_frequent,
            sample_perturbations=sample_perturbations,
        )

    def _update_coherence_with_stability(
        self, coherence: CoherenceAnalysis, stability: StabilityAnalysis
    ) -> CoherenceAnalysis:
        """
        Update coherence analysis with stability results.

        Args:
            coherence: Initial coherence analysis
            stability: Stability analysis results

        Returns:
            Updated CoherenceAnalysis
        """
        # Determine ranking stability classification
        rate = stability.ranking_change_rate

        if rate < 0.1:
            ranking_stability = RankingStability.STABLE
            stability_score = 1.0 - rate
        elif rate < 0.3:
            ranking_stability = RankingStability.SENSITIVE
            stability_score = 0.8 - (rate - 0.1) * 2  # Scale 0.8 to 0.4
        else:
            ranking_stability = RankingStability.UNSTABLE
            stability_score = max(0.0, 0.4 - (rate - 0.3) * 0.6)  # Scale 0.4 to 0.0

        # Add instability warning if needed
        warnings = list(coherence.warnings)
        if ranking_stability == RankingStability.UNSTABLE:
            warnings.append(
                f"Rankings are unstable: {rate*100:.1f}% of perturbations changed top option"
            )
        elif ranking_stability == RankingStability.SENSITIVE:
            warnings.append(
                f"Rankings are sensitive: {rate*100:.1f}% of perturbations changed top option"
            )

        return CoherenceAnalysis(
            top_option_positive=coherence.top_option_positive,
            margin_to_second=coherence.margin_to_second,
            margin_to_second_pct=coherence.margin_to_second_pct,
            ranking_stability=ranking_stability,
            stability_score=round(stability_score, 3),
            warnings=warnings,
        )

    def _generate_recommendations(
        self,
        coherence: CoherenceAnalysis,
        stability: StabilityAnalysis,
        options: List[RankedOption],
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis.

        Args:
            coherence: Coherence analysis results
            stability: Stability analysis results
            options: Sorted list of options

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Negative expected value recommendation
        if not coherence.top_option_positive:
            recommendations.append(
                "Consider if any option has positive expected value, "
                "or re-evaluate the decision framework"
            )

        # Close race recommendation
        if coherence.margin_to_second_pct < 10 and len(options) > 1:
            recommendations.append(
                "Top options are very close in expected value - "
                "decision may depend on risk tolerance or secondary criteria"
            )

        # Instability recommendations
        if coherence.ranking_stability == RankingStability.UNSTABLE:
            recommendations.append(
                "Rankings are unstable under small changes - "
                "consider gathering more data to reduce uncertainty"
            )
            if stability.most_frequent_alternative:
                recommendations.append(
                    f"'{stability.most_frequent_alternative}' frequently takes top spot - "
                    "evaluate it as a robust alternative"
                )
        elif coherence.ranking_stability == RankingStability.SENSITIVE:
            recommendations.append(
                "Rankings are somewhat sensitive to changes - "
                "verify key assumptions before finalizing decision"
            )

        # Confidence interval overlap recommendation
        top_ci = options[0].confidence_interval if options else None
        if top_ci and len(options) > 1:
            second_ci = options[1].confidence_interval
            if second_ci and top_ci[0] < second_ci[1]:  # Intervals overlap
                recommendations.append(
                    "Confidence intervals overlap between top options - "
                    "statistically, they may not be significantly different"
                )

        # If everything looks good
        if not recommendations:
            recommendations.append(
                "Analysis shows stable rankings with clear leader - "
                "decision appears well-supported"
            )

        return recommendations
