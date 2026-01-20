"""
Conditional Recommendation Engine for Phase 4.

Generates conditional recommendations that qualify decisions based on
parameter thresholds, dominance relationships, and risk profiles.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.models.requests import ConditionalRecommendRequest, RankedOption
from src.models.responses import (
    ConditionalRecommendResponse,
    ConditionalRecommendation,
    ConditionExpression,
    PrimaryRecommendation,
    RobustnessSummary,
)
from src.models.shared import DistributionType
from src.constants import ZERO_VARIANCE_TOLERANCE

logger = logging.getLogger(__name__)


class ConditionalRecommendationEngine:
    """
    Engine for generating conditional recommendations.

    Analyzes ranked options and generates conditions under which
    the recommendation would change.
    """

    # Risk profile parameters
    RISK_AVERSE_FACTOR = 0.5  # Weight on downside
    RISK_SEEKING_FACTOR = 1.5  # Weight on upside

    # Monte Carlo settings
    DEFAULT_MC_SAMPLES = 1000

    def __init__(self):
        """Initialize the conditional recommendation engine."""
        self.logger = logger

    def generate_recommendations(
        self,
        request: ConditionalRecommendRequest
    ) -> ConditionalRecommendResponse:
        """
        Generate conditional recommendations for ranked options.

        Args:
            request: Request with ranked options and condition parameters

        Returns:
            ConditionalRecommendResponse with primary and conditional recommendations
        """
        self.logger.info(
            f"Generating conditional recommendations for {len(request.ranked_options)} options"
        )

        # Determine primary recommendation (highest expected value)
        primary_option = self._get_primary_recommendation(request.ranked_options)

        # Generate conditions based on requested types
        conditions: List[ConditionalRecommendation] = []

        if "threshold" in request.condition_types:
            threshold_conditions = self._generate_threshold_conditions(
                request.ranked_options,
                primary_option,
                request.parameters_to_condition_on
            )
            conditions.extend(threshold_conditions)

        if "dominance" in request.condition_types:
            dominance_conditions = self._generate_dominance_conditions(
                request.ranked_options,
                primary_option
            )
            conditions.extend(dominance_conditions)

        if "risk_profile" in request.condition_types:
            risk_conditions = self._generate_risk_profile_conditions(
                request.ranked_options,
                primary_option
            )
            conditions.extend(risk_conditions)

        if "scenario" in request.condition_types:
            scenario_conditions = self._generate_scenario_conditions(
                request.ranked_options,
                primary_option
            )
            conditions.extend(scenario_conditions)

        # Sort by impact and limit
        conditions = self._rank_and_limit_conditions(
            conditions,
            request.max_conditions
        )

        # Calculate robustness summary
        robustness = self._calculate_robustness_summary(
            conditions,
            request.ranked_options,
            primary_option
        )

        # Create primary recommendation
        primary_rec = PrimaryRecommendation(
            option_id=primary_option.option_id,
            label=primary_option.label,
            confidence=self._calculate_confidence(request.ranked_options, primary_option),
            expected_value=primary_option.expected_value
        )

        return ConditionalRecommendResponse(
            primary_recommendation=primary_rec,
            conditional_recommendations=conditions,
            robustness_summary=robustness
        )

    def _get_primary_recommendation(
        self,
        options: List[RankedOption]
    ) -> RankedOption:
        """Get the primary recommended option (highest expected value)."""
        return max(options, key=lambda x: x.expected_value)

    def _calculate_confidence(
        self,
        options: List[RankedOption],
        primary: RankedOption
    ) -> str:
        """
        Calculate confidence level for the primary recommendation.

        Based on gap between primary and next-best option relative to uncertainty.
        """
        if len(options) < 2:
            return "high"

        # Sort options by expected value
        sorted_options = sorted(options, key=lambda x: x.expected_value, reverse=True)

        # Get gap to second best
        gap = sorted_options[0].expected_value - sorted_options[1].expected_value

        # Get uncertainty from primary option's distribution
        primary_std = self._get_distribution_std(primary)

        # Calculate ratio (use tolerance to avoid near-zero division)
        if primary_std >= ZERO_VARIANCE_TOLERANCE:
            gap_ratio = gap / primary_std
        else:
            gap_ratio = float('inf')

        # Classify confidence
        if gap_ratio > 2.0:
            return "high"
        elif gap_ratio > 0.5:
            return "medium"
        else:
            return "low"

    def _get_distribution_std(self, option: RankedOption) -> float:
        """Extract standard deviation from option's distribution."""
        params = option.distribution.parameters

        if option.distribution.type == DistributionType.NORMAL:
            return params.get("std", params.get("scale", 0))
        elif option.distribution.type == DistributionType.UNIFORM:
            # Std of uniform is (b-a)/sqrt(12)
            low = params.get("low", params.get("a", 0))
            high = params.get("high", params.get("b", 1))
            return (high - low) / np.sqrt(12)
        elif option.distribution.type == DistributionType.BETA:
            # Use variance approximation for beta
            a = params.get("a", params.get("alpha", 2))
            b = params.get("b", params.get("beta", 2))
            variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
            return np.sqrt(variance)
        else:
            # Default fallback
            return abs(option.expected_value) * 0.1

    def _generate_threshold_conditions(
        self,
        options: List[RankedOption],
        primary: RankedOption,
        parameters: Optional[List[str]]
    ) -> List[ConditionalRecommendation]:
        """
        Generate threshold-based conditions.

        Identifies parameter thresholds where ranking would flip.
        """
        conditions = []

        # If no parameters specified, auto-detect from distribution
        if parameters is None:
            parameters = self._auto_detect_parameters(options)

        for param in parameters:
            # For each non-primary option, find the threshold where it becomes optimal
            for option in options:
                if option.option_id == primary.option_id:
                    continue

                # Calculate crossover threshold
                threshold_result = self._find_crossover_threshold(
                    primary, option, param
                )

                if threshold_result is not None:
                    threshold_value, direction, probability = threshold_result

                    # Create condition
                    operator = "<" if direction == "below" else ">"
                    description = (
                        f"If {param} {'drops below' if direction == 'below' else 'rises above'} "
                        f"{threshold_value:.2f}"
                    )

                    condition = ConditionalRecommendation(
                        condition_id=f"cond_{uuid.uuid4().hex[:8]}",
                        condition_type="threshold",
                        condition_description=description,
                        condition_expression=ConditionExpression(
                            parameter=param,
                            operator=operator,
                            value=threshold_value
                        ),
                        triggered_recommendation=PrimaryRecommendation(
                            option_id=option.option_id,
                            label=option.label,
                            confidence="medium",
                            expected_value=option.expected_value
                        ),
                        probability_of_condition=probability,
                        impact_magnitude=self._calculate_impact_magnitude(primary, option)
                    )
                    conditions.append(condition)

        return conditions

    def _auto_detect_parameters(self, options: List[RankedOption]) -> List[str]:
        """Auto-detect parameters to condition on from option distributions."""
        parameters = set()

        for option in options:
            params = option.distribution.parameters
            for key in params.keys():
                if key not in ("type", "kind"):
                    parameters.add(f"{option.option_id}_{key}")

        # Add generic parameters if nothing found
        if not parameters:
            parameters = {"mean_adjustment", "uncertainty_factor"}

        return list(parameters)[:5]  # Limit to 5

    def _find_crossover_threshold(
        self,
        primary: RankedOption,
        alternative: RankedOption,
        parameter: str
    ) -> Optional[Tuple[float, str, float]]:
        """
        Find the threshold where alternative becomes preferred.

        Returns: (threshold_value, direction, probability) or None
        """
        # Calculate the gap
        gap = primary.expected_value - alternative.expected_value

        if gap <= 0:
            # Alternative is already preferred
            return None

        # Estimate threshold based on parameter sensitivity
        # Simplified: assume linear relationship with parameter
        primary_std = self._get_distribution_std(primary)

        if primary_std < ZERO_VARIANCE_TOLERANCE:
            return None

        # Threshold where primary drops enough for crossover
        # Assuming parameter affects primary's mean
        threshold_value = primary.expected_value - gap

        # Estimate probability of condition (simplified)
        z_score = gap / primary_std
        probability = 1 - stats.norm.cdf(z_score)

        direction = "below"

        # Normalize threshold to reasonable range
        if "factor" in parameter.lower() or "multiplier" in parameter.lower():
            # For factors, express as fraction
            threshold_value = 1.0 - (gap / primary.expected_value)
            threshold_value = max(0.1, min(0.9, threshold_value))
        else:
            # For absolute parameters, use the raw value
            threshold_value = round(threshold_value, 2)

        return (threshold_value, direction, round(probability, 3))

    def _generate_dominance_conditions(
        self,
        options: List[RankedOption],
        primary: RankedOption
    ) -> List[ConditionalRecommendation]:
        """
        Generate dominance-based conditions.

        Identifies when options dominate others (better on all criteria).
        """
        conditions = []

        for option in options:
            if option.option_id == primary.option_id:
                continue

            # Check if option dominates primary under certain conditions
            dominance_check = self._check_conditional_dominance(primary, option)

            if dominance_check is not None:
                param, operator, value, probability = dominance_check

                description = (
                    f"Choose {option.label} if weighting changes significantly "
                    f"({param} {operator} {value:.2f})"
                )

                condition = ConditionalRecommendation(
                    condition_id=f"cond_{uuid.uuid4().hex[:8]}",
                    condition_type="dominance",
                    condition_description=description,
                    condition_expression=ConditionExpression(
                        parameter=param,
                        operator=operator,
                        value=value
                    ),
                    triggered_recommendation=PrimaryRecommendation(
                        option_id=option.option_id,
                        label=option.label,
                        confidence="medium",
                        expected_value=option.expected_value
                    ),
                    probability_of_condition=probability,
                    impact_magnitude=self._calculate_impact_magnitude(primary, option)
                )
                conditions.append(condition)

        return conditions

    def _check_conditional_dominance(
        self,
        primary: RankedOption,
        alternative: RankedOption
    ) -> Optional[Tuple[str, str, float, float]]:
        """
        Check if alternative can dominate primary under some conditions.

        Returns: (parameter, operator, value, probability) or None
        """
        # Check if alternative has lower variance (safer)
        primary_std = self._get_distribution_std(primary)
        alt_std = self._get_distribution_std(alternative)

        if alt_std < primary_std * 0.7:
            # Alternative is safer - could dominate if risk aversion increases
            param = "risk_aversion_weight"
            operator = ">"
            # Calculate threshold where safer option wins
            ev_diff = primary.expected_value - alternative.expected_value
            variance_diff = primary_std ** 2 - alt_std ** 2

            if variance_diff > 0:
                threshold = ev_diff / variance_diff
                probability = 0.15  # Arbitrary - represents chance of high risk aversion

                return (param, operator, round(threshold, 2), probability)

        return None

    def _generate_risk_profile_conditions(
        self,
        options: List[RankedOption],
        primary: RankedOption
    ) -> List[ConditionalRecommendation]:
        """
        Generate risk-profile based conditions.

        Shows how recommendation changes with different risk tolerances.
        """
        conditions = []

        # Calculate risk-adjusted values for all options
        risk_adjusted = self._calculate_risk_adjusted_rankings(options)

        # Check if ranking changes under different risk profiles
        for profile, ranked_list in risk_adjusted.items():
            if ranked_list[0].option_id != primary.option_id:
                best_under_profile = ranked_list[0]

                description = (
                    f"Choose {best_under_profile.label} if {profile.replace('_', ' ')}"
                )

                # Map profile to parameter
                if profile == "risk_averse":
                    param = "risk_tolerance"
                    operator = "<"
                    value = 0.3
                    probability = 0.20
                else:  # risk_seeking
                    param = "risk_tolerance"
                    operator = ">"
                    value = 0.7
                    probability = 0.15

                condition = ConditionalRecommendation(
                    condition_id=f"cond_{uuid.uuid4().hex[:8]}",
                    condition_type="risk_profile",
                    condition_description=description,
                    condition_expression=ConditionExpression(
                        parameter=param,
                        operator=operator,
                        value=value
                    ),
                    triggered_recommendation=PrimaryRecommendation(
                        option_id=best_under_profile.option_id,
                        label=best_under_profile.label,
                        confidence="medium",
                        expected_value=best_under_profile.expected_value
                    ),
                    probability_of_condition=probability,
                    impact_magnitude=self._calculate_impact_magnitude(
                        primary, best_under_profile
                    )
                )
                conditions.append(condition)

        return conditions

    def _calculate_risk_adjusted_rankings(
        self,
        options: List[RankedOption]
    ) -> Dict[str, List[RankedOption]]:
        """
        Calculate rankings under different risk profiles.

        Uses mean-variance framework for risk adjustment.
        """
        rankings = {}

        # Risk averse: penalize variance
        averse_values = []
        for opt in options:
            std = self._get_distribution_std(opt)
            # Mean - variance penalty
            adjusted = opt.expected_value - self.RISK_AVERSE_FACTOR * (std ** 2)
            averse_values.append((adjusted, opt))

        averse_values.sort(key=lambda x: x[0], reverse=True)
        rankings["risk_averse"] = [x[1] for x in averse_values]

        # Risk seeking: reward variance
        seeking_values = []
        for opt in options:
            std = self._get_distribution_std(opt)
            # Mean + variance bonus
            adjusted = opt.expected_value + self.RISK_SEEKING_FACTOR * std
            seeking_values.append((adjusted, opt))

        seeking_values.sort(key=lambda x: x[0], reverse=True)
        rankings["risk_seeking"] = [x[1] for x in seeking_values]

        return rankings

    def _generate_scenario_conditions(
        self,
        options: List[RankedOption],
        primary: RankedOption
    ) -> List[ConditionalRecommendation]:
        """
        Generate scenario-based conditions.

        Clusters parameter combinations that flip decisions.
        """
        conditions = []

        # Simplified scenario analysis
        scenarios = [
            ("pessimistic", "low_demand", "<", 0.5, 0.2),
            ("optimistic", "high_growth", ">", 1.5, 0.15),
        ]

        for scenario_name, param, operator, value, prob in scenarios:
            # Find best option under scenario
            best_under_scenario = self._find_best_under_scenario(
                options, scenario_name
            )

            if best_under_scenario and best_under_scenario.option_id != primary.option_id:
                description = (
                    f"In {scenario_name} scenario ({param} {operator} {value}), "
                    f"choose {best_under_scenario.label}"
                )

                condition = ConditionalRecommendation(
                    condition_id=f"cond_{uuid.uuid4().hex[:8]}",
                    condition_type="scenario",
                    condition_description=description,
                    condition_expression=ConditionExpression(
                        parameter=param,
                        operator=operator,
                        value=value
                    ),
                    triggered_recommendation=PrimaryRecommendation(
                        option_id=best_under_scenario.option_id,
                        label=best_under_scenario.label,
                        confidence="low",
                        expected_value=best_under_scenario.expected_value
                    ),
                    probability_of_condition=prob,
                    impact_magnitude=self._calculate_impact_magnitude(
                        primary, best_under_scenario
                    )
                )
                conditions.append(condition)

        return conditions

    def _find_best_under_scenario(
        self,
        options: List[RankedOption],
        scenario: str
    ) -> Optional[RankedOption]:
        """Find best option under a given scenario."""
        if scenario == "pessimistic":
            # Prefer lower variance options
            return min(options, key=lambda x: self._get_distribution_std(x))
        elif scenario == "optimistic":
            # Prefer higher upside
            adjusted = []
            for opt in options:
                std = self._get_distribution_std(opt)
                upside = opt.expected_value + 1.5 * std
                adjusted.append((upside, opt))
            adjusted.sort(key=lambda x: x[0], reverse=True)
            return adjusted[0][1]
        return None

    def _calculate_impact_magnitude(
        self,
        primary: RankedOption,
        alternative: RankedOption
    ) -> str:
        """Calculate impact magnitude of switching from primary to alternative."""
        ev_diff = abs(primary.expected_value - alternative.expected_value)
        rel_diff = ev_diff / abs(primary.expected_value) if primary.expected_value != 0 else 0

        if rel_diff > 0.3:
            return "high"
        elif rel_diff > 0.1:
            return "medium"
        else:
            return "low"

    def _rank_and_limit_conditions(
        self,
        conditions: List[ConditionalRecommendation],
        max_conditions: int
    ) -> List[ConditionalRecommendation]:
        """Rank conditions by impact and limit to max_conditions."""
        # Sort by impact magnitude (high > medium > low) and probability
        impact_order = {"high": 3, "medium": 2, "low": 1}

        def sort_key(cond: ConditionalRecommendation):
            impact_score = impact_order.get(cond.impact_magnitude, 0)
            prob_score = cond.probability_of_condition or 0
            return (impact_score, prob_score)

        conditions.sort(key=sort_key, reverse=True)

        return conditions[:max_conditions]

    def _calculate_robustness_summary(
        self,
        conditions: List[ConditionalRecommendation],
        options: List[RankedOption],
        primary: RankedOption
    ) -> RobustnessSummary:
        """Calculate robustness summary for the recommendation."""
        # Count conditions by impact
        high_impact_count = sum(1 for c in conditions if c.impact_magnitude == "high")

        # Determine stability
        if len(conditions) == 0 or high_impact_count == 0:
            stability = "robust"
        elif high_impact_count >= 2 or len(conditions) >= 4:
            stability = "fragile"
        else:
            stability = "moderate"

        # Find closest flip point
        closest_flip = None
        min_probability = 1.0

        for cond in conditions:
            if cond.probability_of_condition and cond.probability_of_condition < min_probability:
                min_probability = cond.probability_of_condition
                closest_flip = cond.condition_expression

        # Calculate safety margin
        safety_margin = None
        if closest_flip is not None and isinstance(closest_flip.value, (int, float)):
            # Rough estimate of margin
            safety_margin = abs(1.0 - closest_flip.value) if closest_flip.value < 2 else 0.1

        return RobustnessSummary(
            recommendation_stability=stability,
            conditions_count=len(conditions),
            closest_flip_point=closest_flip,
            safety_margin=safety_margin
        )
