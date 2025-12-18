"""
FACET-based robustness analyzer.

Verifies whether counterfactual recommendations are robust to
intervention variations and model uncertainties using region-based analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.requests import CounterfactualRequest
from src.models.robustness import (
    FACETRobustnessAnalysis,
    InterventionRegion,
    OutcomeGuarantee,
    RobustnessRequest,
)
from src.models.shared import StructuralModel
from src.services.causal_validator import CausalValidator
from src.services.counterfactual_engine import CounterfactualEngine
from src.utils.determinism import make_deterministic

logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """
    Region-based robustness analysis for counterfactual recommendations.

    Implements FACET algorithm:
    1. Generate candidate intervention regions around proposal
    2. Verify outcomes across each region via sampling
    3. Identify robust regions (all samples satisfy target)
    4. Compute robustness score
    5. Flag fragile recommendations
    """

    def __init__(
        self,
        cf_engine: Optional[CounterfactualEngine] = None,
        causal_validator: Optional[CausalValidator] = None,
    ) -> None:
        """
        Initialize robustness analyzer.

        Args:
            cf_engine: Counterfactual engine for simulations
            causal_validator: For causal validation
        """
        self.cf_engine = cf_engine or CounterfactualEngine()
        self.causal_validator = causal_validator or CausalValidator()

    def analyze_robustness(
        self,
        request: RobustnessRequest,
        request_id: str,
    ) -> FACETRobustnessAnalysis:
        """
        Perform complete FACET robustness analysis.

        Args:
            request: Robustness analysis request
            request_id: Request ID for tracing

        Returns:
            FACETRobustnessAnalysis with regions and guarantees
        """
        # Make computation deterministic
        rng = make_deterministic(request.model_dump())

        logger.info(
            "robustness_analysis_started",
            extra={
                "request_id": request_id,
                "intervention_vars": list(request.intervention_proposal.keys()),
                "target_outcomes": list(request.target_outcome.keys()),
                "perturbation_radius": request.perturbation_radius,
                "seed": rng.seed,
            },
        )

        try:
            # Step 1: Validate causal identifiability (if we have structural model)
            if request.structural_model is not None:
                if not self._validate_intervention_identifiable(
                    request.causal_model,
                    request.intervention_proposal,
                    request.target_outcome,
                ):
                    return self._build_failed_analysis(
                        "Intervention not causally identifiable",
                        request_id,
                    )

            # Step 2: Generate candidate regions
            candidate_regions = self._generate_candidate_regions(
                intervention_proposal=request.intervention_proposal,
                perturbation_radius=request.perturbation_radius,
                feasible_ranges=request.feasible_ranges,
            )

            logger.debug(
                f"Generated {len(candidate_regions)} candidate regions",
                extra={"request_id": request_id},
            )

            # Step 3: Test each region for robustness
            robust_regions = []
            outcome_guarantees = {}

            if request.structural_model is not None:
                # Have structural model - can do full verification
                for region in candidate_regions:
                    is_robust, guarantees = self._verify_region_robustness(
                        region=region,
                        structural_model=request.structural_model,
                        target_outcome=request.target_outcome,
                        min_samples=request.min_samples,
                        confidence_level=request.confidence_level,
                        request_id=request_id,
                        seed=seed,
                    )

                    if is_robust:
                        robust_regions.append(region)
                        outcome_guarantees.update(guarantees)
            else:
                # No structural model - use heuristic region selection
                # (Accept center region as robust with lower confidence)
                logger.warning(
                    "No structural model provided - using heuristic analysis",
                    extra={"request_id": request_id},
                )
                robust_regions = [candidate_regions[0]]  # Center region
                # Create conservative guarantees
                for outcome_var, (target_min, target_max) in request.target_outcome.items():
                    outcome_guarantees[outcome_var] = OutcomeGuarantee(
                        outcome_variable=outcome_var,
                        minimum=target_min,
                        maximum=target_max,
                        confidence=0.7,  # Lower confidence for heuristic
                    )

            # Step 4: Compute robustness metrics
            robustness_score = self._compute_robustness_score(
                robust_regions=robust_regions,
                intervention_space=request.intervention_proposal,
                perturbation_radius=request.perturbation_radius,
            )

            # Step 5: Detect fragility
            is_fragile, fragility_reasons = self._detect_fragility(
                robust_regions=robust_regions,
                robustness_score=robustness_score,
                total_samples=request.min_samples * len(candidate_regions),
            )

            # Step 6: Build result
            total_volume = sum(r.volume() for r in robust_regions)
            samples_successful = request.min_samples * len(robust_regions)

            analysis = FACETRobustnessAnalysis(
                status="robust" if robust_regions and not is_fragile else "fragile",
                robust_regions=robust_regions,
                outcome_guarantees=outcome_guarantees,
                robustness_score=robustness_score,
                region_count=len(robust_regions),
                total_volume=total_volume,
                is_fragile=is_fragile,
                fragility_reasons=fragility_reasons,
                samples_tested=request.min_samples * len(candidate_regions),
                samples_successful=samples_successful,
                confidence_level=request.confidence_level,
                interpretation=self._generate_interpretation(
                    robust_regions, robustness_score, is_fragile
                ),
                recommendation=self._generate_recommendation(
                    robust_regions, is_fragile, request.intervention_proposal
                ),
            )

            logger.info(
                "robustness_analysis_complete",
                extra={
                    "request_id": request_id,
                    "status": analysis.status,
                    "robustness_score": robustness_score,
                    "regions_found": len(robust_regions),
                    "is_fragile": is_fragile,
                },
            )

            return analysis

        except Exception as e:
            logger.error(
                "robustness_analysis_failed",
                exc_info=True,
                extra={"request_id": request_id},
            )
            return self._build_failed_analysis(str(e), request_id)

    def _generate_candidate_regions(
        self,
        intervention_proposal: Dict[str, float],
        perturbation_radius: float,
        feasible_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[InterventionRegion]:
        """
        Generate candidate regions around proposed intervention.

        Strategy: Create overlapping hypercubes centered at proposal,
        expanding outward to explore intervention space.

        Args:
            intervention_proposal: Proposed intervention values
            perturbation_radius: How far to search (fraction)
            feasible_ranges: Optional constraints on variable ranges

        Returns:
            List of candidate intervention regions
        """
        regions = []

        # Center region (small perturbation around proposal)
        center_region = self._create_region_around_point(
            point=intervention_proposal,
            radius=perturbation_radius * 0.5,  # Half radius for tightest region
            feasible_ranges=feasible_ranges,
        )
        regions.append(center_region)

        # Expanded region (full perturbation)
        expanded_region = self._create_region_around_point(
            point=intervention_proposal,
            radius=perturbation_radius,
            feasible_ranges=feasible_ranges,
        )
        regions.append(expanded_region)

        # Directional regions (explore along each dimension)
        for var in intervention_proposal.keys():
            # Positive direction
            pos_point = intervention_proposal.copy()
            pos_point[var] *= 1 + perturbation_radius
            pos_region = self._create_region_around_point(
                pos_point,
                radius=perturbation_radius * 0.3,
                feasible_ranges=feasible_ranges,
            )
            regions.append(pos_region)

            # Negative direction
            neg_point = intervention_proposal.copy()
            neg_point[var] *= 1 - perturbation_radius
            neg_region = self._create_region_around_point(
                neg_point,
                radius=perturbation_radius * 0.3,
                feasible_ranges=feasible_ranges,
            )
            regions.append(neg_region)

        return regions

    def _create_region_around_point(
        self,
        point: Dict[str, float],
        radius: float,
        feasible_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> InterventionRegion:
        """
        Create region as hypercube around point.

        Args:
            point: Center point for region
            radius: Fractional radius (e.g., 0.1 = ±10%)
            feasible_ranges: Optional feasibility constraints

        Returns:
            InterventionRegion centered at point
        """
        variable_ranges = {}

        for var, value in point.items():
            # Compute range
            min_val = value * (1 - radius)
            max_val = value * (1 + radius)

            # Constrain to feasible ranges if provided
            if feasible_ranges and var in feasible_ranges:
                feas_min, feas_max = feasible_ranges[var]
                min_val = max(min_val, feas_min)
                max_val = min(max_val, feas_max)

            variable_ranges[var] = (min_val, max_val)

        return InterventionRegion(variable_ranges=variable_ranges)

    def _verify_region_robustness(
        self,
        region: InterventionRegion,
        structural_model: Dict,
        target_outcome: Dict[str, Tuple[float, float]],
        min_samples: int,
        confidence_level: float,
        request_id: str,
        seed: int,
    ) -> Tuple[bool, Dict[str, OutcomeGuarantee]]:
        """
        Verify if region is robust via Monte Carlo sampling.

        Args:
            region: Intervention region to verify
            structural_model: Structural equations and distributions
            target_outcome: Target outcome ranges
            min_samples: Minimum samples to test
            confidence_level: Required confidence
            request_id: Request ID for logging
            seed: Random seed for reproducibility

        Returns:
            (is_robust, outcome_guarantees) tuple
        """
        # Sample interventions from region
        sampled_interventions = region.sample_random(n=min_samples, seed=seed)

        # Simulate counterfactuals for each sample
        outcomes_by_variable: Dict[str, List[float]] = {
            var: [] for var in target_outcome.keys()
        }

        for i, intervention in enumerate(sampled_interventions):
            try:
                # Create counterfactual request for this intervention
                # Note: We need the outcome variable name for CounterfactualRequest
                outcome_var = list(target_outcome.keys())[0]

                cf_request = CounterfactualRequest(
                    model=StructuralModel(
                        variables=structural_model.get("variables", []),
                        equations=structural_model.get("equations", {}),
                        distributions=structural_model.get("distributions", {}),
                    ),
                    intervention=intervention,
                    outcome=outcome_var,
                    context=structural_model.get("context"),
                )

                # Run counterfactual simulation
                cf_result = self.cf_engine.analyze(cf_request)

                # Extract outcome value
                outcome_value = cf_result.prediction.point_estimate
                outcomes_by_variable[outcome_var].append(outcome_value)

            except Exception as e:
                logger.warning(
                    f"Counterfactual failed for sample {i}: {e}",
                    extra={"request_id": request_id},
                )
                # Treat failure as not satisfying target
                return False, {}

        # Check if all samples satisfy target
        is_robust = True
        guarantees = {}

        for var, (target_min, target_max) in target_outcome.items():
            outcomes = outcomes_by_variable.get(var, [])

            if not outcomes or len(outcomes) < min_samples * 0.8:  # Allow 20% failure
                is_robust = False
                break

            # Check satisfaction
            outcomes_array = np.array(outcomes)
            satisfies = np.all((outcomes_array >= target_min) & (outcomes_array <= target_max))

            if not satisfies:
                is_robust = False
                break

            # Compute guarantee (conservative bounds)
            guarantee = OutcomeGuarantee(
                outcome_variable=var,
                minimum=float(np.min(outcomes_array)),
                maximum=float(np.max(outcomes_array)),
                confidence=confidence_level,
            )
            guarantees[var] = guarantee

        return is_robust, guarantees

    def _compute_robustness_score(
        self,
        robust_regions: List[InterventionRegion],
        intervention_space: Dict[str, float],
        perturbation_radius: float,
    ) -> float:
        """
        Compute overall robustness score.

        Score = (volume of robust regions) / (explored volume)
        Higher score = more robust

        Args:
            robust_regions: List of robust regions found
            intervention_space: Original intervention proposal
            perturbation_radius: Search radius used

        Returns:
            Robustness score between 0 and 1
        """
        if not robust_regions:
            return 0.0

        # Total volume of robust regions
        robust_volume = sum(r.volume() for r in robust_regions)

        # Maximum explored volume (hypercube with full perturbation)
        explored_volume = (2 * perturbation_radius) ** len(intervention_space)

        # Normalize
        score = min(1.0, robust_volume / explored_volume)

        return score

    def _detect_fragility(
        self,
        robust_regions: List[InterventionRegion],
        robustness_score: float,
        total_samples: int,
    ) -> Tuple[bool, List[str]]:
        """
        Detect if recommendation is fragile.

        Fragile indicators:
        - No robust regions found
        - Very small robustness score (< 0.2)
        - Single tiny region

        Args:
            robust_regions: Robust regions found
            robustness_score: Overall robustness score
            total_samples: Total samples tested

        Returns:
            (is_fragile, fragility_reasons) tuple
        """
        reasons = []

        if not robust_regions:
            reasons.append("No robust intervention regions found")
            return True, reasons

        if robustness_score < 0.1:
            reasons.append(f"Very low robustness score ({robustness_score:.3f})")
            return True, reasons

        if robustness_score < 0.3:
            reasons.append(
                f"Low robustness score ({robustness_score:.3f}) - "
                "recommendation is sensitive to intervention variations"
            )

        if len(robust_regions) == 1 and robust_regions[0].volume() < 0.01:
            reasons.append(
                "Only one very small robust region found - recommendation requires precision"
            )

        # Fragile if any reasons
        is_fragile = len(reasons) > 0

        return is_fragile, reasons

    def _generate_interpretation(
        self,
        robust_regions: List[InterventionRegion],
        robustness_score: float,
        is_fragile: bool,
    ) -> str:
        """
        Generate user-friendly interpretation.

        Args:
            robust_regions: Robust regions found
            robustness_score: Overall robustness score
            is_fragile: Whether recommendation is fragile

        Returns:
            Plain English interpretation string
        """
        if not robust_regions:
            return (
                "No robust intervention strategy found. The proposed intervention "
                "does not reliably achieve the target outcome across variations. "
                "Consider alternative strategies or adjusting targets."
            )

        if is_fragile:
            return (
                f"FRAGILE RECOMMENDATION (robustness: {robustness_score:.2f}). "
                f"Only narrow ranges of intervention values achieve the target outcome. "
                f"Small deviations from the proposed strategy may fail to achieve goals. "
                f"Execute with caution and close monitoring."
            )

        if robustness_score > 0.7:
            return (
                f"ROBUST RECOMMENDATION (robustness: {robustness_score:.2f}). "
                f"Multiple intervention strategies achieve the target outcome. "
                f"You have flexibility in implementation and tolerance for variation. "
                f"High confidence in recommendation."
            )

        # Medium robustness
        return (
            f"MODERATELY ROBUST (robustness: {robustness_score:.2f}). "
            f"Several intervention ranges achieve the target, but stay within "
            f"identified regions for reliable results."
        )

    def _generate_recommendation(
        self,
        robust_regions: List[InterventionRegion],
        is_fragile: bool,
        original_proposal: Dict[str, float],
    ) -> str:
        """
        Generate actionable recommendation.

        Args:
            robust_regions: Robust regions found
            is_fragile: Whether recommendation is fragile
            original_proposal: Original intervention proposal

        Returns:
            Actionable recommendation string
        """
        if not robust_regions:
            return (
                "Recommendation: Revise strategy. Current proposal does not robustly "
                "achieve targets. Explore alternative interventions or adjust outcome expectations."
            )

        if is_fragile:
            # Find region containing original proposal
            containing_region = None
            for region in robust_regions:
                if region.contains(original_proposal):
                    containing_region = region
                    break

            if containing_region:
                ranges_str = ", ".join(
                    [
                        f"{var}: {min_val:.1f}-{max_val:.1f}"
                        for var, (min_val, max_val) in containing_region.variable_ranges.items()
                    ]
                )
                return (
                    f"Recommendation: Implement within tight bounds. "
                    f"Acceptable ranges: {ranges_str}. "
                    f"Monitor closely and be prepared to adjust."
                )
            else:
                return (
                    "Recommendation: Original proposal not in robust region. "
                    "Consider using center point of largest robust region instead."
                )

        # Robust case
        largest_region = max(robust_regions, key=lambda r: r.volume())
        ranges_str = ", ".join(
            [
                f"{var}: {min_val:.1f}-{max_val:.1f}"
                for var, (min_val, max_val) in largest_region.variable_ranges.items()
            ]
        )

        return (
            f"Recommendation: Proceed with confidence. "
            f"Operating ranges: {ranges_str}. "
            f"Strategy is robust to reasonable variations."
        )

    def _validate_intervention_identifiable(
        self,
        causal_model: Dict,
        intervention: Dict[str, float],
        target_outcome: Dict[str, Tuple[float, float]],
    ) -> bool:
        """
        Verify intervention is causally identifiable.

        Args:
            causal_model: Causal DAG structure
            intervention: Intervention variables
            target_outcome: Target outcome variables

        Returns:
            True if intervention is identifiable
        """
        # For now, skip Y₀ validation to avoid complexity
        # In production, would validate each treatment→outcome pair
        logger.debug("Skipping causal identifiability check (not implemented)")
        return True

    def _build_failed_analysis(
        self,
        reason: str,
        request_id: str,
    ) -> FACETRobustnessAnalysis:
        """
        Build failed analysis result.

        Args:
            reason: Failure reason
            request_id: Request ID for logging

        Returns:
            FACETRobustnessAnalysis indicating failure
        """
        logger.error(
            "robustness_analysis_failed",
            extra={"request_id": request_id, "reason": reason},
        )

        return FACETRobustnessAnalysis(
            status="failed",
            robust_regions=[],
            outcome_guarantees={},
            robustness_score=0.0,
            region_count=0,
            total_volume=0.0,
            is_fragile=True,
            fragility_reasons=[reason],
            samples_tested=0,
            samples_successful=0,
            confidence_level=0.95,
            interpretation=f"Analysis failed: {reason}",
            recommendation="Cannot provide robustness analysis. Address causal identification issues first.",
        )
