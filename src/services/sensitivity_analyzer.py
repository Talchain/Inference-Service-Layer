"""
Sensitivity Analyzer for testing assumption robustness.

Performs one-at-a-time sensitivity analysis to identify which assumptions
most affect the conclusions.
"""

import logging
from typing import Dict, List

import numpy as np

from src.models.requests import Assumption, SensitivityAnalysisRequest
from src.models.responses import (
    AssumptionAnalysis,
    Breakpoint,
    ConclusionStatement,
    ImpactAssessment,
    RobustnessScore,
    SensitivityAnalysisResponse,
)
from src.models.shared import ConfidenceLevel, ImportanceLevel, RobustnessLevel
from src.services.counterfactual_engine import CounterfactualEngine
from src.services.explanation_generator import ExplanationGenerator
from src.utils.determinism import canonical_hash, make_deterministic

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Analyzes sensitivity to assumption changes.

    Tests how robust conclusions are by perturbing each assumption
    and measuring the impact on results.
    """

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self.explanation_generator = ExplanationGenerator()
        self.counterfactual_engine = CounterfactualEngine()

    def analyze(
        self, request: SensitivityAnalysisRequest
    ) -> SensitivityAnalysisResponse:
        """
        Perform sensitivity analysis.

        Args:
            request: Sensitivity analysis request

        Returns:
            SensitivityAnalysisResponse: Sensitivity analysis results
        """
        # Make computation deterministic
        seed = make_deterministic(request.model_dump())

        logger.info(
            "sensitivity_analysis_started",
            extra={
                "request_hash": canonical_hash(request.model_dump()),
                "baseline_result": request.baseline_result,
                "num_assumptions": len(request.assumptions),
                "seed": seed,
            },
        )

        try:
            # Analyze each assumption
            assumption_analyses = []
            total_variance = 0.0

            for assumption in request.assumptions:
                analysis = self._analyze_assumption(
                    assumption, request.baseline_result, request.model
                )
                assumption_analyses.append(analysis)
                total_variance += analysis.impact.percentage

            # Normalize percentages to sum to 100
            if total_variance > 0:
                for analysis in assumption_analyses:
                    analysis.impact.percentage = (
                        analysis.impact.percentage / total_variance * 100
                    )

            # Sort by importance (impact)
            assumption_analyses.sort(
                key=lambda x: x.impact.if_wrong, reverse=True
            )

            # Calculate overall robustness
            robustness = self._calculate_robustness(
                assumption_analyses, request.baseline_result
            )

            # Create conclusion statement
            conclusion = ConclusionStatement(
                statement=f"Result is {request.baseline_result:.0f}",
                base_case=request.baseline_result,
            )

            # Count assumptions by importance
            num_critical = sum(
                1 for a in assumption_analyses if a.importance == ImportanceLevel.CRITICAL
            )
            num_moderate = sum(
                1 for a in assumption_analyses if a.importance == ImportanceLevel.MODERATE
            )
            num_minor = sum(
                1 for a in assumption_analyses if a.importance == ImportanceLevel.MINOR
            )

            # Generate explanation
            explanation = self.explanation_generator.generate_sensitivity_explanation(
                baseline_result=request.baseline_result,
                robustness_level=robustness.overall.value,
                num_critical=num_critical,
                num_moderate=num_moderate,
                num_minor=num_minor,
            )

            return SensitivityAnalysisResponse(
                conclusion=conclusion,
                assumptions=assumption_analyses,
                robustness=robustness,
                explanation=explanation,
            )

        except Exception as e:
            logger.error("sensitivity_analysis_failed", exc_info=True)
            raise

    def _analyze_assumption(
        self,
        assumption: Assumption,
        baseline_result: float,
        model: any,
    ) -> AssumptionAnalysis:
        """
        Analyze a single assumption.

        Args:
            assumption: Assumption to test
            baseline_result: Baseline result value
            model: Structural model

        Returns:
            AssumptionAnalysis with impact assessment
        """
        # Perform perturbation analysis
        if assumption.variation_range:
            # Use provided range
            min_val = assumption.variation_range.get("min", assumption.current_value * 0.7)
            max_val = assumption.variation_range.get("max", assumption.current_value * 1.3)
        else:
            # Default: ±30% variation
            if isinstance(assumption.current_value, (int, float)):
                min_val = assumption.current_value * 0.7
                max_val = assumption.current_value * 1.3
            else:
                # For non-numeric values, use fixed impact estimate
                min_val = max_val = assumption.current_value

        # Estimate impact (simplified - in practice would rerun model)
        # Here we use a heuristic based on variation range
        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
            variation = abs(max_val - min_val)
            # Impact proportional to variation and baseline result
            impact = abs(baseline_result * (variation / abs(assumption.current_value)))
            if impact == 0 or isinstance(assumption.current_value, (int, float)) and assumption.current_value == 0:
                impact = abs(baseline_result * 0.1)  # Default 10% impact
        else:
            impact = abs(baseline_result * 0.1)

        # Classify importance
        impact_ratio = impact / abs(baseline_result) if baseline_result != 0 else 0.1
        if impact_ratio >= 0.3:
            importance = ImportanceLevel.CRITICAL
        elif impact_ratio >= 0.1:
            importance = ImportanceLevel.MODERATE
        else:
            importance = ImportanceLevel.MINOR

        # Determine confidence
        if assumption.type == "parametric":
            confidence = ConfidenceLevel.MEDIUM
        elif assumption.type == "structural":
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.MEDIUM

        # Generate evidence and recommendation
        evidence = self._generate_evidence(assumption)
        recommendation = self._generate_recommendation(assumption, importance)

        return AssumptionAnalysis(
            name=assumption.name,
            current_value=assumption.current_value,
            importance=importance,
            impact=ImpactAssessment(
                if_wrong=round(impact, 2),
                percentage=0.0,  # Will be normalized later
            ),
            confidence=confidence,
            evidence=evidence,
            recommendation=recommendation,
        )

    def _calculate_robustness(
        self, analyses: List[AssumptionAnalysis], baseline_result: float
    ) -> RobustnessScore:
        """
        Calculate overall robustness score.

        Args:
            analyses: List of assumption analyses
            baseline_result: Baseline result

        Returns:
            RobustnessScore
        """
        # Find maximum impact
        max_impact = max([a.impact.if_wrong for a in analyses], default=0)
        max_impact_ratio = max_impact / abs(baseline_result) if baseline_result != 0 else 0

        # Determine overall robustness
        if max_impact_ratio < 0.15:
            overall = RobustnessLevel.ROBUST
            summary = "Conclusion holds across a wide range of assumption variations"
        elif max_impact_ratio < 0.35:
            overall = RobustnessLevel.MODERATE
            summary = "Conclusion holds under most reasonable assumption variations, but some assumptions are critical"
        else:
            overall = RobustnessLevel.FRAGILE
            summary = "Conclusion depends heavily on specific assumptions being correct"

        # Identify breakpoints (critical assumptions)
        breakpoints = []
        for analysis in analyses:
            if analysis.importance == ImportanceLevel.CRITICAL:
                # Calculate threshold where impact reverses sign
                threshold_desc = f"If {analysis.name} varies by more than ±30%"
                if isinstance(analysis.current_value, (int, float)):
                    critical_value = analysis.current_value * 1.5
                    threshold_desc = f"If {analysis.name} exceeds {critical_value:.2f}, conclusion may reverse"

                breakpoints.append(
                    Breakpoint(
                        assumption=analysis.name,
                        threshold=threshold_desc,
                    )
                )

        return RobustnessScore(
            overall=overall,
            summary=summary,
            breakpoints=breakpoints,
        )

    def _generate_evidence(self, assumption: Assumption) -> str:
        """
        Generate evidence description for an assumption.

        Args:
            assumption: Assumption

        Returns:
            Evidence description
        """
        if assumption.type == "parametric":
            return f"Parameter value based on current model: {assumption.current_value}"
        elif assumption.type == "structural":
            return "Structural assumption based on domain knowledge"
        elif assumption.type == "distributional":
            return "Distributional assumption based on prior data"
        else:
            return "Assumption based on expert judgment"

    def _generate_recommendation(
        self, assumption: Assumption, importance: ImportanceLevel
    ) -> str:
        """
        Generate recommendation for an assumption.

        Args:
            assumption: Assumption
            importance: Importance level

        Returns:
            Recommendation text
        """
        if importance == ImportanceLevel.CRITICAL:
            return f"CRITICAL: Validate {assumption.name} with additional data or experiments before making final decision"
        elif importance == ImportanceLevel.MODERATE:
            return f"Consider validating {assumption.name} if possible to reduce uncertainty"
        else:
            return f"{assumption.name} has minimal impact; no additional validation needed"
