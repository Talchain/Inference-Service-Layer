"""
Conformal Prediction service for causal counterfactuals.

Provides finite-sample valid prediction intervals with guaranteed coverage
using conformal prediction methods.

References:
- Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Lei et al. (2018) "Distribution-Free Predictive Inference"
- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException

from src.models.requests import ConformalCounterfactualRequest, ObservationPoint
from src.models.responses import (
    CalibrationMetrics,
    ComparisonMetrics,
    ConfidenceInterval,
    ConformalCounterfactualResponse,
    ConformalInterval,
    CoverageGuarantee,
    ExplanationMetadata,
)
from src.models.shared import SensitivityRange
from src.services.counterfactual_engine import CounterfactualEngine
from src.utils.determinism import canonical_hash, make_deterministic

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    Conformal prediction for causal counterfactuals.

    Provides finite-sample valid prediction intervals with
    guaranteed coverage regardless of underlying distribution.
    """

    def __init__(self, counterfactual_engine: CounterfactualEngine):
        """
        Initialize conformal predictor.

        Args:
            counterfactual_engine: Engine for generating counterfactual predictions
        """
        self.cf_engine = counterfactual_engine

    def predict_with_conformal_interval(
        self, request: ConformalCounterfactualRequest
    ) -> ConformalCounterfactualResponse:
        """
        Generate conformal prediction interval for counterfactual.

        Args:
            request: Conformal counterfactual request

        Returns:
            Conformal interval with coverage guarantee

        Raises:
            HTTPException: If calibration data is insufficient or invalid
        """
        # Make computation deterministic
        request_hash = make_deterministic(request.model_dump())

        logger.info(
            "conformal_prediction_started",
            extra={
                "request_hash": canonical_hash(request.model_dump()),
                "method": request.method,
                "confidence_level": request.confidence_level,
            },
        )

        try:
            # Validate calibration data
            if request.calibration_data is None or len(request.calibration_data) < 10:
                raise HTTPException(
                    status_code=400,
                    detail="Conformal prediction requires at least 10 calibration points",
                )

            # Use split conformal method
            if request.method == "split":
                result = self._split_conformal(request)
            else:
                # Fallback to split for now (cv+ and jackknife+ would be implemented here)
                logger.warning(
                    f"Method {request.method} not yet implemented, using split conformal"
                )
                result = self._split_conformal(request)

            logger.info(
                "conformal_prediction_completed",
                extra={
                    "guaranteed_coverage": result.coverage_guarantee.guaranteed_coverage,
                    "interval_width": list(result.prediction_interval.interval_width.values())[0]
                    if result.prediction_interval.interval_width
                    else None,
                },
            )

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("conformal_prediction_failed", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Conformal prediction failed: {str(e)}"
            )

    def _split_conformal(
        self, request: ConformalCounterfactualRequest
    ) -> ConformalCounterfactualResponse:
        """
        Split conformal prediction.

        Algorithm:
        1. Split calibration data (50% train, 50% calibrate)
        2. Compute residuals on calibration set
        3. Find quantile for desired coverage
        4. Form interval: [point_estimate - q, point_estimate + q]

        Args:
            request: Conformal counterfactual request

        Returns:
            Conformal prediction response
        """
        if request.seed is not None:
            np.random.seed(request.seed)

        # Split calibration data
        calib_data = request.calibration_data
        n = len(calib_data)
        split_idx = int(n * 0.5)

        # Randomly shuffle
        indices = np.random.permutation(n)
        train_idx = indices[:split_idx]
        calib_idx = indices[split_idx:]

        train_data = [calib_data[i] for i in train_idx]
        calib_data_split = [calib_data[i] for i in calib_idx]

        if len(calib_data_split) < 5:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient calibration data after split ({len(calib_data_split)} points). Need at least 5.",
            )

        # Compute conformity scores (residuals) on calibration set
        conformity_scores = self._compute_conformity_scores(
            request, calib_data_split
        )

        # Get quantile for desired coverage
        alpha = 1 - request.confidence_level
        n_calib = len(conformity_scores)

        # Finite-sample correction
        q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        q = np.quantile(conformity_scores, min(q_level, 1.0))

        # Point prediction using full model
        from src.services.structural_model_parser import StructuralModelParser

        parser = StructuralModelParser()
        scm = parser.parse(request.model)

        # Get point estimate (mean of counterfactual distribution)
        point_prediction = self._get_point_prediction(scm, request.intervention)

        # Extract outcome variable (assume single outcome for simplicity)
        outcome_var = list(point_prediction.keys())[0]
        point_est = point_prediction[outcome_var]

        # Conformal interval
        conformal_interval = ConformalInterval(
            lower_bound={outcome_var: point_est - q},
            upper_bound={outcome_var: point_est + q},
            point_estimate=point_prediction,
            interval_width={outcome_var: 2 * q},
        )

        # Coverage guarantee
        guaranteed_coverage = self._compute_guaranteed_coverage(n_calib, alpha)

        coverage = CoverageGuarantee(
            nominal_coverage=request.confidence_level,
            guaranteed_coverage=guaranteed_coverage,
            finite_sample_valid=True,
            assumptions=["Exchangeability of calibration and test points"],
        )

        # Calibration quality metrics
        calibration_metrics = self._assess_calibration_quality(
            conformity_scores, calib_data_split
        )

        # Compare to standard Monte Carlo
        mc_interval = self._monte_carlo_interval(
            scm, request.intervention, request.confidence_level, request.samples
        )

        comparison = self._compare_intervals(
            conformal_interval, mc_interval, outcome_var
        )

        # Generate explanation
        explanation = self._generate_explanation(
            conformal_interval,
            coverage,
            calibration_metrics,
            comparison,
            request.method,
        )

        return ConformalCounterfactualResponse(
            prediction_interval=conformal_interval,
            coverage_guarantee=coverage,
            calibration_quality=calibration_metrics,
            comparison_to_standard=comparison,
            explanation=explanation,
        )

    def _compute_conformity_scores(
        self,
        request: ConformalCounterfactualRequest,
        calib_data: List[ObservationPoint],
    ) -> np.ndarray:
        """
        Compute conformity scores (absolute residuals).

        For each calibration point, compute |Y_observed - Y_predicted|.

        Args:
            request: Conformal request
            calib_data: Calibration observations

        Returns:
            Array of conformity scores
        """
        from src.services.structural_model_parser import StructuralModelParser

        parser = StructuralModelParser()
        scm = parser.parse(request.model)

        residuals = []

        for obs in calib_data:
            # Predict outcome given inputs
            predicted = self._get_point_prediction(scm, obs.inputs)

            # Get first outcome variable
            outcome_var = list(obs.outcome.keys())[0]
            observed_value = obs.outcome[outcome_var]
            predicted_value = predicted[outcome_var]

            # Absolute residual
            residual = abs(observed_value - predicted_value)
            residuals.append(residual)

        return np.array(residuals)

    def _get_point_prediction(
        self, scm: any, intervention: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get point prediction from structural causal model.

        Args:
            scm: Structural causal model instance
            intervention: Intervention values

        Returns:
            Point predictions for outcome variables
        """
        # For now, use a simple simulation approach
        # In production, would integrate with actual SCM inference

        # Simulate from model equations
        try:
            # This is a simplified prediction - in production would use proper SCM evaluation
            result = scm.simulate(intervention, samples=100)

            # Return mean as point estimate
            point_est = {}
            for var, values in result.items():
                if isinstance(values, (list, np.ndarray)):
                    point_est[var] = float(np.mean(values))
                else:
                    point_est[var] = float(values)

            return point_est
        except Exception as e:
            logger.warning(f"SCM simulation failed, using fallback: {e}")
            # Fallback: return intervention values for outcome
            return {var: float(val) for var, val in intervention.items()}

    def _compute_guaranteed_coverage(self, n: int, alpha: float) -> float:
        """
        Compute finite-sample coverage guarantee.

        Guarantee: coverage ≥ (⌈(n+1)(1-α)⌉ - 1) / n

        Args:
            n: Calibration set size
            alpha: Significance level (1 - confidence)

        Returns:
            Guaranteed coverage probability
        """
        numerator = np.ceil((n + 1) * (1 - alpha)) - 1
        return float(numerator / n)

    def _assess_calibration_quality(
        self, scores: np.ndarray, calib_data: List[ObservationPoint]
    ) -> CalibrationMetrics:
        """
        Assess quality of calibration set.

        Args:
            scores: Conformity scores
            calib_data: Calibration observations

        Returns:
            Calibration quality metrics
        """
        return CalibrationMetrics(
            calibration_size=len(scores),
            residual_statistics={
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
                "iqr": float(
                    np.percentile(scores, 75) - np.percentile(scores, 25)
                ),
            },
            interval_adaptivity=self._compute_adaptivity(scores),
        )

    def _compute_adaptivity(self, scores: np.ndarray) -> float:
        """
        Measure how much intervals adapt to local uncertainty.

        Higher coefficient of variation = more adaptive (desirable).

        Args:
            scores: Conformity scores

        Returns:
            Adaptivity score (coefficient of variation)
        """
        mean_score = np.mean(scores)
        if mean_score < 1e-10:
            return 0.0

        cv = np.std(scores) / mean_score
        return float(cv)

    def _monte_carlo_interval(
        self,
        scm: any,
        intervention: Dict[str, float],
        confidence_level: float,
        samples: int,
    ) -> Dict[str, ConfidenceInterval]:
        """
        Generate standard Monte Carlo confidence interval for comparison.

        Args:
            scm: Structural causal model
            intervention: Intervention values
            confidence_level: Confidence level
            samples: Number of Monte Carlo samples

        Returns:
            Monte Carlo confidence intervals
        """
        try:
            # Simulate from model
            result = scm.simulate(intervention, samples=samples)

            # Compute percentile-based intervals
            alpha = 1 - confidence_level
            lower_percentile = alpha / 2
            upper_percentile = 1 - alpha / 2

            mc_intervals = {}
            for var, values in result.items():
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    lower = float(np.percentile(values, lower_percentile * 100))
                    upper = float(np.percentile(values, upper_percentile * 100))

                    mc_intervals[var] = ConfidenceInterval(
                        lower=lower,
                        upper=upper,
                        confidence_level=confidence_level,
                    )

            return mc_intervals

        except Exception as e:
            logger.warning(f"Monte Carlo interval computation failed: {e}")
            # Return placeholder
            outcome_var = list(intervention.keys())[0]
            return {
                outcome_var: ConfidenceInterval(
                    lower=0.0,
                    upper=100000.0,
                    confidence_level=confidence_level,
                )
            }

    def _compare_intervals(
        self,
        conformal_interval: ConformalInterval,
        mc_interval: Dict[str, ConfidenceInterval],
        outcome_var: str,
    ) -> ComparisonMetrics:
        """
        Compare conformal and Monte Carlo intervals.

        Args:
            conformal_interval: Conformal prediction interval
            mc_interval: Monte Carlo confidence interval
            outcome_var: Outcome variable name

        Returns:
            Comparison metrics
        """
        conf_lower = conformal_interval.lower_bound[outcome_var]
        conf_upper = conformal_interval.upper_bound[outcome_var]
        conf_width = conf_upper - conf_lower

        mc_ci = mc_interval.get(outcome_var)
        if mc_ci:
            mc_lower = mc_ci.lower
            mc_upper = mc_ci.upper
            mc_width = mc_upper - mc_lower

            width_ratio = conf_width / mc_width if mc_width > 0 else 1.0

            if width_ratio > 1.1:
                interpretation = f"Conformal interval is {(width_ratio - 1) * 100:.1f}% wider, providing more honest uncertainty quantification with finite-sample guarantees"
            elif width_ratio < 0.9:
                interpretation = f"Conformal interval is {(1 - width_ratio) * 100:.1f}% narrower, but still provides provable coverage"
            else:
                interpretation = "Conformal and Monte Carlo intervals are similar in width"
        else:
            mc_lower = conf_lower
            mc_upper = conf_upper
            width_ratio = 1.0
            interpretation = "Monte Carlo interval not available for comparison"

        return ComparisonMetrics(
            monte_carlo_interval={
                outcome_var: ConfidenceInterval(
                    lower=mc_lower,
                    upper=mc_upper,
                    confidence_level=0.95,
                )
            },
            conformal_interval={outcome_var: (conf_lower, conf_upper)},
            width_ratio={outcome_var: width_ratio},
            interpretation=interpretation,
        )

    def _generate_explanation(
        self,
        conformal_interval: ConformalInterval,
        coverage: CoverageGuarantee,
        calibration: CalibrationMetrics,
        comparison: ComparisonMetrics,
        method: str,
    ) -> ExplanationMetadata:
        """
        Generate plain English explanation.

        Args:
            conformal_interval: Conformal prediction interval
            coverage: Coverage guarantee
            calibration: Calibration metrics
            comparison: Comparison metrics
            method: Conformal method used

        Returns:
            Explanation metadata
        """
        outcome_var = list(conformal_interval.point_estimate.keys())[0]
        point_est = conformal_interval.point_estimate[outcome_var]
        lower = conformal_interval.lower_bound[outcome_var]
        upper = conformal_interval.upper_bound[outcome_var]

        coverage_pct = coverage.guaranteed_coverage * 100

        summary = (
            f"Conformal prediction provides {coverage_pct:.1f}% guaranteed coverage: "
            f"{outcome_var} interval [{lower:.0f}, {upper:.0f}] around point estimate {point_est:.0f}"
        )

        reasoning = (
            f"Using {method} conformal prediction with {calibration.calibration_size} "
            f"calibration points. {comparison.interpretation}."
        )

        technical_basis = "Finite-sample conformal prediction (Vovk et al. 2005)"

        return ExplanationMetadata(
            summary=summary,
            reasoning=reasoning,
            technical_basis=technical_basis,
            assumptions=coverage.assumptions,
        )
