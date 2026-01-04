"""
Numerical stability utilities for ISL.

P2-ISL-5: Provides epsilon-guarded calculations to prevent inf/nan from
baseline near-zero division.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from src.constants import BASELINE_EPSILON
from src.models.critique import BASELINE_NEAR_ZERO, MONTE_CARLO_FAILED, NUMERICAL_INSTABILITY
from src.models.response_v2 import CritiqueV2

logger = logging.getLogger(__name__)


def safe_sensitivity(
    delta_outcome: float,
    baseline_outcome: float,
    delta_input: float = 1.0,
) -> Tuple[float, bool]:
    """
    Calculate sensitivity with epsilon protection.

    Args:
        delta_outcome: Change in outcome value
        baseline_outcome: Baseline outcome value (potential divisor)
        delta_input: Change in input value (default 1.0)

    Returns:
        (sensitivity_value, was_guarded): Tuple of sensitivity and guard flag
    """
    abs_baseline = abs(baseline_outcome)

    if abs_baseline < BASELINE_EPSILON:
        # Baseline too close to zero — use epsilon-guarded calculation
        guarded_baseline = (
            math.copysign(BASELINE_EPSILON, baseline_outcome)
            if baseline_outcome != 0
            else BASELINE_EPSILON
        )
        sensitivity = delta_outcome / (guarded_baseline * delta_input)
        logger.debug(
            f"Epsilon-guarded sensitivity: baseline={baseline_outcome:.2e}, "
            f"guarded={guarded_baseline:.2e}, result={sensitivity:.4f}"
        )
        return (sensitivity, True)

    sensitivity = delta_outcome / (baseline_outcome * delta_input)
    return (sensitivity, False)


def safe_percent_change(
    new_value: float,
    baseline_value: float,
) -> Tuple[float, bool]:
    """
    Calculate percent change with epsilon protection.

    Args:
        new_value: New value
        baseline_value: Baseline value (potential divisor)

    Returns:
        (percent_change, was_guarded): Tuple of percent change and guard flag
    """
    abs_baseline = abs(baseline_value)

    if abs_baseline < BASELINE_EPSILON:
        guarded_baseline = (
            math.copysign(BASELINE_EPSILON, baseline_value)
            if baseline_value != 0
            else BASELINE_EPSILON
        )
        pct_change = ((new_value - baseline_value) / guarded_baseline) * 100
        return (pct_change, True)

    pct_change = ((new_value - baseline_value) / baseline_value) * 100
    return (pct_change, False)


def check_baseline_near_zero(
    baseline_outcome: float,
    critiques: List[CritiqueV2],
) -> bool:
    """
    Check if baseline is near zero and emit warning critique if so.

    Args:
        baseline_outcome: Baseline outcome value
        critiques: List to append critique to

    Returns:
        True if baseline was near zero, False otherwise
    """
    if abs(baseline_outcome) < BASELINE_EPSILON:
        critiques.append(
            BASELINE_NEAR_ZERO.build(value=f"{baseline_outcome:.2e}")
        )
        logger.warning(
            f"Baseline near zero: {baseline_outcome:.2e}, "
            "sensitivity calculations will be epsilon-guarded"
        )
        return True
    return False


def validate_mc_samples(
    samples: np.ndarray,
) -> Tuple[np.ndarray, List[CritiqueV2]]:
    """
    Validate and clean MC samples, returning critiques for issues.

    Args:
        samples: NumPy array of MC samples

    Returns:
        (cleaned_samples, critiques): Tuple of cleaned samples and issues
    """
    critiques: List[CritiqueV2] = []

    # Check for non-finite values
    non_finite_mask = ~np.isfinite(samples)
    non_finite_count = int(non_finite_mask.sum())

    if non_finite_count > 0:
        # Get valid samples
        valid_samples = samples[np.isfinite(samples)]

        if len(valid_samples) == 0:
            # All samples are non-finite — critical failure
            critiques.append(
                MONTE_CARLO_FAILED.build(
                    reason="All samples produced non-finite values (inf/nan)"
                )
            )
            return samples, critiques

        # Replace with median of valid samples
        median_val = float(np.median(valid_samples))
        cleaned_samples = np.where(non_finite_mask, median_val, samples)

        critiques.append(
            NUMERICAL_INSTABILITY.build(
                invalid_count=non_finite_count,
                total_count=len(samples),
            )
        )

        logger.warning(
            f"MC samples cleaned: {non_finite_count}/{len(samples)} "
            f"non-finite values replaced with median ({median_val:.4f})"
        )

        return cleaned_samples, critiques

    return samples, critiques


def compute_analysis_status_with_numerical_checks(
    base_status: str,
    critiques: List[CritiqueV2],
) -> str:
    """
    Determine status considering numerical warnings.

    If baseline near-zero warning was emitted, status should be 'partial'
    (not 'computed') to indicate results need interpretation.

    Args:
        base_status: Status from normal analysis
        critiques: List of critiques including any numerical warnings

    Returns:
        Adjusted analysis status
    """
    has_numerical_warnings = any(
        c.code in ("BASELINE_NEAR_ZERO", "NUMERICAL_INSTABILITY")
        for c in critiques
    )

    if base_status == "computed" and has_numerical_warnings:
        return "partial"

    return base_status
