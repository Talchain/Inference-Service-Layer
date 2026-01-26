"""
Shared constants for ISL analysis.

These thresholds should be used consistently across all analysis components
and should align with PLoT's thresholds to prevent semantic divergence.
"""

# =============================================================================
# Path Validation Thresholds
# =============================================================================

# Edge must have meaningful probability to be considered in structural path
DEFAULT_EXISTS_PROBABILITY_THRESHOLD = 1e-6

# Edge must have meaningful effect strength to be considered in effective path
DEFAULT_STRENGTH_THRESHOLD = 1e-6


# =============================================================================
# Identical Options Detection
# =============================================================================

# Tolerance for comparing intervention values (floating-point precision)
IDENTICAL_OPTIONS_VALUE_TOLERANCE = 1e-9


# =============================================================================
# Degenerate Outcomes Detection
# =============================================================================

# Relative spread threshold - if max-min is less than 1% of max, outcomes are degenerate
DEGENERATE_RELATIVE_THRESHOLD = 0.01  # 1%


# =============================================================================
# Partial Status Thresholds
# =============================================================================

# Minimum ratio of valid samples for "computed" status (vs "partial")
MIN_VALID_RATIO = 0.8  # 80%


# =============================================================================
# Numerical Stability
# =============================================================================

# Fraction of samples with issues that triggers warning
MAX_NUMERICAL_WARNINGS_RATIO = 0.1  # 10%


# =============================================================================
# Response Versioning
# =============================================================================

# Default response version (for backward compatibility)
DEFAULT_RESPONSE_VERSION = 1

# Current V2 schema version
RESPONSE_SCHEMA_VERSION_V2 = "2.0"


# =============================================================================
# Baseline Protection (P2-ISL-5)
# =============================================================================

# Minimum baseline magnitude for safe division (epsilon guard)
BASELINE_EPSILON = 1e-8


# =============================================================================
# Zero Variance Detection
# =============================================================================

# Values below this threshold are considered effectively zero variance
# (accounts for floating point precision in numerical computations)
ZERO_VARIANCE_TOLERANCE = 1e-10


# =============================================================================
# Factor Sensitivity Elasticity Calculation
# =============================================================================

# Epsilon values for stabilising elasticity computation when baseline or
# factor values are near zero (e.g., binary factors 0/1).
# These prevent division-by-zero while preserving meaningful sensitivity values.
# Note: Values tuned for typical factor scales; may need calibration.
FACTOR_SENSITIVITY_BASELINE_EPSILON = 0.01  # Min denominator for baseline_mean
FACTOR_SENSITIVITY_VALUE_EPSILON = 0.01  # Min denominator for factor mean_value

# Maximum elasticity magnitude for presentation (prevents extreme UX values)
# Elasticity beyond this is clamped to ±ELASTICITY_CLAMP_MAX
ELASTICITY_CLAMP_MAX = 100.0
