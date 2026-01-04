"""
Degenerate outcome detection for ISL V2 response format.

Detects when all options produce nearly identical outcomes,
which indicates a problem with the analysis configuration.
"""

from typing import List, Optional

from src.constants import DEGENERATE_RELATIVE_THRESHOLD
from src.models.critique import DEGENERATE_OUTCOMES
from src.models.response_v2 import CritiqueV2, OptionResultV2


def detect_degenerate_outcomes(
    option_results: List[OptionResultV2],
    threshold: float = DEGENERATE_RELATIVE_THRESHOLD,
) -> Optional[CritiqueV2]:
    """
    Detect if all options produce nearly identical outcomes.

    Uses relative spread calculation:
    - spread = max(p50) - min(p50)
    - relative_spread = spread / max(abs(p50))
    - If relative_spread < threshold, outcomes are degenerate

    Args:
        option_results: List of option results with outcomes
        threshold: Relative spread threshold (default 1%)

    Returns:
        CritiqueV2 if degenerate, None otherwise
    """
    # Only consider computed options
    computed_options = [r for r in option_results if r.status == "computed"]

    if len(computed_options) < 2:
        return None

    # Get p50 (median) values
    p50_values = [r.outcome.p50 for r in computed_options]

    # Calculate relative spread
    max_abs = max(abs(v) for v in p50_values)
    if max_abs == 0:
        # All zeros - degenerate
        return DEGENERATE_OUTCOMES.build()

    spread = max(p50_values) - min(p50_values)
    relative_spread = spread / max_abs

    if relative_spread < threshold:
        return DEGENERATE_OUTCOMES.build()

    return None
