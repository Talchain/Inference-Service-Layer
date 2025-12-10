"""
Risk Adjustment API endpoint.

Provides POST /api/v1/analysis/risk-adjust for computing certainty equivalents
based on user risk profiles.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Header

from src.models.requests import RiskAdjustmentRequest
from src.models.responses import RiskAdjustmentResponse
from src.services.risk_adjuster import risk_adjuster
from src.utils.metadata_builder import MetadataBuilder

router = APIRouter()


@router.post("/risk-adjust", response_model=RiskAdjustmentResponse)
async def adjust_for_risk(
    request: RiskAdjustmentRequest,
    x_request_id: Optional[str] = Header(None)
):
    """
    Apply risk adjustment to option scores based on risk profile.

    Uses mean-variance approach to compute certainty equivalents:
    - **Risk averse**: Penalizes variance (prefers safer options)
    - **Risk neutral**: No adjustment (expected value only)
    - **Risk seeking**: Rewards variance (prefers riskier options)

    **Algorithm**:
    - Risk averse: CE = mean - (coefficient/2) × variance
    - Risk neutral: CE = mean
    - Risk seeking: CE = mean + (coefficient/2) × variance

    **Input Formats**:
    - Mean-variance: Provide `mean` and `std_dev` for each option
    - Percentile: Provide `p10`, `p50`, `p90` for each option

    **Response**:
    - Adjusted scores sorted by certainty equivalent (best to worst)
    - Indicates if rankings changed after adjustment
    - Detailed ranking changes (if applicable)
    - Plain English interpretation of adjustment impact

    **Example Use Case**:
    CEE provides option scores with uncertainty. User's risk profile
    (from CEE risk assessment) determines how to weight variance vs. expected value.

    Args:
        request: Risk adjustment request with options and risk profile
        x_request_id: Optional request ID from header

    Returns:
        Risk-adjusted scores with rankings and interpretation
    """
    # Generate request ID
    request_id = request.request_id or x_request_id or f"req_{uuid.uuid4().hex[:12]}"
    metadata_builder = MetadataBuilder(request_id)

    # Perform risk adjustment
    adjusted_scores, rankings_changed, ranking_changes, interpretation = (
        risk_adjuster.adjust(
            options=request.options,
            risk_coefficient=request.risk_coefficient,
            risk_type=request.risk_type
        )
    )

    # Build algorithm name for metadata
    algorithm_name = f"mean_variance_{request.risk_type}"

    # Build response
    response = RiskAdjustmentResponse(
        adjusted_scores=adjusted_scores,
        rankings_changed=rankings_changed,
        ranking_changes=ranking_changes if rankings_changed else None,
        risk_interpretation=interpretation
    )

    # Add metadata
    response.metadata = metadata_builder.build(algorithm=algorithm_name)

    return response
