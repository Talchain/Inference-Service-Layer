"""
Threshold Identification API endpoint.

Provides POST /api/v1/analysis/thresholds for detecting parameter values
where option rankings change.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header

from src.api.dependencies import get_threshold_identifier
from src.models.requests import ThresholdIdentificationRequest
from src.models.responses import ThresholdIdentificationResponse
from src.services.threshold_identifier import ThresholdIdentifier
from src.models.isl_metadata import MetadataBuilder

router = APIRouter()


@router.post("/thresholds", response_model=ThresholdIdentificationResponse)
async def identify_thresholds(
    request: ThresholdIdentificationRequest,
    x_request_id: Optional[str] = Header(None),
    threshold_identifier: ThresholdIdentifier = Depends(get_threshold_identifier)
):
    """
    Identify parameter thresholds where option rankings change.

    Analyzes parameter sweeps to find critical values where the relative
    ranking of options shifts. This helps users understand:
    - Which parameters are most sensitive (affect rankings most)
    - At what values rankings change (decision thresholds)
    - Which options are affected by parameter changes

    **Algorithm**:
    1. For each parameter sweep:
       - Iterate through values in order
       - Rank options by score at each value
       - Detect where ranking changes from previous value
       - Record threshold value and affected options
    2. Rank parameters by sensitivity (number of changes)
    3. Identify most sensitive range for each parameter

    **Tie Handling**:
    - Options within `confidence_threshold` are considered tied
    - Tied options maintain stable alphabetical ordering
    - Only meaningful ranking changes are reported

    **Use Cases**:
    - Sensitivity analysis: Which parameters matter most?
    - Decision boundaries: When does the best option change?
    - Robustness testing: How stable are rankings?

    **Note**: PLoT provides pre-computed scores at different parameter values.
    ISL only analyzes the provided data (does not run inference).

    Args:
        request: Threshold identification request with parameter sweeps
        x_request_id: Optional request ID from header

    Returns:
        Thresholds, sensitivity ranking, and analysis metadata
    """
    # Generate request ID
    request_id = request.request_id or x_request_id or f"req_{uuid.uuid4().hex[:12]}"
    metadata_builder = MetadataBuilder(request_id)

    # Identify thresholds
    thresholds, sensitivity_ranking, total_thresholds, monotonic_params = (
        threshold_identifier.identify(
            parameter_sweeps=request.parameter_sweeps,
            baseline_ranking=request.baseline_ranking,
            confidence_threshold=request.confidence_threshold
        )
    )

    # Build response
    response = ThresholdIdentificationResponse(
        thresholds=thresholds,
        sensitivity_ranking=sensitivity_ranking,
        total_thresholds=total_thresholds,
        monotonic_parameters=monotonic_params
    )

    # Add metadata
    response.metadata = metadata_builder.build(
        algorithm="sequential_ranking_comparison"
    )

    return response
