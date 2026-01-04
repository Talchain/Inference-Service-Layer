"""
Response builder for ISL V2 response format.

Provides consistent response construction with proper status determination
and error sanitisation.

P2 Brief Alignment:
- Adds seed_used for determinism
- Adds timestamp in ISO 8601 format
- Provides build_422_response for unwrapped 422 errors
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

from src.__version__ import __version__ as engine_version
from src.constants import MIN_VALID_RATIO
from src.models.critique import INTERNAL_ERROR
from src.models.response_v2 import (
    CritiqueV2,
    DiagnosticsV2,
    FactorSensitivityV2,
    ISLV2Error422,
    ISLResponseV2,
    OptionResultV2,
    RequestEchoV2,
    RobustnessResultV2,
)

logger = logging.getLogger(__name__)


def hash_node_id(node_id: str) -> str:
    """
    Hash node ID for logging (no sensitive data exposure).

    Args:
        node_id: Node identifier

    Returns:
        Truncated SHA-256 hash
    """
    return hashlib.sha256(node_id.encode()).hexdigest()[:12]


def determine_option_status(n_valid: int, n_total: int) -> str:
    """
    Determine option status based on valid sample ratio.

    Args:
        n_valid: Number of valid samples
        n_total: Total samples

    Returns:
        Status string: "computed", "partial", or "failed"
    """
    if n_valid == 0:
        return "failed"

    ratio = n_valid / n_total
    if ratio < MIN_VALID_RATIO:
        return "partial"

    return "computed"


class ResponseBuilder:
    """Builds V2 responses consistently."""

    def __init__(
        self,
        request_id: str,
        request_echo: RequestEchoV2,
        seed_used: Optional[str] = None,
    ):
        """
        Initialize response builder.

        Args:
            request_id: Request ID for correlation
            request_echo: Echo of request parameters
            seed_used: RNG seed used for determinism (P2-ISL-1)
        """
        self.request_id = request_id
        self.request_echo = request_echo
        self.seed_used = seed_used
        self.start_time = time.time()

        self.critiques: List[CritiqueV2] = []
        self.diagnostics: Optional[DiagnosticsV2] = None
        self.options: Optional[List[OptionResultV2]] = None
        self.robustness: Optional[RobustnessResultV2] = None
        self.factor_sensitivity: Optional[List[FactorSensitivityV2]] = None

    def add_critique(self, critique: CritiqueV2) -> None:
        """Add a single critique."""
        self.critiques.append(critique)

    def add_critiques(self, critiques: List[CritiqueV2]) -> None:
        """Add multiple critiques."""
        self.critiques.extend(critiques)

    def set_diagnostics(self, diagnostics: DiagnosticsV2) -> None:
        """Set diagnostics."""
        self.diagnostics = diagnostics

    def set_results(
        self,
        options: List[OptionResultV2],
        robustness: Optional[RobustnessResultV2] = None,
        factor_sensitivity: Optional[List[FactorSensitivityV2]] = None,
    ) -> None:
        """Set analysis results."""
        self.options = options
        self.robustness = robustness
        self.factor_sensitivity = factor_sensitivity

    def _determine_analysis_status(self) -> str:
        """Determine overall analysis status."""
        has_blockers = any(c.severity == "blocker" for c in self.critiques)

        if has_blockers:
            return "failed"

        if self.options is None:
            return "failed"

        if all(o.status == "computed" for o in self.options):
            return "computed"

        if any(o.status == "computed" for o in self.options):
            return "partial"

        return "failed"

    def _determine_status_reason(self, analysis_status: str) -> Optional[str]:
        """Determine status reason (sanitised)."""
        if analysis_status == "computed":
            return None

        blockers = [c for c in self.critiques if c.severity == "blocker"]
        if blockers:
            # Return first blocker code, not the full message
            return f"Blocked by: {blockers[0].code}"

        if analysis_status == "partial":
            return "Some options could not be computed"

        return "Analysis could not be completed"

    def get_processing_time_ms(self) -> int:
        """Get current processing time in milliseconds."""
        return int((time.time() - self.start_time) * 1000)

    def build(self) -> ISLResponseV2:
        """Build the final response."""
        processing_time = self.get_processing_time_ms()

        analysis_status = self._determine_analysis_status()
        status_reason = self._determine_status_reason(analysis_status)

        has_blockers = any(c.severity == "blocker" for c in self.critiques)

        # Robustness status
        if self.robustness is not None:
            robustness_status = "computed"
        elif has_blockers:
            robustness_status = "unavailable"
        else:
            robustness_status = "skipped"

        # Factor sensitivity status
        if self.factor_sensitivity is not None:
            factor_sensitivity_status = "computed"
        elif has_blockers:
            factor_sensitivity_status = "unavailable"
        else:
            factor_sensitivity_status = "skipped"

        # P2-ISL-1: Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        return ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version=engine_version,
            timestamp=timestamp,
            analysis_status=analysis_status,
            robustness_status=robustness_status,
            factor_sensitivity_status=factor_sensitivity_status,
            status_reason=status_reason,
            critiques=self.critiques,
            request_echo=self.request_echo,
            diagnostics=self.diagnostics,
            options=self.options,
            robustness=self.robustness,
            factor_sensitivity=self.factor_sensitivity,
            request_id=self.request_id,
            processing_time_ms=processing_time,
            seed_used=self.seed_used,
        )

    def build_422_response(self) -> ISLV2Error422:
        """
        Build unwrapped 422 error response (P2-ISL-3).

        Per P2 brief: Returns ISLV2Error422 directly, NOT wrapped in envelope.
        Use this for validation blockers that prevent analysis.
        """
        blockers = [c for c in self.critiques if c.severity == "blocker"]
        status_reason = blockers[0].message if blockers else "Validation failed"

        return ISLV2Error422(
            analysis_status="blocked",
            status_reason=status_reason,
            critiques=blockers,
            request_id=self.request_id,
        )

    def build_error_response(self, error: Exception) -> ISLResponseV2:
        """
        Build response for unexpected errors (sanitised).

        Args:
            error: The exception that occurred

        Returns:
            ISLResponseV2 with sanitised error information
        """
        processing_time = int((time.time() - self.start_time) * 1000)

        # Log full error internally
        logger.exception(f"Analysis error for request {self.request_id}: {error}")

        # Return sanitised critique
        self.critiques.append(INTERNAL_ERROR.build())

        return ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version=engine_version,
            analysis_status="failed",
            robustness_status="error",
            factor_sensitivity_status="error",
            status_reason="Internal error occurred",  # Sanitised, not str(error)
            critiques=self.critiques,
            request_echo=self.request_echo,
            diagnostics=self.diagnostics,
            request_id=self.request_id,
            processing_time_ms=processing_time,
        )


def build_request_echo(
    graph_node_count: int,
    graph_edge_count: int,
    options_count: int,
    goal_node_id: str,
    n_samples: int,
    response_version: int,
    include_diagnostics: bool,
) -> RequestEchoV2:
    """
    Build request echo from request parameters.

    Args:
        graph_node_count: Number of nodes
        graph_edge_count: Number of edges
        options_count: Number of options
        goal_node_id: Goal node ID (will be hashed)
        n_samples: Number of samples
        response_version: Response version requested
        include_diagnostics: Whether diagnostics were requested

    Returns:
        RequestEchoV2 with hashed sensitive data
    """
    return RequestEchoV2(
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        options_count=options_count,
        goal_node_id_hash=hash_node_id(goal_node_id),
        n_samples=n_samples,
        response_version_requested=response_version,
        include_diagnostics=include_diagnostics,
    )
