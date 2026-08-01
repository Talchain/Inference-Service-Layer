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
import os
import time
from datetime import datetime, timezone
from typing import List, Literal, Optional

from src.__version__ import __version__ as engine_version
from src.constants import MIN_VALID_RATIO
from src.models.critique import INTERNAL_ERROR
from src.models.response_v2 import (
    SUPPRESSED_ATTR_FACTOR_SENSITIVITY,
    ConditionalWinnerV2,
    CorrelationModelV2,
    CritiqueV2,
    DiagnosticsV2,
    FactorFlipValueV2,
    FactorSensitivityV2,
    InferenceWarning,
    ISLV2Error422,
    ISLResponseV2,
    OptionResultV2,
    PathDecompositionV2,
    RequestEchoV2,
    RobustnessResultV2,
    SamplePopulationProvenanceV2,
    StabilityThresholdsResponse,
)

logger = logging.getLogger(__name__)


# Arch step 1 (2026-07-26) — per-metric noise provenance.
#
# One ISLResponseV2 envelope carries metrics from TWO sample populations. The
# split is deliberate and documented at its source (the B2 CRN-fix comment in
# `RobustnessAnalyzerV2.analyze`), but before this it was disclosed only by the
# `auto_noise_applied` boolean, which tells a consumer that noise ran and not
# which of the numbers in front of it the noise reached.
#
# PRE-noise: joint Common-Random-Numbers metrics. The auto-noise draw is
# INDEPENDENT per option, so computing these on noised samples breaks CRN
# alignment (it adds a max-over-independent-noise premium and makes regret
# disagree with win_probability).
PRE_NOISE_METRICS: tuple = (
    "expected_regret",
    "win_probability",
    "factor_evppi",
    "factor_evpc",
    "decision_evpi",
)

# POST-noise: marginal distribution metrics, kept on the noised samples so they
# stay mutually consistent.
POST_NOISE_METRICS: tuple = (
    "p05",
    "p10",
    "p50",
    "p90",
    "mean",
    "cvar_10",
    "probability_of_goal",
)


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
        seed_source: Optional[Literal["client_provided", "server_computed"]] = None,
    ):
        """
        Initialize response builder.

        Args:
            request_id: Request ID for correlation
            request_echo: Echo of request parameters
            seed_used: RNG seed used for determinism (P2-ISL-1)
            seed_source: Origin of seed ('client_provided' or 'server_computed')
        """
        self.request_id = request_id
        self.request_echo = request_echo
        self.seed_used = seed_used
        self.seed_source = seed_source
        self.start_time = time.time()

        self.critiques: List[CritiqueV2] = []
        self.inference_warnings: List[InferenceWarning] = []
        self.diagnostics: Optional[DiagnosticsV2] = None
        self.options: Optional[List[OptionResultV2]] = None
        self.robustness: Optional[RobustnessResultV2] = None
        self.factor_sensitivity: Optional[List[FactorSensitivityV2]] = None
        self.stability_thresholds: Optional[StabilityThresholdsResponse] = None  # 3C
        self.conditional_winners: Optional[List[ConditionalWinnerV2]] = None
        # S2 (D-23.8) HONEST RELABEL: per-factor win-probability sensitivity (was
        # factor_evpi — NOT value-of-information). Enhancement passthrough.
        self.p_win_sensitivity: Optional[list] = None
        # S2 (D-23.8): per-factor regression EVPPI in outcome units. Enhancement
        # passthrough; None => omitted on the wire.
        self.factor_evppi: Optional[list] = None
        # S4 (D-23.8): per-lever EVPC (value of control) in outcome units. Enhancement
        # passthrough; None => omitted on the wire (request-driven by control_candidates).
        self.factor_evpc: Optional[list] = None
        # Decision-level EVPI (S1 — A3 VOI, D-23.8): min_o expected_regret[o] in
        # outcome units. None until set from the option regret population; absent
        # (exclude_none) on the wire when no regret population exists.
        self.decision_evpi: Optional[float] = None
        # Auto-noise disclosure (B3): None until explicitly set from V1 metadata.
        # Preserves False as False; never coerced to None on the path through the route.
        self.auto_noise_applied: Optional[bool] = None
        # Arch step 1: per-metric population provenance, derived alongside the
        # boolean above by set_auto_noise_applied so the two cannot desync.
        self.sample_population_provenance: Optional[SamplePopulationProvenanceV2] = None
        # T1-6: path decomposition passthrough (request-gated; additive)
        self.path_decomposition: Optional[PathDecompositionV2] = None
        # ROADMAP 2.228-F3: per-root-factor flip thresholds (request-gated by
        # include_factor_flips; additive). None => absent on the wire, which is
        # the state every consumer that has not opted in must observe.
        self.factor_flip_values: Optional[List[FactorFlipValueV2]] = None
        # T1-5: reference-option disclosure for sensitivity analyses (additive)
        self.sensitivity_reference_option_id: Optional[str] = None
        # B3-S1: correlated-factors disclosure (present iff correlation active)
        self.correlation_model: Optional[CorrelationModelV2] = None

    def add_critique(self, critique: CritiqueV2) -> None:
        """Add a single critique."""
        self.critiques.append(critique)

    def add_critiques(self, critiques: List[CritiqueV2]) -> None:
        """Add multiple critiques."""
        self.critiques.extend(critiques)

    def set_inference_warnings(self, warnings: List[InferenceWarning]) -> None:
        """Set inference warnings from analyzer output."""
        self.inference_warnings = warnings

    def set_diagnostics(self, diagnostics: DiagnosticsV2) -> None:
        """Set diagnostics."""
        self.diagnostics = diagnostics

    def set_results(
        self,
        options: List[OptionResultV2],
        robustness: Optional[RobustnessResultV2] = None,
        factor_sensitivity: Optional[List[FactorSensitivityV2]] = None,
        stability_thresholds: Optional[StabilityThresholdsResponse] = None,
        p_win_sensitivity: Optional[list] = None,
        factor_evppi: Optional[list] = None,
        factor_evpc: Optional[list] = None,
    ) -> None:
        """Set analysis results."""
        self.options = options
        self.robustness = robustness
        self.factor_sensitivity = factor_sensitivity
        self.stability_thresholds = stability_thresholds
        self.p_win_sensitivity = p_win_sensitivity
        self.factor_evppi = factor_evppi
        self.factor_evpc = factor_evpc

    def set_conditional_winners(
        self, conditional_winners: Optional[List[ConditionalWinnerV2]]
    ) -> None:
        """Set conditional winner analysis results."""
        self.conditional_winners = conditional_winners

    def set_decision_evpi(self, decision_evpi: Optional[float]) -> None:
        """Set the decision-level EVPI (S1 — A3 VOI, D-23.8).

        ``decision_evpi`` = min over options of the pre-noise joint expected regret
        = E[max]−max E on the CRN population, in outcome units. None => omitted.
        """
        self.decision_evpi = decision_evpi

    def set_auto_noise_applied(
        self, flag: Optional[bool], unnoised_constraint_node_ids: Optional[List[str]] = None
    ) -> None:
        """Set the auto-noise disclosure flag for the V2 envelope (B3).

        Arch step 1 (2026-07-26): also derives the per-metric
        `sample_population_provenance` block, so the two disclosures can never
        desync — there is one setter and one source of truth for both.
        """
        self.auto_noise_applied = flag
        if flag is None:
            self.sample_population_provenance = None
            return

        populations: dict = {metric: "model_only" for metric in PRE_NOISE_METRICS}
        populations.update(
            {metric: ("noise_inflated" if flag else "model_only") for metric in POST_NOISE_METRICS}
        )
        self.sample_population_provenance = SamplePopulationProvenanceV2(
            auto_scaled_noise_applied=flag,
            noise_scale=(
                "1.0x model std added per outcome/risk sample (~sqrt(2) spread inflation)"
                if flag
                else None
            ),
            metric_populations=populations,
            unnoised_constraint_node_ids=list(unnoised_constraint_node_ids or []),
        )

    def set_path_decomposition(self, path_decomposition: Optional[PathDecompositionV2]) -> None:
        """Set the path decomposition passthrough (T1-6 wire completeness)."""
        self.path_decomposition = path_decomposition

    def set_factor_flip_values(self, factor_flip_values: Optional[List[FactorFlipValueV2]]) -> None:
        """Set the per-factor flip-threshold block (ROADMAP 2.228-F3)."""
        self.factor_flip_values = factor_flip_values

    def set_sensitivity_reference_option_id(self, option_id: Optional[str]) -> None:
        """Set the reference-option disclosure for sensitivity analyses (T1-5)."""
        self.sensitivity_reference_option_id = option_id

    def set_correlation_model(self, correlation_model: Optional[CorrelationModelV2]) -> None:
        """Set the correlated-factors disclosure block (B3-S1)."""
        self.correlation_model = correlation_model

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
        elif (
            self.correlation_model is not None
            and SUPPRESSED_ATTR_FACTOR_SENSITIVITY
            in self.correlation_model.suppressed_attributions
        ):
            # B3-S1: active correlation deliberately WITHHELD factor_sensitivity
            # because per-factor OAT attributions are non-separable when factors
            # co-move. This is a principled suppression, not a skip — the
            # correlation_model block names the reason. (Previously reported
            # "skipped", which under-explained and read as "nothing to compute".)
            factor_sensitivity_status = "suppressed"
        else:
            factor_sensitivity_status = "skipped"

        # P2-ISL-1: Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Get build commit hash from environment (Render sets RENDER_GIT_COMMIT)
        build_commit = os.environ.get("RENDER_GIT_COMMIT", "dev")[:7]

        return ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version=engine_version,
            build=build_commit,
            timestamp=timestamp,
            analysis_status=analysis_status,
            robustness_status=robustness_status,
            factor_sensitivity_status=factor_sensitivity_status,
            status_reason=status_reason,
            critiques=self.critiques,
            inference_warnings=self.inference_warnings,
            request_echo=self.request_echo,
            diagnostics=self.diagnostics,
            options=self.options,
            robustness=self.robustness,
            factor_sensitivity=self.factor_sensitivity,
            conditional_winners=self.conditional_winners,
            stability_thresholds=self.stability_thresholds,  # 3C
            p_win_sensitivity=self.p_win_sensitivity,  # S2 — A3 VOI relabel (D-23.8)
            factor_evppi=self.factor_evppi,  # S2 — A3 VOI regression EVPPI (D-23.8)
            factor_evpc=self.factor_evpc,  # S4 — A3 value-of-control (D-23.8)
            decision_evpi=self.decision_evpi,  # S1 — A3 VOI (D-23.8)
            path_decomposition=self.path_decomposition,  # T1-6
            factor_flip_values=self.factor_flip_values,  # ROADMAP 2.228-F3
            sensitivity_reference_option_id=self.sensitivity_reference_option_id,  # T1-5
            correlation_model=self.correlation_model,  # B3-S1
            auto_noise_applied=self.auto_noise_applied,
            sample_population_provenance=self.sample_population_provenance,
            request_id=self.request_id,
            processing_time_ms=processing_time,
            seed_used=self.seed_used,
            seed_source=self.seed_source,
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

        # Intentionally omit auto_noise_applied from error responses: on
        # failed / blocked / error paths the analyser may not have populated
        # the metadata flag, so emitting it here would risk surfacing a
        # default (False) that wasn't actually computed. Defaulting to None
        # leaves the field absent under exclude_none=True, which PLoT's
        # extractor already handles conservatively.
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
