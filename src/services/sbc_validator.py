"""
Within-model Simulation-Based Calibration (SBC) for ISL inference (4A).

Generates synthetic ground-truth outcomes from known priors, runs ISL inference,
and checks that posterior CI coverage rates match nominal levels.

This is a **within-model** consistency check — it verifies that ISL's Monte Carlo
inference correctly propagates the uncertainty it is given, NOT that the model
reflects reality.

Usage (direct import)::

    from src.services.sbc_validator import run_sbc_validation

    result = run_sbc_validation(graph, options, goal_node_id, n_trials=200)

Reference:
    Talts et al. (2018) "Validating Bayesian Inference Algorithms with
    Simulation-Based Calibration"
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


# ---------------------------------------------------------------------------
# Result types (Task 2)
# ---------------------------------------------------------------------------


@dataclass
class CICoverageResult:
    """Coverage result for a single nominal CI level."""

    nominal_level: float  # e.g. 0.9
    observed_coverage: float  # fraction of trials where ground truth fell inside CI
    deviation: float  # |observed - nominal|
    flagged: bool  # deviation > 5pp


@dataclass
class SBCResult:
    """Full SBC validation result."""

    n_trials: int
    ci_levels: List[float]
    coverage_results: List[CICoverageResult]
    pit_histogram: List[int]  # counts per bin
    pit_n_bins: int
    pit_chi2_statistic: float
    pit_chi2_p_value: float
    pit_uniform: bool  # p > 0.05
    calibrated: bool  # all CI coverages within tolerance AND PIT uniform
    elapsed_s: float
    ms_per_trial: float
    seed: int
    label: str = "within-model"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "ci_levels": self.ci_levels,
            "coverage_results": [
                {
                    "nominal_level": cr.nominal_level,
                    "observed_coverage": round(cr.observed_coverage, 4),
                    "deviation": round(cr.deviation, 4),
                    "flagged": cr.flagged,
                }
                for cr in self.coverage_results
            ],
            "pit_histogram": self.pit_histogram,
            "pit_n_bins": self.pit_n_bins,
            "pit_chi2_statistic": round(self.pit_chi2_statistic, 4),
            "pit_chi2_p_value": round(self.pit_chi2_p_value, 4),
            "pit_uniform": self.pit_uniform,
            "calibrated": self.calibrated,
            "elapsed_s": round(self.elapsed_s, 3),
            "ms_per_trial": round(self.ms_per_trial, 1),
            "seed": self.seed,
            "label": self.label,
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CI_LEVELS = [0.5, 0.8, 0.9]
_COVERAGE_TOLERANCE = 0.05  # 5 percentage points
_PIT_N_BINS = 10
_PIT_ALPHA = 0.05  # chi-squared significance level


# ---------------------------------------------------------------------------
# Frozen-parameter injection (Task 3)
# ---------------------------------------------------------------------------


def _build_frozen_graph(
    graph: GraphV2,
    true_strengths: Dict[str, float],
    true_existence: Dict[str, bool],
) -> GraphV2:
    """Build a frozen graph with deterministic edge parameters.

    For each edge:
    - Sets mean to the sampled true_strength value.
    - Sets std to 0.0 (bypassing validation via model_construct).
    - Sets exists_probability to 1.0 (if edge exists) or 0.0 (if not).

    Args:
        graph: Original graph with uncertainty.
        true_strengths: edge_key ("from->to") → sampled true strength.
        true_existence: edge_key ("from->to") → whether edge exists in this world.

    Returns:
        New GraphV2 with frozen parameters.
    """
    frozen_edges = []
    for edge in graph.edges:
        edge_key = f"{edge.from_}->{edge.to}"
        exists = true_existence[edge_key]

        frozen_strength = StrengthDistribution.model_construct(
            mean=true_strengths[edge_key] if exists else 0.0,
            std=0.0,
        )
        frozen_edge = EdgeV2.model_construct(
            from_=edge.from_,
            to=edge.to,
            exists_probability=1.0 if exists else 0.0,
            strength=frozen_strength,
            label=edge.label,
            edge_type=getattr(edge, "edge_type", None),
        )
        frozen_edges.append(frozen_edge)

    return GraphV2.model_construct(
        nodes=copy.deepcopy(graph.nodes),
        edges=frozen_edges,
    )


def _sample_true_world(
    graph: GraphV2,
    rng: np.random.Generator,
) -> tuple[Dict[str, float], Dict[str, bool]]:
    """Sample a single 'true world' from the graph's prior distributions.

    For each edge:
    - existence: Bernoulli(exists_probability)
    - strength: Normal(mean, std) clipped to [-1, 1]

    Returns:
        (true_strengths, true_existence) dictionaries keyed by "from->to".
    """
    true_strengths: Dict[str, float] = {}
    true_existence: Dict[str, bool] = {}

    for edge in graph.edges:
        edge_key = f"{edge.from_}->{edge.to}"
        exists = rng.random() < edge.exists_probability
        true_existence[edge_key] = exists
        if exists:
            strength = float(np.clip(
                rng.normal(edge.strength.mean, edge.strength.std),
                -1.0, 1.0,
            ))
        else:
            strength = 0.0
        true_strengths[edge_key] = strength

    return true_strengths, true_existence


def _compute_ground_truth(
    frozen_graph: GraphV2,
    options: List[InterventionOption],
    goal_node_id: str,
    ref_option_id: str,
    seed: int,
) -> float:
    """Run ISL inference on a frozen graph to get the deterministic ground truth.

    Uses n_samples=1 because the frozen graph has no uncertainty (std=0,
    exists_probability ∈ {0, 1}).

    Args:
        frozen_graph: Graph with frozen parameters (std=0, exists ∈ {0,1}).
        options: Same options as the main inference.
        ref_option_id: Which option's outcome to extract.
        goal_node_id: Target node.
        seed: RNG seed (doesn't matter for frozen graph, but required by API).

    Returns:
        Ground-truth outcome value for the reference option.
    """
    analyzer = RobustnessAnalyzerV2()
    request = RobustnessRequestV2.model_construct(
        request_id="sbc-ground-truth",
        graph=frozen_graph,
        options=options,
        goal_node_id=goal_node_id,
        n_samples=1,
        seed=seed,
        analysis_types=["comparison"],
        confidence_level=0.95,
        parameter_uncertainties=None,
        goal_threshold=None,
        goal_constraints=None,
    )
    response = analyzer.analyze(request)
    for opt_result in response.results:
        if opt_result.option_id == ref_option_id:
            return opt_result.outcome_distribution.mean
    raise ValueError(f"Option {ref_option_id!r} not found in frozen-graph response")


# ---------------------------------------------------------------------------
# Core SBC harness (Task 1)
# ---------------------------------------------------------------------------


def run_sbc_validation(
    graph: GraphV2,
    options: List[InterventionOption],
    goal_node_id: str,
    *,
    n_trials: int = 200,
    n_samples: int = 500,
    ci_levels: Optional[List[float]] = None,
    seed: int = 42,
    ref_option_id: Optional[str] = None,
) -> SBCResult:
    """Run Simulation-Based Calibration on ISL inference.

    For each trial:
    1. Sample a "true world" from the graph's priors.
    2. Build a frozen graph and compute the ground-truth outcome.
    3. Run ISL inference with the original (uncertain) graph.
    4. Check whether ground truth falls inside each CI.
    5. Compute PIT value (quantile rank of ground truth in posterior).

    Args:
        graph: Causal graph with edge uncertainty.
        options: Decision options.
        goal_node_id: Target outcome node.
        n_trials: Number of SBC trials (default 200).
        n_samples: MC samples per inference trial (default 500).
        ci_levels: Nominal CI levels to check (default [0.5, 0.8, 0.9]).
        seed: Base RNG seed for reproducibility.
        ref_option_id: Which option's outcome to evaluate. If None, uses
                       the first option.

    Returns:
        SBCResult with coverage, PIT, and calibration diagnostics.
    """
    if ci_levels is None:
        ci_levels = list(_DEFAULT_CI_LEVELS)

    if ref_option_id is None:
        ref_option_id = options[0].id

    rng = np.random.Generator(np.random.PCG64(seed))
    analyzer = RobustnessAnalyzerV2()

    # Accumulators
    ci_hits: Dict[float, int] = {level: 0 for level in ci_levels}
    pit_values: List[float] = []

    t0 = time.time()

    # Check if the goal node is an outcome/risk node (auto-scaled noise applies).
    # ISL adds N(0, σ_model) noise to outcome/risk nodes, inflating the posterior.
    # To match the generative model, ground truth must include a noise draw too.
    goal_node_kind = ""
    for node in graph.nodes:
        if node.id == goal_node_id:
            goal_node_kind = getattr(node, "kind", "").lower()
            break
    goal_receives_noise = goal_node_kind in ("outcome", "risk")

    for trial_idx in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31))

        # Step 1: Sample true world
        true_strengths, true_existence = _sample_true_world(graph, rng)

        # Step 2: Build frozen graph and compute deterministic ground truth
        frozen_graph = _build_frozen_graph(graph, true_strengths, true_existence)
        ground_truth = _compute_ground_truth(
            frozen_graph, options, goal_node_id, ref_option_id, trial_seed,
        )

        # Step 3: Run ISL inference with the original uncertain graph
        request = RobustnessRequestV2(
            request_id=f"sbc-trial-{trial_idx}",
            graph=copy.deepcopy(graph),
            options=options,
            goal_node_id=goal_node_id,
            n_samples=n_samples,
            seed=trial_seed,
            analysis_types=["comparison"],
        )
        response = analyzer.analyze(request)

        # Extract posterior samples for the reference option
        posterior_samples: Optional[List[float]] = None
        for opt_result in response.results:
            if opt_result.option_id == ref_option_id:
                posterior_samples = opt_result.outcome_distribution.samples
                break

        if posterior_samples is None or len(posterior_samples) == 0:
            continue

        samples_arr = np.array(posterior_samples)

        # Step 3b: Match the generative model for outcome/risk nodes.
        # ISL's auto-scaled noise adds N(0, σ_model) to each sample, roughly
        # doubling the variance. The ground truth must include a draw from the
        # same noise distribution: σ_noise ≈ σ_posterior / √2.
        if goal_receives_noise:
            posterior_std = float(np.std(samples_arr))
            if posterior_std > 0:
                noise_std = posterior_std / np.sqrt(2.0)
                ground_truth += rng.normal(0.0, noise_std)

        # Step 4: Check CI coverage for each level
        for level in ci_levels:
            alpha = 1 - level
            lo = float(np.percentile(samples_arr, alpha / 2 * 100))
            hi = float(np.percentile(samples_arr, (1 - alpha / 2) * 100))
            if lo <= ground_truth <= hi:
                ci_hits[level] += 1

        # Step 5: Compute PIT value (quantile rank of ground truth in posterior)
        pit = float(np.mean(samples_arr <= ground_truth))
        pit_values.append(pit)

    elapsed = time.time() - t0

    # --- Build coverage results ---
    coverage_results = []
    for level in ci_levels:
        observed = ci_hits[level] / n_trials if n_trials > 0 else 0.0
        deviation = abs(observed - level)
        coverage_results.append(CICoverageResult(
            nominal_level=level,
            observed_coverage=observed,
            deviation=deviation,
            flagged=deviation > _COVERAGE_TOLERANCE,
        ))

    # --- Build PIT histogram and chi-squared test ---
    pit_arr = np.array(pit_values) if pit_values else np.array([])
    hist, _ = np.histogram(pit_arr, bins=_PIT_N_BINS, range=(0.0, 1.0))
    pit_histogram = hist.tolist()

    if len(pit_values) >= _PIT_N_BINS:
        expected = len(pit_values) / _PIT_N_BINS
        chi2_stat, chi2_p = stats.chisquare(hist, f_exp=[expected] * _PIT_N_BINS)
        chi2_stat = float(chi2_stat)
        chi2_p = float(chi2_p)
    else:
        chi2_stat = 0.0
        chi2_p = 1.0

    pit_uniform = chi2_p > _PIT_ALPHA

    # --- Overall calibration verdict ---
    all_coverage_ok = all(not cr.flagged for cr in coverage_results)
    calibrated = all_coverage_ok and pit_uniform

    return SBCResult(
        n_trials=n_trials,
        ci_levels=ci_levels,
        coverage_results=coverage_results,
        pit_histogram=pit_histogram,
        pit_n_bins=_PIT_N_BINS,
        pit_chi2_statistic=chi2_stat,
        pit_chi2_p_value=chi2_p,
        pit_uniform=pit_uniform,
        calibrated=calibrated,
        elapsed_s=elapsed,
        ms_per_trial=elapsed / n_trials * 1000 if n_trials > 0 else 0.0,
        seed=seed,
    )
