"""Experiment 2 — empirical validation of the EVPI noise floor.

The production floor is z95_worst_case_bernoulli_diff:
    floor(n) = 1.96 * sqrt(0.5 / n)
and entries with |EVPI| < floor are labelled below_resolution (labels only,
never clamped). Question: is the floor conservative, tight, or leaky?

Production caps EVPI at n = min(request.n_samples, 500), so n in {2000, 10000}
cannot be reached through analyze(). This harness replicates the thin
orchestration of `_compute_evpi` exactly (same seed derivations: baseline
streams seed+100/seed+101; per-factor SHA-256 sub-seeds) but with n as a
parameter, and calls the PRODUCTION metric `_compute_evpi_metric` unchanged.

Design: two graphs (knife-edge P(win) ~ 0.5, the floor's worst case; and a
comfortable winner far from 0.5) each with three uncertain factors:
  - driver:   real information value
  - weak:     small true effect
  - stranded: no causal path to the goal => true EVPI is exactly 0
For n in {500, 2000, 10000} and 20 replicate seeds we measure:
  - empirical SD of the EVPI estimator vs the floor's implied worst-case SE
  - the fraction of below-floor estimates whose sign flips across seeds
  - leakage: estimates >= floor (labelled "resolved") for the true-zero factor

Run:  poetry run python benchmarks/science-validation/exp2_evpi_floor.py [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time

from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import SEEDS, build_request, make_evaluator, save_result  # noqa: E402
from graphs import evpi_payload  # noqa: E402

from src.models.robustness_v2 import RobustnessRequestV2  # noqa: E402
from src.services.robustness_analyzer_v2 import (  # noqa: E402
    DualUncertaintySampler,
    FactorSampler,
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
    evpi_noise_floor,
)
from src.utils.rng import SeededRNG  # noqa: E402

N_LEVELS_FULL = [500, 2000, 10000]
N_LEVELS_QUICK = [500, 2000]


def evpi_at_n(
    request: RobustnessRequestV2,
    evaluator: SCMEvaluatorV2,
    seed: int,
    n: int,
    policy_option_id: str,
) -> Dict[str, float]:
    """Replicate _compute_evpi's orchestration with parameterised n.

    Seed derivations and sampler construction mirror
    RobustnessAnalyzerV2._compute_evpi exactly; the metric itself is the
    unmodified production _compute_evpi_metric.
    """
    analyzer = RobustnessAnalyzerV2()
    assert request.parameter_uncertainties

    baseline_sampler = DualUncertaintySampler(request.graph.edges, SeededRNG(seed + 100))
    baseline_factor_sampler = FactorSampler(
        request.graph.nodes, request.parameter_uncertainties, SeededRNG(seed + 101)
    )
    baseline_metric = analyzer._compute_evpi_metric(
        request, baseline_sampler, baseline_factor_sampler, evaluator, n, None, policy_option_id
    )

    out: Dict[str, float] = {}
    for uncertainty in request.parameter_uncertainties:
        modified = [u for u in request.parameter_uncertainties if u.node_id != uncertainty.node_id]
        factor_seed = int(
            hashlib.sha256(f"{seed}:evpi:{uncertainty.node_id}".encode()).hexdigest()[:8], 16
        )
        perfect_sampler = DualUncertaintySampler(request.graph.edges, SeededRNG(factor_seed))
        perfect_factor_sampler = FactorSampler(
            request.graph.nodes, modified if modified else None, SeededRNG(factor_seed + 1)
        )
        perfect_metric = analyzer._compute_evpi_metric(
            request, perfect_sampler, perfect_factor_sampler, evaluator, n, None, policy_option_id
        )
        out[uncertainty.node_id] = perfect_metric - baseline_metric
    return out


def summarise(estimates: List[float], floor: float, reference_sign: int) -> Dict[str, Any]:
    """Summary statistics for one (factor, n) cell across replicate seeds."""
    below = [e for e in estimates if abs(e) < floor]
    resolved = [e for e in estimates if abs(e) >= floor]
    sign_flips_below = [e for e in below if e != 0.0 and (1 if e > 0 else -1) != reference_sign]
    return {
        "n_estimates": len(estimates),
        "mean": statistics.fmean(estimates),
        "sd": statistics.stdev(estimates) if len(estimates) > 1 else 0.0,
        "floor": floor,
        "worst_case_se": floor / 1.96,
        "n_below_floor": len(below),
        "n_resolved": len(resolved),
        "below_floor_sign_flip_fraction": (len(sign_flips_below) / len(below) if below else None),
        "resolved_values": resolved,
        "min": min(estimates),
        "max": max(estimates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer seeds and n levels")
    args = parser.parse_args()

    n_levels = N_LEVELS_QUICK if args.quick else N_LEVELS_FULL
    seeds = SEEDS["exp2_replicate_seeds"][: 5 if args.quick else None]
    policy_seed = SEEDS["exp2_policy_seed"]

    t0 = time.time()
    results = []
    for knife_edge in (False, True):
        payload = evpi_payload(knife_edge=knife_edge)
        request = build_request({**payload, "seed": policy_seed, "request_id": "scival-evpi"})
        # Fixed decision policy from a single production analyze() run — mirrors
        # production semantics (policy held fixed to avoid policy-switch
        # confounding) and keeps it constant across all replicate seeds.
        full_response = RobustnessAnalyzerV2().analyze(request)
        policy = full_response.recommended_option_id
        baseline_p_win = next(
            r.win_probability for r in full_response.results if r.option_id == policy
        )
        request_f, evaluator = make_evaluator(request)

        graph_result: Dict[str, Any] = {
            "graph": "knife_edge" if knife_edge else "comfortable",
            "policy_option_id": policy,
            "policy_p_win": baseline_p_win,
            "cells": [],
        }

        # Per-seed estimates at each n
        for n in n_levels:
            floor = evpi_noise_floor(n)
            per_factor: Dict[str, List[float]] = {}
            for seed in seeds:
                estimates = evpi_at_n(request_f, evaluator, seed, n, policy)
                for factor, value in estimates.items():
                    per_factor.setdefault(factor, []).append(value)
            # Reference sign: sign of the large-n consensus (mean at the
            # largest n gets computed on the final pass; for the true-zero
            # factor the reference is arbitrary — that is the point).
            for factor, values in per_factor.items():
                mean = statistics.fmean(values)
                reference_sign = 1 if mean >= 0 else -1
                cell = {
                    "factor": factor,
                    "true_evpi_is_zero": factor == "stranded",
                    "n": n,
                    **summarise(values, floor, reference_sign),
                    "estimates": values,
                }
                graph_result["cells"].append(cell)
        results.append(graph_result)
    elapsed = time.time() - t0

    path = save_result(
        "exp2_evpi_floor" + ("_quick" if args.quick else ""),
        {
            "config": {
                "n_levels": n_levels,
                "replicate_seeds": seeds,
                "policy_seed": policy_seed,
                "floor_formula": "1.96 * sqrt(0.5 / n)",
            },
            "graphs": results,
            "elapsed_seconds": round(elapsed, 1),
        },
    )
    print(f"exp2 complete in {elapsed:.1f}s -> {path}")
    for g in results:
        print(f"  graph={g['graph']} policy={g['policy_option_id']} p_win={g['policy_p_win']:.3f}")
        for c in g["cells"]:
            flip = c["below_floor_sign_flip_fraction"]
            print(
                f"    {c['factor']:9s} n={c['n']:6d} mean={c['mean']:+.5f} sd={c['sd']:.5f} "
                f"floor={c['floor']:.5f} below={c['n_below_floor']:2d}/{c['n_estimates']:2d} "
                f"sign_flip={flip if flip is not None else 'n/a'}"
            )


if __name__ == "__main__":
    main()
