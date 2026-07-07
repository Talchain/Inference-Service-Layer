"""Experiment 4 (stretch) — calibration groundwork for probability_of_goal.

Question: when ISL reports P(goal >= threshold) = 0.72, is the truth 0.72?

Method: single-edge graphs whose goal distribution is known in closed form —
goal = x * B * T with B ~ Bernoulli(exists_probability) and
T ~ TruncNormal(mean, std, [-1, 1]) — so P(goal >= threshold) is analytic
(truncated-normal CDF x existence mixture). ISL's estimate is compared with
the truth over a grid of (mean, std, exists_probability, threshold) at
n_samples = 10000 across replicate seeds:

- goal kind "chance": auto-scaled noise does NOT apply — this isolates the
  Monte Carlo propagation. Expected: unbiased, ~95% of |z| <= 1.96.
- goal kind "outcome": auto-scaled noise applies (N(0, std(samples)) added
  per sample, the "provisional_pending_pilot_calibration" heuristic). The
  clean truth no longer holds; the noisy truth is computed numerically by
  convolving the analytic goal distribution with the noise, and the
  distortion (reported P vs clean truth) is quantified.

This is within-model calibration groundwork (complements the within-model SBC
in src/services/sbc_validator.py); it does not validate the model against
reality.

Run:  poetry run python benchmarks/science-validation/exp4_calibration.py [--quick]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time

from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.disable(logging.WARNING)

from _lib import SEEDS, build_request, save_result  # noqa: E402
from graphs import calibration_cases  # noqa: E402
from scipy import integrate, stats  # noqa: E402

from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

N_SAMPLES = 10000  # schema maximum


def goal_distribution_moments(mean: float, std: float, exists_p: float, x: float) -> float:
    """Analytic std of goal = x * B * T (see module docstring)."""
    a, b = (-1.0 - mean) / std, (1.0 - mean) / std
    mu_t, var_t = stats.truncnorm.stats(a, b, loc=mean, scale=std, moments="mv")
    e_s = exists_p * float(mu_t)
    e_s2 = exists_p * (float(var_t) + float(mu_t) ** 2)
    return x * math.sqrt(max(e_s2 - e_s**2, 0.0))


def noisy_truth(mean: float, std: float, exists_p: float, x: float, threshold: float) -> float:
    """P(goal + N(0, sigma_goal) >= threshold) with sigma_goal matched to the
    analytic goal std — the population analogue of the auto-noise heuristic."""
    sigma = goal_distribution_moments(mean, std, exists_p, x)
    if sigma <= 1e-12:
        return 1.0 if 0.0 >= threshold else 0.0
    a, b = (-1.0 - mean) / std, (1.0 - mean) / std

    def integrand(s: float) -> float:
        return float(
            stats.truncnorm.pdf(s, a, b, loc=mean, scale=std)
            * stats.norm.cdf((x * s - threshold) / sigma)
        )

    exists_part, _ = integrate.quad(integrand, -1.0, 1.0, limit=200)
    absent_part = float(stats.norm.cdf((0.0 - threshold) / sigma))
    return exists_p * exists_part + (1.0 - exists_p) * absent_part


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer replicate seeds")
    args = parser.parse_args()

    base_seed = SEEDS["exp4_request_seed"]
    seeds = [base_seed + 1000 * i for i in range(2 if args.quick else 5)]

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for goal_kind in ("chance", "outcome"):
        for case in calibration_cases(goal_kind):
            grid = case.payload["graph"]["edges"][0]
            mean = grid["strength"]["mean"]
            std = grid["strength"]["std"]
            exists_p = grid["exists_probability"]
            threshold = case.payload["goal_threshold"]
            for seed in seeds:
                request = build_request(
                    {**case.payload, "seed": seed, "request_id": f"scival-cal-{case.name}"}
                )
                response = RobustnessAnalyzerV2().analyze(request)
                noise_applied = bool(response.metadata.auto_noise_applied)
                for result in response.results:
                    x = {"opt_hi": 1.0, "opt_lo": 0.5}[result.option_id]
                    clean = case.option_truths[result.option_id]
                    truth = (
                        noisy_truth(mean, std, exists_p, x, threshold) if noise_applied else clean
                    )
                    p_hat = result.probability_of_goal
                    if p_hat is None:
                        continue
                    se = math.sqrt(max(truth * (1.0 - truth), 1e-12) / N_SAMPLES)
                    z = (p_hat - truth) / se
                    rows.append(
                        {
                            "goal_kind": goal_kind,
                            "case": case.name,
                            "option": result.option_id,
                            "seed": seed,
                            "auto_noise_applied": noise_applied,
                            "p_hat": p_hat,
                            "clean_truth": clean,
                            "truth_used": truth,
                            "error_vs_truth": p_hat - truth,
                            "error_vs_clean": p_hat - clean,
                            "z_vs_truth": z,
                            "covered_95": abs(z) <= 1.96,
                        }
                    )
    elapsed = time.time() - t0

    def agg(kind: str) -> Dict[str, Any]:
        sub = [r for r in rows if r["goal_kind"] == kind]
        n = len(sub)
        return {
            "cells": n,
            "mean_error_vs_truth": sum(r["error_vs_truth"] for r in sub) / n,
            "max_abs_error_vs_truth": max(abs(r["error_vs_truth"]) for r in sub),
            "coverage_95": sum(1 for r in sub if r["covered_95"]) / n,
            "mean_error_vs_clean": sum(r["error_vs_clean"] for r in sub) / n,
            "max_abs_error_vs_clean": max(abs(r["error_vs_clean"]) for r in sub),
        }

    summary = {"chance_no_noise": agg("chance"), "outcome_auto_noise": agg("outcome")}
    path = save_result(
        "exp4_calibration" + ("_quick" if args.quick else ""),
        {
            "config": {"n_samples": N_SAMPLES, "seeds": seeds},
            "summary": summary,
            "rows": rows,
            "elapsed_seconds": round(elapsed, 1),
        },
    )
    print(f"exp4 complete in {elapsed:.1f}s -> {path}")
    import json as _json

    print(_json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
