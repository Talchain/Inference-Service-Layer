"""Experiment 1 — Higher-K sweep of marginal_switch_probability.

Question: at the production default K=100 (resolution floor 0.01), are the
uniform zeros observed on live graphs TRUE zeros or under-resolution?

Method: the production estimator `_compute_marginal_switch_probability` is
called with explicit k_samples (harness-level parameterisation; no src change)
over K in {100, 1000, 10000, 100000}:

- margin family: single-decisive-edge graphs whose flip probability is known
  in closed form (truncated-normal CDF at zero), spanning knife-edge ->
  effectively-zero, plus a structural TRUE ZERO edge in every graph;
- repo fixtures: every edge of the three pinned graphs in
  tests/benchmarks/sample_variants.json;
- end-to-end: full analyze() runs under the K override, capturing the
  marginal values the wire response actually reports for fragile edges.

Each cell is repeated across pinned global seeds (which rotate every per-edge
SHA-256 sub-seed). Per-edge classification:

- TRUE ZERO               p_hat = 0 at every K and seed (rule-of-three bound)
- UNDER_RESOLUTION_NONZERO p_hat = 0 at K=100 for all seeds, > 0 pooled at K_max
- UNSTABLE                cross-seed dispersion inconsistent with binomial noise
- RESOLVED_NONZERO        otherwise

Run:  poetry run python benchmarks/science-validation/exp1_higher_k.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    SEEDS,
    build_request,
    make_evaluator,
    marginal_switch_probability,
    override_marginal_k,
    rule_of_three_upper,
    save_result,
)
from graphs import MarginCase, fixture_requests, margin_cases  # noqa: E402
from scipy import stats  # noqa: E402

from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

K_LEVELS_QUICK = [100, 1000, 10000]
K_LEVELS_FULL = [100, 1000, 10000, 100000]
CHI2_ALPHA = 0.01


def sweep_edge(
    payload: Dict[str, Any],
    edge: Tuple[str, str],
    k_levels: List[int],
    seeds: List[int],
) -> List[Dict[str, Any]]:
    """Direct production-estimator sweep for one edge; returns result rows."""
    request = build_request(payload)
    request, evaluator = make_evaluator(request)
    rows = []
    for k in k_levels:
        for seed in seeds:
            p_hat = marginal_switch_probability(request, evaluator, edge, seed, k)
            rows.append({"k": k, "seed": seed, "p_hat": p_hat, "flips": round(p_hat * k)})
    return rows


def classify(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify one edge from its sweep rows (see module docstring)."""
    by_k: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_k[r["k"]].append(r)
    k_min, k_max = min(by_k), max(by_k)

    total_draws = sum(r["k"] for r in rows)
    total_flips = sum(r["flips"] for r in rows)
    zero_at_default = all(r["flips"] == 0 for r in by_k[k_min])
    pooled_at_max = sum(r["flips"] for r in by_k[k_max]) / sum(r["k"] for r in by_k[k_max])

    # Cross-seed binomial homogeneity per K (Pearson chi-square)
    min_pvalue = 1.0
    for k, group in by_k.items():
        flips = [r["flips"] for r in group]
        n_seeds = len(flips)
        pooled = sum(flips) / (k * n_seeds)
        if n_seeds < 2 or pooled <= 0.0 or pooled >= 1.0:
            continue
        expected = k * pooled
        chi2 = sum((f - expected) ** 2 for f in flips) / (expected * (1.0 - pooled))
        pvalue = float(stats.chi2.sf(chi2, df=n_seeds - 1))
        min_pvalue = min(min_pvalue, pvalue)

    if total_flips == 0:
        label = "TRUE_ZERO"
    elif min_pvalue < CHI2_ALPHA:
        label = "UNSTABLE"
    elif zero_at_default and pooled_at_max > 0.0:
        label = "UNDER_RESOLUTION_NONZERO"
    else:
        label = "RESOLVED_NONZERO"

    return {
        "classification": label,
        "zero_at_k100_all_seeds": zero_at_default,
        "pooled_p_at_k_max": pooled_at_max,
        "pooled_p_all_draws": total_flips / total_draws,
        "rule_of_three_upper_bound": rule_of_three_upper(total_draws) if total_flips == 0 else None,
        "min_homogeneity_pvalue": min_pvalue,
        "k_max": k_max,
    }


def run_margin_family(
    cases: List[MarginCase], k_levels: List[int], seeds: List[int]
) -> List[Dict[str, Any]]:
    out = []
    for case in cases:
        for edge, kind in [
            (case.decisive_edge, "decisive"),
            (case.structural_zero_edge, "structural_zero"),
        ]:
            if edge is None:
                continue
            rows = sweep_edge(case.payload, edge, k_levels, seeds)
            summary = classify(rows)
            entry: Dict[str, Any] = {
                "family": "margin",
                "case": case.name,
                "regime": case.regime,
                "edge": f"{edge[0]}->{edge[1]}",
                "edge_role": kind,
                "analytic_p": case.analytic_flip_probability if kind == "decisive" else 0.0,
                **summary,
                "rows": rows,
            }
            # Consistency of the pooled estimate with the analytic truth
            if kind == "decisive":
                p = case.analytic_flip_probability
                n = sum(r["k"] for r in rows)
                x = sum(r["flips"] for r in rows)
                # Exact binomial two-sided p-value against the analytic truth
                entry["binom_pvalue_vs_analytic"] = float(
                    stats.binomtest(x, n, p).pvalue if n > 0 else 1.0
                )
            out.append(entry)
    return out


def run_fixture_family(k_levels: List[int], seeds: List[int]) -> List[Dict[str, Any]]:
    out = []
    for idx, payload in enumerate(fixture_requests()):
        for e in payload["graph"]["edges"]:
            edge = (e["from"], e["to"])
            rows = sweep_edge(payload, edge, k_levels, seeds)
            summary = classify(rows)
            out.append(
                {
                    "family": "fixture",
                    "case": f"sample_variants[{idx}]",
                    "edge": f"{edge[0]}->{edge[1]}",
                    "edge_role": "fixture",
                    "analytic_p": None,
                    **summary,
                    "rows": rows,
                }
            )
    return out


def run_end_to_end(k_levels: List[int], seeds: List[int]) -> List[Dict[str, Any]]:
    """Full analyze() under the K override: what the wire response reports."""
    out = []
    for idx, payload in enumerate(fixture_requests()):
        for k in k_levels:
            for seed in seeds:
                req = build_request({**payload, "seed": seed, "request_id": f"scival-e2e-{idx}"})
                with override_marginal_k(k):
                    response = RobustnessAnalyzerV2().analyze(req)
                fragile = response.robustness.fragile_edges_enhanced or []
                out.append(
                    {
                        "case": f"sample_variants[{idx}]",
                        "k": k,
                        "seed": seed,
                        "fragile_edges": [
                            {
                                "edge": f.edge_id,
                                "marginal_switch_probability": f.marginal_switch_probability,
                            }
                            for f in fragile
                        ],
                    }
                )
    return out


def reproducibility_check(k_levels: List[int]) -> Dict[str, Any]:
    """Same seed, same K, run twice -> identical estimates (bitwise)."""
    case = margin_cases()[0]
    request = build_request(case.payload)
    request, evaluator = make_evaluator(request)
    mismatches = []
    for k in k_levels:
        a = marginal_switch_probability(request, evaluator, case.decisive_edge, 42, k)
        b = marginal_switch_probability(request, evaluator, case.decisive_edge, 42, k)
        if a != b:
            mismatches.append({"k": k, "first": a, "second": b})
    return {"identical_on_repeat": not mismatches, "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced seeds and K<=10000")
    args = parser.parse_args()

    k_levels = K_LEVELS_QUICK if args.quick else K_LEVELS_FULL
    seeds = SEEDS["exp1_global_seeds"][: 2 if args.quick else None]

    t0 = time.time()
    margin_results = run_margin_family(margin_cases(), k_levels, seeds)
    fixture_results = run_fixture_family(k_levels, seeds)
    e2e_results = run_end_to_end(K_LEVELS_QUICK if args.quick else K_LEVELS_FULL, seeds[:2])
    repro = reproducibility_check(k_levels)
    elapsed = time.time() - t0

    path = save_result(
        "exp1_higher_k" + ("_quick" if args.quick else ""),
        {
            "config": {"k_levels": k_levels, "seeds": seeds, "chi2_alpha": CHI2_ALPHA},
            "reproducibility": repro,
            "margin_family": margin_results,
            "fixture_family": fixture_results,
            "end_to_end": e2e_results,
            "elapsed_seconds": round(elapsed, 1),
        },
    )

    print(f"exp1 complete in {elapsed:.1f}s -> {path}")
    for r in margin_results + fixture_results:
        print(
            f"  {r['family']:8s} {r['case']:22s} {r['edge']:18s} "
            f"{r['classification']:26s} pooled@Kmax={r['pooled_p_at_k_max']:.6f}"
            + (f" analytic={r['analytic_p']:.6f}" if r.get("analytic_p") is not None else "")
        )


if __name__ == "__main__":
    main()
