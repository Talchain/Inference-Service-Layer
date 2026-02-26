"""
CLI entry point for within-model SBC validation (4A benchmark).

Runs Simulation-Based Calibration on a supplied graph and reports CI coverage
rates, PIT histogram uniformity, and per-trial timing.

This is benchmark infrastructure — not a production endpoint.

Usage::

    python -m tests.benchmarks.sbc_validation \
        --graph path/to/graph.json --seed 42 --n-trials 200

Input format (``graph.json``)::

    {
      "options": [
        {"id": "opt_a", "label": "Option A", "interventions": {"price": 0.3}},
        {"id": "opt_b", "label": "Option B", "interventions": {"price": 0.7}}
      ],
      "goal_node_id": "revenue",
      "graph": {"nodes": [...], "edges": [...]}
    }

Output: JSON report to stdout + human-readable summary to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

from src.models.robustness_v2 import GraphV2, InterventionOption
from src.services.sbc_validator import run_sbc_validation


def _load_graph_data(path: str) -> Dict[str, Any]:
    """Load graph JSON file."""
    with open(path) as f:
        return json.load(f)


def _print_summary(result_dict: Dict[str, Any]) -> None:
    """Print human-readable summary to stderr."""
    print("\n" + "=" * 60, file=sys.stderr)
    print("  SBC Validation Report (within-model)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    print(f"\n  Trials:     {result_dict['n_trials']}", file=sys.stderr)
    print(f"  Seed:       {result_dict['seed']}", file=sys.stderr)
    print(f"  Elapsed:    {result_dict['elapsed_s']:.1f}s  "
          f"({result_dict['ms_per_trial']:.0f} ms/trial)", file=sys.stderr)

    print("\n  CI Coverage:", file=sys.stderr)
    for cr in result_dict["coverage_results"]:
        flag = " *** FLAGGED ***" if cr["flagged"] else ""
        print(f"    {cr['nominal_level']*100:.0f}% CI: "
              f"observed={cr['observed_coverage']*100:.1f}%  "
              f"deviation={cr['deviation']*100:.1f}pp{flag}", file=sys.stderr)

    print(f"\n  PIT Uniformity:", file=sys.stderr)
    print(f"    chi2 = {result_dict['pit_chi2_statistic']:.2f}, "
          f"p = {result_dict['pit_chi2_p_value']:.4f}", file=sys.stderr)
    print(f"    Uniform: {'YES' if result_dict['pit_uniform'] else 'NO'}", file=sys.stderr)
    print(f"    Histogram: {result_dict['pit_histogram']}", file=sys.stderr)

    verdict = "CALIBRATED" if result_dict["calibrated"] else "NOT CALIBRATED"
    print(f"\n  Verdict: {verdict}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run within-model SBC validation on ISL inference (4A benchmark)"
    )
    parser.add_argument(
        "--graph", required=True,
        help="Path to graph JSON file (see module docstring for format)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=200,
        help="Number of SBC trials (default: 200)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="MC samples per trial (default: 500)",
    )
    parser.add_argument(
        "--ref-option", type=str, default=None,
        help="Option ID to evaluate (default: first option)",
    )
    parser.add_argument(
        "--ci-levels", type=str, default="0.5,0.8,0.9",
        help="Comma-separated CI levels (default: 0.5,0.8,0.9)",
    )
    args = parser.parse_args(argv)

    data = _load_graph_data(args.graph)

    graph = GraphV2(**data["graph"])
    options = [InterventionOption(**o) for o in data["options"]]
    goal_node_id = data["goal_node_id"]
    ci_levels = [float(x) for x in args.ci_levels.split(",")]

    t0 = time.time()
    result = run_sbc_validation(
        graph=graph,
        options=options,
        goal_node_id=goal_node_id,
        n_trials=args.n_trials,
        n_samples=args.n_samples,
        ci_levels=ci_levels,
        seed=args.seed,
        ref_option_id=args.ref_option,
    )
    elapsed = time.time() - t0

    result_dict = result.to_dict()
    result_dict["total_elapsed_s"] = round(elapsed, 3)

    # JSON report to stdout
    print(json.dumps(result_dict, indent=2))

    # Human-readable summary to stderr
    _print_summary(result_dict)


if __name__ == "__main__":
    main()
