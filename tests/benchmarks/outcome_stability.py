"""
Outcome stability measurement for the CEE edge-stability benchmark (1A).

Takes N graph variants (same decision, different CEE seeds → different edge
parameters) and reports whether ISL's recommended option changes across them.

This is benchmark infrastructure — not a production endpoint.  It calls
ISL's internal ``RobustnessAnalyzerV2`` directly (no HTTP, no PLoT).

Usage (CLI)::

    python -m tests.benchmarks.outcome_stability \
        --graphs path/to/variants.json --seed 42

Usage (pytest fixture / direct import)::

    from tests.benchmarks.outcome_stability import measure_outcome_stability

    result = measure_outcome_stability(graph_variants, options, goal_node_id, seed=42)

Input format (``variants.json``)::

    {
      "options": [
        {"id": "opt_a", "label": "Option A", "interventions": {"price": 0.3}},
        {"id": "opt_b", "label": "Option B", "interventions": {"price": 0.7}}
      ],
      "goal_node_id": "revenue",
      "seed": 42,
      "n_samples": 1000,
      "graphs": [
        {"nodes": [...], "edges": [...]},
        {"nodes": [...], "edges": [...]}
      ]
    }

Output::

    {
      "recommended_option_changes": bool,
      "outcome_mean_cv": float,
      "option_rank_stability": float,
      "per_run_winners": ["opt_a", "opt_a", ...],
      "n_runs": int
    }
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.robustness_v2 import (
    GraphV2,
    InterventionOption,
    RobustnessRequestV2,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OutcomeStabilityResult:
    """Metrics returned by :func:`measure_outcome_stability`."""

    recommended_option_changes: bool
    outcome_mean_cv: float
    option_rank_stability: float
    per_run_winners: List[str]
    n_runs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_option_changes": self.recommended_option_changes,
            "outcome_mean_cv": self.outcome_mean_cv,
            "option_rank_stability": self.option_rank_stability,
            "per_run_winners": self.per_run_winners,
            "n_runs": self.n_runs,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

_CV_EPSILON = 0.05  # same epsilon-floor as CEE harness


def measure_outcome_stability(
    graphs: List[GraphV2],
    options: List[InterventionOption],
    goal_node_id: str,
    seed: int = 42,
    n_samples: int = 1000,
) -> OutcomeStabilityResult:
    """Run ISL inference on each graph variant and compare results.

    Args:
        graphs: N graph variants (same structure, different edge parameters).
                All must share the same option set.
        options: Decision options (identical across all variants).
        goal_node_id: Target outcome node.
        seed: RNG seed — same for all runs so variance comes only from graph
              differences, not MC sampling.
        n_samples: Monte Carlo samples per run.

    Returns:
        :class:`OutcomeStabilityResult` with the three CEE-harness metrics
        plus debugging info.

    Raises:
        ValueError: If fewer than 1 graph is supplied or option sets don't
                    match across variants.
    """
    if not graphs:
        raise ValueError("At least one graph variant is required")

    option_ids = sorted(opt.id for opt in options)

    # --- Validate option identity precondition ---
    # All graph variants must contain the nodes targeted by option interventions.
    # Options are shared (single list), so IDs/labels are structurally identical.
    # What CAN differ is the graph structure — verify intervention targets exist.
    intervention_node_ids = set()
    for opt in options:
        intervention_node_ids.update(opt.interventions.keys())

    for i, graph in enumerate(graphs):
        graph_node_ids = {n.id for n in graph.nodes}
        missing = intervention_node_ids - graph_node_ids
        if missing:
            raise ValueError(
                f"Graph variant {i} is missing intervention target node(s): "
                f"{sorted(missing)}. All variants must have identical option sets."
            )

    analyzer = RobustnessAnalyzerV2()
    per_run_winners: List[str] = []
    # outcome means for the first-run winner, across all runs
    ref_option_means: List[float] = []
    # per-run ranked option lists (for rank stability)
    per_run_rankings: List[List[str]] = []
    ref_winner: Optional[str] = None

    for i, graph in enumerate(graphs):
        request = RobustnessRequestV2(
            request_id=f"stability-run-{i}",
            graph=graph,
            options=options,
            goal_node_id=goal_node_id,
            n_samples=n_samples,
            seed=seed,
        )
        response = analyzer.analyze(request)

        winner = response.recommended_option_id
        per_run_winners.append(winner)

        if i == 0:
            ref_winner = winner

        # Build win_probability map for ranking
        win_probs: Dict[str, float] = {}
        outcome_means: Dict[str, float] = {}
        for opt_result in response.results:
            win_probs[opt_result.option_id] = opt_result.win_probability
            outcome_means[opt_result.option_id] = opt_result.outcome_distribution.mean

        # Collect the mean outcome of the reference option (run-1 winner)
        ref_option_means.append(outcome_means.get(ref_winner, 0.0))

        # Rank by win_probability descending; tie-break alphabetically by option_id
        ranked = sorted(
            option_ids,
            key=lambda oid: (-win_probs.get(oid, 0.0), oid),
        )
        per_run_rankings.append(ranked)

    # --- Metric 1: recommended_option_changes ---
    recommended_option_changes = len(set(per_run_winners)) > 1

    # --- Metric 2: outcome_mean_cv ---
    means_arr = np.array(ref_option_means)
    if len(means_arr) <= 1:
        outcome_mean_cv = 0.0
    else:
        std = float(np.std(means_arr, ddof=1))
        denom = max(abs(float(np.mean(means_arr))), _CV_EPSILON)
        outcome_mean_cv = std / denom

    # --- Metric 3: option_rank_stability ---
    # Consensus-based: fraction of runs matching the most common ranking.
    # e.g. if 4/5 runs produce the same ranking → 0.8, regardless of which
    # run is first.
    if len(per_run_rankings) <= 1:
        option_rank_stability = 1.0
    else:
        ranking_counts = Counter(tuple(r) for r in per_run_rankings)
        most_common_count = ranking_counts.most_common(1)[0][1]
        option_rank_stability = most_common_count / len(per_run_rankings)

    return OutcomeStabilityResult(
        recommended_option_changes=recommended_option_changes,
        outcome_mean_cv=round(outcome_mean_cv, 8),
        option_rank_stability=round(option_rank_stability, 4),
        per_run_winners=per_run_winners,
        n_runs=len(graphs),
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _load_variants(path: str) -> Dict[str, Any]:
    """Load graph variants JSON file."""
    with open(path) as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure ISL outcome stability across graph variants (1A benchmark)"
    )
    parser.add_argument(
        "--graphs", required=True,
        help="Path to variants JSON file (see module docstring for format)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (same for all runs, default: 42)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Monte Carlo samples per run (default: 1000)",
    )
    args = parser.parse_args(argv)

    data = _load_variants(args.graphs)

    # Parse graphs
    graphs = [GraphV2(**g) for g in data["graphs"]]

    # Parse options
    options = [InterventionOption(**o) for o in data["options"]]

    goal_node_id = data["goal_node_id"]
    seed = args.seed if args.seed is not None else data.get("seed", 42)
    n_samples = args.n_samples or data.get("n_samples", 1000)

    t0 = time.time()
    result = measure_outcome_stability(graphs, options, goal_node_id, seed, n_samples)
    elapsed = time.time() - t0

    output = result.to_dict()
    output["elapsed_s"] = round(elapsed, 3)
    output["ms_per_run"] = round(elapsed / result.n_runs * 1000, 1) if result.n_runs else 0

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
