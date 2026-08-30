"""
Confounding sensitivity analysis for bidirected edges (3A-inference).

Quantifies how sensitive the recommendation is to unmeasured confounding.
For each non-identifiable (treatment, outcome) pair with bidirected edges,
introduces a synthetic confounder at varying strengths and checks whether
the recommended option flips.

This is an operational sensitivity heuristic — NOT a formal causal guarantee.
All outputs are provisional pending scientific review.
"""

from __future__ import annotations

import copy
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConfoundingPairResult:
    """Sensitivity result for a single bidirected pair."""

    node_a: str
    node_b: str
    flip_threshold_raw: Optional[float]  # Absolute edge strength that flips
    flip_threshold_relative: Optional[float]  # Multiple of median edge strength
    baseline_winner: str  # Option ID at confounder_strength=0
    flipped_winner: Optional[str]  # Option ID after flip, if any
    interpretation: str


@dataclass
class ConfoundingSensitivityResult:
    """Aggregate sensitivity result across all bidirected pairs."""

    pairs: List[ConfoundingPairResult]
    overall_robust: bool  # True if no pair flips at any strength
    median_edge_strength: float  # Graph-level reference for normalisation
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Normalised sweep multiples
# ---------------------------------------------------------------------------

# 8 steps as specified: 0.0×, 0.25×, 0.5×, 0.75×, 1.0×, 1.5×, 2.0×, 3.0×
_SWEEP_MULTIPLES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# Minimum median edge strength to avoid degenerate sweeps
_MIN_MEDIAN_STRENGTH = 0.01


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def _compute_median_edge_strength(graph: GraphV2) -> float:
    """Compute median |strength.mean| across all directed edges."""
    strengths = [
        abs(e.strength.mean)
        for e in graph.edges
        if e.edge_type != "bidirected" and abs(e.strength.mean) > 0
    ]
    if not strengths:
        return _MIN_MEDIAN_STRENGTH
    return max(statistics.median(strengths), _MIN_MEDIAN_STRENGTH)


def _extract_bidirected_pairs(graph: GraphV2) -> List[Tuple[str, str]]:
    """Extract deduplicated bidirected pairs from graph edges."""
    seen: set = set()
    pairs: List[Tuple[str, str]] = []
    for edge in graph.edges:
        if edge.edge_type == "bidirected":
            canonical = tuple(sorted([edge.from_, edge.to]))
            if canonical not in seen:
                seen.add(canonical)
                pairs.append(canonical)  # type: ignore[arg-type]
    return pairs


def _inject_confounder(
    graph: GraphV2,
    node_a: str,
    node_b: str,
    strength: float,
) -> GraphV2:
    """Create a copy of the graph with a synthetic confounder injected.

    Adds a virtual node ``__confounder_{a}_{b}`` with directed edges
    to both A and B at the given strength.  The original graph is NOT
    modified.
    """
    confounder_id = f"__confounder_{node_a}_{node_b}"

    # Deep-copy nodes and edges to avoid mutating the original
    new_nodes = [copy.deepcopy(n) for n in graph.nodes]
    new_edges = [copy.deepcopy(e) for e in graph.edges]

    # Add synthetic confounder node
    new_nodes.append(
        NodeV2(
            id=confounder_id,
            kind="chance",
            label=f"Unmeasured confounder ({node_a}, {node_b})",
            observed_state=ObservedState(value=0.5),
            intercept=0.0,
        )
    )

    # Add directed edges from confounder to both nodes
    for target in [node_a, node_b]:
        new_edges.append(
            EdgeV2(
                **{
                    "from": confounder_id,
                    "to": target,
                    "exists_probability": 1.0,
                    "strength": StrengthDistribution(mean=strength, std=0.01),
                    "edge_type": "directed",
                }
            )
        )

    return GraphV2(nodes=new_nodes, edges=new_edges)


def _build_intervention_options(
    options: List[Dict[str, Any]],
) -> List[InterventionOption]:
    """Convert lightweight option dicts to InterventionOption models."""
    result = []
    for opt in options:
        interventions = opt.get("interventions", {})
        float_interventions = {}
        for k, v in interventions.items():
            try:
                float_interventions[k] = float(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Option {opt['id']!r}: intervention value for {k!r} " f"is not numeric: {v!r}"
                )
        result.append(
            InterventionOption(
                id=opt["id"],
                label=opt.get("label", opt["id"]),
                interventions=float_interventions,
            )
        )
    return result


def _get_baseline_winner(
    analyzer: RobustnessAnalyzerV2,
    graph: GraphV2,
    options: List[InterventionOption],
    goal_node_id: str,
    seed: int,
    n_samples: int,
) -> Optional[str]:
    """Run baseline analysis; absence never licenses an invented winner."""
    request = RobustnessRequestV2(
        request_id="confounding-baseline",
        graph=graph,
        options=options,
        goal_node_id=goal_node_id,
        n_samples=n_samples,
        seed=seed,
    )
    response = analyzer.analyze(request)
    return response.recommended_option_id


def analyze_confounding_sensitivity(
    graph: GraphV2,
    options: List[Dict[str, Any]],
    goal_node_id: str,
    bidirected_pairs: List[Tuple[str, str]],
    seed: int = 42,
    n_samples: int = 1000,
) -> Optional[ConfoundingSensitivityResult]:
    """Run confounding sensitivity analysis for non-identifiable bidirected pairs.

    For each bidirected pair, injects a synthetic confounder at normalised
    strength multiples and checks whether the recommended option flips.

    Provisional — pending scientific review.

    Args:
        graph: V2 graph (may contain bidirected edges).
        options: List of option dicts with ``id``, ``label``, ``interventions``.
        goal_node_id: Target outcome node.
        bidirected_pairs: Pairs from identifiability analysis.
        seed: RNG seed (same for all sweeps).
        n_samples: MC samples per simulation run.

    Returns:
        ConfoundingSensitivityResult with per-pair sensitivity and aggregate.
    """
    t0 = time.time()

    if not bidirected_pairs:
        return ConfoundingSensitivityResult(
            pairs=[],
            overall_robust=True,
            median_edge_strength=_compute_median_edge_strength(graph),
            latency_ms=0.0,
        )

    intervention_options = _build_intervention_options(options)
    median_strength = _compute_median_edge_strength(graph)
    analyzer = RobustnessAnalyzerV2()

    # Compute baseline winner (confounder strength = 0)
    baseline_winner = _get_baseline_winner(
        analyzer, graph, intervention_options, goal_node_id, seed, n_samples
    )
    if baseline_winner is None:
        # This endpoint's current request has no objective carrier. Preserve the
        # structural identifiability result, but withhold recommendation sensitivity.
        return None

    pair_results: List[ConfoundingPairResult] = []

    for node_a, node_b in bidirected_pairs:
        flip_raw: Optional[float] = None
        flip_relative: Optional[float] = None
        flipped_winner: Optional[str] = None

        # Sweep through normalised multiples (skip 0.0× — that's the baseline)
        for multiple in _SWEEP_MULTIPLES[1:]:
            confounder_strength = multiple * median_strength
            modified_graph = _inject_confounder(graph, node_a, node_b, confounder_strength)

            request = RobustnessRequestV2(
                request_id=f"confounding-{node_a}-{node_b}-{multiple}x",
                graph=modified_graph,
                options=intervention_options,
                goal_node_id=goal_node_id,
                n_samples=n_samples,
                seed=seed,
            )
            response = analyzer.analyze(request)
            current_winner = response.recommended_option_id
            if current_winner is None:
                return None

            if current_winner != baseline_winner:
                flip_raw = round(confounder_strength, 6)
                flip_relative = multiple
                flipped_winner = current_winner
                break  # Early stopping

        # Build interpretation
        # Look up labels from graph nodes for readable interpretation
        node_labels: Dict[str, str] = {n.id: n.label for n in graph.nodes}
        label_a = node_labels.get(node_a, node_a)
        label_b = node_labels.get(node_b, node_b)

        if flip_relative is not None:
            interpretation = (
                f"Recommendation changes if unmeasured confounding between "
                f"{label_a} and {label_b} exceeds {flip_relative}× median edge "
                f"strength (provisional — pending scientific review)"
            )
        else:
            interpretation = (
                f"Recommendation is robust to unmeasured confounding between "
                f"{label_a} and {label_b} up to 3× median edge strength "
                f"(provisional — pending scientific review)"
            )

        pair_results.append(
            ConfoundingPairResult(
                node_a=node_a,
                node_b=node_b,
                flip_threshold_raw=flip_raw,
                flip_threshold_relative=flip_relative,
                baseline_winner=baseline_winner,
                flipped_winner=flipped_winner,
                interpretation=interpretation,
            )
        )

    overall_robust = all(p.flip_threshold_raw is None for p in pair_results)
    elapsed_ms = (time.time() - t0) * 1000

    return ConfoundingSensitivityResult(
        pairs=pair_results,
        overall_robust=overall_robust,
        median_edge_strength=round(median_strength, 6),
        latency_ms=round(elapsed_ms, 1),
    )
