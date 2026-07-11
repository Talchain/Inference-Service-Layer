"""
Agreement study: analytic flip thresholds vs MC bisection vs #71 stability
bands (Track S Phase 1 spike).

Compares, on the repo's fixtures (the three pinned sample_variants graphs and
the 12-node/17-edge synthetic production-shaped graph from
benchmarks/flip_stability_budget.py):

1. BASE PATH — the flip_mean each SHIPPED analyze() response carries in
   edge_e_values[] (flag off) vs src/services/analytic_flip.py's closed-form
   derivation on the same post-filter graph and deterministic evaluator.
2. STABILITY BANDS — the per-seed flip means the #71 band sweep puts on the
   wire (ISL_FLIP_STABILITY_BANDS=1, stability.seed_flip_means) vs the
   analytic derivation under the SAME SHA-256-derived child-seed backgrounds.
   None (background admits no flip) must agree as None.
3. SPEED — median wall time and exact evaluator.evaluate() call counts for
   the flip-threshold computation alone (bisection vs closed form), and for
   one full band sweep's inner searches.

Run:
    poetry run python benchmarks/analytic_flip_agreement.py

Prints per-fixture agreement tables; numbers are recorded in
docs/lanes/2026-07-11-analytic-flip-threshold-spike.md.
"""

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.analytic_flip import (
    analytic_edge_e_values,
    analytic_flip_mean_under_background,
)
from src.services.robustness_analyzer_v2 import (
    DualUncertaintySampler,
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
    compute_effective_seed,
    filter_inference_graph,
)
from src.utils.rng import SeededRNG

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS_PATH = REPO_ROOT / "tests" / "benchmarks" / "sample_variants.json"

FLIP_MEAN_TOL = 1e-5  # bisection resolution (~2e-6) + 6-dp wire rounding
TIMING_REPETITIONS = 21  # median reported; 1 untimed warm-up
DEFAULT_N_SEEDS = 5  # the #71 default the wire uses


class CountingEvaluator(SCMEvaluatorV2):
    """SCMEvaluatorV2 that counts evaluate() calls (hard cost metric)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def evaluate(self, *args: Any, **kwargs: Any) -> float:
        self.calls += 1
        return super().evaluate(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures — identical to benchmarks/flip_stability_budget.py
# ---------------------------------------------------------------------------


def variant_requests():
    variants = json.loads(VARIANTS_PATH.read_text())
    for idx, graph in enumerate(variants["graphs"]):
        yield f"sample_variants[{idx}]", {
            "request_id": f"analytic-agreement-variant-{idx}",
            "graph": graph,
            "options": variants["options"],
            "goal_node_id": variants["goal_node_id"],
            "seed": 42,
            "n_samples": variants["n_samples"],
            "include_e_values": True,
        }


def synthetic_request():
    """Deterministic 12-node / 17-edge layered DAG, 3 options, n_samples 500."""
    factors = [f"f{i}" for i in range(4)]
    mids = [f"m{i}" for i in range(4)]
    aggs = ["a0", "a1"]
    nodes = (
        [{"id": f, "kind": "factor", "label": f} for f in factors]
        + [{"id": m, "kind": "chance", "label": m} for m in mids]
        + [{"id": a, "kind": "chance", "label": a} for a in aggs]
        + [
            {"id": "risk", "kind": "risk", "label": "risk"},
            {"id": "goal", "kind": "outcome", "label": "goal"},
        ]
    )
    edges = []

    def edge(src, dst, mean, std, ep=1.0):
        edges.append(
            {
                "from": src,
                "to": dst,
                "exists_probability": ep,
                "strength": {"mean": mean, "std": std},
            }
        )

    edge("f0", "m0", 0.5, 0.12, 0.95)
    edge("f0", "m1", -0.3, 0.1)
    edge("f1", "m1", 0.4, 0.15, 0.9)
    edge("f1", "m2", 0.35, 0.08)
    edge("f2", "m2", -0.45, 0.12, 0.85)
    edge("f2", "m3", 0.25, 0.1)
    edge("f3", "m3", 0.6, 0.14, 0.95)
    edge("f3", "m0", -0.2, 0.09)
    edge("m0", "a0", 0.55, 0.1)
    edge("m1", "a0", 0.45, 0.12, 0.9)
    edge("m2", "a1", 0.5, 0.11)
    edge("m3", "a1", -0.35, 0.13, 0.9)
    edge("a0", "risk", -0.3, 0.1)
    edge("a0", "goal", 0.6, 0.1)
    edge("a1", "goal", 0.5, 0.12, 0.95)
    edge("a1", "risk", 0.2, 0.08)
    edge("risk", "goal", -0.4, 0.1)

    options = [
        {"id": "opt_a", "label": "A", "interventions": {"f0": 0.8, "f1": 0.2}},
        {"id": "opt_b", "label": "B", "interventions": {"f0": 0.3, "f2": 0.7}},
        {"id": "opt_c", "label": "C", "interventions": {"f1": 0.6, "f3": 0.4}},
    ]
    return "synthetic_12n_17e", {
        "request_id": "analytic-agreement-synthetic",
        "graph": {"nodes": nodes, "edges": edges},
        "options": options,
        "goal_node_id": "goal",
        "seed": 42,
        "n_samples": 500,
        "include_e_values": True,
    }


# ---------------------------------------------------------------------------
# Reconstruction of the analyzer's flip-search inputs
# ---------------------------------------------------------------------------


def prepared_request(request_dict: dict) -> RobustnessRequestV2:
    """The post-filter request the analyzer's flip search actually sees."""
    request = RobustnessRequestV2(**request_dict)
    filtered = filter_inference_graph(request.graph, log=False)
    if filtered is not request.graph:
        request = request.model_copy(update={"graph": filtered})
    return request


def band_backgrounds(
    request: RobustnessRequestV2, n_seeds: int = DEFAULT_N_SEEDS
) -> List[Dict[Tuple[str, str], float]]:
    """The exact backgrounds _attach_flip_stability_bands samples."""
    master_seed, _ = compute_effective_seed(request)
    child_seeds = [
        int(hashlib.sha256(f"{master_seed}:flip_stability:{i}".encode()).hexdigest()[:8], 16)
        for i in range(n_seeds)
    ]
    return [
        DualUncertaintySampler(request.graph.edges, SeededRNG(s)).sample_edge_configuration()
        for s in child_seeds
    ]


def wire_response(request_dict: dict, flag_on: bool) -> Any:
    import os

    key = "ISL_FLIP_STABILITY_BANDS"
    if flag_on:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        return RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**request_dict))
    finally:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def compare_base_path(name: str, request_dict: dict) -> dict:
    """SHIPPED edge_e_values (flag off) vs analytic closed form."""
    response = wire_response(request_dict, flag_on=False)
    shipped = response.edge_e_values
    assert shipped, f"{name}: no edge_e_values on the wire (budget exceeded?)"

    request = prepared_request(request_dict)
    evaluator = SCMEvaluatorV2(request.graph, epsilon_rng=None)
    analytic = analytic_edge_e_values(request, evaluator)

    assert len(analytic) == len(shipped), f"{name}: entry count mismatch"
    n_flip = n_noflip = n_agree = 0
    max_delta = 0.0
    disagreements: List[str] = []
    for an, ref in zip(analytic, shipped):
        assert an["edge_id"] == ref["edge_id"], f"{name}: entry order mismatch"
        ref_inf = ref["e_value"] == float("inf")
        an_inf = an["e_value"] == float("inf")
        if ref_inf:
            n_noflip += 1
        else:
            n_flip += 1
        if ref_inf != an_inf or an["flip_direction"] != ref["flip_direction"]:
            disagreements.append(
                f"{ref['edge_id']}: verdict/direction mismatch "
                f"(mc inf={ref_inf} dir={ref['flip_direction']} | "
                f"an inf={an_inf} dir={an['flip_direction']})"
            )
            continue
        # No-flip entries carry flip_mean == current_mean on both sides, so
        # the same delta check covers them (delta 0 expected).
        delta = abs(an["flip_mean"] - ref["flip_mean"])
        max_delta = max(max_delta, delta)
        if delta <= FLIP_MEAN_TOL:
            n_agree += 1
        else:
            disagreements.append(f"{ref['edge_id']}: |delta|={delta:.2e}")
    return {
        "edges": len(shipped),
        "flip": n_flip,
        "noflip": n_noflip,
        "agree": n_agree,
        "max_delta": max_delta,
        "disagreements": disagreements,
    }


def compare_bands(name: str, request_dict: dict) -> dict:
    """Wire stability.seed_flip_means (#71, flag on) vs analytic per background."""
    response = wire_response(request_dict, flag_on=True)
    shipped = response.edge_e_values
    assert shipped and all(
        "stability" in e for e in shipped
    ), f"{name}: bands missing on the wire (budget exceeded?)"

    request = prepared_request(request_dict)
    evaluator = SCMEvaluatorV2(request.graph, epsilon_rng=None)
    backgrounds = band_backgrounds(request)
    edges_by_key = {(e.from_, e.to): e for e in request.graph.edges}

    cells = agree = none_cells = 0
    max_delta = 0.0
    disagreements: List[str] = []
    for entry in shipped:
        edge = edges_by_key[(entry["from_id"], entry["to_id"])]
        for i, wire_value in enumerate(entry["stability"]["seed_flip_means"]):
            cells += 1
            an_value: Optional[float] = analytic_flip_mean_under_background(
                request, evaluator, edge, backgrounds[i]
            )
            if wire_value is None:
                none_cells += 1
                if an_value is None:
                    agree += 1
                else:
                    disagreements.append(f"{entry['edge_id']} seed[{i}]: wire None, analytic flip")
            elif an_value is None:
                disagreements.append(f"{entry['edge_id']} seed[{i}]: wire flip, analytic None")
            else:
                delta = abs(an_value - wire_value)
                max_delta = max(max_delta, delta)
                if delta <= FLIP_MEAN_TOL:
                    agree += 1
                else:
                    disagreements.append(f"{entry['edge_id']} seed[{i}]: |delta|={delta:.2e}")
    return {
        "cells": cells,
        "none_cells": none_cells,
        "agree": agree,
        "max_delta": max_delta,
        "disagreements": disagreements,
    }


def compare_speed(name: str, request_dict: dict) -> dict:
    """Median wall ms + exact evaluate() counts, flip computation only."""
    request = prepared_request(request_dict)
    analyzer = RobustnessAnalyzerV2()

    def timed(fn) -> Tuple[float, int]:
        evaluator = CountingEvaluator(request.graph, epsilon_rng=None)
        fn(evaluator)  # warm-up (uncounted timing, counted calls reset below)
        evaluator.calls = 0
        times = []
        for _ in range(TIMING_REPETITIONS):
            t0 = time.perf_counter()
            fn(evaluator)
            times.append((time.perf_counter() - t0) * 1000)
        return statistics.median(times), evaluator.calls // TIMING_REPETITIONS

    mc_ms, mc_calls = timed(lambda ev: analyzer._compute_edge_e_values(request, ev))
    an_ms, an_calls = timed(lambda ev: analytic_edge_e_values(request, ev))

    backgrounds = band_backgrounds(request)

    def band_sweep_mc(ev: SCMEvaluatorV2) -> None:
        for edge in request.graph.edges:
            for background in backgrounds:
                analyzer._flip_mean_under_background(request, ev, edge, background)

    def band_sweep_an(ev: SCMEvaluatorV2) -> None:
        for edge in request.graph.edges:
            for background in backgrounds:
                analytic_flip_mean_under_background(request, ev, edge, background)

    band_mc_ms, band_mc_calls = timed(band_sweep_mc)
    band_an_ms, band_an_calls = timed(band_sweep_an)

    return {
        "mc_ms": mc_ms,
        "an_ms": an_ms,
        "mc_calls": mc_calls,
        "an_calls": an_calls,
        "band_mc_ms": band_mc_ms,
        "band_an_ms": band_an_ms,
        "band_mc_calls": band_mc_calls,
        "band_an_calls": band_an_calls,
    }


def main() -> None:
    fixtures = list(variant_requests()) + [synthetic_request()]

    print("== 1. BASE PATH: shipped edge_e_values (flag off) vs analytic ==")
    print(
        f"{'fixture':24} {'edges':>5} {'flip':>4} {'noflip':>6} {'agree':>5} "
        f"{'max |delta flip_mean|':>21}"
    )
    for name, request_dict in fixtures:
        r = compare_base_path(name, request_dict)
        print(
            f"{name:24} {r['edges']:>5} {r['flip']:>4} {r['noflip']:>6} "
            f"{r['agree']:>2}/{r['edges']:<2} {r['max_delta']:>21.2e}"
        )
        for d in r["disagreements"]:
            print(f"    DISAGREE {d}")

    print()
    print("== 2. STABILITY BANDS (#71): wire seed_flip_means vs analytic ==")
    print(f"{'fixture':24} {'cells':>5} {'none':>5} {'agree':>9} " f"{'max |delta flip_mean|':>21}")
    for name, request_dict in fixtures:
        r = compare_bands(name, request_dict)
        print(
            f"{name:24} {r['cells']:>5} {r['none_cells']:>5} "
            f"{r['agree']:>4}/{r['cells']:<4} {r['max_delta']:>21.2e}"
        )
        for d in r["disagreements"]:
            print(f"    DISAGREE {d}")

    print()
    print(f"== 3. SPEED: flip computation only (median of {TIMING_REPETITIONS}, 1 warm-up) ==")
    print(
        f"{'fixture':24} {'mc ms':>7} {'an ms':>7} {'speedup':>7} "
        f"{'mc evals':>8} {'an evals':>8} | {'band mc':>8} {'band an':>8} {'speedup':>7} "
        f"{'band mc evals':>13} {'band an evals':>13}"
    )
    for name, request_dict in fixtures:
        r = compare_speed(name, request_dict)
        print(
            f"{name:24} {r['mc_ms']:>7.2f} {r['an_ms']:>7.2f} "
            f"{r['mc_ms'] / r['an_ms']:>6.1f}x {r['mc_calls']:>8} {r['an_calls']:>8} | "
            f"{r['band_mc_ms']:>8.2f} {r['band_an_ms']:>8.2f} "
            f"{r['band_mc_ms'] / r['band_an_ms']:>6.1f}x "
            f"{r['band_mc_calls']:>13} {r['band_an_calls']:>13}"
        )


if __name__ == "__main__":
    main()
