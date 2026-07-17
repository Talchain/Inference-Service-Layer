"""
Runtime-budget measurement for seed-sweep flip-threshold stability bands
(Track S Phase 1 — DEFAULT-ON since 2026-07-17, env gating removed).

Measures full analyze() wall time with the band sweep disabled (the
production default before 2026-07-17, simulated by patching
``_attach_flip_stability_bands`` to a no-op) vs the default path (bands on,
N = FLIP_STABILITY_N_SEEDS, currently 10) on:
- the three pinned sample_variants graphs (3 nodes / 3 edges, n_samples 200)
  — the same fixtures the science-validation report used, and
- a larger deterministic synthetic graph (12 nodes / 17 edges, n_samples 500)
  — closer to a production-sized request.

Run:
    poetry run python benchmarks/flip_stability_budget.py

Prints a per-graph table: median ms sweep-off (patched), median ms default,
delta and ratio. Numbers are recorded in
docs/lanes/2026-07-11-flip-stability-bands.md. The comparison is
analyze()-to-analyze(): it captures exactly what the default-on sweep adds,
including the base MC that dominates real requests. The sweep itself is
additionally capped by RobustnessAnalyzerV2.FLIP_STABILITY_BUDGET_MS
(currently 30000 ms, all-or-nothing).
"""

import json
import statistics
import time
from pathlib import Path

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS_PATH = REPO_ROOT / "tests" / "benchmarks" / "sample_variants.json"

REPETITIONS = 21  # median of 21 per cell; 1 warm-up run before timing.
# Note: the three sample_variants fixtures complete in single-digit ms, where
# process noise is the same order as the thing measured — the synthetic
# production-sized graph is the honest budget signal.


def variant_requests():
    variants = json.loads(VARIANTS_PATH.read_text())
    for idx, graph in enumerate(variants["graphs"]):
        yield f"sample_variants[{idx}]", {
            "request_id": f"flip-budget-variant-{idx}",
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

    # factors -> mids (fan-out, mixed signs/eps)
    edge("f0", "m0", 0.5, 0.12, 0.95)
    edge("f0", "m1", -0.3, 0.1)
    edge("f1", "m1", 0.4, 0.15, 0.9)
    edge("f1", "m2", 0.35, 0.08)
    edge("f2", "m2", -0.45, 0.12, 0.85)
    edge("f2", "m3", 0.25, 0.1)
    edge("f3", "m3", 0.6, 0.14, 0.95)
    edge("f3", "m0", -0.2, 0.09)
    # mids -> aggregates
    edge("m0", "a0", 0.55, 0.1)
    edge("m1", "a0", 0.45, 0.12, 0.9)
    edge("m2", "a1", 0.5, 0.11)
    edge("m3", "a1", -0.35, 0.13, 0.9)
    # aggregates -> risk/goal
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
        "request_id": "flip-budget-synthetic",
        "graph": {"nodes": nodes, "edges": edges},
        "options": options,
        "goal_node_id": "goal",
        "seed": 42,
        "n_samples": 500,
        "include_e_values": True,
    }


def median_ms(request_dict, sweep_on: bool) -> tuple[float, int]:
    """Median analyze() wall ms over REPETITIONS (plus 1 untimed warm-up).

    Returns (median_ms, n_entries_with_bands) so the sweep-on cell also
    verifies bands were actually computed (budget not silently exceeded).
    The sweep-off arm patches the sweep to a no-op — the pre-2026-07-17
    default path — since there is no runtime switch any more.
    """
    original_attach = RobustnessAnalyzerV2._attach_flip_stability_bands
    if not sweep_on:
        RobustnessAnalyzerV2._attach_flip_stability_bands = (  # type: ignore[method-assign]
            lambda self, request, evaluator, edge_e_values, master_seed: None
        )
    try:
        request = RobustnessRequestV2(**request_dict)
        RobustnessAnalyzerV2().analyze(request)  # warm-up
        times = []
        banded = 0
        for _ in range(REPETITIONS):
            t0 = time.perf_counter()
            response = RobustnessAnalyzerV2().analyze(request)
            times.append((time.perf_counter() - t0) * 1000)
            banded = sum(1 for e in (response.edge_e_values or []) if "stability" in e)
        return statistics.median(times), banded
    finally:
        RobustnessAnalyzerV2._attach_flip_stability_bands = (  # type: ignore[method-assign]
            original_attach
        )


def main() -> None:
    print(f"repetitions per cell: {REPETITIONS} (median reported; 1 warm-up)")
    header = f"{'graph':24} {'off ms':>9} {'on ms':>9} {'delta':>8} {'ratio':>6}  bands"
    print(header)
    print("-" * len(header))
    for name, request_dict in list(variant_requests()) + [synthetic_request()]:
        off_ms, _ = median_ms(request_dict, sweep_on=False)
        on_ms, banded = median_ms(request_dict, sweep_on=True)
        print(
            f"{name:24} {off_ms:9.1f} {on_ms:9.1f} {on_ms - off_ms:8.1f} "
            f"{on_ms / off_ms:6.2f}  {banded}"
        )


if __name__ == "__main__":
    main()
