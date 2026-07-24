"""OC-1 (D-23.17) EVPPI / EVPC governor recalibration measurement harness.

WHY (D-23.17): the F2 governor fix priced the full-population EVPPI phase as
``(deg+1)*U*(1+K)*O*S`` (coef 5, K=16). But ``_inner_expected_max`` (src/utils/
evppi.py) shares ONE multi-RHS ``np.linalg.lstsq`` SVD across all options, so the
phase wall-time is ~O-INDEPENDENT — the ``*O`` factor over-charges 10-100x, 422-ing
legal requests (OC-1: S=10000/U=20/O=2 charges 34M, runs in ~1s). This harness
MEASURES the real EVPPI + EVPC phase wall-time in isolation so the coefficient can
be re-derived to reflect actual work with a modest (1.5-3x) safety margin.

METHOD
  * Anchor: measure the base-MC phase (S*O*W evaluate()-equivalents) in isolation to
    fix the cost-unit <-> wall-time conversion the whole model + 24M ceiling live in.
    1 cost unit == 1 node-evaluation-equivalent, by the model's own definition.
  * EVPPI: instrument ``_compute_factor_evppi`` (monkeypatched timer) across a grid of
    (S, U, O) with the graph held fixed, so ONLY the EVPPI phase varies. Convert the
    measured phase-ms to "true cost units" via the anchor. Fit the functional form
    (does O matter? confirm the shared-SVD O-independence at the bytes).
  * EVPC: instrument ``_compute_factor_evpc`` across (S, W, grid_points). The phase is
    grid_points full evaluate()s over S samples, structurally identical to base-MC, so
    the adversarial found coef=1 EXACT. Verify by measurement.

Run: PYTHONPATH=<clone> ISL_AUTH_DISABLED=true python benchmarks/oc1_evppi_recal.py
     [--runs N] [--quick] [--out PATH]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("ISL_AUTH_DISABLED", "true")

from src.models.robustness_v2 import RobustnessRequestV2  # noqa: E402
import src.services.robustness_analyzer_v2 as analyzer_mod  # noqa: E402
from src.services.robustness_analyzer_v2 import (  # noqa: E402
    REGRESSION_EVPPI_NULL_PERMUTATIONS,
    RobustnessAnalyzerV2,
    compute_weighted_cost,
)

# --- provenance assert: the module MUST resolve inside this clone -----------
_CLONE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assert analyzer_mod.__file__.startswith(_CLONE + os.sep), (
    f"analyzer resolved OUTSIDE clone: {analyzer_mod.__file__}"
)

K = REGRESSION_EVPPI_NULL_PERMUTATIONS  # 16


# ---------------------------------------------------------------------------
# graph / request construction (mirrors the calibration harness builder)
# ---------------------------------------------------------------------------
def _graph(n_nodes: int, n_edges: int) -> dict:
    nodes = [{"id": f"n{i}", "kind": "factor", "label": f"N{i}"} for i in range(n_nodes)]
    nodes[-1]["kind"] = "outcome"

    def _edge(i: int, j: int) -> dict:
        return {"from": f"n{i}", "to": f"n{j}", "exists_probability": 0.9,
                "strength": {"mean": 0.3, "std": 0.1}}

    edges: List[dict] = []
    seen: set = set()
    for i in range(n_nodes - 1):
        if len(edges) >= n_edges:
            break
        edges.append(_edge(i, i + 1))
        seen.add((i, i + 1))
    for j in range(2, n_nodes):
        for i in range(j - 1):
            if len(edges) >= n_edges:
                break
            if (i, j) not in seen:
                edges.append(_edge(i, j))
                seen.add((i, j))
        if len(edges) >= n_edges:
            break
    return {"nodes": nodes, "edges": edges}


def build(n_nodes: int, n_edges: int, S: int, O: int, *, U: int = 0,
          control: Optional[list] = None) -> RobustnessRequestV2:
    g = _graph(n_nodes, n_edges)
    opts = [{"id": f"o{k}", "label": f"O{k}", "interventions": {"n0": 0.1 * (k + 1)}}
            for k in range(O)]
    body: dict = {"graph": g, "options": opts, "goal_node_id": f"n{n_nodes - 1}",
                  "n_samples": S, "seed": 7, "analysis_types": ["comparison", "robustness"]}
    if U:
        # uncertainties on n1..nU (NOT n0, which options intervene on) so every
        # factor is NON-LEVER -> all U get an EVPPI row (matches the cost formula's
        # unique-factor count u).
        body["include_voi"] = True
        body["parameter_uncertainties"] = [
            {"node_id": f"n{k}", "distribution": "normal", "std": 5.0}
            for k in range(1, U + 1)]
    if control:
        body["control_candidates"] = control
    return RobustnessRequestV2(**body)


# ---------------------------------------------------------------------------
# phase timers via monkeypatch — capture ONLY the target phase wall-time
# ---------------------------------------------------------------------------
_EVPPI_TIMES: List[float] = []
_EVPC_TIMES: List[float] = []

_orig_evppi = RobustnessAnalyzerV2._compute_factor_evppi
_orig_evpc = RobustnessAnalyzerV2._compute_factor_evpc


def _timed_evppi(self, *a, **k):
    t0 = time.perf_counter()
    r = _orig_evppi(self, *a, **k)
    _EVPPI_TIMES.append((time.perf_counter() - t0) * 1000.0)
    return r


def _timed_evpc(self, *a, **k):
    t0 = time.perf_counter()
    r = _orig_evpc(self, *a, **k)
    _EVPC_TIMES.append((time.perf_counter() - t0) * 1000.0)
    return r


RobustnessAnalyzerV2._compute_factor_evppi = _timed_evppi
RobustnessAnalyzerV2._compute_factor_evpc = _timed_evpc


def _median_phase(req: RobustnessRequestV2, bucket: List[float], runs: int) -> float:
    """Run analyze() `runs`+1 times (1 warm-up) and return the median phase-ms.

    gc.collect() before the timed loop + gc disabled during it removes GC-pause
    jitter from the per-phase samples (the phase itself allocates transient numpy
    arrays; a mid-phase collection otherwise inflates a random subset of runs).
    """
    an = RobustnessAnalyzerV2()
    bucket.clear()
    an.analyze(req)  # warm-up (primes numpy, imports, caches)
    bucket.clear()
    gc.collect()
    gc.disable()
    try:
        for _ in range(runs):
            an.analyze(req)
    finally:
        gc.enable()
    # one phase call per analyze() for these single-phase requests
    return statistics.median(bucket)


def _median_full(req: RobustnessRequestV2, runs: int) -> float:
    an = RobustnessAnalyzerV2()
    an.analyze(req)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        an.analyze(req)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def anchor_base_ms_per_unit(runs: int) -> Tuple[float, List[dict]]:
    """Base-MC-only cells (no VOI/EVPC/sensitivity): fit ms per base cost unit.

    Full analyze() wall on a base-only request is dominated by the S*O*W base MC
    loop, so full_ms / base_cost is the node-eval-equivalent conversion. Least
    squares through origin over several shapes.
    """
    cells = [
        (12, 30, 5000, 1), (12, 30, 5000, 4), (12, 30, 5000, 10),
        (22, 40, 10000, 2), (40, 120, 10000, 1), (40, 120, 5000, 4),
        (50, 200, 10000, 1),
    ]
    rows = []
    xs, ys = [], []
    for (N, E, S, O) in cells:
        req = build(N, E, S, O)
        cost = compute_weighted_cost(req)
        base = cost.terms["base_mc"]
        ms = _median_full(req, runs)
        rows.append({"N": N, "E": E, "S": S, "O": O, "W": N + E,
                     "base_units": base, "full_ms": round(ms, 2),
                     "ms_per_unit": ms / base})
        xs.append(base)
        ys.append(ms)
    x = np.array(xs, float)
    y = np.array(ys, float)
    k = float((x @ y) / (x @ x))  # ms per base cost unit, through origin
    return k, rows


def measure_evppi(runs: int, quick: bool) -> List[dict]:
    """EVPPI phase wall across (S, U, O). Graph fixed large enough to hold U+2 nodes."""
    S_grid = [1000, 5000, 10000]
    # U bounded by MAX_GRAPH_NODES(50): 1 lever (n0) + U factor nodes + 1 outcome
    # => max feasible U = 48. (MAX_PARAMETER_UNCERTAINTIES=50 is unreachable in a
    # single graph — a real worst-case finding, noted for the ceiling analysis.)
    U_grid = [2, 10, 20] if quick else [2, 10, 20, 48]
    O_grid = [2, 5, 10]
    rows = []
    for U in U_grid:
        n_nodes = min(50, U + 2)  # n0 (lever) .. nU (uncertain factors) + outcome
        n_edges = max(n_nodes - 1, min(200, n_nodes * 2))
        for S in S_grid:
            for O in O_grid:
                req = build(n_nodes, n_edges, S, O, U=U)
                cost = compute_weighted_cost(req)
                ms = _median_phase(req, _EVPPI_TIMES, runs)
                rows.append({"S": S, "U": U, "O": O, "N": n_nodes, "E": n_edges,
                             "evppi_ms": round(ms, 3),
                             "formula_units": cost.terms.get("evppi_full", 0)})
    return rows


def measure_evpc(runs: int, quick: bool) -> List[dict]:
    """EVPC phase wall across (S, W, grid_points)."""
    configs = [
        # (N, E, S, n_candidates, n_values)
        (22, 40, 5000, 5, 7), (22, 40, 10000, 5, 7),
        (12, 30, 10000, 2, 5), (12, 30, 5000, 3, 3),
        (50, 200, 5000, 5, 7),
    ]
    if not quick:
        configs += [(50, 200, 10000, 5, 7)]  # the F2 worst-case dense cell
    rows = []
    for (N, E, S, ncand, nval) in configs:
        control = [{"factor_id": f"n{1 + k}", "values": [0.1 * (v + 1) for v in range(nval)]}
                   for k in range(ncand)]
        req = build(N, E, S, 1, control=control)
        cost = compute_weighted_cost(req)
        grid = ncand * nval
        ms = _median_phase(req, _EVPC_TIMES, runs)
        rows.append({"N": N, "E": E, "S": S, "W": N + E, "ncand": ncand, "nval": nval,
                     "grid": grid, "evpc_ms": round(ms, 3),
                     "formula_units": cost.terms.get("evpc", 0)})
    return rows


def isolation_sweeps(runs: int) -> Dict[str, List[dict]]:
    """Hold two of (S,U,O) fixed and sweep the third finely, to read the scaling
    exponent of each axis directly (answers: how does EVPPI wall scale with S/U/O?)."""
    out: Dict[str, List[dict]] = {"O_sweep": [], "S_sweep": [], "U_sweep": []}
    # O sweep: S=8000, U=10 fixed; O in 1..10
    for O in [1, 2, 3, 4, 6, 8, 10]:
        req = build(12, 30, 8000, O, U=10)
        ms = _median_phase(req, _EVPPI_TIMES, runs)
        out["O_sweep"].append({"S": 8000, "U": 10, "O": O, "evppi_ms": round(ms, 3)})
    # S sweep: U=10, O=4 fixed; S in grid
    for S in [1000, 2000, 4000, 6000, 8000, 10000]:
        req = build(12, 30, S, 4, U=10)
        ms = _median_phase(req, _EVPPI_TIMES, runs)
        out["S_sweep"].append({"S": S, "U": 10, "O": 4, "evppi_ms": round(ms, 3)})
    # U sweep: S=8000, O=4 fixed; U in grid (n_nodes grows with U, capped at 50)
    for U in [1, 2, 5, 10, 20, 30, 48]:
        n_nodes = min(50, U + 2)
        req = build(n_nodes, max(n_nodes - 1, min(200, 2 * n_nodes)), 8000, 4, U=U)
        ms = _median_phase(req, _EVPPI_TIMES, runs)
        out["U_sweep"].append({"S": 8000, "U": U, "O": 4, "evppi_ms": round(ms, 3)})
    return out


def _loglog_slope(xs: List[float], ys: List[float]) -> float:
    lx = np.log(np.array(xs, float))
    ly = np.log(np.array(ys, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--evpc-runs", type=int, default=3,
                    help="EVPC cells are dense/expensive; fewer runs (coef unchanged).")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print(f"[oc1-recal] runs={args.runs} quick={args.quick}")
    print(f"[oc1-recal] analyzer = {analyzer_mod.__file__}")

    k_base, anchor_rows = anchor_base_ms_per_unit(args.runs)
    print(f"\n=== ANCHOR: base-MC ms per cost unit = {k_base:.6e} ms/unit "
          f"({1.0/k_base:,.0f} units/ms, {1000.0/k_base:,.0f} units/s) ===")
    for r in anchor_rows:
        print(f"  N={r['N']:>3} E={r['E']:>3} S={r['S']:>6} O={r['O']:>2} "
              f"base_units={r['base_units']:>12,} full_ms={r['full_ms']:>9.1f} "
              f"ms/unit={r['ms_per_unit']:.3e}")

    evppi_rows = measure_evppi(args.runs, args.quick)
    print("\n=== EVPPI phase (true units = evppi_ms / k_base) ===")
    print(f"{'S':>6} {'U':>3} {'O':>3} {'evppi_ms':>10} {'true_units':>12} "
          f"{'formula_units':>14} {'over_x':>8} {'ms/(U*17*S)':>14} {'ms/(U*17*O*S)':>16}")
    for r in evppi_rows:
        true_units = r["evppi_ms"] / k_base
        over = r["formula_units"] / true_units if true_units else float("inf")
        per_uks = r["evppi_ms"] / (r["U"] * (1 + K) * r["S"]) * 1e6  # ns per unit
        per_ukos = r["evppi_ms"] / (r["U"] * (1 + K) * r["O"] * r["S"]) * 1e6
        r["true_units"] = round(true_units)
        r["over_x"] = round(over, 2)
        r["ns_per_U17S"] = round(per_uks, 4)
        r["ns_per_U17OS"] = round(per_ukos, 4)
        print(f"{r['S']:>6} {r['U']:>3} {r['O']:>3} {r['evppi_ms']:>10.3f} "
              f"{true_units:>12,.0f} {r['formula_units']:>14,} {over:>7.1f}x "
              f"{per_uks:>13.4f} {per_ukos:>15.4f}")

    evpc_rows = measure_evpc(args.evpc_runs, args.quick)
    print("\n=== EVPC phase (true units = evpc_ms / k_base) ===")
    print(f"{'N':>3} {'E':>3} {'S':>6} {'grid':>5} {'evpc_ms':>10} {'true_units':>12} "
          f"{'formula_units':>14} {'ratio_formula/true':>18}")
    for r in evpc_rows:
        true_units = r["evpc_ms"] / k_base
        ratio = r["formula_units"] / true_units if true_units else float("inf")
        r["true_units"] = round(true_units)
        r["ratio"] = round(ratio, 3)
        print(f"{r['N']:>3} {r['E']:>3} {r['S']:>6} {r['grid']:>5} {r['evpc_ms']:>10.3f} "
              f"{true_units:>12,.0f} {r['formula_units']:>14,} {ratio:>17.3f}x")

    sweeps = isolation_sweeps(args.runs)
    print("\n=== ISOLATION SWEEPS (log-log slope = scaling exponent) ===")
    o = sweeps["O_sweep"]
    # slope over O>=1 (skip O=1 baseline for the ratio print)
    print("O sweep (S=8000,U=10):",
          " ".join(f"O{r['O']}={r['evppi_ms']:.1f}ms" for r in o))
    print(f"  -> O scaling exponent (log-log slope) = "
          f"{_loglog_slope([r['O'] for r in o], [r['evppi_ms'] for r in o]):.3f} "
          f"(1.0=linear, 0.0=flat)")
    print(f"  -> O=1 vs O=10 ratio = {o[-1]['evppi_ms']/o[0]['evppi_ms']:.2f}x "
          f"(formula's *O would predict 10x)")
    s = sweeps["S_sweep"]
    print("S sweep (U=10,O=4):", " ".join(f"S{r['S']}={r['evppi_ms']:.1f}ms" for r in s))
    print(f"  -> S scaling exponent = "
          f"{_loglog_slope([r['S'] for r in s], [r['evppi_ms'] for r in s]):.3f}")
    u = sweeps["U_sweep"]
    print("U sweep (S=8000,O=4):", " ".join(f"U{r['U']}={r['evppi_ms']:.1f}ms" for r in u))
    print(f"  -> U scaling exponent = "
          f"{_loglog_slope([r['U'] for r in u], [r['evppi_ms'] for r in u]):.3f}")

    payload = {
        "k_base_ms_per_unit": k_base,
        "base_units_per_second": 1000.0 / k_base,
        "ceiling_anchor_units_per_ms": 24_000_000 / 25_000,
        "K": K,
        "anchor_rows": anchor_rows,
        "evppi_rows": evppi_rows,
        "evpc_rows": evpc_rows,
        "sweeps": sweeps,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
