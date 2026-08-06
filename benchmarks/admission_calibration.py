"""Admission cost-model calibration harness (Codex F8).

Measures wall-clock for a grid of representative /analyze/v2 requests, relates it
to the structural ``compute_weighted_cost`` cost units, and derives an indicative
admission ceiling toward ``TARGET_WALL_MS`` (half of OVERALL_REQUEST_BUDGET_MS).

WHAT THIS PRODUCES
  * ``k_ms_per_unit`` — least-squares wall-clock per cost unit (through origin).
  * suggested ceiling = floor(TARGET_WALL_MS / k_ms_per_unit).
  * the Codex anchor check (~4.13x runtime 1->10 options at fixed graph/samples).
  * an admit/reject table for realistic graphs at the shipped provisional ceiling.

⚠ The CEILING is PROVISIONAL. This harness on LOCAL hardware only gives an
  indicative fit; the numeric ceiling must be RE-CALIBRATED on the Render
  isl-staging instance before it is finalized (Paul sign-off owed).

USAGE
  poetry run python benchmarks/admission_calibration.py            # full grid
  poetry run python benchmarks/admission_calibration.py --quick    # reduced grid
  poetry run python benchmarks/admission_calibration.py --runs 5   # median of N
  poetry run python benchmarks/admission_calibration.py --out PATH  # write markdown
Run with ISL_AUTH_DISABLED=true (consistent with the repo's other scripts).
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    COMPLEXITY_FORMULA_VERSION,
    DEFAULT_MAX_COST_UNITS,
    TARGET_WALL_MS,
    compute_weighted_cost,
    get_max_cost_units,
)


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------
def _layered_graph(n_nodes: int, n_edges: int, evpi_factors: int = 0) -> dict:
    """A forward DAG with n_nodes nodes and exactly n_edges edges.

    A chain n0->n1->...->n_{last} is laid down FIRST so the goal (last node) is
    always reachable (requires n_edges >= n_nodes-1); remaining budget is filled
    with forward skip edges n_i->n_j (i<j). Edge COUNT (hence cost + loop work)
    is identical to any other fill, so the calibration is unaffected.
    """
    nodes = [{"id": f"n{i}", "kind": "factor", "label": f"N{i}"} for i in range(n_nodes)]
    nodes[-1]["kind"] = "outcome"
    for k in range(min(evpi_factors, n_nodes - 1)):
        nodes[k]["observed_state"] = {"value": 50.0, "std": 5.0}

    def _edge(i: int, j: int) -> dict:
        return {
            "from": f"n{i}",
            "to": f"n{j}",
            "exists_probability": 0.9,
            "strength": {"mean": 0.3, "std": 0.1},
        }

    edges: List[dict] = []
    seen: set = set()
    # 1) reachability chain
    for i in range(n_nodes - 1):
        if len(edges) >= n_edges:
            break
        edges.append(_edge(i, i + 1))
        seen.add((i, i + 1))
    # 2) fill remaining budget with forward skip edges
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


def _root_factor_graph(n_drivers: int, n_edges: int) -> dict:
    """A fan-in decision graph: ``n_drivers`` PARENTLESS factors feeding one outcome.

    ⚠ WHY THE FLIPS CELLS NEED THEIR OWN TOPOLOGY (ROADMAP 2.745).

    ``_layered_graph`` lays a n0->n1->...->n_last chain FIRST, so every node except
    n0 has a parent: it has exactly ONE root. The factor-flip phase screens only
    ROOT factors, so on a layered graph it finds 0 eligible factors and returns
    immediately. Adding ``include_factor_flips`` to a layered cell would therefore
    time a phase that never runs, and the k_ms_per_unit fit would absorb a term
    charging ~25k units for ~3 evaluations — a WORSE calibration than not measuring
    it at all. The flag alone is not the fix; the topology is half of it.

    Every driver carries an ``observed_state.value`` (the other half of the phase's
    eligibility test), so eligible == every driver. Extra edge budget is spent on
    duplicate driver->goal edges, never driver->driver, because a driver that gains
    a parent stops being a root and silently shrinks the phase.
    """
    goal_id = f"n{n_drivers}"
    nodes = [
        {
            "id": f"n{i}",
            "kind": "factor",
            "label": f"N{i}",
            "observed_state": {"value": 0.5, "std": 0.1},
        }
        for i in range(n_drivers)
    ]
    nodes.append({"id": goal_id, "kind": "outcome", "label": "GOAL"})
    edges: List[dict] = [
        {
            "from": f"n{i}",
            "to": goal_id,
            "exists_probability": 0.9,
            "strength": {"mean": 0.2 + 0.05 * (i % 7), "std": 0.1},
        }
        for i in range(n_drivers)
    ]
    while len(edges) < n_edges:
        i = len(edges) % n_drivers
        edges.append(
            {
                "from": f"n{i}",
                "to": goal_id,
                "exists_probability": 0.85,
                "strength": {"mean": 0.15, "std": 0.05},
            }
        )
    return {"nodes": nodes, "edges": edges}


def build_request(
    n_nodes: int,
    n_edges: int,
    n_samples: int,
    n_options: int,
    *,
    evpi_factors: int = 0,
    include_e_values: bool = False,
    include_path: bool = False,
    sensitivity: bool = True,
    include_factor_flips: bool = False,
    root_factors: int = 0,
) -> RobustnessRequestV2:
    # ROADMAP 2.745: flips cells opt into the fan-in topology. `root_factors == 0`
    # leaves every pre-existing cell on the layered graph BYTE-IDENTICALLY, so the
    # historical fit is unchanged and the new cells are additive.
    if root_factors:
        graph = _root_factor_graph(root_factors, n_edges)
    else:
        graph = _layered_graph(n_nodes, n_edges, evpi_factors)
    options = [
        {"id": f"o{k}", "label": f"O{k}", "interventions": {"n0": 0.1 * (k + 1)}}
        for k in range(n_options)
    ]
    analysis_types = ["comparison", "robustness"] + (["sensitivity"] if sensitivity else [])
    body: dict = {
        "graph": graph,
        "options": options,
        # Derived from the graph, not recomputed from n_nodes: identical to the
        # previous `f"n{n_nodes - 1}"` for the layered builder (whose last node IS
        # the outcome), and correct for the fan-in builder too.
        "goal_node_id": graph["nodes"][-1]["id"],
        "n_samples": n_samples,
        "seed": 7,
        "analysis_types": analysis_types,
        "include_e_values": include_e_values,
        "include_path_decomposition": include_path,
    }
    if include_factor_flips:
        body["include_factor_flips"] = True
    if evpi_factors:
        body["include_voi"] = True
        body["parameter_uncertainties"] = [
            {"node_id": f"n{k}", "distribution": "normal", "std": 5.0} for k in range(evpi_factors)
        ]
    return RobustnessRequestV2(**body)


@dataclass
class Cell:
    label: str
    n_nodes: int
    n_edges: int
    n_samples: int
    n_options: int
    evpi_factors: int = 0
    include_e_values: bool = False
    include_path: bool = False
    include_factor_flips: bool = False
    root_factors: int = 0
    cost_units: int = 0
    dominant: str = ""
    t_ms: float = 0.0
    terms: Dict[str, int] = field(default_factory=dict)


def default_grid(quick: bool) -> List[Cell]:
    """Representative cells spanning the realistic cost range."""
    cells: List[Cell] = []

    def add(label, N, E, S, O, evpi=0, ev=False, path=False, flips=False, roots=0):
        cells.append(Cell(label, N, E, S, O, evpi, ev, path, flips, roots))

    # base MC scaling (samples x options)
    add("pilot", 5, 8, 1000, 2)
    add("small-4opt", 8, 20, 2000, 4)
    add("mid", 12, 100, 5000, 3)
    add("mid-10opt", 12, 100, 5000, 10)
    add("dense-mid", 40, 120, 5000, 4)
    add("dense-mid-1opt-deep", 40, 120, 10000, 1)
    add("dense-mid-4opt-deep", 40, 120, 10000, 4)
    # option sweep at fixed graph/samples (Codex 4.13x anchor: 1 -> 10 options)
    add("anchor-1opt", 12, 30, 5000, 1)
    add("anchor-2opt", 12, 30, 5000, 2)
    add("anchor-4opt", 12, 30, 5000, 4)
    add("anchor-10opt", 12, 30, 5000, 10)
    # EVPI phase
    add("evpi-3f", 12, 30, 5000, 3, evpi=3)
    add("evpi-5f-dense", 40, 120, 5000, 4, evpi=5)
    # optional phases
    add("evalues", 12, 40, 3000, 3, ev=True)
    add("path", 20, 60, 3000, 3, path=True)
    # ROADMAP 2.745 — factor flips. PLoT sends `include_factor_flips: true`
    # UNCONDITIONALLY (translator-v3.ts:748), so this term is priced on EVERY live
    # request, and W_FACTOR_FLIP_COEF = 1 gives it ZERO pricing headroom. Until
    # these cells existed, the k_ms_per_unit fit behind the admission ceiling had
    # never once seen it run. `roots=` selects the fan-in topology — without it the
    # phase finds no eligible factor and the cell would time nothing (see
    # _root_factor_graph).
    add("flips-mid", 12, 12, 2000, 3, flips=True, roots=11)
    add("flips-10opt", 12, 12, 2000, 10, flips=True, roots=11)
    add("flips-dense", 40, 120, 5000, 4, flips=True, roots=39)
    if not quick:
        add("upper-poc", 12, 100, 5000, 3)
        add("dense-mid-10opt", 40, 120, 10000, 10)
        add("schema-max-1opt", 50, 200, 10000, 1)
        add("evpi-8f", 20, 60, 5000, 4, evpi=8)
        add("evalues+path", 20, 60, 5000, 3, ev=True, path=True)
        add("full-load", 30, 90, 5000, 4, evpi=4, ev=True, path=True)
        # PLoT's live posture: e-values + VOI + flips together on one request
        # (captured egress carries include_e_values, include_voi,
        # include_factor_flips on every base call).
        add("flips-full-load", 40, 120, 5000, 4, evpi=4, ev=True, flips=True, roots=39)
    return cells


# ---------------------------------------------------------------------------
# Measurement + fit
# ---------------------------------------------------------------------------
def measure(cell: Cell, runs: int) -> Cell:
    # import here so the analyzer instance is fresh; it is stateless anyway
    from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

    req = build_request(
        cell.n_nodes,
        cell.n_edges,
        cell.n_samples,
        cell.n_options,
        evpi_factors=cell.evpi_factors,
        include_e_values=cell.include_e_values,
        include_path=cell.include_path,
        include_factor_flips=cell.include_factor_flips,
        root_factors=cell.root_factors,
    )
    wc = compute_weighted_cost(req)
    cell.cost_units = wc.total
    cell.dominant = wc.dominant_term
    cell.terms = dict(wc.terms)
    analyzer = RobustnessAnalyzerV2()
    # one warm-up run (JIT-free Python, but primes caches / imports)
    analyzer.analyze(req)
    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        analyzer.analyze(req)
        times.append((time.perf_counter() - t0) * 1000.0)
    cell.t_ms = statistics.median(times)
    return cell


def fit_k(cells: List[Cell]) -> float:
    """Least-squares wall-clock per cost unit through the origin."""
    x = np.array([c.cost_units for c in cells], dtype=float)
    y = np.array([c.t_ms for c in cells], dtype=float)
    # k minimizing ||k*x - y||^2  ->  k = (x.y)/(x.x)
    return float((x @ y) / (x @ x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="reduced grid for local runs")
    ap.add_argument("--runs", type=int, default=3, help="median of N timed runs per cell")
    ap.add_argument("--out", type=str, default=None, help="write a markdown report to PATH")
    args = ap.parse_args()

    grid = default_grid(args.quick)
    print(f"Calibrating {len(grid)} cells (median of {args.runs} runs each)...")
    measured: List[Cell] = []
    for c in grid:
        measured.append(measure(c, args.runs))
        print(f"  {c.label:22s} cost={c.cost_units:>12,}  t={c.t_ms:8.1f}ms  dom={c.dominant}")

    k = fit_k(measured)
    suggested = int(TARGET_WALL_MS / k) if k > 0 else 0
    # per-ms-per-unit summary
    ratios = [(c.t_ms / c.cost_units) for c in measured if c.cost_units > 0]

    # Codex anchor check: runtime ratio 1opt -> 10opt at fixed graph/samples
    by_label = {c.label: c for c in measured}
    anchor = None
    if "anchor-1opt" in by_label and "anchor-10opt" in by_label:
        t1 = by_label["anchor-1opt"].t_ms
        t10 = by_label["anchor-10opt"].t_ms
        anchor = (t10 / t1) if t1 > 0 else None

    lines: List[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    out()
    out("=" * 72)
    out(f"FIT: k_ms_per_unit = {k:.6e} ms/unit  (through origin, {len(measured)} cells)")
    out(
        f"     per-cell t_ms/cost_unit range: "
        f"{min(ratios):.3e} .. {max(ratios):.3e} (median {statistics.median(ratios):.3e})"
    )
    out(f"TARGET_WALL_MS = {TARGET_WALL_MS}  ->  suggested ceiling = {suggested:,} cost units")
    out(f"SHIPPED provisional DEFAULT_MAX_COST_UNITS = {DEFAULT_MAX_COST_UNITS:,}")
    out(f"env-resolved ceiling (get_max_cost_units) = {get_max_cost_units():,}")
    if anchor is not None:
        out(f"Codex anchor: 1opt->10opt runtime ratio = {anchor:.2f}x (Codex measured ~4.13x)")
    out("=" * 72)
    out("⚠ LOCAL-HARDWARE indicative fit — staging recalibration owed before finalize.")
    out()

    # markdown table
    md: List[str] = []
    md.append("| cell | N | E | S | O | evpi | phases | cost_units | dominant | t_ms | ms/unit |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in measured:
        phases = (
            ",".join(p for p, on in (("ev", c.include_e_values), ("path", c.include_path)) if on)
            or "-"
        )
        md.append(
            f"| {c.label} | {c.n_nodes} | {c.n_edges} | {c.n_samples} | {c.n_options} | "
            f"{c.evpi_factors} | {phases} | {c.cost_units:,} | {c.dominant} | {c.t_ms:.1f} | "
            f"{(c.t_ms / c.cost_units):.3e} |"
        )
    out("Per-cell measurements:")
    for row in md:
        out(row)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote markdown report to {args.out}")


if __name__ == "__main__":
    main()
