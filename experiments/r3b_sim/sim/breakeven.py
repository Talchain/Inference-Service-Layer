"""Mode X break-even for the price -> churn response on A-180910Z (MVP-B illustration).

The £59 decision turns on how much churn rises per +£10, which no graph quantifies. Instead
of assuming a value, find the response at which each verdict flips:

- ``goal``: £59 + feature still reaches the £100k MRR goal by month 12 (first passage, the frozen
  ``attain_by_H`` semantics);
- ``beats_keep_current``: £59 + feature still ends month 12 above keep-current;
- ``churn_limit``: £59 + feature still keeps monthly churn within the declared limit.

Tier X (``prototype_assumption``), point draws, competitive response 0. Excluded from claims.
Writes ``breakeven.json`` (deterministic). Usage, from ``experiments/r3b_sim``::

    poetry run python -m sim.breakeven
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .corpus import ROOT, load_graph
from .metrics import r6
from .run import model_evals, simulate
from .spec import load_valid_spec
from .static import point_draws

GRAPH = "pj-20260927T180910Z-A"
MODELS = ("X_net_reading", "X_gross_reading")
OPTION = "59_with_feature_release"
SWEPT = "X_price_churn_direct"
FIXED = {"X_price_churn_via_sensitivity": 0.0, "X_feature_competitive_mrr": 0.0}
CHURN = "monthly_churn"
LO, HI, STEPS = 0.0, 10.0, 60


def bisect(margin: Callable[[float], float], lo: float = LO, hi: float = HI) -> float:
    """Largest response in [lo, hi] with margin > 0, for a margin that decreases in it."""
    if margin(lo) <= 0 or margin(hi) > 0:
        raise ValueError("margin does not change sign on the bracket")
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        if margin(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def margins(model_id: str) -> dict[str, Callable[[float], float]]:
    graph = load_graph(GRAPH)
    spec = load_valid_spec(graph)
    model = next(m for m in spec["dynamic_models"] if m["id"] == model_id)
    if spec["goal"]["temporal_semantics"] != "attain_by_H":
        raise ValueError("break-even assumes the frozen attain_by_H goal semantics")
    goal = float(graph.resolve(spec["goal"]["threshold_ptr"]))
    limit = next(c for c in graph.constraints if c["node_id"] == CHURN)
    if limit["operator"] != "<=":
        raise ValueError("churn limit is not an upper bound")
    draws = point_draws(graph)
    base = graph.baseline_option

    def scenario(d: float) -> dict[str, float]:
        return {SWEPT: d, **FIXED}

    def mrr_path(d: float, opt: str) -> list[float]:
        traj = simulate(graph, spec, model, scenario(d), draws)[opt]
        if traj.y is None:
            raise ValueError(f"{opt} not computable under {model_id}")
        return [float(v) for v in traj.y[0]]

    def churn(d: float) -> float:
        nv = model_evals(graph, spec, model, scenario(d), draws)[OPTION].nodes[CHURN]
        if nv.value is None:
            raise ValueError("churn not computable")
        return float(nv.value[0])

    return {
        "goal": lambda d: max(mrr_path(d, OPTION)) - goal,
        "beats_keep_current": lambda d: mrr_path(d, OPTION)[-1] - mrr_path(d, base)[-1],
        "churn_limit": lambda d: float(limit["value"]) - churn(d),
    }


def compute() -> dict[str, Any]:
    graph = load_graph(GRAPH)
    spec = load_valid_spec(graph)
    xe = next(x for x in spec["x_effects"] if x["id"] == SWEPT)
    out: dict[str, Any] = {
        "schema": "r3b-breakeven-v1",
        "graph": GRAPH,
        "option": OPTION,
        "baseline": graph.baseline_option,
        "tier": "X",
        "label": "prototype_assumption",
        "excluded_from_claims": True,
        "swept": {
            "id": SWEPT,
            "unit": f"{xe['amount_unit']} per +£{xe['per_source_change']:g} price",
            "bracket": [LO, HI],
            "mapping_sweep": xe["sweep"],
        },
        "fixed": FIXED,
        "draws": "point",
        "goal": {
            "node": spec["goal"]["node"],
            "threshold": float(graph.resolve(spec["goal"]["threshold_ptr"])),
            "semantics": spec["goal"]["temporal_semantics"],
            "horizon_months": spec["horizon"]["months"],
        },
        "reading": "verdict holds while the price -> churn response is below the threshold",
        "models": {},
    }
    for model_id in MODELS:
        m = margins(model_id)
        out["models"][model_id] = {
            "thresholds": {k: r6(bisect(f)) for k, f in m.items()},
            "margins_at_zero_response": {k: r6(f(0.0)) for k, f in m.items()},
        }
    return out


def main() -> None:
    path = ROOT / "breakeven.json"
    path.write_text(json.dumps(compute(), indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
