"""Monthly stock-and-flow integration (T2 and Mode X only).

Semantics (ANALYSIS-PLAN.md): S_t is the stock at the START of month t; S_0 is the held level.
Levels are the option's static values, constant from month 0 (onset_month_0, no_lag), and are
drawn once per trajectory. One step: flows F_t = f(S_t, L); S_{t+1} = S_t + in_t - out_t;
outputs Y_{t+1} = g(S_{t+1}, L). Y_0 = g(S_0, L) is the month-0 identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .corpus import Graph
from .frames import held_level
from .static import NodeVal


@dataclass
class Trajectory:
    option: str
    y: np.ndarray | None  # shape (n, H+1); for static_at_H, shape (n, 1)
    gaps: list[str] = field(default_factory=list)
    tier: int = 0
    defaults: set[str] = field(default_factory=set)
    assumptions: set[str] = field(default_factory=set)
    used_edges: set[str] = field(default_factory=set)
    negative_stock: bool = False

    @property
    def ok(self) -> bool:
        return self.y is not None


def _need(nodes: dict[str, NodeVal], node: str, gaps: list[str]) -> np.ndarray | None:
    nv = nodes[node]
    if nv.value is None:
        gaps.extend(nv.gaps or [f"NO_LEVEL:{node}"])
        return None
    return nv.value


def run_model(
    graph: Graph,
    model: dict[str, Any],
    option: str,
    nodes: dict[str, NodeVal] | None,
    sq_nodes: dict[str, NodeVal],
    horizon: int,
    tier: int,
    n: int,
) -> Trajectory:
    """Integrate one option through one declared model. ``nodes`` is the option's static
    evaluation at the model tier with the model's replaced edges (None if the option has no
    levels at all)."""
    traj = Trajectory(option=option, y=None, tier=tier)
    traj.defaults |= set(model["defaults"])
    traj.assumptions |= set(model["assumptions"])
    if nodes is None:
        traj.gaps.append("OPTION_LEVELS_MISSING")
        return traj

    referenced: set[str] = {model["output"]}
    if model["type"] == "static_at_H":
        out = _need(nodes, model["output"], traj.gaps)
        if out is None:
            return traj
        nv = nodes[model["output"]]
        traj.tier = max(traj.tier, nv.tier)
        traj.defaults |= nv.defaults
        traj.assumptions |= nv.assumptions
        traj.used_edges |= nv.used_edges
        traj.y = out.reshape(n, 1) if out.size == n else np.full((n, 1), float(out[0]))
        return traj

    stocks: dict[str, np.ndarray] = {}
    s_held0: dict[str, np.ndarray] = {}
    overlay: dict[str, np.ndarray] = {}
    ones = np.ones(n)
    for sid, s in model["stocks"].items():
        referenced.add(sid)
        nv = nodes[sid]
        if nv.delta is None and nv.gaps:
            traj.gaps.extend(nv.gaps)
            return traj
        if s["init"] == "held":
            lvl = held_level(graph, sid)
            if lvl is None or lvl.value is None:
                traj.gaps.append(f"STOCK_LEVEL_MISSING:{sid}")
                return traj
            if lvl.tier > tier:
                traj.gaps.append(f"REQUIRES_T{lvl.tier}:held level {sid}")
                return traj
            s0 = lvl.value * ones
        else:
            inv = s["inversion"]
            tgt = held_level(graph, inv["identity_target"])
            known = held_level(graph, inv["known_operand"])
            if tgt is None or known is None or tgt.value is None or not known.value:
                traj.gaps.append(f"STOCK_LEVEL_MISSING:{sid}")
                return traj
            s0 = (tgt.value / known.value) * ones
        s_held0[sid] = s0
        delta = nv.delta if nv.delta is not None else np.zeros(n)
        if s["option_delta_mode"] == "jump":
            stocks[sid] = s0 + delta
        else:
            stocks[sid] = s0
            overlay[sid] = delta
        traj.defaults |= nv.defaults
        traj.assumptions |= nv.assumptions
        traj.used_edges |= nv.used_edges
        traj.tier = max(traj.tier, nv.tier)

    flow_inputs: dict[str, np.ndarray] = {}
    for f in model["flows"]:
        for key in ("level", "rate", "net"):
            node = f.get(key)
            if node is None:
                continue
            referenced.add(node)
            val = _need(nodes, node, traj.gaps)
            if val is None:
                return traj
            flow_inputs[node] = val * ones
            nv = nodes[node]
            traj.tier = max(traj.tier, nv.tier)
            traj.defaults |= nv.defaults
            traj.assumptions |= nv.assumptions
            traj.used_edges |= nv.used_edges
            if f["form"] in ("rate_delta_x_stock", "derived_gross") and key == "rate":
                sq = sq_nodes[node].value
                if sq is None:
                    traj.gaps.append(f"STATUS_QUO_WITHHELD:{node}")
                    return traj
                flow_inputs[f"sq:{node}"] = sq * ones

    alg_consts: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}
    for a in model["algebraic"]:
        tv = nodes[a["target"]]
        if tv.value is None:
            traj.gaps.extend(tv.gaps or [f"NO_LEVEL:{a['target']}"])
            return traj
        extras[a["target"]] = tv.extra * ones if tv.extra is not None else np.zeros(n)
        traj.defaults |= tv.defaults
        traj.assumptions |= tv.assumptions
        traj.used_edges |= tv.used_edges
        traj.tier = max(traj.tier, tv.tier)
        for op in a["operands"]:
            if (
                op in stocks
                or op in alg_consts
                or any(b["target"] == op for b in model["algebraic"])
            ):
                continue
            val = _need(nodes, op, traj.gaps)
            if val is None:
                return traj
            alg_consts[op] = val * ones

    def outputs(state: dict[str, np.ndarray]) -> np.ndarray:
        env: dict[str, np.ndarray] = {**alg_consts, **state}
        for a in model["algebraic"]:
            arrays = [env[o] for o in a["operands"]]
            base = np.prod(arrays, axis=0) if a["op"] == "product" else np.sum(arrays, axis=0)
            env[a["target"]] = base + extras[a["target"]]
        out = env[model["output"]]
        if model["output"] in overlay:
            out = out + overlay[model["output"]]
        return np.asarray(out, dtype=float)

    y = np.zeros((n, horizon + 1))
    y[:, 0] = outputs(stocks)
    for t in range(horizon):
        change = {sid: np.zeros(n) for sid in stocks}
        for f in model["flows"]:
            s_now = stocks[f["stock"]]
            form = f["form"]
            if form == "level":
                amt = flow_inputs[f["level"]]
            elif form == "rate_x_stock":
                amt = flow_inputs[f["rate"]] * f["rate_scale"] * s_now
            elif form == "rate_delta_x_stock":
                amt = (
                    (flow_inputs[f["rate"]] - flow_inputs[f"sq:{f['rate']}"])
                    * f["rate_scale"]
                    * s_now
                )
            else:  # derived_gross: held net + held churn x held S_0, plus the option's net change
                amt = (
                    flow_inputs[f["net"]]
                    + flow_inputs[f"sq:{f['rate']}"] * f["rate_scale"] * s_held0[f["stock"]]
                )
            change[f["stock"]] = change[f["stock"]] + f["sign"] * amt
        stocks = {sid: stocks[sid] + change[sid] for sid in stocks}
        if any(np.any(v < 0) for v in stocks.values()):
            traj.negative_stock = True
        y[:, t + 1] = outputs(stocks)
    traj.y = y
    return traj
