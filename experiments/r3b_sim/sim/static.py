"""Timeless (H=0) evaluation of one option at one evidence tier, with taint propagation.

Every node value is a numpy array over draws (length 1 at T0/T1). A node is WITHHELD when any
input it needs is inadmissible at the requested tier; the reason is carried as GAP strings.
Nothing is defaulted: an unquantified edge whose source moves taints its target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .corpus import Graph
from .edges import classify_edge, definitional_edges
from .frames import Level, held_level, intervention_level

_EXACT = 1e-9


@dataclass
class NodeVal:
    value: np.ndarray | None
    delta: np.ndarray | None
    gaps: list[str] = field(default_factory=list)
    tier: int = 0
    defaults: set[str] = field(default_factory=set)
    assumptions: set[str] = field(default_factory=set)
    provenance: set[str] = field(default_factory=set)
    conditional_on: list[str] = field(default_factory=list)
    conditional_value: np.ndarray | None = None
    extra: np.ndarray | None = None
    used_edges: set[str] = field(default_factory=set)

    @property
    def withheld(self) -> bool:
        return self.value is None and self.delta is None


@dataclass
class Draws:
    """Per-edge natural-effect amounts and existence indicators, shared across options (CRN)."""

    n: int
    amount: dict[str, np.ndarray]
    exists: dict[str, np.ndarray]
    template: bool


@dataclass(frozen=True)
class XEffect:
    """A Mode X assumption: an effect in user units from path[0] to path[-1]."""

    id: str
    path: tuple[str, ...]
    amount: float
    per: float


@dataclass
class OptionEval:
    option: str
    tier: int
    nodes: dict[str, NodeVal]
    option_gaps: list[str]
    computable: bool


def point_draws(graph: Graph) -> Draws:
    """T0/T1: the natural-effect mean, conditional on the link existing."""
    amount: dict[str, np.ndarray] = {}
    exists: dict[str, np.ndarray] = {}
    for e in graph.behavioural_edges:
        ne = e.natural_effect
        if ne is not None:
            amount[e.key] = np.array([float(ne["amount"])])
            exists[e.key] = np.ones(1)
    return Draws(1, amount, exists, template=False)


def template_draws(
    graph: Graph,
    n: int,
    seed: int,
    sd_ratio_override: float | None = None,
    exists_override: float | None = None,
) -> Draws:
    """T2/X: amount ~ Normal(amount, |amount| x edge sd/mean); existence ~ Bernoulli(p).

    Drawn ONCE per trajectory (the caller never resamples per month). Edges are visited in a
    fixed sorted order so the streams are reproducible."""
    rng = np.random.default_rng(seed)
    amount: dict[str, np.ndarray] = {}
    exists: dict[str, np.ndarray] = {}
    for e in sorted(graph.behavioural_edges, key=lambda x: x.key):
        ne = e.natural_effect
        if ne is None:
            continue
        mean = float(ne["amount"])
        ratio = (
            sd_ratio_override
            if sd_ratio_override is not None
            else abs(e.strength_std / e.strength_mean)
        )
        p = exists_override if exists_override is not None else e.exists_probability
        amount[e.key] = rng.normal(mean, abs(mean) * ratio, size=n)
        exists[e.key] = (rng.random(n) < p).astype(float)
    return Draws(n, amount, exists, template=True)


def _topo_order(graph: Graph, extra_links: list[tuple[str, str]]) -> list[str]:
    nodes = graph.quantity_ids
    preds: dict[str, set[str]] = {n: set() for n in nodes}
    for e in graph.behavioural_edges:
        if e.dst in preds:
            preds[e.dst].add(e.src)
    for a, b in extra_links:
        preds[b].add(a)
    order: list[str] = []
    done: set[str] = set()
    remaining = sorted(nodes)
    while remaining:
        progressed = False
        for n in list(remaining):
            if preds[n] <= done:
                order.append(n)
                done.add(n)
                remaining.remove(n)
                progressed = True
        if not progressed:
            raise ValueError(f"{graph.id}: cycle among {remaining}")
    return order


def _level_ok(level: Level | None, tier: int) -> tuple[bool, str | None]:
    if level is None:
        return False, None
    if not level.ok:
        return False, level.gaps[0] if level.gaps else "FRAME_MISSING"
    if level.tier > tier:
        return False, f"REQUIRES_T{level.tier}"
    return True, None


def evaluate_option(
    graph: Graph,
    spec: dict[str, Any],
    option_id: str,
    tier: int,
    draws: Draws,
    *,
    status_quo: dict[str, NodeVal] | None,
    replaced_edges: frozenset[str],
    active_derived: tuple[str, ...],
    x_effects: tuple[XEffect, ...],
) -> OptionEval:
    """Evaluate ``option_id``. ``status_quo`` is the baseline's evaluation (None when this IS
    the baseline). Derived identities and X effects are only honoured at tier X (3)."""
    n = draws.n
    if tier < 3 and (active_derived or x_effects):
        raise ValueError("derived identities / X effects are Mode X only")
    option_gaps: list[str] = []
    is_baseline = graph.nodes[option_id].get("is_baseline", False)
    interventions: dict[str, dict[str, Any]] = {}
    if not is_baseline:
        raw = graph.interventions(option_id)
        if not raw:
            return OptionEval(option_id, tier, {}, ["OPTION_LEVELS_MISSING"], False)
        for target, iv_spec in raw.items():
            if target not in graph.nodes or graph.is_structural(target):
                option_gaps.append(f"OPTION_TARGET_INVALID:{target}")
                continue
            interventions[target] = iv_spec

    declared_admitted = {
        t: graph.identity(t)
        for t, d in spec["declared_identities"].items()
        if d["status"] == "admitted"
    }
    derived = {d["target"]: d for d in spec["derived_identities"] if d["id"] in active_derived}
    definitional = definitional_edges(graph, spec)
    for d in derived.values():
        for op in d["operands"]:
            definitional.add(f"{op}->{d['target']}")

    # X effects become virtual edges; the real edges on their paths are covered (no taint).
    covered: set[str] = set()
    virtual_in: dict[str, list[XEffect]] = {}
    for xe in x_effects:
        for a, b in zip(xe.path, xe.path[1:]):
            covered.add(f"{a}->{b}")
        virtual_in.setdefault(xe.path[-1], []).append(xe)

    links: list[tuple[str, str]] = [(xe.path[0], xe.path[-1]) for xe in x_effects]
    for d in derived.values():
        links.extend((op, d["target"]) for op in d["operands"])
    order = _topo_order(graph, links)
    vals: dict[str, NodeVal] = {}
    ones = np.ones(n)

    def sq_value(node: str) -> np.ndarray | None:
        if status_quo is None:
            return vals[node].value
        return status_quo[node].value

    for node in order:
        nv = NodeVal(value=None, delta=None)
        held = held_level(graph, node)
        held_ok, held_gap = _level_ok(held, tier)

        # 1. Intervened node: the option sets its level in its own frame (do-operator).
        if node in interventions:
            iv = intervention_level(graph, node, interventions[node])
            iv_ok, iv_gap = _level_ok(iv, tier)
            if not iv_ok:
                nv.gaps.append(f"{iv_gap}:intervention {node}")
                vals[node] = nv
                continue
            assert iv.value is not None
            nv.value = iv.value * ones
            nv.tier = iv.tier
            nv.provenance.add(f"intervention:{iv.source}")
            if held_ok and held is not None and held.value is not None:
                nv.delta = nv.value - held.value
                nv.tier = max(nv.tier, held.tier)
            else:
                nv.gaps.append(f"{held_gap or 'BASELINE_MISSING'}:status quo of {node}")
            vals[node] = nv
            continue

        # 2. Contributions from ordinary incoming edges (and Mode X virtual edges).
        contrib = np.zeros(n)
        tainted: list[str] = []
        for e in graph.incoming(node):
            if e.key in replaced_edges or e.key in definitional or e.key in covered:
                continue
            src = vals[e.src]
            if src.withheld or (src.delta is None and src.gaps):
                tainted.append(f"TAINTED_BY:{e.src}")
                nv.gaps.extend(src.gaps)
                continue
            d_src = src.delta if src.delta is not None else np.zeros(n)
            if np.all(np.abs(d_src) <= _EXACT):
                continue
            ec = classify_edge(graph, spec, e)
            if ec.cls != "behavioural_parameter":
                tainted.append(f"STATIC_COEFFICIENT_NO_TEMPORAL_MEANING:{e.key}")
                continue
            if not ec.units_fit:
                tainted.append(f"UNIT_MISMATCH:{e.key}")
                continue
            assert ec.tier is not None and ec.ne_per is not None
            if ec.tier > tier:
                tainted.append(f"REQUIRES_T{ec.tier}:natural_effect {e.key}")
                continue
            ratio = d_src / ec.ne_per
            if not np.all(np.abs(ratio - 1.0) <= _EXACT):
                if tier < 2:
                    tainted.append(f"REQUIRES_T2:linear_scaling {e.key}")
                    continue
                nv.defaults.add("linear_scaling")
            contrib = contrib + draws.amount[e.key] * draws.exists[e.key] * ratio
            nv.used_edges.add(e.key)
            nv.used_edges |= src.used_edges
            nv.tier = max(nv.tier, ec.tier, src.tier)
            nv.defaults |= src.defaults
            nv.assumptions |= src.assumptions
            nv.provenance |= src.provenance
            if draws.template:
                nv.defaults.add("template_uncertainty")
            else:
                nv.defaults.add("conditional:links_exist")
        for xe in virtual_in.get(node, []):
            src = vals[xe.path[0]]
            if src.delta is None:
                if src.withheld or src.gaps:
                    tainted.append(f"TAINTED_BY:{xe.path[0]}")
                continue
            if np.all(np.abs(src.delta) <= _EXACT):
                continue
            contrib = contrib + xe.amount * (src.delta / xe.per)
            nv.assumptions.add(xe.id)
            nv.assumptions |= src.assumptions
            nv.defaults |= src.defaults
            nv.used_edges |= src.used_edges
            nv.tier = max(nv.tier, 3, src.tier)

        # 3. Identity targets: f(operands) in user units (R3-8), plus ordinary contributions.
        ident_ops: list[str] | None = None
        ident_op = "product"
        overrides_held = False
        if node in declared_admitted:
            ident = declared_admitted[node]
            assert ident is not None
            ident_ops = list(ident["factor_ids"])
        elif node in derived:
            ident_ops = list(derived[node]["operands"])
            ident_op = "sum"
            overrides_held = bool(derived[node]["identity_overrides_held"])
            nv.assumptions.add(derived[node]["id"])
            nv.tier = max(nv.tier, 3)
        if ident_ops is not None:
            op_vals = [vals[o] for o in ident_ops]
            missing = [o for o, v in zip(ident_ops, op_vals) if v.value is None]
            if missing:
                for o in missing:
                    nv.gaps.append(f"OPERAND_WITHHELD:{o}")
                    nv.gaps.extend(g for g in vals[o].gaps if g not in nv.gaps)
                    if not vals[o].gaps:
                        nv.gaps.append(f"FRAME_MISSING:identity operand {o}")
                # Conditional statement: tainted operands held at their status-quo level.
                cond: list[np.ndarray] = []
                for o, v in zip(ident_ops, op_vals):
                    if v.value is not None:
                        cond.append(v.value)
                        continue
                    sq = status_quo[o].value if status_quo is not None else None
                    if sq is None:
                        cond = []
                        break
                    cond.append(sq)
                if cond and status_quo is not None:
                    nv.conditional_on = missing
                    nv.conditional_value = (
                        np.prod(cond, axis=0) if ident_op == "product" else np.sum(cond, axis=0)
                    )
                vals[node] = nv
                continue
            arrays = [v.value for v in op_vals if v.value is not None]
            base = np.prod(arrays, axis=0) if ident_op == "product" else np.sum(arrays, axis=0)
            for v in op_vals:
                nv.tier = max(nv.tier, v.tier)
                nv.defaults |= v.defaults
                nv.assumptions |= v.assumptions
                nv.provenance |= v.provenance
                nv.used_edges |= v.used_edges
            if held is not None and held_ok and held.value is not None and not overrides_held:
                nv.tier = max(nv.tier, held.tier)
                if status_quo is None and abs(float(base[0]) - held.value) > 0.005 * max(
                    1.0, abs(held.value)
                ):
                    nv.gaps.append(f"IDENTITY_INCONSISTENT_WITH_HELD_LEVEL:{node}")
                    vals[node] = nv
                    continue
            if tainted:
                nv.gaps.extend(tainted)
                vals[node] = nv
                continue
            if status_quo is not None and status_quo[node].value is None:
                # The identity was withheld for the status quo (e.g. inconsistent with its held
                # level), so no option's value or change can be stated either.
                nv.gaps.extend(status_quo[node].gaps or [f"STATUS_QUO_WITHHELD:{node}"])
                vals[node] = nv
                continue
            nv.extra = contrib
            nv.value = base + contrib
            sq = sq_value(node) if status_quo is not None else nv.value
            nv.delta = None if sq is None else nv.value - sq
            vals[node] = nv
            continue

        # 4. Ordinary node: held level + contributions.
        if tainted:
            nv.gaps.extend(tainted)
            vals[node] = nv
            continue
        nv.delta = contrib
        if held_ok and held is not None and held.value is not None:
            nv.value = held.value + contrib
            nv.tier = max(nv.tier, held.tier)
            nv.provenance.add(f"held:{held.source}")
        elif held is not None and held_gap is not None:
            # The level exists but is inadmissible here: its CHANGE is still known.
            nv.gaps.append(f"{held_gap}:held level {node}")
        vals[node] = nv

    return OptionEval(option_id, tier, vals, option_gaps, True)


def evaluate_all(
    graph: Graph,
    spec: dict[str, Any],
    tier: int,
    draws: Draws,
    *,
    replaced_edges: frozenset[str] = frozenset(),
    active_derived: tuple[str, ...] = (),
    x_effects: tuple[XEffect, ...] = (),
) -> dict[str, OptionEval]:
    base = graph.baseline_option
    sq = evaluate_option(
        graph,
        spec,
        base,
        tier,
        draws,
        status_quo=None,
        replaced_edges=replaced_edges,
        active_derived=active_derived,
        x_effects=x_effects,
    )
    out = {base: sq}
    for opt in graph.option_ids:
        if opt == base:
            continue
        out[opt] = evaluate_option(
            graph,
            spec,
            opt,
            tier,
            draws,
            status_quo=sq.nodes,
            replaced_edges=replaced_edges,
            active_derived=active_derived,
            x_effects=x_effects,
        )
    return out
