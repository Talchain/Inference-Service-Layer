"""Run the R3-B evaluation over the 12-graph corpus.

Writes ``results.json`` (deterministic: byte-identical across runs) and ``runtime.json``
(wall-clock timings, which are not deterministic). Usage, from ``experiments/r3b_sim``::

    poetry run python -m sim.run
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
import time
from typing import Any, Callable

import numpy as np

from .corpus import ROOT, Graph, graph_ids, load_graph, sha256_file, verify_corpus
from .dynamic import Trajectory, run_model
from .frames import canon_unit, node_unit
from .metrics import (
    evppi_status,
    goal_summary,
    isl_evppi_sha256,
    r6,
    spearman,
    validate_evppi_estimator,
)
from .spec import TIER_NAMES, TIERS, load_valid_spec
from .static import Draws, OptionEval, XEffect, evaluate_all, point_draws, template_draws

SEED = 20260927
N_DRAWS = 2000
STRICT_SEEDS = (SEED, SEED + 1, SEED + 2)
STRICT_N = 50000
CONV_N = (2000, 10000, 50000)
CONVERGENCE_GRAPH = "pj-20260927T180910Z-A"
CONVERGENCE_MODEL = "X_net_reading"
CONVERGENCE_SCENARIO = {
    "X_price_churn_direct": 0.5,
    "X_price_churn_via_sensitivity": 0.0,
    "X_feature_competitive_mrr": 0.0,
}
SPREAD_SCENARIOS = (
    CONVERGENCE_SCENARIO,
    {
        "X_price_churn_direct": 4.25,
        "X_price_churn_via_sensitivity": 0.0,
        "X_feature_competitive_mrr": -31250.0,
    },
)
SPREAD_SD = (0.1, 0.25, 0.5, 1.0)
SPREAD_EXISTS = (1.0, 0.8)
TIME_PATH_GAP = "REQUIRES_T2:time path (level/rate persistence, onset, no lag)"


def gap_codes(gaps: list[str]) -> list[str]:
    return sorted({g.split(":", 1)[0] for g in gaps})


# ---------------------------------------------------------------- static (timeless) blocks


def constraints_block(graph: Graph, evals: dict[str, OptionEval]) -> list[dict[str, Any]]:
    out = []
    for c in graph.constraints:
        node, op, thr = c["node_id"], c["operator"], float(c["value"])
        c_unit, n_unit = canon_unit(c.get("unit")), node_unit(graph, node)
        unit_ok = c_unit is not None and c_unit == n_unit
        unit_assumed = c_unit is not None and n_unit is None
        rows: dict[str, Any] = {}
        for opt in graph.option_ids:
            ev = evals[opt]
            if not ev.computable:
                rows[opt] = {"status": "withheld", "gaps": ev.option_gaps}
                continue
            nv = ev.nodes[node]
            if not unit_ok and not (unit_assumed and nv.tier >= 3):
                code = "UNIT_UNVERIFIABLE" if unit_assumed else "UNIT_MISMATCH"
                rows[opt] = {"status": "withheld", "gaps": [f"{code}:{c['constraint_id']}"]}
                continue
            if nv.value is None:
                rows[opt] = {"status": "withheld", "gaps": sorted(set(nv.gaps))}
                continue
            held = nv.value <= thr if op == "<=" else nv.value >= thr
            row: dict[str, Any] = {
                "status": "computed",
                "kind": "static_timeless",
                "p_held": r6(held.mean()),
                "value_mean": r6(nv.value.mean()),
                "tier": TIER_NAMES[nv.tier],
                "defaults": sorted(nv.defaults),
                "assumptions": sorted(
                    nv.assumptions | ({"X_unit_from_constraint"} if not unit_ok else set())
                ),
                "provenance": sorted(nv.provenance),
                "option_gaps": ev.option_gaps,
            }
            if nv.value.size > 1:
                row["value_q05"] = r6(np.quantile(nv.value, 0.05))
                row["value_q95"] = r6(np.quantile(nv.value, 0.95))
            rows[opt] = row
        out.append(
            {
                "constraint_id": c["constraint_id"],
                "node": node,
                "operator": op,
                "threshold": thr,
                "unit": c.get("unit"),
                "options": rows,
            }
        )
    return out


def identities_block(
    graph: Graph, spec: dict[str, Any], evals: dict[str, OptionEval], derived: tuple[str, ...]
) -> list[dict[str, Any]]:
    targets = [t for t, d in spec["declared_identities"].items()]
    targets += [d["target"] for d in spec["derived_identities"] if d["id"] in derived]
    base = graph.baseline_option
    out = []
    for t in targets:
        rows: dict[str, Any] = {}
        sq_val = evals[base].nodes[t].value if evals[base].computable else None
        for opt in graph.option_ids:
            ev = evals[opt]
            if not ev.computable:
                rows[opt] = {"status": "withheld", "gaps": ev.option_gaps}
                continue
            nv = ev.nodes[t]
            if nv.value is not None:
                rows[opt] = {
                    "status": "computed",
                    "value_mean": r6(nv.value.mean()),
                    "change_vs_status_quo": None if nv.delta is None else r6(nv.delta.mean()),
                    "tier": TIER_NAMES[nv.tier],
                    "defaults": sorted(nv.defaults),
                    "assumptions": sorted(nv.assumptions),
                }
            elif nv.conditional_value is not None:
                rows[opt] = {
                    "status": "conditional",
                    "conditional_on_unchanged": nv.conditional_on,
                    "value_mean": r6(nv.conditional_value.mean()),
                    "change_vs_status_quo": (
                        None if sq_val is None else r6((nv.conditional_value - sq_val).mean())
                    ),
                    "gaps": sorted(set(nv.gaps)),
                }
            else:
                rows[opt] = {"status": "withheld", "gaps": sorted(set(nv.gaps))}
        kind = "declared" if t in spec["declared_identities"] else "derived_X"
        status = spec["declared_identities"][t]["status"] if kind == "declared" else "mode_x"
        out.append({"target": t, "kind": kind, "spec_status": status, "options": rows})
    return out


# ---------------------------------------------------------------- dynamic models


def _active_derived(model: dict[str, Any]) -> tuple[str, ...]:
    ids = {a["derived_id"] for a in model.get("algebraic", []) if a.get("source") == "derived"}
    ids |= {a for a in model["assumptions"] if a.startswith("X_sum")}
    return tuple(sorted(ids))


def _x_effects(spec: dict[str, Any], scenario: dict[str, float]) -> tuple[XEffect, ...]:
    out = []
    for xe in spec["x_effects"]:
        out.append(
            XEffect(
                xe["id"],
                tuple(xe["path"]),
                float(scenario[xe["id"]]),
                float(xe["per_source_change"]),
            )
        )
    return tuple(out)


def scenarios(spec: dict[str, Any]) -> list[dict[str, float]]:
    if not spec["x_effects"]:
        return [{}]
    ids = [xe["id"] for xe in spec["x_effects"]]
    sweeps = [xe["sweep"] for xe in spec["x_effects"]]
    return [dict(zip(ids, combo)) for combo in itertools.product(*sweeps)]


def simulate(
    graph: Graph,
    spec: dict[str, Any],
    model: dict[str, Any],
    scenario: dict[str, float],
    draws: Draws,
) -> dict[str, Trajectory]:
    tier = TIERS[model["tier"]]
    horizon = int(spec["horizon"]["months"])
    # The model evaluates its own algebraic outputs, so their operand edges are handled there.
    replaced = set(model["replaces_edges"])
    for a in model.get("algebraic", []):
        replaced |= {f"{op}->{a['target']}" for op in a["operands"]}
    evals = evaluate_all(
        graph,
        spec,
        tier,
        draws,
        replaced_edges=frozenset(replaced),
        active_derived=_active_derived(model) if tier == 3 else (),
        x_effects=_x_effects(spec, scenario) if tier == 3 else (),
    )
    sq = evals[graph.baseline_option].nodes
    return {
        opt: run_model(
            graph,
            model,
            opt,
            evals[opt].nodes if evals[opt].computable else None,
            sq,
            horizon,
            tier,
            draws.n,
        )
        for opt in graph.option_ids
    }


def _primary(summary: dict[str, Any], semantics: str) -> float:
    if semantics == "attain_by_H" and summary.get("p_by_H") is not None:
        return float(summary["p_by_H"])
    return float(summary["p_at_H"])


def _indicator(y: np.ndarray, thr: float, op: str, first_passage: bool) -> np.ndarray:
    series = y if first_passage else y[:, -1:]
    hit = series >= thr if op == ">=" else series <= thr
    return np.asarray(hit.any(axis=1), dtype=float)


def _theta(draws: Draws, edges: set[str]) -> dict[str, np.ndarray]:
    theta: dict[str, np.ndarray] = {}
    for key in sorted(edges):
        if key in draws.amount and np.unique(draws.amount[key]).size > 1:
            theta[f"amount:{key}"] = draws.amount[key]
        if key in draws.exists and np.unique(draws.exists[key]).size > 1:
            theta[f"exists:{key}"] = draws.exists[key]
    return theta


def model_block(
    graph: Graph,
    spec: dict[str, Any],
    model: dict[str, Any],
    scenario: dict[str, float],
    draws: Draws,
    *,
    strict_check: Callable[[int, int], dict[str, Trajectory]] | None,
    strict_draws: Callable[[int, int], Draws] | None,
) -> dict[str, Any]:
    thr = float(graph.resolve(spec["goal"]["threshold_ptr"]))
    op = spec["goal"]["operator"]
    semantics = spec["goal"]["temporal_semantics"]
    has_path = model["type"] == "stock_flow"
    first_passage = semantics == "attain_by_H" and has_path
    trajs = simulate(graph, spec, model, scenario, draws)
    options: dict[str, Any] = {}
    for opt in graph.option_ids:
        tr = trajs[opt]
        if not tr.ok:
            options[opt] = {"status": "withheld", "gaps": sorted(set(tr.gaps))}
            continue
        assert tr.y is not None
        summ = goal_summary(tr.y, thr, op, has_path)
        options[opt] = {
            "status": "computed",
            "tier": TIER_NAMES[tr.tier],
            "defaults": sorted(tr.defaults),
            "assumptions": sorted(tr.assumptions),
            "n_defaults_and_assumptions": len(tr.defaults) + len(tr.assumptions),
            "negative_stock": tr.negative_stock,
            **summ,
        }
        if has_path:
            options[opt]["h0_control"] = {
                "p_goal_at_month0": summ["p_at_month0"],
                "time_changes_goal_verdict": bool(summ["p_at_month0"] != _primary(summ, semantics)),
            }
    computable = [o for o in graph.option_ids if options[o]["status"] == "computed"]
    block: dict[str, Any] = {
        "model": model["id"],
        "tier": model["tier"],
        "type": model["type"],
        "scenario": scenario,
        "n_draws": draws.n,
        "template_draws": draws.template,
        "primary_metric": "p_by_H" if first_passage else "p_at_H",
        "options": options,
        "computable_options": computable,
    }

    # Whole-decision ranking only when EVERY option is computable at this tier.
    blocking = [o for o in graph.option_ids if o not in computable]
    if not blocking and len(computable) >= 2:
        ranked = sorted(
            computable,
            key=lambda o: (-_primary(options[o], semantics), -options[o]["outcome_at_H_mean"], o),
        )
        p1, p2 = _primary(options[ranked[0]], semantics), _primary(options[ranked[1]], semantics)
        n = draws.n
        se = float(np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)) if n > 1 else 0.0
        block["ranking"] = {
            "status": "computed",
            "order": ranked,
            "best_minus_second_p": r6(p1 - p2),
            "se": r6(se),
            "separated": bool((p1 - p2) > 2 * se) if n > 1 else bool(p1 != p2),
        }
    else:
        block["ranking"] = {"status": "RANKING_WITHHELD", "blocking_options": blocking}

    pair = spec["brief_decision_pair"]
    pair_ok = pair is not None and all(o in computable for o in pair)
    if pair_ok:
        a, b = pair
        pa, pb = _primary(options[a], semantics), _primary(options[b], semantics)
        n = draws.n
        se = float(np.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)) if n > 1 else 0.0
        block["brief_pair"] = {
            "status": "computed",
            "pair": pair,
            "p_goal": {a: r6(pa), b: r6(pb)},
            "outcome_at_H_mean": {
                a: options[a]["outcome_at_H_mean"],
                b: options[b]["outcome_at_H_mean"],
            },
            "delta_p": r6(pb - pa),
            "separated": bool(abs(pb - pa) > 2 * se) if n > 1 else bool(pa != pb),
            "month0_winner_by_outcome": (
                None
                if not has_path
                else (
                    a
                    if options[a]["month0_outcome_mean"] >= options[b]["month0_outcome_mean"]
                    else b
                )
            ),
            "H_winner_by_outcome": a
            if options[a]["outcome_at_H_mean"] >= options[b]["outcome_at_H_mean"]
            else b,
        }
    else:
        block["brief_pair"] = {"status": "withheld", "pair": pair}

    # EVPPI: template draws only; decision = whole set if ranked, else the brief pair.
    decision: list[str] | None = None
    if draws.template:
        if block["ranking"]["status"] == "computed":
            decision = computable
        elif pair_ok:
            decision = list(pair)
    if decision is None:
        if draws.template:
            reason = "FEWER_THAN_2_COMPARABLE_OPTIONS"
        else:
            reason = "POINT_RUN_SEE_MONTE_CARLO_RUN"
        if draws.template and not computable:
            reason = "NO_COMPUTABLE_OPTION"
        block["evppi"] = {"status": "NOT_COMPUTABLE", "reason": reason}
        return block
    used: set[str] = set()
    for o in decision:
        used |= trajs[o].used_edges
    theta = _theta(draws, used)
    ys = {o: trajs[o].y for o in decision}
    outcome_gbp = {o: y[:, -1] for o, y in ys.items() if y is not None}
    outcome_goal = {
        o: _indicator(y, thr, op, first_passage) for o, y in ys.items() if y is not None
    }
    params: dict[str, Any] = {}
    for pid, th in theta.items():
        gbp = evppi_status(th, outcome_gbp, SEED)
        goal = evppi_status(th, outcome_goal, SEED)
        params[pid] = {
            "gbp": gbp,
            "goal_probability": goal,
            "spearman_vs_outcome_at_H": {o: spearman(th, outcome_gbp[o]) for o in decision},
        }
    # Strict status: resolved at 3 seeds AND at 50,000 draws (checked only where resolved).
    if strict_check is not None and strict_draws is not None:
        needs = [
            p
            for p, v in params.items()
            if "resolved" in (v["gbp"]["status"], v["goal_probability"]["status"])
        ]
        if needs:
            reruns = [(s, N_DRAWS) for s in STRICT_SEEDS[1:]] + [(SEED, STRICT_N)]
            stable = {
                p: {
                    "gbp": params[p]["gbp"]["status"] == "resolved",
                    "goal_probability": params[p]["goal_probability"]["status"] == "resolved",
                }
                for p in needs
            }
            for s, n in reruns:
                d2 = strict_draws(s, n)
                t2 = strict_check(s, n)
                th2 = _theta(d2, used)
                y2 = {o: t2[o].y for o in decision}
                g2 = {o: y[:, -1] for o, y in y2.items() if y is not None}
                i2 = {
                    o: _indicator(y, thr, op, first_passage) for o, y in y2.items() if y is not None
                }
                for p in needs:
                    if p not in th2:
                        stable[p] = {"gbp": False, "goal_probability": False}
                        continue
                    stable[p]["gbp"] &= evppi_status(th2[p], g2, s)["status"] == "resolved"
                    stable[p]["goal_probability"] &= (
                        evppi_status(th2[p], i2, s)["status"] == "resolved"
                    )
            for p in params:
                params[p]["strict_resolved"] = stable.get(
                    p, {"gbp": False, "goal_probability": False}
                )
        else:
            for p in params:
                params[p]["strict_resolved"] = {"gbp": False, "goal_probability": False}
    block["evppi"] = {
        "status": "computed",
        "decision_options": decision,
        "decision_scope": "whole" if block["ranking"]["status"] == "computed" else "brief_pair",
        "parameters": params,
        "n_resolved_isl": sum(
            1
            for v in params.values()
            if "resolved" in (v["gbp"]["status"], v["goal_probability"]["status"])
        ),
        "n_resolved_strict": sum(
            1 for v in params.values() if any(v.get("strict_resolved", {}).values())
        ),
    }
    return block


# ---------------------------------------------------------------- per graph


def run_graph(graph: Graph, timings: dict[str, float]) -> dict[str, Any]:
    spec = load_valid_spec(graph)
    res: dict[str, Any] = {
        "journey": graph.journey,
        "brief": graph.brief,
        "horizon": spec["horizon"],
        "goal": {k: spec["goal"][k] for k in ("node", "operator", "temporal_semantics", "gaps")},
        "goal_threshold": (
            float(graph.resolve(spec["goal"]["threshold_ptr"]))
            if spec["goal"]["threshold_ptr"]
            else None
        ),
        "options": graph.option_ids,
        "baseline_option": graph.baseline_option,
        "brief_decision_pair": spec["brief_decision_pair"],
        "tiers": {},
    }
    strict_goal_gaps = sorted(
        set(spec["horizon"]["gaps"]) | set(spec["goal"]["gaps"]) | {TIME_PATH_GAP}
    )

    for tname in ("T0", "T1"):
        t0 = time.perf_counter()
        evals = evaluate_all(graph, spec, TIERS[tname], point_draws(graph))
        res["tiers"][tname] = {
            "constraints": constraints_block(graph, evals),
            "identities": identities_block(graph, spec, evals, ()),
            "goal": {"status": "withheld", "gaps": strict_goal_gaps},
            "evppi": {"status": "NOT_COMPUTABLE", "reason": "UNCERTAINTY_NOT_SPECIFIED"},
        }
        timings[f"{graph.id}:{tname}"] = time.perf_counter() - t0

    # T2: point (links exist, means) and Monte Carlo over the template spread.
    t0 = time.perf_counter()
    t2: dict[str, Any] = {}
    ev_pt = evaluate_all(graph, spec, 2, point_draws(graph))
    ev_mc = evaluate_all(graph, spec, 2, template_draws(graph, N_DRAWS, SEED))
    t2["constraints_point"] = constraints_block(graph, ev_pt)
    t2["constraints_mc"] = constraints_block(graph, ev_mc)
    t2["identities"] = identities_block(graph, spec, ev_pt, ())
    gm = spec["goal_model_t2"]
    if gm["status"] == "model" and spec["horizon"]["status"] == "resolved":
        model = next(m for m in spec["dynamic_models"] if m["id"] == gm["model"])
        t2["goal_models"] = [
            model_block(
                graph, spec, model, {}, point_draws(graph), strict_check=None, strict_draws=None
            ),
            model_block(
                graph,
                spec,
                model,
                {},
                template_draws(graph, N_DRAWS, SEED),
                strict_check=lambda s, n: simulate(
                    graph, spec, model, {}, template_draws(graph, n, s)
                ),
                strict_draws=lambda s, n: template_draws(graph, n, s),
            ),
        ]
    else:
        t2["goal_models"] = []
        t2["goal"] = {
            "status": "withheld",
            "gaps": sorted(set(gm["gaps"]) | set(spec["horizon"]["gaps"])),
        }
    res["tiers"]["T2"] = t2
    timings[f"{graph.id}:T2"] = time.perf_counter() - t0

    # Mode X: static pass per scenario, then every X model per scenario.
    t0 = time.perf_counter()
    xres: dict[str, Any] = {"scenarios": [], "models": []}
    derived_all = tuple(d["id"] for d in spec["derived_identities"])
    if spec["x_effects"] or derived_all:
        for sc in scenarios(spec):
            ev = evaluate_all(
                graph,
                spec,
                3,
                point_draws(graph),
                active_derived=derived_all,
                x_effects=_x_effects(spec, sc),
            )
            xres["scenarios"].append(
                {
                    "scenario": sc,
                    "constraints_point": constraints_block(graph, ev),
                    "identities": identities_block(graph, spec, ev, derived_all),
                }
            )
    x_models = [m for m in spec["dynamic_models"] if m["tier"] == "X"]
    if spec["horizon"]["status"] == "resolved":
        for model in x_models:
            for sc in scenarios(spec):
                xres["models"].append(
                    model_block(
                        graph,
                        spec,
                        model,
                        sc,
                        point_draws(graph),
                        strict_check=None,
                        strict_draws=None,
                    )
                )

                def _chk(
                    s: int, n: int, m: dict[str, Any] = model, c: dict[str, float] = sc
                ) -> dict[str, Trajectory]:
                    return simulate(graph, spec, m, c, template_draws(graph, n, s))

                xres["models"].append(
                    model_block(
                        graph,
                        spec,
                        model,
                        sc,
                        template_draws(graph, N_DRAWS, SEED),
                        strict_check=_chk,
                        strict_draws=lambda s, n: template_draws(graph, n, s),
                    )
                )
    res["tiers"]["X"] = xres
    timings[f"{graph.id}:X"] = time.perf_counter() - t0
    res["summary"] = summarise(graph, spec, res)
    return res


def _computed_rows(blocks: list[dict[str, Any]]) -> int:
    return sum(
        1
        for b in blocks
        for row in b["options"].values()
        if row["status"] in ("computed", "conditional")
    )


def summarise(graph: Graph, spec: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    all_gaps: set[str] = (
        set(spec["graph_gaps"]) | set(spec["goal"]["gaps"]) | set(spec["horizon"]["gaps"])
    )
    for tname in ("T0", "T1"):
        t = res["tiers"][tname]
        n_con = _computed_rows(t["constraints"])
        n_id = _computed_rows(t["identities"])
        for b in t["constraints"] + t["identities"]:
            for row in b["options"].values():
                all_gaps |= set(gap_codes(row.get("gaps", [])))
        out[tname] = {
            "category": "static_only" if (n_con or n_id) else "none",
            "static_limit_verdicts": n_con,
            "identity_values": n_id,
            "deadline_goal": "withheld",
            "time_accumulation": False,
            "b_unique_strong_results": 0,
        }
    t2 = res["tiers"]["T2"]
    models = t2["goal_models"]
    t2_goal_rows = sum(len(m["computable_options"]) for m in models[:1])
    out["T2"] = {
        "category": "deadline_goal"
        if t2_goal_rows
        else (
            "static_only"
            if (_computed_rows(t2["constraints_point"]) or _computed_rows(t2["identities"]))
            else "none"
        ),
        "deadline_goal_options": t2_goal_rows,
        "time_accumulation": bool(models and models[0]["type"] == "stock_flow" and t2_goal_rows),
        "ranking_available": any(m["ranking"]["status"] == "computed" for m in models),
        "brief_pair_available": any(m["brief_pair"]["status"] == "computed" for m in models),
        "time_changes_verdict": any(
            row.get("h0_control", {}).get("time_changes_goal_verdict", False)
            for m in models
            for row in m["options"].values()
        ),
        "evppi_resolved_isl": sum(m["evppi"].get("n_resolved_isl", 0) for m in models),
        "evppi_resolved_strict": sum(m["evppi"].get("n_resolved_strict", 0) for m in models),
    }
    for m in models:
        for row in m["options"].values():
            all_gaps |= set(gap_codes(row.get("gaps", [])))
    xm = res["tiers"]["X"]["models"]
    out["X"] = {
        "models_run": sorted({m["model"] for m in xm}),
        "scenarios": len(res["tiers"]["X"]["scenarios"]) or (1 if xm else 0),
        "deadline_goal_rows": sum(
            len(m["computable_options"]) for m in xm if not m["template_draws"]
        ),
        "ranking_available": any(m["ranking"]["status"] == "computed" for m in xm),
        "brief_pair_available": any(m["brief_pair"]["status"] == "computed" for m in xm),
        "brief_pair_winner_by_scenario": sorted(
            {
                (
                    m["model"],
                    json.dumps(m["scenario"], sort_keys=True),
                    m["brief_pair"]["H_winner_by_outcome"],
                )
                for m in xm
                if not m["template_draws"] and m["brief_pair"]["status"] == "computed"
            }
        ),
        "evppi_resolved_isl": sum(m["evppi"].get("n_resolved_isl", 0) for m in xm),
        "evppi_resolved_strict": sum(m["evppi"].get("n_resolved_strict", 0) for m in xm),
    }
    verdicts: dict[str, set[float]] = {}
    for m in xm:
        if m["template_draws"]:
            continue
        for o, row in m["options"].items():
            if row["status"] == "computed":
                verdicts.setdefault(o, set()).add(_primary(row, res["goal"]["temporal_semantics"]))
    out["X"]["goal_verdicts_by_option"] = {o: sorted(v) for o, v in sorted(verdicts.items())}
    out["X"]["goal_verdict_flips"] = sorted(o for o, v in verdicts.items() if len(v) > 1)
    winners = {w for _, _, w in out["X"]["brief_pair_winner_by_scenario"]}
    out["X"]["brief_pair_winner_flips_across_scenarios"] = len(winners) > 1
    out["gap_classes"] = sorted(
        g for g in all_gaps if not g.startswith(("TAINTED_BY", "OPERAND_WITHHELD"))
    )
    return out


# ---------------------------------------------------------------- corpus-level extras


def convergence() -> dict[str, Any]:
    g = load_graph(CONVERGENCE_GRAPH)
    spec = load_valid_spec(g)
    model = next(m for m in spec["dynamic_models"] if m["id"] == CONVERGENCE_MODEL)
    rows = []
    for n in CONV_N:
        blk = model_block(
            g,
            spec,
            model,
            CONVERGENCE_SCENARIO,
            template_draws(g, n, SEED),
            strict_check=None,
            strict_draws=None,
        )
        rows.append(
            {
                "n_draws": n,
                "options": {
                    o: {
                        k: v
                        for k, v in r.items()
                        if k
                        in (
                            "p_by_H",
                            "p_by_H_mc_se",
                            "p_at_H",
                            "outcome_at_H_mean",
                            "outcome_at_H_quantiles",
                        )
                    }
                    for o, r in blk["options"].items()
                    if r["status"] == "computed"
                },
                "evppi": {
                    p: {"gbp": v["gbp"], "goal_probability": v["goal_probability"]}
                    for p, v in blk["evppi"].get("parameters", {}).items()
                },
            }
        )
    return {
        "graph": CONVERGENCE_GRAPH,
        "model": CONVERGENCE_MODEL,
        "scenario": CONVERGENCE_SCENARIO,
        "runs": rows,
    }


def spread_sweep() -> dict[str, Any]:
    g = load_graph(CONVERGENCE_GRAPH)
    spec = load_valid_spec(g)
    rows = []
    for model in [m for m in spec["dynamic_models"] if m["tier"] == "X"]:
        for sc in SPREAD_SCENARIOS:
            for sd in SPREAD_SD:
                for ex in SPREAD_EXISTS:
                    d = template_draws(g, N_DRAWS, SEED, sd_ratio_override=sd, exists_override=ex)
                    blk = model_block(g, spec, model, sc, d, strict_check=None, strict_draws=None)
                    rows.append(
                        {
                            "model": model["id"],
                            "scenario": sc,
                            "sd_over_mean": sd,
                            "exists_probability": ex,
                            "p_by_H": {
                                o: r["p_by_H"]
                                for o, r in blk["options"].items()
                                if r["status"] == "computed"
                            },
                            "evppi_resolved_isl": blk["evppi"].get("n_resolved_isl", 0),
                            "brief_pair_delta_p": blk["brief_pair"].get("delta_p"),
                        }
                    )
    return {"graph": CONVERGENCE_GRAPH, "rows": rows}


def frozen_check() -> dict[str, Any]:
    lines = (ROOT / "FROZEN.sha256").read_text().split("\n")
    entries = [ln.split() for ln in lines if ln.strip()]
    status = {name: sha256_file(ROOT / name) == digest for digest, name in entries}
    return {"all_match": all(status.values()), "files": {k: status[k] for k in sorted(status)}}


def aggregate(graphs: dict[str, Any], evppi_valid: dict[str, Any]) -> dict[str, Any]:
    cats: dict[str, dict[str, int]] = {}
    for tname in ("T0", "T1", "T2"):
        c: dict[str, int] = {}
        for g in graphs.values():
            k = g["summary"][tname]["category"]
            c[k] = c.get(k, 0) + 1
        cats[tname] = dict(sorted(c.items()))
    gap_freq: dict[str, int] = {}
    for g in graphs.values():
        for code in g["summary"]["gap_classes"]:
            gap_freq[code] = gap_freq.get(code, 0) + 1
    x_deadline = sum(1 for g in graphs.values() if g["summary"]["X"]["deadline_goal_rows"])
    return {
        "graphs": len(graphs),
        "category_by_tier": cats,
        "t2_graphs_with_deadline_goal": sum(
            1 for g in graphs.values() if g["summary"]["T2"]["deadline_goal_options"]
        ),
        "x_graphs_with_deadline_goal": x_deadline,
        "b_unique_strong_results_T0_T1": sum(
            g["summary"][t]["b_unique_strong_results"]
            for g in graphs.values()
            for t in ("T0", "T1")
        ),
        "whole_decision_ranking_available": {
            "T2": sum(1 for g in graphs.values() if g["summary"]["T2"]["ranking_available"]),
            "X": sum(1 for g in graphs.values() if g["summary"]["X"]["ranking_available"]),
        },
        "secondary_diagnostic_share_with_resolved_evppi": {
            "T0_T1": 0.0,
            "T2_isl": r6(
                sum(1 for g in graphs.values() if g["summary"]["T2"]["evppi_resolved_isl"])
                / len(graphs)
            ),
            "T2_strict": r6(
                sum(1 for g in graphs.values() if g["summary"]["T2"]["evppi_resolved_strict"])
                / len(graphs)
            ),
            "X_isl": r6(
                sum(1 for g in graphs.values() if g["summary"]["X"]["evppi_resolved_isl"])
                / len(graphs)
            ),
            "X_strict": r6(
                sum(1 for g in graphs.values() if g["summary"]["X"]["evppi_resolved_strict"])
                / len(graphs)
            ),
            "note": "Secondary diagnostic only; NOT a success criterion. Metric name: "
            + evppi_valid["metric_name"],
        },
        "gap_class_graph_frequency": dict(sorted(gap_freq.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def main() -> None:
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    corpus_ok = verify_corpus()
    frozen = frozen_check()
    if not all(corpus_ok.values()) or not frozen["all_match"]:
        raise SystemExit("corpus or frozen mapping hashes do not match; see AMENDMENTS.md")
    evppi_valid = validate_evppi_estimator()
    graphs = {}
    for gid in graph_ids():
        t0 = time.perf_counter()
        graphs[gid] = run_graph(load_graph(gid), timings)
        timings[f"{gid}:total"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    conv = convergence()
    timings["convergence"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    spread = spread_sweep()
    timings["spread_sweep"] = time.perf_counter() - t0
    results = {
        "meta": {
            "seed": SEED,
            "n_draws": N_DRAWS,
            "strict_evppi": {"seeds": list(STRICT_SEEDS), "n_draws": STRICT_N},
            "python": platform.python_version(),
            "numpy": np.__version__,
            "corpus_manifest_sha256": sha256_file(ROOT / "corpus" / "MANIFEST.json"),
            "corpus_files_verified": all(corpus_ok.values()),
            "frozen": frozen,
            "analysis_plan_sha256": sha256_file(ROOT / "ANALYSIS-PLAN.md"),
            "isl_evppi_sha256": isl_evppi_sha256(),
            "production_engine_used": False,
        },
        "evppi_validation": evppi_valid,
        "aggregate": aggregate(graphs, evppi_valid),
        "graphs": graphs,
        "convergence": conv,
        "spread_sweep": spread,
    }
    text = json.dumps(results, indent=1, sort_keys=True) + "\n"
    (ROOT / "results.json").write_text(text)
    timings["all"] = time.perf_counter() - t_start
    (ROOT / "runtime.json").write_text(
        json.dumps({k: round(v, 4) for k, v in sorted(timings.items())}, indent=1) + "\n"
    )
    print("results.json sha256", hashlib.sha256(text.encode()).hexdigest(), file=sys.stderr)


if __name__ == "__main__":
    main()
